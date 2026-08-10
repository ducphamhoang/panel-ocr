//! L4 — spec §16.38 items 1(g), 3(g), 3(h) and 5(e): the tile loop's write-back and the page
//! assembly. This is where the L0 spike's hardest finding is enforced: *"The model does not
//! composite; the caller must."*

mod common;

use common::{flat_rgb, gray_rect, radii, region, rgba_rect, transparent};
use image::{GrayImage, Rgba, RgbaImage};
use pc_config::InpainterConfig;
use pc_core::{MaskRegionStats, Rect, StageError};
use pc_imageops::{BinaryMask, PIL_BINARY_THRESHOLD};
use pc_inpaint::stub::{StubInpainter, StubMode};
use pc_inpaint::{inpaint_page, PageInput, PageOutput};

const PAGE: (u32, u32) = (600, 600);
const BASE: [u8; 3] = [200, 200, 200];

/// One failed region in the middle of the page, with a 10×10 raw-mask blob inside it. At the ratified
/// defaults its `growth` is `7 + int(30 * 0.2) = 13`, so the padded box is `(272,272,328,328)` and the
/// merged rectangle fits in one centred window.
fn middle_region() -> Vec<MaskRegionStats> {
    vec![region(Rect::new(290, 290, 310, 310), 30.0, true, Some(2))]
}

fn middle_raw() -> GrayImage {
    gray_rect(PAGE, Rect::new(295, 295, 305, 305))
}

/// The `final_mask` of §16.38 item 3(h), recomputed through the individually pinned pure functions
/// rather than read back out of the artifact under test. `l4_fill_mask.rs`, `l4_growth.rs` and
/// `l4_fade.rs` are what make each step of this trustworthy; nothing here derives an expectation from
/// `inpaint_page`'s own output.
fn final_mask_oracle(
    regions: &[MaskRegionStats],
    raw: &GrayImage,
    combined: &RgbaImage,
    config: &InpainterConfig,
) -> GrayImage {
    let frame = combined.dimensions();
    let raw_binary = BinaryMask::from_gray_threshold(raw, PIL_BINARY_THRESHOLD);
    let combined_binary = pc_inpaint::combined_fill_binary(combined);
    let padded: Vec<_> = pc_inpaint::select_regions(regions, config)
        .iter()
        .map(|region| {
            pc_inpaint::padded_region(region, &raw_binary, &combined_binary, 0, config, frame)
        })
        .collect();
    let masks = pc_inpaint::compose_page_masks(&padded, frame, config.inpainting_isolation_radius);
    let faded = pc_inpaint::fade_fill_mask(&masks.fill, config.inpainting_fade_radius);
    pc_inpaint::cut_by_isolation(&faded, &masks.isolation)
}

fn run(
    regions: &[MaskRegionStats],
    raw: &GrayImage,
    combined: &RgbaImage,
    noise: Option<&RgbaImage>,
    config: &InpainterConfig,
    inpainter: &StubInpainter,
    original: &image::RgbImage,
) -> PageOutput {
    inpaint_page(
        PageInput {
            original,
            raw_mask: raw,
            combined_mask: combined,
            noise_mask: noise,
            regions,
            min_mask_thickness: 0,
            config,
        },
        inpainter,
    )
    .expect("the stub inpainter never fails")
}

/// **§16.38 item 1(g), the L0 spike's measurement, and the single most important assertion in L4.**
/// Quoted: *"With an all-zero mask (nothing to fill) only **1 of 786,432** output pixels matched the
/// input exactly — the model regenerates the whole tile. … So blending the unmasked region back from
/// the original is a **hard requirement**, not an optimisation."*
///
/// The stub fills its entire 512×512 tile with a colour nothing on the page has. Three assertions,
/// each catching a different way to get this wrong:
///
///   * a page corner far outside the tile is untouched — catches writing the whole page;
///   * the number of changed pixels equals the number of pixels where the independently recomputed
///     `final_mask` is non-zero — catches writing the whole *tile* (`262,144` pixels), and catches
///     writing a subtly wrong region;
///   * that number sits inside hard-coded bounds derived from the geometry alone (the fill is a 10×10
///     blob dilated by `growth = 13`, so ≈1150 pixels, and everything changed lies inside the
///     isolation mask, ≈1840 pixels) — so the equality above cannot be satisfied by two zeros.
#[test]
fn only_the_final_mask_area_changes_and_the_kept_region_comes_from_the_original() {
    let config = InpainterConfig::default();
    let original = flat_rgb(PAGE, BASE);
    let combined = transparent(PAGE);
    let raw = middle_raw();
    let stub = StubInpainter::flat(255, 0, 0);

    let output = run(
        &middle_region(),
        &raw,
        &combined,
        None,
        &config,
        &stub,
        &original,
    );

    assert_eq!(output.clean_inpaint.dimensions(), PAGE);
    assert_eq!(output.tiles_inferred, 1);
    assert_eq!(
        *output.clean_inpaint.get_pixel(0, 0),
        Rgba([BASE[0], BASE[1], BASE[2], 255]),
        "a corner far outside the 512x512 window must be untouched"
    );

    let expected = final_mask_oracle(&middle_region(), &raw, &combined, &config);
    let mut changed = 0_usize;
    let mut expected_nonzero = 0_usize;
    for y in 0..PAGE.1 {
        for x in 0..PAGE.0 {
            let composed = output.clean_inpaint.get_pixel(x, y);
            let base = original.get_pixel(x, y);
            if composed.0[..3] != base.0[..] {
                changed += 1;
            }
            if expected.get_pixel(x, y).0[0] != 0 {
                expected_nonzero += 1;
            }
        }
    }

    assert_eq!(
        changed, expected_nonzero,
        "exactly the `faded_fill AND isolation` area may change (§16.38 items 3(h), 5(e)); \
         pasting the raw tile would change 262144 pixels"
    );
    assert!(
        (800..2500).contains(&changed),
        "changed = {changed}; the geometry puts it near 1150 (a 10x10 blob dilated by 13) and \
         under the isolation mask's ~1840 — a count outside these hand-derived bounds means the \
         equality above was satisfied by the wrong number"
    );
}

/// §16.38 item 3(h): the fade is a *blend*, not a stencil. Upstream makes the faded mask the inpainted
/// layer's alpha (`inpainting.py:168`) and alpha-composites (`:170`), so the transition band carries
/// intermediate values.
///
/// A hard paste produces **zero** partially blended pixels, which is what the `> 0` assertion rules
/// out; the equality then pins *which* pixels those are. Against a base of `200` and a stub of `0`,
/// `round(200 * (1 - a/255))` lands in `1..=199` for every `a` in `1..=254` and never for `a = 0` or
/// `a = 255`, so the two counts are comparable without a tolerance.
#[test]
fn the_fade_yields_partially_blended_pixels_rather_than_a_hard_edge() {
    let config = InpainterConfig::default();
    let original = flat_rgb(PAGE, BASE);
    let combined = transparent(PAGE);
    let raw = middle_raw();
    let stub = StubInpainter::flat(0, 0, 0);

    let output = run(
        &middle_region(),
        &raw,
        &combined,
        None,
        &config,
        &stub,
        &original,
    );
    let expected = final_mask_oracle(&middle_region(), &raw, &combined, &config);

    let mut partial_composed = 0_usize;
    let mut partial_expected = 0_usize;
    for y in 0..PAGE.1 {
        for x in 0..PAGE.0 {
            let value = output.clean_inpaint.get_pixel(x, y).0[0];
            if (1..=199).contains(&value) {
                partial_composed += 1;
            }
            let alpha = expected.get_pixel(x, y).0[0];
            if (1..=254).contains(&alpha) {
                partial_expected += 1;
            }
        }
    }

    assert!(
        partial_composed > 0,
        "a hard paste produces no partially blended pixels at all"
    );
    assert_eq!(partial_composed, partial_expected);
}

/// §16.9 item 15's lerp, reaching this stage through §16.38 item 3(h)'s alpha composite:
/// `round(base * (1 - alpha) + color * alpha)`. Asserted at the first partially faded pixel found, with
/// the expected value computed in the test from the recomputed alpha — not read back from the output.
/// **Post-§16.45, `alpha_composite_over`'s general rule is real source-over, not this plain lerp** —
/// this citation still holds here only because `clean_inpaint`'s destination is opaque (`da == 255`),
/// the one case where real source-over and item 15's lerp coincide exactly.
#[test]
fn a_partially_faded_pixel_equals_the_hand_computed_source_over_lerp() {
    let config = InpainterConfig::default();
    let original = flat_rgb(PAGE, BASE);
    let combined = transparent(PAGE);
    let raw = middle_raw();
    let stub = StubInpainter::flat(0, 0, 0);

    let output = run(
        &middle_region(),
        &raw,
        &combined,
        None,
        &config,
        &stub,
        &original,
    );
    let expected = final_mask_oracle(&middle_region(), &raw, &combined, &config);

    let mut checked = 0_usize;
    for y in 0..PAGE.1 {
        for x in 0..PAGE.0 {
            let alpha = expected.get_pixel(x, y).0[0];
            if !(1..=254).contains(&alpha) {
                continue;
            }
            let a = f64::from(alpha) / 255.0;
            let want = (f64::from(BASE[0]) * (1.0 - a)).round() as u8;
            assert_eq!(
                output.clean_inpaint.get_pixel(x, y).0[0],
                want,
                "at ({x}, {y}) with alpha {alpha}"
            );
            checked += 1;
            if checked == 40 {
                return;
            }
        }
    }
    panic!("no partially faded pixel was found, so nothing was checked");
}

/// §16.38 item 3(g) / upstream `inpainting.py:131-134`'s `if boxes_to_inpaint:` guard: with nothing
/// eligible the model is not called at all.
///
/// Asserted as a **pair on one fixture**, because `0` on its own is what a completely broken stage
/// would also report. The only difference between the two halves is the region's `failed` flag.
#[test]
fn nothing_eligible_means_zero_model_calls_and_flipping_one_flag_means_one() {
    let config = InpainterConfig::default();
    let original = flat_rgb(PAGE, BASE);
    let combined = transparent(PAGE);
    let raw = middle_raw();

    let ineligible = vec![region(Rect::new(290, 290, 310, 310), 1.0, false, Some(2))];
    let skipped = StubInpainter::flat(255, 0, 0);
    let output = run(
        &ineligible,
        &raw,
        &combined,
        None,
        &config,
        &skipped,
        &original,
    );
    assert_eq!(skipped.calls(), 0);
    assert_eq!(output.tiles_inferred, 0);
    assert!(output.growths.is_empty());
    assert_eq!(
        output.clean_inpaint,
        {
            let mut base = RgbaImage::from_pixel(PAGE.0, PAGE.1, Rgba([200, 200, 200, 255]));
            pc_inpaint::compose::alpha_composite_over(&mut base, &combined, (0, 0));
            base
        },
        "with nothing eligible the cleaned output is the rebuilt base and nothing else"
    );

    let called = StubInpainter::flat(255, 0, 0);
    let output = run(
        &middle_region(),
        &raw,
        &combined,
        None,
        &config,
        &called,
        &original,
    );
    assert_eq!(called.calls(), 1);
    assert_eq!(
        output.growths,
        vec![13],
        "growth = 7 + int(30 * 0.2) = 13 (§16.38 item 3(e))"
    );
}

/// §16.38 item 3(h): *"`cleaned_image` is rebuilt from scratch inside `inpaint_page` —
/// `original_image.convert('RGBA')` (`:149`), the fill mask pasted over it (`:151-153`), then the noise
/// mask when denoising ran (`:155-157`) — so **`_clean_inpaint.png` is not derived from
/// `_clean.png`**."*
///
/// Both layers are checked at a pixel far from anything inpainted, and each is checked against the
/// same pixel with that layer absent — so "the layer was composited" is asserted, not "the pixel
/// happens to be that colour".
#[test]
fn the_cleaned_base_is_rebuilt_from_the_original_plus_the_fill_mask_and_the_noise_mask() {
    let config = InpainterConfig::default();
    let original = flat_rgb(PAGE, BASE);
    let raw = middle_raw();
    let regions = middle_region();

    let combined = rgba_rect(PAGE, Rect::new(50, 50, 100, 100), [0, 255, 0]);
    let noise = rgba_rect(PAGE, Rect::new(400, 50, 450, 100), [0, 0, 255]);

    let with_layers = run(
        &regions,
        &raw,
        &combined,
        Some(&noise),
        &config,
        &StubInpainter::flat(255, 0, 0),
        &original,
    );
    let without_layers = run(
        &regions,
        &raw,
        &transparent(PAGE),
        None,
        &config,
        &StubInpainter::flat(255, 0, 0),
        &original,
    );

    assert_eq!(
        *with_layers.clean_inpaint.get_pixel(75, 75),
        Rgba([0, 255, 0, 255]),
        "the combined fill mask must be composited onto the rebuilt base"
    );
    assert_eq!(
        *without_layers.clean_inpaint.get_pixel(75, 75),
        Rgba([200, 200, 200, 255]),
        "and without it that pixel is the original — so the assertion above is not a coincidence"
    );
    assert_eq!(
        *with_layers.clean_inpaint.get_pixel(425, 75),
        Rgba([0, 0, 255, 255]),
        "the noise mask must be composited next (upstream :155-157)"
    );
    assert_eq!(
        *without_layers.clean_inpaint.get_pixel(425, 75),
        Rgba([200, 200, 200, 255])
    );
    assert!(
        with_layers
            .clean_inpaint
            .pixels()
            .all(|pixel| pixel.0[3] == 255),
        "upstream flattens alpha to 255 at :171"
    );
}

/// §16.38 item 5(e): "write-back touches only owned pixels". Two regions, two windows that **overlap**
/// (`x ∈ [488, 512)`), and a stub whose colour depends on the call index — so the composed page states
/// which window wrote each pixel and a mixed-up coordinate mapping or a whole-tile write is visible.
///
/// The two expected colours come from `StubInpainter::per_call_color`, i.e. from the stub's own
/// contract, not from the output.
#[test]
fn each_window_writes_only_its_own_fill_and_leaves_the_shared_area_alone() {
    let page = (1000_u32, 600_u32);
    let config = InpainterConfig::default();
    let original = flat_rgb(page, BASE);
    let combined = transparent(page);
    let mut raw = GrayImage::new(page.0, page.1);
    for (x0, y0) in [(105_u32, 105_u32), (805, 105)] {
        for y in y0..y0 + 10 {
            for x in x0..x0 + 10 {
                raw.put_pixel(x, y, image::Luma([255]));
            }
        }
    }
    let regions = vec![
        region(Rect::new(100, 100, 120, 120), 30.0, true, Some(2)),
        region(Rect::new(800, 100, 820, 120), 30.0, true, Some(2)),
    ];
    let stub = StubInpainter::new(StubMode::PerCall {
        base: 10,
        step: 100,
    });

    let output = run(&regions, &raw, &combined, None, &config, &stub, &original);

    assert_eq!(stub.calls(), 2, "two merged rectangles, one window each");
    let first = StubInpainter::per_call_color(10, 100, 0);
    let second = StubInpainter::per_call_color(10, 100, 1);
    assert_eq!(
        (first.0[0], second.0[0]),
        (10, 110),
        "the stub's own contract"
    );

    assert_eq!(
        *output.clean_inpaint.get_pixel(110, 110),
        Rgba([first.0[0], first.0[1], first.0[2], 255]),
        "the first window owns the first region's fill"
    );
    assert_eq!(
        *output.clean_inpaint.get_pixel(810, 110),
        Rgba([second.0[0], second.0[1], second.0[2], 255]),
        "the second window owns the second region's fill"
    );
    assert_eq!(
        *output.clean_inpaint.get_pixel(500, 300),
        Rgba([BASE[0], BASE[1], BASE[2], 255]),
        "(500,300) lies inside BOTH windows and inside NO fill region — writing whole tiles would \
         paint it"
    );
    assert_eq!(output.growths, vec![13, 13]);
}

/// §16.38 item 3(h) / upstream `:168`, `:174`: `_inpainting.png` is the inpainted RGB with `final_mask`
/// as its alpha.
///
/// The second assertion records a consequence of `DEVIATION(24)` rather than a parity claim: outside
/// the windows the model never ran, so those pixels carry the **original** RGB under alpha 0.
/// Upstream's own `_inpainting.png` has model output there because it ran the model page-wide. The
/// difference is invisible in `_clean_inpaint.png`, which is why it is asserted here and nowhere else.
#[test]
fn the_inpainting_artifact_carries_the_final_mask_as_alpha_and_the_original_outside_the_windows() {
    let config = InpainterConfig::default();
    let original = flat_rgb(PAGE, BASE);
    let combined = transparent(PAGE);
    let raw = middle_raw();
    let stub = StubInpainter::flat(255, 0, 0);

    let output = run(
        &middle_region(),
        &raw,
        &combined,
        None,
        &config,
        &stub,
        &original,
    );
    let expected = final_mask_oracle(&middle_region(), &raw, &combined, &config);

    assert_eq!(output.inpainting.dimensions(), PAGE);
    for y in 0..PAGE.1 {
        for x in 0..PAGE.0 {
            assert_eq!(
                output.inpainting.get_pixel(x, y).0[3],
                expected.get_pixel(x, y).0[0],
                "alpha at ({x}, {y}) must be `faded_fill AND isolation`"
            );
        }
    }
    assert_eq!(
        *output.inpainting.get_pixel(0, 0),
        Rgba([BASE[0], BASE[1], BASE[2], 0]),
        "outside every 512x512 window the model never ran, so the original shows through at alpha 0"
    );
    assert_eq!(
        output.inpainting.get_pixel(300, 300).0[..3],
        [255, 0, 0],
        "deep inside the fill the model's output is what is kept"
    );

    // §16.38 item 5(e): "write-back touches only owned pixels, masked by `faded_fill AND
    // isolation`". `(100,100)` is INSIDE the single window `(44,44,556,556)` and outside the write
    // region, so the model's output for it exists and must nonetheless be discarded. This is the
    // only place the write-back's own filter is observable: `_clean_inpaint.png` cannot see it,
    // because the page assembly's alpha composite already skips alpha-0 pixels. Dropping the filter
    // turns exactly this assertion red.
    assert_eq!(
        expected.get_pixel(100, 100).0[0],
        0,
        "premise: (100,100) carries no write-region alpha"
    );
    assert_eq!(
        *output.inpainting.get_pixel(100, 100),
        Rgba([BASE[0], BASE[1], BASE[2], 0]),
        "inside the window but outside `faded_fill AND isolation`, the write-back must not write"
    );
}

/// §16.38 item 5(e), the clause that ownership exists for: *"Windows from different merged
/// rectangles may overlap even though the rectangles do not, so ownership is assigned rather than
/// left to write order … a fill pixel belongs to the **first** window in that order."*
///
/// The geometry is built so the two are distinguishable, which the fixtures in
/// `each_window_writes_only_its_own_fill_and_leaves_the_shared_area_alone` are not: region B's fill
/// lies inside **both** centred windows, and window A comes first in `(y1, x1)` order. So under
/// item 5(e) region B's pixels carry the FIRST call's colour, and under a plain last-write-wins loop
/// they would carry the second's. The two expected colours differ by 100 levels.
///
/// Note the consequence the spec accepts and this test therefore pins: window B is still inferred
/// (it intersects the write region, so item 5(c) does not drop it) and then owns nothing.
#[test]
fn when_two_windows_overlap_over_a_filled_area_the_first_in_reading_order_wins() {
    let page = (700_u32, 700_u32);
    let config = InpainterConfig::default();
    let original = flat_rgb(page, BASE);
    let combined = transparent(page);
    let mut raw = GrayImage::new(page.0, page.1);
    for (x0, y0) in [(120_u32, 120_u32), (420, 420)] {
        for y in y0..y0 + 10 {
            for x in x0..x0 + 10 {
                raw.put_pixel(x, y, image::Luma([255]));
            }
        }
    }
    let regions = vec![
        region(Rect::new(115, 115, 135, 135), 30.0, true, Some(2)), // A: window (0,0,512,512)
        region(Rect::new(415, 415, 435, 435), 30.0, true, Some(2)), // B: window (169,169,681,681)
    ];
    let stub = StubInpainter::new(StubMode::PerCall {
        base: 10,
        step: 100,
    });

    let output = run(&regions, &raw, &combined, None, &config, &stub, &original);

    assert_eq!(stub.calls(), 2, "both windows are inferred");
    let first = StubInpainter::per_call_color(10, 100, 0).0[0];
    let second = StubInpainter::per_call_color(10, 100, 1).0[0];
    assert_eq!((first, second), (10, 110), "the stub's own contract");

    // Premise: region B's centre really is inside both windows, or this test proves nothing.
    let inside = |window: (i32, i32, i32, i32), p: (i32, i32)| {
        p.0 >= window.0 && p.0 < window.2 && p.1 >= window.1 && p.1 < window.3
    };
    assert!(
        inside((0, 0, 512, 512), (425, 425)),
        "window A covers B's fill"
    );
    assert!(
        inside((169, 169, 681, 681), (425, 425)),
        "and so does window B"
    );

    assert_eq!(
        output.clean_inpaint.get_pixel(425, 425).0[0],
        first,
        "the earlier window in (y1, x1) order owns the overlap; last-write-wins would give {second}"
    );
    assert_eq!(
        output.clean_inpaint.get_pixel(125, 125).0[0],
        first,
        "region A's own fill is in window A only"
    );
}

/// A mask-frame mismatch between `_raw_mask.png` and the combined mask is a caller bug, not a page
/// defect: both are written in the same scaled frame (§16.38 item 3(a)). Rejected up front so a
/// misaligned crop cannot be mistaken for a masking failure.
#[test]
fn mismatched_mask_frames_and_a_wrongly_sized_noise_mask_are_invalid_input() {
    let config = InpainterConfig::default();
    let original = flat_rgb(PAGE, BASE);
    let stub = StubInpainter::flat(0, 0, 0);

    let mismatch = inpaint_page(
        PageInput {
            original: &original,
            raw_mask: &GrayImage::new(300, 300),
            combined_mask: &transparent(PAGE),
            noise_mask: None,
            regions: &middle_region(),
            min_mask_thickness: 0,
            config: &config,
        },
        &stub,
    );
    assert!(matches!(mismatch, Err(StageError::InvalidInput(_))));

    let bad_noise = inpaint_page(
        PageInput {
            original: &original,
            raw_mask: &middle_raw(),
            combined_mask: &transparent(PAGE),
            noise_mask: Some(&transparent((10, 10))),
            regions: &middle_region(),
            min_mask_thickness: 0,
            config: &config,
        },
        &stub,
    );
    assert!(matches!(bad_noise, Err(StageError::InvalidInput(_))));
}

/// §16.38 item 3(f) / upstream `:127-128`: when the page was scaled, both page masks are NEAREST-resized
/// to the original size before the model runs. The mask frame here is half the original on each axis,
/// so a fill blob must land at twice its mask-frame coordinates — asserted through where the composed
/// page actually changed.
#[test]
fn a_scaled_mask_frame_is_carried_to_the_original_resolution_before_the_tile_loop() {
    let config = radii(2, 2, 0.0, 1, 0);
    let frame = (300_u32, 300_u32);
    let original = flat_rgb(PAGE, BASE);
    let raw = gray_rect(frame, Rect::new(148, 148, 152, 152));
    let combined = transparent(frame);
    let regions = vec![region(Rect::new(145, 145, 155, 155), 0.0, true, Some(1))];
    let stub = StubInpainter::flat(0, 0, 0);

    let output = run(&regions, &raw, &combined, None, &config, &stub, &original);

    let mut changed_bbox: Option<(u32, u32, u32, u32)> = None;
    for y in 0..PAGE.1 {
        for x in 0..PAGE.0 {
            if output.clean_inpaint.get_pixel(x, y).0[0] == BASE[0] {
                continue;
            }
            changed_bbox = Some(match changed_bbox {
                None => (x, y, x + 1, y + 1),
                Some((x1, y1, x2, y2)) => (x1.min(x), y1.min(y), x2.max(x + 1), y2.max(y + 1)),
            });
        }
    }

    let (x1, y1, x2, y2) = changed_bbox.expect("something must have been inpainted");
    // Hand-derived. In the 300-wide mask frame: the 4x4 raw blob at (148..152) is placed at local
    // (6..10) of the 16x16 padded box (142,142,158,158) and dilated by `growth = 2`, whose kernel
    // reaches 2 further on each axis, giving local (4..12) — frame (146..154). `fade_radius = 0`
    // makes `faded` the hard mask, and the isolation mask (fill grown by 1) only ever contains it,
    // so `final_mask` is exactly that fill. The NEAREST resize to 600 doubles it: destination `x` is
    // set iff `floor(x/2) ∈ [146, 154)`, i.e. `x ∈ [292, 308)`.
    assert_eq!(
        (x1, y1, x2, y2),
        (292, 292, 308, 308),
        "the mask frame is half the original, so the fill lands at twice its frame coordinates"
    );
}
