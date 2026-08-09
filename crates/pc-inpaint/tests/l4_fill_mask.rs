//! L4 — spec §16.38 items 3(d), 3(f) and 10: the per-region padded fill mask, the two page-global
//! masks, the near-edge frame question, and the NEAREST resize.

mod common;

use common::{gray_rect, gray_with, pixel_set, radii, region, rgba_rect, set_pixels, transparent};
use pc_core::Rect;
use pc_imageops::morph::{dilate, kernel};
use pc_imageops::{BinaryMask, PIL_BINARY_THRESHOLD};
use pc_inpaint::{
    compose_page_masks, padded_region, resize_nearest_binary, scale_rect_to, select_regions,
};

/// Build the page masks for a whole set of regions, the way `inpaint_page` does, but without any
/// tiling or compositing — so a failure lands on the mask synthesis and nowhere else.
fn page_masks(
    regions: &[pc_core::MaskRegionStats],
    raw: &image::GrayImage,
    combined: &image::RgbaImage,
    min_mask_thickness: u32,
    config: &pc_config::InpainterConfig,
) -> pc_inpaint::PageMasks {
    let canvas = combined.dimensions();
    let raw_binary = BinaryMask::from_gray_threshold(raw, PIL_BINARY_THRESHOLD);
    let combined_binary = pc_inpaint::combined_fill_binary(combined);
    let padded: Vec<_> = select_regions(regions, config)
        .iter()
        .map(|region| {
            padded_region(
                region,
                &raw_binary,
                &combined_binary,
                min_mask_thickness,
                config,
                canvas,
            )
        })
        .collect();
    compose_page_masks(&padded, canvas, config.inpainting_isolation_radius)
}

/// §16.38 item 3(d) / upstream `inpainting.py:93-97`: a **failed** region samples the RAW mask,
/// cropped to the box, and is grown by `masker.min_mask_thickness` *before* the radius growth.
///
/// Hand-derived on a 9×9 page with every radius at zero, so the only growth left is
/// `min_mask_thickness = 1`. `kernel(1)` has diameter 3 with its four corners zeroed (§16.9 item 5),
/// i.e. a plus, so a single raw pixel at `(4,4)` must yield exactly five fill pixels. The combined
/// mask is fully transparent, so a port that read the wrong source would produce an **empty** mask —
/// which the exact set assertion catches, and a count-only assertion would too, but only this one
/// catches a one-pixel misplacement.
#[test]
fn a_failed_region_samples_the_raw_mask_and_is_pre_grown_by_min_mask_thickness() {
    let config = radii(0, 0, 0.0, 0, 0);
    let regions = vec![region(Rect::new(3, 3, 6, 6), 0.0, true, Some(1))];
    let raw = gray_with((9, 9), &[(4, 4)]);
    let combined = transparent((9, 9));

    let masks = page_masks(&regions, &raw, &combined, 1, &config);

    assert_eq!(
        set_pixels(&masks.fill),
        pixel_set(&[(4, 3), (3, 4), (4, 4), (5, 4), (4, 5)]),
        "kernel(1) is a plus (diameter 3 with the corners zeroed, §16.9 item 5)"
    );
    assert_eq!(masks.fill.count_set(), 5);
    assert_eq!(
        set_pixels(&masks.isolation),
        set_pixels(&masks.fill),
        "isolation radius 0 leaves the isolation mask equal to the fill mask"
    );
}

/// §16.38 item 3(d): failed regions sample the **raw** mask (`:93-97`), poorly-fitted regions sample
/// the **combined** fill mask (`:99-101`). Two of them, in one page, with **disjoint content in the
/// two sources** — so swapping the sources produces an empty mask and swapping the *regions*
/// produces two empty crops. Neither survives an exact-set assertion.
#[test]
fn the_two_fill_sources_are_not_interchangeable() {
    let config = radii(0, 0, 0.0, 0, 0);
    let regions = vec![
        region(Rect::new(4, 4, 7, 7), 0.0, true, Some(1)), // failed  -> raw
        region(Rect::new(13, 13, 16, 16), 20.0, false, Some(0)), // poorly -> combined
    ];
    let raw = gray_with((20, 20), &[(5, 5)]);
    let combined = rgba_rect((20, 20), Rect::new(14, 14, 15, 15), [0, 0, 0]);

    let masks = page_masks(&regions, &raw, &combined, 0, &config);

    assert_eq!(
        set_pixels(&masks.fill),
        pixel_set(&[(5, 5), (14, 14)]),
        "one pixel from the raw mask (failed row) and one from the combined mask (poorly-fitted row)"
    );
    assert_eq!(masks.fill.count_set(), 2);
}

/// §16.38 item 3(d): the `min_mask_thickness` growth of a failed region happens in the **box-sized**
/// crop (upstream grows the crop at `:96`, before the paste at `:116`), so it is clipped at the box
/// edge and cannot bleed outside the region's own box.
///
/// Hand-derived: a raw pixel at the box's top-left corner, grown by `kernel(1)`'s plus, loses the
/// two arms that fall outside the box.
#[test]
fn the_min_mask_thickness_growth_is_clipped_at_the_box_edge_not_at_the_page_edge() {
    let config = radii(0, 0, 0.0, 0, 0);
    let regions = vec![region(Rect::new(5, 5, 8, 8), 0.0, true, Some(1))];
    let raw = gray_with((20, 20), &[(5, 5)]);
    let combined = transparent((20, 20));

    let masks = page_masks(&regions, &raw, &combined, 1, &config);

    assert_eq!(
        set_pixels(&masks.fill),
        pixel_set(&[(5, 5), (6, 5), (5, 6)]),
        "the plus's left and upper arms fall outside the box (5,5,8,8) and are clipped there — \
         page coordinates (4,5) and (5,4) are inside the PAGE and must still be clear"
    );
}

/// §16.38 item 3(f) / upstream `:137-144`: the isolation mask is each padded mask grown **again** by
/// `inpainting_isolation_radius`.
///
/// With a single region the page-global compose and a page-global dilation coincide, so the
/// expectation is computed by an independent route (dilating the *page* fill mask) and the
/// hard-coded `21` — `kernel(2)`'s cell count, a 5×5 square with its four corners zeroed — keeps the
/// comparison from passing on two empty masks.
#[test]
fn the_isolation_mask_is_the_fill_mask_grown_again_by_the_isolation_radius() {
    let config = radii(0, 0, 0.0, 2, 0);
    let regions = vec![region(Rect::new(19, 19, 22, 22), 0.0, true, Some(1))];
    let raw = gray_with((40, 40), &[(20, 20)]);
    let combined = transparent((40, 40));

    let masks = page_masks(&regions, &raw, &combined, 0, &config);

    assert_eq!(masks.fill.count_set(), 1, "one raw pixel, no growth");
    assert_eq!(
        masks.isolation.count_set(),
        21,
        "kernel(2) is a 5x5 square with the four corners zeroed = 21 cells"
    );
    assert_eq!(
        set_pixels(&masks.isolation),
        set_pixels(&dilate(&masks.fill, &kernel(2))),
        "for a single region the isolation mask is the fill mask dilated by kernel(isolation)"
    );
    assert!(
        set_pixels(&masks.fill).is_subset(&set_pixels(&masks.isolation)),
        "the isolation mask must contain the fill mask"
    );
}

/// §16.38 item 3(f)'s explicit obligation: *"A direct absolute-frame port is not bit-equivalent, and
/// that is a real subtlety rather than a note … Whichever frame L4 chooses, it must choose
/// deliberately and pin the near-edge case."*
///
/// `fill::padded_region` builds each mask in the **padded box's own frame**. This test pins that
/// against an **independent oracle written in upstream's frame**: a canvas-sized `"1"` buffer with
/// the crop pasted at `(box.x1 - box_padded.x1, box.y1 - box_padded.y1)`, grown by upstream's own
/// `grow_mask` shape — `np.pad(mode="edge")` by `size * 2`, dilate, crop back
/// (`image_ops.py:804-812`) — then pasted at the padded box's origin with the overflow dropped
/// (`:124`). The oracle is a transcription of upstream, not a call into the code under test.
///
/// Both frame-sensitive corners are exercised: a region flush against the top-left, where the pad
/// clamps and the content touches the buffer boundary, and one flush against the bottom-right, where
/// the two frames differ most.
#[test]
fn growth_at_the_frame_edge_matches_the_replicate_padded_oracle() {
    let canvas = (30_u32, 30_u32);
    let config = radii(3, 3, 0.0, 2, 0);
    let regions = vec![
        region(Rect::new(0, 0, 3, 3), 0.0, true, Some(1)),
        region(Rect::new(27, 27, 30, 30), 0.0, true, Some(1)),
    ];
    let raw = gray_with(canvas, &[(0, 0), (29, 29)]);
    let combined = transparent(canvas);

    let masks = page_masks(&regions, &raw, &combined, 0, &config);

    // ---- the oracle, in upstream's canvas-sized frame ----
    let raw_binary = BinaryMask::from_gray_threshold(&raw, PIL_BINARY_THRESHOLD);
    let mut oracle = BinaryMask::new(canvas.0, canvas.1);
    for region in &regions {
        let growth = 3_u32; // min 3, multiplier 0 -> growth 3, exactly
        let padded = region
            .rect
            .pad((growth + config.inpainting_isolation_radius) as i32, canvas);
        let box_w = region.rect.width() as u32;
        let box_h = region.rect.height() as u32;
        let crop = raw_binary.crop_into(region.rect, (box_w, box_h), (0, 0));

        // Upstream: `mask_padded = Image.new("1", mask_image.size, 0)` then paste at the
        // padded-box-relative offset.
        let offset = (region.rect.x1 - padded.x1, region.rect.y1 - padded.y1);
        let mut buffer = BinaryMask::new(canvas.0, canvas.1);
        for y in 0..box_h {
            for x in 0..box_w {
                if !crop.get(x, y) {
                    continue;
                }
                let (tx, ty) = (offset.0 + x as i32, offset.1 + y as i32);
                if tx >= 0 && ty >= 0 && (tx as u32) < canvas.0 && (ty as u32) < canvas.1 {
                    buffer.set(tx as u32, ty as u32, true);
                }
            }
        }

        // Upstream `grow_mask`: pad `mode="edge"` by `size * 2`, convolve, crop back.
        let pad = growth * 2;
        let padded_buffer = pad_replicate(&buffer, pad);
        let grown = center_crop(&dilate(&padded_buffer, &kernel(growth)), pad);

        // Upstream `:124`: paste the canvas-sized buffer at the padded box's origin, overflow
        // dropped by PIL's paste.
        for y in 0..canvas.1 {
            for x in 0..canvas.0 {
                if !grown.get(x, y) {
                    continue;
                }
                let (tx, ty) = (padded.x1 + x as i32, padded.y1 + y as i32);
                if tx >= 0 && ty >= 0 && (tx as u32) < canvas.0 && (ty as u32) < canvas.1 {
                    oracle.set(tx as u32, ty as u32, true);
                }
            }
        }
    }

    assert!(
        oracle.count_set() > 1,
        "the oracle must actually have grown something, or this comparison proves nothing"
    );
    assert_eq!(
        set_pixels(&masks.fill),
        set_pixels(&oracle),
        "the padded-box frame must agree with upstream's canvas frame at both clamped corners"
    );
}

fn pad_replicate(mask: &BinaryMask, pad: u32) -> BinaryMask {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return BinaryMask::new(width + pad * 2, height + pad * 2);
    }
    BinaryMask::from_fn(width + pad * 2, height + pad * 2, |x, y| {
        let source_x = x.saturating_sub(pad).min(width - 1);
        let source_y = y.saturating_sub(pad).min(height - 1);
        mask.get(source_x, source_y)
    })
}

fn center_crop(mask: &BinaryMask, pad: u32) -> BinaryMask {
    let (width, height) = mask.dimensions();
    BinaryMask::from_fn(width - pad * 2, height - pad * 2, |x, y| {
        mask.get(x + pad, y + pad)
    })
}

/// §16.38 item 3(f) / upstream `:127-128`: the page masks are NEAREST-resized to the original size
/// when the page was scaled. The formula is §16.9 item 13's `src = floor(dst * src_len / dst_len)`,
/// and the expected pixel set is worked out by hand: source `(1,2)` in a 4×4 frame maps to the four
/// destination pixels `x ∈ {2,3}, y ∈ {4,5}` in an 8×8 frame.
#[test]
fn the_page_masks_are_nearest_resized_with_the_pinned_floor_formula() {
    let mut mask = BinaryMask::new(4, 4);
    mask.set(1, 2, true);

    let resized = resize_nearest_binary(&mask, (8, 8));

    assert_eq!(
        set_pixels(&resized),
        pixel_set(&[(2, 4), (3, 4), (2, 5), (3, 5)])
    );
    assert_eq!(resized.count_set(), 4);
    assert_eq!(
        resize_nearest_binary(&mask, (4, 4)),
        mask,
        "identity when the sizes already match — the unscaled page pays nothing"
    );
}

/// The padded boxes must survive that same resize, or a resized fill pixel could land outside every
/// merged rectangle and end up **unowned**, which §16.38 item 5(e)'s "exactly once" forbids.
///
/// The **non-integer** ratio is the load-bearing row, and it is here because an integer one cannot
/// discriminate: at exactly 2× a truncating `Rect::scale` and the ceiling mapping agree, so a test
/// using only 4 → 8 passes for a wrong implementation. At 3 → 8 they disagree. Source pixel `x = 1`
/// resizes to the destination pixels satisfying `floor(x * 3 / 8) == 1`, i.e. `x ∈ {3, 4, 5}`;
/// `ceil(1*8/3) = 3` and `ceil(2*8/3) = 6` give `[3, 6)` exactly, while truncation gives `[2, 5)` —
/// which both admits an unfilled pixel and, worse, **excludes** the fill pixel at `x = 5`.
#[test]
fn a_padded_rect_carried_through_the_resize_still_contains_every_resized_fill_pixel() {
    let mut mask = BinaryMask::new(4, 4);
    mask.set(1, 2, true);
    let rect = Rect::new(1, 2, 2, 3);

    assert_eq!(
        scale_rect_to(rect, (4, 4), (8, 8)),
        Rect::new(2, 4, 4, 6),
        "ceiling on BOTH ends"
    );
    assert_eq!(
        resize_nearest_binary(&mask, (8, 8)).bbox(),
        Some(Rect::new(2, 4, 4, 6))
    );
    assert_eq!(
        scale_rect_to(rect, (8, 8), (8, 8)),
        rect,
        "identity when the frames match"
    );

    // The discriminating row: a 3-wide frame resized to 8.
    let mut narrow = BinaryMask::new(3, 3);
    narrow.set(1, 1, true);
    let narrow_rect = Rect::new(1, 1, 2, 2);
    assert_eq!(
        scale_rect_to(narrow_rect, (3, 3), (8, 8)),
        Rect::new(3, 3, 6, 6),
        "ceil(1*8/3) = 3 and ceil(2*8/3) = 6; truncation would say (2,2,5,5)"
    );
    assert_eq!(
        resize_nearest_binary(&narrow, (8, 8)).bbox(),
        Some(Rect::new(3, 3, 6, 6)),
        "and that is exactly where the resized fill pixel lands — a truncating scale would leave \
         (5,5) outside every merged rectangle, and therefore unowned"
    );
}

/// Stated as a property over a sweep rather than at a single ratio, because the containment above is
/// the whole reason `scale_rect_to` exists: for every frame pair tried, every set pixel of the
/// resized mask must lie inside the resized rect.
#[test]
fn the_resized_rect_contains_the_resized_mask_at_every_ratio_tried() {
    let frames = [(3_u32, 3_u32), (4, 4), (7, 5), (16, 16), (100, 60)];
    let mut checked = 0_usize;
    for from in frames {
        for to in frames {
            let mut mask = BinaryMask::new(from.0, from.1);
            mask.set(1, 1, true);
            let rect = Rect::new(1, 1, 2, 2);
            let scaled = scale_rect_to(rect, from, to);
            for (x, y) in set_pixels(&resize_nearest_binary(&mask, to)) {
                assert!(
                    (x as i32) >= scaled.x1
                        && (x as i32) < scaled.x2
                        && (y as i32) >= scaled.y1
                        && (y as i32) < scaled.y2,
                    "{from:?} -> {to:?}: resized pixel ({x}, {y}) fell outside {scaled:?}"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 25,
        "the sweep must actually have examined pixels; got {checked}"
    );
}

/// §16.38 item 10, asserted rather than only commented: for a **failed** region the filled area is a
/// function of `_raw_mask.png`, which v1 writes with `MaskRefineMode::Simple` instead of upstream's
/// `refine_mask` (§14 item 12; cookbook rule 7 measures the agreement at IoU 0.258). So changing the
/// raw mask changes the fill for a failed region and changing the combined mask does not — which is
/// the mechanism by which `DEVIATION(12)` reaches this stage.
#[test]
fn a_failed_regions_fill_tracks_the_raw_mask_and_ignores_the_combined_mask() {
    let config = radii(0, 0, 0.0, 0, 0);
    let regions = vec![region(Rect::new(10, 10, 20, 20), 0.0, true, Some(1))];
    let combined_a = transparent((40, 40));
    let combined_b = rgba_rect((40, 40), Rect::new(10, 10, 20, 20), [255, 255, 255]);

    let raw_small = gray_rect((40, 40), Rect::new(12, 12, 14, 14));
    let raw_large = gray_rect((40, 40), Rect::new(12, 12, 18, 18));

    let small = page_masks(&regions, &raw_small, &combined_a, 0, &config);
    let large = page_masks(&regions, &raw_large, &combined_a, 0, &config);
    let combined_changed = page_masks(&regions, &raw_small, &combined_b, 0, &config);

    assert_eq!(small.fill.count_set(), 4, "a 2x2 raw square");
    assert_eq!(large.fill.count_set(), 36, "a 6x6 raw square");
    assert_eq!(
        set_pixels(&combined_changed.fill),
        set_pixels(&small.fill),
        "a fully covered combined mask must not change a FAILED region's fill"
    );
}
