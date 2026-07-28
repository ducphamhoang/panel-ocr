//! Tasks M5/M6 -- spec §10.3 steps 4-5, §10.7(A)12, §10.7(A)13, §10.7(A)14, §15.3,
//! §16.9 items 13-16, 19. FROZEN.

mod common;

use common::{
    disk_dests, memory_input, page, simple_page, text_raw_mask, uniform_base, uniform_base_rgb,
    PAGE_SIZE,
};
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage, RgbaImage};
use pc_config::MaskerConfig;
use pc_core::{Rect, Stage, StageError};
use pc_imageops::BinaryMask;
use pc_mask::combine::{
    alpha_composite_over, blend_channel, build_combined_mask, cleaned_image, is_achromatic,
    mask_layer_rgba, mask_overlay, resize_nearest_rgba, text_layer,
};
use pc_mask::fit::Fitment;
use pc_mask::{MaskInput, MaskStage};

/// A fitment with an all-set `size`-square mask at `coords`, filled with `color`.
fn solid_fitment(size: u32, coords: (i32, i32), color: [u8; 3]) -> Fitment {
    Fitment {
        mask: Some(BinaryMask::from_fn(size, size, |_, _| true)),
        median_color: color,
        coords,
        std_deviation: 0.0,
        candidate_index: 0,
        thickness: None,
        masking_rect: Rect::new(
            coords.0,
            coords.1,
            coords.0 + size as i32,
            coords.1 + size as i32,
        ),
    }
}

fn rgba_at(image: &RgbaImage, x: u32, y: u32) -> [u8; 4] {
    image.get_pixel(x, y).0
}

// ------------------------------------------------------------ composition (§10.7(A)12)

#[test]
fn a12_overlapping_fitments_composite_in_region_order() {
    // spec §10.7(A)12 / §10.3 step 4: layers are alpha-composited in region order, so
    // the LATER region wins in the overlap, and alpha is exactly 0 or 255 everywhere.
    let first = solid_fitment(4, (0, 0), [10, 20, 30]);
    let second = solid_fitment(4, (2, 2), [200, 100, 50]);

    let combined = build_combined_mask(&[first, second], (10, 10));

    assert_eq!(combined.dimensions(), (10, 10));
    assert_eq!(rgba_at(&combined, 0, 0), [10, 20, 30, 255]);
    assert_eq!(
        rgba_at(&combined, 3, 3),
        [200, 100, 50, 255],
        "later region wins"
    );
    assert_eq!(rgba_at(&combined, 5, 5), [200, 100, 50, 255]);
    assert_eq!(rgba_at(&combined, 9, 9), [0, 0, 0, 0]);
    assert!(
        combined.pixels().all(|pixel| matches!(pixel.0[3], 0 | 255)),
        "combined mask alpha is binary"
    );
}

#[test]
fn a12_failed_fitments_contribute_nothing() {
    // spec §10.3 step 10: a fitment with `mask: None` is still reported in MaskData but
    // paints nothing.
    let mut failed = solid_fitment(4, (0, 0), [10, 20, 30]);
    failed.mask = None;

    let combined = build_combined_mask(&[failed], (6, 6));

    assert!(combined.pixels().all(|pixel| pixel.0[3] == 0));
}

#[test]
fn a12_cleaned_differs_from_the_canvas_only_inside_the_chosen_masks() {
    // spec §10.7(A)12: pixel-set equality between "what changed" and "the union of the
    // chosen masks". Both fills differ from the black canvas, so every masked pixel does
    // change.
    let canvas = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, Rgb([0, 0, 0])));
    let fitments = [
        solid_fitment(4, (0, 0), [10, 20, 30]),
        solid_fitment(4, (5, 5), [200, 100, 50]),
    ];
    let combined = build_combined_mask(&fitments, (10, 10));

    let cleaned = cleaned_image(&canvas, &combined, false).to_rgb8();

    let mut expected = pc_testkit::PixelSet::new((10, 10));
    for (x, y) in (0..10).flat_map(|y| (0..10).map(move |x| (x, y))) {
        if (x < 4 && y < 4) || ((5..9).contains(&x) && (5..9).contains(&y)) {
            expected.insert((x, y));
        }
    }
    let changed = pc_testkit::metrics::diff_set_rgba(
        &DynamicImage::ImageRgb8(canvas.to_rgb8()).to_rgba8(),
        &DynamicImage::ImageRgb8(cleaned).to_rgba8(),
    );
    assert_eq!(changed, expected);
}

// ------------------------------------------------------------ scale != 1 (§10.7(A)13)

#[test]
fn a13_nearest_upscale_of_a_binary_mask_is_exactly_2x2_blocks() {
    // spec §10.7(A)13 / §16.9 item 13: src = floor(dst * src_len / dst_len).
    let mask = build_combined_mask(&[solid_fitment(2, (1, 1), [7, 7, 7])], (5, 5));

    let upscaled = resize_nearest_rgba(&mask, (10, 10));

    assert_eq!(upscaled.dimensions(), (10, 10));
    for block_y in 0..5 {
        for block_x in 0..5 {
            let reference = rgba_at(&upscaled, block_x * 2, block_y * 2);
            for (dx, dy) in [(0, 1), (1, 0), (1, 1)] {
                assert_eq!(
                    rgba_at(&upscaled, block_x * 2 + dx, block_y * 2 + dy),
                    reference,
                    "2x2 block at ({block_x}, {block_y}) is not uniform"
                );
            }
        }
    }
    assert_eq!(rgba_at(&upscaled, 2, 2), [7, 7, 7, 255]);
    assert_eq!(rgba_at(&upscaled, 1, 1), [0, 0, 0, 0]);
}

#[test]
fn a13_cleaned_image_takes_the_canvas_dimensions() {
    // spec §16.9 item 13: the mask is resized to the canvas's ACTUAL dimensions, never
    // to a size computed from `scale`.
    let canvas = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, Rgb([0, 0, 0])));
    let mask = build_combined_mask(&[solid_fitment(2, (1, 1), [255, 255, 255])], (5, 5));

    let cleaned = cleaned_image(&canvas, &mask, false);

    assert_eq!(cleaned.dimensions(), (10, 10));
    assert_eq!(cleaned.to_rgb8().get_pixel(2, 2), &Rgb([255, 255, 255]));
    assert_eq!(cleaned.to_rgb8().get_pixel(0, 0), &Rgb([0, 0, 0]));
}

// ------------------------------------------------------------ output mode (§15.3)

#[test]
fn cleaned_image_is_luma_only_when_the_canvas_is_gray_and_every_fill_is_achromatic() {
    // spec §15.3 / §16.9 item 14: the rule is applied to the in-memory value, so memory
    // mode and disk mode agree pixel-for-pixel.
    let gray = DynamicImage::ImageLuma8(GrayImage::from_pixel(4, 4, Luma([10])));
    let rgb = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([10, 10, 10])));
    let mask = build_combined_mask(&[solid_fitment(2, (0, 0), [90, 90, 90])], (4, 4));

    assert!(matches!(
        cleaned_image(&gray, &mask, true),
        DynamicImage::ImageLuma8(_)
    ));
    assert!(matches!(
        cleaned_image(&gray, &mask, false),
        DynamicImage::ImageRgb8(_)
    ));
    assert!(matches!(
        cleaned_image(&rgb, &mask, true),
        DynamicImage::ImageRgb8(_)
    ));
}

#[test]
fn is_achromatic_is_channel_equality() {
    assert!(is_achromatic([0, 0, 0]));
    assert!(is_achromatic([200, 200, 200]));
    assert!(!is_achromatic([200, 200, 201]));
}

// ------------------------------------------------------------ text layer + overlay

#[test]
fn text_layer_keeps_only_what_the_mask_covers() {
    // spec §10.3 step 4: a transparent canvas with the source pixels showing through the
    // mask -- what the mask covers IS the text.
    let canvas = DynamicImage::ImageRgb8(RgbImage::from_pixel(6, 6, Rgb([12, 34, 56])));
    let mask = build_combined_mask(&[solid_fitment(2, (1, 1), [255, 0, 0])], (6, 6));

    let text = text_layer(&canvas, &mask);

    assert_eq!(text.dimensions(), (6, 6));
    assert_eq!(rgba_at(&text, 1, 1), [12, 34, 56, 255]);
    assert_eq!(rgba_at(&text, 0, 0), [0, 0, 0, 0]);
    assert_eq!(rgba_at(&text, 5, 5), [0, 0, 0, 0]);
}

#[test]
fn mask_overlay_blends_the_debug_colour_at_its_own_alpha() {
    // spec §16.9 item 15: out = round(base*(1-a) + color*a) with a = 127/255, applied
    // only where the combined mask is opaque. Over a black base with the §6 default
    // debug_mask_color [108, 30, 240, 127] that is exactly (54, 15, 120).
    let base = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([0, 0, 0])));
    let mask = build_combined_mask(&[solid_fitment(2, (0, 0), [255, 255, 255])], (4, 4));

    let overlay = mask_overlay(&base, &mask, [108, 30, 240, 127]);

    assert_eq!(overlay.dimensions(), (4, 4));
    assert_eq!(overlay.get_pixel(0, 0), &Rgb([54, 15, 120]));
    assert_eq!(overlay.get_pixel(3, 3), &Rgb([0, 0, 0]));
}

#[test]
fn blend_channel_rounds_half_away_from_zero() {
    assert_eq!(blend_channel(0, 108, 127.0 / 255.0), 54);
    assert_eq!(blend_channel(0, 255, 1.0), 255);
    assert_eq!(blend_channel(80, 200, 0.0), 80);
}

#[test]
fn alpha_composite_over_drops_layers_falling_outside_the_canvas() {
    let mut canvas = RgbaImage::new(4, 4);
    let layer = mask_layer_rgba(&BinaryMask::from_fn(3, 3, |_, _| true), [9, 9, 9]);

    alpha_composite_over(&mut canvas, &layer, (3, 3));

    assert_eq!(rgba_at(&canvas, 3, 3), [9, 9, 9, 255]);
    assert_eq!(rgba_at(&canvas, 0, 0), [0, 0, 0, 0]);
}

// ------------------------------------------------------------ run() (§10.3 steps 0-5)

#[test]
fn run_emits_one_region_and_one_analytic_per_fitment() {
    // spec §10.3 step 5 / §16.9 item 19: MaskData.regions and analytics are 1:1, in
    // region order. The numbers are the ones hand-traced in `m4_fit.rs`.
    let output = pc_mask::run(memory_input(simple_page(), MaskerConfig::default()))
        .expect("masking succeeds");

    assert_eq!(output.mask_data.regions.len(), 1);
    let region = &output.mask_data.regions[0];
    assert_eq!(region.rect, Rect::new(70, 50, 130, 110));
    assert_eq!(region.std_deviation, 0.0);
    assert!(!region.failed);
    assert_eq!(region.thickness, None);

    assert_eq!(output.analytics.len(), 1);
    let analytic = &output.analytics[0];
    assert_eq!(
        analytic.path,
        std::path::PathBuf::from(common::ORIGINAL_PATH)
    );
    assert!(analytic.fit_found);
    assert_eq!(analytic.candidate_index, 11);
    assert_eq!(analytic.std_deviation, 0.0);
    assert_eq!(analytic.thickness, None);
}

#[test]
fn run_passes_through_the_page_identity_fields() {
    // spec §16.9 item 19.
    let input = memory_input(simple_page(), MaskerConfig::default());
    let schema_version = input.schema_version;

    let output = pc_mask::run(input).expect("masking succeeds");

    assert_eq!(output.mask_data.schema_version, schema_version);
    assert_eq!(
        output.mask_data.original_path,
        std::path::PathBuf::from(common::ORIGINAL_PATH)
    );
    assert_eq!(output.mask_data.scale, 1.0);
    assert_eq!(
        output.mask_data.base_image.path,
        Some(std::path::PathBuf::from(common::BASE_IMAGE_PATH))
    );
}

#[test]
fn run_cleans_exactly_the_text_pixels_on_a_uniform_page() {
    // End-to-end consequence of the hand-traced fit: the chosen mask is filled with the
    // background value 200, so the only pixels that change are the 40x40 text block.
    let page = simple_page();
    let base = page.base_image.load().expect("cached base").to_luma8();

    let output =
        pc_mask::run(memory_input(page, MaskerConfig::default())).expect("masking succeeds");

    let cleaned = output.cleaned.load().expect("cleaned is in memory");
    assert_eq!(cleaned.dimensions(), PAGE_SIZE);
    let changed = pc_testkit::metrics::diff_set_gray(&base, &cleaned.to_luma8());
    assert_eq!(changed.len(), 40 * 40);
    assert!(changed.contains((80, 60)));
    assert!(changed.contains((119, 99)));
    assert!(!changed.contains((79, 60)));
}

#[test]
fn run_writes_a_grayscale_cleaned_image_for_a_grayscale_page() {
    // spec §15.3 / §16.9 item 14: gray base + achromatic medians => Luma8.
    let output = pc_mask::run(memory_input(simple_page(), MaskerConfig::default()))
        .expect("masking succeeds");

    let cleaned = output.cleaned.load().expect("cleaned is in memory");
    assert!(matches!(cleaned.as_ref(), DynamicImage::ImageLuma8(_)));
}

#[test]
fn run_keeps_rgb_for_a_colour_page() {
    let text = [Rect::new(80, 60, 120, 100)];
    let page = page(
        uniform_base_rgb(PAGE_SIZE, [200, 150, 100], &text),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(70, 50, 130, 110)],
        20,
        1.0,
    );

    let output =
        pc_mask::run(memory_input(page, MaskerConfig::default())).expect("masking succeeds");

    let cleaned = output.cleaned.load().expect("cleaned is in memory");
    assert!(matches!(cleaned.as_ref(), DynamicImage::ImageRgb8(_)));
    assert_eq!(output.mask_data.regions.len(), 1);
}

#[test]
fn run_on_a_page_with_no_regions_succeeds_with_an_empty_mask() {
    // spec §16.9 item 19 (mirroring §16.8 item 9): an empty page is a success, not an
    // error, and the cleaned image is the canvas untouched.
    let page = page(
        uniform_base(PAGE_SIZE, 200, &[]),
        text_raw_mask(PAGE_SIZE, &[]),
        &[],
        20,
        1.0,
    );
    let base = page.base_image.load().expect("cached base").to_luma8();

    let output =
        pc_mask::run(memory_input(page, MaskerConfig::default())).expect("an empty page is fine");

    assert!(output.mask_data.regions.is_empty());
    assert!(output.analytics.is_empty());
    let combined = output.combined_mask.load().expect("combined mask");
    assert!(combined.to_rgba8().pixels().all(|pixel| pixel.0[3] == 0));
    let cleaned = output.cleaned.load().expect("cleaned");
    assert_eq!(
        pc_testkit::metrics::diff_set_gray(&base, &cleaned.to_luma8()).len(),
        0
    );
}

#[test]
fn a10_a_region_over_no_text_is_absent_from_mask_data_and_analytics() {
    // spec §10.7(A)10: the blank-precise-mask region is dropped entirely, and run()
    // still succeeds.
    let text = [Rect::new(10, 10, 30, 30)];
    let page = page(
        uniform_base(PAGE_SIZE, 200, &text),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(120, 100, 160, 140)],
        20,
        1.0,
    );

    let output =
        pc_mask::run(memory_input(page, MaskerConfig::default())).expect("masking succeeds");

    assert!(output.mask_data.regions.is_empty());
    assert!(output.analytics.is_empty());
}

#[test]
fn run_rejects_a_page_that_fails_its_own_invariants() {
    // spec §16.9 item 19 (mirroring §16.8 item 8): a corrupt #clean.json becomes a clean
    // per-image failure, not a panic in the pixel code.
    let mut page = simple_page();
    page.extended_boxes.clear();

    let error = pc_mask::run(memory_input(page, MaskerConfig::default()))
        .expect_err("validation must reject this page");

    assert!(matches!(error, StageError::InvalidInput(_)));
}

#[test]
fn run_extracts_the_text_layer_only_when_asked() {
    let mut input = memory_input(simple_page(), MaskerConfig::default());
    input.extract_text = true;

    let with_text = pc_mask::run(input).expect("masking succeeds");
    let without_text = pc_mask::run(memory_input(simple_page(), MaskerConfig::default()))
        .expect("masking succeeds");

    assert!(with_text.text_layer.is_some());
    assert!(without_text.text_layer.is_none());
}

#[test]
fn scale_below_one_produces_a_cleaned_image_at_the_original_size() {
    // spec §10.7(A)13 / §16.9 item 13: with scale != 1 the canvas is `original_image`,
    // and the combined mask is upscaled to it with nearest-neighbour.
    let text = [Rect::new(80, 60, 120, 100)];
    let page = page(
        uniform_base(PAGE_SIZE, 200, &text),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(70, 50, 130, 110)],
        20,
        0.5,
    );
    let original = uniform_base((PAGE_SIZE.0 * 2, PAGE_SIZE.1 * 2), 200, &[]);
    let mut input = memory_input(page, MaskerConfig::default());
    input.original_image = pc_core::ImageHandle::with_both(common::ORIGINAL_IMAGE_PATH, original);

    let output = pc_mask::run(input).expect("masking succeeds");

    let cleaned = output.cleaned.load().expect("cleaned is in memory");
    assert_eq!(cleaned.dimensions(), (PAGE_SIZE.0 * 2, PAGE_SIZE.1 * 2));
    let combined = output.combined_mask.load().expect("combined mask");
    assert_eq!(
        combined.dimensions(),
        PAGE_SIZE,
        "the combined mask itself stays at base_image size (§10.2)"
    );
}

// ------------------------------------------------------------ destinations (disk mode)

#[test]
fn run_writes_every_requested_destination_and_returns_path_bearing_handles() {
    // spec §10.2's MaskDests + §10.3 steps 1, 2, 4.
    let dir = tempfile::tempdir().expect("temp dir");
    let mut input = memory_input(simple_page(), MaskerConfig::default());
    input.extract_text = true;
    input.debug_outputs = true;
    input.dests = disk_dests(dir.path());

    let output = pc_mask::run(input).expect("masking succeeds");

    for name in [
        "page_combined_mask.png",
        "page_clean.png",
        "page_text.png",
        "page_box_mask.png",
        "page_cut_mask.png",
        "page_with_masks.png",
    ] {
        assert!(dir.path().join(name).is_file(), "{name} was not written");
    }
    assert_eq!(
        output.combined_mask.path,
        Some(dir.path().join("page_combined_mask.png"))
    );
    assert_eq!(output.cleaned.path, Some(dir.path().join("page_clean.png")));
    assert_eq!(
        output
            .text_layer
            .as_ref()
            .and_then(|handle| handle.path.clone()),
        Some(dir.path().join("page_text.png"))
    );
}

#[test]
fn debug_outputs_are_not_written_when_not_requested() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut input = memory_input(simple_page(), MaskerConfig::default());
    input.dests = pc_mask::MaskDests {
        combined_mask: Some(dir.path().join("page_combined_mask.png")),
        cleaned: Some(dir.path().join("page_clean.png")),
        ..pc_mask::MaskDests::default()
    };

    let output = pc_mask::run(input).expect("masking succeeds");

    assert!(dir.path().join("page_combined_mask.png").is_file());
    assert!(!dir.path().join("page_box_mask.png").exists());
    assert!(output.text_layer.is_none());
}

#[test]
fn mask_data_round_trips_through_json_in_disk_mode() {
    // §16.9 item 16: byte-identity of `#mask_data.json` is only meaningful for
    // materialized handles, which is exactly when the pipeline checkpoints it.
    let dir = tempfile::tempdir().expect("temp dir");
    let mut input = memory_input(simple_page(), MaskerConfig::default());
    input.dests = disk_dests(dir.path());

    let output = pc_mask::run(input).expect("masking succeeds");

    let json = serde_json::to_string(&output.mask_data).expect("materialized handles serialize");
    let restored: pc_core::MaskData = serde_json::from_str(&json).expect("round trip");
    assert_eq!(restored.regions, output.mask_data.regions);
    assert_eq!(
        serde_json::to_string(&restored).expect("re-serialize"),
        json
    );
}

// ------------------------------------------------------------ determinism (§10.7(A)14)

#[test]
fn a14_twenty_runs_produce_identical_masks_and_regions() {
    // spec §10.7(A)14, as amended by §16.9 item 16: compare the combined-mask pixel
    // buffer, the cleaned-image buffer and `MaskData.regions` -- MaskData holds
    // ImageHandles, whose Serialize rejects path-less (memory-mode) handles (§2.3).
    let first = pc_mask::run(memory_input(simple_page(), MaskerConfig::default()))
        .expect("masking succeeds");
    let expected_mask = first.combined_mask.load().expect("mask").to_rgba8();
    let expected_clean = first.cleaned.load().expect("cleaned").to_rgba8();

    for run in 0..20 {
        let output = pc_mask::run(memory_input(simple_page(), MaskerConfig::default()))
            .expect("masking succeeds");
        assert_eq!(
            output.combined_mask.load().expect("mask").to_rgba8(),
            expected_mask,
            "combined mask differs on run {run}"
        );
        assert_eq!(
            output.cleaned.load().expect("cleaned").to_rgba8(),
            expected_clean,
            "cleaned image differs on run {run}"
        );
        assert_eq!(output.mask_data.regions, first.mask_data.regions);
        assert_eq!(output.mask_data.scale, first.mask_data.scale);
        assert_eq!(output.analytics, first.analytics);
    }
}

#[test]
fn a14_eight_threads_agree_with_one() {
    // spec §10.7(A)14: identical output regardless of thread count (§5.7).
    let expected = pc_mask::run(memory_input(simple_page(), MaskerConfig::default()))
        .expect("masking succeeds");
    let expected_mask = expected.combined_mask.load().expect("mask").to_rgba8();

    let handles = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                let output = pc_mask::run(memory_input(simple_page(), MaskerConfig::default()))
                    .expect("masking succeeds");
                (
                    output.combined_mask.load().expect("mask").to_rgba8(),
                    output.mask_data.regions,
                )
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        let (mask, regions) = handle.join().expect("worker thread did not panic");
        assert_eq!(mask, expected_mask);
        assert_eq!(regions, expected.mask_data.regions);
    }
}

#[test]
fn a14_disk_mode_writes_byte_identical_artifacts_twice() {
    // §16.9 item 16: the same destinations written twice must produce identical bytes.
    let dir = tempfile::tempdir().expect("temp dir");
    let mut input = memory_input(simple_page(), MaskerConfig::default());
    input.dests = disk_dests(dir.path());
    let output = pc_mask::run(input).expect("masking succeeds");
    let first_png = std::fs::read(dir.path().join("page_combined_mask.png")).expect("written");
    let first_json = serde_json::to_string(&output.mask_data).expect("serializable");

    let mut input = memory_input(simple_page(), MaskerConfig::default());
    input.dests = disk_dests(dir.path());
    let output = pc_mask::run(input).expect("masking succeeds");
    let second_png = std::fs::read(dir.path().join("page_combined_mask.png")).expect("written");
    let second_json = serde_json::to_string(&output.mask_data).expect("serializable");

    assert_eq!(first_png, second_png);
    assert_eq!(first_json, second_json);
}

// ------------------------------------------------------------ the Stage wrapper (§3)

#[test]
fn mask_stage_matches_the_free_function() {
    let via_trait =
        <MaskStage as Stage>::run(memory_input(simple_page(), MaskerConfig::default()), ())
            .expect("masking succeeds");
    let via_function = pc_mask::run(memory_input(simple_page(), MaskerConfig::default()))
        .expect("masking succeeds");

    assert_eq!(via_trait.mask_data.regions, via_function.mask_data.regions);
    assert_eq!(via_trait.analytics, via_function.analytics);
    assert_eq!(
        via_trait.combined_mask.load().expect("mask").to_rgba8(),
        via_function.combined_mask.load().expect("mask").to_rgba8()
    );
}

#[test]
fn mask_stage_declares_step_mask() {
    assert_eq!(<MaskStage as Stage>::STEP, pc_core::Step::Mask);
}

#[test]
fn mask_input_round_trips_through_json() {
    // §3: every stage Input is serde-derivable, so a checkpoint records exactly which
    // settings produced it.
    let input: MaskInput = memory_input(simple_page(), MaskerConfig::default());
    let json = serde_json::to_string(&input).expect("path-bearing handles serialize");

    let restored: MaskInput = serde_json::from_str(&json).expect("round trip");

    assert_eq!(restored.config, input.config);
    assert_eq!(restored.page.masking_regions, input.page.masking_regions);
    assert_eq!(restored.dests, input.dests);
}
