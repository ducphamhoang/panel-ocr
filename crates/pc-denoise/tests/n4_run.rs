//! Task N4 -- `run()` wiring: spec §11.2, §11.3 steps 1-2 and 5-6,
//! §11.7(A)7/8/10, §11.7(B)13, §16.10 items 4-7, 13, 15, 17, 18. FROZEN.

mod common;

use common::{
    combined_mask, exclusive_nlm, gray_page, memory_input, region, rgb_page, shared_nlm, PAGE_SIZE,
    REGION,
};
use image::{DynamicImage, GenericImageView};
use pc_config::DenoiserConfig;
use pc_core::{ImageHandle, Rect, Stage, SCHEMA_VERSION};
use pc_denoise::{nlm, DenoiseDests, DenoiseInput, DenoiseStage};

/// A 4x4 **1-bit** greyscale PNG, byte-for-byte (IHDR bit depth 1, colour type 0).
/// Written out by hand rather than vendored as a fixture because the `image` crate
/// cannot *encode* 1 bpp, and §16.10 item 6 makes §11.3 step 1's shortcut depend on the
/// real file header — so the test needs a genuine 1-bit file and nothing else will do.
/// Pixels (1 = white): 1010 / 0101 / 1100 / 0011.
const ONE_BIT_PNG: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0x81, 0x8a, 0xa3,
    0xd3, 0x00, 0x00, 0x00, 0x10, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x58, 0xc0, 0x10, 0xc0,
    0x70, 0x80, 0xc1, 0x00, 0x00, 0x08, 0x68, 0x01, 0xe1, 0xa0, 0x19, 0xb0, 0xd7, 0x00, 0x00, 0x00,
    0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
];

// ---------------------------------------------------------------- §16.10 items 4, 6

#[test]
fn the_dests_field_is_named_denoised() {
    // §16.10 item 4: §11.2's `denoised` is authoritative; §4.3's `clean_denoised` is a
    // typo. Pinned through serde so a rename breaks a checkpoint test, not just a
    // compile.
    let dests = DenoiseDests {
        noise_mask: Some("/cache/page_noise_mask.png".into()),
        denoised: Some("/cache/page_clean_denoised.png".into()),
    };
    let json = serde_json::to_value(&dests).expect("dests are plain paths");
    assert!(json.get("denoised").is_some(), "{json}");
    assert!(json.get("noise_mask").is_some(), "{json}");
    assert!(json.get("clean_denoised").is_none(), "{json}");
    assert_eq!(
        DenoiseDests::default(),
        DenoiseDests {
            noise_mask: None,
            denoised: None
        }
    );
}

#[test]
fn the_stage_impl_is_wired_to_step_denoise() {
    // spec §3 / §2.8.
    assert_eq!(DenoiseStage::STEP, pc_core::Step::Denoise);
    assert_eq!(pc_core::Step::Denoise.prev(), Some(pc_core::Step::Mask));
    assert_eq!(
        pc_core::Output::DenoiseMask.cache_suffix(),
        "_noise_mask.png"
    );
    assert_eq!(
        pc_core::Output::DenoisedOutput.cache_suffix(),
        "_clean_denoised.png"
    );
}

#[test]
fn is_one_bit_reads_the_file_header_not_the_decoded_value() {
    // §16.10 item 6: a 1-bit PNG decodes to `Luma8` with values 0/255 AND `image`'s PNG
    // decoder reports `original_color_type() == L8` for it (verified against 0.25.10),
    // so §11.3 step 1's probe MUST parse the IHDR itself. A path-less (memory-only)
    // handle is never 1-bit, because an in-memory `DynamicImage` cannot represent 1 bpp.
    let dir = tempfile::tempdir().expect("a temp dir");
    let one_bit = dir.path().join("one_bit.png");
    std::fs::write(&one_bit, ONE_BIT_PNG).expect("write the 1-bit fixture");
    assert!(
        pc_denoise::is_one_bit(&ImageHandle::from_path(&one_bit)).expect("a readable png"),
        "the hand-built fixture must actually be 1 bpp"
    );
    // ... and it decodes to plain Luma8, which is exactly why the header is needed.
    let decoded = image::open(&one_bit).expect("decodable");
    assert!(matches!(decoded, DynamicImage::ImageLuma8(_)));

    let eight_bit = dir.path().join("eight_bit.png");
    gray_page(PAGE_SIZE, 200, &[])
        .save(&eight_bit)
        .expect("write an 8-bit page");
    assert!(!pc_denoise::is_one_bit(&ImageHandle::from_path(&eight_bit)).expect("readable"));

    assert!(
        !pc_denoise::is_one_bit(&ImageHandle::from_memory(gray_page((4, 4), 128, &[])))
            .expect("a memory handle is never 1-bit")
    );
}

// ---------------------------------------------------------------- §11.7(A)7

#[test]
fn a7_a_one_bit_original_short_circuits_the_whole_stage() {
    // spec §11.7(A)7: a 1-bit original yields `denoised` byte-identical to
    // `masked_image`, a fully transparent noise mask of the correct size, empty
    // analytics, and ZERO NLM invocations. §16.10 item 7 pins the copy semantics: with
    // both paths present and different, the shortcut is a byte-level `fs::copy`.
    let _nlm = exclusive_nlm();
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = dir.path().join("page.png");
    std::fs::write(&original, ONE_BIT_PNG).expect("write the 1-bit original");

    // The stage-3 clean output for a 1-bit page: an ordinary 8-bit PNG.
    let masked_path = dir.path().join("page_clean.png");
    gray_page((4, 4), 170, &[(Rect::new(1, 1, 3, 3), 40)])
        .save(&masked_path)
        .expect("write _clean.png");

    let noise_dest = dir.path().join("page_noise_mask.png");
    let denoised_dest = dir.path().join("page_clean_denoised.png");

    let mut input = memory_input(
        gray_page((4, 4), 170, &[]),
        combined_mask((4, 4), &[]),
        vec![region(Rect::new(0, 0, 4, 4), 9.0, false)],
        DenoiserConfig::default(),
    );
    input.original_image = ImageHandle::from_path(&original);
    input.masked_image = ImageHandle::from_path(&masked_path);
    input.dests = DenoiseDests {
        noise_mask: Some(noise_dest.clone()),
        denoised: Some(denoised_dest.clone()),
    };

    nlm::reset_denoise_call_count();
    let output = pc_denoise::run(input).expect("the 1-bit shortcut never fails");

    assert_eq!(
        nlm::denoise_call_count(),
        0,
        "the 1-bit shortcut must not invoke NLM at all"
    );
    assert_eq!(
        std::fs::read(&denoised_dest).expect("the denoised file"),
        std::fs::read(&masked_path).expect("the masked file"),
        "the denoised output must be byte-identical to _clean.png"
    );

    let noise = output.noise_mask.load().expect("the noise mask").to_rgba8();
    assert_eq!(noise.dimensions(), (4, 4), "original size, not mask size");
    assert!(noise.pixels().all(|pixel| pixel.0[3] == 0));

    assert_eq!(
        output.analytics.path,
        std::path::PathBuf::from(common::ORIGINAL_PATH)
    );
    assert!(
        output.analytics.std_deviations.is_empty(),
        "§11.3 step 1 returns EMPTY analytics on the shortcut"
    );
    assert_eq!(output.analytics.boxes_denoised, 0);
}

// ---------------------------------------------------------------- §11.7(A)10

#[test]
fn a10_zero_qualifying_regions_writes_a_blank_mask_and_the_stage_three_composite() {
    // spec §11.7(A)10: no qualifying region means a fully transparent `_noise_mask.png`
    // is still WRITTEN, `denoised` equals the stage-3 composite, and
    // `boxes_denoised == 0`. Every region here is excluded: one at exactly the cutoff
    // (strict `>`), one failed.
    let _nlm = shared_nlm();
    let dir = tempfile::tempdir().expect("a temp dir");
    let noise_dest = dir.path().join("page_noise_mask.png");
    let denoised_dest = dir.path().join("page_clean_denoised.png");

    let original = gray_page(PAGE_SIZE, 210, &[(REGION, 20)]);
    let mask = combined_mask(PAGE_SIZE, &[(REGION, [210, 210, 210, 255])]);
    let mut input = memory_input(
        original.clone(),
        mask.clone(),
        vec![
            region(REGION, 0.25, false),
            region(Rect::new(0, 0, 20, 20), 8.0, true),
        ],
        DenoiserConfig::default(),
    );
    input.dests = DenoiseDests {
        noise_mask: Some(noise_dest.clone()),
        denoised: Some(denoised_dest.clone()),
    };

    let output = pc_denoise::run(input).expect("an empty selection is a success");
    assert_eq!(output.analytics.boxes_denoised, 0);
    assert!(
        noise_dest.is_file(),
        "the blank noise mask is still written"
    );
    assert!(denoised_dest.is_file());

    let noise = output.noise_mask.load().expect("noise mask").to_rgba8();
    assert_eq!(noise.dimensions(), PAGE_SIZE);
    assert!(noise.pixels().all(|pixel| pixel.0[3] == 0));

    // §11.3 step 2: the stage-3 composite, recomputed at full resolution.
    let expected = pc_denoise::composite_rgb(&original.to_rgb8(), &mask);
    let actual = output.denoised.load().expect("denoised").to_rgb8();
    assert_eq!(actual, expected);

    // §11.3 step 6 / §16.10 item 15: sigma of ALL regions, failures included.
    assert_eq!(output.analytics.std_deviations, vec![0.25, 8.0]);
}

// ---------------------------------------------------------------- §11.7(A)8

#[test]
fn a8_run_changes_nothing_outside_the_padded_region() {
    // spec §11.7(A)8: `denoised` differs from the stage-3 composite ONLY within
    // `region.rect.pad(noise_outline_size + 3 * noise_fade_radius)`.
    let _nlm = shared_nlm();
    let config = DenoiserConfig {
        template_window_size: 3,
        search_window_size: 5,
        ..DenoiserConfig::default()
    };
    // A noisy page so NLM actually has something to change inside the region.
    let noisy = pc_testkit::images::noisy_gray(PAGE_SIZE.0, PAGE_SIZE.1, 150, 18.0, 0x5C0BE_u64);
    let original = DynamicImage::ImageLuma8(noisy);
    let mask = combined_mask(PAGE_SIZE, &[(REGION, [255, 255, 255, 255])]);
    let input = memory_input(
        original.clone(),
        mask.clone(),
        vec![region(REGION, 4.0, false)],
        config.clone(),
    );

    let output = pc_denoise::run(input).expect("a well-formed page");
    assert_eq!(output.analytics.boxes_denoised, 1);

    let baseline = pc_denoise::composite_rgb(&original.to_rgb8(), &mask);
    let actual = output.denoised.load().expect("denoised").to_rgb8();
    let reach = (config.noise_outline_size + 3 * config.noise_fade_radius) as i32;
    let allowed = REGION.pad(reach, PAGE_SIZE);
    let mut changed = 0_usize;
    for (x, y, pixel) in actual.enumerate_pixels() {
        if pixel == baseline.get_pixel(x, y) {
            continue;
        }
        changed += 1;
        assert!(
            allowed.contains((x as i32, y as i32)),
            "pixel ({x}, {y}) changed outside {allowed:?}"
        );
    }
    assert!(changed > 0, "denoising must actually change something");
}

// ---------------------------------------------------------------- §16.10 items 13, 17

#[test]
fn item13_the_upscale_factor_comes_from_the_actual_sizes_not_from_mask_data_scale() {
    // §11.3 step 2 + §16.6 item 3 + §16.10 item 13: the denoiser recomputes
    // `scale_up = cleaned.width / mask.width` from the loaded images and NEVER trusts
    // `MaskData.scale`. Here `scale` is deliberately a lie (0.1) while the true ratio
    // is 2.0; the output must still be correct at the original resolution.
    let _nlm = shared_nlm();
    let half = (PAGE_SIZE.0 / 2, PAGE_SIZE.1 / 2);
    let half_region = Rect::new(20, 15, 35, 30);
    let original = gray_page(PAGE_SIZE, 205, &[(REGION, 30)]);
    let mask = combined_mask(half, &[(half_region, [205, 205, 205, 255])]);

    let mut input = memory_input(
        original.clone(),
        mask.clone(),
        // resolved 2026-07-28 (§16.10 item 22): σ must exceed
        // `noise_min_standard_deviation` (0.25, `DenoiserConfig::default`) or the region
        // is never selected at all, regardless of the scale recompute under test.
        vec![region(half_region, 0.5, false)],
        DenoiserConfig {
            template_window_size: 3,
            search_window_size: 5,
            ..DenoiserConfig::default()
        },
    );
    input.mask_data.scale = 0.1; // a lie the stage must ignore
    input.mask_data.combined_mask =
        ImageHandle::from_memory(DynamicImage::ImageRgba8(mask.clone()));

    let output = pc_denoise::run(input).expect("a half-size mask is normal");
    let denoised = output.denoised.load().expect("denoised");
    let noise = output.noise_mask.load().expect("noise mask").to_rgba8();
    assert_eq!(
        denoised.dimensions(),
        PAGE_SIZE,
        "outputs are at ORIGINAL size"
    );
    assert_eq!(noise.dimensions(), PAGE_SIZE);
    assert_eq!(output.analytics.boxes_denoised, 1);

    // scale_up == 2.0, so the half-size region rect lands on `REGION` (§2.1's
    // truncating `Rect::scale`).
    assert_eq!(half_region.scale(2.0), Rect::new(40, 30, 70, 60));
    let opaque: Vec<(u32, u32)> = noise
        .enumerate_pixels()
        .filter(|(_, _, pixel)| pixel.0[3] > 0)
        .map(|(x, y, _)| (x, y))
        .collect();
    assert!(!opaque.is_empty());
    let reach = 5 + 3;
    let allowed = Rect::new(40, 30, 70, 60).pad(reach, PAGE_SIZE);
    for (x, y) in opaque {
        assert!(allowed.contains((x as i32, y as i32)), "({x}, {y})");
    }
}

#[test]
fn item17_a_grayscale_page_stays_grayscale_and_a_colour_page_stays_rgb() {
    // §16.10 item 17: `ImageLuma8` iff the loaded original is L/LA AND every composited
    // pixel is achromatic, else `ImageRgb8`. This mirrors §16.9 item 14's user-visible
    // outcome so a grayscale page survives the whole pipeline as `L`. The noise mask is
    // ALWAYS RGBA.
    let _nlm = shared_nlm();
    let fast = DenoiserConfig {
        template_window_size: 3,
        search_window_size: 5,
        ..DenoiserConfig::default()
    };

    let gray = memory_input(
        gray_page(PAGE_SIZE, 205, &[(REGION, 30)]),
        combined_mask(PAGE_SIZE, &[(REGION, [205, 205, 205, 255])]),
        vec![region(REGION, 3.0, false)],
        fast.clone(),
    );
    let output = pc_denoise::run(gray).expect("a grayscale page");
    assert!(
        matches!(
            output.denoised.load().expect("denoised").as_ref(),
            DynamicImage::ImageLuma8(_)
        ),
        "a grayscale page with achromatic fills must come back as L"
    );
    assert!(matches!(
        output.noise_mask.load().expect("noise mask").as_ref(),
        DynamicImage::ImageRgba8(_)
    ));

    let colour = memory_input(
        rgb_page(PAGE_SIZE, [200, 40, 40], &[(REGION, [10, 10, 200])]),
        combined_mask(PAGE_SIZE, &[(REGION, [200, 40, 40, 255])]),
        vec![region(REGION, 3.0, false)],
        fast,
    );
    let output = pc_denoise::run(colour).expect("a colour page");
    assert!(
        matches!(
            output.denoised.load().expect("denoised").as_ref(),
            DynamicImage::ImageRgb8(_)
        ),
        "a colour page must come back as RGB"
    );
}

// ---------------------------------------------------------------- §16.10 item 18

#[test]
fn item18_denoising_enabled_is_not_consulted_by_the_stage() {
    // §16.10 item 18: `denoising_enabled` is a pipeline-level skip and an export-level
    // input, never a silent no-op inside `run()`. Both settings must produce the same
    // output for the same page.
    let _nlm = shared_nlm();
    let build = |enabled: bool| {
        memory_input(
            gray_page(PAGE_SIZE, 205, &[(REGION, 30)]),
            combined_mask(PAGE_SIZE, &[(REGION, [205, 205, 205, 255])]),
            vec![region(REGION, 3.0, false)],
            DenoiserConfig {
                denoising_enabled: enabled,
                template_window_size: 3,
                search_window_size: 5,
                ..DenoiserConfig::default()
            },
        )
    };
    let on = pc_denoise::run(build(true)).expect("enabled");
    let off = pc_denoise::run(build(false)).expect("disabled");
    assert_eq!(on.analytics.boxes_denoised, 1);
    assert_eq!(off.analytics.boxes_denoised, 1);
    assert_eq!(
        on.denoised.load().expect("denoised").to_rgb8(),
        off.denoised.load().expect("denoised").to_rgb8()
    );
}

// ---------------------------------------------------------------- determinism

#[test]
fn the_stage_is_deterministic_across_repeats_and_thread_counts() {
    // §5.7, and the §16.9 item 16 precedent: memory mode compares pixel buffers and the
    // analytics record structurally -- `MaskData` holds `ImageHandle`s and
    // `ImageHandle::Serialize` deliberately rejects path-less handles (§2.3, §16.7), so
    // a JSON round-trip is not available here.
    let _nlm = shared_nlm();
    let build = || {
        memory_input(
            DynamicImage::ImageLuma8(pc_testkit::images::noisy_gray(
                PAGE_SIZE.0,
                PAGE_SIZE.1,
                150,
                18.0,
                0xDEC1DE,
            )),
            combined_mask(PAGE_SIZE, &[(REGION, [255, 255, 255, 255])]),
            vec![region(REGION, 4.0, false)],
            DenoiserConfig {
                template_window_size: 3,
                search_window_size: 5,
                ..DenoiserConfig::default()
            },
        )
    };
    let reference = common::with_threads(1, || pc_denoise::run(build())).expect("run");
    let reference_denoised = reference.denoised.load().expect("denoised").to_rgb8();
    let reference_noise = reference.noise_mask.load().expect("noise").to_rgba8();

    for threads in [1_usize, 8] {
        for attempt in 0..3 {
            let output = common::with_threads(threads, || pc_denoise::run(build())).expect("run");
            assert_eq!(
                output.denoised.load().expect("denoised").to_rgb8(),
                reference_denoised,
                "denoised diverged on attempt {attempt} at {threads} thread(s)"
            );
            assert_eq!(
                output.noise_mask.load().expect("noise").to_rgba8(),
                reference_noise
            );
            assert_eq!(output.analytics, reference.analytics);
        }
    }
}

#[test]
fn a_disk_mode_run_is_byte_identical_when_repeated() {
    // The mode in which byte-identity is meaningful (§16.9 item 16's split).
    let _nlm = shared_nlm();
    let dir = tempfile::tempdir().expect("a temp dir");
    let noise_dest = dir.path().join("page_noise_mask.png");
    let denoised_dest = dir.path().join("page_clean_denoised.png");
    let build = || {
        let mut input = memory_input(
            gray_page(PAGE_SIZE, 205, &[(REGION, 30)]),
            combined_mask(PAGE_SIZE, &[(REGION, [205, 205, 205, 255])]),
            vec![region(REGION, 3.0, false)],
            DenoiserConfig {
                template_window_size: 3,
                search_window_size: 5,
                ..DenoiserConfig::default()
            },
        );
        input.dests = DenoiseDests {
            noise_mask: Some(noise_dest.clone()),
            denoised: Some(denoised_dest.clone()),
        };
        input
    };
    pc_denoise::run(build()).expect("first run");
    let first = (
        std::fs::read(&noise_dest).expect("noise mask"),
        std::fs::read(&denoised_dest).expect("denoised"),
    );
    pc_denoise::run(build()).expect("second run");
    let second = (
        std::fs::read(&noise_dest).expect("noise mask"),
        std::fs::read(&denoised_dest).expect("denoised"),
    );
    assert_eq!(
        first, second,
        "two identical runs must write identical bytes"
    );
}

#[test]
fn the_input_round_trips_through_a_json_checkpoint() {
    // §3: `Stage::Input` must be serde-derivable so §4's disk checkpointing works. All
    // handles here carry paths, which is what `ImageHandle::Serialize` requires (§2.3).
    let input = DenoiseInput {
        schema_version: SCHEMA_VERSION,
        mask_data: common::mask_data(
            gray_page((8, 8), 200, &[]),
            combined_mask((8, 8), &[]),
            1.0,
            vec![region(Rect::new(0, 0, 4, 4), 1.5, false)],
        ),
        original_image: ImageHandle::from_path("/synthetic/page.png"),
        masked_image: ImageHandle::from_path("/synthetic/page_clean.png"),
        config: DenoiserConfig::default(),
        dests: DenoiseDests::default(),
    };
    let json = serde_json::to_string(&input).expect("a fully materialized input");
    let back: DenoiseInput = serde_json::from_str(&json).expect("round trip");
    assert_eq!(back.schema_version, SCHEMA_VERSION);
    assert_eq!(back.mask_data.regions, input.mask_data.regions);
    assert_eq!(back.config, input.config);
    assert_eq!(back.dests, input.dests);
}

// ---------------------------------------------------------------- §11.7(B)13

#[test]
#[ignore = "pending task F1: needs the recorded page fixture and its committed golden PNGs"]
fn b13_pending_recorded_page_end_to_end_golden() {
    // spec §11.7(B)13 + §16.10 item 20. On a recorded page (§7.2): `_noise_mask.png`
    // and `_clean_denoised.png` are locked as size/mode/alpha-histogram assertions
    // (`noise_mask::alpha_histogram`) plus `pc_testkit::metrics::ssim_gray >= 0.99`
    // against a committed reference **image file** -- a golden PNG, NOT an `insta`
    // snapshot; this is deliberately not a third snapshot site under §15.10(c).
    // The reference is refreshed only by explicit joint-architect decision, exactly like
    // any other frozen expectation.
    unimplemented!("blocked on F1");
}
