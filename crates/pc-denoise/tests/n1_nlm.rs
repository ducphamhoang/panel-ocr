//! Task N1 -- spec §11.7(A)1-5 and 11, §11.7(B)12. FROZEN.
//!
//! Every gate here uses the **real** §6 defaults (`h = 10`, `template = 7`,
//! `search = 21`), because those are the parameters §11.7(B)12's recorded OpenCV
//! reference was produced with and the ones the perf budget is stated for.

mod common;

use common::{exclusive_nlm, mean, shared_nlm, std_deviation, with_threads};
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage};
use pc_denoise::nlm::{self, NlmParams};
use pc_testkit::images;

fn defaults() -> NlmParams {
    NlmParams::defaults()
}

#[test]
fn nlm_defaults_are_the_profile_defaults() {
    // The parity fixture name `<name>_h10_t7_s21.png` (§11.6) encodes this triple, so
    // a drift between `NlmParams::defaults()` and `DenoiserConfig::default()` would
    // silently invalidate §11.7(B)12.
    let config = pc_config::DenoiserConfig::default();
    let params = defaults();
    assert_eq!(params.h, config.filter_strength as f32);
    assert_eq!(params.template_window, config.template_window_size);
    assert_eq!(params.search_window, config.search_window_size);
    assert_eq!(
        (params.h, params.template_window, params.search_window),
        (10.0, 7, 21)
    );
}

#[test]
fn reflect101_matches_opencv_border_reflect_101() {
    // spec §11.3: `BORDER_REFLECT_101` is `abcd | cba` -- index -1 mirrors to 1, not 0.
    // §16.10 item 9 pins the formula; this test pins the values it must produce.
    let n = 4; // a b c d
    let expected = [
        (-3_i64, 3_usize), // d
        (-2, 2),           // c
        (-1, 1),           // b
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2), // c
        (5, 1), // b
        (6, 0), // a
    ];
    for (index, want) in expected {
        assert_eq!(
            nlm::reflect101(index, n),
            want,
            "reflect101({index}, {n}) must be {want}"
        );
    }
    // A degenerate single-column image has exactly one valid index.
    for index in -5..5 {
        assert_eq!(nlm::reflect101(index, 1), 0);
    }
}

#[test]
fn a1_constant_image_is_returned_bit_exact() {
    let _nlm = shared_nlm();
    // spec §11.7(A)1: all weights are 1 and the weighted mean of a constant is that
    // constant, so a uniform image must come back BIT-EXACT -- any rounding or
    // border bug shows up here first.
    for value in [0_u8, 1, 37, 128, 254, 255] {
        let gray = GrayImage::from_pixel(24, 17, Luma([value]));
        let out = nlm::denoise(&DynamicImage::ImageLuma8(gray.clone()), defaults());
        assert_eq!(
            out.as_luma8().expect("L in, L out (§16.10 item 9)"),
            &gray,
            "constant L image with value {value} was modified"
        );
    }
    let rgb = RgbImage::from_pixel(19, 23, Rgb([12, 200, 77]));
    let out = nlm::denoise(&DynamicImage::ImageRgb8(rgb.clone()), defaults());
    assert_eq!(
        out.as_rgb8().expect("RGB in, RGB out (§16.10 item 9)"),
        &rgb,
        "constant RGB image was modified"
    );
}

#[test]
fn a1_grayscale_and_replicated_rgb_agree_exactly() {
    let _nlm = shared_nlm();
    // §16.10 item 8: `d` divides the per-channel sum by `C`, so a replicated-gray RGB
    // image yields EXACTLY the same numbers as its `L` counterpart. This is what makes
    // the grayscale OpenCV reference (§11.7(B)12) a valid target for our RGB path, and
    // what makes `colored_images` a no-op in v1.
    let gray = images::noisy_gray(23, 19, 128, 12.0, 0xC0FFEE);
    let rgb = RgbImage::from_fn(23, 19, |x, y| {
        let v = gray.get_pixel(x, y).0[0];
        Rgb([v, v, v])
    });

    let denoised_gray = nlm::denoise(&DynamicImage::ImageLuma8(gray), defaults());
    let denoised_rgb = nlm::denoise(&DynamicImage::ImageRgb8(rgb), defaults());

    let l = denoised_gray.as_luma8().expect("L out");
    let r = denoised_rgb.as_rgb8().expect("RGB out");
    for (x, y, pixel) in r.enumerate_pixels() {
        let expected = l.get_pixel(x, y).0[0];
        assert_eq!(
            pixel.0, [expected; 3],
            "channel-count invariance broken at ({x}, {y})"
        );
    }
}

#[test]
fn a2_gaussian_noise_on_flat_gray_is_suppressed() {
    let _nlm = shared_nlm();
    // spec §11.7(A)2: sigma = 12, fixed seed; the denoised standard deviation must be
    // < 40 % of the input's and the mean must be within 0.5 of the input's.
    let noisy = images::noisy_gray(64, 64, 128, 12.0, 0x5EED);
    let input_std = std_deviation(&noisy);
    let input_mean = mean(&noisy);
    assert!(
        input_std > 10.0,
        "the fixture must actually be noisy (measured sigma {input_std})"
    );

    let out = nlm::denoise(&DynamicImage::ImageLuma8(noisy), defaults());
    let out = out.as_luma8().expect("L out");
    let output_std = std_deviation(out);
    let output_mean = mean(out);

    assert!(
        output_std < 0.4 * input_std,
        "denoised sigma {output_std} is not < 40 % of the input's {input_std}"
    );
    pc_testkit::assert_close(output_mean, input_mean, 0.5);
}

#[test]
fn a3_a_sharp_step_edge_is_preserved() {
    let _nlm = shared_nlm();
    // spec §11.7(A)3: plateaus within +/-3 of 0 and 255 at >= 4 px from the edge, and a
    // transition no wider than the template window.
    let (width, height, edge) = (40_u32, 24_u32, 20_u32);
    let step = GrayImage::from_fn(width, height, |x, _| Luma([if x < edge { 0 } else { 255 }]));
    let out = nlm::denoise(&DynamicImage::ImageLuma8(step), defaults());
    let out = out.as_luma8().expect("L out");

    for y in 0..height {
        for x in 0..width {
            let value = out.get_pixel(x, y).0[0];
            let distance = if x < edge { edge - x } else { x - edge + 1 };
            if distance >= 4 {
                let expected = if x < edge { 0_i32 } else { 255 };
                assert!(
                    (i32::from(value) - expected).abs() <= 3,
                    "plateau broken at ({x}, {y}): {value}, expected ~{expected}"
                );
            }
        }
        let transition = (0..width)
            .filter(|x| {
                let value = out.get_pixel(*x, y).0[0];
                value > 3 && value < 252
            })
            .count();
        assert!(
            transition <= defaults().template_window as usize,
            "transition on row {y} is {transition} px wide, wider than the template window"
        );
    }
}

#[test]
fn a4_a_search_window_larger_than_the_image_does_not_panic() {
    let _nlm = shared_nlm();
    // spec §11.7(A)4: a 3x3 constant image with search_window = 21 exercises every
    // reflect-101 index in the padded plane; it must not panic and must return the
    // constant (which also proves the mirrored border carries the right values).
    let tiny = GrayImage::from_pixel(3, 3, Luma([200]));
    let out = nlm::denoise(&DynamicImage::ImageLuma8(tiny.clone()), defaults());
    assert_eq!(out.as_luma8().expect("L out"), &tiny);

    // A 1x1 image is the reflect101(_, 1) degenerate case.
    let single = GrayImage::from_pixel(1, 1, Luma([7]));
    let out = nlm::denoise(&DynamicImage::ImageLuma8(single.clone()), defaults());
    assert_eq!(out.as_luma8().expect("L out"), &single);
}

#[test]
fn a5_output_is_bit_identical_across_thread_counts_and_repeats() {
    let _nlm = shared_nlm();
    // spec §11.7(A)5 / §5.7: 10 runs at 1 and at 8 rayon threads must be bit-identical.
    // §16.10 item 10 is what makes this structural: fixed-size output row bands with
    // per-band scratch, and a fixed offset iteration order.
    let noisy = images::noisy_gray(48, 40, 120, 15.0, 0xD37E);
    let source = DynamicImage::ImageLuma8(noisy);

    let reference = with_threads(1, || nlm::denoise(&source, defaults()));
    let reference = reference.as_luma8().expect("L out").clone();

    for threads in [1_usize, 8] {
        for run in 0..10 {
            let out = with_threads(threads, || nlm::denoise(&source, defaults()));
            assert_eq!(
                out.as_luma8().expect("L out"),
                &reference,
                "run {run} at {threads} thread(s) diverged"
            );
        }
    }
}

#[test]
fn a11_performance_gate_256x256_grayscale_single_threaded() {
    let _nlm = shared_nlm();
    // spec §11.7(A)11 / §11.3: < 400 ms for a 256x256 grayscale crop with the defaults,
    // single-threaded, asserted with the generous 3x margin §11.7(A)11 mandates so CI
    // noise does not flake it. A regression to the naive O(W*H*s^2*t^2) loop is ~49x
    // slower and blows straight through this.
    let noisy = images::noisy_gray(256, 256, 128, 12.0, 0xBEEF);
    let source = DynamicImage::ImageLuma8(noisy);

    let start = std::time::Instant::now();
    let out = with_threads(1, || nlm::denoise(&source, defaults()));
    let elapsed = start.elapsed();

    assert_eq!(out.dimensions(), (256, 256));
    assert!(
        elapsed < std::time::Duration::from_millis(1200),
        "256x256 single-threaded NLM took {elapsed:?}, over the 400 ms budget's 3x margin"
    );
}

#[test]
fn the_invocation_counter_tracks_calls() {
    let _nlm = exclusive_nlm();
    // §16.10 item 11: the counter §11.7(A)7 relies on is always-on, not cfg-gated,
    // because that assertion lives in an integration test.
    nlm::reset_denoise_call_count();
    assert_eq!(nlm::denoise_call_count(), 0);
    let tiny = DynamicImage::ImageLuma8(GrayImage::from_pixel(4, 4, Luma([50])));
    nlm::denoise(&tiny, defaults());
    nlm::denoise(&tiny, defaults());
    assert_eq!(nlm::denoise_call_count(), 2);
}

#[test]
fn b12_recorded_opencv_parity() {
    // spec §11.7(B)12 + §16.10 item 19. Input: the whole
    // `upstream/demo_bubbles/<name>_bubble_raw.png` as luma8 (name in {nightmare, ray});
    // reference: `recorded/nlm/<name>_h10_t7_s21.png`, produced by
    // `cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)`
    // via the committed `cargo xtask record-fixtures` script (§7.2).
    // Thresholds: `GoldenThresholds::nlm_parity()` -- SSIM >= 0.98, mean |delta| <= 1.0,
    // max delta <= 8; the tolerance rationale is §11.7(B)12's own paragraph (OpenCV
    // buckets its weights into a fixed-point LUT and drops sub-threshold ones; we use
    // exact f32 `exp`).
    // §7.3: task F2 (`cargo xtask calibrate-goldens`) must have run and recorded the
    // measured deltas in docs/GOLDEN_CALIBRATION.md BEFORE this test is unignored. If
    // calibration shows the specified tolerances cannot be met, that goes back to the
    // two architects jointly -- never a silent loosening here.
    //
    // UNIGNORED by task F1/F2 (§16.13). The references are REAL OpenCV 5.0.0 output from
    // the committed `xtask/scripts/record_nlm.py`; F2 ran first and recorded the measured
    // numbers in docs/GOLDEN_CALIBRATION.md §1 (nightmare: SSIM 0.999985, mean 0.008946,
    // max 4; ray: SSIM 0.999999, mean 0.003028, max 1) -- comfortably inside the
    // SPECIFIED tolerances, which are what is asserted below and stay frozen.
    let _nlm = shared_nlm();
    let thresholds = pc_testkit::golden::GoldenThresholds::nlm_parity();

    for name in ["nightmare", "ray"] {
        let source = images::load_luma8(pc_testkit::paths::upstream(format!(
            "demo_bubbles/{name}_bubble_raw.png"
        )));
        let ours = match nlm::denoise(&DynamicImage::ImageLuma8(source), defaults()) {
            DynamicImage::ImageLuma8(gray) => gray,
            other => panic!("luma8 input must yield luma8 output, got {other:?}"),
        };
        let reference = images::load_luma8(pc_testkit::paths::recorded(format!(
            "nlm/{name}_h10_t7_s21.png"
        )));
        assert_eq!(
            reference.dimensions(),
            ours.dimensions(),
            "{name}: recorded reference and our output must share dimensions"
        );

        let report = pc_testkit::golden::GoldenReport::compare_gray(name, &reference, &ours);
        thresholds.assert_met(&report);
    }
}
