//! C4 tests -- the golden-compare helpers. Frozen gates.
//!
//! Two behaviours matter and are tested separately: the REPORT (which must never
//! assert -- §15.2 / ATTRIBUTION.md forbid a pass/fail gate against a `*_clean.png`),
//! and the THRESHOLD assertion (used only by the frozen gates §11.7(B)12 and
//! §12.7(A)1/2).

use approx::assert_abs_diff_eq;
use pc_testkit::golden::{
    assert_images_identical_gray, is_within_dilated, GoldenReport, GoldenThresholds,
};
use pc_testkit::images::{gray_from_rows, solid_gray};
use pc_testkit::metrics::PixelSet;

const EPS: f64 = 1e-9;

#[test]
// A report of an image against itself is the perfect score on every metric -- the
// self-check that proves the report is wired to the metrics correctly.
fn report_of_identical_images_is_perfect() {
    let img = gray_from_rows(&[&[0, 128, 255], &[7, 7, 7]]);
    let r = GoldenReport::compare_gray("self", &img, &img);

    assert_eq!(r.name, "self");
    assert_eq!(r.dims, (3, 2));
    assert_abs_diff_eq!(r.exact_fraction, 1.0, epsilon = EPS);
    assert_eq!(r.max_delta, 0);
    assert_abs_diff_eq!(r.mean_abs_diff, 0.0, epsilon = EPS);
    assert_abs_diff_eq!(r.ssim, 1.0, epsilon = EPS);
    // no source image was supplied, so there is no shape metric
    assert_eq!(r.shape_iou, None);
    assert_eq!(r.shape_subset_of_dilated, None);
}

#[test]
// Hand-computed report on a 2x2:
//   expected = [0, 10, 20, 30], actual = [0, 12, 17, 30]
//   -> exact 2/4 = 0.5, max delta 3, mean |delta| 1.25
fn report_values_are_the_underlying_metrics() {
    let expected = gray_from_rows(&[&[0, 10], &[20, 30]]);
    let actual = gray_from_rows(&[&[0, 12], &[17, 30]]);
    let r = GoldenReport::compare_gray("hand", &expected, &actual);

    assert_abs_diff_eq!(r.exact_fraction, 0.5, epsilon = EPS);
    assert_eq!(r.max_delta, 3);
    assert_abs_diff_eq!(r.mean_abs_diff, 1.25, epsilon = EPS);
}

#[test]
// §10.7(B)'s shape metric: with source != expected != actual, G and O are derived from
// the SOURCE, not from each other.
//   source   = [0, 0, 0, 0]
//   upstream = [0, 5, 5, 0]   -> G = {(1,0), (0,1)}
//   ours     = [0, 5, 0, 0]   -> O = {(1,0)}
//   |G intersect O| = 1, |G union O| = 2  -> IoU = 0.5, and O subset dilate(G, 0) so subset = true
fn shape_metric_derives_g_and_o_from_the_source() {
    let source = gray_from_rows(&[&[0, 0], &[0, 0]]);
    let upstream = gray_from_rows(&[&[0, 5], &[5, 0]]);
    let ours = gray_from_rows(&[&[0, 5], &[0, 0]]);

    let r = GoldenReport::compare_gray_with_shape("shape", &source, &upstream, &ours, 0);
    assert_abs_diff_eq!(r.shape_iou.unwrap(), 0.5, epsilon = EPS);
    assert_eq!(r.shape_subset_of_dilated, Some(true));
}

#[test]
// §10.7(B): the subset check uses the DILATED G, so a 1px-off change passes at
// radius 2 and fails at radius 0.
fn shape_subset_check_honours_the_dilation_radius() {
    let source = solid_gray(8, 8, 0);
    let mut upstream = solid_gray(8, 8, 0);
    upstream.put_pixel(4, 4, image::Luma([9]));
    let mut ours = solid_gray(8, 8, 0);
    ours.put_pixel(5, 4, image::Luma([9]));

    let strict = GoldenReport::compare_gray_with_shape("s", &source, &upstream, &ours, 0);
    assert_eq!(strict.shape_subset_of_dilated, Some(false));

    let tolerant = GoldenReport::compare_gray_with_shape("s", &source, &upstream, &ours, 2);
    assert_eq!(tolerant.shape_subset_of_dilated, Some(true));
}

#[test]
// §15.2 / ATTRIBUTION.md: producing a report must NEVER panic, however bad the
// numbers -- the demo_bubbles comparison is a non-gating calibration report and a
// shortfall there is expected evidence, not a build failure.
fn producing_a_report_never_panics_however_bad_the_numbers() {
    let expected = solid_gray(8, 8, 0);
    let actual = solid_gray(8, 8, 255);
    let r = GoldenReport::compare_gray("worst_case", &expected, &actual);
    assert_eq!(r.max_delta, 255);
    assert_abs_diff_eq!(r.exact_fraction, 0.0, epsilon = EPS);
}

#[test]
// F2 writes these rows into docs/GOLDEN_CALIBRATION.md, so a row must be a single line
// with the right column count and must start with the fixture name.
fn markdown_row_is_a_single_well_formed_line() {
    let img = solid_gray(4, 4, 1);
    let row = GoldenReport::compare_gray("nightmare", &img, &img).to_markdown_row();

    assert!(!row.contains('\n'), "a row must be one line: {row:?}");
    assert!(row.starts_with('|') && row.ends_with('|'), "{row:?}");
    assert!(row.contains("nightmare"));

    let header = GoldenReport::markdown_header();
    let header_cols = header.lines().next().unwrap().matches('|').count();
    assert_eq!(
        row.matches('|').count(),
        header_cols,
        "row and header column counts must agree\nheader: {header}\nrow: {row}"
    );
}

// ------------------------------------------------------------- thresholds

#[test]
// §11.7(B)12's tolerances, transcribed: SSIM >= 0.98, mean |delta| <= 1.0, max delta <= 8.
fn nlm_parity_thresholds_match_the_spec() {
    let t = GoldenThresholds::nlm_parity();
    assert_eq!(t.min_ssim, Some(0.98));
    assert_eq!(t.max_mean_abs_diff, Some(1.0));
    assert_eq!(t.max_delta, Some(8));
}

#[test]
// §12.7(A)2's tolerances, transcribed: max delta <= 6, SSIM >= 0.99.
fn jpeg_q95_thresholds_match_the_spec() {
    let t = GoldenThresholds::jpeg_q95();
    assert_eq!(t.max_delta, Some(6));
    assert_eq!(t.min_ssim, Some(0.99));
}

#[test]
// A perfect report meets every threshold.
fn assert_met_passes_on_a_perfect_report() {
    let img = solid_gray(16, 16, 100);
    let r = GoldenReport::compare_gray("perfect", &img, &img);
    GoldenThresholds::nlm_parity().assert_met(&r);
    GoldenThresholds::jpeg_q95().assert_met(&r);
    assert!(GoldenThresholds::nlm_parity().unmet(&r).is_empty());
}

#[test]
// A threshold that is not set is not checked -- a gate constrains only the metrics its
// spec section actually names.
fn unset_thresholds_are_not_checked() {
    let a = solid_gray(8, 8, 0);
    let b = solid_gray(8, 8, 255);
    let r = GoldenReport::compare_gray("bad", &a, &b);
    GoldenThresholds::default().assert_met(&r); // must not panic
}

#[test]
// Every unmet threshold is reported at once, with its measured value, so a parity
// failure does not require one CI run per metric.
fn unmet_lists_every_failing_metric_with_its_value() {
    let a = solid_gray(8, 8, 0);
    let b = solid_gray(8, 8, 255);
    let r = GoldenReport::compare_gray("bad", &a, &b);

    let unmet = GoldenThresholds::nlm_parity().unmet(&r);
    assert_eq!(
        unmet.len(),
        3,
        "expected ssim + mean + max to all fail: {unmet:?}"
    );
    let joined = unmet.join("\n");
    assert!(joined.contains("ssim"), "{joined}");
    assert!(
        joined.contains("255"),
        "the measured max delta must appear: {joined}"
    );
}

#[test]
// The panic message of a failed gate must name the metric -- a bare "assertion failed"
// is useless for a parity investigation.
#[should_panic(expected = "ssim")]
fn assert_met_panics_naming_the_failing_metric() {
    let a = solid_gray(8, 8, 0);
    let b = solid_gray(8, 8, 255);
    let r = GoldenReport::compare_gray("bad", &a, &b);
    GoldenThresholds::nlm_parity().assert_met(&r);
}

// --------------------------------------------------------- exact identity

#[test]
// §12.7(A)1 / §8.7(A)7 need a strict pixel-identity assertion, distinct from any
// tolerance-based one.
fn assert_images_identical_passes_on_equal_images() {
    let img = gray_from_rows(&[&[1, 2], &[3, 4]]);
    assert_images_identical_gray(&img, &img.clone());
}

#[test]
// The failure message must report how many pixels differ and where -- dumping buffers
// is useless on a 1000x8000 strip.
#[should_panic(expected = "1 pixel")]
fn assert_images_identical_reports_the_differing_pixel_count() {
    let a = gray_from_rows(&[&[1, 2], &[3, 4]]);
    let b = gray_from_rows(&[&[1, 2], &[3, 5]]);
    assert_images_identical_gray(&a, &b);
}

#[test]
// The standalone `is_within_dilated` helper agrees with the report's shape check.
fn is_within_dilated_matches_the_report() {
    let g = PixelSet::from_pixels((16, 16), [(5, 5)]);
    let o = PixelSet::from_pixels((16, 16), [(6, 5)]);
    assert!(is_within_dilated(&o, &g, 1));
    assert!(!is_within_dilated(&o, &g, 0));
}
