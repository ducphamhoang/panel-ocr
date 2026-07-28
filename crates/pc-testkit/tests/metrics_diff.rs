//! C4 tests -- mean absolute difference, max per-channel delta, exact-equal fraction.
//! Frozen gates. Every expected value is hand-computed in its comment.
//!
//! These are the numbers §11.7(B)12 (`mean |delta| <= 1.0`, `max delta <= 8`),
//! §12.7(A)2 (`max delta <= 6`) and §10.7(B) ("% exactly equal", "max per-pixel
//! delta") are written in terms of, so their exact definitions are pinned here
//! (spec §16.5 item 10).

use approx::assert_abs_diff_eq;
use pc_testkit::images::{gray_from_rows, rgba_from_rows, solid_gray, solid_rgba};
use pc_testkit::metrics::{
    exact_equal_fraction_gray, exact_equal_fraction_rgba, max_delta_gray, max_delta_rgb,
    max_delta_rgba, mean_abs_diff_gray, mean_abs_diff_rgb, mean_abs_diff_rgba,
};

const EPS: f64 = 1e-12;

#[test]
// Identity: an image against itself has zero difference on every metric and is 100%
// exactly equal.
fn identical_images_score_zero_difference() {
    let img = gray_from_rows(&[&[0, 127, 255], &[3, 3, 3]]);
    assert_abs_diff_eq!(mean_abs_diff_gray(&img, &img), 0.0, epsilon = EPS);
    assert_eq!(max_delta_gray(&img, &img), 0);
    assert_abs_diff_eq!(exact_equal_fraction_gray(&img, &img), 1.0, epsilon = EPS);
}

#[test]
// Hand-computed 2x2 grayscale:
//   a = [0, 10, 20, 30], b = [0, 12, 17, 30]
//   |delta| = [0, 2, 3, 0]  ->  mean = 5/4 = 1.25, max = 3
//   exactly equal: 2 of 4 pixels -> 0.5
fn gray_metrics_hand_computed() {
    let a = gray_from_rows(&[&[0, 10], &[20, 30]]);
    let b = gray_from_rows(&[&[0, 12], &[17, 30]]);
    assert_abs_diff_eq!(mean_abs_diff_gray(&a, &b), 1.25, epsilon = EPS);
    assert_eq!(max_delta_gray(&a, &b), 3);
    assert_abs_diff_eq!(exact_equal_fraction_gray(&a, &b), 0.5, epsilon = EPS);
}

#[test]
// Difference is absolute and therefore symmetric -- b - a must not underflow to a huge
// value, which is the classic u8-subtraction bug this test exists to catch.
fn difference_is_absolute_and_symmetric() {
    let a = gray_from_rows(&[&[0, 255]]);
    let b = gray_from_rows(&[&[255, 0]]);
    assert_eq!(max_delta_gray(&a, &b), 255);
    assert_eq!(max_delta_gray(&b, &a), 255);
    assert_abs_diff_eq!(
        mean_abs_diff_gray(&a, &b),
        mean_abs_diff_gray(&b, &a),
        epsilon = EPS
    );
    assert_abs_diff_eq!(mean_abs_diff_gray(&a, &b), 255.0, epsilon = EPS);
}

#[test]
// Hand-computed 2x1 RGB -- "max PER-CHANNEL delta" means the max over channels, not
// over some per-pixel aggregate:
//   a = [(0,0,0), (255,255,255)], b = [(4,0,9), (250,255,255)]
//   |delta| per channel = [4,0,9, 5,0,0] -> max = 9, mean = 18/6 = 3.0
fn rgb_metrics_hand_computed() {
    let a = image::RgbImage::from_raw(2, 1, vec![0, 0, 0, 255, 255, 255]).unwrap();
    let b = image::RgbImage::from_raw(2, 1, vec![4, 0, 9, 250, 255, 255]).unwrap();
    assert_eq!(max_delta_rgb(&a, &b), 9);
    assert_abs_diff_eq!(mean_abs_diff_rgb(&a, &b), 3.0, epsilon = EPS);
}

#[test]
// spec §16.5 item 10 (decided here): the RGBA variants INCLUDE the alpha channel in
// both the mean and the max, so the divisor is 4 channels, not 3.
//   a = [(10,10,10,255)], b = [(10,10,10,251)]
//   |delta| = [0,0,0,4] -> mean = 4/4 = 1.0, max = 4
// If alpha were excluded, the mean would be 0.0 -- so this test distinguishes the two.
fn rgba_metrics_include_the_alpha_channel() {
    let a = rgba_from_rows(&[&[[10, 10, 10, 255]]]);
    let b = rgba_from_rows(&[&[[10, 10, 10, 251]]]);
    assert_eq!(max_delta_rgba(&a, &b), 4);
    assert_abs_diff_eq!(mean_abs_diff_rgba(&a, &b), 1.0, epsilon = EPS);
}

#[test]
// "Exactly equal" for RGBA means all four channels equal -- a pixel differing only in
// alpha is NOT exactly equal.
fn rgba_exact_equality_requires_every_channel() {
    let a = rgba_from_rows(&[&[[1, 2, 3, 255], [9, 9, 9, 255]]]);
    let b = rgba_from_rows(&[&[[1, 2, 3, 254], [9, 9, 9, 255]]]);
    assert_abs_diff_eq!(exact_equal_fraction_rgba(&a, &b), 0.5, epsilon = EPS);
}

#[test]
// Large flat images: mean must not overflow or lose precision through u8 accumulation.
// 256x256 pixels, every one differing by exactly 200 -> mean is exactly 200.0.
fn mean_does_not_overflow_on_large_images() {
    let a = solid_gray(256, 256, 0);
    let b = solid_gray(256, 256, 200);
    assert_abs_diff_eq!(mean_abs_diff_gray(&a, &b), 200.0, epsilon = EPS);
    assert_eq!(max_delta_gray(&a, &b), 200);
}

#[test]
// A single outlier pixel drives max delta but barely moves the mean -- this is
// exactly the discrimination §11.7(B)12's tolerance rationale relies on ("isolated
// pixels" vs "systematic bias"), so it must actually hold.
fn max_and_mean_measure_different_things() {
    let a = solid_gray(10, 10, 100);
    let mut b = solid_gray(10, 10, 100);
    b.put_pixel(5, 5, image::Luma([108]));
    assert_eq!(max_delta_gray(&a, &b), 8);
    assert_abs_diff_eq!(mean_abs_diff_gray(&a, &b), 8.0 / 100.0, epsilon = EPS);
}

#[test]
#[should_panic(expected = "dimension")]
fn gray_metrics_panic_on_dimension_mismatch() {
    let _ = mean_abs_diff_gray(&solid_gray(4, 4, 0), &solid_gray(4, 5, 0));
}

#[test]
#[should_panic(expected = "dimension")]
fn rgba_metrics_panic_on_dimension_mismatch() {
    let _ = max_delta_rgba(&solid_rgba(4, 4, [0; 4]), &solid_rgba(5, 4, [0; 4]));
}
