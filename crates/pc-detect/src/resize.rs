//! Task D2 -- spec §8.3 step 2: `ctd_interface.calculate_new_size_and_scale` plus an
//! INTER_AREA (box-filter) resampler.
//!
//! Bodies are `todo!()` skeletons: the signatures are frozen alongside the tests, the
//! implementations are task D2's job.
#![allow(unused_variables)]

use image::RgbImage;

/// Round half **away from zero** (spec §8.3 step 2). Python's `round()` is banker's
/// rounding; the spec pins away-from-zero and requires the `.5` boundary to be tested
/// explicitly, so this is a named function rather than an inline `.round()`.
pub fn round_half_away(value: f64) -> i64 {
    todo!("D2: round half away from zero")
}

/// spec §8.3 step 2, verbatim branching. Returns `(new_width, new_height, scale)`.
///
/// DEVIATION(1): in the `lower >= upper` branch upstream sets `new_height =
/// height_target_lower` while computing `scale` from `upper`, which is inconsistent
/// whenever `lower > upper`. We use `upper` for both; identical when `lower == upper`,
/// which is the only sane configuration. See spec §14.1.
pub fn calculate_new_size_and_scale(
    width: u32,
    height: u32,
    target_lower: u32,
    target_upper: u32,
) -> (u32, u32, f64) {
    todo!("D2: port calculate_new_size_and_scale")
}

/// Area-average (box filter) downscale matching `cv2.INTER_AREA` (spec §8.3 step 2):
/// each output pixel is the mean of the source rectangle
/// `[x*sx, (x+1)*sx) x [y*sy, (y+1)*sy)` **with fractional edge weights**, accumulated
/// in f32 and rounded half-away-from-zero to u8.
///
/// `image::imageops::resize` must not be substituted: neither `Triangle` nor `Lanczos3`
/// is INTER_AREA and the difference is visible (spec §8.3 step 2).
pub fn resize_area(image: &RgbImage, new_w: u32, new_h: u32) -> RgbImage {
    todo!("D2: INTER_AREA box-filter resize")
}
