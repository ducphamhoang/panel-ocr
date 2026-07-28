//! Task D2 -- spec §8.3 step 2: `ctd_interface.calculate_new_size_and_scale` plus an
//! INTER_AREA (box-filter) resampler.
//!
//! Implemented (task D2); the signatures are frozen alongside the tests.

use image::RgbImage;

/// Round half **away from zero** (spec §8.3 step 2). Python's `round()` is banker's
/// rounding; the spec pins away-from-zero and requires the `.5` boundary to be tested
/// explicitly, so this is a named function rather than an inline `.round()`.
pub fn round_half_away(value: f64) -> i64 {
    if value.is_sign_negative() {
        (value - 0.5).ceil() as i64
    } else {
        (value + 0.5).floor() as i64
    }
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
    if target_lower == 0 || target_upper == 0 || height <= target_upper {
        return (width, height, 1.0);
    }

    if target_lower >= target_upper {
        let scale = f64::from(target_upper) / f64::from(height);
        // DEVIATION(1): upstream uses `target_lower` for the height while deriving
        // the scale from `target_upper`; use `target_upper` consistently (§14.1).
        return (
            round_half_away(f64::from(width) * scale) as u32,
            target_upper,
            scale,
        );
    }

    let inverse_lower = f64::from(height) / f64::from(target_lower);
    let inverse_upper = f64::from(height) / f64::from(target_upper);
    let inverse_scale = inverse_upper.ceil();

    if inverse_scale <= inverse_lower {
        let scale = 1.0 / inverse_scale;
        (
            round_half_away(f64::from(width) * scale) as u32,
            round_half_away(f64::from(height) * scale) as u32,
            scale,
        )
    } else {
        let max_height = round_half_away(f64::from(height) / inverse_upper) as u32;
        let min_height = round_half_away(f64::from(height) / inverse_lower) as u32;
        let multiple_of_four = (max_height / 4) * 4;
        let new_height = if multiple_of_four < min_height {
            max_height
        } else {
            multiple_of_four
        };
        let scale = f64::from(new_height) / f64::from(height);
        (
            round_half_away(f64::from(width) * scale) as u32,
            new_height,
            scale,
        )
    }
}

/// Area-average (box filter) downscale matching `cv2.INTER_AREA` (spec §8.3 step 2):
/// each output pixel is the mean of the source rectangle
/// `[x*sx, (x+1)*sx) x [y*sy, (y+1)*sy)` **with fractional edge weights**, accumulated
/// in f32 and rounded half-away-from-zero to u8.
///
/// `image::imageops::resize` must not be substituted: neither `Triangle` nor `Lanczos3`
/// is INTER_AREA and the difference is visible (spec §8.3 step 2).
pub fn resize_area(image: &RgbImage, new_w: u32, new_h: u32) -> RgbImage {
    let (source_w, source_h) = image.dimensions();
    if (source_w, source_h) == (new_w, new_h) {
        return image.clone();
    }
    if new_w == 0 || new_h == 0 || source_w == 0 || source_h == 0 {
        return RgbImage::new(new_w, new_h);
    }

    let scale_x = source_w as f32 / new_w as f32;
    let scale_y = source_h as f32 / new_h as f32;
    let area = scale_x * scale_y;

    RgbImage::from_fn(new_w, new_h, |output_x, output_y| {
        let left = output_x as f32 * scale_x;
        let right = (output_x + 1) as f32 * scale_x;
        let top = output_y as f32 * scale_y;
        let bottom = (output_y + 1) as f32 * scale_y;
        let first_x = left.floor() as u32;
        let last_x = right.ceil().min(source_w as f32) as u32;
        let first_y = top.floor() as u32;
        let last_y = bottom.ceil().min(source_h as f32) as u32;
        let mut sums = [0.0_f32; 3];

        for source_y in first_y..last_y {
            let y_weight = bottom.min((source_y + 1) as f32) - top.max(source_y as f32);
            for source_x in first_x..last_x {
                let x_weight = right.min((source_x + 1) as f32) - left.max(source_x as f32);
                let weight = x_weight * y_weight;
                let pixel = image.get_pixel(source_x, source_y).0;
                for channel in 0..3 {
                    sums[channel] += f32::from(pixel[channel]) * weight;
                }
            }
        }

        image::Rgb(sums.map(|sum| round_half_away(f64::from(sum / area)) as u8))
    })
}
