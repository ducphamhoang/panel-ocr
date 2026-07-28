//! Task N2 -- separable Gaussian blur (spec §11.3 step 5, §14.6, §16.10 item 12).
//!
//! DEVIATION(6): PIL's `GaussianBlur(radius=r)` is a **three-pass box-blur
//! approximation**, not a true Gaussian. v1 uses a true separable Gaussian with
//! `sigma = radius`, truncated at `3*sigma`. With the default `noise_fade_radius = 1`
//! the difference is a few levels on a soft edge, well inside the stage tolerance.
//!
//! Pinned by §16.10 item 12: taps `exp(-j^2 / (2*sigma^2))` for `j in -k..=k` with
//! `k = ceil(3*sigma)`, normalised to sum 1 in `f64`; horizontal pass into an `f32`
//! intermediate with **no** intermediate rounding, then the vertical pass, then
//! `clamp(floor(v + 0.5), 0, 255)`. Borders **replicate** (clamp to edge), matching
//! PIL -- deliberately NOT the reflect-101 used inside `nlm` (§16.10 item 12).

use image::GrayImage;

/// The normalised, truncated 1-D Gaussian taps for `radius`, in `f64`.
///
/// `radius == 0` yields the single tap `[1.0]` (the identity kernel), which is what
/// makes `blur(_, 0)` a clone rather than a special case buried in the caller.
pub fn taps(radius: u32) -> Vec<f64> {
    if radius == 0 {
        return vec![1.0];
    }
    let sigma = f64::from(radius);
    let half_width = (3.0 * sigma).ceil() as i64;
    let two_sigma_squared = 2.0 * sigma * sigma;
    let mut weights: Vec<f64> = (-half_width..=half_width)
        .map(|j| (-((j * j) as f64) / two_sigma_squared).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

/// spec §11.3 step 5: a true separable Gaussian with `sigma = radius`, truncated at
/// `3*sigma`, replicate borders. `radius == 0` is the identity.
pub fn blur(image: &GrayImage, radius: u32) -> GrayImage {
    if radius == 0 || image.width() == 0 || image.height() == 0 {
        return image.clone();
    }
    let weights = taps(radius);
    let half_width = (weights.len() / 2) as i64;
    let (width, height) = image.dimensions();
    let (w, h) = (width as i64, height as i64);

    // Horizontal pass -> f32 intermediate, no rounding (§16.10 item 12).
    let mut horizontal = vec![0.0_f32; (width as usize) * (height as usize)];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f64;
            for (index, weight) in weights.iter().enumerate() {
                let source_x = (x + index as i64 - half_width).clamp(0, w - 1);
                sum += weight * f64::from(image.get_pixel(source_x as u32, y as u32).0[0]);
            }
            horizontal[(y as usize) * (width as usize) + (x as usize)] = sum as f32;
        }
    }

    // Vertical pass -> rounded u8.
    GrayImage::from_fn(width, height, |x, y| {
        let mut sum = 0.0_f64;
        for (index, weight) in weights.iter().enumerate() {
            let source_y = (i64::from(y) + index as i64 - half_width).clamp(0, h - 1);
            sum += weight
                * f64::from(horizontal[(source_y as usize) * (width as usize) + (x as usize)]);
        }
        image::Luma([(sum + 0.5).floor().clamp(0.0, 255.0) as u8])
    })
}
