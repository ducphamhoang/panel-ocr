//! Loading and hand-construction helpers. Every loader panics with the offending path
//! in the message -- a fixture path typo must not read as a decode bug.

use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
use std::path::Path;

pub fn load(path: impl AsRef<Path>) -> DynamicImage {
    let path = path.as_ref();
    image::open(path)
        .unwrap_or_else(|error| panic!("failed to decode image `{}`: {error}", path.display()))
}

pub fn load_luma8(path: impl AsRef<Path>) -> GrayImage {
    load(path).to_luma8()
}

pub fn load_rgb8(path: impl AsRef<Path>) -> RgbImage {
    load(path).to_rgb8()
}

pub fn load_rgba8(path: impl AsRef<Path>) -> RgbaImage {
    load(path).to_rgba8()
}

/// A `w x h` image filled with `value`.
pub fn solid_gray(w: u32, h: u32, value: u8) -> GrayImage {
    GrayImage::from_pixel(w, h, image::Luma([value]))
}

pub fn solid_rgba(w: u32, h: u32, px: [u8; 4]) -> RgbaImage {
    RgbaImage::from_pixel(w, h, image::Rgba(px))
}

/// Build a grayscale image from literal rows -- the workhorse for hand-computed tests.
/// Panics if the rows are not all the same length or the image would be empty.
///
/// ```ignore
/// let img = gray_from_rows(&[&[0, 255], &[255, 0]]);
/// ```
pub fn gray_from_rows(rows: &[&[u8]]) -> GrayImage {
    assert!(!rows.is_empty(), "image rows must not be empty");
    let width = rows[0].len();
    assert!(width > 0, "image rows must not be empty");
    assert!(
        rows.iter().all(|row| row.len() == width),
        "image rows must all have the same length"
    );
    let pixels = rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    GrayImage::from_raw(width as u32, rows.len() as u32, pixels)
        .expect("validated grayscale row dimensions")
}

/// Same, for RGBA.
pub fn rgba_from_rows(rows: &[&[[u8; 4]]]) -> RgbaImage {
    assert!(!rows.is_empty(), "image rows must not be empty");
    let width = rows[0].len();
    assert!(width > 0, "image rows must not be empty");
    assert!(
        rows.iter().all(|row| row.len() == width),
        "image rows must all have the same length"
    );
    let pixels = rows
        .iter()
        .flat_map(|row| row.iter().flat_map(|pixel| pixel.iter().copied()))
        .collect::<Vec<_>>();
    RgbaImage::from_raw(width as u32, rows.len() as u32, pixels)
        .expect("validated RGBA row dimensions")
}

/// An 8x8 checkerboard of `a`/`b`, `a` at (0,0). Used by the SSIM tests as a pattern
/// with an exactly known mean and variance.
pub fn checkerboard(w: u32, h: u32, a: u8, b: u8) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| {
        image::Luma([if (x + y) % 2 == 0 { a } else { b }])
    })
}

/// Deterministic (seeded, not `rand`) additive Gaussian-ish noise, for §11.7(A)2.
/// Same seed => same image on every platform.
pub fn noisy_gray(w: u32, h: u32, mean: u8, sigma: f64, seed: u64) -> GrayImage {
    assert!(
        sigma.is_finite() && sigma >= 0.0,
        "sigma must be finite and non-negative"
    );
    let mut state = seed;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = state >> 11;
        (bits as f64 + 0.5) / ((1_u64 << 53) as f64)
    };
    GrayImage::from_fn(w, h, |_x, _y| {
        let u1 = uniform();
        let u2 = uniform();
        let standard_normal = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let value = (f64::from(mean) + sigma * standard_normal)
            .round()
            .clamp(0.0, 255.0);
        image::Luma([value as u8])
    })
}
