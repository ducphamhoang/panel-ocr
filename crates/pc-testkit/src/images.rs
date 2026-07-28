//! Loading and hand-construction helpers. Every loader panics with the offending path
//! in the message -- a fixture path typo must not read as a decode bug.

use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
use std::path::Path;

pub fn load(path: impl AsRef<Path>) -> DynamicImage {
    todo!()
}

pub fn load_luma8(path: impl AsRef<Path>) -> GrayImage {
    todo!()
}

pub fn load_rgb8(path: impl AsRef<Path>) -> RgbImage {
    todo!()
}

pub fn load_rgba8(path: impl AsRef<Path>) -> RgbaImage {
    todo!()
}

/// A `w x h` image filled with `value`.
pub fn solid_gray(w: u32, h: u32, value: u8) -> GrayImage {
    todo!()
}

pub fn solid_rgba(w: u32, h: u32, px: [u8; 4]) -> RgbaImage {
    todo!()
}

/// Build a grayscale image from literal rows -- the workhorse for hand-computed tests.
/// Panics if the rows are not all the same length or the image would be empty.
///
/// ```ignore
/// let img = gray_from_rows(&[&[0, 255], &[255, 0]]);
/// ```
pub fn gray_from_rows(rows: &[&[u8]]) -> GrayImage {
    todo!()
}

/// Same, for RGBA.
pub fn rgba_from_rows(rows: &[&[[u8; 4]]]) -> RgbaImage {
    todo!()
}

/// An 8x8 checkerboard of `a`/`b`, `a` at (0,0). Used by the SSIM tests as a pattern
/// with an exactly known mean and variance.
pub fn checkerboard(w: u32, h: u32, a: u8, b: u8) -> GrayImage {
    todo!()
}

/// Deterministic (seeded, not `rand`) additive Gaussian-ish noise, for §11.7(A)2.
/// Same seed => same image on every platform.
pub fn noisy_gray(w: u32, h: u32, mean: u8, sigma: f64, seed: u64) -> GrayImage {
    todo!()
}
