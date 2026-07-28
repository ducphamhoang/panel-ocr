//! Image comparison metrics -- spec §10.7(B) (shape IoU + value metrics), §11.7(B)12/13
//! (SSIM / mean |delta| / max delta), §12.7(A)2 (SSIM + max delta).
//!
//! # Definitions are part of the contract (spec §16.5 item 7)
//!
//! The spec says "SSIM (8x8 windows, grayscale)" and "IoU" without pinning the
//! estimator. The choices below are frozen by this module's own tests, because the
//! gate numbers (`SSIM >= 0.98`, `IoU >= 0.99`) are only meaningful relative to a
//! fixed definition:
//!
//! * **SSIM** -- non-overlapping 8x8 tiles (NOT a sliding window), uniform (unweighted)
//!   window, **population** variance/covariance (divide by N, not N-1), partial edge
//!   tiles included at their actual size, and the global score is the **unweighted**
//!   mean over tiles. `K1 = 0.01`, `K2 = 0.03`, `L = 255`, so
//!   `C1 = 6.5025`, `C2 = 58.5225`.
//! * **IoU** -- over pixel *sets*; `IoU(empty, empty) == 1.0` (perfect agreement),
//!   `IoU(empty, non-empty) == 0.0`.
//! * **dilate** -- Chebyshev (square) structuring element, clamped to image bounds.
//! * **mean |delta| / max delta** -- typed per colour model. `_rgba` variants include
//!   the alpha channel; that is documented, not incidental.
//!
//! All functions **panic** on mismatched dimensions.

use image::{GrayImage, RgbImage, RgbaImage};
use std::collections::BTreeSet;

// ------------------------------------------------------------------- SSIM

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsimParams {
    /// Tile edge length in pixels. Spec says 8.
    pub window: u32,
    pub k1: f64,
    pub k2: f64,
    /// Dynamic range `L`; 255.0 for 8-bit.
    pub dynamic_range: f64,
}

impl Default for SsimParams {
    /// `window = 8`, `k1 = 0.01`, `k2 = 0.03`, `L = 255.0` (§10.7(B), §11.7(B)).
    fn default() -> Self {
        Self {
            window: 8,
            k1: 0.01,
            k2: 0.03,
            dynamic_range: 255.0,
        }
    }
}

impl SsimParams {
    /// `(k1 * L)^2`
    pub fn c1(&self) -> f64 {
        (self.k1 * self.dynamic_range).powi(2)
    }
    /// `(k2 * L)^2`
    pub fn c2(&self) -> f64 {
        (self.k2 * self.dynamic_range).powi(2)
    }
}

/// Global SSIM with the default 8x8 parameters. Panics on dimension mismatch.
pub fn ssim_gray(a: &GrayImage, b: &GrayImage) -> f64 {
    ssim_gray_with(a, b, SsimParams::default())
}

pub fn ssim_gray_with(a: &GrayImage, b: &GrayImage, params: SsimParams) -> f64 {
    assert_same_dimensions(a.dimensions(), b.dimensions());
    assert!(params.window > 0, "SSIM window must be greater than zero");
    let (width, height) = a.dimensions();
    assert!(width > 0 && height > 0, "SSIM images must not be empty");

    let mut sum = 0.0;
    let mut count = 0_u64;
    for tile_y in (0..height).step_by(params.window as usize) {
        let y_end = tile_y.saturating_add(params.window).min(height);
        for tile_x in (0..width).step_by(params.window as usize) {
            let x_end = tile_x.saturating_add(params.window).min(width);
            let capacity = ((x_end - tile_x) * (y_end - tile_y)) as usize;
            let mut a_tile = Vec::with_capacity(capacity);
            let mut b_tile = Vec::with_capacity(capacity);
            for y in tile_y..y_end {
                for x in tile_x..x_end {
                    a_tile.push(a.get_pixel(x, y)[0]);
                    b_tile.push(b.get_pixel(x, y)[0]);
                }
            }
            sum += ssim_tile(&a_tile, &b_tile, params);
            count += 1;
        }
    }
    (sum / count as f64).clamp(-1.0, 1.0)
}

/// SSIM of a single tile. Exposed so the tile-averaging behaviour is directly testable.
pub fn ssim_tile(a: &[u8], b: &[u8], params: SsimParams) -> f64 {
    assert_eq!(a.len(), b.len(), "SSIM tile dimension mismatch");
    assert!(!a.is_empty(), "SSIM tile must not be empty");
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&value| f64::from(value)).sum::<f64>() / n;
    let mean_b = b.iter().map(|&value| f64::from(value)).sum::<f64>() / n;

    let (variance_a, variance_b, covariance) =
        a.iter()
            .zip(b)
            .fold((0.0, 0.0, 0.0), |(va, vb, cov), (&x, &y)| {
                let dx = f64::from(x) - mean_a;
                let dy = f64::from(y) - mean_b;
                (va + dx * dx, vb + dy * dy, cov + dx * dy)
            });
    let variance_a = variance_a / n;
    let variance_b = variance_b / n;
    let covariance = covariance / n;

    let numerator = (2.0 * mean_a * mean_b + params.c1()) * (2.0 * covariance + params.c2());
    let denominator =
        (mean_a * mean_a + mean_b * mean_b + params.c1()) * (variance_a + variance_b + params.c2());
    (numerator / denominator).clamp(-1.0, 1.0)
}

/// Convenience for RGB inputs: converts both to luma8 first (spec says "grayscale").
pub fn ssim_rgb_as_gray(a: &RgbImage, b: &RgbImage) -> f64 {
    assert_same_dimensions(a.dimensions(), b.dimensions());
    let a = image::DynamicImage::ImageRgb8(a.clone()).to_luma8();
    let b = image::DynamicImage::ImageRgb8(b.clone()).to_luma8();
    ssim_gray(&a, &b)
}

// ------------------------------------------------------------ value metrics

/// Mean over all pixels of `|a - b|`.
pub fn mean_abs_diff_gray(a: &GrayImage, b: &GrayImage) -> f64 {
    mean_abs_diff(a.dimensions(), b.dimensions(), a.as_raw(), b.as_raw())
}

/// Mean over all pixels **and all 3 channels**.
pub fn mean_abs_diff_rgb(a: &RgbImage, b: &RgbImage) -> f64 {
    mean_abs_diff(a.dimensions(), b.dimensions(), a.as_raw(), b.as_raw())
}

/// Mean over all pixels and all **4** channels -- alpha participates (spec §16.5 item 10).
pub fn mean_abs_diff_rgba(a: &RgbaImage, b: &RgbaImage) -> f64 {
    mean_abs_diff(a.dimensions(), b.dimensions(), a.as_raw(), b.as_raw())
}

pub fn max_delta_gray(a: &GrayImage, b: &GrayImage) -> u8 {
    max_delta(a.dimensions(), b.dimensions(), a.as_raw(), b.as_raw())
}

/// Max over all pixels and channels of `|a_c - b_c|` -- "max per-channel delta"
/// (§11.7(B)12, §12.7(A)2).
pub fn max_delta_rgb(a: &RgbImage, b: &RgbImage) -> u8 {
    max_delta(a.dimensions(), b.dimensions(), a.as_raw(), b.as_raw())
}

pub fn max_delta_rgba(a: &RgbaImage, b: &RgbaImage) -> u8 {
    max_delta(a.dimensions(), b.dimensions(), a.as_raw(), b.as_raw())
}

/// Fraction of pixels that are exactly equal in every channel -- §10.7(B)'s
/// "% of pixels exactly equal". Range `0.0..=1.0`.
pub fn exact_equal_fraction_gray(a: &GrayImage, b: &GrayImage) -> f64 {
    exact_equal_fraction(a.dimensions(), b.dimensions(), a.as_raw(), b.as_raw(), 1)
}

pub fn exact_equal_fraction_rgba(a: &RgbaImage, b: &RgbaImage) -> f64 {
    exact_equal_fraction(a.dimensions(), b.dimensions(), a.as_raw(), b.as_raw(), 4)
}

fn assert_same_dimensions(a: (u32, u32), b: (u32, u32)) {
    assert_eq!(a, b, "image dimension mismatch: {a:?} vs {b:?}");
}

fn mean_abs_diff(a_dims: (u32, u32), b_dims: (u32, u32), a: &[u8], b: &[u8]) -> f64 {
    assert_same_dimensions(a_dims, b_dims);
    if a.is_empty() {
        return 0.0;
    }
    let sum = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)))
        .sum::<u64>();
    sum as f64 / a.len() as f64
}

fn max_delta(a_dims: (u32, u32), b_dims: (u32, u32), a: &[u8], b: &[u8]) -> u8 {
    assert_same_dimensions(a_dims, b_dims);
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

fn exact_equal_fraction(
    a_dims: (u32, u32),
    b_dims: (u32, u32),
    a: &[u8],
    b: &[u8],
    channels: usize,
) -> f64 {
    assert_same_dimensions(a_dims, b_dims);
    let pixel_count = (u64::from(a_dims.0) * u64::from(a_dims.1)) as usize;
    if pixel_count == 0 {
        return 1.0;
    }
    let equal = a
        .chunks_exact(channels)
        .zip(b.chunks_exact(channels))
        .filter(|(x, y)| x == y)
        .count();
    equal as f64 / pixel_count as f64
}

// -------------------------------------------------------------- pixel sets

/// A set of `(x, y)` coordinates within a known canvas. `BTreeSet`, not `HashSet`, so
/// iteration order is deterministic (§5.7) and failure messages are reproducible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelSet {
    dims: (u32, u32),
    pixels: BTreeSet<(u32, u32)>,
}

impl PixelSet {
    pub fn new(dims: (u32, u32)) -> Self {
        Self {
            dims,
            pixels: BTreeSet::new(),
        }
    }

    /// Panics if any coordinate is outside `dims`.
    pub fn from_pixels(dims: (u32, u32), pixels: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let mut set = Self::new(dims);
        for pixel in pixels {
            set.insert(pixel);
        }
        set
    }

    pub fn dims(&self) -> (u32, u32) {
        self.dims
    }
    pub fn len(&self) -> usize {
        self.pixels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }
    pub fn contains(&self, p: (u32, u32)) -> bool {
        self.pixels.contains(&p)
    }
    /// Returns `true` if newly inserted. Panics if `p` is out of bounds.
    pub fn insert(&mut self, p: (u32, u32)) -> bool {
        assert!(
            p.0 < self.dims.0 && p.1 < self.dims.1,
            "pixel {p:?} is out of bounds for dimensions {:?}",
            self.dims
        );
        self.pixels.insert(p)
    }
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.pixels.iter().copied()
    }

    /// Panics if `dims` differ.
    pub fn intersection_len(&self, other: &PixelSet) -> usize {
        self.assert_same_dims(other);
        self.pixels.intersection(&other.pixels).count()
    }
    pub fn union_len(&self, other: &PixelSet) -> usize {
        self.assert_same_dims(other);
        self.pixels.union(&other.pixels).count()
    }
    pub fn is_subset_of(&self, other: &PixelSet) -> bool {
        self.assert_same_dims(other);
        self.pixels.is_subset(&other.pixels)
    }

    /// Chebyshev (square) dilation by `radius`, clamped to the canvas (spec §16.5
    /// item 8). `radius == 0` is the identity.
    pub fn dilate(&self, radius: u32) -> PixelSet {
        let mut dilated = PixelSet::new(self.dims);
        for (x, y) in self.iter() {
            let x_start = x.saturating_sub(radius);
            let y_start = y.saturating_sub(radius);
            let x_end = x.saturating_add(radius).min(self.dims.0 - 1);
            let y_end = y.saturating_add(radius).min(self.dims.1 - 1);
            for dy in y_start..=y_end {
                for dx in x_start..=x_end {
                    dilated.insert((dx, dy));
                }
            }
        }
        dilated
    }

    fn assert_same_dims(&self, other: &PixelSet) {
        assert_eq!(
            self.dims, other.dims,
            "pixel-set dimension mismatch: {:?} vs {:?}",
            self.dims, other.dims
        );
    }
}

/// `IoU = |A intersect B| / |A union B|`, with `IoU(empty, empty) = 1.0` (spec §16.5
/// item 9). Panics if the two sets have different `dims`.
pub fn iou(a: &PixelSet, b: &PixelSet) -> f64 {
    let union = a.union_len(b);
    if union == 0 {
        1.0
    } else {
        a.intersection_len(b) as f64 / union as f64
    }
}

/// `{ p : a[p] != b[p] }` -- §10.7(B)'s `G` and `O` sets.
pub fn diff_set_gray(a: &GrayImage, b: &GrayImage) -> PixelSet {
    assert_same_dimensions(a.dimensions(), b.dimensions());
    let mut set = PixelSet::new(a.dimensions());
    for (x, y, pixel) in a.enumerate_pixels() {
        if pixel != b.get_pixel(x, y) {
            set.insert((x, y));
        }
    }
    set
}

pub fn diff_set_rgba(a: &RgbaImage, b: &RgbaImage) -> PixelSet {
    assert_same_dimensions(a.dimensions(), b.dimensions());
    let mut set = PixelSet::new(a.dimensions());
    for (x, y, pixel) in a.enumerate_pixels() {
        if pixel != b.get_pixel(x, y) {
            set.insert((x, y));
        }
    }
    set
}
