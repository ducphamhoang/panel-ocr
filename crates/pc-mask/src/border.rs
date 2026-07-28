//! Task M3 -- edge extraction and border statistics (spec §10.3's
//! `border_std_deviation`, §10.7(A)4-7, §16.9 items 7, 8).
//!
//! **Precision rule (§10.3, §15.9, decided): `f64` only.** No `f32` may appear in this
//! module or in `fit.rs`. Upstream casts to `np.float64` explicitly and the candidate
//! comparison that decides which mask paints the page is pure `f64`; a borderline `f32`
//! rounding flip would change visible output.

use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
use pc_core::Rect;
use pc_imageops::BinaryMask;

/// `||new - old|| < 1e-5` stops Weiszfeld's iteration (§10.3 step 6 of the colour path).
pub const WEISZFELD_EPSILON: f64 = 1e-5;
/// Hard iteration cap (§10.3).
pub const WEISZFELD_MAX_ITERATIONS: usize = 500;

/// The whole region was unusable: the candidate mask has no edge pixels at all, so
/// there is nothing to measure. Upstream's `BlankMaskError`; `fit_region` turns this
/// into `None` for the entire region (§10.3 step 8, §16.9 item 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlankMask;

/// What `border_std_deviation` returns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BorderStats {
    pub std_deviation: f64,
    pub median_color: [u8; 3],
}

/// PIL's ITU-R 601-2 luma transform with integer truncation:
/// `L = (R*299 + G*587 + B*114) / 1000`.
///
/// The `image` crate's `to_luma8` uses different coefficients -- **do not** use it on
/// this path (§10.3 step 1 of `border_std_deviation`).
pub fn pil_luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114) / 1000) as u8
}

/// The analysis canvas a region is scored against (§16.9 item 7). `L`/`LA` sources
/// become [`BaseCanvas::Gray`]; everything else becomes [`BaseCanvas::Rgb`].
#[derive(Debug, Clone, PartialEq)]
pub enum BaseCanvas {
    Gray(GrayImage),
    Rgb(RgbImage),
}

impl BaseCanvas {
    pub fn from_dynamic(image: &DynamicImage) -> BaseCanvas {
        match image {
            DynamicImage::ImageLuma8(_)
            | DynamicImage::ImageLumaA8(_)
            | DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageLumaA16(_) => BaseCanvas::Gray(image.to_luma8()),
            other => BaseCanvas::Rgb(other.to_rgb8()),
        }
    }

    pub fn is_gray(&self) -> bool {
        matches!(self, BaseCanvas::Gray(_))
    }

    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            BaseCanvas::Gray(image) => image.dimensions(),
            BaseCanvas::Rgb(image) => image.dimensions(),
        }
    }

    /// `None` when `rect` is degenerate or fully outside the canvas (§16.9 item 11 --
    /// the caller turns that into a skipped region, never an error).
    pub fn crop(&self, rect: Rect) -> Option<BaseCanvas> {
        let (x, y, width, height) = rect.to_crop(self.dimensions())?;
        Some(match self {
            BaseCanvas::Gray(image) => {
                BaseCanvas::Gray(image::imageops::crop_imm(image, x, y, width, height).to_image())
            }
            BaseCanvas::Rgb(image) => {
                BaseCanvas::Rgb(image::imageops::crop_imm(image, x, y, width, height).to_image())
            }
        })
    }

    /// The `[R, G, B]` at `(x, y)`; a `Gray` canvas replicates its value.
    pub fn color_at(&self, x: u32, y: u32) -> [u8; 3] {
        match self {
            BaseCanvas::Gray(image) => {
                let Luma([value]) = *image.get_pixel(x, y);
                [value, value, value]
            }
            BaseCanvas::Rgb(image) => {
                let Rgb(channels) = *image.get_pixel(x, y);
                channels
            }
        }
    }

    /// The value the grayscale path measures: the stored value for `Gray`, PIL luma for
    /// `Rgb` (§16.9 item 7).
    pub fn luma_at(&self, x: u32, y: u32) -> u8 {
        match self {
            BaseCanvas::Gray(image) => image.get_pixel(x, y).0[0],
            BaseCanvas::Rgb(image) => {
                let Rgb([r, g, b]) = *image.get_pixel(x, y);
                pil_luma(r, g, b)
            }
        }
    }
}

/// spec §10.3 step 2 of `border_std_deviation` -- PIL's `FIND_EDGES` on a mode-`1`
/// mask, reduced to its exact meaning:
///
/// ```text
/// edge(p) = mask[p] && (p is on the 1-pixel image border || any 8-neighbour is 0)
/// ```
///
/// (A set pixel with `k` set neighbours has response `255*(8-k)`, nonzero iff `k < 8`;
/// a clear pixel clamps to 0; PIL copies the outermost ring through unfiltered.)
pub fn is_edge(mask: &BinaryMask, x: u32, y: u32) -> bool {
    if !mask.get(x, y) {
        return false;
    }
    let (width, height) = mask.dimensions();
    if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
        return true;
    }
    for dy in -1_i32..=1 {
        for dx in -1_i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let neighbour_x = (x as i32 + dx) as u32;
            let neighbour_y = (y as i32 + dy) as u32;
            if !mask.get(neighbour_x, neighbour_y) {
                return true;
            }
        }
    }
    false
}

/// Every edge pixel, in **row-major order** -- the order matters for the median
/// heuristic's tie behaviour (§10.3 step 3 of `border_std_deviation`).
pub fn edge_pixels(mask: &BinaryMask) -> Vec<(u32, u32)> {
    let (width, height) = mask.dimensions();
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if is_edge(mask, x, y) {
                pixels.push((x, y));
            }
        }
    }
    pixels
}

/// Population standard deviation (`ddof = 0`).
pub fn population_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / n;
    variance.sqrt()
}

/// Sample standard deviation (`ddof = 1`).
///
/// DEVIATION(4): upstream's `np.std(..., ddof=1)` yields `NaN` for a single border
/// pixel; v1 yields `0.0`, treating that one-sample border as uniform instead of letting
/// NumPy's warning/NaN poison candidate scoring. §14.4, ratified by §15.9.
pub fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (n - 1.0);
    variance.sqrt()
}

/// NumPy-style median of u8 samples, then Python `int()` truncation toward zero
/// (§10.3 step 5 of `border_std_deviation`): odd `n` -> the middle element; even `n` ->
/// `(a + b) / 2.0`, so two middles of 100 and 101 give **100**, not 101.
pub fn numpy_median_u8(values: &[u8]) -> u8 {
    assert!(!values.is_empty(), "median of an empty sample");
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let median = if n % 2 == 1 {
        f64::from(sorted[n / 2])
    } else {
        (f64::from(sorted[n / 2 - 1]) + f64::from(sorted[n / 2])) / 2.0
    };
    median.clamp(0.0, 255.0) as u8
}

/// spec §10.3 step 5 -- grayscale path: population std of the samples, and the
/// truncated NumPy median replicated across the three channels.
pub fn gray_stats(values: &[u8]) -> (f64, [u8; 3]) {
    let as_f64 = values
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let median = numpy_median_u8(values);
    (population_std(&as_f64), [median, median, median])
}

/// spec §10.3 step 6 -- the median heuristic: an exact RGB triple occurring in **more
/// than** `n/2` of the samples wins outright. First such triple in sample order.
pub fn heuristic_median(colors: &[[u8; 3]]) -> Option<[u8; 3]> {
    let n = colors.len();
    colors
        .iter()
        .find(|candidate| {
            let count = colors.iter().filter(|color| color == candidate).count();
            count * 2 > n
        })
        .copied()
}

/// spec §10.3 step 6 -- the geometric median via Weiszfeld: start at the mean; each
/// iteration drop zero-distance points, weight the rest by `1/d` normalised, and stop
/// when the estimate moves less than [`WEISZFELD_EPSILON`] or after
/// [`WEISZFELD_MAX_ITERATIONS`]. If every point coincides with the current estimate,
/// return it. Channels are truncated toward zero.
pub fn geometric_median(colors: &[[u8; 3]]) -> [u8; 3] {
    assert!(!colors.is_empty(), "geometric median of an empty sample");
    let points = colors
        .iter()
        .map(|color| {
            [
                f64::from(color[0]),
                f64::from(color[1]),
                f64::from(color[2]),
            ]
        })
        .collect::<Vec<_>>();

    let mut estimate = mean_color(&points);
    for _ in 0..WEISZFELD_MAX_ITERATIONS {
        let mut weight_total = 0.0;
        let mut accumulator = [0.0_f64; 3];
        for point in &points {
            let distance = euclidean(*point, estimate);
            if distance <= 0.0 {
                continue;
            }
            let weight = 1.0 / distance;
            weight_total += weight;
            for channel in 0..3 {
                accumulator[channel] += weight * point[channel];
            }
        }
        if weight_total == 0.0 {
            // Every point coincides with the estimate.
            break;
        }
        let next = [
            accumulator[0] / weight_total,
            accumulator[1] / weight_total,
            accumulator[2] / weight_total,
        ];
        let moved = euclidean(next, estimate);
        estimate = next;
        if moved < WEISZFELD_EPSILON {
            break;
        }
    }

    [
        truncate_channel(estimate[0]),
        truncate_channel(estimate[1]),
        truncate_channel(estimate[2]),
    ]
}

/// spec §10.3 step 6 -- colour path: sample (`ddof = 1`) std of the per-sample distances
/// to the mean colour, and the heuristic-then-Weiszfeld median.
pub fn color_stats(colors: &[[u8; 3]]) -> (f64, [u8; 3]) {
    assert!(!colors.is_empty(), "colour statistics of an empty sample");
    let points = colors
        .iter()
        .map(|color| {
            [
                f64::from(color[0]),
                f64::from(color[1]),
                f64::from(color[2]),
            ]
        })
        .collect::<Vec<_>>();
    let mean = mean_color(&points);
    let distances = points
        .iter()
        .map(|point| euclidean(*point, mean))
        .collect::<Vec<_>>();
    let median = heuristic_median(colors).unwrap_or_else(|| geometric_median(colors));
    (sample_std(&distances), median)
}

/// spec §10.3 step 7: `min(median_color) > off_white_max_threshold` snaps to pure white.
/// Strictly greater, and on the **minimum** channel -- so `(240,255,255)` with a
/// threshold of 240 is left alone.
pub fn snap_off_white(median_color: [u8; 3], off_white_max_threshold: u8) -> [u8; 3] {
    let minimum = median_color.iter().copied().min().expect("three channels");
    if minimum > off_white_max_threshold {
        [255, 255, 255]
    } else {
        median_color
    }
}

/// spec §10.3's `border_std_deviation`, steps 1-8, in order.
///
/// The grayscale path is taken when the canvas is grayscale **or**
/// `!allow_colored_masks` (§16.9 item 7); in the latter case an RGB canvas is reduced
/// with [`pil_luma`], never with `image`'s `to_luma8`.
pub fn border_std_deviation(
    base_crop: &BaseCanvas,
    mask: &BinaryMask,
    off_white_max_threshold: u8,
    allow_colored_masks: bool,
) -> Result<BorderStats, BlankMask> {
    assert_eq!(
        base_crop.dimensions(),
        mask.dimensions(),
        "border canvas {:?} and candidate mask {:?} must be the same size",
        base_crop.dimensions(),
        mask.dimensions()
    );

    let edges = edge_pixels(mask);
    if edges.is_empty() {
        return Err(BlankMask);
    }

    let use_gray = base_crop.is_gray() || !allow_colored_masks;
    let (std_deviation, median_color) = if use_gray {
        let values = edges
            .iter()
            .map(|(x, y)| base_crop.luma_at(*x, *y))
            .collect::<Vec<_>>();
        gray_stats(&values)
    } else {
        let colors = edges
            .iter()
            .map(|(x, y)| base_crop.color_at(*x, *y))
            .collect::<Vec<_>>();
        color_stats(&colors)
    };

    Ok(BorderStats {
        std_deviation,
        median_color: snap_off_white(median_color, off_white_max_threshold),
    })
}

fn mean_color(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len() as f64;
    let mut mean = [0.0_f64; 3];
    for point in points {
        for channel in 0..3 {
            mean[channel] += point[channel];
        }
    }
    for value in &mut mean {
        *value /= n;
    }
    mean
}

fn euclidean(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn truncate_channel(value: f64) -> u8 {
    value.clamp(0.0, 255.0) as u8
}
