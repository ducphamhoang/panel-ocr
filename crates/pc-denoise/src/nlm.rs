//! Task N1 -- joint-channel non-local-means denoising (spec §11.3's "NLM
//! specification", §11.7(A)1-5/11, §11.7(B)12, §16.10 items 1, 8, 9, 10, 11).
//!
//! This replaces `cv2.fastNlMeansDenoising`. It is **not** a generic NLM: every
//! constant, every window, the border rule, the accumulation type and the rounding are
//! pinned by §16.10 item 9, because §11.7(B)12 compares the result against recorded
//! OpenCV output with a tight tolerance.
//!
//! ```text
//! d(p,q) = ( sum_{o in template} sum_{c<C} (I_c(p+o) - I_c(q+o))^2 ) / (C * tw^2)
//! w(p,q) = exp(-d(p,q) / h^2)                       (sigma^2_est = 0, per §11.3)
//! out_c(p) = sum_q w(p,q) * I_c(q) / sum_q w(p,q)   (self term q == p included)
//! ```
//!
//! Exactness notes that matter for reproducibility (§5.7, §11.7(A)5):
//!   * the squared-difference window sums are **integers**, accumulated in `i64` via a
//!     summed-area table, so they are exact -- no float associativity anywhere in `d`;
//!   * `sum <= 255^2 * 3 * 49 = 9_561_675 < 2^24`, so `sum as f32` is exact too;
//!   * `acc`/`wtot` are `f32` (as §11.3 requires) and offsets are visited in a fixed
//!     order (`dy` ascending, then `dx` ascending), so the accumulation order is fixed;
//!   * work is split into fixed-size output row bands, each with its own scratch, so
//!     the result does not depend on the rayon thread count (§16.10 item 10).
//!
//! The summed-area table is §11.3's mandatory "incremental distance update" in its
//! simplest exact form: `O(W*H*s^2)` instead of the naive `O(W*H*s^2*t^2)`.

use image::{DynamicImage, GrayImage, RgbImage};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// spec §11.3. `h` is `denoiser.filter_strength`; both windows must be odd and `>= 3`
/// (validated in `pc-config`, §6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NlmParams {
    pub h: f32,
    pub template_window: u32,
    pub search_window: u32,
}

impl NlmParams {
    /// The §6 defaults: `h = 10`, `template_window = 7`, `search_window = 21` -- the
    /// same triple §11.7(B)12's recorded OpenCV reference was produced with.
    pub fn defaults() -> Self {
        Self {
            h: 10.0,
            template_window: 7,
            search_window: 21,
        }
    }
}

impl Default for NlmParams {
    fn default() -> Self {
        Self::defaults()
    }
}

/// §16.10 item 11: an always-on (not `cfg`-gated) invocation counter, because
/// §11.7(A)7's "zero NLM invocations" assertion lives in an integration test, which
/// never sees the library's `cfg(test)`. One relaxed increment per call is free next to
/// the filter itself.
static DENOISE_CALLS: AtomicU64 = AtomicU64::new(0);

/// Number of `denoise` calls since the last [`reset_denoise_call_count`].
pub fn denoise_call_count() -> u64 {
    DENOISE_CALLS.load(Ordering::Relaxed)
}

/// Zero the counter read by [`denoise_call_count`].
pub fn reset_denoise_call_count() {
    DENOISE_CALLS.store(0, Ordering::Relaxed);
}

/// Number of output rows one parallel band covers. Fixed in code (never derived from
/// the thread count) so the result is bit-identical at any parallelism (§16.10 item 10).
const BAND_ROWS: usize = 16;

/// spec §11.3 -- joint-channel NLM.
///
/// `ImageLuma8` input yields `ImageLuma8` output (`C = 1`); every other variant is
/// converted to RGB and yields `ImageRgb8` (`C = 3`). Per §16.10 item 8 a replicated-gray
/// RGB image produces *exactly* the same numbers as its `L` counterpart, because both
/// the squared-difference sum and the divisor scale by `C`.
///
/// Panics if either window is even or `< 3` -- a caller bug; `pc-config` validation
/// (§6) rejects such profiles before a stage ever sees them.
pub fn denoise(img: &DynamicImage, params: NlmParams) -> DynamicImage {
    DENOISE_CALLS.fetch_add(1, Ordering::Relaxed);
    assert!(
        params.template_window >= 3 && params.template_window % 2 == 1,
        "template_window must be odd and >= 3, got {}",
        params.template_window
    );
    assert!(
        params.search_window >= 3 && params.search_window % 2 == 1,
        "search_window must be odd and >= 3, got {}",
        params.search_window
    );

    match img {
        DynamicImage::ImageLuma8(gray) => {
            let (width, height) = gray.dimensions();
            let out = filter(gray.as_raw(), width, height, 1, params);
            DynamicImage::ImageLuma8(
                GrayImage::from_raw(width, height, out).expect("preserved dimensions"),
            )
        }
        other => {
            let rgb = other.to_rgb8();
            let (width, height) = rgb.dimensions();
            let out = filter(rgb.as_raw(), width, height, 3, params);
            DynamicImage::ImageRgb8(
                RgbImage::from_raw(width, height, out).expect("preserved dimensions"),
            )
        }
    }
}

/// `BORDER_REFLECT_101` (`abcd | cba`), matching OpenCV's default in
/// `fastNlMeansDenoising`. `n == 1` degenerates to the only valid index.
pub fn reflect101(index: i64, n: i64) -> usize {
    debug_assert!(n >= 1);
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut m = index.rem_euclid(period);
    if m >= n {
        m = period - m;
    }
    m as usize
}

/// The core filter over an interleaved `channels`-per-pixel `u8` plane.
fn filter(source: &[u8], width: u32, height: u32, channels: usize, params: NlmParams) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let template = params.template_window as usize; // tw
    let t = template / 2;
    let s = (params.search_window / 2) as usize;
    let pad = s + t;
    let padded_w = w + 2 * pad;
    let padded_h = h + 2 * pad;

    // Materialise the reflect-101 padded plane once (§16.10 item 9).
    let mut padded = vec![0_u8; padded_w * padded_h * channels];
    let column_map: Vec<usize> = (0..padded_w as i64)
        .map(|x| reflect101(x - pad as i64, w as i64))
        .collect();
    for py in 0..padded_h {
        let source_y = reflect101(py as i64 - pad as i64, h as i64);
        for (px, &source_x) in column_map.iter().enumerate() {
            let source_index = (source_y * w + source_x) * channels;
            let target_index = (py * padded_w + px) * channels;
            padded[target_index..target_index + channels]
                .copy_from_slice(&source[source_index..source_index + channels]);
        }
    }

    let inverse_h_squared = 1.0_f32 / (params.h * params.h);
    let distance_divisor = (channels * template * template) as f32;
    let search_radius = s as i64;

    let mut out = vec![0_u8; w * h * channels];
    out.par_chunks_mut(BAND_ROWS * w * channels)
        .enumerate()
        .for_each(|(band_index, band)| {
            let y0 = band_index * BAND_ROWS;
            let band_rows = band.len() / (w * channels);
            let y1 = y0 + band_rows;

            // The squared-difference plane covers image columns `-t..w+t` and rows
            // `y0-t..y1+t`, i.e. every template window centred inside the band.
            let plane_w = w + 2 * t;
            let plane_h = band_rows + 2 * t;
            let sat_w = plane_w + 1;
            let mut sat = vec![0_i64; sat_w * (plane_h + 1)];

            let mut accumulator = vec![0.0_f32; band_rows * w * channels];
            let mut weight_total = vec![0.0_f32; band_rows * w];

            for dy in -search_radius..=search_radius {
                for dx in -search_radius..=search_radius {
                    // --- squared-difference summed-area table -------------------
                    for j in 0..plane_h {
                        // Image row `y0 + j - t`, i.e. padded row `y0 + j - t + pad`
                        // = `y0 + j + s`, which is always in range (pad = s + t).
                        let row_base = (y0 + j + s) * padded_w;
                        let shifted_row_base = ((y0 + j + s) as i64 + dy) as usize * padded_w;
                        let sat_row = (j + 1) * sat_w;
                        let sat_previous = j * sat_w;
                        let mut running = 0_i64;
                        for i in 0..plane_w {
                            let column = i + s; // image column `i - t`, padded `i + s`
                            let a = (row_base + column) * channels;
                            let b = (shifted_row_base + (column as i64 + dx) as usize) * channels;
                            let mut squared = 0_i64;
                            for c in 0..channels {
                                let delta = i64::from(padded[a + c]) - i64::from(padded[b + c]);
                                squared += delta * delta;
                            }
                            running += squared;
                            sat[sat_row + i + 1] = sat[sat_previous + i + 1] + running;
                        }
                    }

                    // --- weights and accumulation --------------------------------
                    for y in y0..y1 {
                        let j_start = y - y0;
                        let j_end = j_start + 2 * t;
                        let top = j_start * sat_w;
                        let bottom = (j_end + 1) * sat_w;
                        let padded_row = ((y + pad) as i64 + dy) as usize * padded_w;
                        let band_row = (y - y0) * w;
                        for x in 0..w {
                            let i_start = x;
                            let i_end = x + 2 * t;
                            let sum = sat[bottom + i_end + 1]
                                - sat[top + i_end + 1]
                                - sat[bottom + i_start]
                                + sat[top + i_start];
                            let distance = sum as f32 / distance_divisor;
                            let weight = (-distance * inverse_h_squared).exp();
                            let neighbour =
                                (padded_row + ((x + pad) as i64 + dx) as usize) * channels;
                            let target = (band_row + x) * channels;
                            for c in 0..channels {
                                accumulator[target + c] +=
                                    weight * f32::from(padded[neighbour + c]);
                            }
                            weight_total[band_row + x] += weight;
                        }
                    }
                }
            }

            for pixel in 0..band_rows * w {
                let total = weight_total[pixel];
                for c in 0..channels {
                    let value = accumulator[pixel * channels + c] / total;
                    // Round half away from zero; the quotient is non-negative.
                    band[pixel * channels + c] = (value + 0.5).floor().clamp(0.0, 255.0) as u8;
                }
            }
        });

    out
}
