//! Task A1 -- spec §16.37 item 8: the greyscale / histogram / top-k-colour path of
//! `MaskRefineMode::Annotation`, ported from upstream PanelCleaner's
//! `comic_text_detector/utils/textmask.py::get_topk_masklist` and
//! `imgproc_utils.py::expand_textwindow` (pinned commit
//! `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`).
//!
//! **This module is not wired into anything.** `pc_detect::run` still rejects
//! `MaskRefineMode::Annotation` (§16.37's preamble: *"Nothing here asserts that
//! `Annotation` mode works, or exists."*). Wiring is task A4; connected components, the
//! merge loop and hole filling are task A3. Neither is implemented here.
//!
//! What A1 covers, in upstream's own order inside `get_topk_masklist`:
//!   1. [`expand_text_window`] -- the crop window (§16.37 item 5's corrected divisor
//!      formula and `im_w - 1` / `im_h - 1` clamp, bound to `Annotation` only),
//!   2. [`rgb_to_gray`] -- `cv2.cvtColor(..., COLOR_BGR2GRAY)`,
//!   3. [`erode_rect3x3`] -- `cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)`,
//!   4. [`candidate_grey_values`] -- `im_grey[np.where(eroded > 127)]`,
//!   5. [`histogram_255`] -- `np.histogram(candidate_grey_px, bins=255)`,
//!   6. [`top_k_colors`] -- `get_topk_color(his, bin, color_var=10, k=3)`.
//!
//! Step 6 of upstream's function (`cv2.inRange` over each colour, then `minxor_thresh`)
//! is **not** part of A1: `minxor_thresh` is the XOR-minimising selection that
//! §16.37 item 8 assigns to A2. It lives below, with the rest of A2.
//!
//! What A2 adds (spec §16.37 item 8: *"Otsu thresholding and the XOR-minimising channel
//! selection"*), in upstream's own order:
//!   7. [`otsu_threshold`] + [`threshold_binary`] -- `cv2.threshold(c, 1, 255,
//!      THRESH_OTSU + THRESH_BINARY)`,
//!   8. [`xor_sum`] and [`minxor_thresh`] -- `minxor_thresh` (`textmask.py:32`),
//!   9. [`in_range`] and [`top_k_mask_list`] -- step 6 of `get_topk_masklist`, completing
//!      the function A1 started,
//!  10. [`otsu_thresh_mask_list`] -- `get_otsuthresh_masklist(..., per_channel=False)`
//!      (`textmask.py:49`), and
//!  11. [`candidate_mask_list`] -- `refine_mask`'s `mask_list = get_topk_masklist(...)`;
//!      `mask_list += get_otsuthresh_masklist(...)` concatenation, whose **order** A3
//!      depends on.

use image::{GrayImage, Luma, RgbImage};
use pc_core::{Rect, StageError};

/// `expand_r` for `Annotation`: upstream `refine_mask` calls
/// `expand_textwindow(..., expand_r=16)`. **A divisor, not a pad** -- see
/// [`expand_text_window`] and §16.37 item 5.
pub const ANNOTATION_EXPAND_R: i32 = 16;

/// `k` in upstream's `get_topk_color(..., k=3)`.
pub const TOPK_K: usize = 3;
/// `color_var` in upstream's `get_topk_color(..., color_var=10)`.
pub const TOPK_COLOR_VAR: f64 = 10.0;
/// `bin_tol` default of upstream's `get_topk_color` (`bin_tol=0.001`), used as
/// `bin_tol * sum(counts)`.
pub const TOPK_BIN_TOL: f64 = 0.001;

/// Upstream's mask gate: `np.where(eroded_mask > 127)` -- strict.
pub const CANDIDATE_MASK_THRESHOLD: u8 = 127;

/// Number of bins in upstream's `np.histogram(candidate_grey_px, bins=255)`.
pub const HISTOGRAM_BINS: usize = 255;

// ---------------------------------------------------------------- greyscale

/// OpenCV's 15-bit fixed-point BGR->GRAY weights, in R/G/B order. They sum to exactly
/// `1 << GRAY_SHIFT` (9798 + 19235 + 3735 = 32768).
///
/// These are **measured**, not guessed: the classic 14-bit triple (4899, 9617, 1868)
/// disagrees with `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` on 43 864 of the 16 777 216
/// RGB values under the opencv version pinned in
/// `tests/fixtures/recorded/detector/PROVENANCE.json` (`"opencv": "5.0.0"`), while this
/// triple with this shift and this bias reproduces all 16 777 216 exactly. Verified by
/// enumerating the whole cube against `cv2` on 2026-08-06.
const GRAY_R: u32 = 9798;
const GRAY_G: u32 = 19235;
const GRAY_B: u32 = 3735;
const GRAY_SHIFT: u32 = 15;

/// `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`, over an `RgbImage`.
///
/// Upstream reads pages with `cv2.imread`, so its array is **BGR** and
/// `COLOR_BGR2GRAY` applies the 0.299 weight to channel index 2. `RgbImage` stores red
/// at index 0, so the same arithmetic here weights `pixel[0]` by [`GRAY_R`]. The value
/// produced for a given (r, g, b) is identical; only the channel order in memory differs.
///
/// Exact form: `(r * 9798 + g * 19235 + b * 3735 + (1 << 14)) >> 15`. This is
/// **not** `round(0.299 r + 0.587 g + 0.114 b)` and not the 14-bit fixed point either;
/// see [`GRAY_R`]'s note for the measurement.
pub fn rgb_to_gray(image: &RgbImage) -> GrayImage {
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let [r, g, b] = image.get_pixel(x, y).0;
        let sum = u32::from(r) * GRAY_R + u32::from(g) * GRAY_G + u32::from(b) * GRAY_B;
        // `+ (1 << (shift - 1))` then `>> shift` -- OpenCV's CV_DESCALE. The sum plus
        // the bias is at most 255 * 32768 + 16384 < 2^24, so u32 cannot overflow and the
        // result is at most 255.
        Luma([((sum + (1 << (GRAY_SHIFT - 1))) >> GRAY_SHIFT) as u8])
    })
}

// ------------------------------------------------------------- crop window

/// `expand_textwindow(img_size, xyxy, expand_r)` -- upstream
/// `comic_text_detector/utils/imgproc_utils.py:163-173`.
///
/// spec §16.37 item 5 (`CORRECTION`, `SUPERSEDES: §8.3 step 5`): `expand_r` is a
/// **divisor**, not a pad amount, and the far edges clamp to `im_w - 1` / `im_h - 1`:
///
/// ```text
/// paddings = int(round((max(h, w) * 0.25 + min(h, w) * 0.75) / expand_r))
/// x1, y1 = max(0, x1 - paddings), max(0, y1 - paddings)
/// x2, y2 = min(im_w - 1, x2 + paddings), min(im_h - 1, y2 + paddings)
/// ```
///
/// Two details that are easy to get wrong and are both asserted by the A1 tests:
///   * Python's `round` is round-half-to-**even**, so this uses
///     [`f64::round_ties_even`], not `f64::round`.
///   * the far clamp is `- 1` off `pc_core::Rect::pad`'s clamp to `canvas.0`/`canvas.1`.
///
/// **Scope**: this is bound to `Annotation` only. `MaskRefineMode::Simple` keeps its flat
/// `REFINE_EXPAND = 16` pad and `Rect::pad`'s clamp -- §16.37 item 5: *"`Simple`'s
/// padding is therefore left exactly as it is"*, because §8.3 step 5 describes `Simple`
/// as a port of koharu, not of PanelCleaner, and koharu was not consulted.
///
/// `expand_r` must be non-zero; upstream's default is 8 and `refine_mask` passes 16
/// ([`ANNOTATION_EXPAND_R`]). A zero divisor produces a non-finite quotient and the
/// resulting cast saturates, which is why callers pass the constant.
pub fn expand_text_window(rect: Rect, image_size: (u32, u32), expand_r: i32) -> Rect {
    let w = f64::from(rect.width());
    let h = f64::from(rect.height());
    let paddings =
        ((h.max(w) * 0.25 + h.min(w) * 0.75) / f64::from(expand_r)).round_ties_even() as i32;
    Rect::new(
        (rect.x1 - paddings).max(0),
        (rect.y1 - paddings).max(0),
        (rect.x2 + paddings).min(image_size.0 as i32 - 1),
        (rect.y2 + paddings).min(image_size.1 as i32 - 1),
    )
}

// ------------------------------------------------------------------- erode

/// `cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)` -- the minimum over the
/// 3x3 neighbourhood centred on each pixel.
///
/// Border handling is the one non-obvious part and it is *not* zero padding:
/// `cv2.erode`'s default `borderValue` is `morphologyDefaultBorderValue()`, i.e.
/// `+DBL_MAX`, so out-of-image neighbours never lower the minimum and edge pixels are
/// eroded only by their in-image neighbours. Confirmed by running `cv2.erode` on a
/// 5x5 probe (see the A1 test asserting the recorded output).
pub fn erode_rect3x3(mask: &GrayImage) -> GrayImage {
    let (width, height) = mask.dimensions();
    GrayImage::from_fn(width, height, |x, y| {
        let mut min = u8::MAX;
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(height - 1);
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(width - 1);
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                min = min.min(mask.get_pixel(nx, ny).0[0]);
            }
        }
        Luma([min])
    })
}

/// `candidate_grey_px = im_grey[np.where(cv2.erode(msk, ones((3,3)), 1) > 127)]`.
///
/// Returned in `np.where` order, i.e. row-major. Order is irrelevant to
/// [`histogram_255`], which is the only consumer, but it is stated so a future reader
/// does not assume it is sorted.
///
/// Errors (`StageError::InvalidInput`, per-image) when `grey` and `mask` differ in size:
/// upstream indexes one array with the other's boolean mask, which is only defined for
/// equal shapes.
pub fn candidate_grey_values(grey: &GrayImage, mask: &GrayImage) -> Result<Vec<u8>, StageError> {
    if grey.dimensions() != mask.dimensions() {
        return Err(StageError::InvalidInput(format!(
            "grey image {:?} and mask {:?} must have the same size",
            grey.dimensions(),
            mask.dimensions()
        )));
    }
    let eroded = erode_rect3x3(mask);
    let mut values = Vec::new();
    for (grey_pixel, eroded_pixel) in grey.pixels().zip(eroded.pixels()) {
        if eroded_pixel.0[0] > CANDIDATE_MASK_THRESHOLD {
            values.push(grey_pixel.0[0]);
        }
    }
    Ok(values)
}

// --------------------------------------------------------------- histogram

/// `np.histogram(values, bins=255)` for `u8` samples: 255 equal-width bins plus the 256
/// bin edges.
///
/// `edges[i]` is the **left** edge of bin `i`; `edges[255]` is the right edge of the last
/// bin. Upstream's `get_topk_color` is called as `get_topk_color(his, bin, ...)` with
/// `bin, his = np.histogram(...)` -- i.e. numpy's *counts* land in the parameter named
/// `bins` and the *edges* in the one named `color_list`, so a "colour" in
/// [`top_k_colors`] is a bin **left edge**, and a float, not a `u8`.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram255 {
    /// `counts[i]` = number of samples in bin `i`.
    pub counts: [u32; HISTOGRAM_BINS],
    /// The 256 bin edges.
    pub edges: [f64; HISTOGRAM_BINS + 1],
}

/// `np.histogram(values, bins=255)`, replicating numpy's uniform-bin fast path
/// (`numpy/lib/_histograms_impl.py`, the `uniform_bins is not None` branch) rather than
/// a naive `floor((v - min) / step)`.
///
/// The parts that are not obvious, each covered by an A1 test:
///   * range is `[min, max]` of the samples; when `min == max` numpy widens it to
///     `[min - 0.5, max + 0.5]`; for an **empty** input it is `[0, 1]`
///     (`_get_outer_edges`).
///   * edges come from `np.linspace(first, last, 256)`, i.e. `i * step + first` with
///     `step = (last - first) / 255`, and then `edges[255]` is overwritten with `last`
///     exactly.
///   * the index is `((v - first) / (last - first)) * 255` truncated -- **not**
///     `floor((v - first) / step)`. The two disagree for real `u8` inputs: with
///     `min = 0, max = 35`, the sample `21` lands in bin **152** by numpy's formula and
///     bin 153 by the naive one.
///   * `index == 255` is decremented, which is what makes the last bin closed on the
///     right; then numpy's two ULP corrections (`decrement` / `increment`) are applied
///     in that order.
#[must_use]
pub fn histogram_255(values: &[u8]) -> Histogram255 {
    let bins = HISTOGRAM_BINS as f64;
    let (first, last) = match (values.iter().copied().min(), values.iter().copied().max()) {
        // `_get_outer_edges`: "handle empty arrays. Can't determine range, so use 0-1."
        (None, None) => (0.0_f64, 1.0_f64),
        (Some(min), Some(max)) if min == max => (f64::from(min) - 0.5, f64::from(max) + 0.5),
        (Some(min), Some(max)) => (f64::from(min), f64::from(max)),
        _ => unreachable!("min and max are both Some iff values is non-empty"),
    };

    let step = (last - first) / bins;
    let mut edges = [0.0_f64; HISTOGRAM_BINS + 1];
    for (index, edge) in edges.iter_mut().enumerate() {
        *edge = index as f64 * step + first;
    }
    // `np.linspace(..., endpoint=True)`: "if endpoint and num > 1: y[-1] = stop".
    edges[HISTOGRAM_BINS] = last;

    let denominator = last - first;
    let mut counts = [0_u32; HISTOGRAM_BINS];
    for &value in values {
        let sample = f64::from(value);
        // numpy's `keep = (tmp_a >= first_edge) & (tmp_a <= last_edge)`. Unreachable for
        // a range derived from the samples themselves, kept because the guard is what
        // makes the index arithmetic below total.
        if sample < first || sample > last {
            continue;
        }
        let mut index = (((sample - first) / denominator) * bins) as usize;
        if index == HISTOGRAM_BINS {
            index -= 1;
        }
        if sample < edges[index] {
            index -= 1;
        }
        if sample >= edges[index + 1] && index != HISTOGRAM_BINS - 1 {
            index += 1;
        }
        counts[index] += 1;
    }

    Histogram255 { counts, edges }
}

// ------------------------------------------------------------ top-k colour

/// `get_topk_color(color_list, bins, k, color_var, bin_tol)` -- upstream
/// `comic_text_detector/utils/textmask.py:19-30`. Returns bin **left edges** (floats),
/// most populous bin first.
///
/// DEVIATION(23) -- spec §14 item 23 / §16.37 item 2: upstream orders candidates with
/// `idx = np.argsort(bins * -1)`, whose default introsort has **no contractual ordering
/// among equal keys**, and the input carries tie groups of up to 255 equal counts. v1.5
/// therefore declares its own rule: a **stable** sort on
/// `(Reverse(count), ascending bin index)`. §16.37 item 1 measured what the unspecified
/// order costs -- switching upstream to `kind='stable'` alone moves its refined mask by
/// 277 pixels on `…E01P02` and 27 on `…E01P03`, while leaving `…E01P01` byte-identical,
/// and CPU dispatch alone (AVX2 vs SSE) changes how many colours the function returns
/// on synthetic input. Both of those are properties of the pages/inputs they were
/// measured on and neither generalises.
///
/// Everything else is upstream's control flow verbatim, including one ordering quirk
/// that an A1 test pins: the `len(top_colors) >= k or bin < bin_tol` break is evaluated
/// **after** the append, so the candidate that first falls below `bin_tol` is still
/// added before the loop stops.
#[must_use]
pub fn top_k_colors(histogram: &Histogram255, k: usize, color_var: f64, bin_tol: f64) -> Vec<f64> {
    // DEVIATION(23): stable sort on (Reverse(count), ascending bin index). `sort_by_key`
    // is stable, and the keys are visited in ascending bin index, so equal counts keep
    // ascending index order without the index needing to be part of the key.
    let mut order: Vec<usize> = (0..HISTOGRAM_BINS).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(histogram.counts[index]));

    let total: u64 = histogram.counts.iter().copied().map(u64::from).sum();
    let bin_tol = total as f64 * bin_tol;

    let mut top_colors = vec![histogram.edges[order[0]]];
    for &index in &order[1..] {
        let color = histogram.edges[index];
        let count = f64::from(histogram.counts[index]);
        let nearest = top_colors
            .iter()
            .map(|top| (top - color).abs())
            .fold(f64::INFINITY, f64::min);
        if nearest > color_var {
            top_colors.push(color);
        }
        if top_colors.len() >= k || count < bin_tol {
            break;
        }
    }
    top_colors
}

/// [`top_k_colors`] with upstream `get_topk_masklist`'s arguments:
/// `get_topk_color(his, bin, color_var=10, k=3)`.
#[must_use]
pub fn top_k_colors_default(histogram: &Histogram255) -> Vec<f64> {
    top_k_colors(histogram, TOPK_K, TOPK_COLOR_VAR, TOPK_BIN_TOL)
}

// =========================================================== A2: Otsu + XOR selection

/// `color_range` in upstream's `get_topk_masklist` (`textmask.py:78`). The band is
/// `c_top = min(color + color_range, 255)`, `c_bottom = c_top - 2 * color_range` -- anchored
/// at the **top**, so the 255 clamp shifts the whole window down.
pub const TOPK_COLOR_RANGE: f64 = 30.0;

/// `maxval` in upstream's `cv2.threshold(c, 1, 255, ...)`.
pub const THRESHOLD_MAXVAL: u8 = 255;

/// Upstream iterates `channels = [img[..., 0], img[..., 1], img[..., 2]]` over an array read
/// by `cv2.imdecode(..., IMREAD_COLOR)`, i.e. **BGR**. `RgbImage` stores red at index 0, so
/// the same memory order is R-index 2, 1, 0.
///
/// This is not cosmetic: it decides which candidate wins an exact XOR-sum tie, because
/// upstream's `mask_list.sort(key=lambda x: x[1])` is Python's `list.sort`, which is
/// **guaranteed stable**, and [`otsu_thresh_mask_list`] reproduces that with
/// `Iterator::min_by_key` (documented to return the *first* of several equal minima).
const UPSTREAM_CHANNEL_ORDER: [usize; 3] = [2, 1, 0];

/// OpenCV's `FLT_EPSILON`, used by `getThreshVal_Otsu`'s degenerate-split guard. Kept as an
/// `f64` widened from the `f32` constant, which is what the C++ comparison does.
const FLT_EPSILON: f64 = f32::EPSILON as f64;

/// One entry of upstream's `mask_list`: a `[threshed, xor_sum]` pair.
///
/// Upstream's list is a Python `list` of two-element lists, and both of its consumers depend
/// on the **order** of that list -- see [`candidate_mask_list`]. This is a plain struct rather
/// than a tuple so that A3's `mask_list.sort(key=lambda x: x[1])` port cannot sort on the
/// wrong element.
#[derive(Clone, PartialEq, Eq)]
pub struct MaskCandidate {
    /// `threshed` -- a 0/255 mask the size of the crop.
    pub mask: GrayImage,
    /// `xor_sum` -- `cv2.bitwise_xor(threshed, pred_mask).sum()`, the selection key.
    pub xor_sum: u64,
}

impl std::fmt::Debug for MaskCandidate {
    /// Deliberately does **not** print the pixels: a crop is thousands of them, and an
    /// assertion message that scrolls is an assertion message nobody reads.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaskCandidate")
            .field("size", &self.mask.dimensions())
            .field(
                "nonzero",
                &self.mask.pixels().filter(|p| p.0[0] != 0).count(),
            )
            .field("xor_sum", &self.xor_sum)
            .finish()
    }
}

/// `cv2.threshold(src, _, _, cv2.THRESH_OTSU)`'s threshold search, ported from OpenCV's
/// `getThreshVal_Otsu` (`modules/imgproc/src/thresh.cpp`, branch `5.x`) statement for
/// statement -- the histogram, the `1/n` scale, the running `mu1 *= q1` accumulator, the
/// `min(q1,q2) < FLT_EPSILON || max(q1,q2) > 1 - FLT_EPSILON` guard, and the **strict**
/// `sigma > max_sigma` comparison.
///
/// Two consequences of that strictness, both pinned by A2 tests:
///   * of two thresholds with equal between-class variance the **lowest index** wins;
///   * a constant image skips every iteration, so the result is the initial `0` -- and
///     `threshold_binary(_, 0)` then turns every non-zero pixel on.
///
/// **Empty input** returns 0. `cv2.threshold` has no defined behaviour there (it returns no
/// threshold at all), so this is this crate's own totality choice, not upstream parity;
/// upstream cannot reach it, because [`expand_text_window`] never yields an empty window.
///
/// **Measured, and the reason this doc says "OpenCV's C++ reference" and not "cv2".** The
/// opencv-python 5.0.0 wheel is built with Intel IPP 2026.0.0, and `getThreshVal_Otsu_8u`
/// short-circuits to `ippiComputeThreshold_Otsu_8u_C1R` before reaching the code above. Over
/// 20 000 random and degenerate arrays per corpus, on 2026-08-06: with IPP **disabled** `cv2`
/// agreed with this algorithm on every sample of every corpus measured; with IPP **enabled** a
/// handful disagreed per corpus (10, 20, 21 and 54 per 20 000, across four independently
/// generated corpora — the rate is corpus-dependent, not a constant). **Both mechanisms
/// occur**: some disagreements are exact, bit-identical between-class variances — a genuine
/// tie, decided by nothing but the tie rule — and others are ULP-level differences in IPP's
/// arithmetic. IPP's own answer additionally varies with its dispatch level
/// (`OPENCV_IPP=sse42` against `avx2`/`avx512` on the same arrays), so "match IPP" is not
/// merely hard but not well-defined as a target.
///
/// DEVIATION(24) -- spec §14 item 24 / §16.37 item 10: this crate implements the documented
/// reference path (strict `sigma > max_sigma`, lowest bin index wins a tie) and deliberately
/// does **not** reproduce IPP's fast-path near-tie behaviour. Registered rather than left as
/// prose because the two paths are two implementations of one nominally-specified function and
/// a future parity investigation must find the choice immediately.
#[must_use]
pub fn otsu_threshold(image: &GrayImage) -> u8 {
    let mut histogram = [0_u32; 256];
    for pixel in image.pixels() {
        histogram[pixel.0[0] as usize] += 1;
    }

    let count = u64::from(image.width()) * u64::from(image.height());
    if count == 0 {
        return 0;
    }
    let scale = 1.0 / count as f64;

    let mut mu = 0.0_f64;
    for (index, &bin) in histogram.iter().enumerate() {
        mu += index as f64 * f64::from(bin);
    }
    mu *= scale;

    let mut mu1 = 0.0_f64;
    let mut q1 = 0.0_f64;
    let mut max_sigma = 0.0_f64;
    let mut max_val = 0_u8;
    for (index, &bin) in histogram.iter().enumerate() {
        let p_i = f64::from(bin) * scale;
        mu1 *= q1;
        q1 += p_i;
        let q2 = 1.0 - q1;
        if q1.min(q2) < FLT_EPSILON || q1.max(q2) > 1.0 - FLT_EPSILON {
            continue;
        }
        mu1 = (mu1 + index as f64 * p_i) / q1;
        let mu2 = (mu - q1 * mu1) / q2;
        let sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        // DEVIATION(24): strict, so ties keep the lowest index -- OpenCV's reference rule, not
        // IPP's fast path. See this function's doc comment and spec §14 item 24.
        if sigma > max_sigma {
            max_sigma = sigma;
            max_val = index as u8;
        }
    }
    max_val
}

/// `cv2.THRESH_BINARY`: `dst = src > thresh ? 255 : 0`. The comparison is **strict**, so a
/// pixel equal to the threshold is off.
#[must_use]
pub fn threshold_binary(image: &GrayImage, threshold: u8) -> GrayImage {
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let value = image.get_pixel(x, y).0[0];
        Luma([if value > threshold {
            THRESHOLD_MAXVAL
        } else {
            0
        }])
    })
}

/// `cv2.bitwise_xor(a, b).sum()` -- a per-pixel XOR over all eight bits, summed.
///
/// **Not** a count of differing pixels. The pred mask this scores against is the raw U-Net
/// output (255 distinct values on the committed page), so `255 ^ 128 = 127` contributes 127,
/// where a boolean reading of both operands would contribute nothing.
///
/// The accumulator is `u64`: at 255 per pixel a `u32` overflows above 16 843 009 pixels, which
/// a full page-sized mask can exceed.
///
/// Errors (`StageError::InvalidInput`, **per-image**, cookbook rule 4) on a size mismatch;
/// `cv2.bitwise_xor` raises for mismatched shapes, and one bad block must not abort the run.
pub fn xor_sum(a: &GrayImage, b: &GrayImage) -> Result<u64, StageError> {
    if a.dimensions() != b.dimensions() {
        return Err(StageError::InvalidInput(format!(
            "xor operands {:?} and {:?} must have the same size",
            a.dimensions(),
            b.dimensions()
        )));
    }
    Ok(a.pixels()
        .zip(b.pixels())
        .map(|(left, right)| u64::from(left.0[0] ^ right.0[0]))
        .sum())
}

/// `minxor_thresh(threshed, mask, dilate=False)` -- upstream `textmask.py:32-46`.
///
/// Returns whichever of `threshed` and `255 - threshed` has the lower XOR sum against `mask`,
/// with that sum. Upstream's comparison is `if neg_xor_sum < xor_sum` -- **strict** -- so on an
/// exact tie the **positive** `threshed` is returned. Both call sites in upstream pass
/// `dilate=False` (`get_otsuthresh_masklist` explicitly, `get_topk_masklist` by default), so
/// the dilation branch is not ported.
///
/// Errors (`StageError::InvalidInput`, per-image) on a size mismatch.
pub fn minxor_thresh(threshed: &GrayImage, mask: &GrayImage) -> Result<MaskCandidate, StageError> {
    let positive = xor_sum(threshed, mask)?;
    let negated = GrayImage::from_fn(threshed.width(), threshed.height(), |x, y| {
        Luma([255 - threshed.get_pixel(x, y).0[0]])
    });
    let negative = xor_sum(&negated, mask)?;
    if negative < positive {
        Ok(MaskCandidate {
            mask: negated,
            xor_sum: negative,
        })
    } else {
        Ok(MaskCandidate {
            mask: threshed.clone(),
            xor_sum: positive,
        })
    }
}

/// `cv2.inRange(src, lowerb, upperb)` for a single-channel `u8` source and scalar `f64`
/// bounds: 255 where `round(lowerb) <= v <= round(upperb)`, 0 elsewhere.
///
/// The rounding is the part that is easy to get wrong. OpenCV converts each scalar bound to
/// the source depth with `cvRound`, i.e. round-**half-to-even**, and saturates -- it does
/// **not** compare in floating point and does not truncate. Measured against `cv2` 5.0.0 on
/// 2026-08-06: `inRange(0..=255, 2.4, 12.6)` admits `[2, 13]`, and `inRange(0..=255, 0.5,
/// 10.5)` admits `[0, 10]` where round-half-away-from-zero would give `[1, 11]`.
///
/// Bounds outside `0..=255` saturate. For `u8` data a negative lower bound and 0 are
/// indistinguishable, so the saturation is not observable on that side; it is applied anyway
/// to keep the conversion total.
#[must_use]
pub fn in_range(image: &GrayImage, lower: f64, upper: f64) -> GrayImage {
    let bound = |value: f64| value.round_ties_even().clamp(0.0, 255.0) as u8;
    let (low, high) = (bound(lower), bound(upper));
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let value = image.get_pixel(x, y).0[0];
        Luma([if low <= value && value <= high {
            THRESHOLD_MAXVAL
        } else {
            0
        }])
    })
}

/// `get_topk_masklist(im_grey, pred_mask)` -- upstream `textmask.py:63-84`, complete.
///
/// A1 landed steps 2-6a (greyscale, erode, candidate values, histogram, top-k colours); this
/// adds upstream's final loop: one `cv2.inRange` band per top-k colour, each passed through
/// [`minxor_thresh`].
///
/// **The returned order is upstream's colour order**, which is DEVIATION(23)'s declared tie
/// order via [`top_k_colors_default`] -- not sorted by `xor_sum`. See [`candidate_mask_list`]
/// for why the order is load-bearing.
///
/// `grey` must already be greyscale ([`rgb_to_gray`]); upstream's `if len(im_grey.shape) == 3`
/// conversion is handled by the caller, [`candidate_mask_list`].
///
/// Errors (`StageError::InvalidInput`, per-image) when `grey` and `mask` differ in size.
pub fn top_k_mask_list(
    grey: &GrayImage,
    mask: &GrayImage,
) -> Result<Vec<MaskCandidate>, StageError> {
    let values = candidate_grey_values(grey, mask)?;
    let histogram = histogram_255(&values);
    let colors = top_k_colors_default(&histogram);

    let mut list = Vec::with_capacity(colors.len());
    for color in colors {
        // `c_top = min(color + color_range, 255)`; `c_bottom = c_top - 2 * color_range`.
        let upper = (color + TOPK_COLOR_RANGE).min(255.0);
        let lower = upper - 2.0 * TOPK_COLOR_RANGE;
        let threshed = in_range(grey, lower, upper);
        list.push(minxor_thresh(&threshed, mask)?);
    }
    Ok(list)
}

/// `get_otsuthresh_masklist(img, pred_mask, per_channel=False)` -- upstream
/// `textmask.py:49-60`.
///
/// Otsu-thresholds each colour channel, passes each result through [`minxor_thresh`], and
/// returns the **single** lowest-scoring candidate -- upstream's `per_channel=False` branch,
/// `return [mask_list[0]]`, which is the only branch `refine_mask` uses. A one-element `Vec`
/// rather than a bare `MaskCandidate` because the caller concatenates it (see
/// [`candidate_mask_list`]).
///
/// **Tie behaviour, ported exactly, no deviation declared.** Upstream sorts with
/// `mask_list.sort(key=lambda x: x[1])` -- Python's `list.sort`, whose stability is
/// **guaranteed** -- over channels visited in memory order B, G, R. So an exact tie at the
/// minimum resolves to the earliest of those channels. Here the channels are visited in
/// [`UPSTREAM_CHANNEL_ORDER`] and the winner is taken with `Iterator::min_by_key`, documented
/// to return the *first* of several equal minima; `sort_unstable_by_key` would not be
/// equivalent. This is unlike `DEVIATION(23)`, whose upstream tie order comes from
/// `np.argsort`'s introsort and has no ordering contract at all.
///
/// Errors (`StageError::InvalidInput`, per-image) when `image` and `mask` differ in size.
pub fn otsu_thresh_mask_list(
    image: &RgbImage,
    mask: &GrayImage,
) -> Result<Vec<MaskCandidate>, StageError> {
    if image.dimensions() != mask.dimensions() {
        return Err(StageError::InvalidInput(format!(
            "image {:?} and mask {:?} must have the same size",
            image.dimensions(),
            mask.dimensions()
        )));
    }

    let mut candidates = Vec::with_capacity(UPSTREAM_CHANNEL_ORDER.len());
    for &index in &UPSTREAM_CHANNEL_ORDER {
        let channel = GrayImage::from_fn(image.width(), image.height(), |x, y| {
            Luma([image.get_pixel(x, y).0[index]])
        });
        let threshed = threshold_binary(&channel, otsu_threshold(&channel));
        candidates.push(minxor_thresh(&threshed, mask)?);
    }

    let winner = candidates
        .into_iter()
        .min_by_key(|candidate| candidate.xor_sum)
        .expect("UPSTREAM_CHANNEL_ORDER is non-empty");
    Ok(vec![winner])
}

/// `refine_mask`'s candidate list (upstream `textmask.py:206-207`):
///
/// ```text
/// mask_list = get_topk_masklist(im, msk)
/// mask_list += get_otsuthresh_masklist(im, msk, per_channel=False)
/// ```
///
/// **The order is part of the contract, and A3 depends on it.** This is a concatenation, so
/// the Otsu candidate is **last**, after however many top-k candidates there are (1 to 3 --
/// measured on the committed page: 3 for blocks 0-2 and 1 for block 3). `merge_mask_list`
/// then does its own `mask_list.sort(key=lambda x: x[1])` over the combined list, and that
/// sort is stable, so the position of two equally-scoring candidates in *this* list decides
/// which one the merge loop visits first. A `HashMap`, a `HashSet` or any reordering here
/// would silently change the merge, which is why the return type is a `Vec` and why nothing
/// in this function sorts.
///
/// Errors (`StageError::InvalidInput`, per-image) when `image` and `mask` differ in size.
pub fn candidate_mask_list(
    image: &RgbImage,
    mask: &GrayImage,
) -> Result<Vec<MaskCandidate>, StageError> {
    let grey = rgb_to_gray(image);
    let mut list = top_k_mask_list(&grey, mask)?;
    list.extend(otsu_thresh_mask_list(image, mask)?);
    Ok(list)
}
