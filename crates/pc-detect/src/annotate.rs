//! Task A1 -- spec §16.37 item 8: the greyscale / histogram / top-k-colour path of
//! `MaskRefineMode::Annotation`, ported from upstream PanelCleaner's
//! `comic_text_detector/utils/textmask.py::get_topk_masklist` and
//! `imgproc_utils.py::expand_textwindow` (pinned commit
//! `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`).
//!
//! **This module is not wired into anything.** `pc_detect::run` still rejects
//! `MaskRefineMode::Annotation` (§16.37's preamble: *"Nothing here asserts that
//! `Annotation` mode works, or exists."*). Wiring is task A4; Otsu + the XOR-minimising
//! channel selection are task A2; connected components, the merge loop and hole filling
//! are task A3. None of those are implemented here.
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
//! is deliberately **not** here: `minxor_thresh` is the XOR-minimising selection that
//! §16.37 item 8 assigns to A2.

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
