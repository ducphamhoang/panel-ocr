//! Long-strip splitting (task D8).
//!
//! The algorithm is fully pinned by spec §16.6 item 8; §8.7(A)7 and §8.7(B)8 are the
//! acceptance criteria. Nothing here is heuristic-by-choice: every constant and every
//! tie-break is specified, because §5.7 requires identical inputs to produce identical
//! outputs.
//!
//! Coordinate convention: a "split row" `y` means the boundary *above* row `y`, so
//! segments are the half-open row ranges `[0, y0) [y0, y1) ... [yk, height)`. Row `0`
//! can therefore never be a legal split (it would produce an empty leading segment) --
//! which is why `row_scores()[0]` is `+infinity`.

use image::{ImageBuffer, Pixel, RgbImage};
use pc_core::StageError;
use std::ops::Range;

/// Splitting parameters, mirroring the `[general]` keys of spec §6
/// (`preferred_split_height`, `split_tolerance_margin`, `split_long_strips`,
/// `long_strip_aspect_ratio`) without depending on `pc-config` (§1 rule 2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitParams {
    pub preferred_height: u32,
    pub tolerance_margin: u32,
    pub split_long_strips: bool,
    pub max_aspect_ratio: f64,
}

impl SplitParams {
    /// spec §6 `[general]` defaults.
    pub fn defaults() -> Self {
        Self {
            preferred_height: 2000,
            tolerance_margin: 500,
            split_long_strips: true,
            max_aspect_ratio: 0.33,
        }
    }
}

impl Default for SplitParams {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Per-row "how much horizontal change is in this row" score, spec §16.6 item 8:
/// `score[y] = Σ_x (luma[y][x] - luma[y][x+1])²` over the *row* (a horizontal
/// difference, not a row-to-row one). A flat row -- a gutter between panels -- scores
/// near zero, which is what makes the minimum a good cut.
///
/// `score[0]` is `f64::INFINITY`: row 0 is never a legal split (see module docs).
pub fn row_scores(image: &RgbImage) -> Vec<f64> {
    let (width, height) = image.dimensions();
    let mut scores = Vec::with_capacity(height as usize);

    for y in 0..height {
        if y == 0 {
            scores.push(f64::INFINITY);
            continue;
        }
        let mut score = 0.0_f64;
        let mut previous = luma(image, 0, y);
        for x in 1..width {
            let current = luma(image, x, y);
            let delta = previous - current;
            score += delta * delta;
            previous = current;
        }
        scores.push(score);
    }

    scores
}

/// Per-channel-averaged grayscale (spec §16.6 item 8), in f64 so the squared sum has
/// no accumulation surprises on an 8000-row strip.
fn luma(image: &RgbImage, x: u32, y: u32) -> f64 {
    let pixel = image.get_pixel(x, y).0;
    (f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2])) / 3.0
}

/// The half-open search window for each split, spec §16.6 item 8.
///
/// Returns an empty vector when splitting is disabled or the aspect gate rejects the
/// image, so `calculate_best_splits` is exactly "argmin the score inside each range".
/// Windows are `[preferred*i - tolerance, preferred*i + tolerance)` clamped to
/// `1..height`; a window that clamps to empty is dropped.
pub fn search_ranges(image_size: (u32, u32), params: &SplitParams) -> Vec<Range<u32>> {
    let (width, height) = image_size;

    if !params.split_long_strips || params.preferred_height == 0 || height == 0 || width == 0 {
        return Vec::new();
    }

    // spec §16.6 item 8 aspect gate: only tall-and-narrow images are strips.
    if f64::from(width) / f64::from(height) > params.max_aspect_ratio {
        return Vec::new();
    }

    // spec §16.6 item 8: n = round(height / preferred), clamped to >= 1; splits = n - 1.
    let ratio = f64::from(height) / f64::from(params.preferred_height);
    let segments = ratio.round().max(1.0) as u64;

    (1..segments)
        .filter_map(|index| {
            let centre = u64::from(params.preferred_height) * index;
            let start = centre
                .saturating_sub(u64::from(params.tolerance_margin))
                .max(1);
            let end = (centre + u64::from(params.tolerance_margin)).min(u64::from(height));
            (start < end).then(|| start as u32..end as u32)
        })
        .collect()
}

/// The split rows for `image`, spec §16.6 item 8: minimum-score row within each search
/// window, ties broken toward the smallest row index (§5.7 determinism).
///
/// The result is always strictly increasing and inside `1..height`, i.e. always a legal
/// argument to [`split_image`].
pub fn calculate_best_splits(image: &RgbImage, params: &SplitParams) -> Vec<u32> {
    let ranges = search_ranges(image.dimensions(), params);
    if ranges.is_empty() {
        return Vec::new();
    }

    let scores = row_scores(image);
    let mut splits: Vec<u32> = Vec::with_capacity(ranges.len());

    for range in ranges {
        let mut best_row = None;
        let mut best_score = f64::INFINITY;
        for row in range {
            let score = scores[row as usize];
            // Strict `<` while scanning ascending == "ties break toward the smallest row".
            if best_row.is_none() || score < best_score {
                best_row = Some(row);
                best_score = score;
            }
        }
        // Overlapping windows (tolerance > preferred/2) could otherwise yield a
        // non-increasing list, which `split_image` rejects.
        if let Some(row) = best_row {
            if splits.last().is_none_or(|last| row > *last) {
                splits.push(row);
            }
        }
    }

    splits
}

/// Cut `image` into `splits.len() + 1` vertically contiguous segments at the given
/// rows. Segment *i* covers the half-open row range `[splits[i-1], splits[i])`, with
/// the first starting at `0` and the last ending at `height`.
///
/// Bit-exact by construction (raw subpixel memcpy, no resampling) -- §8.7(A)7 requires
/// `split` then `stitch` to reproduce the source exactly.
///
/// `StageError::InvalidInput` if `splits` is not strictly increasing or any row is
/// outside `1..height`.
pub fn split_image<P: Pixel + 'static>(
    image: &ImageBuffer<P, Vec<P::Subpixel>>,
    splits: &[u32],
) -> Result<Vec<ImageBuffer<P, Vec<P::Subpixel>>>, StageError> {
    let (width, height) = image.dimensions();

    let mut previous = 0_u32;
    for &row in splits {
        if row == 0 || row >= height {
            return Err(StageError::InvalidInput(format!(
                "split row {row} is outside 1..{height}"
            )));
        }
        if row <= previous {
            return Err(StageError::InvalidInput(
                "split rows must be strictly increasing".into(),
            ));
        }
        previous = row;
    }

    let channels = usize::from(P::CHANNEL_COUNT);
    let row_len = width as usize * channels;
    let raw = image.as_raw();

    let mut bounds = Vec::with_capacity(splits.len() + 1);
    let mut start = 0_u32;
    for &row in splits {
        bounds.push((start, row));
        start = row;
    }
    bounds.push((start, height));

    bounds
        .into_iter()
        .map(|(top, bottom)| {
            let buffer = raw[top as usize * row_len..bottom as usize * row_len].to_vec();
            ImageBuffer::from_raw(width, bottom - top, buffer).ok_or_else(|| {
                StageError::Empty(format!("segment [{top}, {bottom}) has no pixels"))
            })
        })
        .collect()
}

/// Inverse of [`split_image`]: concatenate `segments` vertically, in order.
///
/// `StageError::InvalidInput` when `segments` is empty or the widths disagree.
pub fn stitch_images<P: Pixel + 'static>(
    segments: &[ImageBuffer<P, Vec<P::Subpixel>>],
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, StageError> {
    let first = segments.first().ok_or_else(|| {
        StageError::InvalidInput("cannot stitch an empty list of segments".into())
    })?;

    let width = first.width();
    if let Some(bad) = segments.iter().find(|segment| segment.width() != width) {
        return Err(StageError::InvalidInput(format!(
            "segment widths disagree: expected {width}, found {}",
            bad.width()
        )));
    }

    let height: u32 = segments.iter().map(ImageBuffer::height).sum();
    let mut buffer =
        Vec::with_capacity(height as usize * width as usize * usize::from(P::CHANNEL_COUNT));
    for segment in segments {
        buffer.extend_from_slice(segment.as_raw());
    }

    ImageBuffer::from_raw(width, height, buffer)
        .ok_or_else(|| StageError::Empty("stitched image has no pixels".into()))
}
