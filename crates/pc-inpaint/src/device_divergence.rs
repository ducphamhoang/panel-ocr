//! GPU-4 (G4-B) — the pure comparator library behind `cargo xtask lama-device-compare`
//! (not yet written; G4-C). Default tier, no `ort`, no model, no GPU.
//!
//! Deliberately does NOT reuse `pc_ocr::device_divergence::compare_tensors`: a
//! `pc-inpaint` -> `pc-ocr` dependency would be a forbidden stage-crate-to-stage-crate
//! edge (§1 rule 2). The future `xtask` producer may import `pc_ocr::device_divergence`
//! directly for any f32-tensor comparison it needs (`xtask` already depends on both
//! stage crates) -- this module holds only what is specific to LaMa's byte-level and
//! quantization-margin comparisons.

/// How two equal-length `u8` buffers differ, byte by byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteDivergence {
    /// The (common) buffer length.
    pub len: usize,
    /// How many positions hold different bytes.
    pub differing_count: usize,
    /// The largest `abs(left[i] - right[i])` across all positions.
    pub max_abs_delta: u8,
}

/// Compare two equal-length `u8` buffers byte by byte. `Err` when the lengths differ --
/// a caller passing different-length buffers is a bug, not a measurement.
pub fn compare_bytes(left: &[u8], right: &[u8]) -> Result<ByteDivergence, String> {
    if left.len() != right.len() {
        return Err(format!(
            "cannot compare byte buffers of different lengths: {} vs {}",
            left.len(),
            right.len()
        ));
    }
    let mut differing_count = 0usize;
    let mut max_abs_delta = 0u8;
    for (&l, &r) in left.iter().zip(right) {
        if l != r {
            differing_count += 1;
            max_abs_delta = max_abs_delta.max(l.abs_diff(r));
        }
    }
    Ok(ByteDivergence {
        len: left.len(),
        differing_count,
        max_abs_delta,
    })
}

/// As [`compare_bytes`], but counting only positions where `selector[i]` is true. `Err`
/// when `selector`'s length does not match `left`/`right`'s. Used to restrict a
/// page-level comparison to the write region (e.g. `final_mask > 0`) rather than the
/// whole artifact, which is mostly untouched original pixels.
pub fn compare_bytes_where(
    left: &[u8],
    right: &[u8],
    selector: &[bool],
) -> Result<ByteDivergence, String> {
    if left.len() != right.len() {
        return Err(format!(
            "cannot compare byte buffers of different lengths: {} vs {}",
            left.len(),
            right.len()
        ));
    }
    if selector.len() != left.len() {
        return Err(format!(
            "the selector has length {} but the buffers have length {}",
            selector.len(),
            left.len()
        ));
    }
    let mut differing_count = 0usize;
    let mut max_abs_delta = 0u8;
    for ((&l, &r), &selected) in left.iter().zip(right).zip(selector) {
        if selected && l != r {
            differing_count += 1;
            max_abs_delta = max_abs_delta.max(l.abs_diff(r));
        }
    }
    Ok(ByteDivergence {
        len: left.len(),
        differing_count,
        max_abs_delta,
    })
}

/// The minimum distance any sample sits from a byte-rounding tie: `min over s of
/// |frac(s * 255.0) - 0.5|`. A value near 0.0 means some sample sits almost exactly on a
/// rounding boundary, where a tiny numeric perturbation could flip the rounded byte; a
/// value near 0.5 means every sample is far from a boundary. Panics on an empty slice --
/// a caller passing zero samples is a bug, not a data condition to handle gracefully.
pub fn min_quantization_margin(samples: &[f32]) -> f32 {
    assert!(
        !samples.is_empty(),
        "min_quantization_margin requires at least one sample"
    );
    let mut minimum = f32::INFINITY;
    for &sample in samples {
        let scaled = sample * 255.0;
        let fraction = scaled - scaled.floor();
        let margin = (fraction - 0.5).abs();
        // NaN propagation done by hand, the same way `pc_ocr::device_divergence`
        // documents it: Rust's `f32::min` treats NaN as less than every number, so
        // `minimum.min(NaN)` would silently rank a NaN margin as "closer than any real
        // margin". A NaN sample must surface as a NaN margin, not be hidden.
        minimum = if margin.is_nan() {
            f32::NAN
        } else {
            minimum.min(margin)
        };
    }
    minimum
}
