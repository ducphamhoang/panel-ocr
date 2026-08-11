//! Structural + numeric comparison of two equal-shape f32 tensors from the same model
//! run on different execution providers/policies. Requirement: distinguish "identical"
//! from "close" from "diverged" without asserting anything about the tolerance a caller
//! should accept — that judgment belongs to the report, not this function.
//!
//! GPU-3 G3-D: these pure functions back the `cargo xtask ocr-device-compare` producer
//! (`xtask/src/ocr_device_compare.rs`) and its tests.

/// How two equal-length `f32` slices from the same model's outputs differ.
///
/// `bitwise_differing_count` counts positions whose f32 bit patterns differ. Two `NaN`s
/// of the same bit pattern are **not** counted as differing: `compare_tensors` compares
/// with bit-equality (`to_bits`), not IEEE 754 `==`, so a matching `NaN` payload is
/// "identical" the same way a matching finite value is. That decision is load-bearing
/// for the encoder hidden states, which can legitimately contain `NaN` if either
/// execution path ever produces one — a same-bit `NaN` on both sides is not evidence of
/// divergence, and a differing bit pattern is.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDivergence {
    pub len: usize,
    pub bitwise_differing_count: usize,
    pub max_abs_delta: f32,
    pub mean_abs_delta: f32,
}

/// Compare two equal-length `f32` slices element by element.
///
/// `Err` when the lengths differ — a caller passing different-length tensors is a bug,
/// not a measurement. `max_abs_delta`/`mean_abs_delta` use IEEE 754 `abs(diff)`; for a
/// `NaN`-on-`NaN` position the delta is `NaN` (IEEE 754 `NaN - NaN`), which propagates
/// into both aggregates. The count is bitwise, so a same-bit `NaN` pair contributes `0`
/// to `bitwise_differing_count` but still contributes `NaN` to the deltas — the two
/// columns answer different questions and are deliberately not forced to agree.
pub fn compare_tensors(left: &[f32], right: &[f32]) -> Result<TensorDivergence, String> {
    if left.len() != right.len() {
        return Err(format!(
            "cannot compare tensors of different lengths: {} vs {}",
            left.len(),
            right.len()
        ));
    }
    let mut bitwise_differing_count = 0usize;
    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f32;
    for (&l, &r) in left.iter().zip(right) {
        if l.to_bits() != r.to_bits() {
            bitwise_differing_count += 1;
        }
        let delta = (l - r).abs();
        // NaN propagation done by hand: Rust's `f32::max` treats NaN as *less than* every
        // number (it is `if rhs > self { rhs } else { self }`), so `0.0f32.max(NaN)` is
        // `0.0` — the opposite of IEEE 754 `fmax`, which returns NaN. A NaN delta is the
        // honest signal that one side had a NaN where the other did not (or both did with
        // different payloads), so it must survive into the aggregates, never be swallowed
        // by a `max` that ranks it as "smallest".
        max_abs_delta = if delta.is_nan() {
            f32::NAN
        } else {
            max_abs_delta.max(delta)
        };
        sum_abs_delta += delta;
    }
    Ok(TensorDivergence {
        len: left.len(),
        bitwise_differing_count,
        max_abs_delta,
        mean_abs_delta: sum_abs_delta / left.len() as f32,
    })
}

/// The first index at which two token-ID sequences differ; `None` if one is a prefix of
/// the other or they're identical up to the shorter length. `Some(min_len)` when the
/// shorter is a strict prefix of the longer (that IS a divergence — the shorter one
/// stopped, e.g. hit EOS, where the other didn't).
pub fn first_divergence_index(left: &[u32], right: &[u32]) -> Option<usize> {
    let min_len = left.len().min(right.len());
    for (index, (&l, &r)) in left.iter().zip(right).enumerate() {
        if l != r {
            return Some(index);
        }
    }
    (left.len() != right.len()).then_some(min_len)
}

/// The minimum (closest) top-2 logit margin across a set of logit rows — the number that
/// tells you how close beam search's decisions were to going the other way. A small
/// margin means a tiny numeric perturbation could have flipped the decoded output; a
/// large margin means the decode was robust to it.
///
/// Panics if any row is empty or has fewer than 2 entries (a decoder vocab always has
/// far more than 2 tokens; treat a smaller row as a caller bug, not a data condition to
/// handle gracefully).
pub fn min_top2_margin(rows: &[Vec<f32>]) -> f32 {
    let mut minimum = f32::INFINITY;
    for row in rows {
        if row.len() < 2 {
            panic!(
                "min_top2_margin requires rows of at least 2 logits; got a row of {}",
                row.len()
            );
        }
        let mut largest = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for &logit in row {
            if logit >= largest {
                second = largest;
                largest = logit;
            } else if logit > second {
                second = logit;
            }
        }
        let margin = largest - second;
        // NaN propagation done by hand, same reason as `compare_tensors`: Rust's
        // `f32::min` treats NaN as less than every number, so `minimum.min(NaN)` would
        // silently rank a NaN margin as "closer than any real margin". A NaN logit must
        // surface as a NaN margin, not be hidden.
        minimum = if margin.is_nan() {
            f32::NAN
        } else {
            minimum.min(margin)
        };
    }
    minimum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_zero_differences() {
        let result = compare_tensors(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).expect("equal length");
        assert_eq!(result.len, 3);
        assert_eq!(result.bitwise_differing_count, 0);
        assert_eq!(result.max_abs_delta, 0.0);
        assert_eq!(result.mean_abs_delta, 0.0);
    }

    #[test]
    fn mismatch_is_a_bitwise_and_numeric_divergence() {
        let result = compare_tensors(&[1.0, 2.0], &[1.0, 2.5]).expect("equal length");
        assert_eq!(result.len, 2);
        assert_eq!(result.bitwise_differing_count, 1);
        assert_eq!(result.max_abs_delta, 0.5);
        assert_eq!(result.mean_abs_delta, 0.25);
    }

    #[test]
    fn same_bit_nan_is_not_bitwise_differing() {
        let result = compare_tensors(&[f32::NAN], &[f32::NAN]).expect("equal length");
        assert_eq!(result.len, 1);
        assert_eq!(
            result.bitwise_differing_count, 0,
            "two NaNs with the same bit pattern are the same value on the wire"
        );
        // IEEE 754: NaN - NaN is NaN, so the deltas are NaN too — honest, not 0.
        assert!(result.max_abs_delta.is_nan());
        assert!(result.mean_abs_delta.is_nan());
    }

    #[test]
    fn mismatched_lengths_are_an_error() {
        let error = compare_tensors(&[1.0], &[1.0, 2.0]).expect_err("lengths differ");
        assert!(error.contains("different lengths"), "{error}");
    }

    #[test]
    fn identical_sequences_have_no_first_divergence() {
        assert_eq!(first_divergence_index(&[1, 2, 3], &[1, 2, 3]), None);
    }

    #[test]
    fn differing_sequences_report_the_first_differing_index() {
        assert_eq!(first_divergence_index(&[1, 2, 3], &[1, 2, 4]), Some(2));
    }

    #[test]
    fn a_strict_prefix_is_a_divergence_at_the_shorter_length() {
        assert_eq!(first_divergence_index(&[1, 2], &[1, 2, 3]), Some(2));
        assert_eq!(first_divergence_index(&[1, 2, 3], &[1, 2]), Some(2));
    }

    #[test]
    fn min_top2_margin_takes_the_minimum_across_rows() {
        let rows = vec![vec![5.0, 3.0, 1.0], vec![5.0, 4.9, 1.0]];
        // Hand-derived f32: 5.0 - 4.9 = 0.099999905 (4.9f32 is 4.900000095367431640625).
        assert_eq!(min_top2_margin(&rows), 0.099999905);
    }

    #[test]
    #[should_panic(expected = "at least 2")]
    fn min_top2_margin_panics_on_a_single_logit_row() {
        min_top2_margin(&[vec![1.0]]);
    }
}
