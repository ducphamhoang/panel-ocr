//! GPU-3 G3-D — the pure comparator library behind the OCR CUDA divergence measurement
//! (`crates/pc-ocr/src/device_divergence.rs`).
//!
//! Every expected literal below was derived by hand (a calculator + the spec of each
//! function), never by running the function under test and copying its output — cookbook
//! rules 7/13. The one case where the hand-derived mathematical value is not exactly
//! representable in f32 (`5.0 - 4.9` = `0.099999905`, not `0.1`) is documented at its
//! test with the by-hand f32 derivation. Part 1 of the G3-D brief is `default`-tier: no
//! `onnx`, no `cuda`, no models, no GPU.

use pc_ocr::device_divergence::{compare_tensors, first_divergence_index, min_top2_margin};

#[test]
fn compare_tensors_on_two_identical_vectors_is_identical() {
    let result = compare_tensors(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).expect("equal lengths");
    assert_eq!(result.len, 3);
    assert_eq!(result.bitwise_differing_count, 0);
    assert_eq!(result.max_abs_delta, 0.0);
    assert_eq!(result.mean_abs_delta, 0.0);
}

#[test]
fn compare_tensors_reports_one_bitwise_and_half_a_unit_delta() {
    let result = compare_tensors(&[1.0, 2.0], &[1.0, 2.5]).expect("equal lengths");
    assert_eq!(result.len, 2);
    assert_eq!(result.bitwise_differing_count, 1);
    assert_eq!(result.max_abs_delta, 0.5);
    assert_eq!(result.mean_abs_delta, 0.25);
}

/// The brief requires this case to be decided and documented, not left implicit. The
/// documented decision (in the module doc + the function doc): `compare_tensors` counts
/// **bitwise** differences (`f32::to_bits`), so two `NaN`s of the same bit pattern are
/// NOT "differing" — the two values are the same value on the wire, and `NaN != NaN` in
/// IEEE 754 is a comparison artifact, not evidence of divergence. The deltas stay `NaN`
/// because `NaN - NaN` is `NaN`; that too is asserted, so the test pins both halves of
/// the behavior.
#[test]
fn compare_tensors_on_same_bit_nan_pairs_is_not_bitwise_differing() {
    let result = compare_tensors(&[f32::NAN], &[f32::NAN]).expect("equal lengths");
    assert_eq!(result.len, 1);
    assert_eq!(result.bitwise_differing_count, 0);
    assert!(result.max_abs_delta.is_nan());
    assert!(result.mean_abs_delta.is_nan());
}

#[test]
fn compare_tensors_on_mismatched_lengths_errors() {
    let error = compare_tensors(&[1.0, 2.0, 3.0], &[1.0, 2.0]).expect_err("lengths differ");
    assert!(
        error.contains("different lengths"),
        "the error must say lengths differ; got {error}"
    );
}

#[test]
fn first_divergence_index_on_identical_sequences_is_none() {
    assert_eq!(first_divergence_index(&[1, 2, 3], &[1, 2, 3]), None);
}

#[test]
fn first_divergence_index_reports_the_first_differing_position() {
    assert_eq!(first_divergence_index(&[1, 2, 3], &[1, 2, 4]), Some(2));
}

#[test]
fn first_divergence_index_counts_a_strict_prefix_as_a_divergence() {
    // The shorter sequence stopped where the longer continued — the shorter hit EOS (or
    // the other kept going), which IS a divergence.
    assert_eq!(first_divergence_index(&[1, 2], &[1, 2, 3]), Some(2));
}

#[test]
fn min_top2_margin_takes_the_minimum_margin_across_rows() {
    // Row [5.0, 3.0, 1.0]: top two are 5.0 and 3.0, margin 2.0 exactly.
    // Row [5.0, 4.9, 1.0]: top two are 5.0 and 4.9, margin 5.0 - 4.9.
    //   In f32 that difference is NOT the decimal 0.1: 4.9f32 is
    //   4.900000095367431640625 (nearest f32 to 4.9), so the margin is
    //   0.099999904632568359375, which is 0.099999905 to 8 significant digits. The
    //   brief's "0.1" is the hand-derived mathematical value; the exact f32 difference
    //   is this literal, derived by hand from the f32 representations — not by running
    //   the function.
    // Minimum across both rows: 0.099999905.
    let rows = vec![vec![5.0, 3.0, 1.0], vec![5.0, 4.9, 1.0]];
    assert_eq!(min_top2_margin(&rows), 0.099999905);
}
