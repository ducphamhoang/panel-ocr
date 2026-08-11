//! GPU-4 (G4-B) — the pure comparator library behind `cargo xtask lama-device-compare`
//! (not yet written; G4-C). Default tier: no `ort`, no model, no GPU.
//!
//! Every literal below is hand-derived, per the cookbook rules (7/13) the brief cites:
//! the expected numbers are computed by arithmetic written out in the comments, never
//! read back from the function under test.

use pc_inpaint::device_divergence::{compare_bytes, compare_bytes_where, min_quantization_margin};

#[test]
fn compare_bytes_on_identical_buffers_reports_no_difference() {
    let result = compare_bytes(&[10, 40, 99], &[10, 40, 99]).expect("equal length");
    assert_eq!(result.len, 3);
    assert_eq!(result.differing_count, 0);
    assert_eq!(result.max_abs_delta, 0);
}

#[test]
fn compare_bytes_reports_the_differing_count_and_max_delta() {
    // [10, 40] vs [10, 43]: position 0 matches (10 == 10, delta 0); position 1 differs,
    // delta = |43 - 40| = 3. So differing_count = 1 and max_abs_delta = max(0, 3) = 3.
    let result = compare_bytes(&[10, 40], &[10, 43]).expect("equal length");
    assert_eq!(result.len, 2);
    assert_eq!(result.differing_count, 1);
    assert_eq!(result.max_abs_delta, 3);
}

#[test]
fn compare_bytes_rejects_mismatched_lengths_naming_the_mismatch() {
    let error = compare_bytes(&[10], &[10, 40]).expect_err("lengths differ");
    assert!(
        error.contains("different lengths"),
        "the error must name the length mismatch, not just be an Err; got {error:?}"
    );
}

#[test]
fn compare_bytes_where_counts_only_the_selected_positions() {
    // Selector [false, true, true] selects positions 1 and 2 only.
    // Position 1: |43 - 40| = 3. Position 2: |5 - 99| = 94. Both differ, so
    // differing_count = 2 and max_abs_delta = max(3, 94) = 94. Position 0 is excluded
    // entirely, even though both buffers carry 10 there.
    let result = compare_bytes_where(&[10, 40, 99], &[10, 43, 5], &[false, true, true])
        .expect("selector length matches");
    assert_eq!(result.len, 3);
    assert_eq!(result.differing_count, 2);
    assert_eq!(result.max_abs_delta, 94);
}

#[test]
fn compare_bytes_where_rejects_a_mismatched_selector() {
    let error =
        compare_bytes_where(&[10, 40], &[10, 43], &[true]).expect_err("selector length differs");
    assert!(
        error.contains("selector"),
        "the error must name the selector length mismatch; got {error:?}"
    );
}

#[test]
fn min_quantization_margin_on_a_sample_exactly_on_the_boundary_is_zero() {
    // 0.5 * 255.0 = 127.5; frac(127.5) = 0.5; margin = |0.5 - 0.5| = 0.0. The single
    // sample sits exactly on the rounding tie, so the minimum margin is 0.0.
    assert_eq!(min_quantization_margin(&[0.5]), 0.0);
}

#[test]
fn min_quantization_margin_derives_half_for_byte_exact_samples() {
    // 1.0 * 255.0 = 255.0; frac(255.0) = 0.0; margin = |0.0 - 0.5| = 0.5.
    // 0.0 * 255.0 = 0.0;   frac(0.0)   = 0.0; margin = |0.0 - 0.5| = 0.5.
    // min(0.5, 0.5) = 0.5.
    assert_eq!(min_quantization_margin(&[0.0, 1.0]), 0.5);
}

#[test]
#[should_panic]
fn min_quantization_margin_panics_on_an_empty_slice() {
    min_quantization_margin(&[]);
}
