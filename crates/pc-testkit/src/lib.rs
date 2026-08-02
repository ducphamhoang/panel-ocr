//! `pc-testkit` -- spec §1 (crate list), §7 (fixtures and the mocked ML boundary),
//! §10.7(B), §11.7(B), §12.7(A). Dev-dependency only; never shipped.
//!
//! Three jobs:
//!   1. **Fixture paths** (`paths`) -- find `tests/fixtures/{upstream,recorded}/...`
//!      from any crate's tests, independent of the current working directory, plus
//!      §7.2's rebasing of relative `ImageHandle` paths in recorded JSON.
//!   2. **Metrics** (`metrics`) -- SSIM, pixel-set IoU, max per-channel delta, mean
//!      absolute difference, exact-equal fraction. These are the numbers §10.7(B),
//!      §11.7(B)12/13 and §12.7(A)2 are written in terms of, so their *definitions*
//!      are part of the test contract and are pinned by this crate's own tests
//!      (spec §16.5 item 7).
//!   3. **Golden compare** (`golden`) -- the report shape that `cargo xtask
//!      calibrate-goldens` (F2) writes into `docs/GOLDEN_CALIBRATION.md`, and the
//!      assertion helpers the frozen parity gates call.
//!
//! Panic policy: this is test-only code, so argument errors (mismatched dimensions,
//! missing fixture files) **panic with a diagnostic message** rather than returning
//! `Result`. Ergonomics in assertions beat error plumbing here.

pub mod golden;
pub mod images;
pub mod metrics;
pub mod model_signature;
pub mod ocr_model_signature;
pub mod paths;
pub mod provenance;

pub use golden::{GoldenReport, GoldenThresholds};
pub use metrics::{PixelSet, SsimParams};
pub use paths::{BubbleKind, DemoBubble, DEMO_BUBBLES, LONG_STRIP_SIZE};

/// `assert!((a - b).abs() <= eps)` with a message that prints all three numbers.
/// Used pervasively by downstream stage tests, so it lives here rather than being
/// re-declared in each crate.
#[track_caller]
pub fn assert_close(actual: f64, expected: f64, eps: f64) {
    assert!(
        eps >= 0.0 && (actual - expected).abs() <= eps,
        "values are not close: actual={actual}, expected={expected}, epsilon={eps}"
    );
}
