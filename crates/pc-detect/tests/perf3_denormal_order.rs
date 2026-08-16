#![cfg(all(feature = "onnx", target_arch = "x86_64"))]
//! FROZEN. Gate for §16.52 item 5's D1: the detector's denormal flushing must not depend
//! on which ONNX Runtime session happened to initialize first in the process.
//!
//! This is deliberately the ONLY test in this file (a single-test-per-file discipline),
//! because it is process-ordering-sensitive and cargo runs tests in threads of one
//! process -- exactly the property whose absence hid the original bug (§16.52 item 1:
//! `perf2_mxcsr_hygiene.rs`'s `flush_denormals_does_not_escape_the_detector_worker_thread`
//! constructs its FLUSHING detector first, so it always wins ONNX Runtime's process-wide
//! `set_denormal_as_zero` once-flag and never observes the real production failure mode).

use pc_detect::onnx::{OnnxDetector, SessionTuning};
use pc_detect::TextDetector;
use std::path::PathBuf;
use std::sync::Mutex;

static REAL_MODEL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn model() -> PathBuf {
    PathBuf::from(
        std::env::var_os("PANEL_OCR_ONNX_MODEL").expect("PANEL_OCR_ONNX_MODEL is required"),
    )
}

fn page() -> image::RgbImage {
    image::RgbImage::from_fn(700, 1000, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 200])
    })
}

/// §16.52 items 1-2. Measured root cause this pins (`ort` 2.0.0-rc.12 + its bundled ONNX
/// Runtime 1.24.2, 2026-08-16): `session.set_denormal_as_zero` reaches a session's own
/// intra-op POOL threads per-session, but reaches the thread that CALLS `Session::run`
/// only through a process-wide `std::call_once` that the first session to initialize in
/// the process consumes. In the real `run_clean`/`run_ocr` CLI paths, the eagerly
/// constructed OCR encoder wins this deterministically (§16.52 item 2) -- so before D1's
/// fix, whether the detector's own worker thread ever flushed denormals during a real
/// inference depended on an ordering the detector itself never controlled.
#[test]
#[ignore = "opt-in: needs PANEL_OCR_ONNX_MODEL and ONNX Runtime; run with \
            `PANEL_OCR_ONNX_MODEL=/path/comictextdetector.pt.onnx cargo test -p pc-detect \
             --features onnx,testkit,bench-tuning --test perf3_denormal_order -- --ignored --nocapture`"]
fn the_detector_worker_flushes_denormals_even_when_a_non_flushing_session_initialized_first() {
    let _test_guard = REAL_MODEL_TEST_LOCK
        .lock()
        .expect("real-model test lock must not be poisoned");

    // Deliberately consume ONNX Runtime's process-wide denormal once-flag with a session
    // that does NOT request flushing, constructed FIRST. This ordering IS the regression
    // §16.52 root-caused -- reproducing it here is the entire point of this test's shape.
    let non_flushing_tuning = SessionTuning {
        flush_denormals: false,
        ..SessionTuning::default()
    };
    let non_flushing = OnnxDetector::from_path_with_tuning(&model(), 0, 0, &non_flushing_tuning)
        .expect("non-flushing session must open");

    let flushing = OnnxDetector::from_path_with_tuning(&model(), 0, 0, &SessionTuning::default())
        .expect("flushing session must open");

    let image = page();

    // Before D1's fix, `last_inference_flush_state()` doesn't exist at all -- this test is
    // genuinely red at compile time until D1 lands, matching this project's test-first
    // convention.
    non_flushing
        .detect(&image)
        .expect("non-flushing detector must complete a real inference");
    let non_flushing_observed = non_flushing
        .last_inference_flush_state()
        .expect("a real inference on x86_64 must record an observed flush state");

    flushing
        .detect(&image)
        .expect("flushing detector must complete a real inference");
    let flushing_observed = flushing
        .last_inference_flush_state()
        .expect("a real inference on x86_64 must record an observed flush state");

    // A PAIR, not a single boolean. `(false, false)` is today's defect (both starved by
    // the poisoned once-flag). `(true, true)` is the tempting wrong fix -- a process-wide
    // or unconditional flush -- which would make `SessionTuning { flush_denormals: false }`
    // unrepresentable and silently change float semantics for every pure-Rust stage
    // sharing a thread (the exact hazard §16.32 exists to prevent). Only `(false, true)`
    // is the ratified behaviour, and only an assertion over both arms can tell the three
    // apart; a one-sided "the flushing detector flushes" passes for all of them.
    assert_eq!(
        (
            non_flushing_observed.flush_to_zero,
            non_flushing_observed.denormals_are_zero,
            flushing_observed.flush_to_zero,
            flushing_observed.denormals_are_zero,
        ),
        (false, false, true, true),
        "(non-flushing worker: flush_to_zero, denormals_are_zero, \
         flushing worker: flush_to_zero, denormals_are_zero) -- construction order must \
         not determine whether the flushing detector actually flushes during real inference"
    );
}
