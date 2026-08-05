//! Private detector-seam gate for GPU-1 G1-C (§16.36 items 4, 6, and 7).
//!
//! FROZEN (CLAUDE.md). `detector.rs` includes this as a test-only child module so the
//! assertion can read `OnnxProvider`'s existing private initialization counter.
#![allow(unexpected_cfgs)]

use super::*;
use pc_core::device::{resolve, Device, DeviceSupport};
use std::sync::atomic::Ordering;

#[cfg(not(feature = "cuda"))]
#[test]
fn a_cuda_device_refusal_is_attempted_once_and_declared_run_fatal() {
    let cache = tempfile::tempdir().expect("cache");
    let provider = OnnxProvider::new(
        None,
        None,
        cache.path(),
        &TextDetectorConfig::default(),
        Device::Cuda,
    );
    let expected = resolve(Device::Cuda, DeviceSupport::CPU_ONLY)
        .expect_err("CPU-only support must refuse cuda")
        .message();

    assert!(DetectorProvider::failures_are_run_fatal(&provider));
    for _ in 0..8 {
        let error = match provider.detector_for(Path::new("page.png")) {
            Err(error) => error,
            Ok(_) => panic!("CPU-only support must refuse the CUDA session"),
        };
        match error {
            StageError::Model(message) => assert_eq!(message, expected),
            other => panic!("expected StageError::Model, got {other:?}"),
        }
    }
    assert_eq!(
        provider.attempts.load(Ordering::SeqCst),
        1,
        "the CUDA refusal must be the one outcome stored in the existing OnceLock"
    );
}
