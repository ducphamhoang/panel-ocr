//! GPU-1 G1-D resolved-policy and provenance-derivation gates — §16.36 item 5.
//!
//! FROZEN (CLAUDE.md). `xtask` is currently binary-only, so the production implementation
//! places its small pure policy seam in `src/device.rs`; including that module here keeps
//! the second guard testable without exporting maintainer internals or constructing ORT.

#[path = "../src/device.rs"]
mod device;

use device::{ensure_recording_policy, recorded_session_pins, RecordingSession};
use pc_core::device::{resolve, Device, DevicePolicy, DeviceSupport};

fn cuda_policy() -> DevicePolicy {
    resolve(Device::Cuda, DeviceSupport::WITH_CUDA)
        .expect("the explicit WITH_CUDA capability makes the resolved-policy arm reachable")
}

#[test]
fn detector_recording_refuses_a_resolved_cuda_policy_before_session_construction() {
    let error = ensure_recording_policy(&cuda_policy(), RecordingSession::DetectorFixture)
        .expect_err("committed detector recording must remain CPU-only after resolution");

    assert_eq!(
        error.to_string(),
        "detector fixture recording refuses resolved execution provider `cuda`; only `cpu` is ratified for committed recordings (§16.22 item 5(b))"
    );
}

#[test]
fn calibrate_goldens_refuses_a_resolved_cuda_policy_before_session_construction() {
    let error = ensure_recording_policy(&cuda_policy(), RecordingSession::CalibrateGoldens)
        .expect_err("golden calibration must remain CPU-only after resolution");

    assert_eq!(
        error.to_string(),
        "golden calibration refuses resolved execution provider `cuda`; only `cpu` is ratified for committed recordings (§16.22 item 5(b))"
    );
}

/// The producer must take the same config value that is handed to
/// `OnnxDetector::from_path_with_config`. Non-default counts make a hidden `0` literal fail.
#[test]
fn detector_provenance_is_derived_from_the_resolved_policy_and_actual_session_config() {
    let mut config = pc_config::TextDetectorConfig::default();
    config.intra_threads = 7;
    config.inter_threads = 3;
    let policy = DevicePolicy::cpu();

    ensure_recording_policy(&policy, RecordingSession::DetectorFixture)
        .expect("CPU recording is ratified");
    let pins = recorded_session_pins(&policy, &config);

    assert_eq!(
        pins.execution_provider,
        policy.provenance_execution_provider()
    );
    assert_eq!(pins.intra_threads, config.intra_threads);
    assert_eq!(pins.inter_threads, config.inter_threads);
    assert_eq!(
        (
            pins.execution_provider,
            pins.intra_threads,
            pins.inter_threads
        ),
        ("cpu", 7, 3)
    );
}
