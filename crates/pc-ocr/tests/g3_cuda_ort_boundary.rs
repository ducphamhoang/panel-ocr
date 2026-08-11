#![cfg(feature = "onnx")]

use pc_core::device::DevicePolicy;
use pc_ocr::onnx::MangaOcrSessions;
use tempfile::tempdir;

/// `WITH_CUDA` explicitly, not `compiled()`: reachable in EVERY build, per §16.36 item 2
/// making `DeviceSupport` a value rather than a `cfg!` read. Only the non-`cuda` build
/// consumes it (the registration-failure arm); a `cuda` build's outcome depends on the
/// running machine, so the helper and its imports are gated with it to avoid an
/// orphaned-item warning under `clippy --all-features -D warnings`.
#[cfg(not(feature = "cuda"))]
fn cuda_policy() -> DevicePolicy {
    use pc_core::device::{resolve, Device, DeviceSupport};
    resolve(Device::Cuda, DeviceSupport::WITH_CUDA)
        .expect("the explicit WITH_CUDA capability resolves cuda in any build")
}

fn junk_path(root: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
    let path = root.path().join(name);
    std::fs::write(&path, b"this is not an ONNX graph").expect("write junk bytes");
    path
}

/// GPU-3 — the anti-vacuity check that proves the CUDA policy was applied **before**
/// `commit_from_file`, not after. Junk (non-ONNX) bytes for both encoder and decoder
/// mean the session construction must fail; under a CUDA policy in a non-`cuda` build
/// the failure is the loud, run-fatal registration error from `pc_ort`'s
/// `apply_device_policy` (which runs before the model is ever committed), NOT a
/// `commit_from_file` load failure.
///
/// Gated to non-`cuda` builds for the same reason `pc-detect`'s
/// `g2_cuda_ort_boundary.rs` gates its equivalent: without `ort/cuda`, `CUDA::register`
/// is compiled to `Err(RegisterError::MissingFeature)` deterministically, independent of
/// hardware/driver/cuDNN. A real `cuda`-feature build's outcome depends on the running
/// machine, not this codebase, so no specific failure string is asserted there.
#[cfg(not(feature = "cuda"))]
#[test]
fn a_cuda_policy_fails_on_registration_before_the_model_is_committed() {
    use pc_core::StageError;

    let root = tempdir().expect("tempdir");
    let encoder = junk_path(&root, "encoder.onnx");
    let decoder = junk_path(&root, "decoder.onnx");

    let error = MangaOcrSessions::from_paths_with_policy(&encoder, &decoder, &cuda_policy())
        .expect_err("a CUDA policy must fail before the model is loaded");
    let StageError::Model(message) = error else {
        panic!("a registration failure is run-fatal `Model`; got {error:?}");
    };

    assert!(
        message.contains("execution provider"),
        "the failure must be the registration error, not a model-load error; got {message:?}"
    );
    assert!(
        !message.contains("failed to load"),
        "the policy must be applied before `commit_from_file`; got {message:?}"
    );
}

/// The control that gives the first test meaning: the same junk bytes under a CPU policy
/// must fail with the model-load message — `"failed to load ..."` — and no registration
/// text, proving the CUDA test's assertion is testing the policy boundary, not just "junk
/// bytes fail".
#[test]
fn a_cpu_policy_with_the_same_junk_bytes_fails_on_the_model_load() {
    use pc_core::StageError;

    let root = tempdir().expect("tempdir");
    let encoder = junk_path(&root, "encoder.onnx");
    let decoder = junk_path(&root, "decoder.onnx");

    let error = MangaOcrSessions::from_paths_with_policy(&encoder, &decoder, &DevicePolicy::cpu())
        .expect_err("junk bytes still fail under a cpu policy");
    let StageError::Model(message) = error else {
        panic!("a model-load failure is run-fatal `Model`; got {error:?}");
    };

    assert!(
        message.contains("failed to load"),
        "a cpu policy must fail on the model load, not registration; got {message:?}"
    );
    assert!(
        !message.contains("execution provider"),
        "a cpu policy registers nothing, so no registration text may appear; got {message:?}"
    );
}
