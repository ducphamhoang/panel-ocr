//! GPU-3 — the OCR seam, `MangaOcrSessions::from_paths_for_device`
//! (`crates/pc-ocr/src/onnx.rs`). Post-deletion, a CUDA-capable build reaches
//! `ensure_model_file`'s pre-flight instead of a stage refusal: the OCR
//! `ensure_stage_supports` gate is gone, so `resolve` is the only device gate left. In a
//! build that cannot register CUDA the GPU-1 `NotCompiledIn` refusal (from `resolve`,
//! which runs first) is the correct outcome. Both arms assert, exactly like
//! `pc_core::device`'s own `device_support_cuda_feature` test: neither is a skip, so
//! this test can never pass vacuously in either tier.
#![cfg(feature = "onnx")]

use pc_core::device::{Device, DeviceRefusal, DeviceSupport};
use pc_core::StageError;
use pc_ocr::onnx::MangaOcrSessions;
use tempfile::tempdir;

#[test]
fn a_cuda_request_reaches_the_model_preflight_instead_of_a_stage_refusal() {
    let root = tempdir().expect("tempdir");
    let missing = root.path().join("missing-encoder.onnx");

    let error = MangaOcrSessions::from_paths_for_device(&missing, &missing, Device::Cuda)
        .expect_err("a missing model file must still fail, in either build");
    let StageError::Model(message) = &error else {
        panic!("run-fatal Model; got {error:?}");
    };

    if DeviceSupport::compiled().supports(Device::Cuda) {
        assert_eq!(
            message,
            &format!(
                "model path is missing or is not a regular file: {}",
                missing.display()
            ),
            "a cuda build must pass straight to the model pre-flight now"
        );
        assert!(!message.contains("no CUDA path is ratified"));
    } else {
        assert_eq!(
            message,
            &DeviceRefusal::NotCompiledIn {
                requested: Device::Cuda
            }
            .message()
        );
    }
}
