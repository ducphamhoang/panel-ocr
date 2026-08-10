//! GPU-2 (§16.47 item 4) — the OCR stage has a refusal path only: `Stage::Ocr` has no
//! ratified CUDA path. In a build that CAN register CUDA the seam refuses with
//! `NoRatifiedStagePath`; in a build that cannot, the GPU-1 `NotCompiledIn` refusal
//! (from `resolve`, which runs first) is the correct outcome. Both arms assert, exactly
//! like `pc_core::device`'s own `device_support_cuda_feature` test: neither is a skip,
//! so this test can never pass vacuously in either tier.
#![cfg(feature = "onnx")]

use pc_core::device::{Device, DeviceRefusal, DeviceSupport, Stage};
use pc_core::StageError;
use pc_ocr::onnx::MangaOcrSessions;
use tempfile::tempdir;

/// §16.47 item 4: OCR has a refusal path only — a build that CAN register CUDA must
/// refuse at the stage seam with `NoRatifiedStagePath`, whose message does not tell the
/// user to rebuild with a flag they already used (which reusing the GPU-1
/// `NotCompiledIn` message would do).
#[test]
fn ocr_refuses_cuda_even_in_a_build_that_can_register_it() {
    let root = tempdir().expect("tempdir");
    let missing = root.path().join("missing-encoder.onnx");

    let error = MangaOcrSessions::from_paths_for_device(&missing, &missing, Device::Cuda)
        .expect_err("OCR must refuse CUDA");
    let StageError::Model(message) = &error else {
        panic!("a device refusal is run-fatal `Model`; got {error:?}");
    };

    match pc_core::device::resolve(Device::Cuda, DeviceSupport::compiled()) {
        // A build that CAN register CUDA: the seam's second gate fires, naming the stage
        // — and the message must not be the GPU-1 one, which would tell the user to
        // rebuild with a flag they already used.
        Ok(_) => {
            let expected = DeviceRefusal::NoRatifiedStagePath {
                requested: Device::Cuda,
                stage: Stage::Ocr,
            };
            assert_eq!(
                message,
                &expected.message(),
                "a CUDA-resolvable build must refuse OCR with the stage-scoped refusal"
            );
            assert_ne!(
                message,
                &DeviceRefusal::NotCompiledIn {
                    requested: Device::Cuda
                }
                .message(),
                "reusing the GPU-1 NotCompiledIn message would tell the user to rebuild \
                 with a flag they already used"
            );
        }
        // A build that cannot register CUDA: `resolve` refuses first with the frozen
        // GPU-1 message, and the seam carries it verbatim.
        Err(refusal) => {
            assert_eq!(message, &refusal.message(), "wrong refusal message");
        }
    }
}
