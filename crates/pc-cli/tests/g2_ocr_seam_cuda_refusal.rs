//! GPU-2 (§16.47 item 4/5) — `pc-cli`'s OCR seam, `build_factory_for_device`
//! (`crates/pc-cli/src/ocr.rs`). Untested until now (independent review, GPU-2 G2-A/B,
//! 2026-08-11): unlike `pc-ocr`'s/`pc-inpaint`'s own seams, `build_factory_for_device`
//! has no injectable `DeviceSupport` — it reads `DeviceSupport::compiled()` internally —
//! so this test uses the same runtime-branch technique `l5_session.rs` and the sibling
//! `g2_stage_refusal.rs` files already use: check `compiled()` at test time and assert
//! the outcome that build is expected to produce. Never a skip, so this can't pass
//! vacuously in either tier.
//!
//! What this closes: a `Device::Cuda` request must be refused (either `NotCompiledIn` or
//! `NoRatifiedStagePath`, per which build this is) **before** `resolve_managed_model` is
//! ever reached — a CUDA user with no cached OCR model must see the ratified device
//! refusal, not a confusing model-download failure.
#![cfg(feature = "onnx")]

use pc_core::device::{Device, DeviceRefusal, DeviceSupport};
use pc_core::StageError;

#[test]
fn a_cuda_request_refuses_before_ever_touching_model_resolution() {
    // A cache root with no models directory at all: if `build_factory_for_device` ever
    // reached `resolve_managed_model`, it would have to touch this path or the network —
    // neither of which this test permits (no tempdir writes expected, no network stub).
    let cache_root = tempfile::tempdir().expect("tempdir");

    let error = pc_cli::ocr::build_factory_for_device(cache_root.path(), Device::Cuda)
        .err()
        .expect("a cuda request must be refused, one way or another, in every build");
    let StageError::Model(message) = error else {
        panic!("a device refusal is run-fatal Model, not any other StageError variant");
    };

    if DeviceSupport::compiled().supports(Device::Cuda) {
        // This build can register CUDA — the OCR-specific stage refusal must fire.
        let expected = DeviceRefusal::NoRatifiedStagePath {
            requested: Device::Cuda,
            stage: pc_core::device::Stage::Ocr,
        }
        .message();
        assert_eq!(message, expected);
    } else {
        // This build cannot register CUDA at all — GPU-1's refusal fires first.
        let expected = DeviceRefusal::NotCompiledIn {
            requested: Device::Cuda,
        }
        .message();
        assert_eq!(message, expected);
    }
}
