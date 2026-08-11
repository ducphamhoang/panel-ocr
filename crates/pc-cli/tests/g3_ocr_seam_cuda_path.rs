//! GPU-3 — `pc-cli`'s OCR seam, `build_factory_for_device`
//! (`crates/pc-cli/src/ocr.rs`). Post-deletion, a `cuda`-capable build reaches model
//! resolution instead of a stage refusal: the OCR `ensure_stage_supports` gate is gone,
//! so a CUDA request on a build that can register CUDA passes straight to
//! `resolve_managed_model` and fails there (in this test's empty-cache-root setup, on a
//! missing cached model).
//!
//! `build_factory_for_device` has no injectable `DeviceSupport` — it reads
//! `DeviceSupport::compiled()` internally — so this test uses the same runtime-branch
//! technique the sibling `g2_stage_refusal.rs` files use: check `compiled()` at test
//! time and assert the outcome that build is expected to produce. Never a skip, so this
//! can't pass vacuously in either tier.
#![cfg(feature = "onnx")]

use pc_core::device::{Device, DeviceRefusal, DeviceSupport};
use pc_core::StageError;

#[test]
fn a_cuda_request_reaches_model_resolution_in_a_cuda_build_and_is_refused_otherwise() {
    let cache_root = tempfile::tempdir().expect("tempdir");

    let error = pc_cli::ocr::build_factory_for_device(cache_root.path(), Device::Cuda)
        .err()
        .expect("an empty cache root has no cached OCR model, so this fails either way");
    let StageError::Model(message) = error else {
        panic!("both outcomes are run-fatal Model");
    };

    if DeviceSupport::compiled().supports(Device::Cuda) {
        // `resolve_managed_model` renders a missing managed model as
        // "managed model is missing at `<path>`; run `panel-ocr models download`"
        // (crates/pc-cli/src/ocr.rs) — so the message names the model resolution
        // rather than any device refusal.
        assert!(
            message.contains("managed model is missing"),
            "must reach model resolution; got {message:?}"
        );
        assert!(
            !message.contains("no CUDA path is ratified"),
            "the deleted stage refusal must not appear; got {message:?}"
        );
    } else {
        assert_eq!(
            message,
            DeviceRefusal::NotCompiledIn {
                requested: Device::Cuda
            }
            .message()
        );
    }
}
