//! GPU-4 — `pc-cli`'s inpainter seam, `OnnxInpainterProvider::initialize`
//! (`crates/pc-cli/src/inpainter.rs`). Post-deletion, the seam resolves the policy once
//! and threads it (mirroring `crate::detector`), so a `cuda`-capable build reaches model
//! resolution instead of a stage refusal: the LaMa `ensure_stage_supports` gate is gone,
//! and `resolve` is the only device gate left.
//!
//! `build_provider` has no injectable `DeviceSupport` — the provider reads
//! `DeviceSupport::compiled()` internally — so this test uses the same runtime-branch
//! technique the sibling `g3_ocr_seam_cuda_path.rs` uses: check `compiled()` at test
//! time and assert the outcome that build is expected to produce. Never a skip, so this
//! can't pass vacuously in either tier.
#![cfg(feature = "onnx")]

use pc_cli::inpainter::build_provider;
use pc_core::device::{Device, DeviceRefusal, DeviceSupport};
use pc_core::StageError;

#[test]
fn a_cuda_request_reaches_model_resolution_in_a_cuda_build_and_carries_pc_cores_refusal_otherwise()
{
    let cache = tempfile::tempdir().expect("tempdir");
    let provider = build_provider(true, None, cache.path(), Device::Cuda)
        .expect("`inpainting_enabled = true` yields a provider");

    let error = provider
        .inpainter()
        .err()
        .expect("an empty cache root has no LaMa artifact, so this fails either way");
    let StageError::Model(message) = error else {
        panic!("every inpainter provisioning failure is run-fatal `Model` (§16.38 item 9)");
    };

    if DeviceSupport::compiled().supports(Device::Cuda) {
        assert!(
            message.contains("the LaMa inpainting model is missing at"),
            "a cuda build must get PAST the device gate to model resolution; got {message:?}"
        );
        assert!(
            !message.contains("no CUDA path is ratified"),
            "GPU-4 deleted the stage refusal; got {message:?}"
        );
    } else {
        assert_eq!(
            message,
            DeviceRefusal::NotCompiledIn {
                requested: Device::Cuda
            }
            .message(),
            "the seam must carry pc_core's refusal verbatim, not grow its own wording"
        );
    }
}

#[test]
fn a_cpu_request_always_reaches_model_resolution_in_every_build() {
    let cache = tempfile::tempdir().expect("tempdir");
    let provider = build_provider(true, None, cache.path(), Device::Cpu)
        .expect("`inpainting_enabled = true` yields a provider");

    let error = provider
        .inpainter()
        .err()
        .expect("no artifact in the cache");
    let StageError::Model(message) = error else {
        panic!("run-fatal Model");
    };
    assert!(
        message.contains("the LaMa inpainting model is missing at"),
        "the CPU path must reach model resolution; got {message:?}"
    );
    assert!(
        !message.contains("execution provider is not available"),
        "the CPU device resolves in every build; got {message:?}"
    );
}
