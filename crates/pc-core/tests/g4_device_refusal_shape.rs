//! GPU-4 — `DeviceRefusal`'s shape after the stage-scoped refusal is deleted. Every
//! device-touching stage (detector §16.47, OCR §16.48, LaMa this task) now has a real,
//! measured CUDA path, so "this build can register the device but this stage may not
//! use it" has no instance left.
//!
//! The exhaustive match is a COMPILE-time gate: re-adding a variant makes this file fail
//! to build, naming the variant — deliberately stronger than a runtime assertion.

use pc_core::device::{Device, DeviceRefusal};

#[test]
fn device_refusal_has_exactly_one_variant_after_the_stage_refusal_deletion() {
    let refusal = DeviceRefusal::NotCompiledIn {
        requested: Device::Cuda,
    };
    match refusal {
        DeviceRefusal::NotCompiledIn { requested } => assert_eq!(requested, Device::Cuda),
    }
}

/// GPU-1's frozen text, hard-coded here rather than derived from `message()` — the same
/// discipline the deleted stage test used, applied to the one message that survives.
#[test]
fn the_not_compiled_in_message_still_names_the_rebuild_flag_and_the_cpu_escape() {
    let message = DeviceRefusal::NotCompiledIn {
        requested: Device::Cuda,
    }
    .message();
    assert_eq!(
        message,
        "the cuda execution provider is not available in this build (panel-ocr was compiled without the `cuda` feature, so no CUDA execution provider is linked in). Rebuild with `--features cuda`, or set `device = \"cpu\"` under `[general]` in your profile to run on the CPU execution provider."
    );
}
