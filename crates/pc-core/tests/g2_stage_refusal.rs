//! GPU-2 (§16.47 item 4) — `ensure_stage_supports` and its refusal messages, tested at
//! the `pc-core` level with no feature and no model needed at all: `resolve(Device::Cuda,
//! DeviceSupport::WITH_CUDA)` reaches the "compiled and resolvable" branch explicitly,
//! same technique `device_policy.rs` already uses for its frozen tests.
//!
//! **Why this file exists (independent review, GPU-2 G2-A/B, 2026-08-11).** Before this
//! file, every check of the stage-refusal text derived its expected value by calling
//! `DeviceRefusal::…::message()` and comparing the result to itself — the message could
//! have swapped its per-stage citation (the exact defect the ratification review caught
//! and fixed) and every existing test would still pass, since nothing hard-coded the
//! ratified text. The tests below assert against **literal strings**, not the function
//! under test.
//!
//! **GPU-3 (2026-08-11)**: the OCR stage variant is deleted — OCR now has a real,
//! ratified CUDA path, so it no longer has this kind of stage-scoped refusal at all.
//! That deletion also removed the Ocr arm of `Stage::as_str()`'s single-source wire
//! spelling; the only remaining stage is Inpaint, and the one live literal below tracks
//! the corrected Inpaint message (which now names the detector **and OCR** as what the
//! measurement covered and what runs on CUDA when inpainting is disabled).
//!
//! Also closes a second gap the same review found: `ensure_stage_supports` itself had no
//! direct test anywhere — deleting its call from a seam only failed the (optional)
//! `cuda`-feature tier, never the standing four-command bar. These tests need no
//! feature, so they run in the default tier, on every future `cargo test --workspace`.

use pc_core::device::{resolve, Device, DeviceRefusal, DeviceSupport, Stage};

/// A Cpu request always has a ratified path, in every build, for every stage — this is
/// what `ensure_stage_supports` must never refuse.
#[test]
fn a_cpu_request_never_refuses_any_stage() {
    let policy = resolve(Device::Cpu, DeviceSupport::WITH_CUDA).expect("cpu always resolves");
    assert!(pc_core::device::ensure_stage_supports(&policy, Stage::Inpaint).is_ok());
}

/// A resolvable Cuda request refuses the inpaint stage with its own message.
#[test]
fn a_resolvable_cuda_request_refuses_the_inpaint_stage() {
    let policy = resolve(Device::Cuda, DeviceSupport::WITH_CUDA)
        .expect("the explicit WITH_CUDA capability resolves cuda");

    let inpaint_error = pc_core::device::ensure_stage_supports(&policy, Stage::Inpaint)
        .expect_err("inpaint has no ratified cuda path");
    assert_eq!(
        inpaint_error,
        DeviceRefusal::NoRatifiedStagePath {
            requested: Device::Cuda,
            stage: Stage::Inpaint,
        }
    );
}

/// The LaMa refusal's literal text — hard-coded, not derived. Falsifier: swapping the
/// citation (e.g. back to OCR's former §16.36 item 3, or the stale §16.47 item 1), changing
/// the enabled-flag name/section, or reverting the detector-and-OCR measurement/flag-advice
/// wording the corrected message now carries.
///
/// **GPU-3 G3-E (2026-08-11)**: the measurement clause's citation is §16.48 item 2, not
/// §16.47 item 1 — item 1 never said anything about OCR being measured; §16.47 item 4 is
/// marked SUPERSEDED IN PART on this exact clause.
#[test]
fn the_inpaint_refusal_message_is_frozen_to_its_own_citation_and_flag() {
    let message = DeviceRefusal::NoRatifiedStagePath {
        requested: Device::Cuda,
        stage: Stage::Inpaint,
    }
    .message();
    assert_eq!(
        message,
        "the cuda execution provider is compiled into this build, but no CUDA path is ratified for the inpaint stage (§16.38 item 15(d) / §16.48 item 2: CUDA has now been measured for the text detector and OCR). Set `inpainting_enabled = false` under `[inpainter]` to run the detector and OCR on CUDA, or set device = \"cpu\" to run every stage on the CPU execution provider."
    );
}

/// Neither refusal message may be `NotCompiledIn`'s text — the ratification's whole
/// reason for adding a second variant was that reusing `NotCompiledIn` in a `cuda` build
/// tells the user to rebuild with a flag they already used.
#[test]
fn a_stage_refusal_never_reuses_not_compiled_ins_message() {
    let not_compiled_in = DeviceRefusal::NotCompiledIn {
        requested: Device::Cuda,
    }
    .message();
    let stage_message = DeviceRefusal::NoRatifiedStagePath {
        requested: Device::Cuda,
        stage: Stage::Inpaint,
    }
    .message();
    assert_ne!(stage_message, not_compiled_in);
}
