//! §16.46 item 8 — the one place a fixture- or report-producing path gets its
//! `TextDetectorConfig`, and the derivation of the provenance fields that describe it.
//!
//! **Why this module exists rather than a literal at each call site.** `pc_detect::run`
//! branches on `mask_refine_mode` for both the mask written to `raw_mask_dest` and the
//! operand the coverage filter scores (`crates/pc-detect/src/lib.rs:112-137`). Three paths
//! here fed it `TextDetectorConfig::default()`, so a change to the shipped default would
//! silently re-point a committed fixture and two committed reports. Measured 2026-08-10:
//! flipping the defaults alone turned `xtask/tests/mask_sweep_cli.rs`'s replay ladder and
//! `crates/pc-detect/tests/d7_run.rs`'s committed-fixture equality red.
//!
//! Everything here is pure and free of `xtask`'s other modules, so
//! `xtask/tests/recording_mode_pin.rs` can include it by path — the shape
//! `g1_d_resolved_policy.rs` already uses for `src/device.rs`.

use pc_config::{MaskRefineMode, TextDetectorConfig};
use serde_json::Value;
use std::collections::BTreeMap;

/// The mask-refine mode every committed artifact in this repository was produced under.
///
/// **A literal, deliberately not `TextDetectorConfig::default().mask_refine_mode`.** It
/// records a fact about artifacts already on disk, which no later change to the shipped
/// default may move. §16.46 item 8's ruling: *"Pin the three xtask sites to
/// `MaskRefineMode::Simple`."* Changing this constant is a re-record, not an edit.
pub const RECORDING_MASK_REFINE_MODE: MaskRefineMode = MaskRefineMode::Simple;

/// The detector config every recording and report path uses.
///
/// Exactly one field is pinned; every other field still tracks the shipped config, so an
/// unrelated default change (thread counts, model path) still reaches recordings rather than
/// being frozen by accident alongside the mode.
pub fn recording_detector_config() -> TextDetectorConfig {
    TextDetectorConfig {
        mask_refine_mode: RECORDING_MASK_REFINE_MODE,
        ..TextDetectorConfig::default()
    }
}

/// How a mode is named in generated human-readable reports, e.g. `MaskRefineMode::Simple`.
///
/// Report generators render this from the config they actually ran, never from a literal
/// (§16.46 item 8(a)'s graft: *"report generators rendering the mode from the config they
/// actually ran, not a hard-coded literal"*). A hard-coded literal is a prose claim that
/// nothing can falsify, which is how `calibrate.rs` came to describe every future run as
/// `Simple` regardless of what it did.
pub fn mode_display(mode: MaskRefineMode) -> &'static str {
    match mode {
        MaskRefineMode::Simple => "MaskRefineMode::Simple",
        MaskRefineMode::Annotation => "MaskRefineMode::Annotation",
    }
}

/// The serde wire spelling of a mode — the same text a profile has to write to reproduce a
/// recording. Taken from serde rather than hand-typed, so the recorded value cannot drift
/// from the accepted TOML text.
pub fn mode_wire(mode: MaskRefineMode) -> Value {
    serde_json::to_value(mode).expect("MaskRefineMode serialises to a string")
}

/// `detector.ours.profile_non_default`, derived by diffing the config a recording actually
/// ran under against the shipped defaults.
///
/// This replaces `record.rs`'s hard-coded `"profile_non_default": {}`.
/// `crates/pc-testkit/src/provenance.rs` documents the field as *"built by diffing against
/// the profile defaults"*, so a literal `{}` was a claim that the recording used a fully
/// default profile — false the moment the recorder pins a mode the shipped default does not
/// carry, and unfalsifiable in the meantime.
///
/// Keys are dotted profile paths (`text_detector.<field>`), so a reader can put the value
/// back into a TOML profile without translating.
pub fn profile_non_default(config: &TextDetectorConfig) -> BTreeMap<String, Value> {
    let shipped = TextDetectorConfig::default();
    let mut out = BTreeMap::new();
    if config.model_path != shipped.model_path {
        out.insert(
            "text_detector.model_path".into(),
            Value::String(config.model_path.clone()),
        );
    }
    if config.concurrent_models != shipped.concurrent_models {
        out.insert(
            "text_detector.concurrent_models".into(),
            Value::from(config.concurrent_models),
        );
    }
    if config.intra_threads != shipped.intra_threads {
        out.insert(
            "text_detector.intra_threads".into(),
            Value::from(config.intra_threads),
        );
    }
    if config.inter_threads != shipped.inter_threads {
        out.insert(
            "text_detector.inter_threads".into(),
            Value::from(config.inter_threads),
        );
    }
    if config.mask_refine_mode != shipped.mask_refine_mode {
        out.insert(
            "text_detector.mask_refine_mode".into(),
            mode_wire(config.mask_refine_mode),
        );
    }
    out
}
