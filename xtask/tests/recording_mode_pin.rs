//! §16.46 items 7 and 8 (Fable tie-break on fixture scope, 2026-08-10) — the two tripwires
//! that keep a shipped-default change from silently re-pointing a committed artifact.
//! **FROZEN once written.**
//!
//! Why these exist, in one measurement. `pc_detect::run` branches on `mask_refine_mode` for
//! both the mask it writes to `raw_mask_dest` and the operand the coverage filter scores
//! (`crates/pc-detect/src/lib.rs:112-137`). Three `xtask` paths used to read
//! `TextDetectorConfig::default()` and feed it straight into that function. Flipping the two
//! shipped defaults on a throwaway worktree at `40e2b5a` took `cargo test --workspace` from
//! 1418 passed / 0 failed to 1409 passed / 9 failed, and two of the nine were behavioural:
//! `xtask/tests/mask_sweep_cli.rs:76` reported a changed ladder and
//! `crates/pc-detect/tests/d7_run.rs:628` lost equality with the committed `#raw.json`.
//!
//! `xtask` is binary-only, so the pure seam is included by path — the same shape
//! `g1_d_resolved_policy.rs` uses for `src/device.rs`, and for the same reason: it keeps the
//! seam testable without exporting maintainer internals.

#[path = "../src/recording_config.rs"]
mod recording_config;

use pc_config::{MaskRefineMode, TextDetectorConfig};
use recording_config::{
    mode_display, profile_non_default, recording_detector_config, RECORDING_MASK_REFINE_MODE,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The mode that is NOT the shipped default, whatever the shipped default currently is.
///
/// Every test below that needs "a non-default value" derives it this way rather than naming
/// one. §16.46 item 1(a) moves that default, so a test naming `Annotation` or `Simple`
/// directly would silently stop testing the non-default case the day it moved — which is the
/// whole failure this file exists to catch, reproduced inside the file meant to catch it.
fn the_other_mode() -> MaskRefineMode {
    match TextDetectorConfig::default().mask_refine_mode {
        MaskRefineMode::Simple => MaskRefineMode::Annotation,
        MaskRefineMode::Annotation => MaskRefineMode::Simple,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tripwire 1 — the pin itself.
// ─────────────────────────────────────────────────────────────────────────────

/// §16.46 item 8: every path that produces a COMMITTED artifact pins its mask-refine mode
/// explicitly instead of inheriting `TextDetectorConfig::default()`.
///
/// `Simple` is asserted as a **literal value**: it is the mode the committed artifacts were
/// produced under, and this line must not follow a future default flip.
///
/// Turns red if the recording mode is changed without re-recording.
#[test]
fn the_recording_detector_config_pins_simple_as_a_literal() {
    assert_eq!(RECORDING_MASK_REFINE_MODE, MaskRefineMode::Simple);
    assert_eq!(
        recording_detector_config().mask_refine_mode,
        MaskRefineMode::Simple
    );
}

/// §16.46 item 8(e) **as amended by the Orchestrator ruling of 2026-08-10**: the second half
/// of the first tripwire — that the pin is provably independent of the shipped default — is a
/// **source-level** check rather than a runtime comparison.
///
/// **Why it is not the obvious `assert_ne!(pin, default)`.** That comparison is exactly what
/// item 8(e) originally specified, and it cannot be satisfied: item 15 requires D1 to be
/// green both before and after the flip, and before the flip the pin and the default are the
/// same value, so the comparison fails precisely when item 15 forbids failure. Deferring it
/// to D2 would have left the pin's independence ungated for a whole task, and writing it
/// anyway would have left D1 red. The ruling substitutes the technique and keeps the intent:
/// what matters is that the constant is a literal variant, not that it currently differs from
/// something. The census tripwire below already reads source structure for the same reason.
///
/// **What it iterates**: the one line of `xtask/src/recording_config.rs` that defines the
/// constant, located by name rather than by line number, with the count of such lines
/// asserted so a second definition cannot hide behind the first.
///
/// Turns red if the constant is ever redefined as
/// `TextDetectorConfig::default().mask_refine_mode` or any other expression that reads the
/// shipped default — which is the one change that would silently make every "pinned" call
/// site inherit again.
#[test]
fn the_recording_mode_constant_is_a_literal_and_never_reads_the_shipped_default() {
    let source = std::fs::read_to_string(
        pc_testkit::paths::workspace_root().join("xtask/src/recording_config.rs"),
    )
    .expect("the pin module is readable");

    let definitions: Vec<&str> = source
        .lines()
        .filter(|line| line.contains("pub const RECORDING_MASK_REFINE_MODE"))
        .collect();
    assert_eq!(
        definitions.len(),
        1,
        "expected exactly one definition of the pin; found {definitions:#?}"
    );

    let definition = definitions[0];
    assert!(
        definition.contains("= MaskRefineMode::Simple;"),
        "the pin must be a literal variant, not an expression: {definition}"
    );
    assert!(
        !definition.contains("default()"),
        "the pin must not read the shipped default -- that is the whole point of pinning it: \
         {definition}"
    );
}

/// The pin moves **exactly one field**. Everything else must keep tracking the shipped
/// config, so an unrelated default change — thread counts, model path — still reaches
/// recordings instead of being frozen by accident alongside the mode.
///
/// Turns red if the pin is widened into a whole hand-built config.
#[test]
fn the_recording_pin_moves_the_mask_mode_and_nothing_else() {
    let pinned = recording_detector_config();
    let shipped = TextDetectorConfig::default();

    assert_eq!(pinned.model_path, shipped.model_path);
    assert_eq!(pinned.concurrent_models, shipped.concurrent_models);
    assert_eq!(pinned.intra_threads, shipped.intra_threads);
    assert_eq!(pinned.inter_threads, shipped.inter_threads);

    // Stated as a whole-struct equality so a field added to `TextDetectorConfig` later is
    // covered without anyone remembering to add a line above: the pinned config must be the
    // shipped one with one field replaced, and nothing else.
    assert_eq!(
        pinned,
        TextDetectorConfig {
            mask_refine_mode: RECORDING_MASK_REFINE_MODE,
            ..TextDetectorConfig::default()
        }
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tripwire 2 — the census. Cookbook rule 13: the pin protects only the call sites
// that USE it, so this enumerates the other population.
// ─────────────────────────────────────────────────────────────────────────────

/// Every remaining literal `TextDetectorConfig::default()` under `xtask/src/`, each declared
/// with the reason it does not produce a committed artifact.
///
/// Asserted as a **map of `(file, count)`**, not as a total: a total is satisfied by a swap
/// (rule 13, "cardinality is not identity"), and a diff over the map can name which file
/// gained an occurrence and which lost one.
///
/// `record.rs` is deliberately **absent**, and its absence is the point of the pin: the
/// detector fixture recorder must reach its config through `recording_detector_config()`.
const DECLARED_BARE_DEFAULTS: &[(&str, usize, &str)] = &[
    (
        "calibrate.rs",
        1,
        "the ONNX session constructor (`OnnxDetector::from_path_with_config`) reads the \
         thread and path fields and never reads `mask_refine_mode`; the `DetectInput` in the \
         same file IS pinned",
    ),
    (
        "mask_sweep.rs",
        1,
        "same session-constructor site as calibrate.rs; the `DetectInput` in the same file \
         IS pinned",
    ),
    (
        "mode_bench.rs",
        3,
        "per-cell configs set `mask_refine_mode` explicitly from the cell (§16.43 item \
         2(a)); the bare defaults are the session constructor, the `..default()` base those \
         per-cell configs are built from, and one test comparison",
    ),
    (
        "recording_config.rs",
        2,
        "the pin itself: `..TextDetectorConfig::default()` keeps every other field tracking \
         the shipped config (which `the_recording_pin_moves_the_mask_mode_and_nothing_else` \
         requires), and `profile_non_default` needs the shipped config as the thing it diffs \
         against",
    ),
];

/// Occurrences of `TextDetectorConfig::default()` on `line`, **excluding line comments**.
///
/// A first draft of the census counted raw substrings and reported `recording_config.rs: 4`
/// against a real call count of 2 — the difference being two doc-comment mentions. Counting
/// prose as call sites makes the declared numbers meaningless and turns the gate red on any
/// comment edit, so the rule is: an occurrence at or after a `//` on the same line is prose.
///
/// **The accept-set of that rule, stated rather than hoped for** (cookbook rule 14c): it
/// misses an occurrence inside a `/* … */` block comment, because it has no block-comment
/// state. That gap is not left as a hope —
/// `the_census_comment_rule_has_no_block_comment_blind_spot` asserts `xtask/src` contains no
/// block comments at all, which makes the gap empty by measurement instead of by assumption.
fn call_occurrences(line: &str) -> usize {
    const NEEDLE: &str = "TextDetectorConfig::default()";
    let code = match line.find("//") {
        Some(at) => &line[..at],
        None => line,
    };
    code.matches(NEEDLE).count()
}

fn xtask_src_rust_files() -> Vec<PathBuf> {
    fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("xtask/src is readable") {
            let path = entry.expect("readable dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    let root = pc_testkit::paths::workspace_root().join("xtask/src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    files.sort();
    files
}

/// §16.46 item 8(e), cookbook rule 13.
///
/// **What it iterates, stated because that is the question rule 13 says finds this class:**
/// every `*.rs` file under `xtask/src/`, discovered by walking the directory — not a list
/// written here, and not the module declarations in `main.rs`, either of which would leave a
/// new file unguarded, and absence is exactly what this is trying to detect.
///
/// Turns red when a bare `TextDetectorConfig::default()` appears in a new `xtask/src` file,
/// or when an existing site's count changes in either direction — which is how
/// `mask_sweep.rs`'s `DetectInput` came to silently track the shipped default in the first
/// place.
#[test]
fn every_bare_default_detector_config_under_xtask_src_is_declared() {
    let files = xtask_src_rust_files();
    assert!(
        files.len() >= 8,
        "the walker found only {} files under xtask/src; a broken walk would report every \
         declared site as missing and read like a real finding",
        files.len()
    );

    let mut observed: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("xtask source is readable");
        let count: usize = text.lines().map(call_occurrences).sum();
        if count > 0 {
            let name = path
                .strip_prefix(pc_testkit::paths::workspace_root().join("xtask/src"))
                .expect("walked from that root")
                .to_string_lossy()
                .replace('\\', "/");
            observed.insert(name, count);
        }
    }

    let declared: BTreeMap<String, usize> = DECLARED_BARE_DEFAULTS
        .iter()
        .map(|(file, count, _)| ((*file).to_owned(), *count))
        .collect();

    let mut unexpected = Vec::new();
    let mut missing = Vec::new();
    for (file, count) in &observed {
        match declared.get(file) {
            Some(expected) if expected == count => {}
            Some(expected) => {
                unexpected.push(format!("{file}: found {count}, declared {expected}"))
            }
            None => unexpected.push(format!(
                "{file}: found {count}, NOT DECLARED — pin it through \
                 `recording_config::recording_detector_config()` if it produces a committed \
                 artifact, or add a row to DECLARED_BARE_DEFAULTS saying why it does not"
            )),
        }
    }
    for (file, expected) in &declared {
        if !observed.contains_key(file) {
            missing.push(format!(
                "{file}: declared {expected}, found none — delete the row if the site is gone"
            ));
        }
    }

    assert!(
        unexpected.is_empty() && missing.is_empty(),
        "the xtask bare-default census changed.\n  unexpected: {unexpected:#?}\n  missing: {missing:#?}"
    );
}

/// The census's comment rule is line-based and therefore blind inside `/* … */`. Rather than
/// leave that as a disclosed hope, this closes it by measurement: `xtask/src` contains no
/// block comments, so there is nowhere for a hidden occurrence to sit.
///
/// Turns red the moment someone introduces a block comment into `xtask/src` — at which point
/// the census's accept-set is no longer known, and either the comment or the rule has to
/// change. That is the intended outcome, not a false alarm.
/// **One named exemption, pinned by its exact content rather than by a category.** This is
/// the only `/*` under `xtask/src`, and it is not a comment: it is a shell-style glob inside
/// the `mode-bench` report template's string literal. A category exemption ("ignore string
/// literals") would need a lexer and would be the bypass wearing different clothes
/// (cookbook rule 13's corollary); pinning the line means that if it ever changes, the gate
/// re-fires and a human re-checks instead of the exemption silently widening.
const BLOCK_COMMENT_EXEMPTIONS: &[&str] =
    &["- The reference is the vendored `demo_bubbles/*_clean.png`. Its producing PanelCleaner"];

#[test]
fn the_census_comment_rule_has_no_block_comment_blind_spot() {
    let mut offenders = Vec::new();
    let mut exemptions_seen = 0usize;
    for path in xtask_src_rust_files() {
        let text = std::fs::read_to_string(&path).expect("xtask source is readable");
        for (index, line) in text.lines().enumerate() {
            if !line.contains("/*") {
                continue;
            }
            if BLOCK_COMMENT_EXEMPTIONS
                .iter()
                .any(|exempt| line.trim_end_matches(['\\', 'n']).contains(exempt))
            {
                exemptions_seen += 1;
                continue;
            }
            offenders.push(format!(
                "{}:{}: {}",
                path.file_name().unwrap_or_default().to_string_lossy(),
                index + 1,
                line.trim()
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "`call_occurrences` strips `//` line comments only, so a block comment would give it \
         an accept-set nobody has enumerated. Either remove the block comment or re-derive \
         the census rule: {offenders:#?}"
    );
    // The exemption must not pass by merely looking exemption-shaped: it has to be there.
    // If the pinned line is edited away, this fails and the exemption gets re-justified
    // rather than quietly protecting nothing.
    assert_eq!(
        exemptions_seen,
        BLOCK_COMMENT_EXEMPTIONS.len(),
        "every pinned exemption must still match exactly one line"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The provenance derivation. §16.46 item 8(d).
// ─────────────────────────────────────────────────────────────────────────────

/// `profile_non_default` is what replaces `record.rs`'s hard-coded `"profile_non_default": {}`.
/// `crates/pc-testkit/src/provenance.rs` documents that field as *"built by diffing against
/// the profile defaults"*, and a literal `{}` is a claim that the recording ran under a fully
/// default profile — false the moment the recorder pins a mode the shipped default does not
/// carry.
///
/// **Every expectation here is default-value-agnostic on purpose.** §16.46 item 1(a) moves
/// the shipped default, so a row naming `Simple` or `Annotation` outright would assert
/// something different before and after that flip. The non-default case is derived through
/// `the_other_mode()` instead.
///
/// Turns red if the diff stops reporting a changed field, starts reporting an unchanged one,
/// or renames a key.
#[test]
fn profile_non_default_reports_exactly_the_fields_that_differ_from_the_shipped_default() {
    let shipped = TextDetectorConfig::default();

    assert_eq!(
        profile_non_default(&shipped),
        BTreeMap::new(),
        "a recording under the shipped config declares no overrides"
    );

    let mut threaded = shipped.clone();
    threaded.intra_threads = 4;
    assert_eq!(
        profile_non_default(&threaded),
        BTreeMap::from([(
            "text_detector.intra_threads".to_owned(),
            serde_json::json!(4)
        )])
    );

    let mut moded = shipped.clone();
    moded.mask_refine_mode = the_other_mode();
    let reported = profile_non_default(&moded);
    assert_eq!(
        reported.keys().collect::<Vec<_>>(),
        vec!["text_detector.mask_refine_mode"],
        "exactly one key, and it names the field that moved"
    );
    assert_eq!(
        reported["text_detector.mask_refine_mode"],
        serde_json::to_value(the_other_mode()).expect("MaskRefineMode serialises"),
        "the value is the serde wire spelling, taken from serde rather than hand-typed"
    );

    let mut both = shipped.clone();
    both.mask_refine_mode = the_other_mode();
    both.model_path = "/tmp/model.onnx".to_owned();
    assert_eq!(
        profile_non_default(&both).keys().collect::<Vec<_>>(),
        vec!["text_detector.mask_refine_mode", "text_detector.model_path"],
        "two changed fields produce two keys; one of them alone would pass a `contains` check"
    );
}

/// The wire spelling the `ours` block records is serde's, not a hand-typed string, so the
/// recorded value cannot drift from the value a profile would have to write to reproduce the
/// recording.
///
/// Turns red if `MaskRefineMode`'s `#[serde(rename_all = "snake_case")]` is removed or a
/// variant is renamed — either of which would change the accepted TOML text too.
#[test]
fn the_recorded_mode_spelling_is_the_same_text_a_profile_would_write() {
    assert_eq!(
        serde_json::to_value(MaskRefineMode::Simple).unwrap(),
        serde_json::json!("simple")
    );
    assert_eq!(
        serde_json::to_value(MaskRefineMode::Annotation).unwrap(),
        serde_json::json!("annotation")
    );
}

/// §16.46 item 8(a): the field the generator adds must survive the typed round-trip, and the
/// pre-field shape must still parse.
///
/// **What this covers and what it does not, stated because the gap matters.** `OursPins` is
/// the READER half, and it is gated here. The WRITER half — the `serde_json::json!` block in
/// `xtask/src/record.rs` — lives inside a `#[cfg(feature = "onnx")]` function and is not
/// reachable from this tier, so nothing here proves the recorder emits the key. That is a
/// disclosed gap, not a covered one; it closes at the next real re-record.
///
/// Turns red if the field stops being optional (which would break the committed file's two
/// frozen readers), or if it stops round-tripping.
#[test]
fn the_ours_pins_schema_accepts_the_mode_field_and_still_parses_a_document_without_it() {
    let with_mode = serde_json::json!({
        "backend": "ort",
        "decoded_rgb_digest": "a".repeat(64),
        "decoded_from": "input_page",
        "execution_provider": "cpu",
        "intra_threads": 0,
        "inter_threads": 0,
        "pad_value": 0,
        "panel_ocr_commit": "b".repeat(40),
        "mask_refine_mode": "simple",
    });
    let parsed: pc_testkit::provenance::OursPins =
        serde_json::from_value(with_mode.clone()).expect("the new field is accepted");
    assert_eq!(parsed.mask_refine_mode.as_deref(), Some("simple"));
    assert_eq!(
        serde_json::to_value(&parsed).expect("re-serialises"),
        with_mode,
        "the field must round-trip unchanged, not be dropped on the way out"
    );

    // The committed PROVENANCE.json predates the field, and `OursPins` carries
    // `deny_unknown_fields`, so this is the shape its two frozen readers must keep parsing.
    let mut without_mode = with_mode;
    without_mode
        .as_object_mut()
        .expect("object")
        .remove("mask_refine_mode");
    let legacy: pc_testkit::provenance::OursPins =
        serde_json::from_value(without_mode).expect("a pre-field document still parses");
    assert_eq!(legacy.mask_refine_mode, None);
}

/// The report generators render this string, so it is part of the committed documents'
/// contract rather than an internal detail. Pinned as literals because the two reports name
/// the mode in prose a human reads.
///
/// Turns red if a variant is renamed or the rendering drops the type prefix.
#[test]
fn the_report_facing_mode_names_are_the_fully_qualified_variants() {
    assert_eq!(
        mode_display(MaskRefineMode::Simple),
        "MaskRefineMode::Simple"
    );
    assert_eq!(
        mode_display(MaskRefineMode::Annotation),
        "MaskRefineMode::Annotation"
    );
}
