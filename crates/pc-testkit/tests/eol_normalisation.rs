//! spec §16.33 item 8 — Windows checkouts must not rewrite the bytes of a digest-pinned text
//! artifact.
//!
//! **The hazard, stated concretely because it is not hypothetical.** Git-for-Windows ships with
//! `core.autocrlf=true`. Any file Git considers text gets its LF rewritten to CRLF in the working
//! tree on checkout. Seven committed *text* artifacts have their exact bytes pinned by a SHA-256
//! digest recorded in a `PROVENANCE.json` and verified by `verify_committed_artifact` in
//! `recorded_provenance.rs`. On a Windows checkout without an end-of-line attribute, all seven
//! digest checks fail — for a reason that has nothing to do with any code under test. That is the
//! worst possible first impression of a newly supported platform, and it is entirely preventable in
//! `.gitattributes`.
//!
//! **This gate points at `.gitattributes`, not at the fixtures** (cookbook rule 12: a gate must
//! point at the artifact carrying the risk). It asks `git check-attr` — the real consumer of that
//! file, not a hand-rolled glob matcher — what attribute each path resolves to. Cookbook rule 14c:
//! never validate with a weaker parser than the real consumer, and for `.gitattributes` the real
//! consumer is `git` itself.
//!
//! **What this gate does NOT establish.** It does not prove any digest is currently correct — that
//! is `recorded_provenance.rs`'s job — and it does not prove a Windows checkout works, which only a
//! Windows runner can show. It proves the attribute is declared for exactly the artifacts whose
//! bytes a `PROVENANCE.json` pins.

use pc_testkit::paths;
use std::collections::BTreeSet;
use std::path::Path;

/// Every other committed text artifact whose bytes are compared exactly by a test. These are not
/// digest-pinned provenance outputs, so they need a separate list rather than being folded into
/// the provenance-derived seven above.
const BYTE_EXACT_NON_PROVENANCE_ARTIFACTS: &[&str] = &[
    "crates/pc-config/src/default_profile.toml",
    ".claude/agents/architect.md",
    ".claude/agents/fable-adjudicator.md",
    ".claude/agents/fresh-reader.md",
    ".claude/agents/rust-engineer.md",
    "tests/fixtures/upstream/ocr_output/good_detected_text.csv",
    "tests/fixtures/upstream/ocr_output/good_detected_text.txt",
    "docs/GOLDEN_CALIBRATION.md",
];
/// Every committed TEXT artifact whose exact bytes a `PROVENANCE.json` record pins with an
/// `output_sha256`. Enumerated on 2026-08-04 by reading every
/// `tests/fixtures/recorded/*/PROVENANCE.json` and taking each record whose `output` is not an image
/// or model file; transcribed into §16.33 item 8.
///
/// Hand-written on purpose. `every_digest_pinned_text_artifact_is_still_pinned_by_a_provenance_record`
/// re-derives the same set from the provenance files and compares — so this list is checked against
/// an independent oracle rather than trusted, and neither side is `.gitattributes` (the artifact
/// under test).
const DIGEST_PINNED_TEXT_ARTIFACTS: &[&str] = &[
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_detector_blocks.json",
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01#raw.json",
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_upstream_oracle.json",
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_upstream_group_output_equality.json",
    "tests/fixtures/recorded/model_signature/comictextdetector.signature.json",
    "tests/fixtures/recorded/ocr_model_signature/encoder_model.signature.json",
    "tests/fixtures/recorded/ocr_model_signature/decoder_model.signature.json",
];

/// §16.24 item 1(f)'s literal-constant pattern: hard-coded, not `.len()`-derived at the point of
/// comparison, so this file cannot pass by pinning zero artifacts.
const EXPECTED_PINNED_ARTIFACT_COUNT: usize = 7;

/// The vendored file that already carries the marker under §16.30 item 2. Listed separately because
/// it is pinned by a hash in Rust source rather than by a `PROVENANCE.json` record, so the
/// provenance-derived oracle below cannot see it — and it must not be lost when `.gitattributes` is
/// rewritten to add the seven above.
const VENDORED_HASH_PINNED_FILE: &str = "crates/pc-ocr/assets/vocab.txt";

/// A file that must stay under Git's ordinary end-of-line handling, so a blanket `* -text` cannot
/// satisfy this gate. §16.33 item 8 requires the marker to stay narrow, so that its presence on a
/// path keeps meaning "these bytes are pinned".
const ORDINARY_TEXT_FILE: &str = "docs/PIPELINE_SPEC_V1.md";

/// Ask git what the `text` attribute resolves to for `relative`. Returns the raw state word:
/// `"unset"` for `-text`, `"set"` for `text`, `"unspecified"` when no pattern matches, or a value.
fn text_attribute(relative: &str) -> String {
    let root = paths::workspace_root();
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["check-attr", "text", "--"])
        .arg(relative)
        .output()
        .expect(
            "`git check-attr` must run: this gate has no fallback, because a hand-rolled \
                 glob matcher would be a weaker parser than the real consumer (cookbook rule 14c)",
        );
    assert!(
        output.status.success(),
        "`git check-attr text -- {relative}` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Format: `<path>: text: <state>`. The path may contain `:`-free but `#`-carrying names, so
    // split from the RIGHT on `": "` to reach the state without parsing the path.
    stdout
        .trim_end()
        .rsplit(": ")
        .next()
        .unwrap_or_default()
        .to_owned()
}

#[test]
// Anti-vacuity: every other test here iterates the list, so all of them pass on an empty one.
fn the_digest_pinned_text_artifact_count_is_pinned() {
    assert_eq!(
        DIGEST_PINNED_TEXT_ARTIFACTS.len(),
        EXPECTED_PINNED_ARTIFACT_COUNT,
        "§16.33 item 8 enumerates {EXPECTED_PINNED_ARTIFACT_COUNT} digest-pinned text artifacts. \
         Recording a new one means raising this constant in the same commit."
    );
}

#[test]
// spec §16.33 item 8 — the primary requirement. What must break for this to fail: remove or narrow
// the `.gitattributes` pattern covering any one of the seven. Today it fails for all seven, because
// `.gitattributes` is the next pipeline step.
fn every_digest_pinned_text_artifact_is_declared_binary_for_end_of_line_purposes() {
    let mut unprotected = Vec::new();
    for relative in DIGEST_PINNED_TEXT_ARTIFACTS
        .iter()
        .chain(BYTE_EXACT_NON_PROVENANCE_ARTIFACTS)
    {
        let state = text_attribute(relative);
        if state != "unset" {
            unprotected.push(format!("{relative} => text: {state}"));
        }
    }
    assert!(
        unprotected.is_empty(),
        "{} digest-pinned text artifact(s) are not `-text` in .gitattributes, so a Windows \
         checkout with core.autocrlf=true rewrites their bytes and fails their SHA-256 checks:\n  \
         {}\n  Fix: add a `-text` pattern covering each one.",
        unprotected.len(),
        unprotected.join("\n  ")
    );
}

#[test]
// The list above must not be arbitrary or stale. Re-derives the same set from the `PROVENANCE.json`
// files — an oracle independent of `.gitattributes`, which is the artifact under test — and compares
// as a SET, so a swap (one artifact dropped, another added) fails where a count would not.
fn every_digest_pinned_text_artifact_is_still_pinned_by_a_provenance_record() {
    let root = paths::workspace_root();
    let mut derived: BTreeSet<String> = BTreeSet::new();
    let groups = std::fs::read_dir(root.join("tests/fixtures/recorded"))
        .expect("the recorded fixtures root must exist");
    for entry in groups {
        let provenance = entry
            .expect("readable dir entry")
            .path()
            .join("PROVENANCE.json");
        if !provenance.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&provenance).expect("readable PROVENANCE.json");
        let json: serde_json::Value =
            serde_json::from_str(&text).expect("parseable PROVENANCE.json");
        let records = json["records"].as_array().cloned().unwrap_or_default();
        for record in records {
            let Some(output) = record["output"].as_str() else {
                continue;
            };
            if record["output_sha256"].as_str().is_none() {
                continue;
            }
            let lowered = output.to_ascii_lowercase();
            let is_binary = [
                ".png", ".jpg", ".jpeg", ".onnx", ".pt", ".webp", ".tiff", ".bmp",
            ]
            .iter()
            .any(|suffix| lowered.ends_with(suffix));
            if !is_binary {
                derived.insert(output.to_owned());
            }
        }
    }

    let listed: BTreeSet<String> = DIGEST_PINNED_TEXT_ARTIFACTS
        .iter()
        .map(|path| (*path).to_owned())
        .collect();

    assert_eq!(
        listed, derived,
        "DIGEST_PINNED_TEXT_ARTIFACTS no longer matches the set derived from the PROVENANCE.json \
         records. Left = this file's list, right = what the provenance files actually pin. A new \
         recorded text artifact needs a row here AND a `-text` pattern in .gitattributes, or it \
         silently loses its byte pinning on Windows."
    );
}

#[test]
// spec §16.33 item 8 — the marker stays narrow. Without this, a blanket `* -text` would satisfy the
// primary gate above while destroying the marker's meaning, and the gate would then prove nothing
// about the seven artifacts in particular.
fn ordinary_documentation_stays_under_gits_default_end_of_line_handling() {
    let state = text_attribute(ORDINARY_TEXT_FILE);
    assert_eq!(
        state, "unspecified",
        "{ORDINARY_TEXT_FILE} resolves to `text: {state}`. §16.33 item 8 keeps the `-text` marker \
         scoped to digest-pinned artifacts, so its presence keeps meaning `these bytes are \
         pinned`; a blanket rule makes it uninformative."
    );
}

#[test]
// spec §16.30 item 2 — regression control. `.gitattributes` has exactly one row today and this
// gate's fix rewrites that file, so the vendored vocab's existing protection is exactly the sort of
// thing an edit loses in passing.
fn the_vendored_vocab_file_keeps_its_end_of_line_protection() {
    let state = text_attribute(VENDORED_HASH_PINNED_FILE);
    assert_eq!(
        state, "unset",
        "{VENDORED_HASH_PINNED_FILE} is a hash-pinned vendored file (§16.30 item 2) and must stay \
         `-text`; it resolves to `text: {state}`"
    );
}

#[test]
// The documentation cross-check: §16.33 item 8 enumerates these seven paths, and a reader who
// updates one list and not the other gets a red test rather than a silent divergence.
fn every_digest_pinned_text_artifact_path_appears_verbatim_in_the_ratification() {
    let spec = std::fs::read_to_string(paths::workspace_root().join("docs/PIPELINE_SPEC_V1.md"))
        .expect("readable spec");
    let mut absent = Vec::new();
    for relative in DIGEST_PINNED_TEXT_ARTIFACTS {
        if !spec.contains(relative) {
            absent.push(*relative);
        }
    }
    assert!(
        absent.is_empty(),
        "{} digest-pinned artifact path(s) are not named verbatim anywhere in the spec, so \
         §16.33 item 8's enumeration and this file's list have diverged: {absent:?}",
        absent.len()
    );
}

#[test]
// Sanity, and a guard against the whole file passing on a broken checkout: the paths must exist.
fn every_digest_pinned_text_artifact_exists_in_the_checkout() {
    let root = paths::workspace_root();
    let mut missing = Vec::new();
    for relative in DIGEST_PINNED_TEXT_ARTIFACTS {
        if !Path::new(&root.join(relative)).is_file() {
            missing.push(*relative);
        }
    }
    assert!(
        missing.is_empty(),
        "missing pinned artifact(s): {missing:?}"
    );
}
