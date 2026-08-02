//! spec §16.17 item 2, as amended by §16.24 — the shared provenance schema and checker.
//! Frozen gates.
//!
//! Two populations, enumerated independently (cookbook rule 13): the *committed* provenance
//! files, discovered from the filesystem, and *synthetic* documents constructed here to prove
//! each rule can fail. Neither population is derived from the other, and no test here branches
//! on whether the detector group exists (§16.24 item 6).

use pc_testkit::paths;
use pc_testkit::provenance::{
    self, ArtifactRecord, Backend, DetectorPins, GroupProvenance, OursPins, UpstreamPins,
    Violation, PROVENANCE_SCHEMA_VERSION,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The real model digest. Written here only so the control document is *well-formed*; the
/// identity against `pc_models::COMIC_TEXT_DETECTOR.sha256` is asserted in
/// `xtask/tests/provenance_digests.rs` (§16.24 item 2), NOT here — `pc-testkit` must not depend
/// on `pc-models` (`tests/model_signature.rs:53-56`).
const MODEL_DIGEST: &str = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f";
/// Upstream's pinned commit (cookbook rule 3, §16.24 item 14).
const UPSTREAM_COMMIT: &str = "0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3";
/// Well-formed and obviously synthetic; the recorder writes the real value.
const OURS_COMMIT: &str = "8a309a0e16c02e725961a88db620a5db74959356";
/// The decoded-RGB-buffer digest both sides must report (§16.24 item 16(b), R23). Synthetic and
/// well-formed: the real value is measured by the recorder and re-derived by
/// `xtask/tests/provenance_digests.rs`, never asserted against reality from this crate.
/// A literal, not `"3".repeat(64)` — `str::repeat` is not `const` and returns `String`.
const DECODED_RGB_DIGEST: &str = "3333333333333333333333333333333333333333333333333333333333333333";

/// Every group directory under `tests/fixtures/recorded/`, discovered from the filesystem —
/// NOT from a list — so a group added without provenance cannot hide.
fn discover_group_dirs() -> Vec<String> {
    let root = paths::recorded_root();
    let mut groups: Vec<String> = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("failed to read `{}`: {error}", root.display()))
        .map(|entry| entry.expect("readable recorded entry"))
        .filter(|entry| {
            entry
                .file_type()
                .expect("stat-able recorded entry")
                .is_dir()
        })
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    groups.sort();
    groups
}

fn read_group(name: &str) -> GroupProvenance {
    let path = paths::recorded(PathBuf::from(name).join(provenance::PROVENANCE_FILE_NAME));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("reading `{}`: {error}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "`{}` does not satisfy the §16.17 item 2 schema: {error}",
            path.display()
        )
    })
}

/// A non-detector document satisfying every rule. Perturbed one field at a time below; a test
/// asserting "exactly one violation" is only falsifiable because
/// `the_valid_control_documents_have_no_violations` proves this baseline is clean.
fn valid_group() -> GroupProvenance {
    GroupProvenance {
        schema_version: PROVENANCE_SCHEMA_VERSION,
        group: "nlm".into(),
        tool: "cv2.fastNlMeansDenoising".into(),
        command_line: "cargo xtask record-fixtures --only nlm".into(),
        tool_versions: BTreeMap::from([("opencv".to_owned(), "5.0.0".to_owned())]),
        records: vec![ArtifactRecord {
            name: "nightmare".into(),
            output: "tests/fixtures/recorded/nlm/nightmare_h10_t7_s21.png".into(),
            output_sha256: "c6cc1002c209ffadd182f2ebd3162b53b55bbb92f99ee038ebaf419bfed94b8d"
                .into(),
            committed: true,
            source: Some("tests/fixtures/upstream/demo_bubbles/nightmare_bubble_raw.png".into()),
            source_sha256: None,
            params: BTreeMap::from([("h".to_owned(), serde_json::json!(10))]),
        }],
        detector: None,
        diagnostics: BTreeMap::new(),
    }
}

/// A detector document satisfying every rule, in the shape §16.24 item 2 mandates: one
/// group-level block, the page declared once as `input_page`, `model_digest` and both
/// `decoded_rgb_digest`s walker-invisible. The record set and the two `decoded_from` values
/// mirror §16.24 item 17's actual data flow: we decode the committed JPEG, upstream decodes the
/// scratch PNG we wrote from that decode.
fn valid_detector_group() -> GroupProvenance {
    let record = |name: &str, file: &str| ArtifactRecord {
        name: name.into(),
        output: format!("tests/fixtures/recorded/detector/{file}"),
        output_sha256: "0".repeat(63) + "a",
        committed: true,
        source: None,
        source_sha256: None,
        params: BTreeMap::new(),
    };

    GroupProvenance {
        schema_version: PROVENANCE_SCHEMA_VERSION,
        group: "detector".into(),
        tool: "comictextdetector.pt.onnx via ort, and upstream PanelCleaner via cv2.dnn".into(),
        command_line: "cargo xtask-onnx record-fixtures --only detector".into(),
        tool_versions: BTreeMap::from([("ort".to_owned(), "2.0.0-rc.12".to_owned())]),
        records: vec![
            record("ours_detector_blocks", "page01_detector_blocks.json"),
            record("ours_detector_mask", "page01_detector_mask.png"),
            record("ours_raw_json", "page01#raw.json"),
            record("upstream_oracle", "page01_upstream_oracle.json"),
            // §16.24 item 17: the PNG upstream decodes, written by us from the JPEG decode.
            // Scratch, so `committed: false` (§16.13 item 6).
            ArtifactRecord {
                name: "decoded_rgb_png".into(),
                output: "target/xtask-scratch/page01_decoded_rgb.png".into(),
                output_sha256: "1".repeat(64),
                committed: false,
                source: Some("tests/fixtures/recorded/detector/page01.jpg".into()),
                source_sha256: None,
                params: BTreeMap::new(),
            },
        ],
        detector: Some(DetectorPins {
            // §16.24 item 17: the byte-for-byte CC-BY JPEG, 441,914 B, cap overage ratified.
            input_page: "tests/fixtures/recorded/detector/page01.jpg".into(),
            input_page_sha256: "2".repeat(64),
            model: "comictextdetector.pt.onnx".into(),
            model_digest: MODEL_DIGEST.into(),
            ours: OursPins {
                backend: Backend::Ort,
                decoded_rgb_digest: DECODED_RGB_DIGEST.into(),
                decoded_from: "input_page".into(),
                execution_provider: "cpu".into(),
                intra_threads: 0,
                inter_threads: 0,
                pad_value: 0,
                panel_ocr_commit: OURS_COMMIT.into(),
                profile_non_default: BTreeMap::new(),
            },
            upstream: UpstreamPins {
                backend: Backend::Cv2Dnn,
                // Equal to ours by R23; upstream reads the scratch PNG, not the JPEG.
                decoded_rgb_digest: DECODED_RGB_DIGEST.into(),
                decoded_from: "decoded_rgb_png".into(),
                version: "2.4.1".into(),
                commit: UPSTREAM_COMMIT.into(),
                command_line: "python -m pcleaner clean page01.png --profile default".into(),
                profile_non_default: BTreeMap::new(),
                dependency_versions: BTreeMap::from([
                    ("opencv-python-headless".to_owned(), "5.0.0".to_owned()),
                    ("torch".to_owned(), "2.13.0+cpu".to_owned()),
                ]),
            },
        }),
        diagnostics: BTreeMap::new(),
    }
}

// ── population 1: the committed files ────────────────────────────────────────

#[test]
// spec §16.17 item 2 ("schedule a shared provenance schema and shared checker") + §16.13
// item 6 ("every recording group writes a PROVENANCE.json next to its outputs"). Enumerates
// group DIRECTORIES from the filesystem and requires each to parse AND validate. Fails if: a
// group has no PROVENANCE.json; a file carries an unknown key — including a misspelled digest
// key such as `sha_256`, which `deny_unknown_fields` rejects and `recorded_provenance.rs:107`
// silently skips; a required field is missing; or any of R1–R22 is broken.
fn every_recorded_group_parses_and_validates() {
    let groups = discover_group_dirs();
    assert!(
        !groups.is_empty(),
        "no recorded groups found — broken checkout"
    );
    for group in &groups {
        let parsed = read_group(group);
        assert_eq!(
            provenance::validate(group, &parsed),
            Vec::<Violation>::new(),
            "`{group}/PROVENANCE.json` violates the §16.17 item 2 schema"
        );
    }
    println!("validated {} group(s): {groups:?}", groups.len());
}

#[test]
// spec §16.24 item 1(a)/(c) and the frozen walker at `recorded_provenance.rs:90-119`. The
// canonical schema must preserve the property that walker depends on: within one JSON object,
// every `*_sha256` key has a string sibling holding its path. Falsifiable: rename
// `output_sha256` to `digest`, or keep `source_sha256` while dropping `source`, and this fails
// HERE with a precise message instead of panicking inside the frozen gate.
//
// It also pins the other half of §16.24 item 2: `model_digest` must NOT be collected, so the
// walked digest count must stay at the 6 declarations of `recorded_provenance.rs:14-21` for the
// three migrated groups — the count is asserted as a floor AND the detector group's model
// digest is asserted absent from the walked set by name.
fn every_walker_visible_digest_has_its_sibling_path() {
    fn walk(value: &serde_json::Value, at: &str, seen: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    if key == "sha256" || key.ends_with("_sha256") {
                        let sibling = match key.as_str() {
                            "sha256" => "output",
                            other => other.strip_suffix("_sha256").expect("checked above"),
                        };
                        let path = map.get(sibling).and_then(serde_json::Value::as_str);
                        assert!(
                            path.is_some(),
                            "digest `{key}` at {at} has no string sibling `{sibling}` — \
                             `recorded_provenance.rs:112-119` would panic"
                        );
                        assert!(
                            child.as_str().is_some(),
                            "digest `{key}` at {at} must be a string"
                        );
                        seen.push(path.expect("checked above").to_owned());
                    }
                    walk(child, &format!("{at}.{key}"), seen);
                }
            }
            serde_json::Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    walk(child, &format!("{at}[{index}]"), seen);
                }
            }
            _ => {}
        }
    }

    let mut walked = Vec::new();
    for group in discover_group_dirs() {
        let path = paths::recorded(PathBuf::from(&group).join(provenance::PROVENANCE_FILE_NAME));
        // Name the file and the reason in both failure paths. `.expect("readable")` /
        // `.expect("valid JSON")` identify neither, and this loop visits every group — so the one
        // thing a reader needs on failure is which file (cookbook rule 13: diagnostics are part of
        // the gate).
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read `{}`: {error}", path.display()));
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|error| panic!("`{}` is not valid JSON: {error}", path.display()));
        walk(&value, &group, &mut walked);
    }

    // Both directions. Lower bound: the loop cannot pass by visiting nothing.
    assert!(
        walked.len() >= 6,
        "expected at least the 6 declarations of `recorded_provenance.rs:14-21`, walked {}: {walked:?}",
        walked.len()
    );
    // Upper bound on the ONE path §16.24 item 2 forbids declaring twice: the bare model
    // filename is `model_signature`'s to declare, and the detector group must not repeat it.
    let model_declarations = walked
        .iter()
        .filter(|path| path.as_str() == "comictextdetector.pt.onnx")
        .count();
    assert_eq!(
        model_declarations, 1,
        "the bare model filename must be declared exactly once tree-wide (§16.24 item 2); \
         a detector-group `model_sha256` would make it 2 and trip `:297-311`"
    );
    println!("walked {} digest declaration(s)", walked.len());
}

// ── population 2: synthetic documents, one rule at a time ────────────────────

#[test]
// The control for every negative test below (cookbook rule 13, both directions). If either
// baseline produced a violation, every "exactly one violation" assertion would be
// unfalsifiable — it would pass on a checker that always returned that one violation.
fn the_valid_control_documents_have_no_violations() {
    assert_eq!(provenance::validate("nlm", &valid_group()), Vec::new());
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &valid_detector_group()),
        Vec::new()
    );
}

#[test]
// spec §16.13 item 6 (group locality). A provenance file describing another group's artifacts
// is the hole `recorded_provenance.rs:153-165` closed at the digest level; this closes it at
// the schema level, where the diagnostic names both sides.
fn a_group_name_disagreeing_with_its_directory_is_rejected() {
    let mut group = valid_group();
    group.group = "inter_area".into();
    assert_eq!(
        provenance::validate("nlm", &group),
        vec![Violation::GroupNameMismatch {
            declared: "inter_area".into(),
            directory: "nlm".into(),
        }]
    );
}

#[test]
// spec §16.24 item 3(c) — the digest sweeps. THE hole this task exists to close: a digest under
// a key the frozen walker's heuristic does not recognise. `deny_unknown_fields` catches it in a
// structural slot; R11 catches the key shape inside the free-form `params`; R12 catches a
// digest-shaped VALUE under an innocent-looking key. All three spellings of the same mistake.
fn a_digest_hidden_in_params_is_rejected_by_key_and_by_value() {
    let hex = "c6cc1002c209ffadd182f2ebd3162b53b55bbb92f99ee038ebaf419bfed94b8d";

    let mut by_key = valid_group();
    by_key.records[0]
        .params
        .insert("sha_256".into(), serde_json::json!("not-a-digest"));
    assert_eq!(
        provenance::validate("nlm", &by_key),
        vec![Violation::DigestLikeKeyOutsideStructuralSlot {
            at: "nightmare".into(),
            key: "sha_256".into(),
        }]
    );

    let mut by_value = valid_group();
    by_value.records[0]
        .params
        .insert("checksum".into(), serde_json::json!(hex));
    assert_eq!(
        provenance::validate("nlm", &by_value),
        vec![Violation::DigestLikeValueOutsideStructuralSlot {
            at: "nightmare".into(),
            key: "checksum".into(),
        }]
    );

    // Nested, because a one-level sweep would pass the two above and still miss this.
    let mut nested = valid_group();
    nested.records[0].params.insert(
        "nested".into(),
        serde_json::json!({ "inner": { "file_hash": "x" } }),
    );
    assert_eq!(
        provenance::validate("nlm", &nested),
        vec![Violation::DigestLikeKeyOutsideStructuralSlot {
            at: "nightmare".into(),
            key: "file_hash".into(),
        }]
    );
}

#[test]
// spec §16.24 item 3(c) — the sweeps must reach EVERY free-form map, not only `params` and
// `diagnostics`. Found by the review gate after F1-A landed: `validate` swept `record.params`
// and `group.diagnostics` but neither `profile_non_default`, so a digest parked in a profile
// map was a digest present in the file and absent from the verified set — precisely the
// invariant the sweeps exist to maintain.
//
// The upstream side matters most: its `profile_non_default` records a profile produced by a
// third-party tool run, so its contents are the least predictable in the document.
//
// Falsifiable in both directions: remove either `validate_freeform` call in `validate_ours` /
// `validate_upstream` and the matching half fails; a profile map holding ordinary values must
// produce NO violation, which the control below asserts so this cannot pass by over-reporting.
fn a_digest_hidden_in_a_detector_profile_map_is_rejected_on_both_sides() {
    let hex = "c6cc1002c209ffadd182f2ebd3162b53b55bbb92f99ee038ebaf419bfed94b8d";

    let mut ours = valid_detector_group();
    ours.detector
        .as_mut()
        .expect("detector pins")
        .ours
        .profile_non_default
        .insert("checksum".into(), serde_json::json!(hex));
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &ours),
        vec![Violation::DigestLikeValueOutsideStructuralSlot {
            at: "detector.ours.profile_non_default".into(),
            key: "checksum".into(),
        }]
    );

    let mut upstream = valid_detector_group();
    upstream
        .detector
        .as_mut()
        .expect("detector pins")
        .upstream
        .profile_non_default
        .insert("sha_256".into(), serde_json::json!("not-a-digest"));
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &upstream),
        vec![Violation::DigestLikeKeyOutsideStructuralSlot {
            at: "detector.upstream.profile_non_default".into(),
            key: "sha_256".into(),
        }]
    );

    // Nested, since a one-level sweep would pass both cases above and still miss this.
    let mut nested = valid_detector_group();
    nested
        .detector
        .as_mut()
        .expect("detector pins")
        .upstream
        .profile_non_default
        .insert(
            "masker".into(),
            serde_json::json!({ "tier": { "file_hash": "x" } }),
        );
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &nested),
        vec![Violation::DigestLikeKeyOutsideStructuralSlot {
            at: "detector.upstream.profile_non_default".into(),
            key: "file_hash".into(),
        }]
    );

    // The other direction: a profile map carrying legitimate non-default keys must be clean, or
    // the rule would fire on every real recording and get widened back out.
    let mut legitimate = valid_detector_group();
    let pins = legitimate.detector.as_mut().expect("detector pins");
    pins.ours
        .profile_non_default
        .insert("intra_threads".into(), serde_json::json!(8));
    pins.upstream
        .profile_non_default
        .insert("mask_growth_step_pixels".into(), serde_json::json!(2));
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &legitimate),
        Vec::<Violation>::new(),
        "ordinary non-default profile keys must not be reported"
    );
}

#[test]
// spec §16.24 item 3(c). `tool_versions` and `dependency_versions` are `BTreeMap<String, String>`,
// so the type stops a nested object — but NOT a digest-shaped value. `{"opencv": "<64 hex>"}`
// parses fine and would be a digest present in the file and outside the verified set. R12 is a
// rule about value *shape*, so the `String` type is not a safeguard against it.
//
// Both directions: real version strings must stay clean, which the control asserts.
fn a_digest_shaped_version_string_is_rejected() {
    let hex = "c6cc1002c209ffadd182f2ebd3162b53b55bbb92f99ee038ebaf419bfed94b8d";

    let mut group = valid_group();
    group.tool_versions.insert("opencv".into(), hex.into());
    assert_eq!(
        provenance::validate("nlm", &group),
        vec![Violation::DigestLikeValueOutsideStructuralSlot {
            at: "tool_versions".into(),
            key: "opencv".into(),
        }]
    );

    let mut deps = valid_detector_group();
    deps.detector
        .as_mut()
        .expect("detector pins")
        .upstream
        .dependency_versions
        .insert("torch".into(), hex.into());
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &deps),
        vec![Violation::DigestLikeValueOutsideStructuralSlot {
            at: "detector.upstream.dependency_versions".into(),
            key: "torch".into(),
        }]
    );

    // Control: ordinary versions are not digests, so neither map may be reported.
    let mut clean = valid_detector_group();
    clean
        .tool_versions
        .insert("opencv".into(), "5.0.0".to_owned());
    clean
        .detector
        .as_mut()
        .expect("detector pins")
        .upstream
        .dependency_versions
        .insert("torch".into(), "2.4.1".to_owned());
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &clean),
        Vec::<Violation>::new(),
        "ordinary version strings must not be reported"
    );
}

#[test]
// spec §16.24 item 3(c) — uppercase-digest rejection, explicitly ratified. An uppercase digest
// VERIFIES at `recorded_provenance.rs:177` (which lowercases first), so it is exactly the
// spelling that passes the existing gate while making two provenance files textually
// incomparable. Length and alphabet are pinned in the same sweep.
fn a_non_canonical_digest_is_rejected() {
    for bad in [
        "C6CC1002C209FFADD182F2EBD3162B53B55BBB92F99EE038EBAF419BFED94B8D",
        "deadbeef",
        "",
        "g6cc1002c209ffadd182f2ebd3162b53b55bbb92f99ee038ebaf419bfed94b8d",
    ] {
        let mut group = valid_group();
        group.records[0].output_sha256 = bad.into();
        assert_eq!(
            provenance::validate("nlm", &group),
            vec![Violation::MalformedDigest {
                at: "nightmare".into(),
                field: "output_sha256",
                value: bad.into(),
            }],
            "digest `{bad}` must be rejected"
        );
    }
}

#[test]
// spec §16.20 item 3's closing ¶: "machine-dependent absolute paths must be normalised out of
// both sides before commit." Across six identical upstream runs those paths were the ONLY
// difference (§16.20 item 7), so an unnormalised path makes two byte-identical runs look
// divergent. `..` is rejected for the same reason a bare exemption is (rule 13's corollary).
fn an_absolute_or_escaping_path_is_rejected() {
    for bad in [
        "/home/maintainer/panel-ocr/tests/fixtures/recorded/nlm/x.png",
        "tests/fixtures/recorded/nlm/../../secret.png",
    ] {
        let mut group = valid_group();
        group.records[0].output = bad.into();
        let violations = provenance::validate("nlm", &group);
        assert!(
            violations.contains(&Violation::NonRelativePath {
                at: "nightmare".into(),
                field: "output",
                path: bad.into(),
            }),
            "path `{bad}` must be rejected; got {violations:?}"
        );
    }
}

#[test]
// spec §16.13 item 6: "Derived intermediates are scratch, not fixtures." `committed: false` is
// legal only for `target/xtask-scratch/` and `committed: true` only inside the declaring group
// — the two halves of the frozen gate's `:153-165` / `:184-208` split, asserted here where the
// message can name the rule. Both directions.
fn the_committed_flag_must_agree_with_the_path_shape() {
    let mut scratch_but_committed = valid_group();
    scratch_but_committed.records[0].output = "target/xtask-scratch/x.png".into();
    assert_eq!(
        provenance::validate("nlm", &scratch_but_committed),
        vec![Violation::CommittedPathOutsideGroup {
            at: "nightmare".into(),
            path: "target/xtask-scratch/x.png".into(),
        }]
    );

    let mut committed_but_flagged_scratch = valid_group();
    committed_but_flagged_scratch.records[0].committed = false;
    assert_eq!(
        provenance::validate("nlm", &committed_but_flagged_scratch),
        vec![Violation::UncommittedPathNotScratch {
            at: "nightmare".into(),
            path: "tests/fixtures/recorded/nlm/nightmare_h10_t7_s21.png".into(),
        }]
    );
}

#[test]
// spec §16.24 item 2 — the erratum's core constraint, and the reason this whole shape changed:
// `recorded_provenance.rs:297-311` rejects any duplicate declared path across the tree,
// unconditionally, and no constant edit can satisfy it. The page is declared exactly once, as
// `input_page`; listing it again as a record `output` is the duplicate. Falsifiable in both
// directions: the control document has the page as `input_page` only and is clean.
fn the_input_page_may_not_also_be_a_record_output() {
    let mut group = valid_detector_group();
    group.records.push(ArtifactRecord {
        name: "page".into(),
        output: "tests/fixtures/recorded/detector/page01.jpg".into(),
        output_sha256: "2".repeat(64),
        committed: true,
        source: None,
        source_sha256: None,
        params: BTreeMap::new(),
    });
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &group),
        vec![Violation::InputPageAlsoDeclaredAsRecordOutput {
            record: "page".into(),
            path: "tests/fixtures/recorded/detector/page01.jpg".into(),
        }]
    );
}

#[test]
// spec §16.24 item 2: the model must be a BARE filename, because the frozen gate's
// deliberately-unverifiable exemption (`:187-191`) is an enumerated allow-list of three bare
// model filenames. A path-shaped value would fall through to the committed branch and fail
// locality. Also pins that the digest slot stays canonical (its VALUE is bound in
// `xtask/tests/provenance_digests.rs`, §16.24 item 2's condition on the walker dodge).
fn the_model_must_be_a_bare_filename_with_a_canonical_digest() {
    let mut pathy = valid_detector_group();
    if let Some(pins) = pathy.detector.as_mut() {
        pins.model = "models/comictextdetector.pt.onnx".into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &pathy),
        vec![Violation::ModelIsNotABareFilename {
            model: "models/comictextdetector.pt.onnx".into(),
        }]
    );

    let mut bad_digest = valid_detector_group();
    if let Some(pins) = bad_digest.detector.as_mut() {
        pins.model_digest = "1A86ACE7".into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &bad_digest),
        vec![Violation::MalformedDigest {
            at: "detector".into(),
            field: "model_digest",
            value: "1A86ACE7".into(),
        }]
    );
}

#[test]
// spec §16.24 item 16(b) — R23, the binding invariant "both detectors must consume
// byte-identical pixel buffers". Under item 17 the two sides read DIFFERENT FILES with DIFFERENT
// DECODERS — we decode the committed JPEG with the `image` crate, upstream decodes the scratch
// PNG with `cv2` — so this equality is live, not vacuous, and it is the only field that can
// witness the invariant.
//
// The negative control is written as a **channel-order** failure specifically, because that is
// the realistic way to break it and the reason §16.20 item 6 calls the backend load-bearing:
// `cv2.imread` yields BGR while the cv2 branch of `preprocess_img` feeds RGB, so an oracle
// script that hashes the unconverted array reports a different digest for the same pixels. The
// violation carries BOTH values so a reviewer sees which side is the odd one out.
//
// A 1-LSB input difference moves confidence 0.079–0.089 (§16.20 item 5), which is why this is a
// hard equality with no tolerance anywhere near it.
fn the_two_sides_must_declare_the_same_decoded_rgb_buffer() {
    // Ours as if hashed after BGR->RGB; upstream as if hashed straight off `cv2.imread`.
    let bgr_by_mistake = "4444444444444444444444444444444444444444444444444444444444444444";

    let mut group = valid_detector_group();
    if let Some(pins) = group.detector.as_mut() {
        pins.upstream.decoded_rgb_digest = bgr_by_mistake.into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &group),
        vec![Violation::DecodedRgbDigestMismatch {
            ours: DECODED_RGB_DIGEST.into(),
            upstream: bgr_by_mistake.into(),
        }],
        "exactly one violation, naming both sides"
    );

    // Both directions of R23's own falsifiability: equal digests must produce NOTHING, or the
    // rule would be unfalsifiable — and a malformed digest must still be caught by R6 rather
    // than being masked by the equality check passing on two equal wrong strings.
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &valid_detector_group()),
        Vec::new()
    );
    let mut both_malformed = valid_detector_group();
    if let Some(pins) = both_malformed.detector.as_mut() {
        pins.ours.decoded_rgb_digest = "deadbeef".into();
        pins.upstream.decoded_rgb_digest = "deadbeef".into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &both_malformed),
        vec![
            Violation::MalformedDigest {
                at: "detector.ours".into(),
                field: "decoded_rgb_digest",
                value: "deadbeef".into(),
            },
            Violation::MalformedDigest {
                at: "detector.upstream".into(),
                field: "decoded_rgb_digest",
                value: "deadbeef".into(),
            },
        ],
        "R23 passing on two equal wrong values must not suppress R6"
    );
}

#[test]
// spec §16.24 items 16(b)/17: `decoded_from` records WHICH artifact each side decoded, so the
// `image`->PNG->`cv2` round-trip is visible to a reviewer rather than folklore. A dangling
// reference would make the field decorative. Both the `"input_page"` sentinel and a record name
// must resolve; the control document uses one of each, so the passing direction is covered by
// `the_valid_control_documents_have_no_violations`.
fn each_side_must_say_which_artifact_it_decoded() {
    let mut group = valid_detector_group();
    if let Some(pins) = group.detector.as_mut() {
        pins.upstream.decoded_from = "typo".into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &group),
        vec![Violation::UnknownDecodedFromReference {
            at: "detector.upstream".into(),
            name: "typo".into(),
        }]
    );
}

#[test]
// spec §16.24 item 2 — the erratum's constraint, checked against the shape we are shipping
// BEFORE the fixture exists. This simulates the frozen walker (`recorded_provenance.rs:90-151`)
// over the serialised detector document and asserts three things no other test can:
//
//   1. every key the walker WOULD collect resolves to a sibling path, so `:112-119` cannot panic
//      (the architect's schema failed exactly here, §16.24 item 2(b));
//   2. no filesystem path is collected twice, so `:297-311`'s unconditional duplicate rejection
//      cannot fire (the failure mode of my own earlier draft, item 2(a));
//   3. the three deliberately walker-invisible digests — `model_digest` and both
//      `decoded_rgb_digest`s — are NOT collected, because each names no path.
//
// Without this, walker-safety is only checked when the detector group records, which is far too
// late: the migration would already be committed.
fn the_detector_document_is_walker_safe() {
    let document = serde_json::to_value(valid_detector_group()).expect("serialisable");

    fn walk(value: &serde_json::Value, at: &str, collected: &mut Vec<(String, String)>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    if key == "sha256" || key.ends_with("_sha256") {
                        let sibling = match key.as_str() {
                            "sha256" => "output",
                            other => other.strip_suffix("_sha256").expect("checked above"),
                        };
                        let path = map.get(sibling).and_then(serde_json::Value::as_str);
                        assert!(
                            path.is_some(),
                            "walker would panic at {at}: `{key}` has no sibling `{sibling}`"
                        );
                        collected.push((key.clone(), path.expect("checked").to_owned()));
                    }
                    walk(child, &format!("{at}.{key}"), collected);
                }
            }
            serde_json::Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    walk(child, &format!("{at}[{index}]"), collected);
                }
            }
            _ => {}
        }
    }

    let mut collected = Vec::new();
    walk(&document, "detector", &mut collected);

    // (3) the invisible slots stay invisible. Falsifiable: rename `model_digest` to
    // `model_sha256` and this fails, which is the whole point of §16.24 item 2.
    let collected_keys: Vec<&str> = collected.iter().map(|(key, _)| key.as_str()).collect();
    for invisible in ["model_digest", "decoded_rgb_digest"] {
        assert!(
            !collected_keys.iter().any(|key| key.contains(invisible)),
            "`{invisible}` must not be walker-visible; collected {collected_keys:?}"
        );
    }

    // (2) no path twice. `:297-311` rejects duplicates across the WHOLE tree, so within one
    // document is the minimum bar.
    let mut paths: Vec<&str> = collected.iter().map(|(_, path)| path.as_str()).collect();
    paths.sort();
    let duplicates: Vec<&&str> = paths
        .windows(2)
        .filter(|pair| pair[0] == pair[1])
        .map(|pair| &pair[0])
        .collect();
    assert!(
        duplicates.is_empty(),
        "duplicate declared path(s): {duplicates:?}"
    );

    // Lower bound, so the walk cannot pass by collecting nothing: the four committed record
    // outputs, the scratch PNG output, and the input page.
    assert_eq!(
        paths.len(),
        6,
        "expected 6 walker-visible declarations in the detector document, got {paths:?}"
    );
}

#[test]
// spec §16.20 item 6: "With a `.onnx` model, upstream does not use PyTorch for inference at
// all" — `cv2.dnn.readNetFromONNX`, and the torch branch feeds BGR while the cv2 branch feeds
// RGB, so "a maintainer who records the oracle from the `.pt` weights ... gets a different code
// path and a different answer." A provenance claiming `upstream` + `ort`, or `ours` + `cv2_dnn`,
// describes a run nobody performed. Biconditional, both directions asserted.
fn each_side_must_declare_its_own_backend() {
    let mut upstream_wrong = valid_detector_group();
    if let Some(pins) = upstream_wrong.detector.as_mut() {
        pins.upstream.backend = Backend::Ort;
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &upstream_wrong),
        vec![Violation::BackendContradictsSide {
            at: "detector.upstream".into(),
            expected: Backend::Cv2Dnn,
            found: Backend::Ort,
        }]
    );

    let mut ours_wrong = valid_detector_group();
    if let Some(pins) = ours_wrong.detector.as_mut() {
        pins.ours.backend = Backend::Cv2Dnn;
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &ours_wrong),
        vec![Violation::BackendContradictsSide {
            at: "detector.ours".into(),
            expected: Backend::Ort,
            found: Backend::Cv2Dnn,
        }]
    );
}

#[test]
// spec §16.20 item 3(a) + §16.24 item 3(a): the upstream side must pin version, a full 40-hex
// commit, "the invoking command line of the upstream run itself", and resolved dependency
// versions. Cookbook rule 3: "an oracle nobody can re-run is not an oracle", and "pin what you
// compared against". §16.20 item 7 makes the dependency versions the ONLY means of attributing
// a later divergence — so an empty map is a violation, not a default.
fn the_upstream_side_must_be_fully_pinned() {
    let mut group = valid_detector_group();
    if let Some(pins) = group.detector.as_mut() {
        pins.upstream.version = String::new();
        pins.upstream.commit = "0afa21f".into();
        pins.upstream.command_line = String::new();
        pins.upstream.dependency_versions.clear();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &group),
        vec![
            Violation::EmptyField {
                at: "detector.upstream".into(),
                field: "command_line"
            },
            Violation::EmptyField {
                at: "detector.upstream".into(),
                field: "version"
            },
            Violation::MalformedCommit {
                at: "detector.upstream".into(),
                field: "commit",
                value: "0afa21f".into(),
            },
            Violation::EmptyDependencyVersions,
        ]
    );
}

#[test]
// spec §16.22 item 1 and item 5(b): CUDA "is quarantined from every gate, fixture and
// recording", and the recorder must refuse a non-CPU execution provider. §16.22 item 6 and
// §16.21 item 5 make the resolved EP and both thread counts mandatory provenance. This pins the
// EP; the thread counts are recorded and deliberately NOT constrained (0 is legal, §16.21
// item 1), which R22 states rather than leaves implied.
fn a_non_cpu_execution_provider_is_rejected_and_thread_counts_are_free() {
    let mut cuda = valid_detector_group();
    if let Some(pins) = cuda.detector.as_mut() {
        pins.ours.execution_provider = "cuda".into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &cuda),
        vec![Violation::ExecutionProviderNotCpu {
            found: "cuda".into()
        }]
    );

    // The other direction: legal thread counts must NOT be reported, or the gate would be
    // asserting a policy no clause states.
    for (intra, inter) in [(0usize, 0usize), (1, 1), (8, 2)] {
        let mut group = valid_detector_group();
        if let Some(pins) = group.detector.as_mut() {
            pins.ours.intra_threads = intra;
            pins.ours.inter_threads = inter;
        }
        assert_eq!(
            provenance::validate(provenance::DETECTOR_GROUP, &group),
            Vec::new(),
            "intra={intra} inter={inter} are legal under §16.21 item 1"
        );
    }
}

#[test]
// spec §16.20 items 5 and 8: `pad_value` and our own resize arithmetic are fixture-affecting
// (the PAD_VALUE erratum measured a confidence change from that constant alone), so the
// producing tree must be pinned with a full commit name.
fn our_side_must_pin_a_full_panel_ocr_commit() {
    let mut group = valid_detector_group();
    if let Some(pins) = group.detector.as_mut() {
        pins.ours.panel_ocr_commit = "8a309a0".into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &group),
        vec![Violation::MalformedCommit {
            at: "detector.ours".into(),
            field: "panel_ocr_commit",
            value: "8a309a0".into(),
        }]
    );
}

#[test]
// spec §16.24 item 2: the pins are group-level and required EXACTLY for the `detector` group.
// Both directions: a detector group without them fails, and a non-detector group that grew them
// (a copy-paste) fails too — the second direction is what stops the block spreading to groups
// whose paths would then be declared twice.
fn the_detector_block_is_required_exactly_in_the_detector_group() {
    let mut missing = valid_detector_group();
    missing.detector = None;
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &missing),
        vec![Violation::MissingDetectorBlock {
            group: "detector".into()
        }]
    );

    let mut unexpected = valid_group();
    unexpected.detector = valid_detector_group().detector;
    let violations = provenance::validate("nlm", &unexpected);
    assert!(
        violations.contains(&Violation::UnexpectedDetectorBlock {
            group: "nlm".into()
        }),
        "got {violations:?}"
    );
}

#[test]
// spec §16.13 item 6's reviewability requirement: two records under one name make "which
// artifact does this digest describe?" unanswerable, and `recorded_provenance.rs:297-311`
// rejects duplicate *paths* only, not duplicate names. §16.24 item 16 also depends on names
// being unique, since `decoded_rgb_record` resolves by name.
fn duplicate_record_names_are_rejected() {
    let mut group = valid_group();
    group.records.push(ArtifactRecord {
        output: "tests/fixtures/recorded/nlm/ray_h10_t7_s21.png".into(),
        output_sha256: "9cdd17366dbabf152f67b8d70bee43745c7e45011f8d1cf2fc55b83646fe1658".into(),
        ..group.records[0].clone()
    });
    assert_eq!(
        provenance::validate("nlm", &group),
        vec![Violation::DuplicateRecordName {
            name: "nightmare".into()
        }]
    );
}

#[test]
// spec §16.24 item 2 — duplicate declared paths are forbidden within one provenance group,
// even when the records have different names. This does NOT cover duplicate paths across
// groups; that whole-tree half remains the frozen walker's responsibility.
fn duplicate_record_outputs_are_rejected_within_a_group() {
    let mut duplicate = valid_group();
    let duplicate_path = duplicate.records[0].output.clone();
    let mut second = duplicate.records[0].clone();
    second.name = "ray".into();
    duplicate.records.push(second);
    let violations = provenance::validate("nlm", &duplicate);
    assert_eq!(
        violations.len(),
        1,
        "expected one violation, got {violations:?}"
    );
    assert_eq!(
        format!("{violations:?}"),
        format!("[DuplicateRecordOutput {{ path: {:?} }}]", duplicate_path),
        "the violation must identify the duplicated output path"
    );

    let mut legal = valid_group();
    let mut distinct = legal.records[0].clone();
    distinct.name = "ray".into();
    distinct.output = "tests/fixtures/recorded/nlm/ray_h10_t7_s21.png".into();
    legal.records.push(distinct);
    assert_eq!(provenance::validate("nlm", &legal), Vec::<Violation>::new());
}

#[test]
// spec §16.13 item 4 ("a missing tool is always a reported skip, never a Rust stand-in") — a
// provenance file with no records describes no recording, letting a group directory exist while
// claiming nothing. `schema_version` is pinned in the same test so a future shape change cannot
// be read by this checker as if it were v1.
fn an_empty_or_stale_document_is_rejected() {
    let mut empty = valid_group();
    empty.records.clear();
    assert_eq!(
        provenance::validate("nlm", &empty),
        vec![Violation::NoRecords]
    );

    let mut stale = valid_group();
    stale.schema_version = PROVENANCE_SCHEMA_VERSION + 1;
    assert_eq!(
        provenance::validate("nlm", &stale),
        vec![Violation::SchemaVersion {
            found: PROVENANCE_SCHEMA_VERSION + 1
        }]
    );
}

#[test]
// spec §16.24 item 1(d): the checker is pure structure over parsed JSON and must round-trip, so
// `xtask` can write through these types and the tests can read the same bytes back. A field
// that serialises under a different name than it deserialises would silently split the writer
// from the checker — the drift §16.13 item 3 exists to prevent.
fn the_canonical_types_round_trip_through_json() {
    for original in [valid_group(), valid_detector_group()] {
        let text = serde_json::to_string_pretty(&original).expect("serialisable");
        let parsed: GroupProvenance = serde_json::from_str(&text).expect("round-trips");
        assert_eq!(parsed, original);
    }
}

#[test]
// spec §16.24 item 1(a) — the migration's acceptance gate, protected at its sharpest edge.
//
// An absent `Option` must be OMITTED from the JSON, never written as `null`. This is not style:
// the frozen walker matches on the KEY (`recorded_provenance.rs:107`), so a
// `"source_sha256": null` is collected, its `source` sibling resolves, and then `:120-122`
// **panics** with "must be a string". `nlm` and `inter_area` both carry `source` with no
// `source_sha256` (§16.24 item 1(e) keeps upstream digests out of these files), so dropping
// `skip_serializing_if` turns `cargo test -p pc-testkit --test recorded_provenance` from green to
// a panic — and the migration would look like the frozen test's fault.
//
// Falsifiable in both directions: remove the attribute and the `null` assertions fail; the
// `detector` half also asserts the key is present when it IS `Some`, so a blanket skip that
// dropped real data would fail too.
fn an_absent_option_is_omitted_from_the_json_never_written_as_null() {
    let plain = serde_json::to_value(valid_group()).expect("serialisable");
    let record = &plain["records"][0];
    assert!(
        record.get("source_sha256").is_none(),
        "`source_sha256` must be absent, not null: {record}"
    );
    assert!(
        record.get("source").is_some(),
        "a present `source` must survive"
    );
    assert!(
        plain.get("detector").is_none(),
        "a non-detector group must not carry a `detector` key at all: {plain}"
    );

    let detector = serde_json::to_value(valid_detector_group()).expect("serialisable");
    assert!(
        detector.get("detector").is_some(),
        "the detector group MUST carry its pins — the skip must not swallow real data"
    );
    // The scratch record is the one with `source` but no `source_sha256`; the same shape the
    // migrated groups use.
    let scratch = detector["records"]
        .as_array()
        .expect("records is an array")
        .iter()
        .find(|record| record["name"] == "decoded_rgb_png")
        .expect("the scratch record exists");
    assert!(scratch.get("source_sha256").is_none(), "{scratch}");
}

// ── the five rules the F1 review found untested or half-tested ───────────────

#[test]
// spec §16.24 item 3(a) — R3, the group-level `tool`. "Every recorded artifact carries the tool
// that produced it": a provenance naming no tool describes a recording nobody can attribute, and
// cookbook rule 3's "pin what you compared against" is unsatisfiable without it.
//
// Untested until now. `the_upstream_side_must_be_fully_pinned` covers the OTHER `EmptyField`
// sites (R19, `at: "detector.upstream"`), so deleting the group-level check at
// `provenance.rs:218-223` left every test green. This test fails on that deletion and that one
// alone, because `at` is the DIRECTORY name — the two sites are distinguishable by `at`.
//
// Does NOT cover: whether a whitespace-only or placeholder tool name (`" "`, `"TODO"`) is
// acceptable. The rule is `String::is_empty`, not `trim().is_empty()`, so `" "` passes today;
// nothing has ratified that either way, so nothing here freezes it.
fn a_group_must_name_the_tool_that_produced_it() {
    let mut plain = valid_group();
    plain.tool = String::new();
    assert_eq!(
        provenance::validate("nlm", &plain),
        vec![Violation::EmptyField {
            at: "nlm".into(),
            field: "tool",
        }]
    );

    // The detector group too, and with a DIFFERENT `at` — which is what proves the reported
    // location is the directory under check rather than a constant.
    let mut detector = valid_detector_group();
    detector.tool = String::new();
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &detector),
        vec![Violation::EmptyField {
            at: "detector".into(),
            field: "tool",
        }]
    );

    // The other direction: a named tool must produce nothing. The controls carry real tool names
    // and are asserted clean by `the_valid_control_documents_have_no_violations`; this adds the
    // minimal non-empty case, so the rule cannot be "any tool shorter than the control's".
    let mut minimal = valid_group();
    minimal.tool = "x".into();
    assert_eq!(
        provenance::validate("nlm", &minimal),
        Vec::<Violation>::new()
    );
}

#[test]
// spec §16.24 item 3(a) — R4, the group-level `command_line`, the field that makes a recording
// re-runnable ("an oracle nobody can re-run is not an oracle", cookbook rule 3).
//
// Untested until now, and the reason it slipped is worth naming: `EmptyField { field:
// "command_line" }` IS asserted by `the_upstream_side_must_be_fully_pinned` — at
// `at: "detector.upstream"`, a different check (R19) on a different struct. Same variant, same
// field name, different rule. A grep for the variant said "covered"; the group-level site had
// nothing.
//
// The third case below is the sharp one: a DETECTOR group whose upstream command line is intact
// and whose group-level one is empty. Exactly one violation, at `at: "detector"`. Delete
// `provenance.rs:224-229` and this fails while every existing test stays green.
//
// Does NOT cover: that the recorded command line actually reproduces the artifact — the string is
// pinned as non-empty, never executed. Nothing here checks it names a real subcommand.
fn a_group_must_record_the_command_line_that_produced_it() {
    let mut plain = valid_group();
    plain.command_line = String::new();
    assert_eq!(
        provenance::validate("nlm", &plain),
        vec![Violation::EmptyField {
            at: "nlm".into(),
            field: "command_line",
        }]
    );

    let mut both_empty = valid_group();
    both_empty.tool = String::new();
    both_empty.command_line = String::new();
    assert_eq!(
        provenance::validate("nlm", &both_empty),
        vec![
            Violation::EmptyField {
                at: "nlm".into(),
                field: "command_line",
            },
            Violation::EmptyField {
                at: "nlm".into(),
                field: "tool",
            },
        ],
        "R3 and R4 are independent: neither may absorb the other"
    );

    // The group-level rule is not the upstream rule. This document's upstream side is fully
    // pinned, so the only violation is the group-level one.
    let mut detector = valid_detector_group();
    detector.command_line = String::new();
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &detector),
        vec![Violation::EmptyField {
            at: "detector".into(),
            field: "command_line",
        }]
    );

    let mut minimal = valid_group();
    minimal.command_line = "x".into();
    assert_eq!(
        provenance::validate("nlm", &minimal),
        Vec::<Violation>::new()
    );
}

#[test]
// spec §16.24 item 1(a)/(c) — R7, and it exists to protect the FROZEN walker, not this checker.
// `recorded_provenance.rs:107` matches on the KEY: a `source_sha256` with no `source` sibling is
// collected, the sibling lookup returns `None`, and `:112-119` **panics**. R7 is the rule that
// turns that panic into a named violation with the record's name in it.
//
// Untested until now — `Violation::DigestWithoutPath` had no test at all, in either direction.
//
// The second case pins that R7 and R6 are INDEPENDENT: an orphan digest that is also malformed
// must report both, or one rule would mask the other exactly the way `the_two_sides_...`'s
// `both_malformed` case guards against for R23/R6.
//
// Does NOT cover: the mirror case (`source` with no `source_sha256`) — that is LEGAL and asserted
// legal below, because §16.24 item 1(e) deliberately keeps upstream digests out of these files.
// It also does not re-verify the walker itself; `every_walker_visible_digest_has_its_sibling_path`
// owns that, over the committed files rather than a synthetic one.
fn a_source_digest_without_its_source_path_is_rejected() {
    let good = "c6cc1002c209ffadd182f2ebd3162b53b55bbb92f99ee038ebaf419bfed94b8d";

    let mut orphan = valid_group();
    orphan.records[0].source = None;
    orphan.records[0].source_sha256 = Some(good.into());
    assert_eq!(
        provenance::validate("nlm", &orphan),
        vec![Violation::DigestWithoutPath {
            at: "nightmare".into(),
            field: "source_sha256",
        }]
    );

    let mut orphan_and_malformed = valid_group();
    orphan_and_malformed.records[0].source = None;
    orphan_and_malformed.records[0].source_sha256 = Some("deadbeef".into());
    assert_eq!(
        provenance::validate("nlm", &orphan_and_malformed),
        vec![
            Violation::MalformedDigest {
                at: "nightmare".into(),
                field: "source_sha256",
                value: "deadbeef".into(),
            },
            Violation::DigestWithoutPath {
                at: "nightmare".into(),
                field: "source_sha256",
            },
        ],
        "R6 and R7 are independent: a malformed orphan digest must report both"
    );

    // Both legal shapes must stay clean, or R7 would fire on every real recording.
    let mut paired = valid_group();
    paired.records[0].source_sha256 = Some(good.into());
    assert_eq!(
        provenance::validate("nlm", &paired),
        Vec::<Violation>::new(),
        "a source WITH its digest is the shape R7 exists to permit"
    );

    let mut neither = valid_group();
    neither.records[0].source = None;
    neither.records[0].source_sha256 = None;
    assert_eq!(
        provenance::validate("nlm", &neither),
        Vec::<Violation>::new(),
        "§16.24 item 1(e): a record may name no source at all"
    );
}

#[test]
// spec §16.24 item 3(c) — R6 at the two digest slots no test reached. R6 is one rule applied at
// SIX slots, and coverage was per-slot, not per-rule:
//
//   already covered — `records[].output_sha256` (`a_non_canonical_digest_is_rejected`, four bad
//   spellings), `detector.model_digest` (`the_model_must_be_a_bare_filename_...`), and both
//   `decoded_rgb_digest`s (`the_two_sides_must_declare_the_same_decoded_rgb_buffer`'s
//   `both_malformed` case);
//   added here — `records[].source_sha256` and `detector.input_page_sha256`.
//
// This is cookbook rule 13 at the slot level: `a_non_canonical_digest_is_rejected` iterates four
// SPELLINGS of one slot, so a `validate_digest` call dropped from either slot below left it green.
// Both new slots are checked with the uppercase spelling, because uppercase is the one that
// VERIFIES at `recorded_provenance.rs:177` (it lowercases first) while making two provenance
// files textually incomparable.
//
// Does NOT cover: that any digest equals real bytes. R6 is shape only — `xtask/tests/
// provenance_digests.rs` owns the identities, and this file must not depend on `pc-models`.
fn r6_reaches_the_source_and_input_page_digest_slots_too() {
    let upper = "C6CC1002C209FFADD182F2EBD3162B53B55BBB92F99EE038EBAF419BFED94B8D";

    // `source` is present, so R7 does not fire and this isolates R6 at the source slot.
    let mut source = valid_group();
    source.records[0].source_sha256 = Some(upper.into());
    assert_eq!(
        provenance::validate("nlm", &source),
        vec![Violation::MalformedDigest {
            at: "nightmare".into(),
            field: "source_sha256",
            value: upper.into(),
        }]
    );

    let mut page = valid_detector_group();
    if let Some(pins) = page.detector.as_mut() {
        pins.input_page_sha256 = upper.into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &page),
        vec![Violation::MalformedDigest {
            at: "detector".into(),
            field: "input_page_sha256",
            value: upper.into(),
        }]
    );

    let mut truncated = valid_detector_group();
    if let Some(pins) = truncated.detector.as_mut() {
        pins.input_page_sha256 = "2".repeat(63);
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &truncated),
        vec![Violation::MalformedDigest {
            at: "detector".into(),
            field: "input_page_sha256",
            value: "2".repeat(63),
        }],
        "63 hex characters is the off-by-one a length check must catch"
    );
}

#[test]
// spec §16.20 item 3's closing ¶ — R8 at the two path slots no test reached, plus the
// `detector.input_page` locality check that shares its failure surface.
//
//   already covered — `records[].output` (`an_absolute_or_escaping_path_is_rejected`, absolute
//   and `..`);
//   added here — `records[].source` and `detector.input_page`.
//
// The `..` case for `input_page` is deliberately kept INSIDE the group prefix, so the locality
// check at `provenance.rs:324-336` cannot fire and R8 is isolated. The absolute case then asserts
// the exact PAIR, which is the property the comment at `provenance.rs:325-331` was written for:
// before it, an absolute misplaced page reported `NonRelativePath` twice and `dedup()` collapsed
// the two into one, making "absolute" and "wrong directory" indistinguishable. Two violations
// with two different `at` values is the observable form of that fix, and nothing asserted it.
//
// Does NOT cover: that any path exists on disk, or that `source` points at a vendored upstream
// asset rather than an arbitrary relative path. R8 is shape only.
fn r8_reaches_the_source_and_input_page_path_slots_too() {
    let absolute_source = "/home/maintainer/panel-ocr/tests/fixtures/upstream/nightmare.png";
    let mut source = valid_group();
    source.records[0].source = Some(absolute_source.into());
    assert_eq!(
        provenance::validate("nlm", &source),
        vec![Violation::NonRelativePath {
            at: "nightmare".into(),
            field: "source",
            path: absolute_source.into(),
        }]
    );

    let escaping_source = "tests/fixtures/upstream/../../../etc/passwd";
    let mut escaping = valid_group();
    escaping.records[0].source = Some(escaping_source.into());
    assert_eq!(
        provenance::validate("nlm", &escaping),
        vec![Violation::NonRelativePath {
            at: "nightmare".into(),
            field: "source",
            path: escaping_source.into(),
        }]
    );

    // Inside the group prefix, so locality is satisfied and R8 fires alone.
    let sneaky_page = "tests/fixtures/recorded/detector/../detector/page01.jpg";
    let mut inside = valid_detector_group();
    if let Some(pins) = inside.detector.as_mut() {
        pins.input_page = sneaky_page.into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &inside),
        vec![Violation::NonRelativePath {
            at: "detector".into(),
            field: "input_page",
            path: sneaky_page.into(),
        }]
    );

    // Absolute: BOTH rules fire, with distinct `at` values, and neither may be swallowed.
    let absolute_page = "/home/maintainer/panel-ocr/tests/fixtures/recorded/detector/page01.jpg";
    let mut outside = valid_detector_group();
    if let Some(pins) = outside.detector.as_mut() {
        pins.input_page = absolute_page.into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &outside),
        vec![
            Violation::NonRelativePath {
                at: "detector".into(),
                field: "input_page",
                path: absolute_page.into(),
            },
            Violation::CommittedPathOutsideGroup {
                at: "detector.input_page".into(),
                path: absolute_page.into(),
            },
        ],
        "an absolute page is BOTH non-relative and outside its group; `dedup()` must not \
         collapse the two"
    );

    // A relative page in the wrong group is misplaced only — not non-relative.
    let wrong_group = "tests/fixtures/recorded/nlm/page01.jpg";
    let mut misplaced = valid_detector_group();
    if let Some(pins) = misplaced.detector.as_mut() {
        pins.input_page = wrong_group.into();
    }
    assert_eq!(
        provenance::validate(provenance::DETECTOR_GROUP, &misplaced),
        vec![Violation::CommittedPathOutsideGroup {
            at: "detector.input_page".into(),
            path: wrong_group.into(),
        }]
    );
}
