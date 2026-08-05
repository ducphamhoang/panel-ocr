//! spec §16.17 item 2 (as amended by §16.24) — the shared `PROVENANCE.json` schema and its
//! checker.
//!
//! This checker is pure structure over already-parsed JSON. It never touches the filesystem
//! and never hashes anything; digest verification remains in the recording tests and xtask.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const PROVENANCE_SCHEMA_VERSION: u32 = 1;
pub const PROVENANCE_FILE_NAME: &str = "PROVENANCE.json";
pub const RECORDED_PREFIX: &str = "tests/fixtures/recorded";
pub const SCRATCH_PREFIX: &str = "target/xtask-scratch/";
pub const DETECTOR_GROUP: &str = "detector";
pub const REQUIRED_EXECUTION_PROVIDER: &str = pc_core::device::Device::Cpu.as_str();

/// Bare model filenames a record may name as `source` with no committed artifact to hash
/// (spec §16.31 item 3). ENUMERATED, not pattern-matched: a broad suffix exemption would
/// silently admit any future `*.onnx`/`*.pt.onnx` name. This is the single copy both
/// `crates/pc-testkit/tests/recorded_provenance.rs` (the enforcing predicate) and
/// `xtask/tests/provenance_digests.rs` (the compensating digest-binding gate) import — two
/// independently-typed-out copies with no comparator between them is exactly the failure
/// mode §16.31 item 2's ruling rejected for the pin constants, and it applies here too.
pub const BARE_MODEL_SOURCES: &[&str] = &[
    "comictextdetector.pt.onnx",
    "encoder_model.onnx",
    "decoder_model.onnx",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupProvenance {
    pub schema_version: u32,
    pub group: String,
    pub tool: String,
    pub command_line: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tool_versions: BTreeMap<String, String>,
    pub records: Vec<ArtifactRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector: Option<DetectorPins>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub diagnostics: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRecord {
    pub name: String,
    pub output: String,
    pub output_sha256: String,
    pub committed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorPins {
    pub input_page: String,
    pub input_page_sha256: String,
    pub model: String,
    pub model_digest: String,
    pub ours: OursPins,
    pub upstream: UpstreamPins,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OursPins {
    pub backend: Backend,
    pub decoded_rgb_digest: String,
    pub decoded_from: String,
    pub execution_provider: String,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub pad_value: u8,
    pub panel_ocr_commit: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub profile_non_default: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamPins {
    pub backend: Backend,
    pub decoded_rgb_digest: String,
    pub decoded_from: String,
    pub version: String,
    pub commit: String,
    pub command_line: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub profile_non_default: BTreeMap<String, Value>,
    pub dependency_versions: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Ort,
    Cv2Dnn,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Violation {
    SchemaVersion {
        found: u32,
    },
    GroupNameMismatch {
        declared: String,
        directory: String,
    },
    EmptyField {
        at: String,
        field: &'static str,
    },
    NoRecords,
    DuplicateRecordName {
        name: String,
    },
    /// Duplicate output paths within one provenance group. Cross-group duplicates remain the
    /// frozen whole-tree walker's responsibility; `validate` only sees one group.
    DuplicateRecordOutput {
        path: String,
    },
    MalformedDigest {
        at: String,
        field: &'static str,
        value: String,
    },
    MalformedCommit {
        at: String,
        field: &'static str,
        value: String,
    },
    DigestWithoutPath {
        at: String,
        field: &'static str,
    },
    NonRelativePath {
        at: String,
        field: &'static str,
        path: String,
    },
    CommittedPathOutsideGroup {
        at: String,
        path: String,
    },
    UncommittedPathNotScratch {
        at: String,
        path: String,
    },
    DigestLikeKeyOutsideStructuralSlot {
        at: String,
        key: String,
    },
    DigestLikeValueOutsideStructuralSlot {
        at: String,
        key: String,
    },
    MissingDetectorBlock {
        group: String,
    },
    UnexpectedDetectorBlock {
        group: String,
    },
    InputPageAlsoDeclaredAsRecordOutput {
        record: String,
        path: String,
    },
    DecodedRgbDigestMismatch {
        ours: String,
        upstream: String,
    },
    UnknownDecodedFromReference {
        at: String,
        name: String,
    },
    ModelIsNotABareFilename {
        model: String,
    },
    BackendContradictsSide {
        at: String,
        expected: Backend,
        found: Backend,
    },
    ExecutionProviderNotCpu {
        found: String,
    },
    EmptyDependencyVersions,
}

pub fn is_canonical_digest(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn is_canonical_commit(text: &str) -> bool {
    text.len() == 40
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn looks_like_digest_key(key: &str) -> bool {
    let normalised = key.to_ascii_lowercase().replace(['-', '_'], "");
    normalised.ends_with("sha256") || normalised.ends_with("digest") || normalised.ends_with("hash")
}

/// Validator rule register. Each line names the check implemented below; the numbering is kept
/// here so readers do not have to reconstruct it from the scattered test and implementation notes.
///
/// R1 — schema version is the current supported version.
/// R2 — the declared group name matches the directory name supplied by the caller.
/// R3 — the group names the tool that produced it.
/// R4 — the group records the command line that produced it.
/// R5 — the group has records with unique names and unique output paths.
/// R6 — every structural digest is a canonical lowercase 64-hex value.
/// R7 — a `source_sha256` digest is not present without its `source` path.
/// R8 — recorded paths are relative and contain no parent-directory component.
/// R9 — committed record outputs stay inside the declaring recorded group.
/// R10 — uncommitted record outputs stay under the scratch prefix.
/// R11 — digest-like keys do not occur outside structural digest slots.
/// R12 — digest-shaped values do not occur outside structural digest slots.
/// R13 — the detector block is present exactly for the `detector` group.
/// R14 — the detector input page is inside the detector recorded group.
/// R15 — the detector input page is not also declared as a record output.
/// R16 — the detector model is a bare filename.
/// R17 — each `decoded_from` reference names `input_page` or a declared record.
/// R18 — our-side pins use the ORT backend.
/// R19 — upstream-side pins identify the cv2.dnn run, version, commit, and command line.
/// R20 — our-side provenance records the panel-ocr commit and resolved execution settings.
/// R21 — upstream dependency versions are present.
/// R22 — the execution provider is CPU and both thread counts are recorded.
/// R23 — both detector sides declare the same decoded RGB buffer digest.
///
/// The rule register records the existing checks only; it does not add validation behavior.
///
/// Validate the structure and invariants of a parsed provenance document.
///
/// The result is sorted and deduplicated. This function deliberately does not inspect files or
/// verify digest contents: those operations belong to the outer recording gates.
pub fn validate(directory_name: &str, group: &GroupProvenance) -> Vec<Violation> {
    let mut violations = Vec::new();

    if group.schema_version != PROVENANCE_SCHEMA_VERSION {
        violations.push(Violation::SchemaVersion {
            found: group.schema_version,
        });
    }
    if group.group != directory_name {
        violations.push(Violation::GroupNameMismatch {
            declared: group.group.clone(),
            directory: directory_name.to_owned(),
        });
    }
    if group.tool.is_empty() {
        violations.push(Violation::EmptyField {
            at: directory_name.to_owned(),
            field: "tool",
        });
    }
    if group.command_line.is_empty() {
        violations.push(Violation::EmptyField {
            at: directory_name.to_owned(),
            field: "command_line",
        });
    }
    if group.records.is_empty() {
        violations.push(Violation::NoRecords);
    }

    let mut names = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    for record in &group.records {
        if !names.insert(record.name.clone()) {
            violations.push(Violation::DuplicateRecordName {
                name: record.name.clone(),
            });
        }
        if !outputs.insert(record.output.clone()) {
            violations.push(Violation::DuplicateRecordOutput {
                path: record.output.clone(),
            });
        }
        validate_digest(
            &mut violations,
            &record.name,
            "output_sha256",
            &record.output_sha256,
        );
        validate_path(&mut violations, &record.name, "output", &record.output);
        if record.committed {
            let prefix = format!("{RECORDED_PREFIX}/{directory_name}/");
            if !record.output.starts_with(&prefix) {
                violations.push(Violation::CommittedPathOutsideGroup {
                    at: record.name.clone(),
                    path: record.output.clone(),
                });
            }
        } else if !record.output.starts_with(SCRATCH_PREFIX) {
            violations.push(Violation::UncommittedPathNotScratch {
                at: record.name.clone(),
                path: record.output.clone(),
            });
        }

        if let Some(source) = &record.source {
            validate_path(&mut violations, &record.name, "source", source);
        }
        if let Some(source_sha256) = &record.source_sha256 {
            validate_digest(
                &mut violations,
                &record.name,
                "source_sha256",
                source_sha256,
            );
            if record.source.is_none() {
                violations.push(Violation::DigestWithoutPath {
                    at: record.name.clone(),
                    field: "source_sha256",
                });
            }
        }
        validate_freeform(&record.params, &record.name, &mut violations);
    }

    let detector_expected = directory_name == DETECTOR_GROUP;
    match (detector_expected, group.detector.as_ref()) {
        (true, None) => violations.push(Violation::MissingDetectorBlock {
            group: directory_name.to_owned(),
        }),
        (false, Some(_)) => violations.push(Violation::UnexpectedDetectorBlock {
            group: directory_name.to_owned(),
        }),
        (true, Some(detector)) => validate_detector(group, detector, &mut violations),
        (false, None) => {}
    }
    validate_freeform(&group.diagnostics, "diagnostics", &mut violations);
    // `tool_versions` is `BTreeMap<String, String>` — see the note in `validate_upstream` on
    // `dependency_versions`: the `String` type stops a nested object, not a digest-shaped value.
    for (key, value) in &group.tool_versions {
        if is_canonical_digest(value) {
            violations.push(Violation::DigestLikeValueOutsideStructuralSlot {
                at: "tool_versions".into(),
                key: key.clone(),
            });
        }
        if looks_like_digest_key(key) {
            violations.push(Violation::DigestLikeKeyOutsideStructuralSlot {
                at: "tool_versions".into(),
                key: key.clone(),
            });
        }
    }

    violations.sort();
    violations.dedup();
    violations
}

fn validate_detector(
    group: &GroupProvenance,
    detector: &DetectorPins,
    violations: &mut Vec<Violation>,
) {
    validate_path(violations, "detector", "input_page", &detector.input_page);
    let expected_prefix = format!("{RECORDED_PREFIX}/{DETECTOR_GROUP}/");
    if !detector.input_page.starts_with(&expected_prefix) {
        // `CommittedPathOutsideGroup`, NOT `NonRelativePath`: a relative path in the wrong
        // directory is misplaced, not non-relative, and saying otherwise is a false message
        // (cookbook rule 1's corollary). It also used to collapse under `dedup()` with
        // `validate_path`'s genuine violation above when the path was absolute, making
        // "absolute" and "wrong directory" indistinguishable in the output. §16.24 item 2 itself
        // only requires the page be declared exactly once and not double-listed as a record
        // output (§16.29 item 2 corrects an earlier over-citation here); the lives-inside-the-
        // group requirement actually flows from item 1's preserved frozen walker plus item 2's
        // walker-visibility mandate, and `:251` already reports exactly this failure class for
        // record outputs.
        violations.push(Violation::CommittedPathOutsideGroup {
            at: "detector.input_page".into(),
            path: detector.input_page.clone(),
        });
    }
    validate_digest(
        violations,
        "detector",
        "input_page_sha256",
        &detector.input_page_sha256,
    );
    for record in &group.records {
        if record.output == detector.input_page {
            violations.push(Violation::InputPageAlsoDeclaredAsRecordOutput {
                record: record.name.clone(),
                path: record.output.clone(),
            });
        }
    }

    if Path::new(&detector.model).components().count() != 1 || detector.model.is_empty() {
        violations.push(Violation::ModelIsNotABareFilename {
            model: detector.model.clone(),
        });
    }
    validate_digest(
        violations,
        "detector",
        "model_digest",
        &detector.model_digest,
    );

    validate_ours(violations, &detector.ours);
    validate_upstream(violations, &detector.upstream);
    if detector.ours.decoded_rgb_digest != detector.upstream.decoded_rgb_digest {
        violations.push(Violation::DecodedRgbDigestMismatch {
            ours: detector.ours.decoded_rgb_digest.clone(),
            upstream: detector.upstream.decoded_rgb_digest.clone(),
        });
    }

    let record_names: BTreeSet<&str> = group
        .records
        .iter()
        .map(|record| record.name.as_str())
        .collect();
    for (at, decoded_from) in [
        ("detector.ours", &detector.ours.decoded_from),
        ("detector.upstream", &detector.upstream.decoded_from),
    ] {
        if decoded_from != "input_page" && !record_names.contains(decoded_from.as_str()) {
            violations.push(Violation::UnknownDecodedFromReference {
                at: at.into(),
                name: decoded_from.clone(),
            });
        }
    }
}

fn validate_ours(violations: &mut Vec<Violation>, ours: &OursPins) {
    validate_digest(
        violations,
        "detector.ours",
        "decoded_rgb_digest",
        &ours.decoded_rgb_digest,
    );
    if ours.backend != Backend::Ort {
        violations.push(Violation::BackendContradictsSide {
            at: "detector.ours".into(),
            expected: Backend::Ort,
            found: ours.backend,
        });
    }
    if ours.execution_provider != REQUIRED_EXECUTION_PROVIDER {
        violations.push(Violation::ExecutionProviderNotCpu {
            found: ours.execution_provider.clone(),
        });
    }
    if !is_canonical_commit(&ours.panel_ocr_commit) {
        violations.push(Violation::MalformedCommit {
            at: "detector.ours".into(),
            field: "panel_ocr_commit",
            value: ours.panel_ocr_commit.clone(),
        });
    }
    // R11/R12 reach EVERY free-form map, not just `params`/`diagnostics`. `profile_non_default`
    // is `BTreeMap<String, Value>` built by diffing against the profile defaults (§16.24 item
    // 3(a)), so it accepts arbitrary JSON and is exactly the shape a digest can hide in. The
    // sweep's whole purpose is that the typed layer enumerates *declarations* while the sweep
    // enumerates *digests present in the file*, and the two populations must match; a map the
    // sweep does not visit breaks that invariant silently.
    validate_freeform(
        &ours.profile_non_default,
        "detector.ours.profile_non_default",
        violations,
    );
}

fn validate_upstream(violations: &mut Vec<Violation>, upstream: &UpstreamPins) {
    validate_digest(
        violations,
        "detector.upstream",
        "decoded_rgb_digest",
        &upstream.decoded_rgb_digest,
    );
    if upstream.backend != Backend::Cv2Dnn {
        violations.push(Violation::BackendContradictsSide {
            at: "detector.upstream".into(),
            expected: Backend::Cv2Dnn,
            found: upstream.backend,
        });
    }
    if upstream.command_line.is_empty() {
        violations.push(Violation::EmptyField {
            at: "detector.upstream".into(),
            field: "command_line",
        });
    }
    if upstream.version.is_empty() {
        violations.push(Violation::EmptyField {
            at: "detector.upstream".into(),
            field: "version",
        });
    }
    if !is_canonical_commit(&upstream.commit) {
        violations.push(Violation::MalformedCommit {
            at: "detector.upstream".into(),
            field: "commit",
            value: upstream.commit.clone(),
        });
    }
    if upstream.dependency_versions.is_empty() {
        violations.push(Violation::EmptyDependencyVersions);
    }
    // Same reasoning as `validate_ours`: the upstream side's `profile_non_default` is free-form
    // and must be swept. This side matters more, not less — it records a profile produced by a
    // *third-party* tool run, so its contents are the least predictable in the document.
    validate_freeform(
        &upstream.profile_non_default,
        "detector.upstream.profile_non_default",
        violations,
    );
    // `dependency_versions` and `tool_versions` are `BTreeMap<String, String>`, so a nested
    // object cannot hide in them, but a digest-shaped *value* still can: `{"opencv": "<64 hex>"}`
    // parses fine and would be a digest present in the file and absent from the verified set.
    // R12 applies by value shape, so check it here rather than assuming the `String` type is a
    // safeguard — it constrains the shape, not the content.
    for (key, value) in &upstream.dependency_versions {
        if is_canonical_digest(value) {
            violations.push(Violation::DigestLikeValueOutsideStructuralSlot {
                at: "detector.upstream.dependency_versions".into(),
                key: key.clone(),
            });
        }
        if looks_like_digest_key(key) {
            violations.push(Violation::DigestLikeKeyOutsideStructuralSlot {
                at: "detector.upstream.dependency_versions".into(),
                key: key.clone(),
            });
        }
    }
}

fn validate_digest(violations: &mut Vec<Violation>, at: &str, field: &'static str, value: &str) {
    if !is_canonical_digest(value) {
        violations.push(Violation::MalformedDigest {
            at: at.to_owned(),
            field,
            value: value.to_owned(),
        });
    }
}

fn validate_path(violations: &mut Vec<Violation>, at: &str, field: &'static str, path: &str) {
    let parsed = Path::new(path);
    // Provenance paths are serialized repository paths, not host-native paths. On Windows,
    // Path::new("/home/...").is_absolute() is false for a root-relative path, so inspect the
    // portable slash spelling as well as the native parser.
    let portable = path.replace('\\', "/");
    let has_drive_root =
        portable.len() >= 3 && portable.as_bytes()[1] == b':' && portable.as_bytes()[2] == b'/';
    if parsed.is_absolute()
        || portable.starts_with('/')
        || has_drive_root
        || portable.split('/').any(|component| component == "..")
    {
        violations.push(Violation::NonRelativePath {
            at: at.to_owned(),
            field,
            path: path.to_owned(),
        });
    }
}

fn validate_freeform(value: &BTreeMap<String, Value>, at: &str, violations: &mut Vec<Violation>) {
    fn walk_value(value: &Value, at: &str, key: Option<&str>, violations: &mut Vec<Violation>) {
        match value {
            Value::Object(map) => {
                for (child_key, child) in map {
                    if looks_like_digest_key(child_key) {
                        violations.push(Violation::DigestLikeKeyOutsideStructuralSlot {
                            at: at.to_owned(),
                            key: child_key.clone(),
                        });
                    }
                    walk_value(child, at, Some(child_key), violations);
                }
            }
            Value::Array(items) => {
                for item in items {
                    walk_value(item, at, key, violations);
                }
            }
            Value::String(text) if key.is_some() && is_canonical_digest(text) => {
                violations.push(Violation::DigestLikeValueOutsideStructuralSlot {
                    at: at.to_owned(),
                    key: key.expect("key checked").to_owned(),
                });
            }
            _ => {}
        }
    }

    let object = value
        .iter()
        .map(|(key, child)| (key.clone(), child.clone()))
        .collect();
    walk_value(&Value::Object(object), at, None, violations);
}
