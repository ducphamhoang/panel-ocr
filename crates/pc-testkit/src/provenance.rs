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
pub const REQUIRED_EXECUTION_PROVIDER: &str = "cpu";

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
    for record in &group.records {
        if !names.insert(record.name.clone()) {
            violations.push(Violation::DuplicateRecordName {
                name: record.name.clone(),
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
        violations.push(Violation::NonRelativePath {
            at: "detector".into(),
            field: "input_page",
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
    if parsed.is_absolute()
        || parsed
            .components()
            .any(|component| component == std::path::Component::ParentDir)
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
