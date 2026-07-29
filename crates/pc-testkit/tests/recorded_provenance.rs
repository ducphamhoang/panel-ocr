//! Recorded-fixture provenance gates (spec §16.13 item 6).
//!
//! The declarations, rather than a hand-maintained list of artifacts, are the coverage
//! set. Every digest declaration is either checked against a committed file or rejected
//! unless it has one of the known deliberately-unverifiable forms.

use pc_testkit::paths;
use serde_json::Value;
use std::path::{Path, PathBuf};

const EXPECTED_GROUPS: usize = 3;
const RECORDED_PREFIX: &str = "tests/fixtures/recorded";
const SCRATCH_PREFIX: &str = "target/xtask-scratch/";
const EXPECTED_DECLARED_PATHS: &[&str] = &[
    "tests/fixtures/recorded/nlm/nightmare_h10_t7_s21.png",
    "tests/fixtures/recorded/nlm/ray_h10_t7_s21.png",
    "tests/fixtures/recorded/inter_area/long_strip_inter_area_500x4000.png",
    "tests/fixtures/recorded/model_signature/comictextdetector.signature.json",
    "target/xtask-scratch/long_strip_inter_area_500x4000_cv2jpeg.png",
    "comictextdetector.pt.onnx",
];
const EXPECTED_COMMITTED_PATHS: &[&str] = &[
    "tests/fixtures/recorded/nlm/nightmare_h10_t7_s21.png",
    "tests/fixtures/recorded/nlm/ray_h10_t7_s21.png",
    "tests/fixtures/recorded/inter_area/long_strip_inter_area_500x4000.png",
    "tests/fixtures/recorded/model_signature/comictextdetector.signature.json",
];

#[derive(Debug)]
struct DigestDeclaration {
    path: String,
    sha256: String,
    declaring_group: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_provenance(group: &Path) -> Value {
    // Keep this lookup panicking: a recorded group without parseable provenance is a
    // broken checkout, not a reason to skip the coverage gate.
    let path = paths::recorded(group.join("PROVENANCE.json"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read provenance `{}`: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("failed to parse provenance `{}`: {error}", path.display()))
}

fn discover_groups() -> Vec<PathBuf> {
    let root = paths::recorded_root();
    let mut groups = std::fs::read_dir(&root)
        .unwrap_or_else(|error| {
            panic!(
                "failed to read recorded fixture root `{}`: {error}",
                root.display()
            )
        })
        .filter_map(|entry| {
            let entry = entry.unwrap_or_else(|error| {
                panic!(
                    "failed to read an entry under `{}`: {error}",
                    root.display()
                )
            });
            let file_type = entry.file_type().unwrap_or_else(|error| {
                panic!(
                    "failed to inspect recorded entry `{}`: {error}",
                    entry.path().display()
                )
            });
            file_type.is_dir().then(|| entry.file_name().into())
        })
        .collect::<Vec<PathBuf>>();

    groups.sort();
    assert_eq!(
        groups.len(),
        EXPECTED_GROUPS,
        "recorded fixture group count changed; inspect this test and update its expected count"
    );
    groups
}

fn path_key_for_digest(digest_key: &str) -> String {
    match digest_key {
        "sha256" | "output_sha256" => "output".to_owned(),
        "source_sha256" => "source".to_owned(),
        other => other.strip_suffix("_sha256").unwrap_or(other).to_owned(),
    }
}

fn collect_digest_declarations(
    value: &Value,
    location: &str,
    declaring_group: &str,
    declarations: &mut Vec<DigestDeclaration>,
) {
    match value {
        Value::Object(object) => {
            for (key, digest) in object {
                if key != "sha256" && !key.ends_with("_sha256") {
                    continue;
                }

                let path_key = path_key_for_digest(key);
                let declared_path = object
                    .get(&path_key)
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| {
                        panic!(
                            "digest declaration `{key}` at {location} has no resolvable path sibling `{path_key}`"
                        )
                    });
                let sha256 = digest.as_str().unwrap_or_else(|| {
                    panic!("digest declaration `{key}` at {location} must be a string")
                });
                declarations.push(DigestDeclaration {
                    path: declared_path.to_owned(),
                    sha256: sha256.to_owned(),
                    declaring_group: declaring_group.to_owned(),
                });
            }

            for (key, child) in object {
                collect_digest_declarations(
                    child,
                    &format!("{location}.{key}"),
                    declaring_group,
                    declarations,
                );
            }
        }
        Value::Array(array) => {
            for (index, child) in array.iter().enumerate() {
                collect_digest_declarations(
                    child,
                    &format!("{location}[{index}]"),
                    declaring_group,
                    declarations,
                );
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn assert_committed_artifact_locality(declaration: &DigestDeclaration, relative: &Path) {
    let actual_group = relative
        .components()
        .next()
        .and_then(|component| component.as_os_str().to_str())
        .unwrap_or("<recorded-root>");
    assert!(
        actual_group == declaration.declaring_group,
        "committed artifact `{}` was declared by group `{}` but lives in group `{actual_group}`",
        declaration.path,
        declaration.declaring_group,
    );
}

fn verify_committed_artifact(declaration: &DigestDeclaration, relative: &Path) {
    let path = paths::recorded(relative);
    let actual_sha256 = sha256_hex(&std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "failed to read declared artifact `{}`: {error}",
            path.display()
        )
    }));
    assert_eq!(
        actual_sha256,
        declaration.sha256.to_ascii_lowercase(),
        "SHA-256 mismatch for declared artifact `{}`",
        declaration.path
    );
    println!("verified committed artifact: {}", declaration.path);
}

fn assert_known_non_committed_form(declared_path: &str) {
    let path = Path::new(declared_path);
    let is_generated_scratch = declared_path.starts_with(SCRATCH_PREFIX);
    let is_bare_model_filename = path.components().count() == 1
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".pt.onnx"));

    assert!(
        is_generated_scratch || is_bare_model_filename,
        "declared non-committed artifact `{declared_path}` is not a known generated scratch path or bare model filename"
    );
    if is_generated_scratch {
        println!(
            "deliberately unverifiable generated artifact: {}",
            declared_path
        );
    } else {
        println!(
            "deliberately unverifiable bare model artifact: {}",
            declared_path
        );
    }
}

fn sorted_expected_paths(expected_paths: &[&str]) -> Vec<String> {
    let mut paths = expected_paths
        .iter()
        .map(|path| (*path).to_owned())
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn path_set_difference(paths: &[String], other: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| !other.contains(path))
        .cloned()
        .collect()
}

fn collect_recorded_files(directory: &Path, relative: &Path, files: &mut Vec<String>) {
    let entries = std::fs::read_dir(directory).unwrap_or_else(|error| {
        panic!(
            "failed to read recorded fixture directory `{}`: {error}",
            directory.display()
        )
    });

    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "failed to read an entry under `{}`: {error}",
                directory.display()
            )
        });
        let entry_path = entry.path();
        let entry_relative = relative.join(entry.file_name());
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!(
                "failed to inspect recorded entry `{}`: {error}",
                entry_path.display()
            )
        });

        if file_type.is_dir() {
            collect_recorded_files(&entry_path, &entry_relative, files);
        } else {
            files.push(
                Path::new(RECORDED_PREFIX)
                    .join(entry_relative)
                    .display()
                    .to_string(),
            );
        }
    }
}

fn is_exempt_candidate(path: &str) -> bool {
    path == format!("{RECORDED_PREFIX}/.gitkeep")
        || Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "PROVENANCE.json" || name == ".gitkeep")
}

#[test]
fn recorded_provenance_declared_digests_are_covered() {
    let groups = discover_groups();
    let mut declarations = Vec::new();

    for group in &groups {
        let provenance = read_provenance(group);
        let group_name = group
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| panic!("recorded fixture group has no directory name: {group:?}"));
        collect_digest_declarations(
            &provenance,
            &group.join("PROVENANCE.json").display().to_string(),
            group_name,
            &mut declarations,
        );
    }

    let mut declared_paths = declarations
        .iter()
        .map(|declaration| declaration.path.clone())
        .collect::<Vec<_>>();
    declared_paths.sort();
    let expected_declared_paths = sorted_expected_paths(EXPECTED_DECLARED_PATHS);
    let duplicate_paths = declared_paths
        .windows(2)
        .filter(|pair| pair[0] == pair[1])
        .map(|pair| pair[0].clone())
        .collect::<Vec<_>>();
    let unexpected_declared_paths = path_set_difference(&declared_paths, &expected_declared_paths);
    let missing_declared_paths = path_set_difference(&expected_declared_paths, &declared_paths);
    assert!(
        duplicate_paths.is_empty(),
        "duplicate declared path(s): {duplicate_paths:?}; declared path set mismatch; unexpected paths: {unexpected_declared_paths:?}; missing paths: {missing_declared_paths:?}"
    );
    assert_eq!(
        declared_paths, expected_declared_paths,
        "declared path set mismatch; unexpected paths: {unexpected_declared_paths:?}; missing paths: {missing_declared_paths:?}"
    );

    let mut committed_paths = Vec::new();
    for declaration in &declarations {
        if let Ok(relative) = Path::new(&declaration.path).strip_prefix(RECORDED_PREFIX) {
            assert_committed_artifact_locality(declaration, relative);
            verify_committed_artifact(declaration, relative);
            committed_paths.push(declaration.path.clone());
        } else {
            assert_known_non_committed_form(&declaration.path);
        }
    }

    committed_paths.sort();
    let expected_committed_paths = sorted_expected_paths(EXPECTED_COMMITTED_PATHS);
    let unexpected_committed_paths =
        path_set_difference(&committed_paths, &expected_committed_paths);
    let missing_committed_paths = path_set_difference(&expected_committed_paths, &committed_paths);
    assert_eq!(
        committed_paths, expected_committed_paths,
        "committed path set mismatch; unexpected paths: {unexpected_committed_paths:?}; missing paths: {missing_committed_paths:?}"
    );

    let mut recorded_files = Vec::new();
    collect_recorded_files(&paths::recorded_root(), Path::new(""), &mut recorded_files);
    recorded_files.sort();

    let mut exempt_paths = recorded_files
        .iter()
        .filter(|path| is_exempt_candidate(path))
        .cloned()
        .collect::<Vec<_>>();
    exempt_paths.sort();
    let mut expected_exempt_paths = groups
        .iter()
        .map(|group| {
            Path::new(RECORDED_PREFIX)
                .join(group.file_name().expect("recorded group must have a name"))
                .join("PROVENANCE.json")
                .display()
                .to_string()
        })
        .collect::<Vec<_>>();
    expected_exempt_paths.push(format!("{RECORDED_PREFIX}/.gitkeep"));
    expected_exempt_paths.sort();
    let unexpected_exempt_paths = path_set_difference(&exempt_paths, &expected_exempt_paths);
    let missing_exempt_paths = path_set_difference(&expected_exempt_paths, &exempt_paths);
    assert_eq!(
        exempt_paths, expected_exempt_paths,
        "exempt path set mismatch; unexpected paths: {unexpected_exempt_paths:?}; missing paths: {missing_exempt_paths:?}"
    );

    let committed_files = recorded_files
        .into_iter()
        .filter(|path| !is_exempt_candidate(path))
        .collect::<Vec<_>>();
    let unexpected_committed_files = path_set_difference(&committed_files, &committed_paths);
    let missing_committed_files = path_set_difference(&committed_paths, &committed_files);
    assert_eq!(
        committed_files, committed_paths,
        "committed fixture file set mismatch; unexpected files: {unexpected_committed_files:?}; missing declared files: {missing_committed_files:?}"
    );
}
