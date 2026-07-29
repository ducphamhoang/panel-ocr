//! `cargo xtask record-fixtures` — spec §7.2, task F1 (maintainer-run).
//!
//! Each recordable group is independent and independently skippable, so a partially
//! capable environment records what it can and reports exactly what it could not.

use crate::env::{detector_backend_status, PythonTooling, NO_PYTHON_HELP};
use crate::model_signature;
use crate::paths;
use anyhow::{bail, Context, Result};
use pc_testkit::provenance::{ArtifactRecord, GroupProvenance};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Group {
    /// §11.6/§11.7(B)12 — `cv2.fastNlMeansDenoising` references. Needs cv2.
    Nlm,
    /// §8.7(A)2 — the `cv2.INTER_AREA` downscale reference. Needs cv2.
    InterArea,
    /// §10.3 step 2 / §16.9 item 21 — PIL `FIND_EDGES` cross-check. Needs PIL.
    FindEdges,
    /// §7.2/§7.2.1 — detector-boundary + whole-page recordings. Needs ONNX, weights, and pages.
    Detector,
    /// §16.16 — declared graph metadata from the sha256-verified ONNX artifact. Needs weights.
    ModelSignature,
}

impl Group {
    pub const ALL: &'static [Group] = &[
        Group::Nlm,
        Group::InterArea,
        Group::FindEdges,
        Group::Detector,
        Group::ModelSignature,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Nlm => "nlm",
            Self::InterArea => "inter-area",
            Self::FindEdges => "find-edges",
            Self::Detector => "detector",
            Self::ModelSignature => "model-signature",
        }
    }
}

/// What actually happened for one group — printed as the run summary and reused by
/// `calibrate-goldens` when it explains a missing input.
#[derive(Debug)]
pub enum Outcome {
    Recorded { detail: String },
    Skipped { reason: String },
    Failed { reason: String },
}

/// §16.10 item 19: the two bubbles the NLM reference is recorded for.
pub const NLM_BUBBLES: &[&str] = &["nightmare", "ray"];

/// §8.7(A)2: `long_strip.jpg` 1000x8000 → 500x4000.
pub const INTER_AREA_TARGET: (u32, u32) = (500, 4000);
pub const INTER_AREA_REFERENCE: &str = "inter_area/long_strip_inter_area_500x4000.png";

pub fn run(
    groups: &[Group],
    python: Option<&Path>,
    detector: Option<&str>,
    model_signature_path: Option<&Path>,
    force: bool,
) -> Result<Vec<(Group, Outcome)>> {
    let needs_python = groups
        .iter()
        .any(|group| matches!(group, Group::Nlm | Group::InterArea | Group::FindEdges));
    let tooling = if needs_python {
        let tooling = PythonTooling::discover(python)?;
        match &tooling {
            Some(found) => println!("python tooling: {}", found.describe()),
            None => println!("python tooling: NOT FOUND"),
        }
        tooling
    } else {
        None
    };

    let mut results = Vec::new();
    for &group in groups {
        println!("\n── {} ─────────────────────────────", group.label());
        let attempt = match group {
            Group::Nlm => match &tooling {
                Some(tooling) => record_nlm(tooling, force),
                None => Ok(Outcome::Skipped {
                    reason: NO_PYTHON_HELP.into(),
                }),
            },
            Group::InterArea => match &tooling {
                Some(tooling) => record_inter_area(tooling, force),
                None => Ok(Outcome::Skipped {
                    reason: NO_PYTHON_HELP.into(),
                }),
            },
            Group::FindEdges => match &tooling {
                Some(tooling) => verify_find_edges(tooling),
                None => Ok(Outcome::Skipped {
                    reason: NO_PYTHON_HELP.into(),
                }),
            },
            Group::Detector => Ok(Outcome::Skipped {
                reason: detector_backend_status(detector)?.explain(),
            }),
            Group::ModelSignature => model_signature::record(model_signature_path, force),
        };
        let outcome = match attempt {
            Ok(outcome) => outcome,
            Err(error) => Outcome::Failed {
                reason: format!("{error:#}"),
            },
        };
        match &outcome {
            Outcome::Recorded { detail } => println!("recorded: {detail}"),
            Outcome::Skipped { reason } => println!("SKIPPED\n{reason}"),
            Outcome::Failed { reason } => println!("FAILED\n{reason}"),
        }
        results.push((group, outcome));
    }
    Ok(results)
}

/// Is this group's recorded state **complete** — its outputs *and* a `PROVENANCE.json` that parses
/// at the current schema version?
///
/// The skip-if-present checks below used to look only at the output artifacts, which reports
/// `Outcome::Recorded` for a group whose provenance is missing or on a pre-migration shape. That is
/// a **false success**: the maintainer ran the recorder precisely to regenerate that file, and was
/// told there was nothing to do. It matters more since §16.24 item 1 changed the schema — a
/// checkout carrying pre-migration provenance would be told "already present" and left stale,
/// surfacing later in `provenance_schema.rs` pointing at the file rather than at the recorder that
/// declined to write it.
///
/// Returning `false` on any read or parse failure is deliberate and is NOT the silent-path defect
/// fixed elsewhere in this crate: false means "not current", which causes the caller to do *more*
/// work (re-record and rewrite the provenance), never less. The conservative answer is the safe one
/// here, which is exactly why the polarity is worth stating.
///
/// **Four conditions, arrived at by three rounds of patching the reported case — which is itself
/// the lesson.** The first version compared only `schema_version`, so a provenance parsing at v1
/// while *misdescribing its own artifacts* satisfied the skip (the "actively misdescribes itself"
/// state `crates/pc-testkit/tests/model_signature.rs` exists to catch, reached here by the recorder
/// declining to fix it). The second added `validate()` and digest matching, but only in the
/// provenance→disk direction, so **unbound fixture state** — a stray or renamed artifact declared by
/// nothing and verified by nothing — still passed.
///
/// The conditions are therefore enumerated here rather than left implicit, because each round of
/// "fix the case that was reported" produced a helper that looked finished and was not:
///
/// 1. `PROVENANCE.json` reads and parses at `PROVENANCE_SCHEMA_VERSION`.
/// 2. `provenance::validate` returns no violations — every rule the checker owns, not one field.
/// 3. **provenance → disk:** every committed declaration's digest matches the bytes.
/// 4. **disk → provenance:** every file in the group directory is declared.
///
/// 3 and 4 are the same bidirectional rule `recorded_provenance.rs` needed five iterations to get
/// right (cookbook rule 13: cardinality is not identity; rule 14: each reader of a format needs the
/// whole invariant, not the convenient half). Having helped harden that gate and then written this
/// one half-way is the recurrence worth naming.
fn provenance_is_current(group: &str) -> bool {
    provenance_is_current_under(&paths::workspace_root(), group)
}

/// [`provenance_is_current`] against an explicit workspace root.
///
/// Extracted for exactly one reason: with the root hard-wired to `paths::workspace_root()`, the
/// only way to make any of the four conditions FAIL is to mutate the committed fixture tree — and
/// a leaked probe is precisely the corruption the recorded-fixture gates exist to detect (cookbook
/// rule 13's closing note). Every condition was therefore asserted only in its passing case, so
/// any one of them could be deleted with the suite still green. `provenance_is_current` is this
/// function at `paths::workspace_root()` and nothing else.
fn provenance_is_current_under(workspace_root: &Path, group: &str) -> bool {
    let recorded_root = workspace_root.join(pc_testkit::provenance::RECORDED_PREFIX);
    let path = recorded_root
        .join(group)
        .join(pc_testkit::provenance::PROVENANCE_FILE_NAME);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<GroupProvenance>(&text) else {
        return false;
    };

    // 1. The schema this binary writes.
    if parsed.schema_version != pc_testkit::provenance::PROVENANCE_SCHEMA_VERSION {
        return false;
    }
    // 2. Structurally valid — every rule the checker enforces, not just the version field.
    if !pc_testkit::provenance::validate(group, &parsed).is_empty() {
        return false;
    }
    // 3. PROVENANCE -> DISK. Every committed declaration matches the bytes on disk: an output
    //    edited or truncated after recording leaves a provenance that parses and validates while
    //    describing different bytes, and re-recording is exactly the fix.
    let declared_matches = parsed
        .records
        .iter()
        .filter(|record| record.committed)
        .all(|record| {
            let artifact = workspace_root.join(&record.output);
            pc_models::sha256_hex(&artifact)
                .is_ok_and(|actual| actual.eq_ignore_ascii_case(&record.output_sha256))
        });
    if !declared_matches {
        return false;
    }

    // 4. DISK -> PROVENANCE, the other direction. Every file in the group directory must be
    //    declared. Without this the skip accepts *unbound fixture state*: a stray, renamed or
    //    left-behind artifact sitting beside the declared ones, describing nothing and verified by
    //    nothing, while the recorder reports the group complete.
    //
    //    This is `recorded_provenance.rs`'s bidirectional coverage rule (its declared/committed
    //    path sets are asserted in BOTH directions) applied to the recorder. Checking only
    //    direction 3 is the same defect that gate took five iterations to remove — cookbook rule
    //    13's "cardinality is not identity", and rule 14's lesson that the readers of a format
    //    each need the whole invariant, not the half that was convenient.
    let group_dir = recorded_root.join(group);
    let declared: std::collections::BTreeSet<String> = parsed
        .records
        .iter()
        .filter(|record| record.committed)
        .map(|record| record.output.clone())
        .collect();
    let Ok(entries) = std::fs::read_dir(&group_dir) else {
        return false;
    };
    for entry in entries {
        let Ok(entry) = entry else { return false };
        let name = entry.file_name();
        // The provenance file describes the others and does not describe itself.
        if name == pc_testkit::provenance::PROVENANCE_FILE_NAME {
            continue;
        }
        let Ok(relative) = entry
            .path()
            .strip_prefix(workspace_root)
            .map(Path::to_path_buf)
        else {
            return false;
        };
        if !declared.contains(&relative.to_string_lossy().replace('\\', "/")) {
            return false;
        }
    }
    true
}

// ------------------------------------------------------------------ group: nlm

fn record_nlm(tooling: &PythonTooling, force: bool) -> Result<Outcome> {
    let out_dir = paths::recorded_root().join("nlm");
    if !force {
        let existing: Vec<_> = NLM_BUBBLES
            .iter()
            .map(|name| out_dir.join(nlm_reference_name(name)))
            .filter(|path| path.is_file())
            .collect();
        // Outputs AND current-schema provenance: a group is "already recorded" only when its
        // recorded state is complete, and the recorded state includes its provenance.
        if existing.len() == NLM_BUBBLES.len() && provenance_is_current("nlm") {
            return Ok(Outcome::Recorded {
                detail: format!(
                    "{} reference(s) and a current-schema PROVENANCE.json already present \
                     (use --force to re-record)",
                    existing.len()
                ),
            });
        }
    }
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let mut command = Command::new(&tooling.interpreter);
    command
        .arg(paths::script("record_nlm.py"))
        .arg(paths::upstream_root())
        .arg(&out_dir);
    for name in NLM_BUBBLES {
        command.arg(name);
    }
    let mut manifest = run_script(&mut command, "record_nlm.py")?;
    paths::relativize_manifest(&mut manifest);
    write_provenance(
        &out_dir.join("PROVENANCE.json"),
        &nlm_provenance(&manifest)?,
    )?;

    Ok(Outcome::Recorded {
        detail: format!(
            "{} cv2.fastNlMeansDenoising reference(s) in {}",
            NLM_BUBBLES.len(),
            paths::display_relative(&out_dir)
        ),
    })
}

pub fn nlm_reference_name(bubble: &str) -> String {
    // §11.6's literal name: `<name>_h10_t7_s21.png`.
    format!("{bubble}_h10_t7_s21.png")
}

// ----------------------------------------------------------- group: inter-area

fn record_inter_area(tooling: &PythonTooling, force: bool) -> Result<Outcome> {
    let reference = paths::recorded_root().join(INTER_AREA_REFERENCE);
    // Same completeness rule as `record_nlm`: the output alone is not the recorded state.
    if reference.is_file() && provenance_is_current("inter_area") && !force {
        return Ok(Outcome::Recorded {
            detail: format!(
                "{} and a current-schema PROVENANCE.json already present (use --force to re-record)",
                paths::display_relative(&reference)
            ),
        });
    }
    let dir = reference
        .parent()
        .expect("INTER_AREA_REFERENCE has a parent")
        .to_path_buf();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    // §16.13 item 5: decode the progressive JPEG once with the SAME decoder the Rust
    // test will use, and hand OpenCV that lossless PNG. Otherwise the recorded
    // reference would fold libjpeg-turbo-vs-`image` decode differences into a gate
    // that is supposed to measure the resize arithmetic alone. The re-decode is
    // reproducible from `long_strip.jpg` at any time, so it lives in scratch and is
    // NOT committed (§16.13 item 6).
    let scratch = paths::scratch_dir()?;
    let strip = paths::upstream_root().join("long_strip.jpg");
    let decoded = image::open(&strip)
        .with_context(|| format!("decoding {}", strip.display()))?
        .to_rgb8();
    let decoded_path = scratch.join("long_strip_decoded_rgb.png");
    decoded
        .save(&decoded_path)
        .with_context(|| format!("writing {}", decoded_path.display()))?;

    let mut manifest = run_script(
        Command::new(&tooling.interpreter)
            .arg(paths::script("record_inter_area.py"))
            .arg(&decoded_path)
            .arg(&reference)
            .arg(INTER_AREA_TARGET.0.to_string())
            .arg(INTER_AREA_TARGET.1.to_string()),
        "record_inter_area.py",
    )?;

    // Diagnostic pass: the same resize starting from OpenCV's own JPEG decode. Never
    // asserted against; it exists so GOLDEN_CALIBRATION.md can quantify how much of any
    // residual delta is decoder rather than resize. Only its *metrics* are kept — the
    // 1.5 MB image itself is scratch.
    let from_jpeg = scratch.join("long_strip_inter_area_500x4000_cv2jpeg.png");
    let mut diagnostic = run_script(
        Command::new(&tooling.interpreter)
            .arg(paths::script("record_inter_area.py"))
            .arg(&strip)
            .arg(&from_jpeg)
            .arg(INTER_AREA_TARGET.0.to_string())
            .arg(INTER_AREA_TARGET.1.to_string()),
        "record_inter_area.py (jpeg-decode diagnostic)",
    )?;
    let reference_image = image::open(&reference)
        .with_context(|| format!("decoding {}", reference.display()))?
        .to_rgb8();
    let from_jpeg_image = image::open(&from_jpeg)
        .with_context(|| format!("decoding {}", from_jpeg.display()))?
        .to_rgb8();
    diagnostic["metrics_vs_reference"] = serde_json::json!({
        "mean_abs_diff": pc_testkit::metrics::mean_abs_diff_rgb(&reference_image, &from_jpeg_image),
        "max_delta": pc_testkit::metrics::max_delta_rgb(&reference_image, &from_jpeg_image),
        "ssim_as_gray": pc_testkit::metrics::ssim_rgb_as_gray(&reference_image, &from_jpeg_image),
    });

    paths::relativize_manifest(&mut manifest);
    paths::relativize_manifest(&mut diagnostic);
    write_provenance(
        &dir.join("PROVENANCE.json"),
        &inter_area_provenance(&manifest, &diagnostic)?,
    )?;

    Ok(Outcome::Recorded {
        detail: format!(
            "{} ({}x{}) plus a decoder diagnostic",
            paths::display_relative(&reference),
            INTER_AREA_TARGET.0,
            INTER_AREA_TARGET.1
        ),
    })
}

// ----------------------------------------------------------- group: find-edges

fn verify_find_edges(tooling: &PythonTooling) -> Result<Outcome> {
    let manifest = run_script(
        Command::new(&tooling.interpreter).arg(paths::script("verify_find_edges.py")),
        "verify_find_edges.py",
    )?;
    let mismatches = manifest["mismatches"].as_u64().unwrap_or(u64::MAX);
    let full_3x3 = manifest["full_3x3_edge_count"].as_u64().unwrap_or(0);
    let checked = manifest["cases_checked"].as_u64().unwrap_or(0);
    if mismatches != 0 {
        bail!(
            "PIL FIND_EDGES disagrees with spec §10.3 step 2's closed form on {mismatches} \
             of {checked} cases — this contradicts a FROZEN test and must go back to the \
             two architects jointly (CLAUDE.md), not be patched here.\n{manifest:#}"
        );
    }
    Ok(Outcome::Recorded {
        detail: format!(
            "PIL {} agrees with §10.3 step 2 on all {checked} cases; \
             fully-set 3x3 mask yields {full_3x3} edges (§16.9 item 21 expects 8)",
            manifest["pillow_version"].as_str().unwrap_or("?"),
        ),
    })
}

// ----------------------------------------------------------------------- utils

fn run_script(command: &mut Command, label: &str) -> Result<serde_json::Value> {
    let output = command
        .output()
        .with_context(|| format!("spawning {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "{label} did not print a JSON manifest; stdout was:\n{}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn write_provenance(path: &Path, provenance: &GroupProvenance) -> Result<()> {
    let mut text = serde_json::to_string_pretty(provenance)?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

fn manifest_string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("recording manifest is missing string field `{key}`"))
}

fn manifest_params(value: &Value, keys: &[&str]) -> BTreeMap<String, Value> {
    keys.iter()
        .filter_map(|key| value.get(*key).map(|value| ((*key).into(), value.clone())))
        .collect()
}

fn manifest_versions(value: &Value) -> BTreeMap<String, String> {
    ["python", "opencv_version", "numpy_version"]
        .iter()
        .filter_map(|key| {
            manifest_string(value, key)
                .ok()
                .map(|version| (key.trim_end_matches("_version").into(), version.into()))
        })
        .collect()
}

fn nlm_provenance(manifest: &Value) -> Result<GroupProvenance> {
    let records = manifest
        .get("records")
        .and_then(Value::as_array)
        .context("NLM manifest has no records array")?
        .iter()
        .map(|record| {
            Ok(ArtifactRecord {
                name: manifest_string(record, "name")?.into(),
                output: manifest_string(record, "output")?.into(),
                output_sha256: manifest_string(record, "sha256")?.into(),
                committed: true,
                source: Some(manifest_string(record, "source")?.into()),
                source_sha256: None,
                params: manifest_params(record, &["size"]),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GroupProvenance {
        schema_version: pc_testkit::provenance::PROVENANCE_SCHEMA_VERSION,
        group: "nlm".into(),
        tool: manifest_string(manifest, "tool")?.into(),
        command_line: "cargo xtask record-fixtures --only nlm".into(),
        tool_versions: manifest_versions(manifest),
        records: records
            .into_iter()
            .map(|mut record| {
                record.params.extend([
                    ("h".into(), serde_json::json!(10)),
                    ("searchWindowSize".into(), serde_json::json!(21)),
                    ("templateWindowSize".into(), serde_json::json!(7)),
                ]);
                record
            })
            .collect(),
        detector: None,
        diagnostics: BTreeMap::new(),
    })
}

fn inter_area_provenance(reference: &Value, diagnostic: &Value) -> Result<GroupProvenance> {
    let make_record = |value: &Value, name: &str, committed: bool| -> Result<ArtifactRecord> {
        Ok(ArtifactRecord {
            name: name.into(),
            output: manifest_string(value, "output")?.into(),
            output_sha256: manifest_string(value, "sha256")?.into(),
            committed,
            source: Some(manifest_string(value, "source")?.into()),
            source_sha256: None,
            params: manifest_params(value, &["output_size", "source_size"]),
        })
    };
    let metrics = diagnostic
        .get("metrics_vs_reference")
        .cloned()
        .context("INTER_AREA diagnostic has no metrics_vs_reference")?;
    Ok(GroupProvenance {
        schema_version: pc_testkit::provenance::PROVENANCE_SCHEMA_VERSION,
        group: "inter_area".into(),
        tool: manifest_string(reference, "tool")?.into(),
        command_line: "cargo xtask record-fixtures --only inter-area".into(),
        tool_versions: manifest_versions(reference),
        records: vec![
            make_record(reference, "reference", true)?,
            make_record(diagnostic, "jpeg_decode_diagnostic", false)?,
        ],
        detector: None,
        diagnostics: BTreeMap::from([("metrics_vs_reference".into(), metrics)]),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// spec §16.13 item 6 + §16.24 item 1 — `provenance_is_current`'s four conditions, falsified.
//
// The helper took three rounds to get right and every round was accepted on the PASSING case
// alone: nothing anywhere asserted that a stale, invalid, tampered or unbound group reads as NOT
// current. Any one condition could be deleted and the suite stayed green. Each test below asserts
// BOTH polarities on the SAME tree — `true` before the perturbation and `false` after — so a
// helper that always answers `true` and one that always answers `false` both fail (cookbook
// rule 6: prove the check ran; rule 1: name the concrete wrong value).
//
// All of these are CI-safe: temp directory, `serde_json`, `sha2`. No model, no network, no Python,
// no committed fixture touched. §16.20 item 11's split puts comparing frozen JSON in CI and
// producing artifacts on the maintainer's machine, and these only compare.
#[cfg(test)]
mod provenance_is_current_falsification {
    use super::*;
    use pc_testkit::provenance::{
        ArtifactRecord, GroupProvenance, PROVENANCE_FILE_NAME, PROVENANCE_SCHEMA_VERSION,
        RECORDED_PREFIX,
    };
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    const GROUP: &str = "nlm";

    /// sha256 over bytes we hold in memory, via `sha2` directly — deliberately NOT
    /// `pc_models::sha256_hex`, which is the function condition 3 calls. Deriving the expected
    /// digest from the implementation under test is cookbook rule 1's `f(x) == f(x)`.
    fn digest(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    struct Tree {
        _dir: tempfile::TempDir,
        root: PathBuf,
    }

    impl Tree {
        fn group_dir(&self) -> PathBuf {
            self.root.join(RECORDED_PREFIX).join(GROUP)
        }
        fn provenance_path(&self) -> PathBuf {
            self.group_dir().join(PROVENANCE_FILE_NAME)
        }
        fn artifact(&self, bubble: &str) -> PathBuf {
            self.group_dir().join(nlm_reference_name(bubble))
        }
        fn read(&self) -> GroupProvenance {
            let text = std::fs::read_to_string(self.provenance_path()).expect("readable");
            serde_json::from_str(&text).expect("parses")
        }
        fn write(&self, provenance: &GroupProvenance) {
            write_provenance(&self.provenance_path(), provenance).expect("writable");
        }
        fn write_raw(&self, bytes: &[u8]) {
            std::fs::write(self.provenance_path(), bytes).expect("writable");
        }
        fn is_current(&self) -> bool {
            provenance_is_current_under(&self.root, GROUP)
        }
    }

    /// A complete `nlm` recorded state in a throwaway workspace: both `NLM_BUBBLES` artifacts and
    /// a provenance describing exactly them, in the shape `nlm_provenance` writes (§16.24 item
    /// 1(e): `source` named, `source_sha256` absent). Canonicalised so `strip_prefix` inside the
    /// helper matches textually on platforms where the temp root is a symlink.
    fn valid_tree() -> Tree {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical temp root");
        let group_dir = root.join(RECORDED_PREFIX).join(GROUP);
        std::fs::create_dir_all(&group_dir).expect("create group dir");

        let records = NLM_BUBBLES
            .iter()
            .map(|bubble| {
                let name = nlm_reference_name(bubble);
                let bytes = format!("not-really-a-png:{bubble}").into_bytes();
                std::fs::write(group_dir.join(&name), &bytes).expect("write artifact");
                ArtifactRecord {
                    name: (*bubble).into(),
                    output: format!("{RECORDED_PREFIX}/{GROUP}/{name}"),
                    output_sha256: digest(&bytes),
                    committed: true,
                    source: Some(format!(
                        "tests/fixtures/upstream/demo_bubbles/{bubble}_bubble_raw.png"
                    )),
                    source_sha256: None,
                    params: BTreeMap::from([
                        ("h".into(), serde_json::json!(10)),
                        ("searchWindowSize".into(), serde_json::json!(21)),
                        ("templateWindowSize".into(), serde_json::json!(7)),
                    ]),
                }
            })
            .collect();

        let tree = Tree { _dir: dir, root };
        tree.write(&GroupProvenance {
            schema_version: PROVENANCE_SCHEMA_VERSION,
            group: GROUP.into(),
            tool: "cv2.fastNlMeansDenoising".into(),
            command_line: "cargo xtask record-fixtures --only nlm".into(),
            tool_versions: BTreeMap::from([("opencv".into(), "5.0.0".into())]),
            records,
            detector: None,
            diagnostics: BTreeMap::new(),
        });
        tree
    }

    #[test]
    // CONDITION 1 — "reads and parses". The `let Ok(..) else { return false }` pair cannot be
    // *deleted* and still compile, so what this binds is the polarity and the absence of a panic:
    // an unreadable or unparsable provenance must answer `false` (re-record) rather than `true`
    // (skip) or an abort that kills the whole run. Replace either `else` arm with `true`, or
    // switch the parse to `unwrap_or_default()`, and this goes red.
    //
    // Case (c) is the one worth having: `deny_unknown_fields`. §16.24 item 1's migration turns a
    // pre-migration file into an unknown-field parse error, which is exactly the state the
    // maintainer ran the recorder to fix — and the state the first version of this helper reported
    // as "already present".
    //
    // Does NOT cover: an I/O failure that is not "absent" (permissions, a directory in place of
    // the file). Both take the same `Err` arm; neither is constructed here.
    fn an_unreadable_or_unparsable_provenance_is_not_current() {
        let tree = valid_tree();
        assert!(
            tree.is_current(),
            "baseline: a complete nlm tree must read as current, or every `false` below is vacuous"
        );
        let good = tree.read();

        std::fs::remove_file(tree.provenance_path()).expect("removable");
        assert!(!tree.is_current(), "(a) no PROVENANCE.json at all");

        tree.write_raw(b"{ this is not json");
        assert!(!tree.is_current(), "(b) present but not JSON");

        // (c) valid JSON, rejected by `deny_unknown_fields` — the pre-migration shape.
        let mut value = serde_json::to_value(&good).expect("serialisable");
        value["records"][0]["sha256"] = serde_json::json!("0".repeat(64));
        tree.write_raw(serde_json::to_string(&value).expect("text").as_bytes());
        assert!(
            !tree.is_current(),
            "(c) an unknown key must not parse: `deny_unknown_fields` is this reader's half of \
             §16.24 item 1"
        );

        // (d) valid JSON, a required field missing.
        let mut value = serde_json::to_value(&good).expect("serialisable");
        value
            .as_object_mut()
            .expect("object")
            .remove("records")
            .expect("records was present");
        tree.write_raw(serde_json::to_string(&value).expect("text").as_bytes());
        assert!(!tree.is_current(), "(d) a required field is missing");

        tree.write(&good);
        assert!(
            tree.is_current(),
            "restoring the file restores currency — every `false` above came from the \
             perturbation, not from a broken tree"
        );
    }

    #[test]
    // CONDITION 2 — "`schema_version` matches the schema this binary writes".
    //
    // **This test does NOT falsify the deletion of condition 2**, and saying so is the point:
    // `validate`'s R1 tests the identical predicate, so condition 3 subsumes condition 2 and the
    // boolean result is unchanged by removing it. What this test binds is the *behaviour* the
    // docstring promises — a provenance at any other version is not current — which fails only if
    // BOTH the version check and the `validate` call are removed. The docstring enumerates four
    // independent conditions; on the evidence here there are three and a fast path. That
    // redundancy is flagged, not resolved.
    //
    // Does NOT cover: a version that parses at v1 while meaning something else (no second version
    // exists yet, so nothing here can distinguish "rejects non-v1" from "rejects != the constant").
    fn a_provenance_at_another_schema_version_is_not_current() {
        let tree = valid_tree();
        assert!(tree.is_current(), "baseline");

        for version in [PROVENANCE_SCHEMA_VERSION + 1, 0] {
            let mut parsed = tree.read();
            parsed.schema_version = version;
            tree.write(&parsed);
            assert!(
                !tree.is_current(),
                "schema_version {version} is not the version this binary writes"
            );
        }
    }

    #[test]
    // CONDITION 3 — "`validate` returns no violations". Both perturbations are invisible to the
    // other three conditions: no digest changes, no filename changes, nothing on disk moves. So
    // deleting the `validate` call turns this red and nothing else.
    //
    // R2 (group name vs directory) and R3 (the group must name its tool) are chosen deliberately:
    // R3 is one of the rules that had no checker test either, and this is the second reader that
    // depends on it (cookbook rule 14 — every reader of the format needs the whole invariant).
    //
    // Does NOT cover: the other 21 rules individually. It proves the verdict is consulted, not
    // that every rule gates the recorder; `crates/pc-testkit/tests/provenance_schema.rs` owns
    // per-rule coverage.
    fn a_structurally_invalid_provenance_is_not_current() {
        let tree = valid_tree();
        assert!(tree.is_current(), "baseline");

        let mut wrong_group = tree.read();
        wrong_group.group = "inter_area".into();
        tree.write(&wrong_group);
        assert!(
            !tree.is_current(),
            "R2: a provenance describing another group is not this group's recorded state"
        );

        let mut no_tool = tree.read();
        no_tool.group = GROUP.into();
        no_tool.tool = String::new();
        tree.write(&no_tool);
        assert!(!tree.is_current(), "R3: the tool must be named");
    }

    #[test]
    // CONDITION 4, PROVENANCE → DISK — "every committed declaration's digest matches the bytes".
    //
    // Isolated from the disk → provenance half: in (a) and (b) the file keeps its name and stays
    // declared, so the reverse direction is satisfied throughout and only this half can see the
    // change. (c) is the sharper isolation — a declared file that is *gone* is invisible to a walk
    // over what IS on disk, so only this direction can report it.
    //
    // Does NOT cover: uncommitted (`committed: false`) declarations, which are scratch and
    // deliberately unverified (§16.13 item 6); or the digest of `source`, which §16.24 item 1(e)
    // keeps out of these files entirely.
    fn a_committed_declaration_whose_bytes_changed_is_not_current() {
        let tree = valid_tree();
        assert!(tree.is_current(), "baseline");
        let artifact = tree.artifact(NLM_BUBBLES[0]);
        let original = std::fs::read(&artifact).expect("readable artifact");

        let mut appended = original.clone();
        appended.push(b'!');
        std::fs::write(&artifact, &appended).expect("writable");
        assert!(
            !tree.is_current(),
            "(a) one appended byte: the provenance parses, validates and names the right file, \
             and describes different bytes"
        );

        std::fs::write(&artifact, &original).expect("writable");
        assert!(
            tree.is_current(),
            "restoring the bytes restores currency — the check is on content, not on mtime"
        );

        std::fs::write(&artifact, b"").expect("writable");
        assert!(!tree.is_current(), "(b) truncated to empty");

        std::fs::remove_file(&artifact).expect("removable");
        assert!(
            !tree.is_current(),
            "(c) a declared artifact missing from disk — only this direction enumerates \
             declarations, so only this direction can see it"
        );
    }

    #[test]
    // CONDITION 4, DISK → PROVENANCE — "every file in the group directory is declared". The
    // orphan probe is cookbook rule 13's own: copying a fixture to `nlm/undeclared_orphan.png`
    // passed the recorded-fixture gate for four iterations. Unbound fixture state is trusted by
    // every consuming golden test while being verified by nothing.
    //
    // Isolated from the provenance → disk half: no declaration changes and no declared file is
    // touched, so that half is satisfied throughout. Delete the `read_dir` loop and only this test
    // goes red.
    //
    // The exempt set is asserted rather than assumed (rule 13's corollary — "a broad exemption is
    // the bypass wearing different clothes"): the baseline passes with `PROVENANCE.json` present,
    // and an exemption-SHAPED name must still fail.
    //
    // Does NOT cover: a file in a nested subdirectory. The walk is non-recursive, which is safe
    // only because the directory entry itself must be declared and a directory cannot match a
    // committed digest — asserted below as the reason, not assumed.
    fn an_undeclared_file_in_the_group_directory_is_not_current() {
        let tree = valid_tree();
        assert!(tree.is_current(), "baseline");

        let orphan = tree.group_dir().join("undeclared_orphan.png");
        std::fs::write(&orphan, b"declared by nothing, verified by nothing").expect("writable");
        assert!(
            !tree.is_current(),
            "an undeclared file is unbound fixture state"
        );
        std::fs::remove_file(&orphan).expect("removable");
        assert!(tree.is_current(), "removing the orphan restores currency");

        let lookalike = tree.group_dir().join("PROVENANCE.json.bak");
        std::fs::write(&lookalike, b"{}").expect("writable");
        assert!(
            !tree.is_current(),
            "only the exact `PROVENANCE.json` name is exempt; a stray file must not pass by \
             merely looking exemption-shaped"
        );
        std::fs::remove_file(&lookalike).expect("removable");

        let stash = tree.group_dir().join("stash");
        std::fs::create_dir(&stash).expect("creatable");
        assert!(
            !tree.is_current(),
            "an undeclared subdirectory is unbound state too — this is why non-recursion is safe"
        );
    }

    #[test]
    // RESIDUAL HOLE 1 — **parameter drift is NOT detected.** A known gap, pinned rather than
    // closed.
    //
    // `params` is a claim about how the bytes were produced, and the four conditions never read
    // it: 1 and 2 look at the header, 3 sweeps `params` for digest shapes but never for agreement
    // with anything, and 4 compares bytes against a digest the same document supplies. So a
    // provenance may declare `h = 999` beside an artifact whose own filename says `_h10_`, and the
    // recorder reports the group complete.
    //
    // Today this is caught only INCIDENTALLY, and only for `nlm`/`inter_area`, because re-running
    // the tool with different parameters changes the FILENAME (`nlm_reference_name`) as well as
    // the bytes — so condition 4 fires for the wrong reason. The detector group does not have that
    // luck: its outputs are `page01_detector_blocks.json` regardless of `pad_value`, thresholds or
    // profile, and a re-record under different parameters lands on the same path.
    //
    // Not closed here, deliberately. Closing it needs a parameter fingerprint the schema does not
    // have — the filename encoding is incidental and group-specific, and a check keyed on it would
    // be exactly the "lucky" coverage that fails silently the first time a group does not encode
    // its parameters. Which mechanism to add is an architect decision, so this test states the
    // current behaviour and its cost.
    //
    // **If this test ever fails, the gap has been closed**: that is a behaviour change to route to
    // both architects (cookbook rule 8, exit 2), not a test to edit.
    //
    // Does NOT cover: params drift that also changes the bytes — condition 4 catches that, for the
    // bytes rather than for the parameters.
    fn parameter_drift_with_unchanged_bytes_is_not_detected_known_gap() {
        let tree = valid_tree();
        assert!(tree.is_current(), "baseline");

        let mut drifted = tree.read();
        assert!(
            drifted.records[0].output.contains("_h10_t7_s21"),
            "the artifact's own filename encodes h=10 — that is what `params` below contradicts"
        );
        drifted.records[0]
            .params
            .insert("h".into(), serde_json::json!(999));
        tree.write(&drifted);

        assert!(
            tree.is_current(),
            "KNOWN GAP (§16.13 item 6): `params` contradicts the filename beside it and the \
             group still reads as current. Nothing compares declared parameters to anything."
        );
    }

    #[test]
    // RESIDUAL HOLE 2 — **tool-version drift is NOT detected, and cannot be by this helper's
    // inputs.** The answer to "can `provenance_is_current` detect a re-record under a different
    // tool version?" is no, for a structural reason rather than a missing line: the helper takes a
    // group name, reads one file and hashes some bytes. It never asks the environment what version
    // of cv2/PIL/numpy is installed, so it has nothing to compare `tool_versions` against — even
    // though `run` has already discovered the interpreter (`PythonTooling::discover`) by the time
    // it is called.
    //
    // The consequence is the one that matters, and it is not hypothetical: on a machine whose cv2
    // WOULD produce different bytes, this helper still answers "current" — the artifacts on disk
    // match the digests recorded by the old version — so the recorder SKIPS and the drift is never
    // observed. Condition 4 can only compare bytes to the digest that shipped with them; it cannot
    // notice bytes that were never produced.
    //
    // The second half is sharper: `tool_versions` is `#[serde(default)]` and no rule requires it
    // non-empty for a non-detector group, so a provenance recording NO versions at all reads as
    // current. Contrast R21, which does require `dependency_versions` non-empty — but only on the
    // detector's upstream side.
    //
    // **If either assertion ever fails, the gap has been closed** — architects, not a test edit.
    //
    // Does NOT cover: the detector group's `dependency_versions`, whose non-emptiness R21 does
    // enforce; nor whether a version change would in fact change any bytes.
    fn tool_version_drift_is_not_detected_known_gap() {
        let tree = valid_tree();
        assert!(tree.is_current(), "baseline");

        let mut drifted = tree.read();
        drifted
            .tool_versions
            .insert("opencv".into(), "1.0.0-never-installed".into());
        tree.write(&drifted);
        assert!(
            tree.is_current(),
            "KNOWN GAP: a version that is not the one installed, and nothing compares the \
             recorded versions with the environment (cookbook rule 3: 'pin what you compared \
             against' — pinned, never checked)"
        );

        let mut absent = tree.read();
        absent.tool_versions.clear();
        tree.write(&absent);
        assert!(
            tree.is_current(),
            "KNOWN GAP: a non-detector group may record NO tool versions at all and still read \
             as current"
        );
    }
}
