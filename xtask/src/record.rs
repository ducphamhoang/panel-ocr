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
    let path = paths::recorded_root()
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
            let artifact = paths::workspace_root().join(&record.output);
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
    let group_dir = paths::recorded_root().join(group);
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
            .strip_prefix(paths::workspace_root())
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
