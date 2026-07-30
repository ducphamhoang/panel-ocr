//! `cargo xtask record-fixtures` — spec §7.2, task F1 (maintainer-run).
//!
//! Each recordable group is independent and independently skippable, so a partially
//! capable environment records what it can and reports exactly what it could not.

use crate::env::{detector_backend_status, DetectorStatus, PythonTooling, NO_PYTHON_HELP};
use crate::model_signature;
use crate::paths;
use anyhow::{bail, Context, Result};
use pc_testkit::provenance::{
    ArtifactRecord, DetectorPins, GroupProvenance, OursPins, UpstreamPins,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const DETECTOR_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";
const DETECTOR_PAGE_SOURCE: &str = "oracle_pages/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg";
const DETECTOR_EXECUTION_PROVIDER: &str = "cpu";
const DETECTOR_PAGE_RECORDED: &str =
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg";

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
    detector_upstream: Option<&Path>,
    model_signature_path: Option<&Path>,
    force: bool,
) -> Result<Vec<(Group, Outcome)>> {
    let needs_python = groups.iter().any(|group| {
        matches!(group, Group::Nlm | Group::InterArea | Group::FindEdges)
            || (cfg!(feature = "onnx") && matches!(group, Group::Detector))
    });
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
            Group::Detector => match detector_backend_status(detector)? {
                DetectorStatus::Ready { model_path } => match &tooling {
                    Some(tooling) => {
                        record_detector_dispatch(tooling, detector_upstream, &model_path, force)
                    }
                    None => Ok(Outcome::Skipped {
                        reason: NO_PYTHON_HELP.into(),
                    }),
                },
                status => Ok(Outcome::Skipped {
                    reason: status.explain(),
                }),
            },
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
/// **Three conditions, arrived at by three rounds of patching the reported case — which is itself
/// the lesson.** The first version compared only `schema_version`, so a provenance parsing at v1
/// while *misdescribing its own artifacts* satisfied the skip (the "actively misdescribes itself"
/// state `crates/pc-testkit/tests/model_signature.rs` exists to catch, reached here by the recorder
/// declining to fix it). The second added `validate()` and digest matching, but only in the
/// provenance→disk direction, so **unbound fixture state** — a stray or renamed artifact declared by
/// nothing and verified by nothing — still passed.
///
/// The checks are therefore enumerated here rather than left implicit, because each round of
/// "fix the case that was reported" produced a helper that looked finished and was not:
///
/// 1. `PROVENANCE.json` reads and parses, and `provenance::validate` returns no violations —
///    including the current schema version and every other rule the checker owns.
/// 2. **provenance → disk:** every committed declaration's digest matches the bytes.
/// 3. **disk → provenance:** every file in the group directory is declared.
///
/// 2 and 3 are the same bidirectional rule `recorded_provenance.rs` needed five iterations to get
/// right (cookbook rule 13: cardinality is not identity; rule 14: each reader of a format needs the
/// whole invariant, not the convenient half). Having helped harden that gate and then written this
/// one half-way is the recurrence worth naming.
fn provenance_is_current(group: &str) -> bool {
    provenance_is_current_under(&paths::workspace_root(), group)
}

/// [`provenance_is_current`] against an explicit workspace root.
///
/// Extracted for exactly one reason: with the root hard-wired to `paths::workspace_root()`, the
/// only way to make any of the three conditions FAIL is to mutate the committed fixture tree — and
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

    // 1. Structurally valid — every rule the checker enforces, including the schema version.
    if !pc_testkit::provenance::validate(group, &parsed).is_empty() {
        return false;
    }
    // 2. PROVENANCE -> DISK. Every committed declaration matches the bytes on disk: an output
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

    // 3. DISK -> PROVENANCE, the other direction. Every file in the group directory must be
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
    let mut declared: std::collections::BTreeSet<String> = parsed
        .records
        .iter()
        .filter(|record| record.committed)
        .map(|record| record.output.clone())
        .collect();
    // The detector group's input page is declared via `detector.input_page`/`input_page_sha256`,
    // not the generic `records` list (§16.24 item 2 forbids double-declaring it as a record
    // output too) — but it is still a real, legitimately-declared file inside this group
    // directory, and condition 3 must know that or the detector group can never be recognized
    // as current: the page would look like an undeclared orphan on every single check.
    if let Some(detector) = &parsed.detector {
        declared.insert(detector.input_page.clone());
    }
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

// ------------------------------------------------------------- group: detector

/// The detector group has one fixed ratified page for v1. The source remains under the
/// upstream fixture tree until the later atomic fixture commit moves the byte-identical page
/// into this group's directory (§16.29 item 2).
pub fn detector_plan(out_dir: &Path, stem: &str) -> std::collections::BTreeSet<PathBuf> {
    [
        out_dir.join(format!("{stem}_detector_mask.png")),
        out_dir.join(format!("{stem}_detector_blocks.json")),
        out_dir.join(format!("{stem}_base.png")),
        out_dir.join(format!("{stem}_raw_mask.png")),
        out_dir.join(format!("{stem}#raw.json")),
        out_dir.join(format!("{stem}_upstream_oracle.json")),
        out_dir.join(format!("{stem}_upstream_group_output_equality.json")),
        out_dir.join(pc_testkit::provenance::PROVENANCE_FILE_NAME),
        // §16.29 item 2: the committed oracle page itself, copied (not just referenced) into
        // the recorded group so `detector.input_page` resolves to a real file here.
        out_dir.join(
            Path::new(DETECTOR_PAGE_RECORDED)
                .file_name()
                .expect("DETECTOR_PAGE_RECORDED names a file"),
        ),
    ]
    .into_iter()
    .collect()
}

fn record_detector_dispatch(
    tooling: &PythonTooling,
    upstream_checkout: Option<&Path>,
    model: &Path,
    force: bool,
) -> Result<Outcome> {
    #[cfg(feature = "onnx")]
    {
        record_detector(tooling, upstream_checkout, model, force)
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (tooling, upstream_checkout, model, force);
        bail!("detector recording requires xtask's `onnx` feature")
    }
}

/// Verify the model before invoking the supplied inference closure. Keeping this seam pure in
/// its ordering makes the dangerous failure mode directly testable without ONNX or a model.
fn with_verified_model<T>(model: &Path, inference: impl FnOnce() -> Result<T>) -> Result<T> {
    pc_models::verify_sha256(model, pc_models::COMIC_TEXT_DETECTOR.sha256)
        .with_context(|| format!("verifying detector model {}", model.display()))?;
    inference()
}

fn ensure_cpu_execution_provider(execution_provider: &str) -> Result<()> {
    if execution_provider != DETECTOR_EXECUTION_PROVIDER {
        bail!(
            "detector recording refuses execution provider `{execution_provider}`; only `cpu` is ratified (§16.22 item 5(b))"
        );
    }
    Ok(())
}

/// Build the canonical detector provenance from a fully normalised manifest. This function does
/// no I/O, so its shape and validator controls can run in the default CI tier without ONNX.
fn provenance_for(manifest: &Value) -> Result<GroupProvenance> {
    let records = manifest
        .get("records")
        .and_then(Value::as_array)
        .context("detector manifest has no records array")?
        .iter()
        .map(|record| {
            Ok(ArtifactRecord {
                name: manifest_string(record, "name")?.into(),
                output: manifest_string(record, "output")?.into(),
                output_sha256: record
                    .get("output_sha256")
                    .or_else(|| record.get("sha256"))
                    .and_then(Value::as_str)
                    .context("detector record has no sha256")?
                    .into(),
                committed: record
                    .get("committed")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                source: None,
                source_sha256: None,
                params: record
                    .get("params")
                    .and_then(Value::as_object)
                    .map(|object| object.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let ours: OursPins = serde_json::from_value(
        manifest
            .get("ours")
            .cloned()
            .context("detector manifest has no ours pins")?,
    )
    .context("malformed detector ours pins")?;
    let upstream: UpstreamPins = serde_json::from_value(
        manifest
            .get("upstream")
            .cloned()
            .context("detector manifest has no upstream pins")?,
    )
    .context("malformed detector upstream pins")?;

    Ok(GroupProvenance {
        schema_version: pc_testkit::provenance::PROVENANCE_SCHEMA_VERSION,
        group: "detector".into(),
        tool: manifest_string(manifest, "tool")?.into(),
        command_line: manifest_string(manifest, "xtask_command_line")?.into(),
        tool_versions: manifest_versions(manifest),
        records,
        detector: Some(DetectorPins {
            input_page: manifest_string(manifest, "input_page")?.into(),
            input_page_sha256: manifest_string(manifest, "input_page_sha256")?.into(),
            model: manifest_string(manifest, "model")?.into(),
            model_digest: manifest_string(manifest, "model_digest")?.into(),
            ours,
            upstream,
        }),
        diagnostics: manifest
            .get("diagnostics")
            .and_then(Value::as_object)
            .map(|object| object.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default(),
    })
}

#[cfg(feature = "onnx")]
fn record_detector(
    tooling: &PythonTooling,
    upstream_checkout: Option<&Path>,
    model: &Path,
    force: bool,
) -> Result<Outcome> {
    let out_dir = paths::recorded_root().join("detector");
    let planned = detector_plan(&out_dir, DETECTOR_STEM);
    if !force
        && planned
            .iter()
            .filter(|path| {
                path.file_name().and_then(|name| name.to_str()) != Some("PROVENANCE.json")
            })
            .all(|path| path.is_file())
        && provenance_is_current("detector")
    {
        return Ok(Outcome::Recorded {
            detail: format!(
                "{} detector artifacts and a current-schema PROVENANCE.json already present (use --force to re-record)",
                planned.len() - 1
            ),
        });
    }

    let Some(upstream_checkout) = upstream_checkout else {
        return Ok(Outcome::Skipped {
            reason: "detector recording needs --detector-upstream PATH pointing at the pinned PanelCleaner checkout; no fixture was emitted.".into(),
        });
    };
    let page = paths::upstream_root().join(DETECTOR_PAGE_SOURCE);
    if !page.is_file() {
        bail!("detector input page is missing: {}", page.display());
    }
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    // §16.29 item 2: `detector.input_page` must resolve to a file inside this recorded group
    // (the shipped provenance validator forces this, and the frozen whole-tree walker later
    // hashes whatever `input_page` names), so the page is copied here, byte-preserving, rather
    // than only referenced by path. This is a plain file copy, not a git operation — nothing is
    // staged or committed by this recorder; the eventual `git add` of this directory, including
    // this copy, is the separate atomic-commit step §16.24 item 6 gates.
    let recorded_page = out_dir.join(
        Path::new(DETECTOR_PAGE_RECORDED)
            .file_name()
            .expect("DETECTOR_PAGE_RECORDED names a file"),
    );
    std::fs::copy(&page, &recorded_page)
        .with_context(|| format!("copying {} to {}", page.display(), recorded_page.display()))?;

    // §16.24 item 14: this is the first operation that can lead to either detector running.
    // The upstream script repeats the same check before importing/invoking cv2.dnn.
    with_verified_model(model, || {
        ensure_cpu_execution_provider(DETECTOR_EXECUTION_PROVIDER)?;

        let config = pc_config::TextDetectorConfig::default();
        let detector = pc_detect::onnx::OnnxDetector::from_path_with_config(model, &config)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let decoded = image::open(&page)
            .with_context(|| format!("decoding {}", page.display()))?
            .to_rgb8();
        let decoded_rgb_digest = sha256_bytes(decoded.as_raw());
        let (new_width, new_height, scale) =
            pc_detect::calculate_new_size_and_scale(decoded.width(), decoded.height(), 1000, 4000);
        let base = pc_detect::resize_area(&decoded, new_width, new_height);

        // Record the detector boundary directly, then prove the serialized replay consumer
        // sees the same pre-filter block list before using it for the whole-page run.
        let direct = pc_detect::TextDetector::detect(&detector, &base)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        pc_detect::write_replay_fixture(&out_dir, DETECTOR_STEM, &direct.mask, &direct.blocks);
        let replay = pc_detect::ReplayDetector::new(&out_dir, DETECTOR_STEM);
        let replayed = pc_detect::TextDetector::detect(&replay, &base)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if direct.blocks != replayed.blocks {
            bail!("direct ONNX and ReplayDetector pre-filter block lists differ");
        }

        let input = pc_detect::DetectInput {
            schema_version: pc_core::SCHEMA_VERSION,
            source: pc_core::ImageHandle::from_path(&page),
            original_path: PathBuf::from(DETECTOR_PAGE_RECORDED),
            target_height_lower: 1000,
            target_height_upper: 4000,
            base_image_dest: Some(out_dir.join(format!("{DETECTOR_STEM}_base.png"))),
            raw_mask_dest: Some(out_dir.join(format!("{DETECTOR_STEM}_raw_mask.png"))),
            min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
            config,
        };
        let output =
            pc_detect::run(input, &replay).map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if output.page.scale != 1.0 || scale != 1.0 {
            bail!(
                "detector recording requires scale == 1.0 on both sides; ours={} upstream-input={scale}",
                output.page.scale
            );
        }
        let mut page_json = output.page.clone();
        let residue = pc_testkit::paths::relativize_page_data_raw(
            &mut page_json,
            &pc_testkit::paths::fixtures_root(),
        );
        if !residue.is_empty() {
            bail!("PageDataRaw contains paths outside fixtures root: {residue:?}");
        }
        let raw_path = out_dir.join(format!("{DETECTOR_STEM}#raw.json"));
        let mut raw_bytes = serde_json::to_vec_pretty(&page_json)?;
        raw_bytes.push(b'\n');
        std::fs::write(&raw_path, raw_bytes)
            .with_context(|| format!("writing {}", raw_path.display()))?;

        let mut upstream_manifest = run_script(
            {
                let decoded_page =
                    paths::scratch_dir()?.join(format!("{DETECTOR_STEM}_decoded_rgb.png"));
                image::DynamicImage::ImageRgb8(decoded.clone())
                    .save(&decoded_page)
                    .with_context(|| format!("writing {}", decoded_page.display()))?;
                Command::new(&tooling.interpreter)
                    .arg(paths::script("record_detector_oracle.py"))
                    .arg(upstream_checkout)
                    .arg(model)
                    .arg(&decoded_page)
                    .arg(&out_dir)
                    .arg(DETECTOR_STEM)
            },
            "record_detector_oracle.py",
        )?;
        let oracle_path = out_dir.join(format!("{DETECTOR_STEM}_upstream_oracle.json"));
        let oracle: Value = serde_json::from_slice(
            &std::fs::read(&oracle_path)
                .with_context(|| format!("reading {}", oracle_path.display()))?,
        )
        .context("parsing upstream detector oracle")?;
        if oracle.get("scale").and_then(Value::as_f64) != Some(1.0) {
            bail!("upstream detector oracle scale is not 1.0");
        }
        let actual_page_digest = pc_models::sha256_hex(&page)?;
        let script_page_digest = pc_models::sha256_hex(
            &paths::scratch_dir()?.join(format!("{DETECTOR_STEM}_decoded_rgb.png")),
        )?;
        if manifest_string(&upstream_manifest, "model_digest")?
            != pc_models::COMIC_TEXT_DETECTOR.sha256
            || manifest_string(&upstream_manifest, "input_page_sha256")? != script_page_digest
        {
            bail!("upstream manifest identity does not match the verified model/decoded page");
        }
        if manifest_string(&upstream_manifest["upstream"], "decoded_rgb_digest")?
            != decoded_rgb_digest
        {
            bail!("ours and upstream decoded RGB digests differ");
        }

        let model_digest = pc_models::COMIC_TEXT_DETECTOR.sha256.to_owned();
        let panel_ocr_commit = current_commit()?;
        let mut records = Vec::new();
        for (name, record_name) in [
            (
                format!("{DETECTOR_STEM}_detector_mask.png"),
                "detector_mask",
            ),
            (
                format!("{DETECTOR_STEM}_detector_blocks.json"),
                "detector_blocks",
            ),
            (format!("{DETECTOR_STEM}_base.png"), "base"),
            (format!("{DETECTOR_STEM}_raw_mask.png"), "raw_mask"),
            (format!("{DETECTOR_STEM}#raw.json"), "raw_page"),
            (
                format!("{DETECTOR_STEM}_upstream_oracle.json"),
                "upstream_oracle",
            ),
            (
                format!("{DETECTOR_STEM}_upstream_group_output_equality.json"),
                "group_output_equality",
            ),
        ] {
            let path = out_dir.join(&name);
            records.push(serde_json::json!({
                "name": record_name,
                "output": paths::display_relative(&path),
                "output_sha256": pc_models::sha256_hex(&path)?,
                "committed": true,
            }));
        }
        let ours = serde_json::json!({
            "backend": "ort",
            "decoded_rgb_digest": decoded_rgb_digest,
            "decoded_from": "input_page",
            "execution_provider": DETECTOR_EXECUTION_PROVIDER,
            "intra_threads": 0,
            "inter_threads": 0,
            "pad_value": pc_detect::onnx::PAD_VALUE,
            "panel_ocr_commit": panel_ocr_commit,
            "profile_non_default": {},
        });
        upstream_manifest["records"] = Value::Array(records);
        upstream_manifest["input_page"] = Value::String(DETECTOR_PAGE_RECORDED.into());
        upstream_manifest["model"] = Value::String(pc_models::COMIC_TEXT_DETECTOR.file_name.into());
        upstream_manifest["model_digest"] = Value::String(model_digest);
        upstream_manifest["input_page_sha256"] = Value::String(actual_page_digest);
        upstream_manifest["ours"] = ours;
        upstream_manifest["xtask_command_line"] =
            Value::String("cargo xtask record-fixtures --only detector".into());
        let provenance = provenance_for(&upstream_manifest)?;
        let violations = pc_testkit::provenance::validate("detector", &provenance);
        if !violations.is_empty() {
            bail!("detector provenance failed validation: {violations:?}");
        }
        write_provenance(
            &out_dir.join(pc_testkit::provenance::PROVENANCE_FILE_NAME),
            &provenance,
        )?;

        Ok(Outcome::Recorded {
            detail: format!(
                "{} detector artifacts in {}",
                records_len(&provenance),
                paths::display_relative(&out_dir)
            ),
        })
    })
}

fn records_len(provenance: &GroupProvenance) -> usize {
    provenance.records.len()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn current_commit() -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .context("reading panel-ocr commit")?;
    if !output.status.success() {
        bail!("git rev-parse HEAD failed: {}", output.status);
    }
    Ok(String::from_utf8(output.stdout)
        .context("git rev-parse HEAD returned non-UTF-8")?
        .trim()
        .to_owned())
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
    ["python", "opencv_version", "numpy_version", "torch_version"]
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
// spec §16.13 item 6 + §16.24 item 1 — `provenance_is_current`'s three conditions, falsified.
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
    // The schema-version part of CONDITION 1 — "`schema_version` matches the schema this binary
    // writes".
    //
    // **This test does NOT falsify the deletion of condition 2**, and saying so is the point:
    // `validate`'s R1 tests the identical predicate, so the explicit version check was redundant.
    // What this test binds is the *behaviour* the docstring promises — a provenance at any other
    // version is not current — which fails if schema validation is removed.
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
    // The remaining structural part of CONDITION 1 — "`validate` returns no violations". Both
    // perturbations are invisible to the other two conditions: no digest changes, no filename
    // changes, nothing on disk moves. So
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
    // CONDITION 2, PROVENANCE → DISK — "every committed declaration's digest matches the bytes".
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
    // CONDITION 3, DISK → PROVENANCE — "every file in the group directory is declared". The
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
    // A real bug, caught by the stop-time review gate on the FIRST end-to-end run of the detector
    // recorder, not by any test: condition 3's `declared` set was built only from `records`, but
    // §16.24 item 2 forbids double-declaring `detector.input_page` as a record output too — it is
    // declared exactly once, via `detector.input_page`/`input_page_sha256`. So the committed page
    // sitting in the group directory looked like an undeclared orphan on EVERY check, and the
    // detector group could never be recognized as current: every invocation re-ran the full
    // recording (real ONNX inference, the Python subprocess) even when nothing had changed.
    //
    // This constructs a detector-shaped group directly (not via the nlm-only `Tree` helper above)
    // with one `records` entry AND a page file declared only through `detector.input_page`, and
    // asserts the whole tree is current. Revert the `declared.insert(detector.input_page...)` line
    // in `provenance_is_current_under` and this goes red.
    fn a_detector_groups_input_page_declared_only_via_detector_input_page_is_still_current() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical temp root");
        let group_dir = root.join(RECORDED_PREFIX).join("detector");
        std::fs::create_dir_all(&group_dir).expect("create group dir");

        let raw_page_bytes = b"not-really-a-page-json".to_vec();
        std::fs::write(group_dir.join("page#raw.json"), &raw_page_bytes).expect("writable");
        let page_bytes = b"not-really-a-jpeg".to_vec();
        std::fs::write(group_dir.join("page.jpg"), &page_bytes).expect("writable");

        let provenance = GroupProvenance {
            schema_version: PROVENANCE_SCHEMA_VERSION,
            group: "detector".into(),
            tool: "PanelCleaner detector oracle recorder".into(),
            command_line: "cargo xtask record-fixtures --only detector".into(),
            tool_versions: BTreeMap::new(),
            records: vec![ArtifactRecord {
                name: "raw_page".into(),
                output: format!("{RECORDED_PREFIX}/detector/page#raw.json"),
                output_sha256: digest(&raw_page_bytes),
                committed: true,
                source: None,
                source_sha256: None,
                params: BTreeMap::new(),
            }],
            detector: Some(pc_testkit::provenance::DetectorPins {
                input_page: format!("{RECORDED_PREFIX}/detector/page.jpg"),
                input_page_sha256: digest(&page_bytes),
                model: "comictextdetector.pt.onnx".into(),
                model_digest: digest(b"model"),
                ours: pc_testkit::provenance::OursPins {
                    backend: pc_testkit::provenance::Backend::Ort,
                    decoded_rgb_digest: digest(b"decoded"),
                    decoded_from: "input_page".into(),
                    execution_provider: "cpu".into(),
                    intra_threads: 0,
                    inter_threads: 0,
                    pad_value: 0,
                    panel_ocr_commit: "b".repeat(40),
                    profile_non_default: BTreeMap::new(),
                },
                upstream: pc_testkit::provenance::UpstreamPins {
                    backend: pc_testkit::provenance::Backend::Cv2Dnn,
                    decoded_rgb_digest: digest(b"decoded"),
                    decoded_from: "input_page".into(),
                    version: "2.11.11".into(),
                    commit: "b".repeat(40),
                    command_line: "record_detector_oracle.py ...".into(),
                    profile_non_default: BTreeMap::new(),
                    dependency_versions: BTreeMap::from([("opencv".into(), "5.0.0".into())]),
                },
            }),
            diagnostics: BTreeMap::new(),
        };
        assert!(
            pc_testkit::provenance::validate("detector", &provenance).is_empty(),
            "the constructed provenance itself must be structurally valid, or this test would \
             pass for the wrong reason (condition 1 failing, not condition 3 passing)"
        );
        write_provenance(&group_dir.join(PROVENANCE_FILE_NAME), &provenance).expect("writable");

        assert!(
            provenance_is_current_under(&root, "detector"),
            "the input page, declared only via `detector.input_page`, must not read as an \
             undeclared orphan"
        );
    }

    #[test]
    // RESIDUAL HOLE 1 — **parameter drift is NOT detected.** A known gap, pinned rather than
    // closed.
    //
    // `params` is a claim about how the bytes were produced, and the three conditions never read
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

#[cfg(test)]
mod detector_recording_tests {
    use super::{detector_plan, provenance_for, with_verified_model, DETECTOR_STEM};
    use pc_testkit::provenance::{Backend, Violation};
    use serde_json::json;
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const COMMIT: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn manifest() -> serde_json::Value {
        json!({
            "tool": "PanelCleaner detector oracle recorder",
            "xtask_command_line": "cargo xtask record-fixtures --only detector",
            "python": "3.12.3",
            "opencv_version": "5.0.0",
            "numpy_version": "2.5.1",
            "torch_version": "2.13.0+cpu",
            "input_page": "tests/fixtures/recorded/detector/page.jpg",
            "input_page_sha256": DIGEST,
            "model": "comictextdetector.pt.onnx",
            "model_digest": DIGEST,
            "records": [{
                "name": "raw_page",
                "output": "tests/fixtures/recorded/detector/page#raw.json",
                "output_sha256": DIGEST,
                "committed": true
            }],
            "ours": {
                "backend": "ort",
                "decoded_rgb_digest": DIGEST,
                "decoded_from": "input_page",
                "execution_provider": "cpu",
                "intra_threads": 0,
                "inter_threads": 0,
                "pad_value": 0,
                "panel_ocr_commit": COMMIT,
                "profile_non_default": {}
            },
            "upstream": {
                "backend": "cv2_dnn",
                "decoded_rgb_digest": DIGEST,
                "decoded_from": "input_page",
                "version": "2.11.11",
                "commit": COMMIT,
                "command_line": "record_detector_oracle.py ...",
                "profile_non_default": {},
                "dependency_versions": {"opencv": "5.0.0"}
            },
            "diagnostics": {}
        })
    }

    #[test]
    fn plan_is_the_literal_detector_artifact_set() {
        let root = PathBuf::from("/tmp/detector-recording");
        let expected = BTreeSet::from([
            root.join("page_detector_mask.png"),
            root.join("page_detector_blocks.json"),
            root.join("page_base.png"),
            root.join("page_raw_mask.png"),
            root.join("page#raw.json"),
            root.join("page_upstream_oracle.json"),
            root.join("page_upstream_group_output_equality.json"),
            root.join("PROVENANCE.json"),
            // §16.29 item 2: the committed oracle page itself, copied into the group.
            root.join("ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg"),
        ]);
        assert_eq!(detector_plan(&root, "page"), expected);
        assert_eq!(DETECTOR_STEM, "ja_Pepper-and-Carrot_by-David-Revoy_E01P01");
    }

    #[test]
    fn provenance_for_is_validated_without_filesystem_or_model() {
        let provenance = provenance_for(&manifest()).expect("hand manifest should parse");
        assert!(pc_testkit::provenance::validate("detector", &provenance).is_empty());
    }

    #[test]
    fn provenance_negative_controls_name_the_specific_violation() {
        let mut wrong_backend = manifest();
        wrong_backend["ours"]["backend"] = json!("cv2_dnn");
        let violations = pc_testkit::provenance::validate(
            "detector",
            &provenance_for(&wrong_backend).expect("manifest shape remains valid"),
        );
        assert!(violations.contains(&Violation::BackendContradictsSide {
            at: "detector.ours".into(),
            expected: Backend::Ort,
            found: Backend::Cv2Dnn,
        }));

        let mut non_cpu = manifest();
        non_cpu["ours"]["execution_provider"] = json!("cuda");
        let violations = pc_testkit::provenance::validate(
            "detector",
            &provenance_for(&non_cpu).expect("manifest shape remains valid"),
        );
        assert!(violations.contains(&Violation::ExecutionProviderNotCpu {
            found: "cuda".into(),
        }));

        let mut malformed_digest = manifest();
        malformed_digest["model_digest"] = json!("not-a-digest");
        let violations = pc_testkit::provenance::validate(
            "detector",
            &provenance_for(&malformed_digest).expect("manifest shape remains valid"),
        );
        assert!(violations.contains(&Violation::MalformedDigest {
            at: "detector".into(),
            field: "model_digest",
            value: "not-a-digest".into(),
        }));
    }

    #[test]
    fn model_digest_failure_precedes_inference_closure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = dir.path().join("wrong.onnx");
        std::fs::write(&model, b"wrong model").expect("model bytes");
        let inference_attempted = AtomicBool::new(false);
        let result = with_verified_model(&model, || {
            inference_attempted.store(true, Ordering::SeqCst);
            Ok(())
        });
        assert!(result.is_err());
        assert!(!inference_attempted.load(Ordering::SeqCst));
    }
}
