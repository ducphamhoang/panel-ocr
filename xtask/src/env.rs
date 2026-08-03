//! Capability probing (spec §7.2's "maintainer machine, real models present").
//!
//! The whole point of this module: `record-fixtures` must be able to say *precisely*
//! what it can and cannot do in the environment it finds itself in, and skip the rest
//! with an actionable message — never silently produce approximated garbage, and never
//! panic with a bare `ModuleNotFoundError`.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// A Python interpreter that has been *verified* to import `cv2` and `PIL`.
#[derive(Debug, Clone)]
pub struct PythonTooling {
    pub interpreter: PathBuf,
    pub python_version: String,
    pub opencv_version: String,
    pub pillow_version: String,
    pub numpy_version: String,
}

const PROBE: &str = "import sys, json, cv2, PIL, numpy\n\
print(json.dumps({\n\
  'python': sys.version.split()[0],\n\
  'cv2': cv2.__version__,\n\
  'PIL': PIL.__version__,\n\
  'numpy': numpy.__version__,\n\
}))";

impl PythonTooling {
    /// Probe one candidate interpreter. `Ok(None)` means "exists but lacks the modules",
    /// which is a skip, not a failure.
    pub fn probe(interpreter: &Path) -> Result<Option<Self>> {
        let output = match Command::new(interpreter).arg("-c").arg(PROBE).output() {
            Ok(output) => output,
            // Interpreter not on PATH at all.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| format!("running {}", interpreter.display()))
            }
        };
        if !output.status.success() {
            return Ok(None);
        }
        let parsed: serde_json::Value =
            serde_json::from_slice(&output.stdout).with_context(|| {
                format!(
                    "parsing the capability probe from {}",
                    interpreter.display()
                )
            })?;
        let field = |key: &str| parsed[key].as_str().unwrap_or("unknown").to_string();
        Ok(Some(Self {
            interpreter: interpreter.to_path_buf(),
            python_version: field("python"),
            opencv_version: field("cv2"),
            pillow_version: field("PIL"),
            numpy_version: field("numpy"),
        }))
    }

    /// Try `explicit` if given, else a short list of conventional names. The first
    /// interpreter with both `cv2` and `PIL` wins.
    pub fn discover(explicit: Option<&Path>) -> Result<Option<Self>> {
        if let Some(path) = explicit {
            return match Self::probe(path)? {
                Some(found) => Ok(Some(found)),
                None => bail!(
                    "--python {} cannot import both `cv2` and `PIL`.\n\
                     Install them there, e.g.:\n  {} -m pip install opencv-python-headless pillow numpy",
                    path.display(),
                    path.display()
                ),
            };
        }
        for candidate in ["python3", "python"] {
            if let Some(found) = Self::probe(Path::new(candidate))? {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    pub fn describe(&self) -> String {
        format!(
            "{} (python {}, opencv {}, pillow {}, numpy {})",
            self.interpreter.display(),
            self.python_version,
            self.opencv_version,
            self.pillow_version,
            self.numpy_version
        )
    }
}

/// The message printed when no capable interpreter is found. Actionable by design
/// (§7.2: recording is maintainer-run; a CI/sandbox checkout must be told what to do,
/// not left guessing).
pub const NO_PYTHON_HELP: &str = "\
no Python interpreter with both `cv2` and `PIL` was found.
These fixtures are REFERENCE outputs of the real third-party implementations; they must
not be approximated in Rust (that would make the parity gates circular). To record them:

    python3 -m venv .venv
    .venv/bin/pip install opencv-python-headless pillow numpy
    cargo xtask record-fixtures --python .venv/bin/python

Until then the dependent tests stay `#[ignore]`d — that is the correct state, not a bug.";

/// Where the detector-boundary recording stands (§7.2.1), computed from this build and
/// the configured model path. An explicit `--detector` spec takes precedence over the
/// `PANEL_OCR_ONNX_MODEL` fallback, just as other explicit xtask options do.
pub fn detector_backend_status(configured_detector: Option<&str>) -> Result<DetectorStatus> {
    detector_backend_status_under(
        configured_detector,
        &crate::paths::workspace_root(),
        cfg!(feature = "onnx"),
    )
}

fn detector_backend_status_under(
    configured_detector: Option<&str>,
    workspace_root: &Path,
    onnx_enabled: bool,
) -> Result<DetectorStatus> {
    let resolution =
        resolve_detector_model_under(configured_detector, workspace_root, onnx_enabled)?;
    let model_path = match resolution {
        ModelResolution::OnnxFeatureDisabled => return Ok(DetectorStatus::OnnxFeatureDisabled),
        ModelResolution::ModelWeightsMissing {
            model_path,
            model_source,
        } => {
            return Ok(DetectorStatus::ModelWeightsMissing {
                model_path,
                model_source,
            })
        }
        ModelResolution::Ready { model_path } => model_path,
    };
    // §16.29 item 2: the atomic recording commit moved P01 out of `oracle_pages/` into the
    // recorded group it belongs to; a fresh `--force` re-record must find it there.
    let page = workspace_root
        .join("tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg");
    if !page.is_file() {
        return Ok(DetectorStatus::MangaPagesMissing);
    }
    Ok(DetectorStatus::Ready { model_path })
}

/// Resolve only the model capability needed by reports that do not consume the signed manga
/// page. In particular, this function must not check `tests/fixtures/recorded/detector` for the
/// maintainer page; `calibrate-goldens`'s demo_bubbles report never touches that artifact.
pub(crate) fn resolve_detector_model(configured_detector: Option<&str>) -> Result<ModelResolution> {
    resolve_detector_model_under(
        configured_detector,
        &crate::paths::workspace_root(),
        cfg!(feature = "onnx"),
    )
}

fn resolve_detector_model_under(
    configured_detector: Option<&str>,
    workspace_root: &Path,
    onnx_enabled: bool,
) -> Result<ModelResolution> {
    let explicit_path = configured_detector.map(parse_detector_spec).transpose()?;
    if !onnx_enabled {
        return Ok(ModelResolution::OnnxFeatureDisabled);
    }
    let (model_path, model_source) = match explicit_path {
        Some(path) => (Some(path), Some("from the --detector onnx:<path> flag")),
        None => match std::env::var_os("PANEL_OCR_ONNX_MODEL") {
            Some(path) => (
                Some(PathBuf::from(path)),
                Some("from the PANEL_OCR_ONNX_MODEL environment variable"),
            ),
            None => (None, None),
        },
    };
    let model_exists = model_path.as_deref().is_some_and(|path| {
        path.is_file() || (path.is_relative() && workspace_root.join(path).is_file())
    });
    if !model_exists {
        return Ok(ModelResolution::ModelWeightsMissing {
            model_path,
            model_source,
        });
    }
    Ok(ModelResolution::Ready {
        model_path: model_path.expect("checked above"),
    })
}

pub(crate) fn parse_detector_spec(spec: &str) -> Result<PathBuf> {
    let Some(path) = spec.strip_prefix("onnx:") else {
        bail!("invalid --detector value `{spec}`; expected the form `onnx:<path>`");
    };
    if path.is_empty() {
        bail!("invalid --detector value `{spec}`; expected the form `onnx:<path>`");
    }
    Ok(PathBuf::from(path))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectorStatus {
    OnnxFeatureDisabled,
    ModelWeightsMissing {
        model_path: Option<PathBuf>,
        model_source: Option<&'static str>,
    },
    MangaPagesMissing,
    Ready {
        model_path: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelResolution {
    OnnxFeatureDisabled,
    ModelWeightsMissing {
        model_path: Option<PathBuf>,
        model_source: Option<&'static str>,
    },
    Ready {
        model_path: PathBuf,
    },
}

impl DetectorStatus {
    pub fn explain(self) -> String {
        match self {
            Self::OnnxFeatureDisabled => ONNX_FEATURE_DISABLED_EXPLANATION.into(),
            Self::ModelWeightsMissing {
                model_path: Some(path),
                model_source: Some(source),
            } => format!(
                "the ONNX feature is enabled but the model weights are not present at {} ({}). \
                 Point it at the sha256-verified comictextdetector.pt.onnx and retry.",
                path.display(),
                source
            ),
            Self::ModelWeightsMissing {
                model_path: None,
                model_source: None,
            } => format!("{MODEL_WEIGHTS_MISSING_EXPLANATION}\nNo model path was supplied."),
            Self::ModelWeightsMissing { .. } => MODEL_WEIGHTS_MISSING_EXPLANATION.into(),
            Self::MangaPagesMissing => MANGA_PAGES_MISSING_EXPLANATION.into(),
            Self::Ready { model_path } => {
                format!("detector recording is ready with {}", model_path.display())
            }
        }
    }
}

const ONNX_FEATURE_DISABLED_EXPLANATION: &str = concat!(
    "detector-boundary recording is UNAVAILABLE because xtask was built without its ",
    "opt-in ONNX feature. Rebuild with `cargo xtask-onnx` (or `cargo run --package ",
    "xtask --features onnx -- ...`) and retry the detector recording command.\n\n",
    "\
These §7.2 artifacts cannot be produced yet:
  tests/fixtures/recorded/detector/<stem>_detector_mask.png     (§7.2.1, raw RawDetection.mask)
  tests/fixtures/recorded/detector/<stem>_detector_blocks.json  (§7.2.1, Vec<RawBlock>)
  tests/fixtures/recorded/detector/<stem>_base.png              (§7.2)
  tests/fixtures/recorded/detector/<stem>_raw_mask.png          (§7.2)
  tests/fixtures/recorded/detector/<stem>#raw.json              (§7.2, PageDataRaw)

and these tests must stay `#[ignore]`d:
  pc-detect     d7_run.rs::a6_pending_recorded_page_equality_and_determinism (§8.7(A)6)
  pc-detect     d7_run.rs::b9_pending_recorded_page_regression_lock        (§8.7(B)9)
  pc-preprocess p5_run.rs::b11_pending_recorded_page_tier_arithmetic         (§9.7(B)11)
  pc-denoise    n4_run.rs::b13_pending_recorded_page_end_to_end_golden     (§11.7(B)13)"
);

const MODEL_WEIGHTS_MISSING_EXPLANATION: &str = concat!(
    "detector-boundary recording is UNAVAILABLE because the ONNX feature is enabled ",
    "but the model weights are missing. Supply them with either the ",
    "`--detector onnx:<path>` flag or the `PANEL_OCR_ONNX_MODEL` environment variable, ",
    "then retry the detector recording command.\n\n",
    "\
These §7.2 artifacts cannot be produced yet:
  tests/fixtures/recorded/detector/<stem>_detector_mask.png     (§7.2.1, raw RawDetection.mask)
  tests/fixtures/recorded/detector/<stem>_detector_blocks.json  (§7.2.1, Vec<RawBlock>)
  tests/fixtures/recorded/detector/<stem>_base.png              (§7.2)
  tests/fixtures/recorded/detector/<stem>_raw_mask.png          (§7.2)
  tests/fixtures/recorded/detector/<stem>#raw.json              (§7.2, PageDataRaw)

and these tests must stay `#[ignore]`d:
  pc-detect     d7_run.rs::a6_pending_recorded_page_equality_and_determinism (§8.7(A)6)
  pc-detect     d7_run.rs::b9_pending_recorded_page_regression_lock        (§8.7(B)9)
  pc-preprocess p5_run.rs::b11_pending_recorded_page_tier_arithmetic         (§9.7(B)11)
  pc-denoise    n4_run.rs::b13_pending_recorded_page_end_to_end_golden     (§11.7(B)13)"
);

const MANGA_PAGES_MISSING_EXPLANATION: &str = concat!(
    "detector-boundary recording is UNAVAILABLE until the maintainer supplies ",
    "license-clean full manga page(s) for §7.2, each no larger than 400 KB. Supply ",
    "those pages, then retry the detector recording command.\n\n",
    "\
These §7.2 artifacts cannot be produced yet:
  tests/fixtures/recorded/detector/<stem>_detector_mask.png     (§7.2.1, raw RawDetection.mask)
  tests/fixtures/recorded/detector/<stem>_detector_blocks.json  (§7.2.1, Vec<RawBlock>)
  tests/fixtures/recorded/detector/<stem>_base.png              (§7.2)
  tests/fixtures/recorded/detector/<stem>_raw_mask.png          (§7.2)
  tests/fixtures/recorded/detector/<stem>#raw.json              (§7.2, PageDataRaw)

and these tests must stay `#[ignore]`d:
  pc-detect     d7_run.rs::a6_pending_recorded_page_equality_and_determinism (§8.7(A)6)
  pc-detect     d7_run.rs::b9_pending_recorded_page_regression_lock        (§8.7(B)9)
  pc-preprocess p5_run.rs::b11_pending_recorded_page_tier_arithmetic         (§9.7(B)11)
  pc-denoise    n4_run.rs::b13_pending_recorded_page_end_to_end_golden     (§11.7(B)13)"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Explicit Fable ruling, accepted and ratified: two residual evasions remain out of scope
    // for source-text guarding because closing them requires deliberate adversarial construction,
    // rather than being a natural regression shape. First, the filename could be split across
    // multiple `.join()`/`format!` fragments so no single occurrence of the literal filename or
    // the four banned substrings appears together. Second, the filename could be imported from
    // another module as a named constant (for example, the existing
    // `crate::record::DETECTOR_PAGE_RECORDED`) and laundered through a helper outside this
    // scanned window that also bypasses `workspace_root` via an ambient `paths::` call. A
    // reintroduction that reuses that constant but still routes through `workspace_root` correctly
    // is caught by the behavioral half: synthetic root lacking the page -> non-Ready result.
    fn model_resolution_is_independent_of_the_maintainer_manga_page() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model = temp.path().join("comictextdetector.pt.onnx");
        std::fs::write(&model, b"synthetic model").expect("model");

        // Exercise the model-present path with an explicitly absent synthetic workspace root;
        // this must fail if the resolver starts checking the maintainer manga page again.
        let resolution = resolve_detector_model_under(
            Some(&format!("onnx:{}", model.display())),
            temp.path(),
            true,
        )
        .expect("model resolution");
        assert_eq!(
            resolution,
            ModelResolution::Ready {
                model_path: model.clone()
            }
        );

        // Keep the resolver's contract explicit: a model-only resolution path must not grow a
        // dependency on the maintainer manga page. This source guard catches a regression that
        // consults the real checkout even when the synthetic root below has no such page.
        let source = include_str!("env.rs");
        let resolver_start = source
            .find("fn resolve_detector_model_under(")
            .expect("resolver source");
        let resolver_end = source[resolver_start..]
            .find("\npub(crate) fn parse_detector_spec")
            .map(|offset| resolver_start + offset)
            .expect("resolver end");
        let resolver_source = &source[resolver_start..resolver_end];
        let status_start = source
            .find("fn detector_backend_status_under(")
            .expect("backend status source");
        let status_end = source[status_start..]
            .find("\n/// Resolve only the model capability")
            .map(|offset| status_start + offset)
            .expect("backend status end");
        let maintainer_page_filename =
            format!("{}{}", "ja_Pepper-and-Carrot_by-David-Revoy_E01P01", ".jpg");
        let filename_offsets: Vec<_> = source
            .match_indices(&maintainer_page_filename)
            .map(|(offset, _)| offset)
            .collect();
        assert!(
            filename_offsets.len() == 1
                && filename_offsets.first().is_some_and(|&offset| {
                    offset >= status_start
                        && offset < status_end
                        && !(offset >= resolver_start && offset < resolver_end)
                }),
            "maintainer page filename must occur exactly once within detector_backend_status_under"
        );
        assert!(!source[resolver_start..resolver_end].contains("tests/fixtures/recorded/detector/"));
        assert!(!resolver_source.contains("recorded_root()"));
        assert!(!resolver_source.contains("fixtures_root()"));
        assert!(!resolver_source.contains("tests/fixtures"));
        assert!(!resolver_source.contains("paths::"));

        let missing = temp.path().join("missing.onnx");
        assert!(matches!(
            resolve_detector_model_under(
                Some(&format!("onnx:{}", missing.display())),
                temp.path(),
                true,
            )
            .expect("missing model resolution"),
            ModelResolution::ModelWeightsMissing { .. }
        ));
    }

    #[test]
    fn detector_backend_status_uses_the_supplied_workspace_root_for_the_page_check() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model = temp.path().join("comictextdetector.pt.onnx");
        std::fs::write(&model, b"synthetic model").expect("model");

        let status = detector_backend_status_under(
            Some(&format!("onnx:{}", model.display())),
            temp.path(),
            true,
        )
        .expect("detector status");
        assert_eq!(status, DetectorStatus::MangaPagesMissing);
    }
}
