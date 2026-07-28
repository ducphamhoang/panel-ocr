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

/// Where the detector-boundary recording stands (§7.2.1). Kept as a function rather
/// than a constant so the reason is computed from the actual build, not from a comment.
pub fn detector_backend_status() -> DetectorStatus {
    // §8.5 tasks D1 (`pc-models`) and D4 (`pc-detect/src/onnx.rs`) are the two pieces
    // that would let this binary run a real detector. Neither exists yet, so there is
    // no code path that could produce `_detector_mask.png` / `_detector_blocks.json`
    // even if weights were present. `cfg!(feature = ...)` cannot express "another
    // crate's module is unimplemented", so this is stated explicitly.
    DetectorStatus::BackendNotImplemented
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorStatus {
    /// `pc-detect`'s ONNX backend (D4) and `pc-models` (D1) are not implemented.
    BackendNotImplemented,
}

impl DetectorStatus {
    pub fn explain(self) -> &'static str {
        match self {
            Self::BackendNotImplemented => {
                "\
detector-boundary recording is UNAVAILABLE in this checkout.
Reason: spec §8.5 task D4 (`crates/pc-detect/src/onnx.rs`, the `ort` session) and task D1
(the `pc-models` crate that resolves/downloads/verifies `comictextdetector.pt.onnx`) are
not implemented yet. Model weights alone would not be enough — there is no inference code
to feed them to.

Consequently NONE of these §7.2 artifacts can be produced yet:
  tests/fixtures/recorded/<stem>_detector_mask.png     (§7.2.1, raw RawDetection.mask)
  tests/fixtures/recorded/<stem>_detector_blocks.json  (§7.2.1, Vec<RawBlock>)
  tests/fixtures/recorded/<stem>_base.png              (§7.2)
  tests/fixtures/recorded/<stem>_raw_mask.png          (§7.2)
  tests/fixtures/recorded/<stem>#raw.json              (§7.2, PageDataRaw)

and these tests must stay `#[ignore]`d:
  pc-detect     d7_run.rs::a6_pending_insta_snapshot_of_recorded_page      (§8.7(A)6)
  pc-detect     d7_run.rs::b9_pending_recorded_page_regression_lock        (§8.7(B)9)
  pc-preprocess p5_run.rs::b11_pending_insta_snapshot_of_recorded_page_tiers (§9.7(B)11)
  pc-denoise    n4_run.rs::b13_pending_recorded_page_end_to_end_golden     (§11.7(B)13)

To finish F1 later: implement D1 + D4, then run on a machine with the weights
  cargo xtask record-fixtures --only detector --detector onnx:<path-to-comictextdetector.pt.onnx>"
            }
        }
    }
}
