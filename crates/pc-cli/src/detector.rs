//! `--detector` → a [`DetectorProvider`] (spec §16.12 items 2 and 3).
//!
//! **This is where v1's honest limitation lives.** Tasks D1 (model provisioning) and D4
//! (the `ort` session) are deferred, and `pc-detect`'s `onnx` feature is not default, so
//! the `onnx` spec cannot be built. It fails here, once, with a message that names the
//! tasks and the workaround — never as a panic, never as a silent empty-detection run.
//!
//! Fully pinned; implemented, not stubbed.

use crate::args::DetectorSpec;
use pc_core::StageError;
use pc_detect::{MockDetector, ReplayDetector, TextDetector};
use pc_pipeline::{DetectorProvider, SharedDetector};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The message the `onnx` spec produces in v1. A *fatal* condition (§5.3 → exit 1).
pub const ONNX_UNAVAILABLE: &str = "the ONNX text detector is not available in this build \
(tasks D1 model provisioning and D4 `ort` session are not implemented yet). \
Pass `--detector replay:<DIR>` to run against recorded detector fixtures, \
or `--detector mock` for a no-text pass-through run.";

/// spec §7.2.1 — one stem-bound `ReplayDetector` per image.
pub struct ReplayProvider {
    dir: PathBuf,
}

impl ReplayProvider {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

impl DetectorProvider for ReplayProvider {
    fn detector_for(&self, original: &Path) -> Result<Arc<dyn TextDetector>, StageError> {
        let stem = original
            .file_stem()
            .ok_or_else(|| {
                StageError::InvalidInput(format!(
                    "input path `{}` has no file stem",
                    original.display()
                ))
            })?
            .to_string_lossy()
            .into_owned();

        // `ReplayDetector::new` panics on a missing fixture (§7.2.1's "fail at
        // construction"). At the pipeline boundary a missing fixture must be a
        // per-image error instead, so probe first.
        let mask = pc_detect::mock::detector_mask_path(&self.dir, &stem);
        let blocks = pc_detect::mock::detector_blocks_path(&self.dir, &stem);
        for path in [&mask, &blocks] {
            if !path.exists() {
                return Err(StageError::Model(format!(
                    "no replay fixture `{}`; run `cargo xtask record-fixtures`",
                    path.display()
                )));
            }
        }
        Ok(Arc::new(ReplayDetector::new(&self.dir, &stem)))
    }
}

/// Build the provider a run will use. `Err` here is fatal (§5.3).
pub fn build_provider(
    spec: &DetectorSpec,
    model_path: Option<&Path>,
) -> Result<Box<dyn DetectorProvider>, StageError> {
    match spec {
        DetectorSpec::Onnx => {
            let _ = model_path;
            Err(StageError::Model(ONNX_UNAVAILABLE.to_string()))
        }
        DetectorSpec::Mock => Ok(Box::new(SharedDetector::new(Arc::new(MockDetector::new())))),
        DetectorSpec::Replay(dir) => Ok(Box::new(ReplayProvider::new(dir.clone()))),
    }
}
