//! The two external resources the pipeline injects into stages (§3's `Ctx`), plus the
//! provider indirection §16.12 item 3 pins.

use pc_core::StageError;
use pc_detect::TextDetector;
use pc_ocr::OcrEngineFactory;
use std::path::Path;
use std::sync::Arc;

/// spec §16.12 item 3 — per-image detector resolution.
///
/// §4.5 wants one shared `ort` session across rayon threads; §7.2.1 binds
/// `ReplayDetector` to one fixture stem at construction. A provider serves both: the
/// shared case returns the same `Arc` every time, the replay case builds a stem-bound
/// detector per image.
pub trait DetectorProvider: Send + Sync {
    fn detector_for(&self, original: &Path) -> Result<Arc<dyn TextDetector>, StageError>;

    /// Whether a failure from [`Self::detector_for`] dooms the whole run (§16.19 item 5).
    ///
    /// The criterion is causal, not textual: `true` iff the failure is independent of
    /// `original`, so every remaining image would fail identically for the same reason.
    /// Default `false` per §16.12 item 3.
    fn failures_are_run_fatal(&self) -> bool {
        false
    }
}

/// The §4.5 case: one detector, shared by every image.
pub struct SharedDetector(Arc<dyn TextDetector>);

impl SharedDetector {
    pub fn new(detector: Arc<dyn TextDetector>) -> Self {
        Self(detector)
    }
}

impl DetectorProvider for SharedDetector {
    fn detector_for(&self, _original: &Path) -> Result<Arc<dyn TextDetector>, StageError> {
        Ok(Arc::clone(&self.0))
    }
}

/// The resources a run injects into the stages. `ocr` remains optional because callers may
/// choose not to construct an engine; `pc_preprocess::run` handles `None` as "no OCR pass".
#[derive(Clone, Copy)]
pub struct PipelineCtx<'a> {
    pub detectors: &'a dyn DetectorProvider,
    pub ocr: Option<&'a dyn OcrEngineFactory>,
}

impl<'a> PipelineCtx<'a> {
    pub fn new(detectors: &'a dyn DetectorProvider) -> Self {
        Self {
            detectors,
            ocr: None,
        }
    }

    pub fn with_ocr(mut self, ocr: &'a dyn OcrEngineFactory) -> Self {
        self.ocr = Some(ocr);
        self
    }
}
