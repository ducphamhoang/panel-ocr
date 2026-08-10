//! Shared helpers for the `pc-pipeline` frozen tests (tasks G1, G2, D9/E5).

#![allow(dead_code)]

use image::{Rgb, RgbImage};
use pc_config::Profile;
use pc_core::{ImageHandle, Rect, StageError};
use pc_detect::{MockDetector, RawBlock, RawDetection, TextDetector};
use pc_pipeline::{Checkpointing, DetectorProvider, PipelineCtx, PipelineOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A small RGB page with a dark rectangle in the middle, written as PNG.
pub fn write_page(dir: &Path, name: &str, size: (u32, u32)) -> PathBuf {
    let (width, height) = size;
    let mut image = RgbImage::from_pixel(width, height, Rgb([255, 255, 255]));
    for y in height / 4..height / 2 {
        for x in width / 4..width / 2 {
            image.put_pixel(x, y, Rgb([16, 16, 16]));
        }
    }
    let path = dir.join(name);
    image
        .save_with_format(&path, image::ImageFormat::Png)
        .expect("write test page");
    path
}

/// Default options pointing at a temp cache + output dir, single-threaded and
/// deterministic. `output_dir` is absolute, so exports land there directly (§16.11
/// item 4).
pub fn options(cache_dir: &Path, output_dir: &Path) -> PipelineOptions {
    let mut profile = Profile::default();
    // These helpers do not construct an OCR engine, so `ctx.ocr` remains `None` in these tests.
    profile.preprocessor.ocr_enabled = false;
    // §16.46 item 13(b) INPUT PINS -- no assertion, name or expected value changes with them.
    // Every `pc-pipeline` integration test builds its options here, and item 1 flips both of
    // these defaults. Left inherited, the whole suite would silently start exercising
    // Annotation masking and an enabled inpainter it never injects a provider for -- green,
    // because nothing on these synthetic pages is eligible, while testing something other
    // than what each test was written for. Pinning keeps the subject fixed; a test that
    // WANTS the shipped defaults sets them itself.
    profile.text_detector.mask_refine_mode = pc_config::MaskRefineMode::Simple;
    profile.inpainter.inpainting_enabled = false;
    PipelineOptions {
        profile,
        cache_dir: cache_dir.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        checkpointing: Checkpointing::Disk,
        threads: 1,
        ..PipelineOptions::default()
    }
}

/// A detector that returns one block covering `rect`, with the mask painted inside it —
/// so the §8.3 step 6 coverage filter keeps it.
pub fn detector_with_block(rect: Rect) -> Arc<dyn TextDetector> {
    Arc::new(
        MockDetector::new()
            .with_blocks(vec![RawBlock {
                rect,
                class_index: 1,
                confidence: 0.9,
            }])
            .with_block_fill(255),
    )
}

/// A detector that finds nothing (`--detector mock`'s behaviour, §16.12 item 2).
pub fn empty_detector() -> Arc<dyn TextDetector> {
    Arc::new(MockDetector::new())
}

/// Fails `detect` for every image.
pub fn failing_detector() -> Arc<dyn TextDetector> {
    Arc::new(MockDetector::new().failing())
}

/// Panics inside `detect` — §5.2's `catch_unwind` boundary must contain this.
#[derive(Debug, Default)]
pub struct PanickingDetector;

impl TextDetector for PanickingDetector {
    fn detect(&self, _image: &image::RgbImage) -> Result<RawDetection, StageError> {
        panic!("detector exploded");
    }
}

/// Routes per image stem: `failing_stems` get `detector`, everything else gets `fallback`.
pub struct SelectiveProvider {
    pub failing_stems: Vec<String>,
    pub failing: Arc<dyn TextDetector>,
    pub fallback: Arc<dyn TextDetector>,
}

impl SelectiveProvider {
    pub fn new(failing_stems: &[&str], failing: Arc<dyn TextDetector>) -> Self {
        Self {
            failing_stems: failing_stems.iter().map(|s| s.to_string()).collect(),
            failing,
            fallback: empty_detector(),
        }
    }

    pub fn with_fallback(mut self, fallback: Arc<dyn TextDetector>) -> Self {
        self.fallback = fallback;
        self
    }
}

impl DetectorProvider for SelectiveProvider {
    fn detector_for(&self, original: &Path) -> Result<Arc<dyn TextDetector>, StageError> {
        let stem = original
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if self.failing_stems.contains(&stem) {
            Ok(Arc::clone(&self.failing))
        } else {
            Ok(Arc::clone(&self.fallback))
        }
    }
}

/// A provider that refuses to build a detector at all (§16.12 item 3: a per-image error,
/// not a fatal one).
pub struct RefusingProvider;

impl DetectorProvider for RefusingProvider {
    fn detector_for(&self, _original: &Path) -> Result<Arc<dyn TextDetector>, StageError> {
        Err(StageError::Model("no detector for you".into()))
    }
}

/// A provider whose declared fatality can be selected by the test, while the underlying
/// provider error remains the same `StageError::Model` in both cases.
pub struct FatalityProvider {
    pub fatal: bool,
    pub attempts: AtomicUsize,
}

impl FatalityProvider {
    pub fn new(fatal: bool) -> Self {
        Self {
            fatal,
            attempts: AtomicUsize::new(0),
        }
    }
}

impl DetectorProvider for FatalityProvider {
    fn detector_for(&self, _original: &Path) -> Result<Arc<dyn TextDetector>, StageError> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        Err(StageError::Model("stub provider refusal".into()))
    }

    fn failures_are_run_fatal(&self) -> bool {
        self.fatal
    }
}

pub fn ctx<'a>(provider: &'a dyn DetectorProvider) -> PipelineCtx<'a> {
    PipelineCtx::new(provider)
}

/// A materialised handle for `path` (path only — the pipeline decodes on demand).
pub fn handle(path: &Path) -> ImageHandle {
    ImageHandle::from_path(path)
}
