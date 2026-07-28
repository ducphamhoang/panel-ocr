//! `pc-detect` -- STAGE 1, text detection (spec §8).
//!
//! Everything model-shaped goes through [`TextDetector`] (§1 rule 4), so the whole
//! stage is testable with mocks and replay fixtures and no ONNX runtime present. The
//! `onnx` feature (task D4, not implemented in this pass) adds the real backend.
//!
//! Bodies here are `todo!()` skeletons for task D7; the signatures are frozen with the
//! tests.
#![allow(unused_variables)]

pub mod detector;
pub mod mask;
pub mod resize;
pub mod yolo;

#[cfg(any(test, feature = "testkit"))]
pub mod mock;

pub use detector::{DetectInput, DetectOutput, RawBlock, RawDetection, TextDetector};
pub use mask::{refine_simple, REFINE_DILATE_RADIUS, REFINE_EXPAND, REFINE_THRESHOLD};
pub use resize::{calculate_new_size_and_scale, resize_area, round_half_away};
pub use yolo::{Candidate, LetterboxGeometry};

#[cfg(any(test, feature = "testkit"))]
pub use mock::{write_replay_fixture, MockDetector, ReplayDetector};

use image::GrayImage;
use pc_core::{DetectAnalytic, DetectedBlock, ImageHandle, PageDataRaw, Rect, StageError, Step};
use std::path::Path;

/// spec §8.3 step 6 / §8.2: fixed at 0.1 in v1.
pub const DEFAULT_MIN_MASK_COVERAGE: f32 = 0.1;

/// `mean(mask over rect) / 255` (spec §8.3 step 6). `rect` uses the codebase-wide
/// exclusive `x2`/`y2` convention (§16.6 item 5); an empty or fully out-of-bounds rect
/// has coverage `0.0`.
pub fn mask_coverage(mask: &GrayImage, rect: Rect) -> f32 {
    todo!("D7: mean mask value inside rect, normalised")
}

/// spec §3 -- the stage contract. `Ctx` is the injected detector.
pub struct DetectStage;

impl pc_core::Stage for DetectStage {
    type Input = DetectInput;
    type Output = DetectOutput;
    type Ctx<'a> = &'a dyn TextDetector;
    const STEP: Step = Step::Detect;

    fn run(input: Self::Input, ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError> {
        run(input, ctx)
    }
}

/// spec §8.3, steps 1-7.
///
/// Rejects `MaskRefineMode::Annotation` with `StageError::InvalidInput` **before** any
/// detection work (§8.3 step 5, §15.2, §16.5 item 3) -- failing after paying for
/// inference would be gratuitous.
pub fn run(input: DetectInput, detector: &dyn TextDetector) -> Result<DetectOutput, StageError> {
    todo!("D7: load/resize, detect, refine, coverage filter, assemble")
}

/// spec §8.3 step 7: `PageDataRaw` from the surviving blocks, in NMS order.
pub fn assemble_page(
    input: &DetectInput,
    base_image: ImageHandle,
    raw_mask: ImageHandle,
    scale: f64,
    image_size: (u32, u32),
    blocks: Vec<DetectedBlock>,
) -> PageDataRaw {
    todo!("D7: assemble PageDataRaw")
}

/// spec §2.7 / §8.3 step 7.
pub fn build_analytic(path: &Path, blocks_detected: usize, blocks_kept: usize) -> DetectAnalytic {
    todo!("D7: build DetectAnalytic")
}
