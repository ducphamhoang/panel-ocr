//! spec §8.2 -- the stage contract and the ML boundary.
//!
//! `TextDetector` is the seam that keeps the `candle` escape hatch real (§1 rule 4) and
//! lets CI run the whole pipeline with zero model files (§7.2).

use image::{GrayImage, RgbImage};
use pc_config::TextDetectorConfig;
use pc_core::{PageDataRaw, Rect, StageError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectInput {
    pub schema_version: u32,
    /// The input file, or one strip segment of it (§4.3).
    pub source: pc_core::ImageHandle,
    /// Path recorded in `PageDataRaw` (the segment path when split).
    pub original_path: PathBuf,
    /// `general.input_height_lower_target`
    pub target_height_lower: u32,
    /// `general.input_height_upper_target`
    pub target_height_upper: u32,
    /// Where to write the scaled PNG; `None` in `Checkpointing::Memory` mode (§4.1).
    /// spec §16.6 item 2: this spelling is authoritative (§4.3's `base_png_dest` is a typo).
    pub base_image_dest: Option<PathBuf>,
    pub raw_mask_dest: Option<PathBuf>,
    /// §8.3 step 6; [`crate::DEFAULT_MIN_MASK_COVERAGE`] in v1.
    pub min_mask_coverage: f32,
    /// spec §16.6 item 1: required because the stage selects the refinement path
    /// `MaskRefineMode::Annotation` (§16.5 item 3), and consistent with §3's
    /// "config by value inside the Input" rule.
    pub config: TextDetectorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectOutput {
    pub page: PageDataRaw,
    pub analytics: pc_core::DetectAnalytic,
}

pub trait TextDetector: Send + Sync {
    /// `image`: RGB8, already scaled to the target height range (§8.3 step 2).
    fn detect(&self, image: &RgbImage) -> Result<RawDetection, StageError>;
}

#[derive(Debug, Clone)]
pub struct RawDetection {
    /// Boxes in `image` coordinates, confidence-filtered and NMS'd, in detector order.
    pub blocks: Vec<RawBlock>,
    /// U-Net text-probability mask, 8-bit, same size as `image`.
    pub mask: GrayImage,
}

/// `Serialize`/`Deserialize` exist for §7.2.1's `<stem>_detector_blocks.json` replay
/// artifact, which records the detector boundary *before* refinement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RawBlock {
    pub rect: Rect,
    pub class_index: u8,
    pub confidence: f32,
}
