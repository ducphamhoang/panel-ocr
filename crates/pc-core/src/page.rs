//! spec §2.4 (`PageDataRaw`, detector output / `#raw.json`)
//! and §2.5 (`PageData`, preprocessor output / `#clean.json`).

use crate::{error::StageError, geometry::Rect, image_handle::ImageHandle, language::Language};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageDataRaw {
    pub schema_version: u32,
    pub original_path: PathBuf,
    pub base_image: ImageHandle,
    pub raw_mask: ImageHandle,
    /// §2.4: `new_height / original_height`; exactly `1.0` when no resize happened.
    pub scale: f64,
    pub image_size: (u32, u32),
    pub blocks: Vec<DetectedBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedBlock {
    pub rect: Rect,
    /// `None` == upstream "unknown".
    pub language: Option<Language>,
    /// YOLO objectness*class score, rounded to 3 dp (upstream parity, §8.3 step 4).
    pub confidence: f32,
    /// `mean(raw_mask within rect) / 255` — analytics/debug only.
    pub mask_coverage: f32,
}

impl PageDataRaw {
    /// Every block rect lies within `image_size` (canvas-flush is legal: x1>=0, y1>=0,
    /// x2<=width, y2<=height).
    pub fn validate(&self) -> Result<(), StageError> {
        todo!()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageData {
    pub schema_version: u32,
    pub original_path: PathBuf,
    pub base_image: ImageHandle,
    pub raw_mask: ImageHandle,
    pub scale: f64,
    pub image_size: (u32, u32),
    pub page_language: Option<Language>,
    /// Tight boxes, reading-order sorted.
    pub text_boxes: Vec<TextBox>,
    /// One per tight box, padded a lot. Rasterised into the "box mask".
    pub extended_boxes: Vec<Rect>,
    /// Overlapping extended boxes merged, each paired with its grown reference box.
    pub masking_regions: Vec<MaskingRegion>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBox {
    pub rect: Rect,
    pub language: Option<Language>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskingRegion {
    /// upstream `merged_extended_boxes[i]`
    pub masking: Rect,
    /// upstream `reference_boxes[i]` — `masking` padded by `box_reference_padding`.
    pub reference: Rect,
}

impl PageData {
    /// spec §2.5 invariants, in callable form (§2.5: "assert in debug, test explicitly"):
    ///   1. `extended_boxes.len() == text_boxes.len()`
    ///   2. every `MaskingRegion.reference` contains its `masking` (`Rect::contains_rect`)
    ///   3. all rects (text_boxes, extended_boxes, and both rects of every
    ///      masking_region) lie within `image_size`, canvas-flush being legal
    /// Returns `StageError::InvalidInput` naming the violated invariant.
    pub fn validate(&self) -> Result<(), StageError> {
        todo!()
    }
}
