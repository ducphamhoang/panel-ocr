//! spec §2.7. These are reporting records, never control flow: a dropped box or an
//! unfitted mask is recorded here, not raised as a `StageError` (§2.9, §5.6).

use crate::geometry::Rect;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrAnalytic {
    pub path: PathBuf,
    /// Boxes considered *before* removal (§9.3 step 7).
    pub num_boxes: usize,
    pub box_areas_ocred: Vec<i64>,
    pub box_areas_removed: Vec<i64>,
    /// Text + box in ORIGINAL image coordinates.
    pub removed: Vec<RemovedBox>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemovedBox {
    pub text: String,
    pub rect: Rect,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskFittingAnalytic {
    pub path: PathBuf,
    pub fit_found: bool,
    pub candidate_index: usize,
    pub std_deviation: f64,
    pub thickness: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenoiseAnalytic {
    pub path: PathBuf,
    /// sigma of ALL mask_data regions, not only the denoised ones (§11.3 step 6).
    pub std_deviations: Vec<f64>,
    pub boxes_denoised: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectAnalytic {
    pub path: PathBuf,
    pub blocks_detected: usize,
    pub blocks_kept: usize,
}
