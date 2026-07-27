//! spec §2.6 — masker output (`#mask_data.json`).

use crate::{geometry::Rect, image_handle::ImageHandle};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskData {
    pub schema_version: u32,
    pub original_path: PathBuf,
    pub base_image: ImageHandle,
    /// RGBA.
    pub combined_mask: ImageHandle,
    pub scale: f64,
    /// One per *attempted* masking region, including failures. Regions whose precise
    /// mask was blank (detector false positive) are ABSENT (§2.6, `masker.py:68`).
    pub regions: Vec<MaskRegionStats>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskRegionStats {
    pub rect: Rect,
    /// Border std dev of the *chosen* candidate. Must be f64 (§10.3, §15.9, decided).
    pub std_deviation: f64,
    /// `std_deviation > masker.mask_max_standard_deviation`.
    pub failed: bool,
    /// `None` when the box mask was chosen.
    pub thickness: Option<u32>,
}
