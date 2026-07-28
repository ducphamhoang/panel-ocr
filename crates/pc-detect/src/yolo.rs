//! Task D5 -- spec §8.3 step 4: YOLO candidate filtering, decode, NMS, rescale,
//! class -> language.
//!
//! Bodies are `todo!()` skeletons: signatures are frozen with the tests, D5 fills them.
#![allow(unused_variables)]

use crate::detector::RawBlock;
use pc_core::Language;

/// `objectness (col 4) > 0.4` -- spec §8.3 step 4, strict.
pub const OBJECTNESS_THRESHOLD: f32 = 0.4;
/// `objectness * best_class_prob > 0.4` -- a **second, separate** gate
/// (`yolov5_utils.py:241`, strict `>`), easy to miss. spec §8.3 step 4.
pub const CLASS_SCORE_THRESHOLD: f32 = 0.4;
pub const NMS_IOU_THRESHOLD: f32 = 0.35;
/// Upstream caps survivors *after* NMS (spec §8.3 step 4).
pub const MAX_DET: usize = 300;
pub const N_CLASSES: usize = 3;
/// `cx, cy, w, h, objectness` + one probability per class.
pub const ROW_STRIDE: usize = 5 + N_CLASSES;

/// A decoded, gated detection candidate in **letterboxed** (network-input) coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// `[x1, y1, x2, y2]`, decoded from `cx, cy, w, h`.
    pub xyxy: [f32; 4],
    pub class_index: u8,
    /// `objectness * class_prob`.
    pub score: f32,
}

/// The letterbox transform that produced the network input (spec §8.3 step 3):
/// square `net_size`, padding added on the right/bottom only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxGeometry {
    pub net_size: u32,
    /// Total padding in x.
    pub dw: f32,
    /// Total padding in y.
    pub dh: f32,
    /// Size of the image that was letterboxed (the base image).
    pub image_size: (u32, u32),
}

impl LetterboxGeometry {
    /// spec §8.3 step 4 computes `(im_w / (net - dw), im_h / (net - dh))`. Letterboxing
    /// scales both axes by the same `r`, so both components are `1/r` — one scalar.
    pub fn resize_ratio(&self) -> f32 {
        todo!("D5: 1/r from the letterbox geometry")
    }
}

/// spec §8.3 step 4: both confidence gates plus the `cx,cy,w,h -> x1,y1,x2,y2` decode.
///
/// `rows` is the flattened `[N, ROW_STRIDE]` block output; its length must be a
/// multiple of `ROW_STRIDE`. Output order is input order (NMS sorts afterwards).
pub fn filter_candidates(rows: &[f32]) -> Vec<Candidate> {
    todo!("D5: objectness gate, best class, class-score gate, xywh->xyxy")
}

/// Intersection over union of two `[x1, y1, x2, y2]` boxes; `0.0` when the union is 0.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    todo!("D5: IoU")
}

/// Greedy NMS, descending score, `NMS_IOU_THRESHOLD`, survivors capped at `MAX_DET`.
///
/// DEVIATION(13): upstream runs **per-class** NMS (`inference.py:115` +
/// `yolov5_utils.py:261`, `agnostic=False`), which can emit duplicate overlapping boxes
/// for one balloon detected under two language classes. v1 deliberately runs
/// **class-agnostic** NMS to remove the duplicate at the source; the surviving box's
/// language is used as-is. See spec §14.13 / §15.1.
pub fn nms(candidates: Vec<Candidate>) -> Vec<Candidate> {
    todo!("D5: class-agnostic greedy NMS")
}

/// Letterboxed coords -> base-image coords (spec §8.3 step 4): multiply by
/// [`LetterboxGeometry::resize_ratio`], truncate to i32, then **clip to the image**.
///
/// spec §16.6 item 4: clamping `x1,y1 >= 0` and `x2,y2 <= image_size` is part of the
/// port (upstream's yolov5 `clip_coords`), and is required for `run()` to produce a
/// `PageDataRaw` that passes its own `validate()` on frame-overhanging detections.
pub fn rescale(candidates: &[Candidate], geometry: &LetterboxGeometry) -> Vec<RawBlock> {
    todo!("D5: rescale + clip to image bounds")
}

/// Round to 3 decimals (upstream `np.round(..., 3)`, spec §8.3 step 4) so golden JSON
/// is stable.
pub fn round3(value: f32) -> f32 {
    todo!("D5: round to 3 decimals")
}

/// spec §8.3 step 4: `0 => English`, `1 => Japanese`, `2 => None`; any other index is
/// `None` plus a `WARN` (a model swap must be diagnosable, not silently mapped).
pub fn class_to_language(class_index: u8) -> Option<Language> {
    todo!("D5: class index -> language")
}

/// `filter_candidates` -> `nms` -> `rescale`, i.e. the whole of spec §8.3 step 4.
pub fn postprocess(rows: &[f32], geometry: &LetterboxGeometry) -> Vec<RawBlock> {
    todo!("D5: compose the postprocess")
}
