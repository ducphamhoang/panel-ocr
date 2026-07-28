//! Task D5 -- spec §8.3 step 4: YOLO candidate filtering, decode, NMS, rescale,
//! class -> language.
//!
//! Implemented (task D5); signatures are frozen with the tests.

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
        self.image_size.0 as f32 / (self.net_size as f32 - self.dw)
    }
}

/// spec §8.3 step 4: both confidence gates plus the `cx,cy,w,h -> x1,y1,x2,y2` decode.
///
/// `rows` is the flattened `[N, ROW_STRIDE]` block output; its length must be a
/// multiple of `ROW_STRIDE`. Output order is input order (NMS sorts afterwards).
pub fn filter_candidates(rows: &[f32]) -> Vec<Candidate> {
    assert_eq!(
        rows.len() % ROW_STRIDE,
        0,
        "YOLO output length must be a multiple of ROW_STRIDE"
    );

    rows.chunks_exact(ROW_STRIDE)
        .filter_map(|row| {
            let objectness = row[4];
            if objectness <= OBJECTNESS_THRESHOLD {
                return None;
            }

            let (class_index, class_probability) = row[5..]
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))?;
            let score = objectness * class_probability;
            if score <= CLASS_SCORE_THRESHOLD {
                return None;
            }

            let half_width = row[2] / 2.0;
            let half_height = row[3] / 2.0;
            Some(Candidate {
                xyxy: [
                    row[0] - half_width,
                    row[1] - half_height,
                    row[0] + half_width,
                    row[1] + half_height,
                ],
                class_index: class_index as u8,
                score,
            })
        })
        .collect()
}

/// Intersection over union of two `[x1, y1, x2, y2]` boxes; `0.0` when the union is 0.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let intersection_width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let intersection_height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Greedy NMS, descending score, `NMS_IOU_THRESHOLD`, survivors capped at `MAX_DET`.
///
/// DEVIATION(13): upstream runs **per-class** NMS (`inference.py:115` +
/// `yolov5_utils.py:261`, `agnostic=False`), which can emit duplicate overlapping boxes
/// for one balloon detected under two language classes. v1 deliberately runs
/// **class-agnostic** NMS to remove the duplicate at the source; the surviving box's
/// language is used as-is. See spec §14.13 / §15.1.
pub fn nms(candidates: Vec<Candidate>) -> Vec<Candidate> {
    let mut ordered = candidates;
    ordered.sort_by(|left, right| right.score.total_cmp(&left.score));

    let mut survivors = Vec::with_capacity(ordered.len().min(MAX_DET));
    // DEVIATION(13): suppress across all classes to eliminate duplicate balloon
    // detections; upstream performs per-class NMS (§14.13 / §15.1).
    for candidate in ordered {
        if survivors
            .iter()
            .all(|kept: &Candidate| iou(&candidate.xyxy, &kept.xyxy) <= NMS_IOU_THRESHOLD)
        {
            survivors.push(candidate);
            if survivors.len() == MAX_DET {
                break;
            }
        }
    }
    survivors
}

/// Letterboxed coords -> base-image coords (spec §8.3 step 4): multiply by
/// [`LetterboxGeometry::resize_ratio`], truncate to i32, then **clip to the image**.
///
/// spec §16.6 item 4: clamping `x1,y1 >= 0` and `x2,y2 <= image_size` is part of the
/// port (upstream's yolov5 `clip_coords`), and is required for `run()` to produce a
/// `PageDataRaw` that passes its own `validate()` on frame-overhanging detections.
pub fn rescale(candidates: &[Candidate], geometry: &LetterboxGeometry) -> Vec<RawBlock> {
    let ratio_x = geometry.image_size.0 as f32 / (geometry.net_size as f32 - geometry.dw);
    let ratio_y = geometry.image_size.1 as f32 / (geometry.net_size as f32 - geometry.dh);
    let max_x = geometry.image_size.0 as i32;
    let max_y = geometry.image_size.1 as i32;

    candidates
        .iter()
        .map(|candidate| {
            let x1 = (candidate.xyxy[0] * ratio_x) as i32;
            let y1 = (candidate.xyxy[1] * ratio_y) as i32;
            let x2 = (candidate.xyxy[2] * ratio_x) as i32;
            let y2 = (candidate.xyxy[3] * ratio_y) as i32;
            RawBlock {
                rect: pc_core::Rect::new(
                    x1.clamp(0, max_x),
                    y1.clamp(0, max_y),
                    x2.clamp(0, max_x),
                    y2.clamp(0, max_y),
                ),
                class_index: candidate.class_index,
                confidence: round3(candidate.score),
            }
        })
        .collect()
}

/// Round to 3 decimals (upstream `np.round(..., 3)`, spec §8.3 step 4) so golden JSON
/// is stable.
pub fn round3(value: f32) -> f32 {
    (value * 1_000.0).round() / 1_000.0
}

/// spec §8.3 step 4: `0 => English`, `1 => Japanese`, `2 => None`; any other index is
/// `None` plus a `WARN` (a model swap must be diagnosable, not silently mapped).
pub fn class_to_language(class_index: u8) -> Option<Language> {
    match class_index {
        0 => Some(Language::English),
        1 => Some(Language::Japanese),
        2 => None,
        _ => {
            tracing::warn!(class_index, "unknown detector class index");
            None
        }
    }
}

/// `filter_candidates` -> `nms` -> `rescale`, i.e. the whole of spec §8.3 step 4.
pub fn postprocess(rows: &[f32], geometry: &LetterboxGeometry) -> Vec<RawBlock> {
    rescale(&nms(filter_candidates(rows)), geometry)
}
