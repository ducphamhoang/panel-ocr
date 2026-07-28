//! Task D6 -- spec §8.3 step 5: U-Net mask postprocess + the v1 "Simple" refinement
//! (ported from koharu's `refine_segmentation_mask`).
//!
//! Bodies are `todo!()` skeletons: signatures are frozen with the tests, D6 fills them.
#![allow(unused_variables)]

use crate::{detector::RawBlock, yolo::LetterboxGeometry};
use image::GrayImage;
use pc_core::{Rect, StageError};

/// `raw_mask[p] > 60` (strict) -- spec §8.3 step 5, refinement step 3.
pub const REFINE_THRESHOLD: u8 = 60;
/// `rect.pad(16)` -- upstream `expand_textwindow(expand_r=16)`, spec §8.3 step 5 step 1.
pub const REFINE_EXPAND: i32 = 16;
/// L1 (diamond) structuring element radius -- spec §8.3 step 5, refinement step 4.
pub const REFINE_DILATE_RADIUS: u32 = 3;

/// Raw U-Net probabilities -> 8-bit mask (spec §8.3 step 5, `postprocess_mask` with
/// `thresh=None`): multiply by 255, clamp to `[0.0, 255.0]`, then **truncate**.
///
/// spec §16.6 item 7: truncation (not rounding) matches upstream's
/// `(img * 255).astype(np.uint8)`; the clamp is the one deliberate correction, since
/// `astype` wraps out-of-range floats.
///
/// `values` must hold exactly `width * height` samples.
pub fn postprocess_mask(values: &[f32], width: u32, height: u32) -> Result<GrayImage, StageError> {
    todo!("D6: probability -> u8 mask")
}

/// Drop the letterbox padding: keep `mask[0..H-dh, 0..W-dw]` (spec §8.3 step 5).
/// Padding is applied right/bottom only, so this is a top-left crop.
pub fn crop_letterbox(mask: &GrayImage, dw: u32, dh: u32) -> Result<GrayImage, StageError> {
    todo!("D6: crop letterbox padding")
}

/// Bilinear resize matching `cv2.INTER_LINEAR` (spec §8.3 step 5).
///
/// spec §16.6 item 6 pins the convention OpenCV actually uses: half-pixel-centre source
/// mapping, `src = (dst + 0.5) * (src_len / dst_len) - 0.5`, clamped at both borders,
/// with the two axes scaled independently.
pub fn resize_bilinear(image: &GrayImage, new_w: u32, new_h: u32) -> GrayImage {
    todo!("D6: OpenCV-convention bilinear resize")
}

/// Dilation with an L1 (diamond) structuring element: output `p` is the maximum over
/// all `q` with `|dx| + |dy| <= radius`. spec §8.3 step 5, refinement step 4.
pub fn dilate_l1(mask: &GrayImage, radius: u32) -> GrayImage {
    todo!("D6: L1/diamond dilation")
}

/// Rasterise the union of `rects` into a `size`-shaped 0/255 mask.
///
/// spec §16.6 item 5 (settled, not open): `x2`/`y2` are **exclusive**, exactly matching
/// `pc_core::Rect::to_crop` and every other rect operation in the codebase, so
/// `mask_coverage` (§8.3 step 6) and the masker (§10) agree on what a box covers.
pub fn rasterize_union(rects: &[Rect], size: (u32, u32)) -> GrayImage {
    todo!("D6: rasterize the union of rects")
}

/// The whole of spec §8.3 step 5 for `MaskRefineMode::Simple`: crop the letterbox
/// padding off `mask`, resize it to `geometry.image_size`, then
///   1. `expanded = rect.pad(REFINE_EXPAND, image_size)` per block,
///   2. rasterise the union of the expanded rects as `in_bounds`,
///   3. `base[p] = 255` iff `in_bounds[p] != 0 && resized[p] > REFINE_THRESHOLD`,
///   4. dilate `base` with an L1 element of radius `REFINE_DILATE_RADIUS`,
///   5. clip back to `in_bounds` (zero outside).
///
/// With no blocks the refined mask is all-zero.
pub fn refine_simple(
    mask: &GrayImage,
    geometry: &LetterboxGeometry,
    blocks: &[RawBlock],
) -> Result<GrayImage, StageError> {
    todo!("D6: simple refinement")
}
