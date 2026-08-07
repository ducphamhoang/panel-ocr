//! Task D6 -- spec §8.3 step 5: U-Net mask postprocess + the v1 "Simple" refinement
//! (ported from koharu's `refine_segmentation_mask`).
//!
//! Implemented (task D6); signatures are frozen with the tests.

use crate::{detector::RawBlock, yolo::LetterboxGeometry};
use image::{GrayImage, Luma};
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
    let expected_len = usize::try_from(u64::from(width) * u64::from(height))
        .map_err(|_| StageError::InvalidInput("mask dimensions are too large".into()))?;
    if values.len() != expected_len {
        return Err(StageError::InvalidInput(format!(
            "mask has {} samples, expected {expected_len}",
            values.len()
        )));
    }

    let pixels = values
        .iter()
        .map(|value| (*value * 255.0).clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(width, height, pixels)
        .ok_or_else(|| StageError::InvalidInput("mask dimensions do not match its samples".into()))
}

/// Drop the letterbox padding: keep `mask[0..H-dh, 0..W-dw]` (spec §8.3 step 5).
/// Padding is applied right/bottom only, so this is a top-left crop.
pub fn crop_letterbox(mask: &GrayImage, dw: u32, dh: u32) -> Result<GrayImage, StageError> {
    if dw >= mask.width() || dh >= mask.height() {
        return Err(StageError::InvalidInput(format!(
            "letterbox padding ({dw}, {dh}) does not fit mask size {:?}",
            mask.dimensions()
        )));
    }
    Ok(image::imageops::crop_imm(mask, 0, 0, mask.width() - dw, mask.height() - dh).to_image())
}

/// Bilinear resize matching `cv2.INTER_LINEAR` (spec §8.3 step 5).
///
/// spec §16.6 item 6 pins the convention OpenCV actually uses: half-pixel-centre source
/// mapping, `src = (dst + 0.5) * (src_len / dst_len) - 0.5`, clamped at both borders,
/// with the two axes scaled independently.
pub fn resize_bilinear(image: &GrayImage, new_w: u32, new_h: u32) -> GrayImage {
    if image.dimensions() == (new_w, new_h) {
        return image.clone();
    }
    if new_w == 0 || new_h == 0 || image.width() == 0 || image.height() == 0 {
        return GrayImage::new(new_w, new_h);
    }

    let scale_x = image.width() as f32 / new_w as f32;
    let scale_y = image.height() as f32 / new_h as f32;
    GrayImage::from_fn(new_w, new_h, |x, y| {
        let source_x =
            ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, image.width().saturating_sub(1) as f32);
        let source_y =
            ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, image.height().saturating_sub(1) as f32);
        let x0 = source_x.floor() as u32;
        let y0 = source_y.floor() as u32;
        let x1 = (x0 + 1).min(image.width() - 1);
        let y1 = (y0 + 1).min(image.height() - 1);
        let weight_x = source_x - x0 as f32;
        let weight_y = source_y - y0 as f32;

        let top = f32::from(image.get_pixel(x0, y0).0[0]) * (1.0 - weight_x)
            + f32::from(image.get_pixel(x1, y0).0[0]) * weight_x;
        let bottom = f32::from(image.get_pixel(x0, y1).0[0]) * (1.0 - weight_x)
            + f32::from(image.get_pixel(x1, y1).0[0]) * weight_x;
        Luma([(top * (1.0 - weight_y) + bottom * weight_y)
            .round()
            .clamp(0.0, 255.0) as u8])
    })
}

pub(crate) fn prepare_mask(
    mask: &GrayImage,
    geometry: &LetterboxGeometry,
) -> Result<GrayImage, StageError> {
    if !geometry.dw.is_finite()
        || !geometry.dh.is_finite()
        || geometry.dw < 0.0
        || geometry.dh < 0.0
    {
        return Err(StageError::InvalidInput(
            "letterbox padding must be finite and non-negative".into(),
        ));
    }

    let cropped = crop_letterbox(mask, geometry.dw as u32, geometry.dh as u32)?;
    Ok(resize_bilinear(
        &cropped,
        geometry.image_size.0,
        geometry.image_size.1,
    ))
}

/// Dilation with an L1 (diamond) structuring element: output `p` is the maximum over
/// all `q` with `|dx| + |dy| <= radius`. spec §8.3 step 5, refinement step 4.
pub fn dilate_l1(mask: &GrayImage, radius: u32) -> GrayImage {
    if radius == 0 {
        return mask.clone();
    }

    let radius = i64::from(radius);
    GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        let x = i64::from(x);
        let y = i64::from(y);
        let mut maximum = 0_u8;
        for dy in -radius..=radius {
            let remaining_x = radius - dy.abs();
            let source_y = y + dy;
            if !(0..i64::from(mask.height())).contains(&source_y) {
                continue;
            }
            for dx in -remaining_x..=remaining_x {
                let source_x = x + dx;
                if (0..i64::from(mask.width())).contains(&source_x) {
                    maximum = maximum.max(mask.get_pixel(source_x as u32, source_y as u32).0[0]);
                }
            }
        }
        Luma([maximum])
    })
}

/// Rasterise the union of `rects` into a `size`-shaped 0/255 mask.
///
/// spec §16.6 item 5 (settled, not open): `x2`/`y2` are **exclusive**, exactly matching
/// `pc_core::Rect::to_crop` and every other rect operation in the codebase, so
/// `mask_coverage` (§8.3 step 6) and the masker (§10) agree on what a box covers.
pub fn rasterize_union(rects: &[Rect], size: (u32, u32)) -> GrayImage {
    let mut mask = GrayImage::new(size.0, size.1);
    for rect in rects {
        if let Some((x, y, width, height)) = rect.to_crop(size) {
            for pixel_y in y..y + height {
                for pixel_x in x..x + width {
                    mask.put_pixel(pixel_x, pixel_y, Luma([255]));
                }
            }
        }
    }
    mask
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
///
/// DEVIATION(12): v1 ships this "Simple" refinement instead of upstream's
/// `refine_mask`/`refine_undetected_mask` (top-k grey/Otsu masks + XOR-minimising merge +
/// hole filling). Decided in §15.2; `MaskRefineMode::Annotation` is the v1.5 door for a
/// full port.
pub fn refine_simple(
    mask: &GrayImage,
    geometry: &LetterboxGeometry,
    blocks: &[RawBlock],
) -> Result<GrayImage, StageError> {
    let prepared = prepare_mask(mask, geometry)?;
    refine_simple_prepared(&prepared, geometry.image_size, blocks)
}

pub(crate) fn refine_simple_prepared(
    resized: &GrayImage,
    image_size: (u32, u32),
    blocks: &[RawBlock],
) -> Result<GrayImage, StageError> {
    let expanded = blocks
        .iter()
        .map(|block| block.rect.pad(REFINE_EXPAND, image_size))
        .collect::<Vec<_>>();
    let in_bounds = rasterize_union(&expanded, image_size);
    let base = GrayImage::from_fn(image_size.0, image_size.1, |x, y| {
        let inside = in_bounds.get_pixel(x, y).0[0] != 0;
        let above_threshold = resized.get_pixel(x, y).0[0] > REFINE_THRESHOLD;
        Luma([if inside && above_threshold { 255 } else { 0 }])
    });
    let dilated = dilate_l1(&base, REFINE_DILATE_RADIUS);

    Ok(GrayImage::from_fn(image_size.0, image_size.1, |x, y| {
        if in_bounds.get_pixel(x, y).0[0] != 0 {
            *dilated.get_pixel(x, y)
        } else {
            Luma([0])
        }
    }))
}
