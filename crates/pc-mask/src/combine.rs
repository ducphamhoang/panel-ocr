//! Task M5 -- composition: the RGBA combined mask, the cleaned image, the text layer
//! and the debug overlay (spec §10.3 step 4, §10.7(A)12/13, §15.3, §16.9 items 13-15).
//!
//! The small pure helpers have fully pinned arithmetic; the four composition entry points
//! are built on them (task M5), with signatures frozen by the tests.

use crate::fit::Fitment;
use image::{DynamicImage, GenericImageView, Rgb, RgbImage, Rgba, RgbaImage};

/// spec §16.9 item 13 -- nearest-neighbour resampling, pinned as
/// `src = floor(dst * src_len / dst_len)`, so a 2x upscale of a binary mask is exactly
/// 2x2 blocks (§10.7(A)13). Identity when the sizes already match.
pub fn resize_nearest_rgba(mask: &RgbaImage, size: (u32, u32)) -> RgbaImage {
    if mask.dimensions() == size || mask.width() == 0 || mask.height() == 0 {
        return if mask.dimensions() == size {
            mask.clone()
        } else {
            RgbaImage::new(size.0, size.1)
        };
    }
    let (source_w, source_h) = mask.dimensions();
    RgbaImage::from_fn(size.0, size.1, |x, y| {
        let source_x = ((u64::from(x) * u64::from(source_w)) / u64::from(size.0)) as u32;
        let source_y = ((u64::from(y) * u64::from(source_h)) / u64::from(size.1)) as u32;
        *mask.get_pixel(source_x.min(source_w - 1), source_y.min(source_h - 1))
    })
}

/// spec §10.3 step 4 -- alpha-composite `layer` onto `dst` at `at`, source-over.
/// Named for what it is even though every alpha here is 0 or 255 (making it equivalent
/// to a paste); pixels landing outside `dst` are dropped.
pub fn alpha_composite_over(dst: &mut RgbaImage, layer: &RgbaImage, at: (i32, i32)) {
    let (width, height) = dst.dimensions();
    for (x, y, pixel) in layer.enumerate_pixels() {
        let target_x = at.0 as i64 + i64::from(x);
        let target_y = at.1 as i64 + i64::from(y);
        if target_x < 0
            || target_y < 0
            || target_x >= i64::from(width)
            || target_y >= i64::from(height)
        {
            continue;
        }
        let Rgba([r, g, b, a]) = *pixel;
        if a == 0 {
            continue;
        }
        let target = dst.get_pixel_mut(target_x as u32, target_y as u32);
        if a == 255 {
            *target = Rgba([r, g, b, 255]);
            continue;
        }
        let alpha = f64::from(a) / 255.0;
        let Rgba([br, bg, bb, ba]) = *target;
        *target = Rgba([
            blend_channel(br, r, alpha),
            blend_channel(bg, g, alpha),
            blend_channel(bb, b, alpha),
            ba.max(a),
        ]);
    }
}

/// `round(base * (1 - alpha) + color * alpha)` (§16.9 item 15).
pub fn blend_channel(base: u8, color: u8, alpha: f64) -> u8 {
    (f64::from(base) * (1.0 - alpha) + f64::from(color) * alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// `r == g == b` -- §15.3's condition for writing the cleaned image as `L`.
pub fn is_achromatic(color: [u8; 3]) -> bool {
    color[0] == color[1] && color[1] == color[2]
}

/// §15.3 / §16.9 item 14: the composited RGB result becomes `L` when the source canvas
/// was grayscale **and** every fill colour used was achromatic; otherwise it stays RGB.
/// Applied to the in-memory value, so memory mode and disk mode agree pixel-for-pixel.
pub fn to_output_mode(composited: RgbImage, prefer_gray: bool) -> DynamicImage {
    if prefer_gray {
        DynamicImage::ImageLuma8(image::DynamicImage::ImageRgb8(composited).to_luma8())
    } else {
        DynamicImage::ImageRgb8(composited)
    }
}

/// spec §10.3 step 4 -- the combined mask: an RGBA image of `size`, all `(0,0,0,0)`,
/// with one opaque `median_color` layer per fitted region alpha-composited at the
/// fitment's `coords`, **in region order** (so a later region wins in an overlap).
/// Fitments with `mask: None` contribute nothing.
pub fn build_combined_mask(fitments: &[Fitment], size: (u32, u32)) -> RgbaImage {
    let mut combined = RgbaImage::new(size.0, size.1);
    for fitment in fitments {
        if let Some(mask) = &fitment.mask {
            let layer = mask_layer_rgba(mask, fitment.median_color);
            alpha_composite_over(&mut combined, &layer, fitment.coords);
        }
    }
    combined
}

/// spec §10.3 step 4 -- the cleaned image: resize `mask` to the canvas with
/// nearest-neighbour if needed, composite it over the canvas using the mask's own
/// alpha, then apply §15.3's output-mode rule (`all_achromatic` is "every
/// `median_color` used was achromatic"; the canvas being grayscale is read off
/// `canvas` itself).
pub fn cleaned_image(
    canvas: &DynamicImage,
    mask: &RgbaImage,
    all_achromatic: bool,
) -> DynamicImage {
    let resized_mask = resize_nearest_rgba(mask, canvas.dimensions());
    let composited = composite_rgb(&canvas.to_rgb8(), &resized_mask);
    to_output_mode(
        composited,
        matches!(
            canvas,
            DynamicImage::ImageLuma8(_) | DynamicImage::ImageLumaA8(_)
        ) && all_achromatic,
    )
}

/// spec §10.3 step 4 -- the text layer: a transparent RGBA canvas of the canvas's size,
/// with the canvas's own pixels kept only where the (nearest-neighbour-resized) mask is
/// opaque. That is, what the mask covers *is* the text.
pub fn text_layer(canvas: &DynamicImage, mask: &RgbaImage) -> RgbaImage {
    let resized_mask = resize_nearest_rgba(mask, canvas.dimensions());
    let source = canvas.to_rgb8();
    RgbaImage::from_fn(source.width(), source.height(), |x, y| {
        if resized_mask.get_pixel(x, y).0[3] == 0 {
            Rgba([0, 0, 0, 0])
        } else {
            let Rgb([r, g, b]) = *source.get_pixel(x, y);
            Rgba([r, g, b, 255])
        }
    })
}

/// spec §10.3 step 4 / §16.9 item 15 -- the only debug visualisation in v1: `base` with
/// the mask recoloured to `debug_mask_color` and blended in at that colour's alpha,
/// wherever the mask is opaque. RGB, base-image size, no text rendering.
pub fn mask_overlay(base: &DynamicImage, mask: &RgbaImage, debug_mask_color: [u8; 4]) -> RgbImage {
    let resized_mask = resize_nearest_rgba(mask, base.dimensions());
    let base = base.to_rgb8();
    let alpha = f64::from(debug_mask_color[3]) / 255.0;
    RgbImage::from_fn(base.width(), base.height(), |x, y| {
        let Rgb([r, g, b]) = *base.get_pixel(x, y);
        if resized_mask.get_pixel(x, y).0[3] == 0 {
            Rgb([r, g, b])
        } else {
            Rgb([
                blend_channel(r, debug_mask_color[0], alpha),
                blend_channel(g, debug_mask_color[1], alpha),
                blend_channel(b, debug_mask_color[2], alpha),
            ])
        }
    })
}

/// An opaque `color` wherever `mask` is set, transparent elsewhere -- the per-fitment
/// layer `build_combined_mask` composites.
pub fn mask_layer_rgba(mask: &pc_imageops::BinaryMask, color: [u8; 3]) -> RgbaImage {
    let (width, height) = mask.dimensions();
    RgbaImage::from_fn(width, height, |x, y| {
        if mask.get(x, y) {
            Rgba([color[0], color[1], color[2], 255])
        } else {
            Rgba([0, 0, 0, 0])
        }
    })
}

/// Composite an RGBA mask (alpha 0 or 255) over an RGB canvas of the same size.
pub fn composite_rgb(canvas: &RgbImage, mask: &RgbaImage) -> RgbImage {
    assert_eq!(
        canvas.dimensions(),
        mask.dimensions(),
        "cleaned-image composition needs matching sizes: {:?} vs {:?}",
        canvas.dimensions(),
        mask.dimensions()
    );
    RgbImage::from_fn(canvas.width(), canvas.height(), |x, y| {
        let Rgba([r, g, b, a]) = *mask.get_pixel(x, y);
        if a == 0 {
            return *canvas.get_pixel(x, y);
        }
        if a == 255 {
            return Rgb([r, g, b]);
        }
        let alpha = f64::from(a) / 255.0;
        let Rgb([br, bg, bb]) = *canvas.get_pixel(x, y);
        Rgb([
            blend_channel(br, r, alpha),
            blend_channel(bg, g, alpha),
            blend_channel(bb, b, alpha),
        ])
    })
}
