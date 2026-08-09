//! Nearest resampling + source-over composition -- the shared pixel arithmetic hoisted
//! here by spec §16.42 ("Composite-helper consolidation").
//!
//! The formulas are unchanged and stay pinned where they were pinned: §16.9 item 13 for
//! `resize_nearest_rgba`'s `src = floor(dst * src_len / dst_len)` mapping, §16.9 item 15
//! for `blend_channel`'s `round(base * (1 - alpha) + color * alpha)` and for
//! `alpha_composite_over`'s `alpha_out = max(base_a, layer_a)`.
//!
//! **What §16.42 changed, and what it did not.** §16.10 item 3 pinned the duplication of
//! these four functions across the stage crates rather than scheduling a hoist; §16.11
//! item 10 separately pinned `pc-export`'s later copy. §16.42 overturns both pins and
//! moves the arithmetic into this crate, exactly the move §16.38 item 16(a) already made
//! for `morph` and §16.38 item 20 for `gaussian`. §1 rule 2 (no stage-crate-to-stage-crate
//! dependency) is untouched and is not what changed: `pc-imageops` is not a stage crate.
//! Every one of the four stage crates keeps its own module path as a re-export, which is
//! what lets the frozen tests that name `pc_mask::combine::*`, `pc_denoise::composite::*`,
//! `pc_export::composite::*` and `pc_inpaint::compose::*` stay unedited.
//!
//! Nothing here touches `pc-config` or any stage type -- only `image` crate types and
//! primitives -- so §16.10 item 1 / §16.38 item 20's "no `pc-config` dependency,
//! independently benchmarkable" property survives by construction (§16.42 item 3).
//!
//! **Scope.** §16.42 item 8 authorises exactly these four functions and nothing else.
//! `pc-mask`'s `cleaned_image`, `text_layer`, `mask_overlay` and `build_combined_mask` are
//! masking *policy* and stay in `pc-mask`.
//!
//! Two cosmetic divergences between the former copies were resolved by §16.42 item 7:
//! `composite_rgb`'s panic message takes the shorter, crate-neutral
//! `"composition needs matching sizes: ..."` wording (`pc-mask`'s longer form embedded a
//! `pc-mask`-specific noun), and `resize_nearest_rgba`'s guards use the two-early-return
//! form the majority of the copies used. Neither is behavioural.

use image::{Rgb, RgbImage, Rgba, RgbaImage};

/// `round(base * (1 - alpha) + color * alpha)`, clamped to `[0, 255]` (§16.9 item 15).
pub fn blend_channel(base: u8, color: u8, alpha: f64) -> u8 {
    (f64::from(base) * (1.0 - alpha) + f64::from(color) * alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Nearest-neighbour resampling, pinned as `src = floor(dst * src_len / dst_len)`
/// (§16.9 item 13), so a 2x upscale of a binary mask is exactly 2x2 blocks
/// (§10.7(A)13). Identity when the sizes already match; an empty source yields a
/// fully transparent image of the requested size.
pub fn resize_nearest_rgba(mask: &RgbaImage, size: (u32, u32)) -> RgbaImage {
    if mask.dimensions() == size {
        return mask.clone();
    }
    if mask.width() == 0 || mask.height() == 0 {
        return RgbaImage::new(size.0, size.1);
    }
    let (source_w, source_h) = mask.dimensions();
    RgbaImage::from_fn(size.0, size.1, |x, y| {
        let source_x = ((u64::from(x) * u64::from(source_w)) / u64::from(size.0)) as u32;
        let source_y = ((u64::from(y) * u64::from(source_h)) / u64::from(size.1)) as u32;
        *mask.get_pixel(source_x.min(source_w - 1), source_y.min(source_h - 1))
    })
}

/// Alpha-composite `layer` onto `dst` at `at`, source-over; pixels landing outside `dst`
/// are dropped. `alpha_out = max(base_a, layer_a)` (§16.9 item 15).
pub fn alpha_composite_over(dst: &mut RgbaImage, layer: &RgbaImage, at: (i32, i32)) {
    let (width, height) = dst.dimensions();
    for (x, y, pixel) in layer.enumerate_pixels() {
        let target_x = i64::from(at.0) + i64::from(x);
        let target_y = i64::from(at.1) + i64::from(y);
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

/// Composite an RGBA layer over an RGB canvas of the same size. Panics on a dimension
/// mismatch (a programming error: the caller resizes first).
pub fn composite_rgb(canvas: &RgbImage, mask: &RgbaImage) -> RgbImage {
    assert_eq!(
        canvas.dimensions(),
        mask.dimensions(),
        "composition needs matching sizes: {:?} vs {:?}",
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
