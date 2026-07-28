//! RGBA composition helpers (spec §11.3 steps 2 and 5, §16.10 item 3).
//!
//! Pinned **identically** to `pc_mask::combine`'s (§16.9 items 13 and 15). §11.3 step 2
//! reproduces Stage 3's clean output at full resolution rather than trusting
//! `_clean.png`, so the two crates' arithmetic must agree pixel-for-pixel. §1 rule 2
//! forbids importing it from `pc-mask`; §16.10 item 3 pins the duplication instead.

use image::{Rgb, RgbImage, Rgba, RgbaImage};

/// `round(base * (1 - alpha) + color * alpha)` (§16.9 item 15).
pub fn blend_channel(base: u8, color: u8, alpha: f64) -> u8 {
    (f64::from(base) * (1.0 - alpha) + f64::from(color) * alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Nearest-neighbour resampling, pinned as `src = floor(dst * src_len / dst_len)`
/// (§16.9 item 13, §16.10 item 13). Identity when the sizes already match.
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

/// Alpha-composite `layer` onto `dst` at `at`, source-over; pixels landing outside
/// `dst` are dropped. `alpha_out = max(base_a, layer_a)` (§16.9 item 15).
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
