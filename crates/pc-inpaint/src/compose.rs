//! RGBA composition helpers — spec §16.38 item 3(h), arithmetic pinned by §16.9 items 13 and 15.
//!
//! **This is a FOURTH copy of arithmetic that already exists three times, and it is flagged rather
//! than quietly added.** `pc_mask::combine` holds the original (§16.9 items 13, 15),
//! `pc_denoise::composite` holds a second copy and `crates/pc-export/src/composite.rs` a third —
//! `pc-export`'s was missed by this module's earlier wording, which said "third". Measured with
//! `grep -n "fn blend_channel\|fn alpha_composite_over\|fn resize_nearest_rgba" -r crates/
//! --include=*.rs`: those three functions exist in **four** crates (`pc-mask`, `pc-denoise`,
//! `pc-export`, `pc-inpaint`); `composite_rgb` exists in **three** (`pc-export` has none).
//! §16.10 item 3 **pins** the duplication — "§1 rule 2 forbids
//! importing it from `pc-mask`; §16.10 item 3 pins the duplication instead" — with a
//! pixel-for-pixel agreement requirement, because §11.3 step 2 reproduces stage 3's composite
//! rather than trusting `_clean.png`. §16.38 item 3(h) puts this stage in the same position: it
//! rebuilds the cleaned page from the original plus the masks (`inpainting.py:149-157`), so
//! "`_clean_inpaint.png` is not derived from `_clean.png`", and it needs the same arithmetic.
//!
//! **Why this was not hoisted the way task L4 hoisted `gaussian`.** The two cases are not alike.
//! §16.10 item **1** placed `gaussian` in `pc-denoise` on a conditional ground and said in the
//! same breath that "a v1.5 hoist is a move plus a re-export" — so hoisting it discharges that
//! item rather than overturning it. §16.10 item **3** does the opposite: it *ratifies the
//! duplication as the design*, with no consolidation ticket, unlike §16.10 item 2 for `morph`
//! (which §16.38 item 16(a) then cashed in). Overturning a ratified pin is a design decision and
//! not a mechanical move, so this copy follows the pin's own precedent instead.
//!
//! **OPEN QUESTION for the architects, recorded here because this is where a reader lands.**
//! §16.38 item 16(a) called a third verbatim copy of `kernel`/`dilate` "not acceptable"; this is a
//! **fourth** verbatim copy of `blend_channel`/`alpha_composite_over`/`resize_nearest_rgba` (and a
//! third of `composite_rgb`), counting `pc-export`'s. Either the same reasoning applies and
//! `composite` owes a hoist into `pc-imageops` (superseding §16.10 item 3), or item 3's pin stands
//! and four copies is the ratified shape. Not decided here.
//!
//! **What the equivalence test does and does NOT hold together, because "holds the copies together"
//! over-claimed it.** `crates/pc-pipeline/tests/l4_composite_equivalence.rs` pins this module
//! against `pc_denoise::composite` and, for `blend_channel` and `composite_rgb`, also against
//! `pc_mask::combine` — the same mechanism, and the same host crate, task L1 used for
//! `l1_morph_equivalence.rs`. It does **not** reach `pc-export`'s copy at all: no test or production
//! code anywhere in the workspace references `pc-export`'s `composite` module or its re-exported
//! `blend_channel` / `alpha_composite_over` / `resize_nearest_rgba`, so that copy is covered by no
//! equivalence check anywhere. (Stated as a claim rather than as a grep command, because the obvious
//! command's pattern now matches this very sentence and would report its own disclosure as a hit.) Per function: `blend_channel` 3 of 4
//! copies, `composite_rgb` 3 of 3, `alpha_composite_over` 2 of 4, `resize_nearest_rgba` 2 of 4. The
//! gap is disclosed here rather than closed — closing it means new assertions in a frozen test file,
//! which is an architects' call, and it is tracked as the separate composite consolidation task.

use image::{Rgb, RgbImage, Rgba, RgbaImage};

/// `round(base * (1 - alpha) + color * alpha)` (§16.9 item 15).
pub fn blend_channel(base: u8, color: u8, alpha: f64) -> u8 {
    (f64::from(base) * (1.0 - alpha) + f64::from(color) * alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Nearest-neighbour resampling, pinned as `src = floor(dst * src_len / dst_len)` (§16.9 item 13,
/// §16.10 item 13). Identity when the sizes already match.
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

/// Alpha-composite `layer` onto `dst` at `at`, source-over; pixels landing outside `dst` are
/// dropped. `alpha_out = max(base_a, layer_a)` (§16.9 item 15).
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

/// Composite an RGBA layer over an RGB canvas of the same size. Panics on a dimension mismatch (a
/// programming error: the caller resizes first).
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
