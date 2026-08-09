//! PRE-HOIST value-lock for `pc_export::composite` (spec §16.42 item 6, binding sequencing
//! requirement). **FROZEN once green against the un-hoisted code.**
//!
//! §16.42 item 6 records that `pc-export`'s copy of `blend_channel` / `resize_nearest_rgba` /
//! `alpha_composite_over` is referenced by **no** test anywhere in the workspace (confirmed
//! again here: `grep -rn "pc_export::composite::" --include=*.rs .` outside `pc-export` itself
//! returns nothing) and mandates that a value-lock test land and be observed GREEN against the
//! **pre-hoist** implementation before the composite-helper hoist into `pc_imageops` begins.
//! A test written only after the hoist would compare `pc-export`'s (by-then identical-by-
//! construction, re-exported) functions against themselves or a sibling and pass vacuously --
//! this file exists to prove the CURRENT, independently-implemented copy is correct *before*
//! it is deleted and replaced with a re-export.
//!
//! Every expected value below is computed by hand from the formulas §16.9 items 13/15 pin, not
//! by running `pc_export::composite` once and pasting its output -- an expectation derived from
//! the artifact under test would prove nothing (cookbook rules 7/13).

use image::{Rgba, RgbaImage};
use pc_export::composite::{alpha_composite_over, blend_channel, resize_nearest_rgba};

/// §16.9 item 15: `round(base * (1 - alpha) + color * alpha)`, clamped to `[0, 255]`.
///
/// Hand-derived table, `alpha = a_u8 / 255`:
///   * `alpha = 0`   -> always `base` (the color contributes nothing).
///   * `alpha = 255` -> always `color` (the base contributes nothing).
///   * `base=100, color=200, a=128` -> `100*127/255 + 200*128/255 = 49.804 + 100.392 = 150.196
///     -> round = 150` (a genuine mid-alpha lerp point, not an edge case).
///   * `base=200, color=0, a=127` -> naive truncation gives `200*128/255 = 100.392 -> trunc 100`;
///     the pinned formula rounds to `100` too here, so a second boundary is also checked at
///     `a=128` where truncation (`200*127/255=99.608 -> trunc 99`) and rounding (`round=100`)
///     actually diverge -- this is the case that would catch a `as u8` cast standing in for
///     `.round()`.
#[test]
fn blend_channel_matches_the_hand_derived_formula() {
    let table: &[(u8, u8, u8, u8)] = &[
        // (base, color, alpha_u8, expected)
        (100, 200, 0, 100),   // alpha=0 -> base
        (100, 200, 255, 200), // alpha=255 -> color
        (100, 200, 128, 150), // midpoint: 100*127/255 + 200*128/255 = 150.196 -> 150
        (200, 0, 128, 100),   // round(200*127/255) = round(99.608) = 100, truncation would give 99
        (0, 255, 128, 128),   // round(255*128/255) = round(128.0) = 128
        (10, 250, 64, 70),    // round(10*191/255 + 250*64/255) = round(7.49+62.75)=round(70.235)=70
    ];

    for (base, color, a, expected) in table {
        let alpha = f64::from(*a) / 255.0;
        assert_eq!(
            blend_channel(*base, *color, alpha),
            *expected,
            "blend_channel({base}, {color}, {a}/255) expected {expected}"
        );
    }
}

/// §16.9 item 13: `src = floor(dst * src_len / dst_len)`. Identity when sizes already match.
///
/// Source is a 4x2 image whose red channel encodes the source column (`x*10`) and whose green
/// channel encodes the source row (`y*10`), so every sampled pixel names exactly which source
/// pixel nearest-neighbour resolved to.
#[test]
fn resize_nearest_rgba_matches_the_hand_derived_floor_mapping_on_upscale() {
    let source = RgbaImage::from_fn(4, 2, |x, y| Rgba([(x * 10) as u8, (y * 10) as u8, 0, 255]));

    // Upscale 4x2 -> 8x4: floor(x*4/8) for x in 0..8 is 0,0,1,1,2,2,3,3.
    let up = resize_nearest_rgba(&source, (8, 4));
    assert_eq!(up.dimensions(), (8, 4));
    let expected_src_x = [0u32, 0, 1, 1, 2, 2, 3, 3];
    for (x, &src_x) in expected_src_x.iter().enumerate() {
        assert_eq!(
            up.get_pixel(x as u32, 0).0[0],
            (src_x * 10) as u8,
            "x={x}: floor({x} * 4 / 8) should be {src_x}"
        );
    }
    // floor(y*2/4) for y in 0..4 is 0,0,1,1.
    let expected_src_y = [0u32, 0, 1, 1];
    for (y, &src_y) in expected_src_y.iter().enumerate() {
        assert_eq!(
            up.get_pixel(0, y as u32).0[1],
            (src_y * 10) as u8,
            "y={y}: floor({y} * 2 / 4) should be {src_y}"
        );
    }
}

#[test]
fn resize_nearest_rgba_matches_the_hand_derived_floor_mapping_on_downscale() {
    let source = RgbaImage::from_fn(4, 2, |x, y| Rgba([(x * 10) as u8, (y * 10) as u8, 0, 255]));

    // Downscale 4x2 -> 3x1: floor(x*4/3) for x in 0..3 is 0,1,2.
    let down = resize_nearest_rgba(&source, (3, 1));
    assert_eq!(down.dimensions(), (3, 1));
    let expected_src_x = [0u32, 1, 2];
    for (x, &src_x) in expected_src_x.iter().enumerate() {
        assert_eq!(
            down.get_pixel(x as u32, 0).0[0],
            (src_x * 10) as u8,
            "x={x}: floor({x} * 4 / 3) should be {src_x}"
        );
    }
}

#[test]
fn resize_nearest_rgba_is_identity_when_sizes_already_match() {
    let source = RgbaImage::from_fn(4, 2, |x, y| Rgba([(x * 10) as u8, (y * 10) as u8, 7, 255]));
    let same = resize_nearest_rgba(&source, (4, 2));
    assert_eq!(same, source);
}

/// A layer pasted fully outside the canvas must be a complete no-op: not one destination pixel
/// changes. What would turn this red: the bounds check dropping the wrong axis, or an off-by-one
/// that lets one row/column of the layer bleed onto the canvas edge.
#[test]
fn alpha_composite_over_fully_outside_bounds_is_a_complete_noop() {
    let make_dst =
        || RgbaImage::from_fn(6, 6, |x, y| Rgba([(x * 40) as u8, (y * 40) as u8, 5, 250]));
    let layer = RgbaImage::from_fn(3, 3, |_, _| Rgba([255, 255, 255, 255]));

    for at in [(-10, -10), (100, 100), (10, 0), (0, 10), (-3, 2)] {
        let mut dst = make_dst();
        alpha_composite_over(&mut dst, &layer, at);
        assert_eq!(dst, make_dst(), "at offset {at:?} must be a no-op");
    }
}

/// An opaque layer pasted partially onscreen must land exact expected pixel values: the
/// dst-only region keeps its original pixels, and the overlap region is fully replaced (a=255
/// takes the `*target = layer pixel` branch, not a blend).
#[test]
fn alpha_composite_over_opaque_layer_partial_overlap_lands_exact_pixels() {
    let dst_pixel = Rgba([10, 20, 30, 200]);
    let mut dst = RgbaImage::from_pixel(6, 6, dst_pixel);
    // 4x4 opaque red layer pasted at (4, 4): only the (4,4)-(5,5) 2x2 corner of dst overlaps.
    let layer = RgbaImage::from_pixel(4, 4, Rgba([255, 0, 0, 255]));
    alpha_composite_over(&mut dst, &layer, (4, 4));

    // Overlap region: fully replaced by the opaque layer color.
    for (x, y) in [(4u32, 4u32), (5, 4), (4, 5), (5, 5)] {
        assert_eq!(
            *dst.get_pixel(x, y),
            Rgba([255, 0, 0, 255]),
            "overlap pixel ({x},{y}) must be fully replaced by the opaque layer"
        );
    }
    // dst-only region (untouched by the layer): unchanged original pixel.
    for (x, y) in [(0u32, 0u32), (3, 3), (0, 5), (5, 0)] {
        assert_eq!(
            *dst.get_pixel(x, y),
            dst_pixel,
            "dst-only pixel ({x},{y}) must be unchanged"
        );
    }
}

/// A fully opaque layer entirely onscreen must exactly overwrite every destination pixel it
/// covers, canvas-sized so every pixel is "overlap".
#[test]
fn alpha_composite_over_opaque_layer_fully_onscreen_replaces_every_pixel() {
    let mut dst = RgbaImage::from_fn(4, 4, |x, y| Rgba([(x * 50) as u8, (y * 50) as u8, 9, 255]));
    let layer = RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 255]));
    alpha_composite_over(&mut dst, &layer, (0, 0));

    for y in 0..4u32 {
        for x in 0..4u32 {
            assert_eq!(
                *dst.get_pixel(x, y),
                Rgba([1, 2, 3, 255]),
                "pixel ({x},{y}) must be fully replaced"
            );
        }
    }
}

/// Added 2026-08-09 after a fresh-reader pass found the alpha=0 skip and the partial-alpha
/// blend branch (the one line most likely to diverge, `alpha_out = max(base_a, layer_a)`)
/// had no hand-derived coverage for `pc-export` specifically — only an agreement check,
/// which §16.42 item 6 explicitly says goes vacuous once the hoist lands. A fully
/// transparent source pixel (`a == 0`) must be skipped entirely: `dst` unchanged.
#[test]
fn alpha_composite_over_fully_transparent_layer_pixel_is_skipped() {
    let mut dst = RgbaImage::from_fn(3, 3, |x, y| Rgba([(x * 40) as u8, (y * 40) as u8, 7, 200]));
    let before = dst.clone();
    let layer = RgbaImage::from_pixel(3, 3, Rgba([9, 9, 9, 0]));
    alpha_composite_over(&mut dst, &layer, (0, 0));
    assert_eq!(
        dst, before,
        "a fully transparent layer pixel must not change dst at all"
    );
}

/// Partial-alpha blend, hand-derived from the pinned formulas (§16.9 item 15):
/// `blend_channel(base, color, alpha) = round(base*(1-alpha) + color*alpha)`,
/// `alpha_out = max(base_a, layer_a)`. base = (100, 50, 200, 180), layer = (20, 250, 0, 128),
/// alpha = 128/255 ≈ 0.501960784:
///   R: round(100*0.498039216 + 20*0.501960784)  = round(59.843137) = 60
///   G: round(50*0.498039216 + 250*0.501960784)  = round(150.392157) = 150
///   B: round(200*0.498039216 + 0*0.501960784)   = round(99.607843)  = 100
///   A: max(180, 128) = 180
///
/// `base_a` (180) is the larger of the two here, so this case alone cannot distinguish
/// `alpha_out = max(base_a, layer_a)` from a bug that just writes `alpha_out = base_a` --
/// see the second test below, added after a fresh-reader pass named exactly this gap.
#[test]
fn alpha_composite_over_partial_alpha_blend_matches_the_hand_derived_result() {
    let mut dst = RgbaImage::from_pixel(1, 1, Rgba([100, 50, 200, 180]));
    let layer = RgbaImage::from_pixel(1, 1, Rgba([20, 250, 0, 128]));
    alpha_composite_over(&mut dst, &layer, (0, 0));
    assert_eq!(
        *dst.get_pixel(0, 0),
        Rgba([60, 150, 100, 180]),
        "partial-alpha blend must match the hand-derived per-channel result and alpha_out = max(base_a, layer_a)"
    );
}

/// Same RGB computation as above (identical `base`/`color`/`alpha` for R/G/B), but with
/// `base_a` (100) SMALLER than `layer_a` (128), so `alpha_out = max(100, 128) = 128`. A
/// bug that writes `alpha_out = base_a` (dropping the layer's alpha entirely) would give
/// 100 here and pass the test above unchanged, which is exactly why this case exists.
#[test]
fn alpha_composite_over_partial_alpha_blend_alpha_out_is_the_max_not_just_base_alpha() {
    let mut dst = RgbaImage::from_pixel(1, 1, Rgba([100, 50, 200, 100]));
    let layer = RgbaImage::from_pixel(1, 1, Rgba([20, 250, 0, 128]));
    alpha_composite_over(&mut dst, &layer, (0, 0));
    assert_eq!(
        *dst.get_pixel(0, 0),
        Rgba([60, 150, 100, 128]),
        "alpha_out must be max(base_a, layer_a) = 128, not base_a = 100"
    );
}
