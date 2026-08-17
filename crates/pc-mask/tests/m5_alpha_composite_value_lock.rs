//! Task M5 -- hand-derived value locks for `pc_mask::combine`'s two composition
//! primitives that no test in the workspace pinned by value: `composite_rgb` and
//! `alpha_composite_over`.
//!
//! **Why this file exists (measured, not argued).** Two independent mutation probes run
//! during the 2026-08-17 joint planning pass found the same hole from opposite
//! directions:
//!
//!   * Re-forking `pc_mask::combine::alpha_composite_over` to the **superseded**
//!     `alpha_out = max(base_a, layer_a)` rule -- the exact regression §16.45 item 4
//!     ratified the fix for -- turned **nothing** in `cargo test --workspace` red.
//!     `crates/pc-mask/tests/m56_run.rs:219`'s
//!     `alpha_composite_over_drops_layers_falling_outside_the_canvas` is the only
//!     `pc-mask` test that names the function at all, and it exercises only the
//!     out-of-bounds rule at a single positive offset with an opaque layer -- the one
//!     regime where every candidate `alpha_out` rule agrees.
//!   * Re-forking `composite_rgb`'s `a == 0` branch to return the layer instead of the
//!     canvas turned nothing in `crates/pc-pipeline/tests/l4_composite_equivalence.rs`
//!     red either, and there is **no hand-derived `composite_rgb` value lock anywhere in
//!     the workspace** (re-confirmed here by grepping `composite_rgb` across
//!     `crates/*/tests/`: `pc-denoise`'s two uses in `n4_run.rs` call the function to
//!     *produce* their own expected image, which is cookbook rule 7's forbidden shape,
//!     and `l4_composite_equivalence.rs` only compares three re-exports of one item
//!     against each other).
//!
//! **Every expected value below is hand-derived from the ratified formula and written out
//! as a literal.** Nothing is read back from `pc_mask::combine`, from `pc_imageops`, or
//! from a sibling test's output (cookbook rules 7 and 13). The arithmetic for each case is
//! written out in the doc comment above it so a reviewer can re-check it without running
//! anything.
//!
//! **Spec trace.**
//!   * `alpha_composite_over` -- **§16.45 item 4**: `out_a = sa + da*(1 - sa)` and
//!     `out_rgb[c] = round((src[c]*sa + dst[c]*da*(1 - sa)) / out_a)`, `sa = src_a/255`,
//!     `da = dst_a/255`, each channel clamped to `0..=255`; pixels landing outside the
//!     destination are dropped.
//!   * `composite_rgb` -- **§16.9 item 15**: `round(base * (1 - alpha) + color * alpha)`
//!     for the partial-alpha branch, with `a == 0` leaving the canvas pixel untouched and
//!     `a == 255` replacing it outright.
//!
//! This file is **additive** (cookbook rule 8 exit 1 -- a frozen-test addition, not an
//! amendment). It edits nothing.

use image::{Rgb, RgbImage, Rgba, RgbaImage};
use pc_mask::combine::{alpha_composite_over, composite_rgb};

// ------------------------------------------------------------------ composite_rgb

/// §16.9 item 15 plus the two exact branches, on one 4x1 canvas that reaches all three.
///
/// Canvas / layer / hand-derived expectation, per column:
///
/// * `x = 0` -- canvas `(200, 10, 90)`, layer `(20, 250, 0, a = 0)`.
///   `a == 0` means the layer contributes **nothing**: expect the canvas pixel back,
///   `(200, 10, 90)`. A branch returning the *layer* here would give `(20, 250, 0)`; a
///   branch that fell through to the lerp would give `round(200*1 + 20*0) = 200`,
///   `round(10*1 + 250*0) = 10`, `round(90*1 + 0*0) = 90` -- i.e. the same answer, so
///   this column discriminates "returns the layer" but not "skips the fast path". That
///   is deliberate and stated rather than overclaimed.
/// * `x = 1` -- canvas `(100, 50, 200)`, layer `(20, 250, 0, a = 255)`.
///   `a == 255` replaces outright: expect `(20, 250, 0)`, which is neither the canvas nor
///   any blend of the two.
/// * `x = 2` -- canvas `(0, 255, 128)`, layer `(20, 250, 0, a = 128)`, with
///   `alpha = 128/255` and `1 - alpha = 127/255`. `B` is the round-vs-truncation
///   discriminator here (truncation would give 63).
///
/// ```text
/// R = (0*127 + 20*128)/255 = 2560/255 = 10.0392 -> 10
/// G = (255*127 + 250*128)/255 = (32385 + 32000)/255 = 64385/255 = 252.4902 -> 252
/// B = (128*127 + 0*128)/255 = 16256/255 = 63.7490 -> 64
/// ```
///
/// * `x = 3` -- canvas `(12, 34, 56)`, layer `(255, 0, 128, a = 64)`, with
///   `alpha = 64/255` and `1 - alpha = 191/255`. `R` discriminates rounding here
///   (truncation would give 72).
///
/// ```text
/// R = (12*191 + 255*64)/255 = (2292 + 16320)/255 = 18612/255 = 72.9882 -> 73
/// G = (34*191 + 0*64)/255 = 6494/255 = 25.4667 -> 25
/// B = (56*191 + 128*64)/255 = (10696 + 8192)/255 = 18888/255 = 74.0706 -> 74
/// ```
///
/// What turns this red: the `a == 0` branch returning the layer; the `a == 255` branch
/// blending instead of replacing; `blend_channel` swapping `round` for a truncating cast;
/// the canvas and layer channels being transposed in the lerp.
#[test]
fn pc_mask_composite_rgb_matches_the_hand_derived_branch_table() {
    const CANVAS: [[u8; 3]; 4] = [[200, 10, 90], [100, 50, 200], [0, 255, 128], [12, 34, 56]];
    const LAYER: [[u8; 4]; 4] = [
        [20, 250, 0, 0],
        [20, 250, 0, 255],
        [20, 250, 0, 128],
        [255, 0, 128, 64],
    ];
    /// Hand-derived in this test's doc comment, one row per column above.
    const EXPECTED: [[u8; 3]; 4] = [[200, 10, 90], [20, 250, 0], [10, 252, 64], [73, 25, 74]];

    let canvas = RgbImage::from_fn(4, 1, |x, _| Rgb(CANVAS[x as usize]));
    let layer = RgbaImage::from_fn(4, 1, |x, _| Rgba(LAYER[x as usize]));

    let out = composite_rgb(&canvas, &layer);

    assert_eq!(
        out.dimensions(),
        (4, 1),
        "composite_rgb preserves the canvas size"
    );
    let actual: Vec<[u8; 3]> = (0..4).map(|x| out.get_pixel(x, 0).0).collect();
    assert_eq!(
        actual,
        EXPECTED.to_vec(),
        "composite_rgb must match the hand-derived §16.9 item 15 branch table"
    );
    // Anti-vacuity: the layer must actually have changed something, so an implementation
    // that returned its canvas argument unchanged cannot satisfy the row vector above by
    // accident on a degenerate input.
    assert_ne!(
        out, canvas,
        "three of the four columns must differ from the canvas"
    );
}

// ----------------------------------------------------------- alpha_composite_over

/// §16.45 item 4's real (Porter-Duff) source-over, four hand-derived cases chosen so that
/// **each one refutes a different wrong rule**, including the specific superseded
/// `alpha_out = max(base_a, layer_a)` rule §16.45 exists to prevent coming back.
///
/// `sa = layer_a/255`, `da = dst_a/255`, `dw = da*(1 - sa)`, `out_a = sa + dw`,
/// `out_rgb[c] = round((src[c]*sa + dst[c]*dw) / out_a)`, `out_alpha_u8 = round(out_a*255)`.
///
/// **Case A -- transparent destination, partial layer.** dst `(0, 0, 0, 0)`,
/// layer `(20, 250, 0, 128)`. `da = 0` so `dw = 0` and `out_a = sa`; every channel is
/// `(src*sa + 0)/sa = src` exactly. Expect **`(20, 250, 0, 128)`** -- the layer verbatim.
/// The pre-§16.45 rule gave `blend_channel(0, src, sa)` with `alpha_out = max(0, 128)`:
/// `(round(20*128/255), round(250*128/255), 0, 128) = (10, 125, 0, 128)`. This is the
/// exact defect §16.45 item 1 root-caused.
///
/// **Case B -- partial over partial, the interior case.** dst `(60, 120, 240, 90)`,
/// layer `(200, 40, 10, 150)`. Exact fractions over a common denominator of
/// `255*255 = 65025`: `sa = 38250/65025`, `dw = (90*105)/65025 = 9450/65025`,
/// `out_a = 47700/65025`.
///   out_alpha = `round(47700*255/65025) = round(12163500/65025) = round(187.0588)` -> **187**
///   R = `(200*38250 + 60*9450)/47700 = 8217000/47700 = 172.2642` -> **172**
///   G = `(40*38250 + 120*9450)/47700 = 2664000/47700 = 55.8491` -> **56**
///   B = `(10*38250 + 240*9450)/47700 = 2650500/47700 = 55.5660` -> **56**
/// Under the superseded rule this pixel would be `alpha_out = max(90, 150) = 150` and
/// `R = round(60*105/255 + 200*150/255) = round(142.353) = 142` -- both coordinates
/// differ, so this case alone refutes it.
///
/// **Case C -- opaque destination.** dst `(200, 10, 90, 255)`, layer `(0, 255, 128, 64)`.
/// `da = 1` so `dw = 1 - sa` and `out_a = 1` -> alpha **255**, and each channel reduces to
/// the plain lerp:
///   R = `(0*64 + 200*191)/255 = 38200/255 = 149.8039` -> **150** (truncation gives 149)
///   G = `(255*64 + 10*191)/255 = 18230/255 = 71.4902` -> **71**
///   B = `(128*64 + 90*191)/255 = 25382/255 = 99.5373` -> **100** (truncation gives 99)
///
/// **Case D -- fully opaque layer over a partial destination.** dst `(10, 20, 30, 77)`,
/// layer `(255, 128, 3, 255)`. The `a == 255` fast path replaces outright: expect
/// **`(255, 128, 3, 255)`**. A general-path implementation that forgot the fast path would
/// still land here (`sa = 1` -> `dw = 0`), so this case pins the *value*, not the branch.
///
/// What turns this red: reverting `alpha_out` to `max(base_a, layer_a)` (case A alpha
/// stays 128 but its RGB moves; case B moves on all four coordinates); ignoring the
/// destination's own alpha (case A and case B); dividing by `sa` instead of `out_a`
/// (case B); a truncating cast in place of `round` (case C).
#[test]
fn pc_mask_alpha_composite_over_matches_the_hand_derived_real_source_over() {
    /// `(destination pixel, layer pixel, hand-derived expected destination pixel)`.
    const CASES: [([u8; 4], [u8; 4], [u8; 4]); 4] = [
        ([0, 0, 0, 0], [20, 250, 0, 128], [20, 250, 0, 128]),
        ([60, 120, 240, 90], [200, 40, 10, 150], [172, 56, 56, 187]),
        ([200, 10, 90, 255], [0, 255, 128, 64], [150, 71, 100, 255]),
        ([10, 20, 30, 77], [255, 128, 3, 255], [255, 128, 3, 255]),
    ];
    /// Anti-vacuity literal: the four `out_alpha` bytes in case order, hand-derived above.
    /// `187` is obtainable from **no** other candidate rule considered by §16.45 item 4 --
    /// plain `base_a` would give 90, the superseded `max(base_a, layer_a)` would give 150 --
    /// so an empty or short loop cannot produce this vector.
    const EXPECTED_OUT_ALPHAS: [u8; 4] = [128, 187, 255, 255];

    let mut out_alphas = Vec::new();
    for (index, (dst_pixel, layer_pixel, expected)) in CASES.iter().enumerate() {
        let mut dst = RgbaImage::from_pixel(1, 1, Rgba(*dst_pixel));
        let layer = RgbaImage::from_pixel(1, 1, Rgba(*layer_pixel));

        alpha_composite_over(&mut dst, &layer, (0, 0));

        let actual = dst.get_pixel(0, 0).0;
        assert_eq!(
            actual, *expected,
            "case {index}: dst {dst_pixel:?} <- layer {layer_pixel:?} must give the \
             hand-derived real source-over result (§16.45 item 4)"
        );
        out_alphas.push(actual[3]);
    }
    assert_eq!(
        out_alphas,
        EXPECTED_OUT_ALPHAS.to_vec(),
        "the four out_alpha bytes must be the hand-derived source-over set; \
         187 in particular is neither base_a (90) nor max(base_a, layer_a) (150)"
    );
}

/// §16.45 item 4's clip rule: "pixels landing outside `dst` are dropped" -- **dropped**,
/// not clamped to the edge and not wrapped around to the opposite side.
///
/// Broader than `crates/pc-mask/tests/m56_run.rs:219`'s
/// `alpha_composite_over_drops_layers_falling_outside_the_canvas`, which checks one
/// positive offset and two pixels. This one covers a **negative** offset (where a
/// `u32`-cast or `saturating_sub` bug clamps instead of dropping), a fully-outside offset
/// on each axis, and asserts the identity of every one of the destination's 16 pixels
/// rather than a count.
///
/// The layer's red channel encodes its own index, `10*(3*y + x) + 1`, so the assertion
/// names *which* layer pixel landed *where*, not merely how many did.
///
/// Hand-derived for `at = (-1, -1)`: layer pixel `(x, y)` targets `(x - 1, y - 1)`, which
/// is on the 4x4 canvas only for `x >= 1 && y >= 1`. That is four pixels:
///   `(1,1)` index 4 -> red 41 -> dst `(0,0)`
///   `(2,1)` index 5 -> red 51 -> dst `(1,0)`
///   `(1,2)` index 7 -> red 71 -> dst `(0,1)`
///   `(2,2)` index 8 -> red 81 -> dst `(1,1)`
/// Every other destination pixel keeps its initial fully transparent `(0,0,0,0)`. A
/// wrapping implementation would additionally light `(3,3)` with the layer's index-0
/// pixel (red 1); a clamping one would pile the dropped column/row onto `(0, *)`/`(*, 0)`
/// and change the reds there.
///
/// What turns this red: replacing the signed bounds test with an unsigned cast; clamping
/// instead of dropping; an off-by-one that lets one extra layer row or column through;
/// a fully-outside offset writing anything at all.
#[test]
fn pc_mask_alpha_composite_over_drops_a_layer_that_lands_outside_the_destination() {
    /// Red channel of every destination pixel, row-major, after `at = (-1, -1)`.
    #[rustfmt::skip]
    const EXPECTED_RED_AFTER_NEGATIVE_OFFSET: [u8; 16] = [
        41, 51, 0, 0,
        71, 81, 0, 0,
         0,  0, 0, 0,
         0,  0, 0, 0,
    ];
    /// Alpha channel of the same 16 pixels: 255 exactly where a layer pixel landed.
    #[rustfmt::skip]
    const EXPECTED_ALPHA_AFTER_NEGATIVE_OFFSET: [u8; 16] = [
        255, 255, 0, 0,
        255, 255, 0, 0,
          0,   0, 0, 0,
          0,   0, 0, 0,
    ];

    let layer = RgbaImage::from_fn(3, 3, |x, y| {
        Rgba([(10 * (3 * y + x) + 1) as u8, 200, 100, 255])
    });

    let mut dst = RgbaImage::new(4, 4);
    alpha_composite_over(&mut dst, &layer, (-1, -1));
    assert_eq!(
        dst.pixels().map(|pixel| pixel.0[0]).collect::<Vec<u8>>(),
        EXPECTED_RED_AFTER_NEGATIVE_OFFSET.to_vec(),
        "at (-1, -1) exactly the four layer pixels with x >= 1 and y >= 1 may land, \
         and they must land at their own shifted positions -- not clamped, not wrapped"
    );
    assert_eq!(
        dst.pixels().map(|pixel| pixel.0[3]).collect::<Vec<u8>>(),
        EXPECTED_ALPHA_AFTER_NEGATIVE_OFFSET.to_vec(),
        "every destination pixel outside the clipped footprint stays fully transparent"
    );

    // Fully outside on either axis, in either direction: a byte-identical no-op.
    let untouched = RgbaImage::new(4, 4);
    for at in [(4_i32, 0_i32), (0, 4), (-3, 0), (0, -3), (-3, -3), (4, 4)] {
        let mut dst = RgbaImage::new(4, 4);
        alpha_composite_over(&mut dst, &layer, at);
        assert_eq!(
            dst, untouched,
            "a 3x3 layer at {at:?} lies entirely outside a 4x4 destination and must \
             change nothing"
        );
    }
}
