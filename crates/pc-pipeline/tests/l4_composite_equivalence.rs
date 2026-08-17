//! Task L4 -- cross-crate equivalence of the RGBA composition arithmetic (spec §16.38 item 3(h),
//! §16.10 item 3, §16.9 items 13 and 15). **FROZEN.**
//!
//! WHY THIS FILE LIVES IN `pc-pipeline`, and why it exists at all.
//!
//! `pc_mask::combine` holds the original arithmetic (§16.9 items 13, 15). `pc_denoise::composite`
//! holds a second verbatim copy, which §16.10 item 3 **pins** rather than schedules for
//! consolidation: "§1 rule 2 forbids importing it from `pc-mask`; §16.10 item 3 pins the duplication
//! instead", because §11.3 step 2 reproduces stage 3's composite at full resolution and the two must
//! agree pixel-for-pixel. §16.38 item 3(h) puts `pc-inpaint` in the same position -- it rebuilds the
//! cleaned page from the original plus the masks (`inpainting.py:149-157`), so
//! "`_clean_inpaint.png` is not derived from `_clean.png`" -- so `pc_inpaint::compose` is a **fourth**
//! copy, not a third: `crates/pc-export/src/composite.rs` holds a further pre-existing copy of
//! `blend_channel` / `alpha_composite_over` / `resize_nearest_rgba`, which was missed by earlier
//! wording here. Measured with
//! `grep -n "fn blend_channel\|fn alpha_composite_over\|fn resize_nearest_rgba" -r crates/ --include=*.rs`:
//! those three exist in **four** crates (`pc-mask`, `pc-denoise`, `pc-export`, `pc-inpaint`), while
//! `composite_rgb` exists in **three** (`pc-mask`, `pc-denoise`, `pc-inpaint` -- `pc-export` has no
//! `composite_rgb`).
//!
//! **WHAT THIS FILE DOES AND DOES NOT COVER, stated per function because "stops them drifting" is
//! broader than the assertions below.** Coverage here is *not* all four copies:
//!
//!   * `blend_channel` -- checked across `pc_mask`, `pc_denoise`, `pc_inpaint`: **3 of 4**.
//!   * `composite_rgb` -- checked across `pc_mask`, `pc_denoise`, `pc_inpaint`: **3 of 3**, complete.
//!   * `alpha_composite_over` -- checked across `pc_denoise` and `pc_inpaint` only: **2 of 4**.
//!   * `resize_nearest_rgba` -- checked across `pc_denoise` and `pc_inpaint` only: **2 of 4**.
//!
//! `pc-export`'s copy is referenced by **no** test in the workspace: nothing outside `pc-export`
//! itself names its `composite` module or its re-exported `blend_channel` / `alpha_composite_over` /
//! `resize_nearest_rgba`, so that copy is uncovered by any equivalence check, here or elsewhere.
//! (Stated as a claim rather than as a grep command: the obvious pattern now matches this very
//! sentence, and a self-referential grep reports its own disclosure as a hit.) That gap is **disclosed, not closed** -- closing it means new assertions in this
//! frozen file (or a new one), which is an architects' call and is tracked as the separate composite
//! consolidation task. Do not read this header as a claim that the four copies cannot drift; the
//! accurate claim is that the rows enumerated above cannot drift.
//!
//! This file lives here for L1's reason, restated: it must
//! name two stage crates in one compilation unit, and §1 rule 2 forbids a stage crate depending on
//! another stage crate even through `[dev-dependencies]`, which would put the forbidden edge in the
//! workspace graph. `pc-pipeline` is "the ONE crate allowed to depend on every stage", so `pc-inpaint`
//! enters its `[dev-dependencies]` here and **no** new dependency edge is created anywhere.
//!
//! **OPEN QUESTION this file does NOT resolve.** §16.38 item 16(a) called a third verbatim copy of
//! `kernel`/`dilate` "not acceptable" and had it hoisted; this is a **fourth** verbatim copy of
//! `blend_channel` / `alpha_composite_over` / `resize_nearest_rgba` (and a third of `composite_rgb`),
//! counting `pc-export`'s. Either the same
//! reasoning applies and `composite` owes a hoist into `pc-imageops` (superseding §16.10 item 3), or
//! item 3's pin stands. L4 did not decide it -- see `crates/pc-inpaint/src/compose.rs`. Note the
//! asymmetry with `l1_morph_equivalence.rs`: that file's agreement half became a tautology once the
//! hoist landed, whereas this one is load-bearing today and stops being so only if the hoist happens.
//!
//! **What keeps this non-vacuous:** every expected value in `the_blend_matches_the_hand_computed_lerp`
//! is a literal computed by hand from §16.9 item 15's formula, so all three crates agreeing on a wrong
//! answer still turns this file red.
//!
//! **ADDITIVE, post-header (§16.42 item 9; cookbook rule 8 exit 1 -- a frozen-test addition, not
//! an amendment).** §16.42 brings `pc-export`'s copy of `blend_channel`, `resize_nearest_rgba` and
//! `alpha_composite_over` into the same cross-crate agreement checks below (it has no
//! `composite_rgb`, so that function's coverage is unchanged at 3 of 3). Nothing above this note
//! was edited; the per-function coverage table above therefore now undercounts `pc-export` by one
//! row each for the three functions it does have -- read the three new tests at the bottom of this
//! file, plus `crates/pc-export/tests/composite_value_lock.rs`, as the corrected count, not the
//! header's "2 of 4" / "3 of 4" prose, which item 10 of §16.42 explicitly leaves untouched as a
//! historical record of what task L4 covered. The same applies to the header's separate sentence
//! *"`pc-export`'s copy is referenced by **no** test in the workspace"* (a fresh-reader pass,
//! 2026-08-09, named this sentence specifically): it too is now stale -- three tests in this file
//! and the whole of `composite_value_lock.rs` reference it -- and is likewise left as historical
//! record rather than edited, per the same item 10 discipline.
//!
//! **2026-08-17 (Fable tie-break, after a joint architect + Senior Rust Engineer planning pass
//! in which the two disagreed): two tests deleted, two stripped-and-renamed.** Recorded here
//! rather than silently applied, so a reader who finds the old names cited elsewhere can see
//! where they went.
//!
//!   * DELETED `composite_rgb_agrees_across_pc_mask_pc_denoise_and_pc_inpaint` and
//!     `alpha_composite_over_agrees_across_pc_denoise_and_pc_inpaint_including_the_offset_clip`.
//!     Both compared three (resp. two) re-exports of one `pc_imageops::composite` item against
//!     each other; after §16.42's hoist neither comparison could fail, and neither test pinned
//!     a value (the `alpha_composite_over` one said so in its own doc comment). They were kept
//!     as re-fork tripwires until that tripwire role was **measured** to be discharged: with
//!     both deleted, a per-call-path re-fork of `pc_inpaint::compose::composite_rgb` (the
//!     `a == 0` branch returning the layer) and of `pc_inpaint::compose::alpha_composite_over`
//!     (reverted to §16.45's superseded `max(base_a, layer_a)` rule) each turn **both** halves
//!     of `crates/pc-testkit/tests/composite_morph_source_sites.rs`'s source-set gate red.
//!     `composite_rgb`'s replacement *value* coverage — which nothing in the workspace had —
//!     landed first as `crates/pc-mask/tests/m5_alpha_composite_value_lock.rs`.
//!   * STRIPPED-AND-RENAMED `pc_export_blend_channel_agrees_with_the_hand_computed_lerp_and_pc_mask`
//!     -> `pc_export_blend_channel_matches_the_hand_computed_lerp`, and
//!     `pc_export_resize_nearest_rgba_agrees_and_matches_the_hand_derived_floor_mapping`
//!     -> `pc_export_resize_nearest_rgba_matches_the_hand_derived_floor_mapping`. Only the
//!     now-identity cross-crate comparison was removed from each; every hand-derived row is
//!     kept, because those rows were measured NOT to duplicate
//!     `crates/pc-export/tests/composite_value_lock.rs` (2 of 8 blend rows overlap; the resize
//!     downscale case differs, `(3, 2)` here vs `(3, 1)` there). See each test's own comment.
//!   * **DELETED** `pc_export_alpha_composite_over_agrees_with_pc_denoise_including_the_offset_clip`
//!     (Fable re-examination, 2026-08-17). It was listed in the strip-and-rename group, but
//!     had **no hand-derived assertions to keep** — its whole body was the now-identity
//!     `pc_export` vs `pc_denoise` comparison plus an `assert_ne!` control, which the
//!     strip instruction would have reduced to asserting only "something changed" (cookbook
//!     rule 1's named defect). Its inclusion in the strip-and-rename group was a transcription
//!     slip: no position anywhere in the record (not the audit, not either Opus side's
//!     original position, not Fable's own stated grounds) argued for that disposition for
//!     this specific test; the architect's own original position placed it in the DELETE
//!     group, on measured grounds (0-red under a full §16.45 alpha-formula revert;
//!     `crates/pc-export/tests/composite_value_lock.rs` measured to catch that same
//!     regression). Fable's re-examination confirmed no unique coverage is lost:
//!     `composite_value_lock.rs` already hand-derives the `pc_export` path's clip behavior
//!     (`alpha_composite_over_fully_outside_bounds_is_a_complete_noop`,
//!     `alpha_composite_over_opaque_layer_partial_overlap_lands_exact_pixels`), and
//!     `crates/pc-mask/tests/m5_alpha_composite_value_lock.rs` hand-derives the shared
//!     implementation's clip rule and alpha-out formula. No rebuild-with-a-new-table
//!     alternative was adopted, to avoid triplicating that coverage.
//!   * Two further now-identity cross-crate comparisons in this file were **out of scope** for
//!     that ruling and are untouched: the `pc_denoise` vs `pc_inpaint` loop inside
//!     `resize_nearest_rgba_agrees_and_matches_the_hand_derived_floor_mapping`, and the
//!     three-crate loop inside `the_blend_matches_the_hand_computed_lerp_in_all_three_crates`.
//!     Both still carry real hand-derived assertions; only their "agrees"/"in all three
//!     crates" naming over-claims.

use image::{Rgba, RgbaImage};

/// §16.9 item 15: `round(base * (1 - alpha) + color * alpha)`, computed by hand.
///
/// `alpha` is `a / 255`, so with `base = 200` and `color = 0`:
///   * `a = 0`   -> `200`
///   * `a = 1`   -> `200 * 254/255 = 199.216 -> 199`
///   * `a = 64`  -> `200 * 191/255 = 149.804 -> 150`
///   * `a = 128` -> `200 * 127/255 =  99.608 -> 100`
///   * `a = 254` -> `200 *   1/255 =   0.784 ->   1`
///   * `a = 255` -> `0`
///
/// and with `base = 0`, `color = 255`, `a = 128`: `255 * 128/255 = 128.0 -> 128`.
#[test]
fn the_blend_matches_the_hand_computed_lerp_in_all_three_crates() {
    let table: &[(u8, u8, u8, u8)] = &[
        // (base, color, a, expected)
        (200, 0, 0, 200),
        (200, 0, 1, 199),
        (200, 0, 64, 150),
        (200, 0, 128, 100),
        (200, 0, 254, 1),
        (200, 0, 255, 0),
        (0, 255, 128, 128),
        (10, 250, 128, 130),
    ];

    for (base, color, a, expected) in table {
        let alpha = f64::from(*a) / 255.0;
        assert_eq!(
            pc_mask::combine::blend_channel(*base, *color, alpha),
            *expected,
            "pc_mask::combine::blend_channel({base}, {color}, {a}/255)"
        );
        assert_eq!(
            pc_denoise::composite::blend_channel(*base, *color, alpha),
            *expected,
            "pc_denoise::composite::blend_channel({base}, {color}, {a}/255)"
        );
        assert_eq!(
            pc_inpaint::compose::blend_channel(*base, *color, alpha),
            *expected,
            "pc_inpaint::compose::blend_channel({base}, {color}, {a}/255)"
        );
    }
}

/// §16.9 item 13 / §16.10 item 13: `src = floor(dst * src_len / dst_len)`.
///
/// The expected mapping is hand-derived rather than compared only across crates: a 4-wide source
/// resized to 8 must repeat each column exactly twice, and resized to 3 must sample columns 0, 1, 2.
#[test]
fn resize_nearest_rgba_agrees_and_matches_the_hand_derived_floor_mapping() {
    let source = RgbaImage::from_fn(4, 2, |x, y| Rgba([(x * 10) as u8, (y * 10) as u8, 0, 255]));

    let up = pc_inpaint::compose::resize_nearest_rgba(&source, (8, 4));
    assert_eq!(up.dimensions(), (8, 4));
    for x in 0..8_u32 {
        assert_eq!(
            up.get_pixel(x, 0).0[0],
            ((x / 2) * 10) as u8,
            "floor({x} * 4 / 8) = {}",
            x / 2
        );
    }

    let down = pc_inpaint::compose::resize_nearest_rgba(&source, (3, 2));
    assert_eq!(
        (0..3)
            .map(|x| down.get_pixel(x, 0).0[0])
            .collect::<Vec<_>>(),
        vec![0, 10, 20],
        "floor(x * 4 / 3) for x = 0,1,2 is 0,1,2"
    );

    for size in [(8_u32, 4_u32), (3, 2), (4, 2), (1, 1)] {
        assert_eq!(
            pc_denoise::composite::resize_nearest_rgba(&source, size),
            pc_inpaint::compose::resize_nearest_rgba(&source, size),
            "at {size:?}"
        );
    }
    assert_eq!(
        pc_inpaint::compose::resize_nearest_rgba(&source, (4, 2)),
        source,
        "identity when the sizes already match"
    );
}

// ---------------------------------------------------------------------------------------------
// ADDITIVE (§16.42 item 9): bring `pc_export::composite`'s `blend_channel`, `resize_nearest_rgba`
// and `alpha_composite_over` into the same cross-crate agreement this file already runs for the
// other three crates. `pc-export` has no `composite_rgb`, so that function's coverage is
// unaffected. These are NEW tests; nothing above this line was edited.
// ---------------------------------------------------------------------------------------------

/// `pc_export::composite::blend_channel` against the hand-computed §16.9 item 15 lerp table.
///
/// **Renamed and stripped 2026-08-17** (Fable tie-break), from
/// `pc_export_blend_channel_agrees_with_the_hand_computed_lerp_and_pc_mask`. The old name's
/// "and pc_mask" half was backed by an `assert_eq!(via_export, pc_mask::combine::blend_channel(..))`
/// that became an identity comparison once §16.42's hoist made both paths re-exports of
/// `pc_imageops::composite::blend_channel` — it could no longer fail, so the name claimed
/// more than the assertion verified (cookbook rule 1). That one line is removed; every
/// hand-derived row below is kept unchanged.
///
/// The rows are **not** redundant with `crates/pc-export/tests/composite_value_lock.rs`: only
/// 2 of these 8 rows appear there, measured row-by-row during the 2026-08-17 planning pass —
/// the near-boundary cases `(200, 0, 1, 199)` and `(200, 0, 254, 1)` are pinned here and
/// nowhere else. What would turn this red: a wrong literal, or `blend_channel` swapping
/// `round` for a truncating cast (rows `a = 1` and `a = 254` both discriminate).
#[test]
fn pc_export_blend_channel_matches_the_hand_computed_lerp() {
    let table: &[(u8, u8, u8, u8)] = &[
        (200, 0, 0, 200),
        (200, 0, 1, 199),
        (200, 0, 64, 150),
        (200, 0, 128, 100),
        (200, 0, 254, 1),
        (200, 0, 255, 0),
        (0, 255, 128, 128),
        (10, 250, 128, 130),
    ];

    for (base, color, a, expected) in table {
        let alpha = f64::from(*a) / 255.0;
        let via_export = pc_export::composite::blend_channel(*base, *color, alpha);
        assert_eq!(
            via_export, *expected,
            "pc_export::composite::blend_channel({base}, {color}, {a}/255)"
        );
    }
}

/// `pc_export::composite::resize_nearest_rgba` against §16.9 item 13's hand-derived
/// `src = floor(dst * src_len / dst_len)` mapping.
///
/// **Renamed and stripped 2026-08-17** (Fable tie-break), from
/// `pc_export_resize_nearest_rgba_agrees_and_matches_the_hand_derived_floor_mapping`. The
/// `for size in [...]` loop comparing `pc_export` against `pc_inpaint` became an identity
/// comparison once §16.42's hoist made both re-exports of one function, so the "agrees" in
/// the old name named something that could no longer fail (cookbook rule 1). That loop is
/// removed; every hand-derived assertion is kept unchanged.
///
/// **Not** redundant with `crates/pc-export/tests/composite_value_lock.rs`: that file's
/// downscale case is `(3, 1)`, this one is `(3, 2)` — a different mapping on the y axis —
/// measured during the 2026-08-17 planning pass. What would turn this red: `floor` becoming
/// `ceil` or `round` on either axis, or the identity fast path resampling anyway.
#[test]
fn pc_export_resize_nearest_rgba_matches_the_hand_derived_floor_mapping() {
    let source = RgbaImage::from_fn(4, 2, |x, y| Rgba([(x * 10) as u8, (y * 10) as u8, 0, 255]));

    let up = pc_export::composite::resize_nearest_rgba(&source, (8, 4));
    assert_eq!(up.dimensions(), (8, 4));
    for x in 0..8_u32 {
        assert_eq!(
            up.get_pixel(x, 0).0[0],
            ((x / 2) * 10) as u8,
            "floor({x} * 4 / 8) = {}",
            x / 2
        );
    }

    let down = pc_export::composite::resize_nearest_rgba(&source, (3, 2));
    assert_eq!(
        (0..3)
            .map(|x| down.get_pixel(x, 0).0[0])
            .collect::<Vec<_>>(),
        vec![0, 10, 20],
        "floor(x * 4 / 3) for x = 0,1,2 is 0,1,2"
    );

    assert_eq!(
        pc_export::composite::resize_nearest_rgba(&source, (4, 2)),
        source,
        "identity when the sizes already match"
    );
}
