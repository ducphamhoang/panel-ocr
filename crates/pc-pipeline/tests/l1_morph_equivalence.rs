//! Task L1 -- cross-crate equivalence of the growth morphology (spec §16.38 item 16(a),
//! §16.10 item 2, §16.9 items 5, 6, §10.3 step 6, §11.3 step 4). **FROZEN.**
//!
//! WHY THIS FILE LIVES IN `pc-pipeline`. It must name **both** `pc_mask::grow` and
//! `pc_denoise::morph` in the same compilation unit, and §1 rule 2 forbids a stage crate
//! depending on another stage crate -- even through `[dev-dependencies]`, which would put
//! the forbidden edge in the workspace graph. `pc-pipeline` is "the ONE crate allowed to
//! depend on every stage" (its own `Cargo.toml` note), so this test adds **no** new
//! dependency edge anywhere. §16.38 item 16(a)'s acceptance gate additionally requires
//! `git diff --stat` over `crates/pc-mask/tests/` and `crates/pc-denoise/tests/` to be
//! empty at the end of L1, which putting the new file here satisfies by construction.
//!
//! WHAT THIS FILE IS FOR. It once had two halves; it now has one.
//!
//! * The **oracle** half (`*_match_a_hand_derived_*_oracle`, plus the three
//!   `both_crates_dilate_*` footprint tests) is what keeps this file non-vacuous after
//!   the hoist, per cookbook rules 7 and 13: every expected value below is hand-derived
//!   from OpenCV's documented `getStructuringElement(MORPH_ELLIPSE)` formula and from
//!   §10.3 step 6's small-diameter branch, written out as literals, and asserted against
//!   **both** crates' public paths. Nothing here is read back out of the code under test.
//! * The **agreement** half (`pc_mask_and_pc_denoise_*_agree_*`) was the pre-hoist
//!   baseline: before L1, `pc_mask::grow::{Kernel, kernel, dilate}` and
//!   `pc_denoise::morph::{Kernel, kernel, dilate}` were two separate verbatim copies
//!   (§16.10 item 2), and those two tests proved they agreed cell-for-cell and
//!   pixel-for-pixel rather than merely "both existing". After the hoist they compared
//!   one item reached by two paths, and were kept only as re-fork tripwires.
//!   **Both were removed 2026-08-17** (Fable tie-break, after a joint architect +
//!   Senior Rust Engineer planning pass in which the two disagreed). This paragraph is
//!   kept rather than deleted, per this project's supersession convention: a future
//!   reader who finds those two names cited elsewhere needs to know where they went.
//!   `pc_mask_and_pc_denoise_kernels_agree_cell_for_cell_...` was undisputed by both
//!   sides -- it is logically subsumed by the two per-path kernel oracles below, and
//!   neither side's probes ever saw it fire alone. `pc_mask_and_pc_denoise_dilate_agree_
//!   pixel_for_pixel_...` was the disputed one, and it was removed only after the
//!   structural re-fork class it guarded was **measured** to be caught by
//!   `crates/pc-testkit/tests/composite_morph_source_sites.rs`'s source-set gate: a
//!   re-fork of `pc_denoise::morph::dilate` diverging at thickness >= 3 turns both halves
//!   of that gate red, plus `crates/pc-denoise/tests/n3_noise_mask.rs`'s
//!   `fade_mask_grows_then_fades_a_single_dot`. Note what that gate does **not** do: it
//!   catches a re-fork *structurally* (a new definition site, or a dropped `pub use`), not
//!   a behavioural edit inside `pc_imageops::morph` itself -- the two per-path oracles
//!   below remain the only thing covering that, which is why they stay.
//!
//! Independent corroboration of the oracle table, so it is not merely self-consistent:
//! the counts it predicts for thicknesses 1, 2, 3, 4 are the same 5 / 21 / 33 / 57 that
//! `crates/pc-mask/tests/m2_grow.rs` pins by hand, and the count for thickness 5 is the
//! same 89 that `crates/pc-denoise/tests/n3_noise_mask.rs` pins by hand. Thickness 0 is
//! likewise already pinned by both -- `m2_grow.rs` asserts `kernel_rows(0) == vec![vec![1]]`
//! and `n3_noise_mask.rs` asserts `identity.as_cells() == &[1]`. Thicknesses 6 and 7 are
//! pinned by no existing test and are new coverage.
//!
//! What would turn each test here red is named in the test body, per cookbook rule 6.

use pc_imageops::BinaryMask;

/// The widest thickness this file's oracle table covers.
const MAX_THICKNESS: u32 = 7;

/// Hand-derived expected structuring elements, as row-major `0`/`1` cell matrices.
///
/// Index is `thickness`; `diameter = thickness * 2 + 1`.
///
/// Thicknesses 0-2 (`diameter <= 5`) come from §10.3 step 6's **small branch**: a full
/// square of 1s with the four corners zeroed, except `diameter == 1` where the "corners"
/// *are* the centre, so the element is the single centre pixel and dilation is the
/// identity (§16.9 item 5). Those three are written out as explicit matrices.
///
/// Thicknesses 3-7 come from OpenCV `MORPH_ELLIPSE`:
/// `dx = round(c * sqrt((r*r - dy*dy) / (r*r)))` with `c == r == thickness`, i.e.
/// `dx = round(sqrt(r*r - dy*dy))`, row `i` set on `[c - dx, c + dx]` inclusive. The
/// half-widths below were computed by hand from that expression, one `dy` at a time:
///
/// * `r = 3`: sqrt(0)=0, sqrt(5)=2.236->2, sqrt(8)=2.828->3, sqrt(9)=3
///   -> `[0, 2, 3, 3, 3, 2, 0]`, widths `1,5,7,7,7,5,1`, 33 cells
/// * `r = 4`: sqrt(0)=0, sqrt(7)=2.646->3, sqrt(12)=3.464->3, sqrt(15)=3.873->4,
///   sqrt(16)=4 -> `[0, 3, 3, 4, 4, 4, 3, 3, 0]`, widths `1,7,7,9,9,9,7,7,1`, 57 cells
/// * `r = 5`: sqrt(0)=0, sqrt(9)=3, sqrt(16)=4, sqrt(21)=4.583->5, sqrt(24)=4.899->5,
///   sqrt(25)=5 -> `[0, 3, 4, 5, 5, 5, 5, 5, 4, 3, 0]`,
///   widths `1,7,9,11,11,11,11,11,9,7,1`, 89 cells
/// * `r = 6`: sqrt(0)=0, sqrt(11)=3.317->3, sqrt(20)=4.472->4, sqrt(27)=5.196->5,
///   sqrt(32)=5.657->6, sqrt(35)=5.916->6, sqrt(36)=6
///   -> `[0, 3, 4, 5, 6, 6, 6, 6, 6, 5, 4, 3, 0]`,
///   widths `1,7,9,11,13,13,13,13,13,11,9,7,1`, 121 cells
/// * `r = 7`: sqrt(0)=0, sqrt(13)=3.606->4, sqrt(24)=4.899->5, sqrt(33)=5.745->6,
///   sqrt(40)=6.325->6, sqrt(45)=6.708->7, sqrt(48)=6.928->7, sqrt(49)=7
///   -> `[0, 4, 5, 6, 6, 7, 7, 7, 7, 7, 6, 6, 5, 4, 0]`,
///   widths `1,9,11,13,13,15,15,15,15,15,13,13,11,9,1`, 169 cells
fn oracle_cells(thickness: u32) -> Vec<u8> {
    #[rustfmt::skip]
    let small: &[&[u8]] = &[
        // thickness 0, diameter 1 -- the single centre pixel (§16.9 item 5).
        &[1],
        // thickness 1, diameter 3 -- full square, four corners zeroed.
        &[0, 1, 0,
          1, 1, 1,
          0, 1, 0],
        // thickness 2, diameter 5 -- full square, four corners zeroed.
        &[0, 1, 1, 1, 0,
          1, 1, 1, 1, 1,
          1, 1, 1, 1, 1,
          1, 1, 1, 1, 1,
          0, 1, 1, 1, 0],
    ];
    if let Some(cells) = small.get(thickness as usize) {
        return cells.to_vec();
    }

    let half_widths: &[u32] = match thickness {
        3 => &[0, 2, 3, 3, 3, 2, 0],
        4 => &[0, 3, 3, 4, 4, 4, 3, 3, 0],
        5 => &[0, 3, 4, 5, 5, 5, 5, 5, 4, 3, 0],
        6 => &[0, 3, 4, 5, 6, 6, 6, 6, 6, 5, 4, 3, 0],
        7 => &[0, 4, 5, 6, 6, 7, 7, 7, 7, 7, 6, 6, 5, 4, 0],
        other => panic!("this file's hand-derived oracle covers thickness 0..=7, not {other}"),
    };
    let diameter = thickness * 2 + 1;
    assert_eq!(
        half_widths.len() as u32,
        diameter,
        "oracle row count for thickness {thickness} must equal its diameter"
    );
    let mut cells = vec![0_u8; (diameter as usize) * (diameter as usize)];
    for (i, half_width) in half_widths.iter().enumerate() {
        for j in 0..diameter {
            if j.abs_diff(thickness) <= *half_width {
                cells[i * (diameter as usize) + (j as usize)] = 1;
            }
        }
    }
    cells
}

/// Hand-derived set-cell counts, one per thickness 0..=7, in that order.
///
/// These are the row-width sums written out in [`oracle_cells`]' doc comment:
/// 1 / 5 / 21 / 33 / 57 / 89 / 121 / 169.
const ORACLE_COUNTS: [usize; 8] = [1, 5, 21, 33, 57, 89, 121, 169];

/// Anti-vacuity literal: `ORACLE_COUNTS.iter().sum()`, written out so the gate cannot
/// pass by finding zero of everything. This number is not computable from the source
/// tree -- it comes from the by-hand derivation above.
const ORACLE_TOTAL_CELLS: usize = 496;

/// Every `(x, y)` this mask has set, in row-major order -- an identity view, not a count
/// (cookbook: cardinality is not identity).
fn set_pixels(mask: &BinaryMask) -> Vec<(u32, u32)> {
    let (width, height) = mask.dimensions();
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if mask.get(x, y) {
                pixels.push((x, y));
            }
        }
    }
    pixels
}

/// A 16x16 canvas with dots placed to exercise all four clipped edges, both corners of a
/// diagonal, and the interior: `(0, 0)`, `(15, 0)`, `(0, 15)`, `(15, 15)`, `(8, 8)`,
/// `(1, 7)`, `(7, 14)` and `(13, 3)`.
fn probe_mask() -> BinaryMask {
    let mut mask = BinaryMask::new(16, 16);
    for (x, y) in [
        (0, 0),
        (15, 0),
        (0, 15),
        (15, 15),
        (8, 8),
        (1, 7),
        (7, 14),
        (13, 3),
    ] {
        mask.set(x, y, true);
    }
    mask
}

// ------------------------------------------------- the oracle half (survives the hoist)

#[test]
fn pc_mask_kernel_matches_the_hand_derived_opencv_ellipse_oracle_for_thickness_0_through_7() {
    // Turns red if `pc_mask::grow::kernel`'s rounding, small-diameter branch, corner
    // zeroing or row extents change for any thickness in 0..=7 -- e.g. swapping
    // `f64::round` for `trunc`, or dropping the `diameter >= 3` guard that keeps
    // `kernel(0)` the identity (§16.9 item 5).
    let mut total = 0_usize;
    for thickness in 0..=MAX_THICKNESS {
        let element = pc_mask::grow::kernel(thickness);
        assert_eq!(
            element.diameter(),
            thickness * 2 + 1,
            "pc_mask::grow::kernel({thickness}) diameter"
        );
        assert_eq!(
            element.radius(),
            thickness,
            "pc_mask::grow::kernel({thickness}) radius"
        );
        assert_eq!(
            element.as_cells(),
            oracle_cells(thickness).as_slice(),
            "pc_mask::grow::kernel({thickness}) cells"
        );
        assert_eq!(
            element.count(),
            ORACLE_COUNTS[thickness as usize],
            "pc_mask::grow::kernel({thickness}) set-cell count"
        );
        total += element.count();
    }
    assert_eq!(
        total, ORACLE_TOTAL_CELLS,
        "set cells summed over pc_mask::grow::kernel(0..=7)"
    );
}

#[test]
fn pc_denoise_kernel_matches_the_hand_derived_opencv_ellipse_oracle_for_thickness_0_through_7() {
    // Same falsifiers as the `pc_mask` case above, applied to the other public path.
    // Asserted separately rather than only through the agreement test, so that the
    // oracle still binds each crate's own name after the L1 hoist makes them aliases.
    let mut total = 0_usize;
    for thickness in 0..=MAX_THICKNESS {
        let element = pc_denoise::morph::kernel(thickness);
        assert_eq!(
            element.diameter(),
            thickness * 2 + 1,
            "pc_denoise::morph::kernel({thickness}) diameter"
        );
        assert_eq!(
            element.radius(),
            thickness,
            "pc_denoise::morph::kernel({thickness}) radius"
        );
        assert_eq!(
            element.as_cells(),
            oracle_cells(thickness).as_slice(),
            "pc_denoise::morph::kernel({thickness}) cells"
        );
        assert_eq!(
            element.count(),
            ORACLE_COUNTS[thickness as usize],
            "pc_denoise::morph::kernel({thickness}) set-cell count"
        );
        total += element.count();
    }
    assert_eq!(
        total, ORACLE_TOTAL_CELLS,
        "set cells summed over pc_denoise::morph::kernel(0..=7)"
    );
}

#[test]
fn both_crates_dilate_a_single_interior_dot_to_the_hand_written_kernel_1_footprint() {
    // spec §16.9 item 6, stamp formulation. The expected set is written out, not derived:
    // kernel(1) is the 3x3 plus sign, so a dot at (3, 3) in a 7x7 canvas yields exactly
    // the five pixels below. Turns red if the stamp offsets lose the `- radius`
    // recentring, or if dilation is replaced by erosion.
    const EXPECTED: [(u32, u32); 5] = [(3, 2), (2, 3), (3, 3), (4, 3), (3, 4)];

    let mut dot = BinaryMask::new(7, 7);
    dot.set(3, 3, true);

    let from_mask = pc_mask::grow::dilate(&dot, &pc_mask::grow::kernel(1));
    let from_denoise = pc_denoise::morph::dilate(&dot, &pc_denoise::morph::kernel(1));
    assert_eq!(set_pixels(&from_mask), EXPECTED.to_vec());
    assert_eq!(set_pixels(&from_denoise), EXPECTED.to_vec());
}

#[test]
fn both_crates_dilate_a_corner_dot_to_the_hand_written_clipped_kernel_2_footprint() {
    // spec §16.9 item 6: writes outside the canvas are **dropped** (zero border), which a
    // reflecting or wrapping implementation would not do. kernel(2) is the 5x5 square
    // with its four corners zeroed; centred on (0, 0) only the offsets `(j, i)` with
    // `j >= 2` and `i >= 2` land on canvas, and of those nine the one at kernel cell
    // (4, 4) is a zeroed corner -- so (2, 2) stays clear and eight pixels are set.
    const EXPECTED: [(u32, u32); 8] = [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
    ];

    let mut dot = BinaryMask::new(7, 7);
    dot.set(0, 0, true);

    let from_mask = pc_mask::grow::dilate(&dot, &pc_mask::grow::kernel(2));
    let from_denoise = pc_denoise::morph::dilate(&dot, &pc_denoise::morph::kernel(2));
    assert_eq!(set_pixels(&from_mask), EXPECTED.to_vec());
    assert_eq!(set_pixels(&from_denoise), EXPECTED.to_vec());
}

#[test]
fn both_crates_dilate_with_the_thickness_0_kernel_as_the_identity() {
    // §16.9 item 5: `kernel(0)` is the single centre pixel, so dilation must return the
    // input unchanged -- not a cleared mask, which is what zeroing a 1x1 kernel's
    // "corners" would produce.
    let probe = probe_mask();
    assert_eq!(
        pc_mask::grow::dilate(&probe, &pc_mask::grow::kernel(0)),
        probe
    );
    assert_eq!(
        pc_denoise::morph::dilate(&probe, &pc_denoise::morph::kernel(0)),
        probe
    );
}
