//! Task A3b -- spec §16.37 item 11: *"`refine_mask`'s page-level driver plus
//! `refine_undetected_mask`"*. FROZEN with the implementation.
//!
//! **Where every expected value comes from.** §16.37 item 3 is explicit that a self-comparison
//! gate proves determinism and not correctness -- *"A port that is wrong in the same way on
//! every run passes it forever."* So every literal below comes from one of exactly two places,
//! never from `pc_detect::annotate_refine`:
//!   * hand derivation, stated in the test's own comment; or
//!   * an independent oracle -- upstream PanelCleaner's `refine_mask`
//!     (`comic_text_detector/utils/textmask.py:195-212`), `refine_undetected_mask` (`:161-192`),
//!     `union_area` (`imgproc_utils.py:15-22`), `expand_textwindow` (`:163-173`) and
//!     `cv2.connectedComponentsWithStats`, at pinned commit
//!     `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, **imported and executed** against `cv2`
//!     **5.0.0** -- the version `tests/fixtures/recorded/detector/PROVENANCE.json` pins -- with
//!     `numpy` 2.2.5 and CPython 3.11.8, on 2026-08-07.
//!
//! **The oracle was run with IPP DISABLED** (`cv2.ipp.setUseIPP(False)`), because
//! `DEVIATION(29)` (§14 item 29 / §16.37 item 10 -- renumbered from 24 on 2026-08-07 to resolve a
//! cross-branch collision with `lama-inpaint`; this file uses 29, the live identifier) commits this crate to OpenCV's documented
//! reference Otsu path rather than IPP's fast path. Comparing against an IPP-enabled oracle
//! would compare our declared behaviour with a path we deliberately do not implement.
//! §16.37 item 10(d) records that the committed page's twelve Otsu thresholds are identical
//! either way, so the real-page tests here are insensitive to that switch; the synthetic ones
//! were not separately measured with IPP on, and nothing here claims they are insensitive.
//!
//! **Every synthetic page in this file is built by a formula, not read from a file** ([`page`]
//! and [`pred`] below), and the same formula was evaluated in Python to produce the oracle
//! values. So no fixture here is a recorded artifact that could drift.
//!
//! **Anti-vacuity.** Every page-level assertion is a **row-by-row ASCII grid** plus a
//! hard-coded non-zero count and a positional fingerprint (`sum of (row-major index + 1) over
//! non-zero pixels`), so a mask with the right population in the wrong place fails. Every
//! `undetected_blocks` assertion is the **exact `Vec<Rect>`**, not its length, so dropping the
//! wrong component fails where a count would not. And
//! [`refine_undetected_mask_on_the_recorded_page_invents_no_block`] -- whose headline assertion
//! is that a list is **empty**, the one shape that can pass by finding nothing -- is paired in
//! the same test with a hard-coded **12** from the same code path under an empty block list, so
//! "found no components at all" and "found twelve and rejected all twelve" are distinguished.
//!
//! # The label-order question this file raised, and how it was closed
//!
//! [`undetected_blocks_matches_upstreams_block_scan_label_order_when_the_background_is_small`]
//! began life asserting a measured **divergence** from upstream on a synthetic input: because
//! `valid_labels[1:]` selects by position in the label sequence, iterating this crate's
//! first-raster-pixel numbering dropped a different component than `cv2` did. That was escalated
//! rather than fixed in place, went to two independent joint reviews, and was resolved by spec
//! §16.37 item 13 -- `undetected_blocks` now iterates
//! `ConnectedComponents::labels_in_opencv_block_scan_order`, and the test asserts **`cv2`'s**
//! answer, `[5, 0, 20, 20]`. The rename is part of that ruling: the old name claimed the
//! numbering *decides* the drop, which is a divergence claim, and what the test verifies now is
//! parity. `connected_components`'s own global numbering is unchanged, and
//! [`connected_components_global_numbering_is_still_first_raster_pixel_after_the_block_scan_accessor`]
//! pins that it was not moved.

use image::{GrayImage, Luma, Rgb, RgbImage};
use pc_core::{Rect, StageError};
use pc_detect::annotate_refine::{
    intersection_area, refine_mask, refine_undetected_mask, undetected_blocks,
    zero_pred_where_refined, UNDETECTED_MIN_COMPONENT_AREA, UNDETECTED_PRED_THRESHOLD,
};
use pc_testkit::paths;

const PAGE_STEM: &str = "detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

/// The synthetic pages' paper colour and ink colour. Chosen only so that the greyscale
/// histogram A1 builds has one dominant light mode and one dark mode; nothing depends on the
/// exact values beyond the oracle having been run with these two.
const LIGHT: u8 = 230;
const DARK: u8 = 20;

// ============================================================ Part 1: the refine_mask driver

/// spec §16.37 item 11(b): *"**Part 1, `refine_mask`'s PAGE-LEVEL DRIVER
/// (`textmask.py:195-212`)** -- a thin per-block composition loop: expand the window with A1's
/// `expand_text_window`, crop image and pred mask, build the candidate list through A1's top-k
/// path and A2's Otsu path ..., call A3's `merge_mask_list`, and OR the result into a
/// page-sized mask **at the window's coordinates**"*.
///
/// **What this test is for: the window's coordinates.** A driver that merges correctly and then
/// writes the result at the *block's* coordinates, or at the origin, or at the wrong axis order,
/// produces exactly the right population in the wrong place -- which is why the assertion is a
/// row-by-row grid and a positional fingerprint, not a count.
///
/// **Fixture, hand-built:** a 30x30 page, paper 230, with a crude 'T' in ink 20 at
/// `[9, 9, 15, 11)` and `[11, 11, 13, 19)`; the pred mask is 255 over `[8, 8, 16, 20)` and 0
/// elsewhere; the single block is `[8, 8, 16, 20]`.
///
/// **Hand-derived window:** `w = 8`, `h = 12`, so
/// `paddings = round((12 * 0.25 + 8 * 0.75) / 16) = round(9.0 / 16) = round(0.5625) = 1`, giving
/// `[max(0, 7), max(0, 7), min(29, 17), min(29, 21)] = [7, 7, 17, 21]`. Upstream's
/// `expand_textwindow` returns the same, verified by running it.
///
/// **Expected page mask: upstream's own output** for this fixture (28 non-zero pixels), which
/// lies wholly inside the window `[7, 7, 17, 21)` -- so the grid also witnesses that nothing
/// was written outside the window.
#[test]
fn refine_mask_writes_each_blocks_merged_mask_at_its_expanded_window() {
    let image = page(30, 30, &[(9, 9, 15, 11, DARK), (11, 11, 13, 19, DARK)]);
    let pred_mask = pred(30, 30, &[(8, 8, 16, 20, 255)]);
    let blocks = [Rect::new(8, 8, 16, 20)];

    let out = refine_mask(&image, &pred_mask, &blocks).expect("equal sizes, non-empty window");

    assert_grid(
        &out,
        &[
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            ".........XXXXXX...............",
            ".........XXXXXX...............",
            "...........XX.................",
            "...........XX.................",
            "...........XX.................",
            "...........XX.................",
            "...........XX.................",
            "...........XX.................",
            "...........XX.................",
            "...........XX.................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
            "..............................",
        ],
    );
    // Upstream's own figures for this fixture. 28 = 2 rows of 6 plus 8 rows of 2.
    assert_eq!(nonzero_count(&out), 28);
    assert_eq!(positional_fingerprint(&out), 10730);
    assert_eq!(
        out.dimensions(),
        (30, 30),
        "the page mask is `np.zeros_like(pred_mask)`"
    );
}

/// spec §16.37 item 5 (`CORRECTION`, `SUPERSEDES: §8.3 step 5`): *"upstream clamps the far
/// edges to **`im_w - 1` / `im_h - 1`**, while `pc_core::Rect::pad` ... clamps to `canvas.0` /
/// `canvas.1` -- off by one at every image edge."*
///
/// **What fails this test.** Substituting `Rect::pad`'s clamp (or `min(im_w, ...)`) for A1's
/// `expand_text_window` in the driver: the window would reach `[13, 13, 24, 24)` instead of
/// `[13, 13, 23, 23)`, the merge would run over an 11x11 crop instead of 10x10, and the final
/// page column and row would become writable. The two explicit column/row assertions below say
/// that in the form a future reader can check without re-deriving the grid.
///
/// **Fixture, hand-built:** a 24x24 page, ink block `[16, 16, 22, 22)`, pred mask 255 over
/// `[14, 14, 24, 24)`, single block `[14, 14, 24, 24]` -- i.e. flush against the bottom-right
/// corner. **Hand-derived window:** `w = h = 10`, `paddings = round((2.5 + 7.5) / 16) =
/// round(0.625) = 1`, so `[13, 13, min(23, 25), min(23, 25)] = [13, 13, 23, 23]`.
///
/// **Expected page mask: upstream's own output** (36 non-zero).
#[test]
fn refine_mask_clamps_the_far_window_edge_one_pixel_inside_the_page() {
    let image = page(24, 24, &[(16, 16, 22, 22, DARK)]);
    let pred_mask = pred(24, 24, &[(14, 14, 24, 24, 255)]);
    let blocks = [Rect::new(14, 14, 24, 24)];

    let out = refine_mask(&image, &pred_mask, &blocks).expect("equal sizes, non-empty window");

    assert_grid(
        &out,
        &[
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "........................",
            "................XXXXXX..",
            "................XXXXXX..",
            "................XXXXXX..",
            "................XXXXXX..",
            "................XXXXXX..",
            "................XXXXXX..",
            "........................",
            "........................",
        ],
    );
    assert_eq!(nonzero_count(&out), 36);
    assert_eq!(positional_fingerprint(&out), 16686);
    // The `- 1` clamp, stated directly: column 23 and row 23 are outside every window this
    // block can produce, so they can never be written.
    assert_eq!(
        (0..24).filter(|&y| out.get_pixel(23, y).0[0] != 0).count(),
        0,
        "the far column is outside the `im_w - 1` clamp"
    );
    assert_eq!(
        (0..24).filter(|&x| out.get_pixel(x, 23).0[0] != 0).count(),
        0,
        "the far row is outside the `im_h - 1` clamp"
    );
}

/// spec §16.37 item 11(b): *"OR the result into a page-sized mask at the window's
/// coordinates"*, which is upstream's `mask_refined[by1:by2, bx1:bx2] = cv2.bitwise_or(
/// mask_refined[by1:by2, bx1:bx2], mask_merged)` (`textmask.py:211`).
///
/// **What fails this test, and it is the whole reason the fixture is shaped the way it is.**
/// Assigning the merged crop into the page instead of OR-ing it. The two expanded windows here
/// overlap in columns 16..20, the **first** block's merged mask is on throughout that overlap,
/// and the **second** block's merged mask is empty there -- so a second-block assignment erases
/// 50 of the first block's pixels. Measured on upstream with an assignment variant substituted:
/// **120** non-zero instead of **170**, 50 differing pixels. The two single-block figures are
/// asserted below so a reader can see the 50 without re-running anything: `170 + 0` under OR,
/// and `170 - 5 columns * 10 rows` under assignment.
///
/// **Fixture, hand-built** (a 44x24 page): ink at `[6, 9, 12, 15)` and `[26, 9, 34, 15)`, plus a
/// single mid-grey (120) pixel at `(15, 10)` whose only job is to make the second block's crop
/// statistics differ from the first's; pred mask 255 over `[5, 8, 20, 16)` and `[17, 8, 36, 16)`;
/// blocks `[5, 8, 20, 16]` and `[17, 8, 36, 16]`. **Hand-derived windows:** block 0 has
/// `w = 15, h = 8`, `paddings = round((3.75 + 6) / 16) = round(0.609) = 1` -> `[4, 7, 21, 17]`;
/// block 1 has `w = 19, h = 8`, `paddings = round((4.75 + 6) / 16) = round(0.672) = 1` ->
/// `[16, 7, 37, 17]`. They overlap in `x in 16..21`.
///
/// **Expected page mask: upstream's own output** (170 non-zero) -- which is exactly block 0's
/// window `[4, 7, 21, 17)` saturated, 17 * 10 = 170, and **nothing** inside block 1's window
/// beyond that overlap. So the grid simultaneously refutes "fill every window" (which would add
/// block 1's 16 remaining columns) and "assign, don't OR".
#[test]
fn refine_mask_ors_overlapping_windows_instead_of_overwriting_them() {
    let image = overlap_page();
    let pred_mask = overlap_pred();
    let blocks = [Rect::new(5, 8, 20, 16), Rect::new(17, 8, 36, 16)];

    let out = refine_mask(&image, &pred_mask, &blocks).expect("equal sizes, non-empty window");
    assert_grid(&out, OVERLAP_EXPECTED);
    assert_eq!(nonzero_count(&out), 170);
    assert_eq!(positional_fingerprint(&out), 88230);

    // Upstream's per-block figures, so the 50-pixel loss an assignment would cause is visible
    // here and not only in this test's comment.
    let first = refine_mask(&image, &pred_mask, &blocks[..1]).expect("equal sizes");
    let second = refine_mask(&image, &pred_mask, &blocks[1..]).expect("equal sizes");
    assert_eq!(
        nonzero_count(&first),
        170,
        "block 0 alone saturates its own window"
    );
    assert_eq!(
        nonzero_count(&second),
        0,
        "block 1 alone contributes nothing"
    );
    assert_eq!(
        (7..17)
            .flat_map(|y| (16..21).map(move |x| (x, y)))
            .filter(|&(x, y)| first.get_pixel(x, y).0[0] != 0)
            .count(),
        50,
        "block 0 owns 50 pixels inside the shared columns, which an assignment would erase"
    );
}

/// The OR at `textmask.py:211` is commutative, so the page mask cannot depend on the order of
/// `blk_list` -- but only because each block's *merge* is computed from `pred_mask` and `img`
/// alone and never from the running `mask_refined`. **What fails this test:** any driver that
/// feeds the accumulating page mask back into the per-block merge, or that seeds
/// `mask_refined` from anything but zeros.
///
/// Uses the same overlapping-window fixture, where order is *observable* under an assignment
/// implementation (the two orders would give 120 and different geometry). Confirmed on upstream:
/// forward and reversed are byte-identical, both 170 non-zero.
#[test]
fn refine_mask_page_mask_is_independent_of_the_block_order() {
    let image = overlap_page();
    let pred_mask = overlap_pred();
    let forward = [Rect::new(5, 8, 20, 16), Rect::new(17, 8, 36, 16)];
    let reversed = [forward[1], forward[0]];

    let a = refine_mask(&image, &pred_mask, &forward).expect("equal sizes");
    let b = refine_mask(&image, &pred_mask, &reversed).expect("equal sizes");
    // Anti-vacuity: both must be upstream's answer, not merely equal to each other.
    assert_eq!(nonzero_count(&a), 170);
    assert_eq!(nonzero_count(&b), 170);
    assert_eq!(a, b, "the page mask must not depend on block order");
}

/// Upstream's `mask_refined = np.zeros_like(pred_mask)` with a `blk_list` the loop never
/// enters. Measured on upstream: 0 non-zero, shape `(24, 44)`.
///
/// **What fails this test:** returning the pred mask, or a mask of the image's size when the
/// two differ, or `None`/an error for the empty case. It is small but it is the base case
/// `refine_undetected_mask` hits whenever nothing is invented.
#[test]
fn refine_mask_returns_an_all_zero_page_mask_for_an_empty_block_list() {
    let image = overlap_page();
    let pred_mask = overlap_pred();
    let out = refine_mask(&image, &pred_mask, &[]).expect("an empty block list is not an error");
    assert_eq!(out.dimensions(), (44, 24));
    assert_eq!(nonzero_count(&out), 0);
}

/// **A totality choice this crate makes, NOT upstream parity, and it is stated as such.**
/// Upstream **raises** here: with a block whose expanded window is empty, `refine_mask` reaches
/// `cv2.cvtColor` on a zero-sized array and dies with
/// `(-215:Assertion failed)` from `modules/imgproc/src/color.cpp:199` (measured 2026-08-07 on
/// `[19, 19, 20, 20]` and `[5, 5, 5, 5]` over a 20x20 page, whose windows are `[19, 19, 19, 19]`
/// and `[5, 5, 5, 5]`). A Python raise aborts the whole page; this crate classifies it
/// **per-image** (`StageError::InvalidInput`) per cookbook rule 4, because one malformed block
/// must not abort a run -- the same call this crate already made for `xor_sum`'s size mismatch.
///
/// The fatality is **declared here**, not inferred: `InvalidInput` is per-image.
///
/// Hand-derived: for `[5, 5, 5, 5]`, `w = h = 0` so `paddings = round(0 / 16) = 0` and the
/// window is `[5, 5, 5, 5]`, which is empty.
#[test]
fn refine_mask_rejects_a_block_whose_expanded_window_is_empty() {
    let image = page(20, 20, &[]);
    let pred_mask = pred(20, 20, &[]);
    let error = refine_mask(&image, &pred_mask, &[Rect::new(5, 5, 5, 5)])
        .expect_err("an empty window has no crop to refine");
    assert!(
        matches!(error, StageError::InvalidInput(_)),
        "an empty window is per-image, not run-fatal: {error:?}"
    );
}

/// Upstream indexes `img` and `pred_mask` with the same window, which numpy only defines for
/// equal shapes. Per-image (`StageError::InvalidInput`), matching every other size guard in the
/// A1-A3 family.
#[test]
fn refine_mask_rejects_an_image_and_pred_mask_of_different_sizes() {
    let image = page(20, 20, &[]);
    let pred_mask = pred(21, 20, &[]);
    let error = refine_mask(&image, &pred_mask, &[]).expect_err("shapes must agree");
    assert!(
        matches!(error, StageError::InvalidInput(_)),
        "a size mismatch is per-image: {error:?}"
    );
}

/// The real-data sanity check for Part 1, on the one page every consumer of this fixture reads.
///
/// **Inputs, both already committed:** `…_base.png` (1200x1660) and `…_detector_mask.png` --
/// the **unrefined** U-Net mask, per §16.37 item 4 -- with the four blocks from
/// `…_detector_blocks.json` verbatim, so the only variable is the refinement.
///
/// **Two independent expectations, and the pair is the point.**
///   * the four per-block page masks have `[1092, 817, 1033, 0]` non-zero pixels. Those four
///     literals are **already frozen** in `a3_annotate_merge.rs`
///     (`EXPECTED_NONZERO`), derived there from upstream's `merge_mask_list` on the four crops.
///     Reproducing them here through a *different* entry point -- the page-level driver, which
///     computes its own windows and crops rather than being handed them -- is the
///     build-vs-known-good signal, and it is what would catch a driver that cropped the right
///     region by luck of a wrong window formula.
///   * the whole-page mask has **2942** non-zero pixels and positional fingerprint
///     **2790551277**. The count alone is `1092 + 817 + 1033 + 0` and so is *derivable* from the
///     first bullet -- the fingerprint is not, and it is what pins that the four crops landed at
///     the four right places on a 1200x1660 canvas. Both from upstream, run on this fixture.
///
/// The four windows are asserted too, since they are the driver's own arithmetic:
/// `[671, 1394, 743, 1441]`, `[563, 70, 667, 127]`, `[602, 626, 729, 708]`,
/// `[435, 1404, 501, 1449]` (upstream's `expand_textwindow` on `img.shape`, run 2026-08-07).
#[test]
fn refine_mask_on_the_recorded_page_reproduces_the_upstream_page_mask() {
    let (image, pred_mask) = recorded_page();
    let blocks = recorded_blocks();
    assert_eq!(
        blocks.len(),
        4,
        "the committed fixture holds four detector blocks"
    );

    // Per-block, through the driver, cross-checked against A3's already-frozen literals.
    const PER_BLOCK: [usize; 4] = [1092, 817, 1033, 0];
    for (index, block) in blocks.iter().enumerate() {
        let one = refine_mask(&image, &pred_mask, &[*block]).expect("equal sizes");
        assert_eq!(
            nonzero_count(&one),
            PER_BLOCK[index],
            "block {index} {block:?} must reproduce A3's frozen merged population"
        );
    }

    let out = refine_mask(&image, &pred_mask, &blocks).expect("equal sizes");
    assert_eq!(out.dimensions(), (1200, 1660));
    assert_eq!(
        nonzero_count(&out),
        2942,
        "upstream's page-level population"
    );
    assert_eq!(
        positional_fingerprint(&out),
        2_790_551_277,
        "upstream's page-level positional fingerprint -- this is what pins placement"
    );
}

/// Upstream's `refine_mask` does **not** mutate its `pred_mask` argument -- only
/// `refine_undetected_mask` does (`textmask.py:169`). Measured on the recorded page: after
/// `refine_mask`, 0 of the 1 992 000 pred pixels differ from the input.
///
/// **What fails this test:** taking `pred_mask` by `&mut` in the driver and reusing that buffer,
/// or zeroing inside the driver instead of inside `refine_undetected_mask`. The distinction is
/// load-bearing because `refine_undetected_mask` computes its residual by *subtracting* the
/// refined mask from the pred mask itself; a driver that had already zeroed it would leave no
/// residual at all.
#[test]
fn refine_mask_does_not_mutate_the_pred_mask() {
    let (image, pred_mask) = recorded_page();
    let before = pred_mask.clone();
    let _ = refine_mask(&image, &pred_mask, &recorded_blocks()).expect("equal sizes");
    assert_eq!(
        pred_mask, before,
        "the driver must leave the pred mask alone"
    );
}

// ================================================== Part 2 (iv): the intersection scorer

/// spec §16.37 item 11(b)(iv): *"`union_area` (`pcleaner/comic_text_detector/utils/
/// imgproc_utils.py:15-22`) is **misnamed**: it computes the **intersection** area of the two
/// boxes, and returns the sentinel **`-1`** when they are disjoint ... Port the behaviour under
/// its own name, not under its upstream name."*
///
/// Every value below was both hand-derived from the six lines of upstream source **and**
/// obtained by importing and calling `union_area` (2026-08-07); the two agreed.
///
/// The `touching` row is the one that is easy to get wrong: sharing an edge gives `x2 == x1`,
/// the guard is `y2 < y1 or x2 < x1` (**strict**), so the result is the area **0**, not the
/// sentinel `-1`. A guard written with `<=` would return `-1` there.
#[test]
fn intersection_area_returns_minus_one_only_for_disjoint_boxes() {
    // Fully disjoint -> sentinel.
    assert_eq!(
        intersection_area(Rect::new(0, 0, 5, 5), Rect::new(10, 10, 20, 20)),
        -1
    );
    // Sharing the edge x = 10 -> zero-width intersection, area 0, NOT the sentinel.
    assert_eq!(
        intersection_area(Rect::new(0, 0, 10, 10), Rect::new(10, 0, 20, 10)),
        0
    );
    // (15 - 5) * (10 - 5) = 50.
    assert_eq!(
        intersection_area(Rect::new(5, 5, 10, 15), Rect::new(5, 5, 15, 15)),
        50
    );
    // (14 - 5) * (10 - 5) = 45.
    assert_eq!(
        intersection_area(Rect::new(5, 5, 10, 14), Rect::new(5, 5, 15, 15)),
        45
    );
    // Nested: (4 - 2) * (4 - 2) = 4, and the argument order does not matter.
    assert_eq!(
        intersection_area(Rect::new(2, 2, 4, 4), Rect::new(0, 0, 10, 10)),
        4
    );
    assert_eq!(
        intersection_area(Rect::new(0, 0, 10, 10), Rect::new(2, 2, 4, 4)),
        4
    );
}

// ============================================ Part 2 (ii)-(iv): which components become blocks

/// spec §16.37 item 11(b)(iii), quoted because the whole test is about not paraphrasing it:
/// *"`valid_labels = np.where(stats[:, -1] > 50)[0]` then `for lab_index in valid_labels[1:]` --
/// and **this is NOT "skip the background label"**. It drops the **first surviving entry** of
/// the area-filtered list, which coincides with label 0 only when the background's own area also
/// exceeds 50. Measured 2026-08-07 on an 8x8 all-foreground image with a single background pixel:
/// `stats` areas are `[1, 63]`, `valid_labels == [1]`, and `valid_labels[1:] == []` -- so the
/// **only real component is silently dropped**."*
///
/// Re-measured here against `cv2` 5.0.0 on 2026-08-07: `num_labels == 2`, areas `[1, 63]`,
/// `valid_labels == [1]`, `valid_labels[1:] == []`.
///
/// **What fails this test:** writing the filter as "skip label 0" -- i.e. iterating
/// `valid_labels` and skipping 0, or iterating `1..num_labels` and filtering on area. Either
/// would return the one 63-pixel component here. The assertion is `is_empty()`, so the test also
/// asserts the component *exists* first, or the emptiness would be vacuous.
#[test]
fn undetected_blocks_drops_the_first_area_filtered_entry_not_the_background_label() {
    // 8x8, every pixel above the pred threshold except (3, 3).
    let mut mask = GrayImage::from_pixel(8, 8, Luma([255]));
    mask.put_pixel(3, 3, Luma([0]));

    // Anti-vacuity: the 63-pixel component really is there and really does clear the area
    // filter, so "empty" below cannot mean "found nothing to consider".
    let components = pc_detect::annotate_merge::connected_components(&mask);
    assert_eq!(components.num_labels(), 2, "background plus one component");
    assert_eq!(components.stats[0].area, 1, "background is the single hole");
    assert_eq!(components.stats[1].area, 63);
    assert!(components.stats[1].area > UNDETECTED_MIN_COMPONENT_AREA);
    assert!(components.stats[0].area < UNDETECTED_MIN_COMPONENT_AREA);

    assert!(
        undetected_blocks(&mask, &[]).is_empty(),
        "the sole surviving entry is the FIRST, and `valid_labels[1:]` drops it"
    );
}

/// The sharper half of the same quirk: when **two** real components survive the area filter and
/// the background does not, the first one is dropped and the second is kept. A "skip label 0"
/// implementation returns **both**; upstream returns **one**.
///
/// **Fixture, hand-built:** a 20x8 mask, 255 over `x in 0..9` and over `x in 10..20`, so column
/// 9 is the only background. Measured against `cv2` 5.0.0 (2026-08-07): areas `[8, 72, 80]`,
/// bounding boxes `[(9,0,1,8), (0,0,9,8), (10,0,10,8)]`, `valid_labels == [1, 2]`,
/// `valid_labels[1:] == [2]`, and `cv2` labels the left blob 1 and the right blob 2. This
/// crate's first-raster-pixel numbering agrees on that fixture (left blob's first pixel is
/// index 0, the right blob's is index 10), so the expected block below is upstream's answer and
/// not merely ours.
///
/// **This fixture IS in the order-sensitive regime -- it just gets the same answer either way,
/// which is a coincidence and not evidence that §16.37 item 13's fix was unnecessary.** Its
/// background area is 8, i.e. `<= 50`, so `valid[0]` is a real component exactly as in
/// [`undetected_blocks_matches_upstreams_block_scan_label_order_when_the_background_is_small`];
/// the block-scan and first-raster-pixel orders merely happen to coincide because each blob's
/// minimum 2x2 block sorts the same way as its first raster pixel here. Both reviewers found
/// this independently.
#[test]
fn undetected_blocks_drops_the_lowest_numbered_survivor_when_the_background_is_filtered_out() {
    let mut mask = GrayImage::new(20, 8);
    for y in 0..8 {
        for x in (0..9).chain(10..20) {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    // Identity, not cardinality: the RIGHT blob, at x in 10..20.
    assert_eq!(undetected_blocks(&mask, &[]), vec![Rect::new(10, 0, 20, 8)]);
}

/// spec §16.37 item 11(b)(iii)'s `stats[:, -1] > 50` -- **strict**, and on a **pixel count**,
/// not a bounding-box area (`CC_STAT_AREA`).
///
/// **Fixture, hand-built** on a 40x20 mask so the background (699 pixels) clears the filter and
/// `valid_labels[0]` is therefore label 0, isolating the area test from the quirk above:
///   * component P: `x in 2..7`, `y in 2..12` -- 5 * 10 = **50** pixels, which `> 50` rejects;
///   * component Q: `x in 20..23`, `y in 2..19` -- 3 * 17 = **51** pixels, which it accepts.
///
/// Measured against `cv2` 5.0.0 (2026-08-07): areas `[699, 50, 51]`, `valid_labels == [0, 2]`,
/// `valid_labels[1:] == [2]`, and `cv2` labels P 1 and Q 2 -- agreeing with this crate's
/// numbering, so the expected block is upstream's.
///
/// **What fails this test:** `>= 50` (which returns both P and Q), or filtering on
/// `width * height` (P's box area is 50 and Q's is 51 as well here, so that mutation is *not*
/// caught by this fixture -- disclosed rather than claimed).
#[test]
fn undetected_blocks_requires_a_component_area_strictly_greater_than_fifty() {
    let mut mask = GrayImage::new(40, 20);
    fill(&mut mask, 2, 2, 7, 12, 255);
    fill(&mut mask, 20, 2, 23, 19, 255);
    assert_eq!(
        undetected_blocks(&mask, &[]),
        vec![Rect::new(20, 2, 23, 19)]
    );
}

/// spec §16.37 item 11(c): *"Port the EFFECTIVE 8, not the literal 4 the source text passes"* --
/// upstream writes `cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)`, whose
/// positionals bind to the `labels`/`stats` **output** slots, leaving `connectivity` at its
/// default 8.
///
/// **Fixture, hand-built:** a 30x30 mask with two 8x8 squares at `[2, 2, 10, 10)` and
/// `[10, 10, 18, 18)`, touching only at the corner pair `(9, 9)`/`(10, 10)`. Measured against
/// `cv2` 5.0.0 (2026-08-07): `connectivity=4` gives **3** labels, `connectivity=8` gives **2**,
/// and the positional call `(m, 4, CV_16U)` gives **2** -- i.e. 8. Areas under 8 are
/// `[772, 128]` with the component's box `(2, 2, 16, 16)`, `valid_labels == [0, 1]`,
/// `valid_labels[1:] == [1]`.
///
/// **What fails this test:** honouring upstream's *written* 4. That would yield two components
/// of 64 pixels each, both over the area threshold, `valid_labels == [0, 1, 2]` and
/// `valid_labels[1:] == [1, 2]` -- two blocks `[2, 2, 10, 10]` and `[10, 10, 18, 18]` instead of
/// the single `[2, 2, 18, 18]` asserted here. Cardinality *and* geometry both move, and the
/// assertion is on the exact vector.
#[test]
fn undetected_blocks_treats_a_corner_touching_pair_as_one_eight_connected_component() {
    let mut mask = GrayImage::new(30, 30);
    fill(&mut mask, 2, 2, 10, 10, 255);
    fill(&mut mask, 10, 10, 18, 18, 255);
    assert_eq!(undetected_blocks(&mask, &[]), vec![Rect::new(2, 2, 18, 18)]);
}

/// spec §16.37 item 11(b)(ii): `cv2.threshold(mask_pred, 30, 255, THRESH_BINARY)` -- **strict**,
/// so a pred pixel of exactly 30 is off. Measured against `cv2` 5.0.0 (2026-08-07): a 10x10
/// blob of value 30 thresholds to nothing at all (`num_labels == 1`, area `[1600]` on a 40x40
/// page), while the same blob at 31 gives `num_labels == 2`, areas `[1500, 100]`.
///
/// **What fails this test:** `>=`. At 30 the blob would become a 100-pixel component and be
/// returned; the `is_empty()` half below would go red. The 31 half is asserted in the same test
/// so the emptiness is not vacuous.
#[test]
fn undetected_blocks_thresholds_the_pred_mask_strictly_above_thirty() {
    assert_eq!(UNDETECTED_PRED_THRESHOLD, 30);

    let mut at_threshold = GrayImage::new(40, 40);
    fill(&mut at_threshold, 5, 5, 15, 15, 30);
    assert!(
        undetected_blocks(&at_threshold, &[]).is_empty(),
        "a pred value of exactly 30 is below the strict threshold"
    );

    let mut above = GrayImage::new(40, 40);
    fill(&mut above, 5, 5, 15, 15, 31);
    assert_eq!(
        undetected_blocks(&above, &[]),
        vec![Rect::new(5, 5, 15, 15)]
    );
}

/// spec §16.37 item 11(b)(iv): *"the invention test, `union_area(blk.xyxy, bbox) / w / h < 0.5`,
/// maximised over every detected block"* -- **strict**, so exactly one half is **rejected**.
///
/// **Fixture, hand-built** on a 40x40 mask with a single 10x10 component at `[5, 5, 15, 15)`, so
/// `w = h = 10` and `w * h = 100`:
///   * detected block `[5, 5, 10, 15]`: intersection `(15 - 5) * (10 - 5) = 50`, ratio exactly
///     **0.5** -> `0.5 < 0.5` is false -> **rejected**;
///   * detected block `[5, 5, 10, 14]`: intersection `(14 - 5) * (10 - 5) = 45`, ratio **0.45**
///     -> **accepted**.
///
/// Both hand-derived and both confirmed end to end on upstream (2026-08-07): the rejecting
/// block leaves the refined mask at 0 non-zero, the accepting one at 64.
///
/// **What fails this test:** `<=` for `<` (the first case would be accepted), or dividing by the
/// component's pixel `area` (100 here, so *that* mutation is invisible on this fixture -- said
/// plainly rather than left implied) -- and, separately, taking the **minimum** or the **first**
/// score instead of the maximum over blocks, which the third case below catches.
#[test]
fn undetected_blocks_rejects_a_component_at_exactly_half_intersection() {
    let mut mask = GrayImage::new(40, 40);
    fill(&mut mask, 5, 5, 15, 15, 200);
    let component = Rect::new(5, 5, 15, 15);

    assert!(
        undetected_blocks(&mask, &[Rect::new(5, 5, 10, 15)]).is_empty(),
        "ratio exactly 0.5 must be rejected by the strict `<`"
    );
    assert_eq!(
        undetected_blocks(&mask, &[Rect::new(5, 5, 10, 14)]),
        vec![component],
        "ratio 0.45 must be accepted"
    );
    // The score is the MAXIMUM over blocks: a far-away block scores the -1 sentinel, and taking
    // the minimum (or the first, or the last) would accept where upstream rejects.
    assert!(
        undetected_blocks(&mask, &[Rect::new(30, 30, 40, 40), Rect::new(5, 5, 10, 15)]).is_empty(),
        "the disjoint block must not be allowed to lower the score"
    );
}

/// The `-1` sentinel's consequence, spelled out by spec §16.37 item 11(b)(iv): *"a component
/// overlapping nothing scores `-1/w/h`, which is `< 0.5`, and is invented"*. With an **empty**
/// block list `bbox_score` never leaves its `-1` initialiser, so every surviving component is
/// invented -- which is also the precondition every `&[]` call in this file relies on.
///
/// **What fails this test:** initialising the score at 0 (then `0 / 100 = 0 < 0.5` still
/// accepts, so that alone is invisible here -- what the *disjoint* case catches is initialising
/// at a large value, or treating "no overlap" as a rejection).
#[test]
fn undetected_blocks_invents_a_component_that_overlaps_no_detected_block() {
    let mut mask = GrayImage::new(40, 40);
    fill(&mut mask, 5, 5, 15, 15, 200);
    let component = Rect::new(5, 5, 15, 15);
    assert_eq!(undetected_blocks(&mask, &[]), vec![component]);
    assert_eq!(
        undetected_blocks(&mask, &[Rect::new(30, 30, 40, 40)]),
        vec![component],
        "a wholly disjoint block scores the -1 sentinel and cannot veto"
    );
}

/// **PARITY with `cv2`'s label ORDER, on the one fixture where the two orders disagree.**
///
/// spec §16.37 item 12 assigns the obligation: *"each new consumer of the label numbering
/// re-establishes order-independence for itself or registers the divergence it has."* This
/// consumer cannot be order-independent -- `valid_labels[1:]` selects by **position** in the
/// label sequence -- so §16.37 item 13 discharges the obligation the third way: `undetected_blocks`
/// iterates `ConnectedComponents::labels_in_opencv_block_scan_order`, i.e. upstream's own order,
/// and therefore has no divergence to register.
///
/// **This test previously asserted `[0, 1, 3, 20]`** -- this crate's answer under
/// first-raster-pixel numbering -- under the name
/// `undetected_blocks_label_numbering_decides_which_component_is_dropped_when_the_background_is_small`.
/// Both the assertion and the name changed under §16.37 item 13's joint ruling (cookbook rule 8:
/// a corrected assertion claiming something *different* about the system, following a joint
/// ruling, is a legitimate exit from a frozen test). The old name asserted that the numbering
/// *decides* the drop, which is a divergence claim; the test now verifies parity, so the name
/// says parity.
///
/// **Where the expected value comes from -- `cv2`, not from this crate.** Measured against
/// **cv2 5.0.0** on this exact 20x20 array, 2026-08-07:
/// `stats` areas `[43, 57, 300]`, `valid_labels == [1, 2]`, `valid_labels[1:] == [2]`, and
/// `stats[2] -> [x, y, x+w, y+h] == [5, 0, 20, 20]`. `cv2` labels **B** 1 and **A** 2 because its
/// block scan reaches B's 2x2 block `(0, 0)` before A's block `(0, 2)`.
///
/// **Why this fixture is the one that matters.** The two orders can only disagree here when
/// `stats[0].area <= 50` -- when the thresholded residual leaves at most 50 background pixels
/// **page-wide**, so `valid[0]` is a real component rather than label 0. Above that both orders
/// put label 0 first and both drop the background. The fixture: component A is `x in 5..20` over
/// every row (first raster pixel `(5, 0)`, area 300); component B is `x in 0..3` over `y in 1..20`
/// (first raster pixel `(0, 1)`, area 57); background is the remaining **43** pixels. Under
/// first-raster-pixel order A is 1 and B is 2, so the pre-fix code invented **B**'s box
/// `[0, 1, 3, 20]`; `cv2` inverts that and invents **A**'s. The committed page is not exposed --
/// its residual background area is **1 987 219**.
///
/// **What turns this red:** reverting `undetected_blocks` to ascending `connected_components`
/// labels (returns `[0, 1, 3, 20]`), or an accessor keyed on the pixel raster index instead of
/// the 2x2-block index (same wrong answer, since SAUF/WU order *is* our order).
#[test]
fn undetected_blocks_matches_upstreams_block_scan_label_order_when_the_background_is_small() {
    let mut mask = GrayImage::new(20, 20);
    fill(&mut mask, 5, 0, 20, 20, 255); // component A
    fill(&mut mask, 0, 1, 3, 20, 255); // component B

    // Anti-vacuity: the regime in which the two orders differ really does hold on this fixture,
    // so a pass here cannot be "the orders happen to agree".
    let components = pc_detect::annotate_merge::connected_components(&mask);
    assert_eq!(components.num_labels(), 3);
    assert_eq!(
        components.stats[0].area, 43,
        "background must fail the > 50 filter"
    );
    assert!(components.stats[0].area <= UNDETECTED_MIN_COMPONENT_AREA);
    // And the two orders really are opposed here. The accessor returns THIS CRATE's labels
    // permuted into cv2's order, so the expected value is `[our label of cv2's first component,
    // our label of cv2's second]`. cv2 5.0.0's first non-background component on this array is B,
    // which this crate numbers 2; its second is A, which this crate numbers 1. Hence `[2, 1]` --
    // a genuine permutation, and the same list ascending would mean the two orders agreed.
    assert_eq!(
        components.labels_in_opencv_block_scan_order(),
        vec![2_u32, 1],
        "cv2 5.0.0 visits B first; B is this crate's label 2"
    );
    assert_eq!(
        components.label_at(5, 0),
        1,
        "this crate's global numbering gives A label 1 -- the opposite first entry"
    );

    // Identity, not cardinality: A's box, which is upstream's answer.
    assert_eq!(
        undetected_blocks(&mask, &[]),
        vec![Rect::new(5, 0, 20, 20)],
        "cv2 5.0.0 drops B (its label 1) and invents A's box [5, 0, 20, 20]"
    );
}

/// **Anti-regression on the thing the fix deliberately did NOT touch:**
/// `connected_components`'s own `labels`/`stats` numbering is still **first raster pixel**, not
/// `cv2`'s block-scan order.
///
/// §16.37 item 13's ruling was to add a *local* accessor, precisely because renumbering globally
/// would contradict A3's frozen
/// `connected_components_label_numbering_is_first_raster_pixel_order_not_opencvs` and silently
/// change `merge_mask_list`/`hole_fill_area_threshold`'s inputs. This test fails if
/// `labels_in_opencv_block_scan_order`'s order ever leaks into the struct's own fields.
///
/// The fixture is the same 20x20 array as the parity test above, where the two orders are
/// opposed, so agreement here cannot be coincidence. Expected values are hard-coded: A's first
/// raster pixel is `(5, 0)` and B's is `(0, 1)`, so ours must give A label 1 and B label 2, with
/// `stats[1].area == 300` (A) and `stats[2].area == 57` (B) -- the exact inverse of cv2 5.0.0's
/// `[43, 57, 300]`.
#[test]
fn connected_components_global_numbering_is_still_first_raster_pixel_after_the_block_scan_accessor()
{
    let mut mask = GrayImage::new(20, 20);
    fill(&mut mask, 5, 0, 20, 20, 255); // component A, first raster pixel (5, 0)
    fill(&mut mask, 0, 1, 3, 20, 255); // component B, first raster pixel (0, 1)

    let components = pc_detect::annotate_merge::connected_components(&mask);
    assert_eq!(components.label_at(5, 0), 1, "A is label 1 -- ours");
    assert_eq!(components.label_at(0, 1), 2, "B is label 2 -- ours");
    assert_eq!(components.stats[1].area, 300, "stats[1] is A, not cv2's B");
    assert_eq!(components.stats[2].area, 57, "stats[2] is B, not cv2's A");
}

// ================================== Part 2 (i): the in-place zeroing of the caller's pred mask

/// spec §16.37 item 11(b)(i): *"`mask_pred[np.where(mask_refined > 30)] = 0` -- a
/// zero-where-already-refined step at threshold **30**, and note it mutates the CALLER's array
/// in place ... (verified by probe, 2026-08-07: a 40x40 pred of uniform 200 with a 5x5 refined
/// patch comes back with **25** pixels zeroed in the caller's own array). A3b decides
/// deliberately whether our port mutates or copies, and records which."*
///
/// **THE DECISION, recorded here and in `annotate_refine`'s module header: this port MUTATES,
/// taking `pred_mask: &mut GrayImage`.** Grounds, in the order they were weighed:
///   1. the mutated array is what upstream's own recursive `refine_mask` call consumes
///      (item 11(b)(v)), so the mutation is **not** optional to the output -- a copy-based port
///      must still zero *something*;
///   2. upstream returns the mutated array to its caller at `inference.py:210`, so the effect is
///      part of the function's observable contract, and a `&GrayImage` signature would silently
///      drop half of it;
///   3. Rust cannot alias-mutate the way Python does, so the choice has to be made explicitly
///      rather than inherited -- an `&mut` parameter is the only form that makes the side effect
///      visible **at every call site**.
///
/// The counter-argument was weighed and rejected: PanelCleaner's own pipeline never reads the
/// mutated array (`ctd_interface.py:182` writes only `mask_refined`), so a copying port would
/// be output-equivalent *in that pipeline* -- but "equivalent for today's single caller" is
/// exactly the reasoning §16.24 item 1(a) is on record as having got wrong.
///
/// Re-measured on upstream 2026-08-07: **25** pixels zeroed in the caller's array, and the
/// caller's array holds exactly 25 zeros afterwards.
#[test]
fn refine_undetected_mask_zeroes_the_callers_pred_mask_where_the_refined_mask_exceeds_thirty() {
    let image = page(40, 40, &[]);
    let mut pred_mask = GrayImage::from_pixel(40, 40, Luma([200]));
    let mut refined = GrayImage::new(40, 40);
    fill(&mut refined, 10, 10, 15, 15, 255);

    let before = pred_mask.clone();
    let _ = refine_undetected_mask(&image, &mut pred_mask, &refined, &[]).expect("equal sizes");

    let changed = before
        .pixels()
        .zip(pred_mask.pixels())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        changed, 25,
        "the 5x5 refined patch zeroes 25 of the caller's pixels"
    );
    assert_eq!(
        pred_mask.pixels().filter(|p| p.0[0] == 0).count(),
        25,
        "and nothing else in the caller's array became zero"
    );
}

/// The threshold in `mask_refined > 30` is **strict**. Measured on upstream 2026-08-07 with a
/// 10x10 pred of uniform 200: a refined value of exactly **30** leaves the pred pixel at 200,
/// and **31** zeroes it.
///
/// **What fails this test:** `>=`, which would zero the first pixel too. Asserted through
/// [`zero_pred_where_refined`] directly so the two pixels are named rather than inferred from a
/// population.
#[test]
fn zero_pred_where_refined_uses_a_strict_threshold_of_thirty() {
    let mut pred_mask = GrayImage::from_pixel(10, 10, Luma([200]));
    let mut refined = GrayImage::new(10, 10);
    refined.put_pixel(0, 0, Luma([30]));
    refined.put_pixel(1, 0, Luma([31]));

    zero_pred_where_refined(&mut pred_mask, &refined).expect("equal sizes");
    assert_eq!(
        pred_mask.get_pixel(0, 0).0[0],
        200,
        "refined == 30 is not > 30"
    );
    assert_eq!(pred_mask.get_pixel(1, 0).0[0], 0, "refined == 31 is > 30");
    assert_eq!(
        pred_mask.pixels().filter(|p| p.0[0] == 0).count(),
        1,
        "exactly one pixel was zeroed"
    );
}

/// Per-image (`StageError::InvalidInput`, cookbook rule 4) on a size mismatch: upstream indexes
/// `mask_pred` with a boolean mask derived from `mask_refined`, which numpy only defines for
/// equal shapes.
#[test]
fn refine_undetected_mask_rejects_size_mismatched_operands() {
    let image = page(20, 20, &[]);
    let mut pred_mask = GrayImage::new(20, 20);
    let refined = GrayImage::new(21, 20);
    let error = refine_undetected_mask(&image, &mut pred_mask, &refined, &[])
        .expect_err("shapes must agree");
    assert!(
        matches!(error, StageError::InvalidInput(_)),
        "a size mismatch is per-image: {error:?}"
    );

    let mut pred_mask = GrayImage::new(21, 20);
    let refined = GrayImage::new(21, 20);
    let error = refine_undetected_mask(&image, &mut pred_mask, &refined, &[])
        .expect_err("the image must match too");
    assert!(matches!(error, StageError::InvalidInput(_)), "{error:?}");
}

// ============================== Part 2 (v): the recursive call, on the ALREADY-ZEROED pred mask

/// spec §16.37 item 11(b)(v): *"the recursive `refine_mask(img, mask_pred, seg_blk_list, ...)`
/// over the invented blocks, taking the **already-zeroed** `mask_pred` from (i), OR-ed into the
/// refined mask -- which is why part 1 must be a callable unit."*
///
/// **This is the test that separates "zeroes the pred mask" from "uses the zeroed pred mask".**
/// Fixture, hand-built on a 40x40 page: ink at `[6, 6, 24, 14)`, pred mask 200 over
/// `[5, 5, 25, 15)`, and an incoming refined mask already saturated over `[5, 5, 16, 15)` --
/// 11 columns x 10 rows = **110** pixels, all `> 30`.
///
/// Measured on upstream 2026-08-07:
///   * **110** pred pixels are zeroed in the caller's array (matching the hand count);
///   * the residual's surviving component has box `(x, y, w, h) = (16, 5, 9, 10)`, i.e.
///     `[16, 5, 25, 15]` -- **not** the pred blob's own `[5, 5, 25, 15]`. That box is the whole
///     point: an implementation that ran the components over the *unzeroed* pred mask would
///     invent `[5, 5, 25, 15]`, whose expanded window is different, and would produce a
///     different page mask;
///   * the returned mask has **174** non-zero pixels and positional fingerprint **68642**.
///
/// 174 is hand-checkable against the grid: 11 + 8 * 19 + 11.
#[test]
fn refine_undetected_mask_feeds_the_zeroed_pred_mask_to_the_recursive_driver() {
    let image = page(40, 40, &[(6, 6, 24, 14, DARK)]);
    let mut pred_mask = GrayImage::new(40, 40);
    fill(&mut pred_mask, 5, 5, 25, 15, 200);
    let mut refined = GrayImage::new(40, 40);
    fill(&mut refined, 5, 5, 16, 15, 255);
    assert_eq!(
        nonzero_count(&refined),
        110,
        "the incoming refined mask, hand-counted"
    );

    let before = pred_mask.clone();
    let out = refine_undetected_mask(&image, &mut pred_mask, &refined, &[]).expect("equal sizes");

    assert_eq!(
        before
            .pixels()
            .zip(pred_mask.pixels())
            .filter(|(a, b)| a != b)
            .count(),
        110,
        "11 columns x 10 rows zeroed in the caller's array"
    );
    // The invented block is the RESIDUAL's box, which is what proves (v) rather than (i).
    assert_eq!(
        undetected_blocks(&pred_mask, &[]),
        vec![Rect::new(16, 5, 25, 15)],
        "the residual's box, not the original pred blob's [5, 5, 25, 15]"
    );

    assert_grid(
        &out,
        &[
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            ".....XXXXXXXXXXX........................",
            ".....XXXXXXXXXXXXXXXXXXX................",
            ".....XXXXXXXXXXXXXXXXXXX................",
            ".....XXXXXXXXXXXXXXXXXXX................",
            ".....XXXXXXXXXXXXXXXXXXX................",
            ".....XXXXXXXXXXXXXXXXXXX................",
            ".....XXXXXXXXXXXXXXXXXXX................",
            ".....XXXXXXXXXXXXXXXXXXX................",
            ".....XXXXXXXXXXXXXXXXXXX................",
            ".....XXXXXXXXXXX........................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
            "........................................",
        ],
    );
    assert_eq!(nonzero_count(&out), 174);
    assert_eq!(positional_fingerprint(&out), 68642);
    // `cv2.bitwise_or(mask_refined, refine_mask(...))`: every incoming pixel survives.
    for y in 5..15 {
        for x in 5..16 {
            assert_eq!(
                out.get_pixel(x, y).0[0],
                255,
                "({x}, {y}) came in refined and must stay on after the OR"
            );
        }
    }
}

/// The invented blocks go through the **same** page-level driver as detected ones, so all of
/// Part 1's behaviour applies to them: window expansion, crop, merge, OR at the window's
/// coordinates.
///
/// **Fixture, hand-built** on a 40x20 page, reusing
/// [`undetected_blocks_requires_a_component_area_strictly_greater_than_fifty`]'s two components
/// so this test also witnesses end to end that P (area 50) contributes nothing:
/// ink at `[3, 3, 6, 11)` and `[21, 3, 22, 18)`, pred 200 over `[2, 2, 7, 12)` and
/// `[20, 2, 23, 19)`, empty detected-block list, empty incoming refined mask.
///
/// Measured on upstream 2026-08-07: the sole invented block `[20, 2, 23, 19]` expands to the
/// window `[19, 1, 24, 20]` and the returned mask has **36** non-zero pixels, fingerprint
/// **15192** -- a hollow outline entirely inside `x in 20..23`, with **nothing** near P.
#[test]
fn refine_undetected_mask_runs_each_invented_block_through_the_page_level_driver() {
    let image = page(40, 20, &[(3, 3, 6, 11, DARK), (21, 3, 22, 18, DARK)]);
    let mut pred_mask = GrayImage::new(40, 20);
    fill(&mut pred_mask, 2, 2, 7, 12, 200);
    fill(&mut pred_mask, 20, 2, 23, 19, 200);
    let refined = GrayImage::new(40, 20);

    let out = refine_undetected_mask(&image, &mut pred_mask, &refined, &[]).expect("equal sizes");
    assert_grid(
        &out,
        &[
            "........................................",
            "........................................",
            "....................XXX.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................X.X.................",
            "....................XXX.................",
            "........................................",
        ],
    );
    assert_eq!(nonzero_count(&out), 36);
    assert_eq!(positional_fingerprint(&out), 15192);
}

/// The base case: nothing invented, so upstream skips the `if len(seg_blk_list) > 0` branch and
/// returns `mask_refined` unchanged. Uses the `valid_labels[1:]` quirk fixture, where a
/// 63-pixel component exists and is nonetheless dropped -- so the "unchanged" result is reached
/// *through* the quirk, not by an absence of input.
///
/// The incoming refined mask carries three arbitrary pixels below the 30 threshold and one
/// above, so "returns the input" is distinguishable from "returns zeros" and from "returns the
/// input with the >30 pixels cleared".
#[test]
fn refine_undetected_mask_returns_the_incoming_refined_mask_when_nothing_is_invented() {
    let image = page(8, 8, &[]);
    let mut pred_mask = GrayImage::from_pixel(8, 8, Luma([255]));
    pred_mask.put_pixel(3, 3, Luma([0]));
    let mut refined = GrayImage::new(8, 8);
    refined.put_pixel(0, 0, Luma([7]));
    refined.put_pixel(1, 1, Luma([30]));
    refined.put_pixel(2, 2, Luma([200]));
    let expected = refined.clone();

    let out = refine_undetected_mask(&image, &mut pred_mask, &refined, &[]).expect("equal sizes");
    assert_eq!(out, expected, "the refined mask is returned untouched");
    // ... while the pred mask still took (i)'s mutation, which happens before the invention test.
    assert_eq!(
        pred_mask.get_pixel(2, 2).0[0],
        0,
        "(i) runs regardless of invention"
    );
    assert_eq!(pred_mask.get_pixel(1, 1).0[0], 255, "30 is not > 30");
}

/// The real-data check for Part 2 -- and spec §16.37 item 11 is explicit that this page is a
/// **degenerate** test for it: *"E01P01 differs by **0** px -- on that page the undetected pass
/// invents no block at all, so its 'agreement' there is degenerate and carries no information
/// about the other two."*
///
/// **So this test's headline assertion is that a list is empty, which is the one shape that can
/// pass by finding nothing at all.** It is therefore paired with a second call on the same
/// residual under an **empty** detected-block list, which must return **12** blocks. Twelve is
/// upstream's own count (measured 2026-08-07: the residual has 17 labels, 13 clear the `> 50`
/// area filter, `valid_labels[1:]` iterates 12, and all 12 are rejected with intersection ratios
/// from 0.8636 to 1.0). Together the two calls say "twelve candidates existed and every one was
/// rejected by the intersection test", which "found nothing" cannot fake.
///
/// The 12 boxes are asserted by identity, sorted by `(y1, x1)`, not by count.
///
/// Also recorded: the residual background area is **1 987 219**, so
/// [`undetected_blocks_matches_upstreams_block_scan_label_order_when_the_background_is_small`]'s
/// order-sensitive condition (`stats[0].area <= 50`) does not hold here, and the drop is the
/// background under both orders. This page therefore cannot corroborate §16.37 item 13's fix and
/// is not offered as doing so.
#[test]
fn refine_undetected_mask_on_the_recorded_page_invents_no_block() {
    let (image, pred_mask) = recorded_page();
    let blocks = recorded_blocks();
    let refined = refine_mask(&image, &pred_mask, &blocks).expect("equal sizes");
    assert_eq!(
        nonzero_count(&refined),
        2942,
        "Part 1's frozen page population"
    );

    let mut mutable = pred_mask.clone();
    let out = refine_undetected_mask(&image, &mut mutable, &refined, &blocks).expect("equal sizes");
    assert_eq!(
        out, refined,
        "nothing is invented, so the page mask does not move"
    );
    assert_eq!(nonzero_count(&out), 2942);
    assert_eq!(positional_fingerprint(&out), 2_790_551_277);
    // (i) still fired: every one of the 2942 refined pixels was non-zero in the pred mask.
    assert_eq!(
        pred_mask
            .pixels()
            .zip(mutable.pixels())
            .filter(|(a, b)| a != b)
            .count(),
        2942,
        "the in-place zeroing ran on the real page too"
    );

    // Anti-vacuity, and the reachability check for the label-order divergence.
    let residual = mutable;
    let components = pc_detect::annotate_merge::connected_components(
        &pc_detect::annotate::threshold_binary(&residual, UNDETECTED_PRED_THRESHOLD),
    );
    assert_eq!(
        components.num_labels(),
        17,
        "upstream's residual label count"
    );
    assert_eq!(
        components.stats[0].area, 1_987_219,
        "the background clears the > 50 filter, so the dropped entry IS label 0 here"
    );
    assert_eq!(
        components
            .stats
            .iter()
            .filter(|s| s.area > UNDETECTED_MIN_COMPONENT_AREA)
            .count(),
        13,
        "13 labels clear the area filter, of which valid_labels[1:] iterates 12"
    );

    let mut candidates = undetected_blocks(&residual, &[]);
    candidates.sort_by_key(|r| (r.y1, r.x1));
    assert_eq!(
        candidates,
        vec![
            Rect::new(577, 77, 636, 98),
            Rect::new(633, 77, 654, 98),
            Rect::new(567, 102, 645, 124),
            Rect::new(646, 103, 663, 124),
            Rect::new(627, 633, 684, 654),
            Rect::new(684, 634, 704, 655),
            Rect::new(628, 657, 703, 680),
            Rect::new(607, 658, 628, 680),
            Rect::new(704, 658, 724, 680),
            Rect::new(646, 684, 685, 706),
            Rect::new(675, 1400, 738, 1438),
            Rect::new(472, 1413, 494, 1433),
        ],
        "the twelve candidates the real detected blocks reject"
    );
    assert!(
        undetected_blocks(&residual, &blocks).is_empty(),
        "all twelve are rejected once the real blocks supply the intersection score"
    );
}

// ================================================================================ helpers

/// A synthetic BGR-equivalent page: [`LIGHT`] paper, with each `(x1, y1, x2, y2, value)`
/// half-open rect filled with `(value, value, value)`. Upstream's arrays come from
/// `cv2.imread`, i.e. BGR; a grey fill is channel-order-independent, which is why the oracle
/// script could use the same numbers.
fn page(width: u32, height: u32, rects: &[(u32, u32, u32, u32, u8)]) -> RgbImage {
    let mut image = RgbImage::from_pixel(width, height, Rgb([LIGHT, LIGHT, LIGHT]));
    for &(x1, y1, x2, y2, value) in rects {
        for y in y1..y2 {
            for x in x1..x2 {
                image.put_pixel(x, y, Rgb([value, value, value]));
            }
        }
    }
    image
}

/// A synthetic pred mask: 0 everywhere, with each half-open rect filled with `value`.
fn pred(width: u32, height: u32, rects: &[(u32, u32, u32, u32, u8)]) -> GrayImage {
    let mut mask = GrayImage::new(width, height);
    for &(x1, y1, x2, y2, value) in rects {
        fill(&mut mask, x1, y1, x2, y2, value);
    }
    mask
}

fn fill(mask: &mut GrayImage, x1: u32, y1: u32, x2: u32, y2: u32, value: u8) {
    for y in y1..y2 {
        for x in x1..x2 {
            mask.put_pixel(x, y, Luma([value]));
        }
    }
}

/// The 44x24 two-overlapping-window fixture, shared by three tests. The mid-grey pixel at
/// `(15, 10)` exists only to make the second block's crop statistics differ from the first's;
/// without it both blocks produce the same mask over the shared columns and the OR-vs-assign
/// distinction disappears (measured: 0 differing pixels).
fn overlap_page() -> RgbImage {
    page(
        44,
        24,
        &[
            (6, 9, 12, 15, DARK),
            (15, 10, 16, 11, 120),
            (26, 9, 34, 15, DARK),
        ],
    )
}

fn overlap_pred() -> GrayImage {
    pred(44, 24, &[(5, 8, 20, 16, 255), (17, 8, 36, 16, 255)])
}

/// Upstream's page mask for [`overlap_page`] / [`overlap_pred`] with blocks
/// `[5, 8, 20, 16]` and `[17, 8, 36, 16]`: block 0's window `[4, 7, 21, 17)` saturated, and
/// nothing else.
const OVERLAP_EXPECTED: &[&str] = &[
    "............................................",
    "............................................",
    "............................................",
    "............................................",
    "............................................",
    "............................................",
    "............................................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "....XXXXXXXXXXXXXXXXX.......................",
    "............................................",
    "............................................",
    "............................................",
    "............................................",
    "............................................",
    "............................................",
    "............................................",
];

/// Row-by-row comparison against a literal grid: `'X'` means non-zero. A population count would
/// pass for a mask of the right size in the wrong place; this does not.
fn assert_grid(mask: &GrayImage, rows: &[&str]) {
    assert_eq!(mask.height() as usize, rows.len(), "fixture row count");
    for (y, row) in rows.iter().enumerate() {
        assert_eq!(mask.width() as usize, row.len(), "fixture row {y} width");
        let actual: String = (0..mask.width())
            .map(|x| {
                if mask.get_pixel(x, y as u32).0[0] != 0 {
                    'X'
                } else {
                    '.'
                }
            })
            .collect();
        assert_eq!(&actual, row, "row {y}");
    }
}

fn nonzero_count(mask: &GrayImage) -> usize {
    mask.pixels().filter(|pixel| pixel.0[0] != 0).count()
}

/// `sum of (row-major index + 1) over non-zero pixels` -- the same positional fingerprint
/// `a2_annotate_otsu.rs` and `a3_annotate_merge.rs` use.
fn positional_fingerprint(mask: &GrayImage) -> u64 {
    mask.pixels()
        .enumerate()
        .filter(|(_, pixel)| pixel.0[0] != 0)
        .map(|(index, _)| index as u64 + 1)
        .sum()
}

/// The committed page and its **unrefined** U-Net mask (§16.37 item 4: `_detector_mask.png`,
/// not `_raw_mask.png`).
fn recorded_page() -> (RgbImage, GrayImage) {
    let image = image::open(paths::recorded(format!("{PAGE_STEM}_base.png")))
        .expect("committed base page must decode")
        .to_rgb8();
    let mask = image::open(paths::recorded(format!("{PAGE_STEM}_detector_mask.png")))
        .expect("committed detector mask must decode")
        .to_luma8();
    assert_eq!(image.dimensions(), (1200, 1660));
    assert_eq!(mask.dimensions(), (1200, 1660));
    (image, mask)
}

/// The four blocks of `…_detector_blocks.json`, in file order. Hard-coded rather than parsed so
/// this test cannot silently follow a fixture edit, and cross-checked against
/// `a3_annotate_merge.rs`'s own copy of the same four rects.
fn recorded_blocks() -> Vec<Rect> {
    vec![
        Rect::new(674, 1397, 740, 1438),
        Rect::new(567, 74, 663, 123),
        Rect::new(607, 631, 724, 703),
        Rect::new(438, 1407, 498, 1446),
    ]
}
