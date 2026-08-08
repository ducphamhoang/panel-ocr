//! Task A3 -- spec §16.37 item 8: *"**A3** (**heavy**): connected components plus the merge
//! and hole-filling loops."* FROZEN with the implementation.
//!
//! **Where every expected value comes from.** §16.37 item 3 is explicit that a self-comparison
//! gate proves determinism and not correctness -- *"A port that is wrong in the same way on
//! every run passes it forever. Correctness evidence therefore comes from A1-A3's per-function
//! hand-derived assertions"*. So every literal below comes from one of exactly two places,
//! never from `pc_detect::annotate_merge`:
//!   * hand derivation, stated in the test's own comment; or
//!   * an independent oracle -- upstream PanelCleaner's `merge_mask_list` /
//!     `connectedComponentsWithStats` / `getStructuringElement` / `erode` at pinned commit
//!     `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3` (`merge_mask_list` at
//!     `comic_text_detector/utils/textmask.py:83-158`), **imported and executed**, against
//!     `cv2` **5.0.0** -- the version `tests/fixtures/recorded/detector/PROVENANCE.json` pins
//!     -- and `numpy` 2.4.4, on 2026-08-07.
//!
//! **The oracle harness was validated before it was trusted.** A line-for-line Python
//! reimplementation of `merge_mask_list` was compared with the imported upstream function on
//! all four blocks of `ja_Pepper-and-Carrot_by-David-Revoy_E01P01` and was byte-identical on
//! each; and the candidate `xor_sum` lists it produced -- `[450963, 435296, 472752, 361494]`
//! for block 2, `[21655, 143200]` for block 3 -- reproduce the literals already frozen in
//! `a2_annotate_otsu.rs`. Only then were the instrumented intermediates (per-candidate label
//! counts, `pred_binary` populations, hole-fill thresholds) read off it.
//!
//! **Anti-vacuity.** Every merge fixture below asserts the merged mask **row by row against a
//! literal ASCII grid**, not a population count, so a swap of two components fails where a
//! count would not; and each grid is accompanied by a hard-coded non-zero count and a
//! positional fingerprint (`sum of (row-major index + 1) over non-zero pixels`) that cannot be
//! computed from the tree by any route these tests take. Several fixtures are specifically
//! constructed so that the *wrong* implementation produces an **empty** mask
//! ([`merge_scores_against_the_eroded_and_binarised_pred_mask_not_the_raw_mask`],
//! [`merge_erodes_the_pred_mask_with_the_plus_kernel_so_diagonal_neighbours_survive`],
//! [`merge_treats_a_diagonal_pair_as_one_eight_connected_component`]) -- so a gate that passed
//! by finding zero of everything would fail those three.
//!
//! **One mutation these tests provably cannot catch, disclosed rather than left for a reader to
//! discover.** Restricting the hole-fill pass to labels `1..num_labels` -- i.e. dropping
//! upstream's inclusion of label 0, which the merge loop *does* skip -- leaves all 22 tests
//! green (re-measured 2026-08-07 with the 22nd test added; the count read 21 before it).
//! That is not a coverage gap: label 0 of `255 - merged` is exactly the set of pixels
//! where `merged == 255`, so OR-ing it in changes nothing, the XOR is unchanged, and the strict
//! comparison rejects it. The mutation is a **no-op**, verified by probe on 2026-08-07, and the
//! proof is in `annotate_merge`'s module header. Every other mutation probed -- the square erode
//! kernel, `MERGE_PRED_THRESHOLD` 60 -> 58, pixel area for bounding-box area, the bounding-box
//! constant 3 -> 2 and 3 -> 4, `sorted_area[-1]` for `[-2]`, a descending candidate sort,
//! 4-connectivity, the raw pred mask, a wholesale bounding-box write instead of OR-with-the-
//! indicator, an unconditional hole fill, and `<=` for `<` -- turns at least one test red.
//!
//! **Two of those probes re-run 2026-08-07, with their exact tallies, because the second one is
//! why the 22nd test exists.** The wholesale bounding-box write fails **10 of 22**, including
//! both order fixtures. **Scoring** the whole bounding box while writing only the label's own
//! pixels fails **4 of 22** -- `annotation_merge_on_the_recorded_page_matches_the_upstream_oracle`,
//! `recorded_page_merge_is_unchanged_when_the_label_order_is_reversed`,
//! `merge_result_depends_on_the_candidate_list_order_when_xor_sums_are_equal`, and
//! [`merge_components_by_xor_is_order_independent_where_the_orders_could_diverge`] -- and
//! **NOT** [`merge_components_by_xor_gives_the_same_mask_for_any_visit_order`], whose uniform-255
//! `pred` accepts every component either way. So at unit level that mutation was caught only by
//! the two recorded-page tests until the 22nd fixture landed.
//!
//! **Scope, corrected 2026-08-07 by spec §16.37 item 11.** A3 covers Annotation primitives; run wiring is A4.
//! This paragraph read *"upstream's
//! `refine_mask` per-block driver, the config gate and §16.37 item 6's coverage decision are A4.
//! Upstream's `refine_undetected_mask` is named by **no** task in §16.37 item 8 and nothing here
//! covers it."* Both halves are now out of date: the per-block driver **and**
//! `refine_undetected_mask` belong to task **A3b** (§16.37 item 11, ratified 2026-08-07, and it
//! was exactly this file's "named by no task" observation that surfaced the gap), while A4 keeps
//! the config gate, item 6's coverage decision and the `DEVIATION(12)` narrowing. What is
//! unchanged is the operative half: **nothing in this file covers either function**. Nothing here
//! asserts anything about `MaskRefineMode::Simple`.

use image::{GrayImage, Luma};
use pc_core::{Rect, StageError};
use pc_detect::annotate::{candidate_mask_list, erode_rect3x3, MaskCandidate};
use pc_detect::annotate_merge::{
    binarize_pred_mask, connected_components, erode_ellipse3x3, hole_fill_area_threshold,
    merge_components_by_xor, merge_mask_list, ComponentStats, ConnectedComponents,
};
use pc_testkit::paths;

const PAGE_STEM: &str = "detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

// ============================================================================ erode

/// spec §16.37 item 8, A3's `merge_mask_list` step at `textmask.py:105-108` --
/// `cv2.erode(pred_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1)))`.
///
/// **Oracle**: `cv2` 5.0.0 on 2026-08-07. The kernel it returns for those arguments is
/// `[[0,1,0],[1,1,1],[0,1,0]]` -- a plus, **not** a square -- and `cv2.erode` on the 5x5 probe
/// below returns the `ELLIPSE` grid asserted here. The `RECT` grid is the output of the same
/// probe under `np.ones((3, 3), np.uint8)`, i.e. what A1's [`erode_rect3x3`] implements, and is
/// asserted as a **difference** so the two kernels cannot be silently interchanged.
///
/// **What fails this test.** Using the 8-neighbourhood minimum (the `RECT` kernel): row 0 comes
/// out `X...X` instead of `XX.XX`. Zero-padding the border instead of `cv2.erode`'s default
/// `borderValue = +DBL_MAX`: every edge pixel drops to 0, so row 0 comes out `.....`.
#[test]
fn erode_ellipse3x3_is_the_plus_shaped_minimum_and_differs_from_the_rect_kernel() {
    let probe = grid(&["XXXXX", "XX.XX", "XXXXX", ".XXXX", "XXXXX"]);

    assert_eq!(
        render(&erode_ellipse3x3(&probe)),
        vec!["XX.XX", "X...X", ".X.XX", "..XXX", ".XXXX"],
        "cv2.erode with getStructuringElement(MORPH_ELLIPSE, (3, 3), (1, 1))"
    );
    assert_eq!(
        render(&erode_rect3x3(&probe)),
        vec!["X...X", "X...X", "....X", "..XXX", "..XXX"],
        "A1's np.ones((3, 3)) erode, asserted so the two kernels stay distinguishable"
    );
}

/// spec §16.37 item 8 -- upstream `textmask.py:109`,
/// `cv2.threshold(pred_mask, 60, 255, cv2.THRESH_BINARY)`, applied after the erode.
///
/// **Oracle**: `cv2.erode` then `cv2.threshold(..., 60, 255, THRESH_BINARY)` on the 1x7 row
/// below returns `.....XX` (measured 2026-08-07); the intermediate eroded row is
/// `[0, 0, 30, 59, 60, 61, 255]`. The erode is a horizontal-only minimum here because the
/// vertical plus-neighbours are all out of image, so each value is replaced by the minimum of
/// itself and its two horizontal neighbours: index 4 (61) drops to 60 and falls out, index 5
/// (255) drops to 61 and survives, index 6 (255) keeps 255. Threshold's strictness is what
/// separates index 4's 60 (off) from index 5's 61 (on).
///
/// The 3x3-island assertion is the erode half on two axes: a 3x3 block of 255 has exactly one
/// pixel all four of whose plus-neighbours are also inside the block, so it erodes to a single
/// pixel (confirmed against `cv2`, same run).
///
/// **What fails this test.** `>=` instead of `>` turns index 4 on. Dropping the erode turns
/// indices 4, 5 and 6 all on and leaves the whole 3x3 island on. Using the square kernel does
/// **not** fail the first assertion (a 1-row image has no diagonal neighbours) and is not
/// claimed to be caught by it -- `erode_ellipse3x3_is_the_plus_shaped_minimum_and_differs_from_the_rect_kernel`
/// and `merge_erodes_the_pred_mask_with_the_plus_kernel_so_diagonal_neighbours_survive` cover
/// that.
#[test]
fn binarize_pred_mask_erodes_then_keeps_only_values_strictly_above_sixty() {
    let row = GrayImage::from_fn(7, 1, |x, _| {
        Luma([[0, 30, 59, 60, 61, 255, 255][x as usize]])
    });
    assert_eq!(
        render(&binarize_pred_mask(&row)),
        vec![".....XX"],
        "the eroded row is [0, 0, 30, 59, 60, 61, 255], of which only 61 and 255 clear 60"
    );

    let mut pred = GrayImage::new(7, 7);
    for y in 2..5 {
        for x in 2..5 {
            pred.put_pixel(x, y, Luma([255]));
        }
    }
    assert_eq!(
        render(&binarize_pred_mask(&pred)),
        vec![".......", ".......", ".......", "...X...", ".......", ".......", "......."],
        "the plus-erode shrinks a 3x3 island of 255 to its centre"
    );
}

// ============================================================ connected components

/// spec §16.37 item 8, A3's `cv2.connectedComponentsWithStats` (`textmask.py:113`, `:137`).
///
/// **Oracle**: `cv2.connectedComponentsWithStats(m, connectivity=8)` at opencv 5.0.0 on
/// 2026-08-07. Every `stats` row below is `cv2`'s, transcribed -- with **one exception, which is
/// this crate's own deliberate substitution rather than `cv2`'s value**: the `single pixel` row's
/// label 0, the one input here with no background pixel at all. There `cv2` returns its
/// uninitialised row `(-1, 2147483647, 0, 0, 0)`; this crate reports `ComponentStats::default()`,
/// i.e. `(0, 0, 0, 0, 0)`, which is what that row asserts. The substitution is documented on
/// [`ConnectedComponents`] and is the subject of
/// [`connected_components_reports_an_empty_label_zero_when_no_pixel_is_background`].
///
/// The `labels` grids are **hand-derived** from this crate's documented rule -- ascending order
/// of each component's first raster pixel -- and coincide with `cv2`'s on all six inputs here;
/// the input on which the two orders *disagree* is
/// [`connected_components_label_numbering_is_first_raster_pixel_order_not_opencvs`].
///
/// **What fails this test.** Reporting `width * height` as `area` (row `three blobs` label 0
/// would become 15, not 11). Omitting label 0 from `stats`, or omitting it from `num_labels`.
/// Computing label 0's bounding box as the whole image rather than the extent of the zero
/// pixels -- see the `all background` and `ring` rows, where they happen to coincide, against
/// `three blobs`, where the zero pixels do span the image and it is the *area* that separates
/// the two readings.
#[test]
fn connected_components_reports_opencv_stats_for_eight_connected_regions() {
    struct Row<'a> {
        name: &'a str,
        input: &'a [&'a str],
        labels: &'a [&'a str],
        /// `cv2`'s `stats`, label 0 first: `(x, y, width, height, area)`.
        stats: &'a [(u32, u32, u32, u32, u32)],
    }

    const ROWS: &[Row] = &[
        Row {
            name: "three blobs",
            input: &["..XX.", ".....", "X...X"],
            labels: &["00110", "00000", "20003"],
            stats: &[
                (0, 0, 5, 3, 11),
                (2, 0, 2, 1, 2),
                (0, 2, 1, 1, 1),
                (4, 2, 1, 1, 1),
            ],
        },
        Row {
            name: "diagonal pair plus a single pixel",
            input: &["......", ".X....", "..X...", "......", "....X.", "......"],
            labels: &["000000", "010000", "001000", "000000", "000020", "000000"],
            stats: &[(0, 0, 6, 6, 33), (1, 1, 2, 2, 2), (4, 4, 1, 1, 1)],
        },
        Row {
            name: "U shape, two provisional runs joined below",
            input: &[".......", ".X...X.", ".XXXXX.", "......."],
            labels: &["0000000", "0100010", "0111110", "0000000"],
            stats: &[(0, 0, 7, 4, 21), (1, 1, 5, 2, 7)],
        },
        Row {
            name: "ring: the enclosed hole is label 0, like the outer background",
            input: &["........", ".XXXX...", ".X..X...", ".XXXX...", "........"],
            labels: &["00000000", "01111000", "01001000", "01111000", "00000000"],
            stats: &[(0, 0, 8, 5, 30), (1, 1, 4, 3, 10)],
        },
        Row {
            name: "all background",
            input: &["...", "..."],
            labels: &["000", "000"],
            stats: &[(0, 0, 3, 2, 6)],
        },
        Row {
            name: "single pixel",
            input: &["X"],
            labels: &["1"],
            stats: &[(0, 0, 0, 0, 0), (0, 0, 1, 1, 1)],
        },
    ];

    for Row {
        name,
        input,
        labels,
        stats,
    } in ROWS
    {
        let components = connected_components(&grid(input));
        assert_eq!(
            components.num_labels() as usize,
            stats.len(),
            "{name}: num_labels"
        );
        assert_eq!(render_labels(&components), *labels, "{name}: labels");
        assert_eq!(
            components.stats,
            stats
                .iter()
                .map(|&(x, y, width, height, area)| ComponentStats {
                    x,
                    y,
                    width,
                    height,
                    area,
                })
                .collect::<Vec<_>>(),
            "{name}: stats"
        );
    }
}

/// spec §16.37 item 8 -- the `all foreground` degenerate case, split out because its `cv2`
/// `stats[0]` row is not a bounding box at all.
///
/// **Oracle**: on `np.full((2, 3), 255)`, `cv2.connectedComponentsWithStats(..., connectivity=8)`
/// returns `num_labels == 2` with `stats[0] == [-1, 2147483647, 0, 0, 0]` -- an uninitialised
/// row, because there is no background pixel to bound (measured 2026-08-07). This crate reports
/// `ComponentStats::default()` for that case instead. The two are behaviourally identical
/// wherever upstream uses the row -- upstream slices `labels[2147483647:2147483647, -1:-1]`,
/// which numpy makes **empty**, exactly as an empty box does here -- and
/// [`hole_fill_area_threshold`]'s doc comment proves the row is unreachable in
/// [`merge_mask_list`] regardless.
///
/// **What fails this test.** Emitting a single label for an all-foreground image (`num_labels`
/// would be 1); or giving label 0 the whole image as its box, which would make it the largest
/// "component" and corrupt [`hole_fill_area_threshold`].
#[test]
fn connected_components_reports_an_empty_label_zero_when_no_pixel_is_background() {
    let components = connected_components(&grid(&["XXX", "XXX"]));
    assert_eq!(components.num_labels(), 2, "background label still counted");
    assert_eq!(
        components.stats[0],
        ComponentStats::default(),
        "no background pixel -> empty box, area 0"
    );
    assert_eq!(
        components.stats[1],
        ComponentStats {
            x: 0,
            y: 0,
            width: 3,
            height: 2,
            area: 6,
        },
        "the one foreground component covers the image"
    );
    assert_eq!(render_labels(&components), vec!["111", "111"]);
}

/// spec §16.37 item 8's *"Connected-component **label order** is a measured non-issue on all
/// three pages … but that is a property of those pages, so A3 states it as a recorded
/// observation, never as a licence to ignore ordering."*
///
/// This test **records the divergence** rather than papering over it. `cv2`'s default 8-way
/// algorithm is block-based (Grana/BBDT) and numbers components by 2x2-block scan order; this
/// crate numbers by first raster pixel. **Oracle**: on the input below, `cv2` returns
/// `labels == [[0,1,0,0,0,3,3,3,3],[0,0,0,2,0,3,3,3,0]]` -- the single pixel at `(1, 3)` is
/// label **2** and the right-hand blob is label **3** -- while first-raster-pixel order gives
/// the blob 2 and the pixel 3 (measured 2026-08-07).
///
/// The two labelings are therefore the **same partition under a permutation**, which is all the
/// merge loops depend on; that the merged mask is order-independent is proved in the module
/// header and tested by
/// [`merge_components_by_xor_gives_the_same_mask_for_any_visit_order`] and
/// [`recorded_page_merge_is_unchanged_when_the_label_order_is_reversed`].
///
/// **What fails this test.** Any change to the numbering rule -- and, deliberately, adopting
/// `cv2`'s BBDT order would fail it too, which is the point: the rule is pinned, not merely
/// described.
#[test]
fn connected_components_label_numbering_is_first_raster_pixel_order_not_opencvs() {
    let components = connected_components(&grid(&[".X...XXXX", "...X.XXX."]));

    assert_eq!(
        render_labels(&components),
        vec!["010002222", "000302220"],
        "ours: the blob (first pixel (0, 5)) precedes the single pixel at (1, 3)"
    );

    // `cv2`'s stats, transcribed, then reordered by the permutation above: cv2 label 1 -> ours
    // 1, cv2 label 3 -> ours 2, cv2 label 2 -> ours 3.
    assert_eq!(
        components.stats,
        vec![
            ComponentStats {
                x: 0,
                y: 0,
                width: 9,
                height: 2,
                area: 9
            },
            ComponentStats {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
                area: 1
            },
            ComponentStats {
                x: 5,
                y: 0,
                width: 4,
                height: 2,
                area: 7
            },
            ComponentStats {
                x: 3,
                y: 1,
                width: 1,
                height: 1,
                area: 1
            },
        ],
        "the same four cv2 stats rows, permuted to our label order"
    );
}

/// spec §16.37 item 8 -- the connectivity actually in force.
///
/// **Oracle and the reason this test exists.** Upstream writes
/// `cv2.connectedComponentsWithStats(candidate_mask, connectivity, cv2.CV_16U)`, but the Python
/// signature is `(image[, labels[, stats[, centroids[, connectivity[, ltype]]]]])`, so those two
/// positionals bind to **output slots** and the connectivity keyword keeps its default 8.
/// Measured on 2026-08-07 on `[[0,0,0,0,255],[0,255,255,0,0],[255,0,255,255,0]]`: the positional
/// calls `(m, 4, CV_16U)` and `(m, 8, CV_16U)` both give `num_labels == 3`, identical to
/// `connectivity=8`, while `connectivity=4` gives 4. On the 2x2 diagonal pair below, `cv2` gives
/// one component of area 2 at `connectivity=8` and two of area 1 at `connectivity=4`.
///
/// For `merge_mask_list` 8 was the intended value, so this port is faithful either way. For
/// upstream's `refine_undetected_mask`, which passes `4`, the effective value is 8 -- an
/// upstream latent defect, recorded in the module header, **outside A3's scope** and not acted
/// on here.
///
/// **What fails this test.** 4-connectivity: the pair splits into two 1x1 components.
#[test]
fn connected_components_joins_diagonal_neighbours_as_opencv_default_eight_connectivity_does() {
    let components = connected_components(&grid(&["X.", ".X"]));
    assert_eq!(
        components.num_labels(),
        2,
        "one foreground component, not two"
    );
    assert_eq!(
        components.stats[1],
        ComponentStats {
            x: 0,
            y: 0,
            width: 2,
            height: 2,
            area: 2
        },
        "cv2 connectivity=8 stats for the diagonal pair"
    );
}

// ================================================================ area threshold

/// spec §16.37 item 8 -- upstream `textmask.py:140-144`'s
/// `area_thresh = sorted_area[-2] if len(sorted_area) > 1 else sorted_area[-1]`.
///
/// **Oracle**: `cv2` on the two inversions below, 2026-08-07. A 1-pixel-thick 9x9 frame inside
/// an 11x11 crop inverts to areas `[32, 40, 49]` (label 0 = the 32-pixel frame, the 40-pixel
/// outer background, the 49-pixel interior), so the threshold is **40**. An all-zero 5x5 mask
/// inverts to areas `[0, 25]`, so the threshold is **0** -- which is the fact that makes
/// `cv2`'s uninitialised `stats[0]` row unreachable, since `0 < 0` is false.
///
/// **What fails this test.** Taking the largest area (49, then 25). Taking the smallest (32,
/// then 0 -- so the second row alone does not catch it, which is why both rows are here).
/// Excluding label 0 from the sort (the first row would give 49).
#[test]
fn hole_fill_area_threshold_is_the_second_largest_component_area() {
    let mut frame = GrayImage::new(11, 11);
    for y in 1..10 {
        for x in 1..10 {
            if y == 1 || y == 9 || x == 1 || x == 9 {
                frame.put_pixel(x, y, Luma([255]));
            }
        }
    }
    let inverted = GrayImage::from_fn(11, 11, |x, y| Luma([255 - frame.get_pixel(x, y).0[0]]));
    let components = connected_components(&inverted);
    assert_eq!(
        sorted_areas(&components),
        vec![32, 40, 49],
        "cv2's three areas for the inverted frame"
    );
    assert_eq!(hole_fill_area_threshold(&components), 40);

    let all_foreground =
        connected_components(&grid(&["XXXXX", "XXXXX", "XXXXX", "XXXXX", "XXXXX"]));
    assert_eq!(sorted_areas(&all_foreground), vec![0, 25]);
    assert_eq!(
        hole_fill_area_threshold(&all_foreground),
        0,
        "so `area < area_thresh` is false for every label, including the empty label 0"
    );
}

// ================================================================= the merge loop

/// spec §16.37 item 8 -- upstream's `if w * h < 3: continue` (`textmask.py:119-120`).
///
/// **Oracle**: upstream's own `merge_mask_list` imported and run on exactly these two arrays on
/// 2026-08-07, returning the grid asserted below. `cv2`'s labelling of the candidate, also
/// transcribed, is what makes the fixture readable: six components with `(w*h, area)` of
/// `(1,1)`, `(2,2)`, `(3,3)`, `(4,2)`, `(2,2)` and `(6,4)`.
///
/// **The discriminating pair.** The `(4, 2)` component is a two-pixel **diagonal**: pixel area
/// 2, bounding-box area 4. It survives. The `(2, 2)` components are two-pixel horizontal and
/// vertical runs: pixel area 2, bounding-box area 2. They are skipped. So the guard is on
/// `w * h`, and an implementation reading `area` instead loses the diagonal and keeps nothing
/// else it should not -- the two readings are separated by this one fixture.
///
/// The `(6, 4)` component is the XOR half: its bounding box passes the filter, but only 1 of its
/// 4 pixels lies on `pred_binary`, so it is rejected. That keeps the test from asserting
/// "everything big enough is kept".
///
/// **What fails this test.** Reading `area` instead of `w * h` (row 5 loses `..X`, row 6 loses
/// `...X`). Dropping the guard entirely (row 2 gains `..X` and `..XX`, row 5 gains `X` at 6).
/// Using `<=` instead of `<` (row 2 loses the 1x3 run `XXX`).
#[test]
fn merge_filters_components_by_bounding_box_area_not_pixel_area() {
    let candidate = grid(&[
        "..............",
        "..............",
        "..X..XX..XXX..",
        "..............",
        "..............",
        "..X...X.......",
        "...X..X.......",
        "..............",
        ".........X....",
        ".........XX...",
        "..........X...",
    ]);
    let mut pred = GrayImage::new(14, 11);
    for y in 1..10 {
        for x in 1..13 {
            pred.put_pixel(x, y, Luma([255]));
        }
    }

    let merged = merge_mask_list(vec![candidate_of(candidate, 0)], &pred).expect("equal sizes");
    assert_eq!(
        render(&merged),
        vec![
            "..............",
            "..............",
            ".........XXX..", // w*h = 3, kept
            "..............",
            "..............",
            "..X...........", // w*h = 4 (diagonal, pixel area 2), kept
            "...X..........",
            "..............",
            "..............", // w*h = 6 but only 1 of 4 pixels on pred_binary, rejected
            "..............",
            "..............",
        ],
        "the 1x1, 1x2 and 2x1 components are skipped by w*h < 3"
    );
    assert_eq!(nonzero_count(&merged), 5);
    assert_eq!(positional_fingerprint(&merged), 278);
}

/// spec §16.37 item 8 -- upstream `textmask.py:103-109`: the merge scores against
/// `cv2.threshold(cv2.erode(pred_mask, ELLIPSE_3x3), 60, 255, THRESH_BINARY)`, **not** against
/// the raw pred mask A1/A2's `xor_sum` uses.
///
/// **Hand-derived, then confirmed against upstream.** On a uniform pred mask of **127** the two
/// readings point in opposite directions. Binarised: `127 > 60`, so every pred pixel is 255 and
/// turning a component pixel on changes its XOR contribution from `0 ^ 255 = 255` to
/// `255 ^ 255 = 0` -- the sum falls, so the component is accepted. Raw: turning it on changes
/// the contribution from `0 ^ 127 = 127` to `255 ^ 127 = 128` -- the sum **rises** by 1 per
/// pixel, so the component is rejected. Upstream's `merge_mask_list` on this input returns the
/// 9-pixel grid asserted below (2026-08-07).
///
/// The second half is the erode: a 3x3 island of pred 255 erodes to a single pixel, so 1 of the
/// component's 9 pixels is on and 8 are off, the XOR rises, and the component is rejected --
/// giving an **all-zero** mask. Without the erode all 9 would be on and the component kept.
///
/// **What fails this test.** Scoring against the raw pred mask (first assertion becomes empty).
/// Skipping the erode, or eroding after thresholding a 0/255 mask *and* also skipping it here
/// (second assertion becomes the full 3x3). Thresholding at 30 rather than 60 does **not** fail
/// this test and is not claimed to be caught by it; `binarize_pred_mask_erodes_then_keeps_only_values_strictly_above_sixty`
/// is what pins the constant.
#[test]
fn merge_scores_against_the_eroded_and_binarised_pred_mask_not_the_raw_mask() {
    let block = grid(&[
        ".......", ".......", "..XXX..", "..XXX..", "..XXX..", ".......", ".......",
    ]);

    let uniform_127 = GrayImage::from_pixel(7, 7, Luma([127]));
    let merged =
        merge_mask_list(vec![candidate_of(block.clone(), 0)], &uniform_127).expect("equal sizes");
    assert_eq!(
        render(&merged),
        vec![".......", ".......", "..XXX..", "..XXX..", "..XXX..", ".......", "......."],
        "pred 127 binarises to 255, so the component lowers the XOR and is accepted"
    );
    assert_eq!(nonzero_count(&merged), 9);
    assert_eq!(positional_fingerprint(&merged), 225);

    let mut island = GrayImage::new(7, 7);
    for y in 2..5 {
        for x in 2..5 {
            island.put_pixel(x, y, Luma([255]));
        }
    }
    let merged = merge_mask_list(vec![candidate_of(block, 0)], &island).expect("equal sizes");
    assert_eq!(
        nonzero_count(&merged),
        0,
        "the pred island erodes to 1 px, so 8 of the component's 9 pixels are off pred and it \
         is rejected"
    );
}

/// spec §16.37 item 8 -- that the pred-mask erode uses `MORPH_ELLIPSE` (a plus) and not
/// `np.ones((3, 3))` (a square), asserted through `merge_mask_list`'s observable output rather
/// than only through [`erode_ellipse3x3`].
///
/// **Oracle**: upstream's `merge_mask_list` on the two arrays below returns the candidate
/// unchanged -- 7 pixels (2026-08-07). The pred mask is uniform 255 with a single 0 at `(3, 3)`.
/// The plus-erode zeroes `pred_binary` at `(3,3)` and its four orthogonal neighbours only; the
/// square-erode would additionally zero the four **diagonal** neighbours `(2,2)`, `(2,4)`,
/// `(4,2)`, `(4,4)` -- and the candidate is one 8-connected component consisting of exactly
/// those four diagonals plus three pixels outside the affected zone. So under the plus kernel
/// all 7 pixels are on pred and the component is accepted; under the square kernel 3 are on and
/// 4 are off, the XOR rises, and the merged mask is **empty**.
///
/// **What fails this test.** Using A1's [`erode_rect3x3`] for the pred mask: the result becomes
/// an all-zero grid.
#[test]
fn merge_erodes_the_pred_mask_with_the_plus_kernel_so_diagonal_neighbours_survive() {
    let candidate = grid(&[
        ".......", "...X...", "..X.X..", ".X.....", "..X.X..", "...X...", ".......",
    ]);
    let mut pred = GrayImage::from_pixel(7, 7, Luma([255]));
    pred.put_pixel(3, 3, Luma([0]));

    let merged = merge_mask_list(vec![candidate_of(candidate, 0)], &pred).expect("equal sizes");
    assert_eq!(
        render(&merged),
        vec![".......", "...X...", "..X.X..", ".X.....", "..X.X..", "...X...", "......."],
        "all seven pixels of the one component sit on pred_binary under the plus kernel"
    );
    assert_eq!(nonzero_count(&merged), 7);
    assert_eq!(positional_fingerprint(&merged), 173);
}

/// spec §16.37 item 8 -- 8-connectivity, asserted through `merge_mask_list`'s output.
///
/// **Oracle**: upstream's `merge_mask_list` returns the 2-pixel grid below (2026-08-07). The
/// candidate holds a diagonal pair at `(1,1)`/`(2,2)` and an isolated pixel at `(4,4)`; pred is
/// uniform 255. Under 8-connectivity the pair is one component with `w*h = 4`, so it clears the
/// `w*h < 3` filter and is accepted, while the isolated pixel has `w*h = 1` and is skipped.
///
/// **What fails this test.** 4-connectivity: the pair becomes two 1x1 components, both fall
/// under `w*h < 3`, and the merged mask is **empty**. This is the one place the connectivity
/// choice is observable through the public merge -- the diagonal-pair CC test asserts the
/// labelling, this one asserts the consequence.
#[test]
fn merge_treats_a_diagonal_pair_as_one_eight_connected_component() {
    let candidate = grid(&["......", ".X....", "..X...", "......", "....X.", "......"]);
    let merged = merge_mask_list(
        vec![candidate_of(candidate, 0)],
        &GrayImage::from_pixel(6, 6, Luma([255])),
    )
    .expect("equal sizes");
    assert_eq!(
        render(&merged),
        vec!["......", ".X....", "..X...", "......", "......", "......"],
        "the diagonal pair clears w*h < 3; the isolated pixel does not"
    );
    assert_eq!(nonzero_count(&merged), 2);
    assert_eq!(positional_fingerprint(&merged), 23);
}

/// spec §16.37 item 8 -- that the acceptance comparison at `textmask.py:131` and `:156` is
/// **strict** (`if xor_merged < xor_origin`), so a component that leaves the XOR *unchanged* is
/// rejected.
///
/// **This test exists because a falsification probe found the gap.** Relaxing the implementation
/// to `<=` left the other nineteen tests in this file green: neither the committed page nor any
/// other fixture here contains a component whose accept/reject decision is an exact tie. This
/// fixture is constructed to be one.
///
/// **Oracle**: upstream's `merge_mask_list` on both arrays below, 2026-08-07. The tie case
/// returns an **all-zero** mask; the control returns the two-pixel candidate.
///
/// **Hand derivation.** A component contributes `(255 ^ p) - p` per not-yet-merged pixel, which
/// is `-255` where `pred_binary` is on and `+255` where it is off -- so the decision is exactly
/// *does the component cover strictly more on-pixels than off-pixels*. The candidate is one
/// diagonal pair `{(1,1),(2,2)}` (`w*h == 4`, so it clears the bounding-box filter). In the tie
/// case a 3x3 pred island erodes to the 2x2 block `(0..2, 0..2)`, which covers `(1,1)` but not
/// `(2,2)`: one on, one off, delta **exactly zero**. In the control a 4x4 island erodes to the
/// 3x3 block `(0..3, 0..3)`, covering both: two on, zero off, delta `-510`.
///
/// **What fails this test.** `<=` instead of `<` (the first assertion becomes 2 pixels). A
/// non-strict comparison in only one of the two upstream loops cannot be distinguished here,
/// because both share [`merge_components_by_xor`]; that sharing is the implementation's choice
/// and is documented as such rather than claimed to be separately verified.
#[test]
fn merge_rejects_a_component_that_leaves_the_xor_unchanged() {
    let candidate = grid(&["......", ".X....", "..X...", "......", "......", "......"]);

    let mut tied_pred = GrayImage::new(6, 6);
    for y in 0..3 {
        for x in 0..3 {
            tied_pred.put_pixel(x, y, Luma([255]));
        }
    }
    assert_eq!(
        render(&binarize_pred_mask(&tied_pred)),
        vec!["XX....", "XX....", "......", "......", "......", "......"],
        "the 3x3 island erodes to a 2x2 block: (1,1) is on, (2,2) is off"
    );
    let merged =
        merge_mask_list(vec![candidate_of(candidate.clone(), 0)], &tied_pred).expect("equal sizes");
    assert_eq!(
        nonzero_count(&merged),
        0,
        "one on-pixel and one off-pixel is a tie, and a tie is rejected"
    );

    let mut better_pred = GrayImage::new(6, 6);
    for y in 0..4 {
        for x in 0..4 {
            better_pred.put_pixel(x, y, Luma([255]));
        }
    }
    let merged =
        merge_mask_list(vec![candidate_of(candidate, 0)], &better_pred).expect("equal sizes");
    assert_eq!(
        render(&merged),
        vec!["......", ".X....", "..X...", "......", "......", "......"],
        "with both pixels on pred the XOR strictly falls and the component is accepted"
    );
    assert_eq!(nonzero_count(&merged), 2);
    assert_eq!(positional_fingerprint(&merged), 23);
}

/// `merge_mask_list` does its own sort of the combined candidate list by each entry's declared
/// `xor_sum`, and that sort's stability decides which of two equally-scoring candidates the merge
/// visits first. **That sentence describes the code, not the spec** -- it is a paraphrase of this
/// module's behaviour and is deliberately not presented as a quotation, because no such sentence
/// appears in `docs/PIPELINE_SPEC_V1.md`. What the spec does supply is the task assignment: spec
/// §16.37 item 8 puts *"connected components plus the merge and hole-filling loops"* in A3, which
/// is what puts this ordering dependence in scope here. The concatenation that produces the
/// equal-key run is `annotate::candidate_mask_list`'s documented contract (Otsu last).
///
/// **Oracle**: upstream's `merge_mask_list` run twice on the same two arrays with the list order
/// swapped, both candidates declaring `xor_sum = 1000` (2026-08-07). `[A, B]` returns the 3-pixel
/// grid; `[B, A]` returns the 4-pixel grid.
///
/// **Hand derivation of why the two differ.** `pred_binary` is on for rows 2-4, columns 2-6.
/// `A` is one component `{(2,2),(2,3),(2,4)}`, all on pred. `B` is one component
/// `{(1,2),(2,2),(2,3)}`, of which `(1,2)` is **off** pred. Visited alone each is accepted (3-0
/// and 2-1 in favour). Visited **second**, `B`'s already-merged pixels `(2,2)`/`(2,3)` no longer
/// contribute, leaving only `(1,2)`, which is off pred -- so `B` is rejected and `(1,2)` stays
/// dark. In the other order `A` visited second still has `(2,4)`, which is on pred, so `A` is
/// accepted and both `(1,2)` and `(2,4)` end up lit.
///
/// **What this test detects, stated exactly.** That `merge_mask_list`'s output depends on the
/// input list's order in this direction -- i.e. that the sort does not reverse or otherwise
/// permute an equal-key run, and that the merge honours the resulting order. It does **not**
/// prove the sort is a stable one in general: `sort_unstable_by_key` on a two-element run of
/// equal keys happens not to swap them either, so that particular mutation would survive this
/// test. The stability guarantee comes from the API choice (`sort_by_key`) and its doc comment;
/// what is asserted here is the observable order dependence, which is what any future change
/// has to preserve.
#[test]
fn merge_result_depends_on_the_candidate_list_order_when_xor_sums_are_equal() {
    let first = grid(&[
        ".........",
        ".........",
        "..XXX....",
        ".........",
        ".........",
        ".........",
        ".........",
    ]);
    let second = grid(&[
        ".........",
        "..X......",
        "..XX.....",
        ".........",
        ".........",
        ".........",
        ".........",
    ]);
    let mut pred = GrayImage::new(9, 7);
    for y in 1..6 {
        for x in 1..8 {
            pred.put_pixel(x, y, Luma([255]));
        }
    }

    let forward = merge_mask_list(
        vec![
            candidate_of(first.clone(), 1000),
            candidate_of(second.clone(), 1000),
        ],
        &pred,
    )
    .expect("equal sizes");
    assert_eq!(
        render(&forward),
        vec![
            ".........",
            ".........",
            "..XXX....",
            ".........",
            ".........",
            ".........",
            "........."
        ],
        "first-listed candidate wins the tie, so (1, 2) is never merged"
    );
    assert_eq!(nonzero_count(&forward), 3);
    assert_eq!(positional_fingerprint(&forward), 66);

    let reversed = merge_mask_list(
        vec![candidate_of(second, 1000), candidate_of(first, 1000)],
        &pred,
    )
    .expect("equal sizes");
    assert_eq!(
        render(&reversed),
        vec![
            ".........",
            "..X......",
            "..XXX....",
            ".........",
            ".........",
            ".........",
            "........."
        ],
        "swapping the equal-scoring candidates changes the merged mask"
    );
    assert_eq!(nonzero_count(&reversed), 4);
    assert_eq!(positional_fingerprint(&reversed), 78);
}

/// spec §16.37 item 8's *"hole-filling loops"* -- upstream `textmask.py:137-157`.
///
/// **Oracle**: upstream's `merge_mask_list` on a 1-pixel-thick 5x5 ring inside a 9x9 crop with
/// pred uniform 255 returns a **solid** 5x5 block, 25 pixels (2026-08-07).
///
/// **Hand derivation.** The ring (16 px) is accepted by the merge loop. Inverting gives areas
/// `[9, 16, 56]` -- the 3x3 interior, label 0 (the ring itself), and the outer background -- so
/// the threshold is 16. The interior's 9 < 16 and all nine of its pixels are on `pred_binary`,
/// so it is filled. Label 0's 16 is not < 16, and would be a no-op anyway. The outer 56 is not
/// < 16.
///
/// **What fails this test.** Omitting the hole-fill pass (25 -> 16, and the interior rows read
/// `..X...X..`). Labelling the holes on `merged` rather than on `255 - merged`, since
/// `connected_components` lumps every zero pixel into label 0 and would find no hole at all.
#[test]
fn merge_fills_a_hole_enclosed_by_an_accepted_component() {
    let ring = grid(&[
        ".........",
        ".........",
        "..XXXXX..",
        "..X...X..",
        "..X...X..",
        "..X...X..",
        "..XXXXX..",
        ".........",
        ".........",
    ]);
    let merged = merge_mask_list(
        vec![candidate_of(ring, 0)],
        &GrayImage::from_pixel(9, 9, Luma([255])),
    )
    .expect("equal sizes");
    assert_eq!(
        render(&merged),
        vec![
            ".........",
            ".........",
            "..XXXXX..",
            "..XXXXX..",
            "..XXXXX..",
            "..XXXXX..",
            "..XXXXX..",
            ".........",
            ".........",
        ],
        "the 3x3 interior is smaller than the second-largest area and lies on pred"
    );
    assert_eq!(nonzero_count(&merged), 25);
    assert_eq!(positional_fingerprint(&merged), 1025);
}

/// spec §16.37 item 8's hole-filling loop -- specifically that its threshold is the **second
/// largest** area (`textmask.py:142`), asserted where the two readings diverge.
///
/// **Oracle**: upstream's `merge_mask_list` on a 1-pixel-thick 9x9 frame inside an 11x11 crop
/// with pred uniform 255 returns the frame **unchanged**, 32 pixels (2026-08-07).
///
/// **Hand derivation.** The frame (32 px) is accepted. Inverting gives areas `[32, 40, 49]`:
/// label 0 is the frame, 40 is the outer background, 49 is the interior. The threshold is the
/// second largest, **40**, so nothing is < 40 and nothing is filled. Under the largest (49) the
/// 40-pixel outer background would be filled, since every one of its pixels is on `pred_binary`
/// -- giving 72 pixels and a mask that is on at the four corners. This is the fixture chosen
/// because the interior hole is deliberately **larger** than the outer background, which is the
/// only arrangement in which the `[-2]` and `[-1]` readings disagree.
///
/// **What fails this test.** `sorted_area[-1]` instead of `sorted_area[-2]` (the frame gains the
/// 40-pixel outer ring). Excluding label 0 from the area sort (threshold becomes 40 still --
/// so this test does not claim to catch that; the `hole_fill_area_threshold` test does).
#[test]
fn merge_leaves_a_hole_larger_than_the_second_largest_component_unfilled() {
    let mut frame = GrayImage::new(11, 11);
    for y in 1..10 {
        for x in 1..10 {
            if y == 1 || y == 9 || x == 1 || x == 9 {
                frame.put_pixel(x, y, Luma([255]));
            }
        }
    }
    let merged = merge_mask_list(
        vec![candidate_of(frame, 0)],
        &GrayImage::from_pixel(11, 11, Luma([255])),
    )
    .expect("equal sizes");
    assert_eq!(
        render(&merged),
        vec![
            "...........",
            ".XXXXXXXXX.",
            ".X.......X.",
            ".X.......X.",
            ".X.......X.",
            ".X.......X.",
            ".X.......X.",
            ".X.......X.",
            ".X.......X.",
            ".XXXXXXXXX.",
            "...........",
        ],
        "area_thresh is 40, so neither the 49-pixel interior nor the 40-pixel outside is filled"
    );
    assert_eq!(nonzero_count(&merged), 32);
    assert_eq!(positional_fingerprint(&merged), 1952);
}

/// spec §16.37 item 8 -- the degenerate path where no component is ever accepted, which
/// `hole_fill_area_threshold`'s doc comment proves cannot reach `cv2`'s uninitialised
/// `stats[0]` row.
///
/// **Oracle**: upstream's `merge_mask_list` with an all-255 candidate against an all-zero pred
/// mask returns an all-zero 5x5 mask (2026-08-07); and `cv2` on `255 - zeros` returns
/// `num_labels == 2` with areas `[0, 25]`, so `area_thresh == 0` and the `area < area_thresh`
/// guard rejects both labels.
///
/// **What fails this test.** Accepting a component that raises the XOR. Treating the empty
/// label 0 as covering the image, which would make it a fillable "hole" and light every pixel.
#[test]
fn merge_returns_an_all_zero_mask_when_every_component_raises_the_xor() {
    let merged = merge_mask_list(
        vec![candidate_of(GrayImage::from_pixel(5, 5, Luma([255])), 0)],
        &GrayImage::new(5, 5),
    )
    .expect("equal sizes");
    assert_eq!(merged.dimensions(), (5, 5));
    assert_eq!(nonzero_count(&merged), 0);
}

/// spec §16.37 item 8 -- the empty-candidate-list case, which A4's driver can reach for a block
/// whose eroded mask yields no candidates.
///
/// **Oracle**: upstream's `merge_mask_list([], pred)` on an all-255 4x4 pred mask returns an
/// all-zero 4x4 array (2026-08-07) -- `np.zeros_like(pred_mask)` survives the hole-fill pass
/// untouched for the reason above.
///
/// **What fails this test.** Returning the pred mask, or a mask of a different size, or
/// panicking on an empty `stats` sort.
#[test]
fn merge_returns_an_all_zero_mask_of_the_pred_mask_size_for_an_empty_candidate_list() {
    let merged = merge_mask_list(Vec::new(), &GrayImage::from_pixel(4, 4, Luma([255])))
        .expect("no operands");
    assert_eq!(merged.dimensions(), (4, 4));
    assert_eq!(nonzero_count(&merged), 0);
}

/// cookbook rule 4 -- a size mismatch is classified **per-image**, not run-fatal.
///
/// Upstream's `cv2.bitwise_xor` raises for mismatched shapes; one malformed block must not
/// abort a whole run, so this returns `StageError::InvalidInput`, the same classification A1's
/// `candidate_grey_values` and A2's `xor_sum` already carry. The classification is *declared*
/// here, not inferred from the signature.
///
/// **What fails this test.** Panicking, silently cropping to the smaller size, or returning a
/// run-fatal variant.
#[test]
fn merge_rejects_a_candidate_whose_size_differs_from_the_pred_mask_as_per_image() {
    let error = merge_mask_list(
        vec![candidate_of(GrayImage::new(4, 4), 0)],
        &GrayImage::new(5, 5),
    )
    .expect_err("size mismatch must be rejected");
    assert!(
        matches!(error, StageError::InvalidInput(_)),
        "per-image, not run-fatal: {error:?}"
    );

    let error = merge_components_by_xor(
        &mut GrayImage::new(4, 4),
        &connected_components(&GrayImage::new(5, 5)),
        &GrayImage::new(5, 5),
        &[],
    )
    .expect_err("size mismatch must be rejected");
    assert!(matches!(error, StageError::InvalidInput(_)), "{error:?}");

    let error = merge_components_by_xor(
        &mut GrayImage::new(4, 4),
        &connected_components(&GrayImage::new(4, 4)),
        &GrayImage::new(4, 4),
        &[7],
    )
    .expect_err("a nonexistent label must be rejected");
    assert!(matches!(error, StageError::InvalidInput(_)), "{error:?}");
}

/// spec §16.37 item 8's *"Connected-component **label order** is a measured non-issue on all
/// three pages … but that is a property of those pages, so A3 states it as a recorded
/// observation, never as a licence to ignore ordering."*
///
/// This test asserts the **structural** claim the module header proves, on a fixture built to
/// stress the one thing that makes the claim non-obvious: two components whose **bounding boxes
/// overlap**. The outer frame's box is the whole 7x5 image and strictly contains the inner
/// bar's, so each component's scoring window covers the other's pixels.
///
/// **Oracle**: upstream's `merge_mask_list` on this candidate with pred uniform 255 returns the
/// candidate unchanged, 20 pixels (2026-08-07); and the same value was produced by the
/// validated reimplementation with the non-background label order **reversed**.
///
/// **What fails this test, corrected 2026-08-07.** This paragraph read: *"writing pixels of a
/// different label inside the box (e.g. `merged[bbox] = 255` wholesale …), which is exactly the
/// mutation that would make the result order-dependent -- the two assertions would then
/// disagree."* **The last clause was false of this fixture, and measured to be false**: with
/// `pred` uniform 255 every component is accepted whatever the order, so the wholesale-write
/// mutant returns **35** pixels under *both* orders (the whole 7x5 bounding box) -- wrong, but
/// order-independent. The assertions that actually catch it here are the per-order
/// `render`/`nonzero_count`/`positional_fingerprint` triple against the hand-derived 20 /
/// **351**: the mutant's 35 pixels and fingerprint 630 match none of them. So this fixture pins
/// the *value*, and what it independently pins about *order* is only that scoring or writing
/// outside a component's own bounding box moves the two orders' results relative to each other
/// when the decisions differ -- which they do not here.
///
/// **That gap is closed by
/// [`merge_components_by_xor_is_order_independent_where_the_orders_could_diverge`] below**, whose
/// fixture rejects one of its three components so the visit orders *can* disagree. There the same
/// wholesale-write mutant measurably returns **49** pixels for the three of the six visit orders
/// that reach label 1 before label 2, and **3** pixels for the three that do not -- six orders
/// split 3-and-3, and no order giving that fixture's correct 27. (The 35 above is *this*
/// fixture's mutant count, identical under both of its two orders; it is not the 22nd test's.)
#[test]
fn merge_components_by_xor_gives_the_same_mask_for_any_visit_order() {
    let candidate = grid(&["XXXXXXX", "X......", "X.XXX..", "X......", "XXXXXXX"]);
    let pred = GrayImage::from_pixel(7, 5, Luma([255]));
    let expected = vec!["XXXXXXX", "X......", "X.XXX..", "X......", "XXXXXXX"];

    let components = connected_components(&candidate);
    assert_eq!(
        components.num_labels(),
        3,
        "two components with nested bounding boxes"
    );
    assert_eq!(
        components.stats[1].bounding_box_area(),
        35,
        "the frame's box is the whole image"
    );
    assert_eq!(components.stats[2].bounding_box_area(), 3);

    let pred_binary = binarize_pred_mask(&pred);
    for order in [vec![1_u32, 2], vec![2, 1]] {
        let mut merged = GrayImage::new(7, 5);
        merge_components_by_xor(&mut merged, &components, &pred_binary, &order)
            .expect("equal sizes");
        assert_eq!(render(&merged), expected, "visit order {order:?}");
        assert_eq!(nonzero_count(&merged), 20, "visit order {order:?}");
        assert_eq!(
            positional_fingerprint(&merged),
            351,
            "visit order {order:?}"
        );
    }

    // and the same through the public entry point, which chooses the order itself.
    let merged = merge_mask_list(vec![candidate_of(candidate, 0)], &pred).expect("equal sizes");
    assert_eq!(render(&merged), expected);
}

/// spec §16.37 item 12's *"the merged mask does not depend on the order at all"*, scoped there
/// to *"`merge_mask_list`'s two loops only"* -- the same claim as the test above, on a fixture
/// that can actually discriminate order.
///
/// **Why a second fixture exists.** The fixture above uses a uniform-255 `pred`, so **every**
/// component lowers the XOR and is accepted regardless of when it is visited; the two orders
/// there cannot diverge even for an implementation that writes other labels' pixels, which is
/// what made the original "the two assertions would then disagree" claim false. Here `pred` is
/// **non-uniform** and one of the three components is **rejected**, which is the configuration
/// in which a wrong write is observably order-dependent.
///
/// **The fixture, and why these exact counts.** 7x7, three components. Label 1 is the frame
/// (24 px, bounding box the whole **49**-px image, so it strictly contains the other two); label
/// 2 is a 3-px bar at row 2; label 3 is a 3-px bar at row 4, two rows away from label 2 so the
/// two stay separate under 8-connectivity. `pred_binary` is hand-built (a 0/255 mask, which is
/// what [`binarize_pred_mask`] produces) with **26** pixels set, split so that:
///   * 15 of the frame's 24 pixels are on and 9 are off, so the frame's own delta is
///     `(-15 + 9) x 255 < 0` and it is **ACCEPTED**;
///   * all 3 of label 2's pixels are on, so its delta is `-3 x 255` and it is **ACCEPTED**;
///   * all 3 of label 3's pixels are off, so its delta is `+3 x 255` and it is **REJECTED** --
///     the discriminating half, absent from the fixture above;
///   * 8 of the remaining 19 in-box pixels are on, chosen so that the *wholesale-write* mutant's
///     decision for the frame lands exactly on the strict-comparison boundary
///     (`23 x 255` against `23 x 255`, which `<` rejects) once label 2 has already written.
///
/// `pred_binary` is passed directly rather than through [`binarize_pred_mask`] because eroding
/// would not preserve this pattern; the loop's contract is over an arbitrary `u8` mask, which is
/// what `cv2.bitwise_xor(...).sum()` gives it upstream.
///
/// **Oracle**, two independent ones agreeing on **27** pixels / fingerprint **654**:
///   * upstream's `merge_mask_list` at the pinned commit, **imported and executed** with
///     `pred_thresh=0` -- which skips its internal erode-and-binarise, so this exact
///     `pred_binary` is what it scores against -- and `refine_mode=REFINEMASK_ANNOTATION`, on
///     2026-08-07 under `cv2` 5.0.0. (Its hole-fill pass adds nothing here, so its output is the
///     merge loop's.)
///   * the hand derivation above: 24 frame + 3 for label 2, label 3 dropped.
///
/// `cv2`'s own labelling of this candidate agrees with ours on all four `stats` rows and on both
/// non-background label numbers: label 1 `(0,0,7,7)` area 24, label 2 `(2,2,3,1)` area 3, label 3
/// `(2,4,3,1)` area 3, label 0 `(1,1,5,5)` area 19.
///
/// **What fails this test, measured rather than argued, on this fixture, 2026-08-07, by an
/// independent Python transcription of `textmask.py:116-132` with the visit order made
/// explicit.** Writing the whole bounding box (`merged[bbox] = 255`) instead of OR-ing the
/// indicator: **49** px for the three orders that visit label 1 before label 2, and **3** px for
/// the three that do not -- the orders disagree, and no order gives 27. Scoring the whole
/// bounding box while writing only the label's own pixels: **27** and **3** by the same split --
/// so that mutant is *right* for one order and wrong for the other, which is precisely the defect
/// a single-order test cannot see.
#[test]
fn merge_components_by_xor_is_order_independent_where_the_orders_could_diverge() {
    let candidate = grid(&[
        "XXXXXXX", "X.....X", "X.XXX.X", "X.....X", "X.XXX.X", "X.....X", "XXXXXXX",
    ]);
    let pred_binary = grid(&[
        "XXXXXXX", "XXXXXXX", "X.XXX.X", "XXXX..X", "X.....X", ".......", ".......",
    ]);
    let expected = vec![
        "XXXXXXX", "X.....X", "X.XXX.X", "X.....X", "X.....X", "X.....X", "XXXXXXX",
    ];

    assert_eq!(
        nonzero_count(&pred_binary),
        26,
        "fixture sanity: the hand-built pred mask has 26 pixels set"
    );

    let components = connected_components(&candidate);
    assert_eq!(components.num_labels(), 4);
    for (label, expected_stats) in [
        (
            1,
            ComponentStats {
                x: 0,
                y: 0,
                width: 7,
                height: 7,
                area: 24,
            },
        ),
        (
            2,
            ComponentStats {
                x: 2,
                y: 2,
                width: 3,
                height: 1,
                area: 3,
            },
        ),
        (
            3,
            ComponentStats {
                x: 2,
                y: 4,
                width: 3,
                height: 1,
                area: 3,
            },
        ),
    ] {
        assert_eq!(
            components.stats[label], expected_stats,
            "label {label} must match cv2's stats row"
        );
    }

    // All six permutations, not two: the divergence the wrong writes produce is between "label 1
    // before label 2" and "label 2 before label 1", which two hand-picked orders could miss.
    for order in [
        vec![1_u32, 2, 3],
        vec![1, 3, 2],
        vec![2, 1, 3],
        vec![2, 3, 1],
        vec![3, 1, 2],
        vec![3, 2, 1],
    ] {
        let mut merged = GrayImage::new(7, 7);
        merge_components_by_xor(&mut merged, &components, &pred_binary, &order)
            .expect("equal sizes");
        // Label 3's row stays empty, so this pins WHICH component was dropped -- a count of 27
        // would be satisfied by dropping label 2 and accepting label 3 instead.
        assert_eq!(render(&merged), expected, "visit order {order:?}");
        assert_eq!(nonzero_count(&merged), 27, "visit order {order:?}");
        assert_eq!(
            positional_fingerprint(&merged),
            654,
            "visit order {order:?}"
        );
    }
}

// ======================================================= the recorded page, end to end

/// spec §16.37 item 3: *"Correctness evidence therefore comes from A1-A3's per-function
/// hand-derived assertions"* -- A3's whole-path check on real committed data, with every
/// expected value produced by **upstream's own Python**, not by `pc-detect`.
///
/// **Inputs**, both already committed: `…_base.png` (the 1200x1660 page) and
/// `…_detector_mask.png` -- the **unrefined** U-Net mask, per §16.37 item 4. The four windows
/// are A1's corrected `expand_textwindow` outputs, the same four `a2_annotate_otsu.rs` uses.
///
/// **Oracle**: upstream's `merge_mask_list` **imported and executed** at the pinned commit
/// against those two files on 2026-08-07 under opencv 5.0.0 with IPP disabled, fed by upstream's
/// own `get_topk_masklist(...) + get_otsuthresh_masklist(..., per_channel=False)`. IPP status is
/// immaterial for this page -- §16.37 item 10(d) measured all 12 per-channel Otsu thresholds as
/// identical on both paths -- and the harness reproduced `a2_annotate_otsu.rs`'s frozen
/// candidate `xor_sum` lists before any number here was read off it.
///
/// **What each column pins.**
///   * `pred_binary` population and fingerprint -- the erode-plus-threshold step, on real
///     non-binary data (the detector mask has 255 distinct values). Block 3's collapse from a
///     populated raw mask to **12** surviving pixels is the sharpest single number here.
///   * `component_counts` -- `num_labels` per candidate **in merged (ascending xor_sum) order**,
///     so a reversed sort fails this column even where it would not change the final mask.
///   * `pre_hole_fill_nonzero` / `pre_hole_fill_fingerprint` -- the mask **after** the merge loop
///     and **before** the hole-fill pass. These are the numbers the oracle reports at that
///     point, and they differ from the final ones on every non-degenerate block, so the two
///     halves of `merge_mask_list` are pinned separately rather than only in combination.
///   * `inverted_areas` + `area_threshold` -- `np.sort(stats[:, -1])` and `sorted_area[-2]` over
///     the labelling of `255 - merged` at that same point.
///   * `nonzero` + `fingerprint` -- the merged mask's population and geometry after hole filling.
///
/// **The hole-fill XOR guard demonstrably fires on real data**, which is why the areas column is
/// here. Block 0's four sub-threshold holes have areas 1, 3, 4 and 21 -- 29 pixels -- and the
/// final mask gains only **28**, so exactly one of them (the single-pixel hole) was rejected for
/// raising the XOR. Blocks 1 and 2 gain 22 = 11 + 11 and 53 = 14 + 39, i.e. all of theirs were
/// accepted. An implementation that filled every sub-threshold hole unconditionally passes
/// blocks 1-3 and fails block 0.
///
/// **Anti-vacuity.** None of these numbers is computable from the tree by any route this test
/// takes. Blocks 0-2 yield 1092, 817 and 1033 pixels, so the test cannot pass by producing empty
/// masks; block 3 yields **0**, and that zero is upstream's own answer, recorded because §16.37
/// item 6 flags exactly this collapse and item 8 defers what to do about it to A4.
///
/// **What fails this test.** Sorting the candidates descending (`component_counts` reverses).
/// Scoring against the raw pred mask, skipping the erode, using the square kernel, dropping the
/// `w*h < 3` filter, using `sorted_area[-1]`, adding the `REFINEMASK_INPAINT` dilation (which
/// would give 1609/1801/2267/0, measured) -- each moves at least one column.
#[test]
fn annotation_merge_on_the_recorded_page_matches_the_upstream_oracle() {
    struct Row {
        rect: Rect,
        window: Rect,
        pred_binary_nonzero: usize,
        pred_binary_fingerprint: u64,
        component_counts: &'static [u32],
        pre_hole_fill_nonzero: usize,
        pre_hole_fill_fingerprint: u64,
        inverted_areas: &'static [u32],
        area_threshold: u32,
        nonzero: usize,
        fingerprint: u64,
    }

    let rows = [
        Row {
            rect: Rect::new(674, 1397, 740, 1438),
            window: Rect::new(671, 1394, 743, 1441),
            pred_binary_nonzero: 1364,
            pred_binary_fingerprint: 2_449_634,
            component_counts: &[8, 8, 5, 16],
            pre_hole_fill_nonzero: 1064,
            pre_hole_fill_fingerprint: 1_913_489,
            inverted_areas: &[1, 3, 4, 21, 1064, 2291],
            area_threshold: 1064,
            nonzero: 1092,
            fingerprint: 1_964_155,
        },
        Row {
            rect: Rect::new(567, 74, 663, 123),
            window: Rect::new(563, 70, 667, 127),
            pred_binary_nonzero: 1532,
            pred_binary_fingerprint: 5_521_257,
            component_counts: &[17, 37, 23, 26],
            pre_hole_fill_nonzero: 795,
            pre_hole_fill_fingerprint: 2_818_713,
            inverted_areas: &[11, 11, 795, 5111],
            area_threshold: 795,
            nonzero: 817,
            fingerprint: 2_907_642,
        },
        Row {
            rect: Rect::new(607, 631, 724, 703),
            window: Rect::new(602, 626, 729, 708),
            pred_binary_nonzero: 2004,
            pred_binary_fingerprint: 10_931_831,
            component_counts: &[22, 22, 32, 93],
            pre_hole_fill_nonzero: 980,
            pre_hole_fill_fingerprint: 5_315_650,
            inverted_areas: &[14, 39, 980, 9381],
            area_threshold: 980,
            nonzero: 1033,
            fingerprint: 5_586_013,
        },
        Row {
            rect: Rect::new(438, 1407, 498, 1446),
            window: Rect::new(435, 1404, 501, 1449),
            pred_binary_nonzero: 12,
            pred_binary_fingerprint: 17_014,
            component_counts: &[8, 7],
            pre_hole_fill_nonzero: 0,
            pre_hole_fill_fingerprint: 0,
            inverted_areas: &[0, 2970],
            area_threshold: 0,
            nonzero: 0,
            fingerprint: 0,
        },
    ];

    for row in &rows {
        let (crop, mask_crop) = recorded_crop(row.rect, row.window);
        let rect = row.rect;

        let pred_binary = binarize_pred_mask(&mask_crop);
        assert_eq!(
            nonzero_count(&pred_binary),
            row.pred_binary_nonzero,
            "block {rect:?} pred_binary population"
        );
        assert_eq!(
            positional_fingerprint(&pred_binary),
            row.pred_binary_fingerprint,
            "block {rect:?} pred_binary geometry"
        );

        let mut candidates = candidate_mask_list(&crop, &mask_crop).expect("equal sizes");
        candidates.sort_by_key(|candidate| candidate.xor_sum);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| connected_components(&candidate.mask).num_labels())
                .collect::<Vec<_>>(),
            row.component_counts,
            "block {rect:?} per-candidate num_labels, in ascending xor_sum order"
        );

        // The merge loop alone, so the state the hole-fill pass starts from is pinned in its own
        // right rather than only through the final mask.
        let mut pre_hole_fill = GrayImage::new(mask_crop.width(), mask_crop.height());
        for candidate in &candidates {
            let components = connected_components(&candidate.mask);
            let labels: Vec<u32> = (1..components.num_labels())
                .filter(|&label| components.stats[label as usize].bounding_box_area() >= 3)
                .collect();
            merge_components_by_xor(&mut pre_hole_fill, &components, &pred_binary, &labels)
                .expect("equal sizes");
        }
        assert_eq!(
            nonzero_count(&pre_hole_fill),
            row.pre_hole_fill_nonzero,
            "block {rect:?} population after the merge loop, before hole filling"
        );
        assert_eq!(
            positional_fingerprint(&pre_hole_fill),
            row.pre_hole_fill_fingerprint,
            "block {rect:?} geometry after the merge loop, before hole filling"
        );

        let inverted = GrayImage::from_fn(pre_hole_fill.width(), pre_hole_fill.height(), |x, y| {
            Luma([255 - pre_hole_fill.get_pixel(x, y).0[0]])
        });
        let holes = connected_components(&inverted);
        assert_eq!(
            sorted_areas(&holes),
            row.inverted_areas,
            "block {rect:?} np.sort(stats[:, -1]) over 255 - merged"
        );
        assert_eq!(
            hole_fill_area_threshold(&holes),
            row.area_threshold,
            "block {rect:?} hole-fill area threshold"
        );

        let merged = merge_mask_list(
            candidate_mask_list(&crop, &mask_crop).expect("equal sizes"),
            &mask_crop,
        )
        .expect("equal sizes");
        assert_eq!(
            nonzero_count(&merged),
            row.nonzero,
            "block {rect:?} merged population"
        );
        assert_eq!(
            positional_fingerprint(&merged),
            row.fingerprint,
            "block {rect:?} merged geometry"
        );
    }
}

/// spec §16.37 item 8's recorded observation about label order, re-measured on the committed
/// page rather than inherited: *"reversing the entire non-background label numbering changes
/// upstream's output by 0 px on E01P01, E01P02 and E01P03"*.
///
/// **Oracle**: upstream's own `merge_mask_list` against the validated reimplementation with the
/// non-background label order reversed in **both** loops, on all four E01P01 blocks, 2026-08-07:
/// byte-identical on each. E01P02/E01P03 are not committed (§16.37 item 9 keeps this work to one
/// page), so only E01P01 is asserted here and the other two pages' figures are cited, not
/// re-derived.
///
/// **Why this is not vacuous.** The three non-degenerate blocks reach the reversed loop with 8,
/// 17, 22 and more labels apiece -- asserted above as `component_counts` -- and their merged
/// masks are non-empty, asserted here as a lower bound so the test cannot pass by comparing two
/// empty masks.
///
/// **What fails this test.** Any implementation in which a component's write escapes its own
/// pixels; see the module header's proof and its stated precondition.
#[test]
fn recorded_page_merge_is_unchanged_when_the_label_order_is_reversed() {
    const WINDOWS: &[(Rect, Rect)] = &[
        (
            Rect {
                x1: 674,
                y1: 1397,
                x2: 740,
                y2: 1438,
            },
            Rect {
                x1: 671,
                y1: 1394,
                x2: 743,
                y2: 1441,
            },
        ),
        (
            Rect {
                x1: 567,
                y1: 74,
                x2: 663,
                y2: 123,
            },
            Rect {
                x1: 563,
                y1: 70,
                x2: 667,
                y2: 127,
            },
        ),
        (
            Rect {
                x1: 607,
                y1: 631,
                x2: 724,
                y2: 703,
            },
            Rect {
                x1: 602,
                y1: 626,
                x2: 729,
                y2: 708,
            },
        ),
        (
            Rect {
                x1: 438,
                y1: 1407,
                x2: 498,
                y2: 1446,
            },
            Rect {
                x1: 435,
                y1: 1404,
                x2: 501,
                y2: 1449,
            },
        ),
    ];
    // Upstream's own answers, from `annotation_merge_on_the_recorded_page_matches_the_upstream_oracle`.
    const EXPECTED_NONZERO: [usize; 4] = [1092, 817, 1033, 0];

    for (index, &(rect, window)) in WINDOWS.iter().enumerate() {
        let (crop, mask_crop) = recorded_crop(rect, window);
        let mut candidates = candidate_mask_list(&crop, &mask_crop).expect("equal sizes");
        candidates.sort_by_key(|candidate| candidate.xor_sum);
        let pred_binary = binarize_pred_mask(&mask_crop);

        let mut merged = GrayImage::new(mask_crop.width(), mask_crop.height());
        for candidate in &candidates {
            let components = connected_components(&candidate.mask);
            let mut labels: Vec<u32> = (1..components.num_labels())
                .filter(|&label| components.stats[label as usize].bounding_box_area() >= 3)
                .collect();
            labels.reverse();
            merge_components_by_xor(&mut merged, &components, &pred_binary, &labels)
                .expect("equal sizes");
        }
        let inverted = GrayImage::from_fn(merged.width(), merged.height(), |x, y| {
            Luma([255 - merged.get_pixel(x, y).0[0]])
        });
        let holes = connected_components(&inverted);
        let threshold = hole_fill_area_threshold(&holes);
        let mut labels: Vec<u32> = (0..holes.num_labels())
            .filter(|&label| holes.stats[label as usize].area < threshold)
            .collect();
        labels.reverse();
        merge_components_by_xor(&mut merged, &holes, &pred_binary, &labels).expect("equal sizes");

        let forward = merge_mask_list(
            candidate_mask_list(&crop, &mask_crop).expect("equal sizes"),
            &mask_crop,
        )
        .expect("equal sizes");
        assert_eq!(
            nonzero_count(&forward),
            EXPECTED_NONZERO[index],
            "block {rect:?}: the forward mask must be upstream's, or the comparison is vacuous"
        );
        assert_eq!(
            merged, forward,
            "block {rect:?}: reversing the label order must not change the mask"
        );
    }
}

// ================================================================================ helpers

/// `'X'` -> 255, anything else -> 0. Every row must have the same length.
fn grid(rows: &[&str]) -> GrayImage {
    let height = rows.len() as u32;
    let width = rows[0].len() as u32;
    assert!(
        rows.iter().all(|row| row.len() as u32 == width),
        "ragged fixture grid"
    );
    GrayImage::from_fn(width, height, |x, y| {
        let byte = rows[y as usize].as_bytes()[x as usize];
        Luma([if byte == b'X' { 255 } else { 0 }])
    })
}

/// The inverse of [`grid`], so a failed assertion prints two readable pictures instead of two
/// pixel counts.
fn render(mask: &GrayImage) -> Vec<String> {
    (0..mask.height())
        .map(|y| {
            (0..mask.width())
                .map(|x| {
                    if mask.get_pixel(x, y).0[0] != 0 {
                        'X'
                    } else {
                        '.'
                    }
                })
                .collect()
        })
        .collect()
}

/// The label array as digits, for labelings with fewer than ten labels.
fn render_labels(components: &ConnectedComponents) -> Vec<String> {
    assert!(
        components.num_labels() < 10,
        "renderer is single-digit only"
    );
    (0..components.height)
        .map(|y| {
            (0..components.width)
                .map(|x| char::from(b'0' + components.label_at(x, y) as u8))
                .collect()
        })
        .collect()
}

fn candidate_of(mask: GrayImage, xor_sum: u64) -> MaskCandidate {
    MaskCandidate { mask, xor_sum }
}

fn nonzero_count(mask: &GrayImage) -> usize {
    mask.pixels().filter(|pixel| pixel.0[0] != 0).count()
}

/// `sum of (row-major index + 1) over non-zero pixels` -- the same positional fingerprint
/// `a2_annotate_otsu.rs` uses, so two masks with equal populations but different geometry are
/// distinguished.
fn positional_fingerprint(mask: &GrayImage) -> u64 {
    mask.pixels()
        .enumerate()
        .filter(|(_, pixel)| pixel.0[0] != 0)
        .map(|(index, _)| index as u64 + 1)
        .sum()
}

fn sorted_areas(components: &ConnectedComponents) -> Vec<u32> {
    let mut areas: Vec<u32> = components.stats.iter().map(|stat| stat.area).collect();
    areas.sort_unstable();
    areas
}

/// The committed page and unrefined detector mask, cropped to `window` -- upstream's
/// `im = img[by1:by2, bx1:bx2]` / `msk = pred_mask[by1:by2, bx1:bx2]`. `rect` is passed only so
/// the assertion messages name the block.
fn recorded_crop(rect: Rect, window: Rect) -> (image::RgbImage, GrayImage) {
    let page = image::open(paths::recorded(format!("{PAGE_STEM}_base.png")))
        .expect("committed base page must decode")
        .to_rgb8();
    let mask = image::open(paths::recorded(format!("{PAGE_STEM}_detector_mask.png")))
        .expect("committed detector mask must decode")
        .to_luma8();
    assert_eq!(page.dimensions(), (1200, 1660), "block {rect:?}");
    assert_eq!(mask.dimensions(), (1200, 1660));

    let crop = image::imageops::crop_imm(
        &page,
        window.x1 as u32,
        window.y1 as u32,
        window.width() as u32,
        window.height() as u32,
    )
    .to_image();
    let mask_crop = image::imageops::crop_imm(
        &mask,
        window.x1 as u32,
        window.y1 as u32,
        window.width() as u32,
        window.height() as u32,
    )
    .to_image();
    (crop, mask_crop)
}
