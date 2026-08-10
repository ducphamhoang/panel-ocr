//! Task A1 -- spec §16.37 item 8: *"**A1** (**heavy**): the greyscale/histogram/top-k-colour
//! path including item 2's tie rule -- pure functions over committed arrays, hand-derivable,
//! no model."* FROZEN with the implementation.
//!
//! **What every expected value here is derived from, and why none of it is circular.**
//! §16.37 item 3 is explicit that a self-comparison gate proves determinism and not
//! correctness -- *"A port that is wrong in the same way on every run passes it forever.
//! Correctness evidence therefore comes from A1-A3's per-function hand-derived
//! assertions"*. So every literal below comes from one of exactly two places, never from
//! `pc_detect::annotate`:
//!   * hand derivation, stated in the test's own comment (the `min = 0, max = 255` histogram,
//!     the four `expand_text_window` paddings, every `top_k_colors` trace); or
//!   * an independent oracle -- upstream PanelCleaner's own Python at pinned commit
//!     `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3` plus `numpy`/`cv2`, run on 2026-08-06
//!     under the versions `tests/fixtures/recorded/detector/PROVENANCE.json` pins
//!     (`"opencv": "5.0.0"`; `numpy` there is `2.5.1`, the run used 2.2.5 -- see the note
//!     on [`histogram_255_matches_numpy_on_the_documented_corner_cases`]).
//!
//! **Scope.** A1 covers Annotation primitives; run wiring is A4. The A4-a integration gate
//! passes; Otsu and the XOR-minimising selection are A2, connected components / merge /
//! hole filling are A3, wiring is A4. Nothing in this file asserts anything about
//! `MaskRefineMode::Simple`, whose behaviour §16.37 leaves untouched.

use image::{GrayImage, Luma, Rgb, RgbImage};
use pc_core::Rect;
use pc_detect::annotate::{
    candidate_grey_values, erode_rect3x3, expand_text_window, histogram_255, rgb_to_gray,
    top_k_colors, top_k_colors_default, Histogram255, ANNOTATION_EXPAND_R, HISTOGRAM_BINS,
    TOPK_BIN_TOL, TOPK_COLOR_VAR, TOPK_K,
};
use pc_testkit::paths;

const PAGE_STEM: &str = "detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

// ============================================================ greyscale (step 2)

/// spec §16.37 item 8, A1's "greyscale" -- upstream `get_topk_masklist`'s
/// `cv2.cvtColor(im_grey, cv2.COLOR_BGR2GRAY)`.
///
/// **Oracle**: `cv2.cvtColor` at opencv 5.0.0 (the version pinned in `PROVENANCE.json`),
/// queried on 2026-08-06 for exactly these 14 (r, g, b) triples. Not computed from any
/// formula in `pc-detect`.
///
/// **Why these triples and not fourteen random ones.** Three plausible-but-wrong
/// implementations have to be excluded, and most pixels exclude none of them:
///   * the classic 14-bit fixed point `(r*4899 + g*9617 + b*1868 + 8192) >> 14`, which
///     disagrees with opencv 5.0.0 on 43 864 of the 16 777 216 RGB values;
///   * `round(0.299 r + 0.587 g + 0.114 b)`;
///   * `trunc(0.299 r + 0.587 g + 0.114 b)`.
///
/// The four rows marked `all-three` below are values where cv2 disagrees with **all
/// three** at once, so this test goes red if the implementation is swapped for any of
/// them. `(92, 124, 186)` additionally separates cv2 from the 14-bit triple and from
/// `round` while agreeing with `trunc`, and `(215, 97, 134)` separates it from `round`
/// alone -- both included so a partially-wrong implementation cannot slip through on the
/// `all-three` rows only.
#[test]
fn rgb_to_gray_matches_opencv_bgr2gray_on_oracle_pixels() {
    // (r, g, b) => cv2.cvtColor(..., COLOR_BGR2GRAY)
    const ORACLE: &[((u8, u8, u8), u8)] = &[
        ((0, 0, 0), 0),
        ((255, 255, 255), 255),
        ((255, 0, 0), 76),  // pure red   -> 0.299 weight
        ((0, 255, 0), 150), // pure green -> 0.587 weight
        ((0, 0, 255), 29),  // pure blue  -> 0.114 weight
        ((128, 128, 128), 128),
        ((1, 2, 3), 2),
        ((106, 15, 0), 41),    // 14-bit fixed point says 40
        ((242, 168, 66), 179), // all-three: 14-bit / round / trunc all say 178
        ((169, 126, 79), 134), // all-three: all three say 133
        ((216, 204, 19), 187), // all-three: all three say 186
        ((74, 234, 44), 165),  // all-three: all three say 164
        ((215, 97, 134), 137), // round says 136
        ((92, 124, 186), 121), // 14-bit and round both say 122
    ];

    let mut image = RgbImage::new(ORACLE.len() as u32, 1);
    for (index, ((r, g, b), _)) in ORACLE.iter().enumerate() {
        image.put_pixel(index as u32, 0, Rgb([*r, *g, *b]));
    }
    let grey = rgb_to_gray(&image);

    assert_eq!(grey.dimensions(), (ORACLE.len() as u32, 1));
    let got = (0..ORACLE.len())
        .map(|index| grey.get_pixel(index as u32, 0).0[0])
        .collect::<Vec<_>>();
    let want = ORACLE.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    assert_eq!(got, want, "cv2.COLOR_BGR2GRAY oracle mismatch");
}

// ================================================================ erode (step 3)

/// spec §16.37 item 8, A1's step 3 -- upstream's
/// `cv2.erode(msk, np.ones((3, 3), np.uint8), iterations=1)`.
///
/// **Oracle**: `cv2.erode` run on this exact 5x5 array on 2026-08-06; the expected block
/// below is its printed output, transcribed.
///
/// **What this excludes.** `cv2.erode`'s default `borderValue` is
/// `morphologyDefaultBorderValue()` = `+DBL_MAX`, so out-of-image neighbours never lower
/// the minimum. A zero-padding erosion -- the obvious implementation -- clears the whole
/// first row and column, so `(0, 0)` would be 0 instead of 255 and this test would fail
/// on four pixels.
#[test]
fn erode_rect3x3_treats_out_of_image_neighbours_as_maximum() {
    let mut mask = GrayImage::new(5, 5);
    for y in 0..3 {
        for x in 0..3 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }

    // cv2.erode output, transcribed:
    //   255 255   0   0   0
    //   255 255   0   0   0
    //     0   0   0   0   0
    //     0   0   0   0   0
    //     0   0   0   0   0
    #[rustfmt::skip]
    const WANT: [[u8; 5]; 5] = [
        [255, 255, 0, 0, 0],
        [255, 255, 0, 0, 0],
        [  0,   0, 0, 0, 0],
        [  0,   0, 0, 0, 0],
        [  0,   0, 0, 0, 0],
    ];

    let eroded = erode_rect3x3(&mask);
    for (y, row) in WANT.iter().enumerate() {
        for (x, want) in row.iter().enumerate() {
            assert_eq!(
                eroded.get_pixel(x as u32, y as u32).0[0],
                *want,
                "eroded pixel ({x}, {y})"
            );
        }
    }
}

/// spec §16.37 item 8, A1's step 3: the structuring element is 3x3 and the operation is
/// the **minimum**.
///
/// Hand-derived, not oracle-derived. The image is `value = 10 * x + y` on a 5x5 grid with
/// one 200 spike at (2, 2) and one 0 pit at (4, 4). For the centre pixel (2, 2) the 3x3
/// neighbourhood is x in 1..=3, y in 1..=3, whose minimum is `10*1 + 1 = 11`; a 5x5
/// element would instead see x in 0..=4 and give `10*0 + 0 = 0`, and a **dilation** would
/// give 200. Pixel (3, 3)'s neighbourhood is x,y in 2..=4 and contains the pit, so it is
/// 0; pixel (1, 1)'s is x,y in 0..=2 and is `0`.
#[test]
fn erode_rect3x3_is_the_3x3_neighbourhood_minimum() {
    let mut mask = GrayImage::from_fn(5, 5, |x, y| Luma([(10 * x + y) as u8]));
    mask.put_pixel(2, 2, Luma([200]));
    mask.put_pixel(4, 4, Luma([0]));

    let eroded = erode_rect3x3(&mask);
    assert_eq!(eroded.dimensions(), (5, 5));
    assert_eq!(
        eroded.get_pixel(2, 2).0[0],
        11,
        "centre: min over x,y in 1..=3"
    );
    assert_eq!(eroded.get_pixel(3, 3).0[0], 0, "sees the pit at (4, 4)");
    assert_eq!(eroded.get_pixel(1, 1).0[0], 0, "sees (0, 0) = 0");
    assert_eq!(
        eroded.get_pixel(4, 0).0[0],
        30,
        "min over x in 3..=4, y in 0..=1"
    );
}

// =================================================== candidate values (step 4)

/// spec §16.37 item 8, A1's step 4 -- upstream's
/// `im_grey[np.where(cv2.erode(msk, ...) > 127)]`, i.e. the gate is **strict** `> 127`
/// on the **eroded** mask, not on the mask itself.
///
/// Hand-derived. The mask is a 5x5 block of 255 at x,y in 1..=5 inside an 8x8 zero field,
/// plus one isolated 128 at (7, 7). A pixel survives the erosion only if its whole 3x3
/// neighbourhood is inside the block, i.e. x,y in 2..=4 -- 9 pixels. Grey is `x + 16 * y`,
/// so the survivors are exactly {34, 35, 36, 50, 51, 52, 66, 67, 68}.
///
/// **What turns this red.** Applying the `> 127` gate to the *un-eroded* mask admits all
/// 25 block pixels **and** the isolated 128 (128 > 127), giving 26 values; erosion is what
/// removes the 128, not the threshold. Eroding with the wrong element also changes the
/// set. Using `>=` instead of `>` would **not** change this case -- the eroded mask's only
/// values are 0 and 255, so no pixel is ever exactly 127 -- which is why the boundary
/// operator is left to upstream's own (`textmask.py:61`) rather than claimed here. The
/// assertion is on the sorted set, not the count, so a same-size wrong set fails too.
#[test]
fn candidate_grey_values_gate_is_strictly_above_127_on_the_eroded_mask() {
    let mut mask = GrayImage::new(8, 8);
    for y in 1..=5 {
        for x in 1..=5 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    mask.put_pixel(7, 7, Luma([128]));
    let grey = GrayImage::from_fn(8, 8, |x, y| Luma([(x + 16 * y) as u8]));

    let mut values = candidate_grey_values(&grey, &mask).expect("equal sizes");
    values.sort_unstable();
    assert_eq!(
        values,
        vec![34, 35, 36, 50, 51, 52, 66, 67, 68],
        "the eroded 3x3 core of the block, and nothing else"
    );
}

/// The size mismatch is a per-image `StageError::InvalidInput` (cookbook rule 4: fatality
/// is declared, not inferred) -- upstream indexes one array with the other's boolean mask,
/// which is only defined for equal shapes.
#[test]
fn candidate_grey_values_rejects_mismatched_sizes_as_invalid_input() {
    let grey = GrayImage::new(4, 4);
    let mask = GrayImage::new(4, 5);
    let error = candidate_grey_values(&grey, &mask).expect_err("sizes differ");
    assert!(
        matches!(error, pc_core::StageError::InvalidInput(_)),
        "expected InvalidInput, got {error:?}"
    );
}

// ================================================================ histogram (step 5)

/// spec §16.37 item 8, A1's step 5 -- upstream's `np.histogram(candidate_grey_px, bins=255)`.
///
/// **Hand-derived, no oracle needed.** With `min = 0` and `max = 255` the width is
/// `255 / 255 = 1` exactly, so `edges[i] == i` for every `i` and bin `i` is `[i, i+1)`
/// -- except the last, which numpy closes on the right. So 254 **and** 255 both land in
/// bin 254, and that is the whole point of the sample set: an implementation that leaves
/// the last bin half-open puts 255 in bin 255, which does not exist, and one that clamps
/// without the `index == 255` decrement loses it entirely.
#[test]
fn histogram_255_closes_the_last_bin_on_the_right() {
    let histogram = histogram_255(&[0, 1, 2, 127, 128, 253, 254, 255]);

    for index in 0..=HISTOGRAM_BINS {
        assert_eq!(
            histogram.edges[index], index as f64,
            "edges[{index}] with a width-1 bin"
        );
    }
    let nonzero = nonzero_bins(&histogram);
    assert_eq!(
        nonzero,
        vec![
            (0, 1),
            (1, 1),
            (2, 1),
            (127, 1),
            (128, 1),
            (253, 1),
            (254, 2)
        ],
        "254 and 255 share the closed last bin"
    );
    assert_eq!(histogram.counts.iter().sum::<u32>(), 8, "no sample dropped");
}

/// spec §16.37 item 8, A1's step 5: the bin index numpy's uniform-bin fast path actually
/// produces, on inputs where a bare `floor((v - min) / step)` produces a different one.
///
/// **Oracle**: `np.histogram(..., bins=255)` run on 2026-08-06 on exactly these two arrays.
///
/// **Why these two arrays.** They are the smallest cases found (by enumerating all 32 640
/// `(min, max)` pairs) where a bare truncating `floor((v - min) / step)` disagrees with
/// numpy's result, and they disagree in opposite directions:
///   * `[0, 21, 35]`: numpy puts 21 in bin **152**, the bare floor puts it in 153;
///   * `[0, 21, 45]`: numpy puts 21 in bin **119**, the bare floor puts it in 118.
///
/// A single case could be passed by an off-by-one; two opposite-direction cases cannot.
///
/// **What this test does and does not distinguish, measured rather than assumed.** numpy
/// computes `((v - first) / (last - first)) * 255`, truncates, decrements an index of 255,
/// and then applies two ULP corrections (`decrement` / `increment`). Mutation probes on
/// 2026-08-06 established: replacing the whole block with a bare clamped floor turns this
/// test red, and keeping numpy's formula while deleting the two corrections also turns it
/// red -- but swapping *only* the initial formula for `(v - first) / step` while keeping
/// the corrections leaves it **green**, because the corrections normalise the difference
/// away. So what is pinned here is the composite index assignment, not the choice of the
/// initial division; the test name says "assigns numpy's bin indices" and not "uses
/// numpy's formula" for exactly that reason.
#[test]
fn histogram_255_assigns_numpys_bin_indices_where_a_bare_floor_disagrees() {
    let histogram = histogram_255(&[0, 21, 35]);
    assert_eq!(histogram.edges[0], 0.0);
    assert_eq!(histogram.edges[HISTOGRAM_BINS], 35.0);
    assert_eq!(
        nonzero_bins(&histogram),
        vec![(0, 1), (152, 1), (254, 1)],
        "np.histogram([0,21,35], bins=255)"
    );

    let histogram = histogram_255(&[0, 21, 45]);
    assert_eq!(histogram.edges[HISTOGRAM_BINS], 45.0);
    assert_eq!(
        nonzero_bins(&histogram),
        vec![(0, 1), (119, 1), (254, 1)],
        "np.histogram([0,21,45], bins=255)"
    );
}

/// spec §16.37 item 8, A1's step 5 -- numpy's `_get_outer_edges` corner cases, which the
/// real page reaches: box `[438, 1407, 498, 1446]` on the committed E01P01 page yields an
/// **empty** candidate set (§16.37 item 4 is the same block, dropped by the coverage
/// filter), so the empty branch is live code, not defensive padding.
///
/// **Oracle**: `np.histogram(..., bins=255)` run on 2026-08-06 on each array below.
///
/// **numpy version note.** `PROVENANCE.json` pins `numpy 2.5.1`; this oracle ran under
/// 2.2.5. The values asserted are stated by `_get_outer_edges`'s own documented
/// behaviour (empty -> `[0, 1]`; `first == last` -> `[min - 0.5, max + 0.5]`), which is
/// unchanged between those versions, but the version gap is recorded rather than hidden.
#[test]
fn histogram_255_matches_numpy_on_the_documented_corner_cases() {
    // min == max: numpy widens the range by half a unit either side, so the single
    // distinct value sits at the centre -- bin 127.
    let histogram = histogram_255(&[5, 5, 5]);
    assert_eq!(histogram.edges[0], 4.5);
    assert_eq!(histogram.edges[HISTOGRAM_BINS], 5.5);
    assert_eq!(histogram.edges[127], 4.998_039_215_686_274_5);
    assert_eq!(histogram.edges[128], 5.001_960_784_313_725_5);
    assert_eq!(nonzero_bins(&histogram), vec![(127, 3)]);

    // empty: "Can't determine range, so use 0-1."
    let histogram = histogram_255(&[]);
    assert_eq!(histogram.edges[0], 0.0);
    assert_eq!(histogram.edges[1], 0.003_921_568_627_450_98);
    assert_eq!(histogram.edges[254], 0.996_078_431_372_549);
    assert_eq!(histogram.edges[HISTOGRAM_BINS], 1.0);
    assert_eq!(histogram.counts.iter().sum::<u32>(), 0);

    // non-integer step: 11 samples spanning [0, 10] land in 11 distinct bins, the last
    // by way of the `index == 255` decrement (10 maps to exactly 255.0 before it).
    let histogram = histogram_255(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert_eq!(histogram.edges[1], 0.039_215_686_274_509_8);
    assert_eq!(histogram.edges[254], 9.960_784_313_725_49);
    assert_eq!(
        nonzero_bins(&histogram),
        vec![
            (0, 1),
            (25, 1),
            (51, 1),
            (76, 1),
            (102, 1),
            (127, 1),
            (153, 1),
            (178, 1),
            (204, 1),
            (229, 1),
            (254, 1),
        ]
    );
}

// ============================================================== top-k colour (step 6)

/// spec §14 item 23 / §16.37 item 2 -- **DEVIATION(23)**, the declared tie rule:
/// *"v1.5 uses a **stable** sort on `(Reverse(count), ascending bin index)`."*
///
/// Hand-derived. Counts are 10 at bins 0, 100 and 200 and nothing else in contention, so
/// all three candidates tie on count and only the tie rule decides their order. Under the
/// declared rule the traversal is bin 0, then 100, then 200; each is more than
/// `color_var = 10` from the ones already kept, so all three are appended and the result
/// is `[0.0, 100.0, 200.0]` **in that order**.
///
/// **What turns this red**: any tie order other than ascending bin index. Reversing the
/// tied group yields `[200.0, 100.0, 0.0]` -- the same *set*, a different *sequence*, and
/// the sequence is what feeds A2's `cv2.inRange` bands and A3's merge order. Asserting
/// the set, or the length, would not catch it (cookbook: cardinality is not identity).
#[test]
fn top_k_colors_breaks_count_ties_toward_the_lower_bin_index() {
    let histogram = synthetic_histogram(&[(0, 10), (100, 10), (200, 10), (5, 7), (50, 3)]);
    assert_eq!(
        top_k_colors_default(&histogram),
        vec![0.0, 100.0, 200.0],
        "ties must resolve to ascending bin index, and order is part of the result"
    );
}

/// spec §16.37 item 8, A1's step 6 -- upstream's
/// `if np.abs(np.array(top_colors) - color).min() > color_var: top_colors.append(color)`
/// with `color_var = 10`.
///
/// Hand-derived. Bin 5 is the second most populous (9) but sits 5 away from the already
/// kept colour 0.0, and `5 > 10` is false, so it is skipped and the next distinct
/// candidate is taken instead. Dropping the `color_var` test entirely gives
/// `[0.0, 5.0, 100.0]`; using `>=` instead of `>` would not change this case, which is
/// why the boundary is left to upstream's own operator rather than claimed here.
#[test]
fn top_k_colors_skips_candidates_within_color_var_of_a_kept_colour() {
    let histogram = synthetic_histogram(&[(0, 10), (5, 9), (100, 8), (200, 7)]);
    assert_eq!(top_k_colors_default(&histogram), vec![0.0, 100.0, 200.0]);
}

/// spec §16.37 item 8, A1's step 6 -- the ordering quirk in upstream's loop body:
/// `bin < bin_tol` is checked **after** the append, so the first candidate to fall below
/// the tolerance is still added before the loop breaks.
///
/// Hand-derived. Counts are 100 000 at bin 0 and 1 at bin 100, so
/// `bin_tol = 0.001 * 100 001 = 100.001`. Bin 100 is 100 away from the kept 0.0, so it is
/// appended; only then does `1 < 100.001` stop the loop. Expected `[0.0, 100.0]`.
///
/// **What turns this red**: hoisting the tolerance check above the append, which is the
/// natural way to write this and yields `[0.0]`.
#[test]
fn top_k_colors_appends_the_candidate_that_trips_bin_tol_before_breaking() {
    let histogram = synthetic_histogram(&[(0, 100_000), (100, 1)]);
    assert_eq!(top_k_colors_default(&histogram), vec![0.0, 100.0]);
}

/// spec §16.37 item 8, A1's step 6 -- `k = 3` caps the result, and the cap is checked
/// after each append.
///
/// Hand-derived. Five well-separated candidates with strictly decreasing counts; the
/// traversal keeps bins 0, 50, 100 and stops, so bins 150 and 200 are never reached even
/// though they are separated and above the tolerance.
#[test]
fn top_k_colors_stops_at_k_and_returns_the_three_most_populous_separated_bins() {
    let histogram = synthetic_histogram(&[(0, 5), (50, 4), (100, 3), (150, 2), (200, 1)]);
    let colors = top_k_colors_default(&histogram);
    assert_eq!(colors, vec![0.0, 50.0, 100.0]);
    assert_eq!(TOPK_K, 3);
    assert!(colors.len() <= TOPK_K);
}

/// spec §16.37 item 8, A1's step 6, on the degenerate input the real page produces: an
/// all-zero histogram, i.e. no candidate grey pixels at all.
///
/// Hand-derived. Every count is 0, so every candidate ties and the declared tie rule
/// (DEVIATION(23)) picks bin 0. `bin_tol = 0.001 * 0 = 0.0` and `0 < 0.0` is false, so
/// the loop never breaks early; every remaining edge is within `color_var = 10` of 0.0
/// (the largest edge is `254/255 < 1`), so nothing is appended. Result: exactly `[0.0]`.
///
/// This is the branch where upstream is at its least defined -- `np.argsort` over 255
/// equal keys -- and it is reached by box `[438, 1407, 498, 1446]` of the committed page
/// (see [`annotation_topk_path_on_the_recorded_page_matches_the_upstream_oracle`]).
#[test]
fn top_k_colors_on_an_all_zero_histogram_returns_the_single_lowest_edge() {
    let histogram = histogram_255(&[]);
    assert_eq!(top_k_colors_default(&histogram), vec![0.0]);
}

/// The three defaults this crate pins are upstream's literal arguments in
/// `get_topk_masklist`: `get_topk_color(his, bin, color_var=10, k=3)` with `get_topk_color`'s
/// own `bin_tol=0.001` default.
#[test]
fn top_k_colors_default_uses_upstreams_literal_arguments() {
    assert_eq!(TOPK_K, 3);
    assert_eq!(TOPK_COLOR_VAR, 10.0);
    assert_eq!(TOPK_BIN_TOL, 0.001);

    let histogram = synthetic_histogram(&[(0, 10), (5, 9), (100, 8), (200, 7)]);
    assert_eq!(
        top_k_colors_default(&histogram),
        top_k_colors(&histogram, 3, 10.0, 0.001)
    );
}

// ============================================================ crop window (step 1)

/// spec §16.37 item 5 (`CORRECTION`, `SUPERSEDES: §8.3 step 5`): *"Upstream's `expand_r`
/// is a **divisor**, not a pad: `paddings = int(round((max(h, w) * 0.25 + min(h, w) *
/// 0.75) / expand_r))` … Run on the four committed E01P01 boxes it yields **3, 4, 5 and
/// 3 px**, against our flat 16."*
///
/// **Hand-derived from the four boxes in the committed
/// `…_detector_blocks.json`**, and the four paddings agree with the spec sentence above:
///   * `[674, 1397, 740, 1438]`: w=66 h=41 -> (66*0.25 + 41*0.75)/16 = 47.25/16 = 2.953 -> 3
///   * `[567, 74, 663, 123]`:    w=96 h=49 -> (96*0.25 + 49*0.75)/16 = 60.75/16 = 3.797 -> 4
///   * `[607, 631, 724, 703]`:   w=117 h=72 -> (117*0.25 + 72*0.75)/16 = 83.25/16 = 5.203 -> 5
///   * `[438, 1407, 498, 1446]`: w=60 h=39 -> (60*0.25 + 39*0.75)/16 = 44.25/16 = 2.766 -> 3
///
/// **What turns this red**: `Rect::pad(16, …)`, i.e. the flat pad §8.3 step 5 currently
/// claims is upstream parity. On box 0 that gives `[658, 1381, 756, 1454]` rather than
/// `[671, 1394, 743, 1441]`.
#[test]
fn expand_text_window_pads_by_upstreams_divisor_formula() {
    const IMAGE_SIZE: (u32, u32) = (1200, 1660);
    const CASES: &[(Rect, Rect)] = &[
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

    assert_eq!(ANNOTATION_EXPAND_R, 16);
    for (input, want) in CASES {
        assert_eq!(
            expand_text_window(*input, IMAGE_SIZE, ANNOTATION_EXPAND_R),
            *want,
            "expand_textwindow({input:?})"
        );
    }
}

/// spec §16.37 item 5's *second* divergence, "not previously recorded anywhere":
/// *"upstream clamps the far edges to **`im_w - 1` / `im_h - 1`**, while `pc_core::Rect::pad`
/// … clamps to `canvas.0` / `canvas.1` -- off by one at every image edge."*
///
/// Hand-derived. The box is `[40, 30, 59, 49]` on a 60x50 image: w = h = 19, so
/// `paddings = round(19/16) = round(1.1875) = 1`, and the unclamped far corner is
/// `(60, 50)`. Upstream clamps it to `(59, 49)`; `Rect::pad`'s convention would leave it
/// at `(60, 50)`. The near corner is clamped at 0 by both, exercised by the second case.
#[test]
fn expand_text_window_clamps_far_edges_to_size_minus_one() {
    assert_eq!(
        expand_text_window(Rect::new(40, 30, 59, 49), (60, 50), ANNOTATION_EXPAND_R),
        Rect::new(39, 29, 59, 49),
        "far edges clamp to (im_w - 1, im_h - 1), not (im_w, im_h)"
    );
    assert_eq!(
        expand_text_window(Rect::new(0, 0, 19, 19), (60, 50), ANNOTATION_EXPAND_R),
        Rect::new(0, 0, 20, 20),
        "near edges clamp at 0"
    );
}

/// spec §16.37 item 5 quotes upstream's `int(round(...))`; Python's `round` on a float is
/// round-half-to-**even**, and Rust's `f64::round` is round-half-away-from-zero.
///
/// Hand-derived. A 40x40 box gives `(40*0.25 + 40*0.75)/16 = 40/16 = 2.5` exactly.
/// Python's `round(2.5)` is **2**; `f64::round(2.5)` is 3. So the expected window is
/// `[28, 28, 72, 72]` and an implementation using `f64::round` produces
/// `[27, 27, 73, 73]`.
#[test]
fn expand_text_window_rounds_halves_to_even_like_python() {
    assert_eq!(
        expand_text_window(Rect::new(30, 30, 70, 70), (100, 100), ANNOTATION_EXPAND_R),
        Rect::new(28, 28, 72, 72),
        "round(2.5) is 2 in Python, not 3"
    );
}

// ============================================================ the recorded page

/// spec §16.37 item 3: *"Correctness evidence therefore comes from A1-A3's per-function
/// hand-derived assertions"* -- this is A1's whole-path check on real committed data, with
/// every expected value produced by **upstream's own Python**, not by `pc-detect`.
///
/// **Inputs**, both already committed and both read here rather than reconstructed:
///   * `…_base.png` -- the 1200x1660 RGB page as `pc_detect::run` writes it;
///   * `…_detector_mask.png` -- the **unrefined** U-Net mask. §16.37 item 4 is explicit
///     that this, not `…_raw_mask.png`, is the unrefined output: *"despite its name
///     `_raw_mask.png` holds the *refined* mask … while `_detector_mask.png` holds the
///     unrefined U-Net output"*. Upstream's `refine_mask` takes the unrefined mask.
///
/// **Oracle**: upstream `get_topk_masklist`'s greyscale/erode/histogram/top-k body plus
/// `expand_textwindow`, at pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, run
/// against these two files on 2026-08-06 under opencv 5.0.0, with `np.argsort(bins * -1)`
/// replaced by DEVIATION(23)'s stable rule. On this page the two tie rules agree --
/// §16.37 item 1 measured `…E01P01` as **byte-identical** between introsort and
/// `kind='stable'` -- so the literals below are not tie-rule-dependent, which is why the
/// tie rule gets its own synthetic test above rather than resting on this one.
///
/// **Anti-vacuity.** `1121` candidate pixels and `237` occupied bins cannot be computed
/// from the tree by any route this test takes; a gate that found zero of everything would
/// fail on both. The second block asserted is the empty-candidate one, so the test also
/// cannot pass by always producing an empty candidate set.
#[test]
fn annotation_topk_path_on_the_recorded_page_matches_the_upstream_oracle() {
    let page = image::open(paths::recorded(format!("{PAGE_STEM}_base.png")))
        .expect("committed base page must decode")
        .to_rgb8();
    let mask = image::open(paths::recorded(format!("{PAGE_STEM}_detector_mask.png")))
        .expect("committed detector mask must decode")
        .to_luma8();
    assert_eq!(page.dimensions(), (1200, 1660));
    assert_eq!(mask.dimensions(), (1200, 1660));

    let grey = rgb_to_gray(&page);

    // ---- block 2 of `…_detector_blocks.json`, the largest, class_index 1 -------------
    let window = expand_text_window(
        Rect::new(607, 631, 724, 703),
        page.dimensions(),
        ANNOTATION_EXPAND_R,
    );
    assert_eq!(
        window,
        Rect::new(602, 626, 729, 708),
        "upstream crop window"
    );
    let grey_crop = crop(&grey, window);
    let mask_crop = crop(&mask, window);
    assert_eq!(grey_crop.dimensions(), (127, 82));
    // Upstream `cv2.cvtColor(im, COLOR_BGR2GRAY).astype(np.uint64).sum()` over this exact
    // crop.
    //
    // Measured limitation, recorded so nobody reads more into this line than it carries:
    // on the whole committed 1200x1660 page there are **zero** pixels where the 14-bit
    // triple or `round(0.299r + 0.587g + 0.114b)` disagree with opencv 5.0.0 (checked
    // 2026-08-06 over all 1 992 000 pixels), so *no* assertion over this fixture can
    // distinguish those two from the shipped weights. This sum does catch a truncating
    // implementation (835 327 pixels differ, page sum 204 600 340 vs 205 435 667), and it
    // is a real-data anti-vacuity literal, but the weights themselves are pinned only by
    // `rgb_to_gray_matches_opencv_bgr2gray_on_oracle_pixels`.
    assert_eq!(
        grey_crop.pixels().map(|p| u64::from(p.0[0])).sum::<u64>(),
        2_119_545,
        "greyscale sum over the crop"
    );

    let values = candidate_grey_values(&grey_crop, &mask_crop).expect("equal sizes");
    assert_eq!(values.len(), 1121, "upstream candidate_grey_px count");
    assert_eq!(values.iter().copied().min(), Some(5), "candidate min grey");
    assert_eq!(
        values.iter().copied().max(),
        Some(254),
        "candidate max grey"
    );

    let histogram = histogram_255(&values);
    assert_eq!(
        histogram.counts.iter().sum::<u32>(),
        1121,
        "no sample dropped"
    );
    assert_eq!(
        histogram.counts.iter().filter(|count| **count > 0).count(),
        237,
        "occupied bins"
    );
    assert_eq!(histogram.edges[0], 5.0);
    assert_eq!(histogram.edges[HISTOGRAM_BINS], 254.0);

    // The first six bins in DEVIATION(23) order, as (bin index, count, left edge). The
    // count-17 tie between bins 25 and 31 is decided by the declared rule.
    let order = deviation23_order(&histogram);
    let want_order: Vec<(usize, u32, f64)> = vec![
        (25, 17, 29.411_764_705_882_35),
        (31, 17, 35.270_588_235_294_11),
        (30, 16, 34.294_117_647_058_826),
        (23, 15, 27.458_823_529_411_763),
        (26, 15, 30.388_235_294_117_646),
        (28, 15, 32.341_176_470_588_24),
    ];
    assert_eq!(
        order[..6].to_vec(),
        want_order,
        "top six bins in DEVIATION(23) order"
    );

    assert_eq!(
        top_k_colors_default(&histogram),
        vec![
            29.411_764_705_882_35,
            40.152_941_176_470_584,
            76.282_352_941_176_47,
        ],
        "upstream get_topk_color on this block"
    );

    // ---- block 3, `[438, 1407, 498, 1446]`: the empty-candidate block ---------------
    // §16.37 item 4's block, the one the coverage filter drops. Its eroded mask crop has
    // no pixel above 127 at all, so A1's degenerate branch is exercised by real data.
    let window = expand_text_window(
        Rect::new(438, 1407, 498, 1446),
        page.dimensions(),
        ANNOTATION_EXPAND_R,
    );
    assert_eq!(window, Rect::new(435, 1404, 501, 1449));
    let values =
        candidate_grey_values(&crop(&grey, window), &crop(&mask, window)).expect("equal sizes");
    assert!(
        values.is_empty(),
        "expected no candidate pixels, got {}",
        values.len()
    );
    let histogram = histogram_255(&values);
    assert_eq!(histogram.edges[0], 0.0);
    assert_eq!(histogram.edges[HISTOGRAM_BINS], 1.0);
    assert_eq!(top_k_colors_default(&histogram), vec![0.0]);
}

// ------------------------------------------------------------------- helpers

/// `(bin index, count)` for every occupied bin, ascending by index.
fn nonzero_bins(histogram: &Histogram255) -> Vec<(usize, u32)> {
    histogram
        .counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(index, count)| (index, *count))
        .collect()
}

/// A histogram whose edges are `edges[i] == i` -- the `min = 0, max = 255` geometry
/// asserted by [`histogram_255_closes_the_last_bin_on_the_right`] -- with the given
/// `(bin, count)` pairs set and every other bin empty. Built by hand so the top-k tests
/// do not depend on `histogram_255` being correct.
fn synthetic_histogram(bins: &[(usize, u32)]) -> Histogram255 {
    let mut counts = [0_u32; HISTOGRAM_BINS];
    for (index, count) in bins {
        counts[*index] = *count;
    }
    let mut edges = [0.0_f64; HISTOGRAM_BINS + 1];
    for (index, edge) in edges.iter_mut().enumerate() {
        *edge = index as f64;
    }
    Histogram255 { counts, edges }
}

/// The DEVIATION(23) traversal order, recomputed here from the counts alone -- a stable
/// sort on `(Reverse(count), ascending bin index)` -- so the recorded-page assertion
/// compares against a rule spelled out in the test rather than against whatever
/// `top_k_colors` happens to do internally.
fn deviation23_order(histogram: &Histogram255) -> Vec<(usize, u32, f64)> {
    let mut order: Vec<(usize, u32, f64)> = (0..HISTOGRAM_BINS)
        .map(|index| (index, histogram.counts[index], histogram.edges[index]))
        .collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    order
}

fn crop(image: &GrayImage, rect: Rect) -> GrayImage {
    image::imageops::crop_imm(
        image,
        rect.x1 as u32,
        rect.y1 as u32,
        rect.width() as u32,
        rect.height() as u32,
    )
    .to_image()
}
