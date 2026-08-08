//! Task A2 -- spec §16.37 item 8: *"**A2** (**heavy**): Otsu thresholding and the
//! XOR-minimising channel selection."* FROZEN with the implementation.
//!
//! **Where every expected value comes from.** §16.37 item 3 is explicit that a
//! self-comparison gate proves determinism and not correctness -- *"A port that is wrong in
//! the same way on every run passes it forever. Correctness evidence therefore comes from
//! A1-A3's per-function hand-derived assertions"*. So every literal below comes from one of
//! exactly two places, never from `pc_detect::annotate`:
//!   * hand derivation, stated in the test's own comment; or
//!   * an independent oracle -- upstream PanelCleaner's `get_otsuthresh_masklist` /
//!     `minxor_thresh` / `get_topk_masklist` at pinned commit
//!     `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3` (`minxor_thresh` at `:32`,
//!     `get_otsuthresh_masklist` at `:49`, fetched from GitHub on 2026-08-06 and confirmed
//!     line-for-line), executed against `cv2` **5.0.0** -- the version
//!     `tests/fixtures/recorded/detector/PROVENANCE.json` pins -- and `numpy` 2.4.4, on
//!     2026-08-06.
//!
//! **The one place `cv2` is not a single-valued oracle, measured not argued.** OpenCV's
//! `getThreshVal_Otsu` (`modules/imgproc/src/thresh.cpp`, branch `5.x`) is guarded by
//! `CV_IPP_RUN_FAST(ipp_getThreshVal_Otsu_8u(...))`, and this wheel is built with Intel IPP
//! 2026.0.0 (`cv2.getBuildInformation()`). The two paths are **not** equivalent:
//!   * with IPP **disabled** (`cv2.ipp.setUseIPP(False)`), OpenCV's own C++ reference
//!     algorithm -- reimplemented independently in Python and validated first -- agreed with
//!     `cv2.threshold(..., THRESH_OTSU)` on **20 000 of 20 000** random/degenerate arrays;
//!   * with IPP **enabled** (the default, i.e. what upstream PanelCleaner actually executes
//!     on x86), **10 of the same 20 000** disagree. **Both mechanisms are present in the
//!     divergence**: some disagreements are ULP-level differences between two candidate
//!     between-class variances, and others are **exact, bit-identical** variances, i.e. a
//!     genuine tie decided by nothing but the tie rule. **Doc-comment correction, 2026-08-06
//!     (§16.37 item 10; no assertion in this file changed):** these three lines previously
//!     read *"None of the 10 is an exact tie in double arithmetic: in every one the two
//!     candidate between-class variances differ in the last few ULPs, so the mechanism is
//!     IPP's arithmetic, not a tie rule"*. That was an overclaim about a corpus, restated as
//!     a claim about the mechanism: re-runs over further independently generated corpora --
//!     this file's own 10 per 20 000, plus three more measuring 20, 21 and 54 per 20 000 --
//!     found exact bit-identical-sigma disagreements arising from **random,
//!     non-constructed** data (15 of the 54 in one corpus), which contradicts the sentence
//!     above directly. The disagreement *rate* is not reproducible and no band is quoted
//!     here: §16.37 item 10(a) tabulates every run on record, from 0 to 460 per 20 000
//!     depending on the corpus's tie density.
//!   * on inputs that are exact ties, the two paths can pick different thresholds and
//!     therefore different masks -- see
//!     [`otsu_threshold_breaks_exact_between_class_variance_ties_toward_the_lowest_index`].
//!   * IPP's own answer varies with its **dispatch level** even with IPP held enabled
//!     (`OPENCV_IPP=sse42` against `avx2`/`avx512` over one fixed array corpus), so "match
//!     IPP" is not merely expensive but not well-defined as a target.
//!
//! This crate implements the **documented C++ reference** path (`sigma > max_sigma`, strict,
//! so the lowest index wins a tie), which is also `cv2`-with-IPP-off. That choice is now
//! settled and registered as **`DEVIATION(29)`** (§14 item 29, ratified by §16.37 item 10);
//! this file's earlier statement that it was "not decided by this file" and "escalated to the
//! two architects" described the state at the time of writing and the escalation has since
//! returned. What is asserted here is only what each test's name says: the reference rule,
//! never parity with upstream-as-executed on an IPP build.
//!
//! **The committed page does not exercise any tie.** Measured on
//! `ja_Pepper-and-Carrot_by-David-Revoy_E01P01`: all four blocks have three strictly
//! distinct per-channel XOR sums (block 0 `[201119, 160916, 161839]`, block 1
//! `[263001, 258511, 261323]`, block 2 `[370402, 361494, 367986]`, block 3
//! `[213650, 151787, 143200]`, in upstream's B, G, R order), and all 12 per-channel Otsu
//! thresholds agree between the IPP and reference paths. Every tie assertion below therefore
//! runs on a constructed input, and each constructed input was executed through `cv2` before
//! its expected value was written down.
//!
//! **Scope.** A2 covers Annotation primitives; run wiring is A4. The A4-a integration gate
//! passes; connected components, the merge loop and hole filling are A3, wiring is A4.
//! Nothing here asserts anything about `MaskRefineMode::Simple`.

use image::{GrayImage, Luma, Rgb, RgbImage};
use pc_core::Rect;
use pc_detect::annotate::{
    candidate_mask_list, in_range, minxor_thresh, otsu_thresh_mask_list, otsu_threshold,
    threshold_binary, top_k_mask_list, xor_sum, MaskCandidate, TOPK_COLOR_RANGE,
};
use pc_testkit::paths;

const PAGE_STEM: &str = "detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

// ==================================================================== Otsu threshold

/// spec §16.37 item 8, A2's "Otsu thresholding" -- upstream
/// `cv2.threshold(c, 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)` (`textmask.py:53`).
///
/// **Oracle**: `cv2.threshold(..., THRESH_OTSU)` at opencv 5.0.0 on 2026-08-06, run on each
/// array below **with IPP disabled** (see the module header for why that qualifier is
/// load-bearing). Ten arrays were probed on both paths: the IPP and reference paths agreed on
/// nine -- the nine asserted here -- and disagreed on the tenth, `[1, 1, 2, 2, 3, 3]` (IPP 2,
/// reference 1, a near-tie differing in the last ULPs), which is therefore **excluded** from
/// this table and recorded in the module header instead.
///
/// **What each row excludes.**
///   * `[10, 10, 200, 200, 200]` -> **10**, not 199 and not 105: OpenCV's search runs over
///     bin *indices* and keeps the **lowest** index of a tied plateau, so the answer is the
///     lower cluster's value, not the top of the empty gap (`>=` instead of `>` gives 199)
///     and not the midpoint a "mean of the two class means" formulation would give.
///   * `[0..=255]` -> **127**, not 128: the flat histogram's variance curve is symmetric, and
///     the tie again resolves downward.
///   * `[42]` -> **0** and `[0]*9 + [200]` -> **0**: every candidate index is rejected by
///     OpenCV's `min(q1,q2) < FLT_EPSILON || max(q1,q2) > 1 - FLT_EPSILON` guard for the
///     single-valued case, and `max_val` keeps its initial 0. An implementation that
///     initialises the answer to 255, or that drops the epsilon guard, fails these two.
///   * `[5,5,5,120,120,121,240,240,240,241]` -> **121**: three clusters, so the answer is not
///     recoverable from "min + max" arithmetic.
#[test]
fn otsu_threshold_matches_opencv_threshold_otsu_on_oracle_arrays() {
    const ORACLE: &[(&str, &[u8], u8)] = &[
        ("two clusters 0/255", &[0, 0, 0, 255, 255, 255], 0),
        ("two clusters 10/200", &[10, 10, 200, 200, 200], 10),
        (
            "extremes",
            &[0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255],
            5,
        ),
        ("tight ramp", &[100, 110, 120, 130, 140], 120),
        ("skewed", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 200], 0),
        (
            "three clusters",
            &[5, 5, 5, 120, 120, 121, 240, 240, 240, 241],
            121,
        ),
        ("single pixel", &[42], 0),
        ("two pixels", &[7, 200], 7),
    ];

    for (name, values, want) in ORACLE {
        let image = row_image(values);
        assert_eq!(otsu_threshold(&image), *want, "otsu threshold for {name}");
    }

    // The 256-value flat histogram, built separately because it does not fit the table.
    let flat: Vec<u8> = (0..=255).collect();
    assert_eq!(
        otsu_threshold(&row_image(&flat)),
        127,
        "flat histogram: the symmetric tie resolves to the lower index"
    );
}

/// spec §16.37 item 8, A2's Otsu step, on an input where two **different** thresholds have
/// bit-identical between-class variance in double arithmetic.
///
/// **What this pins**: OpenCV's `if (sigma > max_sigma)` is strict, so of two exactly tied
/// candidate thresholds the **lower index** is kept. Input `[108,108,129,129,129,129,129,150,150]`
/// has `sigma == 125.99999999999999` at both index 108 and index 129 (bit-identical; verified
/// by comparing the two `f64` values for equality in the independent Python reference), and
/// the two thresholds produce **different masks**: at 108 the five 129s are on, at 129 they
/// are off. So this input, unlike a tie inside a run of empty bins, makes the tie rule
/// observable in the output.
///
/// **What turns this red**: `sigma >= max_sigma`, which yields 129 -- and with it a mask that
/// differs on 5 of the 9 pixels, asserted below so the test fails on the mask and not only on
/// the threshold number.
///
/// **Measured divergence, stated rather than hidden (cookbook rule 1).** `cv2` 5.0.0 with
/// Intel IPP **enabled** -- the default, and what upstream PanelCleaner runs -- returns
/// **129** for this array; with IPP disabled it returns **108**, agreeing with the C++
/// reference this crate ports. This test asserts the **reference** rule. It does **not**
/// assert parity with upstream-as-executed on an IPP build, and its name does not claim to.
#[test]
fn otsu_threshold_breaks_exact_between_class_variance_ties_toward_the_lowest_index() {
    let image = row_image(&[108, 108, 129, 129, 129, 129, 129, 150, 150]);
    assert_eq!(
        otsu_threshold(&image),
        108,
        "of two exactly tied thresholds the lower index wins"
    );
    assert_eq!(
        row_values(&threshold_binary(&image, otsu_threshold(&image))),
        vec![0, 0, 255, 255, 255, 255, 255, 255, 255],
        "the tie is observable: threshold 129 would leave the five 129s off"
    );
}

/// spec §16.37 item 8, A2's Otsu step on the degenerate inputs a crop can actually produce.
///
/// **Hand-derived from OpenCV's guard**, which is quoted in [`otsu_threshold`]'s doc: for a
/// constant image every index has `min(q1, q2) < FLT_EPSILON`, every iteration is skipped,
/// and `max_val` keeps its initial `0`. The consequence is asserted rather than left implicit:
/// a constant image of any **non-zero** value binarises to **all 255** (`v > 0`), and a
/// constant `0` image to all 0.
///
/// **The empty case is ours, not upstream's.** `cv2.threshold` on a `0 x 0` array does not
/// return a threshold at all (it produced `None` and a downstream `AttributeError` when run
/// on 2026-08-06), so there is no upstream value to be faithful to; returning `0` is this
/// crate's own totality choice, and it is stated here so nobody later reads it as parity.
/// Upstream never reaches it: `expand_text_window` cannot produce an empty window from a
/// non-empty rect.
#[test]
fn otsu_threshold_of_a_constant_or_empty_image_is_zero() {
    for value in [1_u8, 7, 128, 255] {
        let image = GrayImage::from_pixel(3, 4, Luma([value]));
        assert_eq!(otsu_threshold(&image), 0, "constant {value}");
        assert!(
            threshold_binary(&image, 0).pixels().all(|p| p.0[0] == 255),
            "constant {value} binarises to all 255"
        );
    }

    let zeros = GrayImage::from_pixel(3, 4, Luma([0]));
    assert_eq!(otsu_threshold(&zeros), 0, "constant 0");
    assert!(
        threshold_binary(&zeros, 0).pixels().all(|p| p.0[0] == 0),
        "constant 0 binarises to all 0"
    );

    assert_eq!(
        otsu_threshold(&GrayImage::new(0, 0)),
        0,
        "our own totality choice, not upstream parity"
    );
}

/// spec §16.37 item 8 -- `cv2.THRESH_BINARY`: `dst = src > thresh ? maxval : 0`, **strict**.
///
/// Hand-derived from that formula. With `threshold = 3` the value 3 itself must be **off**;
/// `>=` would turn it on. The `maxval` upstream passes is `255`.
#[test]
fn threshold_binary_is_strictly_greater_than_the_threshold() {
    let image = row_image(&[0, 1, 2, 3, 4, 5, 254, 255]);
    assert_eq!(
        row_values(&threshold_binary(&image, 3)),
        vec![0, 0, 0, 0, 255, 255, 255, 255],
        "value == threshold stays off"
    );
    assert_eq!(
        row_values(&threshold_binary(&image, 255)),
        vec![0; 8],
        "nothing exceeds 255"
    );
}

// ==================================================================== XOR sum

/// spec §16.37 item 8, A2's "XOR-minimising" selection -- the sum inside upstream's
/// `minxor_thresh`: `cv2.bitwise_xor(threshed, mask).sum()` (`textmask.py:41-42`).
///
/// **Oracle**: `cv2.bitwise_xor` on 2026-08-06 over exactly these two rows, whose printed
/// output was `[127, 128, 55, 255, 0, 255]` with sum `820`.
///
/// **Why not a 0/255-only row.** The pred-mask this is scored against is the raw U-Net
/// output: `…_detector_mask.png` holds **255 distinct values**, and 0.65% of its pixels are
/// neither 0 nor 255 (measured 2026-08-06). So the operation is a per-pixel **bitwise** XOR
/// over all eight bits, not a boolean symmetric difference, and the rows here are chosen so
/// the two disagree: `255 ^ 128 = 127` while a boolean reading would call those two pixels
/// *equal* (both "on") and score 0.
#[test]
fn xor_sum_is_bitwise_over_all_eight_bits() {
    let a = row_image(&[255, 0, 200, 170, 128, 1]);
    let b = row_image(&[128, 128, 255, 85, 128, 254]);
    assert_eq!(
        xor_sum(&a, &b).expect("equal sizes"),
        820,
        "cv2.bitwise_xor(a, b).sum()"
    );
}

/// spec §16.37 item 8, and the plan's arithmetic instruction: the accumulator is `u64`,
/// because `u32` overflows.
///
/// **Hand-derived, and it cannot be satisfied by a `u32` accumulator.** A 4200x4200 image is
/// 17 640 000 pixels; XOR-ing all-255 against all-0 gives 255 everywhere, so the sum is
/// `255 * 17 640 000 = 4 498 200 000`, which exceeds `u32::MAX = 4 294 967 295` by
/// 203 232 705. A `u32` accumulator wraps to 203 232 705 (or panics in a debug build); either
/// way this test is red.
#[test]
fn xor_sum_accumulates_past_the_u32_range() {
    const SIDE: u32 = 4200;
    let ones = GrayImage::from_pixel(SIDE, SIDE, Luma([255]));
    let zeros = GrayImage::from_pixel(SIDE, SIDE, Luma([0]));

    let want = 255_u64 * u64::from(SIDE) * u64::from(SIDE);
    assert_eq!(want, 4_498_200_000, "hand-derived total");
    assert!(want > u64::from(u32::MAX), "the point of the test");
    assert_eq!(xor_sum(&ones, &zeros).expect("equal sizes"), want);
}

/// A size mismatch is a per-image `StageError::InvalidInput`, never run-fatal (cookbook
/// rule 4: fatality is *declared*, not inferred from a signature). Upstream's
/// `cv2.bitwise_xor` raises on mismatched shapes; one bad block must not abort the run.
#[test]
fn xor_sum_rejects_mismatched_sizes_as_invalid_input() {
    let error = xor_sum(&GrayImage::new(4, 4), &GrayImage::new(4, 5)).expect_err("sizes differ");
    assert!(
        matches!(error, pc_core::StageError::InvalidInput(_)),
        "expected InvalidInput, got {error:?}"
    );
}

// ==================================================================== minxor_thresh

/// spec §16.37 item 8, A2's `minxor_thresh` (upstream `textmask.py:32-46`), on an **exact
/// tie** between the positive mask and its inverse.
///
/// **Oracle**: run through `cv2` on 2026-08-06 -- this exact 4x4 pair printed
/// `pos_sum 2040 neg_sum 2040` and upstream's own branch returned the **positive** mask.
///
/// **Hand-derivable too, and the derivation is the point.** `threshed` is the left half (8 of
/// 16 pixels at 255), `mask` the top half (also 8). Their symmetric difference is 8 pixels, so
/// `xor_sum = 8 * 255 = 2040`; the inverse is the right half, whose symmetric difference with
/// the top half is also 8, so `neg_xor_sum = 2040` as well. Upstream's comparison is
/// `if neg_xor_sum < xor_sum` -- **strict** -- so the positive mask wins.
///
/// **What turns this red**: `<=`, which returns the *inverse* -- the right half instead of the
/// left. Both branches return the same `xor_sum` (2040), so only the mask distinguishes them,
/// and the assertion is therefore on the mask's pixels, not on the score. The second
/// assertion states the negative explicitly so a future reader cannot mistake which of the
/// two 2040-scoring masks was chosen.
#[test]
fn minxor_thresh_keeps_the_positive_mask_when_the_two_scores_tie_exactly() {
    let threshed = half_mask(Half::Left);
    let mask = half_mask(Half::Top);

    let candidate = minxor_thresh(&threshed, &mask).expect("equal sizes");
    assert_eq!(candidate.xor_sum, 2040, "8 differing pixels * 255");
    assert_eq!(
        pixels(&candidate.mask),
        pixels(&threshed),
        "the tie keeps the positive mask"
    );
    assert_ne!(
        pixels(&candidate.mask),
        pixels(&half_mask(Half::Right)),
        "and not the inverse, which scores exactly the same 2040"
    );
}

/// spec §16.37 item 8, A2's `minxor_thresh` -- the branch that **does** fire, so the test
/// above cannot pass merely because the function never inverts anything.
///
/// Hand-derived. `threshed` is the left half and `mask` the right half of a 4x4 image: they
/// differ everywhere, so `xor_sum = 16 * 255 = 4080`, while the inverse *is* the right half
/// and scores `0`. `0 < 4080`, so the inverse is returned with score 0.
#[test]
fn minxor_thresh_returns_the_inverted_mask_when_it_scores_strictly_lower() {
    let candidate =
        minxor_thresh(&half_mask(Half::Left), &half_mask(Half::Right)).expect("equal sizes");
    assert_eq!(candidate.xor_sum, 0, "the inverse matches the mask exactly");
    assert_eq!(pixels(&candidate.mask), pixels(&half_mask(Half::Right)));
}

/// Per-image `StageError::InvalidInput` on a size mismatch (cookbook rule 4).
#[test]
fn minxor_thresh_rejects_mismatched_sizes_as_invalid_input() {
    let error =
        minxor_thresh(&GrayImage::new(4, 4), &GrayImage::new(5, 4)).expect_err("sizes differ");
    assert!(
        matches!(error, pc_core::StageError::InvalidInput(_)),
        "expected InvalidInput, got {error:?}"
    );
}

// ==================================================================== in_range

/// spec §16.37 item 8, the `cv2.inRange` band inside upstream's `get_topk_masklist`
/// (`textmask.py:81`), which A1 deliberately deferred to A2 (*"Step 6 of upstream's function
/// (`cv2.inRange` over each colour, then `minxor_thresh`) is deliberately **not** here"*).
///
/// **Oracle**: `cv2.inRange(np.arange(256, dtype=uint8), lo, hi)` on 2026-08-06; each row
/// below is the first and last index of the printed non-zero run.
///
/// **What the rows exclude.** The bounds arrive as floats (a histogram bin edge plus 30) and
/// OpenCV converts them to the source depth with `cvRound`, i.e. round-**half-to-even**, and
/// then compares inclusively. So:
///   * `(2.4, 12.6)` -> `[2, 13]`: the value **2** passes although `2 < 2.4`, and **13**
///     passes although `13 > 12.6`. Truncating or comparing in floating point gives `[3, 12]`.
///   * `(0.5, 10.5)` -> `[0, 10]` and `(1.5, 11.5)` -> `[2, 12]`: both halves round to even.
///     `f64::round` (half away from zero) gives `[1, 11]` and `[2, 12]`, so the **first** row
///     is what separates the two rounding modes.
///   * `(2.5, 3.5)` -> `[2, 4]` and `(3.5, 4.5)` -> `[4, 4]`: the same separation again, in
///     both directions, on adjacent inputs.
///   * `(-0.588…, 59.411…)` -> `[0, 59]`: the real bounds of block 2's first top-k colour;
///     a negative lower bound saturates rather than wrapping.
#[test]
fn in_range_rounds_its_bounds_half_to_even_and_compares_inclusively() {
    const ORACLE: &[(f64, f64, u8, u8)] = &[
        (0.5, 10.5, 0, 10),
        (1.5, 11.5, 2, 12),
        (2.4, 12.6, 2, 13),
        (2.5, 3.5, 2, 4),
        (3.5, 4.5, 4, 4),
        (-0.588_235_294_117_646_4, 59.411_764_705_882_35, 0, 59),
        (46.282_352_941_176_47, 106.282_352_941_176_47, 46, 106),
    ];

    let ramp = row_image(&(0..=255).collect::<Vec<u8>>());
    for (lower, upper, first, last) in ORACLE {
        let inside: Vec<u8> = row_values(&in_range(&ramp, *lower, *upper))
            .into_iter()
            .enumerate()
            .filter(|(_, on)| *on == 255)
            .map(|(index, _)| index as u8)
            .collect();
        let want: Vec<u8> = (*first..=*last).collect();
        assert_eq!(inside, want, "cv2.inRange(ramp, {lower}, {upper})");
    }
}

/// spec §16.37 item 8: upstream's band is `c_top = min(color + 30, 255)`,
/// `c_bottom = c_top - 2 * 30` (`textmask.py:79-80`) -- anchored at the **top**, so the clamp
/// at 255 shifts the whole 60-wide window down rather than truncating it.
///
/// **This test calls [`top_k_mask_list`], not [`in_range`] with bounds recomputed here**
/// (cookbook rule 12: the gate must point at the artifact carrying the risk). A first draft of
/// it asserted the band arithmetic in the test body; a falsification probe that replaced
/// `c_bottom` inside `top_k_mask_list` with the symmetric reading `max(color - 30, 0)` left
/// all 19 tests **green**, because the two formulas agree for every colour on the committed
/// page (they differ only when the upper bound clamps, i.e. above 225, and the page's highest
/// top-k colour is 181.4). Hence this constructed input.
///
/// **Oracle**: upstream `get_topk_masklist` on this exact 8x8 pair, 2026-08-06. The nine
/// candidate greys are all 250, so `np.histogram`'s `min == max` branch widens the range and
/// the single colour is `249.99803921568628`; upstream's band is then `[195, 255]` and it
/// returns one candidate with `xor_sum = 510` over **27** non-zero pixels (fingerprint 765).
/// The symmetric reading's band `[220, 255]` returns `xor_sum = 0` over **25** pixels
/// (fingerprint 700) -- the two 200-valued pixels outside the mask are what separate them.
#[test]
fn top_k_mask_list_anchors_the_band_at_the_clamped_upper_bound() {
    assert_eq!(TOPK_COLOR_RANGE, 30.0);

    // 250 on a 5x5 block at x,y in 1..=5 (its eroded core, x,y in 2..=4, is the candidate
    // set); 200 at two pixels outside the mask; 0 elsewhere.
    let mut grey = GrayImage::new(8, 8);
    for y in 1..=5 {
        for x in 1..=5 {
            grey.put_pixel(x, y, Luma([250]));
        }
    }
    grey.put_pixel(7, 0, Luma([200]));
    grey.put_pixel(0, 7, Luma([200]));
    let mut mask = GrayImage::new(8, 8);
    for y in 1..=5 {
        for x in 1..=5 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }

    let list = top_k_mask_list(&grey, &mask).expect("equal sizes");
    assert_eq!(scores(&list), vec![510], "band [195, 255] admits the 200s");
    assert_eq!(
        nonzero_count(&list[0].mask),
        27,
        "the 25-pixel block plus the two 200s; band [220, 255] gives 25"
    );
    assert_eq!(positional_fingerprint(&list[0].mask), 765);
}

// ============================================== channel selection: the tie sites

/// spec §16.37 item 8, A2's *"XOR-minimising channel selection"* -- upstream
/// `get_otsuthresh_masklist` (`textmask.py:49-60`): `channels = [img[...,0], img[...,1],
/// img[...,2]]` over a **BGR** array, then `mask_list.sort(key=lambda x: x[1])`, which is
/// Python's guaranteed-**stable** `list.sort`, and `return [mask_list[0]]`.
///
/// **Oracle**: this exact 4x4 BGR array and mask were run through upstream's function on
/// 2026-08-06. It printed per-channel sums `B=765, G=1275, R=765` -- an exact tie at the
/// minimum between channel 0 (B) and channel 2 (R) -- a stable order of `[0, 2, 1]`, and
/// returned B's mask.
///
/// **Hand-derivable as well.** Every channel is 0-or-200, so its Otsu threshold is 0 and its
/// binarisation is exactly the 200-set. The mask is the left half (8 of 16 pixels). B's set
/// differs from the mask in 3 pixels and R's in 3 **different** pixels, so both score
/// `3 * 255 = 765`; G's differs in 5, scoring 1275. Neither 765-scoring candidate is inverted
/// (13 pixels differ from the inverse, 3315 > 765).
///
/// **What turns this red.** Iterating an `RgbImage` in index order 0, 1, 2 -- R, G, B --
/// returns **R's** mask, which is a different image with the same score. The assertion is on
/// the mask's pixels and explicitly denies R's mask; a score-only or length-only assertion
/// would pass under the swap (cookbook: cardinality is not identity). `sort_unstable_by_key`
/// would also be able to return either.
///
/// **The committed page cannot substitute for this.** All four of its blocks have three
/// strictly distinct per-channel sums (module header), so on real data the iteration order is
/// unobservable.
#[test]
fn otsu_thresh_mask_list_breaks_a_channel_tie_toward_upstreams_first_channel_blue() {
    let (image, mask, blue, red) = channel_tie_case();

    let list = otsu_thresh_mask_list(&image, &mask).expect("equal sizes");
    assert_eq!(list.len(), 1, "per_channel=False returns one candidate");
    assert_eq!(list[0].xor_sum, 765, "3 differing pixels * 255");
    assert_eq!(
        pixels(&list[0].mask),
        pixels(&blue),
        "upstream iterates B, G, R and a stable sort keeps the first minimum"
    );
    assert_ne!(
        pixels(&list[0].mask),
        pixels(&red),
        "R's candidate ties at 765 with a different mask; picking it means RGB order"
    );
}

/// spec §16.37 item 8: upstream's `get_otsuthresh_masklist(..., per_channel=False)` --
/// *"`if per_channel: return mask_list; else: return [mask_list[0]]`"* -- yields **one**
/// candidate, not three, and it is the lowest-scoring one.
///
/// **Oracle**: upstream's function on block 0 of the committed page (window
/// `[671, 1394, 743, 1441]`) printed per-channel sums `B=201119, G=160916, R=161839`,
/// selected channel 1 (G), and its returned mask had **1025** non-zero pixels.
///
/// **Anti-vacuity and identity.** 1025 cannot be computed from this tree by any route the
/// test takes. The other two candidates have 1314 (B) and 994 (R) non-zero pixels -- both
/// hard-coded here and both denied -- so returning the wrong channel's mask fails even though
/// all three are the same size. Returning all three fails on the length.
#[test]
fn otsu_thresh_mask_list_returns_only_the_lowest_scoring_channels_candidate() {
    let (image, mask) = recorded_crop(
        Rect::new(674, 1397, 740, 1438),
        Rect::new(671, 1394, 743, 1441),
    );

    let list = otsu_thresh_mask_list(&image, &mask).expect("equal sizes");
    assert_eq!(list.len(), 1, "per_channel=False");
    assert_eq!(list[0].xor_sum, 160_916, "upstream's winning xor_sum (G)");
    let nonzero = nonzero_count(&list[0].mask);
    assert_eq!(nonzero, 1025, "upstream's winning mask (G)");
    assert_ne!(nonzero, 1314, "not B's candidate");
    assert_ne!(nonzero, 994, "not R's candidate");
}

/// spec §16.37 item 8, A2's XOR **sum** as the selection key: upstream scores candidates by
/// `cv2.bitwise_xor(...).sum()` over the raw pred mask, not by counting pixels that differ
/// after thresholding the mask.
///
/// **Oracle**: this 1x4 case was run through upstream's `get_otsuthresh_masklist` on
/// 2026-08-06. It printed `ch0(B) sum=509 neg`, `ch1(G) sum=509 pos`, `ch2(R) sum=256 pos`,
/// and returned R's mask `[0, 0, 0, 255]` with sum 256.
///
/// **Why this input.** `mask = [128, 128, 0, 255]`. R's candidate `[0, 0, 0, 255]` disagrees
/// with the mask on **two** pixels but only by 1 bit-weight each (`0 ^ 128 = 128` twice, plus
/// `0 ^ 0 = 0` and `255 ^ 255 = 0`) for a total of 256. G's candidate `[255, 255, 255, 255]`
/// disagrees on **one** pixel yet totals `127 + 127 + 255 + 0 = 509`. So the bitwise sum and a
/// boolean differing-pixel count rank these two candidates in **opposite** orders: measured
/// boolean counts were R 2 versus G 1.
///
/// **What turns this red**: scoring with a boolean symmetric difference (thresholding the
/// mask at 127 and counting mismatches), which selects an all-255 mask instead of
/// `[0, 0, 0, 255]`.
#[test]
fn otsu_thresh_mask_list_scores_candidates_by_bitwise_xor_not_by_a_boolean_pixel_count() {
    let mask = row_image(&[128, 128, 0, 255]);
    let mut image = RgbImage::new(4, 1);
    // (r, g, b) per pixel: upstream channel 0 is B, channel 2 is R.
    for (x, (r, g, b)) in [(0, 200, 0), (0, 200, 0), (0, 200, 0), (200, 200, 0)]
        .into_iter()
        .enumerate()
    {
        image.put_pixel(x as u32, 0, Rgb([r, g, b]));
    }

    let list = otsu_thresh_mask_list(&image, &mask).expect("equal sizes");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].xor_sum, 256, "128 + 128, bitwise");
    assert_eq!(
        row_values(&list[0].mask),
        vec![0, 0, 0, 255],
        "a boolean pixel count would select an all-255 candidate instead"
    );
}

/// Per-image `StageError::InvalidInput` when the crop and the mask crop disagree in size
/// (cookbook rule 4).
#[test]
fn otsu_thresh_mask_list_rejects_mismatched_sizes_as_invalid_input() {
    let error = otsu_thresh_mask_list(&RgbImage::new(4, 4), &GrayImage::new(4, 5))
        .expect_err("sizes differ");
    assert!(
        matches!(error, pc_core::StageError::InvalidInput(_)),
        "expected InvalidInput, got {error:?}"
    );
}

// ==================================================== the ordered candidate list

/// spec §16.37 item 8's sequencing constraint, taken from upstream `refine_mask`
/// (`textmask.py:206-207`): `mask_list = get_topk_masklist(im, msk)` then
/// `mask_list += get_otsuthresh_masklist(im, msk, per_channel=False)` -- a **concatenation**,
/// so the Otsu candidate is **last**, and `merge_mask_list` re-sorts the combined list itself.
/// A2's output must therefore be an ordered sequence; a set or map would destroy the position.
///
/// **Oracle**: upstream's two calls on the committed page, 2026-08-06.
///   * block 2 (window `[602, 626, 729, 708]`): xor sums in list order
///     `[450963, 435296, 472752, 361494]`, the last being the Otsu candidate, whose mask has
///     1077 non-zero pixels;
///   * block 3 (window `[435, 1404, 501, 1449]`): `[21655, 143200]` -- only **two** entries,
///     because that block's eroded mask has no pixel above 127 and `get_topk_color` returns a
///     single colour (§16.37 item 4's block, A1's degenerate branch).
///
/// **What turns this red.** The Otsu candidate is block 2's **lowest**-scoring entry
/// (361494 < 435296), so any implementation that sorts, or that prepends the Otsu candidate,
/// moves it out of last place and the asserted sequence fails. The two blocks also have
/// different lengths, so a hard-coded four-entry list fails on block 3.
#[test]
fn candidate_mask_list_appends_the_otsu_candidate_after_the_top_k_candidates() {
    let (image, mask) = recorded_crop(Rect::new(607, 631, 724, 703), Rect::new(602, 626, 729, 708));
    let list = candidate_mask_list(&image, &mask).expect("equal sizes");
    assert_eq!(
        scores(&list),
        vec![450_963, 435_296, 472_752, 361_494],
        "top-k candidates in colour order, then the Otsu candidate"
    );
    assert_eq!(
        nonzero_count(&list[3].mask),
        1077,
        "the last entry is the Otsu winner, not a top-k candidate"
    );

    let (image, mask) = recorded_crop(
        Rect::new(438, 1407, 498, 1446),
        Rect::new(435, 1404, 501, 1449),
    );
    let list = candidate_mask_list(&image, &mask).expect("equal sizes");
    assert_eq!(
        scores(&list),
        vec![21_655, 143_200],
        "one top-k colour on this block, then the Otsu candidate"
    );
    assert_eq!(nonzero_count(&list[1].mask), 581, "the Otsu winner");
}

// ==================================================== the recorded page, end to end

/// spec §16.37 item 3: *"Correctness evidence therefore comes from A1-A3's per-function
/// hand-derived assertions"* -- A2's whole-path check on real committed data, with every
/// expected value produced by **upstream's own Python**, not by `pc-detect`.
///
/// **Inputs**, both already committed: `…_base.png` (the 1200x1660 page) and
/// `…_detector_mask.png` -- the **unrefined** U-Net mask, per §16.37 item 4: *"despite its
/// name `_raw_mask.png` holds the *refined* mask … while `_detector_mask.png` holds the
/// unrefined U-Net output"*.
///
/// **Oracle**: upstream `get_otsuthresh_masklist` at the pinned commit, run against these two
/// files on 2026-08-06 under opencv 5.0.0. The 12 per-channel Otsu thresholds were **equal on
/// the IPP and reference paths**, so these literals do not depend on that divergence.
///
/// **What each row pins.** The three thresholds are asserted **per channel in upstream's B,
/// G, R order**, so a channel swap fails here even though it cannot fail on the winner alone
/// (the winner is a minimum over the same three scores either way). The winner's positional
/// fingerprint -- `sum of (row-major index + 1) over non-zero pixels` -- is asserted alongside
/// its non-zero count, so two masks with the same population but different geometry are
/// distinguished.
///
/// **Anti-vacuity.** No number below can be computed from the tree by any route this test
/// takes, and block 3 -- the block whose eroded mask is empty, which makes A1's top-k path
/// degenerate -- still has a fully populated Otsu candidate (581 pixels), so the test cannot
/// pass by producing empty masks.
#[test]
fn annotation_otsu_path_on_the_recorded_page_matches_the_upstream_oracle() {
    const ORACLE: &[OtsuOracleRow] = &[
        OtsuOracleRow {
            rect: Rect {
                x1: 674,
                y1: 1397,
                x2: 740,
                y2: 1438,
            },
            window: Rect {
                x1: 671,
                y1: 1394,
                x2: 743,
                y2: 1441,
            },
            thresholds_bgr: [76, 117, 131],
            xor: 160_916,
            nonzero: 1025,
            fingerprint: 1_885_794,
        },
        OtsuOracleRow {
            rect: Rect {
                x1: 567,
                y1: 74,
                x2: 663,
                y2: 123,
            },
            window: Rect {
                x1: 563,
                y1: 70,
                x2: 667,
                y2: 127,
            },
            thresholds_bgr: [122, 153, 134],
            xor: 258_511,
            nonzero: 795,
            fingerprint: 2_818_713,
        },
        OtsuOracleRow {
            rect: Rect {
                x1: 607,
                y1: 631,
                x2: 724,
                y2: 703,
            },
            window: Rect {
                x1: 602,
                y1: 626,
                x2: 729,
                y2: 708,
            },
            thresholds_bgr: [126, 156, 134],
            xor: 361_494,
            nonzero: 1077,
            fingerprint: 5_376_699,
        },
        OtsuOracleRow {
            rect: Rect {
                x1: 438,
                y1: 1407,
                x2: 498,
                y2: 1446,
            },
            window: Rect {
                x1: 435,
                y1: 1404,
                x2: 501,
                y2: 1449,
            },
            thresholds_bgr: [78, 116, 125],
            xor: 143_200,
            nonzero: 581,
            fingerprint: 932_556,
        },
    ];

    for &OtsuOracleRow {
        rect,
        window,
        thresholds_bgr: thresholds,
        xor,
        nonzero,
        fingerprint,
    } in ORACLE
    {
        let (crop, mask_crop) = recorded_crop(rect, window);

        // upstream `channels = [img[..., 0], img[..., 1], img[..., 2]]` over a BGR array;
        // `RgbImage` stores red at index 0, so B is index 2 here.
        for (upstream_index, rgb_index) in [(0_usize, 2_usize), (1, 1), (2, 0)] {
            assert_eq!(
                otsu_threshold(&channel(&crop, rgb_index)),
                thresholds[upstream_index],
                "block {rect:?} upstream channel {upstream_index} ({})",
                ["B", "G", "R"][upstream_index]
            );
        }

        let list = otsu_thresh_mask_list(&crop, &mask_crop).expect("equal sizes");
        assert_eq!(list.len(), 1, "per_channel=False");
        assert_eq!(list[0].xor_sum, xor, "block {rect:?} winning xor_sum");
        assert_eq!(
            nonzero_count(&list[0].mask),
            nonzero,
            "block {rect:?} winning mask population"
        );
        assert_eq!(
            positional_fingerprint(&list[0].mask),
            fingerprint,
            "block {rect:?} winning mask geometry"
        );
    }
}

/// spec §16.37 item 8 -- the half of upstream's `get_topk_masklist` that A1 deferred to A2:
/// `cv2.inRange` over each top-k colour, then `minxor_thresh`.
///
/// **Oracle**: upstream `get_topk_masklist` at the pinned commit, run on the committed page on
/// 2026-08-06 with `get_topk_color`'s `np.argsort` replaced by DEVIATION(23)'s stable rule
/// (§16.37 item 1 measured this page as byte-identical between the two tie rules, so these
/// literals are not tie-rule-dependent).
///   * block 0: colours `[181.41176470588235, 170.81176470588235, 62.94117647058823]`, xor
///     sums `[177954, 171579, 178942]`, non-zero counts `[855, 904, 1257]`;
///   * block 3: a single colour `0.0`, xor sum `21655`, non-zero count `10`.
///
/// **What this pins beyond the sums.** Block 0's second colour, 170.81…, gives an upper bound
/// of 200.81… which `cvRound` takes to **201** while a truncating conversion takes it to 200,
/// and the resulting masks differ -- so this real-data row, not only the synthetic
/// `in_range` table, is sensitive to the rounding mode. The candidates are asserted **in
/// colour order**, which is the order A3's merge consumes.
///
/// **Anti-vacuity.** Block 3's single 10-pixel candidate is the smallest thing here and is
/// still non-empty; 855/904/1257 cannot be derived from the tree.
#[test]
fn annotation_top_k_mask_list_on_the_recorded_page_matches_the_upstream_oracle() {
    let (crop, mask_crop) = recorded_crop(
        Rect::new(674, 1397, 740, 1438),
        Rect::new(671, 1394, 743, 1441),
    );
    let grey = pc_detect::annotate::rgb_to_gray(&crop);
    let list = top_k_mask_list(&grey, &mask_crop).expect("equal sizes");
    assert_eq!(
        scores(&list),
        vec![177_954, 171_579, 178_942],
        "one candidate per top-k colour, in colour order"
    );
    assert_eq!(
        list.iter()
            .map(|c| nonzero_count(&c.mask))
            .collect::<Vec<_>>(),
        vec![855, 904, 1257]
    );

    let (crop, mask_crop) = recorded_crop(
        Rect::new(438, 1407, 498, 1446),
        Rect::new(435, 1404, 501, 1449),
    );
    let grey = pc_detect::annotate::rgb_to_gray(&crop);
    let list = top_k_mask_list(&grey, &mask_crop).expect("equal sizes");
    assert_eq!(scores(&list), vec![21_655], "a single colour on this block");
    assert_eq!(nonzero_count(&list[0].mask), 10);
}

// ------------------------------------------------------------------- helpers

/// One row of [`annotation_otsu_path_on_the_recorded_page_matches_the_upstream_oracle`]'s
/// oracle table. A named struct rather than a tuple so that the numeric columns cannot be
/// transposed unnoticed.
#[derive(Clone, Copy)]
struct OtsuOracleRow {
    /// The block as committed in `…_detector_blocks.json`.
    rect: Rect,
    /// `expand_text_window(rect, (1200, 1660), 16)` -- asserted independently by A1.
    window: Rect,
    /// Otsu thresholds in **upstream's** channel order: B, G, R.
    thresholds_bgr: [u8; 3],
    /// The winning candidate's `xor_sum`.
    xor: u64,
    /// The winning candidate's non-zero pixel count.
    nonzero: usize,
    /// The winning candidate's positional fingerprint, see [`positional_fingerprint`].
    fingerprint: u64,
}

fn row_image(values: &[u8]) -> GrayImage {
    GrayImage::from_fn(values.len() as u32, 1, |x, _| Luma([values[x as usize]]))
}

fn row_values(image: &GrayImage) -> Vec<u8> {
    image.pixels().map(|p| p.0[0]).collect()
}

fn pixels(image: &GrayImage) -> Vec<u8> {
    image.pixels().map(|p| p.0[0]).collect()
}

fn nonzero_count(image: &GrayImage) -> usize {
    image.pixels().filter(|p| p.0[0] != 0).count()
}

/// `sum of (row-major index + 1)` over non-zero pixels -- a position-sensitive fingerprint,
/// computed the same way by the Python oracle (`np.nonzero(mask.ravel())[0] + 1`, summed).
/// The `+ 1` is what makes pixel (0, 0) contribute.
fn positional_fingerprint(image: &GrayImage) -> u64 {
    image
        .pixels()
        .enumerate()
        .filter(|(_, p)| p.0[0] != 0)
        .map(|(index, _)| index as u64 + 1)
        .sum()
}

fn scores(list: &[MaskCandidate]) -> Vec<u64> {
    list.iter().map(|c| c.xor_sum).collect()
}

fn channel(image: &RgbImage, index: usize) -> GrayImage {
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        Luma([image.get_pixel(x, y).0[index]])
    })
}

enum Half {
    Left,
    Right,
    Top,
}

/// A 4x4 mask with one half at 255 -- 8 pixels on, 8 off.
fn half_mask(half: Half) -> GrayImage {
    GrayImage::from_fn(4, 4, |x, y| {
        let on = match half {
            Half::Left => x < 2,
            Half::Right => x >= 2,
            Half::Top => y < 2,
        };
        Luma([if on { 255 } else { 0 }])
    })
}

/// The constructed channel-tie input: `(bgr image, mask, B's candidate mask, R's candidate
/// mask)`. Every channel holds only 0 and 200, so its Otsu threshold is 0 and its
/// binarisation is exactly the 200-set (asserted independently by
/// [`otsu_threshold_matches_opencv_threshold_otsu_on_oracle_arrays`]'s two-cluster rows).
fn channel_tie_case() -> (RgbImage, GrayImage, GrayImage, GrayImage) {
    // mask: left half on.
    let mask = half_mask(Half::Left);
    let base = |x: u32| x < 2;
    // B differs from the mask at (0,0), (2,0), (3,0); R at (0,1), (2,1), (3,1);
    // G at (0,2), (1,2), (2,2), (3,2), (2,3) -- 3, 3 and 5 pixels.
    let blue_on = |x: u32, y: u32| match (x, y) {
        (0, 0) => false,
        (2, 0) | (3, 0) => true,
        _ => base(x),
    };
    let red_on = |x: u32, y: u32| match (x, y) {
        (0, 1) => false,
        (2, 1) | (3, 1) => true,
        _ => base(x),
    };
    let green_on = |x: u32, y: u32| match (x, y) {
        (0, 2) | (1, 2) => false,
        (2, 2) | (3, 2) | (2, 3) => true,
        _ => base(x),
    };

    let image = RgbImage::from_fn(4, 4, |x, y| {
        let v = |on: bool| if on { 200 } else { 0 };
        Rgb([v(red_on(x, y)), v(green_on(x, y)), v(blue_on(x, y))])
    });
    let to_mask = |f: &dyn Fn(u32, u32) -> bool| {
        GrayImage::from_fn(4, 4, |x, y| Luma([if f(x, y) { 255 } else { 0 }]))
    };
    (image, mask, to_mask(&blue_on), to_mask(&red_on))
}

/// The committed page and unrefined detector mask, cropped to `window` -- upstream's
/// `im = img[by1:by2, bx1:bx2]` / `msk = pred_mask[by1:by2, bx1:bx2]`, i.e. the far edge is
/// **exclusive**, which is what `Rect::width()`/`height()` already give. `rect` is passed only
/// so the assertion messages name the block.
fn recorded_crop(rect: Rect, window: Rect) -> (RgbImage, GrayImage) {
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
