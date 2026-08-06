//! Task A3b -- spec §16.37 item 11: *"`refine_mask`'s page-level driver plus
//! `refine_undetected_mask`"*, ported from upstream PanelCleaner's
//! `comic_text_detector/utils/textmask.py` (pinned commit
//! `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`): `refine_mask` at `:195-212` and
//! `refine_undetected_mask` at `:161-192`, plus `imgproc_utils.py::union_area` at `:15-22`.
//!
//! **This module is not wired into anything.** `pc_detect::run` still rejects
//! `MaskRefineMode::Annotation` and `d7_run.rs::run_rejects_annotation_refine_mode` still
//! passes. Wiring is **A4**, together with the config gate, §16.37 item 6's coverage decision,
//! item 7's clause amendments and the `DEVIATION(12)` retirement. None of that is here.
//!
//! **Why a third file rather than more lines in [`crate::annotate`].** §14 items 23 and 24 and
//! §16.37 items 2 and 10 cite six live line numbers inside `annotate.rs`; growing that file
//! shifts them and forces a spec correction, which §16.37 item 2 has already had to make once
//! for exactly that reason. [`crate::annotate_merge`]'s header records the same reasoning. The
//! whole A1-A3b "Annotation refine" family stays in one crate directory.
//!
//! # What this module covers, in upstream's own order
//!
//! **Part 1 -- [`refine_mask`], the page-level driver (`textmask.py:195-212`).** Per block:
//! expand the window with A1's [`crate::annotate::expand_text_window`], crop image and pred mask
//! to it, build the candidate list with A2's [`crate::annotate::candidate_mask_list`], merge with
//! A3's [`crate::annotate_merge::merge_mask_list`], and **OR** the result into a page-sized mask
//! at the window's coordinates.
//!
//! **Part 2 -- [`refine_undetected_mask`] (`textmask.py:161-192`)**, whose five load-bearing
//! details are enumerated by §16.37 item 11(b) and implemented as
//! [`zero_pred_where_refined`] (i), [`undetected_blocks`] (ii)-(iv) and
//! [`refine_undetected_mask`] itself (v).
//!
//! # Five readings of upstream that decide this port
//!
//! **(a) The pred mask is mutated in place, and that is a DECISION -- §16.37 item 11(b)(i)
//! required one to be made and recorded.** [`refine_undetected_mask`] takes
//! `pred_mask: &mut GrayImage`, mirroring upstream's `mask_pred[np.where(mask_refined > 30)] = 0`
//! writing through the caller's own array (verified by probe, 2026-08-07: a 40x40 pred of
//! uniform 200 with a 5x5 refined patch comes back with 25 pixels zeroed in the caller's array).
//! Grounds: the mutated array is what upstream's recursive `refine_mask` call consumes, so the
//! zeroing is not optional to the output; upstream returns that same array to *its* caller at
//! `inference.py:210`, so the effect is part of the observable contract; and Rust cannot
//! alias-mutate silently, so an `&mut` parameter is the only form that shows the side effect at
//! every call site. The rejected alternative -- copy internally, take `&GrayImage` -- is
//! output-equivalent *in PanelCleaner's own pipeline*, which never reads the mutated array
//! (`ctd_interface.py:182` writes only `mask_refined`); "equivalent for today's single caller"
//! is the reasoning §16.24 item 1(a) is on record as having got wrong. [`refine_mask`] by
//! contrast takes `&GrayImage`: upstream's driver does **not** mutate its argument (measured on
//! the committed page: 0 pixels differ afterwards).
//!
//! **(b) `valid_labels[1:]` drops the FIRST SURVIVING ENTRY, not the background label.** §16.37
//! item 11(b)(iii), quoted: *"this is NOT 'skip the background label'. It drops the **first
//! surviving entry** of the area-filtered list, which coincides with label 0 only when the
//! background's own area also exceeds 50."* [`undetected_blocks`] implements the quirk, not the
//! intent. Because the selection is by **position in the label sequence**, that sequence has to
//! be `cv2`'s: [`undetected_blocks`] iterates
//! [`ConnectedComponents::labels_in_opencv_block_scan_order`] rather than ascending
//! [`connected_components`] labels (spec §16.37 item 13). See that function's doc comment.
//!
//! [`ConnectedComponents::labels_in_opencv_block_scan_order`]: crate::annotate_merge::ConnectedComponents::labels_in_opencv_block_scan_order
//!
//! **(c) Connectivity is the EFFECTIVE 8, not the literal 4 (§16.37 item 11(c), which also rules
//! that no `DEVIATION` is needed).** Upstream writes
//! `cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)`, whose positionals bind to the
//! `labels`/`stats` **output** slots, so `connectivity` keeps its default 8 and `ltype` keeps
//! `CV_32S`. Upstream *intends* 4 here and *runs* 8 -- a latent upstream defect, recorded rather
//! than reproduced-as-intended, on the `DEVIATION(29)` precedent: match what upstream does, not
//! what it says. **Citation note:** the Otsu entry this cites is `DEVIATION(29)`, renumbered from
//! 24 on 2026-08-07 to resolve the `lama-inpaint` cross-branch collision recorded in the gap note
//! above §14's item 21. §16.37 item 11(c) spells it `DEVIATION(24)` at commit `dacd666` and
//! earlier, and **29** from the renumber onward; this file uses 29, which is the live identifier.
//! Only the identifier moved -- the precedent itself is unchanged.
//!
//! This module simply calls [`crate::annotate_merge::connected_components`], which is 8-connected.
//!
//! **(d) `union_area` is misnamed and returns a `-1` sentinel.** It computes the **intersection**
//! area, and `-1` when the boxes are disjoint. Ported here as [`intersection_area`] -- under its
//! own name, per §16.37 item 11(b)(iv).
//!
//! **(e) `refine_mode` is not a parameter of this port.** Upstream threads `refine_mode` through
//! both functions, but PanelCleaner's own pipeline passes `REFINEMASK_ANNOTATION`
//! (`ctd_interface.py:161-162`) and A3 did not implement the `REFINEMASK_INPAINT` dilation
//! branch -- see [`crate::annotate_merge`]'s header note (c), which measures what that branch
//! costs. So there is nothing for a mode argument to select.
//!
//! # One error-classification choice, declared rather than inferred
//!
//! Every failure here is `StageError::InvalidInput`, which is **per-image** (cookbook rule 4):
//! one malformed block or mismatched buffer must not abort a run. Two of them have no upstream
//! counterpart at all -- upstream *raises*, and a Python raise takes the whole page down:
//!   * an **empty expanded window** reaches `cv2.cvtColor` on a zero-sized array and dies with
//!     `(-215:Assertion failed)` at `modules/imgproc/src/color.cpp:199` (measured 2026-08-07);
//!   * mismatched shapes fail inside numpy indexing.
//! And [`refine_undetected_mask`] validates **before** mutating, so a rejected call leaves the
//! caller's pred mask untouched. Upstream zeroes first and raises afterwards. That ordering is
//! this crate's choice, stated so it is not mistaken for parity; it is unobservable to a caller
//! that discards the buffer on error, which is the only use this crate has.

use crate::annotate::{
    candidate_mask_list, expand_text_window, threshold_binary, ANNOTATION_EXPAND_R,
};
use crate::annotate_merge::{connected_components, merge_mask_list};
use image::{GrayImage, Luma, RgbImage};
use pc_core::{Rect, StageError};

/// The threshold in upstream's `mask_pred[np.where(mask_refined > 30)] = 0`
/// (`textmask.py:169`). **Strict**: a refined pixel of exactly 30 does not zero the pred pixel.
pub const REFINED_ZERO_THRESHOLD: u8 = 30;

/// The threshold in upstream's `cv2.threshold(mask_pred, 30, 255, THRESH_BINARY)`
/// (`textmask.py:170`). **Strict**: a pred pixel of exactly 30 is off.
///
/// Numerically equal to [`REFINED_ZERO_THRESHOLD`] and kept separate on purpose -- they are two
/// unrelated thresholds in upstream, applied to two different arrays, and collapsing them into
/// one constant would make a future change to either silently change both.
pub const UNDETECTED_PRED_THRESHOLD: u8 = 30;

/// Upstream's `np.where(stats[:, -1] > 50)[0]` (`textmask.py:174`). A **pixel count**
/// (`CC_STAT_AREA`), not a bounding-box area, and the comparison is **strict**.
pub const UNDETECTED_MIN_COMPONENT_AREA: u32 = 50;

/// Upstream's `if bbox_score / w / h < 0.5` (`textmask.py:186`). **Strict**, so a component
/// exactly half-covered by some detected block is **not** invented.
pub const UNDETECTED_MAX_INTERSECTION_RATIO: f64 = 0.5;

/// The value upstream's `union_area` returns for disjoint boxes (`imgproc_utils.py:20`).
///
/// It is a sentinel, not an area: a component overlapping nothing scores `-1 / w / h`, which is
/// below [`UNDETECTED_MAX_INTERSECTION_RATIO`], and is therefore invented. It is also
/// `bbox_score`'s initialiser, which is what makes an empty `blk_list` invent everything.
pub const DISJOINT_SENTINEL: i64 = -1;

// ============================================================ Part 1: the page-level driver

/// `refine_mask(img, pred_mask, blk_list, refine_mode=REFINEMASK_ANNOTATION)` -- upstream
/// `textmask.py:195-212`.
///
/// ```text
/// mask_refined = np.zeros_like(pred_mask)
/// for blk in blk_list:
///     bx1, by1, bx2, by2 = expand_textwindow(img.shape, blk.xyxy, expand_r=16)
///     im  = np.ascontiguousarray(img[by1:by2, bx1:bx2])
///     msk = np.ascontiguousarray(pred_mask[by1:by2, bx1:bx2])
///     mask_list  = get_topk_masklist(im, msk)
///     mask_list += get_otsuthresh_masklist(im, msk, per_channel=False)
///     mask_merged = merge_mask_list(mask_list, msk, ...)
///     mask_refined[by1:by2, bx1:bx2] = cv2.bitwise_or(mask_refined[by1:by2, bx1:bx2], mask_merged)
/// return mask_refined
/// ```
///
/// Three details that are easy to get wrong and are each pinned by an A3b test:
///   * the window comes from A1's [`crate::annotate::expand_text_window`] -- a **divisor**, with
///     the far edges clamped to `im_w - 1` / `im_h - 1` (§16.37 item 5). `Rect::pad`'s clamp is
///     off by one and would make the last page column writable;
///   * the crop is `[by1:by2, bx1:bx2]`, i.e. `x2`/`y2` **exclusive** -- which is `Rect`'s own
///     convention for cropping (`pc-core/src/geometry.rs`'s header);
///   * the write is an **OR**, not an assignment. Two blocks' expanded windows can overlap, and
///     an assignment erases the earlier block's pixels there. The A3b test measures 50 such
///     pixels on a 44x24 fixture.
///
/// The returned mask is the size of `pred_mask` (upstream's `np.zeros_like(pred_mask)`), which is
/// also the image's size -- the two are required to agree.
///
/// Errors (`StageError::InvalidInput`, **per-image**, cookbook rule 4) when `image` and
/// `pred_mask` differ in size, or when some block's expanded window is empty. See the module
/// header's error-classification note: upstream raises in both cases.
pub fn refine_mask(
    image: &RgbImage,
    pred_mask: &GrayImage,
    blocks: &[Rect],
) -> Result<GrayImage, StageError> {
    let (width, height) = pred_mask.dimensions();
    if image.dimensions() != (width, height) {
        return Err(StageError::InvalidInput(format!(
            "image {:?} and pred mask {:?} must have the same size",
            image.dimensions(),
            pred_mask.dimensions()
        )));
    }

    let mut refined = GrayImage::new(width, height);
    for &block in blocks {
        let window = expand_text_window(block, (width, height), ANNOTATION_EXPAND_R);
        if window.width() <= 0 || window.height() <= 0 {
            return Err(StageError::InvalidInput(format!(
                "block {block:?} expands to the empty window {window:?} on a {width}x{height} page"
            )));
        }
        // `expand_text_window` clamps `x1`/`y1` up to 0 and `x2`/`y2` down to `width - 1` /
        // `height - 1`, so a window with positive extent is necessarily in bounds.
        let (x0, y0) = (window.x1 as u32, window.y1 as u32);
        let (crop_w, crop_h) = (window.width() as u32, window.height() as u32);

        let crop = image::imageops::crop_imm(image, x0, y0, crop_w, crop_h).to_image();
        let mask_crop = image::imageops::crop_imm(pred_mask, x0, y0, crop_w, crop_h).to_image();

        let candidates = candidate_mask_list(&crop, &mask_crop)?;
        let merged = merge_mask_list(candidates, &mask_crop)?;

        // `cv2.bitwise_or(mask_refined[bbox], mask_merged)` -- a real bitwise OR, not a
        // set-if-non-zero. Both operands are 0/255 here, so the two coincide; the bitwise form is
        // written because it is what upstream does.
        for y in 0..crop_h {
            for x in 0..crop_w {
                let existing = refined.get_pixel(x0 + x, y0 + y).0[0];
                let incoming = merged.get_pixel(x, y).0[0];
                refined.put_pixel(x0 + x, y0 + y, Luma([existing | incoming]));
            }
        }
    }
    Ok(refined)
}

// ==================================================== Part 2 (iv): the intersection scorer

/// Upstream's `union_area(bboxa, bboxb)` (`imgproc_utils.py:15-22`), ported under an honest name.
///
/// ```text
/// x1 = max(bboxa[0], bboxb[0]); y1 = max(bboxa[1], bboxb[1])
/// x2 = min(bboxa[2], bboxb[2]); y2 = min(bboxa[3], bboxb[3])
/// if y2 < y1 or x2 < x1: return -1
/// return (y2 - y1) * (x2 - x1)
/// ```
///
/// **It is the INTERSECTION area, despite the upstream name** (§16.37 item 11(b)(iv): *"Port the
/// behaviour under its own name, not under its upstream name"*), and it returns
/// [`DISJOINT_SENTINEL`] rather than 0 when the boxes do not meet.
///
/// The guard is **strict**, so boxes sharing exactly one edge are *not* disjoint: they yield a
/// zero-width or zero-height intersection and the result is `0`, not `-1`. Writing the guard
/// with `<=` would change that, and a test pins it.
///
/// Widened to `i64` because a page-sized intersection overflows `i32` on large canvases; upstream
/// is a Python `int` and unbounded.
#[must_use]
pub fn intersection_area(a: Rect, b: Rect) -> i64 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    if y2 < y1 || x2 < x1 {
        return DISJOINT_SENTINEL;
    }
    i64::from(y2 - y1) * i64::from(x2 - x1)
}

// ============================== Part 2 (ii)-(iv): which residual components become new blocks

/// Upstream `textmask.py:170-187` -- the threshold, the labelling, the `valid_labels[1:]` area
/// filter and the intersection test, returning the boxes `refine_undetected_mask` would hand to
/// [`refine_mask`].
///
/// `pred_mask` must already be the **zeroed residual** -- the caller runs
/// [`zero_pred_where_refined`] first. It is a separate parameter rather than an internal step so
/// that (ii)-(iv) are testable without going through (i) and (v).
///
/// The returned boxes are `[x, y, x + w, y + h]` from each component's `stats` row, matching
/// upstream's `bbox = [bx1, by1, bx2, by2]` with `bx2 = x + w` -- `Rect`'s exclusive-`x2`
/// cropping convention, so they can be passed straight to [`refine_mask`].
///
/// # The `valid_labels[1:]` quirk, ported and not fixed
///
/// §16.37 item 11(b)(iii): *"this is **NOT** 'skip the background label'. It drops the **first
/// surviving entry** of the area-filtered list, which coincides with label 0 only when the
/// background's own area also exceeds 50 ... A port that writes 'skip label 0' is a different
/// function."* On an 8x8 all-foreground image with one background pixel the areas are `[1, 63]`,
/// `valid_labels == [1]` and `valid_labels[1:] == []`, so the only real component is silently
/// dropped.
///
/// # Label ORDER: this function is at parity with upstream, and there is no divergence to register
///
/// §16.37 item 12 imposes the obligation this paragraph discharges: *"each new consumer of the
/// label numbering re-establishes order-independence for itself or registers the divergence it
/// has."* This consumer **cannot** be order-independent -- `valid_labels[1:]` selects by
/// **position in the label sequence**, so it drops whichever area-filtered entry sorts first --
/// so the obligation is met the other way: by taking the sequence in **upstream's own order**.
/// The `valid` list below is built from
/// [`ConnectedComponents::labels_in_opencv_block_scan_order`], not from ascending
/// [`connected_components`] labels. Both the algorithm (§16.37 item 11(c)'s effective
/// connectivity 8) and the order now match upstream, so **no `DEVIATION` entry exists or is
/// needed** -- ratified as §16.37 item 13.
///
/// **What the divergence WAS, kept because it is the reason the accessor exists.** Before that
/// change this function iterated first-raster-pixel labels, which diverges from `cv2` whenever
/// the background's own area is at most [`UNDETECTED_MIN_COMPONENT_AREA`] -- i.e. when the
/// thresholded residual leaves at most 50 background pixels page-wide, so that `valid[0]` is a
/// real component rather than label 0. Measured 2026-08-07 on a 20x20 fixture meeting that
/// condition (background 43 px, components of 300 and 57 px): `cv2` invents `[5, 0, 20, 20]`
/// while first-raster-pixel order invented `[0, 1, 3, 20]`. That fixture is now the parity test
/// `undetected_blocks_matches_upstreams_block_scan_label_order_when_the_background_is_small`, and
/// it asserts `cv2`'s answer. Above the 50-pixel threshold the two orders were already
/// indistinguishable here, which is why the committed page -- residual background area
/// **1 987 219** -- never showed the divergence.
///
/// [`ConnectedComponents::labels_in_opencv_block_scan_order`]: crate::annotate_merge::ConnectedComponents::labels_in_opencv_block_scan_order
#[must_use]
pub fn undetected_blocks(pred_mask: &GrayImage, blocks: &[Rect]) -> Vec<Rect> {
    let thresholded = threshold_binary(pred_mask, UNDETECTED_PRED_THRESHOLD);
    let components = connected_components(&thresholded);

    // `valid_labels = np.where(stats[:, -1] > 50)[0]` -- `cv2`'s own label order, label 0 first.
    //
    // The order is `cv2`'s, not ours: `valid_labels[1:]` selects by POSITION, so the sequence
    // this filter runs over has to be the sequence upstream's `stats` rows arrive in. Label 0 is
    // prepended explicitly because the accessor returns non-background labels only, and `cv2`
    // always numbers the background 0 -- so it is always the first row, ahead of every block.
    let valid: Vec<u32> = std::iter::once(0)
        .chain(components.labels_in_opencv_block_scan_order())
        .filter(|&label| components.stats[label as usize].area > UNDETECTED_MIN_COMPONENT_AREA)
        .collect();

    let mut invented = Vec::new();
    // `for lab_index in valid_labels[1:]` -- drops the first SURVIVING entry. Not label 0.
    for &label in valid.iter().skip(1) {
        let stat = components.stats[label as usize];
        let bbox = Rect::new(
            stat.x as i32,
            stat.y as i32,
            (stat.x + stat.width) as i32,
            (stat.y + stat.height) as i32,
        );

        // `bbox_score = -1` then `if bbox_s > bbox_score` over every block: the maximum, with the
        // sentinel as the floor. Taking a minimum, or the first score, changes the decision.
        let mut score = DISJOINT_SENTINEL;
        for &block in blocks {
            score = score.max(intersection_area(block, bbox));
        }

        // `if bbox_score / w / h < 0.5` -- divided by `w` then by `h`, in that order, as upstream
        // writes it. `w` and `h` are the component's box extents, never zero for a label that
        // exists.
        let ratio = score as f64 / f64::from(stat.width) / f64::from(stat.height);
        if ratio < UNDETECTED_MAX_INTERSECTION_RATIO {
            invented.push(bbox);
        }
    }
    invented
}

// ============================================ Part 2 (i) and (v): refine_undetected_mask

/// Upstream's `mask_pred[np.where(mask_refined > 30)] = 0` (`textmask.py:169`) -- step (i) of
/// §16.37 item 11(b).
///
/// **Mutates `pred_mask` in place**, which is upstream's own observable behaviour; the module
/// header records why this port reproduces it rather than copying. The threshold is **strict**,
/// so a refined pixel of exactly [`REFINED_ZERO_THRESHOLD`] leaves the pred pixel alone.
///
/// Errors (`StageError::InvalidInput`, **per-image**) on a size mismatch, and does not mutate
/// anything in that case.
pub fn zero_pred_where_refined(
    pred_mask: &mut GrayImage,
    refined: &GrayImage,
) -> Result<(), StageError> {
    if pred_mask.dimensions() != refined.dimensions() {
        return Err(StageError::InvalidInput(format!(
            "pred mask {:?} and refined mask {:?} must have the same size",
            pred_mask.dimensions(),
            refined.dimensions()
        )));
    }
    let (width, height) = pred_mask.dimensions();
    for y in 0..height {
        for x in 0..width {
            if refined.get_pixel(x, y).0[0] > REFINED_ZERO_THRESHOLD {
                pred_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }
    Ok(())
}

/// `refine_undetected_mask(img, mask_pred, mask_refined, blk_list, refine_mode)` -- upstream
/// `textmask.py:161-192`.
///
/// The five steps of §16.37 item 11(b), in upstream's order:
///   1. [`zero_pred_where_refined`] -- the in-place zeroing of the **caller's** `pred_mask`;
///   2. -- 4. [`undetected_blocks`] -- threshold, 8-connected labelling, the `valid_labels[1:]`
///      area filter and the [`intersection_area`] test;
///   5. [`refine_mask`] over the invented blocks, on the **already-zeroed** `pred_mask`,
///      bitwise-OR-ed into `refined`.
///
/// `refined` is **not** mutated: upstream rebinds `mask_refined = cv2.bitwise_or(...)`, which
/// allocates, so the caller's refined mask is left as it was and the result is returned. When
/// nothing is invented the return value is a clone of `refined`, matching upstream's untaken
/// `if len(seg_blk_list) > 0` branch.
///
/// Errors (`StageError::InvalidInput`, **per-image**, cookbook rule 4) when the three buffers do
/// not all have the same size, or when an invented block's expanded window is empty. Validation
/// happens **before** the mutation in step 1 -- see the module header; that ordering is this
/// crate's choice, not upstream's.
pub fn refine_undetected_mask(
    image: &RgbImage,
    pred_mask: &mut GrayImage,
    refined: &GrayImage,
    blocks: &[Rect],
) -> Result<GrayImage, StageError> {
    let size = pred_mask.dimensions();
    if image.dimensions() != size || refined.dimensions() != size {
        return Err(StageError::InvalidInput(format!(
            "image {:?}, pred mask {:?} and refined mask {:?} must all have the same size",
            image.dimensions(),
            size,
            refined.dimensions()
        )));
    }

    zero_pred_where_refined(pred_mask, refined)?;
    let invented = undetected_blocks(pred_mask, blocks);
    if invented.is_empty() {
        return Ok(refined.clone());
    }

    let extra = refine_mask(image, pred_mask, &invented)?;
    let (width, height) = size;
    Ok(GrayImage::from_fn(width, height, |x, y| {
        Luma([refined.get_pixel(x, y).0[0] | extra.get_pixel(x, y).0[0]])
    }))
}
