//! Task A3 -- spec §16.37 item 8: *"**A3** (**heavy**): connected components plus the merge
//! and hole-filling loops."* A port of upstream PanelCleaner's
//! `comic_text_detector/utils/textmask.py::merge_mask_list` (pinned commit
//! `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, `:83-158`).
//!
//! **Why this is a sibling module and not more lines in [`crate::annotate`].** §14 items 23
//! and 24 and §16.37 items 2 and 10 cite six live line numbers inside `annotate.rs`
//! (`:278`, `:295`, `:410`, `:449`, `:558`, `:600`). Extending that file's module doc comment
//! -- the natural place to describe A3 -- would shift all six and require a spec correction,
//! which §16.37 item 2 has already had to make once for exactly this reason. A1's and A2's
//! functions are used here by path, so the family stays in one crate directory with no
//! line-number churn.
//!
//! **This module is not wired into anything.** `pc_detect::run` still rejects
//! `MaskRefineMode::Annotation` and `d7_run.rs::run_rejects_annotation_refine_mode` still
//! passes. **Corrected 2026-08-07 by spec §16.37 item 11:** this sentence read *"Wiring --
//! upstream's `refine_mask` per-block driver, the config gate, and §16.37 item 6's coverage
//! decision -- is task A4"*, which assigned the per-block driver to the wrong task. The driver
//! belongs to **A3b**, together with `refine_undetected_mask` -- it is numerical work with an
//! upstream oracle, and `refine_undetected_mask` calls into it recursively, so it cannot wait
//! for A4's integration pass. **A4** is the config gate, §16.37 item 6's coverage decision,
//! item 7's clause amendments and the `DEVIATION(12)` retirement.
//!
//! What A3 covers, in upstream's own order inside `merge_mask_list`:
//!   1. [`sort_candidates_by_xor_sum`] -- `mask_list.sort(key=lambda x: x[1])` (`:92`), the
//!      **second** stable sort, over `candidate_mask_list`'s concatenated output;
//!   2. [`binarize_pred_mask`] -- `cv2.erode(pred_mask, ELLIPSE_3x3)` then
//!      `cv2.threshold(pred_mask, 60, 255, THRESH_BINARY)` (`:103-109`), built on
//!      [`erode_ellipse3x3`];
//!   3. [`connected_components`] -- `cv2.connectedComponentsWithStats` (`:113`, `:137`);
//!   4. [`merge_components_by_xor`] -- the accept-a-component-if-it-lowers-the-XOR loop
//!      (`:116-132`), shared verbatim by upstream's merge and hole-fill loops;
//!   5. [`hole_fill_area_threshold`] + [`merge_mask_list`] -- the hole-filling pass
//!      (`:137-157`).
//!
//! # Four upstream readings that are measured, not inferred from the source text
//!
//! **(a) The `connectivity` and `ltype` arguments upstream passes are silently discarded.**
//! Upstream writes `cv2.connectedComponentsWithStats(candidate_mask, connectivity, cv2.CV_16U)`
//! with `connectivity = 8`. The Python binding's signature is
//! `connectedComponentsWithStats(image[, labels[, stats[, centroids[, connectivity[, ltype]]]]])`
//! -- so those two positionals bind to the **`labels` and `stats` output slots**, are ignored,
//! and `connectivity` keeps its default **8** while `ltype` keeps **`CV_32S`**. Measured on
//! opencv 5.0.0 (the version `tests/fixtures/recorded/detector/PROVENANCE.json` pins) on
//! 2026-08-07: on the array `[[0,0,0,0,255],[0,255,255,0,0],[255,0,255,255,0]]`, the positional
//! calls `(m, 4, CV_16U)` and `(m, 8, CV_16U)` both return `num_labels = 3` and are
//! byte-identical to `connectivity=8`, while `connectivity=4` returns `num_labels = 4`; and the
//! returned `labels` array has dtype `int32`, not `uint16`.
//!
//! Two consequences. For `merge_mask_list` there is **none**: 8 was the intended value, so this
//! port is 8-connected and faithful either way, and `CV_16U`'s 65 535-label ceiling never
//! applies (this module's labels are `u32`). For `refine_undetected_mask` (`:170`) the intended
//! value was **4** and the effective value is **8** -- an upstream latent defect. That function
//! is **outside A3**; the finding was reported upward and is now ratified as spec §16.37 item
//! 11(c), which assigns the function to task A3b and rules that the **effective 8** is what
//! gets ported, with no `DEVIATION` needed (the `DEVIATION(29)` precedent: match what upstream
//! does, not what it says).
//!
//! **(b) The `MORPH_ELLIPSE` 3x3 kernel is a plus, not a square.** `cv2.getStructuringElement(
//! cv2.MORPH_ELLIPSE, (3, 3), (1, 1))` returns `[[0,1,0],[1,1,1],[0,1,0]]` (measured, same run).
//! So [`erode_ellipse3x3`] is a 4-neighbourhood minimum, **not** the 8-neighbourhood minimum
//! [`crate::annotate::erode_rect3x3`] implements for `np.ones((3, 3))`. The two differ on real
//! inputs and confusing them is a silent behaviour change; a test pins both against `cv2`.
//!
//! **(c) `refine_mode == REFINEMASK_INPAINT`'s dilation is not ported.** `:134-135` dilates
//! `mask_merged` by a 3x3 square when `refine_mode` is `REFINEMASK_INPAINT`. PanelCleaner's own
//! pipeline never takes that branch: `pcleaner/ctd_interface.py:161-162` calls the detector with
//! `refine_mode=REFINEMASK_ANNOTATION`, and `REFINEMASK_ANNOTATION = 1` while
//! `REFINEMASK_INPAINT = 0`. Same precedent as A2's untaken `dilate=True` branch in
//! `minxor_thresh`. The branch is materially different rather than cosmetic -- measured on the
//! committed page's four blocks, the two branches give merged populations
//! `1092/817/1033/0` (annotation) against `1609/1801/2267/0` (inpaint) -- so it is stated here
//! rather than left implicit.
//!
//! **(d) Upstream's `linemask` is dead.** `:93-101` builds `linemask` under
//! `blk is not None and filter_with_lines`, and then never reads it. `refine_mask` (`:208-210`)
//! never passes `filter_with_lines`, whose default is `False`, so the block does not even run.
//! Nothing to port.
//!
//! # Component visit order
//!
//! [`connected_components`] numbers labels in ascending order of each component's **first
//! raster pixel**. That is **not** OpenCV's order: `cv2`'s default 8-way algorithm is
//! block-based (Grana/BBDT) and numbers by 2x2-block scan order, which differs -- measured on
//! `[[0,255,0,0,0,255,255,255,255],[0,0,0,255,0,255,255,255,0]]`, where `cv2` gives the
//! single pixel at `(1, 3)` label 2 and the right-hand blob label 3, while first-raster-pixel
//! order gives the blob 2 and the pixel 3.
//!
//! **No `DEVIATION` is registered for that, because the merged mask does not depend on the
//! order at all -- and that is a proof, not a measurement.** Scope, stated before the proof
//! because the conclusion must not travel past it: what follows is about **`merge_mask_list`'s
//! two loops only**, and is ratified at that scope by spec §16.37 item 12. Let `merged` be the
//! running mask and let the labels of one [`ConnectedComponents`] be visited in any order.
//! Components with distinct labels are pixel-disjoint, and every pixel of a component lies
//! inside its own bounding box. Visiting label `L` writes
//! `merged[bbox] = merged[bbox] | indicator(L)`, which turns on exactly the pixels of `L` that
//! were off and changes nothing else -- in particular it never touches a pixel of any other
//! label. So the set `{p in L : merged(p) == 0}` at visit time equals the same set taken at
//! loop entry, and `L`'s *effect* is a function of the loop-entry state alone.
//!
//! `L`'s *decision* needs one further step, and omitting it left this proof incomplete until
//! 2026-08-07: the comparison [`merge_components_by_xor`] makes is between two sums taken over
//! the **whole bounding box**, which in general contains pixels of other labels -- pixels that
//! earlier visits in this loop may well have mutated. Those pixels do not have to be
//! unmutated for the argument to work, only *equal on both sides*: at a pixel `p` outside `L`,
//! `candidate(p) == origin(p)` by the write rule above, whatever `origin(p)` has become. So
//! `candidate(p) ^ pred(p)` and `origin(p) ^ pred(p)` are bit-identical, those terms contribute
//! the same amount to `xor_merged` and to `xor_origin`, and they cancel out of the strict
//! inequality `xor_merged < xor_origin`. What survives the cancellation is exactly the
//! contribution of `L`'s own pixels -- which the previous paragraph fixed to the loop-entry
//! state. Hence the decision, like the effect, depends only on the loop-entry state, and the
//! final mask is order-independent. ∎ (Label 0 in the hole-fill pass is consistent with this:
//! its pixels are exactly those where `merged == 255` at entry, so its delta is zero and it is
//! never accepted.)
//!
//! **The scope is load-bearing, not a hedge.** [`ConnectedComponents`] is `pub` with `pub`
//! `labels` and `stats`, so a *different* consumer of the label numbering can be
//! order-sensitive where these two loops are not -- upstream's `refine_undetected_mask` filters
//! by area and then iterates `valid_labels[1:]`, dropping whichever surviving entry sorts
//! first, which is a function of the numbering. Spec §16.37 item 11 assigns that function to
//! task A3b, and §16.37 item 12 records that this proof does **not** cover it. **A3b discharged
//! that obligation by re-establishing parity rather than by registering a divergence** (§16.37
//! item 13): it takes the sequence from `ConnectedComponents::labels_in_opencv_block_scan_order`,
//! which reorders the same components into `cv2`'s order without changing this function's
//! numbering, its `labels`, or its `stats`.
//!
//! The proof's one precondition -- that distinct labels are pixel-disjoint -- is a property of
//! a labeling, so [`merge_components_by_xor`] documents it as a caller obligation rather than
//! assuming it silently.
//!
//! §16.37 item 8's independent measurement agrees and is recorded as it is worded there:
//! *"Connected-component **label order** is a measured non-issue on all three pages --
//! reversing the entire non-background label numbering changes upstream's output by 0 px on
//! E01P01, E01P02 and E01P03 -- but that is a property of those pages, so A3 states it as a
//! recorded observation, never as a licence to ignore ordering."* Re-measured here against
//! upstream's own `merge_mask_list` on E01P01's four blocks on 2026-08-07: reversing the
//! non-background label order leaves all four merged masks byte-identical.

use crate::annotate::{threshold_binary, MaskCandidate, THRESHOLD_MAXVAL};
use image::{GrayImage, Luma};
use pc_core::StageError;

/// `pred_thresh`'s threshold in upstream's `cv2.threshold(pred_mask, 60, 255, THRESH_BINARY)`
/// (`textmask.py:109`). The comparison is **strict**, so a pred pixel of exactly 60 is off.
///
/// Upstream reaches this only under `if pred_thresh > 0`; `pred_thresh` defaults to 30 and
/// `refine_mask` never overrides it, so the branch is unconditional here. Note that the
/// value 30 is **not** the threshold -- 60 is; 30 only decides *whether* to threshold.
pub const MERGE_PRED_THRESHOLD: u8 = 60;

/// Upstream's `if w * h < 3: continue` (`textmask.py:119-120`) -- a **bounding-box** area, not
/// a pixel count. A two-pixel diagonal component has `w * h == 4` and survives; a two-pixel
/// horizontal component has `w * h == 2` and does not.
pub const MERGE_MIN_COMPONENT_BBOX_AREA: u64 = 3;

// ------------------------------------------------------------------- erode

/// `cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1)))` -- the
/// minimum over the **4-neighbourhood plus the centre**.
///
/// The kernel is `[[0,1,0],[1,1,1],[0,1,0]]` (module header note (b), measured against `cv2`
/// 5.0.0), so this is **not** [`crate::annotate::erode_rect3x3`], which takes the minimum over
/// the full 3x3 square for `np.ones((3, 3))`.
///
/// Border handling matches A1's: `cv2.erode`'s default `borderValue` is
/// `morphologyDefaultBorderValue()`, i.e. `+DBL_MAX`, so out-of-image neighbours never lower
/// the minimum and edge pixels are eroded only by their in-image neighbours.
#[must_use]
pub fn erode_ellipse3x3(mask: &GrayImage) -> GrayImage {
    let (width, height) = mask.dimensions();
    GrayImage::from_fn(width, height, |x, y| {
        let mut min = mask.get_pixel(x, y).0[0];
        if x > 0 {
            min = min.min(mask.get_pixel(x - 1, y).0[0]);
        }
        if x + 1 < width {
            min = min.min(mask.get_pixel(x + 1, y).0[0]);
        }
        if y > 0 {
            min = min.min(mask.get_pixel(x, y - 1).0[0]);
        }
        if y + 1 < height {
            min = min.min(mask.get_pixel(x, y + 1).0[0]);
        }
        Luma([min])
    })
}

/// Upstream's `if pred_thresh > 0:` block (`textmask.py:103-109`): erode the pred mask with the
/// plus-shaped 3x3 element, then binarise it strictly above [`MERGE_PRED_THRESHOLD`].
///
/// **This -- not the raw U-Net mask -- is what every XOR comparison in the merge and hole-fill
/// loops scores against**, and the distinction is observable rather than academic. A1/A2's
/// `xor_sum` scores candidates against the *raw* mask (255 distinct values on the committed
/// page); this pass scores components against a 0/255 mask. On a uniform pred mask of **127**
/// the two disagree in opposite directions: raw scoring rejects a fully-covered component
/// (`255 ^ 127 = 128 > 127`, so turning a pixel on *raises* the XOR) while this binarised mask
/// accepts it (`127 > 60`, so the pixel is on and turning it on *lowers* the XOR). A test pins
/// that input.
#[must_use]
pub fn binarize_pred_mask(pred_mask: &GrayImage) -> GrayImage {
    threshold_binary(&erode_ellipse3x3(pred_mask), MERGE_PRED_THRESHOLD)
}

// ---------------------------------------------------- connected components

/// One row of `cv2.connectedComponentsWithStats`'s `stats` output: `[x, y, width, height,
/// area]`, where the first four are the component's bounding box and `area` is its pixel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComponentStats {
    /// `CC_STAT_LEFT`.
    pub x: u32,
    /// `CC_STAT_TOP`.
    pub y: u32,
    /// `CC_STAT_WIDTH`.
    pub width: u32,
    /// `CC_STAT_HEIGHT`.
    pub height: u32,
    /// `CC_STAT_AREA` -- a **pixel count**, not `width * height`.
    pub area: u32,
}

impl ComponentStats {
    /// Upstream's `w * h`, widened so a page-sized bounding box cannot overflow.
    #[must_use]
    pub fn bounding_box_area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// `cv2.connectedComponentsWithStats(mask)`'s three used outputs: `num_labels`, `labels` and
/// `stats`. `centroids` is computed by upstream and never read, so it is not produced here.
///
/// **Label 0 is every zero-valued pixel lumped together, not a component.** OpenCV labels the
/// whole background 0 regardless of connectivity, so a hole fully enclosed by a component
/// carries label 0 exactly like the outer background does -- which is precisely why upstream
/// needs a *second* labeling of `255 - mask_merged` to find holes at all. Verified against
/// `cv2` on a 1-pixel-thick ring: `num_labels == 2`, and the enclosed interior carries label 0.
///
/// `stats[0]` is therefore the bounding box and pixel count of **all** zero pixels. When there
/// are none, `cv2` returns the uninitialised row `[-1, 2147483647, 0, 0, 0]`; this type reports
/// `ComponentStats::default()` (an empty box at the origin, `area = 0`) instead. The two are
/// behaviourally identical wherever upstream uses the row -- both index an empty slice -- and
/// the difference is unreachable in [`merge_mask_list`] anyway; see
/// [`hole_fill_area_threshold`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectedComponents {
    /// Image width, so [`Self::label_at`] can index [`Self::labels`].
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Row-major, `width * height` entries. `0` is background.
    pub labels: Vec<u32>,
    /// `num_labels` entries, indexed by label. Always at least one (label 0).
    pub stats: Vec<ComponentStats>,
}

impl ConnectedComponents {
    /// `num_labels` -- the background label included, so this is never zero.
    #[must_use]
    pub fn num_labels(&self) -> u32 {
        self.stats.len() as u32
    }

    /// The label at `(x, y)`.
    ///
    /// # Panics
    /// If `(x, y)` is outside the labelled image.
    #[must_use]
    pub fn label_at(&self, x: u32, y: u32) -> u32 {
        assert!(
            x < self.width && y < self.height,
            "({x}, {y}) out of bounds"
        );
        self.labels[(y * self.width + x) as usize]
    }

    /// The **non-background** labels of this labeling, reordered into the sequence `cv2` would
    /// have numbered them in -- ascending 2x2-block raster order of each component's
    /// first-reached block.
    ///
    /// # This is the label ORDER OpenCV assigns, NOT a port of the BBDT algorithm
    ///
    /// Nothing here implements Grana/BBDT, SPAGHETTI or any other block-scan labeling. The
    /// pixel-to-label assignment, the `stats` rows and [`Self::labels`] are untouched and stay
    /// [`connected_components`]'s own first-raster-pixel numbering; this method only answers
    /// *"in what order would `cv2` have handed these same components out?"*, which is the only
    /// thing a consumer that selects by **position in the label sequence** actually needs.
    ///
    /// **The rule, measured rather than derived from OpenCV's source:** a component's `cv2` label
    /// rank is the ascending order of `min over its pixels of (y / 2, x / 2)`, compared row-major
    /// (block row first, then block column). Measured against **cv2 5.0.0**, on random binary
    /// fixtures, with **0 mismatches** in every run: the architect measured 2614 + 299 fixtures,
    /// the Senior Rust Engineer 2091 fixtures, and this implementation's own transcription run
    /// re-measured 2000 fixtures / 8318 components. All four **block-based** variants agree
    /// (`CCL_DEFAULT`, `CCL_BBDT`, `CCL_GRANA`, `CCL_SPAGHETTI`: 0 mismatches over 536 fixtures
    /// carrying two or more components); the two **pixel-based** variants do not (`CCL_SAUF`,
    /// `CCL_WU`: 71 of those same 536), and those two match *our* first-raster-pixel order
    /// instead. `cv2`'s effective default for connectivity 8 is the block-based family (module
    /// header note (a)), so the block rule is upstream's genuinely-intended order.
    ///
    /// **The key is a total order, not an arbitrary tie-break.** Two distinct components cannot
    /// share a minimum block: all four pixels of a 2x2 block are mutually 8-adjacent, so pixels
    /// of two labels in one block would be one label. No tie rule is therefore declared here,
    /// and none is needed.
    ///
    /// **Who uses this, and why nothing else should.** Its sole caller is
    /// `annotate_refine::undetected_blocks`, whose ported `valid_labels[1:]` selects by position
    /// in the label sequence (spec §16.37 items 11(b)(iii) and 12). [`merge_mask_list`]'s two
    /// loops are proven order-independent and must keep using the plain numbering -- routing
    /// them through this method would change nothing but would imply an order dependence they
    /// do not have.
    #[must_use]
    pub fn labels_in_opencv_block_scan_order(&self) -> Vec<u32> {
        // `min over pixels (y / 2, x / 2)` per label, taken as a real minimum rather than as
        // "the block of the first pixel the raster scan reaches". Those are NOT the same: a
        // component whose topmost pixel is `(0, 10)` may also hold `(1, 0)`, which shares block
        // row 0 and has the smaller block column, so the first-reached pixel can overstate the
        // key. Comparing every pixel is the only correct form.
        let mut min_block: Vec<Option<(u32, u32)>> = vec![None; self.stats.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                let label = self.labels[(y * self.width + x) as usize] as usize;
                if label == 0 {
                    continue;
                }
                let block = (y / 2, x / 2);
                let slot = &mut min_block[label];
                if slot.is_none_or(|current| block < current) {
                    *slot = Some(block);
                }
            }
        }

        let mut ordered: Vec<(u32, u32, u32)> = min_block
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(label, block)| block.map(|(by, bx)| (by, bx, label as u32)))
            .collect();
        ordered.sort_unstable();
        ordered.into_iter().map(|(_, _, label)| label).collect()
    }
}

/// `cv2.connectedComponentsWithStats(mask)` -- **8-connectivity**, non-zero pixels are
/// foreground.
///
/// 8 is both what upstream intends (`connectivity = 8`) and what `cv2` actually uses at both
/// call sites, for the reason in module header note (a): the positional `connectivity`
/// argument upstream passes never reaches the parameter of that name.
///
/// **Label numbering is ours**, in ascending order of each component's first raster pixel; see
/// the module header for why that is not OpenCV's order. The `stats` rows are OpenCV's
/// semantics exactly and a test pins them against `cv2` on six inputs.
///
/// **The no-`DEVIATION` conclusion for that difference is scoped to `merge_mask_list`'s two
/// loops only** (spec §16.37 item 12), because that is the scope of the proof behind it. It is
/// **not** a general licence to ignore label numbering: this type is `pub`, its `labels` and
/// `stats` are `pub`, and a different caller can be order-sensitive where those two loops are
/// not -- upstream's `refine_undetected_mask` (spec §16.37 item 11's task A3b) iterates
/// `valid_labels[1:]` and so drops whichever area-filtered entry sorts first. A new consumer
/// re-establishes order-independence for itself or registers the divergence it has; it does not
/// inherit this one. That consumer took a third route (§16.37 item 13): it reads
/// [`ConnectedComponents::labels_in_opencv_block_scan_order`], so it is at parity with `cv2`'s
/// order without this function's numbering moving.
#[must_use]
pub fn connected_components(mask: &GrayImage) -> ConnectedComponents {
    let (width, height) = mask.dimensions();
    let count = (width as usize) * (height as usize);

    // Union-find over pixel indices, with the root always the smallest index in the component.
    // That makes a component's root its own first raster pixel, so numbering roots in
    // ascending order is the documented first-raster-pixel order.
    let mut parent: Vec<u32> = (0..count as u32).collect();
    fn find(parent: &mut [u32], mut index: u32) -> u32 {
        while parent[index as usize] != index {
            let grandparent = parent[parent[index as usize] as usize];
            parent[index as usize] = grandparent;
            index = grandparent;
        }
        index
    }
    let union = |parent: &mut Vec<u32>, a: u32, b: u32| {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    };

    let foreground = |x: u32, y: u32| mask.get_pixel(x, y).0[0] != 0;
    for y in 0..height {
        for x in 0..width {
            if !foreground(x, y) {
                continue;
            }
            let here = y * width + x;
            // The four already-scanned 8-neighbours. Scanning only backwards is what makes one
            // pass plus union-find sufficient.
            if x > 0 && foreground(x - 1, y) {
                union(&mut parent, here, here - 1);
            }
            if y > 0 {
                if foreground(x, y - 1) {
                    union(&mut parent, here, here - width);
                }
                if x > 0 && foreground(x - 1, y - 1) {
                    union(&mut parent, here, here - width - 1);
                }
                if x + 1 < width && foreground(x + 1, y - 1) {
                    union(&mut parent, here, here - width + 1);
                }
            }
        }
    }

    // Roots in ascending index order become labels 1, 2, ... -- `labels` starts all-zero, so
    // background pixels need no write.
    let mut labels = vec![0_u32; count];
    let mut stats = vec![ComponentStats::default()];
    let mut label_of_root = vec![0_u32; count];
    for y in 0..height {
        for x in 0..width {
            if !foreground(x, y) {
                continue;
            }
            let index = y * width + x;
            let root = find(&mut parent, index);
            if root == index {
                stats.push(ComponentStats {
                    x,
                    y,
                    width: 1,
                    height: 1,
                    area: 0,
                });
                label_of_root[root as usize] = stats.len() as u32 - 1;
            }
            labels[index as usize] = label_of_root[root as usize];
        }
    }

    // Bounding boxes and areas, label 0 included, in one pass. Accumulated as inclusive
    // min/max and converted at the end, so an empty label 0 stays `ComponentStats::default()`.
    let mut extent: Vec<Option<(u32, u32, u32, u32)>> = vec![None; stats.len()];
    for y in 0..height {
        for x in 0..width {
            let label = labels[(y * width + x) as usize] as usize;
            stats[label].area += 1;
            extent[label] = Some(match extent[label] {
                None => (x, y, x, y),
                Some((x1, y1, x2, y2)) => (x1.min(x), y1.min(y), x2.max(x), y2.max(y)),
            });
        }
    }
    for (stat, extent) in stats.iter_mut().zip(&extent) {
        match *extent {
            None => {
                *stat = ComponentStats::default();
            }
            Some((x1, y1, x2, y2)) => {
                stat.x = x1;
                stat.y = y1;
                stat.width = x2 - x1 + 1;
                stat.height = y2 - y1 + 1;
            }
        }
    }

    ConnectedComponents {
        width,
        height,
        labels,
        stats,
    }
}

// ------------------------------------------------------- the XOR merge loop

/// Upstream's accept-if-it-lowers-the-XOR loop (`textmask.py:116-132`, and again at
/// `:145-157`), with the label sequence hoisted out so the two upstream copies -- which differ
/// only in which labels they visit -- share one body.
///
/// For each label in `labels`, in the order given: restrict to that component's bounding box,
/// form `tmp = merged[bbox] | indicator(label)`, and overwrite `merged[bbox]` with `tmp` iff
/// `xor(tmp, pred_binary[bbox]).sum() < xor(merged[bbox], pred_binary[bbox]).sum()` --
/// **strict**, so a component that leaves the XOR unchanged is rejected.
///
/// `pred_binary` must be [`binarize_pred_mask`]'s output, not a raw U-Net mask; the loop is
/// written against arbitrary `u8` values regardless, because that is what
/// `cv2.bitwise_xor(...).sum()` does.
///
/// **Caller obligation, which the module header's order-independence proof rests on:**
/// distinct labels in `components` must be pixel-disjoint. Every labeling
/// [`connected_components`] produces satisfies that; a hand-built [`ConnectedComponents`] with
/// overlapping labels does not, and then the result *does* depend on `labels`' order.
///
/// A label whose bounding box is empty (`width == 0 || height == 0`, i.e. the empty label 0 of
/// an all-foreground image) is a no-op, matching upstream's empty numpy slice.
///
/// Errors (`StageError::InvalidInput`, **per-image**, cookbook rule 4) when `merged`,
/// `pred_binary` and `components` do not all have the same size, or when `labels` names a
/// label that does not exist: upstream's numpy indexing is only defined for equal shapes, and
/// one malformed block must not abort a run.
pub fn merge_components_by_xor(
    merged: &mut GrayImage,
    components: &ConnectedComponents,
    pred_binary: &GrayImage,
    labels: &[u32],
) -> Result<(), StageError> {
    let size = (components.width, components.height);
    if merged.dimensions() != size || pred_binary.dimensions() != size {
        return Err(StageError::InvalidInput(format!(
            "merge operands {:?} / {:?} must both match the labelled image {size:?}",
            merged.dimensions(),
            pred_binary.dimensions()
        )));
    }
    if let Some(&label) = labels.iter().find(|&&l| l >= components.num_labels()) {
        return Err(StageError::InvalidInput(format!(
            "label {label} does not exist in a labelling with {} labels",
            components.num_labels()
        )));
    }

    for &label in labels {
        let stat = components.stats[label as usize];
        if stat.width == 0 || stat.height == 0 {
            continue;
        }
        let (mut xor_origin, mut xor_merged) = (0_u64, 0_u64);
        for y in stat.y..stat.y + stat.height {
            for x in stat.x..stat.x + stat.width {
                let origin = merged.get_pixel(x, y).0[0];
                // `cv2.bitwise_or(mask_merged[bbox], tmp_merged)`, where `tmp_merged` is 255 on
                // this label and 0 elsewhere.
                let candidate = if components.label_at(x, y) == label {
                    THRESHOLD_MAXVAL
                } else {
                    origin
                };
                let pred = pred_binary.get_pixel(x, y).0[0];
                xor_origin += u64::from(origin ^ pred);
                xor_merged += u64::from(candidate ^ pred);
            }
        }
        if xor_merged < xor_origin {
            for y in stat.y..stat.y + stat.height {
                for x in stat.x..stat.x + stat.width {
                    if components.label_at(x, y) == label {
                        merged.put_pixel(x, y, Luma([THRESHOLD_MAXVAL]));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Upstream's hole-fill threshold (`textmask.py:140-144`):
///
/// ```text
/// sorted_area = np.sort(stats[:, -1])
/// area_thresh = sorted_area[-2] if len(sorted_area) > 1 else sorted_area[-1]
/// ```
///
/// i.e. the **second largest** component area when there is more than one label, otherwise the
/// only one. Label 0's area participates in the sort. Using the largest instead is a different
/// function: on a 1-pixel-thick 9x9 frame inside an 11x11 crop the inverted areas are
/// `[32, 40, 49]`, so the threshold is 40 and nothing is filled, where the largest (49) would
/// pull in the 40-pixel outer background.
///
/// Since `area_thresh` is compared with `area < area_thresh` and label 0 is visited, this is
/// also what makes `cv2`'s uninitialised `stats[0]` row unreachable in [`merge_mask_list`]:
/// label 0 of `255 - merged` has area 0 only when `merged` is all zero, in which case the
/// inversion is uniformly non-zero, so there are exactly two labels, `sorted_area == [0, area]`
/// and the threshold is **0** -- and `0 < 0` is false.
#[must_use]
pub fn hole_fill_area_threshold(components: &ConnectedComponents) -> u32 {
    let mut areas: Vec<u32> = components.stats.iter().map(|stat| stat.area).collect();
    areas.sort_unstable();
    // `sort_unstable` is sound here where A2's `min_by_key` was not: these are `u32` values,
    // not candidates, so equal elements are indistinguishable and no tie order is observable.
    areas[areas.len().saturating_sub(2)]
}

/// `mask_list.sort(key=lambda x: x[1])` (`textmask.py:92`) -- ascending [`MaskCandidate::xor_sum`],
/// **stable**.
///
/// This is upstream's **second** sort of this list. The first is inside
/// `get_otsuthresh_masklist` and ranks channels; this one ranks the concatenation
/// [`crate::annotate::candidate_mask_list`] returns, in which the Otsu candidate is **last**
/// after one to three top-k candidates. Python's `list.sort` is guaranteed stable, so two
/// candidates with equal `xor_sum` keep that concatenation order -- top-k before Otsu -- and
/// [`merge_mask_list`] then visits them in that order. Order across candidates is observable
/// (unlike order *within* a candidate; see the module header), because a candidate visited
/// second is scored against a `merged` the first one has already populated.
pub fn sort_candidates_by_xor_sum(candidates: &mut [MaskCandidate]) {
    candidates.sort_by_key(|candidate| candidate.xor_sum);
}

/// `merge_mask_list(mask_list, pred_mask, refine_mode=REFINEMASK_ANNOTATION)` -- upstream
/// `textmask.py:83-158`.
///
/// Takes [`crate::annotate::candidate_mask_list`]'s output by value because upstream sorts the
/// list in place, and returns the merged 0/255 mask the size of `pred_mask`.
///
/// The `REFINEMASK_INPAINT` dilation at `:134-135` is **not** implemented; module header note
/// (c) records why, and what the two branches measure on the committed page.
///
/// Errors (`StageError::InvalidInput`, **per-image**, cookbook rule 4) when any candidate's
/// mask differs in size from `pred_mask`.
pub fn merge_mask_list(
    mut candidates: Vec<MaskCandidate>,
    pred_mask: &GrayImage,
) -> Result<GrayImage, StageError> {
    let (width, height) = pred_mask.dimensions();
    for candidate in &candidates {
        if candidate.mask.dimensions() != (width, height) {
            return Err(StageError::InvalidInput(format!(
                "candidate mask {:?} must match the pred mask {:?}",
                candidate.mask.dimensions(),
                pred_mask.dimensions()
            )));
        }
    }
    sort_candidates_by_xor_sum(&mut candidates);
    let pred_binary = binarize_pred_mask(pred_mask);

    let mut merged = GrayImage::new(width, height);
    for candidate in &candidates {
        let components = connected_components(&candidate.mask);
        // `if label_index != 0` and `if w * h < 3: continue`, hoisted into the label sequence.
        let labels: Vec<u32> = (1..components.num_labels())
            .filter(|&label| {
                components.stats[label as usize].bounding_box_area()
                    >= MERGE_MIN_COMPONENT_BBOX_AREA
            })
            .collect();
        merge_components_by_xor(&mut merged, &components, &pred_binary, &labels)?;
    }

    // `255 - mask_merged`, then fill every background component smaller than the second
    // largest. Label 0 is **not** skipped here, unlike the loop above.
    let inverted = GrayImage::from_fn(width, height, |x, y| {
        Luma([THRESHOLD_MAXVAL - merged.get_pixel(x, y).0[0]])
    });
    let holes = connected_components(&inverted);
    let area_threshold = hole_fill_area_threshold(&holes);
    let labels: Vec<u32> = (0..holes.num_labels())
        .filter(|&label| holes.stats[label as usize].area < area_threshold)
        .collect();
    merge_components_by_xor(&mut merged, &holes, &pred_binary, &labels)?;

    Ok(merged)
}
