//! Task M4 -- `fit_region`, the heart of the stage (spec §10.3 step 3, §10.7(A)8-11,
//! §16.9 items 9-12).
//!
//! **Precision rule (§15.9): `f64` only, no `f32` in this module.** The comparison
//! `dev_i <= best_dev * (1 - threshold)` decides which mask candidate paints the page.
//!
//! `select_candidate_with_fallback` holds the scored selection policy;
//! `select_candidate` is its compatibility wrapper, while `fit_region_scored` and
//! `fit_region` wire selection and the optional lowest-deviation rescue into fitting.

use crate::border::{border_std_deviation, BlankMask, BorderStats};
use crate::grow::build_candidates;
use pc_config::MaskerConfig;
use pc_core::Rect;
use pc_imageops::BinaryMask;

/// spec §10.3 step 11.
#[derive(Debug, Clone, PartialEq)]
pub struct Fitment {
    /// `None` when the selected (greedy or rescued) candidate's border deviation still
    /// exceeded `mask_max_standard_deviation` -- the box is left uncleaned, but it is
    /// still reported (§10.3 step 10).
    pub mask: Option<BinaryMask>,
    /// Populated even on the failure path (§16.9 item 12).
    pub median_color: [u8; 3],
    /// Where `mask` sits on the page: `(reference.x1, reference.y1)`.
    pub coords: (i32, i32),
    pub std_deviation: f64,
    /// Index of the chosen candidate in the candidate list as ordered by §10.3 step 7.
    pub candidate_index: usize,
    /// `None` when the box candidate was chosen.
    pub thickness: Option<u32>,
    pub masking_rect: Rect,
}

impl Fitment {
    /// `std_deviation > masker.mask_max_standard_deviation` (§2.6's `failed`).
    pub fn failed(&self) -> bool {
        self.mask.is_none()
    }
}

/// The greedy candidate returned by [`select_candidate`]; rescue-aware fitting uses
/// [`Selection`] and [`resolve_fallback`] to choose its final [`Scored`] candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selected {
    pub index: usize,
    pub std_deviation: f64,
    pub median_color: [u8; 3],
}

// ---------------------------------------------------------------------------------------
// §16.35 item 7 -- scored selection and lowest-deviation rescue. `fit_region_scored` and
// `fit_region` use this path; `select_candidate` is a thin compatibility wrapper over it.
// P1-P4 (§16.35 item 2) preserve the outcomes captured by the frozen `m4_fit.rs` and
// `m56_run.rs` traces for every already-passing region.
// ---------------------------------------------------------------------------------------

/// §16.35 item 7 -- one scored candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub index: usize,
    pub std_deviation: f64,
    pub median_color: [u8; 3],
}

impl From<Scored> for Selected {
    fn from(scored: Scored) -> Selected {
        Selected {
            index: scored.index,
            std_deviation: scored.std_deviation,
            median_color: scored.median_color,
        }
    }
}

/// §16.35 item 7 -- everything §10.3 step 9's loop learned. `deviations` holds exactly the
/// candidates that were scored, in candidate order, so a fast-mode run that broke early
/// reports a shorter vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    /// §10.3 step 9's ratchet pick.
    pub greedy: Scored,
    /// The lowest-`std_deviation` candidate among those scored; lowest index on an
    /// exact-equal tie (§16.35 item 5).
    pub lowest: Scored,
    pub deviations: Vec<f64>,
}

/// §16.35 item 7 -- the fail-safe predicate of §10.3 step 10, named once so it has exactly
/// one call site.
pub fn fit_accepted(std_deviation: f64, mask_max_standard_deviation: f64) -> bool {
    std_deviation <= mask_max_standard_deviation
}

/// §16.35 item 7 -- §10.3 step 9's loop, reporting everything it scored.
pub fn select_candidate_with_fallback<F>(
    count: usize,
    fast: bool,
    improvement_threshold: f64,
    scorer: F,
) -> Result<Selection, BlankMask>
where
    F: FnMut(usize) -> Result<BorderStats, BlankMask>,
{
    assert!(
        count > 0,
        "candidate selection needs at least one candidate"
    );

    let mut scorer = scorer;

    let mut greedy: Option<Scored> = None;
    let mut lowest: Option<Scored> = None;
    let mut deviations = Vec::with_capacity(count);

    for index in 0..count {
        let stats = scorer(index)?;
        deviations.push(stats.std_deviation);
        let scored = Scored {
            index,
            std_deviation: stats.std_deviation,
            median_color: stats.median_color,
        };

        let accept = match greedy {
            None => true,
            Some(current) => {
                scored.std_deviation <= current.std_deviation * (1.0 - improvement_threshold)
            }
        };
        if accept {
            greedy = Some(scored);
        }

        if lowest
            .map(|current| scored.std_deviation < current.std_deviation)
            .unwrap_or(true)
        {
            lowest = Some(scored);
        }

        if fast && stats.std_deviation == 0.0 {
            break;
        }
    }

    Ok(Selection {
        greedy: greedy.expect("count > 0 always accepts candidate 0"),
        lowest: lowest.expect("count > 0 always scores candidate 0"),
        deviations,
    })
}

/// §16.35 item 1's rescue policy. `enabled = false` reproduces `select_candidate`'s
/// greedy output bit for bit (§16.35 item 4).
pub fn resolve_fallback(
    selection: &Selection,
    mask_max_standard_deviation: f64,
    enabled: bool,
) -> Scored {
    if !enabled || fit_accepted(selection.greedy.std_deviation, mask_max_standard_deviation) {
        selection.greedy
    } else if fit_accepted(selection.lowest.std_deviation, mask_max_standard_deviation) {
        // DEVIATION(21): retrying with the lowest already-scored candidate when the greedy
        // pick fails the fail-safe has no upstream equivalent; see §14 item 21 and §16.35.
        selection.lowest
    } else {
        selection.greedy
    }
}

/// §16.35 item 7 -- the diagnostic that travels beside `Fitment` instead of inside it
/// (`Fitment` itself gains no fields).
#[derive(Debug, Clone, PartialEq)]
pub struct FitReport {
    pub fitment: Fitment,
    /// The deviations of exactly the candidates that were scored, in candidate order.
    pub candidate_deviations: Vec<f64>,
    /// §10.3 step 9's ratchet pick, reported even when the rescue overrode it.
    pub greedy_index: usize,
    pub rescued: bool,
}

/// §16.35 item 7 -- `fit_region` becomes a thin wrapper over this.
pub fn fit_region_scored(
    base: &crate::border::BaseCanvas,
    cut: &BinaryMask,
    box_mask: &BinaryMask,
    masking: Rect,
    reference: Rect,
    config: &MaskerConfig,
) -> Option<FitReport> {
    let image_size = base.dimensions();
    if reference.to_crop(image_size).is_none() || masking.to_crop(image_size).is_none() {
        tracing::warn!(
            ?masking,
            ?reference,
            ?image_size,
            "skipping degenerate or out-of-canvas masking region"
        );
        return None;
    }

    let x_offset = masking.x1 - reference.x1;
    let y_offset = masking.y1 - reference.y1;
    let base_crop = base
        .crop(reference)
        .expect("reference was validated against the base canvas");
    let crop_size = base_crop.dimensions();
    let precise_cut = cut.crop_into(masking, crop_size, (x_offset, y_offset));
    if precise_cut.is_blank() {
        tracing::warn!(
            ?masking,
            ?reference,
            "skipping masking region with a blank precise cut"
        );
        return None;
    }

    let box_candidate = box_mask.crop_into(masking, crop_size, (x_offset, y_offset));
    let candidates = build_candidates(&precise_cut, box_candidate, config);
    let selection = match select_candidate_with_fallback(
        candidates.len(),
        config.mask_selection_fast,
        config.mask_improvement_threshold,
        |index| {
            border_std_deviation(
                &base_crop,
                &candidates[index].mask,
                config.off_white_max_threshold,
                config.allow_colored_masks,
            )
        },
    ) {
        Ok(selection) => selection,
        Err(BlankMask) => {
            tracing::warn!(
                ?masking,
                ?reference,
                "skipping masking region with an edgeless candidate"
            );
            return None;
        }
    };

    let selected = resolve_fallback(
        &selection,
        config.mask_max_standard_deviation,
        config.mask_fallback_to_lowest_deviation,
    );
    let rescued = selected.index != selection.greedy.index;
    let chosen = &candidates[selected.index];
    let mask = fit_accepted(selected.std_deviation, config.mask_max_standard_deviation)
        .then(|| chosen.mask.clone());

    Some(FitReport {
        fitment: Fitment {
            mask,
            median_color: selected.median_color,
            coords: (reference.x1, reference.y1),
            std_deviation: selected.std_deviation,
            candidate_index: selected.index,
            thickness: chosen.thickness,
            masking_rect: masking,
        },
        candidate_deviations: selection.deviations,
        greedy_index: selection.greedy.index,
        rescued,
    })
}

/// spec §10.3 step 9 -- candidate selection, as a pure policy over a scorer (§16.9
/// item 9), so the rule can be tested on canned deviations and the fast-mode early
/// break can be observed directly.
///
/// ```text
/// accept candidate i  iff  i == 0  or  dev_i <= best_dev * (1.0 - mask_improvement_threshold)
/// ```
///
/// Consequences that are the *intent* of the algorithm and must not be "fixed": a larger
/// mask only wins if it improves border uniformity by at least the threshold, and
/// `best_dev == 0.0` makes `best_dev * (1 - t)` zero, so nothing can beat a perfect mask
/// -- except another perfect mask, since the comparison is `<=` (§10.7(A)8).
///
/// In `fast` mode the loop breaks as soon as a candidate scores exactly `0.0` (§16.9
/// item 10). Any `Err(BlankMask)` from the scorer abandons the whole region (§10.3
/// step 8).
///
/// Panics when `count == 0`; callers always build at least the box candidate.
pub fn select_candidate<F>(
    count: usize,
    fast: bool,
    improvement_threshold: f64,
    scorer: F,
) -> Result<Selected, BlankMask>
where
    F: FnMut(usize) -> Result<BorderStats, BlankMask>,
{
    select_candidate_with_fallback(count, fast, improvement_threshold, scorer)
        .map(|s| s.greedy.into())
}

/// spec §10.3 step 3 -- fit one masking region.
///
/// Returns `None` when the region must be dropped entirely: a blank precise cut
/// (detector noise, §10.3 step 4), a candidate with no edge pixels (§10.3 step 8), or a
/// degenerate/out-of-canvas rect (§16.9 item 11). Each of those is logged at `WARN` and
/// is **not** an error (§5.6).
///
/// Steps, in order:
///  1. `x_offset = masking.x1 - reference.x1`, `y_offset = masking.y1 - reference.y1`
///  2. `base_crop = base.crop(reference)` -- the analysis canvas
///  3. `precise_cut = cut.crop_into(masking, base_crop_size, (x_offset, y_offset))`
///  4. blank `precise_cut` -> `None`
///  5. `box_candidate = box_mask.crop_into(masking, ...)`, `thickness = None`
///  6. growth candidates (`grow::growth_candidates`)
///  7. ordering (`grow::build_candidates`)
///  8. score with `border::border_std_deviation`
///  9. score/select (`select_candidate_with_fallback`)
/// 10. resolve the optional lowest-deviation rescue (`resolve_fallback`)
/// 11. selected `std_deviation > mask_max_standard_deviation` -> `mask: None`
/// 12. assemble `Fitment`
pub fn fit_region(
    base: &crate::border::BaseCanvas,
    cut: &BinaryMask,
    box_mask: &BinaryMask,
    masking: Rect,
    reference: Rect,
    config: &MaskerConfig,
) -> Option<Fitment> {
    fit_region_scored(base, cut, box_mask, masking, reference, config).map(|report| report.fitment)
}
