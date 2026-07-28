//! Task M4 -- `fit_region`, the heart of the stage (spec §10.3 step 3, §10.7(A)8-11,
//! §16.9 items 9-12).
//!
//! **Precision rule (§15.9): `f64` only, no `f32` in this module.** The comparison
//! `dev_i <= best_dev * (1 - threshold)` decides which mask candidate paints the page.
//!
//! `select_candidate` (the policy) is implemented; `fit_region` (the wiring) is a
//! `todo!()` skeleton for task M4 -- its signature is frozen with the tests.
#![allow(unused_variables)]

use crate::border::{BlankMask, BorderStats};
use pc_config::MaskerConfig;
use pc_core::Rect;
use pc_imageops::BinaryMask;

/// spec §10.3 step 11.
#[derive(Debug, Clone, PartialEq)]
pub struct Fitment {
    /// `None` when the best candidate's border deviation exceeded
    /// `mask_max_standard_deviation` -- the box is left uncleaned, but it is still
    /// reported (§10.3 step 10).
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

/// What [`select_candidate`] picked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selected {
    pub index: usize,
    pub std_deviation: f64,
    pub median_color: [u8; 3],
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
    mut scorer: F,
) -> Result<Selected, BlankMask>
where
    F: FnMut(usize) -> Result<BorderStats, BlankMask>,
{
    assert!(
        count > 0,
        "candidate selection needs at least one candidate"
    );

    let mut best: Option<Selected> = None;
    for index in 0..count {
        let stats = scorer(index)?;
        let accept = match &best {
            None => true,
            Some(current) => {
                stats.std_deviation <= current.std_deviation * (1.0 - improvement_threshold)
            }
        };
        if accept {
            best = Some(Selected {
                index,
                std_deviation: stats.std_deviation,
                median_color: stats.median_color,
            });
        }
        if fast && stats.std_deviation == 0.0 {
            break;
        }
    }

    Ok(best.expect("count > 0 always accepts candidate 0"))
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
///  9. select (`select_candidate`)
/// 10. `std_deviation > mask_max_standard_deviation` -> `mask: None`
/// 11. assemble `Fitment`
pub fn fit_region(
    base: &crate::border::BaseCanvas,
    cut: &BinaryMask,
    box_mask: &BinaryMask,
    masking: Rect,
    reference: Rect,
    config: &MaskerConfig,
) -> Option<Fitment> {
    todo!("task M4: spec §10.3 step 3")
}
