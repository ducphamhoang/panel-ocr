//! Task M2 -- the candidate sequence and its padding policy (spec §10.3 step 6,
//! §10.7(A)1-3, §16.9 items 5, 6).
//!
//! **The kernel and the dilation no longer live here.** §16.38 item 16(a) (v1.5, task L1)
//! hoisted `Kernel`, `kernel` and `dilate` into [`pc_imageops::morph`], because
//! `pc-denoise` held a verbatim second copy (§16.10 item 2) and `pc-inpaint` needs a third
//! -- and §1 rule 2 forbids the stage-to-stage edge that would let them share. They are
//! re-exported below, so `pc_mask::grow::{Kernel, kernel, dilate}` still resolves and
//! `crates/pc-mask/tests/m2_grow.rs` is untouched. What stays here is the part that is
//! *masking policy*: the `MaskerConfig`-shaped growth helpers, which is also what keeps
//! `pc-imageops` free of a `pc-config` dependency (§16.9 item 2, reaffirmed by §16.38 item
//! 16(a)).
//!
//! The one subtlety worth reading twice: candidates are produced by dilating a single
//! **replicate-padded buffer in place**, so border-replication effects accumulate from
//! one candidate to the next exactly as they do upstream (which reuses `padded_mask`
//! across iterations). Each candidate is the centre crop of that buffer.

use pc_config::MaskerConfig;
use pc_imageops::BinaryMask;

pub use pc_imageops::morph::{dilate, kernel, Kernel};

/// One entry of §10.3 step 7's candidate list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub mask: BinaryMask,
    /// `None` for the box candidate -- it is not grown from the precise mask
    /// (§10.3 step 5).
    pub thickness: Option<u32>,
}

/// `np.pad(mode="edge")`: `pad` pixels on all four sides, each copied from the nearest
/// edge pixel of `mask` (§10.3 step 6).
pub fn pad_replicate(mask: &BinaryMask, pad: u32) -> BinaryMask {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return BinaryMask::new(width + pad * 2, height + pad * 2);
    }
    BinaryMask::from_fn(width + pad * 2, height + pad * 2, |x, y| {
        let source_x = x.saturating_sub(pad).min(width - 1);
        let source_y = y.saturating_sub(pad).min(height - 1);
        mask.get(source_x, source_y)
    })
}

/// Inverse of [`pad_replicate`]'s framing: drop `pad` pixels from all four sides.
pub fn center_crop(mask: &BinaryMask, pad: u32) -> BinaryMask {
    let (width, height) = mask.dimensions();
    let inner_w = width.saturating_sub(pad * 2);
    let inner_h = height.saturating_sub(pad * 2);
    BinaryMask::from_fn(inner_w, inner_h, |x, y| mask.get(x + pad, y + pad))
}

/// `max(min_mask_thickness, mask_growth_step_pixels) * 2` (§10.3 step 6).
pub fn growth_padding(config: &MaskerConfig) -> u32 {
    config
        .min_mask_thickness
        .max(config.mask_growth_step_pixels)
        * 2
}

/// spec §10.3 step 6: exactly `mask_growth_steps` candidates, thicknesses
/// `min_mask_thickness + i * mask_growth_step_pixels` for `i` in
/// `0..mask_growth_steps`, each the centre crop of one shared padded buffer that is
/// dilated in place (candidate 0 with `kernel(min_mask_thickness)`, the rest with
/// `kernel(mask_growth_step_pixels)`).
pub fn growth_candidates(precise: &BinaryMask, config: &MaskerConfig) -> Vec<Candidate> {
    let pad = growth_padding(config);
    let first = kernel(config.min_mask_thickness);
    let step = kernel(config.mask_growth_step_pixels);

    let mut padded = pad_replicate(precise, pad);
    let mut candidates = Vec::with_capacity(config.mask_growth_steps as usize);
    for i in 0..config.mask_growth_steps {
        let element = if i == 0 { &first } else { &step };
        padded = dilate(&padded, element);
        candidates.push(Candidate {
            mask: center_crop(&padded, pad),
            thickness: Some(config.min_mask_thickness + i * config.mask_growth_step_pixels),
        });
    }
    candidates
}

/// spec §10.3 step 7 -- candidate ordering, which "determines everything downstream":
/// box candidate **first** in `mask_selection_fast` mode, **last** otherwise.
pub fn build_candidates(
    precise: &BinaryMask,
    box_candidate: BinaryMask,
    config: &MaskerConfig,
) -> Vec<Candidate> {
    let growth = growth_candidates(precise, config);
    let box_candidate = Candidate {
        mask: box_candidate,
        thickness: None,
    };

    if config.mask_selection_fast {
        let mut candidates = Vec::with_capacity(growth.len() + 1);
        candidates.push(box_candidate);
        candidates.extend(growth);
        candidates
    } else {
        let mut candidates = growth;
        candidates.push(box_candidate);
        candidates
    }
}
