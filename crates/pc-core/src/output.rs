//! spec §2.8 — `Step` / `Output`, mirroring upstream `output_structures.py` minus
//! inpainting. `cache_suffix` values are upstream's file suffixes VERBATIM so a user
//! can diff our cache against upstream's during parity work.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Detect = 1,
    Preprocess,
    Mask,
    Denoise,
    Export,
}

impl Step {
    /// The step before this one; `None` for `Detect`. Required by §4.4's resume logic.
    pub fn prev(self) -> Option<Step> {
        match self {
            Step::Detect => None,
            Step::Preprocess => Some(Step::Detect),
            Step::Mask => Some(Step::Preprocess),
            Step::Denoise => Some(Step::Mask),
            Step::Export => Some(Step::Denoise),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Output {
    // Detect
    BaseImage,
    RawMask,
    RawJson,
    // Preprocess
    CleanJson,
    // Mask
    BoxMask,
    CutMask,
    FinalMask,
    MaskOverlay,
    IsolatedText,
    MaskedOutput,
    MaskDataJson,
    // Denoise
    DenoiseMask,
    DenoisedOutput,
}

impl Output {
    /// Note: deliberately non-surjective onto `Step` — no variant maps to
    /// `Step::Export` (export writes final user-facing files, not cache artifacts).
    pub fn step(self) -> Step {
        match self {
            Output::BaseImage | Output::RawMask | Output::RawJson => Step::Detect,
            Output::CleanJson => Step::Preprocess,
            Output::BoxMask
            | Output::CutMask
            | Output::FinalMask
            | Output::MaskOverlay
            | Output::IsolatedText
            | Output::MaskedOutput
            | Output::MaskDataJson => Step::Mask,
            Output::DenoiseMask | Output::DenoisedOutput => Step::Denoise,
        }
    }

    /// Upstream suffixes, verbatim (§2.8).
    pub fn cache_suffix(self) -> &'static str {
        match self {
            Output::BaseImage => "_base.png",
            Output::RawMask => "_raw_mask.png",
            Output::RawJson => "#raw.json",
            Output::CleanJson => "#clean.json",
            Output::BoxMask => "_box_mask.png",
            Output::CutMask => "_cut_mask.png",
            Output::FinalMask => "_combined_mask.png",
            Output::MaskOverlay => "_with_masks.png",
            Output::IsolatedText => "_text.png",
            Output::MaskedOutput => "_clean.png",
            Output::MaskDataJson => "#mask_data.json",
            Output::DenoiseMask => "_noise_mask.png",
            Output::DenoisedOutput => "_clean_denoised.png",
        }
    }

    /// All variants, in declaration order. Lets tests enumerate exhaustively without
    /// a hand-maintained duplicate list.
    pub const ALL: &'static [Output] = &[
        Output::BaseImage,
        Output::RawMask,
        Output::RawJson,
        Output::CleanJson,
        Output::BoxMask,
        Output::CutMask,
        Output::FinalMask,
        Output::MaskOverlay,
        Output::IsolatedText,
        Output::MaskedOutput,
        Output::MaskDataJson,
        Output::DenoiseMask,
        Output::DenoisedOutput,
    ];
}
