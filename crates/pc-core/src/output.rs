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
        todo!()
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
        todo!()
    }

    /// Upstream suffixes, verbatim (§2.8).
    pub fn cache_suffix(self) -> &'static str {
        todo!()
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
