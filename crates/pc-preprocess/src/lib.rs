//! `pc-preprocess` — STAGE 2, preprocessing (spec §9).
//!
//! Turns the detector's [`PageDataRaw`] into the masker's [`PageData`]: filter the
//! detected blocks, resolve their overlaps, sort them into reading order, optionally
//! discard the ones OCR says are punctuation noise, then build the three box tiers
//! (tight → extended → masking regions).
//!
//! Everything here is pure geometry over `pc-core` types; the only external resource is
//! the injected [`OcrEngineFactory`] (§3's `Ctx`). Nothing in this crate touches a model.
//!
//! `run()` and the OCR pass are `todo!()` skeletons for tasks P5/P6; their signatures
//! are frozen with the tests. `filter`/`merge`/`order` and the padding tiers are
//! implemented — they are fully specified in §9.3 and hand-traceable.

pub mod filter;
pub mod merge;
pub mod ocr_filter;
pub mod order;

pub use filter::{apply_page_language, assign_languages, filter_boxes, page_language};
pub use merge::{resolve_overlaps, resolve_total_overlaps};
pub use ocr_filter::{compile_blacklist, run_ocr_pass, scale_to_original, OcrPassResult};
pub use order::{is_right_to_left, sort_key, sort_reading_order};

use pc_config::PreprocessorConfig;
use pc_core::{OcrAnalytic, PageData, PageDataRaw, Rect, StageError, Step};
use pc_ocr::OcrEngineFactory;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessInput {
    pub schema_version: u32,
    pub page: PageDataRaw,
    pub config: PreprocessorConfig,
    /// `true` for `panel-ocr ocr` runs. Per spec §16.8 item 12 it gates exactly one
    /// rule — §9.3 step 2's strict-language drop — and nothing else.
    pub performing_ocr: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessOutput {
    pub page: PageData,
    /// spec §16.8 item 3: `Some` iff the OCR pass actually ran, i.e. a factory was
    /// supplied **and** `config.ocr_enabled`.
    pub ocr_analytic: Option<OcrAnalytic>,
}

/// spec §3 — the stage contract. `Ctx` is the optionally-injected OCR factory.
pub struct PreprocessStage;

impl pc_core::Stage for PreprocessStage {
    type Input = PreprocessInput;
    type Output = PreprocessOutput;
    type Ctx<'a> = Option<&'a dyn OcrEngineFactory>;
    const STEP: Step = Step::Preprocess;

    fn run(input: Self::Input, ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError> {
        run(input, ctx)
    }
}

// ------------------------------------------------------------ padding tiers (§9.3 steps 5, 8, 10)

/// spec §9.3 step 5 — the *tight* tier: `pad(box_padding_initial)` then
/// `right_pad(box_right_padding_initial)`, both clamped to the canvas.
pub fn pad_tight(rect: Rect, config: &PreprocessorConfig, canvas: (u32, u32)) -> Rect {
    rect.pad(config.box_padding_initial, canvas)
        .right_pad(config.box_right_padding_initial, canvas)
}

/// spec §9.3 step 8 — the *extended* tier, derived from an already-tight-padded rect:
/// `pad(box_padding_extended)` then `right_pad(box_right_padding_extended)`.
pub fn extend(tight: Rect, config: &PreprocessorConfig, canvas: (u32, u32)) -> Rect {
    tight
        .pad(config.box_padding_extended, canvas)
        .right_pad(config.box_right_padding_extended, canvas)
}

/// spec §9.3 step 10 — the *reference* box of a merged masking rect:
/// `pad(box_reference_padding)`.
pub fn reference_of(masking: Rect, config: &PreprocessorConfig, canvas: (u32, u32)) -> Rect {
    masking.pad(config.box_reference_padding, canvas)
}

// ------------------------------------------------------------ run (§9.3 steps 1-11)

/// spec §9.3, steps 1–11, in exactly that order.
///
/// Validates both ends per spec §16.8 item 8: `input.page.validate()` at entry and
/// `PageData::validate()` on the assembled output, each surfaced as
/// `StageError::InvalidInput`. An empty page is a success, not an error (§16.8 item 9).
pub fn run(
    input: PreprocessInput,
    ocr: Option<&dyn OcrEngineFactory>,
) -> Result<PreprocessOutput, StageError> {
    let _ = (input, ocr);
    todo!("task P5: spec §9.3 steps 1-11")
}
