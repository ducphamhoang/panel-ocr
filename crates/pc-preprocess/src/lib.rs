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
use pc_core::{MaskingRegion, OcrAnalytic, PageData, PageDataRaw, Rect, StageError, Step};
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
    input.page.validate()?;

    let PreprocessInput {
        page,
        config,
        performing_ocr,
        ..
    } = input;
    let canvas = page.image_size;

    let boxes = assign_languages(&page.blocks, config.ocr_language);
    let mut boxes = filter_boxes(boxes, &config, performing_ocr);
    let page_language = page_language(&boxes);
    apply_page_language(&mut boxes, config.ocr_language, page_language);

    let mut boxes = resolve_total_overlaps(boxes);
    for text_box in &mut boxes {
        text_box.rect = pad_tight(text_box.rect, &config, canvas);
    }
    sort_reading_order(&mut boxes, config.reading_order, page_language);

    let (boxes, ocr_analytic) = match (ocr, config.ocr_enabled) {
        (Some(factory), true) => {
            let result = run_ocr_pass(boxes, &page, &config, factory)?;
            (result.boxes, Some(result.analytic))
        }
        _ => (boxes, None),
    };

    let extended_boxes: Vec<Rect> = boxes
        .iter()
        .map(|text_box| extend(text_box.rect, &config, canvas))
        .collect();
    let masking_regions = resolve_overlaps(extended_boxes.clone(), config.box_overlap_threshold)
        .into_iter()
        .map(|masking| MaskingRegion {
            reference: reference_of(masking, &config, canvas),
            masking,
        })
        .collect();

    let output_page = PageData {
        schema_version: page.schema_version,
        original_path: page.original_path,
        base_image: page.base_image,
        raw_mask: page.raw_mask,
        scale: page.scale,
        image_size: page.image_size,
        page_language,
        text_boxes: boxes,
        extended_boxes,
        masking_regions,
    };
    output_page.validate()?;

    Ok(PreprocessOutput {
        page: output_page,
        ocr_analytic,
    })
}
