//! Task P6 (pc-preprocess half) — spec §9.3 step 7: the OCR discard pass and its
//! analytics.
//!
//! [`run_ocr_pass`] implements the full pass (task P6); the signatures are frozen with the
//! tests, as are the two pure-arithmetic pieces ([`compile_blacklist`],
//! [`scale_to_original`]), which are fully specified and hand-checkable.

use pc_config::PreprocessorConfig;
use pc_core::{OcrAnalytic, PageDataRaw, Rect, RemovedBox, StageError, TextBox};
use pc_ocr::OcrEngineFactory;
use regex::Regex;
use std::sync::Arc;

/// What the pass returns: the surviving boxes (order preserved) plus the analytic.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrPassResult {
    pub boxes: Vec<TextBox>,
    pub analytic: OcrAnalytic,
}

/// spec §9.3 step 7 + §16.8 item 5: the blacklist is a **full match** with `(?s)`
/// (DOTALL), so a pattern's `.` also matches a newline in multi-line OCR output.
///
/// Deliberately *not* `PreprocessorConfig::compile_blacklist`: that one (frozen in
/// `pc-config`) omits `(?s)` and exists only as the config-validation compile check.
/// Any pattern that compiles under one compiles under the other.
///
/// A pattern that does not compile is `StageError::InvalidInput` (§16.8 item 6) —
/// config validation normally prevents this, but a hand-built config can reach here.
pub fn compile_blacklist(pattern: &str) -> Result<Regex, StageError> {
    Regex::new(&format!("(?s)^(?:{pattern})$")).map_err(|error| {
        StageError::InvalidInput(format!("ocr_blacklist_pattern does not compile: {error}"))
    })
}

/// spec §9.3 step 7: `RemovedBox.rect` is reported in ORIGINAL-image coordinates, i.e.
/// `rect.scale(1.0 / scale)`.
///
/// §16.8 item 10: a non-finite or non-positive `scale` would saturate `Rect::scale`'s
/// `as i32` cast, so it falls back to a factor of `1.0` with a `WARN`.
pub fn scale_to_original(rect: Rect, scale: f64) -> Rect {
    if !scale.is_finite() || scale <= 0.0 {
        tracing::warn!(
            scale,
            "page scale is not a usable ratio; reporting removed-box coordinates unscaled"
        );
        return rect;
    }
    rect.scale(1.0 / scale)
}

/// spec §9.3 step 7, the whole pass. Called only when a factory was supplied AND
/// `config.ocr_enabled` (§16.8 item 3).
///
/// Contract the frozen tests pin down:
/// * candidates are boxes with `rect.area() < config.ocr_max_size` (strict), judged on
///   the already-padded, already-sorted rects (§16.8 item 11);
/// * `analytic.num_boxes` is `boxes.len()` at entry — before any removal;
/// * a candidate's area goes into `box_areas_ocred` iff an engine actually ran on it;
/// * a full blacklist match drops the box and appends to both `box_areas_removed` and
///   `removed` (the latter in original-image coordinates);
/// * DEVIATION(9): an engine error is logged at `WARN` and the box is **kept**
///   (fail-open — never delete content because OCR broke);
/// * §16.8 item 4: no engine for the box's language ⇒ kept, not OCR'd, not counted;
/// * §16.8 item 7: `page.base_image` is loaded lazily (only if a candidate exists) and a
///   load failure propagates; a rect that does not intersect the canvas is kept and not
///   OCR'd.
pub fn run_ocr_pass(
    boxes: Vec<TextBox>,
    page: &PageDataRaw,
    config: &PreprocessorConfig,
    factory: &dyn OcrEngineFactory,
) -> Result<OcrPassResult, StageError> {
    let blacklist = compile_blacklist(&config.ocr_blacklist_pattern)?;
    let mut analytic = OcrAnalytic {
        path: page.original_path.clone(),
        num_boxes: boxes.len(),
        box_areas_ocred: Vec::new(),
        box_areas_removed: Vec::new(),
        removed: Vec::new(),
    };
    let mut image: Option<Arc<image::DynamicImage>> = None;
    let mut surviving = Vec::with_capacity(boxes.len());

    for text_box in boxes {
        let area = text_box.rect.area();
        if area >= config.ocr_max_size {
            surviving.push(text_box);
            continue;
        }

        let Some(engine) = factory.engine_for(text_box.language) else {
            tracing::debug!(
                language = ?text_box.language,
                "no OCR engine handles the text box language; keeping box"
            );
            surviving.push(text_box);
            continue;
        };

        let Some((x, y, width, height)) = text_box.rect.to_crop(page.image_size) else {
            tracing::debug!(
                rect = ?text_box.rect,
                "text box does not intersect the image canvas; keeping box"
            );
            surviving.push(text_box);
            continue;
        };

        let base_image = match &image {
            Some(image) => Arc::clone(image),
            None => {
                let loaded = page.base_image.load()?;
                image = Some(Arc::clone(&loaded));
                loaded
            }
        };
        let crop = base_image.crop_imm(x, y, width, height);
        analytic.box_areas_ocred.push(area);

        match engine.recognize(&crop) {
            Ok(text) => {
                if blacklist.is_match(&text) {
                    analytic.box_areas_removed.push(area);
                    analytic.removed.push(RemovedBox {
                        text,
                        rect: scale_to_original(text_box.rect, page.scale),
                    });
                } else {
                    surviving.push(text_box);
                }
            }
            Err(error) => {
                tracing::warn!(
                    ?error,
                    rect = ?text_box.rect,
                    "OCR recognition failed; keeping box"
                );
                surviving.push(text_box);
            }
        }
    }

    Ok(OcrPassResult {
        boxes: surviving,
        analytic,
    })
}
