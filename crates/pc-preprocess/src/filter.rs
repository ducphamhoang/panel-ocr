//! Task P2 — spec §9.3 steps 1–3: language assignment/override, the two size filters,
//! and the page-language mode.

use pc_config::{OcrLanguageSetting, PreprocessorConfig};
use pc_core::{DetectedBlock, Language, TextBox};

/// spec §9.3 step 1. `detect_box`/`detect_page` keep each block's detected language;
/// the pinned settings overwrite **every** block's language.
///
/// This is also the `DetectedBlock -> TextBox` conversion: `confidence` and
/// `mask_coverage` are detector analytics (§2.4) and do not survive into `PageData`.
pub fn assign_languages(blocks: &[DetectedBlock], setting: OcrLanguageSetting) -> Vec<TextBox> {
    let fixed = setting.fixed_language();
    blocks
        .iter()
        .map(|block| TextBox {
            rect: block.rect,
            language: match fixed {
                Some(language) => Some(language),
                None => block.language,
            },
        })
        .collect()
}

/// spec §9.3 step 2 — in detector order, keeping order. All three tests are strict `<`.
///
/// `performing_ocr` is the `panel-ocr ocr` flag; per spec §16.8 item 12 it gates the
/// first rule and nothing else.
pub fn filter_boxes(
    boxes: Vec<TextBox>,
    config: &PreprocessorConfig,
    performing_ocr: bool,
) -> Vec<TextBox> {
    boxes
        .into_iter()
        .filter(|text_box| {
            let unknown = text_box.language.is_none();
            let area = text_box.rect.area();

            if performing_ocr && unknown && config.ocr_strict_language {
                return false;
            }
            if area < config.box_min_size {
                return false;
            }
            if unknown && area < config.suspicious_box_min_size {
                return false;
            }
            true
        })
        .collect()
}

/// spec §9.3 step 3 — the mode of the known languages, ties broken by **first
/// occurrence** (Python `Counter.most_common(1)` is insertion-order stable). `None` when
/// no box has a language.
pub fn page_language(boxes: &[TextBox]) -> Option<Language> {
    let mut counts: Vec<(Language, usize)> = Vec::new();
    for language in boxes.iter().filter_map(|text_box| text_box.language) {
        match counts.iter_mut().find(|(seen, _)| *seen == language) {
            Some(entry) => entry.1 += 1,
            // Insertion order IS the tie-break order; do not sort this.
            None => counts.push((language, 1)),
        }
    }

    let mut best: Option<(Language, usize)> = None;
    for (language, count) in counts {
        // Strictly greater, so the first-seen language wins a tie.
        if best.is_none_or(|(_, best_count)| count > best_count) {
            best = Some((language, count));
        }
    }
    best.map(|(language, _)| language)
}

/// spec §9.3 step 3, second half: when `ocr_language == detect_page`, every box takes
/// the page language.
pub fn apply_page_language(
    boxes: &mut [TextBox],
    setting: OcrLanguageSetting,
    page: Option<Language>,
) {
    if setting != OcrLanguageSetting::DetectPage {
        return;
    }
    for text_box in boxes.iter_mut() {
        text_box.language = page;
    }
}
