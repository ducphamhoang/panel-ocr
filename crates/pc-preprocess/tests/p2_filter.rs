//! Task P2 — spec §9.3 steps 1–3, §9.7(A)1. FROZEN.

mod common;

use common::{block, languages, text_box};
use pc_config::{OcrLanguageSetting, PreprocessorConfig};
use pc_core::{Language, Rect};
use pc_preprocess::{apply_page_language, assign_languages, filter_boxes, page_language};

/// `w x h` at the origin, so `area()` is exactly `w * h`.
fn sized(width: i32, height: i32) -> Rect {
    Rect::new(0, 0, width, height)
}

// ------------------------------------------------------------ step 1: language assignment

#[test]
fn detect_modes_keep_each_blocks_detected_language() {
    // spec §9.3 step 1: `detect_box`/`detect_page` keep the detector's per-block call,
    // including "unknown".
    let blocks = vec![
        block(sized(20, 20), Some(Language::Japanese)),
        block(sized(20, 20), Some(Language::English)),
        block(sized(20, 20), None),
    ];

    for setting in [
        OcrLanguageSetting::DetectBox,
        OcrLanguageSetting::DetectPage,
    ] {
        let boxes = assign_languages(&blocks, setting);
        assert_eq!(
            languages(&boxes),
            vec![Some(Language::Japanese), Some(Language::English), None],
            "{setting:?} must not overwrite detected languages at step 1"
        );
    }
}

#[test]
fn pinned_settings_overwrite_every_blocks_language() {
    // spec §9.3 step 1: "Otherwise: overwrite EVERY block's language" — including the
    // blocks the detector marked unknown.
    let blocks = vec![
        block(sized(20, 20), Some(Language::English)),
        block(sized(20, 20), None),
    ];

    let japanese = assign_languages(&blocks, OcrLanguageSetting::Jpn);
    assert_eq!(
        languages(&japanese),
        vec![Some(Language::Japanese), Some(Language::Japanese)]
    );

    let english = assign_languages(&blocks, OcrLanguageSetting::Eng);
    assert_eq!(
        languages(&english),
        vec![Some(Language::English), Some(Language::English)]
    );
}

#[test]
fn language_assignment_preserves_rects_and_detector_order() {
    let blocks = vec![
        block(Rect::new(10, 10, 30, 30), None),
        block(Rect::new(50, 50, 90, 90), Some(Language::Japanese)),
    ];

    let boxes = assign_languages(&blocks, OcrLanguageSetting::DetectBox);

    assert_eq!(common::rects(&boxes), vec![blocks[0].rect, blocks[1].rect]);
}

// ------------------------------------------------------------ step 2: size filters (§9.7(A)1)

#[test]
fn a1_size_filters_are_strict_and_the_suspicious_rule_only_hits_unknown_boxes() {
    // spec §9.7(A)1: areas 399, 400, 39_999(lang=None), 40_000(lang=None) with
    // box_min_size=400 and suspicious_box_min_size=40_000 => exactly the 400 box (with
    // a language) and the 40_000 box survive. `<` is strict in both filters.
    let config = PreprocessorConfig::default();
    assert_eq!(config.box_min_size, 400);
    assert_eq!(config.suspicious_box_min_size, 40_000);

    let boxes = vec![
        // 399: below box_min_size even though it has a language.
        text_box(sized(399, 1), Some(Language::Japanese)),
        // 400: exactly box_min_size, strict `<` keeps it.
        text_box(sized(400, 1), Some(Language::Japanese)),
        // 39_999 unknown: below suspicious_box_min_size.
        text_box(sized(39_999, 1), None),
        // 40_000 unknown: exactly suspicious_box_min_size, strict `<` keeps it.
        text_box(sized(40_000, 1), None),
    ];

    let kept = filter_boxes(boxes, &config, false);

    assert_eq!(
        kept.iter().map(|b| b.rect.area()).collect::<Vec<_>>(),
        vec![400, 40_000]
    );
    assert_eq!(languages(&kept), vec![Some(Language::Japanese), None]);
}

#[test]
fn a1_a_small_box_with_a_language_survives_the_suspicious_rule() {
    // The suspicious-size rule is conditioned on `language.is_none()`; a known-language
    // box of the same area must survive it.
    let config = PreprocessorConfig::default();
    let boxes = vec![
        text_box(sized(39_999, 1), Some(Language::English)),
        text_box(sized(39_999, 1), None),
    ];

    let kept = filter_boxes(boxes, &config, false);

    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].language, Some(Language::English));
}

#[test]
fn filtering_preserves_detector_order() {
    // spec §9.3 step 2: "in detector order, keeping order".
    let config = PreprocessorConfig::default();
    let boxes = vec![
        text_box(Rect::new(900, 0, 940, 40), Some(Language::Japanese)),
        text_box(Rect::new(0, 0, 5, 5), Some(Language::Japanese)),
        text_box(Rect::new(100, 0, 140, 40), Some(Language::English)),
    ];

    let kept = filter_boxes(boxes, &config, false);

    assert_eq!(
        common::rects(&kept),
        vec![Rect::new(900, 0, 940, 40), Rect::new(100, 0, 140, 40)]
    );
}

// ------------------------------------------------------------ step 2: strict language (§16.8 item 12)

#[test]
fn strict_language_drops_unknown_boxes_only_while_performing_ocr() {
    // spec §9.3 step 2 rule 1 + §16.8 item 12: all three conditions must hold.
    let config = PreprocessorConfig {
        ocr_strict_language: true,
        ..PreprocessorConfig::default()
    };
    // Large enough that neither size rule can be the reason it disappears.
    let boxes = vec![
        text_box(sized(400, 200), None),
        text_box(sized(400, 200), Some(Language::Japanese)),
    ];

    let performing = filter_boxes(boxes.clone(), &config, true);
    assert_eq!(languages(&performing), vec![Some(Language::Japanese)]);

    let not_performing = filter_boxes(boxes.clone(), &config, false);
    assert_eq!(
        languages(&not_performing),
        vec![None, Some(Language::Japanese)],
        "performing_ocr == false must leave the unknown box alone"
    );

    let lenient = PreprocessorConfig {
        ocr_strict_language: false,
        ..config
    };
    let lenient_kept = filter_boxes(boxes, &lenient, true);
    assert_eq!(
        languages(&lenient_kept),
        vec![None, Some(Language::Japanese)],
        "ocr_strict_language == false must leave the unknown box alone"
    );
}

// ------------------------------------------------------------ step 3: page language

#[test]
fn page_language_is_the_mode_of_the_known_languages() {
    // spec §9.3 step 3.
    let boxes = vec![
        text_box(sized(20, 20), Some(Language::English)),
        text_box(sized(20, 20), Some(Language::Japanese)),
        text_box(sized(20, 20), Some(Language::Japanese)),
        text_box(sized(20, 20), None),
    ];

    assert_eq!(page_language(&boxes), Some(Language::Japanese));
}

#[test]
fn page_language_ties_are_broken_by_first_occurrence() {
    // spec §9.3 step 3: Python's `Counter.most_common(1)` is insertion-order stable, so
    // a 1-1 tie resolves to whichever language was SEEN first — not to a fixed enum
    // ordering. Both orderings are asserted so an accidental `max_by_key` (which
    // returns the LAST maximum) cannot pass.
    let english_first = vec![
        text_box(sized(20, 20), Some(Language::English)),
        text_box(sized(20, 20), Some(Language::Japanese)),
    ];
    assert_eq!(page_language(&english_first), Some(Language::English));

    let japanese_first = vec![
        text_box(sized(20, 20), Some(Language::Japanese)),
        text_box(sized(20, 20), Some(Language::English)),
    ];
    assert_eq!(page_language(&japanese_first), Some(Language::Japanese));
}

#[test]
fn page_language_is_none_when_no_box_has_one() {
    assert_eq!(page_language(&[]), None);
    assert_eq!(
        page_language(&[text_box(sized(20, 20), None), text_box(sized(20, 20), None)]),
        None
    );
}

#[test]
fn detect_page_overwrites_every_box_language_and_the_other_settings_do_not() {
    // spec §9.3 step 3, second half.
    let original = vec![
        text_box(sized(20, 20), Some(Language::Japanese)),
        text_box(sized(20, 20), None),
        text_box(sized(20, 20), Some(Language::English)),
    ];

    let mut detect_page = original.clone();
    apply_page_language(
        &mut detect_page,
        OcrLanguageSetting::DetectPage,
        Some(Language::Japanese),
    );
    assert_eq!(
        languages(&detect_page),
        vec![
            Some(Language::Japanese),
            Some(Language::Japanese),
            Some(Language::Japanese)
        ]
    );

    for setting in [
        OcrLanguageSetting::DetectBox,
        OcrLanguageSetting::Jpn,
        OcrLanguageSetting::Eng,
    ] {
        let mut untouched = original.clone();
        apply_page_language(&mut untouched, setting, Some(Language::Japanese));
        assert_eq!(
            languages(&untouched),
            languages(&original),
            "{setting:?} must not rewrite per-box languages at step 3"
        );
    }
}

#[test]
fn detect_page_propagates_an_unknown_page_language_to_every_box() {
    // The page language can legitimately be `None` (no box had one); `detect_page` then
    // makes that unanimous rather than leaving a mixture behind.
    let mut boxes = vec![
        text_box(sized(20, 20), None),
        text_box(sized(20, 20), Some(Language::English)),
    ];

    apply_page_language(&mut boxes, OcrLanguageSetting::DetectPage, None);

    assert_eq!(languages(&boxes), vec![None, None]);
}
