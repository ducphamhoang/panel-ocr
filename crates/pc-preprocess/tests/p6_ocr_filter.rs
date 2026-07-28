//! Task P6 (pc-preprocess half) — spec §9.3 step 7, §9.7(A)8, §9.7(A)9, §14.9,
//! §16.8 items 3-7, 10-12. FROZEN.

mod common;

use common::{block, raw_page_with_image, synthetic_base, unpadded_config};
use pc_config::PreprocessorConfig;
use pc_core::{Language, Rect, StageError};
use pc_ocr::{MockOcrEngine, MockOcrFactory, OcrEngineFactory};
use pc_preprocess::{compile_blacklist, scale_to_original, PreprocessInput};

/// The §6 default blacklist.
const DEFAULT_BLACKLIST: &str = "[～．ー！？０-９~.!?0-9-]*";

/// A 200x200 page with unpadded boxes, so every rect a test writes down is the rect the
/// OCR pass sees (§16.8 item 11).
fn ocr_config() -> PreprocessorConfig {
    unpadded_config()
}

fn ocr_input(blocks: Vec<pc_core::DetectedBlock>, config: PreprocessorConfig) -> PreprocessInput {
    PreprocessInput {
        schema_version: pc_core::SCHEMA_VERSION,
        page: raw_page_with_image(synthetic_base(200, 200), 1.0, blocks),
        config,
        performing_ocr: false,
    }
}

fn run_with(
    input: PreprocessInput,
    factory: &dyn OcrEngineFactory,
) -> pc_preprocess::PreprocessOutput {
    pc_preprocess::run(input, Some(factory)).expect("preprocessing succeeds")
}

// ------------------------------------------------------------ blacklist regex (§9.7(A)8)

#[test]
fn a8_the_default_blacklist_drops_punctuation_and_keeps_real_text() {
    // spec §9.7(A)8: "", "...", "！？", "123" are all FULL matches (dropped);
    // "こんにちは", "A1", "1 2" are not (kept). The empty string matching confirms `*`
    // semantics under a full match.
    let blacklist = compile_blacklist(DEFAULT_BLACKLIST).expect("the §6 default compiles");

    for dropped in ["", "...", "！？", "123", "～．ー", "1-2"] {
        assert!(
            blacklist.is_match(dropped),
            "`{dropped}` must fully match the blacklist"
        );
    }
    for kept in ["こんにちは", "A1", "1 2", "12a"] {
        assert!(
            !blacklist.is_match(kept),
            "`{kept}` must NOT fully match the blacklist"
        );
    }
}

#[test]
fn a8_the_blacklist_is_anchored_at_both_ends() {
    // A partial match must not drop a box: "1!x" contains blacklisted characters but is
    // not blacklisted text.
    let blacklist = compile_blacklist(DEFAULT_BLACKLIST).expect("compiles");

    assert!(!blacklist.is_match("1!x"));
    assert!(!blacklist.is_match("x1!"));
}

#[test]
fn a8_the_blacklist_matches_with_dotall_semantics() {
    // spec §9.3 step 7 / §16.8 item 5: the full match uses `(?s)`, so a `.` in the
    // pattern also spans the newlines multi-line OCR output contains.
    let blacklist = compile_blacklist(".*").expect("compiles");

    assert!(blacklist.is_match("first\nsecond"));
    assert!(blacklist.is_match("\n"));
}

#[test]
fn a8_an_uncompilable_blacklist_is_invalid_input() {
    // spec §16.8 item 6.
    let error = compile_blacklist("[unterminated").expect_err("must not compile");

    assert!(
        matches!(error, StageError::InvalidInput(_)),
        "got {error:?}"
    );
}

// ------------------------------------------------------------ coordinate scaling (§9.7(A)9)

#[test]
fn a9_removed_boxes_are_reported_in_original_image_coordinates() {
    // spec §9.7(A)9: with scale = 0.5 a box at (10,10,30,30) is recorded as
    // (20,20,60,60). 0.5 is dyadic, so 1.0/0.5 is exactly 2.0 and the arithmetic is
    // exact.
    assert_eq!(
        scale_to_original(Rect::new(10, 10, 30, 30), 0.5),
        Rect::new(20, 20, 60, 60)
    );
    assert_eq!(
        scale_to_original(Rect::new(10, 10, 30, 30), 1.0),
        Rect::new(10, 10, 30, 30)
    );
}

#[test]
fn a9_an_unusable_page_scale_falls_back_to_an_unscaled_report() {
    // spec §16.8 item 10: `Rect::scale`'s `as i32` would saturate to i32::MAX on an
    // infinite factor, which would silently corrupt the OCR report.
    let rect = Rect::new(10, 10, 30, 30);

    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(scale_to_original(rect, scale), rect, "scale = {scale}");
    }
}

// ------------------------------------------------------------ the pass, end to end

#[test]
fn a8_blacklisted_boxes_are_dropped_and_others_survive() {
    // spec §9.3 step 7 through `run`: the engine's text decides which boxes stay.
    let blocks = vec![
        block(Rect::new(10, 10, 40, 40), Some(Language::Japanese)),
        block(Rect::new(60, 10, 90, 40), Some(Language::Japanese)),
        block(Rect::new(110, 10, 140, 40), Some(Language::Japanese)),
    ];
    // All-Japanese page => right-to-left, so the engine sees x=110, then 60, then 10.
    let factory =
        MockOcrFactory::new(MockOcrEngine::new().with_script(["こんにちは", "！？", "text"]));

    let output = run_with(ocr_input(blocks, ocr_config()), &factory);

    assert_eq!(
        common::rects(&output.page.text_boxes),
        vec![Rect::new(110, 10, 140, 40), Rect::new(10, 10, 40, 40)],
        "only the box whose text fully matched the blacklist is removed"
    );
    assert_eq!(factory.engine().calls(), 3);
}

#[test]
fn a9_the_analytic_records_areas_texts_and_original_coordinates() {
    // spec §9.3 step 7 + §9.7(A)9: `num_boxes` is the PRE-removal count, ocred areas
    // cover every box an engine ran on, and removed rects are in original coordinates.
    let blocks = vec![
        block(Rect::new(10, 10, 30, 30), Some(Language::Japanese)),
        block(Rect::new(60, 10, 90, 40), Some(Language::Japanese)),
    ];
    let mut input = ocr_input(blocks, ocr_config());
    // scale 0.5 => original coordinates are exactly doubled.
    input.page.scale = 0.5;
    // Japanese page => right-to-left, so the 900-area box at x=60 is OCR'd first.
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_script(["こんにちは", "！？"]));

    let output = run_with(input, &factory);
    let analytic = output
        .ocr_analytic
        .expect("the pass ran, so the analytic exists");

    assert_eq!(
        analytic.path,
        std::path::PathBuf::from(common::ORIGINAL_PATH)
    );
    assert_eq!(analytic.num_boxes, 2, "num_boxes is counted before removal");
    assert_eq!(analytic.box_areas_ocred, vec![900, 400]);
    assert_eq!(analytic.box_areas_removed, vec![400]);
    assert_eq!(analytic.removed.len(), 1);
    assert_eq!(analytic.removed[0].text, "！？");
    assert_eq!(analytic.removed[0].rect, Rect::new(20, 20, 60, 60));
    assert_eq!(output.page.text_boxes.len(), 1);
}

#[test]
fn boxes_at_or_above_ocr_max_size_are_never_ocred() {
    // spec §9.3 step 7: the candidate test is `rect.area() < ocr_max_size`, strict.
    let config = PreprocessorConfig {
        ocr_max_size: 900,
        ..ocr_config()
    };
    let blocks = vec![
        // area 899 -> candidate
        block(Rect::new(0, 0, 29, 31), Some(Language::Japanese)),
        // area 900 -> exactly the limit, NOT a candidate
        block(Rect::new(40, 0, 70, 30), Some(Language::Japanese)),
    ];
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_response("！？"));

    let output = run_with(ocr_input(blocks, config), &factory);

    assert_eq!(
        factory.engine().calls(),
        1,
        "only the sub-limit box is OCR'd"
    );
    assert_eq!(
        common::rects(&output.page.text_boxes),
        vec![Rect::new(40, 0, 70, 30)],
        "the un-OCR'd box survives; the OCR'd one was blacklisted"
    );
    let analytic = output.ocr_analytic.expect("the pass ran");
    assert_eq!(analytic.num_boxes, 2);
    assert_eq!(analytic.box_areas_ocred, vec![899]);
}

#[test]
fn a_pass_with_no_candidates_still_emits_an_empty_analytic() {
    // spec §9.3 step 7: "If no box is small enough to be a candidate, emit
    // OcrAnalytic { num_boxes: boxes.len(), .. empty vectors }".
    let config = PreprocessorConfig {
        ocr_max_size: 10,
        ..ocr_config()
    };
    let blocks = vec![block(Rect::new(10, 10, 40, 40), Some(Language::Japanese))];
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_response("！？"));

    let output = run_with(ocr_input(blocks, config), &factory);
    let analytic = output.ocr_analytic.expect("the pass ran");

    assert_eq!(factory.engine().calls(), 0);
    assert_eq!(analytic.num_boxes, 1);
    assert!(analytic.box_areas_ocred.is_empty());
    assert!(analytic.box_areas_removed.is_empty());
    assert!(analytic.removed.is_empty());
    assert_eq!(output.page.text_boxes.len(), 1);
}

#[test]
fn engine_failure_keeps_the_box_and_does_not_fail_the_page() {
    // DEVIATION(9) / spec §9.3 step 7: fail-open — never delete content because OCR
    // broke. Upstream would propagate the exception.
    let blocks = vec![block(Rect::new(10, 10, 40, 40), Some(Language::Japanese))];
    let factory = MockOcrFactory::new(MockOcrEngine::new().failing());

    let output = run_with(ocr_input(blocks, ocr_config()), &factory);

    assert_eq!(output.page.text_boxes.len(), 1);
    assert_eq!(factory.engine().calls(), 1);
    let analytic = output.ocr_analytic.expect("the pass ran");
    assert!(
        analytic.box_areas_removed.is_empty() && analytic.removed.is_empty(),
        "a failed recognition removes nothing"
    );
}

#[test]
fn a_box_with_no_engine_for_its_language_is_kept_and_not_counted() {
    // spec §16.8 item 4.
    let blocks = vec![
        block(Rect::new(10, 10, 40, 40), Some(Language::Japanese)),
        block(Rect::new(60, 10, 90, 40), Some(Language::English)),
    ];
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_response("！？"))
        .accepting(vec![Some(Language::Japanese)]);

    let output = run_with(ocr_input(blocks, ocr_config()), &factory);

    assert_eq!(
        factory.engine().calls(),
        1,
        "only the Japanese box is routed"
    );
    assert_eq!(
        common::rects(&output.page.text_boxes),
        vec![Rect::new(60, 10, 90, 40)],
        "the unroutable box survives un-OCR'd"
    );
    let analytic = output.ocr_analytic.expect("the pass ran");
    assert_eq!(analytic.num_boxes, 2);
    assert_eq!(
        analytic.box_areas_ocred,
        vec![900],
        "an unroutable box is not counted as OCR'd"
    );
}

#[test]
fn a_denying_factory_leaves_every_box_in_place() {
    let blocks = vec![
        block(Rect::new(10, 10, 40, 40), Some(Language::Japanese)),
        block(Rect::new(60, 10, 90, 40), Some(Language::English)),
    ];
    let factory = MockOcrFactory::default().denying_all();

    let output = run_with(ocr_input(blocks, ocr_config()), &factory);

    assert_eq!(factory.engine().calls(), 0);
    assert_eq!(output.page.text_boxes.len(), 2);
    let analytic = output.ocr_analytic.expect("the pass ran");
    assert!(analytic.box_areas_ocred.is_empty());
}

#[test]
fn the_engine_receives_a_crop_the_size_of_the_box() {
    // spec §9.3 step 7: "crop base_image to rect". Exclusive x2/y2 (§2.1), so a
    // (10,10)-(40,40) box is a 30x30 crop.
    let blocks = vec![block(Rect::new(10, 10, 40, 40), Some(Language::Japanese))];
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_response("kept"));

    run_with(ocr_input(blocks, ocr_config()), &factory);

    assert_eq!(factory.engine().crops(), vec![(30, 30)]);
}

#[test]
fn the_ocr_pass_does_not_run_when_ocr_is_disabled() {
    // spec §16.8 item 3: `ocr_analytic` is Some iff a factory was supplied AND
    // `ocr_enabled` — an all-empty analytic must not be mistakable for "ran and found
    // nothing".
    let config = PreprocessorConfig {
        ocr_enabled: false,
        ..ocr_config()
    };
    let blocks = vec![block(Rect::new(10, 10, 40, 40), Some(Language::Japanese))];
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_response("！？"));

    let output = run_with(ocr_input(blocks, config), &factory);

    assert!(output.ocr_analytic.is_none());
    assert_eq!(factory.engine().calls(), 0);
    assert_eq!(output.page.text_boxes.len(), 1);
}

#[test]
fn removal_preserves_the_reading_order_of_the_survivors() {
    // spec §9.3 steps 6-7: the pass runs AFTER the sort and only deletes, so the
    // survivors keep their sorted positions and stay index-paired with the extended
    // tier (§2.5).
    let blocks = vec![
        block(Rect::new(10, 10, 40, 40), Some(Language::Japanese)),
        block(Rect::new(60, 10, 90, 40), Some(Language::Japanese)),
        block(Rect::new(110, 10, 140, 40), Some(Language::Japanese)),
    ];
    // Japanese page => right-to-left => x=110 first, then 60, then 10. Blacklist the
    // middle one.
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_script(["keep", "！？", "keep"]));

    let output = run_with(ocr_input(blocks, ocr_config()), &factory);

    assert_eq!(
        common::rects(&output.page.text_boxes),
        vec![Rect::new(110, 10, 140, 40), Rect::new(10, 10, 40, 40)]
    );
    assert_eq!(
        output.page.extended_boxes.len(),
        output.page.text_boxes.len()
    );
    output.page.validate().expect("invariants survive removal");
}

#[test]
fn an_uncompilable_blacklist_fails_the_stage() {
    // spec §16.8 item 6: raised once, before the pass, as InvalidInput.
    let config = PreprocessorConfig {
        ocr_blacklist_pattern: "[unterminated".into(),
        ..ocr_config()
    };
    let blocks = vec![block(Rect::new(10, 10, 40, 40), Some(Language::Japanese))];
    let factory = MockOcrFactory::default();

    let error = pc_preprocess::run(
        ocr_input(blocks, config),
        Some(&factory as &dyn OcrEngineFactory),
    )
    .expect_err("an uncompilable blacklist must fail the stage");

    assert!(
        matches!(error, StageError::InvalidInput(_)),
        "got {error:?}"
    );
}

#[test]
fn performing_ocr_with_strict_language_drops_unknown_boxes_before_the_pass() {
    // spec §9.3 step 2 / §16.8 item 12: the strict-language drop happens at step 2, so
    // the unknown box never reaches the engine at all.
    let config = PreprocessorConfig {
        ocr_strict_language: true,
        ..ocr_config()
    };
    let blocks = vec![
        block(Rect::new(10, 10, 40, 40), Some(Language::Japanese)),
        // Large enough to clear suspicious_box_min_size were it not for the strict rule.
        block(Rect::new(0, 60, 200, 200), None),
    ];
    let mut input = ocr_input(blocks, config);
    input.performing_ocr = true;
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_response("keep"));

    let output = run_with(input, &factory);

    assert_eq!(
        common::rects(&output.page.text_boxes),
        vec![Rect::new(10, 10, 40, 40)]
    );
    assert_eq!(factory.engine().calls(), 1);
}
