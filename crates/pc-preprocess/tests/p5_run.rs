//! Task P5 — spec §9.3 steps 5, 8, 10, 11; §9.7(A)6, §9.7(A)7, §9.7(A)10;
//! §16.8 items 8, 9, 13. FROZEN.

mod common;

use common::{block, input, no_ocr, page_json, raw_page, text_box, PAGE_SIZE};
use pc_config::{OcrLanguageSetting, PreprocessorConfig, ReadingOrder};
use pc_core::{Language, Rect, Stage, StageError};
use pc_preprocess::{extend, pad_tight, reference_of, PreprocessStage};

// ------------------------------------------------------------ padding tiers (§9.7(A)6)

#[test]
fn a6_right_padding_clamps_to_the_canvas() {
    // spec §9.7(A)6: a box touching the right edge, right_pad(5) on a 100-wide image
    // yields x2 == 100, never 105.
    let canvas = (100, 100);
    let config = PreprocessorConfig {
        box_padding_initial: 0,
        box_right_padding_initial: 5,
        ..PreprocessorConfig::default()
    };

    let padded = pad_tight(Rect::new(60, 10, 100, 40), &config, canvas);

    assert_eq!(padded.x2, 100);
    assert_eq!(padded, Rect::new(60, 10, 100, 40));
}

#[test]
fn a6_every_tier_clamps_on_all_four_sides() {
    // `Rect::pad` clamps x1/y1 up to 0 and x2/y2 down to the canvas (§2.1), so a box
    // filling the page cannot grow at any tier.
    let canvas = (100, 100);
    let config = PreprocessorConfig::default();
    let full = Rect::new(0, 0, 100, 100);

    assert_eq!(pad_tight(full, &config, canvas), full);
    assert_eq!(extend(full, &config, canvas), full);
    assert_eq!(reference_of(full, &config, canvas), full);
}

#[test]
fn padding_tiers_apply_the_configured_amounts_in_order() {
    // spec §9.3 steps 5, 8, 10 with the §6 defaults:
    //   tight     = pad(2)  then right_pad(3)
    //   extended  = pad(5)  then right_pad(5)
    //   reference = pad(20)
    let config = PreprocessorConfig::default();
    let rect = Rect::new(100, 100, 200, 200);

    let tight = pad_tight(rect, &config, PAGE_SIZE);
    assert_eq!(tight, Rect::new(98, 98, 205, 202));

    let extended = extend(tight, &config, PAGE_SIZE);
    assert_eq!(extended, Rect::new(93, 93, 215, 207));

    let reference = reference_of(extended, &config, PAGE_SIZE);
    assert_eq!(reference, Rect::new(73, 73, 235, 227));
    assert!(reference.contains_rect(&extended));
}

// ------------------------------------------------------------ run(): the three tiers

#[test]
fn run_builds_the_three_box_tiers_from_one_block() {
    // spec §9.3 steps 5, 8, 9, 10 end to end, with the numbers hand-traced above.
    let page = raw_page(vec![block(
        Rect::new(100, 100, 200, 200),
        Some(Language::Japanese),
    )]);

    let output = pc_preprocess::run(input(page, PreprocessorConfig::default()), no_ocr())
        .expect("preprocessing succeeds");

    assert_eq!(
        output.page.text_boxes,
        vec![text_box(
            Rect::new(98, 98, 205, 202),
            Some(Language::Japanese)
        )]
    );
    assert_eq!(
        output.page.extended_boxes,
        vec![Rect::new(93, 93, 215, 207)]
    );
    assert_eq!(output.page.masking_regions.len(), 1);
    assert_eq!(
        output.page.masking_regions[0].masking,
        Rect::new(93, 93, 215, 207)
    );
    assert_eq!(
        output.page.masking_regions[0].reference,
        Rect::new(73, 73, 235, 227)
    );
    assert_eq!(output.page.page_language, Some(Language::Japanese));
}

#[test]
fn run_merges_overlapping_extended_boxes_into_fewer_masking_regions() {
    // spec §9.3 step 9: the extended tier is what gets merged, so two tight boxes that
    // do NOT overlap can still collapse into one masking region.
    let page = raw_page(vec![
        block(Rect::new(100, 100, 200, 200), Some(Language::Japanese)),
        block(Rect::new(196, 100, 296, 200), Some(Language::Japanese)),
    ]);

    let output = pc_preprocess::run(input(page, PreprocessorConfig::default()), no_ocr())
        .expect("preprocessing succeeds");

    assert_eq!(
        output.page.text_boxes.len(),
        2,
        "the tight tier must be untouched by the extended-tier merge"
    );
    assert_eq!(output.page.extended_boxes.len(), 2);
    assert_eq!(
        output.page.masking_regions.len(),
        1,
        "the two extended boxes overlap past the 20% default threshold"
    );
    let region = &output.page.masking_regions[0];
    for extended in &output.page.extended_boxes {
        assert!(
            region.masking.contains_rect(extended),
            "the merged masking rect must bound both extended boxes"
        );
    }
}

#[test]
fn run_applies_the_reading_order_to_the_tight_tier() {
    // spec §9.3 step 6 inside `run`: the emitted `text_boxes` are reading-order sorted,
    // and `extended_boxes` are index-paired with them (§2.5).
    let page = raw_page(vec![
        block(Rect::new(0, 0, 100, 100), Some(Language::Japanese)),
        block(Rect::new(500, 0, 600, 100), Some(Language::Japanese)),
    ]);
    let config = PreprocessorConfig {
        reading_order: ReadingOrder::Manga,
        ..PreprocessorConfig::default()
    };

    let output = pc_preprocess::run(input(page, config), no_ocr()).expect("preprocessing succeeds");

    let xs = output
        .page
        .text_boxes
        .iter()
        .map(|text_box| text_box.rect.x1)
        .collect::<Vec<_>>();
    assert_eq!(xs, vec![498, 0], "right-to-left: the x=500 box comes first");
    assert_eq!(
        output.page.extended_boxes[0].x1,
        output.page.text_boxes[0].rect.x1 - 5,
        "extended_boxes must be paired with text_boxes by index, post-sort"
    );
}

#[test]
fn run_sets_the_page_language_and_honours_detect_page() {
    // spec §9.3 step 3 inside `run`.
    let blocks = vec![
        block(Rect::new(0, 0, 100, 100), Some(Language::Japanese)),
        block(Rect::new(200, 0, 300, 100), Some(Language::Japanese)),
        block(Rect::new(400, 0, 500, 100), Some(Language::English)),
    ];

    let detect_box = pc_preprocess::run(
        input(raw_page(blocks.clone()), PreprocessorConfig::default()),
        no_ocr(),
    )
    .expect("preprocessing succeeds");
    assert_eq!(detect_box.page.page_language, Some(Language::Japanese));
    assert!(detect_box
        .page
        .text_boxes
        .iter()
        .any(|text_box| text_box.language == Some(Language::English)));

    let config = PreprocessorConfig {
        ocr_language: OcrLanguageSetting::DetectPage,
        ..PreprocessorConfig::default()
    };
    let detect_page = pc_preprocess::run(input(raw_page(blocks), config), no_ocr())
        .expect("preprocessing succeeds");
    assert_eq!(detect_page.page.page_language, Some(Language::Japanese));
    assert!(
        detect_page
            .page
            .text_boxes
            .iter()
            .all(|text_box| text_box.language == Some(Language::Japanese)),
        "detect_page must rewrite every box language to the page language"
    );
}

// ------------------------------------------------------------ invariants (§9.7(A)7)

#[test]
fn a7_every_emitted_page_satisfies_the_section_2_5_invariants() {
    // spec §9.7(A)7 / §16.8 item 8: `extended_boxes.len() == text_boxes.len()`, every
    // `reference` contains its `masking`, all rects within `image_size`.
    let page = raw_page(vec![
        // Deliberately flush against all four edges so every padding tier clamps.
        block(Rect::new(0, 0, 100, 100), Some(Language::Japanese)),
        block(Rect::new(900, 900, 1000, 1000), Some(Language::English)),
        block(Rect::new(400, 400, 500, 500), None),
    ]);

    let output = pc_preprocess::run(input(page, PreprocessorConfig::default()), no_ocr())
        .expect("preprocessing succeeds");

    output
        .page
        .validate()
        .expect("emitted PageData must satisfy its own invariants");
    assert_eq!(
        output.page.extended_boxes.len(),
        output.page.text_boxes.len()
    );
    for region in &output.page.masking_regions {
        assert!(region.reference.contains_rect(&region.masking));
    }
    for rect in &output.page.extended_boxes {
        assert!(rect.x1 >= 0 && rect.y1 >= 0);
        assert!(rect.x2 <= PAGE_SIZE.0 as i32 && rect.y2 <= PAGE_SIZE.1 as i32);
    }
}

#[test]
fn run_rejects_an_input_page_whose_blocks_escape_the_canvas() {
    // spec §16.8 item 8: a corrupt `#raw.json` becomes a clean per-image failure, not a
    // debug-only assertion.
    let page = raw_page(vec![block(
        Rect::new(900, 900, 1200, 1200),
        Some(Language::Japanese),
    )]);

    let error = pc_preprocess::run(input(page, PreprocessorConfig::default()), no_ocr())
        .expect_err("an out-of-bounds input page must be rejected");

    assert!(
        matches!(error, StageError::InvalidInput(_)),
        "got {error:?}"
    );
}

#[test]
fn a9_an_empty_page_is_a_success_with_empty_tiers() {
    // spec §16.8 item 9 / §5.6: zero boxes is the pipeline's business, not a StageError.
    for page in [
        raw_page(Vec::new()),
        // Every block filtered away by box_min_size.
        raw_page(vec![block(Rect::new(0, 0, 5, 5), Some(Language::Japanese))]),
    ] {
        let output = pc_preprocess::run(input(page, PreprocessorConfig::default()), no_ocr())
            .expect("an empty page is not an error");

        assert!(output.page.text_boxes.is_empty());
        assert!(output.page.extended_boxes.is_empty());
        assert!(output.page.masking_regions.is_empty());
        assert_eq!(output.page.page_language, None);
        output.page.validate().expect("an empty page is valid");
    }
}

// ------------------------------------------------------------ pass-through (§9.4)

#[test]
fn run_passes_the_page_header_through_untouched() {
    // spec §9.4: base_image, raw_mask, scale, image_size and original_path all travel
    // from `PageDataRaw` to `PageData` unchanged.
    let mut page = raw_page(vec![block(
        Rect::new(100, 100, 200, 200),
        Some(Language::Japanese),
    )]);
    page.scale = 0.5;

    let expected = page.clone();
    let output = pc_preprocess::run(input(page, PreprocessorConfig::default()), no_ocr())
        .expect("preprocessing succeeds");

    assert_eq!(output.page.schema_version, expected.schema_version);
    assert_eq!(output.page.original_path, expected.original_path);
    assert_eq!(output.page.base_image, expected.base_image);
    assert_eq!(output.page.raw_mask, expected.raw_mask);
    assert_eq!(output.page.scale, 0.5);
    assert_eq!(output.page.image_size, expected.image_size);
}

// ------------------------------------------------------------ ocr_analytic gating (§16.8 item 3)

#[test]
fn ocr_analytic_is_absent_when_no_factory_is_supplied() {
    let page = raw_page(vec![block(
        Rect::new(100, 100, 130, 130),
        Some(Language::Japanese),
    )]);

    let output = pc_preprocess::run(input(page, PreprocessorConfig::default()), no_ocr())
        .expect("preprocessing succeeds");

    assert!(output.ocr_analytic.is_none());
}

// ------------------------------------------------------------ stage contract (§3)

#[test]
fn preprocess_stage_matches_the_free_function() {
    let page = raw_page(vec![block(
        Rect::new(100, 100, 200, 200),
        Some(Language::Japanese),
    )]);

    let direct = pc_preprocess::run(input(page.clone(), PreprocessorConfig::default()), no_ocr())
        .expect("preprocessing succeeds");
    let via_trait = PreprocessStage::run(input(page, PreprocessorConfig::default()), no_ocr())
        .expect("preprocessing succeeds");

    assert_eq!(page_json(&direct.page), page_json(&via_trait.page));
    assert_eq!(direct.ocr_analytic, via_trait.ocr_analytic);
    assert_eq!(PreprocessStage::STEP, pc_core::Step::Preprocess);
}

// ------------------------------------------------------------ serde (§3, §16.8 item 13)

#[test]
fn preprocess_input_round_trips_through_json() {
    let page = raw_page(vec![block(
        Rect::new(100, 100, 200, 200),
        Some(Language::Japanese),
    )]);
    let value = input(page, PreprocessorConfig::default());

    let json = serde_json::to_string(&value).expect("PreprocessInput is Serialize");
    let restored: pc_preprocess::PreprocessInput =
        serde_json::from_str(&json).expect("PreprocessInput is Deserialize");

    assert_eq!(restored.schema_version, value.schema_version);
    assert_eq!(restored.config, value.config);
    assert_eq!(restored.performing_ocr, value.performing_ocr);
    assert_eq!(restored.page, value.page);
}

#[test]
fn preprocess_output_round_trips_through_json() {
    let page = raw_page(vec![block(
        Rect::new(100, 100, 200, 200),
        Some(Language::Japanese),
    )]);

    let output = pc_preprocess::run(input(page, PreprocessorConfig::default()), no_ocr())
        .expect("preprocessing succeeds");
    let json = serde_json::to_string(&output).expect("PreprocessOutput is Serialize");
    let restored: pc_preprocess::PreprocessOutput =
        serde_json::from_str(&json).expect("PreprocessOutput is Deserialize");

    assert_eq!(page_json(&restored.page), page_json(&output.page));
    assert_eq!(restored.ocr_analytic, output.ocr_analytic);
}

// ------------------------------------------------------------ determinism (§9.7(A)10)

#[test]
fn a10_run_is_byte_identical_across_one_hundred_iterations() {
    // spec §9.7(A)10: guards against any accidental HashSet/HashMap ordering leaking
    // into the merge passes or the page-language mode.
    // §16.8 item 13: comparing canonical JSON is legal here because the preprocess stage
    // only ever forwards the input's (path-bearing) ImageHandles.
    let blocks = vec![
        block(Rect::new(100, 100, 200, 200), Some(Language::Japanese)),
        block(Rect::new(196, 100, 296, 200), Some(Language::English)),
        block(Rect::new(400, 400, 500, 500), None),
        block(Rect::new(0, 700, 120, 820), Some(Language::English)),
    ];

    let first = page_json(
        &pc_preprocess::run(
            input(raw_page(blocks.clone()), PreprocessorConfig::default()),
            no_ocr(),
        )
        .expect("preprocessing succeeds")
        .page,
    );

    for iteration in 1..100 {
        let next = page_json(
            &pc_preprocess::run(
                input(raw_page(blocks.clone()), PreprocessorConfig::default()),
                no_ocr(),
            )
            .expect("preprocessing succeeds")
            .page,
        );
        assert_eq!(next, first, "iteration {iteration} diverged");
    }
}

// ------------------------------------------------------------ pending F1 (§9.7(B)11)

#[test]
#[ignore = "pending task F1: needs `cargo xtask record-fixtures` output under tests/fixtures/recorded/"]
fn b11_pending_insta_snapshot_of_recorded_page_tiers() {
    // spec §9.7(B)11 / §15.10: the single permitted `assert_json_snapshot!` site for
    // this stage — the recorded page fixture run with a `MockOcrEngine` returning "" for
    // every crop, locking the three box tiers. Safeguards binding on whoever unignores
    // this: (a) the first accepted snapshot must be reviewed against a hand-traced
    // expected value, with reviewer/date/method recorded in docs/GOLDEN_CALIBRATION.md,
    // before commit; (b) the snapshot is frozen exactly like a hand-written test --
    // `cargo insta accept` is forbidden in CI and any change needs joint-architect
    // sign-off; (c) no further snapshot sites may be added without the same §15-style
    // process.
    unimplemented!("blocked on F1");
}
