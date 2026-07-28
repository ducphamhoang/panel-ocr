//! Task D7 -- spec §8.3 steps 6-7, §8.7(A)5-6, §16.5 item 3. FROZEN.

mod common;

use common::{
    memory_input, synthetic_blocks, synthetic_page, synthetic_raw_mask, synthetic_replay_fixture,
    COVERED_RECT, REPLAY_SIZE, UNCOVERED_RECT,
};
use image::{GrayImage, Luma};
use pc_config::{MaskRefineMode, TextDetectorConfig};
use pc_core::{Language, Rect, Stage, StageError};
use pc_detect::{DetectStage, MockDetector, ReplayDetector, TextDetector};

fn replay_detector() -> (tempfile::TempDir, ReplayDetector) {
    let (dir, stem) = synthetic_replay_fixture();
    let detector = ReplayDetector::new(dir.path(), stem);
    (dir, detector)
}

fn page_json(output: &pc_detect::DetectOutput) -> String {
    serde_json::to_string(&output.page).expect("PageDataRaw is serializable")
}

// ------------------------------------------------------------ mask_coverage (§8.7(A)5)

#[test]
fn a5_coverage_filter_boundary_is_25_vs_26_over_255() {
    // spec §8.7(A)5: mean 25/255 (~0.098) is dropped, 26/255 (~0.102) is kept, at the
    // fixed v1 threshold of 0.1.
    let rect = Rect::new(0, 0, 4, 4);

    let low = GrayImage::from_pixel(4, 4, Luma([25]));
    let high = GrayImage::from_pixel(4, 4, Luma([26]));

    assert!(pc_detect::mask_coverage(&low, rect) < pc_detect::DEFAULT_MIN_MASK_COVERAGE);
    assert!(pc_detect::mask_coverage(&high, rect) >= pc_detect::DEFAULT_MIN_MASK_COVERAGE);
}

#[test]
fn mask_coverage_is_the_mean_over_the_exclusive_rect() {
    // spec §16.6 item 5: x2/y2 exclusive. Half of a 2x4 rect saturated -> 0.5.
    let mask = pc_testkit::images::gray_from_rows(&[
        &[255, 255, 0, 0],
        &[255, 255, 0, 0],
        &[0, 0, 0, 0],
        &[0, 0, 0, 0],
    ]);

    assert_eq!(pc_detect::mask_coverage(&mask, Rect::new(0, 0, 4, 2)), 0.5);
    assert_eq!(pc_detect::mask_coverage(&mask, Rect::new(0, 0, 2, 2)), 1.0);
    assert_eq!(pc_detect::mask_coverage(&mask, Rect::new(2, 2, 4, 4)), 0.0);
}

#[test]
fn mask_coverage_of_a_degenerate_rect_is_zero() {
    let mask = GrayImage::from_pixel(4, 4, Luma([255]));

    assert_eq!(pc_detect::mask_coverage(&mask, Rect::new(2, 2, 2, 2)), 0.0);
    assert_eq!(
        pc_detect::mask_coverage(&mask, Rect::new(10, 10, 20, 20)),
        0.0
    );
}

// ------------------------------------------------------------ run() wiring

#[test]
fn run_drops_uncovered_blocks_and_keeps_covered_ones() {
    // spec §8.3 step 6: the coverage filter runs against the *refined* mask.
    let detector = MockDetector::new()
        .with_blocks(synthetic_blocks())
        .with_mask(synthetic_raw_mask());
    let input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));

    let output = pc_detect::run(input, &detector).expect("detection succeeds");

    assert_eq!(output.page.blocks.len(), 1);
    assert_eq!(output.page.blocks[0].rect, COVERED_RECT);
    assert_eq!(output.analytics.blocks_detected, 2);
    assert_eq!(output.analytics.blocks_kept, 1);
}

#[test]
fn run_preserves_detector_block_order() {
    // spec §8.3 step 7: surviving blocks stay in NMS (detector) order.
    let blocks = vec![
        pc_detect::RawBlock {
            rect: Rect::new(4, 4, 20, 20),
            class_index: 0,
            confidence: 0.9,
        },
        pc_detect::RawBlock {
            rect: Rect::new(36, 4, 52, 20),
            class_index: 1,
            confidence: 0.8,
        },
        pc_detect::RawBlock {
            rect: Rect::new(4, 36, 20, 52),
            class_index: 2,
            confidence: 0.7,
        },
    ];
    let detector = MockDetector::new()
        .with_blocks(blocks.clone())
        .with_block_fill(255);
    let input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));

    let output = pc_detect::run(input, &detector).expect("detection succeeds");

    assert_eq!(
        output
            .page
            .blocks
            .iter()
            .map(|b| b.rect)
            .collect::<Vec<_>>(),
        blocks.iter().map(|b| b.rect).collect::<Vec<_>>()
    );
}

#[test]
fn run_maps_class_index_to_language_per_block() {
    // spec §8.3 step 4: 0 => English, 1 => Japanese, 2 => None.
    let blocks = vec![
        pc_detect::RawBlock {
            rect: Rect::new(4, 4, 20, 20),
            class_index: 0,
            confidence: 0.9,
        },
        pc_detect::RawBlock {
            rect: Rect::new(36, 4, 52, 20),
            class_index: 1,
            confidence: 0.8,
        },
        pc_detect::RawBlock {
            rect: Rect::new(4, 36, 20, 52),
            class_index: 2,
            confidence: 0.7,
        },
    ];
    let detector = MockDetector::new().with_blocks(blocks).with_block_fill(255);
    let input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));

    let output = pc_detect::run(input, &detector).expect("detection succeeds");

    assert_eq!(
        output
            .page
            .blocks
            .iter()
            .map(|block| block.language)
            .collect::<Vec<_>>(),
        vec![Some(Language::English), Some(Language::Japanese), None]
    );
}

#[test]
fn run_rejects_annotation_refine_mode() {
    // spec §16.5 item 3 / §15.2: config accepts `annotation`, the stage rejects it --
    // and must do so BEFORE paying for inference.
    let detector = MockDetector::new().with_blocks(synthetic_blocks());
    let mut input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));
    input.config = TextDetectorConfig {
        mask_refine_mode: MaskRefineMode::Annotation,
        ..TextDetectorConfig::default()
    };

    let error = pc_detect::run(input, &detector).expect_err("annotation mode is v1.5");

    assert!(matches!(error, StageError::InvalidInput(_)));
    assert_eq!(
        detector.calls(),
        0,
        "must fail before invoking the detector"
    );
}

#[test]
fn run_propagates_detector_failure() {
    let detector = MockDetector::new().failing();
    let input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));

    let error = pc_detect::run(input, &detector).expect_err("detector fails");

    assert!(matches!(error, StageError::Inference(_)));
}

#[test]
fn memory_mode_writes_nothing_to_disk() {
    // spec §4.1: with `base_image_dest` / `raw_mask_dest` both None the stage must not
    // touch the filesystem, and the resulting handles carry no path.
    let scratch = tempfile::TempDir::new().expect("temp dir");
    let detector = MockDetector::new()
        .with_blocks(synthetic_blocks())
        .with_mask(synthetic_raw_mask());
    let input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));
    assert!(input.base_image_dest.is_none() && input.raw_mask_dest.is_none());

    let output = pc_detect::run(input, &detector).expect("detection succeeds");

    assert!(output.page.base_image.path.is_none());
    assert!(output.page.raw_mask.path.is_none());
    assert_eq!(
        std::fs::read_dir(scratch.path())
            .expect("scratch dir readable")
            .count(),
        0
    );
}

#[test]
fn disk_mode_writes_both_artifacts_to_their_destinations() {
    let scratch = tempfile::TempDir::new().expect("temp dir");
    let base_dest = scratch.path().join("page_base.png");
    let mask_dest = scratch.path().join("page_raw_mask.png");

    let detector = MockDetector::new()
        .with_blocks(synthetic_blocks())
        .with_mask(synthetic_raw_mask());
    let mut input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));
    input.base_image_dest = Some(base_dest.clone());
    input.raw_mask_dest = Some(mask_dest.clone());

    let output = pc_detect::run(input, &detector).expect("detection succeeds");

    assert!(base_dest.is_file());
    assert!(mask_dest.is_file());
    assert_eq!(
        output.page.base_image.path.as_deref(),
        Some(base_dest.as_path())
    );
    assert_eq!(
        output.page.raw_mask.path.as_deref(),
        Some(mask_dest.as_path())
    );
}

#[test]
fn run_emits_a_page_that_passes_its_own_validate() {
    // spec §16.6 item 4: clipping is what makes this hold for real detector output.
    let detector = MockDetector::new()
        .with_blocks(synthetic_blocks())
        .with_mask(synthetic_raw_mask());
    let input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));

    let output = pc_detect::run(input, &detector).expect("detection succeeds");

    assert_eq!(output.page.image_size, REPLAY_SIZE);
    assert_eq!(output.page.scale, 1.0);
    assert_eq!(output.page.schema_version, pc_core::SCHEMA_VERSION);
    output.page.validate().expect("emitted page must be valid");
}

#[test]
fn run_records_mask_coverage_on_surviving_blocks() {
    let detector = MockDetector::new()
        .with_blocks(synthetic_blocks())
        .with_mask(synthetic_raw_mask());
    let input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));

    let output = pc_detect::run(input, &detector).expect("detection succeeds");

    assert_eq!(output.page.blocks[0].mask_coverage, 1.0);
    assert_ne!(UNCOVERED_RECT, output.page.blocks[0].rect);
}

// ------------------------------------------------------------ Stage conformance

#[test]
fn detect_stage_matches_the_free_function() {
    // spec §3: `Stage::run` is a thin wrapper, not a second code path.
    let detector = MockDetector::new()
        .with_blocks(synthetic_blocks())
        .with_mask(synthetic_raw_mask());

    let direct = pc_detect::run(
        memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
        &detector,
    )
    .expect("detection succeeds");
    let via_trait = DetectStage::run(
        memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
        &detector as &dyn TextDetector,
    )
    .expect("detection succeeds");

    assert_eq!(page_json(&direct), page_json(&via_trait));
    assert_eq!(direct.analytics, via_trait.analytics);
    assert_eq!(DetectStage::STEP, pc_core::Step::Detect);
}

// ------------------------------------------------------------ serde

#[test]
fn detect_output_round_trips_through_json() {
    // Handles must be materialized for a checkpoint write (§2.3), so this test uses the
    // disk-mode destinations.
    let scratch = tempfile::TempDir::new().expect("temp dir");
    let detector = MockDetector::new()
        .with_blocks(synthetic_blocks())
        .with_mask(synthetic_raw_mask());
    let mut input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));
    input.base_image_dest = Some(scratch.path().join("base.png"));
    input.raw_mask_dest = Some(scratch.path().join("mask.png"));

    let output = pc_detect::run(input, &detector).expect("detection succeeds");
    let json = serde_json::to_string(&output).expect("DetectOutput is Serialize");
    let restored: pc_detect::DetectOutput =
        serde_json::from_str(&json).expect("DetectOutput is Deserialize");

    assert_eq!(
        serde_json::to_string(&restored.page).expect("re-serializable"),
        serde_json::to_string(&output.page).expect("re-serializable")
    );
    assert_eq!(restored.analytics, output.analytics);
}

#[test]
fn detect_input_round_trips_through_json() {
    // §3: the Input must be fully serde-derivable so a checkpoint records exactly which
    // settings produced it.
    let scratch = tempfile::TempDir::new().expect("temp dir");
    let source_path = scratch.path().join("source.png");
    synthetic_page(8, 8)
        .save(&source_path)
        .expect("write source");

    let mut input = memory_input(synthetic_page(8, 8));
    input.source = pc_core::ImageHandle::from_path(&source_path);

    let json = serde_json::to_string(&input).expect("DetectInput is Serialize");
    let restored: pc_detect::DetectInput =
        serde_json::from_str(&json).expect("DetectInput is Deserialize");

    assert_eq!(restored.source.path, input.source.path);
    assert_eq!(restored.config, input.config);
    assert_eq!(restored.min_mask_coverage, input.min_mask_coverage);
    assert_eq!(restored.base_image_dest, input.base_image_dest);
}

// ------------------------------------------------------------ determinism (§8.7(A)6)

#[test]
fn a6_replay_run_is_byte_identical_across_ten_runs() {
    // spec §8.7(A)6, hand-written half: the JSON must be byte-stable run to run.
    let (_dir, detector) = replay_detector();

    let first = page_json(
        &pc_detect::run(
            memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
            &detector,
        )
        .expect("detection succeeds"),
    );

    for run in 1..10 {
        let next = page_json(
            &pc_detect::run(
                memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
                &detector,
            )
            .expect("detection succeeds"),
        );
        assert_eq!(next, first, "run {run} diverged");
    }
    assert_eq!(detector.calls(), 10);
}

#[test]
fn a6_replay_run_is_identical_under_concurrent_shared_detector_use() {
    // spec §8.7(A)6 / §4.5: `run()` is single-image and single-threaded; the
    // parallelism lives above it and shares ONE `&dyn TextDetector` across threads.
    let (_dir, detector) = replay_detector();
    let shared: &dyn TextDetector = &detector;

    let expected = page_json(
        &pc_detect::run(
            memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
            shared,
        )
        .expect("detection succeeds"),
    );

    let results: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(move || {
                    page_json(
                        &pc_detect::run(
                            memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
                            shared,
                        )
                        .expect("detection succeeds"),
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker did not panic"))
            .collect()
    });

    assert_eq!(results.len(), 8);
    assert!(results.iter().all(|json| *json == expected));
    assert_eq!(detector.calls(), 9);
}

// ------------------------------------------------------------ pending F1

#[test]
#[ignore = "pending task F1: needs `cargo xtask record-fixtures` output under tests/fixtures/recorded/"]
fn a6_pending_insta_snapshot_of_recorded_page() {
    // spec §8.7(A)6 / §15.10: the single permitted `assert_json_snapshot!` site for
    // this stage. Safeguards binding on whoever unignores this: (a) the first accepted
    // snapshot must be reviewed against a hand-traced expected value, with
    // reviewer/date/method recorded in docs/GOLDEN_CALIBRATION.md, before commit;
    // (b) the snapshot is frozen exactly like a hand-written test -- `cargo insta
    // accept` is forbidden in CI and any change needs joint-architect sign-off;
    // (c) no further snapshot sites may be added without the same §15-style process.
    unimplemented!("blocked on F1");
}

#[test]
#[ignore = "pending task F1: needs the recorded page fixture"]
fn b9_pending_recorded_page_regression_lock() {
    // spec §8.7(B)9: box count and each box's coordinates must match the recorded
    // `#raw.json` exactly. A regression lock against our own recorded output, NOT a
    // Python-parity claim (§15.1).
    unimplemented!("blocked on F1");
}
