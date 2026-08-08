//! Task D7 -- spec §8.3 steps 6-7, §8.7(A)5-6, §16.5 item 3. FROZEN.

mod common;

use common::{
    memory_input, synthetic_blocks, synthetic_page, synthetic_raw_mask, synthetic_replay_fixture,
    COVERED_RECT, REPLAY_SIZE, UNCOVERED_RECT,
};
use image::{GrayImage, Luma};
use pc_config::TextDetectorConfig;
use pc_core::{Language, Rect, Stage, StageError};
use pc_detect::{DetectStage, MockDetector, ReplayDetector, TextDetector};
use std::path::{Path, PathBuf};

fn replay_detector() -> (tempfile::TempDir, ReplayDetector) {
    let (dir, stem) = synthetic_replay_fixture();
    let detector = ReplayDetector::new(dir.path(), stem);
    (dir, detector)
}

/// resolved 2026-07-28 (spec §16.7, joint-architect decision): this used to be
/// `page_json()`, serializing `output.page` to a JSON string. That is impossible for a
/// memory-mode output — `ImageHandle`'s hand-written `Serialize` (§2.3) deliberately errors
/// on `path.is_none()`, and §4.1 memory mode necessarily yields path-less handles. The
/// properties these three call sites actually assert (stage-wrapper equivalence, run-to-run
/// determinism, determinism under a shared detector) only need a deterministic, comparable
/// representation of the page; JSON was a stand-in for one. `PageDataRaw: PartialEq`
/// compares its structural fields, but its `ImageHandle` fields compare by `path` only;
/// in memory mode both paths are `None`, so `PartialEq` does not compare pixel content.
/// The `a6_replay_image_content_is_byte_identical_across_ten_runs_and_eight_threads` test
/// below covers image-content identity by loading and comparing raw base-image and mask
/// bytes. JSON byte-stability itself stays covered by
/// `detect_output_round_trips_through_json`, which runs in disk mode.
fn page_of(output: &pc_detect::DetectOutput) -> pc_core::PageDataRaw {
    output.page.clone()
}

fn image_content_of(output: &pc_detect::DetectOutput) -> (Vec<u8>, Vec<u8>) {
    let base_image = output
        .page
        .base_image
        .load()
        .expect("base image is loaded in memory")
        .to_rgb8();
    let raw_mask = output
        .page
        .raw_mask
        .load()
        .expect("raw mask is loaded in memory")
        .to_luma8();
    (base_image.into_raw(), raw_mask.into_raw())
}

fn directory_snapshot(path: &Path) -> Vec<PathBuf> {
    let mut entries = std::fs::read_dir(path)
        .expect("directory readable")
        .map(|entry| entry.expect("directory entry readable").path())
        .collect::<Vec<_>>();
    entries.sort();
    entries
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
fn memory_mode_does_not_write_destination_scratch_or_source_parent() {
    let source_dir = tempfile::TempDir::new().expect("source dir");
    let destination_scratch = tempfile::TempDir::new().expect("destination scratch");
    let source_path = source_dir.path().join("source.png");
    synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)
        .save(&source_path)
        .expect("write source image");
    let source_parent_before = directory_snapshot(source_dir.path());

    let mut input = memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1));
    input.source = pc_core::ImageHandle::from_path(&source_path);
    input.original_path = source_path.clone();
    assert!(input.base_image_dest.is_none() && input.raw_mask_dest.is_none());

    let detector = MockDetector::new()
        .with_blocks(synthetic_blocks())
        .with_mask(synthetic_raw_mask());
    let output = pc_detect::run(input, &detector).expect("detection succeeds");

    assert!(output.page.base_image.path.is_none());
    assert!(output.page.raw_mask.path.is_none());
    assert!(directory_snapshot(destination_scratch.path()).is_empty());
    assert_eq!(
        directory_snapshot(source_dir.path()),
        source_parent_before,
        "memory mode must not alter the original image's parent"
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

    // resolved 2026-07-28 (§16.7): memory-mode pages cannot be JSON-serialized
    // (ImageHandle guard); compare the `PageDataRaw` structurally instead.
    assert_eq!(page_of(&direct), page_of(&via_trait));
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
    // spec §8.7(A)6, hand-written half: the emitted page must be stable run to run.
    // resolved 2026-07-28 (§16.7): memory-mode pages cannot be JSON-serialized (ImageHandle
    // guard); compare `PageDataRaw` structurally instead of comparing JSON strings.
    let (_dir, detector) = replay_detector();

    let first = page_of(
        &pc_detect::run(
            memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
            &detector,
        )
        .expect("detection succeeds"),
    );

    for run in 1..10 {
        let next = page_of(
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
    // resolved 2026-07-28 (§16.7): memory-mode pages cannot be JSON-serialized (ImageHandle
    // guard); compare `PageDataRaw` structurally instead of comparing JSON strings.
    let (_dir, detector) = replay_detector();
    let shared: &dyn TextDetector = &detector;

    let expected = page_of(
        &pc_detect::run(
            memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
            shared,
        )
        .expect("detection succeeds"),
    );

    let results: Vec<pc_core::PageDataRaw> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(move || {
                    page_of(
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
    assert!(results.iter().all(|page| *page == expected));
    assert_eq!(detector.calls(), 9);
}

#[test]
fn a6_replay_image_content_is_byte_identical_across_ten_runs_and_eight_threads() {
    let (_dir, detector) = replay_detector();

    let expected_output = pc_detect::run(
        memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
        &detector,
    )
    .expect("detection succeeds");
    let expected = image_content_of(&expected_output);

    for run in 1..10 {
        let output = pc_detect::run(
            memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
            &detector,
        )
        .expect("detection succeeds");
        assert!(image_content_of(&output) == expected, "run {run} diverged");
    }

    let shared: &dyn TextDetector = &detector;
    let results: Vec<(Vec<u8>, Vec<u8>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(move || {
                    let output = pc_detect::run(
                        memory_input(synthetic_page(REPLAY_SIZE.0, REPLAY_SIZE.1)),
                        shared,
                    )
                    .expect("detection succeeds");
                    image_content_of(&output)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker did not panic"))
            .collect()
    });

    assert_eq!(results.len(), 8);
    assert!(
        results.iter().all(|content| content == &expected),
        "concurrent image content diverged"
    );
}

// ------------------------------------------------------------ recorded page (F1)
//
// The `_pending_` in the two test names below is kept verbatim: spec §16.24 item 8 names both
// by that exact identifier, and renaming them here would desync the spec from the suite. The
// `#[ignore]`d `unimplemented!()` bodies are what F1 lifts, not the names.

/// The recorded detector page (§7.2), and the four `DetectInput` knobs the recorder used
/// (`xtask/src/record.rs`) — so a test reproducing its output reproduces its inputs too.
const RECORDED_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";
const RECORDED_ORIGINAL_PATH: &str =
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg";
const RECORDED_HEIGHT_LOWER: u32 = 1000;
const RECORDED_HEIGHT_UPPER: u32 = 4000;

fn recorded_replay_detector() -> ReplayDetector {
    ReplayDetector::new(
        &pc_testkit::paths::recorded_root().join("detector"),
        RECORDED_STEM,
    )
}

/// `DetectInput` over the recorded page. `dests` is `None` for memory mode (§4.1), or the
/// directory the two PNGs should be written into.
fn recorded_input(dests: Option<&Path>) -> pc_detect::DetectInput {
    let page = pc_testkit::paths::recorded(format!("detector/{RECORDED_STEM}.jpg"));
    pc_detect::DetectInput {
        schema_version: pc_core::SCHEMA_VERSION,
        source: pc_core::ImageHandle::from_path(&page),
        original_path: PathBuf::from(RECORDED_ORIGINAL_PATH),
        target_height_lower: RECORDED_HEIGHT_LOWER,
        target_height_upper: RECORDED_HEIGHT_UPPER,
        base_image_dest: dests.map(|dir| dir.join(format!("{RECORDED_STEM}_base.png"))),
        raw_mask_dest: dests.map(|dir| dir.join(format!("{RECORDED_STEM}_raw_mask.png"))),
        min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
        config: TextDetectorConfig::default(),
    }
}

/// The committed `#raw.json`, deserialized WITHOUT rebasing — its two handle paths stay relative
/// to the fixtures root, which is the form §7.2 requires them to be committed in and therefore
/// the form the reproduction below has to match.
fn committed_recorded_page() -> pc_core::PageDataRaw {
    let path = pc_testkit::paths::recorded(format!("detector/{RECORDED_STEM}#raw.json"));
    let bytes = std::fs::read(&path).expect("committed #raw.json is readable");
    serde_json::from_slice(&bytes).expect("committed #raw.json parses as PageDataRaw")
}

#[test]
fn a6_pending_recorded_page_equality_and_determinism() {
    // spec §8.7(A)6 / §16.20 item 1(b): this test is TWO hand-written parts: (i) the
    // `PageDataRaw` JSON is byte-identical across 10 runs and across 1 vs 8 rayon
    // threads — runs are compared against each other, which never needed a snapshot;
    // and (ii) one hand-written equality asserts that the produced `PageDataRaw`
    // equals the committed `#raw.json`, which locks strictly more than a snapshot
    // would (every field: `confidence`, `language`, `mask_coverage`, `scale`,
    // `image_size`, schema shape).
    // §16.20 item 2 caveat: `ReplayDetector` ignores the image it is passed (§7.2.1),
    // and `crates/pc-detect/src/lib.rs` copies `rect`/`class_index`/`confidence` out
    // of the fixture untouched — so the values this test actually computes are
    // `mask_coverage`, the survivor set, `scale` and `image_size`. The fixture's own
    // correctness is gated separately by §16.20 item 3's committed-oracle review,
    // NOT here.
    //
    // §16.7 item 4 amends part (i)'s mechanism: identity is checked STRUCTURALLY on
    // `PageDataRaw` in memory mode, because a memory-mode page has path-less handles that
    // §2.3 refuses to serialize. Part (ii) needs the handles, so it runs in disk mode into a
    // temp tree shaped like the fixtures root, and `relativize_page_data_raw` — the recorder's
    // own inverse (§16.24 item 13) — puts the paths back into committed form.
    let detector = recorded_replay_detector();

    // ── part (i): determinism ────────────────────────────────────────────────
    let first = pc_detect::run(recorded_input(None), &detector).expect("detection succeeds");
    let expected = page_of(&first);

    for run in 1..10 {
        let page = page_of(&pc_detect::run(recorded_input(None), &detector).expect("run"));
        assert_eq!(page, expected, "run {run} diverged");
    }

    let shared: &dyn TextDetector = &detector;
    for threads in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        let pages: Vec<pc_core::PageDataRaw> = pool.install(|| {
            use rayon::prelude::*;
            (0..8)
                .into_par_iter()
                .map(|_| page_of(&pc_detect::run(recorded_input(None), shared).expect("run")))
                .collect()
        });
        assert_eq!(pages.len(), 8);
        assert!(
            pages.iter().all(|page| *page == expected),
            "diverged under {threads} rayon threads"
        );
    }
    // Memory mode is what makes the comparison above structural rather than textual; assert it,
    // because `PageDataRaw`'s `PartialEq` compares handles by `path` and two `None`s comparing
    // equal is only meaningful once it is known both sides are path-less by construction.
    assert_eq!(expected.base_image.path, None);
    assert_eq!(expected.raw_mask.path, None);

    // ── part (ii): equality with the committed fixture ───────────────────────
    let temp = tempfile::TempDir::new().expect("temp dir");
    let out_dir = temp.path().join("recorded/detector");
    std::fs::create_dir_all(&out_dir).expect("temp fixture tree");

    let output = pc_detect::run(recorded_input(Some(&out_dir)), &detector).expect("run");
    let mut produced = output.page.clone();
    let residue = pc_testkit::paths::relativize_page_data_raw(&mut produced, temp.path());
    assert_eq!(
        residue,
        Vec::<PathBuf>::new(),
        "paths outside the temp root"
    );

    assert_eq!(produced, committed_recorded_page());

    // `ImageHandle`'s `PartialEq` compares `path` only (§2.3), so the equality above says
    // nothing about the two PNGs. Compare their bytes directly — they are the artifacts a
    // wrong `PAD_VALUE`, a mis-cropped mask or a changed encoder would corrupt, and the
    // committed copies are what every downstream golden reads.
    for suffix in ["_base.png", "_raw_mask.png"] {
        let produced_bytes =
            std::fs::read(out_dir.join(format!("{RECORDED_STEM}{suffix}"))).expect("written png");
        let committed_bytes = std::fs::read(pc_testkit::paths::recorded(format!(
            "detector/{RECORDED_STEM}{suffix}"
        )))
        .expect("committed png");
        assert!(
            produced_bytes == committed_bytes,
            "{suffix} differs from the committed fixture ({} vs {} bytes)",
            produced_bytes.len(),
            committed_bytes.len()
        );
    }
}

#[test]
fn b9_pending_recorded_page_regression_lock() {
    // spec §8.7(B)9: box count and each box's coordinates must match the recorded
    // `#raw.json` exactly. A regression lock against our own recorded output, NOT a
    // Python-parity claim (§15.1).
    //
    // The expected coordinates are written out as literals rather than read from the fixture:
    // a comparison against the file it is meant to lock would pass for any file (cookbook rule
    // 7). The fixture is then asserted to agree with the same literals, so the lock binds BOTH
    // the freshly produced page and the committed one — if they ever disagree, the two
    // assertions name which side moved.
    const EXPECTED_RECTS: [Rect; 3] = [
        Rect {
            x1: 674,
            y1: 1397,
            x2: 740,
            y2: 1438,
        },
        Rect {
            x1: 567,
            y1: 74,
            x2: 663,
            y2: 123,
        },
        Rect {
            x1: 607,
            y1: 631,
            x2: 724,
            y2: 703,
        },
    ];
    // §8.3 step 4's class -> language mapping, per surviving box, in the same order.
    const EXPECTED_LANGUAGES: [Option<Language>; 3] = [
        Some(Language::English),
        Some(Language::English),
        Some(Language::Japanese),
    ];

    let detector = recorded_replay_detector();
    let output = pc_detect::run(recorded_input(None), &detector).expect("detection succeeds");

    // The pre-filter fixture has 4 blocks and exactly one is dropped by §8.3 step 6, so both
    // counts are real assertions and not the length of whatever came back. `blocks_detected` is
    // the replayed pre-filter list; `blocks_kept` is the survivor set this stage computes.
    assert_eq!(output.analytics.blocks_detected, 4);
    assert_eq!(output.analytics.blocks_kept, 3);
    assert_eq!(output.page.blocks.len(), 3);

    assert_eq!(
        output
            .page
            .blocks
            .iter()
            .map(|block| block.rect)
            .collect::<Vec<_>>(),
        EXPECTED_RECTS.to_vec()
    );
    assert_eq!(
        output
            .page
            .blocks
            .iter()
            .map(|block| block.language)
            .collect::<Vec<_>>(),
        EXPECTED_LANGUAGES.to_vec()
    );

    let committed = committed_recorded_page();
    assert_eq!(
        committed
            .blocks
            .iter()
            .map(|block| block.rect)
            .collect::<Vec<_>>(),
        EXPECTED_RECTS.to_vec(),
        "the committed #raw.json no longer carries the locked boxes"
    );
    assert_eq!(committed.blocks.len(), 3);
    assert_eq!(committed.image_size, (1200, 1660));
    assert_eq!(committed.scale, 1.0);
}
