//! Task **G1-chain** — the five-stage single-image orchestration (spec §4.3, §4.4,
//! §5.6, §16.12 items 5 and 12). Heavy task; every test here fails until
//! `single::process_image` is implemented.
//!
//! FROZEN (CLAUDE.md).

mod common;

use pc_core::{Output, Rect, Step};
use pc_pipeline::cache::CachePaths;
use pc_pipeline::{Checkpointing, ImageOutcome, PipelineCtx, SharedDetector, SkipReason};
use std::path::Path;

fn cleaned_export(output_dir: &Path, stem: &str) -> std::path::PathBuf {
    output_dir.join(format!("{stem}_clean.png"))
}

/// §4.3: a page with text runs all five stages and reports `Completed`; §4.3's
/// checkpoint rule means all three JSON checkpoints exist afterwards.
#[test]
fn a_page_with_text_completes_and_leaves_every_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = dir.path().join("in");
    std::fs::create_dir_all(&inputs).unwrap();
    let page = common::write_page(&inputs, "page01.png", (64, 64));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let mut options = common::options(&cache, &out);
    options.profile.general.preferred_file_type = ".png".to_string();

    let provider = SharedDetector::new(common::detector_with_block(Rect::new(8, 8, 48, 48)));
    let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));

    match &outcome {
        ImageOutcome::Completed { files_written, .. } => {
            assert!(
                files_written.iter().all(|path| path.exists()),
                "files_written must list files that exist: {files_written:?}"
            );
            assert!(files_written.contains(&cleaned_export(&out, "page01")));
        }
        other => panic!("expected Completed, got {other:?}"),
    }

    let entry = CachePaths::discover(&cache, &page).unwrap().unwrap();
    for output in [Output::RawJson, Output::CleanJson, Output::MaskDataJson] {
        assert!(
            entry.for_output(output).exists(),
            "missing checkpoint for {output:?}"
        );
    }
    for output in [
        Output::BaseImage,
        Output::RawMask,
        Output::FinalMask,
        Output::MaskedOutput,
        Output::DenoisedOutput,
    ] {
        assert!(
            entry.for_output(output).exists(),
            "missing cache artifact for {output:?}"
        );
    }
}

/// §5.6 / §16.12 item 12: no detected text is a `Skipped`, but the page is still
/// exported — this is exactly what `--detector mock` does in v1 (§16.12 item 2).
#[test]
fn a_page_without_text_is_skipped_but_still_exported() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "empty.png", (48, 48));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let mut options = common::options(&cache, &out);
    options.profile.general.preferred_file_type = ".png".to_string();

    let provider = SharedDetector::new(common::empty_detector());
    let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));

    match &outcome {
        ImageOutcome::Skipped {
            reason,
            files_written,
            ..
        } => {
            assert_eq!(*reason, SkipReason::NoTextDetected);
            assert!(
                files_written.contains(&cleaned_export(&out, "empty")),
                "§5.6: the untouched page is still exported, got {files_written:?}"
            );
        }
        other => panic!("expected Skipped(NoTextDetected), got {other:?}"),
    }
}

/// §5.1: a stage error is a `Failed` naming the step that failed; no panic escapes.
#[test]
fn a_detector_error_fails_the_image_at_the_detect_step() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (32, 32));
    let options = common::options(&dir.path().join("cache"), &dir.path().join("out"));

    let provider = SharedDetector::new(common::failing_detector());
    let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));

    match outcome {
        ImageOutcome::Failed { step, .. } => assert_eq!(step, Step::Detect),
        other => panic!("expected Failed at Detect, got {other:?}"),
    }
}

/// §16.12 item 3: a provider that cannot supply a detector is a per-image failure, not
/// a fatal one.
#[test]
fn a_refusing_detector_provider_fails_only_that_image() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (32, 32));
    let options = common::options(&dir.path().join("cache"), &dir.path().join("out"));

    let provider = common::RefusingProvider;
    let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));

    assert!(outcome.is_failed(), "expected Failed, got {outcome:?}");
}

/// §4.1: Memory mode writes no cache at all, only the requested export.
#[test]
fn memory_mode_writes_exports_but_no_cache() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (48, 48));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let mut options = common::options(&cache, &out);
    options.checkpointing = Checkpointing::Memory;
    options.profile.general.preferred_file_type = ".png".to_string();

    let provider = SharedDetector::new(common::detector_with_block(Rect::new(6, 6, 40, 40)));
    let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));

    assert!(
        !outcome.files_written().is_empty(),
        "memory mode still exports: {outcome:?}"
    );
    assert!(
        !cache.exists() || std::fs::read_dir(&cache).unwrap().next().is_none(),
        "memory mode must not populate the cache dir"
    );
}

/// §16.12 item 5: with denoising disabled, stage 4 never runs — no `_noise_mask.png`,
/// no `_clean_denoised.png` — and the exported cleaned image is the masker's.
#[test]
fn skip_denoise_leaves_no_denoise_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (64, 64));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let mut options = common::options(&cache, &out);
    options.skips.denoise = true;
    options.profile.general.preferred_file_type = ".png".to_string();

    let provider = SharedDetector::new(common::detector_with_block(Rect::new(8, 8, 48, 48)));
    let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));

    assert!(!outcome.is_failed(), "{outcome:?}");
    let entry = CachePaths::discover(&cache, &page).unwrap().unwrap();
    assert!(!entry.for_output(Output::DenoiseMask).exists());
    assert!(!entry.for_output(Output::DenoisedOutput).exists());
}

/// §4.4: `--skip-text-detection` loads `#raw.json` instead of running the detector.
/// The mock's call counter proves the stage really was skipped.
#[test]
fn skip_text_detection_reuses_the_cached_raw_json() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (64, 64));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let mut options = common::options(&cache, &out);
    options.profile.general.preferred_file_type = ".png".to_string();

    let detector = std::sync::Arc::new(
        pc_detect::MockDetector::new()
            .with_blocks(vec![pc_detect::RawBlock {
                rect: Rect::new(8, 8, 48, 48),
                class_index: 1,
                confidence: 0.9,
            }])
            .with_block_fill(255),
    );
    let provider = SharedDetector::new(detector.clone());
    let first = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));
    assert!(!first.is_failed(), "{first:?}");
    assert_eq!(detector.calls(), 1);

    options.skips.text_detection = true;
    let second = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));

    assert!(!second.is_failed(), "{second:?}");
    assert_eq!(
        detector.calls(),
        1,
        "the detector must not run again under --skip-text-detection"
    );
}

/// §4.4: a resume with nothing in the cache is a per-image failure, not a silent
/// fresh run.
#[test]
fn resuming_without_a_cache_entry_fails_the_image() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (32, 32));
    let mut options = common::options(&dir.path().join("cache"), &dir.path().join("out"));
    options.skips.preprocess = true;

    let provider = SharedDetector::new(common::empty_detector());
    let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));

    assert!(outcome.is_failed(), "expected Failed, got {outcome:?}");
}

/// §5.7: identical inputs produce identical outputs — same outcome shape, same exported
/// files, byte-identical cleaned image.
#[test]
fn two_identical_runs_produce_identical_exports() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (64, 64));
    let provider = SharedDetector::new(common::detector_with_block(Rect::new(8, 8, 48, 48)));

    let mut bytes = Vec::new();
    for run in 0..2 {
        let cache = dir.path().join(format!("cache{run}"));
        let out = dir.path().join(format!("out{run}"));
        let mut options = common::options(&cache, &out);
        options.profile.general.preferred_file_type = ".png".to_string();
        let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&provider));
        assert!(!outcome.is_failed(), "{outcome:?}");
        bytes.push(std::fs::read(cleaned_export(&out, "page01")).unwrap());
    }

    assert_eq!(bytes[0], bytes[1], "§5.7: runs must be byte-identical");
}
