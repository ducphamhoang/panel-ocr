//! §16.14 item 1 — §4.3's `[split?]` box, wired into the live pipeline.
//!
//! `strip.rs`'s pieces were already unit-tested by `d9_strip.rs`; these tests are the
//! *integration* half the review found missing: that a default-configured run of a long
//! strip actually splits, that a normal page is untouched by the new branch, and that a
//! split image is still exactly ONE row in the batch summary.
//!
//! FROZEN (CLAUDE.md).

mod common;

use pc_core::Rect;
use pc_pipeline::cache::CachePaths;
use pc_pipeline::{Checkpointing, ImageOutcome, PipelineCtx, SharedDetector};
use std::path::Path;

/// Every `#splits.json` in `cache_dir`, and every `_seg{NNN}.png`.
fn split_artifacts(cache_dir: &Path) -> (Vec<String>, Vec<String>) {
    let mut manifests = Vec::new();
    let mut segments = Vec::new();
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(pc_pipeline::SPLITS_SUFFIX) {
                manifests.push(name);
            } else if name.contains(pc_pipeline::SEGMENT_INFIX) && name.ends_with(".png") {
                // Only the segment images themselves, not the per-segment stage artifacts
                // (`..._seg000_clean.png` and friends).
                if name
                    .rsplit_once(pc_pipeline::SEGMENT_INFIX)
                    .is_some_and(|(_, tail)| tail.len() == "000.png".len())
                {
                    segments.push(name);
                }
            }
        }
    }
    manifests.sort();
    segments.sort();
    (manifests, segments)
}

/// A profile whose strip settings are the shipped defaults, with denoising off so the
/// test exercises the split wiring rather than NLM's runtime.
fn strip_options(cache: &Path, out: &Path) -> pc_pipeline::PipelineOptions {
    let mut options = common::options(cache, out);
    options.profile.general.preferred_file_type = ".png".to_string();
    options.save_only = Some(pc_pipeline::SaveOnly::Cleaned);
    options.skips.denoise = true;
    options
}

/// §4.3 + §13 row 26 + §16.14 item 1: with `general.split_long_strips` at its **default**
/// (`true`), a 1000x8000 strip is cut into segments before stage 1 — `#splits.json` and
/// the four `_seg{NNN}.png` files exist — and the stitched export is the full strip.
///
/// This is the regression test for the review finding: before §16.14 item 1 the split
/// module had no live caller, so a default `clean` run produced none of these artifacts.
#[test]
fn a_long_strip_is_split_by_a_default_run() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let strip = pc_testkit::paths::long_strip();
    let options = strip_options(&cache, &out);
    assert!(
        options.profile.general.split_long_strips,
        "the default profile must have splitting on, or this test proves nothing"
    );

    let provider = SharedDetector::new(common::empty_detector());
    let outcome =
        pc_pipeline::process_image_with_splitting(&strip, &options, &PipelineCtx::new(&provider));

    assert!(!outcome.is_failed(), "{outcome:?}");
    let (manifests, segments) = split_artifacts(&cache);
    assert_eq!(manifests.len(), 1, "exactly one manifest: {manifests:?}");
    assert_eq!(
        segments.len(),
        4,
        "§8.7(B)8: 8000/2000 plans 3 splits => 4 segments, got {segments:?}"
    );

    // The manifest is readable and describes the whole strip.
    let entry = CachePaths::discover(&cache, &strip).unwrap().unwrap();
    let manifest = pc_pipeline::strip::read_manifest(&entry.splits_manifest()).unwrap();
    assert_eq!(manifest.original, strip);
    assert_eq!(manifest.image_size, (1000, 8000));
    assert_eq!(manifest.segments.len(), 4);

    // §12.7(B)11 / §16.11 item 1: one export, named after the ORIGINAL strip, full size.
    let written = outcome.files_written();
    assert_eq!(written.len(), 1, "one merged export: {written:?}");
    assert_eq!(
        written[0].file_name().unwrap().to_string_lossy(),
        "long_strip_clean.png"
    );
    assert_eq!(image::image_dimensions(&written[0]).unwrap(), (1000, 8000));
}

/// §16.14 item 1's collapse rule: a split image is ONE `ImageOutcome` for the original
/// input file, never one per segment — so the batch summary and the exit code still count
/// user inputs.
#[test]
fn a_split_image_collapses_to_one_outcome_per_original_input() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let page = common::write_page(dir.path(), "page01.png", (64, 64));
    let strip = pc_testkit::paths::long_strip();
    let options = strip_options(&cache, &out);

    let provider = SharedDetector::new(common::empty_detector());
    let summary = pc_pipeline::run_batch(
        &[strip.clone(), page.clone()],
        &options,
        &PipelineCtx::new(&provider),
    );

    assert_eq!(
        summary.outcomes.len(),
        2,
        "one row per ORIGINAL input, not per segment: {:?}",
        summary
            .outcomes
            .iter()
            .map(|outcome| outcome.original().clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(summary.outcomes[0].original(), &strip);
    assert_eq!(summary.outcomes[1].original(), &page);
    assert_eq!(summary.failed(), 0, "{}", summary.render());
    // No segment path ever surfaces as an outcome's `original`.
    for outcome in &summary.outcomes {
        assert!(
            !outcome
                .original()
                .to_string_lossy()
                .contains(pc_pipeline::SEGMENT_INFIX),
            "a segment leaked into the summary: {outcome:?}"
        );
    }
}

/// §16.6 item 8's aspect gate at the live-pipeline level: a normal page goes down exactly
/// the pre-§16.14 path — no manifest, no segments, and the same outcome
/// [`pc_pipeline::process_image`] gives.
#[test]
fn a_normal_page_is_untouched_by_the_split_branch() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let page = common::write_page(dir.path(), "page01.png", (64, 64));
    let provider = SharedDetector::new(common::detector_with_block(Rect::new(8, 8, 48, 48)));

    let split_cache = dir.path().join("cache-split");
    let split_options = strip_options(&split_cache, &out);
    let with_branch = pc_pipeline::process_image_with_splitting(
        &page,
        &split_options,
        &PipelineCtx::new(&provider),
    );

    let plain_cache = dir.path().join("cache-plain");
    let plain_options = strip_options(&plain_cache, &out);
    let without_branch =
        pc_pipeline::process_image(&page, &plain_options, &PipelineCtx::new(&provider));

    assert!(
        matches!(with_branch, ImageOutcome::Completed { .. }),
        "{with_branch:?}"
    );
    assert_eq!(
        with_branch.files_written(),
        without_branch.files_written(),
        "the split branch must not change a normal page's exports"
    );
    let (manifests, segments) = split_artifacts(&split_cache);
    assert!(manifests.is_empty(), "{manifests:?}");
    assert!(segments.is_empty(), "{segments:?}");
}

/// `general.split_long_strips = false` really turns the branch off, even for a strip that
/// would otherwise qualify.
#[test]
fn the_config_key_disables_splitting() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let strip = pc_testkit::paths::long_strip();
    let mut options = strip_options(&cache, &out);
    options.profile.general.split_long_strips = false;

    let provider = SharedDetector::new(common::empty_detector());
    let outcome =
        pc_pipeline::process_image_with_splitting(&strip, &options, &PipelineCtx::new(&provider));

    assert!(!outcome.is_failed(), "{outcome:?}");
    let (manifests, segments) = split_artifacts(&cache);
    assert!(manifests.is_empty(), "{manifests:?}");
    assert!(segments.is_empty(), "{segments:?}");
}

/// §16.14 item 1's Memory-mode carve-out: segments have to be materialised, so
/// `Checkpointing::Memory` processes the strip whole. It must still export (never a silent
/// drop) and must still write nothing to the cache dir.
#[test]
fn memory_mode_processes_the_strip_whole_and_still_exports() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let strip = pc_testkit::paths::long_strip();
    let mut options = strip_options(&cache, &out);
    options.checkpointing = Checkpointing::Memory;

    let provider = SharedDetector::new(common::empty_detector());
    let outcome =
        pc_pipeline::process_image_with_splitting(&strip, &options, &PipelineCtx::new(&provider));

    assert!(!outcome.is_failed(), "{outcome:?}");
    assert!(
        !outcome.files_written().is_empty(),
        "memory mode still exports: {outcome:?}"
    );
    assert!(
        !cache.exists() || std::fs::read_dir(&cache).unwrap().next().is_none(),
        "memory mode must not populate the cache dir"
    );
}
