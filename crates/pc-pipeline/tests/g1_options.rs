//! Task **G1** — run options: skip levels, requested outputs, checkpointing mode,
//! thread budget, dest builders (spec §4.1, §4.4, §4.5, §16.12 items 5, 6, 14–17).
//!
//! FROZEN (CLAUDE.md).

mod common;

use pc_core::{Output, Step};
use pc_pipeline::cache::CachePaths;
use pc_pipeline::options::{
    requested_outputs, resolve_threads, select_checkpointing, Checkpointing, PipelineOptions,
    SaveOnly, SkipFlags,
};
use pc_pipeline::single::{cache_paths_for, denoise_dests, detect_dests, mask_dests};
use std::path::Path;

/// §16.12 item 6: skipping step *n* implies every earlier step.
#[test]
fn skip_flags_normalise_to_a_prefix() {
    let only_mask = SkipFlags {
        mask: true,
        ..SkipFlags::default()
    };

    let normalised = only_mask.normalized();

    assert!(normalised.text_detection && normalised.preprocess && normalised.mask);
    assert!(!normalised.denoise);
    assert!(only_mask.implies_more());
    assert!(!normalised.implies_more());
}

/// §16.12 item 6's start-step table.
#[test]
fn start_step_follows_the_skip_level() {
    let none = SkipFlags::default();
    assert_eq!(none.start_step(), Step::Detect);

    assert_eq!(
        SkipFlags {
            text_detection: true,
            ..none
        }
        .start_step(),
        Step::Preprocess
    );
    assert_eq!(
        SkipFlags {
            preprocess: true,
            ..none
        }
        .start_step(),
        Step::Mask
    );
    assert_eq!(SkipFlags { mask: true, ..none }.start_step(), Step::Denoise);
}

/// §16.12 item 5: `--skip-denoise` is *disable*, not a resume level — it must not move
/// the start step, and it must turn `denoising_enabled` off.
#[test]
fn skip_denoise_disables_the_stage_without_moving_the_start_step() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = common::options(dir.path(), dir.path());
    assert!(options.denoising_enabled());

    options.skips.denoise = true;

    assert_eq!(options.start_step(), Step::Detect);
    assert!(!options.denoising_enabled());
}

/// §16.12 item 5: a profile that disables denoising does so regardless of the flag.
#[test]
fn a_profile_can_disable_denoising_on_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = common::options(dir.path(), dir.path());
    options.profile.denoiser.denoising_enabled = false;

    assert!(!options.denoising_enabled());
}

/// §16.12 item 16 — the default run requests all three categories, text only with
/// `--extract-text`.
#[test]
fn requested_outputs_follow_the_category_map() {
    assert_eq!(
        requested_outputs(None, false),
        vec![
            Output::MaskedOutput,
            Output::DenoisedOutput,
            Output::FinalMask,
            Output::DenoiseMask
        ]
    );
    assert_eq!(
        requested_outputs(None, true),
        vec![
            Output::MaskedOutput,
            Output::DenoisedOutput,
            Output::FinalMask,
            Output::DenoiseMask,
            Output::IsolatedText
        ]
    );
    assert_eq!(
        requested_outputs(Some(SaveOnly::Mask), true),
        vec![Output::FinalMask, Output::DenoiseMask]
    );
    assert_eq!(
        requested_outputs(Some(SaveOnly::Cleaned), false),
        vec![Output::MaskedOutput, Output::DenoisedOutput]
    );
    assert_eq!(
        requested_outputs(Some(SaveOnly::Text), false),
        vec![Output::IsolatedText]
    );
}

/// §4.1 / §16.12 item 14 — Memory needs all three conditions.
#[test]
fn memory_mode_needs_one_image_no_debug_and_no_cache() {
    assert_eq!(select_checkpointing(1, false, true), Checkpointing::Memory);
    assert_eq!(select_checkpointing(2, false, true), Checkpointing::Disk);
    assert_eq!(select_checkpointing(1, true, true), Checkpointing::Disk);
    assert_eq!(select_checkpointing(1, false, false), Checkpointing::Disk);
    assert_eq!(Checkpointing::default(), Checkpointing::Disk);
}

/// §4.5 / §16.12 item 17 — `0` means all cores, and the budget never exceeds the number
/// of images or drops below 1.
#[test]
fn thread_budget_is_bounded_by_the_image_count() {
    assert_eq!(resolve_threads(8, 3), 3);
    assert_eq!(resolve_threads(2, 10), 2);
    assert_eq!(resolve_threads(4, 0), 1);
    assert!(resolve_threads(0, 64) >= 1);
}

/// §4.1: every destination is `None` in Memory mode, so nothing is persisted.
#[test]
fn memory_mode_produces_no_destinations() {
    assert_eq!(detect_dests(None), (None, None));
    assert_eq!(mask_dests(None, true, true), pc_mask::MaskDests::default());
    assert_eq!(
        denoise_dests(None, true),
        pc_denoise::DenoiseDests::default()
    );
}

/// §10.2 / §16.12 item 15 — the debug trio is gated, `_text.png` needs `--extract-text`.
#[test]
fn disk_mode_destinations_are_gated_by_their_flags() {
    let cache = CachePaths::from_parts(Path::new("/cache"), "page01", uuid::Uuid::nil());

    let plain = mask_dests(Some(&cache), false, false);
    assert_eq!(
        plain.combined_mask,
        Some(cache.for_output(Output::FinalMask))
    );
    assert_eq!(plain.cleaned, Some(cache.for_output(Output::MaskedOutput)));
    assert_eq!(plain.text_layer, None);
    assert_eq!(plain.box_mask, None);
    assert_eq!(plain.cut_mask, None);
    assert_eq!(plain.mask_overlay, None);

    let full = mask_dests(Some(&cache), true, true);
    assert_eq!(
        full.text_layer,
        Some(cache.for_output(Output::IsolatedText))
    );
    assert_eq!(full.box_mask, Some(cache.for_output(Output::BoxMask)));
    assert_eq!(full.cut_mask, Some(cache.for_output(Output::CutMask)));
    assert_eq!(
        full.mask_overlay,
        Some(cache.for_output(Output::MaskOverlay))
    );
}

/// §16.12 item 5 — a disabled denoiser writes no denoise artifacts at all.
#[test]
fn denoise_destinations_vanish_when_denoising_is_off() {
    let cache = CachePaths::from_parts(Path::new("/cache"), "page01", uuid::Uuid::nil());

    let on = denoise_dests(Some(&cache), true);
    assert_eq!(on.noise_mask, Some(cache.for_output(Output::DenoiseMask)));
    assert_eq!(on.denoised, Some(cache.for_output(Output::DenoisedOutput)));

    assert_eq!(
        denoise_dests(Some(&cache), false),
        pc_denoise::DenoiseDests::default()
    );
}

/// §16.12 item 8: a fresh run mints a uuid; a resume without cached artifacts is an
/// error rather than a silent fresh start.
#[test]
fn cache_paths_resolution_follows_the_start_step() {
    let dir = tempfile::tempdir().unwrap();
    let original = common::write_page(dir.path(), "page01.png", (8, 8));
    let cache_dir = dir.path().join("cache");
    let mut options = common::options(&cache_dir, dir.path());

    let fresh = cache_paths_for(&original, &options).unwrap().unwrap();
    assert_eq!(fresh.stem(), "page01");

    options.checkpointing = Checkpointing::Memory;
    assert!(cache_paths_for(&original, &options).unwrap().is_none());

    options.checkpointing = Checkpointing::Disk;
    options.skips = SkipFlags {
        text_detection: true,
        ..SkipFlags::default()
    };
    assert!(cache_paths_for(&original, &options).is_err());
}

/// §4.1: debug outputs are a Disk-mode feature.
#[test]
fn debug_outputs_are_inactive_in_memory_mode() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = PipelineOptions {
        debug_outputs: true,
        ..common::options(dir.path(), dir.path())
    };
    assert!(options.debug_outputs_active());

    options.checkpointing = Checkpointing::Memory;
    assert!(!options.debug_outputs_active());
}
