//! Tasks **D9/E5** — long-strip split orchestration and merged export
//! (spec §13 row 26, §12.6, §12.7(B)11, §16.6 item 8, §16.12 item 10).
//!
//! FROZEN (CLAUDE.md).

mod common;

use pc_config::GeneralConfig;
use pc_core::{ImageHandle, Output, SCHEMA_VERSION};
use pc_export::ExportSources;
use pc_pipeline::cache::CachePaths;
use pc_pipeline::strip::{
    merged_strip_export, plan_and_write_segments, read_manifest, should_split, split_params,
};
use uuid::Uuid;

/// §16.6 item 8's aspect gate, in `[general]` terms.
#[test]
fn only_tall_narrow_images_are_split() {
    let general = GeneralConfig::default();

    assert!(should_split((1000, 8000), &general));
    assert!(!should_split((1000, 1000), &general));
    assert!(!should_split((4000, 1000), &general));

    let disabled = GeneralConfig {
        split_long_strips: false,
        ..GeneralConfig::default()
    };
    assert!(!should_split((1000, 8000), &disabled));
}

/// The `[general]` → `SplitParams` mapping (`pc-imageops` takes no `pc-config`).
#[test]
fn split_params_mirror_the_general_section() {
    let general = GeneralConfig::default();

    let params = split_params(&general);

    assert_eq!(params.preferred_height, general.preferred_split_height);
    assert_eq!(params.tolerance_margin, general.split_tolerance_margin);
    assert_eq!(params.split_long_strips, general.split_long_strips);
    assert_eq!(params.max_aspect_ratio, general.long_strip_aspect_ratio);
}

/// §8.7(B)8 + §16.12 item 10: `long_strip.jpg` (1000×8000) splits into 4 segments whose
/// heights sum to the original, named `{uuid}_{stem}_seg{NNN}.png`, with a manifest that
/// round-trips.
#[test]
fn the_long_strip_splits_into_four_segments_and_a_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let cache = CachePaths::from_parts(dir.path(), "long_strip", Uuid::nil());
    let general = GeneralConfig::default();

    let manifest = plan_and_write_segments(&pc_testkit::paths::long_strip(), &cache, &general)
        .expect("planning must succeed")
        .expect("the long strip qualifies for splitting");

    assert_eq!(manifest.image_size, (1000, 8000));
    assert_eq!(manifest.split_rows.len(), 3);
    assert_eq!(manifest.segments.len(), 4);
    assert_eq!(manifest.schema_version, SCHEMA_VERSION);

    let mut total_height = 0;
    for (index, path) in manifest.segments.iter().enumerate() {
        assert_eq!(path, &cache.segment(index));
        let (width, height) = image::image_dimensions(path).unwrap();
        assert_eq!(width, 1000);
        total_height += height;
    }
    assert_eq!(total_height, 8000);

    let reloaded = read_manifest(&cache.splits_manifest()).unwrap();
    assert_eq!(reloaded, manifest);
}

/// §16.6 item 8's gate again, at the orchestration level: a square page is not split
/// and nothing is written.
#[test]
fn a_normal_page_is_not_split() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (400, 400));
    let cache = CachePaths::from_parts(dir.path(), "page01", Uuid::nil());

    let manifest = plan_and_write_segments(&page, &cache, &GeneralConfig::default()).unwrap();

    assert!(manifest.is_none());
    assert!(!cache.splits_manifest().exists());
    assert!(!cache.segment(0).exists());
}

/// §12.7(B)11 — split → (pass-through mock pipeline) → stitch → export reproduces the
/// original 1000×8000 image exactly. Task E5.
#[test]
fn merged_strip_export_reproduces_the_original_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();
    let out = dir.path().join("out");
    let cache = CachePaths::from_parts(&cache_dir, "long_strip", Uuid::nil());
    let general = GeneralConfig::default();
    let manifest = plan_and_write_segments(&pc_testkit::paths::long_strip(), &cache, &general)
        .unwrap()
        .unwrap();

    // The "mock pipeline": every segment's cleaned artifact is the segment itself.
    let sources: Vec<ExportSources> = manifest
        .segments
        .iter()
        .map(|path| ExportSources {
            masked: Some(ImageHandle::from_path(path)),
            ..ExportSources::default()
        })
        .collect();

    let mut options = common::options(&cache_dir, &out);
    options.profile.general.preferred_file_type = ".png".to_string();
    options.save_only = Some(pc_pipeline::SaveOnly::Cleaned);
    options.skips.denoise = true;

    let exported = merged_strip_export(&manifest, &sources, &options).expect("export");

    assert_eq!(exported.files_written.len(), 1);
    let written = &exported.files_written[0];
    assert_eq!(
        written.file_name().unwrap().to_string_lossy(),
        "long_strip_clean.png"
    );
    assert_eq!(image::image_dimensions(written).unwrap(), (1000, 8000));

    let expected = image::open(pc_testkit::paths::long_strip())
        .unwrap()
        .to_rgb8();
    let actual = image::open(written).unwrap().to_rgb8();
    assert_eq!(
        actual.as_raw(),
        expected.as_raw(),
        "§12.7(B)11: a pass-through pipeline must round-trip the strip exactly"
    );
}

/// §16.12 item 10: the manifest suffix is pipeline-local, never an `Output` variant.
#[test]
fn the_manifest_suffix_is_not_an_output_variant() {
    let cache = CachePaths::from_parts(std::path::Path::new("/cache"), "s", Uuid::nil());

    let manifest = cache.splits_manifest();

    assert!(manifest.to_string_lossy().ends_with("#splits.json"));
    assert!(!Output::ALL
        .iter()
        .any(|output| manifest.to_string_lossy().ends_with(output.cache_suffix())));
}
