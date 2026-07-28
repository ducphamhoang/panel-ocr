//! Task **G1** — checkpoint read/write (spec §4.3, §4.4, §16.12 item 13).
//!
//! FROZEN (CLAUDE.md).

mod common;

use pc_core::{
    DetectedBlock, ImageHandle, Language, MaskData, PageDataRaw, Rect, StageError, SCHEMA_VERSION,
};
use pc_pipeline::checkpoint;
use std::path::Path;

fn page_on_disk(dir: &Path) -> PageDataRaw {
    let base = common::write_page(dir, "page01_base.png", (32, 32));
    let mask = common::write_page(dir, "page01_raw_mask.png", (32, 32));
    PageDataRaw {
        schema_version: SCHEMA_VERSION,
        original_path: dir.join("page01.png"),
        base_image: ImageHandle::from_path(&base),
        raw_mask: ImageHandle::from_path(&mask),
        scale: 1.0,
        image_size: (32, 32),
        blocks: vec![DetectedBlock {
            rect: Rect::new(4, 4, 20, 20),
            language: Some(Language::Japanese),
            confidence: 0.875,
            mask_coverage: 0.5,
        }],
    }
}

/// §4.3: the detector's `PageDataRaw` survives a `#raw.json` round-trip unchanged.
/// Compared structurally, per §16.7 item 2.
#[test]
fn page_raw_round_trips_through_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let page = page_on_disk(dir.path());
    let path = dir.path().join("uuid_page01#raw.json");

    checkpoint::write_page_raw(&page, &path).expect("write");
    let loaded = checkpoint::read_page_raw(&path).expect("read");

    assert_eq!(loaded, page);
}

/// §2.3 + §16.12 item 13: a Memory-mode handle must never reach a checkpoint, and the
/// pre-flight gives the *named* error rather than an opaque serde failure.
#[test]
fn writing_an_unmaterialized_handle_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut page = page_on_disk(dir.path());
    page.base_image = ImageHandle::from_memory(image::DynamicImage::new_rgb8(4, 4));

    let error = checkpoint::write_page_raw(&page, &dir.path().join("x#raw.json")).unwrap_err();

    assert!(
        matches!(error, StageError::UnmaterializedHandle),
        "expected UnmaterializedHandle, got {error:?}"
    );
    assert!(!dir.path().join("x#raw.json").exists());
}

/// §4.4: an unknown `schema_version` is a per-image error, not a panic.
#[test]
fn an_unknown_schema_version_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let mut page = page_on_disk(dir.path());
    page.schema_version = 99;
    let path = dir.path().join("uuid_page01#raw.json");
    checkpoint::write_json(&page, &path).unwrap();

    let error = checkpoint::read_page_raw(&path).unwrap_err();

    assert!(
        matches!(error, StageError::InvalidInput(ref message) if message.contains("99")),
        "error must name the offending version, got {error:?}"
    );
    assert!(checkpoint::SUPPORTED_SCHEMA_VERSIONS.contains(&SCHEMA_VERSION));
}

/// §4.4: a checkpoint whose images have been deleted is a per-image error.
#[test]
fn a_checkpoint_referencing_missing_images_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let page = page_on_disk(dir.path());
    let path = dir.path().join("uuid_page01#raw.json");
    checkpoint::write_page_raw(&page, &path).unwrap();
    std::fs::remove_file(page.raw_mask.path.as_ref().unwrap()).unwrap();

    let error = checkpoint::read_page_raw(&path).unwrap_err();

    assert!(
        matches!(error, StageError::Io { .. }),
        "expected an Io error naming the missing file, got {error:?}"
    );
}

/// §2.6: `MaskData` gets the same treatment as the page checkpoints.
#[test]
fn mask_data_round_trips_and_validates_its_handles() {
    let dir = tempfile::tempdir().unwrap();
    let base = common::write_page(dir.path(), "base.png", (16, 16));
    let combined = common::write_page(dir.path(), "combined.png", (16, 16));
    let mask_data = MaskData {
        schema_version: SCHEMA_VERSION,
        original_path: dir.path().join("page01.png"),
        base_image: ImageHandle::from_path(&base),
        combined_mask: ImageHandle::from_path(&combined),
        scale: 1.0,
        regions: Vec::new(),
    };
    let path = dir.path().join("uuid_page01#mask_data.json");

    checkpoint::write_mask_data(&mask_data, &path).unwrap();
    let loaded = checkpoint::read_mask_data(&path).unwrap();

    assert_eq!(loaded.original_path, mask_data.original_path);
    assert_eq!(loaded.base_image.path, mask_data.base_image.path);
    assert_eq!(loaded.combined_mask.path, mask_data.combined_mask.path);
    assert_eq!(loaded.regions.len(), 0);
}

/// The parent directory is created on demand — the pipeline may write a checkpoint
/// before anything else has touched the cache dir.
#[test]
fn write_json_creates_the_parent_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deep").join("nested").join("m.json");

    checkpoint::write_json(&vec![1_u32, 2, 3], &path).unwrap();

    let loaded: Vec<u32> = checkpoint::read_json(&path).unwrap();
    assert_eq!(loaded, vec![1, 2, 3]);
}
