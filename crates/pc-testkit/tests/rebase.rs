//! C4 tests -- spec §7.2's `ImageHandle` path rebasing. Frozen gates.
//!
//! "The recorded JSON's `ImageHandle` paths are stored **relative to the fixtures
//! root** and rebased on load by `pc-testkit`."
//!
//! Note: `PageDataRaw` and `MaskData` now implement `PartialEq`; these tests compare
//! handle paths field-wise because `ImageHandle` equality is deliberately path-based.

use pc_core::{ImageHandle, MaskData, PageDataRaw, SCHEMA_VERSION};
use pc_testkit::paths;
use std::path::{Path, PathBuf};

fn page_with(base: &str, mask: &str) -> PageDataRaw {
    PageDataRaw {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from("black_bubble_raw.png"),
        base_image: ImageHandle::from_path(base),
        raw_mask: ImageHandle::from_path(mask),
        scale: 1.0,
        image_size: (202, 319),
        blocks: vec![],
    }
}

#[test]
// spec §7.2: relative recorded paths are rebased onto the given root, so the recorded
// JSON stays checkout-location-independent
fn relative_handles_are_rebased_onto_the_root() {
    let root = Path::new("/fixtures");
    let mut page = page_with("recorded/black_base.png", "recorded/black_raw_mask.png");

    paths::rebase_page_data_raw(&mut page, root);

    assert_eq!(
        page.base_image.path.as_deref(),
        Some(Path::new("/fixtures/recorded/black_base.png"))
    );
    assert_eq!(
        page.raw_mask.path.as_deref(),
        Some(Path::new("/fixtures/recorded/black_raw_mask.png"))
    );
}

#[test]
// spec §7.2: an already-absolute path is left alone -- rebasing must be idempotent, so
// loading a rebased structure twice cannot double-prefix it
fn absolute_handles_are_left_alone_and_rebasing_is_idempotent() {
    let root = Path::new("/fixtures");
    let mut page = page_with("recorded/a.png", "/elsewhere/b.png");

    paths::rebase_page_data_raw(&mut page, root);
    let after_once = (page.base_image.path.clone(), page.raw_mask.path.clone());

    paths::rebase_page_data_raw(&mut page, root);
    assert_eq!(
        (page.base_image.path.clone(), page.raw_mask.path.clone()),
        after_once,
        "rebasing must be idempotent"
    );
    assert_eq!(
        page.raw_mask.path.as_deref(),
        Some(Path::new("/elsewhere/b.png"))
    );
}

#[test]
// spec §7.2: MaskData's two handles get the same treatment as PageDataRaw's
fn mask_data_handles_are_rebased() {
    let mut mask = MaskData {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from("black_bubble_raw.png"),
        base_image: ImageHandle::from_path("recorded/black_base.png"),
        combined_mask: ImageHandle::from_path("recorded/black_combined_mask.png"),
        scale: 1.0,
        regions: vec![],
    };

    paths::rebase_mask_data(&mut mask, Path::new("/fixtures"));

    assert_eq!(
        mask.base_image.path.as_deref(),
        Some(Path::new("/fixtures/recorded/black_base.png"))
    );
    assert_eq!(
        mask.combined_mask.path.as_deref(),
        Some(Path::new("/fixtures/recorded/black_combined_mask.png"))
    );
}

#[test]
// spec §7.2: rebasing touches ONLY the handles -- original_path, scale, image_size and
// the block list are recorded data and must survive untouched
fn rebasing_does_not_disturb_other_fields() {
    let mut page = page_with("recorded/a.png", "recorded/b.png");
    paths::rebase_page_data_raw(&mut page, Path::new("/fixtures"));

    assert_eq!(page.original_path, PathBuf::from("black_bubble_raw.png"));
    assert_eq!(page.scale, 1.0);
    assert_eq!(page.image_size, (202, 319));
    assert_eq!(page.schema_version, SCHEMA_VERSION);
    assert!(page.blocks.is_empty());
}
