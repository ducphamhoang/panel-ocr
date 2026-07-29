//! spec §16.24 item 13 — `relativize_*` as the exact inverse of `rebase_*`. Frozen gates.
//!
//! Companion to `tests/rebase.rs`, which covers the forward direction only.

use pc_core::{ImageHandle, MaskData, PageDataRaw, SCHEMA_VERSION};
use pc_testkit::paths;
use std::path::{Path, PathBuf};

fn page_with(base: &str, mask: &str) -> PageDataRaw {
    PageDataRaw {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from("page01.png"),
        base_image: ImageHandle::from_path(base),
        raw_mask: ImageHandle::from_path(mask),
        scale: 1.0,
        image_size: (1024, 1434),
        blocks: vec![],
    }
}

fn handle_paths(page: &PageDataRaw) -> (Option<PathBuf>, Option<PathBuf>) {
    (page.base_image.path.clone(), page.raw_mask.path.clone())
}

#[test]
// spec §16.24 item 13 / §7.2: `relativize` is the exact inverse of `rebase` on relative input —
// the composition a recorder performs in reverse. Asserts the recovered value equals the
// ORIGINAL, not merely that it is relative: a function returning `PathBuf::new()` for
// everything would satisfy "is relative" and fail here.
fn relativize_undoes_rebase_on_relative_handles() {
    let root = Path::new("/fixtures");
    let original = page_with(
        "recorded/detector/page01_base.png",
        "recorded/detector/page01_raw_mask.png",
    );

    let mut page = original.clone();
    paths::rebase_page_data_raw(&mut page, root);
    assert_eq!(
        handle_paths(&page),
        (
            Some(PathBuf::from("/fixtures/recorded/detector/page01_base.png")),
            Some(PathBuf::from(
                "/fixtures/recorded/detector/page01_raw_mask.png"
            )),
        ),
        "precondition: rebase must actually have changed both handles"
    );

    let residue = paths::relativize_page_data_raw(&mut page, root);
    assert!(residue.is_empty(), "unrelativized: {residue:?}");
    assert_eq!(handle_paths(&page), handle_paths(&original));
}

#[test]
// spec §16.24 item 13: the direction the recorder actually runs. `pc_detect::run` produces
// ABSOLUTE handles (`crates/pc-detect/src/lib.rs:121-128`), so `rebase(relativize(p)) == p` on
// absolute-under-root input is the property that makes a committed `#raw.json` loadable again.
fn rebase_undoes_relativize_on_absolute_handles_under_the_root() {
    let root = Path::new("/fixtures");
    let original = page_with(
        "/fixtures/recorded/detector/page01_base.png",
        "/fixtures/recorded/detector/page01_raw_mask.png",
    );

    let mut page = original.clone();
    let residue = paths::relativize_page_data_raw(&mut page, root);
    assert!(residue.is_empty(), "unrelativized: {residue:?}");
    assert_eq!(
        handle_paths(&page),
        (
            Some(PathBuf::from("recorded/detector/page01_base.png")),
            Some(PathBuf::from("recorded/detector/page01_raw_mask.png")),
        ),
        "precondition: relativize must actually have stripped the root"
    );

    paths::rebase_page_data_raw(&mut page, root);
    assert_eq!(handle_paths(&page), handle_paths(&original));
}

#[test]
// spec §16.20 item 3's closing ¶ — the failure this function exists to prevent. A handle outside
// the fixtures root cannot be relativized, and silence there is precisely the defect: the
// recorder would commit one machine's absolute path. The path is REPORTED and left unchanged, so
// the caller can refuse; asserting both halves is what stops a "fix" that quietly drops it.
fn a_handle_outside_the_root_is_reported_and_left_unchanged() {
    let root = Path::new("/fixtures");
    let mut page = page_with(
        "/fixtures/recorded/detector/page01_base.png",
        "/elsewhere/mask.png",
    );

    let residue = paths::relativize_page_data_raw(&mut page, root);

    assert_eq!(residue, vec![PathBuf::from("/elsewhere/mask.png")]);
    assert_eq!(
        handle_paths(&page),
        (
            Some(PathBuf::from("recorded/detector/page01_base.png")),
            Some(PathBuf::from("/elsewhere/mask.png")),
        ),
        "the relativizable handle is rewritten; the foreign one is untouched, not dropped"
    );
}

#[test]
// spec §16.24 item 13: relativizing twice must not corrupt an already-relative path, mirroring
// `rebase`'s idempotence on absolutes (`tests/rebase.rs:47`). Falsifiable: a `strip_prefix` that
// ran unconditionally would mangle the second pass.
fn relativize_is_idempotent() {
    let root = Path::new("/fixtures");
    let mut page = page_with(
        "/fixtures/recorded/detector/page01_base.png",
        "recorded/detector/m.png",
    );

    assert!(paths::relativize_page_data_raw(&mut page, root).is_empty());
    let after_once = handle_paths(&page);
    assert!(paths::relativize_page_data_raw(&mut page, root).is_empty());
    assert_eq!(
        handle_paths(&page),
        after_once,
        "relativizing must be idempotent"
    );
}

#[test]
// spec §16.24 item 13: `MaskData`'s two handles get the same treatment as `PageDataRaw`'s, the
// symmetric obligation `tests/rebase.rs:68` establishes for the forward direction.
fn mask_data_handles_are_relativized_and_round_trip() {
    let root = Path::new("/fixtures");
    let original = MaskData {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from("page01.png"),
        base_image: ImageHandle::from_path("/fixtures/recorded/detector/page01_base.png"),
        combined_mask: ImageHandle::from_path(
            "/fixtures/recorded/detector/page01_combined_mask.png",
        ),
        scale: 1.0,
        regions: vec![],
    };

    let mut mask = original.clone();
    assert!(paths::relativize_mask_data(&mut mask, root).is_empty());
    assert_eq!(
        mask.base_image.path.as_deref(),
        Some(Path::new("recorded/detector/page01_base.png"))
    );
    assert_eq!(
        mask.combined_mask.path.as_deref(),
        Some(Path::new("recorded/detector/page01_combined_mask.png"))
    );

    paths::rebase_mask_data(&mut mask, root);
    assert_eq!(mask.base_image.path, original.base_image.path);
    assert_eq!(mask.combined_mask.path, original.combined_mask.path);
}

#[test]
// spec §16.24 item 13 + `tests/rebase.rs:93`'s obligation in the other direction: relativizing
// touches ONLY the handles. `original_path`, `scale`, `image_size`, `schema_version` and the
// block list are recorded data and must survive untouched.
fn relativizing_does_not_disturb_other_fields() {
    let mut page = page_with("/fixtures/a.png", "/fixtures/b.png");
    paths::relativize_page_data_raw(&mut page, Path::new("/fixtures"));

    assert_eq!(page.original_path, PathBuf::from("page01.png"));
    assert_eq!(page.scale, 1.0);
    assert_eq!(page.image_size, (1024, 1434));
    assert_eq!(page.schema_version, SCHEMA_VERSION);
    assert!(page.blocks.is_empty());
}

#[test]
// spec §2.3 + §7.2: a path-less (memory-mode) handle has nothing to relativize and must not be
// invented into one. `ImageHandle::Serialize` (`crates/pc-core/src/image_handle.rs:21-31`)
// rejects a path-less handle outright, so fabricating an empty path here would convert a loud
// serialisation failure into a silently wrong committed path.
fn a_path_less_handle_is_left_as_none_and_not_reported() {
    let mut page = page_with("/fixtures/a.png", "/fixtures/b.png");
    page.raw_mask = ImageHandle::from_memory(image::DynamicImage::new_rgb8(1, 1));

    let residue = paths::relativize_page_data_raw(&mut page, Path::new("/fixtures"));

    assert!(
        residue.is_empty(),
        "a path-less handle is not an unrelativizable path: {residue:?}"
    );
    assert_eq!(page.raw_mask.path, None);
    assert_eq!(page.base_image.path.as_deref(), Some(Path::new("a.png")));
}
