//! C2 tests — spec §2.3 `ImageHandle` and its materialization invariant.

use image::{DynamicImage, GrayImage, RgbImage};
use pc_core::{ImageHandle, StageError};
use std::path::{Path, PathBuf};

fn rgb(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, image::Rgb([7, 8, 9])))
}

fn gray(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageLuma8(GrayImage::from_pixel(w, h, image::Luma([42])))
}

fn write_png(dir: &Path, name: &str, img: &DynamicImage) -> PathBuf {
    let p = dir.join(name);
    img.save(&p).expect("write fixture png");
    p
}

#[test]
// spec §2.3: from_path records the path and does not decode; a path that does not
// exist yet is a legal handle (the pipeline names destinations before writing them)
fn from_path_records_path_without_decoding() {
    let h = ImageHandle::from_path("/definitely/does/not/exist.png");
    assert_eq!(h.path.as_deref(), Some(Path::new("/definitely/does/not/exist.png")));
    assert!(!h.is_materialized(), "a nonexistent path is not materialized");
    assert!(h.load().is_err(), "loading a missing file must fail, not panic");
}

#[test]
// spec §2.3: is_materialized() means "path exists on disk"
fn is_materialized_tracks_disk_presence() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_png(dir.path(), "on_disk.png", &rgb(4, 6));
    assert!(ImageHandle::from_path(&p).is_materialized());

    std::fs::remove_file(&p).unwrap();
    assert!(!ImageHandle::from_path(&p).is_materialized());

    // an in-memory-only handle has no path and therefore cannot be materialized
    assert!(!ImageHandle::from_memory(rgb(4, 6)).is_materialized());
}

#[test]
// spec §2.3: from_memory yields path: None and load() serves the cached image
fn from_memory_serves_the_cache_with_no_path() {
    let h = ImageHandle::from_memory(rgb(11, 13));
    assert_eq!(h.path, None);
    let loaded = h.load().expect("cached load");
    assert_eq!(loaded.width(), 11);
    assert_eq!(loaded.height(), 13);
    // dimensions() must work with no path at all (cache-first)
    assert_eq!(h.dimensions().unwrap(), (11, 13));
}

#[test]
// spec §2.3: load() prefers the cache over the path — with_both against a path
// that does not exist must still succeed, proving no decode was attempted
fn load_prefers_cache_over_path() {
    let h = ImageHandle::with_both("/definitely/does/not/exist.png", gray(5, 9));
    let loaded = h.load().expect("cache must be used, not the missing path");
    assert_eq!((loaded.width(), loaded.height()), (5, 9));
    assert_eq!(h.dimensions().unwrap(), (5, 9));
}

#[test]
// spec §2.3: with no cache, load() decodes from `path` and dimensions() reads the
// header only — both must agree with the file on disk
fn load_and_dimensions_fall_back_to_path() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_png(dir.path(), "page.png", &rgb(23, 31));
    let h = ImageHandle::from_path(&p);
    assert_eq!(h.dimensions().unwrap(), (23, 31));
    let loaded = h.load().unwrap();
    assert_eq!((loaded.width(), loaded.height()), (23, 31));
}

#[test]
// spec §2.3: decoding a corrupt file is an error, not a panic
fn load_of_a_corrupt_file_errors() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("broken.png");
    std::fs::write(&p, b"this is not a png").unwrap();
    let h = ImageHandle::from_path(&p);
    assert!(h.load().is_err());
    assert!(h.dimensions().is_err());
}

#[test]
// spec §2.3: `cached` is #[serde(skip)], so after a round-trip the handle carries
// only its path. Proven behaviourally: the pre-round-trip handle loads from cache
// even though its path is missing, and the post-round-trip handle cannot.
fn serde_round_trip_drops_the_cache() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("never_written.png");
    let before = ImageHandle::with_both(&missing, rgb(3, 3));
    assert!(before.load().is_ok(), "cache present before round-trip");

    let json = serde_json::to_string(&before).expect("serialize");
    let after: ImageHandle = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(after.path.as_deref(), Some(missing.as_path()));
    assert!(
        after.load().is_err(),
        "cache must not survive the round-trip; load must fall back to the missing path"
    );
    assert!(after.dimensions().is_err());
}

#[test]
// spec §2.3: the serialized form carries `path` and nothing else — no `cached` key
// may appear in a checkpoint JSON
fn serialized_shape_is_path_only() {
    let h = ImageHandle::with_both("/a/b.png", rgb(2, 2));
    let json = serde_json::to_string(&h).expect("serialize");
    assert_eq!(json, r#"{"path":"/a/b.png"}"#);
    assert!(!json.contains("cached"));
}

#[test]
// spec §2.3 materialization invariant: a handle with path: None cannot be
// checkpointed — the pipeline must materialize it first
fn unmaterialized_handle_cannot_be_checkpointed() {
    let mem = ImageHandle::from_memory(rgb(2, 2));
    match mem.ensure_materialized() {
        Err(StageError::UnmaterializedHandle) => {}
        other => panic!("expected UnmaterializedHandle, got {other:?}"),
    }
    // ...and serialization itself must refuse, so no code path can smuggle a
    // path-less handle into a JSON checkpoint.
    assert!(
        serde_json::to_string(&mem).is_err(),
        "serializing a path-less handle must fail (§2.3)"
    );
}

#[test]
// spec §2.3: a handle that names a path is checkpointable even before the file has
// been written — checkpointing records intent, and the pipeline writes the bytes
fn handle_with_a_path_is_checkpointable() {
    assert!(ImageHandle::from_path("/tmp/planned.png").ensure_materialized().is_ok());
    assert!(ImageHandle::with_both("/tmp/planned.png", rgb(1, 1))
        .ensure_materialized()
        .is_ok());
    assert!(serde_json::to_string(&ImageHandle::from_path("/tmp/planned.png")).is_ok());
}

#[test]
// spec §2.3: deserializing a null path yields a handle that is itself
// unmaterialized — reading a malformed checkpoint must be detectable, not silently
// accepted as a working handle
fn deserialized_null_path_is_reported_as_unmaterialized() {
    let h: ImageHandle = serde_json::from_str(r#"{"path":null}"#).expect("deserialize");
    assert_eq!(h.path, None);
    assert!(matches!(
        h.ensure_materialized(),
        Err(StageError::UnmaterializedHandle)
    ));
    assert!(matches!(h.load(), Err(StageError::UnmaterializedHandle)));
}

#[test]
// spec §2.3: cloning shares the decoded image (Arc) rather than re-decoding, so a
// clone of a cached handle still loads without touching disk
fn clone_shares_the_cache() {
    let h = ImageHandle::with_both("/definitely/does/not/exist.png", rgb(6, 6));
    let c = h.clone();
    assert!(c.load().is_ok());
    assert_eq!(c.dimensions().unwrap(), (6, 6));
}
