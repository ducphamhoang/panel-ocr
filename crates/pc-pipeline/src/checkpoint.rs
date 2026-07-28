//! spec §4.3/§4.4 — disk checkpoints between stages, and their two validation rules
//! (§16.12 item 13).
//!
//! A checkpoint is the stage's persisted output struct (`PageDataRaw` → `#raw.json`,
//! `PageData` → `#clean.json`, `MaskData` → `#mask_data.json`). Writing one is only
//! legal when every `ImageHandle` it references is materialised on disk (§2.3) —
//! otherwise a later `--skip-*` run would load a handle it can never decode.
//!
//! Fully pinned; implemented, not stubbed.

use pc_core::{ImageHandle, MaskData, PageData, PageDataRaw, StageError};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;

/// §16.12 item 13. `pc_core::SCHEMA_VERSION` is the only version v1 knows.
pub const SUPPORTED_SCHEMA_VERSIONS: &[u32] = &[pc_core::SCHEMA_VERSION];

/// `Ok(())` iff `found` is a version this build understands (§4.4).
pub fn check_schema_version(found: u32, path: &Path) -> Result<(), StageError> {
    if SUPPORTED_SCHEMA_VERSIONS.contains(&found) {
        Ok(())
    } else {
        Err(StageError::InvalidInput(format!(
            "checkpoint `{}` has unsupported schema_version {found} (supported: {:?})",
            path.display(),
            SUPPORTED_SCHEMA_VERSIONS
        )))
    }
}

/// `Ok(())` iff every handle has a path and that path exists (§4.4: a checkpoint whose
/// referenced images are gone is a per-image error, not a fatal one).
pub fn ensure_handles_readable(handles: &[&ImageHandle]) -> Result<(), StageError> {
    for handle in handles {
        match handle.path.as_ref() {
            None => return Err(StageError::UnmaterializedHandle),
            Some(path) if !path.exists() => {
                return Err(StageError::Io {
                    path: path.clone(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "checkpoint references an image that no longer exists",
                    ),
                })
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// Serialise `value` to `path` as pretty JSON, creating the parent directory.
///
/// `ImageHandle`'s own `Serialize` guard (§2.3) turns a path-less handle into a serde
/// error here; callers that want the friendlier `UnmaterializedHandle` should call the
/// typed wrappers below, which pre-flight with `ensure_materialized()`.
pub fn write_json<T: Serialize>(value: &T, path: &Path) -> Result<(), StageError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
    }
    let text = serde_json::to_string_pretty(value)?;
    std::fs::write(path, text).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Read and deserialise a checkpoint. Schema/handle validation is the typed wrappers'
/// job — this is the raw read.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, StageError> {
    let text = std::fs::read_to_string(path).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(serde_json::from_str(&text)?)
}

// ------------------------------------------------------------ typed wrappers

/// `#raw.json` (§2.4).
pub fn write_page_raw(page: &PageDataRaw, path: &Path) -> Result<(), StageError> {
    page.base_image.ensure_materialized()?;
    page.raw_mask.ensure_materialized()?;
    write_json(page, path)
}

/// `#raw.json`, validated per §16.12 item 13.
pub fn read_page_raw(path: &Path) -> Result<PageDataRaw, StageError> {
    let page: PageDataRaw = read_json(path)?;
    check_schema_version(page.schema_version, path)?;
    ensure_handles_readable(&[&page.base_image, &page.raw_mask])?;
    Ok(page)
}

/// `#clean.json` (§2.5).
pub fn write_page(page: &PageData, path: &Path) -> Result<(), StageError> {
    page.base_image.ensure_materialized()?;
    page.raw_mask.ensure_materialized()?;
    write_json(page, path)
}

/// `#clean.json`, validated.
pub fn read_page(path: &Path) -> Result<PageData, StageError> {
    let page: PageData = read_json(path)?;
    check_schema_version(page.schema_version, path)?;
    ensure_handles_readable(&[&page.base_image, &page.raw_mask])?;
    Ok(page)
}

/// `#mask_data.json` (§2.6).
pub fn write_mask_data(mask_data: &MaskData, path: &Path) -> Result<(), StageError> {
    mask_data.base_image.ensure_materialized()?;
    mask_data.combined_mask.ensure_materialized()?;
    write_json(mask_data, path)
}

/// `#mask_data.json`, validated.
pub fn read_mask_data(path: &Path) -> Result<MaskData, StageError> {
    let mask_data: MaskData = read_json(path)?;
    check_schema_version(mask_data.schema_version, path)?;
    ensure_handles_readable(&[&mask_data.base_image, &mask_data.combined_mask])?;
    Ok(mask_data)
}
