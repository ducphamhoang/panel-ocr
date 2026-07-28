//! Input discovery: turning the CLI's `<PATHS>...` into a deterministic list of images.
//!
//! spec §8.3 step 1 owns the supported-input list (upstream's `SUPPORTED_IMG_TYPES`);
//! §5.3 makes "no input images found" a fatal condition, and §5.7 requires the order to
//! be deterministic. Fully pinned; implemented, not stubbed.

use pc_core::StageError;
use std::path::{Path, PathBuf};

/// spec §8.3 step 1 (upstream `SUPPORTED_IMG_TYPES`). `.jp2` is a valid *input* suffix
/// even though §12.3 step 6 rejects it as an *output* one.
pub const SUPPORTED_INPUT_SUFFIXES: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".dib", ".jp2", ".ppm",
];

/// ASCII-case-insensitive suffix test, matching `pc_config`'s normalisation rule.
pub fn is_supported_input(path: &Path) -> bool {
    match path.extension() {
        Some(extension) => {
            let suffix = format!(".{}", extension.to_string_lossy().to_ascii_lowercase());
            SUPPORTED_INPUT_SUFFIXES.contains(&suffix.as_str())
        }
        None => false,
    }
}

/// The suffix [`is_supported_input`] judged, normalised the same way, for
/// [`crate::outcome::SkipReason::UnsupportedFormat`]. `""` when the path has no extension.
pub fn input_suffix(path: &Path) -> String {
    match path.extension() {
        Some(extension) => format!(".{}", extension.to_string_lossy().to_ascii_lowercase()),
        None => String::new(),
    }
}

/// Expand the user's paths into images, in a deterministic order (§5.7):
///
///   * a file is taken as-is (even when its suffix is unsupported — the per-image
///     boundary reports that as `Skipped { UnsupportedFormat }`, §5.6, so a typo'd
///     file is visible in the summary rather than silently dropped);
///   * a directory contributes its **direct** children with supported suffixes, sorted
///     by path (no recursion in v1);
///   * a path that does not exist is a `StageError::Io` — a fatal condition (§5.3),
///     since the user named something that isn't there.
///
/// Duplicates are removed, keeping first-occurrence order.
pub fn expand_inputs(paths: &[PathBuf]) -> Result<Vec<PathBuf>, StageError> {
    let mut images: Vec<PathBuf> = Vec::new();

    for path in paths {
        if path.is_dir() {
            let entries = std::fs::read_dir(path).map_err(|source| StageError::Io {
                path: path.clone(),
                source,
            })?;
            let mut children: Vec<PathBuf> = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|source| StageError::Io {
                    path: path.clone(),
                    source,
                })?;
                let child = entry.path();
                if child.is_file() && is_supported_input(&child) {
                    children.push(child);
                }
            }
            children.sort();
            images.extend(children);
        } else if path.exists() {
            images.push(path.clone());
        } else {
            return Err(StageError::Io {
                path: path.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "input path does not exist",
                ),
            });
        }
    }

    let mut seen = std::collections::HashSet::new();
    images.retain(|path| seen.insert(path.clone()));
    Ok(images)
}
