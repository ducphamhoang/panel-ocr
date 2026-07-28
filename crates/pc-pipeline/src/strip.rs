//! spec §13 row 26 (tasks **D9** + **E5**) — long-strip orchestration.
//!
//! A webtoon-style strip is split into segments *before* stage 1, each segment runs the
//! normal five-stage chain, and (when `general.merge_after_split`) the per-segment
//! outputs are stitched back into one full-height image whose `export_path` is the
//! original file's (§16.11 item 1: `pc-export` knows nothing about any of this).
//!
//! Planning, the manifest and the segment naming are fully pinned (§16.6 item 8,
//! §16.12 item 10) and implemented here. [`merged_strip_export`] is `todo!()` for Codex.

use crate::cache::CachePaths;
use crate::options::PipelineOptions;
use pc_config::GeneralConfig;
use pc_core::StageError;
use pc_export::{ExportOutput, ExportSources};
use pc_imageops::SplitParams;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// spec §13 row 26's `splits.json`, written to `{uuid}_{stem}#splits.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitManifest {
    pub schema_version: u32,
    /// The user's input file.
    pub original: PathBuf,
    /// Its full size, so the stitched export can be size-checked (§12.7(B)11).
    pub image_size: (u32, u32),
    /// The split rows, as [`pc_imageops::calculate_best_splits`] returned them.
    pub split_rows: Vec<u32>,
    /// One path per segment, in top-to-bottom order. `segments.len() == split_rows.len() + 1`.
    pub segments: Vec<PathBuf>,
}

/// spec §16.6 item 8's aspect gate, in `[general]` terms.
pub fn should_split(image_size: (u32, u32), general: &GeneralConfig) -> bool {
    let (width, height) = image_size;
    general.split_long_strips
        && height > 0
        && f64::from(width) / f64::from(height) <= general.long_strip_aspect_ratio
}

/// `[general]` → [`SplitParams`] (`pc-imageops` deliberately does not depend on
/// `pc-config`; §1 rule 2).
pub fn split_params(general: &GeneralConfig) -> SplitParams {
    SplitParams {
        preferred_height: general.preferred_split_height,
        tolerance_margin: general.split_tolerance_margin,
        split_long_strips: general.split_long_strips,
        max_aspect_ratio: general.long_strip_aspect_ratio,
    }
}

/// Split `original` into segment PNGs under `cache`, and write the manifest.
///
/// Returns `Ok(None)` when the image does not qualify for splitting (§16.6 item 8's
/// gate, or a single-segment plan), in which case nothing is written.
pub fn plan_and_write_segments(
    original: &Path,
    cache: &CachePaths,
    general: &GeneralConfig,
) -> Result<Option<SplitManifest>, StageError> {
    let image = image::open(original).map_err(|source| StageError::Decode {
        path: original.to_path_buf(),
        source,
    })?;
    let rgb = image.to_rgb8();
    let image_size = (rgb.width(), rgb.height());

    if !should_split(image_size, general) {
        return Ok(None);
    }

    let params = split_params(general);
    let split_rows = pc_imageops::calculate_best_splits(&rgb, &params);
    if split_rows.is_empty() {
        return Ok(None);
    }

    let segments = pc_imageops::split_image(&rgb, &split_rows)?;
    let mut segment_paths = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        let path = cache.segment(index);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        segment
            .save_with_format(&path, image::ImageFormat::Png)
            .map_err(|error| StageError::Io {
                path: path.clone(),
                source: std::io::Error::other(error),
            })?;
        segment_paths.push(path);
    }

    let manifest = SplitManifest {
        schema_version: pc_core::SCHEMA_VERSION,
        original: original.to_path_buf(),
        image_size,
        split_rows,
        segments: segment_paths,
    };
    crate::checkpoint::write_json(&manifest, &cache.splits_manifest())?;
    Ok(Some(manifest))
}

/// Read a manifest back, validating its schema version (§16.12 item 13).
pub fn read_manifest(path: &Path) -> Result<SplitManifest, StageError> {
    let manifest: SplitManifest = crate::checkpoint::read_json(path)?;
    crate::checkpoint::check_schema_version(manifest.schema_version, path)?;
    Ok(manifest)
}

/// spec §12.5 task **E5** — stitch the per-segment stage outputs and export once, with
/// `export_path` re-pointed at the original strip.
///
/// `segment_sources[i]` is the availability set stage 3/4 produced for segment `i`
/// (`crate::single::export_sources`), in manifest order.
///
/// Contract Codex must satisfy:
///   * only the categories the run actually requests are stitched (§12.3's out-of-scope
///     note rules out the `stitch_all` debug variant); a category is stitched iff every
///     segment supplies it, otherwise it is dropped with a `WARN`;
///   * stitching uses [`pc_imageops::stitch_images`] in manifest segment order, so the
///     result has exactly `manifest.image_size` (§12.7(B)11);
///   * exactly one `pc_export::run` call, with `original_path == export_path ==
///     manifest.original`, `output_dir`/suffixes from `options` (§16.11 items 1 and 4);
///   * stitched intermediates are written under the same cache entry as the strip, so
///     the `ImageHandle`s handed to `pc-export` are materialised (§2.3);
///   * `general.merge_after_split == false` means: no stitching, and each segment is
///     exported separately with its own `export_path`.
pub fn merged_strip_export(
    manifest: &SplitManifest,
    segment_sources: &[ExportSources],
    options: &PipelineOptions,
) -> Result<ExportOutput, StageError> {
    let _ = (manifest, segment_sources, options);
    todo!("tasks D9/E5 (spec §12.5, §13 row 26): stitch segment outputs and export once")
}
