//! spec §13 row 26 (tasks **D9** + **E5**) — long-strip orchestration.
//!
//! A webtoon-style strip is split into segments *before* stage 1, each segment runs the
//! normal five-stage chain, and (when `general.merge_after_split`) the per-segment
//! outputs are stitched back into one full-height image whose `export_path` is the
//! original file's (§16.11 item 1: `pc-export` knows nothing about any of this).
//!
//! Planning, the manifest and the segment naming are fully pinned (§16.6 item 8,
//! §16.12 item 10) and implemented here. [`merged_strip_export`] stitches and exports.

use crate::cache::CachePaths;
use crate::options::PipelineOptions;
use pc_config::GeneralConfig;
use pc_core::{ImageHandle, Output, StageError};
use pc_export::{ExportInput, ExportOutput, ExportSources};
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
/// Contract (each clause is a frozen test):
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
    if manifest.segments.len() != segment_sources.len() {
        return Err(StageError::InvalidInput(format!(
            "split manifest has {} segments but {} export-source sets were supplied",
            manifest.segments.len(),
            segment_sources.len()
        )));
    }

    if !options.profile.general.merge_after_split {
        let mut files_written = Vec::new();
        for (segment, sources) in manifest.segments.iter().zip(segment_sources) {
            let output = pc_export::run(ExportInput {
                schema_version: pc_core::SCHEMA_VERSION,
                original_path: segment.clone(),
                export_path: segment.clone(),
                output_dir: options.output_dir.clone(),
                outputs: options.requested_outputs(),
                sources: sources.clone(),
                preferred_file_type: Some(options.profile.general.preferred_file_type.clone()),
                preferred_mask_file_type: options.profile.general.preferred_mask_file_type.clone(),
                denoising_enabled: options.denoising_enabled(),
                inpainting_enabled: options.profile.inpainter.inpainting_enabled,
            })?;
            files_written.extend(output.files_written);
        }
        return Ok(ExportOutput { files_written });
    }

    let first_segment = manifest.segments.first().ok_or_else(|| {
        StageError::InvalidInput("cannot merge a split manifest with no segments".into())
    })?;
    let cache_dir = first_segment.parent().ok_or_else(|| {
        StageError::InvalidInput(format!(
            "segment path `{}` has no parent directory",
            first_segment.display()
        ))
    })?;
    let cache = CachePaths::discover(cache_dir, &manifest.original)?.ok_or_else(|| {
        StageError::InvalidInput(format!(
            "cannot discover original strip cache for `{}`",
            manifest.original.display()
        ))
    })?;
    let requested = options.requested_outputs();

    let sources = ExportSources {
        masked: stitch_requested(
            &requested,
            &[Output::MaskedOutput, Output::DenoisedOutput],
            segment_sources
                .iter()
                .map(|sources| sources.masked.as_ref()),
            cache.for_output(Output::MaskedOutput),
            manifest.image_size,
            "cleaned",
        )?,
        denoised: stitch_requested(
            &requested,
            &[Output::MaskedOutput, Output::DenoisedOutput],
            segment_sources
                .iter()
                .map(|sources| sources.denoised.as_ref()),
            cache.for_output(Output::DenoisedOutput),
            manifest.image_size,
            "cleaned",
        )?,
        inpainted: stitch_requested(
            &requested,
            &[Output::MaskedOutput, Output::DenoisedOutput],
            segment_sources
                .iter()
                .map(|sources| sources.inpainted.as_ref()),
            cache.for_suffix(crate::single::CLEAN_INPAINT_SUFFIX),
            manifest.image_size,
            "inpainted",
        )?,
        final_mask: stitch_requested(
            &requested,
            &[Output::FinalMask, Output::DenoiseMask],
            segment_sources
                .iter()
                .map(|sources| sources.final_mask.as_ref()),
            cache.for_output(Output::FinalMask),
            manifest.image_size,
            "mask",
        )?,
        denoise_mask: stitch_requested(
            &requested,
            &[Output::FinalMask, Output::DenoiseMask],
            segment_sources
                .iter()
                .map(|sources| sources.denoise_mask.as_ref()),
            cache.for_output(Output::DenoiseMask),
            manifest.image_size,
            "mask",
        )?,
        inpainted_mask: stitch_requested(
            &requested,
            &[Output::FinalMask, Output::DenoiseMask],
            segment_sources
                .iter()
                .map(|sources| sources.inpainted_mask.as_ref()),
            cache.for_suffix(crate::single::INPAINTING_SUFFIX),
            manifest.image_size,
            "inpainted mask",
        )?,
        isolated_text: stitch_requested(
            &requested,
            &[Output::IsolatedText],
            segment_sources
                .iter()
                .map(|sources| sources.isolated_text.as_ref()),
            cache.for_output(Output::IsolatedText),
            manifest.image_size,
            "text",
        )?,
    };

    pc_export::run(ExportInput {
        schema_version: pc_core::SCHEMA_VERSION,
        original_path: manifest.original.clone(),
        export_path: manifest.original.clone(),
        output_dir: options.output_dir.clone(),
        outputs: requested,
        sources,
        preferred_file_type: Some(options.profile.general.preferred_file_type.clone()),
        preferred_mask_file_type: options.profile.general.preferred_mask_file_type.clone(),
        denoising_enabled: options.denoising_enabled(),
        inpainting_enabled: options.profile.inpainter.inpainting_enabled,
    })
}

/// Stitch one availability candidate when its export category was requested. A partial
/// category is deliberately discarded: exporting only some strip segments would produce
/// a misleading, truncated image.
fn stitch_requested<'a>(
    requested: &[Output],
    category_outputs: &[Output],
    handles: impl Iterator<Item = Option<&'a ImageHandle>>,
    destination: PathBuf,
    expected_size: (u32, u32),
    category: &str,
) -> Result<Option<ImageHandle>, StageError> {
    if !requested
        .iter()
        .any(|output| category_outputs.contains(output))
    {
        return Ok(None);
    }

    let handles: Option<Vec<&ImageHandle>> = handles.collect();
    let Some(handles) = handles else {
        tracing::warn!(%category, "not every split segment supplied this export category; dropping it");
        return Ok(None);
    };

    let segments = handles
        .iter()
        .map(|handle| handle.load().map(|image| image.to_rgba8()))
        .collect::<Result<Vec<_>, _>>()?;
    let stitched = pc_imageops::stitch_images(&segments)?;
    if stitched.dimensions() != expected_size {
        return Err(StageError::InvalidInput(format!(
            "stitched {category} image has dimensions {:?}, expected {:?}",
            stitched.dimensions(),
            expected_size
        )));
    }

    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    stitched
        .save_with_format(&destination, image::ImageFormat::Png)
        .map_err(|error| StageError::Io {
            path: destination.clone(),
            source: std::io::Error::other(error),
        })?;
    Ok(Some(ImageHandle::from_path(destination)))
}
