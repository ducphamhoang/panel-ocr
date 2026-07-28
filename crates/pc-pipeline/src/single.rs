//! spec §4.3 — the single-image chain: `PageDataRaw` → `PageData` → `MaskData` →
//! denoised → exported files, with checkpoint writes between stages and the four skip
//! levels loading from cache instead.
//!
//! The dest-builders and the cache-path resolver are fully pinned and implemented here.
//! [`process_image`] itself is `todo!()` for Codex (task G1-chain, **heavy** per §16.12
//! item 19): it is the one place where five stage contracts, two checkpointing modes and
//! four skip levels meet.

use crate::cache::CachePaths;
use crate::checkpoint;
use crate::ctx::PipelineCtx;
use crate::options::{Checkpointing, PipelineOptions};
use crate::outcome::{ImageAnalytics, ImageOutcome, SkipReason};
use pc_core::{ImageHandle, Output, Stage, StageError, Step};
use pc_denoise::{DenoiseDests, DenoiseInput, DenoiseOutput, DenoiseStage};
use pc_detect::{DetectInput, DetectStage};
use pc_export::{ExportInput, ExportSources, ExportStage};
use pc_mask::{MaskDests, MaskInput, MaskOutput, MaskStage};
use pc_preprocess::{PreprocessInput, PreprocessStage};
use std::path::{Path, PathBuf};

/// spec §4.3 / §8.2 — `(base_image_dest, raw_mask_dest)`. Both `None` in Memory mode.
pub fn detect_dests(cache: Option<&CachePaths>) -> (Option<PathBuf>, Option<PathBuf>) {
    match cache {
        Some(paths) => (
            Some(paths.for_output(Output::BaseImage)),
            Some(paths.for_output(Output::RawMask)),
        ),
        None => (None, None),
    }
}

/// spec §10.2 / §16.12 item 15 — the masker's six destinations.
///
/// `text_layer` is `Some` only with `--extract-text`; the three debug artifacts only
/// with debug outputs active (which §4.1 restricts to `Disk` mode).
pub fn mask_dests(
    cache: Option<&CachePaths>,
    extract_text: bool,
    debug_outputs: bool,
) -> MaskDests {
    let Some(paths) = cache else {
        return MaskDests::default();
    };
    MaskDests {
        combined_mask: Some(paths.for_output(Output::FinalMask)),
        cleaned: Some(paths.for_output(Output::MaskedOutput)),
        text_layer: extract_text.then(|| paths.for_output(Output::IsolatedText)),
        box_mask: debug_outputs.then(|| paths.for_output(Output::BoxMask)),
        cut_mask: debug_outputs.then(|| paths.for_output(Output::CutMask)),
        mask_overlay: debug_outputs.then(|| paths.for_output(Output::MaskOverlay)),
    }
}

/// spec §11.2 — the denoiser's two destinations. All `None` when denoising is disabled
/// (§16.12 item 5) or in Memory mode.
pub fn denoise_dests(cache: Option<&CachePaths>, denoising_enabled: bool) -> DenoiseDests {
    match cache {
        Some(paths) if denoising_enabled => DenoiseDests {
            noise_mask: Some(paths.for_output(Output::DenoiseMask)),
            denoised: Some(paths.for_output(Output::DenoisedOutput)),
        },
        _ => DenoiseDests::default(),
    }
}

/// spec §12.3 step 2 — **availability**, which is the pipeline's half of the export
/// contract (`pc-export` owns precedence). A `None` stage output contributes nothing.
pub fn export_sources(mask: Option<&MaskOutput>, denoise: Option<&DenoiseOutput>) -> ExportSources {
    ExportSources {
        masked: mask.map(|output| output.cleaned.clone()),
        denoised: denoise.map(|output| output.denoised.clone()),
        final_mask: mask.map(|output| output.combined_mask.clone()),
        denoise_mask: denoise.map(|output| output.noise_mask.clone()),
        isolated_text: mask.and_then(|output| output.text_layer.clone()),
    }
}

/// spec §4.2/§4.4 + §16.12 item 8 — the cache entry this image uses.
///
///   * Memory mode → `Ok(None)`: nothing is persisted, so nothing needs a uuid.
///   * Disk mode, starting at `Detect` → a fresh [`CachePaths::new`].
///   * Disk mode, resuming → [`CachePaths::discover`]; a miss is an error, because the
///     user asked to load artifacts that do not exist.
pub fn cache_paths_for(
    original: &Path,
    options: &PipelineOptions,
) -> Result<Option<CachePaths>, StageError> {
    if options.checkpointing == Checkpointing::Memory {
        return Ok(None);
    }
    if options.start_step() == pc_core::Step::Detect {
        return Ok(Some(CachePaths::new(original, &options.cache_dir)));
    }
    CachePaths::discover(&options.cache_dir, original)?
        .map(Some)
        .ok_or_else(|| {
            StageError::InvalidInput(format!(
                "no cached artifacts for `{}` in `{}`; cannot honour --skip-*",
                original.display(),
                options.cache_dir.display()
            ))
        })
}

/// spec §4.3 — run all five stages for one image.
///
/// Contract Codex must satisfy (each clause is already a frozen test):
///   1. Stages run in `Step` order starting at `options.start_step()`; skipped steps
///      load their predecessor's checkpoint through `crate::checkpoint` instead
///      (§4.4), and a malformed/missing checkpoint is a `Failed`, never a panic.
///   2. In `Disk` mode each stage's persisted struct is written to
///      `CachePaths::for_output` immediately after that stage returns (§4.3), before
///      the next stage runs.
///   3. Denoising is skipped entirely when `options.denoising_enabled()` is false, and
///      `ExportInput.denoising_enabled` carries the same value (§16.12 item 5).
///   4. A page whose `PageData::text_boxes` is empty still exports, and the outcome is
///      `Skipped { NoTextDetected, files_written }` (§5.6, §16.12 item 12).
///   5. The returned `ImageOutcome::Failed.step` names the stage that actually failed.
///   6. No `catch_unwind` here — that is [`crate::batch`]'s boundary (§5.2).
pub fn process_image(
    original: &Path,
    options: &PipelineOptions,
    ctx: &PipelineCtx<'_>,
) -> ImageOutcome {
    let original_buf = original.to_path_buf();
    let flags = options.skips.normalized();
    if options.skips.implies_more() {
        tracing::warn!("normalising skip flags so resumed stages have their predecessors");
    }

    let cache = match cache_paths_for(original, options) {
        Ok(cache) => cache,
        Err(error) => return failed(original_buf, options.start_step(), error),
    };
    if let Some(cache) = cache.as_ref() {
        if let Err(source) = std::fs::create_dir_all(cache.cache_dir()) {
            return failed(
                original_buf,
                options.start_step(),
                StageError::Io {
                    path: cache.cache_dir().to_path_buf(),
                    source,
                },
            );
        }
    }
    let source = ImageHandle::from_path(original);
    let mut analytics = ImageAnalytics::default();

    let raw = if flags.text_detection {
        let Some(cache) = cache.as_ref() else {
            return failed(
                original_buf,
                Step::Detect,
                StageError::InvalidInput(
                    "cannot resume text detection without disk checkpoints".into(),
                ),
            );
        };
        match checkpoint::read_page_raw(&cache.for_output(Output::RawJson)) {
            Ok(page) => page,
            Err(error) => return failed(original_buf, Step::Detect, error),
        }
    } else {
        let detector = match ctx.detectors.detector_for(original) {
            Ok(detector) => detector,
            Err(error) => return failed(original_buf, Step::Detect, error),
        };
        let (base_image_dest, raw_mask_dest) = detect_dests(cache.as_ref());
        let output = match DetectStage::run(
            DetectInput {
                schema_version: pc_core::SCHEMA_VERSION,
                source: source.clone(),
                original_path: original_buf.clone(),
                target_height_lower: options.profile.general.input_height_lower_target,
                target_height_upper: options.profile.general.input_height_upper_target,
                base_image_dest,
                raw_mask_dest,
                min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
                config: options.profile.text_detector.clone(),
            },
            detector.as_ref(),
        ) {
            Ok(output) => output,
            Err(error) => return failed(original_buf, Step::Detect, error),
        };
        if let Some(cache) = cache.as_ref() {
            if let Err(error) =
                checkpoint::write_page_raw(&output.page, &cache.for_output(Output::RawJson))
            {
                return failed(original_buf, Step::Detect, error);
            }
        }
        analytics.detect = Some(output.analytics.clone());
        output.page
    };

    let page = if flags.preprocess {
        let Some(cache) = cache.as_ref() else {
            return failed(
                original_buf,
                Step::Preprocess,
                StageError::InvalidInput(
                    "cannot resume preprocessing without disk checkpoints".into(),
                ),
            );
        };
        match checkpoint::read_page(&cache.for_output(Output::CleanJson)) {
            Ok(page) => page,
            Err(error) => return failed(original_buf, Step::Preprocess, error),
        }
    } else {
        let output = match PreprocessStage::run(
            PreprocessInput {
                schema_version: pc_core::SCHEMA_VERSION,
                page: raw,
                config: options.profile.preprocessor.clone(),
                performing_ocr: options.performing_ocr,
            },
            ctx.ocr,
        ) {
            Ok(output) => output,
            Err(error) => return failed(original_buf, Step::Preprocess, error),
        };
        if let Some(cache) = cache.as_ref() {
            if let Err(error) =
                checkpoint::write_page(&output.page, &cache.for_output(Output::CleanJson))
            {
                return failed(original_buf, Step::Preprocess, error);
            }
        }
        analytics.ocr = output.ocr_analytic.clone();
        output.page
    };
    let no_text = page.text_boxes.is_empty();

    let mask = if flags.mask {
        let Some(cache) = cache.as_ref() else {
            return failed(
                original_buf,
                Step::Mask,
                StageError::InvalidInput("cannot resume masking without disk checkpoints".into()),
            );
        };
        let mask_data = match checkpoint::read_mask_data(&cache.for_output(Output::MaskDataJson)) {
            Ok(mask_data) => mask_data,
            Err(error) => return failed(original_buf, Step::Mask, error),
        };
        cached_mask_output(cache, mask_data, options.extract_text)
    } else {
        let output = match MaskStage::run(
            MaskInput {
                schema_version: pc_core::SCHEMA_VERSION,
                page,
                original_image: source.clone(),
                config: options.profile.masker.clone(),
                extract_text: options.extract_text,
                debug_outputs: options.debug_outputs_active(),
                dests: mask_dests(
                    cache.as_ref(),
                    options.extract_text,
                    options.debug_outputs_active(),
                ),
            },
            (),
        ) {
            Ok(output) => output,
            Err(error) => return failed(original_buf, Step::Mask, error),
        };
        if let Some(cache) = cache.as_ref() {
            if let Err(error) = checkpoint::write_mask_data(
                &output.mask_data,
                &cache.for_output(Output::MaskDataJson),
            ) {
                return failed(original_buf, Step::Mask, error);
            }
        }
        analytics.mask_fitting = output.analytics.clone();
        output
    };

    let denoise = if options.denoising_enabled() {
        match DenoiseStage::run(
            DenoiseInput {
                schema_version: pc_core::SCHEMA_VERSION,
                mask_data: mask.mask_data.clone(),
                original_image: source.clone(),
                masked_image: mask.cleaned.clone(),
                config: options.profile.denoiser.clone(),
                dests: denoise_dests(cache.as_ref(), true),
            },
            (),
        ) {
            Ok(output) => {
                analytics.denoise = Some(output.analytics.clone());
                Some(output)
            }
            Err(error) => return failed(original_buf, Step::Denoise, error),
        }
    } else {
        None
    };

    let files_written = match ExportStage::run(
        ExportInput {
            schema_version: pc_core::SCHEMA_VERSION,
            original_path: original_buf.clone(),
            export_path: original_buf.clone(),
            output_dir: options.output_dir.clone(),
            outputs: options.requested_outputs(),
            sources: export_sources(Some(&mask), denoise.as_ref()),
            preferred_file_type: options.profile.general.cleaned_suffix(),
            preferred_mask_file_type: options.profile.general.preferred_mask_file_type.clone(),
            denoising_enabled: options.denoising_enabled(),
        },
        (),
    ) {
        Ok(output) => output.files_written,
        Err(error) => return failed(original_buf, Step::Export, error),
    };

    if no_text {
        ImageOutcome::Skipped {
            original: original_buf,
            reason: SkipReason::NoTextDetected,
            files_written,
        }
    } else {
        ImageOutcome::Completed {
            original: original_buf,
            files_written,
            analytics: Box::new(analytics),
        }
    }
}

fn failed(original: PathBuf, step: Step, error: StageError) -> ImageOutcome {
    ImageOutcome::Failed {
        original,
        step,
        error,
    }
}

fn cached_mask_output(
    cache: &CachePaths,
    mask_data: pc_core::MaskData,
    extract_text: bool,
) -> MaskOutput {
    let text_path = cache.for_output(Output::IsolatedText);
    MaskOutput {
        combined_mask: mask_data.combined_mask.clone(),
        cleaned: ImageHandle::from_path(cache.for_output(Output::MaskedOutput)),
        text_layer: (extract_text && text_path.exists()).then(|| ImageHandle::from_path(text_path)),
        mask_data,
        analytics: Vec::new(),
    }
}
