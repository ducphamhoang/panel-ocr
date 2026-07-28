//! spec §4.3 — the single-image chain: `PageDataRaw` → `PageData` → `MaskData` →
//! denoised → exported files, with checkpoint writes between stages and the four skip
//! levels loading from cache instead.
//!
//! Three layers, per §16.14 item 1:
//!   * [`run_stages`] — stages 1–4 for one image *or one strip segment*, no export;
//!   * [`process_image`] — `run_stages` + one export, for one image (task G1-chain);
//!   * [`process_image_with_splitting`] — §4.3's `[split?]` box above both: the entry
//!     point [`crate::batch`] calls, which turns a qualifying long strip into segments,
//!     runs the chain per segment and exports the stitched result **once**.

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

/// Everything stages 1–4 produced for one image — or for one strip segment (§16.14
/// item 1) — before stage 5 turns it into user-facing files.
#[derive(Debug)]
pub struct ChainOutputs {
    /// §12.3 step 2's availability set, ready for `pc_export`.
    pub sources: ExportSources,
    pub analytics: ImageAnalytics,
    /// `true` when `PageData::text_boxes` was empty (§5.6's `NoTextDetected`).
    pub no_text: bool,
}

/// spec §4.3 — stages 1–4 for one image, with **no** export.
///
/// Split out of [`process_image`] by §16.14 item 1 so the long-strip path can run the
/// chain per segment and export **once**, for the whole strip. `cache` is the entry this
/// image's artifacts belong to (`None` in `Checkpointing::Memory`), already created on
/// disk by the caller. The `Err` half names the step that failed, so the caller can build
/// an `ImageOutcome::Failed` for whichever *original* input this call belongs to.
///
/// Contract (each clause is a frozen test):
///   1. Stages run in `Step` order starting at `options.start_step()`; skipped steps
///      load their predecessor's checkpoint through `crate::checkpoint` instead
///      (§4.4), and a malformed/missing checkpoint is an `Err`, never a panic.
///   2. In `Disk` mode each stage's persisted struct is written to
///      `CachePaths::for_output` immediately after that stage returns (§4.3), before
///      the next stage runs.
///   3. Denoising is skipped entirely when `options.denoising_enabled()` is false
///      (§16.12 item 5).
///   4. No `catch_unwind` here — that is [`crate::batch`]'s boundary (§5.2).
pub fn run_stages(
    original: &Path,
    cache: Option<&CachePaths>,
    options: &PipelineOptions,
    ctx: &PipelineCtx<'_>,
) -> Result<ChainOutputs, (Step, StageError)> {
    let original_buf = original.to_path_buf();
    let flags = options.skips.normalized();
    let source = ImageHandle::from_path(original);
    let mut analytics = ImageAnalytics::default();

    let raw = if flags.text_detection {
        let Some(cache) = cache else {
            return Err((
                Step::Detect,
                StageError::InvalidInput(
                    "cannot resume text detection without disk checkpoints".into(),
                ),
            ));
        };
        checkpoint::read_page_raw(&cache.for_output(Output::RawJson))
            .map_err(|error| (Step::Detect, error))?
    } else {
        let detector = ctx
            .detectors
            .detector_for(original)
            .map_err(|error| (Step::Detect, error))?;
        let (base_image_dest, raw_mask_dest) = detect_dests(cache);
        let output = DetectStage::run(
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
        )
        .map_err(|error| (Step::Detect, error))?;
        if let Some(cache) = cache {
            checkpoint::write_page_raw(&output.page, &cache.for_output(Output::RawJson))
                .map_err(|error| (Step::Detect, error))?;
        }
        analytics.detect = Some(output.analytics.clone());
        output.page
    };

    let page = if flags.preprocess {
        let Some(cache) = cache else {
            return Err((
                Step::Preprocess,
                StageError::InvalidInput(
                    "cannot resume preprocessing without disk checkpoints".into(),
                ),
            ));
        };
        checkpoint::read_page(&cache.for_output(Output::CleanJson))
            .map_err(|error| (Step::Preprocess, error))?
    } else {
        let output = PreprocessStage::run(
            PreprocessInput {
                schema_version: pc_core::SCHEMA_VERSION,
                page: raw,
                config: options.profile.preprocessor.clone(),
                performing_ocr: options.performing_ocr,
            },
            ctx.ocr,
        )
        .map_err(|error| (Step::Preprocess, error))?;
        if let Some(cache) = cache {
            checkpoint::write_page(&output.page, &cache.for_output(Output::CleanJson))
                .map_err(|error| (Step::Preprocess, error))?;
        }
        analytics.ocr = output.ocr_analytic.clone();
        output.page
    };
    let no_text = page.text_boxes.is_empty();

    let mask = if flags.mask {
        let Some(cache) = cache else {
            return Err((
                Step::Mask,
                StageError::InvalidInput("cannot resume masking without disk checkpoints".into()),
            ));
        };
        let mask_data = checkpoint::read_mask_data(&cache.for_output(Output::MaskDataJson))
            .map_err(|error| (Step::Mask, error))?;
        cached_mask_output(cache, mask_data, options.extract_text)
    } else {
        let output = MaskStage::run(
            MaskInput {
                schema_version: pc_core::SCHEMA_VERSION,
                page,
                original_image: source.clone(),
                config: options.profile.masker.clone(),
                extract_text: options.extract_text,
                debug_outputs: options.debug_outputs_active(),
                dests: mask_dests(cache, options.extract_text, options.debug_outputs_active()),
            },
            (),
        )
        .map_err(|error| (Step::Mask, error))?;
        if let Some(cache) = cache {
            checkpoint::write_mask_data(&output.mask_data, &cache.for_output(Output::MaskDataJson))
                .map_err(|error| (Step::Mask, error))?;
        }
        analytics.mask_fitting = output.analytics.clone();
        output
    };

    let denoise = if options.denoising_enabled() {
        let output = DenoiseStage::run(
            DenoiseInput {
                schema_version: pc_core::SCHEMA_VERSION,
                mask_data: mask.mask_data.clone(),
                original_image: source.clone(),
                masked_image: mask.cleaned.clone(),
                config: options.profile.denoiser.clone(),
                dests: denoise_dests(cache, true),
            },
            (),
        )
        .map_err(|error| (Step::Denoise, error))?;
        analytics.denoise = Some(output.analytics.clone());
        Some(output)
    } else {
        None
    };

    Ok(ChainOutputs {
        sources: export_sources(Some(&mask), denoise.as_ref()),
        analytics,
        no_text,
    })
}

/// spec §4.3 — run all five stages for one image, exporting that image's own outputs.
///
/// This is the non-split path (and the per-image path a caller who has already handled
/// splitting uses). §16.14 item 1 puts the long-strip branch **above** it, in
/// [`process_image_with_splitting`]; `process_image` itself is unchanged.
///
/// Contract (each clause is a frozen test):
///   1. Everything [`run_stages`] guarantees.
///   2. `ExportInput.denoising_enabled` carries `options.denoising_enabled()`
///      (§16.12 item 5).
///   3. A page whose `PageData::text_boxes` is empty still exports, and the outcome is
///      `Skipped { NoTextDetected, files_written }` (§5.6, §16.12 item 12).
///   4. The returned `ImageOutcome::Failed.step` names the stage that actually failed.
///   5. No `catch_unwind` here — that is [`crate::batch`]'s boundary (§5.2).
pub fn process_image(
    original: &Path,
    options: &PipelineOptions,
    ctx: &PipelineCtx<'_>,
) -> ImageOutcome {
    let original_buf = original.to_path_buf();
    warn_on_implied_skips(options);

    let cache = match prepare_cache(original, options) {
        Ok(cache) => cache,
        Err((step, error)) => return failed(original_buf, step, error),
    };

    let chain = match run_stages(original, cache.as_ref(), options, ctx) {
        Ok(chain) => chain,
        Err((step, error)) => return failed(original_buf, step, error),
    };

    let files_written = match export_once(&original_buf, chain.sources, options) {
        Ok(files_written) => files_written,
        Err((step, error)) => return failed(original_buf, step, error),
    };

    outcome_for(original_buf, chain.no_text, files_written, chain.analytics)
}

/// spec §4.3's `[split?]` box + §13 row 26 — the **entry point a batch uses**.
///
/// §16.14 item 1: `general.split_long_strips` is on by default, so this branch (not
/// [`process_image`]) is what `crate::batch` calls. A qualifying long strip is cut into
/// segments *before* stage 1, every segment runs [`run_stages`] against its own cache
/// entry, and [`crate::strip::merged_strip_export`] stitches and exports **once** — so the
/// batch summary still shows exactly one row per ORIGINAL input file.
///
/// Splitting is not attempted (and the call is a plain [`process_image`]) when:
///   * `general.split_long_strips` is false, or the image fails §16.6 item 8's aspect
///     gate / plans to a single segment;
///   * `options.checkpointing == Checkpointing::Memory` — segments must be materialised
///     to be handed to five stages and stitched, and Memory mode writes nothing (§4.1).
///     Because that combination is reachable from a default-on config key, it logs a
///     `WARN` rather than silently no-oping;
///   * `options.start_step() != Step::Detect` — a resumed run's cache entries belong to
///     the segments of the earlier run, and re-planning would orphan them. Also a `WARN`.
pub fn process_image_with_splitting(
    original: &Path,
    options: &PipelineOptions,
    ctx: &PipelineCtx<'_>,
) -> ImageOutcome {
    let original_buf = original.to_path_buf();
    let general = &options.profile.general;

    if !general.split_long_strips {
        return process_image(original, options, ctx);
    }
    // Cheap header-only probe: never decode the image twice just to answer the gate.
    let Ok(size) = image::image_dimensions(original) else {
        // Not a decodable image — let `process_image` report the real error.
        return process_image(original, options, ctx);
    };
    if !crate::strip::should_split(size, general) {
        return process_image(original, options, ctx);
    }
    if options.checkpointing == Checkpointing::Memory {
        tracing::warn!(
            image = %original.display(),
            "general.split_long_strips is on but long-strip splitting needs the cache; \
             processing this strip as a single page (drop --no-cache to enable splitting)"
        );
        return process_image(original, options, ctx);
    }
    if options.start_step() != Step::Detect {
        tracing::warn!(
            image = %original.display(),
            "general.split_long_strips is on but a resumed run reuses the earlier run's \
             segments; processing this strip as a single page"
        );
        return process_image(original, options, ctx);
    }

    warn_on_implied_skips(options);
    let cache = match prepare_cache(original, options) {
        Ok(Some(cache)) => cache,
        Ok(None) => return process_image(original, options, ctx),
        Err((step, error)) => return failed(original_buf, step, error),
    };

    let manifest = match crate::strip::plan_and_write_segments(original, &cache, general) {
        // §16.6 item 8: a plan with no split rows is an ordinary page after all.
        Ok(None) => return process_image(original, options, ctx),
        Ok(Some(manifest)) => manifest,
        Err(error) => return failed(original_buf, Step::Detect, error),
    };

    let mut segment_sources = Vec::with_capacity(manifest.segments.len());
    let mut analytics = ImageAnalytics::default();
    let mut no_text = true;
    for segment in &manifest.segments {
        let segment_cache = CachePaths::new(segment, &options.cache_dir);
        match run_stages(segment, Some(&segment_cache), options, ctx) {
            Ok(chain) => {
                no_text &= chain.no_text;
                merge_analytics(&mut analytics, chain.analytics, &original_buf);
                segment_sources.push(chain.sources);
            }
            Err((step, error)) => return failed(original_buf, step, error),
        }
    }

    match crate::strip::merged_strip_export(&manifest, &segment_sources, options) {
        Ok(output) => outcome_for(original_buf, no_text, output.files_written, analytics),
        Err(error) => failed(original_buf, Step::Export, error),
    }
}

/// §16.12 item 6's `WARN`, hoisted so both entry points emit it exactly once.
fn warn_on_implied_skips(options: &PipelineOptions) {
    if options.skips.implies_more() {
        tracing::warn!("normalising skip flags so resumed stages have their predecessors");
    }
}

/// [`cache_paths_for`] plus the `mkdir -p` both entry points need.
fn prepare_cache(
    original: &Path,
    options: &PipelineOptions,
) -> Result<Option<CachePaths>, (Step, StageError)> {
    let cache =
        cache_paths_for(original, options).map_err(|error| (options.start_step(), error))?;
    if let Some(cache) = cache.as_ref() {
        std::fs::create_dir_all(cache.cache_dir()).map_err(|source| {
            (
                options.start_step(),
                StageError::Io {
                    path: cache.cache_dir().to_path_buf(),
                    source,
                },
            )
        })?;
    }
    Ok(cache)
}

/// The single `pc_export::run` call of the non-split path (§16.11 item 1 owns the
/// destination rules; the pipeline only supplies `export_path` and `output_dir`).
fn export_once(
    original: &Path,
    sources: ExportSources,
    options: &PipelineOptions,
) -> Result<Vec<PathBuf>, (Step, StageError)> {
    ExportStage::run(
        ExportInput {
            schema_version: pc_core::SCHEMA_VERSION,
            original_path: original.to_path_buf(),
            export_path: original.to_path_buf(),
            output_dir: options.output_dir.clone(),
            outputs: options.requested_outputs(),
            sources,
            preferred_file_type: options.profile.general.cleaned_suffix(),
            preferred_mask_file_type: options.profile.general.preferred_mask_file_type.clone(),
            denoising_enabled: options.denoising_enabled(),
        },
        (),
    )
    .map(|output| output.files_written)
    .map_err(|error| (Step::Export, error))
}

/// §5.6 / §16.12 item 12 — one `ImageOutcome` per ORIGINAL input file.
fn outcome_for(
    original: PathBuf,
    no_text: bool,
    files_written: Vec<PathBuf>,
    analytics: ImageAnalytics,
) -> ImageOutcome {
    if no_text {
        ImageOutcome::Skipped {
            original,
            reason: SkipReason::NoTextDetected,
            files_written,
        }
    } else {
        ImageOutcome::Completed {
            original,
            files_written,
            analytics: Box::new(analytics),
        }
    }
}

/// §16.14 item 2 — collapse one segment's analytics into the strip's.
///
/// The per-page singletons accumulate (counts add, σ lists concatenate) and every `path`
/// is rewritten to the ORIGINAL strip, so the analytics printout has one row per user
/// input rather than one per segment.
fn merge_analytics(into: &mut ImageAnalytics, segment: ImageAnalytics, original: &Path) {
    if let Some(detect) = segment.detect {
        let total = into.detect.get_or_insert(pc_core::DetectAnalytic {
            path: original.to_path_buf(),
            blocks_detected: 0,
            blocks_kept: 0,
        });
        total.blocks_detected += detect.blocks_detected;
        total.blocks_kept += detect.blocks_kept;
    }
    if let Some(ocr) = segment.ocr {
        let total = into.ocr.get_or_insert(pc_core::OcrAnalytic {
            path: original.to_path_buf(),
            num_boxes: 0,
            box_areas_ocred: Vec::new(),
            box_areas_removed: Vec::new(),
            removed: Vec::new(),
        });
        total.num_boxes += ocr.num_boxes;
        total.box_areas_ocred.extend(ocr.box_areas_ocred);
        total.box_areas_removed.extend(ocr.box_areas_removed);
        total.removed.extend(ocr.removed);
    }
    for mut fitting in segment.mask_fitting {
        fitting.path = original.to_path_buf();
        into.mask_fitting.push(fitting);
    }
    if let Some(denoise) = segment.denoise {
        let total = into.denoise.get_or_insert(pc_core::DenoiseAnalytic {
            path: original.to_path_buf(),
            std_deviations: Vec::new(),
            boxes_denoised: 0,
        });
        total.std_deviations.extend(denoise.std_deviations);
        total.boxes_denoised += denoise.boxes_denoised;
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
