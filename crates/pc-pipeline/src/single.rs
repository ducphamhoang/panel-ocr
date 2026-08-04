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
use crate::outcome::{ImageAnalytics, ImageOutcome, PipelineError, SkipReason};
use pc_core::{ImageHandle, OcrAnalytic, Output, Stage, StageError, Step};
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

/// §5 item 6's skip decision, declared per path rather than inferred from one field: the
/// `clean` and `ocr` paths mean different things by an empty `text_boxes` (§16.34).
///
///   * `clean` (`performing_ocr == false`) — unchanged, bit for bit: `text_boxes.is_empty()`
///     is exactly §5 item 6's trigger.
///   * `ocr` with an OCR analytic present — §15's overrides move every OCR'd box out of
///     `text_boxes` into `OcrAnalytic.removed` (§16.11 item 11), so `text_boxes` is empty *by
///     design*. The population §5 item 6 counts is `OcrAnalytic.num_boxes`, already defined by
///     §16.8 item 11 as "the number of tight boxes at entry to step 7 (pre-removal)" — no new
///     field is added anywhere.
///   * `ocr` with no analytic (no factory, or the pass never ran) — the pass never ran, so
///     nothing was consumed and `text_boxes` is still the whole population; falls back to the
///     `clean` reading.
fn no_text_for(performing_ocr: bool, text_boxes_empty: bool, ocr: Option<&OcrAnalytic>) -> bool {
    match (performing_ocr, ocr) {
        (true, Some(ocr)) => ocr.num_boxes == 0,
        _ => text_boxes_empty,
    }
}
/// Everything stages 1–4 produced for one image — or for one strip segment (§16.14
/// item 1) — before stage 5 turns it into user-facing files.
#[derive(Debug)]
pub struct ChainOutputs {
    /// §12.3 step 2's availability set, ready for `pc_export`.
    pub sources: ExportSources,
    pub analytics: ImageAnalytics,
    /// The per-path §5 item 6 `no_text_for` decision for this image or strip segment.
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
) -> Result<ChainOutputs, (Step, PipelineError)> {
    let original_buf = original.to_path_buf();
    let flags = options.skips.normalized();
    let source = ImageHandle::from_path(original);
    let mut analytics = ImageAnalytics::default();

    let raw = if flags.text_detection {
        let Some(cache) = cache else {
            return Err((
                Step::Detect,
                PipelineError::Stage(StageError::InvalidInput(
                    "cannot resume text detection without disk checkpoints".into(),
                )),
            ));
        };
        checkpoint::read_page_raw(&cache.for_output(Output::RawJson))
            .map_err(|error| (Step::Detect, PipelineError::Stage(error)))?
    } else {
        let detector = ctx.detectors.detector_for(original).map_err(|error| {
            // Fatality is declared by the provider, never inferred from the `StageError`
            // variant; both a provisioning refusal and a missing replay fixture arrive as
            // `StageError::Model`.
            let failure = if ctx.detectors.failures_are_run_fatal() {
                PipelineError::RunFatal(error)
            } else {
                PipelineError::Stage(error)
            };
            (Step::Detect, failure)
        })?;
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
        .map_err(|error| (Step::Detect, PipelineError::Stage(error)))?;
        if let Some(cache) = cache {
            checkpoint::write_page_raw(&output.page, &cache.for_output(Output::RawJson))
                .map_err(|error| (Step::Detect, PipelineError::Stage(error)))?;
        }
        analytics.detect = Some(output.analytics.clone());
        output.page
    };

    let page = if flags.preprocess {
        let Some(cache) = cache else {
            return Err((
                Step::Preprocess,
                PipelineError::Stage(StageError::InvalidInput(
                    "cannot resume preprocessing without disk checkpoints".into(),
                )),
            ));
        };
        checkpoint::read_page(&cache.for_output(Output::CleanJson))
            .map_err(|error| (Step::Preprocess, PipelineError::Stage(error)))?
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
        .map_err(|error| (Step::Preprocess, PipelineError::Stage(error)))?;
        if let Some(cache) = cache {
            checkpoint::write_page(&output.page, &cache.for_output(Output::CleanJson))
                .map_err(|error| (Step::Preprocess, PipelineError::Stage(error)))?;
        }
        analytics.ocr = output.ocr_analytic.clone();
        output.page
    };
    let no_text = no_text_for(
        options.performing_ocr,
        page.text_boxes.is_empty(),
        analytics.ocr.as_ref(),
    );

    // §13.1: `panel-ocr ocr`'s job is "run OCR over the detected boxes and write a
    // CSV/TXT report" — stages 1–2 plus a report, and it has no `--output-dir` to write
    // images to. Stop here so an `ocr` run never masks, denoises or exports.
    if options.performing_ocr {
        return Ok(ChainOutputs {
            sources: ExportSources::default(),
            analytics,
            no_text,
        });
    }

    let mask = if flags.mask {
        let Some(cache) = cache else {
            return Err((
                Step::Mask,
                PipelineError::Stage(StageError::InvalidInput(
                    "cannot resume masking without disk checkpoints".into(),
                )),
            ));
        };
        let mask_data = checkpoint::read_mask_data(&cache.for_output(Output::MaskDataJson))
            .map_err(|error| (Step::Mask, PipelineError::Stage(error)))?;
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
        .map_err(|error| (Step::Mask, PipelineError::Stage(error)))?;
        if let Some(cache) = cache {
            checkpoint::write_mask_data(&output.mask_data, &cache.for_output(Output::MaskDataJson))
                .map_err(|error| (Step::Mask, PipelineError::Stage(error)))?;
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
        .map_err(|error| (Step::Denoise, PipelineError::Stage(error)))?;
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
///   6. An input whose suffix is not in [`crate::discovery::SUPPORTED_INPUT_SUFFIXES`] is
///      `Skipped { UnsupportedFormat, files_written: [] }` — §5.1/§5.6 make an unsupported
///      format a graceful skip, so one stray non-image file among named inputs must not
///      turn an otherwise-successful batch into exit code 2. This is the per-image
///      boundary `crate::discovery::expand_inputs` defers to for explicitly named files.
///   7. With `options.performing_ocr` (`panel-ocr ocr`, §13.1) the chain stops after
///      preprocessing and **nothing is exported**: the run's only product is the report
///      `pc-cli` renders from `ImageAnalytics::ocr`.
pub fn process_image(
    original: &Path,
    options: &PipelineOptions,
    ctx: &PipelineCtx<'_>,
) -> ImageOutcome {
    let original_buf = original.to_path_buf();
    if !crate::discovery::is_supported_input(original) {
        return ImageOutcome::Skipped {
            original: original_buf,
            reason: SkipReason::UnsupportedFormat {
                suffix: crate::discovery::input_suffix(original),
            },
            files_written: Vec::new(),
        };
    }
    warn_on_implied_skips(options);

    let cache = match prepare_cache(original, options) {
        Ok(cache) => cache,
        Err((step, error)) => return failed(original_buf, step, error),
    };

    let chain = match run_stages(original, cache.as_ref(), options, ctx) {
        Ok(chain) => chain,
        Err((step, PipelineError::Stage(error))) => return failed(original_buf, step, error),
        Err((step, PipelineError::RunFatal(error))) => return run_fatal(original_buf, step, error),
    };

    if options.performing_ocr {
        return outcome_for(original_buf, chain.no_text, Vec::new(), chain.analytics);
    }

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
///   * `options.performing_ocr` — the whole point of the split path is
///     [`crate::strip::merged_strip_export`]'s single stitched *export*, and an `ocr` run
///     exports nothing (§13.1). Silent, because this is not a user-visible degradation:
///     `ocr` reports boxes in original-image space either way.
pub fn process_image_with_splitting(
    original: &Path,
    options: &PipelineOptions,
    ctx: &PipelineCtx<'_>,
) -> ImageOutcome {
    let original_buf = original.to_path_buf();
    let general = &options.profile.general;

    if options.performing_ocr {
        return process_image(original, options, ctx);
    }
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
            Err((step, PipelineError::Stage(error))) => return failed(original_buf, step, error),
            Err((step, PipelineError::RunFatal(error))) => {
                return run_fatal(original_buf, step, error)
            }
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

fn run_fatal(original: PathBuf, step: Step, error: StageError) -> ImageOutcome {
    ImageOutcome::RunFatal {
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

#[cfg(test)]
mod tests {
    use super::*;
    use pc_core::{OcrAnalytic, Rect, RemovedBox};

    /// An `OcrAnalytic` carrying the two fields these tests vary. `num_boxes` is §16.8
    /// item 11's "number of tight boxes at entry to step 7 (pre-removal)"; `removed` is
    /// the post-OCR survivor list §16.34 item 2 explicitly refuses to key the skip
    /// decision on.
    fn analytic(num_boxes: usize, removed: Vec<RemovedBox>) -> OcrAnalytic {
        OcrAnalytic {
            path: PathBuf::from("page01.png"),
            num_boxes,
            box_areas_ocred: Vec::new(),
            box_areas_removed: Vec::new(),
            removed,
        }
    }

    fn a_removed_box() -> RemovedBox {
        RemovedBox {
            text: "ハロー".to_string(),
            rect: Rect::new(18, 18, 85, 62),
        }
    }

    /// §16.34 item 2 — the whole truth table of `no_text_for`, one row per line, each
    /// expectation transcribed from item 2's doc comment rather than computed from the
    /// function:
    ///
    /// | `performing_ocr` | `text_boxes_empty` | `ocr`             | expected | why (§16.34 item 2) |
    /// |---|---|---|---|---|
    /// | `false` | `true`  | `None`            | `true`  | `clean`, unchanged: §5 item 6's trigger |
    /// | `false` | `true`  | `Some{num_boxes:2}` | `true` | `clean` **control**: the OCR filter discarded every box, so the page has nothing to mask and is still a skip — the predicate must not key off `ocr` being `Some` |
    /// | `false` | `false` | `None`            | `false` | `clean`, boxes survive |
    /// | `false` | `false` | `Some{num_boxes:2}` | `false` | same, with an analytic present |
    /// | `true`  | `true`  | `Some{num_boxes:2}` | `false` | the defect row: §15's overrides emptied `text_boxes` by design, and 2 boxes entered step 7 |
    /// | `true`  | `true`  | `Some{num_boxes:0}` | `true`  | the `ocr` path genuinely saw no boxes |
    /// | `true`  | `true`  | `None`            | `true`  | the fallback arm: the pass never ran, so `text_boxes` is still the whole population |
    ///
    /// The `(false, false, _)` row of item 2's enumeration is expanded into both of its
    /// `ocr` instantiations, which is strictly stronger than the single `_` row.
    ///
    /// What turns this red: reverting the call site to `text_boxes.is_empty()` fails row 5;
    /// dropping the `performing_ocr` guard (keying on `ocr.is_some()` alone) fails row 2;
    /// matching `(true, None)` to `false` fails row 7.
    #[test]
    fn no_text_for_matches_the_truth_table_declared_in_16_34_item_2() {
        // clean path (`performing_ocr == false`): `text_boxes_empty` decides, alone.
        assert!(
            no_text_for(false, true, None),
            "row 1: clean + empty text_boxes is §5 item 6's skip"
        );
        assert!(
            no_text_for(false, true, Some(&analytic(2, Vec::new()))),
            "row 2: clean + every box OCR-filtered away is still a skip (the control)"
        );
        assert!(
            !no_text_for(false, false, None),
            "row 3: clean + surviving boxes is not a skip"
        );
        assert!(
            !no_text_for(false, false, Some(&analytic(2, Vec::new()))),
            "row 4: clean + surviving boxes is not a skip, analytic or not"
        );

        // report path (`performing_ocr == true`): `OcrAnalytic::num_boxes` decides when an
        // analytic exists, and `text_boxes_empty` decides when it does not.
        assert!(
            !no_text_for(true, true, Some(&analytic(2, vec![a_removed_box()]))),
            "row 5: the defect — 2 boxes entered step 7, so the page HAS text"
        );
        assert!(
            no_text_for(true, true, Some(&analytic(0, Vec::new()))),
            "row 6: the ocr path saw zero boxes"
        );
        assert!(
            no_text_for(true, true, None),
            "row 7: no analytic — fall back to the clean reading of text_boxes"
        );
    }

    /// §16.34 item 2's *rejected candidate*, pinned so it cannot be reintroduced: gating on
    /// `ocr.removed.is_empty()` instead of `ocr.num_boxes == 0` "would report an OCR
    /// **failure** as an **absence**".
    ///
    /// Both `removed` shapes are pinned to a hard-coded expectation rather than to each
    /// other (a `f(a) == f(b)` pair would also hold for a predicate that returned the same
    /// wrong value twice — cookbook rule 1). The `num_boxes == 0` row carries a non-empty
    /// `removed`, a shape the pipeline never produces, asserted only to prove the predicate
    /// ignores that field in the other direction too.
    ///
    /// What turns this red: implementing the rejected `removed.is_empty()` predicate — it
    /// flips rows 1 and 3 below.
    #[test]
    fn no_text_for_ignores_the_removed_list_and_reads_num_boxes_only() {
        assert!(
            !no_text_for(true, true, Some(&analytic(2, Vec::new()))),
            "an all-boxes-failed-OCR page (num_boxes 2, removed []) has text, not absence"
        );
        assert!(
            !no_text_for(true, true, Some(&analytic(2, vec![a_removed_box()]))),
            "the same num_boxes with a populated removed list reads the same way"
        );
        assert!(
            no_text_for(true, true, Some(&analytic(0, vec![a_removed_box()]))),
            "num_boxes == 0 reads as no-text regardless of removed"
        );
    }
}
