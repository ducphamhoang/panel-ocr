//! spec §4.3 — the single-image chain: `PageDataRaw` → `PageData` → `MaskData` →
//! denoised → exported files, with checkpoint writes between stages and the four skip
//! levels loading from cache instead.
//!
//! The dest-builders and the cache-path resolver are fully pinned and implemented here.
//! [`process_image`] itself is `todo!()` for Codex (task G1-chain, **heavy** per §16.12
//! item 19): it is the one place where five stage contracts, two checkpointing modes and
//! four skip levels meet.

use crate::cache::CachePaths;
use crate::ctx::PipelineCtx;
use crate::options::{Checkpointing, PipelineOptions};
use crate::outcome::ImageOutcome;
use pc_core::{Output, StageError};
use pc_denoise::{DenoiseDests, DenoiseOutput};
use pc_export::ExportSources;
use pc_mask::{MaskDests, MaskOutput};
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
    let _ = (original, options, ctx);
    todo!("task G1-chain (spec §4.3): five-stage orchestration, checkpoints, skips")
}
