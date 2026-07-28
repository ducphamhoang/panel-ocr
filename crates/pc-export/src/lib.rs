//! `pc-export` — STAGE 5, export (spec §12).
//!
//! The pipeline's last step: take the artifacts the earlier stages produced for one
//! page and write the *user-facing* files — the cleaned image, optionally the mask,
//! optionally the isolated text layer — in the user's chosen formats, at the original
//! image's size and colour mode.
//!
//! Module map (§12.1, §12.5, §16.11 item 15):
//!   * `formats`    — suffix→format, per-format save options, colour modes, dpi (E1)
//!   * `discover`   — availability→precedence resolution + `--save-only-*` (E2)
//!   * `composite`  — nearest resize + source-over, for the mask branch (§16.11 item 10)
//!   * `ocr_report` — CSV/TXT report writers for `panel-ocr ocr` (E4)
//!   * this file    — destination resolution and `run()` wiring (E3)
//!
//! Two scope notes, both settled in §16.11 item 1:
//!   * **E5 (merged-strip export) is `pc-pipeline` work**, per §13 row 26. This crate
//!     receives an already re-pointed `export_path` and knows nothing about splits.
//!   * **PSD / layered export is v1.5** (§12.3's out-of-scope paragraph, §16).
//!
//! §1 rule 1's one documented exception lives here: `pc-export` resolves its own
//! *destination* paths (absolute-vs-relative `output_dir`, per-artifact filenames,
//! `mkdir -p`). Cache-path resolution and artifact *availability* remain exclusively
//! `pc-pipeline`'s job (§12.3 step 2).
//!
//! `run()` and its three per-category helpers are `todo!()` skeletons for task E3;
//! their signatures are frozen with the tests.
#![allow(unused_variables)]

pub mod composite;
pub mod discover;
pub mod formats;
pub mod ocr_report;

pub use composite::{alpha_composite_over, blend_channel, resize_nearest_rgba};
pub use discover::{Category, MaskChoice, Selection};
pub use formats::{ColorMode, OutputFormat};
pub use ocr_report::{render as render_ocr_report, ReportFormat};

use pc_core::{ImageHandle, Output, StageError, Step};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// spec §12.2. Not `PartialEq`: `ExportSources` holds `ImageHandle`s whose equality is
/// path-only, so tests compare the fields they actually mean (§16.11 item 13).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInput {
    pub schema_version: u32,
    /// For metadata (dpi, colour mode) and for the original size the mask is scaled to.
    pub original_path: PathBuf,
    /// Logical output identity — differs from `original_path` for merged strips (E5).
    pub export_path: PathBuf,
    /// Absolute => used as-is; relative => relative to `export_path.parent()`.
    pub output_dir: PathBuf,
    /// Requested outputs, NOT precedence-resolved (§12.3 step 2). Read through
    /// `discover::Category` (§16.11 item 2); an empty list requests nothing.
    pub outputs: Vec<Output>,
    pub sources: ExportSources,
    /// `None` (or `Some("")`) => keep the original's suffix (§16.11 item 4).
    pub preferred_file_type: Option<String>,
    /// Never empty; `".png"` by default (§6).
    pub preferred_mask_file_type: String,
    /// Config/`--skip-denoise` state. Excludes the denoise candidates from precedence
    /// even when they happen to be populated by a stale cache (§12.3 step 2).
    pub denoising_enabled: bool,
}

/// **Availability**, resolved by `pc-pipeline` before the call: `Some` iff that stage
/// artifact actually exists for this image (§12.3 step 2).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportSources {
    /// `_clean.png`
    pub masked: Option<ImageHandle>,
    /// `_clean_denoised.png`
    pub denoised: Option<ImageHandle>,
    /// `_combined_mask.png`
    pub final_mask: Option<ImageHandle>,
    /// `_noise_mask.png`
    pub denoise_mask: Option<ImageHandle>,
    /// `_text.png`
    pub isolated_text: Option<ImageHandle>,
}

/// spec §12.2 / §12.3 step 8. Ordered **cleaned, mask, text** (§16.11 item 4) and
/// listing exactly the files that exist on disk afterwards (§12.7(A)5).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportOutput {
    pub files_written: Vec<PathBuf>,
}

/// The three destination paths §12.3 step 1 derives, plus the `base` directory that
/// `run()` must `mkdir -p`. Computed for all three categories regardless of which are
/// requested — resolution is pure, writing is not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Destinations {
    pub base: PathBuf,
    /// `{base}/{stem}_clean{suffix}`
    pub cleaned: PathBuf,
    /// `{base}/{stem}_mask{preferred_mask_file_type}`
    pub mask: PathBuf,
    /// `{base}/{stem}_text{preferred_mask_file_type}`
    pub text: PathBuf,
}

/// spec §3 — the stage contract. Export needs no external resource, so `Ctx` is `()`.
pub struct ExportStage;

impl pc_core::Stage for ExportStage {
    type Input = ExportInput;
    type Output = ExportOutput;
    type Ctx<'a> = ();
    const STEP: Step = Step::Export;

    fn run(input: Self::Input, _ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError> {
        run(input)
    }
}

/// spec §12.3 step 1, pinned by §16.11 item 4. **Pure** — it creates no directories.
///
///   * `base = if output_dir.is_absolute() { output_dir } else { export_path.parent()
///     .unwrap_or("") / output_dir }`
///   * `stem = export_path.file_stem()`; missing => `StageError::InvalidInput`
///   * cleaned suffix = `preferred_file_type` (normalised) when `Some` and non-empty,
///     else `original_path`'s extension (normalised); neither => `InvalidInput`
///   * mask/text suffix = `preferred_mask_file_type` (normalised)
///   * names are `{stem}_clean{…}`, `{stem}_mask{…}`, `{stem}_text{…}` — upstream's
///     `OutputPathGenerator(export_mode=True)` suffixes.
pub fn destinations(input: &ExportInput) -> Result<Destinations, StageError> {
    let base = if input.output_dir.is_absolute() {
        input.output_dir.clone()
    } else {
        input
            .export_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(&input.output_dir)
    };

    let stem = input
        .export_path
        .file_stem()
        .ok_or_else(|| {
            formats::invalid_input(format!(
                "export_path `{}` has no file stem",
                input.export_path.display()
            ))
        })?
        .to_string_lossy()
        .into_owned();

    let cleaned_suffix = match input
        .preferred_file_type
        .as_deref()
        .filter(|suffix| !suffix.is_empty())
    {
        Some(suffix) => formats::normalize_suffix(suffix),
        None => formats::suffix_of(&input.original_path).ok_or_else(|| {
            formats::invalid_input(format!(
                "no preferred_file_type and original_path `{}` has no extension",
                input.original_path.display()
            ))
        })?,
    };
    let mask_suffix = formats::normalize_suffix(&input.preferred_mask_file_type);

    Ok(Destinations {
        cleaned: formats::artifact_path(&base, &stem, "_clean", &cleaned_suffix),
        mask: formats::artifact_path(&base, &stem, "_mask", &mask_suffix),
        text: formats::artifact_path(&base, &stem, "_text", &mask_suffix),
        base,
    })
}

/// spec §12.3 step 3 — the cleaned image (task **E3**).
///
/// Load `source`, convert it to `original_mode` (§16.11 item 5), then
/// `formats::save` it with the format `dest`'s suffix names, carrying `dpi` over when
/// the target format supports it (§16.11 item 8). The format-specific colour coercion
/// (§16.11 item 6) happens inside `formats::encode_to_vec`, so this function must NOT
/// pre-flatten.
pub fn export_cleaned(
    source: &ImageHandle,
    dest: &Path,
    original_mode: ColorMode,
    dpi: Option<(u32, u32)>,
) -> Result<(), StageError> {
    todo!("task E3: cleaned-image export (§12.3 step 3)")
}

/// spec §12.3 step 4 — the mask image (task **E3**).
///
/// Both branches load the combined mask as RGBA and
/// `composite::resize_nearest_rgba` it to `original_size` — nearest-neighbour
/// uniformly, `// DEVIATION(8)` / §15.8. `MaskChoice::WithDenoise` additionally
/// resizes the noise mask to the same size when it differs and
/// `composite::alpha_composite_over`s it at `(0, 0)`. The result is `Rgba8` before
/// `formats`' coercion, which is what makes §12.7(A)7's "every exported pixel's colour
/// occurs in the source mask" assertion exact. No dpi (§16.11 item 8).
pub fn export_mask(
    choice: &MaskChoice,
    dest: &Path,
    original_size: (u32, u32),
) -> Result<(), StageError> {
    todo!("task E3: mask export (§12.3 step 4)")
}

/// spec §12.3 step 5 — the isolated text layer (task **E3**).
///
/// RGBA is preserved. When `dest`'s format has no alpha channel
/// (`!OutputFormat::supports_alpha()`), emit exactly one `WARN` naming the suffix and
/// let `formats`' coercion flatten onto white — this must never be an error
/// (§12.7(A)8, §16.11 item 6). The text layer is written at its own size (it is
/// already an original-resolution artifact); no resize, no dpi.
pub fn export_text(source: &ImageHandle, dest: &Path) -> Result<(), StageError> {
    todo!("task E3: text-layer export (§12.3 step 5)")
}

/// spec §12.3, steps 1–8, in exactly that order (task **E3**).
///
/// 1. `destinations(&input)?`, then `std::fs::create_dir_all(&dests.base)` mapped to
///    `StageError::Io` (§16.11 item 4).
/// 2. `discover::resolve(&input.sources, &input.outputs, input.denoising_enabled)`.
/// 3. For each selected category, in the fixed order **cleaned, mask, text** (steps
///    3-5), call the helper above and push the destination onto `files_written`. The
///    original's colour mode comes from `formats::read_color_mode(&input.original_path)`
///    and its size from `image::image_dimensions` — both header-only reads (§16.11
///    items 5, 9). The cleaned dpi comes from `formats::read_dpi_lossy(...)`.
/// 8. `Ok(ExportOutput { files_written })`.
///
/// §16.11 item 14: an empty selection is `files_written: vec![]`, **not** an error.
/// Errors are reserved for an unresolvable destination, an unsupported suffix, a failed
/// `mkdir -p`, an unloadable source handle, and an encode/write failure.
pub fn run(input: ExportInput) -> Result<ExportOutput, StageError> {
    todo!("task E3: run() wiring (§12.3 steps 1-8)")
}
