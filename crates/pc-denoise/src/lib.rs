//! `pc-denoise` — STAGE 4, denoising (spec §11).
//!
//! Masking leaves a hard-edged fill sitting on top of JPEG ringing. This stage hides
//! that seam: for every region whose mask fit was *imperfect but not failed*, it grows
//! and fades the region's mask, non-local-means-denoises the corresponding crop of the
//! stage-3 composite, and blends the denoised crop back through the faded mask. It
//! never touches a pixel outside those grown+faded region masks (§11.7(A)8).
//!
//! Module map (§11.1, §11.5, §16.10 items 1-3):
//!   * `nlm`        — joint-channel non-local-means (N1, heavy)
//!   * `gaussian`   — separable Gaussian blur, `sigma = radius` (N2)
//!   * `morph`      — growth kernel + binary dilation (§16.10 item 2)
//!   * `composite`  — RGBA source-over + nearest resize (§16.10 item 3)
//!   * `noise_mask` — region selection, crop, grow, fade, alpha attach, compose (N3)
//!   * this file    — `run()` wiring, the 1-bit shortcut, analytics (N4)
//!
//! §16.10 item 1: `nlm` and `gaussian` live here rather than in `pc-imageops` (which
//! §11.1 named) because `pc-imageops` was frozen at the end of Stage 3 and no other v1
//! stage uses either module. Both are `pub` and config-free, so a v1.5 hoist is a move
//! plus a re-export.
//!
//! `run()`'s wiring, the 1-bit shortcut and the analytics are implemented here (N4); its
//! signature is frozen with the tests.

pub mod composite;
pub mod gaussian;
pub mod morph;
pub mod nlm;
pub mod noise_mask;

pub use composite::{alpha_composite_over, blend_channel, composite_rgb, resize_nearest_rgba};
pub use morph::{dilate, kernel, Kernel};
pub use nlm::{denoise_call_count, reset_denoise_call_count, NlmParams};
pub use noise_mask::{
    alpha_binary, attach_alpha, blank_noise_mask, build_noise_mask, fade_mask, nlm_params,
    select_regions,
};

use image::DynamicImage;
use pc_config::DenoiserConfig;
use pc_core::{DenoiseAnalytic, ImageHandle, MaskData, StageError, Step};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenoiseInput {
    pub schema_version: u32,
    pub mask_data: MaskData,
    /// `= mask_data.original_path`, resolved. Needed for the 1-bit probe (§11.3 step 1)
    /// and as the full-resolution base canvas (§11.3 step 2).
    pub original_image: ImageHandle,
    /// Stage 3's `_clean.png`; used **only** for the 1-bit shortcut (§11.3 step 1).
    pub masked_image: ImageHandle,
    pub config: DenoiserConfig,
    pub dests: DenoiseDests,
}

/// Every destination is optional: `None` means "keep it in memory only" (§4.1's
/// `Checkpointing::Memory`). §16.10 item 4: the second field is `denoised` per §11.2;
/// §4.3's `clean_denoised` is a typo.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenoiseDests {
    /// `_noise_mask.png` (RGBA).
    pub noise_mask: Option<PathBuf>,
    /// `_clean_denoised.png`.
    pub denoised: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenoiseOutput {
    /// RGB(A)/L, original size (§16.10 item 17).
    pub denoised: ImageHandle,
    /// RGBA, original size; fully transparent when nothing was denoised.
    pub noise_mask: ImageHandle,
    /// One record per page (§16.10 item 5), `path = mask_data.original_path`.
    pub analytics: DenoiseAnalytic,
}

/// spec §3 — the stage contract. Denoising needs no external resource, so `Ctx` is `()`.
pub struct DenoiseStage;

impl pc_core::Stage for DenoiseStage {
    type Input = DenoiseInput;
    type Output = DenoiseOutput;
    type Ctx<'a> = ();
    const STEP: Step = Step::Denoise;

    fn run(input: Self::Input, _ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError> {
        run(input)
    }
}

/// spec §11.3 step 1's mode probe (§16.10 item 6) — a 26-byte header read.
///
/// Two facts force this shape, both verified against `image` 0.25.10: the crate has no
/// 1-bit `DynamicImage` variant (a 1 bpp PNG decodes to `Luma8` with values `0`/`255`),
/// **and** its PNG decoder reports `original_color_type() == L8` for such a file, so
/// neither the decoded value nor the decoder API carries the signal. The header does.
///
///   * PNG: `true` iff the `IHDR` bit depth is `1` and the colour type is `0`
///     (greyscale) — exactly PIL's mode `"1"`. Colour type `3` (1 bpp palette) is PIL
///     mode `"P"` and is therefore `false`.
///   * PBM (`P1`/`P4`): `true`, also PIL mode `"1"`.
///   * anything else, and any path-less (memory-only) handle: `false`.
pub fn is_one_bit(handle: &ImageHandle) -> Result<bool, StageError> {
    const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

    let Some(path) = handle.path.as_ref() else {
        return Ok(false);
    };
    if !path.exists() {
        return Ok(false);
    }
    let mut file = std::fs::File::open(path).map_err(|source| StageError::Io {
        path: path.clone(),
        source,
    })?;
    let mut header = [0_u8; 26];
    let mut filled = 0_usize;
    while filled < header.len() {
        let read = std::io::Read::read(&mut file, &mut header[filled..]).map_err(|source| {
            StageError::Io {
                path: path.clone(),
                source,
            }
        })?;
        if read == 0 {
            break;
        }
        filled += read;
    }

    // IHDR layout: 8 signature | 4 length | 4 "IHDR" | 4 width | 4 height | depth | type
    if filled == header.len() && header[..8] == PNG_SIGNATURE && &header[12..16] == b"IHDR" {
        return Ok(header[24] == 1 && header[25] == 0);
    }
    if filled >= 2 && header[0] == b'P' && matches!(header[1], b'1' | b'4') {
        return Ok(true);
    }
    Ok(false)
}

/// spec §11.3, steps 1–6, in exactly that order.
///
/// Step 1 — **1-bit shortcut** (§16.10 items 6, 7). If `is_one_bit(&original_image)`,
/// copy `masked_image` to `dests.denoised` (a byte-level `std::fs::copy` when both
/// paths are `Some` and differ; otherwise write/keep the loaded image), emit a fully
/// transparent noise mask of the **original** size, return
/// `DenoiseAnalytic { path, std_deviations: vec![], boxes_denoised: 0 }`, and call
/// `nlm::denoise` **zero** times.
///
/// Step 2 — **base canvas.** `cleaned = original.to_rgb8()`;
/// `mask = mask_data.combined_mask.load()?.to_rgba8()`. If the sizes differ,
/// `scale_up = cleaned.width() as f64 / mask.width() as f64` and the mask is
/// nearest-resized to `cleaned`'s dimensions (§16.10 item 13 — `mask_data.scale` is
/// never read); else `scale_up = 1.0` exactly. Then
/// `cleaned = composite::composite_rgb(&cleaned, &mask)`, reproducing stage 3's clean
/// output at full resolution rather than trusting `_clean.png`.
///
/// Step 3 — **select regions**: `noise_mask::select_regions(&mask_data.regions,
/// config.noise_min_standard_deviation)`.
///
/// Steps 4–5 — **build and composite**: `noise_mask::build_noise_mask(&cleaned, &mask,
/// &selected, scale_up, &config)` gives the RGBA `noise_mask` and `boxes_denoised`;
/// `denoised = composite::composite_rgb(&cleaned, &noise_mask)`, then §16.10 item 17's
/// output-mode rule (`ImageLuma8` iff the loaded original is `L`/`LA` **and** every
/// composited pixel is achromatic, else `ImageRgb8`). Write each destination that is
/// `Some`; the noise mask is always RGBA.
///
/// Step 6 — **analytics**: `std_deviations` is the sigma of **all** `mask_data.regions`
/// in region order (failed ones included — the CLI histogram shows them against the
/// cutoff), `boxes_denoised` is what step 4–5 returned (§16.10 item 15), and `path` is
/// `mask_data.original_path` (§16.10 item 5).
///
/// §16.10 item 18: `config.denoising_enabled` is **not** consulted here — skipping the
/// step is `pc-pipeline`'s job.
pub fn run(input: DenoiseInput) -> Result<DenoiseOutput, StageError> {
    let DenoiseInput {
        mask_data,
        original_image,
        masked_image,
        config,
        dests,
        ..
    } = input;

    if is_one_bit(&original_image)? {
        let original_size = original_image.dimensions()?;
        let masked = masked_image.load()?.as_ref().clone();
        let noise_mask = noise_mask::blank_noise_mask(original_size);

        if let Some(path) = &dests.denoised {
            match masked_image.path.as_ref() {
                Some(source) if source != path => {
                    std::fs::copy(source, path).map_err(|source| StageError::Io {
                        path: path.clone(),
                        source,
                    })?;
                }
                _ => write_png(&masked, path)?,
            }
        }
        let noise_image = DynamicImage::ImageRgba8(noise_mask.clone());
        if let Some(path) = &dests.noise_mask {
            write_png(&noise_image, path)?;
        }

        return Ok(DenoiseOutput {
            denoised: image_handle(&dests.denoised, masked),
            noise_mask: image_handle(&dests.noise_mask, noise_image),
            analytics: DenoiseAnalytic {
                path: mask_data.original_path,
                std_deviations: Vec::new(),
                boxes_denoised: 0,
            },
        });
    }

    let original = original_image.load()?;
    let original_is_gray = matches!(
        original.as_ref(),
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageLumaA8(_)
    );
    let mut cleaned = original.to_rgb8();
    let raw_mask = mask_data.combined_mask.load()?.to_rgba8();
    let scale_up = if raw_mask.dimensions() == cleaned.dimensions() {
        1.0
    } else {
        f64::from(cleaned.width()) / f64::from(raw_mask.width())
    };
    let mask = composite::resize_nearest_rgba(&raw_mask, cleaned.dimensions());
    cleaned = composite::composite_rgb(&cleaned, &mask);

    let selected =
        noise_mask::select_regions(&mask_data.regions, config.noise_min_standard_deviation);
    let (noise_mask, boxes_denoised) =
        noise_mask::build_noise_mask(&cleaned, &mask, &selected, scale_up, &config);
    let denoised_rgb = composite::composite_rgb(&cleaned, &noise_mask);
    let denoised = if original_is_gray && is_achromatic_image(&denoised_rgb) {
        DynamicImage::ImageRgb8(denoised_rgb).to_luma8().into()
    } else {
        DynamicImage::ImageRgb8(denoised_rgb)
    };
    let noise_image = DynamicImage::ImageRgba8(noise_mask);

    if let Some(path) = &dests.denoised {
        write_png(&denoised, path)?;
    }
    if let Some(path) = &dests.noise_mask {
        write_png(&noise_image, path)?;
    }

    Ok(DenoiseOutput {
        denoised: image_handle(&dests.denoised, denoised),
        noise_mask: image_handle(&dests.noise_mask, noise_image),
        analytics: DenoiseAnalytic {
            path: mask_data.original_path,
            std_deviations: mask_data
                .regions
                .iter()
                .map(|region| region.std_deviation)
                .collect(),
            boxes_denoised,
        },
    })
}

/// `ImageHandle::with_both` when a destination exists, `from_memory` otherwise — the
/// same idiom `pc-mask` uses.
pub fn image_handle(path: &Option<PathBuf>, image: DynamicImage) -> ImageHandle {
    match path {
        Some(path) => ImageHandle::with_both(path, image),
        None => ImageHandle::from_memory(image),
    }
}

/// Save `image` as PNG at `path`, mapping failures to `StageError::Io`.
pub fn write_png(image: &DynamicImage, path: &Path) -> Result<(), StageError> {
    image
        .save_with_format(path, image::ImageFormat::Png)
        .map_err(|error| StageError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::other(error),
        })
}

/// §16.10 item 17: `true` when every pixel of an RGB image satisfies `r == g == b`.
pub fn is_achromatic_image(image: &image::RgbImage) -> bool {
    image
        .pixels()
        .all(|pixel| pixel.0[0] == pixel.0[1] && pixel.0[1] == pixel.0[2])
}
