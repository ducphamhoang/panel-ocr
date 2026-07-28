//! `pc-mask` — STAGE 3, masking (spec §10). **This is the heart of the project.**
//!
//! Rather than inpainting, PanelCleaner grows the AI text mask outward and scores the
//! *uniformity of the pixels the mask's own border sits on*, keeps the largest mask
//! whose border is most uniform, and fills it with that border's median colour. If
//! nothing is uniform enough, the box is left alone and reported as a failure.
//!
//! Module map (§10.1, §10.5, §16.9 item 2):
//!   * `grow`    — kernels + iterative padded dilation → the candidate sequence (M2)
//!   * `border`  — edge extraction + border std dev + median colours (M3)
//!   * `fit`     — `fit_region`, the candidate scoring/selection policy (M4)
//!   * `combine` — RGBA composition, cleaned image, text layer, overlay (M5)
//!   * this file — `run()` wiring, `MaskData`/analytics emission, debug writes (M6)
//!
//! `BinaryMask` and box-mask rasterisation live in `pc-imageops` (M1).
//!
//! `run()` is a `todo!()` skeleton for task M6; its signature is frozen with the tests.
#![allow(unused_variables)]

pub mod border;
pub mod combine;
pub mod fit;
pub mod grow;

pub use border::{border_std_deviation, BaseCanvas, BlankMask, BorderStats};
pub use combine::{build_combined_mask, cleaned_image, mask_overlay, text_layer};
pub use fit::{fit_region, select_candidate, Fitment, Selected};
pub use grow::{build_candidates, growth_candidates, kernel, Candidate, Kernel};

use pc_config::MaskerConfig;
use pc_core::{
    ImageHandle, MaskData, MaskFittingAnalytic, MaskRegionStats, PageData, StageError, Step,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskInput {
    pub schema_version: u32,
    pub page: PageData,
    /// The user's original (unscaled) image — needed to produce the cleaned output when
    /// `scale != 1.0`. Never loaded when `scale == 1.0` (§16.9 item 13).
    pub original_image: ImageHandle,
    pub config: MaskerConfig,
    pub extract_text: bool,
    /// `--cache-masks`.
    pub debug_outputs: bool,
    pub dests: MaskDests,
}

/// Every destination is optional: `None` means "keep it in memory only" (§4.1's
/// `Checkpointing::Memory`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskDests {
    /// `_combined_mask.png` (RGBA).
    pub combined_mask: Option<PathBuf>,
    /// `_clean.png`.
    pub cleaned: Option<PathBuf>,
    /// `_text.png` — iff `extract_text`.
    pub text_layer: Option<PathBuf>,
    /// `_box_mask.png` — iff `debug_outputs`.
    pub box_mask: Option<PathBuf>,
    /// `_cut_mask.png` — iff `debug_outputs`.
    pub cut_mask: Option<PathBuf>,
    /// `_with_masks.png` — iff `debug_outputs`.
    pub mask_overlay: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskOutput {
    pub mask_data: MaskData,
    /// RGBA, `base_image` size.
    pub combined_mask: ImageHandle,
    /// Original size (§16.9 item 13).
    pub cleaned: ImageHandle,
    pub text_layer: Option<ImageHandle>,
    /// 1:1 with `mask_data.regions`, in region order (§16.9 item 19).
    pub analytics: Vec<MaskFittingAnalytic>,
}

/// spec §3 — the stage contract. Masking needs no external resource, so `Ctx` is `()`.
pub struct MaskStage;

impl pc_core::Stage for MaskStage {
    type Input = MaskInput;
    type Output = MaskOutput;
    type Ctx<'a> = ();
    const STEP: Step = Step::Mask;

    fn run(input: Self::Input, _ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError> {
        run(input)
    }
}

/// spec §10.3, steps 0–5, in exactly that order.
///
/// Step 0: load `base_image`, threshold `raw_mask` at `> 127` into the precise mask.
/// Step 1: rasterise `page.extended_boxes` into the box mask (exclusive rects, §16.9
/// item 1); write it iff `dests.box_mask`.
/// Step 2: `cut = precise AND box_mask`; write iff `dests.cut_mask`.
/// Step 3: `fit_region` per `MaskingRegion`, **in order**; `None` drops the region
/// entirely (no analytics entry, no `MaskData` entry).
/// Step 4: compose the combined mask, the cleaned image, and — when asked — the text
/// layer and the debug overlay; write each destination that is `Some`.
/// Step 5: emit `MaskData` (one `MaskRegionStats` per fitment, failures included, in
/// region order) and the matching `Vec<MaskFittingAnalytic>`.
///
/// Per §16.9 item 19 this validates `page` at entry (`StageError::InvalidInput`), and a
/// page with no masking regions is a success: a fully transparent combined mask, a
/// cleaned image equal to the canvas, and empty `regions`/`analytics`.
pub fn run(input: MaskInput) -> Result<MaskOutput, StageError> {
    input.page.validate()?;

    let MaskInput {
        schema_version,
        page,
        original_image,
        config,
        extract_text,
        debug_outputs,
        dests,
    } = input;
    let base_image = page.base_image.load()?;
    let base = BaseCanvas::from_dynamic(&base_image);
    let precise = pc_imageops::BinaryMask::from_gray_threshold(
        &page.raw_mask.load()?.to_luma8(),
        pc_imageops::PIL_BINARY_THRESHOLD,
    );
    let box_mask = pc_imageops::rasterize_boxes(&page.extended_boxes, page.image_size);
    if debug_outputs {
        if let Some(path) = &dests.box_mask {
            write_png(&image::DynamicImage::ImageLuma8(box_mask.to_gray()), path)?;
        }
    }
    let cut = precise.and(&box_mask);
    if debug_outputs {
        if let Some(path) = &dests.cut_mask {
            write_png(&image::DynamicImage::ImageLuma8(cut.to_gray()), path)?;
        }
    }

    let fitments = page
        .masking_regions
        .iter()
        .filter_map(|region| {
            fit_region(
                &base,
                &cut,
                &box_mask,
                region.masking,
                region.reference,
                &config,
            )
        })
        .collect::<Vec<_>>();
    let combined = build_combined_mask(&fitments, page.image_size);
    let combined_image = image::DynamicImage::ImageRgba8(combined.clone());
    if let Some(path) = &dests.combined_mask {
        write_png(&combined_image, path)?;
    }
    let combined_handle = image_handle(&dests.combined_mask, combined_image);

    let canvas = if page.scale != 1.0 {
        original_image.load()?.as_ref().clone()
    } else {
        base_image.as_ref().clone()
    };
    let all_achromatic = fitments
        .iter()
        .all(|fitment| combine::is_achromatic(fitment.median_color));
    let cleaned = cleaned_image(&canvas, &combined, all_achromatic);
    if let Some(path) = &dests.cleaned {
        write_png(&cleaned, path)?;
    }
    let cleaned_handle = image_handle(&dests.cleaned, cleaned);

    let text_layer = extract_text
        .then(|| {
            let text = text_layer(&canvas, &combined);
            if let Some(path) = &dests.text_layer {
                write_png(&image::DynamicImage::ImageRgba8(text.clone()), path)?;
            }
            Ok::<ImageHandle, StageError>(image_handle(
                &dests.text_layer,
                image::DynamicImage::ImageRgba8(text),
            ))
        })
        .transpose()?;

    if debug_outputs {
        if let Some(path) = &dests.mask_overlay {
            let overlay = mask_overlay(&base_image, &combined, config.debug_mask_color);
            write_png(&image::DynamicImage::ImageRgb8(overlay), path)?;
        }
    }

    let regions = fitments
        .iter()
        .map(|fitment| MaskRegionStats {
            rect: fitment.masking_rect,
            std_deviation: fitment.std_deviation,
            failed: fitment.failed(),
            thickness: fitment.thickness,
        })
        .collect();
    let analytics = fitments
        .iter()
        .map(|fitment| MaskFittingAnalytic {
            path: page.original_path.clone(),
            fit_found: !fitment.failed(),
            candidate_index: fitment.candidate_index,
            std_deviation: fitment.std_deviation,
            thickness: fitment.thickness,
        })
        .collect();
    let mask_data = MaskData {
        schema_version,
        original_path: page.original_path,
        base_image: page.base_image,
        combined_mask: combined_handle.clone(),
        scale: page.scale,
        regions,
    };

    Ok(MaskOutput {
        mask_data,
        combined_mask: combined_handle,
        cleaned: cleaned_handle,
        text_layer,
        analytics,
    })
}

fn image_handle(path: &Option<PathBuf>, image: image::DynamicImage) -> ImageHandle {
    match path {
        Some(path) => ImageHandle::with_both(path, image),
        None => ImageHandle::from_memory(image),
    }
}

fn write_png(image: &image::DynamicImage, path: &std::path::Path) -> Result<(), StageError> {
    image
        .save_with_format(path, image::ImageFormat::Png)
        .map_err(|error| StageError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::other(error),
        })
}
