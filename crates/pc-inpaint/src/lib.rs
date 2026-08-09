//! `pc-inpaint` — STAGE 5, LaMa inpainting (spec §16.38).
//!
//! Task **L4** (§16.38 item 16(c)) is everything in this crate except the ONNX session:
//! the eligibility filter (item 3(c)), the two fill-mask sources (3(d)), the growth
//! arithmetic (3(e)), the page-global fill and isolation masks (3(f)), the tile cover
//! (items 5(b)–(e)) and the compositing (3(h)). All of it sits behind the [`Inpainter`]
//! trait, whose test double returns a fixed tile, so **every L4 test here runs in the
//! default no-`onnx` tier** — the tier cookbook rule 6 records as the one that actually
//! executes.
//!
//! Task **L5** (item 16(d)) has since added the `ort`-backed implementation of that one
//! trait method, in `onnx`, behind the non-default `onnx` feature. It changed nothing L4
//! pinned: no L4 test was edited, and `onnx.rs`'s own tensor arithmetic and runtime
//! output-shape validation are ungated, so they run in the default tier too.
//!
//! Module map:
//!   * `eligible` — the two eligibility sets and their fill-mask sources (items 3(c), 3(d))
//!   * `growth`   — the growth arithmetic and the padded box (item 3(e))
//!   * `fill`     — the page-global fill and isolation masks (items 3(d), 3(f), 10)
//!   * `tile`     — merged cover, the 512-lattice, and pixel ownership (items 5(b)–(e))
//!   * `fade`     — the Gaussian fade and the isolation cut (item 3(h))
//!   * `onnx`     — L5: the two NCHW input tensors (item 1(b)), the **runtime** output-shape validation (item 1(c)), and — behind the `onnx` feature — `OnnxInpainter`, built through `pc_core::device::resolve` (item 16(d))
//!   * `compose`  — RGBA source-over + nearest resize; a **re-export** of [`pc_imageops::composite`], hoisted from what was a fourth pinned copy (after `pc-mask`, `pc-denoise` and `pc-export`) by §16.42, which overturned §16.10 item 3's per-crate pin
//!   * `stub`     — the `testkit`-gated test double
//!   * this file  — [`inpaint_page`], the tile loop and the page assembly
//!
//! **What this crate deliberately does NOT do.** No `Step::Inpaint`, no cache-suffix
//! constants, no `ExportSources` and no `--skip-inpaint`: those are L6 (§16.38 item 16(e)).
//! No model **acquisition** and no lazy latch either — item 8 puts the provider in
//! `crates/pc-cli/` (`DEVIATION(27)`), which is where L5 landed it, so this crate takes a
//! path and nothing else. And per item 18(c) it does not modify `pc-mask` —
//! `MaskRegionStats` already matches upstream's `boxes_with_stats` field for field (item
//! 3(b)).

pub mod compose;
pub mod eligible;
pub mod fade;
pub mod fill;
pub mod growth;
pub mod onnx;
pub mod tile;

#[cfg(any(test, feature = "testkit"))]
pub mod stub;

pub use eligible::{select_regions, EligibleRegion, FillSource};
pub use fade::{cut_by_isolation, fade_fill_mask};
pub use fill::{
    combined_fill_binary, compose_page_masks, padded_region, resize_nearest_binary, scale_rect_to,
    PaddedRegion, PageMasks,
};
pub use growth::{growth, padded_box};
pub use tile::{merge_rects, owner_map, tile_cover, tile_windows, TileWindow, UNOWNED};

use image::{GrayImage, Rgb, RgbImage, Rgba, RgbaImage};
use pc_config::InpainterConfig;
use pc_core::{MaskRegionStats, StageError};
use pc_imageops::{BinaryMask, PIL_BINARY_THRESHOLD};

/// The tile side length, in pixels.
///
/// **This is a hard constraint of the pinned artifact, not a tuning knob.** §16.38 item
/// 1(b) decoded `lama-manga.onnx`'s `image` input as `[batch, 3, 512, 512]` and its `mask`
/// input as `[batch, 1, 512, 512]`, with the `512`s as literal `dim_value`s; item 1(d)
/// then fed a real `ort` session 256×256, 640×512 and 1024×1024 inputs and got three
/// identical rejections. Changing this constant does not change what the model accepts.
pub const TILE: u32 = 512;

/// The seam L5 replaces, and the only thing it replaces.
///
/// One call per 512×512 tile. §16.38 item 1(d) measured the `batch` axis as genuinely
/// dynamic but **linear** in wall clock (1.58 / 3.03 / 6.55 s for b = 1 / 2 / 4), so
/// batching buys no throughput and is deliberately not in this signature.
///
/// **Fatality is declared, not inferred** (cookbook rule 4). This method receives the
/// image, so its failures are **per-image**: an implementation returns
/// [`StageError::Inference`] and the pipeline records `Failed { step: Inpaint }`, exit 2
/// (§16.38 item 9(g)). *Construction* failures are the other half of that split and are
/// **run-fatal**, rendered as [`StageError::Model`] — but construction is not part of this
/// trait, precisely because it takes no image (item 9(b)); it lives in L5's provider in
/// `pc-cli`, lazily and latched (item 8, `DEVIATION(27)`).
pub trait Inpainter: Send + Sync {
    /// `tile`: RGB8, exactly [`TILE`]×[`TILE`].
    ///
    /// `mask`: exactly [`TILE`]×[`TILE`]; **set = fill, clear = keep**. §16.38 item 1(g)
    /// confirmed that convention against the real artifact (value 1 = fill, 0 = keep).
    ///
    /// The return must be RGB8 and exactly [`TILE`]×[`TILE`]; §16.38 item 1(c) requires L5
    /// to validate that at *run time* and never to trust the graph's declared output shape,
    /// three of whose four axes are symbolic and one of which reuses the batch symbol for
    /// height.
    ///
    /// **The implementation is NOT expected to composite.** §16.38 item 1(g) measured a
    /// zero mask leaving only 1 of 786,432 pixels bit-identical to the input, so the caller
    /// blends the kept region back from the original. [`inpaint_page`] does that; an
    /// implementation of this trait must not try to help.
    fn inpaint_tile(&self, tile: &RgbImage, mask: &BinaryMask) -> Result<RgbImage, StageError>;
}

/// Everything [`inpaint_page`] reads, already decoded. Nothing here touches the filesystem:
/// L6 owns loading and writing (§16.38 item 16(e)).
pub struct PageInput<'a> {
    /// The original at full resolution, already RGB (upstream `inpainting.py:65`).
    pub original: &'a RgbImage,
    /// `_raw_mask.png` — `PageData::raw_mask` (§16.38 item 3(a)), in the **mask frame**.
    pub raw_mask: &'a GrayImage,
    /// `MaskData::combined_mask` (RGBA), in the **mask frame**. Both mask-frame images must
    /// share dimensions; a mismatch is [`StageError::InvalidInput`].
    pub combined_mask: &'a RgbaImage,
    /// `_noise_mask.png`, at the original size, when denoising ran (upstream `:155-157`).
    pub noise_mask: Option<&'a RgbaImage>,
    /// `MaskData::regions` — upstream's `boxes_with_stats`, field for field (item 3(b)).
    pub regions: &'a [MaskRegionStats],
    /// `masker.min_mask_thickness`: the growth applied to a **failed** region's raw-mask
    /// crop before the radius growth (item 3(d), upstream `:96`).
    pub min_mask_thickness: u32,
    pub config: &'a InpainterConfig,
}

/// What the stage produces. Both images are RGBA and at the original size.
#[derive(Debug, Clone)]
pub struct PageOutput {
    /// `_inpainting.png` (upstream `output_structures.py:409`): the inpainted RGB with
    /// `final_mask` as its alpha.
    pub inpainting: RgbaImage,
    /// `_clean_inpaint.png` (`:413`): the rebuilt cleaned page with the inpainted areas
    /// composited over it, alpha flattened to 255.
    pub clean_inpaint: RgbaImage,
    /// One `growth` per eligible region, in eligibility order — upstream's
    /// `analytics_thicknesses` (`:119`, `:178`).
    pub growths: Vec<u32>,
    /// How many times [`Inpainter::inpaint_tile`] was called. `0` when nothing was
    /// eligible, which is upstream's `if boxes_to_inpaint:` guard at `:131`.
    pub tiles_inferred: usize,
}

/// spec §16.38 items 3(c)–3(h) plus items 5(b)–(e), in that order.
///
/// The shape, so the tile loop below is readable:
///
/// 1. eligibility (item 3(c)) and the two fill sources (3(d));
/// 2. per region: growth, padded box, padded-frame fill mask (3(e));
/// 3. page-global fill and isolation masks, NEAREST-resized to the original size (3(f));
/// 4. the merged tile cover and per-pixel ownership (5(b)–(e));
/// 5. the fade and the isolation cut (3(h));
/// 6. the tile loop — `DEVIATION(24)`, see below;
/// 7. the page assembly (3(h)).
pub fn inpaint_page(
    input: PageInput<'_>,
    inpainter: &dyn Inpainter,
) -> Result<PageOutput, StageError> {
    let PageInput {
        original,
        raw_mask,
        combined_mask,
        noise_mask,
        regions,
        min_mask_thickness,
        config,
    } = input;

    let mask_frame = combined_mask.dimensions();
    if raw_mask.dimensions() != mask_frame {
        return Err(StageError::InvalidInput(format!(
            "raw mask is {:?} but the combined mask is {mask_frame:?}; \
             both are written in the same (scaled) frame",
            raw_mask.dimensions()
        )));
    }
    let original_size = original.dimensions();
    if let Some(noise) = noise_mask {
        if noise.dimensions() != original_size {
            return Err(StageError::InvalidInput(format!(
                "noise mask is {:?} but the original is {original_size:?}",
                noise.dimensions()
            )));
        }
    }

    // (1) + (2).
    let eligible = select_regions(regions, config);
    // §16.38 item 21(d): these two lines binarise two masks TWO DIFFERENT WAYS, deliberately.
    // Do not unify them. `_raw_mask.png` is greyscale, has no alpha to discard and is already
    // strictly bilevel, so PIL's `convert("1")` dithering is a no-op on it and a strict `> 127`
    // luma threshold (§16.9 item 4) is exactly upstream's behaviour. `_combined_mask.png` is
    // RGBA, and reading its luma is DEVIATION(28)'s bug (§14 item 28) — it must be read on
    // alpha. Routing the combined mask through `from_gray_threshold` reintroduces that bug;
    // routing the raw mask through an alpha test has no channel to read.
    let raw_binary = BinaryMask::from_gray_threshold(raw_mask, PIL_BINARY_THRESHOLD);
    let combined_binary = combined_fill_binary(combined_mask);
    let padded: Vec<PaddedRegion> = eligible
        .iter()
        .map(|region| {
            padded_region(
                region,
                &raw_binary,
                &combined_binary,
                min_mask_thickness,
                config,
                mask_frame,
            )
        })
        .collect();
    let growths: Vec<u32> = padded.iter().map(|region| region.growth).collect();

    // (3).
    let frame_masks = compose_page_masks(&padded, mask_frame, config.inpainting_isolation_radius);
    let fill = resize_nearest_binary(&frame_masks.fill, original_size);
    let isolation = resize_nearest_binary(&frame_masks.isolation, original_size);

    // (5) before (4), because the tile cover is a function of the write region and the write
    // region is `final_mask`. See the note below.
    let faded = fade_fill_mask(&fill, config.inpainting_fade_radius);
    let final_mask = cut_by_isolation(&faded, &isolation);

    // (4). The padded boxes are in the mask frame; the tiling runs at the original size, so
    // they are carried across with the SAME nearest-resize mapping the masks used —
    // `fill::scale_rect_to` inverts `resize_nearest_binary` exactly, which is what keeps
    // every resized write pixel inside some merged rectangle and therefore owned.
    //
    // **The write region is `final_mask > 0`, NOT the fill mask — and this reading of §16.38
    // item 5(e) is a decision, escalated rather than assumed.** Item 5(e)'s two halves pull
    // apart: its heading and its ownership sentence both say *"fill pixel"* ("Every fill pixel
    // is written exactly once"; "a fill pixel belongs to the first window in that order"),
    // while the same sentence ends "and write-back touches only owned pixels, **masked by
    // `faded_fill AND isolation`**" — a set that strictly CONTAINS the fill mask, since the
    // fade spreads outward and the isolation mask is the fill grown by
    // `inpainting_isolation_radius`. Under the narrow reading that trailing clause is
    // redundant, and worse, the fade band would blend the original against the original and
    // produce no soft edge at all — silently disabling the feature item 3(h) exists for.
    //
    // Upstream decides it, and it was READ rather than recalled (`inpainting.py:167-170`):
    // `inpainted_image.putalpha(final_mask)` then `cleaned_image.alpha_composite(...)`, so the
    // fade band composites MODEL output over the cleaned page. Item 3(h) transcribes exactly
    // that. So ownership is assigned over `final_mask > 0`; "every fill pixel is written
    // exactly once" stays true, because the fill mask is a subset of it.
    //
    // Consequence, stated because it is a visible difference from item 5(c)'s literal text:
    // windows are dropped when they miss the **write region**, not when they miss the fill
    // mask. That drops a subset of what the literal rule drops — never more — so no write
    // pixel can end up unowned, at the cost of an occasional extra inference call for a window
    // that holds only fade band. The alternative leaves silent holes.
    let write_region = BinaryMask::from_gray_threshold(&final_mask, 0);
    let scaled: Vec<pc_core::Rect> = padded
        .iter()
        .map(|region| fill::scale_rect_to(region.padded, mask_frame, original_size))
        .collect();
    let merged = merge_rects(&scaled);
    let tiling_canvas = (original_size.0.max(TILE), original_size.1.max(TILE));
    let cover = tile_cover(&merged, &write_region, tiling_canvas);
    let owners = owner_map(&cover, &write_region);

    // (6) DEVIATION(24) — per-tile 512x512 inference in place of upstream's single
    // full-page call (§14 item 24, §16.38 item 4). Upstream calls its model once per page
    // on the full-resolution original with one page-global mask (`inpainting.py:132`,
    // guarded by `if boxes_to_inpaint:` at `:131`) and tolerates any page size, because its
    // artifact is a fully-convolutional TorchScript generator. This is FORCED, not chosen:
    // §16.38 item 1(b) measured the ONNX artifact's H and W axes as literal 512s and item
    // 1(d) measured three off-size inputs being rejected. The 512-px context per call is a
    // genuine behaviour difference from upstream's page-wide context, its visual
    // consequence is UNMEASURED, and §16.38 item 17(b) (decision point D2) keeps that open
    // — no clause anywhere may claim visual parity with upstream's inpainting.
    //
    // Two consequences of tiling that are visible in the artifact and are stated rather
    // than left to be discovered:
    //   * outside the windows the model never ran, so `_inpainting.png` carries the
    //     ORIGINAL pixels there rather than model output. Those pixels have alpha 0 in
    //     `final_mask`, so `_clean_inpaint.png` is unaffected; upstream's own
    //     `_inpainting.png` has model output there because it ran the model page-wide.
    //   * item 5(f)'s seam risk: a fill region spanning a lattice boundary is generated
    //     from two different 512 contexts. Item 5(c)'s centred window removes this for
    //     every region that fits, which the default radii make the common case.
    let mut inpainted = original.clone();
    let mut tiles_inferred = 0_usize;
    for (index, window) in cover.iter().enumerate() {
        let tile = crop_replicate_rgb(original, window.window, TILE);
        let tile_mask = crop_replicate_mask(&fill, window.window, TILE);
        let filled = inpainter.inpaint_tile(&tile, &tile_mask)?;
        tiles_inferred += 1;
        if filled.dimensions() != (TILE, TILE) {
            return Err(StageError::Inference(format!(
                "inpainter returned a {:?} tile; the pinned artifact is {TILE}x{TILE} \
                 (§16.38 item 1(b))",
                filled.dimensions()
            )));
        }
        write_back_owned(
            &mut inpainted,
            &filled,
            window,
            index,
            &owners,
            &final_mask,
            original_size,
        );
    }

    // (7) — upstream `:146-171`, in upstream's order.
    let inpainting = attach_alpha(&inpainted, &final_mask);
    let mut clean_inpaint = RgbaImage::from_fn(original_size.0, original_size.1, |x, y| {
        let Rgb([r, g, b]) = *original.get_pixel(x, y);
        Rgba([r, g, b, 255])
    });
    let combined_full = compose::resize_nearest_rgba(combined_mask, original_size);
    compose::alpha_composite_over(&mut clean_inpaint, &combined_full, (0, 0));
    if let Some(noise) = noise_mask {
        compose::alpha_composite_over(&mut clean_inpaint, noise, (0, 0));
    }
    compose::alpha_composite_over(&mut clean_inpaint, &inpainting, (0, 0));
    for pixel in clean_inpaint.pixels_mut() {
        pixel.0[3] = 255;
    }

    Ok(PageOutput {
        inpainting,
        clean_inpaint,
        growths,
        tiles_inferred,
    })
}

/// spec §16.38 item 5(e): "write-back touches only owned pixels, masked by `faded_fill AND
/// isolation`", where `final_mask` **is** that conjunction.
///
/// Writing the model's RGB in hard, rather than blending here, is not a shortcut: the blend
/// happens once at the page assembly, where `final_mask` becomes the alpha and the
/// source-over against the cleaned base performs `round(base*(1-a) + model*a)`. A pixel
/// with `final_mask == 0` is never read by that composite, which is why it is skipped here.
fn write_back_owned(
    inpainted: &mut RgbImage,
    filled: &RgbImage,
    window: &TileWindow,
    index: usize,
    owners: &[u32],
    final_mask: &GrayImage,
    size: (u32, u32),
) {
    for local_y in 0..TILE {
        for local_x in 0..TILE {
            let page_x = window.window.x1 + local_x as i32;
            let page_y = window.window.y1 + local_y as i32;
            if page_x < 0 || page_y < 0 || page_x >= size.0 as i32 || page_y >= size.1 as i32 {
                continue;
            }
            let (page_x, page_y) = (page_x as u32, page_y as u32);
            if owners[(page_y as usize) * (size.0 as usize) + (page_x as usize)] != index as u32 {
                continue;
            }
            if final_mask.get_pixel(page_x, page_y).0[0] == 0 {
                continue;
            }
            inpainted.put_pixel(page_x, page_y, *filled.get_pixel(local_x, local_y));
        }
    }
}

/// `rgb` with `alpha` as its alpha channel — upstream's `putalpha` at `inpainting.py:168`.
/// Panics on a dimension mismatch (both come from the same page).
pub fn attach_alpha(rgb: &RgbImage, alpha: &GrayImage) -> RgbaImage {
    assert_eq!(
        rgb.dimensions(),
        alpha.dimensions(),
        "alpha attach needs matching sizes: {:?} vs {:?}",
        rgb.dimensions(),
        alpha.dimensions()
    );
    RgbaImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let Rgb([r, g, b]) = *rgb.get_pixel(x, y);
        Rgba([r, g, b, alpha.get_pixel(x, y).0[0]])
    })
}

/// A `side`×`side` crop of `image` at `window`'s top-left, sampling **edge-replicated**
/// outside the canvas.
///
/// spec §16.38 item 5(d): "Pages smaller than 512 on either axis are edge-replicated up to
/// 512 and the result cropped back. Edge replication rather than a constant fill is the
/// in-family choice: upstream's own `grow_mask` pads with `mode="edge"`
/// (`image_ops.py:812`)." Replicating here, at sample time, is the same thing as
/// materialising a padded page and cropping it, without the copy.
pub fn crop_replicate_rgb(image: &RgbImage, window: pc_core::Rect, side: u32) -> RgbImage {
    let (width, height) = image.dimensions();
    assert!(width > 0 && height > 0, "cannot replicate an empty page");
    RgbImage::from_fn(side, side, |x, y| {
        let source_x = (window.x1 + x as i32).clamp(0, width as i32 - 1) as u32;
        let source_y = (window.y1 + y as i32).clamp(0, height as i32 - 1) as u32;
        *image.get_pixel(source_x, source_y)
    })
}

/// [`crop_replicate_rgb`] for a [`BinaryMask`] — the model's second input (§16.38 item
/// 1(b)), with the same edge replication for the same reason.
pub fn crop_replicate_mask(mask: &BinaryMask, window: pc_core::Rect, side: u32) -> BinaryMask {
    let (width, height) = mask.dimensions();
    assert!(width > 0 && height > 0, "cannot replicate an empty mask");
    BinaryMask::from_fn(side, side, |x, y| {
        let source_x = (window.x1 + x as i32).clamp(0, width as i32 - 1) as u32;
        let source_y = (window.y1 + y as i32).clamp(0, height as i32 - 1) as u32;
        mask.get(source_x, source_y)
    })
}
