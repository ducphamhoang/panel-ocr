//! spec §16.38 items 3(d), 3(f) and 10 — the per-region padded fill mask, the two
//! page-global masks, and the NEAREST resize to the original size.
//!
//! **The frame choice, made deliberately because item 3(f) requires it to be.** Item 3(f)
//! warns that upstream builds each region's padded mask in a **canvas-sized** buffer whose
//! content sits at `(box.x1 - box_padded.x1, box.y1 - box_padded.y1)` (`inpainting.py:114-116`)
//! and only then pastes it back at the padded box's origin (`:124`) — so "a port that reads
//! (e) and (f) as operating in one absolute frame would place every fill region at the wrong
//! coordinates", and, because `grow_mask` pads with `mode="edge"` (`image_ops.py:812`),
//! "growth near a canvas boundary replicates whatever sits at that boundary — and the
//! boundary differs between upstream's padded-box-relative canvas and an absolute page
//! canvas".
//!
//! This module builds each region's mask in the **padded box's own frame** — neither
//! upstream's canvas-sized buffer nor an absolute page canvas — and the choice is safe for a
//! reason that is checkable rather than asserted:
//!
//!   * *Sizing.* [`crate::growth::padded_box`] pads by `growth + inpainting_isolation_radius`,
//!     so a crop grown by `growth` and then again by `inpainting_isolation_radius` fits the
//!     padded frame exactly. Where the padded box was clamped to the canvas, the overflow is
//!     clipped — and it is clipped at the same absolute coordinate the page canvas would
//!     clip it at, because the clamp *is* the canvas edge.
//!   * *Borders.* For the ellipse structuring elements of [`pc_imageops::morph::kernel`],
//!     edge-replicated dilation and zero-bordered dilation agree on every **in-frame** output
//!     pixel. Sketch: a replicated read at an out-of-bounds `q` returns the value at the
//!     clamped position `q'`, and for these row-convex symmetric kernels `q'` is itself
//!     reachable from the same output pixel by a shorter in-kernel offset, so the OR over the
//!     footprint is unchanged. That is a claim about a family of kernels, not a proof about
//!     this code, which is why `growth_at_the_frame_edge_matches_the_replicate_padded_oracle`
//!     pins it against an independent replicate-padded oracle built in upstream's own frame.
//!
//! Whoever changes the kernel family owes that test a second look.

use crate::eligible::{EligibleRegion, FillSource};
use crate::growth::{growth, padded_box};
use image::RgbaImage;
use pc_config::InpainterConfig;
use pc_core::Rect;
use pc_imageops::morph::{dilate, kernel};
use pc_imageops::BinaryMask;

/// One region's contribution, in the padded box's own frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaddedRegion {
    /// Index into `MaskData::regions`.
    pub index: usize,
    /// The region's own box, in the mask frame.
    pub rect: Rect,
    /// `rect` padded by `growth + inpainting_isolation_radius`, clamped to the canvas.
    pub padded: Rect,
    /// The `growth` of item 3(e); upstream's `analytics_thicknesses` entry.
    pub growth: u32,
    /// `padded`-sized: the box crop, grown per item 3(d)/3(e). `padded.x1/y1` is its origin.
    pub fill: BinaryMask,
}

/// The two page-global masks of item 3(f), in whatever frame they were composed in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageMasks {
    /// Upstream's `combined_mask` (`:122-124`) — what the model is told to fill.
    pub fill: BinaryMask,
    /// Upstream's `isolated_combined_mask` (`:137-144`) — each padded mask grown *again* by
    /// `inpainting_isolation_radius`. What cuts the inpainted area out at composite time.
    pub isolation: BinaryMask,
}

/// spec §16.38 item 3(d) — binarising the combined fill mask.
///
/// DEVIATION(28): **the coverage signal comes from ALPHA, not from luma.** Ratified by
/// §16.38 item 21 (Fable tie-break) and registered at §14 item 28. Upstream does
/// `mask_image.convert("1")` (`inpainting.py:63`) on an RGBA image — RGBA because upstream
/// builds it through `combine_best_masks` / `convert_mask_to_rgba` — and PIL routes
/// `convert("1")` through `L`, the luma of the RGB channels, with alpha **discarded**. For a
/// black-filled mask (the `black_bubble` demo page — a demo fixture, not a profile — fill
/// `(0,0,0,255)`) the luma is `0`, the
/// binary mask comes out empty, and every poorly-fitted region silently gets nothing
/// inpainted.
///
/// **This is its OWN register entry, NOT `DEVIATION(5)` propagating** — an earlier draft of
/// this comment said the latter and §16.38 item 21 ruled against it. Two facts separate them.
/// (i) A different upstream call: item 5 is `denoiser.py:62`'s path into `grow_mask` and its
/// `mask.convert("L")`; this is `inpainting.py:63`'s `convert("1")`. (ii) `convert("1")` also
/// **dithers** — PIL's documented default for mode `"1"` is Floyd-Steinberg error diffusion —
/// so on intermediate luma values upstream produces a stippled checkerboard, not a clean
/// bilevel cut. Item 5's text describes luma-versus-alpha and nothing else. The dithering is
/// also why "replicate upstream verbatim, bug included" is not an option here: it would mean
/// porting error diffusion, not changing a threshold.
///
/// What item 5 does supply is the GROUND, and it is reused verbatim rather than widened:
/// *"A mask's 'is this pixel covered' signal must not depend on the brightness of its fill
/// color"* (§14 item 5, decided in §15.6, implemented there as `alpha > 0` in
/// `crates/pc-denoise/src/noise_mask.rs`'s `alpha_binary`).
///
/// The companion raw-mask binarisation deliberately does **not** match this one; see the note
/// at the pair of call sites in [`crate::inpaint_page`] and §16.38 item 21(d).
pub fn combined_fill_binary(combined: &RgbaImage) -> BinaryMask {
    BinaryMask::from_fn(combined.width(), combined.height(), |x, y| {
        combined.get_pixel(x, y).0[3] > 0
    })
}

/// spec §16.38 items 3(d) and 3(e) for one region.
///
/// A **failed** region samples `raw`, cropped to the box and grown by `min_mask_thickness`
/// (upstream `:93-97`); a **poorly-fitted** one samples `combined`, cropped to the box and
/// not pre-grown (`:99-101`). The `min_mask_thickness` growth happens in the **box-sized**
/// frame, before the paste, so it is clipped at the box edge — upstream-faithful: `:94-96`
/// grows the box-sized crop, not the padded buffer.
pub fn padded_region(
    region: &EligibleRegion,
    raw: &BinaryMask,
    combined: &BinaryMask,
    min_mask_thickness: u32,
    config: &InpainterConfig,
    canvas: (u32, u32),
) -> PaddedRegion {
    let growth = growth(region.std_deviation, config);
    let padded = padded_box(region.rect, growth, config, canvas);

    let box_w = region.rect.width().max(0) as u32;
    let box_h = region.rect.height().max(0) as u32;
    let pad_w = padded.width().max(0) as u32;
    let pad_h = padded.height().max(0) as u32;

    let source = match region.source {
        FillSource::RawMask => raw,
        FillSource::CombinedMask => combined,
    };
    let mut crop = source.crop_into(region.rect, (box_w, box_h), (0, 0));
    if region.source == FillSource::RawMask {
        crop = dilate(&crop, &kernel(min_mask_thickness));
    }

    let offset = (region.rect.x1 - padded.x1, region.rect.y1 - padded.y1);
    let placed = crop.crop_into(
        Rect::new(0, 0, box_w as i32, box_h as i32),
        (pad_w, pad_h),
        offset,
    );
    let fill = dilate(&placed, &kernel(growth));

    PaddedRegion {
        index: region.index,
        rect: region.rect,
        padded,
        growth,
        fill,
    }
}

/// spec §16.38 item 3(f) — one page-global fill mask and one page-global isolation mask.
///
/// Both are ORs of the per-region padded masks pasted at `padded.x1/y1`; the isolation mask's
/// contribution is that same mask dilated by `inpainting_isolation_radius` first (`:139`).
/// Upstream's `paste(mask, (x1, y1), mask)` uses the mask as its own paste stencil, so the
/// paste writes only where the mask is set — that is an OR, and it is why regions may overlap
/// without erasing each other.
pub fn compose_page_masks(
    regions: &[PaddedRegion],
    canvas: (u32, u32),
    isolation_radius: u32,
) -> PageMasks {
    let mut fill = BinaryMask::new(canvas.0, canvas.1);
    let mut isolation = BinaryMask::new(canvas.0, canvas.1);
    let isolation_kernel = kernel(isolation_radius);

    for region in regions {
        let at = (region.padded.x1, region.padded.y1);
        paste_or(&mut fill, &region.fill, at);
        paste_or(&mut isolation, &dilate(&region.fill, &isolation_kernel), at);
    }

    PageMasks { fill, isolation }
}

/// OR `src` into `dst` with `src`'s top-left at `at`; writes outside `dst` are dropped.
pub fn paste_or(dst: &mut BinaryMask, src: &BinaryMask, at: (i32, i32)) {
    let (width, height) = dst.dimensions();
    for y in 0..src.height() {
        for x in 0..src.width() {
            if !src.get(x, y) {
                continue;
            }
            let target_x = i64::from(at.0) + i64::from(x);
            let target_y = i64::from(at.1) + i64::from(y);
            if target_x < 0
                || target_y < 0
                || target_x >= i64::from(width)
                || target_y >= i64::from(height)
            {
                continue;
            }
            dst.set(target_x as u32, target_y as u32, true);
        }
    }
}

/// spec §16.38 item 3(f) / upstream `:127-128`, `:141-144`: NEAREST resize to the original
/// size when the page was scaled.
///
/// The formula is the one §16.9 item 13 and §16.10 item 13 already pin for the RGBA masks —
/// `src = floor(dst * src_len / dst_len)` — restated here for [`BinaryMask`], which neither
/// of those two sites handles. Identity when the sizes already match, so the unscaled page
/// pays nothing.
pub fn resize_nearest_binary(mask: &BinaryMask, size: (u32, u32)) -> BinaryMask {
    if mask.dimensions() == size {
        return mask.clone();
    }
    let (source_w, source_h) = mask.dimensions();
    if source_w == 0 || source_h == 0 {
        return BinaryMask::new(size.0, size.1);
    }
    BinaryMask::from_fn(size.0, size.1, |x, y| {
        let source_x = ((u64::from(x) * u64::from(source_w)) / u64::from(size.0)) as u32;
        let source_y = ((u64::from(y) * u64::from(source_h)) / u64::from(size.1)) as u32;
        mask.get(source_x.min(source_w - 1), source_y.min(source_h - 1))
    })
}

/// Carry a mask-frame rect into the resized frame **exactly as [`resize_nearest_binary`]
/// carries the pixels**, so that no resized fill pixel can land outside it.
///
/// Why the arithmetic is what it is, since a plain `Rect::scale` is the obvious wrong answer:
/// `resize_nearest_binary` sets destination pixel `x` from source `floor(x * from / to)`, so
/// `x` is covered by a source range `[x1, x2)` exactly when `x * from / to ∈ [x1, x2)`, i.e.
/// when `x ∈ [ceil(x1 * to / from), ceil(x2 * to / from))`. Both ends therefore take a
/// **ceiling**, not a truncation. `Rect::scale`'s truncation (§2.1, Python `int()`) would
/// shrink the interval on the left and could leave a resized fill pixel outside every merged
/// rectangle — which §16.38 item 5(e)'s "every fill pixel is written exactly once" would then
/// fail on, as an unowned pixel rather than a double-written one.
///
/// Identity when `from == to`.
pub fn scale_rect_to(rect: Rect, from: (u32, u32), to: (u32, u32)) -> Rect {
    if from == to {
        return rect;
    }
    Rect::new(
        ceil_scale(rect.x1, to.0, from.0),
        ceil_scale(rect.y1, to.1, from.1),
        ceil_scale(rect.x2, to.0, from.0),
        ceil_scale(rect.y2, to.1, from.1),
    )
}

/// `ceil(value * numerator / denominator)` in `i64`, for non-negative `denominator`.
fn ceil_scale(value: i32, numerator: u32, denominator: u32) -> i32 {
    if denominator == 0 {
        return value;
    }
    let numerator = i64::from(numerator);
    let denominator = i64::from(denominator);
    let product = i64::from(value) * numerator;
    let quotient = product.div_euclid(denominator);
    let remainder = product.rem_euclid(denominator);
    let ceiling = if remainder == 0 {
        quotient
    } else {
        quotient + 1
    };
    ceiling.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}
