//! `panel-ocr inpaint` — turn a user-painted RGBA mask into the `MaskRegionStats` +
//! raw-mask pair `pc_inpaint::inpaint_page` needs, with no cache/uuid state.

use crate::args::InpaintArgs;
use crate::{inpainter, paths, setup};
use anyhow::{anyhow, bail, Context, Result};
use image::{GrayImage, Luma, RgbaImage};
use pc_core::{MaskRegionStats, Rect};
use pc_detect::annotate_merge::connected_components;
use pc_pipeline::EXIT_OK;

/// One region per disjoint painted blob (8-connectivity, reusing the detector's own
/// connected-components pass rather than writing a second one), each marked `failed: true`
/// so `pc_inpaint::select_regions` treats it as unconditionally eligible with
/// `FillSource::RawMask` — the ONLY fill source `failed` regions use (see
/// `crates/pc-inpaint/src/fill.rs::padded_region`), which is why the returned `GrayImage`
/// (not the RGBA mask itself) is what actually drives what gets inpainted.
///
/// `std_deviation: 0.0` is deliberate, not a placeholder: `FillSource::RawMask`'s padded
/// region never reads it as the *source crop* (only `growth()`'s dilation amount, which
/// `0.0` still yields non-zero for).
///
/// Empty when `mask` has no painted pixel (alpha 0 everywhere) — the caller decides whether
/// that is an error.
pub fn regions_from_brush_mask(mask: &RgbaImage) -> (Vec<MaskRegionStats>, GrayImage) {
    let (width, height) = mask.dimensions();
    let raw_mask = GrayImage::from_fn(width, height, |x, y| {
        Luma([if mask.get_pixel(x, y).0[3] > 0 {
            255
        } else {
            0
        }])
    });

    let components = connected_components(&raw_mask);
    let regions = components
        .stats
        .iter()
        .skip(1) // label 0 is background, not a component (see the type's own doc)
        .map(|stats| MaskRegionStats {
            rect: Rect::new(
                stats.x as i32,
                stats.y as i32,
                (stats.x + stats.width) as i32,
                (stats.y + stats.height) as i32,
            ),
            std_deviation: 0.0,
            failed: true,
            thickness: None,
        })
        .collect();

    (regions, raw_mask)
}

/// `panel-ocr inpaint`'s contract, mirroring `run_clean`/`run_ocr`'s style in `lib.rs`:
/// reads `IMAGE`/`--mask`, validates that the mask matches the image's pixel dimensions,
/// derives the regions from the mask's alpha, builds the inpainter unconditionally (this
/// command never consults `[inpainter].inpainting_enabled`), and writes `--output`.
///
/// Every error — including a per-tile inference failure, which is a per-image failure at
/// exit 2 in the batch pipeline — becomes exit 1 here, because this command has no batch
/// to isolate a failure within.
pub fn run(args: InpaintArgs) -> Result<i32> {
    let image = image::open(&args.image)
        .with_context(|| format!("reading IMAGE {}", args.image.display()))?
        .into_rgb8();
    let mask_image = image::open(&args.mask)
        .with_context(|| format!("reading --mask {}", args.mask.display()))?
        .into_rgba8();

    if mask_image.dimensions() != image.dimensions() {
        bail!(
            "--mask is {:?} but IMAGE is {:?}; they must be the same pixel dimensions",
            mask_image.dimensions(),
            image.dimensions()
        );
    }

    let (regions, raw_mask) = regions_from_brush_mask(&mask_image);
    if regions.is_empty() {
        bail!("--mask has no painted pixels (alpha is 0 everywhere); nothing to inpaint");
    }

    let config = setup::load_app_config()?;
    let profile = setup::load_profile(
        args.profile.as_deref(),
        args.profile_path.as_deref(),
        &config,
    )?;
    let cache_root = paths::resolve_cache_root(args.cache_dir.as_deref(), &config);

    // `enabled: true` unconditionally — this command's whole purpose is to inpaint, so it
    // does not consult `[inpainter].inpainting_enabled` the way `clean` does.
    let provider = inpainter::build_provider(
        true,
        args.model_path.as_deref(),
        &cache_root,
        profile.general.device,
    )
    .expect("build_provider(true, ..) always returns Some");
    let inpainter = provider.inpainter().map_err(|error| anyhow!("{error}"))?;

    // A fully transparent no-op layer, deliberately NOT the user's mask pixels: `inpaint_page`
    // composites `combined_mask` as a page-wide layer UNDER the final inpainting result
    // (`crates/pc-inpaint/src/lib.rs`'s page assembly). Since every region here is `failed`
    // and `failed` regions are filled from `raw_mask` alone (`fill.rs::padded_region`),
    // `combined_mask`'s content plays no role in *what* gets inpainted — but if it carried
    // the brush's own opaque paint, it would visibly leak through as a flat-color ring
    // wherever `final_mask`'s alpha is below 255 — which the fade radius produces in a thin
    // band just inside the silhouette's own edge, not outside it. Transparent means it
    // composites as a true no-op.
    let combined_mask = RgbaImage::new(image.width(), image.height());

    let output = pc_inpaint::inpaint_page(
        pc_inpaint::PageInput {
            original: &image,
            raw_mask: &raw_mask,
            combined_mask: &combined_mask,
            noise_mask: None,
            regions: &regions,
            min_mask_thickness: profile.masker.min_mask_thickness,
            config: &profile.inpainter,
        },
        inpainter.as_ref(),
    )?;

    output
        .clean_inpaint
        .save(&args.output)
        .with_context(|| format!("writing --output {}", args.output.display()))?;

    Ok(EXIT_OK)
}
