//! `panel-ocr residual-check` — run the existing text detector over an already-*cleaned*
//! (post-inpaint) image and report whether it still finds text, as an objective replacement
//! for subjective "does this look clean" visual judgment.
//!
//! Non-gating by design: the *presence* of residual detections is diagnostic information,
//! never a process failure. Only read failures, dimension mismatches, and detector-init
//! failures are errors. Stateless — no cache/uuid state, mirroring `standalone_inpaint.rs`.

use crate::args::{ResidualCheckArgs, ResidualFormat};
use crate::{detector, paths, setup};
use anyhow::{bail, Context, Result};
use image::RgbaImage;
use pc_core::Rect;
use pc_detect::yolo::CLASS_SCORE_THRESHOLD;
use pc_detect::RawBlock;
use pc_pipeline::EXIT_OK;

/// Does `block.rect` overlap any pixel of `mask` whose alpha is > 0?
///
/// A pixel/rect overlap test only — no connected components needed here. `x2`/`y2` are
/// treated as exclusive here, matching mask rasterization's convention (spec line 1543,
/// "box rasterization is EXCLUSIVE on `x2`/`y2` everywhere") — NOT `Rect::contains`'s own
/// inclusive convention, which is a deliberately non-uniform rule (`pc_core::Rect`'s doc
/// comment: exclusive for cropping/rasterising, inclusive for `contains()`; do not "fix"
/// this file to match `contains()` instead).
pub fn block_overlaps_mask(rect: Rect, mask: &RgbaImage) -> bool {
    let x1 = rect.x1.clamp(0, mask.width() as i32);
    let y1 = rect.y1.clamp(0, mask.height() as i32);
    let x2 = rect.x2.clamp(0, mask.width() as i32);
    let y2 = rect.y2.clamp(0, mask.height() as i32);

    for y in y1..y2 {
        for x in x1..x2 {
            if mask.get_pixel(x as u32, y as u32).0[3] > 0 {
                return true;
            }
        }
    }
    false
}

/// The confident blocks, optionally restricted to those overlapping a masked region.
///
/// `threshold` is the minimum confidence to count as "residual text found"; with no mask,
/// every confident block counts.
pub fn residual_blocks(
    blocks: &[RawBlock],
    threshold: f32,
    mask: Option<&RgbaImage>,
) -> Vec<RawBlock> {
    blocks
        .iter()
        .copied()
        // Strictly greater, matching `pc_detect::yolo`'s own postprocessing boundary (a block
        // at exactly the threshold was already discarded there) rather than disagreeing with
        // it at a threshold a caller might pass explicitly.
        .filter(|block| block.confidence > threshold)
        .filter(|block| {
            mask.map(|mask| block_overlaps_mask(block.rect, mask))
                .unwrap_or(true)
        })
        .collect()
}

fn render_text(
    image: &std::path::Path,
    threshold: f32,
    total_blocks: usize,
    residuals: &[RawBlock],
) -> String {
    let mut lines = String::new();
    lines.push_str(&format!(
        "residual-check {}: {} total block(s) found, {} residual block(s) at confidence > {threshold}\n",
        image.display(),
        total_blocks,
        residuals.len()
    ));
    for block in residuals {
        lines.push_str(&format!(
            "  rect=({}, {}, {}, {}) confidence={} class={}\n",
            block.rect.x1,
            block.rect.y1,
            block.rect.x2,
            block.rect.y2,
            block.confidence,
            block.class_index
        ));
    }
    lines
}

fn render_json(
    image: &std::path::Path,
    threshold: f32,
    total_blocks: usize,
    residuals: &[RawBlock],
) -> Result<String> {
    let blocks = residuals
        .iter()
        .map(|block| {
            serde_json::json!({
                "rect": [block.rect.x1, block.rect.y1, block.rect.x2, block.rect.y2],
                "confidence": block.confidence,
                "class_index": block.class_index,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string_pretty(&serde_json::json!({
        "image": image.display().to_string(),
        "threshold": threshold,
        "total_blocks": total_blocks,
        "residual_count": residuals.len(),
        "blocks": blocks,
    }))
    .context("serializing residual report")
}

fn emit(report: String, output: Option<&std::path::Path>) -> Result<()> {
    match output {
        Some(path) => std::fs::write(path, report)
            .with_context(|| format!("writing --output {}", path.display())),
        None => {
            print!("{report}");
            Ok(())
        }
    }
}

/// `panel-ocr residual-check`'s contract: load the already-cleaned image, build the
/// detector, run it, filter confident blocks (optionally to those overlapping the masked
/// region), and report. Returns `EXIT_OK` regardless of how many residual blocks were
/// found — this tool is diagnostic, not a gate.
pub fn run(args: ResidualCheckArgs) -> Result<i32> {
    let image = image::open(&args.image)
        .with_context(|| format!("reading IMAGE {}", args.image.display()))?
        .into_rgb8();

    let mask = match &args.mask {
        Some(path) => {
            let mask = image::open(path)
                .with_context(|| format!("reading --mask {}", path.display()))?
                .into_rgba8();
            if mask.dimensions() != image.dimensions() {
                bail!(
                    "--mask is {:?} but IMAGE is {:?}; they must be the same pixel dimensions",
                    mask.dimensions(),
                    image.dimensions()
                );
            }
            Some(mask)
        }
        None => None,
    };

    let threshold = args.threshold.unwrap_or(CLASS_SCORE_THRESHOLD);

    let config = setup::load_app_config()?;
    let profile = setup::load_profile(
        args.profile.as_deref(),
        args.profile_path.as_deref(),
        &config,
    )?;
    let cache_root = paths::resolve_cache_root(args.cache_dir.as_deref(), &config);
    let provider = detector::build_provider(
        &args.detector,
        args.model_path.as_deref(),
        profile.text_detector.model_path(),
        &cache_root,
        &profile.text_detector,
        profile.general.device,
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    let detector = provider
        .detector_for(&args.image)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let detection = detector
        .detect(&image)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let residuals = residual_blocks(&detection.blocks, threshold, mask.as_ref());
    let total_blocks = detection.blocks.len();

    let report = match args.format {
        ResidualFormat::Text => render_text(&args.image, threshold, total_blocks, &residuals),
        ResidualFormat::Json => render_json(&args.image, threshold, total_blocks, &residuals)?,
    };
    emit(report, args.output.as_deref())?;

    Ok(EXIT_OK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn mask_with_rect(w: u32, h: u32, rect: Rect) -> RgbaImage {
        let mut mask = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
        for y in rect.y1..rect.y2 {
            for x in rect.x1..rect.x2 {
                mask.put_pixel(x as u32, y as u32, Rgba([255, 255, 255, 255]));
            }
        }
        mask
    }

    #[test]
    fn a_block_overlapping_a_masked_pixel_is_flagged() {
        let mask = mask_with_rect(100, 100, Rect::new(10, 10, 30, 30));
        let block = Rect::new(20, 20, 40, 40);
        assert!(block_overlaps_mask(block, &mask));
    }

    #[test]
    fn a_block_without_overlap_is_excluded() {
        let mask = mask_with_rect(100, 100, Rect::new(10, 10, 30, 30));
        let block = Rect::new(50, 50, 70, 70);
        assert!(!block_overlaps_mask(block, &mask));
    }

    #[test]
    fn residual_blocks_applies_threshold_and_mask_filtering() {
        let mask = mask_with_rect(100, 100, Rect::new(0, 0, 50, 50));
        let blocks = vec![
            RawBlock {
                rect: Rect::new(10, 10, 20, 20),
                class_index: 0,
                confidence: 0.8,
            },
            RawBlock {
                rect: Rect::new(60, 60, 70, 70),
                class_index: 0,
                confidence: 0.9,
            },
            RawBlock {
                rect: Rect::new(10, 10, 20, 20),
                class_index: 0,
                confidence: 0.1,
            },
        ];
        let residuals = residual_blocks(&blocks, 0.5, Some(&mask));
        assert_eq!(residuals.len(), 1);
        assert_eq!(residuals[0].confidence, 0.8);
    }

    #[test]
    fn no_mask_means_every_confident_block_counts() {
        let blocks = vec![
            RawBlock {
                rect: Rect::new(0, 0, 10, 10),
                class_index: 0,
                confidence: 0.8,
            },
            RawBlock {
                rect: Rect::new(20, 20, 30, 30),
                class_index: 0,
                confidence: 0.2,
            },
        ];
        let residuals = residual_blocks(&blocks, 0.5, None);
        assert_eq!(residuals.len(), 1);
        assert_eq!(residuals[0].confidence, 0.8);
    }
}
