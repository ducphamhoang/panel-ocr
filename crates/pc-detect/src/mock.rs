//! spec §7.2 / §7.2.1 -- the mocked ML boundary. Test-only (`cfg(any(test, feature =
//! "testkit"))`), so per `pc-testkit`'s panic policy fixture problems panic with a
//! diagnostic rather than returning `Result`.
//!
//! This is test infrastructure, not part of the shipped stage: the frozen D3/D7 tests are
//! written against it, and it is compiled out of a normal build.

use crate::detector::{RawBlock, RawDetection, TextDetector};
use image::{GrayImage, Luma, RgbImage};
use pc_core::StageError;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// `<stem>_detector_mask.png` (§7.2.1).
pub fn detector_mask_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}_detector_mask.png"))
}

/// `<stem>_detector_blocks.json` (§7.2.1).
pub fn detector_blocks_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}_detector_blocks.json"))
}

/// Write the §7.2.1 replay pair. Used by the D3/D7 tests and by `cargo xtask
/// record-fixtures` (task F1), so both write byte-identical layouts.
pub fn write_replay_fixture(dir: &Path, stem: &str, mask: &GrayImage, blocks: &[RawBlock]) {
    std::fs::create_dir_all(dir)
        .unwrap_or_else(|error| panic!("cannot create `{}`: {error}", dir.display()));

    let mask_path = detector_mask_path(dir, stem);
    mask.save(&mask_path)
        .unwrap_or_else(|error| panic!("cannot write `{}`: {error}", mask_path.display()));

    let blocks_path = detector_blocks_path(dir, stem);
    let json = serde_json::to_string_pretty(blocks).expect("RawBlock list is serializable");
    std::fs::write(&blocks_path, json)
        .unwrap_or_else(|error| panic!("cannot write `{}`: {error}", blocks_path.display()));
}

/// A programmable detector: fixed blocks plus a synthetic mask (§7.2).
#[derive(Debug, Default)]
pub struct MockDetector {
    blocks: Vec<RawBlock>,
    mask: Option<GrayImage>,
    mask_fill: u8,
    block_fill: Option<u8>,
    failing: bool,
    calls: AtomicUsize,
}

impl MockDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_blocks(mut self, blocks: Vec<RawBlock>) -> Self {
        self.blocks = blocks;
        self
    }

    /// Paint `value` inside every block rect of the generated mask (exclusive `x2`/`y2`,
    /// spec §16.6 item 5).
    pub fn with_block_fill(mut self, value: u8) -> Self {
        self.block_fill = Some(value);
        self
    }

    /// Background value of the generated mask.
    pub fn with_mask_fill(mut self, value: u8) -> Self {
        self.mask_fill = value;
        self
    }

    /// Return this exact mask, ignoring the input image size and both fills.
    pub fn with_mask(mut self, mask: GrayImage) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Make `detect` return `StageError::Inference`.
    pub fn failing(mut self) -> Self {
        self.failing = true;
        self
    }

    /// Number of `detect` calls so far. Interior mutability because `detect` takes
    /// `&self` (the trait is shared across rayon threads, §4.5).
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl TextDetector for MockDetector {
    fn detect(&self, image: &RgbImage) -> Result<RawDetection, StageError> {
        self.calls.fetch_add(1, Ordering::SeqCst);

        if self.failing {
            return Err(StageError::Inference("mock detector failure".into()));
        }

        let mask = match &self.mask {
            Some(mask) => mask.clone(),
            None => {
                let (width, height) = image.dimensions();
                let mut mask = GrayImage::from_pixel(width, height, Luma([self.mask_fill]));
                if let Some(fill) = self.block_fill {
                    for block in &self.blocks {
                        if let Some((x, y, w, h)) = block.rect.to_crop((width, height)) {
                            for py in y..y + h {
                                for px in x..x + w {
                                    mask.put_pixel(px, py, Luma([fill]));
                                }
                            }
                        }
                    }
                }
                mask
            }
        };

        Ok(RawDetection {
            blocks: self.blocks.clone(),
            mask,
        })
    }
}

/// spec §7.2.1: bound to **one fixture stem at construction**, because the
/// `TextDetector` boundary is stem-less (image in, detection out). `detect()` ignores
/// the image and replays the recorded pair.
#[derive(Debug)]
pub struct ReplayDetector {
    dir: PathBuf,
    stem: String,
    mask: GrayImage,
    blocks: Vec<RawBlock>,
    calls: AtomicUsize,
}

impl ReplayDetector {
    /// Reads `<stem>_detector_mask.png` and `<stem>_detector_blocks.json` from `dir`
    /// eagerly, so a missing fixture fails at construction with a clear message rather
    /// than mid-pipeline.
    pub fn new(dir: &Path, stem: &str) -> Self {
        let mask_path = detector_mask_path(dir, stem);
        let mask = image::open(&mask_path)
            .unwrap_or_else(|error| {
                panic!(
                    "missing or undecodable replay mask `{}`: {error}; run `cargo xtask record-fixtures`",
                    mask_path.display()
                )
            })
            .to_luma8();

        let blocks_path = detector_blocks_path(dir, stem);
        let bytes = std::fs::read(&blocks_path).unwrap_or_else(|error| {
            panic!(
                "missing replay blocks `{}`: {error}; run `cargo xtask record-fixtures`",
                blocks_path.display()
            )
        });
        let blocks: Vec<RawBlock> = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            panic!(
                "malformed replay blocks `{}`: {error}",
                blocks_path.display()
            )
        });

        Self {
            dir: dir.to_path_buf(),
            stem: stem.to_string(),
            mask,
            blocks,
            calls: AtomicUsize::new(0),
        }
    }

    pub fn mask_path(&self) -> PathBuf {
        detector_mask_path(&self.dir, &self.stem)
    }

    pub fn blocks_path(&self) -> PathBuf {
        detector_blocks_path(&self.dir, &self.stem)
    }

    pub fn stem(&self) -> &str {
        &self.stem
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl TextDetector for ReplayDetector {
    fn detect(&self, _image: &RgbImage) -> Result<RawDetection, StageError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(RawDetection {
            blocks: self.blocks.clone(),
            mask: self.mask.clone(),
        })
    }
}
