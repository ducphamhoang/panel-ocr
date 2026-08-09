//! The `testkit`-gated test double for [`crate::Inpainter`] (spec §16.38 item 16(c): "All of it
//! behind an `Inpainter` trait whose test double returns a fixed tile, so **every test in L4 runs
//! in the default no-`onnx` tier**").
//!
//! This is a stub in the strict sense: it produces *recognisable* pixels, never plausible ones.
//! Its whole job is to make the geometry and the compositing falsifiable — which window wrote a
//! pixel, whether the kept region was blended back, whether a tile was 512×512 — none of which a
//! realistic inpainter would let a test see.

use crate::{Inpainter, TILE};
use image::{Rgb, RgbImage};
use pc_core::StageError;
use pc_imageops::BinaryMask;
use std::sync::Mutex;

/// What the stub returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StubMode {
    /// Every call returns a tile flat-filled with `color`.
    Flat(Rgb<u8>),
    /// The `n`-th call (0-based) returns a tile flat-filled with `Rgb([base + n*step; 3])`, so the
    /// composed page says **which** window wrote each pixel. A test that only checks "some pixel
    /// changed" cannot tell an ownership bug from a correct write; this mode can.
    PerCall { base: u8, step: u8 },
    /// The tile, shifted by `(dx, dy)` with edge clamping — the "copy from a fixed offset" stub,
    /// useful when a test needs the output to depend on the *input* pixels.
    CopyOffset { dx: i32, dy: i32 },
    /// Return a `side`×`side` tile instead of a `TILE`×`TILE` one — the shape §16.38 item 1(c) warns
    /// about, where the graph's *declared* output shape is symbolic and unusable and only the runtime
    /// shape can be trusted.
    WrongSize { side: u32 },
    /// Fail with [`StageError::Inference`] — the per-image half of §16.38 item 9(g)'s split.
    Fail,
}

/// One recorded call, so a test can assert what the model was actually shown.
#[derive(Debug, Clone)]
pub struct SeenTile {
    pub tile: RgbImage,
    pub mask: BinaryMask,
}

pub struct StubInpainter {
    mode: StubMode,
    seen: Mutex<Vec<SeenTile>>,
}

impl StubInpainter {
    pub fn new(mode: StubMode) -> Self {
        Self {
            mode,
            seen: Mutex::new(Vec::new()),
        }
    }

    /// A flat mid-grey filler, distinguishable from both black and white page content.
    pub fn flat(r: u8, g: u8, b: u8) -> Self {
        Self::new(StubMode::Flat(Rgb([r, g, b])))
    }

    pub fn calls(&self) -> usize {
        self.seen.lock().expect("stub mutex").len()
    }

    pub fn seen(&self) -> Vec<SeenTile> {
        self.seen.lock().expect("stub mutex").clone()
    }

    /// The colour [`StubMode::PerCall`] returns on call `n` — exposed so a test states the expected
    /// colour by *calling this*, and does not silently re-derive it from the composed output.
    pub fn per_call_color(base: u8, step: u8, n: usize) -> Rgb<u8> {
        let value = base.wrapping_add(step.wrapping_mul(n as u8));
        Rgb([value, value, value])
    }
}

impl Inpainter for StubInpainter {
    fn inpaint_tile(&self, tile: &RgbImage, mask: &BinaryMask) -> Result<RgbImage, StageError> {
        assert_eq!(
            tile.dimensions(),
            (TILE, TILE),
            "the trait contract is a {TILE}x{TILE} tile (§16.38 item 1(b))"
        );
        assert_eq!(
            mask.dimensions(),
            (TILE, TILE),
            "the trait contract is a {TILE}x{TILE} mask (§16.38 item 1(b))"
        );
        let call_index = {
            let mut seen = self.seen.lock().expect("stub mutex");
            seen.push(SeenTile {
                tile: tile.clone(),
                mask: mask.clone(),
            });
            seen.len() - 1
        };

        match self.mode {
            StubMode::Flat(color) => Ok(RgbImage::from_pixel(TILE, TILE, color)),
            StubMode::PerCall { base, step } => Ok(RgbImage::from_pixel(
                TILE,
                TILE,
                Self::per_call_color(base, step, call_index),
            )),
            StubMode::CopyOffset { dx, dy } => Ok(RgbImage::from_fn(TILE, TILE, |x, y| {
                let source_x = (x as i32 + dx).clamp(0, TILE as i32 - 1) as u32;
                let source_y = (y as i32 + dy).clamp(0, TILE as i32 - 1) as u32;
                *tile.get_pixel(source_x, source_y)
            })),
            StubMode::WrongSize { side } => Ok(RgbImage::from_pixel(side, side, Rgb([0, 0, 0]))),
            StubMode::Fail => Err(StageError::Inference(
                "stub inpainter refused (§16.38 item 9(g): per-tile failures are per-image)".into(),
            )),
        }
    }
}
