//! Task M1 -- `BinaryMask` and box-mask rasterisation (spec §10.2, §10.3 steps 0-2,
//! §16.9 items 1-4).
//!
//! `BinaryMask` is one byte per pixel, not bit-packed: dilation (§10.3 step 6) is
//! memory-bandwidth bound anyway, and the byte layout keeps every operation here a
//! two-line loop that a reader can check against the spec.
//!
//! Coordinate convention: `x2`/`y2` are **exclusive** everywhere, matching
//! `pc_core::Rect::to_crop` and §16.6 item 5 / §16.9 item 1. §10.3 step 1's "inclusive"
//! wording is superseded -- rasterising inclusively would make the masker cover a
//! different region than the `mask_coverage` filter that produced the boxes.

use image::{GrayImage, Luma};
use pc_core::Rect;

/// PIL's `L -> 1` conversion without dither thresholds at `value > 127` (spec §10.3
/// step 0). Our refined mask is strictly 0 or 255, so `> 0` and `> 127` agree; `127` is
/// specified for upstream parity and for hand-authored fixtures.
pub const PIL_BINARY_THRESHOLD: u8 = 127;

/// A `w x h` binary image, one byte per pixel, each byte exactly `0` or `1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryMask {
    w: u32,
    h: u32,
    bits: Vec<u8>,
}

impl BinaryMask {
    /// An all-zero mask.
    pub fn new(w: u32, h: u32) -> Self {
        let len = usize::try_from(u64::from(w) * u64::from(h)).expect("mask fits in memory");
        Self {
            w,
            h,
            bits: vec![0; len],
        }
    }

    pub fn from_fn(w: u32, h: u32, mut f: impl FnMut(u32, u32) -> bool) -> Self {
        let mut mask = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                if f(x, y) {
                    mask.set(x, y, true);
                }
            }
        }
        mask
    }

    pub fn width(&self) -> u32 {
        self.w
    }

    pub fn height(&self) -> u32 {
        self.h
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    /// `false` for any out-of-bounds coordinate (so neighbour scans need no bounds
    /// special-casing).
    pub fn get(&self, x: u32, y: u32) -> bool {
        if x >= self.w || y >= self.h {
            return false;
        }
        self.bits[self.index(x, y)] == 1
    }

    /// Panics on an out-of-bounds coordinate -- a write outside the canvas is a
    /// programming error, unlike a read (which is defined as `false`).
    pub fn set(&mut self, x: u32, y: u32, value: bool) {
        assert!(
            x < self.w && y < self.h,
            "pixel ({x}, {y}) is out of bounds for a {}x{} mask",
            self.w,
            self.h
        );
        let index = self.index(x, y);
        self.bits[index] = u8::from(value);
    }

    /// Raw bytes, row-major, each `0` or `1`.
    pub fn as_bits(&self) -> &[u8] {
        &self.bits
    }

    pub fn count_set(&self) -> usize {
        self.bits.iter().filter(|bit| **bit == 1).count()
    }

    pub fn is_blank(&self) -> bool {
        self.bits.iter().all(|bit| *bit == 0)
    }

    /// Bounding box of the set pixels in the exclusive `x2`/`y2` convention (§16.9
    /// item 3); `None` when the mask is blank.
    pub fn bbox(&self) -> Option<Rect> {
        let mut found = false;
        let (mut x1, mut y1, mut x2, mut y2) = (u32::MAX, u32::MAX, 0_u32, 0_u32);
        for y in 0..self.h {
            for x in 0..self.w {
                if self.get(x, y) {
                    found = true;
                    x1 = x1.min(x);
                    y1 = y1.min(y);
                    x2 = x2.max(x + 1);
                    y2 = y2.max(y + 1);
                }
            }
        }
        found.then(|| Rect::new(x1 as i32, y1 as i32, x2 as i32, y2 as i32))
    }

    /// Pixel-wise AND. Panics on a dimension mismatch (§16.9 item 3).
    pub fn and(&self, other: &BinaryMask) -> BinaryMask {
        self.assert_same_dimensions(other);
        BinaryMask {
            w: self.w,
            h: self.h,
            bits: self
                .bits
                .iter()
                .zip(&other.bits)
                .map(|(a, b)| a & b)
                .collect(),
        }
    }

    /// Pixel-wise OR. Panics on a dimension mismatch (§16.9 item 3).
    pub fn or(&self, other: &BinaryMask) -> BinaryMask {
        self.assert_same_dimensions(other);
        BinaryMask {
            w: self.w,
            h: self.h,
            bits: self
                .bits
                .iter()
                .zip(&other.bits)
                .map(|(a, b)| a | b)
                .collect(),
        }
    }

    /// spec §10.3 step 3 / §16.9 item 3: copy the `rect` window of `self` into a fresh
    /// zeroed `target`-sized mask, placing `rect`'s top-left at `offset`. Source pixel
    /// `(sx, sy)` lands at `(offset.0 + sx - rect.x1, offset.1 + sy - rect.y1)`; the
    /// source read is clamped to this mask (via `Rect::to_crop`) and destinations
    /// outside `target` are dropped, so a partially out-of-canvas `rect` still lands in
    /// the right place.
    pub fn crop_into(&self, rect: Rect, target: (u32, u32), offset: (i32, i32)) -> BinaryMask {
        let mut out = BinaryMask::new(target.0, target.1);
        let Some((x, y, width, height)) = rect.to_crop(self.dimensions()) else {
            return out;
        };
        for source_y in y..y + height {
            for source_x in x..x + width {
                if !self.get(source_x, source_y) {
                    continue;
                }
                let destination_x = offset.0 as i64 + i64::from(source_x) - i64::from(rect.x1);
                let destination_y = offset.1 as i64 + i64::from(source_y) - i64::from(rect.y1);
                if destination_x < 0
                    || destination_y < 0
                    || destination_x >= i64::from(target.0)
                    || destination_y >= i64::from(target.1)
                {
                    continue;
                }
                out.set(destination_x as u32, destination_y as u32, true);
            }
        }
        out
    }

    /// `1 -> 255`, `0 -> 0`.
    pub fn to_gray(&self) -> GrayImage {
        GrayImage::from_fn(self.w, self.h, |x, y| {
            Luma([if self.get(x, y) { 255 } else { 0 }])
        })
    }

    /// `value > threshold` -- **strict** (§16.9 item 4). Use [`PIL_BINARY_THRESHOLD`]
    /// for the §10.3 step 0 precise mask.
    pub fn from_gray_threshold(image: &GrayImage, threshold: u8) -> BinaryMask {
        BinaryMask::from_fn(image.width(), image.height(), |x, y| {
            image.get_pixel(x, y).0[0] > threshold
        })
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.w as usize) + (x as usize)
    }

    fn assert_same_dimensions(&self, other: &BinaryMask) {
        assert_eq!(
            self.dimensions(),
            other.dimensions(),
            "binary mask dimension mismatch: {:?} vs {:?}",
            self.dimensions(),
            other.dimensions()
        );
    }
}

/// spec §10.3 step 1 -- the "box mask": the union of `rects` rasterised into a
/// `size`-shaped mask, with `x2`/`y2` **exclusive** (§16.9 item 1). Rects are clamped to
/// the canvas; empty or fully out-of-bounds rects contribute nothing.
pub fn rasterize_boxes(rects: &[Rect], size: (u32, u32)) -> BinaryMask {
    let mut mask = BinaryMask::new(size.0, size.1);
    for rect in rects {
        let Some((x, y, width, height)) = rect.to_crop(size) else {
            continue;
        };
        for pixel_y in y..y + height {
            for pixel_x in x..x + width {
                mask.set(pixel_x, pixel_y, true);
            }
        }
    }
    mask
}
