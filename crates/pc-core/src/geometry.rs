//! spec §2.1 — upstream `Box` (`structures.py:25-145`) ported as `Rect`.
//!
//! Coordinate convention (§2.1, deliberately mixed, matching upstream):
//!   `x1,y1` inclusive; `x2,y2` **exclusive** for cropping/rasterising, but
//!   **inclusive** for `contains()`. Do not "fix" this — box merging depends on it.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rect {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl Rect {
    pub fn new(_x1: i32, _y1: i32, _x2: i32, _y2: i32) -> Self {
        todo!()
    }

    /// `x2 - x1`
    pub fn width(&self) -> i32 {
        todo!()
    }

    /// `y2 - y1`
    pub fn height(&self) -> i32 {
        todo!()
    }

    /// `width * height` widened to i64 (upstream Python ints are unbounded).
    pub fn area(&self) -> i64 {
        todo!()
    }

    /// `((x1+x2)/2, (y1+y2)/2)` with Python floor-division semantics:
    /// use `(x1 + x2).div_euclid(2)`, NOT `/ 2` (they differ for negative sums).
    pub fn center(&self) -> (i32, i32) {
        todo!()
    }

    /// Point containment, **inclusive on both ends**: `x1 <= x <= x2 && y1 <= y <= y2`.
    pub fn contains(&self, _p: (i32, i32)) -> bool {
        todo!()
    }

    /// Rect containment, inclusive on all four sides (`self.x1<=o.x1 && self.y1<=o.y1
    /// && self.x2>=o.x2 && self.y2>=o.y2`); `self.contains_rect(&self)` is true. Added
    /// beyond the original method list because §2.5's "every `reference` superset-of
    /// `masking`" invariant needs it and `contains()` only takes a point.
    pub fn contains_rect(&self, _other: &Rect) -> bool {
        todo!()
    }

    /// Bounding union.
    pub fn merge(&self, _other: &Rect) -> Rect {
        todo!()
    }

    /// spec §2.1: `inter / min(area, other.area)` — with the divisor forced to 1 when
    /// that minimum is 0 — compared **strictly greater** against `threshold_percent / 100.0`.
    pub fn overlaps(&self, _other: &Rect, _threshold_percent: f64) -> bool {
        todo!()
    }

    /// spec §2.1: `other.contains(self.center()) || self.contains(other.center())`.
    pub fn overlaps_center(&self, _other: &Rect) -> bool {
        todo!()
    }

    /// spec §2.1: `max(x1-a,0)`, `max(y1-a,0)`, `min(x2+a, canvas.0)`, `min(y2+a, canvas.1)`.
    pub fn pad(&self, _amount: i32, _canvas: (u32, u32)) -> Rect {
        todo!()
    }

    /// spec §2.1: only `x2' = min(x2 + a, canvas.0)`.
    pub fn right_pad(&self, _amount: i32, _canvas: (u32, u32)) -> Rect {
        todo!()
    }

    /// spec §2.1: per-coordinate `(c as f64 * factor) as i32` — truncation toward zero
    /// (Python int()), **not** rounding.
    pub fn scale(&self, _factor: f64) -> Rect {
        todo!()
    }

    pub fn translate(&self, _dx: i32, _dy: i32) -> Rect {
        todo!()
    }

    /// `width <= 0 || height <= 0`
    pub fn is_empty(&self) -> bool {
        todo!()
    }

    /// Clamped `(x, y, w, h)` for the `image` crate's crop API. `x1/y1` clamp up to 0;
    /// `x2/y2` (treated as exclusive, per §2.1's cropping convention) clamp down to the
    /// canvas. Returns `None` when the clamped width or height is `<= 0` (fully
    /// degenerate or fully out-of-bounds).
    pub fn to_crop(&self, _canvas: (u32, u32)) -> Option<(u32, u32, u32, u32)> {
        todo!()
    }
}
