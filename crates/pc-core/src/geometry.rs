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
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// `x2 - x1`
    pub fn width(&self) -> i32 {
        self.x2 - self.x1
    }

    /// `y2 - y1`
    pub fn height(&self) -> i32 {
        self.y2 - self.y1
    }

    /// `width * height` widened to i64 (upstream Python ints are unbounded).
    pub fn area(&self) -> i64 {
        i64::from(self.width()) * i64::from(self.height())
    }

    /// `((x1+x2)/2, (y1+y2)/2)` with Python floor-division semantics:
    /// use `(x1 + x2).div_euclid(2)`, NOT `/ 2` (they differ for negative sums).
    pub fn center(&self) -> (i32, i32) {
        (
            (self.x1 + self.x2).div_euclid(2),
            (self.y1 + self.y2).div_euclid(2),
        )
    }

    /// Point containment, **inclusive on both ends**: `x1 <= x <= x2 && y1 <= y <= y2`.
    pub fn contains(&self, p: (i32, i32)) -> bool {
        self.x1 <= p.0 && p.0 <= self.x2 && self.y1 <= p.1 && p.1 <= self.y2
    }

    /// Rect containment, inclusive on all four sides (`self.x1<=o.x1 && self.y1<=o.y1
    /// && self.x2>=o.x2 && self.y2>=o.y2`); `self.contains_rect(&self)` is true. Added
    /// beyond the original method list because §2.5's "every `reference` superset-of
    /// `masking`" invariant needs it and `contains()` only takes a point.
    pub fn contains_rect(&self, other: &Rect) -> bool {
        self.x1 <= other.x1 && self.y1 <= other.y1 && self.x2 >= other.x2 && self.y2 >= other.y2
    }

    /// Bounding union.
    pub fn merge(&self, other: &Rect) -> Rect {
        Rect::new(
            self.x1.min(other.x1),
            self.y1.min(other.y1),
            self.x2.max(other.x2),
            self.y2.max(other.y2),
        )
    }

    /// spec §2.1: `inter / min(area, other.area)` — with the divisor forced to 1 when
    /// that minimum is 0 — compared **strictly greater** against `threshold_percent / 100.0`.
    pub fn overlaps(&self, other: &Rect, threshold_percent: f64) -> bool {
        let x_overlap = 0.max(self.x2.min(other.x2) - self.x1.max(other.x1));
        let y_overlap = 0.max(self.y2.min(other.y2) - self.y1.max(other.y1));
        let intersection = i64::from(x_overlap) * i64::from(y_overlap);
        let smaller = self.area().min(other.area());
        let divisor = if smaller == 0 { 1 } else { smaller };

        intersection as f64 / divisor as f64 > threshold_percent / 100.0
    }

    /// spec §2.1: `other.contains(self.center()) || self.contains(other.center())`.
    pub fn overlaps_center(&self, other: &Rect) -> bool {
        other.contains(self.center()) || self.contains(other.center())
    }

    /// spec §2.1: `max(x1-a,0)`, `max(y1-a,0)`, `min(x2+a, canvas.0)`, `min(y2+a, canvas.1)`.
    pub fn pad(&self, amount: i32, canvas: (u32, u32)) -> Rect {
        Rect::new(
            (self.x1 - amount).max(0),
            (self.y1 - amount).max(0),
            (self.x2 + amount).min(canvas.0 as i32),
            (self.y2 + amount).min(canvas.1 as i32),
        )
    }

    /// spec §2.1: only `x2' = min(x2 + a, canvas.0)`.
    pub fn right_pad(&self, amount: i32, canvas: (u32, u32)) -> Rect {
        Rect::new(
            self.x1,
            self.y1,
            (self.x2 + amount).min(canvas.0 as i32),
            self.y2,
        )
    }

    /// spec §2.1: per-coordinate `(c as f64 * factor) as i32` — truncation toward zero
    /// (Python int()), **not** rounding.
    pub fn scale(&self, factor: f64) -> Rect {
        Rect::new(
            (f64::from(self.x1) * factor) as i32,
            (f64::from(self.y1) * factor) as i32,
            (f64::from(self.x2) * factor) as i32,
            (f64::from(self.y2) * factor) as i32,
        )
    }

    pub fn translate(&self, dx: i32, dy: i32) -> Rect {
        Rect::new(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)
    }

    /// `width <= 0 || height <= 0`
    pub fn is_empty(&self) -> bool {
        self.width() <= 0 || self.height() <= 0
    }

    /// Clamped `(x, y, w, h)` for the `image` crate's crop API. `x1/y1` clamp up to 0;
    /// `x2/y2` (treated as exclusive, per §2.1's cropping convention) clamp down to the
    /// canvas. Returns `None` when the clamped width or height is `<= 0` (fully
    /// degenerate or fully out-of-bounds).
    pub fn to_crop(&self, canvas: (u32, u32)) -> Option<(u32, u32, u32, u32)> {
        let x1 = i64::from(self.x1).clamp(0, i64::from(canvas.0));
        let y1 = i64::from(self.y1).clamp(0, i64::from(canvas.1));
        let x2 = i64::from(self.x2).clamp(0, i64::from(canvas.0));
        let y2 = i64::from(self.y2).clamp(0, i64::from(canvas.1));

        if x2 <= x1 || y2 <= y1 {
            return None;
        }

        Some((x1 as u32, y1 as u32, (x2 - x1) as u32, (y2 - y1) as u32))
    }
}
