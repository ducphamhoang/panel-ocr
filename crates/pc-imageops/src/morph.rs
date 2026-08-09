//! Task L1 -- the single shared growth kernel + binary dilation (spec §16.38 item 16(a),
//! §10.3 step 6, §11.3 step 4, §16.9 items 5, 6).
//!
//! This module is the **one** implementation. It arrived here by hoisting two verbatim
//! copies -- `pc_mask::grow::{Kernel, kernel, dilate}` and
//! `pc_denoise::morph::{Kernel, kernel, dilate}` -- which §16.10 item 2 had deliberately
//! duplicated because §1 rule 2 forbids a stage crate depending on another stage crate,
//! and which the same item scheduled for a "v1.5 consolidation ticket". §16.38 item 16(a)
//! calls that consolidation in, as a **prerequisite** rather than cleanup: `pc-inpaint`
//! needs the same primitives and a third copy is not acceptable. Both stage crates now
//! re-export from here.
//!
//! Both hoisted copies were confirmed byte-identical in their bodies before the move, and
//! `crates/pc-pipeline/tests/l1_morph_equivalence.rs` holds the frozen proof: it pins both
//! crates' public paths against a hand-derived OpenCV `MORPH_ELLIPSE` oracle for
//! thicknesses 0..=7 and against each other, and was green on the pre-hoist code.
//!
//! §16.38 item 16(a) requires §16.9 item 2's "no `pc-config` dependency" property of this
//! crate to survive the hoist, so nothing here takes a config type: callers pass plain
//! `u32` thicknesses. The `MaskerConfig`-shaped helpers (`growth_padding`,
//! `growth_candidates`, `build_candidates`) stay in `pc-mask`, which is where the *policy*
//! lives.
//!
//! Upstream grows the precise mask with `scipy.signal.convolve2d(mask, kernel) > 0`. For a
//! non-negative kernel that is **exactly** binary dilation, so the reformulation here is
//! an exact -- and vastly faster -- restatement, not an approximation (§10.3 step 6
//! requires this equivalence to be stated at the implementation site).

use crate::BinaryMask;

/// A square, symmetric structuring element of odd `diameter = thickness * 2 + 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    radius: u32,
    diameter: u32,
    cells: Vec<u8>,
}

impl Kernel {
    pub fn radius(&self) -> u32 {
        self.radius
    }

    pub fn diameter(&self) -> u32 {
        self.diameter
    }

    /// `(column, row)` indexing, both in `0..diameter`.
    pub fn get(&self, j: u32, i: u32) -> bool {
        if j >= self.diameter || i >= self.diameter {
            return false;
        }
        self.cells[(i as usize) * (self.diameter as usize) + (j as usize)] == 1
    }

    pub fn count(&self) -> usize {
        self.cells.iter().filter(|cell| **cell == 1).count()
    }

    /// Row-major cells, each `0` or `1` -- for whole-matrix assertions.
    pub fn as_cells(&self) -> &[u8] {
        &self.cells
    }
}

/// spec §10.3 step 6 / §11.3 step 4.
///
/// * `diameter <= 5` (i.e. `thickness <= 2`): a full square of 1s with the four
///   corners zeroed -- **except** for `diameter == 1`, where the "corners" are the
///   centre itself, so the kernel is the single centre pixel and dilation is the
///   identity (§16.9 item 5; §11.3 step 4's "`size == 0` -> identity").
/// * otherwise: OpenCV's `MORPH_ELLIPSE`, reproduced from `getStructuringElement`'s own
///   code path: `dx = round(c * sqrt((r*r - dy*dy) / (r*r)))`, row `i` set on
///   `[max(c-dx,0), min(c+dx+1, diameter))`. `saturate_cast<int>` rounds to nearest;
///   `f64::round` (half away from zero) matches -- exact ties do not occur for these
///   square-root values.
pub fn kernel(thickness: u32) -> Kernel {
    let radius = thickness;
    let diameter = thickness * 2 + 1;
    let size = (diameter as usize) * (diameter as usize);
    let mut cells = vec![0_u8; size];

    if diameter <= 5 {
        cells.fill(1);
        if diameter >= 3 {
            let last = diameter - 1;
            for (j, i) in [(0, 0), (last, 0), (0, last), (last, last)] {
                cells[(i as usize) * (diameter as usize) + (j as usize)] = 0;
            }
        }
        return Kernel {
            radius,
            diameter,
            cells,
        };
    }

    let r = i64::from(radius);
    let c = i64::from(radius);
    for i in 0..i64::from(diameter) {
        let dy = i - r;
        let inverse_r2 = 1.0 / ((r * r) as f64);
        let dx = ((c as f64) * (((r * r - dy * dy) as f64) * inverse_r2).sqrt()).round() as i64;
        let j1 = (c - dx).max(0);
        let j2 = (c + dx + 1).min(i64::from(diameter));
        for j in j1..j2 {
            cells[(i as usize) * (diameter as usize) + (j as usize)] = 1;
        }
    }

    Kernel {
        radius,
        diameter,
        cells,
    }
}

/// Binary dilation, stamp formulation (§16.9 item 6): every set input pixel stamps the
/// whole kernel footprint centred on it, writes outside the canvas being dropped (zero
/// border). Kernels here are symmetric, so this equals the reflect-then-max definition.
pub fn dilate(mask: &BinaryMask, kernel: &Kernel) -> BinaryMask {
    let (width, height) = mask.dimensions();
    let mut out = BinaryMask::new(width, height);
    let radius = i64::from(kernel.radius());
    for y in 0..height {
        for x in 0..width {
            if !mask.get(x, y) {
                continue;
            }
            for i in 0..kernel.diameter() {
                for j in 0..kernel.diameter() {
                    if !kernel.get(j, i) {
                        continue;
                    }
                    let target_x = i64::from(x) + i64::from(j) - radius;
                    let target_y = i64::from(y) + i64::from(i) - radius;
                    if target_x < 0
                        || target_y < 0
                        || target_x >= i64::from(width)
                        || target_y >= i64::from(height)
                    {
                        continue;
                    }
                    out.set(target_x as u32, target_y as u32, true);
                }
            }
        }
    }
    out
}
