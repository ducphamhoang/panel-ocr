//! Growth kernels and binary dilation for the noise mask (spec §11.3 step 4,
//! §16.10 item 2).
//!
//! DEVIATION-FREE COPY: this is a verbatim restatement of `pc_mask::grow::{kernel,
//! dilate}`, which is itself a restatement of OpenCV `getStructuringElement`. §11.3
//! step 4 says to use "the **same** `kernel()` from `pc-imageops`", but per §16.9 item 2
//! that function actually lives in `pc-mask`, and §1 rule 2 forbids one stage crate
//! depending on another. §16.10 item 2 resolves this in favour of rule 2 and requires a
//! frozen test that pins the full 11x11 cell matrix, so the two copies cannot drift.
//! v1.5 consolidation: hoist into `pc_imageops::morph` and re-export from both stages.

use pc_imageops::BinaryMask;

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
/// * `diameter <= 5` (i.e. `thickness <= 2`): a full square of 1s with the four corners
///   zeroed -- **except** for `diameter == 1`, where the "corners" are the centre
///   itself, so the kernel is the single centre pixel and dilation is the identity
///   (§16.9 item 5; §11.3 step 4's "`size == 0` -> identity").
/// * otherwise: OpenCV's `MORPH_ELLIPSE`:
///   `dx = round(c * sqrt((r*r - dy*dy) / (r*r)))`, row `i` set on
///   `[max(c-dx,0), min(c+dx+1, diameter))`.
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
/// whole kernel footprint centred on it, writes outside the canvas being dropped.
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
