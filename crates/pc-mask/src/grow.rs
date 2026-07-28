//! Task M2 -- growth kernels and the candidate sequence (spec §10.3 step 6,
//! §10.7(A)1-3, §16.9 items 5, 6).
//!
//! Upstream grows the precise mask with `scipy.signal.convolve2d(mask, kernel) > 0`.
//! For a non-negative kernel that is **exactly** binary dilation, so the reformulation
//! here is an exact -- and vastly faster -- restatement, not an approximation (§10.3
//! step 6 requires this equivalence to be stated at the implementation site).
//!
//! The one subtlety worth reading twice: candidates are produced by dilating a single
//! **replicate-padded buffer in place**, so border-replication effects accumulate from
//! one candidate to the next exactly as they do upstream (which reuses `padded_mask`
//! across iterations). Each candidate is the centre crop of that buffer.

use pc_config::MaskerConfig;
use pc_imageops::BinaryMask;

/// One entry of §10.3 step 7's candidate list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub mask: BinaryMask,
    /// `None` for the box candidate -- it is not grown from the precise mask
    /// (§10.3 step 5).
    pub thickness: Option<u32>,
}

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

/// spec §10.3 step 6.
///
/// * `diameter <= 5` (i.e. `thickness <= 2`): a full square of 1s with the four
///   corners zeroed -- **except** for `diameter == 1`, where the "corners" are the
///   centre itself, so the kernel is the single centre pixel and dilation is the
///   identity (§16.9 item 5).
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

/// `np.pad(mode="edge")`: `pad` pixels on all four sides, each copied from the nearest
/// edge pixel of `mask` (§10.3 step 6).
pub fn pad_replicate(mask: &BinaryMask, pad: u32) -> BinaryMask {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return BinaryMask::new(width + pad * 2, height + pad * 2);
    }
    BinaryMask::from_fn(width + pad * 2, height + pad * 2, |x, y| {
        let source_x = x.saturating_sub(pad).min(width - 1);
        let source_y = y.saturating_sub(pad).min(height - 1);
        mask.get(source_x, source_y)
    })
}

/// Inverse of [`pad_replicate`]'s framing: drop `pad` pixels from all four sides.
pub fn center_crop(mask: &BinaryMask, pad: u32) -> BinaryMask {
    let (width, height) = mask.dimensions();
    let inner_w = width.saturating_sub(pad * 2);
    let inner_h = height.saturating_sub(pad * 2);
    BinaryMask::from_fn(inner_w, inner_h, |x, y| mask.get(x + pad, y + pad))
}

/// `max(min_mask_thickness, mask_growth_step_pixels) * 2` (§10.3 step 6).
pub fn growth_padding(config: &MaskerConfig) -> u32 {
    config
        .min_mask_thickness
        .max(config.mask_growth_step_pixels)
        * 2
}

/// spec §10.3 step 6: exactly `mask_growth_steps` candidates, thicknesses
/// `min_mask_thickness + i * mask_growth_step_pixels` for `i` in
/// `0..mask_growth_steps`, each the centre crop of one shared padded buffer that is
/// dilated in place (candidate 0 with `kernel(min_mask_thickness)`, the rest with
/// `kernel(mask_growth_step_pixels)`).
pub fn growth_candidates(precise: &BinaryMask, config: &MaskerConfig) -> Vec<Candidate> {
    let pad = growth_padding(config);
    let first = kernel(config.min_mask_thickness);
    let step = kernel(config.mask_growth_step_pixels);

    let mut padded = pad_replicate(precise, pad);
    let mut candidates = Vec::with_capacity(config.mask_growth_steps as usize);
    for i in 0..config.mask_growth_steps {
        let element = if i == 0 { &first } else { &step };
        padded = dilate(&padded, element);
        candidates.push(Candidate {
            mask: center_crop(&padded, pad),
            thickness: Some(config.min_mask_thickness + i * config.mask_growth_step_pixels),
        });
    }
    candidates
}

/// spec §10.3 step 7 -- candidate ordering, which "determines everything downstream":
/// box candidate **first** in `mask_selection_fast` mode, **last** otherwise.
pub fn build_candidates(
    precise: &BinaryMask,
    box_candidate: BinaryMask,
    config: &MaskerConfig,
) -> Vec<Candidate> {
    let growth = growth_candidates(precise, config);
    let box_candidate = Candidate {
        mask: box_candidate,
        thickness: None,
    };

    if config.mask_selection_fast {
        let mut candidates = Vec::with_capacity(growth.len() + 1);
        candidates.push(box_candidate);
        candidates.extend(growth);
        candidates
    } else {
        let mut candidates = growth;
        candidates.push(box_candidate);
        candidates
    }
}
