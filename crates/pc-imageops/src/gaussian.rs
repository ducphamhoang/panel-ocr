//! Task N2 -- separable Gaussian blur (spec §11.3 step 5, §14.6, §16.10 item 12).
//!
//! **Placement.** Written for `pc-denoise` (task N2) and hoisted here by task L4, body
//! unchanged, ratified by §16.38 item 20. §16.10 item 1 placed it in `pc-denoise` on a
//! ground it stated as conditional -- paraphrasing, not quoting: `pc-imageops` had been
//! frozen and committed at the end of Stage 3, and no other v1 stage used either module,
//! so re-opening the crate bought nothing. That item then anticipated this move, and
//! *this* sentence is quoted verbatim from §16.10 item 1:
//!
//! > Both stay `pub` (`pc_denoise::nlm`, `pc_denoise::gaussian`) and free of any
//! > `pc-config` dependency, so §11.1's "independently benchmarkable" property is
//! > preserved and a v1.5 hoist into `pc-imageops` is a mechanical move plus a re-export.
//!
//! v1.5's `pc-inpaint` needs the same blur for `inpainting_fade_radius` (§16.38 item 3(h),
//! porting `image_ops.py:820-830`'s `fade_mask_edges`), and §1 rule 2 forbids the
//! stage-to-stage edge that would let it borrow `pc-denoise`'s copy. `pc_denoise::gaussian`
//! is now a re-export, so `crates/pc-denoise/tests/n2_gaussian.rs` is untouched.
//!
//! **Scope, stated because widening it would be wrong:** §16.38 item 20 ratifies the move
//! for `gaussian` ALONE. `nlm` stays in `pc-denoise` -- it still has exactly one consumer --
//! and §16.10 item 1 remains the live placement decision for it. §16.38 item 16(a)'s
//! separate hoist names `kernel` and `dilate` only, and is not authority for this one.
//!
//! **DEVIATION(6) travels with the code and now has a second consumer.** `pc-inpaint`'s
//! fade inherits it, since upstream's `fade_mask_edges` is the same PIL
//! `GaussianBlur(radius)` call. That gets no new register number -- it is DEVIATION(6)
//! propagating into a new consumer, the shape §16.38 item 10 uses for DEVIATION(12) -- but
//! it is stated at the new call site too.
//!
//! DEVIATION(6): PIL's `GaussianBlur(radius=r)` is a **three-pass box-blur
//! approximation**, not a true Gaussian. v1 uses a true separable Gaussian with
//! `sigma = radius`, truncated at `3*sigma`. With the default `noise_fade_radius = 1`
//! the difference is a few levels on a soft edge, well inside the stage tolerance.
//!
//! Pinned by §16.10 item 12: taps `exp(-j^2 / (2*sigma^2))` for `j in -k..=k` with
//! `k = ceil(3*sigma)`, normalised to sum 1 in `f64`; horizontal pass into an `f32`
//! intermediate with **no** intermediate rounding, then the vertical pass, then
//! `clamp(floor(v + 0.5), 0, 255)`. Borders **replicate** (clamp to edge), matching
//! PIL -- deliberately NOT the reflect-101 used inside `nlm` (§16.10 item 12).

use image::GrayImage;

/// The normalised, truncated 1-D Gaussian taps for `radius`, in `f64`.
///
/// `radius == 0` yields the single tap `[1.0]` (the identity kernel), which is what
/// makes `blur(_, 0)` a clone rather than a special case buried in the caller.
pub fn taps(radius: u32) -> Vec<f64> {
    if radius == 0 {
        return vec![1.0];
    }
    let sigma = f64::from(radius);
    let half_width = (3.0 * sigma).ceil() as i64;
    let two_sigma_squared = 2.0 * sigma * sigma;
    let mut weights: Vec<f64> = (-half_width..=half_width)
        .map(|j| (-((j * j) as f64) / two_sigma_squared).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

/// spec §11.3 step 5: a true separable Gaussian with `sigma = radius`, truncated at
/// `3*sigma`, replicate borders. `radius == 0` is the identity.
pub fn blur(image: &GrayImage, radius: u32) -> GrayImage {
    if radius == 0 || image.width() == 0 || image.height() == 0 {
        return image.clone();
    }
    let weights = taps(radius);
    let half_width = (weights.len() / 2) as i64;
    let (width, height) = image.dimensions();
    let (w, h) = (width as i64, height as i64);

    // Horizontal pass -> f32 intermediate, no rounding (§16.10 item 12).
    let mut horizontal = vec![0.0_f32; (width as usize) * (height as usize)];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f64;
            for (index, weight) in weights.iter().enumerate() {
                let source_x = (x + index as i64 - half_width).clamp(0, w - 1);
                sum += weight * f64::from(image.get_pixel(source_x as u32, y as u32).0[0]);
            }
            horizontal[(y as usize) * (width as usize) + (x as usize)] = sum as f32;
        }
    }

    // Vertical pass -> rounded u8.
    GrayImage::from_fn(width, height, |x, y| {
        let mut sum = 0.0_f64;
        for (index, weight) in weights.iter().enumerate() {
            let source_y = (i64::from(y) + index as i64 - half_width).clamp(0, h - 1);
            sum += weight
                * f64::from(horizontal[(source_y as usize) * (width as usize) + (x as usize)]);
        }
        image::Luma([(sum + 0.5).floor().clamp(0.0, 255.0) as u8])
    })
}
