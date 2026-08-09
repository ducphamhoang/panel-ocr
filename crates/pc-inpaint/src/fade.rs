//! spec §16.38 item 3(h) — the Gaussian fade and the isolation cut.
//!
//! Upstream, `pcleaner/inpainting.py:159-167`:
//!
//! ```text
//! if i_conf.inpainting_fade_radius:
//!     mask_faded = ops.fade_mask_edges(combined_mask, i_conf.inpainting_fade_radius)
//! else:
//!     mask_faded = combined_mask.convert("L")
//! final_mask = Image.new("L", original_image.size, 0)
//! final_mask.paste(mask_faded, (0, 0), isolated_combined_mask)
//! ```
//!
//! and `fade_mask_edges` is `mask.convert("L").filter(ImageFilter.GaussianBlur(fade_radius))`
//! (`image_ops.py:820-830`).

use image::{GrayImage, Luma};
use pc_imageops::{gaussian, BinaryMask};

/// The faded fill mask: `L`-mode, `0` or `255` before the blur, blurred by
/// `inpainting_fade_radius`.
///
/// `radius == 0` is upstream's `else` branch — a plain `convert("L")` — and this returns
/// exactly that, because [`pc_imageops::gaussian::blur`] is the identity at radius 0. So there
/// is no branch here and no second code path to keep in step; upstream's `if` and its `else`
/// are the same expression once the blur is defined that way.
///
/// **`DEVIATION(6)` reaches a second consumer here** (§14 item 6). PIL's
/// `GaussianBlur(radius=r)` is a three-pass box-blur approximation; this project uses a true
/// separable Gaussian with `sigma = radius` truncated at `3*sigma`, pinned by §16.10 item 12.
/// The register entry was written for the *noise* fade (`noise_fade_radius = 1`, where **§11.3
/// step 5** — not §14 item 6, which is one sentence on the box-versus-true-Gaussian choice and
/// carries no such phrase — states the difference as "a few levels on a soft edge"); at the
/// inpainting default
/// `inpainting_fade_radius = 4` the kernel is wider and the difference is correspondingly
/// larger, and **nothing here measures it**. This gets no new register number — it is an
/// existing deviation reaching a new reader, the shape §16.38 item 10 uses for `DEVIATION(12)` —
/// but the un-measured half is stated rather than inherited silently.
pub fn fade_fill_mask(fill: &BinaryMask, fade_radius: u32) -> GrayImage {
    gaussian::blur(&fill.to_gray(), fade_radius)
}

/// `final_mask`: `faded` where `isolation` is set, `0` elsewhere — upstream's
/// `final_mask.paste(mask_faded, (0, 0), isolated_combined_mask)` onto a zeroed `L` canvas.
///
/// This is what makes the fade one-sided in effect: the blur spreads the fill mask outward past
/// its own edge, and the isolation mask — the fill grown by `inpainting_isolation_radius`, which
/// is `>= ` the fade's reach at the defaults — decides how much of that spread survives. Panics
/// on a dimension mismatch; both come from the same page.
pub fn cut_by_isolation(faded: &GrayImage, isolation: &BinaryMask) -> GrayImage {
    assert_eq!(
        (faded.width(), faded.height()),
        isolation.dimensions(),
        "the fade and the isolation mask must share the page: {:?} vs {:?}",
        faded.dimensions(),
        isolation.dimensions()
    );
    GrayImage::from_fn(faded.width(), faded.height(), |x, y| {
        if isolation.get(x, y) {
            *faded.get_pixel(x, y)
        } else {
            Luma([0])
        }
    })
}
