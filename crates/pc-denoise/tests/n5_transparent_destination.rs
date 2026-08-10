//! §16.45 items 1, 3 and 9 (H2) — the transparent-destination compositing defect, gated
//! at **`pc_denoise::run`'s own output**, not at the `alpha_composite_over` primitive.
//!
//! Why this file is separate from the primitive-level locks in
//! `crates/pc-export/tests/composite_value_lock.rs`: §16.45 item 1 root-causes the visible
//! artifact to a *two-step* composite — `build_noise_mask` composites each region's faded
//! layer onto a **fully transparent** canvas (`noise_mask.rs`'s `blank_noise_mask`,
//! `(0,0,0,0)`), and `run` then composites that layer onto the real, opaque page. Only the
//! first step is wrong; the second is correct source-over onto an opaque canvas and is what
//! *reveals* the first as a visible darkening. A gate on the primitive alone would not show
//! that this stage reaches the broken regime at all, which is `docs/COOKBOOK.md`'s "gate the
//! risk, not a copy of it".
//!
//! The oracle is independent of the code under test: on a **uniform** page whose mask fill
//! equals the paper colour, the denoised cutout is the same uniform colour as the page, so
//! source-over of that layer at *any* alpha must reproduce the page exactly —
//! `base·(1−a) + base·a = base`. No expected value here is read back from a run. §16.45
//! item 1 states the defect's shape for exactly this case: `out = base·(1 − a + a²)`,
//! bottoming out at `0.75·base` at `a = 0.5`.

mod common;

use common::{combined_mask, fast_config, memory_input, region, shared_nlm, PAGE_SIZE, REGION};
use image::{DynamicImage, Rgb, RgbImage};
use pc_core::MaskRegionStats;
use pc_denoise::run;

/// Deliberately non-grey and per-channel distinct, for two reasons: §16.10 item 17 then
/// keeps the output `ImageRgb8` instead of collapsing it to `L` (so the assertion sees
/// three channels), and a darkening bug shows a *different* absolute error on each
/// channel, so a single-channel coincidence cannot hide it.
const PAPER: [u8; 3] = [236, 232, 224];

/// Minimum number of partially-transparent (`0 < a < 255`) pixels the fixture's noise mask
/// must contain, hard-coded and derived from the fixture's geometry rather than from any
/// run: [`REGION`] is 30x30, dilation by `kernel(noise_outline_size = 5)` grows it to
/// roughly 40x40, so the faded perimeter is on the order of 160 px long, and the
/// `sigma = noise_fade_radius = 1` Gaussian spreads it over at least one pixel on each
/// side of that perimeter. 200 is a deliberately loose floor under that estimate; its job
/// is only to make the "denoises back to itself" assertion above impossible to satisfy
/// vacuously by a fixture with no faded rim at all (in which case every alpha would be
/// 0 or 255 and both the old and the new formula would agree).
const MINIMUM_FADED_RIM_PIXELS: usize = 200;

fn selected_region() -> Vec<MaskRegionStats> {
    // std_deviation 5.0 is comfortably above the default cutoff (0.25), and `failed` is
    // false, so §11.3 step 3 selects it and the denoise path actually runs.
    vec![region(REGION, 5.0, false)]
}

/// §16.45 items 1 and 3: with `denoising_enabled` true and a region past the sigma cutoff,
/// a page that is one uniform colour must survive the denoise stage **bit-for-bit**.
///
/// What turns this red: `alpha_composite_over` ignoring the destination's alpha. Under that
/// formula every rim pixel of the noise mask is written as `round(0·(1−a) + PAPER·a)` — the
/// paper colour premultiplied once against the transparent canvas's black RGB — while its
/// stored alpha stays `a`; `run`'s second composite then produces `PAPER·(1 − a + a²)`, a
/// visible darkening ring bottoming out at `0.75·PAPER` (177/174/168 here) at `a = 128`.
#[test]
fn a_uniform_page_denoises_back_to_itself_exactly() {
    let _nlm = shared_nlm();
    let page = RgbImage::from_pixel(PAGE_SIZE.0, PAGE_SIZE.1, Rgb(PAPER));
    // The mask fill is the paper colour, so stage 3's composite leaves the page uniform:
    // the only thing the denoise stage can change is the compositing arithmetic itself.
    let mask = combined_mask(PAGE_SIZE, &[(REGION, [PAPER[0], PAPER[1], PAPER[2], 255])]);
    let input = memory_input(
        DynamicImage::ImageRgb8(page),
        mask,
        selected_region(),
        fast_config(),
    );

    let output = run(input).expect("denoise succeeds on a synthetic uniform page");

    let noise_mask = output
        .noise_mask
        .load()
        .expect("the noise mask handle carries its image")
        .to_rgba8();

    // Anti-vacuity, asserted BEFORE the equality below so a fixture that stopped producing
    // a faded rim fails as a fixture problem rather than passing as a clean page.
    let faded = noise_mask
        .pixels()
        .filter(|pixel| pixel.0[3] > 0 && pixel.0[3] < 255)
        .count();
    assert!(
        faded >= MINIMUM_FADED_RIM_PIXELS,
        "fixture must actually exercise the partial-alpha composite: only {faded} \
         partially transparent noise-mask pixels, expected at least \
         {MINIMUM_FADED_RIM_PIXELS}"
    );

    let denoised = output
        .denoised
        .load()
        .expect("the denoised handle carries its image")
        .to_rgb8();
    assert_eq!(denoised.dimensions(), PAGE_SIZE);

    let mut worst: Option<(u32, u32, [u8; 3])> = None;
    for (x, y, pixel) in denoised.enumerate_pixels() {
        if pixel.0 != PAPER {
            let replace = worst.is_none_or(|(_, _, current)| {
                let error = |value: [u8; 3]| {
                    (0..3)
                        .map(|channel| {
                            i32::from(PAPER[channel]).abs_diff(i32::from(value[channel]))
                        })
                        .max()
                        .unwrap_or(0)
                };
                error(pixel.0) > error(current)
            });
            if replace {
                worst = Some((x, y, pixel.0));
            }
        }
    }
    assert_eq!(
        worst, None,
        "a uniform page must denoise to itself exactly; the worst deviating pixel is \
         listed (expected {PAPER:?})"
    );
}

/// The same fixture with the noise mask inspected directly: every rim pixel's **colour**
/// must still be the paper colour, only its alpha faded. This separates the two halves of
/// the assertion above — it is the intermediate artifact `pc-export` also consumes
/// (§16.45 item 6), so a fix that repaired the final page while still storing
/// premultiplied colour in `_noise_mask.png` would pass the test above and fail here.
///
/// What turns this red: the same destination-alpha-ignoring blend; it stores
/// `round(PAPER·a)` in the rim pixels' RGB.
#[test]
fn the_noise_masks_faded_rim_keeps_the_layer_colour_and_only_fades_its_alpha() {
    let _nlm = shared_nlm();
    let page = RgbImage::from_pixel(PAGE_SIZE.0, PAGE_SIZE.1, Rgb(PAPER));
    let mask = combined_mask(PAGE_SIZE, &[(REGION, [PAPER[0], PAPER[1], PAPER[2], 255])]);
    let input = memory_input(
        DynamicImage::ImageRgb8(page),
        mask,
        selected_region(),
        fast_config(),
    );

    let output = run(input).expect("denoise succeeds on a synthetic uniform page");
    let noise_mask = output
        .noise_mask
        .load()
        .expect("the noise mask handle carries its image")
        .to_rgba8();

    let mut faded = 0_usize;
    let mut wrong: Option<(u32, u32, [u8; 4])> = None;
    for (x, y, pixel) in noise_mask.enumerate_pixels() {
        let [r, g, b, a] = pixel.0;
        if a == 0 {
            continue;
        }
        if a < 255 {
            faded += 1;
        }
        if [r, g, b] != PAPER && wrong.is_none() {
            wrong = Some((x, y, pixel.0));
        }
    }
    assert!(
        faded >= MINIMUM_FADED_RIM_PIXELS,
        "fixture must actually exercise the partial-alpha composite: only {faded} \
         partially transparent noise-mask pixels, expected at least \
         {MINIMUM_FADED_RIM_PIXELS}"
    );
    assert_eq!(
        wrong, None,
        "every non-transparent noise-mask pixel's RGB must be the denoised layer's own \
         colour {PAPER:?}, never that colour premultiplied against the transparent \
         canvas (§16.45 item 1)"
    );
}
