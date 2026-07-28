//! Task N2 -- spec §11.3 step 5, §14.6 (DEVIATION 6), §16.10 item 12. FROZEN.
//!
//! §11.3 step 5 replaces PIL's `GaussianBlur(radius=r)` -- a three-pass box-blur
//! approximation -- with a true separable Gaussian, `sigma = radius`, truncated at
//! `3*sigma`. §16.10 item 12 pins every remaining degree of freedom: tap formula,
//! `f64` normalisation, `f32` intermediate with no mid-rounding, replicate borders,
//! and `clamp(floor(v + 0.5), 0, 255)` at the end.

mod common;

use image::{GrayImage, Luma};
use pc_denoise::gaussian;
use pc_testkit::images;

#[test]
fn taps_are_normalised_symmetric_and_truncated_at_three_sigma() {
    // §16.10 item 12: half-width `k = ceil(3*sigma)`, so `radius = 1` gives 7 taps and
    // `radius = 2` gives 13. A wrong truncation changes the fade profile of every
    // noise mask, so the count is pinned, not just the shape.
    assert_eq!(
        gaussian::taps(0),
        vec![1.0],
        "radius 0 is the identity kernel"
    );
    assert_eq!(gaussian::taps(1).len(), 7);
    assert_eq!(gaussian::taps(2).len(), 13);
    assert_eq!(gaussian::taps(3).len(), 19);

    for radius in 1..=4_u32 {
        let taps = gaussian::taps(radius);
        let total: f64 = taps.iter().sum();
        pc_testkit::assert_close(total, 1.0, 1e-12);
        for (left, right) in taps.iter().zip(taps.iter().rev()) {
            pc_testkit::assert_close(*left, *right, 1e-15);
        }
        let centre = taps.len() / 2;
        assert!(
            taps.iter()
                .enumerate()
                .all(|(i, w)| i == centre || *w < taps[centre]),
            "the centre tap must be the largest"
        );
    }

    // The radius-1 taps, hand-computed: exp(-j^2/2) / sum, j in -3..=3.
    let taps = gaussian::taps(1);
    pc_testkit::assert_close(taps[3], 0.398_942_3 / 0.999_730_0, 1e-4);
    pc_testkit::assert_close(taps[3], 1.0 / 2.505_949_878_974_976, 1e-12);
    pc_testkit::assert_close(
        taps[2],
        0.606_530_659_712_633_4 / 2.505_949_878_974_976,
        1e-12,
    );
}

#[test]
fn radius_zero_is_the_identity() {
    // §11.3 step 4's kernel has the same convention: a zero size never modifies the
    // mask. A `noise_fade_radius = 0` profile must produce a hard-edged noise mask,
    // not a blank one.
    let source = images::noisy_gray(11, 9, 128, 20.0, 0xA1);
    assert_eq!(gaussian::blur(&source, 0), source);
}

#[test]
fn a_constant_image_survives_the_replicate_border() {
    // Replicate borders (§16.10 item 12) mean the taps that fall outside the canvas
    // repeat the edge pixel, so a uniform image is unchanged everywhere including the
    // corners. A zero-padded implementation darkens the border and fails here.
    for radius in 1..=3_u32 {
        for value in [0_u8, 97, 255] {
            let flat = GrayImage::from_pixel(9, 7, Luma([value]));
            assert_eq!(
                gaussian::blur(&flat, radius),
                flat,
                "radius {radius}, value {value}"
            );
        }
    }
}

#[test]
fn a_single_bright_pixel_blurs_to_the_hand_computed_profile() {
    // A 7x1 row with a single 255 at the centre and `radius = 1`. Hand-computed from
    // §16.10 item 12's own formula (taps exp(-j^2/2)/2.505949878974976, replicate
    // borders, round half away from zero):
    //   w0 = 0.39905022, w1 = 0.24203450, w2 = 0.05400642, w3 = 0.00443295
    //   255 * w = 101.758, 61.719, 13.772, 1.130  ->  102, 62, 14, 1
    // This is the value gate: it pins the tap normalisation, the border rule and the
    // rounding all at once.
    let mut row = GrayImage::from_pixel(7, 1, Luma([0]));
    row.put_pixel(3, 0, Luma([255]));
    let blurred = gaussian::blur(&row, 1);
    let values: Vec<u8> = blurred.pixels().map(|pixel| pixel.0[0]).collect();
    assert_eq!(values, vec![1, 14, 62, 102, 62, 14, 1]);
}

#[test]
fn blurring_is_separable_and_monotone_away_from_a_disc() {
    // The property §11.3 step 5 actually needs: "fade its edges". A solid disc's
    // blurred edge must decrease monotonically outwards, with the interior untouched.
    let mut disc = GrayImage::from_pixel(31, 31, Luma([0]));
    for y in 0..31_i32 {
        for x in 0..31_i32 {
            if (x - 15).pow(2) + (y - 15).pow(2) <= 8 * 8 {
                disc.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }
    }
    let blurred = gaussian::blur(&disc, 2);
    assert_eq!(
        blurred.get_pixel(15, 15).0[0],
        255,
        "the interior is untouched"
    );

    let profile: Vec<u8> = (15..31).map(|x| blurred.get_pixel(x, 15).0[0]).collect();
    for pair in profile.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "the blurred edge profile must be non-increasing: {profile:?}"
        );
    }
    assert_eq!(*profile.last().expect("non-empty"), 0, "the fade dies out");
    assert!(
        profile.iter().any(|value| *value > 0 && *value < 255),
        "there must actually be a soft edge: {profile:?}"
    );
}
