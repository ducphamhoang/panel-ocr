//! C4 tests -- SSIM. Frozen gates.
//!
//! These tests do not merely check that SSIM "looks right"; they PIN the estimator
//! (spec §16.5 item 7): non-overlapping 8x8 tiles, uniform window, POPULATION
//! variance, unweighted mean over tiles, K1=0.01/K2=0.03/L=255. Every expected value
//! below is hand-derived from the SSIM formula and shown in its comment. If the
//! architects change the estimator, these tests change with it -- Codex may not.

use approx::assert_abs_diff_eq;
use pc_testkit::images::{checkerboard, gray_from_rows, solid_gray};
use pc_testkit::metrics::{ssim_gray, ssim_gray_with, ssim_tile, SsimParams};

const EPS: f64 = 1e-9;

#[test]
// The constants are part of the contract: C1 = (0.01*255)^2 = 6.5025,
// C2 = (0.03*255)^2 = 58.5225. Every hand-computed expectation below uses them.
fn stability_constants() {
    let p = SsimParams::default();
    assert_eq!(p.window, 8);
    assert_eq!(p.k1, 0.01);
    assert_eq!(p.k2, 0.03);
    assert_eq!(p.dynamic_range, 255.0);
    assert_abs_diff_eq!(p.c1(), 6.502_500_000_000_001, epsilon = EPS);
    assert_abs_diff_eq!(p.c2(), 58.522_499_999_999_994, epsilon = EPS);
}

#[test]
// SSIM(x, x) == 1.0 exactly, for every kind of image: flat, structured, and
// non-multiple-of-8. This is the single most important self-check -- a metric that
// cannot score an image against itself as 1.0 cannot gate anything.
fn ssim_of_an_image_against_itself_is_one() {
    let cases = [
        solid_gray(8, 8, 0),
        solid_gray(8, 8, 255),
        solid_gray(16, 16, 128),
        checkerboard(8, 8, 0, 255),
        checkerboard(37, 19, 10, 200), // deliberately not a multiple of 8
        gray_from_rows(&[&[1, 2, 3], &[4, 5, 6]]),
    ];
    for img in cases {
        assert_abs_diff_eq!(ssim_gray(&img, &img), 1.0, epsilon = EPS);
    }
}

#[test]
// Two flat images differing by a constant offset exercise ONLY the luminance term:
//   mu_x = 100, mu_y = 150, sigma_x = sigma_y = sigma_xy = 0
//   SSIM = (2*100*150 + C1)/(100^2 + 150^2 + C1) * (0 + C2)/(0 + C2)
//        = 30006.5025 / 32506.5025 * 1
//        = 0.923092310530793
// This also pins that a zero-variance tile is NOT special-cased to 0 or 1: the
// stability constants must carry it.
fn ssim_of_two_constant_images_is_the_luminance_term() {
    let a = solid_gray(8, 8, 100);
    let b = solid_gray(8, 8, 150);
    assert_abs_diff_eq!(ssim_gray(&a, &b), 0.923_092_310_530_793, epsilon = EPS);
}

#[test]
// A checkerboard against its inverse exercises ONLY the contrast/structure term and
// pins POPULATION variance (divide by N):
//   mu_x = mu_y = 127.5           -> luminance term = 1 exactly
//   sigma_x^2 = sigma_y^2 = 127.5^2 = 16256.25   (population, N=64)
//   sigma_xy  = -16256.25
//   SSIM = (2*(-16256.25) + C2) / (16256.25 + 16256.25 + C2)
//        = -32453.9775 / 32571.0225
//        = -0.9964064683569576
// An implementation using the UNBIASED (N-1) variance would give a different number
// here, which is exactly why this case is frozen.
fn ssim_of_a_checkerboard_and_its_inverse_is_negative() {
    let a = checkerboard(8, 8, 0, 255);
    let b = checkerboard(8, 8, 255, 0);
    assert_abs_diff_eq!(ssim_gray(&a, &b), -0.996_406_468_356_957_6, epsilon = EPS);
}

#[test]
// SSIM is symmetric in its arguments.
fn ssim_is_symmetric() {
    let a = checkerboard(16, 16, 20, 200);
    let b = solid_gray(16, 16, 90);
    assert_abs_diff_eq!(ssim_gray(&a, &b), ssim_gray(&b, &a), epsilon = EPS);
}

#[test]
// The global score is the UNWEIGHTED MEAN over non-overlapping 8x8 tiles (spec §16.5
// item 7). A 16x8 image = exactly 2 tiles: tile 0 identical (SSIM 1.0), tile 1
// constant 100-vs-150 (SSIM 0.923092310530793).
//   expected = (1.0 + 0.923092310530793) / 2 = 0.9615461552653965
// A sliding-window implementation would NOT produce this number.
fn global_ssim_is_the_unweighted_mean_over_tiles() {
    let mut a = solid_gray(16, 8, 100);
    let mut b = solid_gray(16, 8, 100);
    // right-hand tile only
    for y in 0..8 {
        for x in 8..16 {
            a.put_pixel(x, y, image::Luma([100]));
            b.put_pixel(x, y, image::Luma([150]));
        }
    }
    assert_abs_diff_eq!(ssim_gray(&a, &b), 0.961_546_155_265_396_5, epsilon = EPS);
}

#[test]
// Partial edge tiles are INCLUDED at their real size and count as one tile each, with
// the same unweighted mean (spec §16.5 item 7). A 12x8 image = one 8-wide tile + one
// 4-wide tile:
//   tile 0 identical (1.0), tile 1 constant 100-vs-150 (0.923092310530793)
//   expected = 0.9615461552653965 -- the same as the 16x8 case above, because the
//   partial tile is NOT area-weighted.
fn partial_edge_tiles_are_included_and_not_area_weighted() {
    let mut a = solid_gray(12, 8, 100);
    let mut b = solid_gray(12, 8, 100);
    for y in 0..8 {
        for x in 8..12 {
            a.put_pixel(x, y, image::Luma([100]));
            b.put_pixel(x, y, image::Luma([150]));
        }
    }
    assert_abs_diff_eq!(ssim_gray(&a, &b), 0.961_546_155_265_396_5, epsilon = EPS);
}

#[test]
// An image smaller than one window is a single (partial) tile, not an error and not
// a zero score. Tiny inputs must be comparable.
fn images_smaller_than_the_window_are_a_single_tile() {
    let a = gray_from_rows(&[&[100, 100], &[100, 100]]);
    let b = gray_from_rows(&[&[150, 150], &[150, 150]]);
    assert_abs_diff_eq!(ssim_gray(&a, &a), 1.0, epsilon = EPS);
    assert_abs_diff_eq!(ssim_gray(&a, &b), 0.923_092_310_530_793, epsilon = EPS);
}

#[test]
// `ssim_tile` is the per-tile primitive the global score averages, so the same
// hand-computed values must come out of it directly.
fn ssim_tile_matches_the_hand_computed_values() {
    let p = SsimParams::default();
    let flat_a = [100u8; 64];
    let flat_b = [150u8; 64];
    assert_abs_diff_eq!(ssim_tile(&flat_a, &flat_a, p), 1.0, epsilon = EPS);
    assert_abs_diff_eq!(
        ssim_tile(&flat_a, &flat_b, p),
        0.923_092_310_530_793,
        epsilon = EPS
    );
}

#[test]
// SSIM never escapes [-1, 1], for any input pair.
fn ssim_stays_within_minus_one_and_one() {
    let cases = [
        (solid_gray(8, 8, 0), solid_gray(8, 8, 255)),
        (checkerboard(24, 24, 0, 255), checkerboard(24, 24, 255, 0)),
        (checkerboard(9, 17, 3, 250), solid_gray(9, 17, 128)),
    ];
    for (a, b) in cases {
        let s = ssim_gray(&a, &b);
        assert!((-1.0..=1.0).contains(&s), "SSIM out of range: {s}");
    }
}

#[test]
// Parameters are honoured: a larger window changes the tiling and therefore the score
// on the 16x8 two-tile case (one 16-wide tile instead of two 8-wide ones).
fn window_size_parameter_changes_the_tiling() {
    let mut a = solid_gray(16, 8, 100);
    let mut b = solid_gray(16, 8, 100);
    for y in 0..8 {
        for x in 8..16 {
            a.put_pixel(x, y, image::Luma([100]));
            b.put_pixel(x, y, image::Luma([150]));
        }
    }
    let tiled_8 = ssim_gray(&a, &b);
    let one_tile = ssim_gray_with(
        &a,
        &b,
        SsimParams {
            window: 16,
            ..Default::default()
        },
    );
    assert!(
        (tiled_8 - one_tile).abs() > 1e-6,
        "the window parameter must actually change the tiling ({tiled_8} vs {one_tile})"
    );
}

#[test]
// Mismatched dimensions are a programming error in a test, so they panic immediately
// rather than silently comparing a prefix.
#[should_panic(expected = "dimension")]
fn ssim_panics_on_dimension_mismatch() {
    let _ = ssim_gray(&solid_gray(8, 8, 0), &solid_gray(8, 9, 0));
}
