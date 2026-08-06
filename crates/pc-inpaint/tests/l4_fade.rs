//! L4 — spec §16.38 item 3(h): the Gaussian fade for `inpainting_fade_radius`, and the isolation cut.
//!
//! Upstream is `mask.convert("L").filter(ImageFilter.GaussianBlur(fade_radius))`
//! (`image_ops.py:820-830`), with `combined_mask.convert("L")` when the radius is `0`
//! (`inpainting.py:159-163`). `DEVIATION(6)` reaches this consumer: v1 uses a **true** separable
//! Gaussian with `sigma = radius` truncated at `3*sigma` (§16.10 item 12) rather than PIL's
//! three-pass box-blur approximation, and nothing measures the difference at the inpainting default
//! of `4` — see `fade::fade_fill_mask`.

mod common;

use pc_imageops::morph::{dilate, kernel};
use pc_imageops::BinaryMask;
use pc_inpaint::{cut_by_isolation, fade_fill_mask};

/// A vertical half-plane: set for `x >= edge`. The profile of its fade is a pure 1-D function of `x`,
/// which is what makes the expected values hand-computable.
fn half_plane(size: (u32, u32), edge: u32) -> BinaryMask {
    BinaryMask::from_fn(size.0, size.1, |x, _| x >= edge)
}

/// §16.38 item 3(h)'s `else` branch: with `inpainting_fade_radius = 0` the mask is used as-is, lifted
/// to `L`. Asserted against the mask's own `to_gray()` **and** with a non-blank precondition, so it
/// cannot pass on two empty images.
#[test]
fn a_fade_radius_of_zero_is_the_hard_mask_lifted_to_eight_bit() {
    let fill = half_plane((9, 4), 5);
    assert_eq!(fill.count_set(), 16, "4 columns x 4 rows");

    let faded = fade_fill_mask(&fill, 0);

    assert_eq!(faded, fill.to_gray());
    assert_eq!(faded.get_pixel(4, 0).0[0], 0);
    assert_eq!(faded.get_pixel(5, 0).0[0], 255);
}

/// §16.10 item 12 pins the truncation at `k = ceil(3*sigma)` taps each side, and §16.38 item 3(h) makes
/// `sigma = inpainting_fade_radius`. So for radius 2 the kernel reaches exactly 6 pixels, and the fade
/// of a half-plane whose edge is at `x = 10` has three hand-derivable landmarks:
///
///   * `x <= 3` is exactly `0` — six taps short of the edge, no weight reaches it;
///   * `x = 4` is the first non-zero column: only the outermost tap `exp(-36/8)/Z ≈ 0.0022182` reaches,
///     giving `floor(255 * 0.0022182 + 0.5) = 1`;
///   * `x = 15` is the last column below saturation (`255 * (1 - 0.0022182) ≈ 254.43 -> 254`), and
///     `x >= 16` is exactly `255`.
///
/// `Z = 1 + 2*(e^-0.125 + e^-0.5 + e^-1.125 + e^-2 + e^-3.125 + e^-4.5) ≈ 5.008122`, computed by hand
/// from §16.10 item 12's formula rather than from the implementation. A wrong sigma, a wrong truncation
/// width or a missing blur moves at least one of these four columns.
#[test]
fn the_fade_of_a_half_plane_reaches_exactly_three_sigma_with_the_hand_computed_endpoints() {
    let fill = half_plane((30, 7), 10);
    let faded = fade_fill_mask(&fill, 2);
    let row = 3; // the middle row, so the vertical pass's replicate border is irrelevant

    for x in 0..=3_u32 {
        assert_eq!(
            faded.get_pixel(x, row).0[0],
            0,
            "x = {x} is more than 3*sigma = 6 from the edge at 10"
        );
    }
    assert_eq!(
        faded.get_pixel(4, row).0[0],
        1,
        "the outermost tap alone: floor(255 * 0.0022182 + 0.5) = 1"
    );
    assert_eq!(
        faded.get_pixel(15, row).0[0],
        254,
        "one tap short of saturation: floor(255 * (1 - 0.0022182) + 0.5) = 254"
    );
    for x in 16..30_u32 {
        assert_eq!(
            faded.get_pixel(x, row).0[0],
            255,
            "x = {x} is at least 3*sigma inside the mask, so every tap lands on a set pixel"
        );
    }
}

/// The fade must be **monotone** across the transition and must actually produce intermediate levels.
/// A box blur, a wrong normalisation or an off-by-one in the taps can still satisfy the endpoints
/// above; a non-monotone or two-level profile cannot satisfy this.
#[test]
fn the_fade_rises_monotonically_across_the_transition_band_with_intermediate_levels() {
    let fill = half_plane((30, 7), 10);
    let faded = fade_fill_mask(&fill, 2);
    let row = 3;

    let profile: Vec<u8> = (0..20).map(|x| faded.get_pixel(x, row).0[0]).collect();
    for window in profile.windows(2) {
        assert!(
            window[1] >= window[0],
            "the fade profile must not go down: {profile:?}"
        );
    }
    let intermediate = profile.iter().filter(|v| **v > 0 && **v < 255).count();
    assert_eq!(
        intermediate, 12,
        "columns 4..=15 inclusive are strictly between 0 and 255 — 12 of them, which is 2*3*sigma"
    );
}

/// A wider radius spreads further, and these are the hand-computed columns at which the fade first
/// becomes **visible** — which is *not* the same as the kernel's support, and the difference is the
/// point of this test rather than a caveat on it.
///
/// The support reaches `k = ceil(3*sigma)` (§16.10 item 12), so radius 4's kernel touches column
/// `30 - 12 = 18`. But the value there is the outermost tap alone: `255 * exp(-4.5)/Z_4` with
/// `Z_4 ≈ 10.009173`, i.e. `0.283`, and §16.10 item 12's rounding rule is
/// `floor(v + 0.5)` — so column 18 comes out **0** and the first visible column is 19. At radius 2
/// the same tap is `255 * exp(-4.5)/5.008122 = 0.566`, which rounds to 1, and at radius 1 it is
/// `255 * exp(-4.5)/2.50595 = 1.130`. So the visible reach is `3*sigma` for radius 1 and 2 and one
/// short of it for radius 4.
///
/// All three columns were computed from §16.10 item 12's closed form outside this codebase (a Python
/// evaluation of `exp(-j^2/(2*sigma^2))`, normalised, then `floor(255*s + 0.5)`), so they are an
/// independent oracle and not a transcription of what the code returned. Naming the test after
/// "3*sigma" would have been a claim one of the three rows falsifies.
#[test]
fn the_fade_becomes_visible_at_the_hand_computed_column_for_each_radius() {
    let fill = half_plane((60, 5), 30);
    let first_nonzero = |radius: u32| {
        let faded = fade_fill_mask(&fill, radius);
        (0..60)
            .find(|x| faded.get_pixel(*x, 2).0[0] > 0)
            .expect("the fade must reach somewhere")
    };

    assert_eq!(
        first_nonzero(1),
        27,
        "support 3; outermost tap x 255 = 1.130 -> 1"
    );
    assert_eq!(
        first_nonzero(2),
        24,
        "support 6; outermost tap x 255 = 0.566 -> 1"
    );
    assert_eq!(
        first_nonzero(4),
        19,
        "the ratified default: support reaches 18, but its outermost tap x 255 = 0.283 -> 0"
    );
}

/// §16.38 item 3(h) / upstream `:166-167`: `final_mask` is the faded mask pasted through the isolation
/// mask onto a zeroed `L` canvas, so fade that spread beyond the isolation mask is discarded.
///
/// The pair of assertions is what makes this falsifiable: a pixel where the fade is non-zero **and**
/// the isolation mask is clear must come out `0`, and the same pixel's pre-cut value must be non-zero —
/// otherwise "the cut removed it" would be true of a cut that does nothing.
#[test]
fn the_isolation_cut_discards_fade_that_spread_outside_the_isolation_mask() {
    let mut fill = BinaryMask::new(21, 21);
    fill.set(10, 10, true);
    let isolation = dilate(&fill, &kernel(1)); // a plus: 5 pixels

    let faded = fade_fill_mask(&fill, 2);
    let cut = cut_by_isolation(&faded, &isolation);

    assert_eq!(isolation.count_set(), 5, "kernel(1) is a plus");
    assert!(
        faded.get_pixel(10, 6).0[0] > 0,
        "premise: the fade reaches (10,6), four pixels out"
    );
    assert!(
        !isolation.get(10, 6),
        "premise: the isolation mask does not"
    );
    assert_eq!(
        cut.get_pixel(10, 6).0[0],
        0,
        "so the cut must zero it (upstream pastes THROUGH the isolation mask)"
    );

    assert_eq!(
        cut.get_pixel(10, 10).0[0],
        faded.get_pixel(10, 10).0[0],
        "inside the isolation mask the faded value survives unchanged"
    );
    let surviving = (0..21)
        .flat_map(|y| (0..21).map(move |x| (x, y)))
        .filter(|(x, y)| cut.get_pixel(*x, *y).0[0] > 0)
        .count();
    assert!(
        surviving <= 5,
        "nothing outside the 5-pixel isolation mask may survive; got {surviving}"
    );
    assert!(surviving > 0, "and the cut must not have zeroed everything");
}

/// The cut must not *raise* anything: `final_mask <= faded` everywhere, which is what makes the fade
/// one-sided in effect. Asserted over the whole canvas rather than at a sample.
#[test]
fn the_isolation_cut_never_increases_a_value() {
    let mut fill = BinaryMask::new(21, 21);
    for x in 8..13 {
        for y in 8..13 {
            fill.set(x, y, true);
        }
    }
    let isolation = dilate(&fill, &kernel(2));
    let faded = fade_fill_mask(&fill, 3);
    let cut = cut_by_isolation(&faded, &isolation);

    for y in 0..21 {
        for x in 0..21 {
            assert!(
                cut.get_pixel(x, y).0[0] <= faded.get_pixel(x, y).0[0],
                "the cut raised ({x}, {y})"
            );
        }
    }
    // The peak must survive the cut. Note it is NOT 255 here: a 5x5 fill blurred with sigma = 3
    // (a 19-tap kernel) never saturates, so asserting `== 255` would be asserting something false
    // about the Gaussian rather than something true about the cut.
    let peak_faded = faded.pixels().map(|pixel| pixel.0[0]).max().unwrap();
    let peak_cut = cut.pixels().map(|pixel| pixel.0[0]).max().unwrap();
    assert!(peak_faded > 0, "the fade must produce something");
    assert!(
        peak_faded < 255,
        "premise: a 5x5 blob at sigma 3 does not saturate, so the check below is about the cut"
    );
    assert_eq!(
        peak_cut, peak_faded,
        "the peak lies inside the isolation mask and survives"
    );
}
