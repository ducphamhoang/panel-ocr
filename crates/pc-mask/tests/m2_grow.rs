//! Task M2 -- spec §10.3 step 6, §10.7(A)1, §10.7(A)2, §10.7(A)3, §16.9 items 5, 6.
//! FROZEN.

mod common;

use common::set_pixels;
use pc_config::MaskerConfig;
use pc_imageops::BinaryMask;
use pc_mask::grow::{
    build_candidates, center_crop, dilate, growth_candidates, growth_padding, kernel, pad_replicate,
};

/// Row-major `0`/`1` matrix of a kernel, for whole-matrix assertions.
fn kernel_rows(thickness: u32) -> Vec<Vec<u8>> {
    let element = kernel(thickness);
    (0..element.diameter())
        .map(|i| {
            (0..element.diameter())
                .map(|j| u8::from(element.get(j, i)))
                .collect()
        })
        .collect()
}

/// The set pixels of a footprint centred at `center` in a `size`-square canvas,
/// described per row as an inclusive `|dx| <= bound` half-width.
fn footprint(size: u32, center: u32, bounds: &[u32]) -> Vec<(u32, u32)> {
    let radius = (bounds.len() as u32 - 1) / 2;
    let mut pixels = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let dy = y as i64 - center as i64;
            let dx = x as i64 - center as i64;
            if dy.unsigned_abs() > u64::from(radius) {
                continue;
            }
            let bound = bounds[(dy + i64::from(radius)) as usize];
            if dx.unsigned_abs() <= u64::from(bound) {
                pixels.push((x, y));
            }
        }
    }
    pixels
}

// ------------------------------------------------------------ kernels (§10.7(A)1)

#[test]
fn a1_kernel_1_is_a_3x3_square_with_zeroed_corners() {
    // spec §10.7(A)1: diameter 3 <= 5, so a full square with the four corners cleared.
    assert_eq!(
        kernel_rows(1),
        vec![vec![0, 1, 0], vec![1, 1, 1], vec![0, 1, 0]]
    );
    assert_eq!(kernel(1).count(), 5);
}

#[test]
fn a1_kernel_2_is_a_5x5_square_with_zeroed_corners() {
    // spec §10.7(A)1: 25 cells minus 4 corners = 21 set.
    assert_eq!(
        kernel_rows(2),
        vec![
            vec![0, 1, 1, 1, 0],
            vec![1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![0, 1, 1, 1, 0],
        ]
    );
    assert_eq!(kernel(2).count(), 21);
}

#[test]
fn a1_kernel_3_is_the_exact_opencv_7x7_ellipse() {
    // spec §10.7(A)1, hand-traced from `getStructuringElement`:
    //   dy=-3 -> dx=0            -> [3,4)
    //   dy=-2 -> dx=round(sqrt 5)=2 -> [1,6)
    //   dy=-1 -> dx=round(sqrt 8)=3 -> [0,7)
    //   dy= 0 -> dx=3            -> [0,7)
    // mirrored below. Total set = 1+5+7+7+7+5+1 = 33.
    assert_eq!(
        kernel_rows(3),
        vec![
            vec![0, 0, 0, 1, 0, 0, 0],
            vec![0, 1, 1, 1, 1, 1, 0],
            vec![1, 1, 1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1, 1, 1],
            vec![0, 1, 1, 1, 1, 1, 0],
            vec![0, 0, 0, 1, 0, 0, 0],
        ]
    );
    assert_eq!(kernel(3).count(), 33);
}

#[test]
fn a1_kernel_0_is_the_single_centre_pixel() {
    // spec §16.9 item 5: corner-zeroing applies only from diameter 3 upward -- for
    // diameter 1 the "corners" are the centre, and §10.3 step 6 requires dilation with
    // this kernel to be the identity. Reachable via `min_mask_thickness = 0`.
    assert_eq!(kernel_rows(0), vec![vec![1]]);

    let mask = BinaryMask::from_fn(5, 5, |x, y| x == 2 && y == 2);
    assert_eq!(dilate(&mask, &kernel(0)), mask);
}

// ------------------------------------------------------------ dilation (§10.7(A)2)

#[test]
fn a2_single_pixel_dilated_by_kernel_4_is_the_kernel_footprint() {
    // spec §10.7(A)2: one set pixel at the centre of a 21x21 mask, dilated with
    // kernel(4) (the 9x9 ellipse), equals the kernel footprint translated -- bit-exact.
    // Per-row half-widths, hand-traced: dx = round(sqrt(16 - dy^2)) for dy = -4..4.
    let bounds = [0, 3, 3, 4, 4, 4, 3, 3, 0];
    let mask = BinaryMask::from_fn(21, 21, |x, y| x == 10 && y == 10);

    let dilated = dilate(&mask, &kernel(4));

    assert_eq!(set_pixels(&dilated), footprint(21, 10, &bounds));
    assert_eq!(dilated.count_set(), 57);
    assert_eq!(kernel(4).count(), 57);
}

#[test]
fn a2_two_dilations_by_kernel_2_equal_the_minkowski_sum() {
    // spec §10.7(A)2: two successive dilations with kernel(2) equal one dilation with
    // the Minkowski sum of the two kernels, confirming the accumulation semantics.
    // Hand-computed 9x9 reference: K = [-2,2]^2 minus its four corners, so K (+) K
    // reaches |u| = 4 only when |v| <= 2, and |v| = 4 only when |u| <= 2.
    // Per-row half-widths for v = -4..4: [2,3,4,4,4,4,4,3,2]  (69 cells).
    let bounds = [2, 3, 4, 4, 4, 4, 4, 3, 2];
    let mask = BinaryMask::from_fn(21, 21, |x, y| x == 10 && y == 10);

    let twice = dilate(&dilate(&mask, &kernel(2)), &kernel(2));

    assert_eq!(set_pixels(&twice), footprint(21, 10, &bounds));
    assert_eq!(twice.count_set(), 69);
}

#[test]
fn a2_dilation_drops_writes_outside_the_canvas() {
    // spec §16.9 item 6: out-of-bounds stamping is dropped (zero border), never wrapped.
    let mask = BinaryMask::from_fn(3, 3, |x, y| x == 0 && y == 0);

    let dilated = dilate(&mask, &kernel(1));

    assert_eq!(set_pixels(&dilated), vec![(0, 0), (1, 0), (0, 1)]);
}

#[test]
fn pad_replicate_copies_the_edge_pixels_and_center_crop_inverts_it() {
    // `np.pad(mode="edge")` (§10.3 step 6): the border ring repeats the nearest edge
    // pixel, and the centre crop is its exact inverse.
    let mask = BinaryMask::from_fn(2, 2, |x, y| x == 0 && y == 0);

    let padded = pad_replicate(&mask, 1);

    assert_eq!(padded.dimensions(), (4, 4));
    assert_eq!(
        set_pixels(&padded),
        vec![(0, 0), (1, 0), (0, 1), (1, 1)],
        "the set corner pixel is replicated into the pad ring"
    );
    assert_eq!(center_crop(&padded, 1), mask);
}

// ------------------------------------------------------------ candidates (§10.7(A)3)

#[test]
fn a3_growth_candidate_count_and_thicknesses_match_the_defaults() {
    // spec §10.7(A)3: min_mask_thickness=4, mask_growth_step_pixels=2,
    // mask_growth_steps=11 -> thicknesses [4,6,8,...,24] (11 values).
    let config = MaskerConfig::default();
    let precise = BinaryMask::from_fn(41, 41, |x, y| x == 20 && y == 20);

    let candidates = growth_candidates(&precise, &config);

    assert_eq!(candidates.len(), 11);
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.thickness)
            .collect::<Vec<_>>(),
        (0..11).map(|i| Some(4 + i * 2)).collect::<Vec<_>>()
    );
    assert!(candidates
        .iter()
        .all(|candidate| candidate.mask.dimensions() == (41, 41)));
}

#[test]
fn a3_growth_padding_is_twice_the_larger_of_thickness_and_step() {
    // spec §10.3 step 6: pad = max(min_mask_thickness, mask_growth_step_pixels) * 2.
    assert_eq!(growth_padding(&MaskerConfig::default()), 8);
    assert_eq!(
        growth_padding(&MaskerConfig {
            min_mask_thickness: 1,
            mask_growth_step_pixels: 6,
            ..MaskerConfig::default()
        }),
        12
    );
}

#[test]
fn a3_growth_candidates_are_monotonically_larger() {
    // Each candidate is the previous one dilated again, so the set of covered pixels
    // only ever grows -- the property the improvement threshold trades against.
    let config = MaskerConfig::default();
    let precise = BinaryMask::from_fn(81, 81, |x, y| x == 40 && y == 40);

    let candidates = growth_candidates(&precise, &config);

    let mut previous = 0;
    for candidate in &candidates {
        let count = candidate.mask.count_set();
        assert!(
            count > previous,
            "candidate {:?} did not grow: {count} <= {previous}",
            candidate.thickness
        );
        previous = count;
    }
}

#[test]
fn a3_candidate_ordering_is_box_first_only_in_fast_mode() {
    // spec §10.3 step 7 / §10.7(A)3: 12 candidates in total; the box candidate
    // (thickness None) is first iff mask_selection_fast, last otherwise.
    let precise = BinaryMask::from_fn(21, 21, |x, y| x == 10 && y == 10);
    let box_mask = BinaryMask::from_fn(21, 21, |x, y| (5..15).contains(&x) && (5..15).contains(&y));

    let slow = build_candidates(&precise, box_mask.clone(), &MaskerConfig::default());
    assert_eq!(slow.len(), 12);
    assert_eq!(slow[0].thickness, Some(4));
    assert_eq!(slow[11].thickness, None);
    assert_eq!(slow[11].mask, box_mask);

    let fast = build_candidates(
        &precise,
        box_mask.clone(),
        &MaskerConfig {
            mask_selection_fast: true,
            ..MaskerConfig::default()
        },
    );
    assert_eq!(fast.len(), 12);
    assert_eq!(fast[0].thickness, None);
    assert_eq!(fast[0].mask, box_mask);
    assert_eq!(fast[1].thickness, Some(4));
}
