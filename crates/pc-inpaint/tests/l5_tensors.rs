//! Task **L5** — the tensor construction and the runtime output validation, in the DEFAULT
//! no-`onnx` tier (cookbook rule 6: the tier that actually executes).
//!
//! Every assertion here traces to a numbered clause of spec §16.38 and every expected value
//! is either hand-computed or a literal taken from the spec. Nothing here reads the pinned
//! artifact, so nothing here can pass by agreeing with it.

use image::{Rgb, RgbImage};
use pc_core::StageError;
use pc_imageops::BinaryMask;
use pc_inpaint::onnx::{
    decode_output_tile, image_to_nchw, mask_to_nchw, EXPECTED_OUTPUT_SHAPE, IMAGE_CHANNELS,
    IMAGE_INPUT, MASK_CHANNELS, MASK_INPUT, SAMPLE_SCALE,
};
use pc_inpaint::TILE;

const SIDE: usize = 512;
const PLANE: usize = SIDE * SIDE;

/// The interface literals of §16.38 item 1(b), spelled out here so a change to any of them
/// is a deliberate frozen-test question rather than a free edit:
///
/// * *"`image`: `elem_type = 1` (float32), dims `[dim_param "batch", 3, 512, 512]`"*
/// * *"`mask`: `elem_type = 1` (float32), dims `[dim_param "batch", 1, 512, 512]`"*
/// * *"There is **no** concatenated 4-channel input."*
///
/// The names are asserted because `run_raw` binds the two inputs **by name**; a typo in
/// either constant would silently feed the mask where the image belongs, which no shape
/// check downstream could catch (both are `f32` `512×512`).
#[test]
fn the_two_input_names_channel_counts_and_the_tile_side_are_the_literals_item_1b_decoded() {
    assert_eq!(IMAGE_INPUT, "image");
    assert_eq!(MASK_INPUT, "mask");
    assert_eq!(IMAGE_CHANNELS, 3);
    assert_eq!(MASK_CHANNELS, 1);
    assert_eq!(TILE, 512);
    // Anti-vacuity: the expected runtime output shape is a literal, not a value read off a
    // session. §16.38 item 1(c): "must never derive geometry from the declared one."
    assert_eq!(EXPECTED_OUTPUT_SHAPE, [1, 3, 512, 512]);
    // The sigmoid-output scale, MEASURED at L5 against the pinned artifact and recorded in
    // `onnx.rs`. Pinned here so a silent change to it must be argued.
    assert_eq!(SAMPLE_SCALE, 255.0);
}

/// §16.38 item 1(b): the `image` input is **planar NCHW**, `[batch, 3, 512, 512]`.
///
/// Three pixels at three distinct positions are checked in all three planes, with
/// hand-computed flat indices and hand-computed normalised values. An HWC layout, a plane
/// ordering of B-G-R, or a divisor other than 255 each turns this red.
#[test]
fn image_to_nchw_writes_three_planes_in_r_g_b_order_each_divided_by_255() {
    let mut tile = RgbImage::from_pixel(TILE, TILE, Rgb([0, 0, 0]));
    tile.put_pixel(0, 0, Rgb([255, 0, 0]));
    tile.put_pixel(7, 3, Rgb([0, 128, 64]));
    tile.put_pixel(511, 511, Rgb([1, 2, 3]));

    let tensor = image_to_nchw(&tile).expect("a 512x512 tile is accepted");

    assert_eq!(tensor.len(), 3 * PLANE, "3 * 512 * 512 = 786432 samples");
    assert_eq!(tensor.len(), 786_432);

    // (0, 0) -> flat index 0 in each plane.
    assert_eq!(tensor[0], 1.0, "R plane at (0,0): 255/255");
    assert_eq!(tensor[PLANE], 0.0, "G plane at (0,0)");
    assert_eq!(tensor[2 * PLANE], 0.0, "B plane at (0,0)");

    // (7, 3) -> row-major flat index 3 * 512 + 7 = 1543.
    let index = 3 * SIDE + 7;
    assert_eq!(index, 1543);
    assert_eq!(tensor[index], 0.0, "R plane at (7,3)");
    assert_eq!(tensor[PLANE + index], 128.0 / 255.0, "G plane at (7,3)");
    assert_eq!(tensor[2 * PLANE + index], 64.0 / 255.0, "B plane at (7,3)");

    // (511, 511) -> flat index 511 * 512 + 511 = 262143, the last sample of each plane.
    let index = 511 * SIDE + 511;
    assert_eq!(index, 262_143);
    assert_eq!(tensor[index], 1.0 / 255.0);
    assert_eq!(tensor[PLANE + index], 2.0 / 255.0);
    assert_eq!(tensor[2 * PLANE + index], 3.0 / 255.0);

    // And the untouched bulk really is zero, so the three checks above are not the only
    // non-zero samples by accident.
    let non_zero = tensor.iter().filter(|sample| **sample != 0.0).count();
    assert_eq!(
        non_zero, 6,
        "exactly the six non-zero channel values placed above"
    );
}

/// §16.38 item 1(g): *"Mask convention confirmed: value 1 = fill, value 0 = keep."*
///
/// Asserted as the **set** of positions holding `1.0` rather than as a count — cardinality is
/// not identity, and a transposed `(x, y)` keeps the count while filling the wrong pixels.
#[test]
fn mask_to_nchw_writes_one_plane_holding_1_at_exactly_the_set_positions() {
    let set = [(0_u32, 0_u32), (7, 3), (511, 0), (0, 511)];
    let mask = BinaryMask::from_fn(TILE, TILE, |x, y| set.contains(&(x, y)));

    let tensor = mask_to_nchw(&mask).expect("a 512x512 mask is accepted");

    assert_eq!(tensor.len(), PLANE, "one channel, 512 * 512 = 262144");
    assert_eq!(tensor.len(), 262_144);

    let expected: std::collections::BTreeSet<usize> = set
        .iter()
        .map(|(x, y)| (*y as usize) * SIDE + (*x as usize))
        .collect();
    // 3*512+7 = 1543, 0*512+511 = 511, 511*512+0 = 261632, plus 0.
    assert_eq!(
        expected,
        [0_usize, 511, 1543, 261_632].into_iter().collect(),
        "the hand-computed flat indices"
    );

    let actual: std::collections::BTreeSet<usize> = tensor
        .iter()
        .enumerate()
        .filter(|(_, sample)| **sample == 1.0)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(actual, expected, "1 = fill, and at exactly those pixels");

    let others: Vec<f32> = tensor
        .iter()
        .enumerate()
        .filter(|(index, _)| !expected.contains(index))
        .map(|(_, sample)| *sample)
        .collect();
    assert!(
        others.iter().all(|sample| *sample == 0.0),
        "0 = keep everywhere else"
    );
}

/// §16.38 item 1(b) measured the H/W axes as literal `512`s and item 1(d) measured three
/// off-size inputs being rejected by a real `ort` session. Refusing the wrong size before
/// building a tensor turns that into a named error instead of an ORT dimension complaint.
///
/// §16.38 item 9(g) fixes the classification: this code path only runs with an image in
/// hand, so the variant is `Inference` (per-image), never `Model` (run-fatal).
#[test]
fn image_and_mask_tensor_builders_reject_a_non_512_square_input_as_an_inference_error() {
    let short = RgbImage::from_pixel(511, TILE, Rgb([0, 0, 0]));
    let tall = RgbImage::from_pixel(TILE, 513, Rgb([0, 0, 0]));
    for wrong in [&short, &tall] {
        let error = image_to_nchw(wrong).expect_err("an off-size tile is refused");
        assert!(
            matches!(error, StageError::Inference(_)),
            "got {error:?} for {:?}",
            wrong.dimensions()
        );
        assert!(error.to_string().contains("512"));
    }

    for size in [(256_u32, 256_u32), (640, 512), (1024, 1024)] {
        // The three sizes §16.38 item 1(d) actually fed the real session.
        let mask = BinaryMask::new(size.0, size.1);
        let error = mask_to_nchw(&mask).expect_err("an off-size mask is refused");
        assert!(matches!(error, StageError::Inference(_)), "got {error:?}");
    }
}

/// §16.38 item 1(c): *"the implementation must validate the **runtime** output shape and must
/// never derive geometry from the declared one."*
///
/// The accepted shape is the single literal `[1, 3, 512, 512]`. The rejected list includes
/// the two forms item 1(c) names as hazards — a symbolic `-1` axis, and a height axis that
/// took the batch value — plus a near-miss and a wrong rank, because a check written as
/// "rank is 4 and channels are 3" would pass those.
#[test]
fn decode_output_tile_accepts_only_the_runtime_shape_1_3_512_512() {
    let good = vec![0.0_f32; 3 * PLANE];
    let decoded =
        decode_output_tile(&[1, 3, 512, 512], &good).expect("the measured runtime shape decodes");
    assert_eq!(decoded.dimensions(), (512, 512));

    let rejected: Vec<Vec<i64>> = vec![
        vec![-1, 3, -1, -1],  // item 1(c)'s symbolic axes, as `ort` renders a dim_param
        vec![1, 3, 1, 1],     // the height axis carrying the batch VALUE
        vec![1, 3, 512, 511], // a one-pixel near-miss
        vec![1, 3, 512],      // wrong rank, right prefix
        vec![2, 3, 512, 512], // a batch this stage never asks for (item 1(d): linear, so no batching)
        vec![1, 1, 512, 512], // the mask's channel count, not the image's
        vec![1, 512, 512, 3], // NHWC
    ];
    assert_eq!(
        rejected.len(),
        7,
        "anti-vacuity: seven rejection cases are asserted below, and this count cannot be \
         computed from the code under test"
    );
    for shape in &rejected {
        let error = decode_output_tile(shape, &good)
            .expect_err("only the exact runtime shape may be accepted");
        assert!(
            matches!(error, StageError::Inference(_)),
            "shape {shape:?} gave {error:?}"
        );
    }
}

/// A right-shaped header over the wrong number of samples must not be indexed into.
#[test]
fn decode_output_tile_rejects_a_sample_count_that_does_not_match_its_shape() {
    let short = vec![0.0_f32; 3 * PLANE - 1];
    let error = decode_output_tile(&[1, 3, 512, 512], &short).expect_err("short buffer");
    assert!(matches!(error, StageError::Inference(_)), "got {error:?}");
    assert!(error.to_string().contains("786432"));
}

/// The sample→byte mapping, as hand-computed pairs. `SAMPLE_SCALE` is 255 and `f32::round`
/// is documented to round half **away from zero**, so `0.5 * 255 = 127.5` becomes `128`.
///
/// Out-of-range samples are clamped rather than refused: the graph ends in a sigmoid, so a
/// hair over `1.0` is rounding noise (L5 measured a maximum of `0.99999857`).
#[test]
fn decode_output_tile_scales_by_255_rounds_half_away_from_zero_and_clamps_out_of_range() {
    // (sample, expected byte) — every expectation hand-computed, none read from a model.
    let cases: [(f32, u8); 9] = [
        (0.0, 0),
        (1.0, 255),
        (0.5, 128),           // 127.5 rounds away from zero
        (0.5 / 255.0, 1),     // exactly 0.5 after scaling -> 1
        (0.49 / 255.0, 0),    // 0.49 -> 0
        (128.0 / 255.0, 128), // an exact byte round-trips
        (0.99999857, 255),    // the maximum L5 measured
        (-1.0, 0),            // clamped low
        (2.0, 255),           // clamped high
    ];
    assert_eq!(cases.len(), 9, "anti-vacuity: nine hand-computed pairs");

    for (sample, expected) in cases {
        let mut values = vec![0.0_f32; 3 * PLANE];
        // One pixel, all three channels, at (5, 9) -> flat index 9 * 512 + 5 = 4613.
        let index = 9 * SIDE + 5;
        assert_eq!(index, 4613);
        values[index] = sample;
        values[PLANE + index] = sample;
        values[2 * PLANE + index] = sample;

        let tile = decode_output_tile(&[1, 3, 512, 512], &values).expect("decodes");
        assert_eq!(
            *tile.get_pixel(5, 9),
            Rgb([expected, expected, expected]),
            "sample {sample} must become byte {expected}"
        );
        // And the untouched neighbour stays 0, which pins the flat-index arithmetic: an
        // off-by-one in the row stride would put the value here instead.
        assert_eq!(*tile.get_pixel(6, 9), Rgb([0, 0, 0]));
    }
}

/// The three planes must land in R, G, B order and be indexed row-major, checked with three
/// mutually distinguishable values at one position.
#[test]
fn decode_output_tile_maps_plane_0_to_red_plane_1_to_green_and_plane_2_to_blue() {
    let mut values = vec![0.0_f32; 3 * PLANE];
    let index = 300 * SIDE + 200; // (x = 200, y = 300) -> 153800
    assert_eq!(index, 153_800);
    values[index] = 10.0 / 255.0;
    values[PLANE + index] = 20.0 / 255.0;
    values[2 * PLANE + index] = 30.0 / 255.0;

    let tile = decode_output_tile(&[1, 3, 512, 512], &values).expect("decodes");

    assert_eq!(*tile.get_pixel(200, 300), Rgb([10, 20, 30]));
    assert_eq!(
        *tile.get_pixel(300, 200),
        Rgb([0, 0, 0]),
        "the transposed position must be untouched, or x and y are swapped"
    );
}

/// Declared, because the spec is silent: a non-finite sample is refused rather than clamped.
///
/// `f32::clamp` panics on `NaN`, and black or white would both be an invented pixel, so the
/// tile is refused with the per-image classification of §16.38 item 9(g).
#[test]
fn decode_output_tile_refuses_a_non_finite_sample_instead_of_inventing_a_byte_for_it() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = vec![0.0_f32; 3 * PLANE];
        values[PLANE + 12] = bad;
        let error = decode_output_tile(&[1, 3, 512, 512], &values)
            .expect_err("a non-finite sample is refused");
        assert!(matches!(error, StageError::Inference(_)), "got {error:?}");
        assert!(
            error.to_string().contains("262156"),
            "the message names the offending flat index (512*512 + 12); got: {error}"
        );
    }
}
