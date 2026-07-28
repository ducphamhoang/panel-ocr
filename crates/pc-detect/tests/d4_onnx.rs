//! Task D4a -- spec §8.3 step 3 (letterbox, NCHW normalisation, output binding + swap
//! guard, tensor decode) and §8.6's "synthetic 1024x1024 letterbox round-trip fixture
//! (exercise dw/dh when aspect != 1)". FROZEN per CLAUDE.md.
//!
//! **This file is deliberately NOT gated on the `onnx` feature and never mentions `ort`.**
//! The workspace pins `ort` with `default-features = false`, so building with
//! `--features onnx` needs an ONNX Runtime shared library that is not available in this
//! checkout -- anything gated on that feature cannot even link, let alone run. Every part
//! of D4 that is pure arithmetic therefore lives outside the gate (task D4a) and is
//! exercised by a plain `cargo test`; the `ort` session is D4b, in `d4_session.rs`.
//!
//! No model file is required by, or fabricated for, any test in this file.

use image::{Rgb, RgbImage};
use pc_core::Rect;
use pc_detect::onnx::{
    bind_outputs, decode_blocks, decode_mask, describe_outputs, ensure_model_file, letterbox,
    resize_bilinear_rgb, to_nchw, OutputBinding, OutputMeta, INTRA_THREADS, NET_SIZE, PAD_VALUE,
    STRIDE,
};
use pc_detect::yolo::{self, Candidate, LetterboxGeometry, ROW_STRIDE};

fn solid_rgb(width: u32, height: u32, pixel: [u8; 3]) -> RgbImage {
    RgbImage::from_pixel(width, height, Rgb(pixel))
}

fn geometry(net_size: u32, dw: f32, dh: f32, image_size: (u32, u32)) -> LetterboxGeometry {
    LetterboxGeometry {
        net_size,
        dw,
        dh,
        image_size,
    }
}

fn meta(name: &str, shape: &[i64]) -> OutputMeta {
    OutputMeta {
        name: name.to_string(),
        shape: shape.to_vec(),
    }
}

/// The output set the shipped `comictextdetector.pt.onnx` declares: `blk` with 7 columns,
/// then a 1-channel segmentation mask, then a 2-channel lines map. The literal `7` is
/// certified by `d4_signature.rs::row_stride_matches_the_models_declared_blk_arity`, whose
/// recorded signature was measured from the sha256-verified third-party artifact (group F3,
/// sha256 `1a86ace7…`). Do not change this back to `ROW_STRIDE as i64`: that would restore
/// the self-referential tripwire this fixture is meant to close.
fn shipped_outputs() -> [OutputMeta; 3] {
    [
        meta("blk", &[1, 64_512, 7]),
        meta("seg", &[1, 1, 1024, 1024]),
        meta("det", &[1, 2, 1024, 1024]),
    ]
}

/// `byte / 255.0`, compared with a ~1 ULP tolerance so the implementation stays free to
/// divide in `f32` or in `f64`. The *index mapping* is what these assertions pin.
#[track_caller]
fn assert_normalised(actual: f32, byte: u8) {
    let expected = f32::from(byte) / 255.0;
    assert!(
        (actual - expected).abs() <= 1e-7,
        "expected {byte}/255 = {expected}, got {actual}"
    );
}

// ------------------------------------------------------------ constants

#[test]
fn constants_match_the_spec() {
    // spec §8.3 step 3: letterbox to 1024x1024 with `stride = 64`, pad with
    // (114, 114, 114), CPU EP with `intra_threads = 1` (the parallelism is at the image
    // level, §4.5). `INTRA_THREADS` is pinned as a constant because the session option it
    // feeds is not otherwise observable from a test.
    assert_eq!(NET_SIZE, 1024);
    assert_eq!(STRIDE, 64);
    assert_eq!(PAD_VALUE, 114);
    assert_eq!(INTRA_THREADS, 1);
    // `auto = false`, so the stride never reduces the padding and the network input is
    // always exactly NET_SIZE square; the stride is nonetheless a divisor of it.
    assert_eq!(NET_SIZE % STRIDE, 0);
}

// ------------------------------------------------------------ letterbox geometry

#[test]
fn a_letterbox_geometry_matches_the_spec_formula() {
    // spec §8.3 step 3: `r = min(1024/h, 1024/w)`, resize to `round(w*r) x round(h*r)`,
    // pad right/bottom to 1024, and record `(dw, dh)` as the TOTAL padding per axis.
    //
    // Note a structural consequence, so nobody later "fixes" a case that looks
    // half-tested: with a square target and `auto = false`, the axis that determines `r`
    // fills the net exactly, so **exactly one of dw/dh is always 0**. Aspect != 1 is
    // exercised by having different cases put the padding on different axes.
    let cases: [((u32, u32), (f32, f32)); 4] = [
        // (w, h)      (dw, dh)
        ((512, 256), (0.0, 512.0)), // r = 1024/256 = 2      -> unpad 1024x512
        ((1000, 1500), (341.0, 0.0)), // r = 1024/1500         -> unpad  683x1024
        ((300, 200), (0.0, 341.0)), // r = 1024/200 = 5.12   -> unpad 1024x683 (upscale)
        ((1024, 1024), (0.0, 0.0)), // r = 1                 -> no resize, no padding
    ];

    for ((width, height), (dw, dh)) in cases {
        let boxed = letterbox(&solid_rgb(width, height, [0, 0, 0]));

        assert_eq!(
            boxed.image.dimensions(),
            (NET_SIZE, NET_SIZE),
            "{width}x{height}: `auto = false` always yields a square net input"
        );
        assert_eq!(boxed.geometry.net_size, NET_SIZE, "{width}x{height}");
        assert_eq!(
            boxed.geometry.image_size,
            (width, height),
            "{width}x{height}: geometry records the ORIGINAL size, which is what \
             `yolo::rescale` divides by"
        );
        assert_eq!(
            (boxed.geometry.dw, boxed.geometry.dh),
            (dw, dh),
            "{width}x{height}"
        );
    }
}

#[test]
fn letterbox_pads_the_right_edge_and_leaves_the_top_left_as_content() {
    // spec §8.3 step 3: pad **right/bottom** with (114, 114, 114) -- NOT yolov5's centred
    // padding, which halves dw/dh and pads all four sides. The source is split left/right
    // into two flat colours, so every assertion below is exact after a bilinear resize and
    // would fail if any padding were placed on the left or the top.
    let source = RgbImage::from_fn(1000, 1500, |x, _| {
        if x < 500 {
            Rgb([10, 20, 30])
        } else {
            Rgb([40, 50, 60])
        }
    });

    let boxed = letterbox(&source);

    assert_eq!((boxed.geometry.dw, boxed.geometry.dh), (341.0, 0.0));
    let content_width = NET_SIZE - boxed.geometry.dw as u32; // 683
    assert_eq!(
        *boxed.image.get_pixel(0, 0),
        Rgb([10, 20, 30]),
        "the top-left pixel is CONTENT: nothing is padded on the left or the top"
    );
    assert_eq!(
        *boxed.image.get_pixel(0, NET_SIZE - 1),
        Rgb([10, 20, 30]),
        "and the bottom-left too: dh is 0 here"
    );
    assert_eq!(
        *boxed.image.get_pixel(content_width - 1, 0),
        Rgb([40, 50, 60]),
        "the last content column is the source's right half"
    );
    assert_eq!(
        *boxed.image.get_pixel(content_width, 0),
        Rgb([PAD_VALUE; 3]),
        "padding starts exactly at NET_SIZE - dw"
    );
    assert_eq!(
        *boxed.image.get_pixel(NET_SIZE - 1, NET_SIZE - 1),
        Rgb([PAD_VALUE; 3])
    );
    let padded = boxed
        .image
        .pixels()
        .filter(|pixel| pixel.0 == [PAD_VALUE; 3])
        .count();
    assert_eq!(
        padded as u32,
        boxed.geometry.dw as u32 * NET_SIZE,
        "the padded area is exactly dw full-height columns, so no pad pixel leaked into \
         the content region"
    );
}

#[test]
fn letterbox_pads_the_bottom_edge_when_the_page_is_wider_than_tall() {
    // The mirror case: the source is split top/bottom into two flat colours, so the
    // assertions fail if the padding lands on the top instead of the bottom.
    let source = RgbImage::from_fn(512, 256, |_, y| {
        if y < 128 {
            Rgb([1, 2, 3])
        } else {
            Rgb([4, 5, 6])
        }
    });

    let boxed = letterbox(&source);

    assert_eq!((boxed.geometry.dw, boxed.geometry.dh), (0.0, 512.0));
    let content_height = NET_SIZE - boxed.geometry.dh as u32; // 512
    assert_eq!(
        *boxed.image.get_pixel(0, 0),
        Rgb([1, 2, 3]),
        "the top row is CONTENT: nothing is padded above it"
    );
    assert_eq!(
        *boxed.image.get_pixel(0, 400),
        Rgb([4, 5, 6]),
        "the source's bottom half is inside the content band"
    );
    assert_eq!(
        *boxed.image.get_pixel(0, content_height),
        Rgb([PAD_VALUE; 3]),
        "padding starts exactly at NET_SIZE - dh"
    );
    let padded = boxed
        .image
        .pixels()
        .filter(|pixel| pixel.0 == [PAD_VALUE; 3])
        .count();
    assert_eq!(padded as u32, boxed.geometry.dh as u32 * NET_SIZE);
}

#[test]
fn letterbox_of_an_exactly_net_sized_page_is_the_identity() {
    // r = 1 with no padding, so the content must survive bit-for-bit. Resampling here
    // would silently blur every page that is already 1024 on its long axis.
    let source = RgbImage::from_fn(NET_SIZE, NET_SIZE, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });

    let boxed = letterbox(&source);

    assert_eq!((boxed.geometry.dw, boxed.geometry.dh), (0.0, 0.0));
    assert_eq!(boxed.image.as_raw(), source.as_raw());
}

#[test]
fn letterbox_upscales_small_pages_because_upstream_keeps_scaleup() {
    // spec §8.3 step 3 keeps upstream `letterbox`'s `scaleup = True` explicitly, so `r`
    // may exceed 1. If `r` were clamped to 1, this 300x200 page would sit in a corner of
    // the net input and dh would be 824, not 341 -- a difference that changes every
    // detection on every small page.
    let boxed = letterbox(&solid_rgb(300, 200, [10, 20, 30]));

    assert_eq!((boxed.geometry.dw, boxed.geometry.dh), (0.0, 341.0));
    assert_eq!(
        *boxed.image.get_pixel(1023, 682),
        Rgb([10, 20, 30]),
        "content really was enlarged to fill 1024x683 (a uniform input stays exact under \
         bilinear resampling)"
    );
    assert_eq!(*boxed.image.get_pixel(0, 683), Rgb([PAD_VALUE; 3]));
}

// ------------------------------------------------------------ §8.6's round trip

#[test]
fn a_letterbox_round_trip_is_exact_for_a_dyadic_geometry() {
    // spec §8.6: "a synthetic 1024x1024 letterbox round-trip fixture for D4 (exercise
    // dw/dh when aspect != 1)". 512x256 gives r = 2 and dh = 512, so the forward map (x2)
    // and `yolo::rescale`'s inverse (x0.5) are both exact in binary -- no tolerance is
    // needed, and any off-by-one in dw/dh shows up immediately.
    //
    // This is also the exact geometry the FROZEN `d5_yolo.rs::geometry()` helper
    // documents, so D4's output and D5's fixture are locked to each other.
    let boxed = letterbox(&solid_rgb(512, 256, [0, 0, 0]));
    let scale_x = (NET_SIZE as f32 - boxed.geometry.dw) / 512.0; // 2.0
    let scale_y = (NET_SIZE as f32 - boxed.geometry.dh) / 256.0; // 2.0
    let source = Rect::new(100, 40, 300, 200);

    let blocks = yolo::rescale(
        &[Candidate {
            xyxy: [
                source.x1 as f32 * scale_x,
                source.y1 as f32 * scale_y,
                source.x2 as f32 * scale_x,
                source.y2 as f32 * scale_y,
            ],
            class_index: 0,
            score: 0.75,
        }],
        &boxed.geometry,
    );

    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].rect, source);
}

#[test]
fn a_letterbox_round_trip_survives_a_non_dyadic_geometry_within_one_pixel() {
    // The same §8.6 fixture with dw != 0: 1000x1500 -> unpad 683x1024, dw = 341. Neither
    // ratio is binary-exact and §8.3 step 4 truncates (`astype(np.int32)`), so 1 px is the
    // honest bound. A wrong dw would be off by tens of pixels, not one.
    let boxed = letterbox(&solid_rgb(1000, 1500, [0, 0, 0]));
    assert_eq!((boxed.geometry.dw, boxed.geometry.dh), (341.0, 0.0));
    let scale_x = (NET_SIZE as f32 - boxed.geometry.dw) / 1000.0;
    let scale_y = (NET_SIZE as f32 - boxed.geometry.dh) / 1500.0;
    let source = Rect::new(200, 300, 600, 900);

    let blocks = yolo::rescale(
        &[Candidate {
            xyxy: [
                source.x1 as f32 * scale_x,
                source.y1 as f32 * scale_y,
                source.x2 as f32 * scale_x,
                source.y2 as f32 * scale_y,
            ],
            class_index: 0,
            score: 0.75,
        }],
        &boxed.geometry,
    );

    assert_eq!(blocks.len(), 1);
    let rect = blocks[0].rect;
    for (actual, expected, name) in [
        (rect.x1, source.x1, "x1"),
        (rect.y1, source.y1, "y1"),
        (rect.x2, source.x2, "x2"),
        (rect.y2, source.y2, "y2"),
    ] {
        assert!(
            (actual - expected).abs() <= 1,
            "{name}: round trip gave {actual}, expected {expected} +/- 1"
        );
    }
}

// ------------------------------------------------------------ the RGB bilinear resize

#[test]
fn resize_bilinear_rgb_uses_the_same_half_pixel_convention_as_the_mask_resize() {
    // spec §16.6 item 6 pinned `cv2.INTER_LINEAR` for the mask resize
    // (`src = (dst + 0.5) * (src_len / dst_len) - 0.5`, clamped at both borders). The
    // letterbox resize (§8.3 step 3) is the same OpenCV call on three channels, so it must
    // agree channel-for-channel: per channel this is exactly the frozen `d6_mask.rs`
    // `[0, 25, 75, 100]` case.
    let mut source = RgbImage::new(2, 1);
    source.put_pixel(0, 0, Rgb([0, 0, 100]));
    source.put_pixel(1, 0, Rgb([100, 0, 0]));

    let resized = resize_bilinear_rgb(&source, 4, 1);

    assert_eq!(resized.dimensions(), (4, 1));
    assert_eq!(
        resized.pixels().map(|pixel| pixel.0[0]).collect::<Vec<_>>(),
        vec![0, 25, 75, 100]
    );
    assert_eq!(
        resized.pixels().map(|pixel| pixel.0[2]).collect::<Vec<_>>(),
        vec![100, 75, 25, 0]
    );
    assert!(
        resized.pixels().all(|pixel| pixel.0[1] == 0),
        "channels must not bleed into one another"
    );
}

#[test]
fn resize_bilinear_rgb_to_the_same_size_is_the_identity() {
    let source = RgbImage::from_fn(3, 2, |x, y| Rgb([(x * 9) as u8, (y * 7) as u8, 5]));

    assert_eq!(resize_bilinear_rgb(&source, 3, 2).as_raw(), source.as_raw());
}

// ------------------------------------------------------------ NCHW normalisation

#[test]
fn to_nchw_lays_out_planar_rgb_at_the_documented_indices() {
    // spec §8.3 step 3: "Channel order: RGB, NCHW, f32 / 255.0", i.e.
    // `index = c * h * w + y * w + x`. Four distinct pixels with twelve distinct values,
    // so an interleaved (HWC) layout or a BGR swap changes every assertion below.
    let mut image = RgbImage::new(2, 2);
    image.put_pixel(0, 0, Rgb([1, 2, 3]));
    image.put_pixel(1, 0, Rgb([4, 5, 6]));
    image.put_pixel(0, 1, Rgb([7, 8, 9]));
    image.put_pixel(1, 1, Rgb([10, 11, 12]));

    let tensor = to_nchw(&image);

    assert_eq!(tensor.len(), 3 * 2 * 2);
    for (index, byte) in [
        // R plane
        (0, 1),
        (1, 4),
        (2, 7),
        (3, 10),
        // G plane
        (4, 2),
        (5, 5),
        (6, 8),
        (7, 11),
        // B plane
        (8, 3),
        (9, 6),
        (10, 9),
        (11, 12),
    ] {
        assert_normalised(tensor[index], byte);
    }
}

#[test]
fn to_nchw_maps_the_endpoints_exactly() {
    // 0 -> 0.0 and 255 -> 1.0 exactly. This pins division by 255 rather than
    // multiplication by a precomputed 1/255 reciprocal (which yields 0.99999994 for 255),
    // matching upstream's `img / 255`.
    let black = to_nchw(&solid_rgb(1, 1, [0, 0, 0]));
    let white = to_nchw(&solid_rgb(1, 1, [255, 255, 255]));

    assert_eq!(black, vec![0.0, 0.0, 0.0]);
    assert_eq!(white, vec![1.0, 1.0, 1.0]);
}

#[test]
fn to_nchw_divides_by_255_and_not_by_256() {
    // The classic off-by-one that shifts every activation slightly and is invisible in any
    // dimension check. 114/255 = 0.447059 vs 114/256 = 0.445313.
    let tensor = to_nchw(&solid_rgb(1, 1, [PAD_VALUE, PAD_VALUE, PAD_VALUE]));

    assert_normalised(tensor[0], PAD_VALUE);
    assert!(
        (tensor[0] - f32::from(PAD_VALUE) / 256.0).abs() > 1e-3,
        "must divide by 255, not 256: got {}",
        tensor[0]
    );
}

// ------------------------------------------------------------ output binding + swap guard

#[test]
fn bind_outputs_binds_by_index_for_the_shipped_model_layout() {
    // spec §8.3 step 3: "Bind by index". The shipped `comictextdetector.pt.onnx` emits
    // `blk`, then a 1-channel `seg` mask, then a 2-channel `det` lines map -- so the swap
    // guard does NOT fire on the real weights, which is exactly why it needs a unit test
    // rather than being left to be discovered in production.
    assert_eq!(
        bind_outputs(&shipped_outputs()).expect("a well-formed output set"),
        OutputBinding::BY_INDEX
    );
    assert_eq!(
        OutputBinding::BY_INDEX,
        OutputBinding {
            blks: 0,
            mask: 1,
            lines_map: 2
        }
    );
}

#[test]
fn bind_outputs_swaps_a_mask_and_lines_map_that_arrive_in_the_other_order() {
    // spec §8.3 step 3's guard (upstream `inference.py`, ~lines 181-185): if output 1 has
    // 2 channels and output 2 has 1, the two are the other way round. Channel count is
    // dimension 1 of each `[1, C, H, W]` shape.
    let outputs = [
        meta("blk", &[1, 64_512, ROW_STRIDE as i64]),
        meta("det", &[1, 2, 1024, 1024]),
        meta("seg", &[1, 1, 1024, 1024]),
    ];

    let binding = bind_outputs(&outputs).expect("a well-formed output set");

    assert_eq!(
        binding,
        OutputBinding {
            blks: 0,
            mask: 2,
            lines_map: 1
        }
    );
}

#[test]
fn bind_outputs_rejects_a_model_with_the_wrong_number_of_outputs() {
    // Binding by index into a 2-output model would index out of bounds. §5.2 contains
    // panics rather than welcoming them, and §8.3 step 3 wants a model swap to be
    // *diagnosable* -- so this is an error, not a panic.
    let two = [
        meta("blk", &[1, 64_512, ROW_STRIDE as i64]),
        meta("seg", &[1, 1, 1024, 1024]),
    ];
    let four = [
        meta("blk", &[1, 64_512, ROW_STRIDE as i64]),
        meta("seg", &[1, 1, 1024, 1024]),
        meta("det", &[1, 2, 1024, 1024]),
        meta("extra", &[1, 1, 4, 4]),
    ];

    assert!(bind_outputs(&two).is_err());
    assert!(bind_outputs(&four).is_err());
}

#[test]
fn bind_outputs_errors_name_the_observed_outputs_and_shapes() {
    let outputs = [
        meta("actual_blk", &[1, 64_512, 6]),
        meta("actual_mask", &[1, 3, 1024, 1024]),
        meta("actual_lines", &[1, 4, 1024, 1024]),
    ];

    let error = bind_outputs(&outputs).expect_err("the blk column count is invalid");
    let message = error.to_string();

    for observed in [
        "actual_blk",
        "[1, 64512, 6]",
        "actual_mask",
        "[1, 3, 1024, 1024]",
        "actual_lines",
        "[1, 4, 1024, 1024]",
    ] {
        assert!(
            message.contains(observed),
            "{observed:?} missing from {message:?}"
        );
    }
}

#[test]
fn bind_outputs_rejects_a_channel_pair_that_is_neither_arrangement() {
    // The spec is silent on this case. Ratified as an error because the alternative --
    // falling back to index order and feeding a 2-channel lines map into the mask decoder
    // -- produces a plausible-looking wrong mask with no diagnostic at all, the same class
    // of silent failure as the column-count coincidence below.
    let both_two = [
        meta("blk", &[1, 64_512, ROW_STRIDE as i64]),
        meta("det", &[1, 2, 4, 4]),
        meta("det2", &[1, 2, 4, 4]),
    ];
    let both_one = [
        meta("blk", &[1, 64_512, ROW_STRIDE as i64]),
        meta("seg", &[1, 1, 4, 4]),
        meta("seg2", &[1, 1, 4, 4]),
    ];

    assert!(bind_outputs(&both_two).is_err());
    assert!(bind_outputs(&both_one).is_err());
}

#[test]
fn bind_outputs_rejects_a_blk_column_count_that_is_not_the_frozen_row_stride() {
    // THE TRIPWIRE, promoted into the guard that runs on every session construction
    // rather than living only in an `#[ignore]`d smoke test.
    //
    // Why it cannot be inferred from the payload length: divisibility is NOT a validity
    // check. Concretely, before spec §16.15 corrected `N_CLASSES` from 3 to 2, the shipped
    // model's `blk` output `[1, 64512, 7]` gave 64512 * 7 = 451_584 values, which is
    // *exactly* divisible by the then-current stride of 8 (= 8 * 56_448) -- so a 7-column
    // tensor read as 8-column rows would have passed a modulo check by pure coincidence
    // and yielded 56,448 plausible-looking garbage boxes, silently. Only an explicit
    // comparison against `yolo::ROW_STRIDE` catches that, and the message must name both
    // numbers so the diagnosis is immediate.
    let mut outputs = shipped_outputs();
    outputs[0] = meta("blk", &[1, 64_512, ROW_STRIDE as i64 + 1]);

    let error = bind_outputs(&outputs).expect_err("the declared blk arity must be checked");

    let message = error.to_string();
    assert!(
        message.contains(&ROW_STRIDE.to_string()),
        "must name the expected column count: {message}"
    );
    assert!(
        message.contains(&(ROW_STRIDE + 1).to_string()),
        "must name the model's actual column count: {message}"
    );
}

#[test]
fn describe_outputs_names_every_output_and_its_shape() {
    // spec §8.3 step 3: "log the actual output names/shapes once at DEBUG so a model swap
    // is diagnosable". The rendering is what makes that diagnostic possible, so it is part
    // of this module's contract rather than an incidental `tracing` format string -- and
    // it has to survive the case where binding FAILED, which is exactly when it matters.
    let rendered = describe_outputs(&shipped_outputs());

    for name in ["blk", "seg", "det"] {
        assert!(
            rendered.contains(name),
            "must name output `{name}`: {rendered}"
        );
    }
    assert!(
        rendered.contains("64512") || rendered.contains("64_512"),
        "must render the blk shape: {rendered}"
    );
    assert!(
        rendered.contains(&ROW_STRIDE.to_string()),
        "must render the blk column count: {rendered}"
    );
    assert!(
        rendered.contains("1024"),
        "must render the mask/lines shapes: {rendered}"
    );
}

// ------------------------------------------------------------ mask tensor decode

#[test]
fn decode_mask_produces_a_mask_at_base_image_size() {
    // spec §8.2 says `RawDetection.mask` is "8-bit, same size as `image`", so the backend
    // must apply §8.3 step 5's first two bullets (crop the letterbox padding, then
    // INTER_LINEAR resize) itself, before `run()` ever sees the mask. Uniform input, so
    // the resize path cannot make the expectation approximate.
    let geometry = geometry(8, 2.0, 3.0, (3, 2));
    let values = [0.5_f32; 8 * 8];

    let mask = decode_mask(&values, &geometry).expect("a well-sized tensor");

    assert_eq!(mask.dimensions(), (3, 2));
    assert!(
        mask.pixels().all(|pixel| pixel.0[0] == 127),
        "spec §16.6 item 7: 0.5 * 255 = 127.5 TRUNCATES to 127"
    );
}

#[test]
fn decode_mask_crops_the_letterbox_padding_before_resizing() {
    // spec §8.3 step 5: `mask[0..H-dh, 0..W-dw]`. The padded band is set to 1.0 (255) and
    // the content to 0.0, so a single leaked pad sample is visible. `image_size` equals
    // the cropped size here, which makes the resize an identity and isolates the crop.
    let geometry = geometry(8, 2.0, 3.0, (6, 5));
    let mut values = [0.0_f32; 8 * 8];
    for y in 0..8 {
        for x in 0..8 {
            if x >= 6 || y >= 5 {
                values[y * 8 + x] = 1.0;
            }
        }
    }

    let mask = decode_mask(&values, &geometry).expect("a well-sized tensor");

    assert_eq!(mask.dimensions(), (6, 5));
    assert!(
        mask.pixels().all(|pixel| pixel.0[0] == 0),
        "no letterbox padding may survive the crop"
    );
}

#[test]
fn decode_mask_rejects_a_tensor_that_is_not_net_size_squared() {
    // The `[1, 1, H, W]` payload is at the letterbox resolution (H = W = net_size); that
    // is the assumption §8.3 step 5's dw/dh crop silently depends on, so a tensor of any
    // other size is rejected instead of being cropped by the wrong amount.
    let geometry = geometry(8, 0.0, 0.0, (8, 8));

    assert!(decode_mask(&[0.0_f32; 63], &geometry).is_err());
    assert!(decode_mask(&[0.0_f32; 65], &geometry).is_err());
}

// ------------------------------------------------------------ blks tensor decode

#[test]
fn decode_blocks_reads_the_documented_column_meaning() {
    // spec §8.3 step 3/step 4: a `blks` row is `cx, cy, w, h, objectness, cls0..clsN`.
    // Hand-traced: cx=200, cy=160, w=64, h=64 -> xyxy (168, 128, 232, 192) in letterbox
    // space; with net 1024, dh 512 and a 512x256 base image both ratios are 0.5, giving
    // (84, 64, 116, 96). objectness 0.875 * cls0 0.9375 = 0.8203125, round3 -> 0.82.
    // Every number is dyadic, so nothing here can go flaky on rounding.
    let geometry = geometry(1024, 0.0, 512.0, (512, 256));
    let mut row = [0.0_f32; ROW_STRIDE];
    row[0] = 200.0;
    row[1] = 160.0;
    row[2] = 64.0;
    row[3] = 64.0;
    row[4] = 0.875;
    row[5] = 0.9375; // class 0

    let blocks = decode_blocks(&row, ROW_STRIDE, &geometry).expect("a well-formed tensor");

    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].rect, Rect::new(84, 64, 116, 96));
    assert_eq!(blocks[0].class_index, 0);
    assert_eq!(blocks[0].confidence, 0.82);
}

#[test]
fn decode_blocks_delegates_to_the_frozen_yolo_postprocess() {
    // spec §8.3 step 4 is task D5's, already frozen and calibrated. The backend must call
    // it, not grow a second decode path that can drift from it.
    let geometry = geometry(1024, 0.0, 512.0, (512, 256));
    let mut rows = [0.0_f32; ROW_STRIDE * 2];
    rows[0] = 200.0;
    rows[1] = 160.0;
    rows[2] = 64.0;
    rows[3] = 64.0;
    rows[4] = 0.875;
    rows[5] = 0.9375;
    rows[ROW_STRIDE] = 700.0;
    rows[ROW_STRIDE + 1] = 300.0;
    rows[ROW_STRIDE + 2] = 32.0;
    rows[ROW_STRIDE + 3] = 32.0;
    rows[ROW_STRIDE + 4] = 0.75;
    rows[ROW_STRIDE + 6] = 0.875; // class 1

    assert_eq!(
        decode_blocks(&rows, ROW_STRIDE, &geometry).expect("a well-formed tensor"),
        yolo::postprocess(&rows, &geometry)
    );
}

#[test]
fn decode_blocks_rejects_a_column_count_that_is_not_the_frozen_row_stride() {
    // The decode-time half of the tripwire above. `bind_outputs` catches a bad *declared*
    // shape at session construction; this catches a bad column count at the point the
    // values are actually reinterpreted, which is where the silent mis-decode would occur.
    // `ROW_STRIDE * (ROW_STRIDE + 1)` reproduces the divisibility coincidence by
    // construction, for any value of `ROW_STRIDE`.
    let geometry = geometry(1024, 0.0, 512.0, (512, 256));
    const COLUMNS: usize = ROW_STRIDE + 1;
    let payload = [0.0_f32; ROW_STRIDE * COLUMNS];
    assert_eq!(
        payload.len() % ROW_STRIDE,
        0,
        "the fixture must reproduce the divisibility coincidence, or it proves nothing"
    );

    let error = decode_blocks(&payload, COLUMNS, &geometry)
        .expect_err("the column count must be checked explicitly");

    let message = error.to_string();
    assert!(
        message.contains(&ROW_STRIDE.to_string()),
        "must name the expected column count: {message}"
    );
    assert!(
        message.contains(&COLUMNS.to_string()),
        "must name the model's actual column count: {message}"
    );
}

#[test]
fn decode_blocks_rejects_a_payload_that_is_not_a_whole_number_of_rows() {
    // `yolo::filter_candidates` asserts (i.e. panics) on this; §5.2 contains panics but
    // does not want them, so the backend guards before delegating.
    let geometry = geometry(1024, 0.0, 512.0, (512, 256));

    assert!(decode_blocks(&[0.0_f32; ROW_STRIDE + 1], ROW_STRIDE, &geometry).is_err());
}

#[test]
fn decode_blocks_of_an_empty_tensor_is_an_empty_block_list() {
    // A page with no candidates is a normal outcome, not an error (§5.6 -> `Skipped {
    // NoTextDetected }` further up the pipeline).
    let geometry = geometry(1024, 0.0, 512.0, (512, 256));

    assert!(decode_blocks(&[], ROW_STRIDE, &geometry)
        .expect("an empty tensor is well-formed")
        .is_empty());
}

// ------------------------------------------------------------ model file pre-flight

#[test]
fn ensure_model_file_reports_a_missing_file_by_name() {
    // spec §5.3 makes a missing model a FATAL condition, so the message has to name the
    // path the user (or the cache) pointed at. Deliberately ungated and `ort`-free: it is
    // the pre-flight `OnnxDetector::from_path` must run *before* touching the runtime,
    // which is what lets this behaviour be tested with no ONNX Runtime present at all.
    let root = tempfile::TempDir::new().expect("temp dir");
    let missing = root.path().join("comictextdetector.pt.onnx");

    let error = ensure_model_file(&missing).expect_err("no such file");

    assert!(matches!(error, pc_core::StageError::Model(_)));
    assert!(
        error.to_string().contains("comictextdetector.pt.onnx"),
        "must name the path: {error}"
    );
}

#[test]
fn ensure_model_file_rejects_a_directory_and_accepts_a_regular_file() {
    // A cache directory that was created but never populated is a realistic state; it must
    // fail as clearly as an absent path, not as an opaque runtime load error.
    let root = tempfile::TempDir::new().expect("temp dir");
    let directory = root.path().join("models");
    std::fs::create_dir_all(&directory).expect("create the directory");
    let file = root.path().join("model.onnx");
    std::fs::write(&file, b"not a real model, and never loaded by this test")
        .expect("write the file");

    assert!(ensure_model_file(&directory).is_err());
    ensure_model_file(&file).expect("a regular file passes the pre-flight");
}
