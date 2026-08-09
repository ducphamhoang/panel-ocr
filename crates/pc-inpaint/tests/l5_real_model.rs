//! Task **L5** — the pinned `lama-manga.onnx` artifact, run for real.
//!
//! **This file is what resolves §16.38 item 1(c).** That clause is a correction to the L0
//! spike's own report: the graph's declared output shape is
//! `[dim_param "batch", 3, dim_param "batch", dim_param "Sigmoidoutput_dim_3"]`, and its
//! *"Binding consequence for L5: the implementation must validate the **runtime** output
//! shape and must never derive geometry from the declared one."* Item 1(c) does not say what
//! the runtime shape **is**; only running the artifact answers that, and these tests are
//! where the answer is pinned.
//!
//! **Gating, and why it is `#[ignore]` rather than a silent skip.** These need the 207 MB
//! optional artifact on disk (§16.38 item 1(a); `Requirement::Optional` per item 19), so they
//! follow the shape `crates/pc-ocr/tests/p7_session.rs` established for real manga-ocr
//! weights: `#[ignore]`, plus an env-var pointing at the file, plus a `runtime_available`
//! probe. Run them with:
//!
//! ```text
//! PANEL_OCR_LAMA_MODEL=<cache>/lama-manga.onnx \
//!   cargo test -p pc-inpaint --features onnx --test l5_real_model -- --ignored
//! ```
//!
//! **Executed and green during L5** (2026-08-07) against sha256
//! `50a1abae0d73bd46d08eae36c8590cd59ad09029494c9698702b050ef00b0100`, 207,482,644 bytes.
#![cfg(feature = "onnx")]

use image::{Rgb, RgbImage};
use pc_imageops::BinaryMask;
use pc_inpaint::onnx::{runtime_available, OnnxInpainter};
use pc_inpaint::{Inpainter, TILE};

/// `None` means "not runnable here"; the caller prints why and returns.
fn open_pinned_model() -> Option<OnnxInpainter> {
    let Some(path) = std::env::var_os("PANEL_OCR_LAMA_MODEL") else {
        eprintln!(
            "skipping: set PANEL_OCR_LAMA_MODEL to the pinned lama-manga.onnx \
             (`panel-ocr models download --include-optional`)"
        );
        return None;
    };
    if !runtime_available() {
        eprintln!("skipping: no ONNX Runtime shared library could be loaded");
        return None;
    }
    Some(
        OnnxInpainter::from_path(std::path::Path::new(&path))
            .expect("the pinned artifact must open on the CPU execution provider"),
    )
}

/// A recognisable 512x512 page: an x-ramp in red, a y-ramp in green, flat blue.
fn ramp_tile() -> RgbImage {
    RgbImage::from_fn(TILE, TILE, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, 128])
    })
}

/// A 100x100 hole in the middle of the tile.
fn centre_hole() -> BinaryMask {
    BinaryMask::from_fn(TILE, TILE, |x, y| {
        (200..300).contains(&x) && (200..300).contains(&y)
    })
}

/// **§16.38 item 1(c), resolved.** The runtime output shape is asserted against a hard-coded
/// literal `[1, 3, 512, 512]` and a hard-coded sample count `786_432`; neither is read from
/// the session, so this cannot pass by agreeing with whatever the artifact happens to return.
///
/// The value range is asserted as `0.0 ..= 1.0` because the graph ends in a sigmoid — the name
/// `Sigmoidoutput_dim_3` in item 1(c)'s decoded output symbol is the only textual hint, and
/// the `SAMPLE_SCALE = 255.0` in `onnx.rs` depends on it. A model emitting `0..255` directly
/// would make every decoded pixel white, and this is the assertion that would catch it.
#[test]
#[ignore = "opt-in: needs the 207 MB optional lama-manga.onnx and an ONNX Runtime shared library"]
fn the_runtime_output_shape_is_1_3_512_512_and_every_sample_lies_in_the_unit_interval() {
    let Some(inpainter) = open_pinned_model() else {
        return;
    };

    let (shape, values) = inpainter
        .run_raw(&ramp_tile(), &centre_hole())
        .expect("one tile runs");

    assert_eq!(
        shape,
        vec![1_i64, 3, 512, 512],
        "the RUNTIME shape, which item 1(c) left open"
    );
    assert_eq!(values.len(), 786_432, "3 * 512 * 512");
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (0.0..=1.0).contains(&min) && (0.0..=1.0).contains(&max),
        "a sigmoid output must lie in [0, 1]; measured min={min} max={max}"
    );
    assert!(
        max > 0.5,
        "and the output must not be degenerate: measured max={max}"
    );
}

/// §16.38 item 1(b)'s decoded input signature, asserted against the live graph.
///
/// The expected values are the literals item 1(b) states — *"`image`: `elem_type = 1`
/// (float32), dims `[dim_param "batch", 3, 512, 512]`"* and *"`mask`: … `[dim_param "batch",
/// 1, 512, 512]`"* — with `-1` standing for the `batch` `dim_param`, which is how `ort`
/// renders a symbolic axis. Nothing here is copied out of the session first.
///
/// This is also the test that would catch a **swap**: two `f32` tensors of the same spatial
/// size are indistinguishable downstream, so the channel counts paired with the names are the
/// only signal that `image` is the 3-channel one.
#[test]
#[ignore = "opt-in: needs the 207 MB optional lama-manga.onnx and an ONNX Runtime shared library"]
fn the_live_graph_declares_exactly_the_two_separate_inputs_item_1b_decoded() {
    let Some(inpainter) = open_pinned_model() else {
        return;
    };

    assert_eq!(
        inpainter.declared_inputs(),
        vec![
            (
                "image".to_owned(),
                "f32".to_owned(),
                vec![-1_i64, 3, 512, 512]
            ),
            (
                "mask".to_owned(),
                "f32".to_owned(),
                vec![-1_i64, 1, 512, 512]
            ),
        ],
        "two SEPARATE tensors, not one concatenated 4-channel input (§16.38 item 1(b))"
    );
}

/// **§16.38 item 1(c)'s narrowing note, pinned.** That note reports *two* figures — the shape
/// `ort` **declares** after its load-time inference pass, and the shape inference actually
/// **produces** — and calls both "measured by running the model" in this file. The second is
/// asserted by `the_runtime_output_shape_is_1_3_512_512_and_every_sample_lies_in_the_unit_interval`
/// above; this test is the first, so neither figure rests on a one-off probe nobody re-runs.
///
/// The expected value is the literal the note states — *"`session.outputs()` reports the output
/// as `("output", f32, [-1, 3, 512, 512])`"* — hard-coded here, not read from the session.
///
/// **What would turn this red, which is the point of having it:** the note's own reasoning is
/// that an inference pass is a property of the **runtime version**, not of the pinned artifact,
/// so a future `ort` that stops concretising the spatial symbols would report the protobuf's
/// `[-1, 3, "batch", "Sigmoidoutput_dim_3"]` instead and this assertion would fail — which is
/// the signal the note wants, and is why it must not go red silently on the runtime shape alone.
/// Going red here is **not** a defect in the inference path: nothing in this crate reads the
/// declared output shape (item 1(c) forbids it), so the correct response is to re-transcribe the
/// note, not to change `EXPECTED_OUTPUT_SHAPE`.
#[test]
#[ignore = "opt-in: needs the 207 MB optional lama-manga.onnx and an ONNX Runtime shared library"]
fn ort_declares_one_output_named_output_with_a_free_batch_axis_and_concrete_512_spatial_axes() {
    let Some(inpainter) = open_pinned_model() else {
        return;
    };

    assert_eq!(
        inpainter.declared_outputs(),
        vec![(
            "output".to_owned(),
            "f32".to_owned(),
            vec![-1_i64, 3, 512, 512]
        )],
        "§16.38 item 1(c): ONNX Runtime's load-time shape inference concretises BOTH spatial \
         symbols, so the shape an `ort` caller is shown is not the protobuf's fully symbolic one"
    );
}

/// **A MEASURED fact recorded because nothing in §16.38 states it, and getting it wrong is
/// silent: the exported graph zeroes the masked region of `image` ITSELF.**
///
/// LaMa's generator computes `masked_img = img * (1 - mask)` and concatenates the mask to make
/// the 4 channels the sibling weights repo declares (§16.38 item 6(a):
/// `"input_channels": 4`), and this export puts that inside the graph. Measured at L5 by
/// running three tiles that differ **only inside the mask** — original content, black, and a
/// loud magenta — and getting **bit-identical** outputs.
///
/// Two consequences this pins: the caller must **not** pre-zero the hole (that would be dead
/// work), and page content inside the hole cannot leak into the fill. If a future artifact
/// moves the masking out of the graph, this goes red and the caller owes the zeroing.
#[test]
#[ignore = "opt-in: needs the 207 MB optional lama-manga.onnx and an ONNX Runtime shared library"]
fn the_graph_zeroes_the_masked_region_itself_so_hole_content_cannot_reach_the_output() {
    let Some(inpainter) = open_pinned_model() else {
        return;
    };
    let mask = centre_hole();
    let as_is = ramp_tile();

    let mut blacked = as_is.clone();
    let mut loud = as_is.clone();
    for y in 0..TILE {
        for x in 0..TILE {
            if mask.get(x, y) {
                blacked.put_pixel(x, y, Rgb([0, 0, 0]));
                loud.put_pixel(x, y, Rgb([255, 0, 255]));
            }
        }
    }
    // The three inputs really do differ, or the comparison below is vacuous.
    assert_ne!(as_is.get_pixel(250, 250), blacked.get_pixel(250, 250));
    assert_ne!(as_is.get_pixel(250, 250), loud.get_pixel(250, 250));

    let (_, from_as_is) = inpainter.run_raw(&as_is, &mask).expect("run 1");
    let (_, from_blacked) = inpainter.run_raw(&blacked, &mask).expect("run 2");
    let (_, from_loud) = inpainter.run_raw(&loud, &mask).expect("run 3");

    assert_eq!(
        from_as_is, from_blacked,
        "hole content must not reach the output at all"
    );
    assert_eq!(
        from_as_is, from_loud,
        "including a loud, out-of-family fill"
    );
}

/// §16.38 item 1(g): *"With an all-zero mask (nothing to fill) only **1 of 786,432** output
/// pixels matched the input exactly — the model regenerates the whole tile. … So blending the
/// unmasked region back from the original is a **hard requirement**, not an optimisation."*
///
/// Asserted as a hard-coded ceiling of 10% of the tile's 262,144 pixels rather than as item
/// 1(g)'s exact `1`, because that figure was a per-sample count on one machine and the
/// property this test protects is *"regenerates the whole tile"*, not that one number.
///
/// The complementary half — that the regeneration is nonetheless **close** to the input — is
/// what pins the `/255` normalisation, and it is asserted as a mean absolute error bound. L5
/// measured 0.0082 in `0..1` units (about 2.1/255); the bound of 8/255 leaves headroom while
/// still failing loudly if the model were fed unnormalised `0..255` samples.
#[test]
#[ignore = "opt-in: needs the 207 MB optional lama-manga.onnx and an ONNX Runtime shared library"]
fn a_zero_mask_still_regenerates_the_tile_but_stays_close_to_the_input_which_pins_the_scaling() {
    let Some(inpainter) = open_pinned_model() else {
        return;
    };
    let tile = ramp_tile();
    let empty = BinaryMask::new(TILE, TILE);
    assert_eq!(empty.count_set(), 0, "nothing to fill");

    let filled = inpainter
        .inpaint_tile(&tile, &empty)
        .expect("a zero mask still runs — upstream's own guard is at the page level");
    assert_eq!(filled.dimensions(), (512, 512));

    let mut identical = 0_usize;
    let mut absolute_error = 0_u64;
    for y in 0..TILE {
        for x in 0..TILE {
            let before = *tile.get_pixel(x, y);
            let after = *filled.get_pixel(x, y);
            if before == after {
                identical += 1;
            }
            for channel in 0..3 {
                absolute_error += u64::from(before.0[channel].abs_diff(after.0[channel]));
            }
        }
    }

    assert!(
        identical < 26_214,
        "the model regenerates the tile, so at most a small fraction of 262144 pixels may \
         come back bit-identical; {identical} did"
    );
    let mean_error = absolute_error as f64 / (3.0 * 262_144.0);
    assert!(
        mean_error < 8.0,
        "mean absolute channel error was {mean_error}/255; a value this large means the \
         image input is not being divided by 255 (or the output is not being multiplied by \
         it)"
    );
}

/// The whole trait contract, end to end on the real artifact: a 512x512 RGB tile out, the
/// masked region visibly changed, and the unmasked region still recognisably the page.
///
/// The "changed" assertion uses a hole placed over a **flat** area whose input value is
/// known, so the expectation is a property of the input and not of the output.
#[test]
#[ignore = "opt-in: needs the 207 MB optional lama-manga.onnx and an ONNX Runtime shared library"]
fn inpaint_tile_returns_a_512_square_rgb_tile_whose_masked_region_no_longer_matches_the_page() {
    let Some(inpainter) = open_pinned_model() else {
        return;
    };
    // A flat white page with a solid black blob — the shape this stage exists to remove.
    let mut tile = RgbImage::from_pixel(TILE, TILE, Rgb([255, 255, 255]));
    for y in 220..300 {
        for x in 200..320 {
            tile.put_pixel(x, y, Rgb([0, 0, 0]));
        }
    }
    let mask = BinaryMask::from_fn(TILE, TILE, |x, y| {
        (195..325).contains(&x) && (215..305).contains(&y)
    });

    let filled = inpainter.inpaint_tile(&tile, &mask).expect("one tile runs");

    assert_eq!(filled.dimensions(), (512, 512));
    // The blob's centre was pure black; the fill must have moved it toward the surrounding
    // white. The threshold is a hand-picked literal, not a measurement of the output.
    let centre = *filled.get_pixel(260, 260);
    assert!(
        centre.0.iter().all(|channel| *channel > 128),
        "the black blob's centre must be filled from the white surroundings, got {centre:?}"
    );
    // And a far-away unmasked pixel must still be white-ish, so the whole tile was not
    // replaced by mush.
    let corner = *filled.get_pixel(20, 20);
    assert!(
        corner.0.iter().all(|channel| *channel > 200),
        "an unmasked corner of a white page must stay white-ish, got {corner:?}"
    );
}
