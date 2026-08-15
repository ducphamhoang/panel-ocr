//! Task B2 (§16.51) — `MangaOcrSessions::decode_step_batch` against the REAL pinned
//! decoder. FROZEN per CLAUDE.md.
//!
//! `#[ignore]`d for the same reason `p7_session.rs`'s real-weights test is: it needs
//! both the pinned weights on disk and an ONNX Runtime shared library. Opt in with
//!   PANEL_OCR_MANGA_OCR_ENCODER=... PANEL_OCR_MANGA_OCR_DECODER=... \
//!     cargo test -p pc-ocr --features onnx --test p7f_decode_step_batch -- --ignored
//!
//! MEASURED PROVENANCE (2026-08-16, decoder sha256 ef7765..f5e8, CPU provider), from a
//! throwaway spike run BEFORE this file was written, so nothing here is read back off
//! the implementation it gates:
//!     batch=1: OK  logits shape=[1, 5, 6144]
//!     batch=2: OK  logits shape=[2, 5, 6144]
//!     batch=4: OK  logits shape=[4, 5, 6144]
//!     row 0..3: max |batched - single| = 0e0   (BIT-IDENTICAL, hence no tolerance below)
//!     MIN cross-row diff = 3.4635007e-1        (rows are NOT a broadcast of row 0)
#![cfg(feature = "onnx")]

use pc_core::StageError;
use pc_ocr::onnx::{runtime_available, MangaOcrSessions};

/// The four equal-length prefixes used below. Equal length is not incidental: beam
/// search advances every beam in lockstep, so a batched decoder never needs padding.
const PREFIXES: [[u32; 5]; 4] = [
    [2, 101, 202, 303, 404],
    [2, 111, 212, 313, 414],
    [2, 121, 222, 323, 424],
    [2, 131, 232, 333, 434],
];

fn sessions() -> Option<MangaOcrSessions> {
    let (Some(encoder), Some(decoder)) = (
        std::env::var_os("PANEL_OCR_MANGA_OCR_ENCODER"),
        std::env::var_os("PANEL_OCR_MANGA_OCR_DECODER"),
    ) else {
        eprintln!("skipping: set PANEL_OCR_MANGA_OCR_ENCODER and PANEL_OCR_MANGA_OCR_DECODER");
        return None;
    };
    if !runtime_available() {
        eprintln!("skipping: no ONNX Runtime shared library could be loaded");
        return None;
    }
    Some(
        MangaOcrSessions::from_paths(
            &std::path::PathBuf::from(encoder),
            &std::path::PathBuf::from(decoder),
        )
        .expect("the pinned real sessions open"),
    )
}

fn encoder_output(sessions: &MangaOcrSessions) -> pc_ocr::onnx::EncoderOutput {
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(300, 120, |x, y| {
        image::Rgb([((x * 7 + y * 3) % 256) as u8; 3])
    }));
    sessions
        .encode(&pc_ocr::preprocess::pixel_values(&image))
        .expect("encodes")
}

#[test]
#[ignore = "opt-in: needs BOTH real manga-ocr weights and an ONNX Runtime shared library"]
fn batched_row_i_is_bit_identical_to_the_single_prefix_decode_step_for_prefix_i() {
    // THE load-bearing test of task B2. If row slicing is off by one stride, this test
    // is the only thing standing between a wrong beam's logits and a silently drifted
    // decode. Exact equality, not a tolerance: 0e0 was MEASURED on the pinned artifact
    // (see header), and cookbook rule 7 prefers an exact identity where one holds.
    let Some(sessions) = sessions() else { return };
    let encoder = encoder_output(&sessions);

    let borrowed: Vec<&[u32]> = PREFIXES.iter().map(|p| &p[..]).collect();
    let batched = sessions
        .decode_step_batch(&borrowed, &encoder)
        .expect("batched decode step");

    assert_eq!(batched.len(), 4, "one row per prefix");
    for (index, prefix) in PREFIXES.iter().enumerate() {
        let single = sessions
            .decode_step(prefix, &encoder)
            .expect("single decode step");
        assert_eq!(
            batched[index], single,
            "batched row {index} differs from the batch-1 run for the same prefix"
        );
    }

    // ANTI-VACUITY, and it is not optional. If the graph broadcast row 0 across the
    // batch, every assertion above would still pass. The measured minimum adjacent-row
    // separation was 3.46e-1; 1e-2 is an order of magnitude below that, so this
    // threshold cannot fail on noise and cannot pass on a broadcast.
    for index in 1..batched.len() {
        let separation = batched[index - 1]
            .iter()
            .zip(&batched[index])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            separation > 1e-2,
            "rows {} and {index} are indistinguishable (max diff {separation:e}); the \
             equality assertions above would pass vacuously on a broadcast",
            index - 1
        );
    }
}

#[test]
#[ignore = "opt-in: needs BOTH real manga-ocr weights and an ONNX Runtime shared library"]
fn every_batched_row_has_the_recorded_vocabulary_width() {
    // The expected width comes from the RECORDED signature fixture, an oracle
    // independent of the decoder call under test, not from a constant in our own crate.
    let Some(sessions) = sessions() else { return };
    let encoder = encoder_output(&sessions);
    let recorded_width = pc_testkit::ocr_model_signature::manga_ocr_decoder_signature()
        .outputs
        .iter()
        .find(|meta| meta.name == "logits")
        .and_then(|meta| meta.shape.last())
        .and_then(pc_testkit::model_signature::Dim::fixed)
        .and_then(|value| usize::try_from(value).ok())
        .expect("the recorded decoder signature pins a fixed logits width");
    // Anti-vacuity literal, so an emptied or mis-parsed fixture cannot make this
    // tautological: the recorded fixture says 6144.
    assert_eq!(recorded_width, 6144);

    let borrowed: Vec<&[u32]> = PREFIXES.iter().map(|p| &p[..]).collect();
    let batched = sessions
        .decode_step_batch(&borrowed, &encoder)
        .expect("batched decode step");

    assert_eq!(batched.len(), 4);
    for (index, row) in batched.iter().enumerate() {
        assert_eq!(row.len(), recorded_width, "row {index} has the wrong width");
    }
}

#[test]
#[ignore = "opt-in: needs BOTH real manga-ocr weights and an ONNX Runtime shared library"]
fn prefixes_of_unequal_length_are_rejected_before_any_inference() {
    // The ONNX graph takes a rectangular `[batch, seq]` tensor, so ragged prefixes have
    // no correct encoding. Refusing is the only safe answer; padding would silently
    // feed the decoder tokens no beam ever generated.
    let Some(sessions) = sessions() else { return };
    let encoder = encoder_output(&sessions);
    let short: [u32; 3] = [2, 101, 202];
    let ragged: Vec<&[u32]> = vec![&PREFIXES[0][..], &short[..]];

    let error = sessions
        .decode_step_batch(&ragged, &encoder)
        .expect_err("ragged prefixes are rejected");

    assert!(matches!(error, StageError::InvalidInput(_)));
    let message = error.to_string();
    assert!(
        message.contains('5') && message.contains('3'),
        "the message must name both lengths that disagree, got: {message}"
    );
}

#[test]
#[ignore = "opt-in: needs BOTH real manga-ocr weights and an ONNX Runtime shared library"]
fn an_empty_prefix_list_is_rejected_rather_than_producing_a_zero_batch_tensor() {
    let Some(sessions) = sessions() else { return };
    let encoder = encoder_output(&sessions);

    let error = sessions
        .decode_step_batch(&[], &encoder)
        .expect_err("an empty batch is rejected");

    assert!(matches!(error, StageError::InvalidInput(_)));
}
