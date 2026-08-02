//! Task P7c Phase 3 — manga-ocr ONNX session construction and live metadata cross-check.
//!
//! This cannot run in this checkout (the workspace pins `ort` with `default-features = false`,
//! so no ONNX Runtime shared library is linkable here), the same documented situation as
//! `crates/pc-detect/tests/d4_session.rs`.
#![cfg(feature = "onnx")]

use pc_core::StageError;
use pc_ocr::onnx::{runtime_available, MangaOcrSessions, TensorMeta};
use pc_testkit::{model_signature::Dim, ocr_model_signature::manga_ocr_encoder_signature};
use tempfile::TempDir;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn manga_ocr_sessions_are_send_and_sync() {
    // Compile-time only: no weights and no ONNX Runtime session construction.
    assert_send_sync::<MangaOcrSessions>();
}

#[test]
fn missing_encoder_is_reported_by_its_own_path_before_decoder_loading() {
    let root = TempDir::new().expect("temp dir");
    let encoder = root.path().join("encoder_model.onnx");
    let decoder = root.path().join("decoder_model.onnx");
    std::fs::write(&decoder, b"not an ONNX graph").expect("garbage decoder");

    let error = MangaOcrSessions::from_paths(&encoder, &decoder).expect_err("missing encoder");

    assert!(matches!(error, StageError::Model(_)));
    assert!(error.to_string().contains(&encoder.display().to_string()));
    assert!(!error.to_string().contains(&decoder.display().to_string()));
}

#[test]
fn missing_decoder_is_reported_by_its_own_path() {
    let root = TempDir::new().expect("temp dir");
    let encoder = root.path().join("encoder_model.onnx");
    let decoder = root.path().join("decoder_model.onnx");
    std::fs::write(&encoder, b"not an ONNX graph").expect("garbage encoder");

    let error = MangaOcrSessions::from_paths(&encoder, &decoder).expect_err("missing decoder");

    assert!(matches!(error, StageError::Model(_)));
    assert!(error.to_string().contains(&decoder.display().to_string()));
    assert!(!error.to_string().contains(&encoder.display().to_string()));
}

#[test]
fn both_missing_paths_fail_on_the_encoder_preflight() {
    let root = TempDir::new().expect("temp dir");
    let encoder = root.path().join("encoder_model.onnx");
    let decoder = root.path().join("decoder_model.onnx");

    let error = MangaOcrSessions::from_paths(&encoder, &decoder).expect_err("missing files");

    assert!(matches!(error, StageError::Model(_)));
    assert!(error.to_string().contains(&encoder.display().to_string()));
    assert!(!error.to_string().contains("ONNX Runtime"));
}

#[test]
#[ignore = "opt-in: needs BOTH real manga-ocr weights and an ONNX Runtime shared library"]
fn manga_ocr_real_sessions_match_recorded_signatures_without_running_inference() {
    let (Some(encoder_path), Some(decoder_path)) = (
        std::env::var_os("PANEL_OCR_MANGA_OCR_ENCODER"),
        std::env::var_os("PANEL_OCR_MANGA_OCR_DECODER"),
    ) else {
        eprintln!("skipping: set both PANEL_OCR_MANGA_OCR_ENCODER and PANEL_OCR_MANGA_OCR_DECODER to the pinned manga-ocr weights");
        return;
    };
    if !runtime_available() {
        eprintln!("skipping: no ONNX Runtime shared library could be loaded");
        return;
    }

    let encoder_path = std::path::PathBuf::from(encoder_path);
    let decoder_path = std::path::PathBuf::from(decoder_path);
    let sessions = MangaOcrSessions::from_paths(&encoder_path, &decoder_path)
        .expect("the pinned real sessions open");
    let encoder_recorded = manga_ocr_encoder_signature();

    assert_live_matches_recorded(
        sessions.encoder_inputs(),
        &encoder_recorded.inputs,
        "encoder inputs",
    );
    assert_live_matches_recorded(
        sessions.encoder_outputs(),
        &encoder_recorded.outputs,
        "encoder outputs",
    );

    let decoder_recorded = pc_testkit::ocr_model_signature::manga_ocr_decoder_signature();
    assert_live_matches_recorded(
        sessions.decoder_inputs(),
        &decoder_recorded.inputs,
        "decoder inputs",
    );
    assert_live_matches_recorded(
        sessions.decoder_outputs(),
        &decoder_recorded.outputs,
        "decoder outputs",
    );

    let live_logits_width = sessions
        .decoder_outputs()
        .iter()
        .find(|meta| meta.name == "logits")
        .and_then(|meta| meta.shape.last().copied())
        .expect("live decoder has logits");
    assert_eq!(live_logits_width, pc_ocr::vocab::VOCAB_LEN as i64);
}

fn assert_live_matches_recorded(
    live: &[TensorMeta],
    recorded: &[pc_testkit::model_signature::SymbolicTensorSignature],
    label: &str,
) {
    assert_eq!(
        live.len(),
        recorded.len(),
        "{label}: metadata count differs"
    );
    for (actual, expected) in live.iter().zip(recorded) {
        assert_eq!(actual.name, expected.name, "{label}: tensor name differs");
        assert_eq!(
            actual.element_type, expected.element_type,
            "{label}: dtype differs"
        );
        assert_eq!(
            actual.shape.len(),
            expected.shape.len(),
            "{label}: rank differs"
        );
        assert_eq!(
            actual.symbols.len(),
            actual.shape.len(),
            "{label}: symbols are not parallel"
        );

        for (index, (dimension, recorded_dimension)) in
            actual.shape.iter().zip(&expected.shape).enumerate()
        {
            match recorded_dimension {
                Dim::Fixed(value) => assert_eq!(
                    (*dimension, actual.symbols[index].as_str()),
                    (*value, ""),
                    "{label}: concrete dimension {index} differs"
                ),
                Dim::Symbolic(symbol) => assert_eq!(
                    (*dimension, actual.symbols[index].as_str()),
                    (-1, symbol.as_str()),
                    "{label}: symbolic dimension {index} differs"
                ),
            }
        }
    }
}
