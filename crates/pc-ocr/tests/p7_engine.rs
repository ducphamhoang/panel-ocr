//! Task P7e — `OcrEngine`/`OcrEngineFactory` wiring (spec §9.2, §15.4 item 4). FROZEN per
//! CLAUDE.md.
//!
//! `#![cfg(feature = "onnx")]`: this crate's OCR engine only exists behind the `onnx`
//! feature. Unlike `p7_session.rs`'s documented "cannot run here" situation, `ort` DOES
//! link and run in this environment (measured during P7c) -- the non-ignored tests below
//! are expected to actually execute, not merely compile.
#![cfg(feature = "onnx")]

use pc_core::Language;
use pc_ocr::OcrEngineFactory;

#[test]
fn from_paths_reports_a_missing_encoder_run_fatally_before_any_recognize_call() {
    // Construction failures are run-fatal and image-independent -- this is just
    // MangaOcrSessions::from_paths's existing contract (P7c), re-asserted at the
    // MangaOcrEngine level so a future refactor of the wrapper can't silently drop it.
    let root = tempfile::TempDir::new().expect("temp dir");
    let encoder = root.path().join("encoder_model.onnx");
    let decoder = root.path().join("decoder_model.onnx");
    std::fs::write(&decoder, b"not an ONNX graph").expect("garbage decoder");

    let error =
        pc_ocr::manga::MangaOcrEngine::from_paths(&encoder, &decoder).expect_err("missing encoder");

    assert!(error.to_string().contains(&encoder.display().to_string()));
}

#[test]
#[ignore = "opt-in: needs BOTH real manga-ocr weights (PANEL_OCR_MANGA_OCR_ENCODER / \
            PANEL_OCR_MANGA_OCR_DECODER) and an ONNX Runtime shared library. Run with \
            `cargo test -p pc-ocr --features onnx --test p7_engine -- --ignored --nocapture`"]
fn the_factory_serves_every_language_including_unknown_and_recognize_returns_non_empty_text() {
    let (Some(encoder_path), Some(decoder_path)) = (
        std::env::var_os("PANEL_OCR_MANGA_OCR_ENCODER"),
        std::env::var_os("PANEL_OCR_MANGA_OCR_DECODER"),
    ) else {
        eprintln!("skipping: set both PANEL_OCR_MANGA_OCR_ENCODER and PANEL_OCR_MANGA_OCR_DECODER");
        return;
    };
    if !pc_ocr::onnx::runtime_available() {
        eprintln!("skipping: no ONNX Runtime shared library could be loaded");
        return;
    }

    let engine = pc_ocr::manga::MangaOcrEngine::from_paths(
        std::path::Path::new(&encoder_path),
        std::path::Path::new(&decoder_path),
    )
    .expect("the real sessions open");
    let factory = pc_ocr::manga::MangaOcrFactory::new(engine);

    // spec §15.4 item 4: manga-ocr serves EVERY language, including an unknown one -- a
    // factory that special-cased `None` to return `None` was explicitly rejected during
    // P7 planning, since it would silently disable OCR on every box a detector could not
    // classify.
    assert!(factory.engine_for(Some(Language::Japanese)).is_some());
    assert!(factory.engine_for(Some(Language::English)).is_some());
    assert!(factory.engine_for(None).is_some());

    let engine = factory.engine_for(None).expect("served");
    assert_eq!(engine.languages(), &[Language::Japanese, Language::English]);

    let crop = image::open(pc_testkit::paths::upstream(
        "demo_bubbles/handwritten_bubble_raw.png",
    ))
    .expect("committed fixture");
    let text = engine.recognize(&crop).expect("real inference succeeds");

    // No published label exists for this specific fixture (cookbook rule 7: do not
    // assert exact text derived from our own run as if it were an oracle). What IS
    // assertable: post_process's own guarantee that its output contains no whitespace,
    // and that recognition produced *something* rather than an empty string.
    assert!(!text.is_empty(), "recognize produced empty text");
    assert!(
        !text.chars().any(char::is_whitespace),
        "post_process strips all whitespace; found some in {text:?}"
    );
    eprintln!("recognized (no oracle, informational only): {text:?}");
}
