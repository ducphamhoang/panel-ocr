//! Task B3 (§16.51) — end-to-end: batching the beams must not change a single decoded
//! character. FROZEN per CLAUDE.md.
//!
//! THE EXPECTED STRING BELOW IS AN UPSTREAM VALUE, NOT OURS. Produced 2026-08-16 by
//! running real `manga_ocr` (`transformers` 5.14.1, `torch` 2.13.0+cpu, model
//! `kha-white/manga-ocr-base`) on this exact committed fixture, per cookbook rule 3's
//! "consult upstream by RUNNING it" and rule 7's "the break in the circle must come from
//! outside the code under test":
//!     OCR OUTPUT: それがゴブタを隊長に選抜した理由ですー
//! This project's own pre-batching `MangaOcrEngine::recognize` independently produced
//! the identical string on the same fixture in the same session, so this literal is
//! corroborated by two implementations that share no code.
#![cfg(feature = "onnx")]

use pc_ocr::OcrEngine;

/// Upstream `manga_ocr`'s output for `nightmare_bubble_raw.png`. See the module header
/// for how it was obtained.
const UPSTREAM_NIGHTMARE_BUBBLE_TEXT: &str = "それがゴブタを隊長に選抜した理由ですー";

#[test]
#[ignore = "opt-in: needs BOTH real manga-ocr weights and an ONNX Runtime shared library"]
fn the_batched_engine_reproduces_upstreams_own_decoded_text() {
    let (Some(encoder), Some(decoder)) = (
        std::env::var_os("PANEL_OCR_MANGA_OCR_ENCODER"),
        std::env::var_os("PANEL_OCR_MANGA_OCR_DECODER"),
    ) else {
        eprintln!("skipping: set PANEL_OCR_MANGA_OCR_ENCODER and PANEL_OCR_MANGA_OCR_DECODER");
        return;
    };
    if !pc_ocr::onnx::runtime_available() {
        eprintln!("skipping: no ONNX Runtime shared library could be loaded");
        return;
    }
    let engine = pc_ocr::MangaOcrEngine::from_paths(
        &std::path::PathBuf::from(encoder),
        &std::path::PathBuf::from(decoder),
    )
    .expect("the pinned real engine opens");

    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/upstream/demo_bubbles/nightmare_bubble_raw.png");
    let crop = image::open(&fixture).expect("the committed fixture is readable");

    let text = engine.recognize(&crop).expect("recognizes");

    // Exact string equality against an upstream-produced literal. A batched decoder that
    // sliced the wrong row would still return plausible Japanese; only an exact
    // comparison against an independently-produced expectation catches that.
    assert_eq!(text, UPSTREAM_NIGHTMARE_BUBBLE_TEXT);
    // Anti-vacuity: a `recognize` that returned the empty string, or that returned its
    // input unchanged, would be caught here rather than by the comparison above alone.
    assert_eq!(
        text.chars().count(),
        UPSTREAM_NIGHTMARE_BUBBLE_TEXT.chars().count()
    );
    assert!(!text.is_empty());
}
