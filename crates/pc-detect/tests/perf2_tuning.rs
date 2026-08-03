#![cfg(feature = "onnx")]

use pc_config::TextDetectorConfig;
use pc_detect::onnx::{decode_outputs, OnnxDetector, SessionTuning};
use pc_detect::{calculate_new_size_and_scale, resize_area, TextDetector};
use std::path::PathBuf;

const STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

#[test]
fn session_tuning_default_is_the_shipped_flush_session() {
    let tuning = SessionTuning::default();
    assert!(tuning.flush_denormals);
    assert!(!tuning.parallel_execution);
    assert_eq!(tuning.intra_op_spinning, None);
    assert_eq!(tuning.dynamic_block_base, None);
}

#[test]
fn the_tuning_knobs_are_not_exposed_through_text_detector_config() {
    let rendered = toml_edit::ser::to_document(&TextDetectorConfig::default())
        .expect("TextDetectorConfig must serialize as TOML")
        .to_string();
    for forbidden in [
        "flush_denormals",
        "parallel_execution",
        "intra_op_spinning",
        "dynamic_block_base",
    ] {
        assert!(
            !rendered.contains(forbidden),
            "investigation-only tuning knob leaked into config TOML: {forbidden}\n{rendered}"
        );
    }
    assert!(
        rendered.contains("intra_threads"),
        "serialization did not render the pinned config fields: {rendered}"
    );
}

#[test]
#[ignore = "opt-in: set PANEL_OCR_ONNX_MODEL and run `PANEL_OCR_ONNX_MODEL=/path/comictextdetector.pt.onnx cargo test -p pc-detect --features onnx,testkit --test perf2_tuning -- --ignored --nocapture`"]
fn detect_raw_tensors_decodes_to_exactly_what_detect_returns() {
    let model = PathBuf::from(
        std::env::var_os("PANEL_OCR_ONNX_MODEL").expect("PANEL_OCR_ONNX_MODEL is required"),
    );
    let page = pc_testkit::paths::recorded(format!("detector/{STEM}.jpg"));
    let decoded = image::open(&page)
        .expect("recorded page must decode")
        .to_rgb8();
    let (width, height, _) =
        calculate_new_size_and_scale(decoded.width(), decoded.height(), 1000, 4000);
    let base = resize_area(&decoded, width, height);
    let detector = OnnxDetector::from_path_with_tuning(&model, 0, 0, &SessionTuning::default())
        .expect("real ONNX session must open");

    let (metas, values, geometry) = detector
        .detect_raw_tensors(&base)
        .expect("raw detector tensors must be available");
    let decoded_raw = decode_outputs(&metas, &values, &geometry).expect("raw tensors decode");
    let ordinary = TextDetector::detect(&detector, &base).expect("ordinary detector path");
    assert_eq!(decoded_raw.blocks, ordinary.blocks);
    assert_eq!(decoded_raw.mask, ordinary.mask);
}
