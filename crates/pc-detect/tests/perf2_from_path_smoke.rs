#![cfg(feature = "onnx")]

use pc_detect::onnx::OnnxDetector;
use pc_detect::{calculate_new_size_and_scale, resize_area, TextDetector};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[test]
#[ignore = "opt-in: needs PANEL_OCR_ONNX_MODEL and ONNX Runtime"]
fn normal_from_path_detect_uses_the_shipped_flush_default() {
    let model = PathBuf::from(
        std::env::var_os("PANEL_OCR_ONNX_MODEL").expect("PANEL_OCR_ONNX_MODEL is required"),
    );
    let page =
        pc_testkit::paths::recorded("detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg");
    let decoded = image::open(page)
        .expect("recorded page must decode")
        .to_rgb8();
    let (width, height, _) =
        calculate_new_size_and_scale(decoded.width(), decoded.height(), 1000, 4000);
    let base = resize_area(&decoded, width, height);
    let detector = OnnxDetector::from_path(&model).expect("real ONNX session must open");

    let started = Instant::now();
    TextDetector::detect(&detector, &base).expect("normal detector path must succeed");
    let elapsed = started.elapsed();
    println!("from_path_default_detect_elapsed={elapsed:?}");
    assert!(
        elapsed < Duration::from_secs(5),
        "normal from_path detect took {elapsed:?}; shipped flush default may not be active"
    );
}
