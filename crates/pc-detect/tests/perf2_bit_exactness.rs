#![cfg(feature = "onnx")]

use image::GrayImage;
use pc_core::Rect;
use pc_detect::onnx::{OnnxDetector, SessionTuning};
use pc_detect::{calculate_new_size_and_scale, resize_area, RawBlock, TextDetector};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";
const EXPECTED_RAW_FLOATS: usize = 3_597_312;

fn expected_blocks() -> Vec<RawBlock> {
    vec![
        RawBlock {
            rect: Rect::new(674, 1397, 740, 1438),
            class_index: 0,
            confidence: 0.713,
        },
        RawBlock {
            rect: Rect::new(567, 74, 663, 123),
            class_index: 0,
            confidence: 0.704,
        },
        RawBlock {
            rect: Rect::new(607, 631, 724, 703),
            class_index: 1,
            confidence: 0.643,
        },
        RawBlock {
            rect: Rect::new(438, 1407, 498, 1446),
            class_index: 0,
            confidence: 0.627,
        },
    ]
}

fn model() -> PathBuf {
    PathBuf::from(
        std::env::var_os("PANEL_OCR_ONNX_MODEL").expect("PANEL_OCR_ONNX_MODEL is required"),
    )
}

fn base_image() -> image::RgbImage {
    let page = pc_testkit::paths::recorded(format!("detector/{STEM}.jpg"));
    let decoded = image::open(page)
        .expect("recorded page must decode")
        .to_rgb8();
    let (width, height, _) =
        calculate_new_size_and_scale(decoded.width(), decoded.height(), 1000, 4000);
    resize_area(&decoded, width, height)
}

fn digest(values: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    for output in values {
        for value in output {
            hasher.update(value.to_ne_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn committed_mask() -> GrayImage {
    image::open(pc_testkit::paths::recorded(format!(
        "detector/{STEM}_detector_mask.png"
    )))
    .expect("committed detector mask must decode")
    .to_luma8()
}

fn run_level_a(intra_threads: usize, tuning: &SessionTuning, label: &str) -> String {
    let detector = OnnxDetector::from_path_with_tuning(&model(), intra_threads, 0, tuning)
        .expect("real ONNX session must open");
    let base = base_image();
    let (metas, values, geometry) = detector
        .detect_raw_tensors(&base)
        .expect("raw detector tensors must be available");
    let count: usize = metas
        .iter()
        .zip(&values)
        .map(|(meta, output)| {
            let shape_count: usize = meta
                .shape
                .iter()
                .map(|dimension| *dimension as usize)
                .product();
            assert_eq!(
                shape_count,
                output.len(),
                "output shape/value count mismatch"
            );
            shape_count
        })
        .sum();
    assert_eq!(
        count, EXPECTED_RAW_FLOATS,
        "recorded output shape total changed"
    );

    let decoded = pc_detect::onnx::decode_outputs(&metas, &values, &geometry)
        .expect("raw outputs must decode");
    assert_eq!(decoded.blocks, expected_blocks(), "decoded blocks changed");
    let json_path = pc_testkit::paths::recorded(format!("detector/{STEM}_detector_blocks.json"));
    let json_blocks: Vec<RawBlock> = serde_json::from_slice(
        &std::fs::read(&json_path).expect("committed detector JSON must be readable"),
    )
    .expect("committed detector JSON must parse");
    assert_eq!(
        json_blocks,
        expected_blocks(),
        "committed detector JSON moved"
    );
    assert_eq!(
        decoded.mask,
        committed_mask(),
        "committed detector mask moved"
    );

    let ordinary = TextDetector::detect(&detector, &base).expect("ordinary detector path");
    assert_eq!(ordinary.blocks, expected_blocks());
    assert_eq!(ordinary.mask, committed_mask());
    let digest = digest(&values);
    println!("{label} intra_threads={intra_threads} raw_f32_count={count} sha256={digest}");
    digest
}

fn run_level_a_default(intra_threads: usize) -> String {
    let flush_off = SessionTuning {
        flush_denormals: false,
        ..SessionTuning::default()
    };
    run_level_a(intra_threads, &flush_off, "flush_denormals=off")
}

#[test]
#[ignore = "opt-in: needs PANEL_OCR_ONNX_MODEL and ONNX Runtime; run `PANEL_OCR_ONNX_MODEL=/home/ducph/.cache/panel-ocr/models/comictextdetector.pt.onnx cargo test -p pc-detect --features onnx,testkit --test perf2_bit_exactness -- --ignored --nocapture`"]
fn the_real_detector_reproduces_the_committed_fixture_at_every_thread_count() {
    let flush_on = SessionTuning {
        flush_denormals: true,
        ..SessionTuning::default()
    };
    for (label, tuning) in [
        (
            "flush_denormals=off",
            SessionTuning {
                flush_denormals: false,
                ..SessionTuning::default()
            },
        ),
        ("flush_denormals=on", flush_on),
    ] {
        let digests: Vec<_> = [0, 1, 8]
            .into_iter()
            .map(|intra_threads| run_level_a(intra_threads, &tuning, label))
            .collect();
        assert!(
            digests.windows(2).all(|pair| pair[0] == pair[1]),
            "{label} raw output digests differ across thread counts: {digests:?}"
        );
    }
}

#[test]
#[ignore = "opt-in: needs PANEL_OCR_ONNX_MODEL and ONNX Runtime; set PANEL_OCR_PERF_BASELINE to a scratch file and run `PANEL_OCR_ONNX_MODEL=/home/ducph/.cache/panel-ocr/models/comictextdetector.pt.onnx PANEL_OCR_PERF_BASELINE=/tmp/panel-ocr-baseline cargo test -p pc-detect --features onnx,testkit --test perf2_bit_exactness -- --ignored --nocapture`"]
fn the_raw_float_digests_match_the_recorded_scratch_baseline() {
    let Some(path) = std::env::var_os("PANEL_OCR_PERF_BASELINE") else {
        eprintln!(
            "SKIPPED: PANEL_OCR_PERF_BASELINE is unset; Level B has no scratch baseline to compare"
        );
        return;
    };
    let digest = run_level_a_default(0);
    let path = Path::new(&path);
    if std::env::var_os("PANEL_OCR_PERF_BASELINE_WRITE").is_some() {
        std::fs::write(path, format!("raw_sha256={digest}\n")).expect("write scratch baseline");
        panic!("wrote PANEL_OCR_PERF_BASELINE and deliberately failed the recording run");
    }
    let baseline = std::fs::read_to_string(path).expect("scratch baseline must be readable");
    let expected = baseline
        .trim()
        .strip_prefix("raw_sha256=")
        .expect("scratch baseline must contain raw_sha256=<digest>");
    assert_eq!(
        digest, expected,
        "raw float digest differs from scratch baseline"
    );
}
