#![cfg(all(feature = "onnx", target_arch = "x86_64"))]

use pc_detect::onnx::{
    panic_next_infer_for_test, panic_next_preprocess_for_test, panic_next_session_build_for_test,
    OnnxDetector, SessionTuning,
};
use pc_detect::TextDetector;
use sha2::{Digest, Sha256};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

static REAL_MODEL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn denormal_survives_on_this_thread() -> bool {
    let smallest_normal = std::hint::black_box(f32::MIN_POSITIVE);
    let divisor = std::hint::black_box(2.0_f32);
    std::hint::black_box(smallest_normal / divisor) != 0.0
}

fn model() -> PathBuf {
    PathBuf::from(
        std::env::var_os("PANEL_OCR_ONNX_MODEL").expect("PANEL_OCR_ONNX_MODEL is required"),
    )
}

fn page() -> image::RgbImage {
    image::RgbImage::from_fn(700, 1000, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 200])
    })
}

#[test]
fn session_build_panic_reaches_caller_with_original_payload() {
    let tempdir = tempfile::tempdir().expect("temporary model directory must be created");
    let model = tempdir.path().join("dummy.onnx");
    std::fs::write(&model, b"not a real model").expect("dummy model file must be created");

    panic_next_session_build_for_test();
    let result = catch_unwind(AssertUnwindSafe(|| {
        OnnxDetector::from_path_with_tuning(&model, 0, 0, &SessionTuning::default())
    }));
    let message = result.as_ref().err().and_then(|payload| {
        payload
            .downcast_ref::<&'static str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
    });

    assert_eq!(message, Some("injected session build panic"));
}

fn output_digest(detector: &OnnxDetector, image: &image::RgbImage) -> [u8; 32] {
    let (_, values, _) = detector
        .detect_raw_tensors(image)
        .expect("raw tensors must be available for digesting");
    let mut digest = Sha256::new();
    for output in values {
        digest.update((output.len() as u64).to_le_bytes());
        for value in output {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    digest.finalize().into()
}

#[test]
#[ignore = "opt-in: needs PANEL_OCR_ONNX_MODEL and ONNX Runtime; run with `PANEL_OCR_ONNX_MODEL=/path/comictextdetector.pt.onnx cargo test -p pc-detect --features onnx,testkit,bench-tuning --test perf2_mxcsr_hygiene -- --ignored --nocapture`"]
fn flush_denormals_does_not_escape_the_detector_worker_thread() {
    let _test_guard = REAL_MODEL_TEST_LOCK
        .lock()
        .expect("real-model test lock must not be poisoned");
    assert!(
        denormal_survives_on_this_thread(),
        "test caller already has DAZ/FTZ enabled before detector construction"
    );

    let flush_tuning = SessionTuning {
        flush_denormals: true,
        ..SessionTuning::default()
    };
    let baseline_tuning = SessionTuning {
        flush_denormals: false,
        ..SessionTuning::default()
    };
    let flush_detector = OnnxDetector::from_path_with_tuning(&model(), 0, 0, &flush_tuning)
        .expect("real ONNX session must open");
    let baseline_detector = OnnxDetector::from_path_with_tuning(&model(), 0, 0, &baseline_tuning)
        .expect("baseline real ONNX session must open");
    let image = page();

    flush_detector
        .detect_raw_tensors(&image)
        .expect("flush detector tensors must be available");
    baseline_detector
        .detect_raw_tensors(&image)
        .expect("baseline detector tensors must be available");

    let baseline_started = Instant::now();
    baseline_detector
        .detect(&image)
        .expect("baseline detector path must be available");
    let baseline_elapsed = baseline_started.elapsed();

    let flush_started = Instant::now();
    flush_detector
        .detect(&image)
        .expect("flush detector path must be available");
    let flush_elapsed = flush_started.elapsed();
    let speedup = baseline_elapsed.as_secs_f64() / flush_elapsed.as_secs_f64();
    println!(
        "flush_denormals=false baseline_elapsed={baseline_elapsed:?}; \
         flush_denormals=true inference_elapsed={flush_elapsed:?}; speedup={speedup:.2}x"
    );
    assert!(
        speedup >= 3.0,
        "flush_denormals=true inference took {flush_elapsed:?} versus baseline {baseline_elapsed:?}; \
         this timing proxy suggests the ORT denormal-flush option was not applied (the ratio is \
         not a performance SLA)"
    );

    assert!(
        denormal_survives_on_this_thread(),
        "DAZ/FTZ leaked onto the detector caller thread"
    );

    let fresh_thread = std::thread::spawn(denormal_survives_on_this_thread)
        .join()
        .expect("fresh hygiene probe thread must not panic");
    assert!(fresh_thread, "DAZ/FTZ leaked into a freshly spawned thread");
}

#[test]
#[ignore = "opt-in: needs PANEL_OCR_ONNX_MODEL and ONNX Runtime; run with `PANEL_OCR_ONNX_MODEL=/path/comictextdetector.pt.onnx cargo test -p pc-detect --features onnx,testkit,bench-tuning --test perf2_mxcsr_hygiene -- --ignored --nocapture`"]
fn preprocessing_panic_is_per_image_and_does_not_kill_the_detector_worker() {
    let _test_guard = REAL_MODEL_TEST_LOCK
        .lock()
        .expect("real-model test lock must not be poisoned");
    let tuning = SessionTuning::default();
    let detector = OnnxDetector::from_path_with_tuning(&model(), 0, 0, &tuning)
        .expect("real ONNX session must open");
    let image = page();

    detector
        .detect(&image)
        .expect("first good image must succeed");
    panic_next_preprocess_for_test();
    let bad = catch_unwind(AssertUnwindSafe(|| detector.detect(&image)));
    let bad_message = bad.as_ref().err().and_then(|payload| {
        payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
    });
    let after_bad = detector.detect(&image);
    let after_second_good = detector.detect(&image);

    assert!(
        bad_message == Some("injected preprocessing panic")
            && after_bad.is_ok()
            && after_second_good.is_ok(),
        "bad_message={bad_message:?} after_bad={after_bad:?} \
         after_second_good={after_second_good:?}"
    );
}

#[test]
#[ignore = "opt-in: needs PANEL_OCR_ONNX_MODEL and ONNX Runtime; run with `PANEL_OCR_ONNX_MODEL=/path/comictextdetector.pt.onnx cargo test -p pc-detect --features onnx,testkit,bench-tuning --test perf2_mxcsr_hygiene -- --ignored --nocapture`"]
fn inference_panic_is_per_request_and_does_not_kill_the_detector_worker() {
    let _test_guard = REAL_MODEL_TEST_LOCK
        .lock()
        .expect("real-model test lock must not be poisoned");
    let detector = OnnxDetector::from_path_with_tuning(&model(), 0, 0, &SessionTuning::default())
        .expect("real ONNX session must open");
    let image = page();

    detector
        .detect(&image)
        .expect("first good image must succeed");
    let reference_digest = output_digest(&detector, &image);
    panic_next_infer_for_test();
    let panicking_request = catch_unwind(AssertUnwindSafe(|| detector.detect(&image)));
    let after_panic_digest = output_digest(&detector, &image);
    let after_second_good_digest = output_digest(&detector, &image);

    let panicking_message = panicking_request.as_ref().err().and_then(|payload| {
        payload
            .downcast_ref::<&'static str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
    });
    assert!(
        panicking_message == Some("injected ONNX output extraction panic")
            && after_panic_digest == reference_digest
            && after_second_good_digest == reference_digest,
        "panicking_request={panicking_request:?} reference_digest={reference_digest:?} \
         after_panic_digest={after_panic_digest:?} after_second_good_digest={after_second_good_digest:?}"
    );
}
