//! GPU-1 G1-C — §16.36 items 3, 4, 6, and 7.
//!
//! FROZEN (CLAUDE.md). GPU-2 is pre-authorized to make the `cuda` feature real; the
//! refusal-only assertions are therefore gated exactly as §16.36 item 7 requires.
#![allow(unexpected_cfgs)]

use image::{Rgb, RgbImage};
use pc_core::device::{resolve, Device, DeviceRefusal, DeviceSupport};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_panel-ocr"))
}

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("run panel-ocr")
}

fn write_page(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    RgbImage::from_pixel(32, 32, Rgb([255, 255, 255]))
        .save(&path)
        .expect("write input page");
    path
}

fn write_cuda_profile(dir: &Path, ocr_enabled: bool) -> PathBuf {
    let profile = pc_config::DEFAULT_PROFILE_TOML
        .replace(
            "device                       = \"cpu\"",
            "device                       = \"cuda\"",
        )
        .replace(
            "ocr_enabled                  = true",
            if ocr_enabled {
                "ocr_enabled                  = true"
            } else {
                "ocr_enabled                  = false"
            },
        );
    assert!(profile.contains("device                       = \"cuda\""));
    assert_eq!(
        profile.contains("ocr_enabled                  = true"),
        ocr_enabled
    );

    let path = dir.join(if ocr_enabled {
        "cuda-with-ocr.toml"
    } else {
        "cuda-without-ocr.toml"
    });
    std::fs::write(&path, profile).expect("write profile");
    path
}

fn cuda_refusal() -> DeviceRefusal {
    resolve(Device::Cuda, DeviceSupport::CPU_ONLY)
        .expect_err("the explicit CPU-only capability must refuse cuda")
}

// GPU-2 (§16.47 item 10): used only by the three `#[cfg(all(feature = "onnx", not(feature
// = "cuda")))]`-gated tests below it; the sixth, ungated test doesn't call it. Gated
// identically to its call sites -- not just `not(feature = "cuda")` -- so it isn't dead
// code on the default (non-onnx) tier, where none of its callers compile either
// (confirmed 2026-08-17: `RUSTFLAGS="-D warnings" cargo check -p pc-cli --test
// g1_c_device_refusal --all-targets` on the default tier failed with exactly this
// function reported never used, before this gate was corrected).
#[cfg(all(feature = "onnx", not(feature = "cuda")))]
fn assert_exact_cli_refusal(output: &Output) {
    let expected = format!("error: model error: {}\n", cuda_refusal().message());
    assert_eq!(
        output.status.code(),
        Some(1),
        "a session-construction refusal is run-fatal, not per-image or a panic"
    );
    assert_eq!(String::from_utf8_lossy(&output.stderr), expected);
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .matches(&cuda_refusal().message())
            .count()
            + String::from_utf8_lossy(&output.stderr)
                .matches(&cuda_refusal().message())
                .count(),
        1,
        "the latched run-level refusal is rendered once"
    );
}

/// §16.36 items 4, 6, and 7: the detector's image-independent CUDA refusal goes through
/// the existing `OnceLock` outcome path as `StageError::Model`; the provider declares
/// every such initialization failure run-fatal. Repeated calls exercise the same public
/// boundary used by rayon workers; the private companion test pins the attempt count.
#[cfg(all(feature = "onnx", not(feature = "cuda")))]
#[test]
fn a_cuda_device_refusal_is_model_and_declared_run_fatal() {
    use pc_cli::args::DetectorSpec;
    use pc_cli::detector;
    use pc_core::StageError;

    let cache = tempfile::tempdir().expect("cache");
    let provider = detector::build_provider(
        &DetectorSpec::Onnx,
        None,
        None,
        cache.path(),
        &pc_config::TextDetectorConfig::default(),
        Device::Cuda,
    )
    .expect("the lazy provider constructs before session initialization");

    assert!(provider.failures_are_run_fatal());
    for _ in 0..8 {
        let error = match provider.detector_for(Path::new("page.png")) {
            Err(error) => error,
            Ok(_) => panic!("a CPU-only build must refuse the requested CUDA session"),
        };
        match error {
            StageError::Model(message) => assert_eq!(message, cuda_refusal().message()),
            other => panic!("expected StageError::Model, got {other:?}"),
        }
    }
}

/// §16.36 item 3: pc-ocr owns an equivalent refusal at its eager session seam. Missing
/// paths are deliberate negative controls: device resolution must precede model preflight,
/// otherwise this test reports a path error and the requested device was silently dropped.
#[cfg(all(feature = "onnx", not(feature = "cuda")))]
#[test]
fn the_eager_ocr_session_factory_refuses_cuda_with_the_exact_model_error() {
    use pc_core::StageError;
    use pc_ocr::onnx::MangaOcrSessions;

    let root = tempfile::tempdir().expect("tempdir");
    let error = MangaOcrSessions::from_paths_for_device(
        &root.path().join("missing-encoder.onnx"),
        &root.path().join("missing-decoder.onnx"),
        Device::Cuda,
    )
    .expect_err("device refusal must happen before either missing-model preflight");

    match error {
        StageError::Model(message) => assert_eq!(message, cuda_refusal().message()),
        other => panic!("expected StageError::Model, got {other:?}"),
    }
}

/// §16.36 item 6's named plumbing hazard. This starts from real TOML, not a direct
/// `Device` argument, disables eager OCR so the detector is the first session seam, and
/// leaves the model cache empty. The exact device refusal (rather than a missing-model
/// error) proves `profile.general.device` survived run_clean -> build_provider ->
/// OnnxProvider -> initialize_detector and was checked before model resolution.
#[cfg(all(feature = "onnx", not(feature = "cuda")))]
#[test]
fn cuda_in_a_profile_reaches_initialize_detector_end_to_end() {
    let dir = tempfile::tempdir().expect("tempdir");
    let inputs = dir.path().join("inputs");
    std::fs::create_dir_all(&inputs).expect("inputs");
    write_page(&inputs, "page01.png");
    write_page(&inputs, "page02.png");
    let profile = write_cuda_profile(dir.path(), false);

    let output = run(&[
        "clean",
        inputs.to_str().expect("utf-8 input path"),
        "--profile-path",
        profile.to_str().expect("utf-8 profile path"),
        "--detector",
        "onnx",
        "--cache-dir",
        dir.path().join("cache").to_str().expect("utf-8 cache path"),
        "--output-dir",
        dir.path().join("out").to_str().expect("utf-8 output path"),
    ]);

    assert_exact_cli_refusal(&output);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!combined.contains("page01.png"), "{combined}");
    assert!(!combined.contains("page02.png"), "{combined}");
    assert!(!combined.contains("models download"), "{combined}");
}

/// §16.36 item 3's eager-ordering consequence. A mock detector cannot produce this
/// refusal, and the empty model cache would otherwise produce a provisioning error, so
/// the exact text proves the profile's device reached the eager OCR factory first.
#[cfg(all(feature = "onnx", not(feature = "cuda")))]
#[test]
fn cuda_in_a_profile_reaches_the_eager_ocr_factory_before_model_resolution() {
    let dir = tempfile::tempdir().expect("tempdir");
    let page = write_page(dir.path(), "page.png");
    let profile = write_cuda_profile(dir.path(), true);

    let output = run(&[
        "clean",
        page.to_str().expect("utf-8 input path"),
        "--profile-path",
        profile.to_str().expect("utf-8 profile path"),
        "--detector",
        "mock",
        "--cache-dir",
        dir.path().join("cache").to_str().expect("utf-8 cache path"),
        "--output-dir",
        dir.path().join("out").to_str().expect("utf-8 output path"),
    ]);

    assert_exact_cli_refusal(&output);
    assert!(
        !String::from_utf8_lossy(&output.stderr).contains("managed model is missing"),
        "device refusal must precede managed-model lookup"
    );

    let ocr_output = run(&[
        "ocr",
        page.to_str().expect("utf-8 input path"),
        "--profile-path",
        profile.to_str().expect("utf-8 profile path"),
        "--detector",
        "mock",
        "--cache-dir",
        dir.path()
            .join("ocr-cache")
            .to_str()
            .expect("utf-8 cache path"),
    ]);

    assert_exact_cli_refusal(&ocr_output);
    assert!(
        !String::from_utf8_lossy(&ocr_output.stderr).contains("managed model is missing"),
        "device refusal must precede managed-model lookup"
    );
}

/// §16.36 item 4: a configured device is inert when neither selected backend creates an
/// ONNX session. Both non-ONNX provider variants must construct under CUDA, and a real
/// mock CLI run with OCR disabled must complete without a refusal or WARN.
#[test]
fn replay_and_mock_with_ocr_disabled_accept_cuda_without_creating_a_session() {
    use pc_cli::args::DetectorSpec;
    use pc_cli::detector;

    let dir = tempfile::tempdir().expect("tempdir");
    for spec in [
        DetectorSpec::Mock,
        DetectorSpec::Replay(dir.path().join("replay")),
    ] {
        let provider = detector::build_provider(
            &spec,
            None,
            None,
            &dir.path().join("cache"),
            &pc_config::TextDetectorConfig::default(),
            Device::Cuda,
        )
        .expect("a non-ONNX provider must not resolve an unused device");
        assert!(!provider.failures_are_run_fatal());
    }

    let page = write_page(dir.path(), "page.png");
    let profile = write_cuda_profile(dir.path(), false);
    let output = run(&[
        "clean",
        page.to_str().expect("utf-8 input path"),
        "--profile-path",
        profile.to_str().expect("utf-8 profile path"),
        "--detector",
        "mock",
        "--cache-dir",
        dir.path().join("cache").to_str().expect("utf-8 cache path"),
        "--output-dir",
        dir.path().join("out").to_str().expect("utf-8 output path"),
    ]);

    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!combined.contains(&cuda_refusal().message()), "{combined}");
    assert!(!combined.contains("WARN"), "{combined}");
}
