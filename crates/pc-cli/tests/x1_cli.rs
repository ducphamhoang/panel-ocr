//! Task **X1** — end-to-end runs of the `panel-ocr` binary (spec §5.3, §5.5, §5.6,
//! §13.1, §16.12 item 2).
//!
//! These are the tests that prove what v1's CLI can actually promise today: a full
//! five-stage run with `--detector mock`, and a clear, non-panicking refusal for the
//! not-yet-implemented ONNX backend.
//!
//! FROZEN (CLAUDE.md).

use image::{Rgb, RgbImage};
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
    let mut image = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]));
    for y in 16..32 {
        for x in 16..32 {
            image.put_pixel(x, y, Rgb([20, 20, 20]));
        }
    }
    let path = dir.join(name);
    image.save(&path).unwrap();
    path
}

#[test]
fn help_and_version_work() {
    assert!(run(&["--help"]).status.success());
    assert!(run(&["--version"]).status.success());
    assert!(run(&["clean", "--help"]).status.success());
}

/// §16.12 item 2: the default `onnx` backend is a *fatal* error with an explanation —
/// exit code 1, not a panic (which would be 101), and not a silent no-op run.
#[test]
fn the_onnx_detector_fails_cleanly_with_an_explanation() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");

    let output = run(&[
        "clean",
        page.to_str().unwrap(),
        "--cache-dir",
        dir.path().join("cache").to_str().unwrap(),
        "--output-dir",
        dir.path().join("out").to_str().unwrap(),
    ]);

    assert_eq!(
        output.status.code(),
        Some(1),
        "must be a fatal, not a panic"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not available in this build"), "{stderr}");
    assert!(stderr.contains("replay"), "{stderr}");
}

/// §5.6 + §16.12 item 2: `--detector mock` runs the whole chain on a page with no
/// detected text and still exports it, exiting 0.
#[test]
fn a_mock_run_exports_the_page_and_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let out = dir.path().join("out");

    let output = run(&[
        "clean",
        page.to_str().unwrap(),
        "--detector",
        "mock",
        "--cache-dir",
        dir.path().join("cache").to_str().unwrap(),
        "--output-dir",
        out.to_str().unwrap(),
    ]);

    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        out.join("page01_clean.png").exists(),
        "expected an exported cleaned page in {}",
        out.display()
    );
}

/// §5.3: a path that does not exist is a fatal condition, exit code 1.
#[test]
fn a_missing_input_is_fatal() {
    let dir = tempfile::tempdir().unwrap();

    let output = run(&[
        "clean",
        dir.path().join("nope.png").to_str().unwrap(),
        "--detector",
        "mock",
        "--cache-dir",
        dir.path().join("cache").to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(1));
}

/// §6/§12.7(A)9: an unsupported output suffix fails **config validation**, fatally, with
/// the supported list in the message.
#[test]
fn an_unsupported_output_suffix_fails_config_validation() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let profile = dir.path().join("bad.toml");
    // resolved 2026-07-28 (§16.12 item 22): DEFAULT_PROFILE_TOML already has a
    // `[general]` table, so appending a second one is a TOML duplicate-table parse
    // error that masks the suffix-validation failure this test exists to check.
    // Override the existing key in place instead.
    let bad = pc_config::DEFAULT_PROFILE_TOML.replace(
        "preferred_file_type          = \"\"",
        "preferred_file_type          = \".xyz\"",
    );
    assert!(
        bad.contains("\".xyz\""),
        "the default profile's preferred_file_type line changed shape"
    );
    std::fs::write(&profile, bad).unwrap();

    let output = run(&[
        "clean",
        page.to_str().unwrap(),
        "--detector",
        "mock",
        "--profile-path",
        profile.to_str().unwrap(),
        "--cache-dir",
        dir.path().join("cache").to_str().unwrap(),
    ]);

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(".png"),
        "must name the supported list: {stderr}"
    );
}

/// §5.5: a batch where one image fails exits 2, and the summary names the failure.
/// `broken.png` is a text file with a `.png` suffix — undecodable, so stage 1 fails on
/// it while the real page completes.
#[test]
fn a_partly_failing_batch_exits_two() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = dir.path().join("in");
    std::fs::create_dir_all(&inputs).unwrap();
    write_page(&inputs, "good.png");
    std::fs::write(inputs.join("broken.png"), b"not an image at all").unwrap();

    let output = run(&[
        "clean",
        inputs.to_str().unwrap(),
        "--detector",
        "mock",
        "--cache-dir",
        dir.path().join("cache").to_str().unwrap(),
        "--output-dir",
        dir.path().join("out").to_str().unwrap(),
    ]);

    assert_eq!(
        output.status.code(),
        Some(2),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(combined.contains("broken.png"), "{combined}");
    assert!(
        dir.path().join("out").join("good_clean.png").exists(),
        "the healthy page must still be exported"
    );
}
