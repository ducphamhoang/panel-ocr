//! Task **X1**, v1 review finding 4 (CLI half) — a stray non-image among named inputs is
//! a skip, not an exit-code-2 batch (§5.1, §5.5, §5.6).
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

fn write_no_ocr_profile(dir: &Path) -> PathBuf {
    let path = dir.join("no-ocr.toml");
    let profile = pc_config::DEFAULT_PROFILE_TOML.replace(
        "ocr_enabled                  = true",
        "ocr_enabled                  = false",
    );
    std::fs::write(&path, profile).unwrap();
    path
}

/// A `.xyz` file named alongside a real page: exit code 0, the summary says SKIPPED with
/// the reason, and the real page is still exported.
#[test]
fn a_stray_unsupported_file_does_not_fail_the_batch() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let stray = dir.path().join("notes.xyz");
    std::fs::write(&stray, b"not an image").unwrap();
    let out = dir.path().join("out");
    let profile = write_no_ocr_profile(dir.path());

    let output = run(&[
        "clean",
        page.to_str().unwrap(),
        stray.to_str().unwrap(),
        "--detector",
        "mock",
        "--profile-path",
        profile.to_str().unwrap(),
        "--cache-dir",
        dir.path().join("cache").to_str().unwrap(),
        "--output-dir",
        out.to_str().unwrap(),
    ]);

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "an unsupported format is a skip, not a failure:\n{combined}"
    );
    assert!(combined.contains("SKIPPED"), "{combined}");
    assert!(combined.contains("notes.xyz"), "{combined}");
    assert!(
        combined.contains("unsupported image format"),
        "the summary must give the reason:\n{combined}"
    );
    assert!(
        out.join("page01_clean.png").exists(),
        "the real page must still be exported"
    );
}

/// A batch of nothing but unsupported files is still exit 0 (no failures), and writes no
/// exports at all.
#[test]
fn only_unsupported_files_is_still_success() {
    let dir = tempfile::tempdir().unwrap();
    let stray = dir.path().join("notes.xyz");
    std::fs::write(&stray, b"not an image").unwrap();
    let out = dir.path().join("out");
    let profile = write_no_ocr_profile(dir.path());

    let output = run(&[
        "clean",
        stray.to_str().unwrap(),
        "--detector",
        "mock",
        "--profile-path",
        profile.to_str().unwrap(),
        "--cache-dir",
        dir.path().join("cache").to_str().unwrap(),
        "--output-dir",
        out.to_str().unwrap(),
    ]);

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(0), "{combined}");
    assert!(!out.exists(), "nothing was cleaned, so nothing is exported");
}
