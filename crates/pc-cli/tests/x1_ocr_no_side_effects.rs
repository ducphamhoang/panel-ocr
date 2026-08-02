//! Task **X1**, v1 review finding 3 — `panel-ocr ocr` has no cleaning side effects.
//!
//! §13.1 gives `ocr` the job "run OCR over the detected boxes and write a CSV/TXT report"
//! and gives it no `--output-dir`: it is stages 1–2 plus a report. Cleaning outputs
//! (`_clean.png`, `_combined_mask.png`, ... under `cleaned/`) are `clean`'s product, and an
//! `ocr` invocation must not manufacture them as a side effect of reusing the five-stage
//! chain. The no-ONNX cases also verify that an unavailable eager factory writes no report.
//!
//! FROZEN (CLAUDE.md).

use image::{Rgb, RgbImage};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_panel-ocr"))
}

/// Runs the binary with `cwd` as the working directory, so a *relative* `--output-dir`
/// default (`cleaned`, §13.1) would land inside the temp dir we then inspect.
fn run_in(cwd: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .current_dir(cwd)
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

/// Every file under `root`, recursively, as paths relative to `root`.
fn walk(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                found.push(path.strip_prefix(root).unwrap().to_path_buf());
            }
        }
    }
    found.sort();
    found
}

#[cfg(not(feature = "onnx"))]
fn assert_no_cleaning_artifacts(root: &Path) {
    let files = walk(root);
    let rendered: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();

    assert!(
        !root.join("cleaned").exists(),
        "`ocr` must not create a `cleaned/` directory; found {rendered:?}"
    );
    for file in &rendered {
        assert!(
            !file.contains("_clean") && !file.contains("_mask"),
            "`ocr` must not write cleaning artifacts; found `{file}` among {rendered:?}"
        );
    }
}

/// An unavailable OCR factory fails before writing a report or cleaning output.
#[cfg(not(feature = "onnx"))]
#[test]
fn ocr_writes_only_its_report() {
    let dir = tempfile::tempdir().unwrap();
    let work = dir.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    let page = write_page(&work, "page01.png");
    let report = work.join("report.csv");

    let output = run_in(
        &work,
        &[
            "ocr",
            page.to_str().unwrap(),
            "--detector",
            "mock",
            "--cache-dir",
            work.join("cache").to_str().unwrap(),
            "--output",
            report.to_str().unwrap(),
        ],
    );

    assert_eq!(output.status.code(), Some(1));
    assert!(
        !report.exists(),
        "the unavailable engine must not write a report"
    );
    assert_no_cleaning_artifacts(&work);
}

/// The same no-side-effect guarantee holds for the report-to-stdout path.
#[cfg(not(feature = "onnx"))]
#[test]
fn ocr_to_stdout_writes_nothing_to_disk() {
    let dir = tempfile::tempdir().unwrap();
    let work = dir.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    let page = write_page(&work, "page01.png");

    let output = run_in(
        &work,
        &[
            "ocr",
            page.to_str().unwrap(),
            "--detector",
            "mock",
            "--format",
            "txt",
            "--cache-dir",
            work.join("cache").to_str().unwrap(),
        ],
    );

    assert_eq!(output.status.code(), Some(1));
    assert_no_cleaning_artifacts(&work);
    assert_eq!(
        walk(&work),
        vec![PathBuf::from("page01.png")],
        "only the input page may remain on disk after an `ocr` run"
    );
}

/// The contrast case, so the test above cannot pass by the run silently doing nothing:
/// `clean` on the same page, with the same detector, *does* write `cleaned/`.
#[test]
fn clean_still_writes_its_cleaned_directory() {
    let dir = tempfile::tempdir().unwrap();
    let work = dir.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    let page = write_page(&work, "page01.png");
    let profile = pc_config::DEFAULT_PROFILE_TOML.replace(
        "ocr_enabled                  = true",
        "ocr_enabled                  = false",
    );
    let profile_path = work.join("no-ocr.toml");
    std::fs::write(&profile_path, profile).unwrap();

    let output = run_in(
        &work,
        &[
            "clean",
            page.to_str().unwrap(),
            "--detector",
            "mock",
            "--profile-path",
            profile_path.to_str().unwrap(),
            "--cache-dir",
            work.join("cache").to_str().unwrap(),
        ],
    );

    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        work.join("cleaned").join("page01_clean.png").exists(),
        "`clean` must still export: {:?}",
        walk(&work)
    );
}
