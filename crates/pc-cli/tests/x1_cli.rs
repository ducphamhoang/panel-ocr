//! Task **X1** — end-to-end runs of the `panel-ocr` binary (spec §5.3, §5.5, §5.6,
//! §13.1, §16.12 item 2).
//!
//! These are the tests that prove what v1's CLI can actually promise today: a full
//! five-stage run with `--detector mock`, and a clear, non-panicking refusal for the
//! not-yet-implemented ONNX backend.
//!
//! FROZEN (CLAUDE.md).

use image::{GrayImage, Luma, Rgb, RgbImage};
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

/// The first cache entry in `dir` ending in `suffix` (§4.2 names them
/// `{uuid}_{stem}{suffix}`, so the uuid is not predictable from the test).
#[cfg(feature = "onnx")]
fn cache_entry_with_suffix(dir: &Path, suffix: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.flatten().find_map(|entry| {
        let path = entry.path();
        path.file_name()?
            .to_str()?
            .ends_with(suffix)
            .then_some(path)
    })
}

#[test]
fn help_and_version_work() {
    assert!(run(&["--help"]).status.success());
    assert!(run(&["--version"]).status.success());
    assert!(run(&["clean", "--help"]).status.success());
}

/// §16.12 item 2: the default `onnx` backend is a *fatal* error with an explanation —
/// exit code 1, not a panic (which would be 101), and not a silent no-op run.
// Gate added 2026-07-28 (authorized directly, no assertion text changed): this asserts the
// content of `detector::ONNX_UNAVAILABLE`, which is `#[cfg(not(feature = "onnx"))]`. Under
// `--features onnx` the ONNX arm is real, so this invocation stopped being a refusal at all
// — it exited 0 after ~150s, having downloaded ~90 MB and run real inference into a
// discarded TempDir. PER-TEST, never file-level: the five other tests here are
// feature-independent. The `--features onnx` counterpart is the test immediately below.
#[cfg(not(feature = "onnx"))]
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

/// §5.3 + the "`clean` never provisions" reversal — the `--features onnx` counterpart of
/// `the_onnx_detector_fails_cleanly_with_an_explanation`.
///
/// With the feature on, `clean` with no `--detector` takes the real ONNX arm. It must still
/// be a *fatal* (exit 1, not a panic's 101), because provisioning happens only in the
/// explicit `models` subcommands — and the message must hand the user that command and the
/// path it will populate.
///
/// Fable's standing rule for every `clean` invocation in this file, and the reason this test
/// pins **both** a cache dir and a profile: `--cache-dir` alone is not sufficient isolation.
/// An ambient profile with `text_detector.model_path` set takes the *override* arm, bypasses
/// the managed cache entirely, and could turn this test back into a real inference run on a
/// developer machine. Prefer an explicit `--detector` and an isolated cache dir, both.
///
/// **Under lazy-in-provider resolution this test also pins the runner's `Model` carve-out.**
/// Resolution no longer happens in `run_clean`; it happens in `detector_for`, on first use,
/// i.e. *inside* per-image work. `single.rs` maps that `Err` to `(Step::Detect, error)`, which
/// is a per-image failure and would yield `EXIT_PARTIAL` (2). Asserting **1** here is
/// therefore the end-to-end proof that a `StageError::Model` from the provider aborts the
/// whole run instead of degrading into one failure per image. `1` versus `2` is the entire
/// discriminator; see `a_batch_refuses_once_when_the_model_is_missing` for the plural case.
#[cfg(feature = "onnx")]
#[test]
fn clean_never_provisions_and_names_the_command_that_does() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let cache_root = dir.path().join("cache");
    let models_dir = pc_cli::paths::models_dir(&cache_root);
    let profile = dir.path().join("default.toml");
    std::fs::write(&profile, pc_config::DEFAULT_PROFILE_TOML).unwrap();

    let output = run(&[
        "clean",
        page.to_str().unwrap(),
        "--profile-path",
        profile.to_str().unwrap(),
        "--cache-dir",
        cache_root.to_str().unwrap(),
        "--output-dir",
        dir.path().join("out").to_str().unwrap(),
    ]);

    assert_eq!(
        output.status.code(),
        Some(1),
        "must be a fatal, not a panic (101): {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("panel-ocr models download"),
        "must name the explicit provisioning command: {stderr}"
    );
    assert!(
        stderr.contains(&models_dir.display().to_string()),
        "must name the cache path it expects the model in: {stderr}"
    );
    assert!(
        !stderr.contains("not available in this build"),
        "that is the no-feature message; this build HAS the backend: {stderr}"
    );

    // The observable form of "never provisions": no timing is asserted, the untouched cache
    // is the evidence. Before the reversal this run wrote ~90 MB here.
    assert!(
        !models_dir.exists()
            || std::fs::read_dir(&models_dir)
                .expect("readable")
                .next()
                .is_none(),
        "`clean` must not have provisioned anything into {}",
        models_dir.display()
    );
}

/// §5.1 / §5.5 + the lazy-in-provider ruling: a missing model is **one** refusal for the
/// whole run, not one failure per image.
///
/// This is the plural case `clean_never_provisions_and_names_the_command_that_does` cannot
/// express. With resolution deferred into `detector_for`, the natural implementation makes
/// every image fail identically — `single.rs` maps the provider's `Err` to
/// `(Step::Detect, error)` — which produces `EXIT_PARTIAL` (2) and an N-line summary of the
/// same fact. The ruling requires a fatal instead, so the exit code is the discriminator:
/// **1 = the carve-out holds; 2 = it does not.** The occurrence count is asserted as the
/// secondary signal because "repeated across the batch" is the regression's visible shape.
#[cfg(feature = "onnx")]
#[test]
fn a_batch_refuses_once_when_the_model_is_missing() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = dir.path().join("in");
    std::fs::create_dir_all(&inputs).unwrap();
    write_page(&inputs, "page01.png");
    write_page(&inputs, "page02.png");
    let cache_root = dir.path().join("cache");
    let profile = dir.path().join("default.toml");
    std::fs::write(&profile, pc_config::DEFAULT_PROFILE_TOML).unwrap();

    let output = run(&[
        "clean",
        inputs.to_str().unwrap(),
        "--profile-path",
        profile.to_str().unwrap(),
        "--cache-dir",
        cache_root.to_str().unwrap(),
        "--output-dir",
        dir.path().join("out").to_str().unwrap(),
    ]);

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "a missing model must abort the whole run (1), not fail per image (2): {combined}"
    );
    // "Not repeated across the batch", expressed as the absence of per-image reporting: a
    // fatal aborts before per-image work, so no `ImageOutcome` row exists for either input.
    // Counting occurrences of the message itself would NOT work -- one fatal is already
    // printed twice today (a `tracing` ERROR plus anyhow's `error:` line), so a substring
    // count cannot tell double-reporting apart from per-image repetition. The input passed on
    // the command line is the *directory*, so a page name can only appear via a summary row.
    for page in ["page01.png", "page02.png"] {
        assert!(
            !combined.contains(page),
            "`{page}` was reported individually, so the missing model became a per-image \
             failure rather than one refusal: {combined}"
        );
    }
}

/// §16.12 item 3 / §16.19 item 5: replay fixture refusal is per-image, not run-fatal.
#[test]
fn a_missing_replay_fixture_does_not_abort_the_batch() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = dir.path().join("in");
    std::fs::create_dir_all(&inputs).unwrap();
    write_page(&inputs, "page01.png");
    write_page(&inputs, "page02.png");
    write_page(&inputs, "page03.png");

    let replay_dir = dir.path().join("replay");
    let mask = GrayImage::from_pixel(64, 64, Luma([0]));
    pc_detect::write_replay_fixture(&replay_dir, "page01", &mask, &[]);
    pc_detect::write_replay_fixture(&replay_dir, "page03", &mask, &[]);
    let replay_spec = format!("replay:{}", replay_dir.display());
    let cache_dir = dir.path().join("cache");
    let output_dir = dir.path().join("out");

    let output = run(&[
        "clean",
        inputs.to_str().unwrap(),
        "--detector",
        &replay_spec,
        "--cache-dir",
        cache_dir.to_str().unwrap(),
        "--output-dir",
        output_dir.to_str().unwrap(),
    ]);

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.status.code(),
        Some(2),
        "a missing replay fixture must be per-image (2), not fatal (1): {combined}"
    );
    assert!(
        combined.contains("0 completed, 2 skipped, 1 failed"),
        "{combined}"
    );
    assert!(combined.contains("FAILED"), "{combined}");
    assert!(
        combined.contains("page02.png") && combined.contains("Detect"),
        "{combined}"
    );
    assert!(output_dir.join("page01_clean.png").is_file(), "{combined}");
    assert!(output_dir.join("page03_clean.png").is_file(), "{combined}");
}

/// §4.4 + the lazy-in-provider ruling — **the regression the ruling exists to fix.**
///
/// A resumed run whose `#raw.json` is cached never executes stage 1, so it must not need a
/// model at all. Under eager resolution (`run_clean` resolving before the batch) it hard-fails
/// demanding a 90 MB file it would never read; `single.rs`'s `flags.text_detection` arm reads
/// the checkpoint and never calls `detector_for`, so with resolution deferred this run
/// succeeds with an empty model cache.
///
/// No new fixtures: pass one seeds the cache with `--detector mock --keep-cache`, pass two
/// resumes it with the **default** `onnx` detector and no model present. Both passes pin the
/// same isolated cache dir and profile, per the standing rule.
#[cfg(feature = "onnx")]
#[test]
fn a_resumed_run_needs_no_model_when_detection_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let cache_root = dir.path().join("cache");
    let models_dir = pc_cli::paths::models_dir(&cache_root);
    let profile = dir.path().join("default.toml");
    std::fs::write(&profile, pc_config::DEFAULT_PROFILE_TOML).unwrap();

    // Pass 1 — seed the cache. `--keep-cache` is what makes the entry survive the run
    // (`clean` deletes its cache dir otherwise), and `--detector mock` keeps it model-free.
    let seed = run(&[
        "clean",
        page.to_str().unwrap(),
        "--detector",
        "mock",
        "--keep-cache",
        "--profile-path",
        profile.to_str().unwrap(),
        "--cache-dir",
        cache_root.to_str().unwrap(),
        "--output-dir",
        dir.path().join("out1").to_str().unwrap(),
    ]);
    assert_eq!(
        seed.status.code(),
        Some(0),
        "seeding pass: {}",
        String::from_utf8_lossy(&seed.stderr)
    );

    // Assert the precondition explicitly, so a failure in pass 2 cannot be misdiagnosed as
    // "the resume worked but there was nothing to resume from".
    let images = pc_cli::paths::image_cache_dir(&cache_root);
    assert!(
        cache_entry_with_suffix(&images, "#raw.json").is_some(),
        "pass 1 must have written a `#raw.json` checkpoint into {}",
        images.display()
    );

    // Pass 2 — resume with the DEFAULT detector (`onnx`) and no model anywhere.
    let out = dir.path().join("out2");
    let resumed = run(&[
        "clean",
        page.to_str().unwrap(),
        "--skip-text-detection",
        "--profile-path",
        profile.to_str().unwrap(),
        "--cache-dir",
        cache_root.to_str().unwrap(),
        "--output-dir",
        out.to_str().unwrap(),
    ]);

    assert_eq!(
        resumed.status.code(),
        Some(0),
        "a resumed run must not demand a model it will never read: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert!(
        out.join("page01_clean.png").exists(),
        "the resumed run must still export, into {}",
        out.display()
    );
    assert!(
        !models_dir.exists()
            || std::fs::read_dir(&models_dir)
                .expect("readable")
                .next()
                .is_none(),
        "and must not have provisioned anything into {}",
        models_dir.display()
    );
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
