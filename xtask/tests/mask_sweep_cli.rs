//! Task T3 (§16.35 item 8) -- CI-runnable `cargo xtask mask-sweep --replay`. FROZEN.
//!
//! Existing xtask convention splits pure renderer checks into the source module and keeps a
//! process-level CLI assertion in `xtask/tests/` (`calibrate_cli.rs`). Because T3's source module
//! does not exist in this planning pass, these black-box tests freeze the command/report contract;
//! T3 should additionally place pure aggregation/renderer controls in `mask_sweep.rs`.

use std::process::Command;

const STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

#[test]
fn replay_mode_measures_the_committed_detector_fixture_and_writes_a_non_gating_report() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = temp.path().join("MASK_QUALITY_CALIBRATION.md");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "mask-sweep",
            "--replay",
            "--reviewer",
            "CI committed-fixture replay",
            "--date",
            "2026-08-05",
            "--method",
            "ReplayDetector over the committed detector fixture",
            "--out",
            report.to_str().expect("utf-8 report path"),
        ])
        .output()
        .expect("run xtask mask-sweep");

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let document = std::fs::read_to_string(&report).expect("mask-sweep report");

    // docs/GOLDEN_CALIBRATION.md's convention: generated-in-full warning, numbered
    // sections, explicit non-gating status, and a final verdict/summary.
    assert!(document.starts_with("# Mask quality calibration (spec §16.35, task T3)\n"));
    assert!(
        document.contains("**Generated in full by `cargo xtask mask-sweep` — do not hand-edit.**")
    );
    assert!(document.contains("non-gating"));
    assert!(document.contains("## 1. Method and provenance"));
    assert!(document.contains("## 2. Summary"));
    assert!(document.contains("## 3. Region measurements"));
    assert!(document.contains("## 4. Verdict"));

    // §16.35 explicitly requires reviewer/date/method fields, not an unsigned report.
    assert!(document.contains("- **Reviewer:** CI committed-fixture replay"));
    assert!(document.contains("- **Date:** 2026-08-05"));
    assert!(document.contains("- **Method:** ReplayDetector over the committed detector fixture"));
    assert!(document.contains(STEM));
    assert!(document.contains("tests/fixtures/recorded/detector/PROVENANCE.json"));

    // Hand-measured from the committed replay page on 2026-08-05: three regions reach
    // fitting, two greedy picks already pass, and the remaining region is unrecoverable.
    // The zero rescue count is a measured result, not a skipped sweep.
    assert!(document.contains(
        "| Pages | Masking regions | Reached fitting | Already accepted | Rescue-eligible | Failed even with rescue | Dropped |"
    ));
    assert!(document.contains("| 1 | 3 | 3 | 2 | 0 | 1 | 0 |"));

    // "Against what deviation values": every row carries the complete twelve-value
    // ladder, plus separate greedy/lowest columns. Six decimals mirrors GOLDEN's tables.
    assert!(document.contains(
        "| Page | Region | Greedy index | Greedy deviation | Lowest index | Lowest deviation | Outcome | Candidate deviations |"
    ));
    for ladder in [
        "[13.143410, 13.918179, 14.692733, 14.989392, 15.320318, 34.633578, 53.482887, 54.342167, 50.611816, 47.943391, 44.657286, 48.969334]",
        "[12.424840, 12.458071, 12.692986, 12.567281, 12.569554, 13.287905, 38.344053, 46.132005, 43.101108, 39.311065, 36.409029, 49.824835]",
        "[34.022327, 29.129848, 31.198768, 32.081043, 37.543467, 41.801678, 39.391925, 40.149008, 38.748077, 34.959623, 35.487528, 29.154271]",
    ] {
        assert!(document.contains(ladder), "missing full ladder: {ladder}");
    }
    assert!(document.contains("| 0 | 13.143410 | 0 | 13.143410 | accepted |"));
    assert!(document.contains("| 0 | 12.424840 | 0 | 12.424840 | accepted |"));
    assert!(document.contains("| 1 | 29.129848 | 1 | 29.129848 | failed-even-with-rescue |"));
}

#[test]
fn help_exposes_the_replay_and_maintainer_local_modes_without_loading_a_model() {
    // CI only has to parse/compile the local form. `--help` exercises clap's real command
    // surface while guaranteeing no filesystem walk, model verification, or ONNX session.
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["mask-sweep", "--help"])
        .output()
        .expect("run mask-sweep help");
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let help = String::from_utf8_lossy(&result.stdout);
    for flag in [
        "--replay",
        "--pages <DIR>",
        "--detector <SPEC>",
        "--reviewer <REVIEWER>",
        "--date <DATE>",
        "--method <METHOD>",
        "--out <OUT>",
    ] {
        assert!(help.contains(flag), "help omitted {flag}: {help}");
    }
    assert!(
        help.contains("onnx:"),
        "local detector syntax is undocumented: {help}"
    );
}
