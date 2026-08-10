//! `cargo xtask mode-bench`'s input-source/detector coupling (spec §16.43 item 8).
//!
//! Separate from `mode_bench_cli.rs` deliberately: that file is frozen, and these assert a
//! rule the measurement-driver task introduced rather than one the ratification froze —
//! `--demo-bubbles` must run the **real** detector, because `ReplayDetector` carries
//! exactly one recorded page and would replay that page's boxes for all 7 crops.

use std::process::Command;

fn xtask(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .output()
        .expect("run xtask")
}

const CLAP_MISSING_REQUIRED: &str = "the following required arguments were not provided";

#[test]
fn demo_bubbles_without_a_detector_is_a_usage_error_naming_the_missing_flag() {
    let result = xtask(&["mode-bench", "--demo-bubbles"]);
    assert!(
        !result.status.success(),
        "--demo-bubbles ran with no detector; it would have replayed one recorded page's \
         boxes across all 7 crops. stdout: {}",
        String::from_utf8_lossy(&result.stdout)
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains(CLAP_MISSING_REQUIRED), "{stderr}");
    assert!(stderr.contains("--detector"), "{stderr}");
}

#[test]
fn supplying_a_detector_satisfies_the_requirement_so_the_test_above_is_not_vacuous() {
    // Falsification control: if `mode-bench --demo-bubbles` failed at clap for some
    // unrelated reason, the test above would pass without proving anything about
    // `--detector`. With the flag present the run gets past argument parsing — it then
    // fails for a *different*, tier-dependent reason (no `onnx` feature, or a model that
    // fails its digest check), which is not clap's missing-argument error.
    let result = xtask(&[
        "mode-bench",
        "--demo-bubbles",
        "--detector",
        "onnx:/nonexistent/comictextdetector.pt.onnx",
    ]);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !stderr.contains(CLAP_MISSING_REQUIRED),
        "clap still reports a missing required argument with --detector supplied: {stderr}"
    );
}

#[test]
fn replay_refuses_a_detector_so_the_ci_runnable_source_can_never_load_a_model() {
    let result = xtask(&[
        "mode-bench",
        "--replay",
        "--detector",
        "onnx:/nonexistent/comictextdetector.pt.onnx",
    ]);
    assert!(
        !result.status.success(),
        "--replay accepted a --detector spec; the no-model source is no longer no-model"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("cannot be used with"),
        "the refusal is not clap's conflict error: {stderr}"
    );
}
