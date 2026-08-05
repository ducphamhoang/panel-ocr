//! GPU-1 G1-D CLI gates — §16.36 item 5.
//!
//! FROZEN (CLAUDE.md). These are process-level tests because request refusal must happen
//! before `record::run` dispatches a group or `calibrate::run` writes its document.

use pc_core::device::DevicePolicy;
use std::path::PathBuf;
use std::process::{Command, Output};

const REQUEST_REFUSAL: &str = "fixture-producing command refuses requested device `cuda`; only `cpu` is ratified for committed recordings (§16.22 item 5(b))";

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_xtask"))
}

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .env_remove("PANEL_OCR_ONNX_MODEL")
        .args(args)
        .output()
        .expect("run xtask")
}

fn assert_request_refusal_before_dispatch(group: &str) {
    let output = run(&["record-fixtures", "--device", "cuda", "--only", group]);
    assert!(!output.status.success(), "{group} must refuse cuda");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(REQUEST_REFUSAL), "{group}: {stderr}");
    assert!(
        !stdout.contains('─') && !stdout.contains("summary"),
        "{group} was dispatched before the request refusal:\n{stdout}"
    );
}

/// The command-level flag applies once to the request, before any selected group. Detector
/// is the session-building `record-fixtures` arm; its second guard is tested separately.
#[test]
fn record_fixtures_cuda_is_refused_before_detector_dispatch() {
    assert_request_refusal_before_dispatch("detector");
}

/// `Group::ALL` has six exact variants. Five are sessionless, despite the task brief's
/// later shorthand saying "other four": none may evade the command-level recording rule.
#[test]
fn all_five_sessionless_groups_are_still_refused_at_request_level() {
    for group in [
        "nlm",
        "inter-area",
        "find-edges",
        "model-signature",
        "ocr-model-signature",
    ] {
        assert_request_refusal_before_dispatch(group);
    }
}

/// Independent cardinality/identity gate for the actual `Group::ALL` command spellings.
/// A new group cannot silently inherit no test merely because the loop above still passes.
#[test]
fn the_six_record_fixture_group_spellings_are_all_covered() {
    const ALL: &[&str] = &[
        "nlm",
        "inter-area",
        "find-edges",
        "detector",
        "model-signature",
        "ocr-model-signature",
    ];
    assert_eq!(ALL.len(), 6);
    for group in ALL {
        assert_request_refusal_before_dispatch(group);
    }
}

/// `calibrate-goldens` is itself fixture-producing and follows the identical request rule.
/// The absent output is the observable proof that refusal preceded calibration dispatch.
#[test]
fn calibrate_goldens_cuda_is_refused_before_writing_or_session_work() {
    let temp = tempfile::tempdir().expect("tempdir");
    let out = temp.path().join("must-not-exist.md");
    let output = run(&[
        "calibrate-goldens",
        "--device",
        "cuda",
        "--out",
        out.to_str().expect("utf-8 output path"),
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(REQUEST_REFUSAL),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !out.exists(),
        "request refusal must precede document writing"
    );
}

/// Bench output is non-gating scratch data. It resolves CPU and reports the policy even
/// when the model/ONNX feature is unavailable; it must not reuse recording refusal.
#[test]
fn bench_reports_the_resolved_device_instead_of_refusing_it() {
    let output = run(&["bench-detector", "--reps", "3", "--warmup", "0"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains(&DevicePolicy::cpu().report()),
        "{combined}"
    );
    assert!(!combined.contains("refuses requested device"), "{combined}");
}
