use std::process::Command;

#[test]
fn calibrate_goldens_prints_a_downgrade_warning_to_stderr() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output = temp.path().join("prior-calibration.md");
    std::fs::write(&output, include_str!("../../docs/GOLDEN_CALIBRATION.md"))
        .expect("prior document");

    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .env_remove("PANEL_OCR_ONNX_MODEL")
        .args([
            "calibrate-goldens",
            "--out",
            output.to_str().expect("utf-8 output path"),
        ])
        .output()
        .expect("run xtask");

    assert!(result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("WARNING: this run could not reproduce a previously measured Section 4"),
        "expected downgrade warning on stderr, got: {stderr}"
    );
}
