//! Process-level contract for `cargo xtask mode-bench` (spec §16.43). FROZEN.
//!
//! §16.43 item 10 places these tests **first, before either heavy task starts**, as
//! ordinary red-first TDD. `replay_mode_writes_a_report_with_the_ratified_structure` is
//! therefore RED BY DESIGN right now: the pure layer (types, CLI parsing, device
//! disclosure, renderer) exists, but the Simple/Annotation measurement driver that fills
//! the table does not, so every cell currently renders BLOCKED. That test is
//! `#[ignore]`d only so the workspace suite stays readable; the measurement-driver task
//! (§16.43 item 9, heavy task 2) un-ignores it and must make it pass unchanged.
//!
//! The other tests here exercise the pure layer only and are green as of this task.

use std::process::Command;

const REPLAY_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

/// `pc_core::device::DeviceRefusal::NotCompiledIn { requested: Cuda }`'s message, copied
/// character-for-character from `crates/pc-core/src/device.rs`. Hard-coded rather than
/// imported so this asserts the *text a user sees*, not that two call sites agree.
const CUDA_REFUSAL: &str = "the cuda execution provider is not available in this build (panel-ocr was compiled without the `cuda` feature, so no CUDA execution provider is linked in). Rebuild with `--features cuda`, or set `device = \"cpu\"` under `[general]` in your profile to run on the CPU execution provider.";

/// The one device statement §16.43 item 6 requires verbatim, for `--device cpu`.
const CPU_DEVICE_STATEMENT: &str = "device: requested cpu; no execution provider registered explicitly (ONNX Runtime's built-in CPU provider is implicit); per-node operator placement is not claimed";

fn xtask(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .output()
        .expect("run xtask")
}

/// Pull the `| ... |` row for one `(page, cell)` pair out of the section-3 table.
fn table_row<'a>(document: &'a str, page: &str, cell: &str) -> &'a str {
    let needle = format!("| {page} | {cell} |");
    document
        .lines()
        .find(|line| line.starts_with(&needle))
        .unwrap_or_else(|| panic!("no table row for ({page}, {cell}):\n{document}"))
}

fn column(row: &str, index: usize) -> &str {
    row.split('|').nth(index).expect("column index").trim()
}

#[test]
fn replay_mode_writes_a_report_with_the_ratified_structure() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = temp.path().join("MODE_COMPARISON.md");
    let result = xtask(&[
        "mode-bench",
        "--replay",
        "--out",
        report.to_str().expect("utf-8 report path"),
    ]);
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let document = std::fs::read_to_string(&report).expect("mode-bench report");

    // §16.43 item 1: the MASK_QUALITY_CALIBRATION.md banner convention, and the
    // non-gating status stated in the document itself.
    assert!(document.starts_with("# Mode comparison benchmark (spec §16.43)\n"));
    assert!(
        document.contains("**Generated in full by `cargo xtask mode-bench` — do not hand-edit.**")
    );
    assert!(document.contains("**non-gating**"));

    // Section skeleton.
    for heading in [
        "## 1. Method and provenance",
        "## 2. Device",
        "## 3. Per-page, per-cell measurements",
        "## 4. Eligibility segments",
        "### 4.1 Full population",
        "### 4.2 Per-cell eligible subsets",
        "### 4.3 Common-eligible intersection over (simple, annotation, simple+lama)",
        "## 5. Reference comparison",
        "## 6. Verdict",
    ] {
        assert!(document.contains(heading), "missing heading `{heading}`");
    }

    // §16.43 item 6: one device statement, verbatim, once.
    assert_eq!(document.matches(CPU_DEVICE_STATEMENT).count(), 1);
    assert_eq!(document.matches("device: requested").count(), 1);
    assert!(document.contains(
        "its session constructor takes no device argument at all — device reaches the \
         detector path only as an up-front refusal, never as a registration"
    ));

    // D1's default cell set, present as rows — identity, not a count.
    assert!(document.contains("- **Cells run:** simple, annotation, simple+lama"));
    for cell in ["simple", "annotation", "simple+lama"] {
        let row = table_row(&document, REPLAY_STEM, cell);
        assert!(
            !row.contains("**BLOCKED**") && !row.contains("**NOT RECORDED**"),
            "cell `{cell}` produced no measurement on the replay fixture: {row}"
        );
    }

    // Anti-vacuity literals, taken from an independent oracle: `cargo xtask mask-sweep
    // --replay` measures the SAME committed fixture through the same
    // `pc_detect::run` -> `pc_preprocess::run` sequence and records 1 page, 3 masking
    // regions and 0 dropped (docs/MASK_QUALITY_CALIBRATION.md's summary row
    // `| 1 | 3 | 3 | 2 | 0 | 1 | 0 |`). The Simple cell here must agree.
    let simple = table_row(&document, REPLAY_STEM, "simple");
    assert_eq!(column(simple, 5), "3", "masking regions: {simple}");
    assert_eq!(column(simple, 8), "0", "dropped regions: {simple}");
    // Exactly one page on this source.
    assert_eq!(
        document
            .lines()
            .filter(|line| line.starts_with(&format!("| {REPLAY_STEM} | simple |")))
            .count(),
        1
    );

    // §16.43 item 4: the captions that carry the segmentation's meaning.
    assert!(document.contains("**cross-cell mean is permitted**"));
    assert!(document.contains("not cross-comparable"));

    // §16.43 item 8: `--replay` carries no reference, and the run does not close D2.
    assert!(document.contains("no reference"));
    assert!(document.contains("does **not** close §16.38 item 17(b) decision point D2"));
}

#[test]
fn cuda_is_a_hard_refusal_carrying_pc_cores_own_message_and_writes_no_report() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = temp.path().join("MODE_COMPARISON.md");
    let result = xtask(&[
        "mode-bench",
        "--replay",
        "--device",
        "cuda",
        "--out",
        report.to_str().expect("utf-8 report path"),
    ]);

    // §16.43 item 2(b) / item 8: non-zero exit, never a silent downgrade to CPU.
    assert!(
        !result.status.success(),
        "`--device cuda` succeeded on a build with no cuda feature; stdout: {}",
        String::from_utf8_lossy(&result.stdout)
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains(CUDA_REFUSAL),
        "refusal message is not pc_core's own, verbatim: {stderr}"
    );
    assert!(
        !report.exists(),
        "a refused run still wrote a report — the downgrade this forbids happened silently"
    );
}

#[test]
fn cpu_is_accepted_so_the_cuda_refusal_test_is_refusing_cuda_and_not_the_subcommand() {
    // Falsification control for the test above: if `mode-bench --replay` failed for some
    // unrelated reason, that test would pass vacuously.
    let temp = tempfile::tempdir().expect("tempdir");
    let report = temp.path().join("MODE_COMPARISON.md");
    let result = xtask(&[
        "mode-bench",
        "--replay",
        "--device",
        "cpu",
        "--out",
        report.to_str().expect("utf-8 report path"),
    ]);
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(report.exists());
    let document = std::fs::read_to_string(&report).expect("report");
    assert_eq!(document.matches(CPU_DEVICE_STATEMENT).count(), 1);
}

#[test]
fn help_exposes_the_three_input_sources_and_the_cell_and_device_flags() {
    let result = xtask(&["mode-bench", "--help"]);
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let help = String::from_utf8_lossy(&result.stdout);
    for flag in [
        "--replay",
        "--demo-bubbles",
        "--pages <DIR>",
        "--detector <SPEC>",
        "--cells <CELLS>",
        "--device <DEVICE>",
        "--out <OUT>",
    ] {
        assert!(help.contains(flag), "help omitted {flag}: {help}");
    }
}

#[test]
fn the_default_cell_set_is_d1s_three_cells_and_all_four_are_accepted_values() {
    let help = String::from_utf8_lossy(&xtask(&["mode-bench", "--help"]).stdout).into_owned();
    // §16.43 item 3: the default is the three-cell subset, not all four.
    assert!(
        help.contains("[default: simple,annotation,simple+lama]"),
        "default cell set is not D1's three: {help}"
    );
    for name in [
        "simple",
        "annotation",
        "simple+lama",
        "annotation+lama",
        "all",
    ] {
        assert!(
            help.contains(name),
            "help omits accepted cell `{name}`: {help}"
        );
    }
}

#[test]
fn an_unknown_cell_name_is_a_usage_error_naming_the_accepted_values() {
    let result = xtask(&["mode-bench", "--replay", "--cells", "simple,bogus"]);
    assert!(
        !result.status.success(),
        "an unknown cell name was accepted"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("bogus"), "{stderr}");
    assert!(stderr.contains("annotation+lama"), "{stderr}");
}

#[test]
fn the_three_input_sources_are_mutually_exclusive_and_one_is_required() {
    let both = xtask(&["mode-bench", "--replay", "--demo-bubbles"]);
    assert!(
        !both.status.success(),
        "--replay and --demo-bubbles were both accepted"
    );
    let neither = xtask(&["mode-bench"]);
    assert!(
        !neither.status.success(),
        "mode-bench ran with no input source"
    );
}
