//! spec §16.33 items 9 and 10 — what CI must and must not do on the third platform.
//!
//! Two ratified requirements are mechanical enough to gate:
//!
//! - **Item 10 (E6).** The default feature tier ships on Windows now, regardless of whether the
//!   `ort` Windows linking issue closes. So the default `test` job's operating-system set must
//!   contain a Windows runner. The `onnx` tier is a ratified best-effort partial and must never make
//!   Windows a *required* check unless that is separately ratified — so if the `test-onnx` job lists
//!   a Windows runner at all, it must carry `continue-on-error: true`.
//! - **Item 9 (E4).** `xtask` must compile and its non-model tests must run on Windows, which
//!   `--workspace` already delivers because `xtask` is a workspace member (§16.13 item 2). Its
//!   Python-subprocess paths stay maintainer-local, so CI must not invoke `cargo xtask` on a Windows
//!   runner.
//!
//! **Parser caveat, stated rather than glossed (cookbook rule 14c).** This file parses `ci.yml` with
//! `yaml-rust2`, which is a PROXY for the real consumer. The real consumer is GitHub Actions' own
//! YAML parser plus its expression evaluator, and neither is this crate. In particular a matrix
//! built by an expression (`${{ ... }}`) rather than a literal list would read here as a string and
//! this gate would report it as an unexpected shape rather than resolving it — which is the correct
//! failure mode for a proxy, but it is a limitation, not thoroughness.

use pc_testkit::paths;
use std::collections::BTreeSet;
use yaml_rust2::{Yaml, YamlLoader};

/// Every job `ci.yml` is expected to declare, as a SET. Pinned so the accessors below cannot
/// silently find nothing: if a job is renamed, this fails and names it, instead of every other test
/// here quietly passing over an absent key. Cardinality would not catch a rename plus an addition.
const EXPECTED_JOB_IDS: &[&str] = &[
    "fmt",
    "clippy",
    "test",
    "test-onnx",
    "frozen-snapshot-guard",
];

/// The three runners the default tier must cover from v1.1. Asserted as a set, not a count: a count
/// passes when `windows-latest` replaces `macos-latest`.
const EXPECTED_DEFAULT_TIER_RUNNERS: &[&str] = &["ubuntu-latest", "macos-latest", "windows-latest"];

/// Substring identifying a Windows runner label, so `windows-latest`, `windows-2022` and
/// `windows-11-arm` are all recognised rather than only the label in use today.
const WINDOWS_RUNNER_MARKER: &str = "windows";

fn workflow() -> Yaml {
    let path = paths::workspace_root().join(".github/workflows/ci.yml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read `{}`: {error}", path.display()));
    let mut documents = YamlLoader::load_from_str(&text)
        .unwrap_or_else(|error| panic!("`{}` is not parseable YAML: {error}", path.display()));
    assert_eq!(
        documents.len(),
        1,
        "ci.yml must be a single YAML document; found {}",
        documents.len()
    );
    documents.remove(0)
}

fn job(workflow: &Yaml, id: &str) -> Yaml {
    let job = &workflow["jobs"][id];
    assert!(
        !job.is_badvalue(),
        "ci.yml declares no job `{id}`; if it was renamed, update EXPECTED_JOB_IDS deliberately"
    );
    job.clone()
}

/// The literal `strategy.matrix.os` list of a job, as a set. `None` when the job declares no matrix
/// at all; panics when it declares one in a shape this proxy parser cannot read, rather than
/// treating an unreadable matrix as an empty one.
fn matrix_runners(job: &Yaml) -> Option<BTreeSet<String>> {
    let os = &job["strategy"]["matrix"]["os"];
    if os.is_badvalue() {
        return None;
    }
    let Yaml::Array(entries) = os else {
        panic!(
            "strategy.matrix.os is not a literal list ({os:?}); an expression-built matrix is \
             outside what this proxy parser resolves — see this file's parser caveat"
        );
    };
    Some(
        entries
            .iter()
            .map(|entry| {
                entry
                    .as_str()
                    .expect("each matrix.os entry must be a string")
                    .to_owned()
            })
            .collect(),
    )
}

/// Every `run:` script body in a job's steps, concatenated.
fn run_scripts(job: &Yaml) -> String {
    let Yaml::Array(steps) = &job["steps"] else {
        panic!("a job must declare a `steps` array");
    };
    steps
        .iter()
        .filter_map(|step| step["run"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
// Anti-vacuity for every accessor below: they all index by job id, and an index into a renamed key
// yields BadValue. Pinning the job-id set means a rename fails here with a name, rather than
// downstream with a confusing "declares no job" panic in an unrelated assertion.
fn the_workflow_declares_the_expected_job_set() {
    let workflow = workflow();
    let Yaml::Hash(jobs) = &workflow["jobs"] else {
        panic!("ci.yml declares no `jobs` mapping");
    };
    let found: BTreeSet<String> = jobs
        .keys()
        .map(|key| key.as_str().expect("job id must be a string").to_owned())
        .collect();
    let expected: BTreeSet<String> = EXPECTED_JOB_IDS.iter().map(|id| (*id).to_owned()).collect();
    assert_eq!(
        found, expected,
        "the CI job set changed. Adding or renaming a job means updating EXPECTED_JOB_IDS in the \
         same commit, so this file's other gates keep pointing at real jobs."
    );
}

#[test]
// spec §16.33 item 10 (E6) — the default tier ships on Windows. Asserted as a SET so replacing one
// runner with another fails; a length check would not (cookbook rule 13, cardinality is not
// identity). What must break for this to fail: drop `windows-latest` from the `test` job's matrix.
fn the_default_test_tier_runs_on_all_three_supported_platforms() {
    let workflow = workflow();
    let runners = matrix_runners(&job(&workflow, "test"))
        .expect("the `test` job must run under an os matrix");
    let expected: BTreeSet<String> = EXPECTED_DEFAULT_TIER_RUNNERS
        .iter()
        .map(|runner| (*runner).to_owned())
        .collect();
    assert_eq!(
        runners, expected,
        "§16.33 item 10 ships the DEFAULT feature tier on Linux, macOS and Windows. Left = what \
         ci.yml lists, right = what is ratified."
    );
}

#[test]
// spec §16.33 item 10 (E6) — the ratified partial, made mechanical. The `onnx` tier on Windows is
// best-effort: it may be absent, and it may be present as an advisory job, but it may not be a
// required check without a separate ratification. Both accepted shapes are spelled out here so the
// implementation is free to choose either and neither can drift into a blocking check by accident.
//
// What must break for this to fail: add a Windows runner to `test-onnx` without
// `continue-on-error: true`.
fn the_onnx_tier_never_makes_windows_a_required_check() {
    let workflow = workflow();
    let onnx = job(&workflow, "test-onnx");
    let Some(runners) = matrix_runners(&onnx) else {
        panic!("the `test-onnx` job must run under an os matrix");
    };
    let windows: Vec<&String> = runners
        .iter()
        .filter(|runner| runner.contains(WINDOWS_RUNNER_MARKER))
        .collect();
    if windows.is_empty() {
        // The deferred shape: nothing to check, and §16.33 item 10 permits it explicitly.
        return;
    }
    let advisory = onnx["continue-on-error"].as_bool() == Some(true);
    assert!(
        advisory,
        "the `test-onnx` job lists Windows runner(s) {windows:?} but does not carry \
         `continue-on-error: true`. §16.33 item 10 ratified the onnx tier on Windows as a \
         best-effort partial; making it a required check needs its own ratification."
    );
}

#[test]
// spec §16.33 item 9 (E4) — the "compiles" half. `xtask` is a workspace member (§16.13 item 2), so
// `--workspace` is what makes the Windows runner compile it and run its unit tests. A job that
// tested only `-p pc-cli` would satisfy the matrix gate above while leaving xtask unbuilt on
// Windows, which is exactly the gap E4 was asked about.
fn the_default_test_tier_builds_the_whole_workspace() {
    let workflow = workflow();
    let scripts = run_scripts(&job(&workflow, "test"));
    assert!(
        scripts.contains("--workspace"),
        "the `test` job must build the whole workspace so `xtask` is compiled on every matrix \
         runner (§16.33 item 9); its run scripts were:\n{scripts}"
    );
}

#[test]
// spec §16.33 item 9 (E4) — the "and nothing more" half. `xtask`'s fixture-recording and
// calibration paths spawn a Python interpreter and stay maintainer-local on Linux and macOS. A
// Windows job that invoked `cargo xtask` would be asserting parity that was explicitly NOT ratified.
//
// Scoped to jobs that actually list a Windows runner, so it does not forbid a maintainer-local
// Linux-only xtask job from ever being added.
fn no_ci_job_that_targets_windows_invokes_cargo_xtask() {
    let workflow = workflow();
    let mut offenders = Vec::new();
    for id in EXPECTED_JOB_IDS {
        let job = job(&workflow, id);
        let targets_windows = matrix_runners(&job).is_some_and(|runners| {
            runners
                .iter()
                .any(|runner| runner.contains(WINDOWS_RUNNER_MARKER))
        }) || job["runs-on"]
            .as_str()
            .is_some_and(|runner| runner.contains(WINDOWS_RUNNER_MARKER));
        if targets_windows && run_scripts(&job).contains("cargo xtask") {
            offenders.push(*id);
        }
    }
    assert!(
        offenders.is_empty(),
        "job(s) {offenders:?} target a Windows runner and invoke `cargo xtask`. §16.33 item 9 \
         keeps xtask's Python-subprocess paths maintainer-local on Linux and macOS; running them \
         on Windows CI asserts a parity that was explicitly not ratified."
    );
}
