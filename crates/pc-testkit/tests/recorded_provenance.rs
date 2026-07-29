//! Recorded-fixture provenance gates (spec §16.13 item 6).
//!
//! Every committed artifact named by the NLM and inter-area provenance files is checked
//! byte-for-byte against its recorded SHA-256 digest. Generated diagnostic outputs are
//! declared in provenance too, but are intentionally outside this committed-fixture gate.

use pc_testkit::paths;
use serde_json::Value;
use std::path::Path;

const EXPECTED_VERIFIED_ARTIFACTS: usize = 3;
const RECORDED_PREFIX: &str = "tests/fixtures/recorded";

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_provenance(group: &str) -> Value {
    let path = paths::recorded(Path::new(group).join("PROVENANCE.json"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read provenance `{}`: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("failed to parse provenance `{}`: {error}", path.display()))
}

fn verify_declared_artifact(output: &str, recorded_sha256: &str) {
    let relative = Path::new(output)
        .strip_prefix(RECORDED_PREFIX)
        .unwrap_or_else(|_| panic!("declared output `{output}` is not under {RECORDED_PREFIX}"));
    let path = paths::recorded(relative);
    let actual_sha256 = sha256_hex(&std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "failed to read declared artifact `{}`: {error}",
            path.display()
        )
    }));
    assert_eq!(
        actual_sha256,
        recorded_sha256.to_ascii_lowercase(),
        "SHA-256 mismatch for declared artifact `{output}`"
    );
}

#[test]
fn committed_recorded_fixture_digests_match_provenance() {
    let mut verified_artifact_count = 0;

    let nlm = read_provenance("nlm");
    let records = nlm["records"]
        .as_array()
        .expect("nlm/PROVENANCE.json must contain a records array");
    assert_eq!(
        records.len(),
        2,
        "nlm/PROVENANCE.json must declare both committed NLM artifacts"
    );

    for record in records {
        let output = record["output"]
            .as_str()
            .expect("each NLM provenance record must contain an output string");
        let recorded_sha256 = record["sha256"]
            .as_str()
            .expect("each NLM provenance record must contain a sha256 string");
        verify_declared_artifact(output, recorded_sha256);
        verified_artifact_count += 1;
        println!("verified committed artifact: {output}");
    }

    let inter_area = read_provenance("inter_area");
    let reference = inter_area["reference"]
        .as_object()
        .expect("inter_area/PROVENANCE.json must contain a reference object");
    let reference_output = reference["output"]
        .as_str()
        .expect("inter_area reference must contain an output string");
    let reference_sha256 = reference["sha256"]
        .as_str()
        .expect("inter_area reference must contain a sha256 string");
    verify_declared_artifact(reference_output, reference_sha256);
    verified_artifact_count += 1;
    println!("verified committed artifact: {reference_output}");

    let diagnostic = inter_area["jpeg_decode_diagnostic"]
        .as_object()
        .expect("inter_area/PROVENANCE.json must contain a jpeg_decode_diagnostic object");
    let diagnostic_output = diagnostic["output"]
        .as_str()
        .expect("inter_area JPEG diagnostic must contain an output string");
    assert!(
        diagnostic_output.starts_with("target/xtask-scratch/"),
        "the JPEG diagnostic output must remain a generated scratch path"
    );
    diagnostic["sha256"]
        .as_str()
        .expect("inter_area JPEG diagnostic must contain a sha256 string");
    println!(
        "skipped generated artifact: {diagnostic_output} (target/xtask-scratch is not committed)"
    );

    assert_eq!(verified_artifact_count, EXPECTED_VERIFIED_ARTIFACTS);
}
