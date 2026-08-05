//! GPU-1 G1-D — §16.36 item 5's one-constant rule and byte-stability gates.
//!
//! FROZEN (CLAUDE.md). `OursPins.execution_provider` deliberately remains `String`;
//! `provenance_schema.rs` continues to own that schema assertion unchanged.

use pc_core::device::Device;
use pc_testkit::provenance::REQUIRED_EXECUTION_PROVIDER;
use sha2::{Digest, Sha256};

#[test]
fn required_execution_provider_is_the_cpu_device_spelling() {
    assert_eq!(REQUIRED_EXECUTION_PROVIDER, Device::Cpu.as_str());
}

/// §16.36 says the producer changes but the committed schema/bytes do not. These hashes
/// were taken before G1-D and cover every committed PROVENANCE.json, not only Detector.
#[test]
fn committed_provenance_files_remain_byte_for_byte_unchanged() {
    const EXPECTED: &[(&str, &str)] = &[
        (
            "detector/PROVENANCE.json",
            "0667a2fbba275c5f63d11a16be0c56c870609ec5f3e605dc0df14a7c1e74147d",
        ),
        (
            "inter_area/PROVENANCE.json",
            "a56d2b5d0c8855844f50855142fbd83249bae12ea277dd35d2e2fb06f1ff4582",
        ),
        (
            "model_signature/PROVENANCE.json",
            "63c6371718282fcf029eb48a9625a44269c16350e786b6fe043a1db36ffa99b0",
        ),
        (
            "nlm/PROVENANCE.json",
            "5bcba4578563db4007ff5aafc4181970f497f3ac5e3ec22f5799feb5b7807580",
        ),
        (
            "ocr_model_signature/PROVENANCE.json",
            "d8371a23ae384be73b252aaff19b8f5e0fc77c64aa08e21ddeab10653c8bb18c",
        ),
    ];

    assert_eq!(EXPECTED.len(), 5);
    for (relative, expected) in EXPECTED {
        let bytes = std::fs::read(pc_testkit::paths::recorded(relative)).expect("provenance bytes");
        let actual = format!("{:x}", Sha256::digest(bytes));
        assert_eq!(&actual, expected, "{relative} changed byte-for-byte");
    }
}

/// `docs/GOLDEN_CALIBRATION.md` is a second reader of Detector provenance. Pin the
/// complete fenced JSON content, rather than one convenient field or digest; line endings
/// are normalized because the repository deliberately supports CRLF worktrees.
#[test]
fn golden_calibration_inlines_the_complete_detector_provenance() {
    let relative = "tests/fixtures/recorded/detector/PROVENANCE.json";
    let provenance = std::fs::read_to_string(pc_testkit::paths::workspace_root().join(relative))
        .expect("detector provenance");
    let document = std::fs::read_to_string(
        pc_testkit::paths::workspace_root().join("docs/GOLDEN_CALIBRATION.md"),
    )
    .expect("golden calibration document");
    let marker = format!("Recording provenance (`{relative}`)</summary>");
    let after_marker = document
        .split_once(&marker)
        .expect("detector provenance details block")
        .1;
    let fenced = after_marker
        .split_once("```json\n")
        .expect("opening detector JSON fence")
        .1
        .split_once("\n```")
        .expect("closing detector JSON fence")
        .0;

    let normalized = provenance.replace("\r\n", "\n");
    assert_eq!(fenced, normalized.trim_end());
}
