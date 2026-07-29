//! spec §16.24 item 1(e) and item 2 — the digest identities the pure schema checker cannot
//! reach, closed ADDITIVELY (cookbook rule 8 exit 1) rather than by editing the frozen gate.
//!
//! Why this file exists at all, in one sentence per hole:
//!   * **H2** — `model_signature`'s `source_sha256` is today verified only *inside the writer*
//!     (`xtask/src/model_signature.rs:212-218`). A writer verifying its own output is cookbook
//!     rule 12 exactly: the gate guards a copy, and nothing checks the committed file against
//!     the single source of truth.
//!   * **H1** — `nlm`/`inter_area` name upstream sources whose digests are NOT in the
//!     provenance files, because `assert_known_non_committed_form`
//!     (`recorded_provenance.rs:184-208`) rejects that path form and loosening the exemption is
//!     rule 13's "bypass wearing different clothes" (§16.24 item 1(e)). So the digests live
//!     here, as literals measured with coreutils `sha256sum` — an implementation independent of
//!     our own `sha2` (§16.24 item 15: no digest is ever hand-typed).
//!   * **§16.24 item 2's condition** — `model_digest` dodges the frozen walker. That is
//!     permitted ONLY because this file asserts it equals `pc_models::COMIC_TEXT_DETECTOR.sha256`.
//!     Deleting `the_detector_group_model_digest_equals_the_pc_models_constant` re-opens the hole
//!     it was traded for.
//!   * **The same condition for `decoded_rgb_digest`** (§16.24 items 16(b)/17) — it dodges the
//!     walker too, and `validate`'s R23 only proves the two sides AGREE. Two equal wrong values
//!     pass R23. `our_recorded_decoded_rgb_digest_equals_the_decoded_committed_page` is the other
//!     leg: it pins our side to reality by decoding the committed JPEG. Both legs are load-bearing
//!     together, exactly as §16.16 item 5 describes for the model-identity pair.

use pc_models::COMIC_TEXT_DETECTOR;
use pc_testkit::paths;
use pc_testkit::provenance::{self, GroupProvenance};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_group(name: &str) -> GroupProvenance {
    let path = paths::recorded(PathBuf::from(name).join(provenance::PROVENANCE_FILE_NAME));
    serde_json::from_slice(&std::fs::read(&path).expect("readable provenance"))
        .unwrap_or_else(|error| panic!("`{}` is not canonical provenance: {error}", path.display()))
}

/// Measured with `sha256sum` at `8a309a0`, NOT with any panel-ocr code path (§16.24 item 15).
/// `(repo-relative source path, digest)`.
const UPSTREAM_SOURCE_DIGESTS: &[(&str, &str)] = &[
    (
        "tests/fixtures/upstream/demo_bubbles/nightmare_bubble_raw.png",
        "634f509b0a5c57953a42b720d34da2241504e56c28310d785166a71d04df4773",
    ),
    (
        "tests/fixtures/upstream/demo_bubbles/ray_bubble_raw.png",
        "f72e735d7c66fac7fde5a922a19f3903bafabeba77986309751722e7c0aff7da",
    ),
    (
        "tests/fixtures/upstream/long_strip.jpg",
        "56d40c06190334e999d90ca59b81d0b099876a8e820df5de198d5012d0e3a5c0",
    ),
];

#[test]
// spec §16.24 item 1(e), hole H2. `pc-models` owns model identity (§16.16 item 5: "the single
// source of truth for model identity lives in `pc-models`"), and the committed
// `model_signature/PROVENANCE.json` claims that identity. Nothing compared the two: the
// recorder's own post-write verification protects the maintainer who ran it and says nothing
// about the state everyone else pulls. Falsifiable: change either side by one nibble.
fn the_model_signature_source_digest_equals_the_pc_models_constant() {
    let group = read_group("model_signature");
    let record = group
        .records
        .iter()
        .find(|record| record.source.as_deref() == Some(COMIC_TEXT_DETECTOR.file_name))
        .unwrap_or_else(|| {
            panic!(
                "no record names `{}` as its source; records: {:?}",
                COMIC_TEXT_DETECTOR.file_name,
                group
                    .records
                    .iter()
                    .map(|record| &record.name)
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(
        record.source_sha256.as_deref(),
        Some(COMIC_TEXT_DETECTOR.sha256),
        "the recorded model source digest must equal `pc_models::COMIC_TEXT_DETECTOR.sha256`"
    );
}

#[test]
// spec §16.24 item 1(e), hole H1. Every upstream file a recording group consumed must hash to
// its measured literal, so a silently re-encoded or truncated vendored fixture cannot slip
// through — the parity gates that consume the recordings would then be comparing against a
// reference derived from a different input. Bidirectional by construction: the loop asserts a
// digest per named source, AND asserts that every source named by `nlm`/`inter_area` appears in
// the table, so adding a group source without a literal fails.
fn every_upstream_source_consumed_by_a_recording_group_hashes_to_its_literal() {
    for (relative, expected) in UPSTREAM_SOURCE_DIGESTS {
        let path = paths::workspace_root().join(relative);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("reading `{}`: {error}", path.display()));
        assert_eq!(
            &sha256_hex(&bytes),
            expected,
            "digest changed for `{relative}`"
        );
    }

    let mut named: Vec<String> = ["nlm", "inter_area"]
        .into_iter()
        .flat_map(|group| read_group(group).records)
        .filter_map(|record| record.source)
        .filter(|source| source.starts_with("tests/fixtures/upstream/"))
        .collect();
    named.sort();
    named.dedup();

    let mut covered: Vec<String> = UPSTREAM_SOURCE_DIGESTS
        .iter()
        .map(|(relative, _)| (*relative).to_owned())
        .collect();
    covered.sort();

    // The set, not the count: a swapped entry keeps the count and breaks the set.
    assert_eq!(
        named,
        covered,
        "uncovered upstream sources: {:?}; stale literals: {:?}",
        named
            .iter()
            .filter(|source| !covered.contains(source))
            .collect::<Vec<_>>(),
        covered
            .iter()
            .filter(|source| !named.contains(source))
            .collect::<Vec<_>>()
    );
}

#[test]
// spec §16.24 item 2 — THE compensating check the walker dodge was traded for: "A digest slot
// dodging the walker is acceptable ONLY because that test verifies it against the single source
// of truth; that condition is part of this ratification, not an implementation detail."
//
// Runs unconditionally and degrades to nothing: with no detector group there is no provenance
// to check, and the assertion below is the whole of the obligation once there is. There is no
// conditional-return *gate* here (§16.24 item 6) — the gate is the frozen provenance test; this
// is an identity between a constant and a committed field.
fn the_detector_group_model_digest_equals_the_pc_models_constant() {
    let path = paths::recorded_root()
        .join(provenance::DETECTOR_GROUP)
        .join(provenance::PROVENANCE_FILE_NAME);
    if !path.is_file() {
        // DORMANT, and said out loud on purpose (§16.24 item 19(f) / cookbook rule 6): this test
        // is one of the compensating legs §16.24 item 2 traded the walker dodge for, and with no
        // detector group recorded it executes ZERO assertions while reporting as passed. A silent
        // early return here is a green test concealing an unenforced condition. Falsifying all
        // three legs -- corrupt each digest by one nibble and watch them fail -- is a named line
        // item on the atomic recording commit.
        println!(
            "DORMANT: no detector group recorded yet, so this compensating leg asserted nothing \
             (§16.24 item 2's condition is UNENFORCED until the recording commit)"
        );
        return;
    }
    let group = read_group(provenance::DETECTOR_GROUP);
    let pins = group
        .detector
        .expect("the detector group carries group-level pins (R13)");
    assert_eq!(pins.model, COMIC_TEXT_DETECTOR.file_name);
    assert_eq!(
        pins.model_digest, COMIC_TEXT_DETECTOR.sha256,
        "§16.24 item 2: `model_digest` is walker-invisible ONLY because this identity holds"
    );
}

#[test]
// spec §16.20 item 8 (the PAD_VALUE erratum: "it must land before F1 records anything, since
// the pad colour perturbs every recorded box") and §16.24 item 3(b), which makes `pad_value`
// fixture-affecting provenance. `pc-testkit` cannot reach `pc-detect`, so the identity lives
// here — `pc_detect::onnx::PAD_VALUE` is ungated (`crates/pc-detect/src/onnx.rs:30`), so this
// runs in the default tier with no ONNX Runtime.
fn the_detector_group_pad_value_equals_the_implementation_constant() {
    let path = paths::recorded_root()
        .join(provenance::DETECTOR_GROUP)
        .join(provenance::PROVENANCE_FILE_NAME);
    if !path.is_file() {
        // DORMANT, and said out loud on purpose (§16.24 item 19(f) / cookbook rule 6): this test
        // is one of the compensating legs §16.24 item 2 traded the walker dodge for, and with no
        // detector group recorded it executes ZERO assertions while reporting as passed. A silent
        // early return here is a green test concealing an unenforced condition. Falsifying all
        // three legs -- corrupt each digest by one nibble and watch them fail -- is a named line
        // item on the atomic recording commit.
        println!(
            "DORMANT: no detector group recorded yet, so this compensating leg asserted nothing \
             (§16.24 item 2's condition is UNENFORCED until the recording commit)"
        );
        return;
    }
    let pins = read_group(provenance::DETECTOR_GROUP)
        .detector
        .expect("the detector group carries group-level pins (R13)");
    assert_eq!(pins.ours.pad_value, pc_detect::onnx::PAD_VALUE);
}

#[test]
// spec §16.24 items 16(b) and 17 — the second leg of the `decoded_rgb_digest` dodge, and the only
// check that pins either side to a real pixel buffer. R23 in `validate` proves the two sides agree
// and CANNOT prove either is real: two equal wrong values pass it. This decodes the committed JPEG
// with the same `image` crate our detector uses and re-derives the digest.
//
// The digest definition is normative in `OursPins::decoded_rgb_digest`'s doc comment: sha256 over
// the row-major RGB8 buffer, `height * width * 3` bytes, channel order R,G,B, no header, no
// padding. `to_rgb8().as_raw()` is exactly that layout, which is why the upstream script's numpy
// hash must be taken in C order AFTER BGR->RGB (§16.20 item 6) — hashing `cv2.imread`'s array
// directly is the channel-order failure `provenance_schema.rs`'s negative control covers.
//
// Falsifiable: re-encode the page, or drop the BGR->RGB conversion on either side, and this fails.
fn our_recorded_decoded_rgb_digest_equals_the_decoded_committed_page() {
    let path = paths::recorded_root()
        .join(provenance::DETECTOR_GROUP)
        .join(provenance::PROVENANCE_FILE_NAME);
    if !path.is_file() {
        // DORMANT, and said out loud on purpose (§16.24 item 19(f) / cookbook rule 6): this test
        // is one of the compensating legs §16.24 item 2 traded the walker dodge for, and with no
        // detector group recorded it executes ZERO assertions while reporting as passed. A silent
        // early return here is a green test concealing an unenforced condition. Falsifying all
        // three legs -- corrupt each digest by one nibble and watch them fail -- is a named line
        // item on the atomic recording commit.
        println!(
            "DORMANT: no detector group recorded yet, so this compensating leg asserted nothing \
             (§16.24 item 2's condition is UNENFORCED until the recording commit)"
        );
        return;
    }
    let group = read_group(provenance::DETECTOR_GROUP);
    let pins = group
        .detector
        .expect("the detector group carries group-level pins (R13)");

    // Ours decodes `input_page` itself (§16.24 item 17); assert that rather than assume it, since
    // the whole point of `decoded_from` is that the two sides read different artifacts.
    assert_eq!(pins.ours.decoded_from, "input_page");

    let page = paths::workspace_root().join(&pins.input_page);
    let decoded = image::open(&page)
        .unwrap_or_else(|error| panic!("decoding `{}`: {error}", page.display()))
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    let buffer = decoded.as_raw();
    assert_eq!(
        buffer.len(),
        width as usize * height as usize * 3,
        "the normative definition is 3 bytes per pixel with no padding"
    );
    assert_eq!(
        sha256_hex(buffer),
        pins.ours.decoded_rgb_digest,
        "our recorded decoded-RGB digest must be reproducible from the committed page"
    );

    // And the file digest itself, so a re-encoded page cannot pass by having its buffer digest
    // updated in isolation (§16.24 item 17 commits the JPEG byte-for-byte).
    let bytes = std::fs::read(&page).expect("readable page");
    assert_eq!(sha256_hex(&bytes), pins.input_page_sha256);
}

#[test]
// spec §16.24 item 15's cleanup, verified rather than assumed: `xtask`'s hand-rolled 75-line
// SHA-256 is deleted in favour of `sha2`, which is already in the build graph via `pc-models`.
// This asserts the two implementations agreed on a known vector before the copy went away, so
// the deletion is a measured equivalence and not a hope. (Delete this test WITH the copy if the
// architects prefer; it documents the migration, and its value expires once the copy is gone.)
fn sha256_agrees_with_a_known_vector() {
    // FIPS-180-2 / RFC 6234 test vector for "abc".
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}
