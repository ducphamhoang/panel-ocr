//! Task F3 -- the recorded model signature schema (spec §16.16). Frozen gates.
//!
//! `pc-testkit` is the SOLE owner of this schema: the F3 `xtask` probe writes it, and
//! `pc-detect`'s `d4_signature.rs` and `pc-models`' `d1_resolve.rs` both read it through
//! this crate. Duplicating the shape in either consumer would let the two drift, which is
//! the same failure mode §16.13 item 3 gives as the reason `xtask` may consume this crate.
//!
//! These tests deliberately assert *schema* properties -- that the committed file parses,
//! that lookups behave, and that the recorded tensors satisfy the semantic facts consumers
//! rely on. The vocabulary constant remains the single spelling authority; these tests also
//! make sure the fixture really uses that spelling.

use pc_testkit::model_signature::{
    self, comic_text_detector_signature, load_model_signature, COMIC_TEXT_DETECTOR_SIGNATURE,
    ELEMENT_TYPE_F32,
};
use pc_testkit::paths;

#[test]
// spec §16.16: the committed signature lives under `recorded/model_signature/`, a
// directory of its own -- deliberately NOT under `recorded/detector/`, whose BLOCKED status
// (§16.13 item 8: needs D4b, weights AND license-clean manga pages) must stay
// unambiguous. Underscored to match the existing `nlm` / `inter_area` group directories.
fn the_signature_fixture_lives_in_its_own_group_directory() {
    let path = paths::recorded(COMIC_TEXT_DETECTOR_SIGNATURE);

    assert!(path.is_file());
    assert_eq!(
        path.parent().expect("the fixture has a parent directory"),
        paths::recorded_root().join("model_signature")
    );
    assert!(
        !path.starts_with(paths::recorded_root().join("detector")),
        "the model signature must not live in the blocked `detector` group's directory"
    );
}

#[test]
// spec §16.13 item 6: every recording group writes a `PROVENANCE.json` next to its
// outputs -- a reviewer cannot review a recorded artifact without knowing what produced
// it, and §7.2 calls these "a checked-in artifact, reviewed like code".
fn the_signature_group_records_its_provenance() {
    let provenance = paths::recorded("model_signature/PROVENANCE.json");

    assert!(std::fs::metadata(&provenance).expect("readable").len() > 0);
    let parsed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&provenance).expect("readable"))
            .expect("PROVENANCE.json is valid JSON");
    assert!(parsed.is_object());
}

/// `sha256(bytes)` as lowercase hex.
///
/// Local to this test file on purpose. `pc-testkit` must not depend on `pc-models` (that
/// would invert a dependency edge, since `pc-models` dev-depends on this crate), and
/// hashing one 772-byte fixture does not justify new public API on the schema owner.
/// `{:02x}` rather than `{:x}`, so a byte below `0x10` keeps its leading zero.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
// spec §16.13 item 6 / §16.16: the committed PROVENANCE.json must describe the committed
// signature *accurately*, not merely exist.
//
// The hole this closes cannot be closed from inside the recorder. Recording commits TWO
// files with TWO renames, and without a journal that pair can never be made atomic -- so a
// crash between them leaves a NEW signature beside a STALE PROVENANCE.json. Because
// provenance carries the signature's own `output_sha256`, such a pair does not just go
// stale, it *actively misdescribes itself*, which destroys exactly the reviewability
// §16.13 item 6 exists to provide ("a reviewer cannot review a PNG without knowing what
// produced it"). The recorder's own post-write verification protects the maintainer who
// ran it; it says nothing about the state everyone else pulls.
//
// In practice the mundane causes are likelier than a crash: a hand-edited signature, or a
// half-committed pair with one file staged and the other not.
//
// INTERNAL CONSISTENCY ONLY -- no digest literal is pinned, so a legitimate re-recording
// (new weights, a schema change) stays green while an inconsistent pair fails loudly.
fn the_committed_provenance_describes_the_committed_signature_byte_for_byte() {
    // Prove the instrument before trusting it: FIPS 180-4's `sha256("abc")`. Without this,
    // a hex-encoding bug in the helper above would turn the real assertion into a
    // permanent failure whose message pointed at the fixture instead of at the test.
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let signature_path = paths::recorded(COMIC_TEXT_DETECTOR_SIGNATURE);
    let provenance_path = paths::recorded("model_signature/PROVENANCE.json");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&provenance_path).expect("readable"))
            .expect("PROVENANCE.json is valid JSON");

    // Guard the premise of everything below: the provenance must be describing THIS file
    // and not some other output that happens to sit in the same directory.
    let described = provenance["output"]
        .as_str()
        .expect("PROVENANCE.json records `output` as a string");
    assert_eq!(
        paths::workspace_root().join(described),
        signature_path,
        "PROVENANCE.json describes `{described}`, which is not the committed signature"
    );

    let recorded = provenance["output_sha256"]
        .as_str()
        .expect("PROVENANCE.json records `output_sha256` as a string");
    let actual = sha256_hex(&std::fs::read(&signature_path).expect("readable"));

    assert!(
        recorded.eq_ignore_ascii_case(&actual),
        "the committed fixture pair is INCONSISTENT -- provenance and signature do not \
         describe the same bytes.\n  \
         PROVENANCE.json records output_sha256 = {recorded}\n  \
         {} actually hashes to        {actual}\n\
         Either the signature was hand-edited, or only one of the two files reached the \
         commit (recording performs two renames and cannot make them atomic). Both are \
         fixed the same way: re-run `cargo xtask record-fixtures --only model-signature` \
         and commit BOTH files together.",
        signature_path.display()
    );

    // Secondary, and deliberately redundant with `pc-models`'
    // `the_recorded_model_signature_was_taken_from_the_declared_artifact`: provenance names
    // the source model's digest as well, and that is the one field a hand-edit could
    // desynchronise without disturbing `output_sha256` above.
    let recorded_source = provenance["source_sha256"]
        .as_str()
        .expect("PROVENANCE.json records `source_sha256` as a string");
    let signature = comic_text_detector_signature();
    assert!(
        recorded_source.eq_ignore_ascii_case(&signature.sha256),
        "PROVENANCE.json records source_sha256 = {recorded_source}, but the signature it \
         sits beside describes {}",
        signature.sha256
    );
}

#[test]
// spec §16.16: the committed file parses into the typed schema, and the schema version is
// pinned so a future probe format change is a loud failure rather than a silent
// field-defaulting one.
fn the_committed_signature_parses_into_the_typed_schema() {
    let signature = comic_text_detector_signature();

    assert_eq!(signature.schema_version, 1);
    assert!(!signature.model_file_name.is_empty());
    assert!(!signature.sha256.is_empty());
    assert!(signature.size_bytes > 0);
    assert!(signature.opset > 0);
    assert!(!signature.inputs.is_empty());
    assert!(!signature.outputs.is_empty());
    for tensor in signature.inputs.iter().chain(signature.outputs.iter()) {
        assert!(!tensor.name.is_empty());
        assert!(!tensor.element_type.is_empty());
        assert!(
            !tensor.shape.is_empty(),
            "`{}` recorded an empty shape",
            tensor.name
        );
    }
}

#[test]
// The named accessor is the whole reason consumers do not index by position when they mean
// to look something up by name. This convenience helper must point at the comic-text-
// detector signature, not merely return the same file as the general loader.
fn the_named_convenience_loader_matches_the_general_one() {
    let signature = comic_text_detector_signature();

    assert_eq!(signature.model_file_name, "comictextdetector.pt.onnx");
    assert_eq!(signature.input("images").shape, vec![1, 3, 1024, 1024]);
    assert_eq!(signature.output("blk").shape, vec![1, 64512, 7]);
    assert_eq!(signature.output("seg").shape, vec![1, 1, 1024, 1024]);
    assert_eq!(signature.output("det").shape, vec![1, 2, 1024, 1024]);
}

#[test]
// Lookup by name resolves the tensors `d4_signature.rs` asks for, and the recorded order
// is preserved on parse (spec §8.3 step 3 binds outputs BY INDEX, so a loader that sorted
// or reordered would quietly invalidate that instruction).
fn lookup_by_name_resolves_and_parse_preserves_recorded_order() {
    let signature = comic_text_detector_signature();

    assert_eq!(signature.input("images").name, "images");
    for name in ["blk", "seg", "det"] {
        assert_eq!(signature.output(name).name, name);
    }
    assert_eq!(signature.outputs[0].name, signature.output("blk").name);
    assert_eq!(signature.outputs[1].name, signature.output("seg").name);
    assert_eq!(signature.outputs[2].name, signature.output("det").name);
}

#[test]
// Per this crate's panic policy (a fixture problem panics with a diagnostic rather than
// returning `Result`), an unknown tensor name must list the valid ones -- a typo in a
// stage test is then self-diagnosing.
#[should_panic(expected = "blk")]
fn an_unknown_output_name_panics_listing_the_recorded_names() {
    // The requested name is deliberately NOT a substring of any valid one (`"blks"` would
    // contain `"blk"`, so the assertion would pass on a message that merely echoed the
    // key). Expecting `blk` therefore proves the panic really enumerates what IS recorded.
    let _ = comic_text_detector_signature().output("mask");
}

#[test]
// spec §16.16: the signature is a COMMITTED artifact, so its absence is a broken checkout.
// It must be loaded through the panicking `recorded` helper and never through
// `recorded_opt` -- a skip-gated version of `d4_signature.rs` would silently restore the
// self-reference that file exists to close.
#[should_panic(expected = "record-fixtures")]
fn a_missing_signature_panics_with_the_xtask_hint_rather_than_skipping() {
    let _ = load_model_signature("model_signature/definitely_not_recorded.signature.json");
}

#[test]
// The element-type spelling is pinned in exactly ONE place -- `ELEMENT_TYPE_F32` -- so that
// `d4_signature.rs` can assert the semantic fact ("this tensor is f32") through `is_f32()`.
// This test must nevertheless prove both sides of that contract: the recorded tensors really
// carry the spelling, and `is_f32()` is not a constant that happens to pass for this fixture.
fn the_element_type_spelling_is_pinned_once_and_is_f32_agrees_with_it() {
    assert_eq!(ELEMENT_TYPE_F32, "f32");

    let signature = comic_text_detector_signature();
    for tensor in signature.inputs.iter().chain(signature.outputs.iter()) {
        assert!(
            tensor.is_f32(),
            "`{}` recorded element type `{}`",
            tensor.name,
            tensor.element_type
        );
        assert_eq!(
            tensor.element_type, ELEMENT_TYPE_F32,
            "`{}` must be recorded as f32",
            tensor.name
        );
    }

    let non_f32 = model_signature::TensorSignature {
        name: "not-f32".to_owned(),
        element_type: "i64".to_owned(),
        shape: vec![1],
    };
    assert!(!non_f32.is_f32(), "is_f32() must inspect element_type");
}

#[test]
// Guards a real hazard in a hand-written schema: `model_signature` must not silently
// accept a JSON document that is missing fields the consumers rely on. `serde` without
// `#[serde(default)]` gives this, and this test is what keeps a well-meaning `default`
// attribute from being added later.
fn a_signature_missing_required_fields_is_rejected() {
    let incomplete = r#"{"schema_version": 1, "model_file_name": "x.onnx"}"#;

    assert!(
        serde_json::from_str::<model_signature::ModelSignature>(incomplete).is_err(),
        "a partial signature must not deserialize with defaulted tensors"
    );
}
