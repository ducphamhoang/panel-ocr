//! Task **L3** — `models verify`'s optionality policy (spec §13.1 as superseded by
//! §16.38 item 19, clause (d)). FROZEN per CLAUDE.md.
//!
//! **No test in this file touches the network.** `pc_models::verify` takes no
//! `ModelFetcher`, so there is no transport in reach at all — that is the point of
//! extracting it from `pc-cli`'s loop, and it is what keeps §16.18 item 3's prohibition on
//! test-reachable network I/O satisfied while the policy becomes assertable.
//!
//! The specs below are local literals rather than the real registry entries: the real ones
//! are 90-343 MB artifacts, and what is under test is the classification, not the bytes.

mod common;

use common::{PAYLOAD, PAYLOAD_SHA256};
use pc_models::{verify, ModelSpec, Requirement, VerifyStatus};

/// A `Required` model whose payload is the 28-byte fake. Distinct `file_name` from
/// [`optional`]'s, so both can be seeded into one `models/` directory.
const fn required() -> ModelSpec {
    ModelSpec {
        name: "fake-required",
        file_name: "fake-required.onnx",
        // RFC 6761's reserved TLD: a test that somehow reached a transport would fail DNS
        // rather than silently hit the network. Nothing here has a transport to reach.
        url: "https://example.invalid/fake-required.onnx",
        sha256: PAYLOAD_SHA256,
        requirement: Requirement::Required,
    }
}

/// The same artifact declared `Optional` — the shape of the LaMa registry entry.
const fn optional() -> ModelSpec {
    ModelSpec {
        name: "fake-optional",
        file_name: "fake-optional.onnx",
        url: "https://example.invalid/fake-optional.onnx",
        sha256: PAYLOAD_SHA256,
        requirement: Requirement::Optional,
    }
}

// ------------------------------------------------------------ absence

#[test]
fn a_required_model_absent_from_the_cache_is_a_failure_labelled_missing() {
    // THE REGRESSION LOCK named by §16.38 item 19(d) as the specific way this change could
    // go wrong: `models verify` set `all_ok = false` on ANY `Missing` before optionality
    // existed, and the whole point of the flag is that adding a non-failing absent state
    // must not weaken this one. If this goes green while `Missing` stops being a failure,
    // every CI preflight that relies on `models verify` silently starts passing with no
    // detector weights at all.
    let (_root, models) = common::models_dir();

    let verification = verify(&required(), &models, None);

    assert_eq!(verification.status, VerifyStatus::Missing);
    assert!(
        verification.status.is_failure(),
        "an absent Required model must clear all_ok and exit EXIT_FATAL"
    );
    assert_eq!(verification.status.label(), "MISSING");
    assert_eq!(verification.path, models.join("fake-required.onnx"));
}

#[test]
fn an_optional_model_absent_from_the_cache_is_reported_and_is_not_a_failure() {
    // §16.38 item 19(d): the NEW state. Absent + `Optional` is reported, does not clear
    // `all_ok`, and does not exit `EXIT_FATAL`.
    let (_root, models) = common::models_dir();

    let verification = verify(&optional(), &models, None);

    assert_eq!(verification.status, VerifyStatus::NotInstalled);
    assert!(
        !verification.status.is_failure(),
        "an absent Optional model must NOT clear all_ok: {:?}",
        verification.status
    );
    assert_eq!(verification.path, models.join("fake-optional.onnx"));
}

#[test]
fn the_absent_optional_label_is_textually_distinguishable_from_missing() {
    // §16.38 item 19(d) requires the new status not to use the literal string `MISSING`, so
    // a script grepping the old word cannot mistake an optional model it never asked for
    // for a required one that vanished. Asserted as a substring rule, not just inequality:
    // `MISSING (optional)` would pass an `!=` check and still break every such script.
    let absent_optional = VerifyStatus::absent(Requirement::Optional);
    let absent_required = VerifyStatus::absent(Requirement::Required);

    assert_eq!(absent_optional.label(), "NOT INSTALLED (optional)");
    assert!(
        !absent_optional.label().contains("MISSING"),
        "the optional-absent label must not contain the required-absent word: {}",
        absent_optional.label()
    );
    assert_eq!(absent_required.label(), "MISSING");
    assert_ne!(absent_optional.label(), absent_required.label());
}

// ------------------------------------------------------------ corruption

#[test]
fn an_optional_model_present_with_the_wrong_bytes_is_still_a_failure() {
    // §16.38 item 19(d), the clause that keeps optionality narrow: "optionality licenses
    // absence only, never corruption." A present-but-wrong optional file is a `Requirement`
    // -independent failure — the user has a broken artifact on disk, and silently reporting
    // it as fine is worse than reporting a model they never installed.
    let (_root, models) = common::models_dir();
    let spec = optional();
    std::fs::write(models.join(spec.file_name), b"truncated garbage")
        .expect("seed a corrupt entry");

    let verification = verify(&spec, &models, None);

    match &verification.status {
        VerifyStatus::HashMismatch { actual, expected } => {
            assert_eq!(expected, PAYLOAD_SHA256);
            assert_ne!(actual, PAYLOAD_SHA256);
        }
        other => panic!("expected a hash mismatch, got {other:?}"),
    }
    assert!(
        verification.status.is_failure(),
        "a corrupt Optional model must clear all_ok"
    );
    assert_eq!(verification.status.label(), "HASH MISMATCH");
}

#[test]
fn an_optional_model_present_with_the_wrong_size_is_still_a_failure() {
    // Same clause, the other corruption channel. The size check runs before the digest, so
    // a truncated download is reported as a size mismatch — with both numbers, so "which
    // one is wrong" is answerable without re-running anything.
    let (_root, models) = common::models_dir();
    let spec = optional();
    std::fs::write(models.join(spec.file_name), PAYLOAD).expect("seed the entry");

    let verification = verify(&spec, &models, Some(PAYLOAD.len() as u64 + 1));

    assert_eq!(
        verification.status,
        VerifyStatus::SizeMismatch {
            actual: PAYLOAD.len() as u64,
            expected: PAYLOAD.len() as u64 + 1,
        }
    );
    assert!(verification.status.is_failure());
    assert_eq!(verification.status.label(), "SIZE MISMATCH");
}

#[test]
fn a_required_model_present_with_the_wrong_bytes_is_a_failure() {
    // The control for the two above: corruption reporting is unchanged for `Required`
    // models, so nothing about the new state altered the pre-existing path.
    let (_root, models) = common::models_dir();
    let spec = required();
    std::fs::write(models.join(spec.file_name), b"truncated garbage")
        .expect("seed a corrupt entry");

    let verification = verify(&spec, &models, None);

    assert!(matches!(
        verification.status,
        VerifyStatus::HashMismatch { .. }
    ));
    assert!(verification.status.is_failure());
}

// ------------------------------------------------------------ the happy path

#[test]
fn a_present_model_with_the_declared_bytes_verifies_ok_at_either_requirement() {
    // Both requirements, so the `Ok` path is proven not to have picked up an optionality
    // branch: an implementation that reported `NotInstalled` for any `Optional` model, or
    // skipped the digest check for one, would fail here.
    let (_root, models) = common::models_dir();
    for spec in [required(), optional()] {
        std::fs::write(models.join(spec.file_name), PAYLOAD).expect("seed the entry");

        let verification = verify(&spec, &models, Some(PAYLOAD.len() as u64));

        assert_eq!(verification.status, VerifyStatus::Ok, "{}", spec.name);
        assert!(!verification.status.is_failure(), "{}", spec.name);
        assert_eq!(verification.status.label(), "OK");
        assert_eq!(verification.path, models.join(spec.file_name));
    }
}

#[test]
fn verify_creates_nothing_and_modifies_nothing() {
    // `models verify` is documented as checking "without modifying the cache" (§13.1), and
    // §5.3 makes "cache dir not creatable" a startup decision — so verification must not
    // mkdir either. Both models are absent and the directory itself does not exist.
    let root = tempfile::TempDir::new().expect("temp dir");
    let models = root.path().join("models");

    assert_eq!(
        verify(&required(), &models, None).status,
        VerifyStatus::Missing
    );
    assert_eq!(
        verify(&optional(), &models, None).status,
        VerifyStatus::NotInstalled
    );

    assert!(!models.exists(), "verification must not mkdir");
    assert_eq!(common::entries(root.path()), Vec::<String>::new());
}

// ------------------------------------------------------------ the exit-code rule

#[test]
fn not_installed_is_the_only_non_ok_status_that_is_not_a_failure() {
    // The exit-code rule stated over the whole enum rather than one variant at a time.
    // Enumerated literally: this list is the gate's input set, and cookbook rule 13's
    // question — "what does it iterate to find them?" — is answered by a hand-written
    // table, nothing derived from the enum.
    //
    // The limit of that table, stated rather than implied: `is_failure` is
    // `!matches!(self, Self::Ok | Self::NotInstalled)`, which is NOT an exhaustive match,
    // so a new `VerifyStatus` variant silently becomes "is a failure" with no compile
    // error and without joining the list below. The compile-time backstop is elsewhere —
    // `VerifyStatus::label()` IS an exhaustive `match`, so adding a variant breaks the
    // build there and forces a reader past this file — but it is not `is_failure`, and it
    // does not force anyone to extend this table. What this test does pin is that the six
    // variants that exist today classify as written.
    let failures = [
        VerifyStatus::Missing,
        VerifyStatus::SizeMismatch {
            actual: 1,
            expected: 2,
        },
        VerifyStatus::HashMismatch {
            actual: "a".into(),
            expected: "b".into(),
        },
        VerifyStatus::Error("unreadable".into()),
    ];
    for status in &failures {
        assert!(status.is_failure(), "{status:?} must be a failure");
    }

    for status in [VerifyStatus::Ok, VerifyStatus::NotInstalled] {
        assert!(!status.is_failure(), "{status:?} must not be a failure");
    }
}
