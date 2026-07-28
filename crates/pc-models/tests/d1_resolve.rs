//! Task D1 -- spec §8.3 step 3 (the model's URL and digest), §6 (`text_detector.model_path`),
//! §13.1 (`--model-path FILE`), §8.5's D1 row. FROZEN per CLAUDE.md.
//!
//! No test in this file touches the network or the user's real cache directory.

mod common;

use common::{FAKE_SPEC, PAYLOAD, PAYLOAD_SHA256};
use pc_core::StageError;
use pc_models::{resolve, sha256_hex, verify_sha256, ModelError, Resolution, COMIC_TEXT_DETECTOR};
use std::path::PathBuf;
use tempfile::TempDir;

// ------------------------------------------------------------ the declared models

#[test]
fn comic_text_detector_spec_matches_the_spec_url_and_digest() {
    // spec §8.3 step 3, transcribed literally: upstream's `comictextdetector.pt.onnx`.
    // Frozen here because a typo in either string turns every download into a confusing
    // HashMismatch (or a 404) rather than an obvious mistake.
    assert_eq!(
        COMIC_TEXT_DETECTOR.url,
        "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
    );
    assert_eq!(
        COMIC_TEXT_DETECTOR.sha256,
        "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
    );
    assert_eq!(COMIC_TEXT_DETECTOR.file_name, "comictextdetector.pt.onnx");
    assert!(
        !COMIC_TEXT_DETECTOR.name.is_empty(),
        "errors quote the name"
    );
}

#[test]
fn the_recorded_model_signature_was_taken_from_the_declared_artifact() {
    // Task F3 / spec §16.16 -- THE KEYSTONE of the recorded signature.
    //
    // `pc-detect`'s `d4_signature.rs` asserts the model's tensor arity against
    // `tests/fixtures/recorded/model_signature/`, which is what finally ties
    // `yolo::ROW_STRIDE` to the shipped artifact instead of to our reading of §8.3 step 3.
    // That whole chain is worthless unless the signature was recorded from *this* model:
    // without this assertion the JSON is unfalsifiable and could describe any export.
    //
    // It lives here rather than in `pc-detect` because this crate owns
    // `COMIC_TEXT_DETECTOR.sha256` and spec §1 (as amended) forbids `pc-detect` from
    // depending on `pc-models`. It also catches the reverse staleness: if
    // `COMIC_TEXT_DETECTOR` is ever repointed at a new model file, this fails until the
    // signature is re-recorded, so the arity assertions can never silently describe the
    // previous artifact.
    //
    // Runnable offline with no weights: the digest is embedded in the ~300-byte signature,
    // and the probe verified it against the real 90 MB file at record time.
    let signature = pc_testkit::model_signature::comic_text_detector_signature();

    assert!(
        signature
            .sha256
            .eq_ignore_ascii_case(COMIC_TEXT_DETECTOR.sha256),
        "the recorded signature describes sha256 {}, but COMIC_TEXT_DETECTOR declares {}; \
         re-record with `cargo xtask record-fixtures --only model-signature`",
        signature.sha256,
        COMIC_TEXT_DETECTOR.sha256
    );
    assert_eq!(signature.model_file_name, COMIC_TEXT_DETECTOR.file_name);
}

#[test]
fn all_lists_every_model_the_cli_can_manage() {
    // spec §13.1's `models download|verify` must iterate a registry rather than accreting
    // per-model knowledge in `pc-cli` (§1 rule 3: no algorithm or policy code there).
    // Task P7's manga-ocr weights join this list without touching `pc-cli`.
    assert!(
        pc_models::ALL
            .iter()
            .any(|spec| spec.url == COMIC_TEXT_DETECTOR.url),
        "the detector model must be reachable through the registry"
    );

    let mut file_names: Vec<&str> = pc_models::ALL.iter().map(|spec| spec.file_name).collect();
    let declared = file_names.len();
    file_names.sort_unstable();
    file_names.dedup();
    assert_eq!(
        file_names.len(),
        declared,
        "cache file names must be unique: two models sharing one entry would clobber \
         each other"
    );
}

#[test]
fn every_declared_digest_is_lowercase_hex_of_the_right_length() {
    // `verify_sha256` compares case-insensitively, but a *declared* digest that is not
    // 64 lowercase hex characters is a transcription error, not a style choice.
    for spec in pc_models::ALL {
        assert_eq!(spec.sha256.len(), 64, "{}", spec.name);
        assert!(
            spec.sha256
                .chars()
                .all(|character| matches!(character, '0'..='9' | 'a'..='f')),
            "{} has a non-lowercase-hex digest",
            spec.name
        );
        assert!(!spec.name.is_empty());
        assert!(!spec.file_name.is_empty());
        assert!(!spec.url.is_empty());
    }
}

// ------------------------------------------------------------ path resolution

#[test]
fn resolve_without_an_override_points_into_the_managed_cache() {
    // spec §6: `text_detector.model_path = ""` means "use the managed cache". The cache
    // directory itself is `pc-cli`'s to choose (`paths::models_dir`), so it arrives as a
    // parameter -- this crate never consults the environment.
    let (_root, models) = common::models_dir();

    let resolved = resolve(&FAKE_SPEC, &models, None).expect("resolution succeeds");

    assert_eq!(
        resolved,
        Resolution::Missing(models.join(FAKE_SPEC.file_name))
    );
}

#[test]
fn resolve_reports_an_existing_cache_entry_as_cached() {
    // `Cached` means "a file is present at the managed path" and nothing more -- digest
    // checking belongs to `ensure_available`, which is the layer that can act on a
    // mismatch by re-downloading. Resolution stays a pure query.
    let (_root, models) = common::models_dir();
    let dest = common::dest_of(&models);
    std::fs::write(&dest, PAYLOAD).expect("seed the cache entry");

    assert_eq!(
        resolve(&FAKE_SPEC, &models, None).expect("resolution succeeds"),
        Resolution::Cached(dest)
    );
}

#[test]
fn resolve_reports_a_corrupt_cache_entry_as_cached_not_missing() {
    // Same contract, stated as the case that could plausibly have gone the other way: a
    // present-but-wrong file is `Cached`, so the digest failure is reported by the layer
    // that can recover from it rather than being disguised as absence.
    let (_root, models) = common::models_dir();
    let dest = common::dest_of(&models);
    std::fs::write(&dest, b"truncated garbage").expect("seed a corrupt entry");

    assert_eq!(
        resolve(&FAKE_SPEC, &models, None).expect("resolution succeeds"),
        Resolution::Cached(dest)
    );
}

#[test]
fn resolve_honours_an_explicit_override_verbatim() {
    // spec §13.1's `--model-path FILE` and §6's `text_detector.model_path`: an explicit
    // path is used exactly as given -- never relocated into the cache, never downloaded.
    let root = TempDir::new().expect("temp dir");
    let elsewhere = root.path().join("my-own-export.onnx");
    std::fs::write(&elsewhere, b"a locally exported model").expect("write the override");
    let unrelated_cache = root.path().join("cache/models");

    let resolved =
        resolve(&FAKE_SPEC, &unrelated_cache, Some(&elsewhere)).expect("resolution succeeds");

    assert_eq!(resolved, Resolution::Override(elsewhere.clone()));
    assert_eq!(resolved.path(), elsewhere.as_path());
}

#[test]
fn resolve_rejects_an_override_that_does_not_exist() {
    // spec §16.13 item 4's principle applied to a model file: a missing file the user
    // explicitly named is a reported error. It must never fall back to the managed copy
    // or to a download -- silently using a different model than the one requested is the
    // worst available outcome.
    let (_root, models) = common::models_dir();
    let missing = models.join("definitely-not-here.onnx");

    let error = resolve(&FAKE_SPEC, &models, Some(&missing)).expect_err("the override is absent");

    assert!(matches!(error, ModelError::OverrideMissing { .. }));
    assert!(
        error.to_string().contains("definitely-not-here.onnx"),
        "the message must name the path the user gave: {error}"
    );
}

#[test]
fn resolve_does_not_create_the_models_directory() {
    // §5.3 makes "cache dir not creatable" a *fatal startup* condition, decided once at
    // startup. Resolution is a pure query and must not have that side effect -- among
    // other things `models path` and `--model-path` must work read-only.
    let root = TempDir::new().expect("temp dir");
    let models = root.path().join("models");

    let resolved = resolve(&FAKE_SPEC, &models, None).expect("resolution succeeds");

    assert_eq!(
        resolved,
        Resolution::Missing(models.join(FAKE_SPEC.file_name))
    );
    assert!(!models.exists(), "resolution must not mkdir");
}

// ------------------------------------------------------------ sha256

#[test]
fn sha256_hex_matches_the_published_test_vectors() {
    // FIPS 180-4 vectors, so this pins the digest itself and not merely self-consistency.
    let root = TempDir::new().expect("temp dir");
    let empty = root.path().join("empty");
    let abc = root.path().join("abc");
    std::fs::write(&empty, b"").expect("write empty");
    std::fs::write(&abc, b"abc").expect("write abc");

    assert_eq!(
        sha256_hex(&empty).expect("hashable"),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(&abc).expect("hashable"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn sha256_hex_returns_lowercase_hex() {
    let root = TempDir::new().expect("temp dir");
    let path = root.path().join("payload");
    std::fs::write(&path, PAYLOAD).expect("write payload");

    let digest = sha256_hex(&path).expect("hashable");

    assert_eq!(digest, PAYLOAD_SHA256);
    assert_eq!(digest, digest.to_lowercase());
}

#[test]
fn sha256_hex_of_a_missing_file_is_an_io_error() {
    let root = TempDir::new().expect("temp dir");

    let error = sha256_hex(&root.path().join("nope")).expect_err("no such file");

    assert!(matches!(error, ModelError::Io { .. }));
}

#[test]
fn verify_sha256_is_case_insensitive_and_reports_both_digests() {
    // Case-insensitive so an uppercase digest pasted into a config is not a spurious
    // mismatch; both digests in the message so "which one is wrong" is answerable
    // without re-running anything.
    let root = TempDir::new().expect("temp dir");
    let path = root.path().join("payload");
    std::fs::write(&path, PAYLOAD).expect("write payload");
    let wrong = "0".repeat(64);

    verify_sha256(&path, PAYLOAD_SHA256).expect("the digest matches");
    verify_sha256(&path, &PAYLOAD_SHA256.to_uppercase()).expect("ASCII-case-insensitive");

    let error = verify_sha256(&path, &wrong).expect_err("the digest does not match");

    assert!(matches!(error, ModelError::HashMismatch { .. }));
    let message = error.to_string();
    assert!(
        message.contains(&wrong),
        "must quote the expected digest: {message}"
    );
    assert!(
        message.contains(PAYLOAD_SHA256),
        "must quote the actual digest: {message}"
    );
}

// ------------------------------------------------------------ error plumbing

#[test]
fn model_errors_convert_into_stage_errors_without_losing_the_message() {
    // `ModelError` is this crate's own typed error (so tests match variants instead of
    // string-matching), and §2.9's `StageError::Model(String)` is the variant it surfaces
    // through for the consumers in §1's graph. §5.3 then treats it as fatal.
    let error = ModelError::OverrideMissing {
        path: PathBuf::from("/nope/model.onnx"),
    };
    let message = error.to_string();

    let converted: StageError = error.into();

    match converted {
        StageError::Model(text) => assert_eq!(text, message),
        other => panic!("expected StageError::Model, got {other:?}"),
    }
}
