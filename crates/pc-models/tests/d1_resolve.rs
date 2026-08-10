//! Task D1 -- spec §8.3 step 3 (the model's URL and digest), §6 (`text_detector.model_path`),
//! §13.1 (`--model-path FILE`), §8.5's D1 row. FROZEN per CLAUDE.md.
//!
//! No test in this file touches the network or the user's real cache directory.

mod common;

use common::{FAKE_SPEC, PAYLOAD, PAYLOAD_SHA256};
use pc_core::StageError;
use pc_models::{
    resolve, sha256_hex, verify_sha256, ModelError, Requirement, Resolution, VerifyStatus,
    COMIC_TEXT_DETECTOR, LAMA_MANGA_INPAINTER, MANGA_OCR_DECODER, MANGA_OCR_ENCODER,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use tempfile::TempDir;

/// The registry entry names at a given [`Requirement`], as a SET — cardinality is not
/// identity (cookbook rule 13), so a swap between the two partitions must be visible.
fn names_at(requirement: Requirement) -> BTreeSet<&'static str> {
    pc_models::ALL
        .iter()
        .filter(|spec| spec.requirement == requirement)
        .map(|spec| spec.name)
        .collect()
}

fn names_of(specs: &[&'static pc_models::ModelSpec]) -> BTreeSet<&'static str> {
    specs.iter().map(|spec| spec.name).collect()
}

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
fn manga_ocr_encoder_spec_matches_the_spec_url_and_digest() {
    // spec §16.30 item 1, transcribed literally: upstream's encoder artifact.
    // Frozen here because a typo in either string turns every download into a confusing
    // HashMismatch (or a 404) rather than an obvious mistake.
    assert_eq!(
        MANGA_OCR_ENCODER.url,
        "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/24b12778d85800835e2ca409236de281b8ab7b9f/encoder_model.onnx"
    );
    assert_eq!(
        MANGA_OCR_ENCODER.sha256,
        "15fa8155fe9bc1a7d25d9bb353debaa4def033d0174e907dbd2dd6d995def85f"
    );
    assert_eq!(MANGA_OCR_ENCODER.file_name, "encoder_model.onnx");
    assert_eq!(MANGA_OCR_ENCODER.name, "manga-ocr-encoder");
}

#[test]
fn manga_ocr_decoder_spec_matches_the_spec_url_and_digest() {
    // spec §16.30 item 1, transcribed literally: upstream's decoder artifact.
    // Frozen here because a typo in either string turns every download into a confusing
    // HashMismatch (or a 404) rather than an obvious mistake.
    assert_eq!(
        MANGA_OCR_DECODER.url,
        "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/24b12778d85800835e2ca409236de281b8ab7b9f/decoder_model.onnx"
    );
    assert_eq!(
        MANGA_OCR_DECODER.sha256,
        "ef7765261e9d1cdc34d89356986c2bbc2a082897f753a89605ae80fdfa61f5e8"
    );
    assert_eq!(MANGA_OCR_DECODER.file_name, "decoder_model.onnx");
    assert_eq!(MANGA_OCR_DECODER.name, "manga-ocr-decoder");
}

#[test]
fn lama_manga_inpainter_spec_matches_the_ratified_artifact_pin() {
    // spec §16.38 item 1(a), transcribed literally: the L0 spike's pinned revision of
    // `mayocream/koharu`, whose size and digest were re-verified for that entry against
    // Hugging Face's `X-Linked-Size` / `X-Linked-ETag` headers. Frozen here for the same
    // reason as the three siblings above: a typo in either string turns every download into
    // a confusing HashMismatch or a 404 rather than an obvious mistake. Registered at §14
    // as DEVIATION(25), the substitution for upstream's TorchScript `.pt`.
    assert_eq!(
        LAMA_MANGA_INPAINTER.url,
        "https://huggingface.co/mayocream/koharu/resolve/15439cba09df388c51de6e47c6020bc31edab41f/lama-manga.onnx"
    );
    assert_eq!(
        LAMA_MANGA_INPAINTER.sha256,
        "50a1abae0d73bd46d08eae36c8590cd59ad09029494c9698702b050ef00b0100"
    );
    assert_eq!(LAMA_MANGA_INPAINTER.file_name, "lama-manga.onnx");
    assert_eq!(LAMA_MANGA_INPAINTER.name, "lama-manga-inpainter");
    // The revision is what makes the URL reproducible: a `main`-branch URL would silently
    // start serving different bytes and the digest above would become a mystery failure.
    assert!(
        LAMA_MANGA_INPAINTER
            .url
            .contains("/resolve/15439cba09df388c51de6e47c6020bc31edab41f/"),
        "the URL must pin the ratified revision, not a branch: {}",
        LAMA_MANGA_INPAINTER.url
    );
}

#[test]
fn the_required_optional_partition_is_pinned_by_name() {
    // spec §16.38 item 19 (decision D1), and the specific graft the Fable tie-break made a
    // condition of the ruling: the partition is pinned BY NAME, so a future edit that flips
    // any one entry's `Requirement` turns this red instead of silently changing what
    // `models download` fetches by default (or silently adding 207 MB to every user's).
    //
    // Sets, not counts: a swap — the detector marked `Optional` and LaMa marked `Required`
    // — keeps both cardinalities and is exactly the mistake this exists to catch.
    assert_eq!(
        names_at(Requirement::Required),
        BTreeSet::from([
            "comic-text-detector",
            "manga-ocr-encoder",
            "manga-ocr-decoder",
            "lama-manga-inpainter",
        ]),
        "§16.46 item 11(b): every model a DEFAULT run needs must be Required, and item 1(b) turns inpainting on by default, so that now includes the LaMa weights"
    );
    assert_eq!(
        names_at(Requirement::Optional),
        BTreeSet::new(),
        "§16.46 item 11(c): the Optional side is now EMPTY -- the partition machinery and the `--include-optional` flag stay in place, per that item, with no member"
    );

    // Anti-vacuity: a hard-coded total that cannot be computed from the registry, so the
    // two set assertions above cannot both pass over an empty or truncated `ALL`.
    assert_eq!(pc_models::ALL.len(), 4);

    // And the field is stated per entry, not inferred: assert the two constants directly,
    // so this test still fails if `ALL` stops containing one of them.
    assert_eq!(COMIC_TEXT_DETECTOR.requirement, Requirement::Required);
    assert_eq!(MANGA_OCR_ENCODER.requirement, Requirement::Required);
    assert_eq!(MANGA_OCR_DECODER.requirement, Requirement::Required);
    assert_eq!(LAMA_MANGA_INPAINTER.requirement, Requirement::Required);
}

#[test]
fn download_without_the_flag_selects_exactly_the_required_models() {
    // §16.38 item 19(b): `models download` with no flag fetches only `Required` models.
    // The set is written out literally rather than derived from `Requirement::Required`,
    // so this cannot pass by `selected` and `names_at` sharing one bug.
    //
    // §16.46 item 11(b) makes that set ALL FOUR entries: a fresh install's plain
    // `models download` now fetches the 207,482,644-byte LaMa artifact. That is the point of
    // the promotion -- inpainting is on by default, so a run that cannot reach the weights
    // aborts the whole batch (§16.38 item 9: exit 1, zero pages exported).
    assert_eq!(
        names_of(&pc_models::selected(false)),
        BTreeSet::from([
            "comic-text-detector",
            "manga-ocr-encoder",
            "manga-ocr-decoder",
            "lama-manga-inpainter",
        ])
    );
    assert_eq!(pc_models::selected(false).len(), 4);
}

#[test]
fn download_with_the_flag_selects_every_registry_entry() {
    // §16.38 item 19(b): with `--include-optional`, `Required` + `Optional`.
    assert_eq!(
        names_of(&pc_models::selected(true)),
        BTreeSet::from([
            "comic-text-detector",
            "manga-ocr-encoder",
            "manga-ocr-decoder",
            "lama-manga-inpainter",
        ])
    );
    assert_eq!(pc_models::selected(true).len(), 4);
}

#[test]
fn the_skipped_set_is_empty_at_both_flag_settings_now_that_nothing_is_optional() {
    // §16.38 item 19(c) requires `models download` to NAME what it left out. §16.46 item
    // 11(c) empties the Optional side, so it now leaves out nothing at either setting.
    //
    // A COVERAGE LOSS, recorded rather than hidden: with no Optional entry in the real
    // registry, `skipped`'s non-empty branch and `models download`'s "SKIPPED (optional)"
    // line are no longer exercised by anything that goes through `ALL`, and
    // `--include-optional` is a no-op for every current entry. The machinery is kept
    // deliberately (§16.46 item 11(c)) against a future optional model; it is simply
    // unexercised until there is one. `selected_and_skipped_partition_the_whole_registry_at_both_flag_settings`
    // still holds and is now the only thing binding the two functions together over `ALL`.
    // The Optional SEMANTICS are not lost with it: `l3_verify.rs` drives `VerifyStatus::absent`
    // and the whole verify path through SYNTHETIC `Required`/`Optional` specs, so the
    // non-failing-absence branch stays covered by tests that never read `ALL`.
    assert_eq!(names_of(&pc_models::skipped(false)), BTreeSet::new());
    assert_eq!(names_of(&pc_models::skipped(true)), BTreeSet::new());
}

#[test]
fn an_absent_lama_artifact_now_fails_models_verify_instead_of_being_reported_as_optional() {
    // §16.46 item 11(b), the user-visible half of the promotion. This is the behaviour the
    // promotion exists for, and it is asserted through the REAL registry entry rather than a
    // synthetic spec, because what changed is that entry's `requirement` field.
    //
    // `VerifyStatus::absent` is the one place `Requirement` reaches the verify policy, and
    // `pc-cli`'s `run_models` sets `all_ok = false` on any `is_failure()` status and returns
    // `EXIT_FATAL`. So this pair is what makes `panel-ocr models verify` exit non-zero on a
    // machine that never fetched the weights -- where before the promotion the row read
    // "NOT INSTALLED (optional)" and the command exited 0.
    //
    // Turns red if the entry is demoted back to `Optional` without a ratification, or if
    // `absent` stops routing `Required` to a failing status.
    let status = VerifyStatus::absent(LAMA_MANGA_INPAINTER.requirement);
    assert_eq!(status, VerifyStatus::Missing);
    assert!(status.is_failure());
    assert_eq!(status.label(), "MISSING");

    // The control, so the assertions above are not satisfied by an `absent` that fails for
    // every requirement: the other arm must still be the non-failing one.
    let optional = VerifyStatus::absent(Requirement::Optional);
    assert_eq!(optional, VerifyStatus::NotInstalled);
    assert!(!optional.is_failure());
}

#[test]
fn selected_and_skipped_partition_the_whole_registry_at_both_flag_settings() {
    // Cookbook rule 13's bidirectional-coverage question, asked of the selector: enumerate
    // BOTH populations independently and assert they cover `ALL` with no overlap. Without
    // this, a model could be dropped from `selected` and never appear in `skipped` either —
    // silently unfetchable and unreported, which is the one outcome neither option of D1
    // permitted.
    let everything: BTreeSet<&str> = pc_models::ALL.iter().map(|spec| spec.name).collect();

    for include_optional in [false, true] {
        let selected = names_of(&pc_models::selected(include_optional));
        let skipped = names_of(&pc_models::skipped(include_optional));

        assert!(
            selected.is_disjoint(&skipped),
            "include_optional={include_optional}: a model is both selected and skipped: {:?}",
            selected.intersection(&skipped).collect::<Vec<_>>()
        );
        assert_eq!(
            selected.union(&skipped).copied().collect::<BTreeSet<_>>(),
            everything,
            "include_optional={include_optional}: selected + skipped must cover the registry"
        );
    }
}

#[test]
fn the_manga_ocr_encoder_registry_entry_matches_the_pc_testkit_pin() {
    let pin = pc_testkit::ocr_model_signature::MANGA_OCR_ENCODER;

    assert_eq!(MANGA_OCR_ENCODER.file_name, pin.file_name);
    assert_eq!(MANGA_OCR_ENCODER.sha256, pin.sha256);
}

#[test]
fn the_manga_ocr_decoder_registry_entry_matches_the_pc_testkit_pin() {
    let pin = pc_testkit::ocr_model_signature::MANGA_OCR_DECODER;

    assert_eq!(MANGA_OCR_DECODER.file_name, pin.file_name);
    assert_eq!(MANGA_OCR_DECODER.sha256, pin.sha256);
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
