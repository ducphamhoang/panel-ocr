//! Task **L5** — the inpainting provider's refusals, as a user reads them.
//!
//! These are integration tests on purpose: the messages here are the *only* thing a user sees
//! when an opt-in 207 MB artifact is absent or corrupt, and §16.38 item 19(b)'s ruling makes
//! this runtime check the reason a plain `models verify` is allowed to omit the optional row
//! at all. A test on a private helper would not prove the message reaches the boundary.
//!
//! The latch itself is unit-tested next to its `OnceLock`, in
//! `crates/pc-cli/src/inpainter.rs`, because the attempt counter it asserts on is deliberately
//! `#[cfg(test)]`-private.

use pc_cli::inpainter::{build_provider, InpainterProvider, UnavailableInpainterProvider};
use pc_core::device::Device;
use pc_core::StageError;
use tempfile::TempDir;

/// §16.38 item 13(a): `inpainting_enabled` defaults to `false`. A default run must not even
/// have a provider to ask, which is what keeps item 8(c)'s "inpaint nothing" run free.
#[test]
fn the_disabled_flag_yields_no_provider_at_all_rather_than_one_that_refuses() {
    let cache = TempDir::new().expect("temp dir");

    assert!(build_provider(false, None, cache.path(), Device::Cpu).is_none());
    assert!(
        build_provider(true, None, cache.path(), Device::Cpu).is_some(),
        "and the enabled flag does produce one, or the assertion above is vacuous"
    );
}

/// §16.38 item 9(g): construction failures are **run-fatal** and render as `StageError::Model`.
/// The provider declares that rather than leaving it to be inferred from a signature
/// (cookbook rule 4).
#[test]
fn every_provider_this_build_can_produce_declares_its_construction_failures_run_fatal() {
    let cache = TempDir::new().expect("temp dir");
    let provider =
        build_provider(true, None, cache.path(), Device::Cpu).expect("the flag is enabled");

    assert!(
        provider.failures_are_run_fatal(),
        "§16.38 item 9 ruled a missing or uninitializable inpainting model RUN-FATAL"
    );

    let refusing = UnavailableInpainterProvider::new("no backend here");
    assert!(refusing.failures_are_run_fatal());
    let Err(error) = refusing.inpainter() else {
        panic!("an unavailable provider must refuse");
    };
    assert!(matches!(error, StageError::Model(_)), "got {error:?}");
}

/// §16.38 item 19(g): the refusal hint for the **optional** artifact must name
/// `--include-optional`, because *"a user who has enabled `inpainting_enabled` and run
/// `models download` must not be left inferring why the stage still refuses"* (item 19(c)).
///
/// The expected substring comes from `pc_cli::models::models_download_optional_command`, which
/// is the function item 19(g) landed for exactly this caller — an independent oracle rather
/// than a second copy of the string.
#[test]
#[cfg(feature = "onnx")]
fn a_missing_optional_model_refuses_by_naming_the_include_optional_download() {
    let cache = TempDir::new().expect("temp dir");
    let provider =
        build_provider(true, None, cache.path(), Device::Cpu).expect("the flag is enabled");

    let Err(error) = provider.inpainter() else {
        panic!("nothing is cached in a fresh temp dir, so this must refuse");
    };

    let StageError::Model(message) = &error else {
        panic!("a missing model is run-fatal `Model`; got {error:?}");
    };
    assert!(
        message.contains("--include-optional"),
        "the hint must name the flag that fetches it; got {message}"
    );
    assert!(
        message.contains("lama-manga.onnx"),
        "and the artifact whose absence caused it; got {message}"
    );
}

/// §16.38 item 19(b)'s RULING, verbatim on the point this test exists for: *"L5's inpainter
/// runtime path must independently verify the LaMa artifact's integrity before use, regardless
/// of what `models verify` last reported, so a corrupt optional model yields a run-fatal
/// refusal at the stage rather than corrupted inpainting"*. Item 19(d) states the rule it
/// rests on: *"optionality licenses **absence only, never corruption**."*
///
/// A present-but-wrong file is therefore NOT treated as absent, and the two refusals are
/// distinguishable: this one names the digest mismatch. Without the runtime hash check this
/// test fails by getting the *missing*-model message — or worse, by succeeding.
#[test]
#[cfg(feature = "onnx")]
fn a_present_but_corrupt_optional_model_is_refused_for_its_digest_and_not_reported_as_missing() {
    let cache = TempDir::new().expect("temp dir");
    let models = cache.path().join("models");
    std::fs::create_dir_all(&models).expect("models dir");
    // Present, named exactly as the registry expects, and not the pinned artifact.
    std::fs::write(
        models.join(pc_models::LAMA_MANGA_INPAINTER.file_name),
        b"corrupt",
    )
    .expect("write a corrupt artifact");

    let provider =
        build_provider(true, None, cache.path(), Device::Cpu).expect("the flag is enabled");
    let Err(error) = provider.inpainter() else {
        panic!("a corrupt artifact must be refused");
    };

    let StageError::Model(message) = &error else {
        panic!("corruption is run-fatal `Model`; got {error:?}");
    };
    assert!(
        message.contains("sha256 mismatch"),
        "the refusal must name the integrity failure, not the absence; got {message}"
    );
    assert!(
        message.contains(pc_models::LAMA_MANGA_INPAINTER.sha256),
        "and the expected digest, so the user can compare; got {message}"
    );
    assert!(
        !message.contains("is missing at"),
        "a present-but-corrupt artifact must not be reported as absent; got {message}"
    );
}

/// §16.38 item 13(c), applied to the build tier rather than to config: *"config accepts, the
/// stage refuses."* In a build without `onnx` the provider still exists — the flag loaded fine
/// — and refuses on first use with a message naming the feature.
#[test]
#[cfg(not(feature = "onnx"))]
fn without_the_onnx_feature_the_provider_refuses_and_names_the_feature_to_rebuild_with() {
    let cache = TempDir::new().expect("temp dir");
    let provider =
        build_provider(true, None, cache.path(), Device::Cpu).expect("the flag is enabled");

    let Err(error) = provider.inpainter() else {
        panic!("a build without `onnx` has no backend to hand out");
    };

    let StageError::Model(message) = &error else {
        panic!("got {error:?}");
    };
    assert_eq!(message, pc_cli::inpainter::INPAINT_ONNX_UNAVAILABLE);
    assert!(message.contains("--features onnx"), "{message}");
    assert!(message.contains("inpainting_enabled = false"), "{message}");
}
