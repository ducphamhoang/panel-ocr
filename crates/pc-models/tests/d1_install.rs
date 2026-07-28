//! Task D1 -- spec §8.5's D1 row (download with progress, sha256 verify, atomic rename,
//! offline error message) and §5.3 (a missing or mismatched model is fatal). FROZEN.
//!
//! **No test here performs network I/O.** Every download runs against
//! `common::MemoryFetcher`. The real ~90 MB transfer is exercised only by
//! `d1_real_download_smoke_test` at the bottom, which is `#[ignore]`d *and* env-gated.

mod common;

use common::{dest_of, entries, models_dir, MemoryFetcher, RecordingProgress, FAKE_SPEC, PAYLOAD};
use pc_models::{
    ensure_available, install, partial_path, verify_sha256, ModelError, ModelFetcher, ModelSpec,
    NoProgress, COMIC_TEXT_DETECTOR,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

// ------------------------------------------------------------ happy path

#[test]
fn install_writes_the_verified_payload_to_the_managed_cache_entry() {
    // spec §8.5 D1: download, verify, install. One `fetch` per install, at the spec's URL.
    let (_root, models) = models_dir();
    let fetcher = MemoryFetcher::new(PAYLOAD);
    let mut progress = RecordingProgress::default();

    let installed =
        install(&FAKE_SPEC, &models, &fetcher, &mut progress).expect("install succeeds");

    assert_eq!(installed, dest_of(&models));
    assert_eq!(std::fs::read(&installed).expect("readable"), PAYLOAD);
    assert_eq!(fetcher.urls(), vec![FAKE_SPEC.url.to_string()]);
    assert_eq!(
        entries(&models),
        vec![FAKE_SPEC.file_name.to_string()],
        "no partial file may survive a successful install"
    );
}

#[test]
fn install_creates_the_models_directory_when_it_is_absent() {
    // Unlike `resolve` (a pure query), installing is allowed to create its destination
    // directory: §5.3 lists "cache dir not creatable" as a fatal condition, which only
    // makes sense if something tries to create it.
    let root = tempfile::TempDir::new().expect("temp dir");
    let models = root.path().join("cache/models");
    assert!(!models.exists());

    let installed = install(
        &FAKE_SPEC,
        &models,
        &MemoryFetcher::new(PAYLOAD),
        &mut NoProgress,
    )
    .expect("install succeeds");

    assert!(models.is_dir());
    assert_eq!(std::fs::read(&installed).expect("readable"), PAYLOAD);
}

// ------------------------------------------------------------ progress

#[test]
fn install_reports_start_then_every_byte_then_finish() {
    // spec §8.5 D1's "download with progress". Pinned as a contract rather than as a
    // rendered bar so it is observable without a terminal: `start` once with the known
    // total, `advance` summing to the payload length, `finish` once. The `indicatif`
    // implementation of this trait lives in `pc-cli`, which owns all terminal output.
    let (_root, models) = models_dir();
    let mut progress = RecordingProgress::default();

    install(
        &FAKE_SPEC,
        &models,
        &MemoryFetcher::new(PAYLOAD),
        &mut progress,
    )
    .expect("install succeeds");

    assert_eq!(progress.starts(), 1);
    assert_eq!(progress.started_with(), Some(Some(PAYLOAD.len() as u64)));
    assert_eq!(progress.total_advanced(), PAYLOAD.len() as u64);
    assert_eq!(progress.finishes(), 1);
}

#[test]
fn install_reports_an_unknown_total_as_none_and_still_succeeds() {
    // A server that sends no `Content-Length` must degrade to a spinner, not to a failure.
    let (_root, models) = models_dir();
    let fetcher = MemoryFetcher::new(PAYLOAD).with_content_length(None);
    let mut progress = RecordingProgress::default();

    install(&FAKE_SPEC, &models, &fetcher, &mut progress).expect("install succeeds");

    assert_eq!(progress.started_with(), Some(None));
    assert_eq!(progress.total_advanced(), PAYLOAD.len() as u64);
    assert_eq!(progress.finishes(), 1);
}

// ------------------------------------------------------------ digest gate + atomicity

#[test]
fn install_refuses_to_publish_a_payload_whose_digest_is_wrong() {
    // spec §8.5 D1 ("sha256 verify") and §5.3 (a hash mismatch is fatal). The point is
    // the *negative*: a wrong payload must not reach the destination path, because
    // everything that reads the cache afterwards trusts that path implicitly.
    let (_root, models) = models_dir();
    let fetcher = MemoryFetcher::new(b"this is not the model");

    let error = install(&FAKE_SPEC, &models, &fetcher, &mut NoProgress)
        .expect_err("the digest does not match");

    assert!(matches!(error, ModelError::HashMismatch { .. }));
    assert!(
        !dest_of(&models).exists(),
        "a corrupt download must never be published"
    );
    assert!(
        entries(&models).is_empty(),
        "and must leave no partial file behind: {:?}",
        entries(&models)
    );
}

#[test]
fn install_never_exposes_a_partial_file_at_the_destination_path() {
    // spec §8.5 D1's "atomic rename", asserted through the property that actually
    // matters: at no point during the transfer does the destination path exist. The
    // observer runs after every read of the fake stream, i.e. strictly before the rename.
    let (_root, models) = models_dir();
    let dest = dest_of(&models);
    let observations: Arc<Mutex<Vec<bool>>> = Arc::default();
    let probe = {
        let observations = Arc::clone(&observations);
        let dest = dest.clone();
        move || {
            observations
                .lock()
                .expect("observation lock")
                .push(dest.exists());
        }
    };
    let fetcher = MemoryFetcher::new(PAYLOAD).observing(probe);

    install(&FAKE_SPEC, &models, &fetcher, &mut NoProgress).expect("install succeeds");

    let observations = observations.lock().expect("observation lock").clone();
    assert!(
        !observations.is_empty(),
        "the fake transport must actually have been read from"
    );
    assert!(
        observations.iter().all(|exists| !exists),
        "the destination must not exist until the fully verified payload is renamed \
         into place, observed: {observations:?}"
    );
    assert!(dest.is_file(), "and must exist once the install returns");
}

#[test]
fn partial_path_is_a_sibling_of_the_destination() {
    // Why this is pinned: `std::fs::rename` is only atomic within one filesystem, so the
    // partial download has to live in the destination's own directory. The *name* is
    // deliberately left free.
    let dest = Path::new("/cache/models/comictextdetector.pt.onnx");

    let partial = partial_path(dest);

    assert_eq!(
        partial.parent(),
        dest.parent(),
        "the rename must be same-filesystem to be atomic"
    );
    assert_ne!(partial, dest);
}

#[test]
fn install_cleans_up_after_an_interrupted_transfer() {
    // A dropped connection must leave the cache exactly as it was -- no destination file
    // and no orphaned partial that a later run would mistake for a download in progress.
    let (_root, models) = models_dir();
    let fetcher = MemoryFetcher::new(PAYLOAD).failing_after(8);

    let error =
        install(&FAKE_SPEC, &models, &fetcher, &mut NoProgress).expect_err("the transfer dies");

    assert!(
        !matches!(error, ModelError::HashMismatch { .. }),
        "a dropped connection must be reported as a transfer failure, not as a digest \
         mismatch (hashing a truncated stream and blaming the digest sends the user \
         hunting the wrong problem): {error}"
    );
    assert!(!dest_of(&models).exists());
    assert!(
        entries(&models).is_empty(),
        "leftovers after an interrupted transfer: {:?}",
        entries(&models)
    );
}

// ------------------------------------------------------------ ensure_available

#[test]
fn ensure_available_returns_a_valid_cache_entry_without_fetching() {
    // The whole reason the managed cache exists: a second run must not move 90 MB again.
    // The entry's digest IS re-verified every run (that is what makes the corrupt-entry
    // recovery below reachable), but a matching digest must not trigger a fetch.
    let (_root, models) = models_dir();
    let dest = dest_of(&models);
    std::fs::write(&dest, PAYLOAD).expect("seed the cache entry");
    let fetcher = MemoryFetcher::new(PAYLOAD);

    let path = ensure_available(&FAKE_SPEC, &models, None, &fetcher, &mut NoProgress)
        .expect("the cached model is usable");

    assert_eq!(path, dest);
    assert_eq!(
        fetcher.calls(),
        0,
        "a valid cache entry must not be re-fetched"
    );
}

#[test]
fn ensure_available_replaces_a_corrupt_cache_entry() {
    // spec §5.3 makes a hash mismatch fatal only "*(and download unavailable)*", so a
    // mismatch with a working transport is recoverable: re-download once, then use it. A
    // half-written file from an earlier crash must not brick every later run.
    //
    // The replacement is preceded by a `WARN` naming the expected and actual digests --
    // silently discarding a file the user may have placed there deliberately is not
    // acceptable. That is a logging side effect and so is not asserted here, matching how
    // every other WARN in this codebase is treated (cf. `d5_yolo.rs`'s unknown-class test).
    let (_root, models) = models_dir();
    let dest = dest_of(&models);
    std::fs::write(&dest, b"truncated garbage").expect("seed a corrupt entry");
    let fetcher = MemoryFetcher::new(PAYLOAD);

    let path = ensure_available(&FAKE_SPEC, &models, None, &fetcher, &mut NoProgress)
        .expect("a corrupt entry is replaced");

    assert_eq!(path, dest);
    assert_eq!(std::fs::read(&path).expect("readable"), PAYLOAD);
    assert_eq!(fetcher.calls(), 1, "exactly one recovery attempt");
    verify_sha256(&path, FAKE_SPEC.sha256).expect("the installed file verifies");
}

#[test]
fn ensure_available_fails_when_the_replacement_download_also_mismatches() {
    // One recovery attempt, not a loop: if the freshly downloaded bytes are also wrong,
    // the problem is upstream (a re-tagged release asset, a corrupting proxy) and no
    // amount of retrying inside one run will fix it. The message must therefore hand the
    // user the deliberate, explicit retry path rather than implying a transient glitch.
    let (_root, models) = models_dir();
    std::fs::write(dest_of(&models), b"truncated garbage").expect("seed a corrupt entry");
    let fetcher = MemoryFetcher::new(b"the replacement is also wrong");

    let error = ensure_available(&FAKE_SPEC, &models, None, &fetcher, &mut NoProgress)
        .expect_err("the replacement does not verify either");

    assert_eq!(
        fetcher.calls(),
        1,
        "exactly one recovery attempt, not a loop"
    );
    let message = error.to_string();
    assert!(
        message.contains(FAKE_SPEC.sha256),
        "must quote the digest that was expected: {message}"
    );
    assert!(
        message.contains("panel-ocr models download"),
        "must name the explicit retry path (spec §13.1): {message}"
    );
}

#[test]
fn ensure_available_fails_when_a_corrupt_entry_cannot_be_replaced() {
    // The other half of §5.3: mismatch AND no download -> fatal, never a silent fallback
    // to the corrupt bytes.
    let (_root, models) = models_dir();
    std::fs::write(dest_of(&models), b"truncated garbage").expect("seed a corrupt entry");
    let fetcher = MemoryFetcher::offline("connect error: network is unreachable");

    let error = ensure_available(&FAKE_SPEC, &models, None, &fetcher, &mut NoProgress)
        .expect_err("nothing can fix this");

    assert!(matches!(error, ModelError::Unavailable { .. }));
}

#[test]
fn ensure_available_offline_message_is_actionable() {
    // spec §8.5 D1's "offline error message", and §5.3's "download unavailable" fatal.
    // Substrings, not an exact string: the wording is free, the *content* is not. A user
    // who cannot download must be able to fetch the file by hand and place it correctly,
    // or point at a copy they already have, from this one message.
    let (_root, models) = models_dir();
    let fetcher = MemoryFetcher::offline("dns error: failed to lookup address information");

    let error = ensure_available(&FAKE_SPEC, &models, None, &fetcher, &mut NoProgress)
        .expect_err("offline");

    assert!(matches!(error, ModelError::Unavailable { .. }));
    let message = error.to_string();
    assert!(
        message.contains(FAKE_SPEC.url),
        "must name the URL to fetch by hand: {message}"
    );
    assert!(
        message.contains(&models.display().to_string()),
        "must name where to put the file: {message}"
    );
    assert!(
        message.contains("--model-path"),
        "must name the escape hatch (spec §13.1): {message}"
    );
    assert!(
        message.contains("dns error: failed to lookup address information"),
        "must preserve the transport's own reason: {message}"
    );
}

#[test]
fn ensure_available_uses_an_override_verbatim_without_a_digest_requirement() {
    // Ratified decision: `--model-path` / §6's `model_path` is used as given and is NOT
    // checked against the spec's digest. The flag exists precisely so a maintainer can
    // supply a different or locally exported model -- verifying it against
    // `COMIC_TEXT_DETECTOR.sha256` would make the flag unusable. A *missing* override is
    // still a hard error (`d1_resolve.rs::resolve_rejects_an_override_that_does_not_exist`).
    let (root, models) = models_dir();
    let elsewhere = root.path().join("my-own-export.onnx");
    std::fs::write(
        &elsewhere,
        b"a locally exported model with a different digest",
    )
    .expect("write the override");
    let fetcher = MemoryFetcher::new(PAYLOAD);

    let path = ensure_available(
        &FAKE_SPEC,
        &models,
        Some(&elsewhere),
        &fetcher,
        &mut NoProgress,
    )
    .expect("an override is used as-is");

    assert_eq!(path, elsewhere);
    assert_eq!(
        fetcher.calls(),
        0,
        "an override must never trigger a download"
    );
    assert!(
        !dest_of(&models).exists(),
        "an override must not populate the managed cache"
    );
}

// ------------------------------------------------------------ shareability

#[test]
fn the_fetcher_seam_is_object_safe_and_shareable() {
    // §4.5 shares work across rayon threads, so the transport is `Send + Sync` and is
    // consumed as `&dyn ModelFetcher` -- which is also what makes it injectable, and what
    // keeps every test above off the network.
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_object_safe(_fetcher: &dyn ModelFetcher) {}

    assert_send_sync::<ModelSpec>();
    assert_send_sync::<MemoryFetcher>();
    assert_object_safe(&MemoryFetcher::new(PAYLOAD));
}

// ------------------------------------------------------------ opt-in: the real transport

#[test]
#[ignore = "opt-in: performs a REAL ~90 MB network download. Run with \
            `PANEL_OCR_ALLOW_NETWORK=1 cargo test -p pc-models --test d1_install -- \
            --ignored --nocapture`"]
fn d1_real_download_smoke_test() {
    // The ONLY test that exercises the real transport: reqwest + rustls, GitHub's 302 to
    // release-assets.githubusercontent.com, the full 94,669,756-byte body, and the real
    // sha256 from spec §8.3 step 3. Two gates (`#[ignore]` *and* the env var) because
    // `cargo test` -- including `cargo test -- --ignored` -- must never move 90 MB and
    // must pass on a machine with no network at all. An absent gate is a reported skip,
    // never a failure, following spec §16.13 item 4's principle.
    if std::env::var_os("PANEL_OCR_ALLOW_NETWORK").is_none() {
        eprintln!(
            "skipping the real-download smoke test: set PANEL_OCR_ALLOW_NETWORK=1 to \
             permit a ~90 MB transfer"
        );
        return;
    }

    let root = tempfile::TempDir::new().expect("temp dir");
    let models = root.path().join("models");

    let installed = ensure_available(
        &COMIC_TEXT_DETECTOR,
        &models,
        None,
        &pc_models::ReqwestFetcher::new(),
        &mut NoProgress,
    )
    .expect("the real model downloads and verifies");

    assert_eq!(installed, models.join(COMIC_TEXT_DETECTOR.file_name));
    assert_eq!(
        std::fs::metadata(&installed).expect("installed").len(),
        94_669_756,
        "the release asset's content-length"
    );
    verify_sha256(&installed, COMIC_TEXT_DETECTOR.sha256).expect("the digest matches");
}
