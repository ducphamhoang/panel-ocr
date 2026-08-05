//! C3 tests -- spec §6 round-trip. Frozen gates.
//!
//! "Unknown keys produce a WARN and are **preserved** on round-trip (that is why
//! `toml_edit`)." These tests are the reason the crate may not switch to `toml`.

use pc_config::{ConfigWarning, Profile, ProfileDocument, DEFAULT_PROFILE_TOML};
use std::io::Write;
use std::sync::{Arc, Mutex};

// ------------------------------------------------- tracing capture harness
//
// §6 says unknown keys produce a WARN. `ProfileDocument::warnings()` is the
// deterministic mirror of that, but a mirror is not proof, so one test below also
// captures real `tracing` output. Scoped via `with_default`, so it is thread-local
// and cannot leak into other tests.

#[derive(Clone, Default)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl CapturedLog {
    fn contents(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

impl Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
    type Writer = CapturedLog;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn capture_warnings<R>(f: impl FnOnce() -> R) -> (R, String) {
    let log = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(log.clone())
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .without_time()
        .finish();
    let out = tracing::subscriber::with_default(subscriber, f);
    (out, log.contents())
}

// ------------------------------------------------------------- preservation

const WITH_UNKNOWNS: &str = "\
# a user's own comment, which must survive
[general]
preferred_mask_file_type = \".png\"
mystery_key              = 42          # unknown, must be preserved

[masker]
mask_growth_steps = 11

[future_stage]
enabled = true
";

#[test]
// spec §6: unknown keys are PRESERVED on round-trip -- byte-identical output for an
// untouched document, comments and alignment included
fn unknown_keys_survive_round_trip_byte_for_byte() {
    let doc = ProfileDocument::parse(WITH_UNKNOWNS).unwrap();
    assert_eq!(doc.to_toml_string(), WITH_UNKNOWNS);
}

#[test]
// spec §6: the default profile itself round-trips byte-for-byte, comments included
fn default_profile_round_trips_byte_for_byte() {
    let doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    assert_eq!(doc.to_toml_string(), DEFAULT_PROFILE_TOML);
}

#[test]
// spec §6: unknown keys WARN -- they are neither silently dropped nor an error
fn unknown_keys_are_reported_as_warnings() {
    let doc = ProfileDocument::parse(WITH_UNKNOWNS).expect("unknown keys are not an error");

    assert!(doc.warnings().contains(&ConfigWarning::UnknownKey {
        table: "general".into(),
        key: "mystery_key".into(),
    }));
    // §16.5 item 11: an unrecognised table is treated the same way.
    assert!(doc.warnings().contains(&ConfigWarning::UnknownTable {
        table: "future_stage".into()
    }));
    assert_eq!(
        doc.warnings().len(),
        2,
        "no spurious warnings: {:?}",
        doc.warnings()
    );
}

#[test]
// spec §6: "produce a WARN" means the tracing WARN level, not just a return value --
// the user must actually see it
fn unknown_keys_emit_a_tracing_warn() {
    let (_doc, log) = capture_warnings(|| ProfileDocument::parse(WITH_UNKNOWNS).unwrap());
    assert!(
        log.contains("WARN"),
        "expected a WARN-level record, got: {log:?}"
    );
    assert!(
        log.contains("mystery_key"),
        "WARN must name the key: {log:?}"
    );
    assert!(
        log.contains("future_stage"),
        "WARN must name the table: {log:?}"
    );
}

#[test]
// spec §6: a document with no unknown keys emits nothing -- no WARN noise on the
// happy path
fn known_only_document_emits_no_warn() {
    let (_doc, log) = capture_warnings(|| ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap());
    assert!(log.is_empty(), "expected silence, got: {log:?}");
}

// ------------------------------------------------------------- edit-in-place

#[test]
// spec §6: writing an edited profile back updates the known key in place and leaves
// the surrounding document -- comments, unknown keys, key order -- undisturbed
fn set_profile_preserves_unknown_keys_and_comments() {
    let mut doc = ProfileDocument::parse(WITH_UNKNOWNS).unwrap();

    let mut profile = doc.profile().clone();
    profile.masker.mask_growth_steps = 7;
    doc.set_profile(&profile).unwrap();

    let out = doc.to_toml_string();
    assert!(out.contains("mask_growth_steps = 7"));
    assert!(out.contains("# a user's own comment, which must survive"));
    assert!(out.contains("mystery_key"));
    assert!(out.contains("[future_stage]"));

    // and the edit is stable: re-parsing yields the value we wrote
    assert_eq!(
        ProfileDocument::parse(&out)
            .unwrap()
            .profile()
            .masker
            .mask_growth_steps,
        7
    );
}

#[test]
// spec §6: `set_profile` re-validates -- an edit may not smuggle in an invalid value
fn set_profile_rejects_an_invalid_edit() {
    let mut doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    let mut profile = doc.profile().clone();
    profile.denoiser.search_window_size = 20; // even -> §6 violation
    let err = doc
        .set_profile(&profile)
        .expect_err("an even NLM search window must not be writable");
    assert_eq!(err.field(), Some("denoiser.search_window_size"));
}

#[test]
// spec §6: a key absent from the user's file but present in the profile is appended
// when written, so `profile edit` never silently loses a setting
fn set_profile_appends_keys_absent_from_the_document() {
    let mut doc = ProfileDocument::parse("[masker]\nmask_growth_steps = 11\n").unwrap();
    let mut profile = doc.profile().clone();
    profile.masker.off_white_max_threshold = 200;
    doc.set_profile(&profile).unwrap();

    let out = doc.to_toml_string();
    assert!(out.contains("off_white_max_threshold = 200"), "got: {out}");
    assert_eq!(
        ProfileDocument::parse(&out)
            .unwrap()
            .profile()
            .masker
            .off_white_max_threshold,
        200
    );
}

// --------------------------------------------------------- value round-trip

#[test]
// spec §6: a profile serialised from scratch and re-parsed is the same profile --
// the typed layer and the text layer agree in both directions
fn from_profile_round_trips_through_text() {
    let mut original = Profile::default();
    original.general.preferred_file_type = ".webp".into();
    original.preprocessor.ocr_strict_language = true;
    original.masker.debug_mask_color = [1, 2, 3, 4];
    original.denoiser.noise_min_standard_deviation = 0.75;

    let text = ProfileDocument::from_profile(&original).to_toml_string();
    let reparsed = ProfileDocument::parse(&text).unwrap();

    assert_eq!(*reparsed.profile(), original);
    assert_eq!(reparsed.warnings(), &[]);
}

#[test]
// spec §16.35 items 4 and 8: `[masker] mask_fallback_to_lowest_deviation` round-trips
// through both from-scratch serialization and in-place `toml_edit` updates. This mirrors
// the v1.5 device-key gate while keeping the two config-surface tasks in separate calls.
fn masker_fallback_round_trips_through_text_and_set_profile() {
    let mut original = Profile::default();
    original.masker.mask_fallback_to_lowest_deviation = false;
    let text = ProfileDocument::from_profile(&original).to_toml_string();
    assert!(
        text.contains("mask_fallback_to_lowest_deviation = false"),
        "the wire value must be a TOML boolean: {text}"
    );
    assert!(
        !ProfileDocument::parse(&text)
            .unwrap()
            .profile()
            .masker
            .mask_fallback_to_lowest_deviation
    );

    let mut doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    assert!(doc.profile().masker.mask_fallback_to_lowest_deviation);
    let mut edited = doc.profile().clone();
    edited.masker.mask_fallback_to_lowest_deviation = false;
    doc.set_profile(&edited).unwrap();

    // Read through toml_edit rather than depending on column alignment in the shipped file.
    assert_eq!(
        doc.document()["masker"]["mask_fallback_to_lowest_deviation"].as_bool(),
        Some(false),
        "got: {}",
        doc.to_toml_string()
    );
    let out = doc.to_toml_string();
    assert!(
        out.contains("mask_selection_fast"),
        "neighbouring [masker] keys must survive: {out}"
    );
    let reparsed = ProfileDocument::parse(&out).unwrap();
    assert!(!reparsed.profile().masker.mask_fallback_to_lowest_deviation);
    assert_eq!(reparsed.warnings(), &[]);
}

#[test]
// spec §16.36 item 1: `[general] device` round-trips through `toml_edit` like every other
// key -- written as the snake_case wire spelling `"cuda"` (the file-format contract, not
// the Rust variant name), edited in place without disturbing the rest of the document,
// and read back as the same typed value. Without this, `profile edit`/`profile set` could
// serialise a device the loader cannot read, or silently reset it to cpu.
fn general_device_round_trips_through_text_and_set_profile() {
    use pc_core::device::Device;

    let mut original = Profile::default();
    original.general.device = Device::Cuda;
    let text = ProfileDocument::from_profile(&original).to_toml_string();
    assert!(
        text.contains(r#"device = "cuda""#),
        "the wire spelling is snake_case `cuda`: {text}"
    );
    assert_eq!(
        ProfileDocument::parse(&text)
            .unwrap()
            .profile()
            .general
            .device,
        Device::Cuda
    );

    // in-place edit of the shipped document, comments and neighbours preserved
    let mut doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    assert_eq!(doc.profile().general.device, Device::Cpu);
    let mut edited = doc.profile().clone();
    edited.general.device = Device::Cuda;
    doc.set_profile(&edited).unwrap();

    // read through `toml_edit`, not by substring: the shipped document is
    // column-aligned, so the emitted line is not literally `device = "cuda"`.
    assert_eq!(
        doc.document()["general"]["device"].as_str(),
        Some("cuda"),
        "got: {}",
        doc.to_toml_string()
    );
    let out = doc.to_toml_string();
    assert!(
        out.contains("max_threads"),
        "the neighbouring [general] keys must survive: {out}"
    );
    let reparsed = ProfileDocument::parse(&out).unwrap();
    assert_eq!(reparsed.profile().general.device, Device::Cuda);
    assert_eq!(reparsed.warnings(), &[]);
}

// ------------------------------------------------------------------ on disk

#[test]
// spec §6: load/save go through the same round-trip machinery as parse/to_string
fn save_then_load_preserves_the_document() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("profile.toml");

    ProfileDocument::parse(WITH_UNKNOWNS)
        .unwrap()
        .save(&path)
        .unwrap();
    let reloaded = ProfileDocument::load(&path).unwrap();

    assert_eq!(reloaded.to_toml_string(), WITH_UNKNOWNS);
}

#[test]
// spec §6: a missing profile file is an Io error naming the path, not a panic and
// not a silent fallback to defaults
fn loading_a_missing_file_is_an_io_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nope.toml");
    let err = ProfileDocument::load(&path).expect_err("missing file must error");
    assert!(err.to_string().contains("nope.toml"), "{err}");
}

#[test]
// spec §6: malformed TOML is a Parse error, distinguishable from a validation error
fn malformed_toml_is_a_parse_error() {
    let err = ProfileDocument::parse("[general\npreferred_file_type = ").unwrap_err();
    assert_eq!(err.field(), None, "a syntax error has no validated field");
}
