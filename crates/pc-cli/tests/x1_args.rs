//! Task **X1** — the `clap` surface (spec §13.1, §16.12 items 2, 6, 16, 21).
//!
//! These test the parser, not the pipeline, so they pass today.
//!
//! FROZEN (CLAUDE.md).

use clap::Parser;
use pc_cli::args::{Cli, Command, DetectorSpec};
use pc_cli::{detector, logging, paths};
use pc_core::device::Device;
use pc_pipeline::SaveOnly;
use std::path::PathBuf;
use std::str::FromStr;

fn clean(args: &[&str]) -> pc_cli::args::CleanArgs {
    let mut argv = vec!["panel-ocr", "clean"];
    argv.extend_from_slice(args);
    match Cli::try_parse_from(argv).expect("parse").command {
        Command::Clean(args) => args,
        other => panic!("expected clean, got {other:?}"),
    }
}

/// §13.1: `clean` takes one or more paths and defaults `--output-dir` to `cleaned`.
#[test]
fn clean_parses_paths_and_defaults() {
    let args = clean(&["a.png", "b.png"]);

    assert_eq!(
        args.paths,
        vec![PathBuf::from("a.png"), PathBuf::from("b.png")]
    );
    assert_eq!(args.output_dir, PathBuf::from("cleaned"));
    assert_eq!(args.detector, DetectorSpec::Onnx);
    assert!(args.save_only().is_none());
    assert_eq!(args.skip_flags(), pc_pipeline::SkipFlags::default());
    assert!(!args.fail_fast && !args.no_cache && !args.extract_text);
}

/// §13.1 requires at least one path.
#[test]
fn clean_requires_at_least_one_path() {
    assert!(Cli::try_parse_from(["panel-ocr", "clean"]).is_err());
}

/// §16.12 item 16 — the three `--save-only-*` flags are mutually exclusive.
#[test]
fn save_only_flags_are_mutually_exclusive() {
    assert_eq!(
        clean(&["a.png", "--save-only-mask"]).save_only(),
        Some(SaveOnly::Mask)
    );
    assert_eq!(
        clean(&["a.png", "--save-only-cleaned"]).save_only(),
        Some(SaveOnly::Cleaned)
    );
    assert_eq!(
        clean(&["a.png", "--extract-text", "--save-only-text"]).save_only(),
        Some(SaveOnly::Text)
    );

    assert!(Cli::try_parse_from([
        "panel-ocr",
        "clean",
        "a.png",
        "--save-only-mask",
        "--save-only-cleaned"
    ])
    .is_err());
    // `--save-only-text` without `--extract-text` would request a file nothing produces.
    assert!(Cli::try_parse_from(["panel-ocr", "clean", "a.png", "--save-only-text"]).is_err());
}

/// §4.1: `--no-cache` cannot be combined with flags that need a cache.
#[test]
fn no_cache_conflicts_with_the_cache_flags() {
    assert!(
        Cli::try_parse_from(["panel-ocr", "clean", "a.png", "--no-cache", "--cache-masks"])
            .is_err()
    );
    assert!(
        Cli::try_parse_from(["panel-ocr", "clean", "a.png", "--no-cache", "--keep-cache"]).is_err()
    );
}

/// §13.1: `--profile` and `--profile-path` are alternatives.
#[test]
fn profile_selection_is_exclusive() {
    assert!(Cli::try_parse_from([
        "panel-ocr",
        "clean",
        "a.png",
        "--profile",
        "manga",
        "--profile-path",
        "p.toml"
    ])
    .is_err());
}

/// §4.4 / §16.12 item 6 — the flags are parsed raw; normalisation happens later.
#[test]
fn skip_flags_are_parsed_without_normalisation() {
    let args = clean(&["a.png", "--skip-mask"]);

    let flags = args.skip_flags();

    assert!(flags.mask && !flags.preprocess && !flags.text_detection);
    assert!(flags.implies_more());
    assert!(flags.normalized().text_detection);
}

/// §16.12 item 2 — the detector-spec grammar.
#[test]
fn detector_spec_grammar() {
    assert_eq!(DetectorSpec::from_str("onnx").unwrap(), DetectorSpec::Onnx);
    assert_eq!(DetectorSpec::from_str("mock").unwrap(), DetectorSpec::Mock);
    assert_eq!(
        DetectorSpec::from_str("replay:/tmp/fixtures").unwrap(),
        DetectorSpec::Replay(PathBuf::from("/tmp/fixtures"))
    );

    assert!(DetectorSpec::from_str("replay:").is_err());
    assert!(DetectorSpec::from_str("candle").is_err());
    assert_eq!(
        clean(&["a.png", "--detector", "mock"]).detector,
        DetectorSpec::Mock
    );
}

/// §16.12 item 2 — `onnx` is a fatal, explained failure in v1; `mock`/`replay` build.
// Gate added 2026-07-28 (authorized directly, no assertion text changed): this asserts the
// content of `detector::ONNX_UNAVAILABLE`, which is itself `#[cfg(not(feature = "onnx"))]`.
// Without the same `cfg` it asserts the contents of nothing — under `--features onnx` the
// ONNX arm is real and the "D1/D4" wording does not exist. PER-TEST, never file-level: the
// other tests here are feature-independent and must keep running in both configurations.
// The `--features onnx` counterparts are the two tests immediately below.
#[cfg(not(feature = "onnx"))]
#[test]
fn only_the_mock_and_replay_providers_can_be_built_in_v1() {
    let cache_root = tempfile::tempdir().unwrap();
    let error = detector::build_provider(
        &DetectorSpec::Onnx,
        None,
        None,
        cache_root.path(),
        &pc_config::TextDetectorConfig::default(),
        Device::Cpu,
    )
    .err()
    .expect("onnx must not be buildable in v1");
    let message = error.to_string();
    assert!(message.contains("D1"), "{message}");
    assert!(message.contains("D4"), "{message}");
    assert!(message.contains("--detector replay"), "{message}");

    assert!(detector::build_provider(
        &DetectorSpec::Mock,
        None,
        None,
        cache_root.path(),
        &pc_config::TextDetectorConfig::default(),
        Device::Cpu,
    )
    .is_ok());
    assert!(
        detector::build_provider(
            &DetectorSpec::Replay(PathBuf::from("/nope")),
            None,
            None,
            cache_root.path(),
            &pc_config::TextDetectorConfig::default(),
            Device::Cpu,
        )
        .is_ok(),
        "a replay provider builds; a missing fixture is a per-image error (§16.12 item 3)"
    );
}

/// §5.3 + the "`clean` never provisions" reversal — the `--features onnx` counterpart of
/// `only_the_mock_and_replay_providers_can_be_built_in_v1`.
///
/// With the feature on, the ONNX arm is real, so the default detector's failure mode is no
/// longer "not in this build" but "the managed model is not provisioned". Provisioning
/// happens **only** in the explicit `models` subcommands, so this must be a fatal that
/// tells the user the command to run — never a download.
///
/// `resolve_detector_model` is the unit under test rather than `build_provider`, because it
/// is the function that owns the managed-cache case and the only one that receives a cache
/// root at all; `build_provider` takes an already-resolved path and so cannot name a cache
/// location. See the companion test below for `build_provider`'s half.
#[cfg(feature = "onnx")]
#[test]
fn onnx_resolution_against_an_empty_managed_cache_tells_the_user_to_download() {
    let cache_root = tempfile::tempdir().unwrap();
    let models_dir = paths::models_dir(cache_root.path());

    // Both overrides `None`, i.e. the managed-cache path — the only arm the reversal
    // changes. Passing them explicitly also makes this test immune to an ambient profile
    // that happens to set `text_detector.model_path`, which would otherwise take the
    // override arm and skip the case under test entirely.
    let error =
        pc_cli::models::resolve_detector_model(&DetectorSpec::Onnx, None, None, cache_root.path())
            .expect_err("an unprovisioned managed cache must be fatal, never a download");

    let message = error.to_string();
    assert!(
        message.contains("panel-ocr models download"),
        "must name the explicit provisioning command: {message}"
    );
    assert!(
        message.contains(&models_dir.display().to_string()),
        "must name where the model is expected to be: {message}"
    );
    assert!(
        !message.contains("not available in this build"),
        "that is the no-feature message; this build HAS the backend: {message}"
    );

    // The observable form of "never provisions": under the reversed behaviour this call
    // downloaded ~90 MB into `models_dir`. No timing is asserted -- the empty cache is the
    // evidence.
    assert!(
        !models_dir.exists()
            || std::fs::read_dir(&models_dir)
                .expect("readable")
                .next()
                .is_none(),
        "resolution must neither create nor populate the managed cache: {}",
        models_dir.display()
    );
}

/// Lazy-in-provider: construction must SUCCEED with no model (a run that never detects needs
/// none), and the actionable refusal must arrive on first use instead.
#[cfg(feature = "onnx")]
#[test]
fn onnx_provider_defers_model_resolution_to_first_use() {
    let cache_root = tempfile::tempdir().unwrap();
    let models_dir = paths::models_dir(cache_root.path());

    let provider = detector::build_provider(
        &DetectorSpec::Onnx,
        None,
        None,
        cache_root.path(),
        &pc_config::TextDetectorConfig::default(),
        Device::Cpu,
    )
    .expect("construction must not resolve, so it cannot fail on a missing model");

    let error = provider
        .detector_for(std::path::Path::new("page01.png"))
        .err()
        .expect("first use must refuse");
    let message = error.to_string();
    assert!(
        matches!(error, pc_core::StageError::Model(_)),
        "the runner carves out `Model`: {message}"
    );
    assert!(message.contains("panel-ocr models download"), "{message}");
    assert!(
        message.contains(&models_dir.display().to_string()),
        "{message}"
    );
    assert!(
        !message.to_lowercase().contains("runtime"),
        "must fail on the model, before any ONNX Runtime probe: {message}"
    );

    // Coverage the `cfg(not(onnx))` gate removes from this configuration.
    assert!(detector::build_provider(
        &DetectorSpec::Mock,
        None,
        None,
        cache_root.path(),
        &pc_config::TextDetectorConfig::default(),
        Device::Cpu,
    )
    .is_ok());
}

/// §6 / §13.1 — the precedence boundary between an explicit override and the managed cache.
///
/// `resolve_detector_model` evaluates `cli_override.or(profile_override)` **before** the
/// managed arm, so a user who names a model file can never be told to download a different
/// one. That guarantee holds by control flow for a *valid* override; this pins it for an
/// *invalid* one, which is where the reversal's new error plumbing could plausibly wrap a
/// path error with a provisioning hint. Suggesting `panel-ocr models download` here would be
/// actively wrong advice: the managed model is not the model that was asked for.
///
/// Both arms are exercised, because both feed the same `.or()` and a fix applied to one
/// could easily miss the other. No fixtures are needed — the profile override arrives as a
/// plain parameter, not through a profile file.
///
/// Three distinct failure modes now exist for `DetectorSpec::Onnx` (no feature / absent
/// override / unprovisioned cache), and none of them may emit another's message. Each of the
/// three counterparts therefore asserts the absence of the other two's wording.
#[cfg(feature = "onnx")]
#[test]
fn a_missing_explicit_model_path_is_not_answered_with_a_download_suggestion() {
    let cache_root = tempfile::tempdir().unwrap();
    // A separate directory, so the named-but-absent file cannot be mistaken for the managed
    // cache entry and the "nothing was provisioned" check below stays unambiguous.
    let elsewhere = tempfile::tempdir().unwrap();

    for (label, via_cli, file_name) in [
        ("--model-path", true, "my-own-export.onnx"),
        ("profile model_path", false, "profile-export.onnx"),
    ] {
        let named = elsewhere.path().join(file_name);
        let (cli, profile) = if via_cli {
            (Some(named.as_path()), None)
        } else {
            (None, Some(named.as_path()))
        };

        let error = pc_cli::models::resolve_detector_model(
            &DetectorSpec::Onnx,
            cli,
            profile,
            cache_root.path(),
        )
        .err()
        .unwrap_or_else(|| panic!("{label}: a named-but-absent model file is fatal"));

        let message = error.to_string();
        assert!(
            message.contains(file_name),
            "{label}: must name the path the user gave: {message}"
        );
        assert!(
            !message.contains("models download"),
            "{label}: the managed model is not what was requested: {message}"
        );
        assert!(
            !message.contains("not available in this build"),
            "{label}: that is the no-feature message; this build HAS the backend: {message}"
        );
    }

    // An override failure must not have touched the managed cache either way.
    let models_dir = paths::models_dir(cache_root.path());
    assert!(
        !models_dir.exists()
            || std::fs::read_dir(&models_dir)
                .expect("readable")
                .next()
                .is_none(),
        "an override failure must not provision anything: {}",
        models_dir.display()
    );
}

/// §5.3 — the recovery suggestion carried by the "not provisioned" fatal must be safe to
/// **paste**, for any cache root.
///
/// That is the suggestion's entire justification: it exists so a user with a non-default
/// cache root can copy the line verbatim and have it work. An unquoted path defeats that on
/// its own criterion — a space splits it into extra arguments, and `;` / `&` / `$(...)` /
/// backticks stop being path characters and become shell *syntax*. Neither is exotic: a
/// `--cache-dir` beneath a spaced directory is entirely ordinary on Windows/WSL
/// (`/mnt/d/Duc/Manga/Choujin Locke/...`).
///
/// **`/bin/sh` is the authority here, not a hand-rolled quoter.** Asserting an exact quoted
/// string would reject a different-but-valid quoting style, and re-implementing POSIX
/// quoting in the test would only prove the test agrees with itself — the same circularity
/// §16.13 item 4 rules out for reference fixtures. So the test hands the emitted segment to a
/// real shell and asserts the shell recovers the original path exactly. v1 is Linux + macOS
/// only, so a POSIX shell is always present (§16.33); `#[cfg(unix)]` keeps the file compiling anyway.
///
/// One assertion catches every failure mode at once: a broken or unterminated quote makes
/// `sh` exit non-zero, and word splitting, globbing, parameter expansion and command
/// substitution all change the recovered bytes.
#[cfg(all(feature = "onnx", unix))]
#[test]
fn the_models_download_suggestion_is_paste_safe_for_hostile_cache_paths() {
    const PREFIX: &str = "panel-ocr models download --cache-dir ";

    // Every payload is harmless IF EXECUTED, deliberately: an unquoted `$( )` or backtick
    // would be run by the shell below, so nothing here may have a side effect. `*` is
    // omitted on purpose — an unmatched glob is left literal by POSIX sh, so it would be a
    // false negative rather than a probe.
    let hostile = [
        "space dir",            // the ordinary case: word splitting
        "it's-a-cache",         // the case naive quoting gets wrong: ' must become '\''
        "dollar$(echo pwned)",  // command substitution
        "back`echo pwned`tick", // the older substitution syntax
        "semi;colon",           // would terminate the command
        "amp&ersand",           // would background it
        "pipe|and\"quote",      // pipeline plus a double quote
        "tab\tseparated",       // IFS splitting on whitespace that is not a space
        "line\nbreak",          // legal in a POSIX path, and the nastiest case
    ];

    let root = tempfile::tempdir().unwrap();

    for hostile in hostile {
        let path = root.path().join(hostile);
        let expected = path.display().to_string();
        let command = pc_cli::models::models_download_command(Some(path.as_path()));

        assert!(
            command.starts_with(PREFIX),
            "unexpected suggestion shape for {hostile:?}: {command}"
        );
        let segment = &command[PREFIX.len()..];

        let script = format!("printf '%s\\n' {segment}");
        let output = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(&script)
            .output()
            .expect("run /bin/sh");

        assert!(
            output.status.success(),
            "a POSIX shell cannot even parse the suggestion for {hostile:?}\n  \
             command: {command}\n  script:  {script}\n  stderr:  {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!("{expected}\n"),
            "the shell did not recover the original path for {hostile:?} — word splitting, \
             globbing, parameter/command expansion, or a mis-escaped quote\n  \
             command: {command}"
        );
    }
}

/// With no `--cache-dir` override the suggestion stays the bare command: the flag is carried
/// only when it is needed to make the line followable, so quoting must not cause it to be
/// emitted unconditionally.
#[cfg(feature = "onnx")]
#[test]
fn the_models_download_suggestion_omits_the_flag_for_the_default_cache() {
    assert_eq!(
        pc_cli::models::models_download_command(None),
        "panel-ocr models download"
    );
}

/// §16.12 item 3 — a missing replay fixture is reported per image, not at build time.
#[test]
fn a_missing_replay_fixture_is_a_per_image_error() {
    use pc_pipeline::DetectorProvider;

    let dir = tempfile::tempdir().unwrap();
    let provider = detector::ReplayProvider::new(dir.path());

    let error = provider
        .detector_for(std::path::Path::new("page01.png"))
        .err()
        .expect("no fixture => error");

    assert!(error.to_string().contains("page01"), "{error}");
}

/// §13.1: verbosity maps to tracing levels.
#[test]
fn verbosity_maps_to_tracing_levels() {
    assert_eq!(logging::filter_directive(0, false), "warn");
    assert_eq!(logging::filter_directive(1, false), "info");
    assert_eq!(logging::filter_directive(2, false), "debug");
    assert_eq!(logging::filter_directive(9, false), "trace");
    assert_eq!(logging::filter_directive(3, true), "error");

    let cli = Cli::try_parse_from(["panel-ocr", "-vv", "clean", "a.png"]).unwrap();
    assert_eq!(cli.verbose, 2);
    assert!(Cli::try_parse_from(["panel-ocr", "-q", "-v", "clean", "a.png"]).is_err());
}

/// §13.1's other four subcommands parse.
#[test]
fn the_management_subcommands_parse() {
    assert!(matches!(
        Cli::try_parse_from(["panel-ocr", "profile", "list"])
            .unwrap()
            .command,
        Command::Profile { .. }
    ));
    assert!(matches!(
        Cli::try_parse_from(["panel-ocr", "cache", "clear", "--images"])
            .unwrap()
            .command,
        Command::Cache { .. }
    ));
    assert!(matches!(
        Cli::try_parse_from(["panel-ocr", "models", "path"])
            .unwrap()
            .command,
        Command::Models { .. }
    ));
    assert!(matches!(
        Cli::try_parse_from(["panel-ocr", "ocr", "a.png", "--format", "txt"])
            .unwrap()
            .command,
        Command::Ocr(_)
    ));
}

/// §16.12 item 21 — the default cache location, and the hidden override.
#[test]
fn the_cache_directory_is_overridable() {
    assert!(paths::default_cache_dir().ends_with("panel-ocr"));
    assert!(paths::default_config_path().ends_with("config.toml"));

    let args = clean(&["a.png", "--cache-dir", "/tmp/pc-cache"]);
    assert_eq!(args.cache_dir, Some(PathBuf::from("/tmp/pc-cache")));
}

// ============================================================ task L3: --include-optional
//
// spec §13.1 as superseded by §16.38 item 19. Additive to this frozen file (cookbook rule
// 8's exit 1): no existing assertion is touched.

/// Parse a `models` subcommand, or fail the test.
fn models(args: &[&str]) -> pc_cli::args::ModelsCommand {
    let mut argv = vec!["panel-ocr", "models"];
    argv.extend_from_slice(args);
    match Cli::try_parse_from(argv).expect("parse").command {
        Command::Models { command } => command,
        other => panic!("expected models, got {other:?}"),
    }
}

/// §16.38 item 19(b): the flag exists on `models download` and defaults to off, so the
/// default invocation cannot pull the 207 MB optional artifact.
#[test]
fn models_download_takes_include_optional_and_defaults_it_off() {
    match models(&["download"]) {
        pc_cli::args::ModelsCommand::Download {
            include_optional, ..
        } => assert!(
            !include_optional,
            "the default must not fetch optional models"
        ),
        other => panic!("expected download, got {other:?}"),
    }
    match models(&["download", "--include-optional"]) {
        pc_cli::args::ModelsCommand::Download {
            include_optional, ..
        } => assert!(include_optional),
        other => panic!("expected download, got {other:?}"),
    }
}

/// §16.38 item 19(b) put the flag on `verify` as well as `download`, deliberately — "not
/// download alone". A parser accepting it only on `download` would make the ruling's
/// preflight story unreachable, and fails here rather than at review time.
#[test]
fn models_verify_takes_include_optional_and_defaults_it_off() {
    match models(&["verify"]) {
        pc_cli::args::ModelsCommand::Verify {
            include_optional, ..
        } => assert!(!include_optional),
        other => panic!("expected verify, got {other:?}"),
    }
    match models(&["verify", "--include-optional"]) {
        pc_cli::args::ModelsCommand::Verify {
            include_optional, ..
        } => assert!(include_optional),
        other => panic!("expected verify, got {other:?}"),
    }
}

/// The flag's scope is `download` and `verify` only. `models path` already lists every
/// registry entry unconditionally with `EXIT_OK` (§16.38 item 19(b) says it needs no
/// change), so an `--include-optional` there would be a no-op flag implying the listing is
/// otherwise filtered. It must be a parse error, not silently accepted.
#[test]
fn models_path_rejects_include_optional() {
    assert!(
        Cli::try_parse_from(["panel-ocr", "models", "path", "--include-optional"]).is_err(),
        "`models path` must not accept a flag that changes nothing there"
    );
    // Control: the subcommand itself parses, so the assertion above is about the flag.
    assert!(Cli::try_parse_from(["panel-ocr", "models", "path"]).is_ok());
}

/// §16.38 item 19(g): the refusal hint L5 will use for the optional model names the flag
/// that actually fetches it. Literals, not a value derived from `models_download_command` —
/// a hint assembled from the artifact it is meant to describe would pass vacuously.
#[test]
fn the_optional_download_hint_names_the_include_optional_flag() {
    assert_eq!(
        pc_cli::models::models_download_optional_command(None),
        "panel-ocr models download --include-optional"
    );

    // A temp dir is never the default cache root, so this exercises the `--cache-dir` branch.
    let root = tempfile::tempdir().unwrap();
    let with_root = pc_cli::models::models_download_optional_command(Some(root.path()));
    assert!(
        with_root.starts_with("panel-ocr models download --include-optional --cache-dir "),
        "the flag must survive the cache-dir branch: {with_root}"
    );
    let quoted = paths::Shell::HOST.quote(root.path());
    assert!(
        with_root.ends_with(&quoted),
        "the hint must name the cache root it applies to: {with_root}"
    );
}

/// §16.38 item 19(c): a skipped optional model is announced. The notice must name the model
/// AND the flag — a line saying only "skipped" leaves the user with no next step, which is
/// the silent-skip outcome the ruling forbids.
#[test]
fn the_skipped_optional_notice_names_the_model_and_the_flag() {
    let notice = pc_cli::models::skipped_optional_notice(&pc_models::LAMA_MANGA_INPAINTER);

    assert!(
        notice.contains("lama-manga-inpainter"),
        "must name the model: {notice}"
    );
    assert!(
        notice.contains("--include-optional"),
        "must name the flag that fetches it: {notice}"
    );
    assert!(
        !notice.contains("MISSING"),
        "a deliberate skip is not a missing model: {notice}"
    );
}

/// §16.38 item 1(a)'s measured byte length, hard-coded here from the spec rather than read
/// off any artifact: `expected_size` is what makes a truncated 207 MB download a reported
/// `SIZE MISMATCH` instead of a mystery digest failure, and a `None` here would disable that
/// check for the one model most likely to be interrupted mid-transfer.
#[test]
fn the_lama_registry_entry_has_a_pinned_expected_size() {
    assert_eq!(
        pc_cli::models::expected_size(&pc_models::LAMA_MANGA_INPAINTER),
        Some(207_482_644)
    );
}

/// The row `models verify` prints for each status. Pinned because the labels are a script
/// interface: the optional-absent row must be greppable as its own thing, and the two
/// mismatch rows must carry both numbers.
#[test]
fn the_verify_row_renders_each_status_with_its_diagnostic() {
    use pc_models::{Verification, VerifyStatus};
    let path = PathBuf::from("m.onnx");
    let row = |status| {
        pc_cli::models::verify_row(
            "m",
            &Verification {
                path: path.clone(),
                status,
            },
        )
    };

    assert_eq!(row(VerifyStatus::Ok), "m\tOK\tm.onnx");
    assert_eq!(row(VerifyStatus::Missing), "m\tMISSING\tm.onnx");
    assert_eq!(
        row(VerifyStatus::NotInstalled),
        "m\tNOT INSTALLED (optional)\tm.onnx"
    );
    assert_eq!(
        row(VerifyStatus::SizeMismatch {
            actual: 1,
            expected: 2
        }),
        "m\tSIZE MISMATCH\tm.onnx\tactual=1\texpected=2"
    );
    assert_eq!(
        row(VerifyStatus::HashMismatch {
            actual: "aa".into(),
            expected: "bb".into()
        }),
        "m\tHASH MISMATCH\tm.onnx\tactual=aa\texpected=bb"
    );
    assert_eq!(
        row(VerifyStatus::Error("unreadable".into())),
        "m\tERROR\tm.onnx\tunreadable"
    );
}
