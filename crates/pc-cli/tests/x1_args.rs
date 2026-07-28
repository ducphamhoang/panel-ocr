//! Task **X1** — the `clap` surface (spec §13.1, §16.12 items 2, 6, 16, 21).
//!
//! These test the parser, not the pipeline, so they pass today.
//!
//! FROZEN (CLAUDE.md).

use clap::Parser;
use pc_cli::args::{Cli, Command, DetectorSpec};
use pc_cli::{detector, logging, paths};
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
#[test]
fn only_the_mock_and_replay_providers_can_be_built_in_v1() {
    let error = detector::build_provider(&DetectorSpec::Onnx, None)
        .err()
        .expect("onnx must not be buildable in v1");
    let message = error.to_string();
    assert!(message.contains("D1"), "{message}");
    assert!(message.contains("D4"), "{message}");
    assert!(message.contains("--detector replay"), "{message}");

    assert!(detector::build_provider(&DetectorSpec::Mock, None).is_ok());
    assert!(
        detector::build_provider(&DetectorSpec::Replay(PathBuf::from("/nope")), None).is_ok(),
        "a replay provider builds; a missing fixture is a per-image error (§16.12 item 3)"
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
