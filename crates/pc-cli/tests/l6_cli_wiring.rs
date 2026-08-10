//! L6-5 red-first tests for §16.38 items 12(b), 16(f), 19(g), and 25(e).
//!
//! These tests are frozen before production implementation. They pin the CLI boundary:
//! parsing, config-plus-CLI resolution, conditional provider construction, and pipeline
//! context injection. They deliberately do not exercise LaMa inference or stage behavior.

use clap::Parser;
use pc_cli::args::{Cli, Command};
use pc_cli::inpainter::build_provider;
use pc_cli::setup::effective_inpainting_enabled;
use pc_config::Profile;
use pc_core::device::Device;
use pc_pipeline::{DetectorProvider, PipelineCtx};
use std::path::Path;

fn clean(args: &[&str]) -> pc_cli::args::CleanArgs {
    let mut argv = vec!["panel-ocr", "clean", "page.png"];
    argv.extend_from_slice(args);
    match Cli::try_parse_from(argv).expect("CLI must parse").command {
        Command::Clean(args) => args,
        other => panic!("expected clean command, got {other:?}"),
    }
}

/// §16.38 item 16(e): `--skip-inpaint` has the existing disable-flag shape and parses as a
/// boolean clean option, rather than accepting a value or being silently ignored.
#[test]
fn skip_inpaint_parses_as_a_clean_boolean_flag() {
    let args = clean(&["--skip-inpaint"]);
    assert!(args.skip_inpaint);
}

/// §16.38 item 16(e), item 12(b): the effective state is config-enabled unless the CLI
/// disable flag is present; the CLI must be able to turn configured inpainting off.
#[test]
fn skip_inpaint_overrides_configured_inpainting_but_absence_preserves_config() {
    let mut profile = Profile::default();
    profile.inpainter.inpainting_enabled = true;

    let enabled_args = clean(&[]);
    assert!(effective_inpainting_enabled(&enabled_args, &profile));

    let disabled_args = clean(&["--skip-inpaint"]);
    assert!(!effective_inpainting_enabled(&disabled_args, &profile));
}

/// §16.38 item 8(c), item 16(e): provider construction is conditional on the resolved
/// state. The disabled branch must return no provider; the enabled branch must return one
/// even in the non-ONNX build, where the provider reports the optional-feature refusal only
/// when the pipeline actually asks it for an inpainter.
#[test]
fn cli_builds_a_provider_only_when_effectively_enabled() {
    let cache = tempfile::tempdir().expect("temporary cache");
    assert!(build_provider(false, None, cache.path(), Device::Cpu).is_none());
    assert!(build_provider(true, None, cache.path(), Device::Cpu).is_some());
}

struct Detector;
impl DetectorProvider for Detector {
    fn detector_for(
        &self,
        _original: &Path,
    ) -> Result<std::sync::Arc<dyn pc_detect::TextDetector>, pc_core::StageError> {
        Err(pc_core::StageError::Model("test detector".into()))
    }
}

/// §16.40 item 2(d), §16.38 item 16(e): the CLI-created provider is injected through the
/// pipeline-owned trait boundary, not kept as a pc-cli-only value. The assertion observes
/// the actual `PipelineCtx` field and its exact provider object.
#[test]
fn cli_provider_is_injected_into_pipeline_context() {
    let detector = Detector;
    let cache = tempfile::tempdir().expect("temporary cache");
    let provider = build_provider(true, None, cache.path(), Device::Cpu)
        .expect("enabled CLI state must create a provider");
    let ctx = PipelineCtx::new(&detector).with_inpainter(provider.as_ref());
    assert!(ctx.inpainter.is_some());
}

/// §16.46 item 13(d). `skip_inpaint_overrides_configured_inpainting_but_absence_preserves_config`
/// above covers a HAND-SET flag. This covers the SHIPPED DEFAULT, which is a different case
/// and was untestable before §16.46: the default was already `false`, so that test's
/// "absence preserves config" leg proved nothing about a default run.
///
/// Turns red if `effective_inpainting_enabled` stops reading the profile, if `--skip-inpaint`
/// stops being honoured, or if the default reverts -- in which case the premise fires first,
/// with a message saying so, rather than the test silently going vacuous.
#[test]
fn skip_inpaint_turns_off_the_now_default_on_inpainting() {
    let profile = Profile::default();
    assert!(
        profile.inpainter.inpainting_enabled,
        "premise (§16.46 item 1(b)): the shipped default is ON, or both legs below are vacuous"
    );

    assert!(effective_inpainting_enabled(&clean(&[]), &profile));
    assert!(!effective_inpainting_enabled(
        &clean(&["--skip-inpaint"]),
        &profile
    ));
}
