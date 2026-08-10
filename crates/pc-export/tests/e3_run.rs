//! Task **E3** -- destination resolution, the mask branch, and `run()` wiring:
//! spec §12.2, §12.3 steps 1, 3-5 and 8, §12.7(A)4/5/6/7/8, §16.11 items 4, 9, 13, 14.
//! FROZEN.

mod common;

use common::{
    all_outputs, full_sources, gray_page, handle, input_with, only_cleaned, only_mask, only_text,
    rgba_mask, write_original_png, PAGE_SIZE,
};
use image::DynamicImage;
use pc_core::{ImageHandle, Rect, Stage, StageError, Step, SCHEMA_VERSION};
use pc_export::{destinations, run, ExportInput, ExportOutput, ExportSources, ExportStage};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------ §12.2 / §3 / §2.8

#[test]
fn the_stage_impl_is_wired_to_step_export() {
    assert_eq!(ExportStage::STEP, Step::Export);
    assert_eq!(Step::Export.prev(), Some(Step::Inpaint));
    // §2.8: no `Output` variant maps to `Step::Export` -- export writes user-facing
    // files, not cache artifacts. §16.11 item 2 exists precisely because of this.
    assert!(pc_core::Output::ALL
        .iter()
        .all(|output| output.step() != Step::Export));
}

#[test]
fn the_input_carries_a_schema_version_and_serialises_when_its_handles_have_paths() {
    // §2 (schema_version first) and §16.11 item 13: `ExportSources` holds
    // `ImageHandle`s, so a memory-only input is deliberately NOT serialisable
    // (§2.3's Serialize guard) and structural comparison is the only honest check.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let input = input_with(&original, dir.path(), full_sources(), all_outputs(), true);
    assert_eq!(input.schema_version, SCHEMA_VERSION);
    let json = serde_json::to_value(&input).expect("every handle carries a path");
    assert_eq!(json["schema_version"], SCHEMA_VERSION);
    assert!(json["sources"]["denoised"]["path"].is_string(), "{json}");

    let path_less = ExportInput {
        sources: ExportSources {
            masked: Some(ImageHandle::from_memory(gray_page(PAGE_SIZE, 200, &[]))),
            ..ExportSources::default()
        },
        ..input_with(
            &original,
            dir.path(),
            ExportSources::default(),
            all_outputs(),
            true,
        )
    };
    assert!(
        serde_json::to_value(&path_less).is_err(),
        "a memory-only handle must never reach a JSON checkpoint (§2.3)"
    );
}

// ------------------------------------------------------- §12.3 step 1 / §12.7(A)4

#[test]
fn a4_an_absolute_output_dir_is_used_as_is() {
    // §12.7(A)4 first half.
    let input = destination_input("/pages/page.png", "/exports", None, ".png");
    let dests = destinations(&input).expect("resolvable destinations");
    assert_eq!(dests.base, PathBuf::from("/exports"));
    assert_eq!(dests.cleaned, PathBuf::from("/exports/page_clean.png"));
    assert_eq!(dests.mask, PathBuf::from("/exports/page_mask.png"));
    assert_eq!(dests.text, PathBuf::from("/exports/page_text.png"));
}

#[test]
fn a4_a_relative_output_dir_is_relative_to_the_export_paths_parent() {
    // §12.7(A)4 second half: relative `cleaned` => `{input_parent}/cleaned/...`.
    let input = destination_input("/pages/page.png", "cleaned", None, ".png");
    let dests = destinations(&input).expect("resolvable destinations");
    assert_eq!(dests.base, PathBuf::from("/pages/cleaned"));
    assert_eq!(
        dests.cleaned,
        PathBuf::from("/pages/cleaned/page_clean.png")
    );
    assert_eq!(dests.mask, PathBuf::from("/pages/cleaned/page_mask.png"));
    assert_eq!(dests.text, PathBuf::from("/pages/cleaned/page_text.png"));
}

#[test]
fn the_cleaned_suffix_falls_back_to_the_originals_extension() {
    // §12.3 step 1: `suffix = preferred_file_type.unwrap_or(original_path.extension())`.
    // Mask and text always use `preferred_mask_file_type`.
    let kept = destination_input("/pages/page.JPG", "/out", None, ".png");
    let dests = destinations(&kept).expect("resolvable");
    assert_eq!(dests.cleaned, PathBuf::from("/out/page_clean.jpg"));
    assert_eq!(dests.mask, PathBuf::from("/out/page_mask.png"));

    let preferred = destination_input("/pages/page.jpg", "/out", Some(".WEBP"), "tiff");
    let dests = destinations(&preferred).expect("resolvable");
    assert_eq!(dests.cleaned, PathBuf::from("/out/page_clean.webp"));
    assert_eq!(dests.mask, PathBuf::from("/out/page_mask.tiff"));
    assert_eq!(dests.text, PathBuf::from("/out/page_text.tiff"));

    // §16.11 item 4: an empty `preferred_file_type` means "keep the original suffix",
    // matching `GeneralConfig::cleaned_suffix`'s empty-string convention (§6).
    let empty = destination_input("/pages/page.bmp", "/out", Some(""), ".png");
    assert_eq!(
        destinations(&empty).expect("resolvable").cleaned,
        PathBuf::from("/out/page_clean.bmp")
    );
}

#[test]
fn the_export_path_supplies_the_stem_not_the_original_path() {
    // §12.2: `export_path` is the logical output identity and differs from
    // `original_path` for merged strips (E5 re-points it before calling us).
    let mut input = destination_input("/pages/page_seg_0.png", "/out", None, ".png");
    input.export_path = PathBuf::from("/pages/page.png");
    let dests = destinations(&input).expect("resolvable");
    assert_eq!(dests.cleaned, PathBuf::from("/out/page_clean.png"));
}

#[test]
fn unresolvable_destinations_are_invalid_input_errors() {
    // §16.11 item 4: a stem-less `export_path`, and no suffix from either source.
    let mut no_stem = destination_input("/pages/page.png", "/out", None, ".png");
    no_stem.export_path = PathBuf::from("/pages/..");
    assert!(matches!(
        destinations(&no_stem),
        Err(StageError::InvalidInput(_))
    ));

    let no_suffix = destination_input("/pages/page", "/out", None, ".png");
    let error = destinations(&no_suffix).expect_err("no suffix is resolvable");
    assert!(matches!(error, StageError::InvalidInput(_)), "{error:?}");
    assert!(error.to_string().contains("preferred_file_type"), "{error}");
}

fn destination_input(
    original: &str,
    output_dir: &str,
    preferred_file_type: Option<&str>,
    preferred_mask_file_type: &str,
) -> ExportInput {
    ExportInput {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from(original),
        export_path: PathBuf::from(original),
        output_dir: PathBuf::from(output_dir),
        outputs: all_outputs(),
        sources: ExportSources::default(),
        preferred_file_type: preferred_file_type.map(str::to_string),
        preferred_mask_file_type: preferred_mask_file_type.to_string(),
        denoising_enabled: true,
        inpainting_enabled: false,
    }
}

// ------------------------------------------------------------- §12.3 steps 2-8

#[test]
fn a4_run_creates_missing_parent_directories() {
    // §12.7(A)4: "Parent directories are created."
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("deeply/nested/out");
    let output = run(input_with(
        &original,
        &base,
        full_sources(),
        only_cleaned(),
        true,
    ))
    .expect("export succeeds");
    assert!(base.is_dir(), "mkdir -p must have run");
    assert_eq!(output.files_written, vec![base.join("page_clean.png")]);
    assert!(base.join("page_clean.png").is_file());
}

#[test]
fn a5_exactly_one_cleaned_file_is_written_and_it_is_the_denoised_one() {
    // §12.7(A)5, end to end: precedence chooses the denoised source, and
    // `files_written` lists exactly the files that exist on disk afterwards.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");
    let output = run(input_with(
        &original,
        &base,
        full_sources(),
        all_outputs(),
        true,
    ))
    .expect("export succeeds");

    assert_eq!(
        output.files_written,
        vec![
            base.join("page_clean.png"),
            base.join("page_mask.png"),
            base.join("page_text.png"),
        ],
        "ordered cleaned, mask, text (§16.11 item 4)"
    );
    assert_eq!(on_disk(&base), files_as_set(&output));

    // The denoised source is the one that landed (its text rect is value 20, the
    // masked source's is 10 -- see `common::full_sources`).
    let exported = pc_testkit::images::load_luma8(base.join("page_clean.png"));
    assert_eq!(exported.get_pixel(5, 5).0[0], 20);
}

#[test]
fn a5_with_denoising_disabled_the_masked_source_is_exported() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");
    run(input_with(
        &original,
        &base,
        full_sources(),
        only_cleaned(),
        false,
    ))
    .expect("export succeeds");
    let exported = pc_testkit::images::load_luma8(base.join("page_clean.png"));
    assert_eq!(exported.get_pixel(5, 5).0[0], 10);
}

#[test]
fn a6_save_only_mask_writes_no_cleaned_and_no_text_file() {
    // §12.7(A)6.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");
    let output = run(input_with(
        &original,
        &base,
        full_sources(),
        only_mask(),
        true,
    ))
    .expect("export succeeds");
    assert_eq!(output.files_written, vec![base.join("page_mask.png")]);
    assert_eq!(on_disk(&base), files_as_set(&output));
}

#[test]
fn a1_the_cleaned_export_keeps_the_originals_colour_mode() {
    // §12.7(A)1's second half / §12.3 step 3: grayscale in => grayscale out, even
    // though the pipeline's internal composites are RGB(A) (§16.9 item 14 / §16.10 17).
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");
    let sources = ExportSources {
        masked: Some(handle(
            "/cache/page_clean.png",
            DynamicImage::ImageRgb8(gray_page(PAGE_SIZE, 200, &[]).to_rgb8()),
        )),
        ..ExportSources::default()
    };
    run(input_with(&original, &base, sources, only_cleaned(), false)).expect("export succeeds");
    assert_eq!(
        pc_export::formats::read_color_mode(&base.join("page_clean.png"))
            .expect("a readable header"),
        pc_export::ColorMode::L
    );
}

#[test]
fn a7_the_mask_is_upscaled_with_nearest_neighbour_only() {
    // §12.7(A)7, the regression test for §14.8/§15.8: every pixel of the exported mask
    // must be one of the source mask's own colours -- bilinear would manufacture
    // interpolated values at the fill's edges.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path()); // PAGE_SIZE = 40x24
    let base = dir.path().join("out");
    let half = (PAGE_SIZE.0 / 2, PAGE_SIZE.1 / 2);
    let small_mask = rgba_mask(half, &[(Rect::new(3, 3, 9, 8), [255, 255, 255, 255])]);
    let sources = ExportSources {
        final_mask: Some(handle(
            "/cache/page_combined_mask.png",
            DynamicImage::ImageRgba8(small_mask.clone()),
        )),
        ..ExportSources::default()
    };
    run(input_with(&original, &base, sources, only_mask(), false)).expect("export succeeds");

    let exported = pc_testkit::images::load_rgba8(base.join("page_mask.png"));
    assert_eq!(
        exported.dimensions(),
        PAGE_SIZE,
        "the mask is upscaled to the ORIGINAL image's size (§12.3 step 4)"
    );
    let source_colours = small_mask
        .pixels()
        .map(|pixel| pixel.0)
        .collect::<HashSet<_>>();
    for pixel in exported.pixels() {
        assert!(
            source_colours.contains(&pixel.0),
            "colour {:?} exists nowhere in the source mask -- that is interpolation",
            pixel.0
        );
    }
}

#[test]
fn the_denoise_branch_composites_the_noise_mask_over_the_upscaled_combined_mask() {
    // §12.3 step 4's second bullet, with §16.11 item 3's nearest-everywhere rule. §16.11
    // item 9's compositing half is superseded by §16.45 item 4: real (Porter-Duff)
    // source-over, `out_a = sa + da*(1 - sa)`, not `alpha_out = max(base_a, layer_a)`.
    // The layers below are fully opaque (a = 255), the regime both formulas agree on, so
    // no assertion in this test moves.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");
    let combined = rgba_mask(
        PAGE_SIZE,
        &[(Rect::new(4, 4, 12, 12), [255, 255, 255, 255])],
    );
    let noise = rgba_mask(PAGE_SIZE, &[(Rect::new(6, 6, 10, 10), [0, 0, 0, 255])]);
    let sources = ExportSources {
        final_mask: Some(handle("/cache/m.png", DynamicImage::ImageRgba8(combined))),
        denoise_mask: Some(handle("/cache/n.png", DynamicImage::ImageRgba8(noise))),
        ..ExportSources::default()
    };
    run(input_with(&original, &base, sources, only_mask(), true)).expect("export succeeds");

    let exported = pc_testkit::images::load_rgba8(base.join("page_mask.png"));
    assert_eq!(exported.dimensions(), PAGE_SIZE);
    assert_eq!(
        exported.get_pixel(0, 0).0,
        [0, 0, 0, 0],
        "untouched stays transparent"
    );
    assert_eq!(
        exported.get_pixel(5, 5).0,
        [255, 255, 255, 255],
        "combined mask only"
    );
    assert_eq!(
        exported.get_pixel(7, 7).0,
        [0, 0, 0, 255],
        "noise mask on top"
    );
}

#[test]
fn a8_a_text_export_to_a_non_alpha_format_flattens_onto_white_instead_of_failing() {
    // §12.7(A)8 / §12.3 step 5: warn and flatten, never error.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");
    let text = rgba_mask(PAGE_SIZE, &[(Rect::new(4, 4, 12, 12), [0, 0, 0, 255])]);
    let sources = ExportSources {
        isolated_text: Some(handle("/cache/t.png", DynamicImage::ImageRgba8(text))),
        ..ExportSources::default()
    };
    let mut input = input_with(&original, &base, sources, only_text(), false);
    input.preferred_mask_file_type = ".jpg".into();

    let output = run(input).expect("a non-alpha text target must not be an error");
    assert_eq!(output.files_written, vec![base.join("page_text.jpg")]);
    let exported = pc_testkit::images::load(base.join("page_text.jpg"));
    assert!(
        !pc_export::formats::color_mode_of(&exported).has_alpha(),
        "JPEG cannot carry alpha"
    );
    // The transparent surround became white; the opaque glyph area stayed black.
    let rgb = exported.to_rgb8();
    assert!(
        rgb.get_pixel(0, 0).0.iter().all(|c| *c >= 250),
        "{:?}",
        rgb.get_pixel(0, 0)
    );
    assert!(
        rgb.get_pixel(8, 8).0.iter().all(|c| *c <= 5),
        "{:?}",
        rgb.get_pixel(8, 8)
    );
}

#[test]
fn the_text_layer_keeps_its_alpha_when_the_target_supports_it() {
    // §12.3 step 5: "RGBA preserved".
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");
    let text = rgba_mask(PAGE_SIZE, &[(Rect::new(4, 4, 12, 12), [0, 0, 0, 255])]);
    let sources = ExportSources {
        isolated_text: Some(handle(
            "/cache/t.png",
            DynamicImage::ImageRgba8(text.clone()),
        )),
        ..ExportSources::default()
    };
    run(input_with(&original, &base, sources, only_text(), false)).expect("export succeeds");
    assert_eq!(
        pc_testkit::images::load_rgba8(base.join("page_text.png")),
        text
    );
}

#[test]
fn nothing_requested_and_nothing_available_are_both_empty_successes() {
    // §16.11 item 14 / §5.6: export of a page with no artifacts is a normal outcome.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");

    let nothing_requested = run(input_with(&original, &base, full_sources(), vec![], true))
        .expect("an empty request is not an error");
    assert_eq!(nothing_requested, ExportOutput::default());

    let nothing_available = run(input_with(
        &original,
        &base,
        ExportSources::default(),
        all_outputs(),
        true,
    ))
    .expect("empty sources are not an error");
    assert_eq!(nothing_available.files_written, Vec::<PathBuf>::new());
}

#[test]
fn an_unsupported_preferred_suffix_is_a_stage_error_at_runtime() {
    // §12.7(A)9's runtime backstop: config validation is the primary gate (pc-config),
    // but a hand-built input must not silently write a `.xyz` file.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let base = dir.path().join("out");
    let mut input = input_with(&original, &base, full_sources(), only_cleaned(), true);
    input.preferred_file_type = Some(".xyz".into());
    assert!(matches!(run(input), Err(StageError::UnsupportedFormat(_))));
}

#[test]
fn export_is_deterministic_for_identical_inputs() {
    // §5.7: identical inputs and config produce identical outputs, including ordering.
    let dir = tempfile::tempdir().expect("a temp dir");
    let original = write_original_png(dir.path());
    let first = dir.path().join("a");
    let second = dir.path().join("b");
    let left = run(input_with(
        &original,
        &first,
        full_sources(),
        all_outputs(),
        true,
    ))
    .expect("export succeeds");
    let right = run(input_with(
        &original,
        &second,
        full_sources(),
        all_outputs(),
        true,
    ))
    .expect("export succeeds");

    let names = |output: &ExportOutput| {
        output
            .files_written
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&left), names(&right));
    for (a, b) in left.files_written.iter().zip(&right.files_written) {
        assert_eq!(
            std::fs::read(a).unwrap(),
            std::fs::read(b).unwrap(),
            "{a:?}"
        );
    }
}

fn on_disk(base: &Path) -> HashSet<PathBuf> {
    std::fs::read_dir(base)
        .expect("the export directory exists")
        .map(|entry| entry.expect("a readable dir entry").path())
        .collect()
}

fn files_as_set(output: &ExportOutput) -> HashSet<PathBuf> {
    output.files_written.iter().cloned().collect()
}
