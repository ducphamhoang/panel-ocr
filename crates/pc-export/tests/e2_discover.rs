//! Task **E2** -- availability -> precedence resolution and `--save-only-*` narrowing:
//! spec §12.3 step 2, §12.7(A)5/6, §16.11 items 2 and 3. FROZEN.

mod common;

use common::{full_sources, only_cleaned, only_mask, only_text};
use pc_core::Output;
use pc_export::discover::{all_exportable_outputs, is_requested, resolve, Category, MaskChoice};
use pc_export::ExportSources;

#[test]
fn the_category_map_covers_exactly_the_five_export_relevant_outputs() {
    // §16.11 item 2: `pc_core::Output` has no export-side variants (§2.8 documents
    // `Output::step()` as deliberately non-surjective), so requested outputs are read
    // through this map. Cache-only artifacts map to `None`.
    assert_eq!(Category::of(Output::MaskedOutput), Some(Category::Cleaned));
    assert_eq!(
        Category::of(Output::DenoisedOutput),
        Some(Category::Cleaned)
    );
    assert_eq!(Category::of(Output::FinalMask), Some(Category::Mask));
    assert_eq!(Category::of(Output::DenoiseMask), Some(Category::Mask));
    assert_eq!(Category::of(Output::IsolatedText), Some(Category::Text));
    for output in [
        Output::BaseImage,
        Output::RawMask,
        Output::RawJson,
        Output::CleanJson,
        Output::BoxMask,
        Output::CutMask,
        Output::MaskOverlay,
        Output::MaskDataJson,
    ] {
        assert_eq!(Category::of(output), None, "{output:?}");
    }
    // The helper the pipeline uses for an unrestricted run is exactly those five,
    // in `Output::ALL` declaration order (§5.7 determinism).
    assert_eq!(
        all_exportable_outputs(),
        vec![
            Output::FinalMask,
            Output::IsolatedText,
            Output::MaskedOutput,
            Output::DenoiseMask,
            Output::DenoisedOutput,
        ]
    );
    assert_eq!(
        Category::ALL,
        &[Category::Cleaned, Category::Mask, Category::Text],
        "the order `files_written` follows (§16.11 item 4)"
    );
}

#[test]
fn a_category_is_requested_when_any_of_its_variants_is_listed() {
    // §16.11 item 2, including the empty-list case: `outputs: []` requests nothing.
    assert!(is_requested(&only_cleaned(), Category::Cleaned));
    assert!(!is_requested(&only_cleaned(), Category::Mask));
    assert!(!is_requested(&only_cleaned(), Category::Text));
    assert!(is_requested(&[Output::DenoiseMask], Category::Mask));
    for category in Category::ALL {
        assert!(!is_requested(&[], *category));
        assert!(!is_requested(&[Output::BaseImage], *category));
        assert!(is_requested(&all_exportable_outputs(), *category));
    }
}

#[test]
fn a5_the_denoised_image_wins_over_the_masked_one() {
    // §12.7(A)5 / §12.3 step 2: `cleaned: denoised > masked`, exactly one is chosen.
    let sources = full_sources();
    let selection = resolve(&sources, &all_exportable_outputs(), true);
    assert_eq!(selection.cleaned, sources.denoised);
    assert_ne!(selection.cleaned, sources.masked);
}

#[test]
fn a5_with_denoising_disabled_the_masked_image_wins_even_if_a_denoised_one_is_cached() {
    // §12.3 step 2's explicit requirement: "a stale cached denoise artifact from a
    // previous run must not resurrect itself when denoising is now disabled".
    let sources = full_sources();
    let selection = resolve(&sources, &all_exportable_outputs(), false);
    assert_eq!(selection.cleaned, sources.masked);
    assert_eq!(
        selection.mask,
        Some(MaskChoice::FinalOnly(sources.final_mask.clone().unwrap())),
        "and the noise mask is likewise excluded from the mask composite"
    );
}

#[test]
fn the_masked_image_is_used_when_no_denoised_one_exists() {
    let sources = ExportSources {
        denoised: None,
        denoise_mask: None,
        ..full_sources()
    };
    let selection = resolve(&sources, &all_exportable_outputs(), true);
    assert_eq!(selection.cleaned, sources.masked);
    assert_eq!(
        selection.mask,
        Some(MaskChoice::FinalOnly(sources.final_mask.clone().unwrap()))
    );
}

#[test]
fn the_denoise_composite_branch_needs_both_mask_handles() {
    // §16.11 item 3: §12.3 step 4 composites the noise mask *over* the resized combined
    // mask, so `WithDenoise` requires both. A noise mask with no combined mask has no
    // defined output -- it WARNs and exports no mask, rather than erroring (§5.6).
    let sources = full_sources();
    assert_eq!(
        resolve(&sources, &all_exportable_outputs(), true).mask,
        Some(MaskChoice::WithDenoise {
            final_mask: sources.final_mask.clone().unwrap(),
            denoise_mask: sources.denoise_mask.clone().unwrap(),
        })
    );

    let orphan = ExportSources {
        final_mask: None,
        ..full_sources()
    };
    assert_eq!(resolve(&orphan, &all_exportable_outputs(), true).mask, None);

    let no_denoise_mask = ExportSources {
        denoise_mask: None,
        ..full_sources()
    };
    assert_eq!(
        resolve(&no_denoise_mask, &all_exportable_outputs(), true).mask,
        Some(MaskChoice::FinalOnly(
            no_denoise_mask.final_mask.clone().unwrap()
        ))
    );
}

#[test]
fn a6_save_only_mask_selects_no_cleaned_and_no_text() {
    // §12.7(A)6.
    let sources = full_sources();
    let selection = resolve(&sources, &only_mask(), true);
    assert!(selection.cleaned.is_none());
    assert!(selection.text.is_none());
    assert!(selection.mask.is_some());
    assert!(!selection.is_empty());
}

#[test]
fn save_only_cleaned_and_save_only_text_narrow_the_same_way() {
    let sources = full_sources();

    let cleaned_only = resolve(&sources, &only_cleaned(), true);
    assert_eq!(cleaned_only.cleaned, sources.denoised);
    assert!(cleaned_only.mask.is_none() && cleaned_only.text.is_none());

    let text_only = resolve(&sources, &only_text(), true);
    assert_eq!(text_only.text, sources.isolated_text);
    assert!(text_only.cleaned.is_none() && text_only.mask.is_none());
}

#[test]
fn narrowing_is_applied_after_precedence_so_the_two_cannot_interact() {
    // §16.11 item 3: `--save-only-cleaned` must not make the MASK's precedence rules
    // change what "cleaned" means, and vice versa. Denoised still wins under every
    // narrowing that includes the cleaned category.
    let sources = full_sources();
    for outputs in [only_cleaned(), all_exportable_outputs()] {
        assert_eq!(resolve(&sources, &outputs, true).cleaned, sources.denoised);
    }
}

#[test]
fn an_empty_request_or_empty_sources_selects_nothing() {
    // §16.11 item 14: a valid, non-error outcome.
    assert!(resolve(&full_sources(), &[], true).is_empty());
    assert!(resolve(&ExportSources::default(), &all_exportable_outputs(), true).is_empty());
    assert!(resolve(&ExportSources::default(), &[], false).is_empty());
    assert_eq!(
        resolve(&ExportSources::default(), &all_exportable_outputs(), true),
        pc_export::Selection::default()
    );
}

#[test]
fn text_selection_is_independent_of_the_denoise_state() {
    // §12.3 step 2: "text: isolated_text (independent)".
    let sources = full_sources();
    for denoising_enabled in [true, false] {
        assert_eq!(
            resolve(&sources, &all_exportable_outputs(), denoising_enabled).text,
            sources.isolated_text
        );
    }
}
