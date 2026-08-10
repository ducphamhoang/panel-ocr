//! C1 tests — spec §2.2 (`Language`), §2.8 (`Step` / `Output` / `cache_suffix`).

use pc_core::{Language, Output, Step, RTL_BOX_ORDER_LANGUAGES};

#[test]
// spec §2.2: Language serializes lowercase; "unknown" is modelled as Option::None
fn language_json_is_lowercase_and_unknown_is_null() {
    assert_eq!(
        serde_json::to_string(&Language::Japanese).unwrap(),
        r#""japanese""#
    );
    assert_eq!(
        serde_json::to_string(&Language::English).unwrap(),
        r#""english""#
    );
    let unknown: Option<Language> = None;
    assert_eq!(serde_json::to_string(&unknown).unwrap(), "null");
    let back: Option<Language> = serde_json::from_str("null").unwrap();
    assert_eq!(back, None);
    let back: Option<Language> = serde_json::from_str(r#""japanese""#).unwrap();
    assert_eq!(back, Some(Language::Japanese));
}

#[test]
// spec §2.2: the RTL box-order set contains Japanese and not English — this drives
// the reading-order flip in §9.3 step 6
fn rtl_set_contains_only_japanese_in_v1() {
    assert!(RTL_BOX_ORDER_LANGUAGES.contains(&Language::Japanese));
    assert!(!RTL_BOX_ORDER_LANGUAGES.contains(&Language::English));
    assert_eq!(RTL_BOX_ORDER_LANGUAGES.len(), 1);
}

#[test]
// spec §2.8: Step is ordered so the pipeline can compare "have we reached X yet",
// with Detect == 1
fn step_ordering_follows_pipeline_order() {
    assert!(Step::Detect < Step::Preprocess);
    assert!(Step::Preprocess < Step::Mask);
    assert!(Step::Mask < Step::Denoise);
    assert!(Step::Denoise < Step::Export);
    assert!(Step::Denoise < Step::Inpaint);
    assert!(Step::Inpaint < Step::Export);
    assert_eq!(Step::Detect as i32, 1);
    assert_eq!(Step::Export as i32, 6);
}

#[test]
// spec §4.4: resume loads the checkpoint for Step::prev(); Detect has no predecessor
fn step_prev_walks_backwards_and_bottoms_out() {
    assert_eq!(Step::Detect.prev(), None);
    assert_eq!(Step::Preprocess.prev(), Some(Step::Detect));
    assert_eq!(Step::Mask.prev(), Some(Step::Preprocess));
    assert_eq!(Step::Denoise.prev(), Some(Step::Mask));
    assert_eq!(Step::Inpaint.prev(), Some(Step::Denoise));
    assert_eq!(Step::Export.prev(), Some(Step::Inpaint));
}

#[test]
// spec §2.8: cache_suffix values are upstream's file suffixes VERBATIM, so our
// cache can be diffed against upstream's during parity work. All 13 are frozen.
fn cache_suffixes_are_upstream_verbatim() {
    assert_eq!(Output::BaseImage.cache_suffix(), "_base.png");
    assert_eq!(Output::RawMask.cache_suffix(), "_raw_mask.png");
    assert_eq!(Output::RawJson.cache_suffix(), "#raw.json");
    assert_eq!(Output::CleanJson.cache_suffix(), "#clean.json");
    assert_eq!(Output::BoxMask.cache_suffix(), "_box_mask.png");
    assert_eq!(Output::CutMask.cache_suffix(), "_cut_mask.png");
    assert_eq!(Output::FinalMask.cache_suffix(), "_combined_mask.png");
    assert_eq!(Output::MaskOverlay.cache_suffix(), "_with_masks.png");
    assert_eq!(Output::IsolatedText.cache_suffix(), "_text.png");
    assert_eq!(Output::MaskedOutput.cache_suffix(), "_clean.png");
    assert_eq!(Output::MaskDataJson.cache_suffix(), "#mask_data.json");
    assert_eq!(Output::DenoiseMask.cache_suffix(), "_noise_mask.png");
    assert_eq!(Output::DenoisedOutput.cache_suffix(), "_clean_denoised.png");
}

#[test]
// spec §2.8: exactly 13 outputs in v1 (upstream's set minus inpainting), and every
// suffix is distinct — a collision would silently overwrite a cache artifact
fn output_set_is_complete_and_suffixes_are_unique() {
    assert_eq!(Output::ALL.len(), 13);
    let mut suffixes: Vec<&str> = Output::ALL.iter().map(|o| o.cache_suffix()).collect();
    suffixes.sort_unstable();
    let before = suffixes.len();
    suffixes.dedup();
    assert_eq!(suffixes.len(), before, "cache suffixes must be unique");
}

#[test]
// spec §2.8: Output::step() maps each artifact to the step that produces it
fn output_step_mapping() {
    for o in [Output::BaseImage, Output::RawMask, Output::RawJson] {
        assert_eq!(o.step(), Step::Detect, "{o:?}");
    }
    assert_eq!(Output::CleanJson.step(), Step::Preprocess);
    for o in [
        Output::BoxMask,
        Output::CutMask,
        Output::FinalMask,
        Output::MaskOverlay,
        Output::IsolatedText,
        Output::MaskedOutput,
        Output::MaskDataJson,
    ] {
        assert_eq!(o.step(), Step::Mask, "{o:?}");
    }
    for o in [Output::DenoiseMask, Output::DenoisedOutput] {
        assert_eq!(o.step(), Step::Denoise, "{o:?}");
    }
    // §2.8 defines no Export-step artifact: export writes final files, not cache ones
    assert!(Output::ALL.iter().all(|o| o.step() != Step::Export));
}

#[test]
// spec §2: "JSON is snake_case" applied to the §2.8 enums (decided: both carry
// #[serde(rename_all = "snake_case")]).
fn step_and_output_json_are_snake_case() {
    assert_eq!(serde_json::to_string(&Step::Detect).unwrap(), r#""detect""#);
    assert_eq!(
        serde_json::to_string(&Step::Preprocess).unwrap(),
        r#""preprocess""#
    );
    assert_eq!(
        serde_json::to_string(&Output::BaseImage).unwrap(),
        r#""base_image""#
    );
    assert_eq!(
        serde_json::to_string(&Output::MaskDataJson).unwrap(),
        r#""mask_data_json""#
    );
    assert_eq!(
        serde_json::to_string(&Output::DenoisedOutput).unwrap(),
        r#""denoised_output""#
    );
}

#[test]
// spec §2.8: every Step and Output round-trips through JSON regardless of the
// chosen wire representation (representation-agnostic companion to the test above)
fn step_and_output_round_trip() {
    for s in [
        Step::Detect,
        Step::Preprocess,
        Step::Mask,
        Step::Denoise,
        Step::Inpaint,
        Step::Export,
    ] {
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Step>(&json).unwrap(), s);
    }
    for o in Output::ALL {
        let json = serde_json::to_string(o).unwrap();
        assert_eq!(&serde_json::from_str::<Output>(&json).unwrap(), o);
    }
}
