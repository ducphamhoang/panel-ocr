//! C3 tests -- spec §6 defaults. Frozen gates.
//!
//! The contract these lock: `DEFAULT_PROFILE_TOML` (the literal §6 block) and
//! `Profile::default()` are the *same* configuration. Per §16.5 item 4 the
//! comparison is by **value**, not by text: §6 writes `filter_strength = 10` as an
//! integer while the field is `f64`, so a serialised default would say `10.0`.

use pc_config::{
    MaskRefineMode, OcrLanguageSetting, Profile, ProfileDocument, ReadingOrder,
    DEFAULT_PROFILE_TOML,
};

#[test]
// spec §6: the literal default TOML block parses, validates, and equals the
// programmatic default. This is the single test that keeps the two in sync.
fn default_toml_equals_default_profile() {
    let doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML)
        .expect("the spec §6 default profile must load and validate");
    assert_eq!(*doc.profile(), Profile::default());
}

#[test]
// spec §6: the shipped default profile contains no unknown keys -- if it did, every
// `profile new` would immediately WARN at the user.
fn default_toml_produces_no_warnings() {
    let doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    assert_eq!(doc.warnings(), &[]);
}

#[test]
// spec §6: `Profile::TABLES` is the unknown-key oracle, so it must describe exactly
// the tables and keys present in the default document -- no more, no less.
fn table_registry_matches_default_document() {
    let doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    let document = doc.document();

    let registry_tables: Vec<&str> = Profile::TABLES.iter().map(|(t, _)| *t).collect();
    let document_tables: Vec<String> = document.iter().map(|(k, _)| k.to_string()).collect();
    assert_eq!(registry_tables, document_tables);

    for (table, keys) in Profile::TABLES {
        let item = document
            .get(table)
            .unwrap_or_else(|| panic!("default document is missing table [{table}]"));
        let doc_keys: Vec<String> = item
            .as_table()
            .unwrap_or_else(|| panic!("[{table}] is not a table"))
            .iter()
            .map(|(k, _)| k.to_string())
            .collect();
        let known: Vec<String> = keys.iter().map(|k| k.to_string()).collect();
        assert_eq!(known, doc_keys, "key registry mismatch for [{table}]");
    }
}

// ------------------------------------------------------------------ [general]

#[test]
// spec §6 [general] -- every value transcribed from upstream config.py
fn general_defaults() {
    let g = Profile::default().general;
    assert_eq!(g.preferred_file_type, "");
    assert_eq!(g.preferred_mask_file_type, ".png");
    assert_eq!(g.input_height_lower_target, 1000);
    assert_eq!(g.input_height_upper_target, 4000);
    assert!(g.split_long_strips);
    assert_eq!(g.preferred_split_height, 2000);
    assert_eq!(g.split_tolerance_margin, 500);
    assert_eq!(g.long_strip_aspect_ratio, 0.33);
    assert!(g.merge_after_split);
    assert_eq!(g.max_threads, 0);
    assert!(!g.always_cache_masks);
}

#[test]
// spec §6: empty preferred_file_type means "keep the original suffix", which is a
// distinct state from any concrete suffix -- it must not be normalised into ".png"
fn empty_preferred_file_type_means_keep_original() {
    assert_eq!(Profile::default().general.cleaned_suffix(), None);

    let mut p = Profile::default();
    p.general.preferred_file_type = ".JPG".into();
    assert_eq!(p.general.cleaned_suffix(), Some(".jpg".to_string()));
}

// ------------------------------------------------------------- [text_detector]

#[test]
// spec §6 [text_detector]
fn text_detector_defaults() {
    let t = Profile::default().text_detector;
    assert_eq!(t.model_path, "");
    assert_eq!(t.model_path(), None);
    assert_eq!(t.concurrent_models, 1);
    assert_eq!(t.intra_threads, 0);
    assert_eq!(t.inter_threads, 0);
    // §8.3 step 5 / §15.2: v1 ships Simple.
    assert_eq!(t.mask_refine_mode, MaskRefineMode::Simple);
}

// ------------------------------------------------------------- [preprocessor]

#[test]
// spec §6 [preprocessor]
fn preprocessor_defaults() {
    let p = Profile::default().preprocessor;
    assert_eq!(p.box_min_size, 400); // 20*20
    assert_eq!(p.suspicious_box_min_size, 40_000); // 200*200
    assert_eq!(p.box_overlap_threshold, 20.0);
    assert!(p.ocr_enabled);
    assert_eq!(p.ocr_language, OcrLanguageSetting::DetectBox);
    assert_eq!(p.reading_order, ReadingOrder::Auto);
    assert_eq!(p.ocr_max_size, 3000); // 30*100
    assert_eq!(p.ocr_blacklist_pattern, "[～．ー！？０-９~.!?0-9-]*");
    assert!(!p.ocr_strict_language);
    assert_eq!(p.box_padding_initial, 2);
    assert_eq!(p.box_right_padding_initial, 3);
    assert_eq!(p.box_padding_extended, 5);
    assert_eq!(p.box_right_padding_extended, 5);
    assert_eq!(p.box_reference_padding, 20);
}

#[test]
// spec §2.2 + §6: the two pinned OCR language settings map onto pc-core's Language;
// the two detect modes deliberately map to None (they are resolved at runtime)
fn ocr_language_setting_maps_to_core_language() {
    use pc_core::Language;
    assert_eq!(
        OcrLanguageSetting::Jpn.fixed_language(),
        Some(Language::Japanese)
    );
    assert_eq!(
        OcrLanguageSetting::Eng.fixed_language(),
        Some(Language::English)
    );
    assert_eq!(OcrLanguageSetting::DetectBox.fixed_language(), None);
    assert_eq!(OcrLanguageSetting::DetectPage.fixed_language(), None);
}

// ------------------------------------------------------------------- [masker]

#[test]
// spec §6 [masker]
fn masker_defaults() {
    let m = Profile::default().masker;
    assert_eq!(m.mask_growth_step_pixels, 2);
    assert_eq!(m.mask_growth_steps, 11);
    assert_eq!(m.min_mask_thickness, 4);
    assert!(m.allow_colored_masks);
    assert_eq!(m.off_white_max_threshold, 240);
    assert_eq!(m.mask_max_standard_deviation, 15.0);
    assert_eq!(m.mask_improvement_threshold, 0.1);
    assert!(!m.mask_selection_fast);
    assert_eq!(m.debug_mask_color, [108, 30, 240, 127]);
}

// ----------------------------------------------------------------- [denoiser]

#[test]
// spec §6 [denoiser]
fn denoiser_defaults() {
    let d = Profile::default().denoiser;
    assert!(d.denoising_enabled);
    assert_eq!(d.noise_min_standard_deviation, 0.25);
    assert_eq!(d.noise_outline_size, 5);
    assert_eq!(d.noise_fade_radius, 1);
    // §15.7: upstream's own default is false, so v1's default path is unaffected by
    // the joint-channel approximation.
    assert!(!d.colored_images);
    assert_eq!(d.filter_strength, 10.0);
    assert_eq!(d.color_filter_strength, 10.0);
    assert_eq!(d.template_window_size, 7);
    assert_eq!(d.search_window_size, 21);
}

#[test]
// spec §16.5 item 4: §6 writes `filter_strength = 10` (integer) for an f64 field.
// Both spellings must load to the same value; neither may be a parse error.
fn float_fields_accept_integer_and_float_literals() {
    let integral =
        ProfileDocument::parse("[denoiser]\nfilter_strength = 10\ncolor_filter_strength = 10\n")
            .unwrap();
    let fractional = ProfileDocument::parse(
        "[denoiser]\nfilter_strength = 10.0\ncolor_filter_strength = 10.0\n",
    )
    .unwrap();
    assert_eq!(integral.profile().denoiser.filter_strength, 10.0);
    assert_eq!(fractional.profile().denoiser.filter_strength, 10.0);
    assert_eq!(integral.profile(), fractional.profile());
}

// ----------------------------------------------------------- missing keys

#[test]
// spec §6: "Missing keys take defaults." An empty document is a valid profile.
fn empty_document_is_the_default_profile() {
    let doc = ProfileDocument::parse("").expect("an empty profile is valid");
    assert_eq!(*doc.profile(), Profile::default());
    assert_eq!(doc.warnings(), &[]);
}

#[test]
// spec §6: a partial table takes defaults for its absent keys and does not disturb
// the other tables
fn partial_table_takes_defaults_for_absent_keys() {
    let doc = ProfileDocument::parse("[masker]\nmask_growth_steps = 3\n").unwrap();
    let p = doc.profile();
    assert_eq!(p.masker.mask_growth_steps, 3);
    assert_eq!(p.masker.mask_growth_step_pixels, 2); // untouched default
    assert_eq!(p.denoiser, Profile::default().denoiser); // other tables untouched
}

// --------------------------------------------------------- enum spellings

#[test]
// spec §6: the enum spellings in the TOML are snake_case and are part of the file
// format contract -- renaming a variant must not silently change the accepted text
fn enum_spellings_are_snake_case() {
    let doc = ProfileDocument::parse(concat!(
        "[text_detector]\nmask_refine_mode = \"annotation\"\n",
        "[preprocessor]\nocr_language = \"detect_page\"\nreading_order = \"comic\"\n",
    ))
    .unwrap();
    let p = doc.profile();
    assert_eq!(p.text_detector.mask_refine_mode, MaskRefineMode::Annotation);
    assert_eq!(p.preprocessor.ocr_language, OcrLanguageSetting::DetectPage);
    assert_eq!(p.preprocessor.reading_order, ReadingOrder::Comic);
}

#[test]
// spec §16.5 item 3 (decided): §8.3 step 5 / §15.2 make `annotation` a *stage* error
// (`StageError::InvalidInput` from pc-detect), not a config-load error, so config must
// accept it.
fn annotation_refine_mode_loads_successfully() {
    let doc = ProfileDocument::parse("[text_detector]\nmask_refine_mode = \"annotation\"\n")
        .expect("config accepts annotation; pc-detect is what rejects it");
    assert_eq!(
        doc.profile().text_detector.mask_refine_mode,
        MaskRefineMode::Annotation
    );
}

#[test]
// spec §6: an enum value outside the documented set fails the load and names the key
fn unknown_enum_value_fails_load() {
    let err = ProfileDocument::parse("[preprocessor]\nreading_order = \"vertical\"\n")
        .expect_err("`vertical` is not a documented reading order");
    let msg = err.to_string();
    assert!(
        msg.contains("reading_order"),
        "error must name the key: {msg}"
    );
}
