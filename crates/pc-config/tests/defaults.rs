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
// spec §16.36 item 2: `Device` lives in `pc_core::device`, NOT in `pc-config` -- so that
// `pc-ocr` (which depends on `pc-core` only) can name it without gaining a `pc-config`
// edge that §16.5 item 1's dependency list does not mandate.
use pc_core::device::Device;

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
    // §16.36 item 1 + §16.22 item 5(a): the device key lives in [general] and defaults
    // to cpu -- "cuda" must be explicit, never auto-detected (DEVIATION(22)).
    assert_eq!(g.device, Device::Cpu);
}

#[test]
// spec §16.36 item 1: `device` must be a REGISTERED [general] key in the SHIPPED default
// profile, otherwise every `profile new` WARNs at the user about its own output. The key
// is named as a literal string here on purpose: `table_registry_matches_default_document`
// compares the registry against the document, so it stays green if BOTH lack `device`.
// The text value is read out of the `toml_edit` document, an oracle independent of the
// serde layer that produces `profile().general.device`.
fn the_general_device_key_is_registered_and_ships_as_cpu() {
    let (_, general_keys) = Profile::TABLES
        .iter()
        .find(|(table, _)| *table == "general")
        .expect("[general] must be in the key registry");
    assert!(
        general_keys.contains(&"device"),
        "`device` must be a known [general] key: {general_keys:?}"
    );

    let doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    let text_value = doc.document()["general"]["device"]
        .as_str()
        .expect("[general] device must be a TOML string in the shipped default profile");
    assert_eq!(text_value, "cpu");
    assert_eq!(doc.profile().general.device, Device::Cpu);
    assert_eq!(doc.warnings(), &[]);
}

#[test]
// spec §16.36 item 6 + §16.5 item 3's ratified split: config ACCEPTS `device = "cuda"`
// (parses AND validates); the refusal belongs at session creation (§16.36 items 3-4), not
// at config load -- otherwise `--detector replay`/`mock` runs, which create no session at
// all, would break for a device nothing in them ever uses. Mirrors
// `annotation_refine_mode_loads_successfully`.
fn cuda_device_loads_successfully() {
    let doc = ProfileDocument::parse("[general]\ndevice = \"cuda\"\n")
        .expect("config accepts cuda; session creation is what refuses it");
    assert_eq!(doc.profile().general.device, Device::Cuda);
    assert_eq!(doc.warnings(), &[]);
    // Accepting the value means accepting it in `validate()` too, not only in
    // `Deserialize` -- `ProfileDocument::parse` runs both, this pins the second.
    doc.profile()
        .validate()
        .expect("device = \"cuda\" is a valid profile in every build");
}

#[test]
// spec §16.36 item 1: `cpu | cuda` is the closed documented set. A v2 device name
// (CoreML/DirectML are v2 per §16's out-of-scope list) must fail the LOAD naming the key,
// rather than silently degrading to the cpu default -- §14 item 7's "an opt-in setting
// must not silently behave differently". Mirrors `unknown_enum_value_fails_load`.
fn unsupported_device_values_fail_load_naming_the_key() {
    for bad in ["mps", "directml", "coreml", "CPU"] {
        let err = ProfileDocument::parse(&format!("[general]\ndevice = \"{bad}\"\n"))
            .expect_err("an undocumented device must not load");
        let msg = err.to_string();
        assert!(
            msg.contains("general.device"),
            "`{bad}`: error must name the key: {msg}"
        );
        assert!(
            msg.contains("cpu") && msg.contains("cuda"),
            "`{bad}`: error must name the accepted spellings: {msg}"
        );
    }
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
    assert!(m.mask_fallback_to_lowest_deviation);
    assert_eq!(m.debug_mask_color, [108, 30, 240, 127]);
}

#[test]
// spec §16.35 items 4 and 8: this is a registered [masker] key in the shipped default
// document. The literal registry/document checks are separate from
// `default_toml_equals_default_profile`: that equality stays green if all three surfaces
// accidentally omit the key together.
fn the_mask_fallback_key_is_registered_and_ships_as_true() {
    let (_, masker_keys) = Profile::TABLES
        .iter()
        .find(|(table, _)| *table == "masker")
        .expect("[masker] must be in the key registry");
    assert!(
        masker_keys.contains(&"mask_fallback_to_lowest_deviation"),
        "the fallback must be a known [masker] key: {masker_keys:?}"
    );

    let doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    let text_value = doc.document()["masker"]["mask_fallback_to_lowest_deviation"]
        .as_bool()
        .expect("the shipped fallback value must be a TOML boolean");
    assert!(text_value);
    assert!(doc.profile().masker.mask_fallback_to_lowest_deviation);
    assert_eq!(doc.warnings(), &[]);
}

#[test]
// spec §16.35 item 4's DEVIATION(21) control: false is accepted by deserialization AND
// by config-load validation. The stage, not the loader, decides how the flag changes
// candidate selection.
fn mask_fallback_false_loads_and_validates() {
    let doc = ProfileDocument::parse("[masker]\nmask_fallback_to_lowest_deviation = false\n")
        .expect("the documented compatibility value must load and validate");
    assert!(!doc.profile().masker.mask_fallback_to_lowest_deviation);
    assert_eq!(doc.warnings(), &[]);
    doc.profile()
        .validate()
        .expect("mask_fallback_to_lowest_deviation = false is valid");
}

#[test]
// The key is boolean at the config boundary; a string that merely looks boolean must not
// silently fall back to true, and the load error must identify the key the user mistyped.
fn a_non_boolean_mask_fallback_value_fails_load_naming_the_key() {
    let error = ProfileDocument::parse("[masker]\nmask_fallback_to_lowest_deviation = \"false\"\n")
        .expect_err("a string is not a boolean config value");
    assert!(
        error
            .to_string()
            .contains("mask_fallback_to_lowest_deviation"),
        "error must name the key: {error}"
    );
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
        .expect("config accepts annotation as an opt-in refinement mode");
    assert_eq!(
        doc.profile().text_detector.mask_refine_mode,
        MaskRefineMode::Annotation
    );
}

// ---------------------------------------------------------------- [inpainter]

#[test]
// spec §6 as superseded by §16.38 item 13(a) -- the eight v1.5 inpainting keys. Every value
// is `config.py:817-824`'s DATACLASS default, which §16.38 item 7(b) established is
// upstream's runtime authority: `media/default.conf` disagrees on `min_inpainting_radius`
// (5 vs 7) and is read by zero upstream Python files, so it is documentation only. Taking
// the wrong file's values would move the eligibility filter's threshold.
fn inpainter_defaults() {
    let i = Profile::default().inpainter;
    // §16.38 item 15(e): the default-off flag is what makes LaMa imply no fixture re-record.
    assert!(!i.inpainting_enabled);
    assert_eq!(i.inpainting_min_std_dev, 15.0);
    assert_eq!(i.inpainting_max_mask_radius, 6);
    assert_eq!(i.min_inpainting_radius, 7);
    assert_eq!(i.max_inpainting_radius, 20);
    assert_eq!(i.inpainting_radius_multiplier, 0.2);
    assert_eq!(i.inpainting_isolation_radius, 5);
    assert_eq!(i.inpainting_fade_radius, 4);
}

#[test]
// spec §16.38 item 13(a): the eight keys must be REGISTERED and present in the SHIPPED
// default profile, otherwise every `profile new` WARNs at the user about its own output.
// The key names are literals here on purpose, for the reason
// `the_general_device_key_is_registered_and_ships_as_cpu` gives: the registry-vs-document
// comparison stays green if BOTH surfaces omit a key. The values are read out of the
// `toml_edit` document -- an oracle independent of the serde layer `inpainter_defaults`
// goes through.
fn the_inpainter_keys_are_registered_and_ship_the_documented_values() {
    let (_, keys) = Profile::TABLES
        .iter()
        .find(|(table, _)| *table == "inpainter")
        .expect("[inpainter] must be in the key registry");
    for key in [
        "inpainting_enabled",
        "inpainting_min_std_dev",
        "inpainting_max_mask_radius",
        "min_inpainting_radius",
        "max_inpainting_radius",
        "inpainting_radius_multiplier",
        "inpainting_isolation_radius",
        "inpainting_fade_radius",
    ] {
        assert!(
            keys.contains(&key),
            "`{key}` must be a known [inpainter] key: {keys:?}"
        );
    }
    // Anti-vacuity: a hard-coded count so the loop above cannot pass over a registry that
    // gained an extra ninth key nobody ratified.
    assert_eq!(
        keys.len(),
        8,
        "§16.38 item 13 declares eight keys: {keys:?}"
    );

    let doc = ProfileDocument::parse(DEFAULT_PROFILE_TOML).unwrap();
    let table = &doc.document()["inpainter"];
    assert_eq!(
        table["inpainting_enabled"].as_bool(),
        Some(false),
        "the shipped flag must be a TOML boolean set to false"
    );
    assert_eq!(table["inpainting_min_std_dev"].as_float(), Some(15.0));
    assert_eq!(table["inpainting_max_mask_radius"].as_integer(), Some(6));
    assert_eq!(table["min_inpainting_radius"].as_integer(), Some(7));
    assert_eq!(table["max_inpainting_radius"].as_integer(), Some(20));
    assert_eq!(table["inpainting_radius_multiplier"].as_float(), Some(0.2));
    assert_eq!(table["inpainting_isolation_radius"].as_integer(), Some(5));
    assert_eq!(table["inpainting_fade_radius"].as_integer(), Some(4));
    assert_eq!(doc.warnings(), &[]);
}

#[test]
// spec §16.38 item 13(c), the same config-accepts/stage-refuses split §16.36 item 6 ruled
// for `device = "cuda"`: `inpainting_enabled = true` must LOAD and VALIDATE in a build with
// no ONNX at all -- this test binary is one. Otherwise `--detector replay`/`mock` runs,
// which never construct an inpainting session, would break for a model nothing in them
// reads. `ProfileDocument::parse` runs deserialization and validation; the explicit
// `validate()` pins the second half rather than trusting the first.
fn inpainting_enabled_true_loads_and_validates() {
    let doc = ProfileDocument::parse("[inpainter]\ninpainting_enabled = true\n")
        .expect("config accepts inpainting_enabled; the stage is what refuses it");
    assert!(doc.profile().inpainter.inpainting_enabled);
    assert_eq!(doc.warnings(), &[]);
    doc.profile()
        .validate()
        .expect("inpainting_enabled = true is a valid profile in every build");
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
