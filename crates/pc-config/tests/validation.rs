//! C3 tests -- spec §6's validation paragraph. Frozen gates.
//!
//! Style: each test starts from a valid `Profile::default()` and breaks exactly one
//! field, so no test depends on the order in which rules are checked. Both sides of
//! every boundary are asserted.
//!
//! The handful of rules whose bound is unrepresentable in the field's Rust type
//! (spec §16.5 item 5 -- `off_white_max_threshold in 0..=255`, `min_mask_thickness >= 0`,
//! `noise_outline_size >= 0`, `noise_fade_radius >= 0`) are asserted at the *load*
//! level as "fails and names the key", since whether they surface as `Parse` or as
//! `Invalid` depends on a typing decision the spec does not make.

use pc_config::{ConfigError, Profile, ProfileDocument, SUPPORTED_OUTPUT_SUFFIXES};

fn assert_valid(p: &Profile) {
    assert!(
        p.validate().is_ok(),
        "expected valid, got {:?}",
        p.validate().err()
    );
}

fn assert_invalid_field(p: &Profile, field: &str) {
    let err = p
        .validate()
        .expect_err(&format!("expected `{field}` to be rejected"));
    assert_eq!(err.field(), Some(field), "wrong rule fired: {err}");
}

/// For rules whose violation cannot be represented in the field's Rust type.
fn assert_load_fails_naming(toml: &str, key: &str) {
    let err = ProfileDocument::parse(toml)
        .err()
        .unwrap_or_else(|| panic!("expected `{key}` to fail config load"));
    assert!(
        err.to_string().contains(key),
        "error must name `{key}`: {err}"
    );
}

#[test]
// spec §6: the shipped defaults must themselves pass every validation rule
fn defaults_are_valid() {
    assert_valid(&Profile::default());
}

#[test]
// §16.21 item 3: zero means "let ONNX Runtime choose", while recorded fixtures remain
// reproducible because explicit counts such as 1 and 8 must still parse and serialize.
fn onnx_thread_counts_default_and_explicit_values_round_trip() {
    let defaults =
        ProfileDocument::parse("[text_detector]\nintra_threads = 0\ninter_threads = 0\n").unwrap();
    assert_eq!(defaults.profile().text_detector.intra_threads, 0);
    assert_eq!(defaults.profile().text_detector.inter_threads, 0);

    let mut explicit = Profile::default();
    explicit.text_detector.intra_threads = 1;
    explicit.text_detector.inter_threads = 8;
    let serialized = ProfileDocument::from_profile(&explicit).to_toml_string();
    let restored = ProfileDocument::parse(&serialized).unwrap();
    assert_eq!(restored.profile().text_detector.intra_threads, 1);
    assert_eq!(restored.profile().text_detector.inter_threads, 8);
}

#[test]
// §16.21 item 3: a count beyond the signed 32-bit value accepted by ORT's C API is an
// invalid configuration and must fail during profile load, before any image is processed.
fn onnx_thread_counts_above_ort_limit_fail_at_config_load() {
    for key in ["intra_threads", "inter_threads"] {
        let text = format!("[text_detector]\n{key} = 2147483648\n");
        assert_load_fails_naming(&text, key);
    }
}

// ============================================================ [general]

#[test]
// spec §6: long_strip_aspect_ratio > 0 (strict)
fn long_strip_aspect_ratio_must_be_positive() {
    let mut p = Profile::default();

    p.general.long_strip_aspect_ratio = 0.0;
    assert_invalid_field(&p, "general.long_strip_aspect_ratio");

    p.general.long_strip_aspect_ratio = -0.1;
    assert_invalid_field(&p, "general.long_strip_aspect_ratio");

    // NaN is not > 0, so it must be rejected too -- otherwise it silently disables the
    // §8.6/§8.7(B)8 aspect gate.
    p.general.long_strip_aspect_ratio = f64::NAN;
    assert_invalid_field(&p, "general.long_strip_aspect_ratio");

    p.general.long_strip_aspect_ratio = f64::MIN_POSITIVE;
    assert_valid(&p);

    p.general.long_strip_aspect_ratio = 0.33;
    assert_valid(&p);
}

// --------------------------------------------- suffix validation (§6 + §12.3 step 6)

#[test]
// spec §6 / §12.7(A)9: an unknown preferred_file_type fails CONFIG VALIDATION (not at
// runtime) with an error naming the supported list
fn unknown_preferred_file_type_fails_validation_naming_the_list() {
    let mut p = Profile::default();
    p.general.preferred_file_type = ".xyz".into();

    let err = p
        .validate()
        .expect_err(".xyz is not a supported output suffix");
    assert_eq!(err.field(), Some("general.preferred_file_type"));

    let msg = err.to_string();
    assert!(
        msg.contains(".xyz"),
        "error must quote the bad value: {msg}"
    );
    for suffix in SUPPORTED_OUTPUT_SUFFIXES {
        assert!(msg.contains(suffix), "error must name `{suffix}`: {msg}");
    }
}

#[test]
// spec §6: every suffix in the §12.3 step 6 table is accepted for both keys
fn all_supported_suffixes_are_accepted() {
    for suffix in SUPPORTED_OUTPUT_SUFFIXES {
        let mut p = Profile::default();
        p.general.preferred_file_type = (*suffix).into();
        p.general.preferred_mask_file_type = (*suffix).into();
        assert_valid(&p);
    }
}

#[test]
// spec §6: `.jp2` is a valid INPUT suffix but is rejected as an output suffix,
// because the `image` crate has no JPEG2000 encoder (§12.3 step 6 note, workspace
// Cargo.toml comment). The message must make the input/output distinction clear
// rather than just saying "unsupported".
fn jp2_is_rejected_as_an_output_suffix() {
    let mut p = Profile::default();
    p.general.preferred_file_type = ".jp2".into();

    let err = p.validate().expect_err(".jp2 cannot be written");
    assert_eq!(err.field(), Some("general.preferred_file_type"));
    let msg = err.to_string();
    assert!(msg.contains(".jp2"), "{msg}");
    assert!(
        msg.contains("input"),
        "message must explain .jp2 is input-only: {msg}"
    );
}

#[test]
// spec §6: the mask suffix is validated by the same rule as the cleaned suffix
fn unknown_preferred_mask_file_type_fails_validation() {
    let mut p = Profile::default();
    p.general.preferred_mask_file_type = ".xyz".into();
    assert_invalid_field(&p, "general.preferred_mask_file_type");
}

#[test]
// spec §6: empty preferred_file_type is legal ("keep original suffix");
// spec §16.5 item 6 (decided here): empty preferred_mask_file_type is NOT -- there is
// no "original" mask file to keep the suffix of.
fn empty_suffix_is_legal_only_for_the_cleaned_output() {
    let mut p = Profile::default();
    p.general.preferred_file_type = String::new();
    assert_valid(&p);

    p.general.preferred_mask_file_type = String::new();
    assert_invalid_field(&p, "general.preferred_mask_file_type");
}

#[test]
// spec §16.5 item 6 (decided here): suffix matching is ASCII-case-insensitive
// and the leading dot is REQUIRED -- "png" is a mistake worth reporting, not a synonym
fn suffix_matching_is_case_insensitive_but_requires_the_dot() {
    let mut p = Profile::default();

    p.general.preferred_file_type = ".PNG".into();
    assert_valid(&p);
    p.general.preferred_file_type = ".Jpeg".into();
    assert_valid(&p);

    p.general.preferred_file_type = "png".into();
    assert_invalid_field(&p, "general.preferred_file_type");
}

// ======================================================= [preprocessor]

#[test]
// spec §6: 0 <= box_overlap_threshold <= 100 (inclusive both ends -- it is a percent)
fn box_overlap_threshold_bounds() {
    let mut p = Profile::default();

    p.preprocessor.box_overlap_threshold = -0.001;
    assert_invalid_field(&p, "preprocessor.box_overlap_threshold");

    p.preprocessor.box_overlap_threshold = 0.0;
    assert_valid(&p);

    p.preprocessor.box_overlap_threshold = 100.0;
    assert_valid(&p);

    p.preprocessor.box_overlap_threshold = 100.001;
    assert_invalid_field(&p, "preprocessor.box_overlap_threshold");
}

#[test]
// spec §6: ocr_blacklist_pattern must compile as a regex -- a bad pattern is a config
// error, not a per-image failure at OCR time (§9.7(A)8 depends on it compiling)
fn ocr_blacklist_pattern_must_compile() {
    let mut p = Profile::default();

    p.preprocessor.ocr_blacklist_pattern = "[unclosed".into();
    assert_invalid_field(&p, "preprocessor.ocr_blacklist_pattern");

    // the default pattern, and the inert ".*" used by run_ocr (§15.5), both compile
    p.preprocessor.ocr_blacklist_pattern = "[～．ー！？０-９~.!?0-9-]*".into();
    assert_valid(&p);
    p.preprocessor.ocr_blacklist_pattern = ".*".into();
    assert_valid(&p);

    // an empty pattern is a legal regex (it matches the empty string) -- do not
    // special-case it into an error
    p.preprocessor.ocr_blacklist_pattern = String::new();
    assert_valid(&p);
}

// ============================================================= [masker]

#[test]
// spec §6: mask_growth_step_pixels >= 1
fn mask_growth_step_pixels_at_least_one() {
    let mut p = Profile::default();
    p.masker.mask_growth_step_pixels = 0;
    assert_invalid_field(&p, "masker.mask_growth_step_pixels");
    p.masker.mask_growth_step_pixels = 1;
    assert_valid(&p);
}

#[test]
// spec §6: mask_growth_steps >= 1
fn mask_growth_steps_at_least_one() {
    let mut p = Profile::default();
    p.masker.mask_growth_steps = 0;
    assert_invalid_field(&p, "masker.mask_growth_steps");
    p.masker.mask_growth_steps = 1;
    assert_valid(&p);
}

#[test]
// spec §6: 0.0 <= mask_improvement_threshold < 1.0 -- the upper bound is EXCLUSIVE,
// because at 1.0 the §10.7(A)8 acceptance test `dev <= best * (1 - threshold)`
// degenerates to `dev <= 0`
fn mask_improvement_threshold_bounds() {
    let mut p = Profile::default();

    p.masker.mask_improvement_threshold = -0.001;
    assert_invalid_field(&p, "masker.mask_improvement_threshold");

    p.masker.mask_improvement_threshold = 0.0;
    assert_valid(&p);

    p.masker.mask_improvement_threshold = 0.999;
    assert_valid(&p);

    p.masker.mask_improvement_threshold = 1.0;
    assert_invalid_field(&p, "masker.mask_improvement_threshold");
}

#[test]
// spec §6: mask_max_standard_deviation > 0 (strict)
fn mask_max_standard_deviation_must_be_positive() {
    let mut p = Profile::default();
    p.masker.mask_max_standard_deviation = 0.0;
    assert_invalid_field(&p, "masker.mask_max_standard_deviation");
    p.masker.mask_max_standard_deviation = -1.0;
    assert_invalid_field(&p, "masker.mask_max_standard_deviation");
    p.masker.mask_max_standard_deviation = 0.001;
    assert_valid(&p);
}

#[test]
// spec §6: off_white_max_threshold in 0..=255 -- both ends inclusive and both legal
fn off_white_max_threshold_endpoints_are_legal() {
    let mut p = Profile::default();
    p.masker.off_white_max_threshold = 0;
    assert_valid(&p);
    p.masker.off_white_max_threshold = 255;
    assert_valid(&p);
}

#[test]
// spec §6 + §16.5 item 5 (flagged): an out-of-range off_white_max_threshold must fail
// CONFIG LOAD and name the key. Whether that surfaces as a type error or a validation
// error is a typing decision the spec does not make, so only the observable behaviour
// is frozen here.
fn off_white_max_threshold_out_of_range_fails_load() {
    assert_load_fails_naming(
        "[masker]\noff_white_max_threshold = 256\n",
        "off_white_max_threshold",
    );
    assert_load_fails_naming(
        "[masker]\noff_white_max_threshold = -1\n",
        "off_white_max_threshold",
    );
}

#[test]
// spec §6 + §16.5 item 5: min_mask_thickness >= 0 -- 0 is legal, negative fails load
fn min_mask_thickness_non_negative() {
    let mut p = Profile::default();
    p.masker.min_mask_thickness = 0;
    assert_valid(&p);
    assert_load_fails_naming("[masker]\nmin_mask_thickness = -1\n", "min_mask_thickness");
}

// =========================================================== [denoiser]

#[test]
// spec §6: template_window_size odd and >= 3
fn template_window_size_odd_and_at_least_three() {
    let mut p = Profile::default();

    p.denoiser.template_window_size = 1; // odd but < 3
    assert_invalid_field(&p, "denoiser.template_window_size");

    p.denoiser.template_window_size = 2; // even and < 3
    assert_invalid_field(&p, "denoiser.template_window_size");

    p.denoiser.template_window_size = 4; // even
    assert_invalid_field(&p, "denoiser.template_window_size");

    p.denoiser.template_window_size = 0;
    assert_invalid_field(&p, "denoiser.template_window_size");

    p.denoiser.template_window_size = 3;
    assert_valid(&p);
    p.denoiser.template_window_size = 7; // the default
    assert_valid(&p);
}

#[test]
// spec §6: search_window_size odd and >= 3 -- the same rule, separately enforced, so
// that the error names the right key
fn search_window_size_odd_and_at_least_three() {
    let mut p = Profile::default();

    p.denoiser.search_window_size = 20;
    assert_invalid_field(&p, "denoiser.search_window_size");

    p.denoiser.search_window_size = 2;
    assert_invalid_field(&p, "denoiser.search_window_size");

    p.denoiser.search_window_size = 3;
    assert_valid(&p);
    p.denoiser.search_window_size = 21; // the default
    assert_valid(&p);
}

#[test]
// spec §6: filter_strength > 0.0 (strict) -- h = 0 would make every NLM weight 1
fn filter_strength_must_be_positive() {
    let mut p = Profile::default();
    p.denoiser.filter_strength = 0.0;
    assert_invalid_field(&p, "denoiser.filter_strength");
    p.denoiser.filter_strength = -1.0;
    assert_invalid_field(&p, "denoiser.filter_strength");
    p.denoiser.filter_strength = 0.001;
    assert_valid(&p);
}

#[test]
// spec §6: color_filter_strength > 0.0 -- still validated even though v1 ignores it
// (§14.7), because a v1.5 upgrade must not suddenly surface a bad stored value
fn color_filter_strength_must_be_positive() {
    let mut p = Profile::default();
    p.denoiser.color_filter_strength = 0.0;
    assert_invalid_field(&p, "denoiser.color_filter_strength");
    p.denoiser.color_filter_strength = 0.001;
    assert_valid(&p);
}

#[test]
// spec §6 + §16.5 item 5: noise_outline_size >= 0 and noise_fade_radius >= 0 --
// 0 is legal for both, negative fails load
fn noise_sizes_non_negative() {
    let mut p = Profile::default();
    p.denoiser.noise_outline_size = 0;
    p.denoiser.noise_fade_radius = 0;
    assert_valid(&p);

    assert_load_fails_naming(
        "[denoiser]\nnoise_outline_size = -1\n",
        "noise_outline_size",
    );
    assert_load_fails_naming("[denoiser]\nnoise_fade_radius = -1\n", "noise_fade_radius");
}

// ================================================================== [inpainter]
//
// spec §6 as superseded by §16.38 item 13(b), mirroring `config.py:917-932`'s `fix()` as
// ERRORS rather than silent clamps.

#[test]
// §16.38 item 13(b): `inpainting_min_std_dev >= 0.0`. Both sides of the boundary, and 0.0
// itself is valid -- `fix()` clamps at 0, it does not require a positive value.
fn inpainting_min_std_dev_must_not_be_negative() {
    let mut p = Profile::default();
    p.inpainter.inpainting_min_std_dev = 0.0;
    assert_valid(&p);

    p.inpainter.inpainting_min_std_dev = -0.001;
    assert_invalid_field(&p, "inpainter.inpainting_min_std_dev");
}

#[test]
// §16.38 item 13(b) singles this key out because it is the block's only non-`Pixels`,
// non-threshold key and therefore the easy one to omit -- and omitting it would be a real
// gap, not a tidiness one: the multiplier is LIVE (it feeds item 3(e)'s
// `growth = min(min_inpainting_radius + int(deviation * multiplier), max_inpainting_radius)`),
// so a negative value drives `growth` below `min_inpainting_radius` and potentially negative,
// into a growth kernel whose `size` is unsigned.
fn inpainting_radius_multiplier_must_not_be_negative() {
    let mut p = Profile::default();
    p.inpainter.inpainting_radius_multiplier = 0.0;
    assert_valid(&p);

    p.inpainter.inpainting_radius_multiplier = -0.2;
    assert_invalid_field(&p, "inpainter.inpainting_radius_multiplier");
}

#[test]
// §16.38 item 13(b): `max_inpainting_radius >= min_inpainting_radius`, which is upstream's
// own `config.py:932` invariant. Equality is the boundary and must be ACCEPTED -- a
// strictly-greater reading would reject a profile upstream repairs to exactly this state.
fn max_inpainting_radius_must_be_at_least_min() {
    let mut p = Profile::default();
    p.inpainter.min_inpainting_radius = 12;
    p.inpainter.max_inpainting_radius = 12;
    assert_valid(&p);

    p.inpainter.max_inpainting_radius = 11;
    assert_invalid_field(&p, "inpainter.max_inpainting_radius");

    // The message must name the other key, or the user cannot tell which of the two to move.
    let error = p
        .validate()
        .expect_err("11 < 12 must be rejected")
        .to_string();
    assert!(
        error.contains("min_inpainting_radius"),
        "the error must name the key it is compared against: {error}"
    );
}

#[test]
// §16.38 item 13(b) requires each of the five `Pixels` keys to be `>= 0`. That bound is
// unrepresentable in their `u32` type -- exactly the situation §16.5 item 5 and this file's
// header already handle for `min_mask_thickness` / `noise_outline_size` / `noise_fade_radius`
// -- so it is asserted at the LOAD level as "fails and names the key". Falsifiable: change
// any of these fields to `i64` and the load starts succeeding.
fn each_inpainter_pixel_radius_rejects_a_negative_literal_naming_the_key() {
    for key in [
        "inpainting_max_mask_radius",
        "min_inpainting_radius",
        "max_inpainting_radius",
        "inpainting_isolation_radius",
        "inpainting_fade_radius",
    ] {
        assert_load_fails_naming(&format!("[inpainter]\n{key} = -1\n"), key);
    }
}

#[test]
// §16.38 item 13(a): zero is a legal value for every one of the five radius keys (upstream's
// `fix()` clamps TO zero, so zero must be reachable). Written as a control for the test
// above, so "fails the load" is not mistaken for "any small value is rejected".
fn zero_is_valid_for_every_inpainter_pixel_radius() {
    let doc = ProfileDocument::parse(concat!(
        "[inpainter]\n",
        "inpainting_max_mask_radius = 0\n",
        "min_inpainting_radius = 0\n",
        "max_inpainting_radius = 0\n",
        "inpainting_isolation_radius = 0\n",
        "inpainting_fade_radius = 0\n",
    ))
    .expect("zero is a legal radius");
    let i = &doc.profile().inpainter;
    assert_eq!(i.inpainting_max_mask_radius, 0);
    assert_eq!(i.min_inpainting_radius, 0);
    assert_eq!(i.max_inpainting_radius, 0);
    assert_eq!(i.inpainting_isolation_radius, 0);
    assert_eq!(i.inpainting_fade_radius, 0);
}

// ==================================================== load-time integration

#[test]
// spec §6: validation runs at LOAD -- a syntactically fine but invalid document must
// not produce a usable ProfileDocument
fn parse_runs_validation() {
    let err = ProfileDocument::parse("[masker]\nmask_growth_steps = 0\n")
        .expect_err("load must fail on an invalid value");
    assert_eq!(err.field(), Some("masker.mask_growth_steps"));
}

#[test]
// spec §6: validate_all reports every violation, so `profile validate` can show the
// user all their mistakes at once instead of one per run
fn validate_all_reports_every_violation() {
    let mut p = Profile::default();
    p.masker.mask_growth_steps = 0;
    p.denoiser.search_window_size = 20;
    p.general.preferred_file_type = ".xyz".into();

    let errors: Vec<ConfigError> = p.validate_all();
    let fields: Vec<Option<&str>> = errors.iter().map(ConfigError::field).collect();

    assert_eq!(errors.len(), 3, "got {fields:?}");
    // order is Profile::TABLES order, so it is deterministic
    assert_eq!(
        fields,
        vec![
            Some("general.preferred_file_type"),
            Some("masker.mask_growth_steps"),
            Some("denoiser.search_window_size"),
        ]
    );
}
