//! C3 tests -- the app-level `Config` (spec §6).
//!
//! SPEC GAP (§16.5 item 2): §6 names this type and specifies nothing about it. Every
//! assertion here encodes a provisional design and must be reviewed by the architects
//! before it is treated as frozen.

use pc_config::{Config, ConfigDocument};
use std::path::PathBuf;

#[test]
// provisional: an absent config file behaves like an all-defaults config -- no saved
// profiles, no pinned default, platform cache dir
fn defaults_are_empty() {
    let c = Config::default();
    assert_eq!(c.default_profile, None);
    assert!(c.saved_profiles.is_empty());
    assert_eq!(c.cache_dir, None);
    assert!(c.validate().is_ok());
}

#[test]
// provisional: `default_profile` must name a saved profile, otherwise every run would
// fail later with a confusing "profile not found"
fn default_profile_must_be_a_saved_profile() {
    let mut c = Config::default();
    c.default_profile = Some("mine".into());
    let err = c
        .validate()
        .expect_err("unknown default profile must be rejected");
    assert_eq!(err.field(), Some("default_profile"));

    c.saved_profiles
        .insert("mine".into(), PathBuf::from("/tmp/mine.toml"));
    assert!(c.validate().is_ok());
    assert_eq!(
        c.profile_path("mine"),
        Some(std::path::Path::new("/tmp/mine.toml"))
    );
    assert_eq!(c.profile_path("other"), None);
}

#[test]
// §5.7 determinism: saved profiles serialise in sorted key order regardless of
// insertion order, so the config file does not churn between runs
fn saved_profiles_serialise_deterministically() {
    let mut c = Config::default();
    c.saved_profiles
        .insert("zulu".into(), PathBuf::from("/z.toml"));
    c.saved_profiles
        .insert("alpha".into(), PathBuf::from("/a.toml"));
    c.saved_profiles
        .insert("mike".into(), PathBuf::from("/m.toml"));

    let text = ConfigDocument::from_config(&c).to_toml_string();
    let alpha = text.find("alpha").unwrap();
    let mike = text.find("mike").unwrap();
    let zulu = text.find("zulu").unwrap();
    assert!(alpha < mike && mike < zulu, "not sorted:\n{text}");
}

#[test]
// spec §6: the app config gets the same unknown-key preservation as the profile
fn unknown_keys_are_preserved_and_warned() {
    let text = "# user comment\nfuture_option = 7\n";
    let doc = ConfigDocument::parse(text).unwrap();
    assert_eq!(doc.to_toml_string(), text);
    assert_eq!(doc.warnings().len(), 1, "{:?}", doc.warnings());
}

#[test]
// provisional: round-trip through text preserves every field
fn config_round_trips() {
    let mut c = Config::default();
    c.saved_profiles
        .insert("mine".into(), PathBuf::from("/tmp/mine.toml"));
    c.default_profile = Some("mine".into());
    c.cache_dir = Some(PathBuf::from("/var/cache/panel-ocr"));

    let text = ConfigDocument::from_config(&c).to_toml_string();
    assert_eq!(*ConfigDocument::parse(&text).unwrap().config(), c);
}
