//! Task **X1/W1** — the platform-parameterized cache and config resolver (spec §16.33 items 3, 4, 5
//! and 13; §16.12 item 21 for everything those items leave standing).
//!
//! FROZEN once committed (CLAUDE.md). These tests are written before the implementation exists and
//! do not compile until `pc_cli::paths` gains `Platform`, `EnvSource`, `ProcessEnv`, `MapEnv` and
//! `DirEnv`. That is the intended starting state.
//!
//! **Why the resolver takes a platform and an environment instead of reading `cfg!` and `std::env`.**
//! Two thirds of what §16.33 ratifies is unreachable from a Linux or macOS test host: the whole
//! Windows column of both resolution tables, including the one decision that diverges from upstream
//! (`%LOCALAPPDATA%` for cache, `%APPDATA%` for config). A resolver that consults `cfg!` internally
//! can only ever be tested on the platform it is running on, which means CI would gate two of the
//! three supported platforms and the third would be gated by a Windows runner alone. Parameterizing
//! makes every cell of both tables reachable everywhere.
//!
//! **Where the expected values come from.** Each expectation is built as
//! `Path::new(<a root literal typed out here>).join(<a name literal typed out here>)`. `Path::join`
//! is `std`, not the artifact under test; the artifact under test is *which* environment variable and
//! *which* subdirectory the resolver picks. Composing the expectation with `join` rather than writing
//! a whole path string keeps the assertions separator-portable — a hard-coded
//! `"C:\\Users\\u\\AppData\\Local\\panel-ocr"` would not compare equal on a Linux host, where `\`
//! is not a separator — without letting the expectation come from the resolver (cookbook rules 7
//! and 13).

use pc_cli::paths::{self, DirEnv, MapEnv, Platform, ProcessEnv, APP_DIR_NAME, CONFIG_FILE_NAME};
use std::path::{Path, PathBuf};

/// All three supported platforms, so a table-driven test cannot quietly skip one.
const EVERY_PLATFORM: [Platform; 3] = [Platform::Linux, Platform::MacOs, Platform::Windows];

fn cache_dir(platform: Platform, env: &[(&str, &str)]) -> PathBuf {
    let env = MapEnv::from_pairs(env);
    DirEnv::new(platform, &env).cache_dir()
}

fn config_dir(platform: Platform, env: &[(&str, &str)]) -> PathBuf {
    let env = MapEnv::from_pairs(env);
    DirEnv::new(platform, &env).config_dir()
}

fn config_path(platform: Platform, env: &[(&str, &str)]) -> PathBuf {
    let env = MapEnv::from_pairs(env);
    DirEnv::new(platform, &env).config_path()
}

/// An environment where every root variable this project reads is set to a DISTINCT decoy, so an
/// assertion that the right one was chosen is not satisfiable by choosing any other.
fn every_root_set() -> Vec<(&'static str, &'static str)> {
    vec![
        ("XDG_CACHE_HOME", "/xdg/cache"),
        ("XDG_CONFIG_HOME", "/xdg/config"),
        ("HOME", "/home/u"),
        ("LOCALAPPDATA", r"C:\Users\u\AppData\Local"),
        ("APPDATA", r"C:\Users\u\AppData\Roaming"),
        ("USERPROFILE", r"C:\Users\u"),
    ]
}

#[test]
// spec §16.33 item 4 (E3) — `XDG_CACHE_HOME` wins on every platform, Windows included. Asserted for
// all three platforms in one loop over EVERY_PLATFORM so adding a platform cannot leave a gap, and
// asserted as an inequality against each platform's own default root as well, so "XDG wins" is not
// satisfied by a resolver that happens to produce the XDG path for another reason.
fn the_xdg_cache_variable_wins_on_every_platform() {
    let expected = Path::new("/xdg/cache").join(APP_DIR_NAME);
    for platform in EVERY_PLATFORM {
        assert_eq!(
            cache_dir(platform, &every_root_set()),
            expected,
            "{platform:?}: $XDG_CACHE_HOME must win over the platform default root"
        );
    }
    // The decoys must really be decoys: each platform's own default must differ from the XDG answer.
    for (platform, decoy_root) in [
        (Platform::Linux, "/home/u/.cache"),
        (Platform::MacOs, "/home/u/Library/Caches"),
        (Platform::Windows, r"C:\Users\u\AppData\Local"),
    ] {
        assert_ne!(
            Path::new(decoy_root).join(APP_DIR_NAME),
            expected,
            "{platform:?}: fixture sanity — the decoy root must differ from the XDG answer, or the \
             assertion above proves nothing"
        );
    }
}

#[test]
// spec §16.33 item 4 (E3) — the same for `XDG_CONFIG_HOME`, on every platform.
fn the_xdg_config_variable_wins_on_every_platform() {
    let expected = Path::new("/xdg/config").join(APP_DIR_NAME);
    for platform in EVERY_PLATFORM {
        assert_eq!(
            config_dir(platform, &every_root_set()),
            expected,
            "{platform:?}: $XDG_CONFIG_HOME must win over the platform default root"
        );
    }
}

#[test]
// spec §16.33 items 3 and 5 (E1), plus §14 item 19 — the ratified split, and the assertion that
// distinguishes it from upstream's measured behaviour.
//
// Upstream PanelCleaner puts BOTH its cache and its config under `%APPDATA%`. The two `assert_ne!`
// legs are what make this test fail on a resolver that ports upstream instead of implementing the
// ratified deviation, and they are what make it fail on a resolver that swaps the two variables —
// a swap that leaves every path shape and every path count unchanged (cookbook rule 13: cardinality
// is not identity, and neither is shape).
fn the_windows_cache_root_is_localappdata_and_the_config_root_is_appdata() {
    let local = r"C:\Users\u\AppData\Local";
    let roaming = r"C:\Users\u\AppData\Roaming";
    let env = [("LOCALAPPDATA", local), ("APPDATA", roaming)];

    assert_eq!(
        cache_dir(Platform::Windows, &env),
        Path::new(local).join(APP_DIR_NAME),
        "the Windows cache root is %LOCALAPPDATA%\\panel-ocr (§16.33 item 3)"
    );
    assert_eq!(
        config_dir(Platform::Windows, &env),
        Path::new(roaming).join(APP_DIR_NAME),
        "the Windows config root is %APPDATA%\\panel-ocr (§16.33 item 3)"
    );

    assert_ne!(
        cache_dir(Platform::Windows, &env),
        Path::new(roaming).join(APP_DIR_NAME),
        "the cache must NOT land under %APPDATA%: that is upstream's measured behaviour, and \
         §14 item 19 registers our divergence from it because %APPDATA% roams and a regenerable \
         cache must not"
    );
    assert_ne!(
        config_dir(Platform::Windows, &env),
        Path::new(local).join(APP_DIR_NAME),
        "the config must NOT land under %LOCALAPPDATA%: config is the half that is supposed to roam"
    );
}

#[test]
// spec §16.33 item 5 — the Linux rows of both tables are today's behaviour and must stay so. Windows
// support is not licence to touch them; this is the regression control for that.
fn the_linux_default_roots_are_unchanged_by_windows_support() {
    let env = [("HOME", "/home/u")];
    assert_eq!(
        cache_dir(Platform::Linux, &env),
        Path::new("/home/u/.cache").join(APP_DIR_NAME)
    );
    assert_eq!(
        config_dir(Platform::Linux, &env),
        Path::new("/home/u/.config").join(APP_DIR_NAME)
    );
}

#[test]
// spec §16.33 item 5 — the macOS rows, likewise unchanged. Note `Application Support` with its
// space: it is the one default root that a naive shell-quoting bug would visibly break, which is why
// the literal is spelled out rather than composed.
fn the_macos_default_roots_are_unchanged_by_windows_support() {
    let env = [("HOME", "/Users/u")];
    assert_eq!(
        cache_dir(Platform::MacOs, &env),
        Path::new("/Users/u/Library/Caches").join(APP_DIR_NAME)
    );
    assert_eq!(
        config_dir(Platform::MacOs, &env),
        Path::new("/Users/u/Library/Application Support").join(APP_DIR_NAME)
    );
}

#[test]
// spec §16.33 item 5 — `EnvSource::var` returns `None` for an empty value as well as an unset one.
// This is today's `env_path` behaviour and it must survive the refactor: an exported-but-empty
// `HOME` or `LOCALAPPDATA` is common in stripped CI environments and container entrypoints, and
// treating it as set would produce a root at the filesystem root.
fn an_empty_environment_variable_is_treated_as_unset() {
    for (platform, empty_key) in [
        (Platform::Linux, "HOME"),
        (Platform::MacOs, "HOME"),
        (Platform::Windows, "LOCALAPPDATA"),
    ] {
        let env = [("XDG_CACHE_HOME", ""), (empty_key, "")];
        assert_eq!(
            cache_dir(platform, &env),
            PathBuf::from(".panel-ocr-cache"),
            "{platform:?}: an empty {empty_key} (and an empty XDG_CACHE_HOME) must both count as \
             unset, leaving the last-resort relative fallback"
        );
    }
}

#[test]
// spec §16.33 item 5 — the last-resort relative fallback, on every platform, from a completely empty
// environment. Unchanged from §16.12 item 21's `./.panel-ocr-cache`.
fn the_last_resort_fallback_is_relative_on_every_platform() {
    for platform in EVERY_PLATFORM {
        assert_eq!(
            cache_dir(platform, &[]),
            PathBuf::from(".panel-ocr-cache"),
            "{platform:?}: empty environment => relative cache fallback"
        );
        assert_eq!(
            config_dir(platform, &[]),
            PathBuf::from(".panel-ocr"),
            "{platform:?}: empty environment => relative config fallback"
        );
    }
}

#[test]
// spec §16.33 item 5, the interface detail that entry flags as settled by transcription rather than
// by the ratification: on Windows the resolver consults neither `HOME` nor `USERPROFILE`.
//
// Both are commonly set on Windows — MSYS2, Cygwin and Git-for-Windows all set `HOME` to a
// POSIX-shaped path — so a resolver that fell through to either would put the cache somewhere that
// depends on which terminal launched the binary. The environment here sets both and no
// `%LOCALAPPDATA%`/`%APPDATA%`, so a resolver that consults either produces a HOME-shaped path and
// this test names which variable leaked.
fn the_windows_resolver_ignores_home_and_userprofile() {
    let env = [("HOME", "/home/u"), ("USERPROFILE", r"C:\Users\u")];
    assert_eq!(
        cache_dir(Platform::Windows, &env),
        PathBuf::from(".panel-ocr-cache"),
        "the Windows cache root must not be derived from $HOME or %USERPROFILE%"
    );
    assert_eq!(
        config_dir(Platform::Windows, &env),
        PathBuf::from(".panel-ocr"),
        "the Windows config root must not be derived from $HOME or %USERPROFILE%"
    );
}

#[test]
// spec §16.33 item 4 — the two XDG variables are independent. A resolver that gates both on one
// variable's presence would pass every test above.
fn the_two_xdg_variables_are_resolved_independently() {
    let env = [
        ("XDG_CACHE_HOME", "/xdg/cache"),
        ("LOCALAPPDATA", r"C:\Users\u\AppData\Local"),
        ("APPDATA", r"C:\Users\u\AppData\Roaming"),
    ];
    assert_eq!(
        cache_dir(Platform::Windows, &env),
        Path::new("/xdg/cache").join(APP_DIR_NAME),
        "XDG_CACHE_HOME is set, so it wins for the cache root"
    );
    assert_eq!(
        config_dir(Platform::Windows, &env),
        Path::new(r"C:\Users\u\AppData\Roaming").join(APP_DIR_NAME),
        "XDG_CONFIG_HOME is UNSET, so the config root must fall through to %APPDATA% — the cache \
         variable must not stand in for it"
    );
}

#[test]
// spec §16.33 item 5 — `config_path` is `config_dir().join(CONFIG_FILE_NAME)` on every platform.
// Kept separate from the `config_dir` tests because the frozen assertion in item 13 is about
// `default_config_path`, not `default_config_dir`.
fn the_config_path_is_the_config_file_inside_the_config_directory() {
    for platform in EVERY_PLATFORM {
        let env = every_root_set();
        assert_eq!(
            config_path(platform, &env),
            config_dir(platform, &env).join(CONFIG_FILE_NAME),
            "{platform:?}: config_path must be config_dir + {CONFIG_FILE_NAME}"
        );
        assert_eq!(
            config_path(platform, &env),
            Path::new("/xdg/config")
                .join(APP_DIR_NAME)
                .join("config.toml"),
            "{platform:?}: and the whole path, spelled out independently of the accessor above"
        );
    }
}

#[test]
// spec §16.33 item 13 — the DEVIATION-adjacent confirmation, asserted rather than left as prose.
//
// `crates/pc-cli/tests/x1_args.rs`'s frozen `the_cache_directory_is_overridable` asserts
// `default_cache_dir().ends_with("panel-ocr")` and `default_config_path().ends_with("config.toml")`.
// `Path::ends_with` compares whole trailing COMPONENTS, and every non-fallback row of both tables
// ends in `.join(APP_DIR_NAME)`, so the ratified `%LOCALAPPDATA%\panel-ocr` satisfies it exactly as
// `$HOME/.cache/panel-ocr` does. This checks the property for all three platforms and both
// non-fallback branches, so the frozen assertion cannot be broken by a Windows-only change without a
// second test going red and saying why.
fn every_platform_root_still_ends_with_the_app_directory_name() {
    let platform_only: [(Platform, Vec<(&str, &str)>); 3] = [
        (Platform::Linux, vec![("HOME", "/home/u")]),
        (Platform::MacOs, vec![("HOME", "/Users/u")]),
        (
            Platform::Windows,
            vec![
                ("LOCALAPPDATA", r"C:\Users\u\AppData\Local"),
                ("APPDATA", r"C:\Users\u\AppData\Roaming"),
            ],
        ),
    ];
    for (platform, env) in &platform_only {
        assert!(
            cache_dir(*platform, env).ends_with(APP_DIR_NAME),
            "{platform:?}: the platform-default cache root must keep {APP_DIR_NAME} as its final \
             component, or x1_args.rs's frozen the_cache_directory_is_overridable breaks"
        );
        assert!(
            config_path(*platform, env).ends_with(CONFIG_FILE_NAME),
            "{platform:?}: the platform-default config path must keep {CONFIG_FILE_NAME} as its \
             final component"
        );
    }
    // And under XDG, which is the other non-fallback branch.
    for platform in EVERY_PLATFORM {
        let env = [
            ("XDG_CACHE_HOME", "/xdg/cache"),
            ("XDG_CONFIG_HOME", "/xdg/config"),
        ];
        assert!(cache_dir(platform, &env).ends_with(APP_DIR_NAME));
        assert!(config_path(platform, &env).ends_with(CONFIG_FILE_NAME));
    }
}

#[test]
// spec §16.33 item 13, the other half — the last-resort fallback does NOT end with the app directory
// name, and that is a PRE-EXISTING conditionality of the frozen assertion, identical on all three
// platforms and not introduced by E1.
//
// Asserted rather than described, because item 13's claim is precisely that E1 changed the platform
// count and not the shape. If a future change made the fallback end in `panel-ocr`, this test goes
// red and the reader learns that item 13's reasoning no longer describes the code — which is more
// useful than a silently stronger guarantee.
fn the_last_resort_fallback_does_not_end_with_the_app_directory_name() {
    for platform in EVERY_PLATFORM {
        assert!(
            !cache_dir(platform, &[]).ends_with(APP_DIR_NAME),
            "{platform:?}: the relative fallback is `.panel-ocr-cache`, whose final component is \
             not {APP_DIR_NAME} — the pre-existing gap §16.33 item 13 records"
        );
    }
}

#[test]
// spec §16.33 item 5 — `Platform::HOST` must name the compilation target. The expectation is a
// `cfg!` cascade written HERE, in the test, rather than read from the source: two independent
// statements of the same rule, which is the only way this assertion is worth anything.
fn the_host_platform_constant_matches_the_compilation_target() {
    let expected = if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "macos") {
        Platform::MacOs
    } else {
        Platform::Linux
    };
    assert_eq!(Platform::HOST, expected);
}

#[test]
// spec §16.33 item 5 — the three existing free functions must delegate to the host platform's
// resolver rather than keeping a second copy of the rules. This is a consistency check, not an
// independent oracle, and its value is narrow and specific: it fails if a wrapper hard-codes a
// platform (e.g. `Platform::Linux`) or keeps the old `cfg!` body while `DirEnv` gains the ratified
// one. `paths::resolve_cache_root`'s own precedence is §16.12 item 21's and is untouched here.
fn the_host_bound_free_functions_delegate_to_the_host_platform_resolver() {
    let host = DirEnv::new(Platform::HOST, &ProcessEnv);
    assert_eq!(paths::default_cache_dir(), host.cache_dir());
    assert_eq!(paths::default_config_dir(), host.config_dir());
    assert_eq!(paths::default_config_path(), host.config_path());
}
