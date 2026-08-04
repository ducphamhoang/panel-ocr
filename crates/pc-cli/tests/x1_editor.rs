//! Task **X1/W3** — editor resolution for `profile edit` (spec §16.33 item 7, E5).
//!
//! FROZEN once committed (CLAUDE.md). Does not compile until `pc_cli` gains `resolve_editor`.
//!
//! **Why this is a separate pure function rather than an assertion about `profile edit`.** The
//! command's behaviour is "spawn a process and wait", which a test cannot exercise without either
//! spawning a real editor or mocking the spawn. What §16.33 item 7 ratifies is not the spawn — it is
//! *which program name* gets spawned, on which platform, from which environment. Factoring that
//! decision out is what makes all six cells of item 7's table reachable from any host, the same
//! reasoning as the resolver in `x1_platform_paths.rs`.
//!
//! **What these tests do NOT cover, stated so a green run is not over-read.** They say nothing about
//! the spawn, the exit-status handling, or the rendered error text — `resolve_editor` returning `None`
//! is asserted here, but that `None` still produces the exact message `"$EDITOR is not set"` is a
//! property of `run_profile`'s error path and is not gated by this file.

use pc_cli::paths::{MapEnv, Platform};
use pc_cli::resolve_editor;
use std::ffi::OsString;

/// All three supported platforms, so a table-driven test cannot quietly skip one.
const EVERY_PLATFORM: [Platform; 3] = [Platform::Linux, Platform::MacOs, Platform::Windows];

fn resolved(platform: Platform, env: &[(&str, &str)]) -> Option<OsString> {
    let env = MapEnv::from_pairs(env);
    resolve_editor(platform, &env)
}

#[test]
// spec §16.33 item 7 — the three "`$EDITOR` set" cells. `$EDITOR` wins on every platform, so the
// Windows fallback must never shadow an explicit choice. The value is deliberately not a real editor
// name: nothing here spawns anything, and a distinctive value makes a fallback leak obvious.
fn an_explicit_editor_variable_wins_on_every_platform() {
    for platform in EVERY_PLATFORM {
        assert_eq!(
            resolved(platform, &[("EDITOR", "my-chosen-editor")]),
            Some(OsString::from("my-chosen-editor")),
            "{platform:?}: $EDITOR must win over any platform fallback"
        );
    }
}

#[test]
// spec §16.33 item 7 — the Windows "`$EDITOR` unset" cell, which is the whole of E5. Asserted with
// the literal program name, because that name IS the ratified decision.
fn an_unset_editor_variable_falls_back_to_notepad_on_windows() {
    assert_eq!(
        resolved(Platform::Windows, &[]),
        Some(OsString::from("notepad.exe")),
        "§16.33 item 7 (E5): Windows falls back to notepad.exe when $EDITOR is unset"
    );
}

#[test]
// spec §16.33 item 7 — the Linux and macOS "`$EDITOR` unset" cells. `None` keeps today's exact
// behaviour: `profile edit` fails with `"$EDITOR is not set"` rather than guessing. E5 is scoped to
// Windows, and this is the assertion that stops it from widening into a `nano`/`vi` fallback nobody
// asked for.
fn an_unset_editor_variable_has_no_fallback_on_linux_or_macos() {
    for platform in [Platform::Linux, Platform::MacOs] {
        assert_eq!(
            resolved(platform, &[]),
            None,
            "{platform:?}: E5 is scoped to Windows; the POSIX platforms must keep failing loudly \
             rather than gaining an unratified default editor"
        );
    }
}

#[test]
// spec §16.33 item 7 — an empty `$EDITOR` counts as unset, matching the `EnvSource::var` contract in
// item 5. An exported-but-empty `EDITOR` is common in stripped environments, and treating it as set
// would try to spawn the empty string.
fn an_empty_editor_variable_is_treated_as_unset() {
    assert_eq!(
        resolved(Platform::Windows, &[("EDITOR", "")]),
        Some(OsString::from("notepad.exe")),
        "an empty $EDITOR must fall through to the Windows fallback"
    );
    for platform in [Platform::Linux, Platform::MacOs] {
        assert_eq!(
            resolved(platform, &[("EDITOR", "")]),
            None,
            "{platform:?}: an empty $EDITOR must be treated as unset"
        );
    }
}

#[test]
// spec §16.33 item 7 — `$VISUAL` is deliberately NOT consulted, on any platform. Today's
// `profile edit` reads `EDITOR` only; adding `VISUAL` would be a behaviour change on Linux and macOS
// that no requirement asked for, and "the convention is VISUAL first" is exactly the kind of
// plausible unrequested addition a review is supposed to catch. Pinned so it cannot arrive quietly.
fn the_visual_variable_is_not_consulted_on_any_platform() {
    for platform in EVERY_PLATFORM {
        let with_visual_only = resolved(platform, &[("VISUAL", "visual-editor")]);
        assert_ne!(
            with_visual_only,
            Some(OsString::from("visual-editor")),
            "{platform:?}: $VISUAL must not be consulted — §16.33 item 7 reads $EDITOR only"
        );
        assert_eq!(
            with_visual_only,
            resolved(platform, &[]),
            "{platform:?}: setting $VISUAL must change nothing at all"
        );
        // And it must not shadow an explicit $EDITOR either, in either direction.
        assert_eq!(
            resolved(
                platform,
                &[("VISUAL", "visual-editor"), ("EDITOR", "editor-wins")]
            ),
            Some(OsString::from("editor-wins")),
            "{platform:?}: $EDITOR is the only variable consulted"
        );
    }
}
