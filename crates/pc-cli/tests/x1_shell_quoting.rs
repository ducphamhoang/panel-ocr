//! Task **X1/W2** — shell-flavor selection and quoting for pasted recovery commands (spec §16.33
//! item 6; §16.12 item 21 for the POSIX rule it leaves standing).
//!
//! FROZEN once committed (CLAUDE.md). Does not compile until `pc_cli::paths` gains `Shell`.
//!
//! **The asymmetry between the two flavors is deliberate and is a limitation, not a design.** §16.12
//! item 21's justification for emitting a recovery command at all is that it can be pasted verbatim,
//! and the existing gate on that (`x1_args.rs`'s
//! `the_models_download_suggestion_is_paste_safe_for_hostile_cache_paths`) hands the emitted segment
//! to a real `/bin/sh` and asserts the shell recovers the original bytes — a genuine oracle, and one
//! that deliberately avoids asserting an exact quoted string, because a different-but-valid quoting
//! style would be a false failure. That test is gated `#[cfg(all(feature = "onnx", unix))]`
//! (`x1_args.rs:369`) and stays so — it is unix-only *and* only on the `onnx` feature tier, which is
//! precisely why `posix_quoting_round_trips_through_a_real_posix_shell` below exists to cover the
//! quoter on the default tier.
//!
//! No such oracle is available for PowerShell on the Linux and macOS runners, which have no
//! PowerShell. So PowerShell quoting is checked two ways, and each test's name says which:
//! `powershell_quoting_matches_the_single_quote_doubling_rule` asserts conformance to the *rule as
//! written in §16.33 item 6* against a hard-coded table — it does NOT establish paste safety in
//! general — and `powershell_quoting_round_trips_through_a_real_powershell` is the real oracle and
//! runs on a Windows host only. Reading the first as if it were the second is exactly cookbook rule
//! 1's defect class, which is why they are two tests with two names.

use pc_cli::paths::{Platform, Shell};
use std::path::Path;

/// Payloads that are shell-dangerous AND legal in a path on the platform under test. Every one is
/// harmless if executed, deliberately: an unquoted `$( )` or backtick would be run by the shell in
/// the round-trip tests below, so nothing here may have a side effect.
///
/// `#[cfg(unix)]` because its only consumer is the POSIX-only round trip below, and an unused
/// constant is a `-D warnings` clippy failure on Windows — the same reasoning already applied to
/// `HOSTILE_WINDOWS_SEGMENTS` below.
#[cfg(unix)]
const HOSTILE_POSIX_SEGMENTS: &[&str] = &[
    "space dir",            // word splitting
    "it's-a-cache",         // the case naive quoting gets wrong
    "dollar$(echo pwned)",  // command substitution
    "back`echo pwned`tick", // the older substitution syntax
    "semi;colon",           // would terminate the command
    "amp&ersand",           // would background it
    "pipe|and\"quote",      // pipeline plus a double quote
    "tab\tseparated",       // IFS splitting on whitespace that is not a space
];

/// The same idea restricted to characters Windows actually permits in a path: `|`, `"`, `<`, `>`,
/// `?`, `*` and `:` are reserved there, so including them would test a string no cache root can be.
///
/// `#[cfg(windows)]` because its only consumer is the Windows-only round trip below, and an unused
/// constant is a `-D warnings` clippy failure on the other two platforms.
#[cfg(windows)]
const HOSTILE_WINDOWS_SEGMENTS: &[&str] = &[
    "space dir",
    "it's-a-cache",
    "dollar$(echo pwned)",
    "back`echo pwned`tick",
    "semi;colon",
    "amp&ersand",
    "percent%VAR%sign",
    "brace{}bracket[]caret^",
];

#[test]
// spec §16.33 item 6 — the mapping, exhaustively. Three separate assertions rather than a loop so a
// failure names the platform.
fn shell_for_target_maps_each_platform_to_its_flavor() {
    assert_eq!(Shell::for_target(Platform::Linux), Shell::Posix);
    assert_eq!(Shell::for_target(Platform::MacOs), Shell::Posix);
    assert_eq!(Shell::for_target(Platform::Windows), Shell::PowerShell);
}

#[test]
// spec §16.33 item 6 — `Shell::HOST` must agree with the compilation target. The expectation is a
// `cfg!` cascade written here rather than read from the source, so this is two independent statements
// of one rule and not a tautology. It also pins that `HOST` is derived from `Platform::HOST` rather
// than hard-coded to `Posix`, which would be invisible on a Linux runner.
fn the_host_shell_flavor_matches_the_compilation_target() {
    let expected = if cfg!(target_os = "windows") {
        Shell::PowerShell
    } else {
        Shell::Posix
    };
    assert_eq!(Shell::HOST, expected);
    assert_eq!(Shell::HOST, Shell::for_target(Platform::HOST));
}

#[cfg(unix)]
#[test]
// spec §16.12 item 21's POSIX rule, which §16.33 item 6 quotes as unchanged: *"wrap the path in
// single quotes and represent each embedded single quote as `'\''`"*.
//
// **A ROUND-TRIP assertion, not a string-equality one**, for the reason `x1_args.rs` records:
// asserting an exact quoted string would reject a different-but-valid style, and re-implementing
// POSIX quoting in the test would only prove the test agrees with itself. `/bin/sh` is the authority.
//
// Distinct from the existing `x1_args.rs` gate: that one goes through `models_download_command` and
// needs the `onnx` feature, while this exercises `Shell::Posix.quote` directly — so the quoter stays
// covered after the refactor moves it out of `models.rs`, on the default feature tier.
fn posix_quoting_round_trips_through_a_real_posix_shell() {
    for segment in HOSTILE_POSIX_SEGMENTS {
        let path = Path::new("/tmp/pc-cache").join(segment);
        let expected = path.display().to_string();
        let quoted = Shell::Posix.quote(&path);

        let script = format!("printf '%s\\n' {quoted}");
        let output = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(&script)
            .output()
            .expect("run /bin/sh");

        assert!(
            output.status.success(),
            "a POSIX shell cannot even parse the quoting for {segment:?}\n  quoted: {quoted}\n  \
             stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!("{expected}\n"),
            "the shell did not recover the original path for {segment:?} — word splitting, \
             globbing, parameter/command expansion, or a mis-escaped quote\n  quoted: {quoted}"
        );
    }
}

#[test]
// spec §16.33 item 6 — conformance to the PowerShell rule AS WRITTEN there: *"wraps in single quotes
// and doubles each embedded single quote (`''`)"*.
//
// **What this test establishes and what it does not.** It establishes that `Shell::PowerShell.quote`
// implements the recorded rule, on a table of expectations typed out here. It does NOT establish
// paste safety — only a real PowerShell can, and that oracle is the Windows-only test below. The
// exact-string form is accepted here, against `x1_args.rs`'s general preference for a round trip,
// because on a Linux or macOS runner there is no PowerShell to ask and the alternative is no
// assertion at all. Every expected string is written out; none is produced by the quoter.
fn powershell_quoting_matches_the_single_quote_doubling_rule() {
    let table: &[(&str, &str)] = &[
        (r"C:\cache", r"'C:\cache'"),
        (r"C:\space dir", r"'C:\space dir'"),
        // The one character that needs escaping, and the whole reason the rule is worth pinning.
        (r"C:\it's-a-cache", r"'C:\it''s-a-cache'"),
        // Two quotes, non-adjacent, to catch a quoter that escapes only the first occurrence.
        (r"C:\it's-a-user's-cache", r"'C:\it''s-a-user''s-cache'"),
        // Adjacent quotes, to catch a doubling that loses one.
        (r"C:\odd''name", r"'C:\odd''''name'"),
        // Everything below is literal inside PowerShell single quotes and must NOT be escaped.
        (r"C:\dollar$(echo pwned)", r"'C:\dollar$(echo pwned)'"),
        (r"C:\back`echo pwned`tick", r"'C:\back`echo pwned`tick'"),
        (r"C:\percent%VAR%sign", r"'C:\percent%VAR%sign'"),
        (r"C:\semi;colon&amp", r"'C:\semi;colon&amp'"),
    ];
    for (input, expected) in table {
        assert_eq!(
            Shell::PowerShell.quote(Path::new(input)),
            *expected,
            "PowerShell quoting of {input:?} must wrap in single quotes and double each embedded \
             single quote, escaping nothing else"
        );
    }
}

#[cfg(windows)]
#[test]
// spec §16.33 item 6 — the real oracle, on the only host that has one. Mirrors
// `x1_args.rs`'s POSIX round trip: hand the quoted segment to a real PowerShell and assert it
// recovers the original bytes. One assertion pair catches every failure mode at once — a broken quote
// makes PowerShell exit non-zero, and expansion or splitting changes the recovered text.
//
// `-NoProfile` matters: a user profile could define an alias or a function that changes how
// `Write-Output` behaves, which would make this test's result depend on the runner's user state.
fn powershell_quoting_round_trips_through_a_real_powershell() {
    for segment in HOSTILE_WINDOWS_SEGMENTS {
        let path = Path::new(r"C:\pc-cache").join(segment);
        let expected = path.display().to_string();
        let quoted = Shell::PowerShell.quote(&path);

        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command"])
            .arg(format!("Write-Output {quoted}"))
            .output()
            .expect("run powershell");

        assert!(
            output.status.success(),
            "PowerShell cannot even parse the quoting for {segment:?}\n  quoted: {quoted}\n  \
             stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim_end_matches(['\r', '\n']),
            expected,
            "PowerShell did not recover the original path for {segment:?}\n  quoted: {quoted}"
        );
    }
}

#[cfg(all(feature = "onnx", windows))]
#[test]
// spec §16.33 item 6 — the one call site. `models_download_command`'s suggestion must be quoted with
// the HOST flavor, so a Windows user gets a PowerShell-quoted line rather than a POSIX-quoted one.
// The expected command is written out in full here rather than composed from `Shell`, so a quoter
// that is wrong in the same way at both ends cannot make this pass.
//
// The path is deliberately not the default cache root, because
// `the_models_download_suggestion_omits_the_flag_for_the_default_cache` pins that the flag is absent
// in that case.
fn the_models_download_suggestion_is_powershell_quoted_on_windows() {
    let path = Path::new(r"C:\somewhere\odd cache");
    assert_eq!(
        pc_cli::models::models_download_command(Some(path)),
        r"panel-ocr models download --cache-dir 'C:\somewhere\odd cache'"
    );
}
