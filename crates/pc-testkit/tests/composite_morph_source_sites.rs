//! The **structural** re-fork tripwire for the six pixel primitives §16.42 and §16.38
//! item 16(a) hoisted into `pc-imageops`: `kernel`, `dilate`, `blend_channel`,
//! `resize_nearest_rgba`, `alpha_composite_over`, `composite_rgb`.
//!
//! **What this gate does and does not catch, stated first because the distinction is the
//! whole point.** It catches a *structural* re-fork -- a crate growing its own
//! `fn alpha_composite_over` instead of re-exporting `pc_imageops`', or quietly dropping
//! a `pub use` and pointing the module path somewhere else. It does **not** catch a
//! *behavioural* change inside `pc-imageops`' own definitions: that is what the
//! hand-derived per-path value locks are for
//! (`crates/pc-export/tests/composite_value_lock.rs`,
//! `crates/pc-mask/tests/m5_alpha_composite_value_lock.rs`,
//! `crates/pc-pipeline/tests/l1_morph_equivalence.rs`'s two kernel oracles). Reading this
//! file as "the composite/morph tests" would be exactly the over-claim cookbook rule 1
//! is about.
//!
//! **Why a file-scanning gate rather than a compile-time or pointer check.** Both cheaper
//! candidates were built and measured during the 2026-08-17 joint planning pass and both
//! failed:
//!   * `const _: fn(u32) -> Kernel = pc_mask::grow::kernel;` was compiled against a live,
//!     genuinely re-forked `pc_mask::grow::kernel` and compiled **clean** while six of
//!     seven `l1_morph_equivalence.rs` tests went red. It proves signature compatibility,
//!     not function-item identity -- and every re-fork preserves the signature, since
//!     that is what a re-fork is.
//!   * Runtime function-pointer equality (`a as usize == b as usize`) is unsound as a
//!     gate: Rust does not guarantee address-uniqueness for function items, so
//!     identical-code folding can merge two genuinely distinct functions to one address
//!     and let a real re-fork pass.
//!
//! Precedent for the shape: `crates/pc-testkit/tests/spec_supersession.rs` and
//! `a4d_claim_sites.rs` already parse repository files at runtime and pin the parsed set
//! against a hard-coded literal, including a hard-coded row count so silently emptying
//! the scan fails loudly instead of passing vacuously.

use pc_testkit::paths;
use std::path::{Path, PathBuf};

/// The six function names §16.42 item 8 and §16.38 item 16(a) consolidated.
const WATCHED: [&str; 6] = [
    "alpha_composite_over",
    "blend_channel",
    "composite_rgb",
    "dilate",
    "kernel",
    "resize_nearest_rgba",
];

/// Hard-coded expected row count for the definition-site scan (`spec_supersession.rs`'s
/// `EXPECTED_PARSED_CLAIMS` pattern). Never derived from the tree: a scan that matched
/// nothing would otherwise satisfy an empty-set comparison silently.
///
/// 7 = the 6 real definitions in `pc-imageops` + the 1 pinned exemption below.
const EXPECTED_DEFINITION_SITES: usize = 7;

/// Hard-coded expected row count for the re-export scan, same reasoning.
///
/// 7 -- note this is one MORE than the six `pub use` lines the 2026-08-17 audit and the
/// architect's first draft both enumerated. The seventh is
/// `crates/pc-denoise/src/lib.rs`'s whole-module `pub use pc_imageops::morph;`, which the
/// six-row list missed because it names no function. Re-derived here by scanning rather
/// than copied from either document.
const EXPECTED_REEXPORT_SITES: usize = 7;

/// A floor on the number of `crates/*/src/**/*.rs` files the walk must visit. Hard-coded,
/// deliberately well below the observed count (107 at the time of writing) so ordinary
/// growth does not churn it, but far enough above zero that a broken walk -- a wrong
/// anchor path, a directory rename, a filter typo -- cannot report "no definitions found,
/// therefore nothing changed."
const MINIMUM_SCANNED_FILES: usize = 80;

/// Every `fn <watched name>` definition site the scan is allowed to find, as
/// `(repo-relative slash path, function name)`, sorted.
///
/// Six of these are the real, single implementations. The seventh is an **exemption**:
///
/// `crates/pc-testkit/src/metrics.rs`'s `PixelSet::dilate` is an unrelated inherent
/// method on this crate's own test-metric type. It happens to share a name with
/// `pc_imageops::morph::dilate` and has nothing to do with it -- different receiver,
/// different type, different purpose. It is pinned as an expected row rather than
/// filtered out of the scan, so that this gate stays red if `pc-testkit` ever grows a
/// *second* `dilate`, or if this one moves. A path-based filter would have made both of
/// those invisible.
const EXPECTED_DEFINITIONS: &[(&str, &str)] = &[
    (
        "crates/pc-imageops/src/composite.rs",
        "alpha_composite_over",
    ),
    ("crates/pc-imageops/src/composite.rs", "blend_channel"),
    ("crates/pc-imageops/src/composite.rs", "composite_rgb"),
    ("crates/pc-imageops/src/composite.rs", "resize_nearest_rgba"),
    ("crates/pc-imageops/src/morph.rs", "dilate"),
    ("crates/pc-imageops/src/morph.rs", "kernel"),
    // EXEMPTION, see the doc comment above: unrelated `PixelSet::dilate`, not a re-fork.
    ("crates/pc-testkit/src/metrics.rs", "dilate"),
];

/// The placeholder recorded for a whole-module re-export (`pub use pc_imageops::morph;`),
/// which imports no individual names.
const WHOLE_MODULE: &str = "<whole module>";

/// Every `pub use pc_imageops::{morph,composite}...` site, as
/// `(repo-relative slash path, module, comma-joined sorted imported names)`, sorted.
///
/// Two rows share `crates/pc-denoise/src/lib.rs` -- the whole-module re-export at line 54
/// and the named one at line 67 -- which is why this is a sorted multiset keyed on the
/// whole row and not a map keyed on the file.
///
/// **Scope, stated so it is not read wider than it is.** These are the *direct*
/// `pc_imageops::morph` / `pc_imageops::composite` re-exports only. A crate's own
/// second-hop re-export of its own module (`pc-denoise/src/lib.rs`'s
/// `pub use composite::{...}`, `pc-export/src/lib.rs`'s, `pc-imageops/src/lib.rs`'s own)
/// is not listed: it inherits whatever the first hop resolved to, so it cannot introduce
/// an independent implementation without also tripping the definition-site scan above.
/// `pub use pc_imageops::gaussian;` is likewise absent -- `gaussian` is not one of the six
/// functions this gate watches.
const EXPECTED_REEXPORTS: &[(&str, &str, &str)] = &[
    (
        "crates/pc-denoise/src/composite.rs",
        "composite",
        "alpha_composite_over,blend_channel,composite_rgb,resize_nearest_rgba",
    ),
    ("crates/pc-denoise/src/lib.rs", "morph", WHOLE_MODULE),
    (
        "crates/pc-denoise/src/lib.rs",
        "morph",
        "Kernel,dilate,kernel",
    ),
    (
        "crates/pc-export/src/composite.rs",
        "composite",
        "alpha_composite_over,blend_channel,resize_nearest_rgba",
    ),
    (
        "crates/pc-inpaint/src/compose.rs",
        "composite",
        "alpha_composite_over,blend_channel,composite_rgb,resize_nearest_rgba",
    ),
    (
        "crates/pc-mask/src/combine.rs",
        "composite",
        "alpha_composite_over,blend_channel,composite_rgb,resize_nearest_rgba",
    ),
    (
        "crates/pc-mask/src/grow.rs",
        "morph",
        "Kernel,dilate,kernel",
    ),
];

/// Every `crates/*/src/**/*.rs`, repo-relative with forward slashes, sorted.
fn source_files() -> Vec<(String, PathBuf)> {
    let root = paths::workspace_root();
    let mut files = Vec::new();
    let crates_dir = root.join("crates");
    let mut crate_dirs: Vec<PathBuf> = std::fs::read_dir(&crates_dir)
        .expect("crates/ must be readable")
        .map(|entry| entry.expect("crates/ entry must be readable").path())
        .filter(|path| path.is_dir())
        .collect();
    crate_dirs.sort();
    for crate_dir in crate_dirs {
        collect_rs(&crate_dir.join("src"), &root, &mut files);
    }
    files.sort();
    files
}

fn collect_rs(dir: &Path, root: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return; // a crate without a src/ directory is not an error here
    };
    for entry in entries {
        let path = entry.expect("directory entry must be readable").path();
        if path.is_dir() {
            collect_rs(&path, root, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let relative = path
                .strip_prefix(root)
                .expect("scanned file must live under the workspace root");
            out.push((paths::slash_separated(relative), path.clone()));
        }
    }
}

/// The name in `fn <name>(` on this line, if the line declares a function at all.
///
/// Deliberately syntactic and narrow: it looks for the token `fn` followed by an
/// identifier and an opening parenthesis, so a doc comment mentioning `fn dilate(` in
/// prose is the one shape it would over-report. There is no such comment in the tree
/// today, and over-reporting fails the gate loudly rather than silently -- the failure
/// mode this parser must not have is the other one.
fn declared_fn_name(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") {
        return None;
    }
    let rest = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
    let rest = rest.split_once("fn ").map(|(before, after)| {
        // Only accept a `fn` that is at the start of the (visibility-stripped) item, or
        // preceded solely by the modifiers Rust allows there.
        let before = before.trim();
        let modifiers_only = before.is_empty()
            || before
                .split_whitespace()
                .all(|word| matches!(word, "const" | "async" | "unsafe" | "extern" | "default"));
        modifiers_only.then_some(after)
    })??;
    let name_end = rest.find(['(', '<', ' '])?;
    let name = &rest[..name_end];
    (!name.is_empty()).then_some(name)
}

/// `pub use pc_imageops::<module>...;` -- returns `(module, comma-joined sorted names)`.
///
/// Joins a multi-line brace list into one statement before parsing, because rustfmt wraps
/// the four-name imports across three lines and a line-at-a-time scan would miss the
/// names entirely.
fn parse_reexports(text: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for module in ["morph", "composite"] {
        let needle = format!("pub use pc_imageops::{module}");
        let mut cursor = 0usize;
        while let Some(offset) = text[cursor..].find(&needle) {
            let start = cursor + offset;
            // Reject `pub use pc_imageops::morphology` and friends: the character after
            // the module name must end the path segment.
            let after = &text[start + needle.len()..];
            let terminator = after.chars().next();
            if !matches!(terminator, Some(':') | Some(';') | Some(' ') | Some('\n')) {
                cursor = start + needle.len();
                continue;
            }
            let end = after
                .find(';')
                .expect("a `pub use` statement must be terminated by `;`");
            let statement = &after[..end];
            let names = match statement.find('{') {
                // No brace list: either the whole module (`pub use pc_imageops::morph;`)
                // or exactly one name (`pub use pc_imageops::composite::blend_channel;`).
                // Recording the single name rather than collapsing it to the whole-module
                // placeholder keeps the row honest about what it actually re-exports.
                None => {
                    let single = statement.trim().trim_start_matches("::").trim();
                    if single.is_empty() {
                        WHOLE_MODULE.to_string()
                    } else {
                        single.to_string()
                    }
                }
                Some(brace) => {
                    let close = statement
                        .find('}')
                        .expect("a braced `use` list must be closed");
                    let mut names: Vec<&str> = statement[brace + 1..close]
                        .split(',')
                        .map(str::trim)
                        .filter(|name| !name.is_empty())
                        .collect();
                    names.sort_unstable();
                    names.join(",")
                }
            };
            found.push((module.to_string(), names));
            cursor = start + needle.len() + end;
        }
    }
    found
}

/// Whichever crate grows a second implementation of one of the six hoisted primitives,
/// this fails and names the file. Turning it red: add
/// `pub fn blend_channel(base: u8, color: u8, alpha: f64) -> u8 { .. }` to any crate's
/// `src/`, or move one of `pc-imageops`' six definitions to another file.
#[test]
fn the_six_hoisted_primitives_are_defined_only_where_this_gate_pins_them() {
    let files = source_files();
    assert!(
        files.len() >= MINIMUM_SCANNED_FILES,
        "the source walk found only {} files under crates/*/src; expected at least \
         {MINIMUM_SCANNED_FILES}. The walk is broken, not the tree.",
        files.len()
    );

    let mut found: Vec<(String, String)> = Vec::new();
    for (relative, path) in &files {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"));
        for line in text.lines() {
            if let Some(name) = declared_fn_name(line) {
                if WATCHED.contains(&name) {
                    found.push((relative.clone(), name.to_string()));
                }
            }
        }
    }
    found.sort();

    let expected: Vec<(String, String)> = EXPECTED_DEFINITIONS
        .iter()
        .map(|(path, name)| ((*path).to_string(), (*name).to_string()))
        .collect();
    assert_eq!(
        expected.len(),
        EXPECTED_DEFINITION_SITES,
        "EXPECTED_DEFINITIONS and EXPECTED_DEFINITION_SITES must be raised together"
    );
    assert_eq!(
        found, expected,
        "the set of `fn` definition sites for {WATCHED:?} changed. A new row outside \
         `pc-imageops` means a primitive was re-forked; a missing row means one moved or \
         was deleted. Neither is a change this gate may be edited to accommodate without \
         the ratification that authorised it."
    );
    assert_eq!(
        found.len(),
        EXPECTED_DEFINITION_SITES,
        "definition-site count"
    );
}

/// The other half of the same structural claim: every stage crate's module path is still
/// a re-export of `pc_imageops`, with exactly the names it re-exported before. Turning it
/// red: delete `pc-mask/src/grow.rs`'s `pub use pc_imageops::morph::{dilate, kernel,
/// Kernel};`, or narrow any of these lists, or add a re-export in a new crate.
#[test]
fn every_pc_imageops_morph_and_composite_reexport_site_is_the_pinned_set() {
    let files = source_files();
    let mut found: Vec<(String, String, String)> = Vec::new();
    for (relative, path) in &files {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"));
        for (module, names) in parse_reexports(&text) {
            found.push((relative.clone(), module, names));
        }
    }
    found.sort();

    let expected: Vec<(String, String, String)> = EXPECTED_REEXPORTS
        .iter()
        .map(|(path, module, names)| {
            (
                (*path).to_string(),
                (*module).to_string(),
                (*names).to_string(),
            )
        })
        .collect();
    assert_eq!(
        expected.len(),
        EXPECTED_REEXPORT_SITES,
        "EXPECTED_REEXPORTS and EXPECTED_REEXPORT_SITES must be raised together"
    );
    assert_eq!(
        found, expected,
        "the set of direct `pub use pc_imageops::{{morph,composite}}` re-export sites \
         changed. A missing row means a stage crate stopped delegating -- the first half \
         of a re-fork, whose second half the definition-site gate would catch."
    );
    assert_eq!(found.len(), EXPECTED_REEXPORT_SITES, "re-export-site count");
}
