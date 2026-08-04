//! spec §16.33 item 2 — every site that asserts this project's supported-platform list must point
//! at the ratification that changed it.
//!
//! **Why this gate exists, and why it flattens whitespace before scanning.** §16.33 item 2's
//! enumeration was taken twice. A line-based `grep -rn "Linux + macOS"` over the tracked tree
//! returned 7 hits. The same scan over whitespace-flattened file text returned **8** — §16.12 item
//! 21's claim wraps across two physical lines (`... and v1 is Linux +` / `macOS only). ...`), so no
//! single line contains the phrase. The missed site was the most important one in the list: it is
//! the clause that fixes the cache and config roots, i.e. the clause E1 actually changes. So the
//! naive form of this check was measurably blind to the one site that mattered, and
//! `a_claim_that_wraps_across_lines_is_invisible_to_a_line_based_scan_and_visible_to_this_one`
//! pins that difference rather than trusting the author to remember it.
//!
//! **What this gate does NOT establish.** It does not check that any site's *replacement wording*
//! is correct, and it cannot: "says the right thing about Windows" is not mechanisable. It checks
//! two things only — that each enumerated site mentions `§16.33` at all, and that no site leaves a
//! two-platform phrase sitting further than `POINTER_WINDOW` flattened characters from such a
//! mention. Both are drift guards on a claim that already drifted across five files with no owner.

use pc_testkit::paths;
use std::path::PathBuf;

/// Every file this gate holds to the pointer requirement — six entries, and they are NOT six scan
/// results. **Five** of them are the files the flattened scan of 2026-08-04 found carrying the
/// two-platform claim (8 occurrences: `README.md` x2, `docs/PIPELINE_SPEC_V1.md` x3,
/// `crates/pc-cli/src/paths.rs` x1, `crates/pc-cli/tests/x1_args.rs` x1,
/// `.github/workflows/ci.yml` x1), transcribed into §16.33 item 2's table.
///
/// The sixth, `docs/ARCHITECTURE_DECISIONS.md`, had **zero** occurrences of the claim and so is not
/// a scan result at all. It is included on a separate, weaker ground, stated rather than blurred into
/// the measurement: it is the document two of the five scanned sites forward to as the platform
/// authority, and item 2 measured that it had no platform-support section to forward *to*. A gate on
/// the five citers with nothing on the cited authority would leave the one site that is supposed to
/// own the claim as the only one free to go silent.
///
/// This is a hand-written enumeration on purpose. Deriving it by scanning the tree for the phrase
/// would make the gate vacuous the moment the phrase is reworded — the list would empty itself and
/// the suite would stay green, which is cookbook rule 13's collapse (the expectation coming from
/// the artifact under test).
const PLATFORM_CLAIM_SITES: &[(&str, &str)] = &[
    (
        "docs/PIPELINE_SPEC_V1.md",
        "the `Fixed constraints` preamble (line 6), §16.12 item 21's default cache/config roots, \
         and §16's out-of-scope bullet",
    ),
    (
        "docs/ARCHITECTURE_DECISIONS.md",
        "the `Platform support` section — the authority the other two sites below forward to, and \
         which did not exist before §16.33",
    ),
    (
        "README.md",
        "the user-facing platform statement and the roadmap checkbox",
    ),
    (
        ".github/workflows/ci.yml",
        "the matrix comment, which cites docs/ARCHITECTURE_DECISIONS.md by name",
    ),
    (
        "crates/pc-cli/src/paths.rs",
        "the module doc justifying hand-rolled directory discovery over a `dirs` dependency",
    ),
    (
        "crates/pc-cli/tests/x1_args.rs",
        "the doc comment justifying the `unix` gate on the paste-safety test — whose attribute is \
         `#[cfg(all(feature = \"onnx\", unix))]`, not the bare `#[cfg(unix)]` its own prose says. \
         §16.33 item 2 authorises a COMMENT-ONLY edit here: no assertion, `cfg`, name or fixture \
         may change",
    ),
];

/// §16.24 item 1(f)'s literal-constant pattern: a hard-coded count that cannot be computed from the
/// tree, so this file cannot pass by enumerating nothing, and shrinking the list is an edit that
/// cannot be skipped.
const EXPECTED_SITE_COUNT: usize = 6;

/// The stale claim, in the exact form all eight measured occurrences use.
const TWO_PLATFORM_PHRASE: &str = "Linux + macOS";

/// The ratification every site must forward to.
const POINTER: &str = "§16.33";

/// How far, in flattened characters, a `§16.33` mention may sit from a two-platform phrase and
/// still count as qualifying it. 400 is roughly four wrapped prose lines: close enough that a
/// reader who lands on the phrase sees the pointer, far enough that the pointer need not be
/// crammed into the same sentence.
const POINTER_WINDOW: usize = 400;

fn site_path(relative: &str) -> PathBuf {
    paths::workspace_root().join(relative)
}

fn read_site(relative: &str) -> String {
    let path = site_path(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read `{}`: {error}", path.display()))
}

/// The heading that opens the ratification's own section.
const RATIFICATION_HEADING: &str = "## 16.33 ";

/// The text of a site MINUS the ratification's own section.
///
/// **Why the exemption is necessary and why it is not a loophole.** §16.33 quotes all three clauses
/// it replaces verbatim, because this project's transcription rule requires a claim's scope to be
/// quoted rather than paraphrased — three over-generalisations in the F1 sequence all happened while
/// paraphrasing. Those quotations are claim-shaped to a scanner and are not claims. Requiring
/// §16.33 to cite `§16.33` beside its own quotations would be self-reference, so the section that
/// IS the pointer is skipped instead. `the_ratification_section_exemption_is_load_bearing_and_narrow`
/// pins that this removes one section and not the file.
fn scannable(relative: &str) -> String {
    let text = read_site(relative);
    let Some(start) = text.find(RATIFICATION_HEADING) else {
        return text;
    };
    // The next `## ` heading at column 0 ends the section; end of file otherwise.
    let after = &text[start + RATIFICATION_HEADING.len()..];
    let end = after
        .match_indices("\n## ")
        .next()
        .map_or(text.len(), |(offset, _)| {
            start + RATIFICATION_HEADING.len() + offset
        });
    format!("{}{}", &text[..start], &text[end..])
}

/// Collapse every run of whitespace to a single space, so a claim that wraps across physical lines
/// reads as one contiguous phrase. This is the whole reason the gate is not a `grep`.
fn flatten(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Byte offsets of every two-platform phrase in `flat` that has NO `§16.33` mention within
/// `POINTER_WINDOW` characters either side.
fn unqualified_phrase_offsets(flat: &str) -> Vec<usize> {
    let mut offsets = Vec::new();
    for (offset, _) in flat.match_indices(TWO_PLATFORM_PHRASE) {
        let start = offset.saturating_sub(POINTER_WINDOW);
        let end = (offset + TWO_PLATFORM_PHRASE.len() + POINTER_WINDOW).min(flat.len());
        // Snap to char boundaries: the spec is full of multi-byte characters (`§`, em dashes).
        let start = (start..=offset)
            .find(|i| flat.is_char_boundary(*i))
            .unwrap();
        let end = (end..=flat.len())
            .find(|i| flat.is_char_boundary(*i))
            .unwrap_or(flat.len());
        if !flat[start..end].contains(POINTER) {
            offsets.push(offset);
        }
    }
    offsets
}

/// A 120-character window around `offset`, for a failure message that names the site rather than
/// only its file.
fn excerpt(flat: &str, offset: usize) -> String {
    let start = offset.saturating_sub(60);
    let end = (offset + TWO_PLATFORM_PHRASE.len() + 60).min(flat.len());
    let start = (start..=offset)
        .find(|i| flat.is_char_boundary(*i))
        .unwrap();
    let end = (end..=flat.len())
        .find(|i| flat.is_char_boundary(*i))
        .unwrap_or(flat.len());
    format!("...{}...", &flat[start..end])
}

#[test]
// Anti-vacuity. Every other test in this file iterates PLATFORM_CLAIM_SITES, so all of them pass
// trivially on an empty list. This one cannot: the number is written here and is not derivable from
// the tree.
fn the_enumerated_platform_claim_site_count_is_pinned() {
    assert_eq!(
        PLATFORM_CLAIM_SITES.len(),
        EXPECTED_SITE_COUNT,
        "this gate holds {EXPECTED_SITE_COUNT} files to the pointer requirement: the five files \
         §16.33 item 2 measured as carrying the claim, plus `docs/ARCHITECTURE_DECISIONS.md`, the \
         authority two of them cite (see PLATFORM_CLAIM_SITES for why the sixth is not a scan \
         result). Adding or removing one means changing EXPECTED_SITE_COUNT in the same commit; a \
         count derived from the list would let this file silently enumerate nothing."
    );
}

#[test]
// The allowlist must not rot: a site that has been renamed or deleted would otherwise leave a row
// covering nothing, which is the failure mode of every allowlist that outlives its subject.
fn every_enumerated_platform_claim_site_still_exists() {
    let mut missing = Vec::new();
    for (relative, governs) in PLATFORM_CLAIM_SITES {
        if !site_path(relative).is_file() {
            missing.push(format!("{relative} (governs: {governs})"));
        }
    }
    assert!(
        missing.is_empty(),
        "enumerated platform-claim site(s) no longer exist; move the row rather than leaving it \
         covering nothing: {missing:?}"
    );
}

#[test]
// spec §16.33 item 2 — the primary requirement. Cookbook rule 14 applied to a documentation claim:
// when a ratified decision changes a shared claim, every reader of that claim must be reachable
// from the decision AND must reach the decision. This asserts the second direction, which is the
// one a future reader needs: they land on the site, not on the ratification.
//
// What must break for this to fail: delete `§16.33` from any of the six files. Today it fails for
// four of them, because the source-side and README-side updates are the next pipeline step.
fn every_enumerated_platform_claim_site_points_at_the_ratification() {
    let mut silent = Vec::new();
    for (relative, governs) in PLATFORM_CLAIM_SITES {
        if !read_site(relative).contains(POINTER) {
            silent.push(format!("{relative} — governs {governs}"));
        }
    }
    assert!(
        silent.is_empty(),
        "{} platform-claim site(s) never mention {POINTER}, so a reader who lands there learns \
         nothing about the v1.1 platform change:\n  {}",
        silent.len(),
        silent.join("\n  ")
    );
}

#[test]
// spec §16.33 item 2 — the stale phrase itself, wherever it survives, must sit beside the pointer.
// Distinct from the test above: a file may cite §16.33 in one place and still leave an unqualified
// "Linux + macOS" three screens away, which is exactly the reading order that produced the drift.
fn every_two_platform_phrase_at_an_enumerated_site_has_a_nearby_ratification_pointer() {
    let mut unqualified = Vec::new();
    for (relative, _) in PLATFORM_CLAIM_SITES {
        let flat = flatten(&scannable(relative));
        for offset in unqualified_phrase_offsets(&flat) {
            unqualified.push(format!("{relative}: {}", excerpt(&flat, offset)));
        }
    }
    assert!(
        unqualified.is_empty(),
        "{} occurrence(s) of {TWO_PLATFORM_PHRASE:?} have no {POINTER} within {POINTER_WINDOW} \
         flattened characters. Either reword the claim or put the pointer beside it:\n  {}",
        unqualified.len(),
        unqualified.join("\n  ")
    );
}

#[test]
// The exemption above is the one thing in this file that can weaken it, so it gets its own gate.
// Three assertions: (1) it is load-bearing — §16.33's verbatim quotations really do contain
// unqualified two-platform phrases, so the exemption is not decoration; (2) it removes exactly one
// section, not the file; (3) two-platform phrases still survive OUTSIDE it, so the main gate above
// is still looking at something.
fn the_ratification_section_exemption_is_load_bearing_and_narrow() {
    const SPEC: &str = "docs/PIPELINE_SPEC_V1.md";
    let whole = read_site(SPEC);
    let without = scannable(SPEC);

    assert!(
        whole.contains(RATIFICATION_HEADING),
        "the ratification section must exist for the exemption to mean anything"
    );
    assert!(
        !without.contains(RATIFICATION_HEADING),
        "the exemption must actually remove §16.33's heading"
    );

    let removed = whole.len() - without.len();
    assert!(
        removed > 1_000,
        "the removed span is {removed} bytes, which is too small to be §16.33 — the section-end \
         search probably matched the wrong heading"
    );
    assert!(
        removed < whole.len() / 3,
        "the exemption removed {removed} of {} bytes; it must remove ONE section, not the file",
        whole.len()
    );

    let inside = flatten(&whole).matches(TWO_PLATFORM_PHRASE).count();
    let outside = flatten(&without).matches(TWO_PLATFORM_PHRASE).count();
    assert!(
        inside > outside,
        "§16.33 quotes {TWO_PLATFORM_PHRASE:?} verbatim (it must, per the transcription rule), so \
         the exempted span must contain at least one occurrence; found {inside} in the whole file \
         and {outside} outside the exemption"
    );
    assert!(
        outside > 0,
        "every remaining occurrence is inside the exemption, so the main gate is scanning nothing"
    );
}

#[test]
// The measured regression control, on synthetic text, and the reason this file exists at all.
//
// Three assertions, and all three are needed. (1) The wrapped, unqualified claim is invisible to a
// line-based scan — this is the actual defect, reproduced, not asserted from memory. (2) The
// flattened scanner sees it. (3) The same wrapped claim with a nearby pointer is accepted, so the
// scanner is not simply flagging everything, and assertion 2's failure is caused by the missing
// pointer rather than by the fixture.
fn a_claim_that_wraps_across_lines_is_invisible_to_a_line_based_scan_and_visible_to_this_one() {
    // Deliberately shaped like the real §16.12 item 21 site: the phrase straddles the line break.
    let wrapped_unqualified = "\
    taken for this (`dirs` is not in `[workspace.dependencies]` and v1 is Linux +
    macOS only). The recovery suggestion is derived from the resolved cache root.
";
    let wrapped_qualified = "\
    taken for this (`dirs` is not in `[workspace.dependencies]` and v1 is Linux +
    macOS only — the second half of that parenthetical is replaced by §16.33 item 3).
";

    assert!(
        !wrapped_unqualified
            .lines()
            .any(|line| line.contains(TWO_PLATFORM_PHRASE)),
        "fixture sanity: no single LINE may contain the phrase, or the control proves nothing \
         about the line-based blindness it is here to reproduce"
    );

    assert_eq!(
        unqualified_phrase_offsets(&flatten(wrapped_unqualified)).len(),
        1,
        "the flattened scanner must see the wrapped, unqualified claim a line-based grep misses"
    );

    assert!(
        unqualified_phrase_offsets(&flatten(wrapped_qualified)).is_empty(),
        "a wrapped claim with a nearby {POINTER} pointer is qualified and must NOT be reported; \
         otherwise the scanner flags every occurrence and the assertion above is vacuous"
    );
}
