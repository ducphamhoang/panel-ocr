//! spec §16.24 item 19(b) + cookbook rule 14, applied to the SPEC instead of to code — for every
//! supersession claim, the site it names must point back at the claiming section.
//!
//! Why this exists: the pipeline has an adjudicator for disagreement and no adversary for
//! consensus. Three supersession markers were missed in one session — §16.20 item 3(d), §16.24
//! item 6, §16.24 item 9 — not because anyone disputed them but because everyone agreed and nobody
//! checked. This gate does not need anyone to remember.
//!
//! **Two layers, and the split is deliberate.** Layer A parses an explicit `SUPERSEDES:` marker and
//! enforces the back-pointer: exact, zero false positives. Layer B scans PROSE claims and pins the
//! result as a set: its job is only to fail when a new claim is written in prose instead of the
//! marker form. Prose was measured at a ~44% false-positive rate — quoted anchors, slash-lists, and
//! citations of someone else's amendment are structurally ambiguous, not tunable — so prose must
//! never carry enforcement. A gate that cries wolf half the time gets allowlisted into uselessness.

use pc_testkit::paths;
use std::collections::BTreeSet;

/// The exact marker a claim uses to declare its targets. Fixed syntax so the parser needs no verb
/// list, no distance heuristic and no `regex` dependency.
const MARKER: &str = "**SUPERSEDES:";

/// §16.24 item 1(f)'s literal-constant pattern: a hard-coded expected number of parsed claims,
/// never derived from the file, so the gate cannot pass by finding zero claims and raising the
/// number is an edit that cannot be skipped.
const EXPECTED_PARSED_CLAIMS: usize = 3;

/// The three claims the maintainer's diagnosis named, listed individually so deleting any one
/// marker fails with THAT site in the message rather than a bare count mismatch.
const RATIFIED_SUPERSESSIONS: &[(&str, &str)] = &[
    ("16.25", "16.20 item 3"),
    ("16.25", "16.24 item 6"),
    ("16.25", "16.24 item 9"),
];

/// Every PROSE-form claim in the file today, measured at `87c74c6`. Layer B's pinned set.
///
/// This is NOT a list of sites needing fixes — roughly eight of these are false positives of prose
/// parsing (quoted anchors, slash-lists, citations of another section's amendment), and adjudicating
/// which is which is the retroactive pass's job, not this gate's. Asserted as a sorted SET, so it
/// can neither grow silently nor shrink silently: converting a prose claim to the marker form
/// removes its row here in the same commit.
const PRE_CONVENTION_PROSE_CLAIMS: &[(&str, &str)] = &[
    ("15", "8.7"),
    ("15", "9.7"),
    ("16.10", "11.1"),
    ("16.10", "11.5"),
    ("16.15", "8.3 step 3"),
    ("16.19", "16.12 item 2"),
    ("16.20", "1"),
    ("16.20", "15 item 10"),
    ("16.20", "2.4"),
    ("16.20", "2.5"),
    ("16.20", "8.3 step 3"),
    ("16.20", "8.3 step 6"),
    ("16.21", "16.20 item 3"),
    ("16.21", "8.3 step 3"),
    ("16.22", "16.20 item 3"),
    ("16.24", "16.20 item 12"),
    ("16.24", "16.20 item 3"),
    ("16.24", "2.4"),
    ("16.24", "2.5"),
    ("16.25", "16.24 item 9"),
    // §16.26 item 6: item 1's verbatim quotes of real spec lines, claim-shaped to the scanner.
    // Quotations, not claims. Rewording them would destroy item 1's virtue (each example is a
    // real line of this spec), so they are pinned instead.
    // Quoted-anchor false positive: this is §16.26 item 1's verbatim quote of §13.
    ("16.26", "13"),
    // Quoted-anchor false positive: this is §16.26 item 1's verbatim quote of §16.20 item 3.
    ("16.26", "16.20 item 3"),
    ("16.6", "11.3"),
    ("16.6", "2.4"),
    ("16.6", "8.3 step 4"),
    ("16.7", "8.7"),
    ("16.9", "13"),
    // Verb-and-unrelated-anchor-on-one-line: line 741's clause names §15.10, not §7.2.1.
    ("8.7", "7.2.1"),
];

/// Verbs Layer B recognises. Deliberately unused by Layer A.
const PROSE_VERBS: &[&str] = &[
    "SUPERSEDED",
    "superseded",
    "supersedes",
    "AMENDED",
    "Amended",
    "amended",
    "ERRATUM",
    "QUALIFIED",
    "withdrawn",
    "NARROWED",
    "RE-GROUNDED",
];

// ── the parser: pure over `&str`, so the synthetic controls can drive it ──────

#[derive(Debug, Clone, PartialEq, Eq)]
struct Anchor {
    section: String,
    /// `Some` only for `item N`; `step N` and bare sections resolve to the whole section.
    item: Option<String>,
    /// Rendered form, for messages.
    label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Claim {
    /// Section containing the claim.
    host: String,
    line: usize,
    target: Anchor,
}

/// `(start, end)` line indices of every numbered section, and of every `^N. ` item within one.
struct Outline {
    sections: Vec<(String, usize, usize)>,
    items: Vec<(String, String, usize, usize)>,
}

fn outline(lines: &[&str]) -> Outline {
    let mut heads: Vec<(String, usize)> = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let rest = line
            .strip_prefix("### ")
            .or_else(|| line.strip_prefix("## "));
        if let Some(rest) = rest {
            let number: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            // Trailing `.` belongs to the prose (`## 16. Summary`), not to the number.
            let number = number.trim_end_matches('.').to_owned();
            if !number.is_empty() && number.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                heads.push((number, index));
            }
        }
    }
    let mut sections = Vec::new();
    for (position, (number, start)) in heads.iter().enumerate() {
        let end = heads.get(position + 1).map_or(lines.len(), |next| next.1);
        sections.push((number.clone(), *start, end));
    }

    let mut items = Vec::new();
    for (number, start, end) in &sections {
        let marks: Vec<(String, usize)> = (*start..*end)
            .filter_map(|index| item_marker(lines[index]).map(|n| (n, index)))
            .collect();
        for (position, (item, at)) in marks.iter().enumerate() {
            let item_end = marks.get(position + 1).map_or(*end, |next| next.1);
            items.push((number.clone(), item.clone(), *at, item_end));
        }
    }
    Outline { sections, items }
}

/// `^N. ` at column 0.
fn item_marker(line: &str) -> Option<String> {
    let digits: String = line.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    line[digits.len()..].starts_with(". ").then_some(digits)
}

/// Parse one `§`-anchor beginning at `text`, e.g. `§16.20 item 3(d)` or `§8.3 step 6`.
fn parse_anchor(text: &str) -> Option<Anchor> {
    let rest = text.strip_prefix('§')?;
    let raw_section: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let raw_section = raw_section.trim_end_matches('.').to_owned();
    if raw_section.is_empty() {
        return None;
    }
    // REVERTED (Orchestrator, 2026-07-29). This read:
    //     let section = raw_section.split('.').take(2).collect::<Vec<_>>().join(".");
    // which resolved a three-component anchor against its two-component parent, so `§7.2.1`
    // resolved to §7.2 -- and so would `§7.2.999`. That silences
    // `every_pinned_pre_convention_target_still_resolves` by making an unresolvable anchor
    // resolve, which is exactly what §16.26 item 4 ratified against ("an unresolvable anchor is
    // a FAILURE, not a skip"), and it is the same bypass the drafting measurement script had as
    // `if target not in spans: continue`. The anchor is taken whole, as drafted.
    //
    // Consequence, now ratified: the anchor is taken whole here, and target_span applies §16.26
    // item 3(d): a three-component anchor resolves through its two-component parent's span only
    // when its literal sub-section token appears as a bold heading inside that span. §7.2.1
    // therefore resolves to the ReplayDetector clause inside §7.2, while §7.2.999 does not;
    // §7.2.1 is the spec's only three-component citation, and it always means that clause.
    let section = raw_section.clone();
    let after = &rest[raw_section.len()..];
    let (item, label_tail) = if let Some(tail) = after.strip_prefix(" item ") {
        let n: String = tail.chars().take_while(char::is_ascii_digit).collect();
        if n.is_empty() {
            (None, String::new())
        } else {
            (Some(n.clone()), format!(" item {n}"))
        }
    } else if let Some(tail) = after.strip_prefix(" step ") {
        let n: String = tail.chars().take_while(char::is_ascii_digit).collect();
        // `step` targets scope to the whole section: §8.3 uses `**N — Title.**` markers, not
        // `^N. `, so it has zero parseable items. Lenient but never wrong (plan §2 point 1).
        (
            None,
            if n.is_empty() {
                String::new()
            } else {
                format!(" step {n}")
            },
        )
    } else {
        (None, String::new())
    };
    Some(Anchor {
        label: format!("§{raw_section}{label_tail}"),
        section,
        item,
    })
}

/// Layer A: every `**SUPERSEDES: ...**` marker, with its declared targets.
fn marker_claims(lines: &[&str], outline: &Outline) -> Vec<Claim> {
    let mut claims = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(at) = line.find(MARKER) else {
            continue;
        };
        let after = &line[at + MARKER.len()..];
        let declaration = after.split("**").next().unwrap_or(after);
        let host = section_of(outline, index);
        for (offset, _) in declaration.match_indices('§') {
            if let Some(target) = parse_anchor(&declaration[offset..]) {
                claims.push(Claim {
                    host: host.clone(),
                    line: index + 1,
                    target,
                });
            }
        }
    }
    claims
}

fn section_of(outline: &Outline, line: usize) -> String {
    outline
        .sections
        .iter()
        .find(|(_, start, end)| (*start..*end).contains(&line))
        .map(|(number, _, _)| number.clone())
        .unwrap_or_default()
}

/// Numeric ordering, so `16.24 > 16.20 > 14 > 8.3`. `None` when unparseable.
fn version(number: &str) -> Option<Vec<u32>> {
    number.split('.').map(|part| part.parse().ok()).collect()
}

/// The span of an anchor's target: the item's lines when it names one, else the whole section.
/// `None` when the anchor does not resolve — a FAILURE for the caller, never a skip.
fn target_span(lines: &[&str], outline: &Outline, target: &Anchor) -> Option<(usize, usize)> {
    if let Some(item) = &target.item {
        return outline
            .items
            .iter()
            .find(|(section, number, _, _)| section == &target.section && number == item)
            .map(|(_, _, start, end)| (*start, *end));
    }
    if target.section.split('.').count() == 3 {
        let mut components = target.section.rsplitn(2, '.');
        components.next();
        let parent = components.next()?;
        let span = outline
            .sections
            .iter()
            .find(|(number, _, _)| number == parent)
            .map(|(_, start, end)| (*start, *end))?;
        let heading_prefix = format!("**{} ", target.section);
        let exists = lines[span.0..span.1].iter().any(|line| {
            let heading = line.trim();
            heading.starts_with(&heading_prefix) && heading.ends_with("**")
        });
        return exists.then_some(span);
    }
    outline
        .sections
        .iter()
        .find(|(number, _, _)| number == &target.section)
        .map(|(_, start, end)| (*start, *end))
}

/// Does `span` reference `§<host>`? Requires the sigil and a non-digit boundary, so `§16.2` does not
/// satisfy a claim by `§16.25`.
fn contains_back_pointer(lines: &[&str], span: (usize, usize), host: &str) -> bool {
    let needle = format!("§{host}");
    for line in &lines[span.0..span.1] {
        for (offset, _) in line.match_indices(needle.as_str()) {
            let after = &line[offset + needle.len()..];
            let is_numeric_extension = after.starts_with(|c: char| c.is_ascii_digit())
                || after
                    .strip_prefix('.')
                    .is_some_and(|tail| tail.starts_with(|c: char| c.is_ascii_digit()));
            if !is_numeric_extension {
                return true;
            }
        }
    }
    false
}

fn mask_marker_declarations(line: &str) -> String {
    let mut masked = String::with_capacity(line.len());
    let mut cursor = 0;
    while let Some(relative) = line[cursor..].find(MARKER) {
        let start = cursor + relative;
        masked.push_str(&line[cursor..start]);
        let declaration_start = start + MARKER.len();
        let end = line[declaration_start..]
            .find("**")
            .map_or(line.len(), |relative| declaration_start + relative + 2);
        masked.push(' ');
        cursor = end;
        if cursor == line.len() {
            break;
        }
    }
    masked.push_str(&line[cursor..]);
    masked
}

/// Layer B: prose claims, as `(host, target label without the sigil)`.
fn prose_claims(lines: &[&str], outline: &Outline) -> BTreeSet<(String, String)> {
    let mut found = BTreeSet::new();
    for (index, line) in lines.iter().enumerate() {
        // Masking beats skipping: the marker verb and declared anchors remain inside
        // the masked span, so Layer A and Layer B stay structurally disjoint even if
        // that verb is added to PROSE_VERBS. A prose claim outside the declaration
        // on the same physical line remains visible to Layer B.
        let scan_line = mask_marker_declarations(line);
        if !PROSE_VERBS.iter().any(|verb| scan_line.contains(verb)) {
            continue;
        }
        let host = section_of(outline, index);
        let Some(host_version) = version(&host) else {
            continue;
        };
        for (offset, _) in scan_line.match_indices('§') {
            let Some(target) = parse_anchor(&scan_line[offset..]) else {
                continue;
            };
            let Some(target_version) = version(&target.section) else {
                continue;
            };
            // Direction: a CLAIM names an older target; a BACK-POINTER names a newer claimer.
            if target_version >= host_version {
                continue;
            }
            let before = &scan_line[..offset];
            // `superseded BY §X` — §X is the superseder, so this is a back-pointer.
            if before.trim_end().ends_with("by") {
                continue;
            }
            // `(ERRATUM, §X …)` — a parenthesised back-pointer marker inside the target.
            if ["(ERRATUM", "(QUALIFIED", "(SUPERSEDED", "(NARROWED"]
                .iter()
                .any(|open| before.contains(open))
            {
                continue;
            }
            found.insert((
                host.clone(),
                target.label.trim_start_matches('§').to_owned(),
            ));
        }
    }
    found
}

fn spec() -> String {
    // This gate deliberately reads only docs/PIPELINE_SPEC_V1.md. docs/COOKBOOK.md rule 14b
    // legitimately contains the literal marker text inside inline code while discussing it;
    // widening the scan to all docs would therefore trip on prose about the gate.
    let path = paths::workspace_root().join("docs/PIPELINE_SPEC_V1.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read `{}`: {error}", path.display()))
}

// ── Layer A ──────────────────────────────────────────────────────────────────

#[test]
// The gate itself: for every `SUPERSEDES:` marker, the named site must point back at the claiming
// section. This is cookbook rule 14 applied to the spec — "when a ratified decision changes a shared
// artifact, enumerate every reader, record the enumeration" — where the artifact is a spec clause and
// the reader is anyone who cites it without noticing it was superseded.
//
// What must break for this to fail: delete a back-pointer from a superseded site, or add a
// `SUPERSEDES:` marker without adding one. Nothing else. The check is scoped to the target's own
// span, so a pointer elsewhere in the file does not satisfy it — proven separately by
// `a_back_pointer_outside_the_target_span_does_not_satisfy_the_gate`.
fn every_supersession_marker_has_a_back_pointer_at_its_target() {
    let text = spec();
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);
    let claims = marker_claims(&lines, &outline);

    let mut missing = Vec::new();
    for claim in &claims {
        let span = target_span(&lines, &outline, &claim.target).unwrap_or_else(|| {
            panic!(
                "§{} (line {}) claims to supersede {}, which does not resolve to any section or \
                 item. An unresolvable anchor is a FAILURE, not a skip: a typo'd anchor would \
                 otherwise exempt the claim from its own gate.",
                claim.host, claim.line, claim.target.label
            )
        });
        if !contains_back_pointer(&lines, span, &claim.host) {
            missing.push(format!(
                "\n  {} (spec line {}) is superseded by §{}, but {}'s own text (lines {}-{}) \
                 never mentions §{}.\n    Fix: add a marker at the START of {} — e.g.\n      \
                 **SUPERSEDED by §{} — read it before citing this clause.**\n    Why: anyone \
                 reading {} in isolation must learn it has been superseded from {} itself. A \
                 pointer that exists only in §{} is invisible to them.",
                claim.target.label,
                claim.line,
                claim.host,
                claim.target.label,
                span.0 + 1,
                span.1,
                claim.host,
                claim.target.label,
                claim.host,
                claim.target.label,
                claim.target.label,
                claim.host,
            ));
        }
    }
    assert!(
        missing.is_empty(),
        "{} supersession claim(s) have no back-pointer at the site they name:{}",
        missing.len(),
        missing.join("")
    );
}

#[test]
// Anti-vacuity, §16.24 item 1(f)'s literal-constant pattern: a hard-coded count that cannot be
// derived from the file, so the gate above cannot pass by finding zero markers — and raising it is
// an edit that cannot be skipped. Deriving this number from the tree is forbidden: that is cookbook
// rule 13's iteration-2 collapse, where the expectation comes from the artifact under test.
fn the_parsed_supersession_claim_count_is_pinned() {
    let text = spec();
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);
    let claims = marker_claims(&lines, &outline);

    assert_eq!(
        claims.len(),
        EXPECTED_PARSED_CLAIMS,
        "expected {EXPECTED_PARSED_CLAIMS} parsed supersession claim(s), found {}. Adding a supersession \
         claim means raising EXPECTED_PARSED_CLAIMS in the same commit; a count that tracks the file \
         automatically would make this gate pass on zero claims.\n  found: {:?}",
        claims.len(),
        claims
            .iter()
            .map(|claim| format!("§{} -> {}", claim.host, claim.target.label))
            .collect::<Vec<_>>()
    );
}

#[test]
// The three sites the maintainer's diagnosis named, asserted INDIVIDUALLY rather than by count, so
// deleting any one marker fails with that site in the message. Both directions: the set must match
// exactly, so an extra unrecorded marker also fails.
fn the_three_ratified_supersessions_are_covered() {
    let text = spec();
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);

    let found: BTreeSet<(String, String)> = marker_claims(&lines, &outline)
        .into_iter()
        .map(|claim| {
            (
                claim.host,
                claim.target.label.trim_start_matches('§').to_owned(),
            )
        })
        .collect();
    let expected: BTreeSet<(String, String)> = RATIFIED_SUPERSESSIONS
        .iter()
        .map(|(host, target)| ((*host).to_owned(), (*target).to_owned()))
        .collect();

    let absent: Vec<_> = expected.difference(&found).collect();
    let extra: Vec<_> = found.difference(&expected).collect();
    assert!(
        absent.is_empty() && extra.is_empty(),
        "the marker set changed.\n  missing (a ratified supersession lost its marker): {absent:?}\n  \
         unrecorded (add it to RATIFIED_SUPERSESSIONS deliberately): {extra:?}"
    );
}

// ── Layer B ──────────────────────────────────────────────────────────────────

#[test]
// The tripwire. Its job is NOT to find missing back-pointers — prose parsing was measured at a ~44%
// false-positive rate and cannot carry enforcement (see this file's header). Its job is to fail when
// a NEW claim is written in prose instead of the `SUPERSEDES:` marker form, so the convention cannot
// erode quietly.
//
// The set is pinned, not counted: a swap keeps a count and breaks a set (cookbook rule 13's
// "cardinality is not identity"). A REMOVAL fails too, which is correct — converting a prose claim
// to the marker form is a deliberate act that updates this constant in the same commit.
fn prose_form_claims_match_the_recorded_pre_convention_set() {
    let text = spec();
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);
    let found = prose_claims(&lines, &outline);
    let expected: BTreeSet<(String, String)> = PRE_CONVENTION_PROSE_CLAIMS
        .iter()
        .map(|(host, target)| ((*host).to_owned(), (*target).to_owned()))
        .collect();

    let new: Vec<_> = found.difference(&expected).collect();
    let gone: Vec<_> = expected.difference(&found).collect();
    assert!(
        new.is_empty() && gone.is_empty(),
        "the prose-claim set changed.\n  \
         NEW prose claim(s) — write these as `{MARKER} §X item N**` instead, so the back-pointer \
         gate can enforce them: {new:?}\n  \
         GONE — if you converted one to the marker form, delete its row from \
         PRE_CONVENTION_PROSE_CLAIMS in this commit: {gone:?}\n\
         Roughly a third of the pinned rows are false positives of prose parsing (quoted anchors, \
         slash-lists, citations of another section's amendment). Adjudicating them is the \
         retroactive pass's job, not this gate's."
    );
}

#[test]
// The allowlist must not rot. Every pinned row's target must still resolve, so deleting or
// renumbering a section cannot leave a stale exemption behind that silently covers nothing —
// the failure mode of every allowlist that outlives its subject.
fn every_pinned_pre_convention_target_still_resolves() {
    let text = spec();
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);

    let mut stale = Vec::new();
    for (host, target) in PRE_CONVENTION_PROSE_CLAIMS {
        let anchor = parse_anchor(&format!("§{target}"))
            .unwrap_or_else(|| panic!("pinned target `{target}` is not a parseable anchor"));
        if target_span(&lines, &outline, &anchor).is_none() {
            stale.push(format!("§{host} -> §{target}"));
        }
    }
    assert!(
        stale.is_empty(),
        "pinned pre-convention row(s) name a target that no longer exists; remove them rather than \
         leaving an exemption covering nothing: {stale:?}"
    );
}

// ── the parser's own controls, on synthetic text ──────────────────────────────

#[test]
// spec §16.24 item 19(b)'s prohibition, made mechanical: a back-pointer ELSEWHERE in the file must
// not satisfy the gate. Driven on synthetic text so it tests the scoping rule rather than the
// current spec, and asserts both directions — the pointer inside the target's span passes, the same
// text one item away does not.
fn a_back_pointer_outside_the_target_span_does_not_satisfy_the_gate() {
    let inside = "\
## 16.20 Older section

1. First item, untouched.

2. **SUPERSEDED by §16.25.** The clause this test cares about.

## 16.25 Newer section

1. **SUPERSEDES: §16.20 item 2** — the claim.
";
    let outside = "\
## 16.20 Older section

1. First item, and the pointer is parked HERE instead: §16.25.

2. The clause that should carry the pointer and does not.

## 16.25 Newer section

1. **SUPERSEDES: §16.20 item 2** — the claim.
";

    for (label, text, expect_pointer) in [("inside", inside, true), ("outside", outside, false)] {
        let lines: Vec<&str> = text.lines().collect();
        let outline = outline(&lines);
        let claims = marker_claims(&lines, &outline);
        assert_eq!(claims.len(), 1, "{label}: one marker expected");
        let span = target_span(&lines, &outline, &claims[0].target)
            .expect("§16.20 item 2 resolves in the synthetic text");
        assert_eq!(
            contains_back_pointer(&lines, span, &claims[0].host),
            expect_pointer,
            "{label}: item-scoped back-pointer detection is wrong; a pointer in item 1 must not \
             satisfy a claim against item 2"
        );
    }
}

#[test]
fn a_same_line_prose_claim_survives_marker_masking_without_emitting_the_marker_target() {
    let text = "\
## 16.10 Older target

1. The older clause.

## 16.20 Marker target

1. The other older clause.

## 16.24 Newer section

1. **SUPERSEDES: §16.20 item 1** — The class list is also amended at §16.10 item 1.
";
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);
    let found = prose_claims(&lines, &outline);

    // Assertion 1 pins the hole closed; assertion 2 pins the two layers disjoint.
    assert!(found.contains(&("16.24".to_owned(), "16.10 item 1".to_owned())));
    assert!(!found.contains(&("16.24".to_owned(), "16.20 item 1".to_owned())));
}

#[test]
// This checks anchor existence only; it does NOT cover back-pointer detection.
fn nonexistent_three_component_anchor_does_not_resolve() {
    let text = spec();
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);
    let target = parse_anchor("§7.2.999").expect("control anchor should parse");

    assert!(
        target_span(&lines, &outline, &target).is_none(),
        "§7.2.999 must not resolve without a matching bold heading in §7.2"
    );
}

#[test]
// §X is the stated reason the count test sees three parsed claims versus four physical markers.
fn parse_anchor_rejects_non_numeric_section_placeholder() {
    assert!(parse_anchor("§X item N").is_none());
}

#[test]
// The boundary rule in `contains_back_pointer`: `§16.2` must not satisfy a claim by `§16.25`, or
// every claim by a section whose number extends another's would be spuriously green. Prefix
// matching on section numbers is the subtlest way this gate could have been vacuous.
fn a_numeric_prefix_does_not_satisfy_a_back_pointer() {
    let lines = vec!["1. See §16.2 for context.", "2. See §16.25 item 5."];
    assert!(!contains_back_pointer(&lines, (0, 1), "16.25"));
    assert!(contains_back_pointer(&lines, (1, 2), "16.25"));
    // And the reverse direction: §16.2 IS matched by its own number.
    assert!(contains_back_pointer(&lines, (0, 1), "16.2"));
}
