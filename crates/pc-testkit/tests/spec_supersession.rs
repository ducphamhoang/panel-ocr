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
use std::collections::{BTreeMap, BTreeSet};

/// The exact marker a claim uses to declare its targets. Fixed syntax so the parser needs no verb
/// list, no distance heuristic and no `regex` dependency.
const MARKER: &str = "**SUPERSEDES:";

/// §16.24 item 1(f)'s literal-constant pattern: a hard-coded expected number of parsed claims,
/// never derived from the file, so the gate cannot pass by finding zero claims and raising the
/// number is an edit that cannot be skipped.
const EXPECTED_PARSED_CLAIMS: usize = 41;

/// Claims whose declared sub-item identity must scope the target-side back-pointer independently.
/// Most historical lettered claims retain §16.26 item 3(a)'s parent-item span. These two are pinned
/// because §16.40 resolves two siblings in the same item and either sibling's pointer would otherwise
/// satisfy both claims, making each dedicated annotation decorative.
const SUB_ITEM_SCOPED_BACKPOINTERS: &[(&str, &str)] =
    &[("16.40", "16.38 item 25(a)"), ("16.40", "16.38 item 25(d)")];

/// Every ratified supersession claim, keyed by `(host, MARKER IDENTITY)` — the identity being the
/// target label plus whatever sub-item letter the marker's own anchor text declares (see
/// `marker_sub_item`). One row per marker, not one row per resolved span.
///
/// **Why the identity carries the letter.** §16.27 declares three separate markers against §16.24
/// item 18, one per sub-item erratum: 18(b) (leg-2 scope), 18(i) (derivation-conditional message),
/// 18(k) ("rare-but-real" conflation). Span resolution deliberately collapses all three to item
/// 18's whole span — ratified §16.26 item 3(a) leniency, and correct, because the real
/// back-pointers sit inside the parent item. But collapsing the *span* is not a reason to collapse
/// the *row*: while all three read `§16.24 item 18` with no letter, the three markers were
/// indistinguishable at this level, so the marker backing 18(i) could be deleted and a duplicate
/// `item 18` marker added back — total count still 3, the multiset row for
/// `("16.27", "16.24 item 18")` still "3 found, 3 expected", suite still green, and 18(i)'s
/// coverage silently gone. The three markers now declare their letters, so each has its own row
/// below and losing any one names *which* erratum lost its marker.
///
/// `("16.25", "16.20 item 3(d)")` carries a letter for the same reason and by the same parse; that
/// marker has declared `item 3(d)` since it was written.
///
/// This stays a `Vec` compared as a MULTISET (`tally`/`tally_diff`) even though every row below
/// happens to be distinct today: nothing forbids a future entry from legitimately carrying two
/// markers with the same identity, and a `BTreeSet` would silently absorb the second. Uniqueness of
/// today's rows is a fact about today's spec, not a property the comparison may assume.
const RATIFIED_SUPERSESSIONS: &[(&str, &str)] = &[
    ("16.25", "16.20 item 3(d)"),
    ("16.25", "16.24 item 6"),
    ("16.25", "16.24 item 9"),
    ("16.27", "16.20 item 3"),
    ("16.27", "16.24 item 18(b)"),
    ("16.27", "16.24 item 18(i)"),
    ("16.27", "16.24 item 18(k)"),
    ("16.27", "16.24 item 20"),
    ("16.27", "16.24 item 4"),
    ("16.27", "16.25 item 10"),
    ("16.27", "16.6 item 4"),
    ("16.27", "16.24 item 21"),
    ("16.28", "16.27 item 1(g)"),
    // §16.30's three markers. The first is item-scoped; the other two are BARE-SECTION targets, and
    // deliberately so — §9.5's P7 row and §13's row 30 live inside markdown tables, which carry no
    // `^N. ` item marker, so no item-scoped anchor exists to name. That is ratified §16.26 item
    // 3(c) leniency ("a bare `§N` target with no item scopes to the whole section"), the same class
    // as the `step N` fallback in item 3(b). Consequence worth stating rather than discovering: a
    // back-pointer anywhere in §9.5 or anywhere in §13 satisfies the gate, so the row-level
    // precision of those two annotations is a convention this constant cannot enforce.
    ("16.30", "16.12 item 4"),
    ("16.30", "9.5"),
    ("16.30", "13"),
    // §16.31's single marker. Item-scoped, and narrower than a decision reversal: §16.29 item 2's
    // RULING stands whole, and what §16.31 item 3(e) qualifies is one parenthetical of its GROUNDS
    // — the description of `assert_known_non_committed_form`'s accepted set as "a bare `*.pt.onnx`
    // filename", which stops describing the predicate once the enumerated three-name allow-list
    // lands. Recorded here rather than left to the reader because the target site is the one a
    // future reader lands on, which is the whole point of §16.26.
    ("16.31", "16.29 item 2"),
    ("16.32", "16.19 item 10"),
    ("16.32", "14 item 15"),
    ("16.32", "4.5"),
    // §16.33's three markers. `("16.33", "16")` is a BARE-SECTION target and deliberately so: the
    // superseded text is a bullet in §16's markdown-style out-of-scope list, which carries no
    // `^N. ` item marker, so no item-scoped anchor exists to name. That is ratified §16.26 item
    // 3(c) leniency, the same class as §16.30's `9.5` and `13` rows above, with the same
    // consequence: a back-pointer anywhere in §16 satisfies the gate, so the bullet-level
    // precision of that annotation is a convention this constant cannot enforce.
    //
    // §16 is also the LAST section in the file, so its span runs to EOF. Nothing else can be
    // appended after it without landing inside that span; a future §16.34 must be inserted BEFORE
    // §16, exactly as §16.33 was.
    ("16.33", "16"),
    ("16.33", "16.12 item 21"),
    ("16.33", "16.22 item 1"),
    // §16.34's two markers. `("16.34", "9.5")` is a BARE-SECTION target, deliberately so — §9.5's
    // P8 row lives inside a markdown table, the same shape §16.30's `9.5`/`13` rows above already
    // document; a back-pointer anywhere in §9.5 satisfies the gate. §13's row 31 is deliberately
    // NOT superseded: unlike §9.5's row, it never enumerated the overrides it references (a bare
    // pointer to §15.5), so nothing there is stale — the step-1a fresh-reader pass caught the
    // first transcription's mistaken claim that both rows needed a marker.
    ("16.34", "15 item 5"),
    ("16.34", "9.5"),
    // §16.35's single marker: the mask-quality-polish rescue tier amends the failure-threshold
    // clause (step 10 has no `^N. ` item marker at the section level it lives in, so `step 10`
    // resolves to the whole §10.3 span per the `step N` fallback documented above).
    ("16.35", "10.3 step 10"),
    // §16.36's single marker: pins the device config key's section, previously unstated.
    ("16.36", "8.3 step 3"),
    // §16.38's NINE markers (LaMa inpainting) — eight landed with the ratification, the ninth
    // (`13.1`) with task L3; every "eight" in the paragraphs below is a measurement on the
    // original eight and is deliberately not restated as a claim about nine. No §16.37 row
    // exists on this branch: that number is
    // occupied by the sibling `mask-parity` branch, which forked from the same base commit, so this
    // entry is numbered 16.38 to avoid two `## 16.37` headings after a merge. Whoever merges the two
    // branches owns reconciling BOTH branches' rows here and BOTH branches' raises of
    // EXPECTED_PARSED_CLAIMS, since each was measured against the shared base (§16.38 item 17(d)).
    //
    // **A parallel-branch collision this file does NOT gate, recorded here because this is where a
    // reader will look for it.** The §-section number above is one of THREE collisions between these
    // branches; the other two are the §14 DEVIATION register number — both branches independently
    // reached for 23, since the shared base ended at `…19, 21, 22` with 20 unused, and
    // `mask-parity`'s `1b4e12f` had already committed `DEVIATION(23)`, so §16.38 takes 24-27 — and
    // this constant together with EXPECTED_PARSED_CLAIMS. **Nothing in this repo gates the
    // register-number half:** no test reads §14's item numbering or cross-checks it against the
    // `// DEVIATION(n)` comments in code, so two branches can still merge into two different
    // deviations sharing one n. That gap is disclosed, not closed; §16.38 item 17(d)(ii) and §14's
    // own "item 23 is not absent by accident" note are the record.
    //
    // Four of the eight are BARE-SECTION targets — `2.8`, `6`, `12.2`, `12.3` — and deliberately so:
    // none of those four sections carries a `^N. ` item marker (§2.8 and §12.2 are prose plus a code
    // fence; §6 is prose plus TOML; §12.3 numbers its steps as `**N — Title.**`, the same shape §8.3
    // uses and the reason the `step N` fallback exists). That is ratified §16.26 item 3(c) leniency,
    // the same class as §16.30's `9.5`/`13` rows and §16.33's `16` row, with the same consequence: a
    // back-pointer anywhere in the section satisfies the gate, so the clause-level precision of
    // those four annotations is a convention this constant cannot enforce.
    //
    // The four item-scoped rows carry their back-pointers at the END of the target item rather than
    // its start, against §16.26's usual preference, and for a measured reason: all four of those
    // items OPEN on a physical line that already cites an older section (§10.1, §11.3, §12.3,
    // §16.21/§16.22), so placing a supersession verb on that line would manufacture a phantom Layer
    // B prose claim out of a line break. Cookbook rule 14b, applied before it bit rather than after.
    //
    // **Seven of these eight back-pointers are individually load-bearing; `("16.38", "6")` is NOT,
    // and the difference is sharper than the leniency noted above.** A step-1a fresh reader deleted
    // each of the eight in turn and watched the gate: seven turn it red, and §6's does not. Cause,
    // re-measured here: §6's span carries **five** separate `§16.38` mentions, because §16.38 also
    // added the `[inpainter]` TOML block and a validation clause to that section, and each of those
    // cites §16.38 in its own comments — while §2.8, §12.2 and §12.3 carry exactly one each.
    // `contains_back_pointer` asks only whether the span mentions the host, so any one of the five
    // satisfies the claim and the header note could be deleted with the suite green. The leniency
    // paragraph above says PLACEMENT within a section is unenforced; this is one step further — for
    // §6 alone, the marker's EXISTENCE is unenforced too. Do not read a green suite as proof that
    // all eight notes are present. Closing this would need a per-target expected mention count,
    // which is a hard-coded number derived from prose and would go stale on every edit to §6; the
    // trade was made deliberately in favour of disclosure.
    //
    // **The ninth row, `("16.38", "13.1")`, landed later than the other eight** — with task
    // L3, when decision D1 was resolved and transcribed as §16.38 item 19. It is
    // item-scoped in neither direction: `13.1` is a `### ` heading, so `outline()` resolves
    // it as a two-component SECTION and the back-pointer may sit anywhere inside §13.1's
    // span (which runs from that heading to `## 14.`). Unlike the `("16.38", "6")` row
    // below, this one IS load-bearing today: §13.1's span carries exactly one `§16.38`
    // mention — the back-pointer note itself — so deleting it turns the gate red. That is a
    // fact about §13.1's current text, not a property of the row; a future edit adding a
    // second `§16.38` citation anywhere in §13.1 would quietly make it non-load-bearing in
    // the same way §6's is, and nothing here would notice.
    ("16.38", "13.1"),
    ("16.38", "2.8"),
    ("16.38", "6"),
    ("16.38", "12.2"),
    ("16.38", "12.3"),
    ("16.38", "16.9 item 2"),
    ("16.38", "16.10 item 2"),
    ("16.38", "16.11 item 3"),
    ("16.38", "16.23 item 5"),
    // §16.38 item 20's TWO markers (the L4 Gaussian hoist), taking this section from nine
    // markers to eleven and EXPECTED_PARSED_CLAIMS from 36 to 38.
    //
    // `("16.38", "16.10 item 1")` is item-scoped, and its back-pointer sits at the END of
    // §16.10 item 1 for the same reason the four item-scoped rows above do: that item OPENS on a
    // line already citing §11.1 and §11.5, so a supersession verb there would manufacture a
    // phantom Layer B prose claim out of a line break (cookbook rule 14b). It is load-bearing
    // today, but LESS SO than "exactly one mention" would suggest, and the count is stated
    // precisely because the earlier wording here rounded it wrong: §16.10 item 1's span carries
    // exactly **one LINE** mentioning §16.38 — the back-pointer sentence — and that single line
    // carries **TWO** `§16.38` tokens (the `SUPERSEDED in part by §16.38 item 20` back-pointer, and
    // the later `§16.38 item 20 is explicitly not authority for hoisting it` scope clause).
    // Consequence, since this gate matches on the token and not on the verb: deleting the
    // back-pointer clause alone would leave the second token in the span and the gate would stay
    // GREEN. Only removing the whole line turns it red. All of that is a fact about today's text,
    // not a property of the row.
    //
    // **The marker's own text is SCOPED to `gaussian`, and this constant cannot enforce that.**
    // §16.10 item 1 places `nlm` AND `gaussian`; item 20 hoists `gaussian` only, and `nlm`
    // stays. The gate checks that a back-pointer exists in the target span, never what it says,
    // so if someone later widens the claim to `nlm` on this entry's authority nothing here goes
    // red. Same class of unenforced-content gap as the placement leniency documented above.
    //
    // `("16.38", "13")` is a BARE-SECTION target for §13's row 14, which attributes "Gaussian
    // blur" to `pc-denoise`. Deliberately bare: §13's rows live in a markdown table and carry no
    // `^N. ` item marker, so no item-scoped anchor exists — ratified §16.26 item 3(c) leniency,
    // the same class as §16.30's `13` row above, with the same consequence that a back-pointer
    // anywhere in §13 satisfies the gate. It IS load-bearing today: §13's span (from `## 13.` to
    // `### 13.1`) carries exactly one `§16.38` mention, the row-14 annotation, since §16.38's
    // other §13 marker targets §13.1 and lands in a later span.
    //
    // **§11.5's N2 row is a third target that gets NO row here, and its absence is deliberate,
    // not an omission.** The hoist RESTORES what that row always said (`pc-imageops`), and a
    // restoration is not a supersession, so §16.38 item 20(c) puts a plain note there and no
    // marker. Consequence, disclosed rather than left to be discovered: nothing in this file
    // gates that note's existence and deleting it leaves the suite green — the same shape as the
    // `("16.38", "6")` row's non-load-bearing back-pointer, accepted for the same reason.
    ("16.38", "16.10 item 1"),
    ("16.38", "13"),
    // §16.38 item 22's single marker (the L6 mask-precedence correction), taking this section
    // from eleven markers to twelve and EXPECTED_PARSED_CLAIMS from 38 to 39.
    //
    // **This is the file's first SELF-referential row: host and target are the same section.**
    // §16.38 item 22 corrects §16.38 item 12(a) — an entry amending its own earlier item rather
    // than an older section's. Layer A handles it without special-casing (`marker_claims` never
    // compares host to target), and `target_span` resolves `16.38 item 12` through the ordinary
    // `^N. ` item outline, so the back-pointer must sit inside item 12's own span and a pointer
    // parked in item 11 or item 13 would not satisfy it.
    //
    // Two consequences of the self-reference worth stating, because neither is obvious:
    //
    //   1. **Layer B cannot see this claim at all**, in either direction. `prose_claims` drops
    //      any candidate whose `target_version >= host_version`, and here they are EQUAL — so a
    //      future editor who rewrites this correction in prose instead of the marker form would
    //      NOT trip `prose_form_claims_match_the_recorded_pre_convention_set`. For same-section
    //      claims the prose tripwire is inert, and Layer A is the only thing enforcing the
    //      convention. That is a gap in the tripwire's coverage, disclosed rather than closed:
    //      relaxing the direction filter to admit equal versions would re-open the false-positive
    //      class the filter exists to suppress (every back-pointer inside §16.38 that names
    //      §16.38 would become a "claim").
    //
    //   2. **It IS load-bearing today**, and the count is stated precisely rather than rounded:
    //      §16.38 item 12's span carries exactly one line mentioning §16.38 — the
    //      `SUPERSEDED IN PART by §16.38 item 22` back-pointer added with item 22 — and that line
    //      carries exactly one `§16.38` token. Deleting the line turns
    //      `every_supersession_marker_has_a_back_pointer_at_its_target` red. Unlike the
    //      `("16.38", "16.10 item 1")` row above, there is no second token on the line to keep the
    //      gate green after a partial deletion. That is a fact about item 12's current text, not a
    //      property of this row: any later edit adding another `§16.38` citation anywhere inside
    //      item 12's span would quietly make this row non-load-bearing, exactly as the
    //      `("16.38", "6")` row already is, and nothing here would notice.
    //
    // The marker declares the sub-item letter (`item 12(a)`), so `Claim::identity` keys this row
    // as `16.24`-style lettered identities are keyed: the row below reads `12(a)`, while the
    // resolved SPAN is item 12 whole (ratified §16.26 item 3(a)). If a later erratum against a
    // different sub-item of item 12 is added, it gets its own row and losing either names which.
    ("16.38", "16.38 item 12(a)"),
    // §16.40 resolves two independently identified sub-items inside §16.38 item 25. Ordinary
    // `parse_anchor` / `target_span` first identifies parent item 25 under §16.26 item 3(a), but
    // `claim_target_span` deliberately narrows these two `SUB_ITEM_SCOPED_BACKPOINTERS` identities
    // to their respective (a)/(d) spans so each target annotation is independently load-bearing.
    // The two rows raise EXPECTED_PARSED_CLAIMS from 39 to 41.
    ("16.40", "16.38 item 25(a)"),
    ("16.40", "16.38 item 25(d)"),
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
    /// LAYER A ONLY. The marker's declared target *identity*: `target.label` without the sigil, plus
    /// the sub-item letter the marker's own anchor text spells out, when it spells one
    /// (`16.24 item 18(b)`). Constructed by `marker_claims` alone — `Claim` has no other
    /// constructor — so nothing in Layer B or in `target_span` can see it.
    ///
    /// Historical lettered identities remain identities only and resolve to their parent item under
    /// §16.26 item 3(a). The two identities in `SUB_ITEM_SCOPED_BACKPOINTERS` are the explicit narrow
    /// exception: `claim_target_span` resolves those letters to separate sub-item spans so sibling
    /// target annotations are independently falsifiable. Layer B and bare `target_span` remain
    /// unchanged.
    identity: String,
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

/// LAYER A ONLY. The sub-item letter(s) an anchor's text spells out immediately after the item
/// digits: `§16.24 item 18(b)` → `Some("b")`; `§16.24 item 18` → `None`; `§8.3 step 4` → `None`.
///
/// **Deliberately NOT folded into `parse_anchor`, and the cost of doing so was MEASURED, not
/// assumed.** `parse_anchor` is also called by `prose_claims` (Layer B's scanner over the whole
/// ~5 500-line file), by `target_span`, and by
/// `every_pinned_pre_convention_target_still_resolves`. Layer B's pinned 28-row set was measured
/// with every lettered prose citation collapsed to a bare `item N`. Teaching the same letter rule to
/// `parse_anchor`'s `label` and re-running this suite (2026-07-30, on this spec) turns
/// `prose_form_claims_match_the_recorded_pre_convention_set` red with **4** rows relabelled:
/// `("16.21", "16.20 item 3")`, `("16.22", "16.20 item 3")`, `("16.24", "16.20 item 3")` and
/// `("16.26", "16.20 item 3")` become `... item 3(a)` / `... item 3(e)`, so all four pin nothing and
/// four NEW claims appear in their place. That is the audit the narrow scoping avoids. The letter is
/// therefore read here, by a function only `marker_claims` calls.
fn marker_sub_item(text: &str) -> Option<String> {
    let rest = text.strip_prefix('§')?;
    let section_len = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .count();
    // ASCII-only prefix, so char count is the byte offset.
    let tail = rest[section_len..].strip_prefix(" item ")?;
    let digits = tail.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let inner = tail[digits..].strip_prefix('(')?;
    let letters: String = inner
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect();
    if letters.is_empty() {
        return None;
    }
    inner[letters.len()..].starts_with(')').then_some(letters)
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
            let anchor_text = &declaration[offset..];
            if let Some(target) = parse_anchor(anchor_text) {
                let bare = target.label.trim_start_matches('§').to_owned();
                let identity = match marker_sub_item(anchor_text) {
                    Some(letters) => format!("{bare}({letters})"),
                    None => bare,
                };
                claims.push(Claim {
                    host: host.clone(),
                    line: index + 1,
                    target,
                    identity,
                });
            }
        }
    }
    claims
}

/// Multiset tally of `(host, target)` pairs, keyed by count rather than mere presence. Used by
/// `the_ratified_supersessions_are_each_covered` (and its synthetic control) so that a target with
/// N real markers is distinguished from one with N-1 markers plus an unrelated duplicate elsewhere
/// — a distinction a `BTreeSet` cannot make.
fn tally(items: impl Iterator<Item = (String, String)>) -> BTreeMap<(String, String), usize> {
    let mut counts = BTreeMap::new();
    for item in items {
        *counts.entry(item).or_insert(0usize) += 1;
    }
    counts
}

/// Diff two multiset tallies. Returns `(missing, extra)` messages: `missing` for a key whose
/// expected count exceeds its found count (a marker disappeared, even if some other marker for the
/// same target remains), `extra` for the reverse (an unrecorded or duplicated marker).
fn tally_diff(
    found: &BTreeMap<(String, String), usize>,
    expected: &BTreeMap<(String, String), usize>,
) -> (Vec<String>, Vec<String>) {
    let mut missing = Vec::new();
    let mut extra = Vec::new();
    let keys: BTreeSet<&(String, String)> = found.keys().chain(expected.keys()).collect();
    for key in keys {
        let found_count = found.get(key).copied().unwrap_or(0);
        let expected_count = expected.get(key).copied().unwrap_or(0);
        if expected_count > found_count {
            missing.push(format!(
                "{key:?}: expected {expected_count} marker(s), found {found_count}"
            ));
        } else if found_count > expected_count {
            extra.push(format!(
                "{key:?}: expected {expected_count} marker(s), found {found_count}"
            ));
        }
    }
    (missing, extra)
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

fn sub_item_marker(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('(')?;
    let letters_len = rest
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .map(char::len_utf8)
        .sum();
    if letters_len == 0 || !rest[letters_len..].starts_with(')') {
        return None;
    }
    Some(&rest[..letters_len])
}

/// Resolve a claim's target span. Historical lettered claims keep §16.26 item 3(a)'s parent-item
/// leniency unless their identity is explicitly pinned in `SUB_ITEM_SCOPED_BACKPOINTERS`.
fn claim_target_span(lines: &[&str], outline: &Outline, claim: &Claim) -> Option<(usize, usize)> {
    let parent = target_span(lines, outline, &claim.target)?;
    if !SUB_ITEM_SCOPED_BACKPOINTERS.contains(&(claim.host.as_str(), claim.identity.as_str())) {
        return Some(parent);
    }

    let letter = claim.identity.strip_suffix(')')?.rsplit_once('(')?.1;
    let start =
        (parent.0..parent.1).find(|index| sub_item_marker(lines[*index]) == Some(letter))?;
    let end = ((start + 1)..parent.1)
        .find(|index| sub_item_marker(lines[*index]).is_some())
        .unwrap_or(parent.1);
    Some((start, end))
}

fn has_pinned_supersession_annotation(line: &str, host: &str) -> bool {
    let line = line.trim_start();
    for prefix in ["**SUPERSEDED by §", "**SUPERSEDED IN PART by §"] {
        let Some(after_prefix) = line.strip_prefix(prefix) else {
            continue;
        };
        let Some(after_host) = after_prefix.strip_prefix(host) else {
            continue;
        };
        // After the host, accept only end-of-line or the exact live separator ` —` (ASCII space
        // followed by em dash). Tabs, bare spaces, and every other continuation are malformed.
        let allowed_host_boundary = after_host.is_empty() || after_host.starts_with(" —");
        if allowed_host_boundary {
            return true;
        }
    }
    false
}

fn contains_claim_back_pointer(lines: &[&str], span: (usize, usize), claim: &Claim) -> bool {
    if SUB_ITEM_SCOPED_BACKPOINTERS.contains(&(claim.host.as_str(), claim.identity.as_str())) {
        // For independently pinned sibling claims, a mere citation of the host is insufficient.
        // After indentation, the dedicated line must begin with exactly one of these Markdown forms:
        // `**SUPERSEDED by §<host>` or `**SUPERSEDED IN PART by §<host>`. After the host, accept only
        // end-of-line or the exact live separator ` —`; arbitrary words between the verb and `by`,
        // and every other post-host continuation, are rejected.
        return lines[span.0..span.1]
            .iter()
            .any(|line| has_pinned_supersession_annotation(line, &claim.host));
    }
    contains_back_pointer(lines, span, &claim.host)
}

fn missing_back_pointers(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);
    let claims = marker_claims(&lines, &outline);
    let mut missing = Vec::new();
    for claim in &claims {
        let span = claim_target_span(&lines, &outline, claim).unwrap_or_else(|| {
            panic!(
                "§{} (line {}) claims to supersede {}, which does not resolve to any section, item, or pinned sub-item",
                claim.host, claim.line, claim.identity
            )
        });
        if !contains_claim_back_pointer(&lines, span, claim) {
            missing.push(claim.identity.clone());
        }
    }
    missing
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
// What must break for this to fail differs by claim class. Historical claims fail when their target
// span lacks a citation of the host section. The two `SUB_ITEM_SCOPED_BACKPOINTERS` identities are
// stricter: after indentation, each individual sub-item span must have a line beginning with exactly
// `**SUPERSEDED by §<host>` or `**SUPERSEDED IN PART by §<host>`. After the host, only end-of-line
// or the exact live separator ` —` is accepted. Thus removing the annotation,
// replacing/negating/extending its verb, inserting arbitrary words between the verb and `by`, or
// using any other post-host continuation fails even when another host citation remains. Adding a `SUPERSEDES:` marker without the required target-side form also fails. In both
// classes, a pointer outside the target span does not satisfy the gate, as separately proven by
// `a_back_pointer_outside_the_target_span_does_not_satisfy_the_gate`.
fn every_supersession_marker_has_a_back_pointer_at_its_target() {
    let text = spec();
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);
    let claims = marker_claims(&lines, &outline);

    let mut missing = Vec::new();
    for claim in &claims {
        let span = claim_target_span(&lines, &outline, claim).unwrap_or_else(|| {
            panic!(
                "§{} (line {}) claims to supersede {}, which does not resolve to any section or \
                 item. An unresolvable anchor is a FAILURE, not a skip: a typo'd anchor would \
                 otherwise exempt the claim from its own gate.",
                claim.host, claim.line, claim.target.label
            )
        });
        if !contains_claim_back_pointer(&lines, span, claim) {
            if SUB_ITEM_SCOPED_BACKPOINTERS
                .contains(&(claim.host.as_str(), claim.identity.as_str()))
            {
                missing.push(format!(
                    "\n  {} (spec line {}) is superseded by §{}, but its individual sub-item span \
                     (lines {}-{}) lacks a dedicated annotation beginning with exactly \
                     `**SUPERSEDED by §{}` or `**SUPERSEDED IN PART by §{}`.\n    \
                     Fix: restore one of those two forms, followed only by end-of-line or the exact \
                     separator ` —`.\n    Why: another §{} citation or any other verb/post-host \
                     syntax does not identify this target annotation.",
                    claim.identity,
                    claim.line,
                    claim.host,
                    span.0 + 1,
                    span.1,
                    claim.host,
                    claim.host,
                    claim.host,
                ));
            } else {
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
// Every ratified supersession site, asserted INDIVIDUALLY rather than by count, so deleting any one
// marker fails with that site in the message. Both directions: the multiset must match exactly, so
// an extra unrecorded marker also fails.
//
// Keyed on `claim.identity` — the label PLUS the sub-item letter the marker declares — not on the
// resolved span label. §16.27's three markers against §16.24 item 18 back three different errata
// (18(b), 18(i), 18(k)); all three resolve to item 18's whole span by ratified §16.26 item 3(a),
// so keying on the span label made them one indistinguishable row and let a deleted marker be
// replaced by a duplicate of a sibling. See RATIFIED_SUPERSESSIONS' comment.
//
// Still compared as a MULTISET (via `tally`/`tally_diff`) rather than a `BTreeSet`, so two markers
// that legitimately share one identity in future cannot silently become one row.
//
// Renamed from `the_three_ratified_supersessions_are_covered` when §16.27 took the constant from 3
// rows to 12: a name pinning a cardinality the constant no longer has is cookbook rule 1's dominant
// defect class, and this is the file whose job is to prevent it.
fn the_ratified_supersessions_are_each_covered() {
    let text = spec();
    let lines: Vec<&str> = text.lines().collect();
    let outline = outline(&lines);

    let found = tally(
        marker_claims(&lines, &outline)
            .into_iter()
            .map(|claim| (claim.host, claim.identity)),
    );
    let expected = tally(
        RATIFIED_SUPERSESSIONS
            .iter()
            .map(|(host, target)| ((*host).to_owned(), (*target).to_owned())),
    );

    let (missing, extra) = tally_diff(&found, &expected);
    assert!(
        missing.is_empty() && extra.is_empty(),
        "the marker multiset changed. Rows are keyed by declared identity, so a sub-item letter is \
         part of the key: losing §16.27's 18(i) marker is reported as `16.24 item 18(i)` missing \
         even while 18(b) and 18(k) remain.\n  \
         missing (a ratified supersession lost a marker, even if other markers for a neighbouring \
         sub-item remain): {missing:?}\n  \
         extra (an unrecorded or duplicated marker — add it to RATIFIED_SUPERSESSIONS \
         deliberately): {extra:?}"
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
fn section_16_40s_two_item_25_back_pointers_are_independently_load_bearing() {
    let text = spec();
    let annotations = [
        (
            "16.38 item 25(a)",
            "    **SUPERSEDED IN PART by §16.40 — this back-pointer qualifies the former OPEN, DEFERRED status and the former omission of strip tests; the measurement above remains unchanged.**\n",
        ),
        (
            "16.38 item 25(d)",
            "    **SUPERSEDED IN PART by §16.40 — this back-pointer resolves the OPEN provider-location choice; the measurements and dependency-edge analysis above remain unchanged.**\n",
        ),
    ];

    assert!(
        missing_back_pointers(&text).is_empty(),
        "the unmodified live spec must satisfy every target-side back-pointer"
    );
    for (identity, annotation) in annotations {
        assert_eq!(
            text.matches(annotation).count(),
            1,
            "the mutation must identify exactly one dedicated annotation for {identity}"
        );
        let deleted = text.replacen(annotation, "", 1);
        assert_ne!(
            deleted, text,
            "the {identity} deletion must change the spec"
        );
        assert_eq!(
            missing_back_pointers(&deleted),
            vec![identity.to_owned()],
            "deleting only {identity}'s annotation must fail that identity even while the sibling and other §16.40 mentions remain"
        );

        for (label, replacement_prefix) in [
            ("replaced verb", "**QUALIFIED"),
            ("negated verb", "NOT SUPERSEDED"),
            ("token-extended verb", "**SUPERSEDEDNESS"),
            ("post-verb negation", "**SUPERSEDED NOT"),
            ("altered IN PART phrase", "**SUPERSEDED IN NO PART"),
        ] {
            let rejected_annotation =
                annotation.replacen("**SUPERSEDED IN PART", replacement_prefix, 1);
            assert_ne!(
                rejected_annotation, annotation,
                "the {identity} {label} mutation must alter the annotation"
            );
            let mutated = text.replacen(annotation, &rejected_annotation, 1);
            assert_ne!(mutated, text, "the {identity} {label} mutation must land");
            assert!(
                mutated.contains(&rejected_annotation),
                "the {identity} {label} mutation must leave its rejected form in the document"
            );
            assert_eq!(
                missing_back_pointers(&mutated),
                vec![identity.to_owned()],
                "the {identity} {label} form must fail only that identity while its host citation, sibling annotation, and other §16.40 citations remain"
            );
        }
        for (label, replacement_host) in [
            ("alphabetic host extension", "§16.40x"),
            ("hyphen host extension", "§16.40-extra"),
        ] {
            let rejected_annotation = annotation.replacen("§16.40", replacement_host, 1);
            assert_ne!(
                rejected_annotation, annotation,
                "the {identity} {label} mutation must alter the annotation"
            );
            let mutated = text.replacen(annotation, &rejected_annotation, 1);
            assert_ne!(mutated, text, "the {identity} {label} mutation must land");
            assert!(
                mutated.contains(&rejected_annotation),
                "the {identity} {label} mutation must leave its rejected form in the document"
            );
            assert_eq!(
                missing_back_pointers(&mutated),
                vec![identity.to_owned()],
                "the {identity} {label} form must fail only that identity while its sibling annotation and other §16.40 citations remain"
            );
        }

        let misleading_suffix = annotation.replacen("§16.40 —", "§16.40 NOT superseded. —", 1);
        assert_ne!(
            misleading_suffix, annotation,
            "the {identity} post-host prose mutation must alter the annotation"
        );
        let mutated = text.replacen(annotation, &misleading_suffix, 1);
        assert_ne!(
            mutated, text,
            "the {identity} post-host prose mutation must land"
        );
        assert!(
            mutated.contains(&misleading_suffix),
            "the {identity} post-host prose mutation must remain in the document"
        );
        assert_eq!(
            missing_back_pointers(&mutated),
            vec![identity.to_owned()],
            "post-host ` NOT superseded.` must fail only {identity} while its sibling annotation and other §16.40 citations remain"
        );
    }
}

#[test]
fn pinned_annotation_accepts_only_the_two_declared_markdown_forms() {
    for accepted in [
        "    **SUPERSEDED by §16.40",
        "    **SUPERSEDED IN PART by §16.40",
        "    **SUPERSEDED by §16.40 — annotation.**",
        "    **SUPERSEDED IN PART by §16.40 — annotation.**",
    ] {
        assert!(has_pinned_supersession_annotation(accepted, "16.40"));
    }
    for rejected in [
        "    NOT SUPERSEDED by §16.40 — annotation.",
        "    **SUPERSEDEDNESS by §16.40 — annotation.**",
        "    **QUALIFIED by §16.40 — annotation.**",
        "    **SUPERSEDED NOT by §16.40 — annotation.**",
        "    **SUPERSEDED IN NO PART by §16.40 — annotation.**",
        "    **SUPERSEDED by §16.400 — annotation.**",
        "    **SUPERSEDED IN PART by §16.40.1 — annotation.**",
        "    **SUPERSEDED by §16.40x — annotation.**",
        "    **SUPERSEDED IN PART by §16.40-extra — annotation.**",
        "    **SUPERSEDED by §16.40 NOT superseded.",
        "    **SUPERSEDED by §16.40 plain space",
        "    **SUPERSEDED IN PART by §16.40\tannotation.",
    ] {
        assert!(
            !has_pinned_supersession_annotation(rejected, "16.40"),
            "rejected pinned annotation form passed: {rejected}"
        );
    }
}

#[test]
fn pinned_sibling_sub_items_use_distinct_spans_and_require_annotations() {
    let text = "\
## 16.38 Older

25. Parent cites §16.40, which must satisfy neither sibling by itself.

    (a) A cites §16.40 in prose.

    **SUPERSEDED IN PART by §16.40 — dedicated A annotation.**

    (b) Untouched.

    (d) D has no dedicated annotation, though it cites §16.40.

## 16.40 Newer

1. **SUPERSEDES: §16.38 item 25(a); §16.38 item 25(d)** — two claims.
";
    assert_eq!(
        missing_back_pointers(text),
        vec!["16.38 item 25(d)".to_owned()],
        "a parent citation, sibling annotation, and same-sub-item prose citation must not satisfy 25(d)"
    );
}

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

#[test]
// Regression control for the multiset fix in `the_ratified_supersessions_are_each_covered`: proves
// that a `BTreeSet` comparison of ratified markers can miss "delete the marker backing one erratum
// on a multi-marker target, and separately add an unrelated duplicate marker re-targeting a target
// already in the set" — while a multiset (`tally`/`tally_diff`) comparison catches it. This is the
// exact class of gap fixed by comparing RATIFIED_SUPERSESSIONS as a multiset instead of a set.
fn a_multiset_diff_catches_a_moved_marker_that_a_set_diff_would_miss() {
    // "Before": three markers target §16.24 item 18 (standing in for 18(b)/18(i)/18(k)), one
    // targets §16.24 item 20.
    let before = "\
## 16.24 Older section

1. First item.

18. Eighteenth item, three sub-errata correct it.

20. Twentieth item.

## 16.27 Newer section

1. **SUPERSEDES: §16.24 item 18** -- erratum for 18(b).

2. **SUPERSEDES: §16.24 item 18** -- erratum for 18(i).

3. **SUPERSEDES: §16.24 item 18** -- erratum for 18(k).

4. **SUPERSEDES: §16.24 item 20** -- an unrelated erratum.
";
    // "After": 18(i)'s marker was deleted (only 2 of the 3 markers on item 18 remain), and a
    // duplicate marker was added re-targeting item 20 (already in the set) instead. The set of
    // DISTINCT `(host, target)` pairs is unchanged either way (`{item 18, item 20}`), but the true
    // per-target count moved -- exactly the swap a `BTreeSet` comparison cannot see.
    let after = "\
## 16.24 Older section

1. First item.

18. Eighteenth item, three sub-errata correct it.

20. Twentieth item.

## 16.27 Newer section

1. **SUPERSEDES: §16.24 item 18** -- erratum for 18(b).

2. **SUPERSEDES: §16.24 item 18** -- erratum for 18(k).

3. **SUPERSEDES: §16.24 item 20** -- an unrelated erratum.

4. **SUPERSEDES: §16.24 item 20** -- a duplicate, standing in for 18(i)'s deleted marker.
";

    fn tally_of(text: &str) -> BTreeMap<(String, String), usize> {
        let lines: Vec<&str> = text.lines().collect();
        let outline = outline(&lines);
        tally(marker_claims(&lines, &outline).into_iter().map(|claim| {
            (
                claim.host,
                claim.target.label.trim_start_matches('§').to_owned(),
            )
        }))
    }

    let expected_targets = [
        ("16.27".to_owned(), "16.24 item 18".to_owned()),
        ("16.27".to_owned(), "16.24 item 18".to_owned()),
        ("16.27".to_owned(), "16.24 item 18".to_owned()),
        ("16.27".to_owned(), "16.24 item 20".to_owned()),
    ];
    let expected_tally = tally(expected_targets.iter().cloned());
    let expected_set: BTreeSet<(String, String)> = expected_targets.iter().cloned().collect();

    let before_found = tally_of(before);
    let after_found = tally_of(after);

    // The bug this test guards against: a SET comparison cannot distinguish "before" from "after"
    // -- both have the exact same distinct pairs -- so it would silently pass on both.
    let before_set: BTreeSet<(String, String)> = before_found.keys().cloned().collect();
    let after_set: BTreeSet<(String, String)> = after_found.keys().cloned().collect();
    assert_eq!(
        before_set, expected_set,
        "sanity: 'before' matches on distinct pairs"
    );
    assert_eq!(
        after_set, expected_set,
        "sanity: the set of distinct pairs is unchanged by the move -- this is precisely why a \
         BTreeSet comparison would miss it"
    );

    // The multiset comparison, however, must tell them apart.
    let (before_missing, before_extra) = tally_diff(&before_found, &expected_tally);
    assert!(
        before_missing.is_empty() && before_extra.is_empty(),
        "'before' has the exact expected multiset and must pass: missing={before_missing:?} \
         extra={before_extra:?}"
    );

    let (after_missing, after_extra) = tally_diff(&after_found, &expected_tally);
    assert!(
        !after_missing.is_empty() || !after_extra.is_empty(),
        "'after' deleted 18(i)'s marker and added an unrelated duplicate elsewhere; the multiset \
         diff must catch this even though the set of distinct pairs did not change"
    );
}

#[test]
// The gap one level deeper than the test above, and the reason `Claim::identity` exists. The
// multiset fix catches "3 markers became 2". It does NOT catch "delete the marker backing 18(i) and
// add back a DUPLICATE marker declaring the same target as 18(b)'s" — because while all three
// markers read `§16.24 item 18` with no letter, they were the same key: raw count still 3, multiset
// row still `3 found / 3 expected`, suite still green, 18(i)'s coverage gone.
//
// Driven on synthetic text, so it tests the keying rule rather than today's spec, and the expected
// identities are typed out as literals — never read back from the parser, which would be cookbook
// rule 13's vacuous gate. All three assertion blocks are needed: block 2 shows the two pre-existing
// checks are blind to this swap, block 3 shows the identity-keyed diff is not, and block 1 shows the
// unmutated text passes (so block 3's failure is caused by the mutation, not by the fixture).
fn a_deleted_sub_item_marker_padded_by_a_duplicate_sibling_fails_the_identity_diff() {
    // Three markers, one per erratum, each declaring its own sub-item letter.
    let honest = "\
## 16.24 Older section

18. Eighteenth item; three of its sub-items are corrected separately.

20. Twentieth item.

## 16.27 Newer section

4. **SUPERSEDES: §16.24 item 18(b)** -- erratum 1 of 6, leg-2 scope.

6. **SUPERSEDES: §16.24 item 18(i)** -- erratum 3 of 6, derivation-conditional message.

8. **SUPERSEDES: §16.24 item 18(k)** -- erratum 5 of 6, the rare-but-real conflation.
";
    // The attack: 18(i)'s marker is gone, and a second 18(b) marker pads the count back to three.
    // Nothing new is corrected; the letter is the only thing that distinguishes this from `honest`.
    let padded = "\
## 16.24 Older section

18. Eighteenth item; three of its sub-items are corrected separately.

20. Twentieth item.

## 16.27 Newer section

4. **SUPERSEDES: §16.24 item 18(b)** -- erratum 1 of 6, leg-2 scope.

6. **SUPERSEDES: §16.24 item 18(b)** -- padding, declaring a target already covered.

8. **SUPERSEDES: §16.24 item 18(k)** -- erratum 5 of 6, the rare-but-real conflation.
";

    fn claims_of(text: &str) -> Vec<Claim> {
        let lines: Vec<&str> = text.lines().collect();
        let outline = outline(&lines);
        marker_claims(&lines, &outline)
    }
    /// Keyed the way the gate is keyed now: identity, letter included.
    fn identities(text: &str) -> BTreeMap<(String, String), usize> {
        tally(
            claims_of(text)
                .into_iter()
                .map(|claim| (claim.host, claim.identity)),
        )
    }
    /// Keyed the way the gate was keyed before: the resolved span label, letter discarded.
    fn labels(text: &str) -> BTreeMap<(String, String), usize> {
        tally(claims_of(text).into_iter().map(|claim| {
            (
                claim.host,
                claim.target.label.trim_start_matches('§').to_owned(),
            )
        }))
    }

    // Hard-coded oracle. These three strings are written here, not computed from either fixture.
    let expected = tally(
        [
            ("16.27".to_owned(), "16.24 item 18(b)".to_owned()),
            ("16.27".to_owned(), "16.24 item 18(i)".to_owned()),
            ("16.27".to_owned(), "16.24 item 18(k)".to_owned()),
        ]
        .into_iter(),
    );

    // 1. The honest text matches the ratified identities exactly, both directions.
    let (missing, extra) = tally_diff(&identities(honest), &expected);
    assert!(
        missing.is_empty() && extra.is_empty(),
        "the three lettered markers must match the recorded identities exactly: \
         missing={missing:?} extra={extra:?}"
    );

    // 2. Both pre-existing checks are blind to the swap — which is why point 3 is not redundant.
    assert_eq!(
        claims_of(honest).len(),
        3,
        "fixture sanity: three markers parsed"
    );
    assert_eq!(
        claims_of(padded).len(),
        3,
        "the raw marker count is unchanged by the swap, so the pinned-count test cannot catch it"
    );
    assert_eq!(
        labels(honest),
        labels(padded),
        "keyed on the resolved span label, the two fixtures are IDENTICAL (both are \
         `16.24 item 18` x3) — this is precisely the multiset gap the identity keying closes"
    );

    // 3. The identity-keyed diff catches it, and names which sub-item lost its marker and which
    //    gained a duplicate. Naming the erratum is the point: "some marker moved" is not actionable.
    let (missing, extra) = tally_diff(&identities(padded), &expected);
    assert_eq!(
        missing.len(),
        1,
        "exactly one identity should be missing, got {missing:?}"
    );
    assert!(
        missing[0].contains("16.24 item 18(i)"),
        "the failure must name 18(i) as the erratum whose marker vanished, got {missing:?}"
    );
    assert_eq!(
        extra.len(),
        1,
        "exactly one identity should be over-represented, got {extra:?}"
    );
    assert!(
        extra[0].contains("16.24 item 18(b)"),
        "the failure must name 18(b) as the duplicated identity, got {extra:?}"
    );
}
