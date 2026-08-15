//! A4-d site gate for the ratified Annotation claim cleanup.
//!
//! Every row is a named `(path, line, substring)` site from §16.39 items 5(c),
//! 5(d), 5(d.i), or 5(g).  The line coordinate is intentional: a file-wide
//! `contains()` check can pass after a claim moves to an unrelated paragraph.

use pc_testkit::paths;
use std::fs;

#[derive(Clone, Copy)]
struct Site {
    path: &'static str,
    line: usize,
    text: &'static str,
}

// 2026-08-10: dropped one ABSENT and one PRESENT site each, both at `docs/HANDOVER.md:39`.
// That file was self-marked "TEMPORARY, delete when consumed" and was deleted once its
// content was fully consumed into `docs/WORKSTATE.md` (see WORKSTATE's "Known stale
// sources" entry and CLAUDE.md's removed `## Handover` section) -- the site this gate
// pinned no longer exists, not because the claim it recorded became false again.
// Then README.md:188→189 (2026-08-10, same author): the final review's byte-total fix
// rewrapped the LaMa sentence in "Opting out" from two lines to three, one line above
// the `### v2` heading. The other three README rows sit above that point and did not
// move. Re-derived by searching for the heading text.
// Then +2 again (2026-08-10, same author): D3's review added the missing §14 register
// entry for `DEVIATION(30)`, two lines at `docs/PIPELINE_SPEC_V1.md:1438`. Only the rows
// BELOW that point moved (501 and 701 did not); each was re-derived by searching for its
// own text, and the two ABSENT-only rows by exact whole-line match against `git show HEAD`.
// 1450→1452, 1485→1487, 1537→1539, 2821→2823, 3822→3824, 3831→3833, 3834→3836,
// 3852→3854, 3871→3873, 3909→3911, 3916→3918, 9278→9280.
// 2026-08-10, Senior Rust Engineer, MECHANICAL RE-PIN ONLY — no row added, removed or
// re-worded, and neither count below changes. Transcribing §16.46 (the shipped-defaults
// flip) inserted a new section plus ten supersession back-pointers into
// `docs/PIPELINE_SPEC_V1.md`, moving 19 of this gate's rows: all 7 spec rows in ABSENT and
// all 12 in PRESENT. Every one was re-derived by SEARCHING FOR ITS OWN TEXT in the new
// file, never by adding a delta — the two ABSENT-only rows that pin text which no longer
// occurs (`3821`, `3899`) were re-derived instead by an exact whole-line match against
// `git show HEAD:docs/PIPELINE_SPEC_V1.md`, each returning exactly one hit. Old → new:
// 499→501, 699→701, 1442→1450, 1475→1485, 1527→1537, 2811→2821, 3812→3822, 3821→3831,
// 3824→3834, 3842→3852, 3861→3871, 3899→3909, 3906→3916, 9106→9274.
// Then 9276→9278 (2026-08-10, same author): D1's independent review ruled an amendment
// into §16.46 item 8(e), adding two more lines above this point. Re-derived by searching
// for the row's own text, as above.
// Then 9274→9276 (2026-08-10, same author, same session): the step-1a fresh-reader pass
// returned four blocking corrections to §16.46's prose, whose fixes added two net lines
// above this point. Re-derived by searching for the row's own text, as above.
//
// **CORRECTED 2026-08-10 after D2's independent review: the paragraph that stood here said
// "no row added, removed or re-worded", and by then that was false.** The MECHANICAL RE-PIN
// note above describes the transcription pass accurately; D2 then went further, because
// flipping the shipped defaults FALSIFIED three rows' pinned text at their own sites. Those
// three were re-worded, each carrying its own inline note:
//
//   * `crates/pc-config/src/default_profile.toml:20` — "Annotation is opt-in; Simple remains
//     the default" → "Annotation is the shipped default; `simple` is the opt-out"
//   * `crates/pc-config/src/profile.rs:208` — "Simple remains the shipped default" →
//     "as superseded by §16.46 items 1(a), 2 and 4"
//   * `crates/pc-config/src/profile.rs:214→217` — "DEVIATION(12): the shipped default is
//     `Simple`" → "`DEVIATION(12)` is RETIRED here (§16.46 item 2)"
//
// They were pulled forward out of D4 because D2's own edit is what made the old text false;
// leaving them would have shipped a gate that was green over a claim the tree contradicts.
// Neither count below changed — three rows moved text, none was added or removed.
//
// **What is still pinned-and-false, and therefore still D4's:** the two spec-prose rows at
// `docs/PIPELINE_SPEC_V1.md:501` ("Annotation is opt-in; Simple remains the default", in §6's
// shipped-default TOML block, which §16.46 item 5 requires rewriting) and `:2821`
// ("DEVIATION(12)` on the default `Simple` variant of `MaskRefineMode`"). Until D4 lands, a
// green run over THOSE rows means the pinned text is where it was pinned — not that it is
// true. The two claims this paragraph previously named as still-false are NOT among them:
// both were the re-worded rows above, and one of them no longer exists in this file at all.
//
// ============================================================================================
// **D4, 2026-08-10, Senior Rust Engineer (fresh spawn; §16.46 item 15's claim-site sweep).**
// Every line number below was re-derived by SEARCHING FOR THE ROW'S OWN TEXT in the current
// tree — a script that, per row, listed every line containing the pinned substring and
// compared it against the declared coordinate. No row's number was obtained by adding a delta.
// The pre-edit run of that script reported 31/31 ABSENT rows satisfied and 29/33 PRESENT rows
// satisfied; the four PRESENT rows it reported unsatisfied are exactly the four this pass
// moves, and each moved for a stated reason rather than because a line drifted:
//
//   * `docs/PIPELINE_SPEC_V1.md:501` — TEXT CHANGED, not re-pinned. §16.46 item 5 required §6's
//     shipped-default TOML line rewritten; D4 did it, so the old pinned string is gone from the
//     file except inside item 5's own verbatim quote at `:9167`. The row now pins the NEW
//     wording, and a paired ABSENT row (added below) pins the OLD wording's departure from that
//     same line, so the rewrite is gated in both directions instead of only forward.
//   * `crates/pc-detect/src/mask.rs:172 → :173` — the `refine_simple` doc comment gained a
//     sentence recording §16.46 item 2's retirement of `DEVIATION(12)`; the pinned phrase
//     ("default-value divergence from") survives verbatim, one line lower. Re-derived by
//     content: exactly one hit, at 173.
//   * `README.md:173 → :188` — the README was reframed around opting OUT (§16.46 item 15), which
//     added net 15 lines above the unchanged `### v2` heading. Re-derived by content: exactly
//     one hit, at 188.
//   * `docs/DETECTOR_ORACLE.md:52` — TEXT CHANGED, not re-pinned. That row's `grep` transcript
//     described `DEVIATION(12)` as sitting "at the default `Simple` variant"; `Simple` is no
//     longer the default variant, so the sentence was corrected in place and re-measured. The
//     row pins the corrected wording and a paired ABSENT row (added below) pins the old
//     wording's departure.
//
// **Five rows ADDED, so both counts move — +4 ABSENT, +1 PRESENT (31→35, 33→34).** Four of the
// five exist because D4's own prose corrections were otherwise ungated: a PRESENT row proves
// the new sentence is at the line, and nothing proved the OLD sentence had left it. Cardinality
// is not identity, so each addition names its own text at its own line:
//
//   * ABSENT `docs/PIPELINE_SPEC_V1.md:501` — "Annotation is opt-in; Simple remains the default".
//   * ABSENT `crates/pc-config/src/profile.rs:208` — "Simple remains the shipped default". This
//     is D2's independent review finding LOW-4: the PRESENT row at that same line is
//     citation-only ("as superseded by §16.46 items 1(a), 2 and 4"), so it guards a
//     cross-reference and not the original claim. Note the wording is deliberately NOT "the
//     shipped default is `Simple`" — that exact phrase occurs LEGITIMATELY at `:218`, inside the
//     `DEVIATION(12)`-retirement doc's verbatim quote of what it is retiring, and an ABSENT row
//     must not forbid a correctly-scoped quotation.
//   * ABSENT `docs/DETECTOR_ORACLE.md:52` — "at the default `Simple` variant".
//   * ABSENT + PRESENT `README.md:76` — the section heading. Before D4, NOTHING in this gate
//     pinned any of README's default-related prose except the `### v2` roadmap heading, which is
//     a position marker and asserts nothing about defaults. This pair gates the one edit a user
//     actually reads.
//
// **For an ABSENT row the gate checks only that THIS line lacks the text**, never that the text
// occurs nowhere else, and three of the four additions rely on that: "Annotation is opt-in;
// Simple remains the default" still occurs at `docs/PIPELINE_SPEC_V1.md:9167` (§16.46 item 5's
// verbatim quote of what it superseded) and must, since deleting a supersession's quoted subject
// is how scope gets lost. That is expected, not a defect, and is stated here rather than left for
// a later reader to rediscover as one.
//
// **ESCALATED, NOT FIXED, and deliberately left pinned-and-false: `docs/PIPELINE_SPEC_V1.md:2823`**
// ("DEVIATION(12)` on the default `Simple` variant of `MaskRefineMode`"). The paragraph above
// assigns it to D4. D4 declines to edit it, and says why rather than quietly leaving it: that
// sentence is §16.14 item 5's dated record of a comment-only change MADE on 2026-07-28, when
// `Simple` was in fact the default variant, so it is a true historical statement of a past act
// rather than a live claim. §16.46's supersession target list does not include §16.14 item 5
// (see `spec_supersession.rs`'s ten `("16.46", …)` rows), so marking it here would be an
// implementer inventing a ratification, which CLAUDE.md step 1a forbids outright. The correct
// exit is a ratified entry naming that target; D4 reports it to the Orchestrator instead of
// acting. Until then the row stays green over text that reads as false out of context — stated
// plainly so nobody later mistakes the green for a verification of the claim.
// ============================================================================================
const EXPECTED_ABSENT: usize = 35;
const EXPECTED_PRESENT: usize = 34;

const ABSENT: &[Site] = &[
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 501,
        text: "annotation\" rejected in v1",
    },
    Site {
        // Added 2026-08-10 (D4): pairs with the PRESENT row at this same line. §16.46 item 5
        // required §6's shipped-default TOML line rewritten; this is the wording it replaced.
        // Still present at `:9167`, inside item 5's own verbatim quote of what it superseded —
        // which is correct, and harmless here because an ABSENT row checks only THIS line.
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 501,
        text: "Annotation is opt-in; Simple remains the default",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 701,
        text: "only `Simple` is implemented in v1",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1487,
        text: "rejected by `pc-detect`",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1539,
        text: "can reject `MaskRefineMode::Annotation`",
    },
    Site {
        path: "crates/pc-config/src/default_profile.toml",
        line: 20,
        text: "annotation is rejected",
    },
    Site {
        path: "crates/pc-config/src/profile.rs",
        line: 208,
        text: "Only `Simple` is implemented",
    },
    Site {
        // Added 2026-08-10 (D4), from D2's independent review finding LOW-4: the PRESENT row at
        // this same line is citation-only, so nothing asserted the ORIGINAL claim had left. The
        // wording is deliberately not "the shipped default is `Simple`" — that phrase occurs
        // legitimately at `:218` as the verbatim quote of what `DEVIATION(12)`'s retirement is
        // retiring, and an ABSENT row must not forbid a correctly-scoped quotation.
        path: "crates/pc-config/src/profile.rs",
        line: 208,
        text: "Simple remains the shipped default",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 11,
        text: "still rejects",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 14,
        text: "still rejects",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 17,
        text: "still rejects",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 86,
        text: "Rejects `MaskRefineMode::Annotation`",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 90,
        text: "return Err(StageError::InvalidInput",
    },
    Site {
        path: "crates/pc-detect/src/detector.rs",
        line: 29,
        text: "so the stage can reject",
    },
    Site {
        path: "crates/pc-config/tests/defaults.rs",
        line: 113,
        text: "literal-§6-block",
    },
    Site {
        // Re-pinned 2026-08-10 (D2): the three new §16.46 item 13(d) tests and the
        // item 13(a) comment corrections added two lines above this point in
        // `defaults.rs`. Re-derived by searching for the row's own text; the paired
        // ABSENT and PRESENT rows move together so the pairing stays meaningful.
        path: "crates/pc-config/tests/defaults.rs",
        line: 358,
        text: "pc-detect is what rejects",
    },
    Site {
        // Re-pinned 2026-08-10 (D2): the three new §16.46 item 13(d) tests and the
        // item 13(a) comment corrections added two lines above this point in
        // `defaults.rs`. Re-derived by searching for the row's own text; the paired
        // ABSENT and PRESENT rows move together so the pairing stays meaningful.
        path: "crates/pc-config/tests/defaults.rs",
        line: 363,
        text: "pc-detect is what rejects",
    },
    Site {
        path: "crates/pc-detect/tests/d7_run.rs",
        line: 196,
        text: "run_rejects_annotation_refine_mode",
    },
    Site {
        path: "crates/pc-detect/src/mask.rs",
        line: 150,
        text: "DEVIATION(12): v1 ships",
    },
    Site {
        // Re-pinned 2026-08-11 (v1.5 closure + README rewrite, §16.50): the old "in
        // progress" v1.5 checklist this ABSENT row guarded against is gone entirely --
        // Annotation shipped at v1.2 and the checklist item never described v1.5 scope in
        // the current README anyway. Vacuously true at any line lacking the exact text,
        // same discipline this file already documents for other ABSENT rows; pointed at
        // the line that now carries Annotation's real, shipped status for meaningfulness.
        //   → 301 (2026-08-11, `panel-ocr inpaint` standalone subcommand (brief
        //     BRIEF_INPAINT_STANDALONE.md): README section documenting `clean`/`ocr`/
        //     `inpaint` usage gained the new command, +12 net lines above this point,
        //     re-derived by content). Pointed at the line that now carries Annotation's
        //     real, shipped status for meaningfulness.
        //   → 306 (2026-08-11, fix round (brief BRIEF_INPAINT_STANDALONE_FIXES.md): M5/M6/
        //     L10 added six net lines to the `inpaint` paragraph above this point; re-derived
        //     by content, single occurrence confirmed).
        path: "README.md",
        line: 306,
        text: "[ ] `Mask RefineMode::Annotation`",
    },
    Site {
        path: "crates/pc-detect/src/annotate.rs",
        line: 7,
        text: "not wired into anything",
    },
    Site {
        path: "crates/pc-detect/src/annotate_merge.rs",
        line: 14,
        text: "not wired into anything",
    },
    Site {
        path: "crates/pc-detect/src/annotate_refine.rs",
        line: 7,
        text: "not wired into anything",
    },
    Site {
        path: "crates/pc-detect/tests/a1_annotate_topk.rs",
        line: 19,
        text: "is *not* Annotation mode",
    },
    Site {
        path: "crates/pc-detect/tests/a2_annotate_otsu.rs",
        line: 65,
        text: "is *not* Annotation mode",
    },
    Site {
        path: "crates/pc-detect/tests/a3_annotate_merge.rs",
        line: 60,
        text: "is *not* Annotation mode",
    },
    Site {
        path: "crates/pc-detect/src/annotate_merge.rs",
        line: 22,
        text: "DEVIATION(12) retirement",
    },
    Site {
        path: "crates/pc-detect/src/annotate_refine.rs",
        line: 10,
        text: "DEVIATION(12) retirement",
    },
    Site {
        path: "crates/pc-detect/tests/a3_annotate_merge.rs",
        line: 68,
        text: "DEVIATION(12) retirement",
    },
    Site {
        path: "docs/DETECTOR_ORACLE.md",
        line: 52,
        text: "crates/pc-detect/src/mask.rs:150",
    },
    Site {
        // Added 2026-08-10 (D4): pairs with the PRESENT row at this same line. That row's grep
        // transcript called `DEVIATION(12)`'s site "the default `Simple` variant"; §16.46 item
        // 1(a) made `Annotation` the default, so the phrase was corrected and re-measured.
        path: "docs/DETECTOR_ORACLE.md",
        line: 52,
        text: "at the default `Simple` variant",
    },
    Site {
        // Added 2026-08-10 (D4): §16.46 item 15 required README's "opt into Annotation and LaMa"
        // section reframed around opting OUT. Before this pair, the only README row in this gate
        // was the `### v2` heading, which asserts nothing about defaults — so the one artifact an
        // end user actually reads was ungated for exactly the claim D4 exists to correct.
        // Re-pinned 2026-08-11 (v1.5 closure + README rewrite, §16.50): README restructured
        // (new Prerequisites/GPU acceleration/FAQ sections inserted above this point).
        // Content at line 76 changed, but the checked-absent text never occurred anywhere
        // in the file either before or after -- vacuously true either way.
        path: "README.md",
        line: 76,
        text: "### Getting the best cleaning quality",
    },
    Site {
        // Re-pin history (each event distinct, none overwritten):
        //   2779→2804 (2026-08-09, §16.42 spec insertions, task #22 ratification +
        //     its two back-pointers, +25 net lines above this point)
        //   →2811 (2026-08-10, Orchestrator, mechanical — §16.45's two small marker
        //     insertions at §16.10 item 3 and §16.11 item 9, above this point)
        // Content at the new line unchanged from what previously sat at the old line.
        // Note (added after a fresh-reader pass, 2026-08-09): for an ABSENT row the
        // gate only checks THIS line does not contain the text — it cannot verify
        // "occurs nowhere else in the file", so that is not claimed here.
        //   → 2821 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 2823,
        text: "pc-detect's `refine_simple`",
    },
    Site {
        // Re-pinned 2026-08-09: shifted 3798→3823 by the §16.42 spec insertions
        // (task #22 ratification + its two back-pointers, +25 net lines above this
        // point). Corrected 2026-08-09 (a fresh-reader pass caught the same off-by-two
        // this exact site had in §16.39 item 5(f) — see docs/PIPELINE_SPEC_V1.md's own
        // correction there): the intended target is 3821, not 3823 (3798→3821 net,
        // since the pre-§16.42 pin was already 2 lines short of its target). Also note:
        // this exact text string (with the closing backtick immediately before
        // `; and`) doesn't occur anywhere in the current file, so this row is
        // vacuously true regardless of the line number — flagged rather than left
        // silently unfalsifiable. For an ABSENT row the gate only checks THIS line
        // does not contain the text — it cannot verify "occurs nowhere else."
        //   → 3831 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3833,
        text: "`MaskRefineMode::Annotation`; and",
    },
    Site {
        // Re-pinned 2026-08-09: shifted 3874→3899 by the §16.42 spec insertions
        // (task #22 ratification + its two back-pointers, +25 net lines above this
        // point). Content at the new line unchanged from what previously sat at the
        // old line. Note (added after a fresh-reader pass, 2026-08-09): a
        // fresh-reader pass found this exact phrase DOES occur elsewhere in the file
        // (§16.23 item 6's verbatim quote at line 7700, and this entry's own citation
        // of it at §16.39 item 5(d)) — that is expected and harmless for an ABSENT
        // row, since the gate only checks line 3899 specifically, not the whole file;
        // the original comment's "no longer occurs anywhere" claim was wrong and is
        // removed rather than repeated.
        //   → 3909 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        //   → 3913 (2026-08-11, §16.50's back-pointer at §16.23 item 1 inserted above
        //     this point; re-derived by content — the ABSENT check still holds vacuously
        //     at the new line, same as the existing note above documents for prior shifts).
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3913,
        text: "**Annotation** last",
    },
];

const PRESENT: &[Site] = &[
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 11,
        text: "path wired through [`run`]",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 14,
        text: "used by the path wired through [`run`]",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 17,
        text: "used by the path wired through",
    },
    Site {
        path: "crates/pc-detect/src/annotate.rs",
        line: 7,
        text: "supplies Annotation primitives",
    },
    Site {
        path: "crates/pc-detect/src/annotate_merge.rs",
        line: 14,
        text: "supplies Annotation merge primitives",
    },
    Site {
        path: "crates/pc-detect/src/annotate_refine.rs",
        line: 7,
        text: "supplies Annotation page-level refinement",
    },
    Site {
        path: "crates/pc-detect/tests/a1_annotate_topk.rs",
        line: 19,
        text: "A1 covers Annotation primitives",
    },
    Site {
        path: "crates/pc-detect/tests/a2_annotate_otsu.rs",
        line: 65,
        text: "A2 covers Annotation primitives",
    },
    Site {
        path: "crates/pc-detect/tests/a3_annotate_merge.rs",
        line: 60,
        text: "A3 covers Annotation primitives",
    },
    Site {
        // 2026-08-10 (D4): TEXT CHANGED, not re-pinned. §16.46 item 5 required this line's value
        // and its comment rewritten so the two agree; the old wording survives only inside item
        // 5's own verbatim quote at `:9167`, which the paired ABSENT row at this line accounts
        // for. Re-derived by content: exactly one hit for the new text, at 501.
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 501,
        text: "Annotation is the shipped default; `simple` is the opt-out",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 701,
        text: "`Annotation` is the opt-in upstream refinement path",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1487,
        text: "accepted by config and selects the upstream refinement path",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1539,
        text: "stage selects the `MaskRefineMode::Annotation` refinement path",
    },
    Site {
        // 2026-08-10 (D2): §16.46 item 1 flipped the shipped defaults, which FALSIFIED this
        // row's pinned text at its own site. That is a content change, not a line-number
        // re-pin, and it is pulled forward out of D4 because D2's own edit is what made the
        // old text false -- leaving it for D4 would mean shipping a red gate in between.
        // Every OTHER D4 row (README, DETECTOR_ORACLE.md, the spec prose, the code comments
        // elsewhere) is deliberately untouched here.
        path: "crates/pc-config/src/default_profile.toml",
        line: 20,
        text: "Annotation is the shipped default; `simple` is the opt-out",
    },
    Site {
        // 2026-08-10 (D2): §16.46 item 1 flipped the shipped defaults, which FALSIFIED this
        // row's pinned text at its own site. That is a content change, not a line-number
        // re-pin, and it is pulled forward out of D4 because D2's own edit is what made the
        // old text false -- leaving it for D4 would mean shipping a red gate in between.
        // Every OTHER D4 row (README, DETECTOR_ORACLE.md, the spec prose, the code comments
        // elsewhere) is deliberately untouched here.
        path: "crates/pc-config/src/profile.rs",
        line: 208,
        text: "as superseded by §16.46 items 1(a), 2 and 4",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 86,
        text: "Supports the `MaskRefineMode::Annotation` path",
    },
    Site {
        path: "crates/pc-detect/src/detector.rs",
        line: 29,
        text: "because the stage selects the refinement path",
    },
    Site {
        // Re-pinned 2026-08-10 (D2): the three new §16.46 item 13(d) tests and the
        // item 13(a) comment corrections added two lines above this point in
        // `defaults.rs`. Re-derived by searching for the row's own text; the paired
        // ABSENT and PRESENT rows move together so the pairing stays meaningful.
        path: "crates/pc-config/tests/defaults.rs",
        line: 363,
        text: "config accepts annotation as an opt-in refinement mode",
    },
    Site {
        // Re-pinned 172→173 (2026-08-10, D4): `refine_simple`'s doc gained a sentence recording
        // §16.46 item 2's retirement of `DEVIATION(12)`, one line above this phrase. The phrase
        // itself is unchanged. Re-derived by content: exactly one hit, at 173.
        path: "crates/pc-detect/src/mask.rs",
        line: 173,
        text: "default-value divergence from",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1452,
        text: "The door is open: A4 ships `Annotation`",
    },
    Site {
        // Re-pin history: 3780→3805 (2026-08-09, §16.42 spec insertions) →3812
        // (2026-08-10, Orchestrator, mechanical — §16.45's two small marker
        // insertions above this point). Content unchanged, single occurrence verified.
        //   → 3822 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3824,
        text: "§16.39 later ratified and landed that opt-in path",
    },
    Site {
        // Re-pin history: 3792→3817 (2026-08-09, §16.42 spec insertions) →3824
        // (2026-08-10, Orchestrator, mechanical — §16.45's two small marker
        // insertions above this point). Content unchanged, single occurrence verified.
        //   → 3834 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3836,
        text: "The 2026-07-29 plan recorded these as v1.5 scope",
    },
    Site {
        // Re-pin history: 3810→3835 (2026-08-09, §16.42 spec insertions) →3842
        // (2026-08-10, Orchestrator, mechanical — §16.45's two small marker
        // insertions above this point). Content unchanged, single occurrence verified.
        //   → 3852 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        //   → 3856 (2026-08-11, §16.50's back-pointer at §16.23 item 1 inserted above
        //     this point; re-derived by content, single occurrence confirmed).
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3856,
        text: "Historical sequencing (2026-07-29)",
    },
    Site {
        // Re-pin history: 3829→3854 (2026-08-09, §16.42 spec insertions) →3861
        // (2026-08-10, Orchestrator, mechanical — §16.45's two small marker
        // insertions above this point). Content unchanged, single occurrence verified.
        //   → 3871 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        //   → 3875 (2026-08-11, §16.50's back-pointer at §16.23 item 1 inserted above
        //     this point; re-derived by content, single occurrence confirmed).
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3875,
        text: "§16.39 subsequently landed the path",
    },
    Site {
        // Re-pin history: 3874→3899 (2026-08-09, §16.42 spec insertions) →3906
        // (2026-08-10, Orchestrator, mechanical — §16.45's two small marker
        // insertions above this point). Content unchanged, single occurrence verified.
        //   → 3916 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        //   → 3920 (2026-08-11, §16.50's back-pointer at §16.23 item 1 inserted above
        //     this point; re-derived by content, single occurrence confirmed).
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3920,
        text: "historical final step; §16.39 subsequently landed it",
    },
    Site {
        // Re-pin history, each a DISTINCT event (not one claim restated — earlier
        // attributions are preserved here rather than overwritten, since a provenance
        // overwrite is the failure class CLAUDE.md step 1a exists to catch):
        //   7666 (mask-parity's standalone spec, pre-integration)
        //   → 8170, 8363, 8366, 8373 (2026-08-09, joint architect + rust-engineer
        //     ruling, unanimous — shifts during §16.42's own step-1a review cycle)
        //   → 8608 (2026-08-09, Orchestrator, mechanical — the §16.43 mode-bench
        //     ratification inserted directly above "## 16. Summary of what v1 is NOT")
        //   → 8638, 8667, 8803, 8815, 8823 (2026-08-09/10, Orchestrator,
        //     mechanical — §16.43's own step-1a review fix-up passes, then the §16.44
        //     Fable tie-break ratification (2026-08-10) and its own step-1a review
        //     fix-up passes, each inserted/edited directly above "## 16. Summary of
        //     what v1 is NOT", each adding more lines above this point)
        //   → 9027 (2026-08-10, Orchestrator, mechanical — the §16.45 alpha-composite
        //     ratification and its three supersession back-pointers, inserted directly
        //     above "## 16. Summary of what v1 is NOT")
        //   → 9106 (2026-08-10, Orchestrator, mechanical — §16.45's own step-1a
        //     fresh-reader review fix-up pass: one BLOCKING finding (a frozen test's
        //     name/message would have stated the superseded rule) plus several
        //     non-blocking content corrections, all applied above this point)
        //   → 9274 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point)
        //   → 9276 (2026-08-10, same author — the step-1a fresh-reader pass's four
        //     blocking prose corrections to §16.46, two net lines above this point)
        //   → now 9278 (2026-08-10, same author — D1's independent review ruled an
        //     amendment into §16.46 item 8(e), two more lines above this point)
        // Every step above was re-derived by searching for this row's own text, never by
        // adding a delta. Content unchanged, single occurrence verified each time. Given how
        // many times this one site has moved, ALWAYS re-verify with `grep -n
        // "Upstream .refine_mask" docs/PIPELINE_SPEC_V1.md` before trusting this
        // number if this test ever fails again — do not just add/subtract a delta.
        //   → 9376 (2026-08-11, §16.47 GPU-2 ratification inserted above this point;
        //     re-derived by content, single occurrence confirmed).
        //   → 9387 (2026-08-11, same task, fresh-reader fix round added 11 net lines
        //     above this point; re-derived by content, single occurrence confirmed).
        //   → 9393 (2026-08-11, same task, second fresh-reader fix round added 6 more
        //     net lines above this point; re-derived by content, single occurrence
        //     confirmed).
        //   → 9420 (2026-08-11, Orchestrator, mechanical — the §16.48 GPU-3 ratification
        //     (real OCR CUDA divergence measurement, correcting §16.47 item 4's stale
        //     citation) inserted above this point; re-derived by content, single
        //     occurrence confirmed).
        //   → 9422 (2026-08-11, same task, two net lines added rewording §16.48 item 4's
        //     text to dodge a Layer-B prose-claim false positive rather than pin it;
        //     re-derived by content, single occurrence confirmed).
        //   → 9429 (2026-08-11, same task, the step-1a fresh-reader pass's four blocking
        //     findings (B1-B4: a widened citation claim, an overclaimed "mechanism
        //     unaffected", a false "no other assertion changes" claim, an undisclosed
        //     3-part message rewrite) plus five non-blocking corrections fixed, net lines
        //     added above this point; re-derived by content, single occurrence confirmed).
        //   → 9485 (2026-08-11, GPU-4 §16.49 ratification inserted above this point —
        //     the real, measured LaMa CUDA divergence measurement and the deletion of
        //     Stage/ensure_stage_supports entirely, plus five supersession back-pointers
        //     added at the §16.47/§16.48 sites it names; re-derived by content, single
        //     occurrence confirmed).
        //   → 9495 (2026-08-11, same task, the step-1a fresh-reader pass's one blocking
        //     finding (§16.47 item 6's own "Replacement sentence, for all four" quote left
        //     unmarked as superseded, and item 4's enumeration omitted DETECTOR_DEVICE_MECHANISM)
        //     plus six non-blocking corrections fixed, net lines added above this point;
        //     re-derived by content, single occurrence confirmed).
        //   → 9515 (2026-08-11, §16.50 inserted above this point -- the maintainer's direct
        //     decision to defer legacy INI import/Lab NLM/PSD/DBNet lines and close v1.5 on
        //     GPU-2/3/4 alone; re-derived by content, single occurrence confirmed).
        //   → 9517 (2026-08-11, same task -- the `## 16. Summary of what v1 is NOT` heading
        //     was accidentally deleted by the §16.50 edit and restored 2 lines further down;
        //     re-derived by content, single occurrence confirmed).
        //   → 9523 (2026-08-11, same task -- the step-1a fresh-reader pass's two blocking
        //     findings (B1: a truncated §16.23 quote read as narrowing the whole list rather
        //     than four clauses; B2: two more "still v1.5 scope" sites found unmarked) plus
        //     five non-blocking corrections fixed, net lines added above this point;
        //     re-derived by content, single occurrence confirmed).
        //   → 9543 (2026-08-16, §16.51's OCR beam-batching ratification added 20 lines
        //     before this site; re-derived by grep for the exact claim text, single
        //     occurrence confirmed, not incremented blindly).
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 9543,
        text: "Upstream `refine_mask`/`refine_undetected_mask`",
    },
    Site {
        // Re-pinned 103->173 (2026-08-10, README rewritten as end-user guide for the
        // v1.2.0 release; the v2 heading is unchanged content, just moved further down).
        //   → 188 (2026-08-10, D4, §16.46 item 15's README reframing around opting OUT:
        //     +15 net lines above this point).
        //   → 190 (2026-08-11, GPU-1 checklist correction: the README's v1.5 section
        //     falsely claimed "not started" while GPU-1/§16.36 had actually merged
        //     2026-08-05, verified against git — checking the box and adding the
        //     shipped-date note added one net line above this point). Re-derived by
        //     content: exactly one hit.
        //   → 328 (2026-08-11, v1.5 closure + README rewrite, §16.50: new Prerequisites,
        //     GPU acceleration, and FAQ sections inserted above this point; the v2 heading
        //     is unchanged content, just moved further down). Re-derived by content:
        //     exactly one hit.
        //   → 340 (2026-08-11, `panel-ocr inpaint` standalone subcommand (brief
        //     BRIEF_INPAINT_STANDALONE.md): README section documenting `clean`/`ocr`/
        //     `inpaint` usage gained the new command, +12 net lines above this point; the
        //     v2 heading is unchanged content, just moved further down). Re-derived by
        //     content: exactly one hit.
        //   → 345 (2026-08-11, fix round (brief BRIEF_INPAINT_STANDALONE_FIXES.md): M5/M6/
        //     L10 added six net lines to the `inpaint` paragraph above this point; re-derived
        //     by content, single occurrence confirmed).
        path: "README.md",
        line: 345,
        text: "### v2",
    },
    Site {
        // Added 2026-08-10 (D4): pairs with the ABSENT row at this same line, and is the only
        // row in this gate asserting anything about README's default-related PROSE.
        //   → 95 (2026-08-11, v1.5 closure + README rewrite, §16.50: a new Prerequisites
        //     section inserted above this point). Re-derived by content: exactly one hit.
        //   → 107 (2026-08-11, `panel-ocr inpaint` standalone subcommand (brief
        //     BRIEF_INPAINT_STANDALONE.md): the usage section gained the new command, +12 net
        //     lines above this point). Re-derived by content: exactly one hit.
        //   → 112 (2026-08-11, fix round (brief BRIEF_INPAINT_STANDALONE_FIXES.md): M5/M6/
        //     L10 added six net lines to the `inpaint` paragraph above this point; re-derived
        //     by content, single occurrence confirmed).
        path: "README.md",
        line: 112,
        text: "### Opting out of the default refinement and inpainting",
    },
    Site {
        // 2026-08-10 (D4): TEXT CHANGED, not re-pinned. `Simple` is no longer the default
        // variant (§16.46 item 1(a)), so this row's grep transcript was corrected in place and
        // re-measured; the paired ABSENT row at this line pins the old wording's departure.
        path: "docs/DETECTOR_ORACLE.md",
        line: 52,
        text: "DEVIATION(12)` on the `Simple` variant in `crates/pc-config/src/profile.rs`",
    },
    Site {
        // Re-pin history (each event distinct, none overwritten):
        //   2779→2804 (2026-08-09, §16.42 spec insertions, task #22 ratification +
        //     its two back-pointers, +25 net lines above this point)
        //   →2811 (2026-08-10, Orchestrator, mechanical — §16.45's two small marker
        //     insertions at §16.10 item 3 and §16.11 item 9, above this point)
        // Content unchanged, single occurrence verified each time.
        //   → 2821 (2026-08-10, Senior Rust Engineer, mechanical — the §16.46
        //     shipped-defaults ratification plus its ten supersession back-pointers,
        //     inserted above this point). Re-derived by content, not by adding a delta.
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 2823,
        text: "DEVIATION(12)` on the default `Simple` variant of `MaskRefineMode`",
    },
    Site {
        // 2026-08-10 (D2): §16.46 item 1 flipped the shipped defaults, which FALSIFIED this
        // row's pinned text at its own site. That is a content change, not a line-number
        // re-pin, and it is pulled forward out of D4 because D2's own edit is what made the
        // old text false -- leaving it for D4 would mean shipping a red gate in between.
        // Every OTHER D4 row (README, DETECTOR_ORACLE.md, the spec prose, the code comments
        // elsewhere) is deliberately untouched here.
        path: "crates/pc-config/src/profile.rs",
        line: 217,
        text: "`DEVIATION(12)` is RETIRED here (§16.46 item 2)",
    },
    Site {
        path: "crates/pc-detect/src/annotate_merge.rs",
        line: 21,
        text: "DEVIATION(12)` narrowing",
    },
    Site {
        path: "crates/pc-detect/src/annotate_refine.rs",
        line: 10,
        text: "DEVIATION(12)` narrowing",
    },
    Site {
        path: "crates/pc-detect/tests/a3_annotate_merge.rs",
        line: 67,
        text: "DEVIATION(12)` narrowing",
    },
];

fn line(site: Site) -> String {
    let path = paths::workspace_root().join(site.path);
    assert!(
        path.is_file(),
        "A4-d site path does not exist: {}",
        path.display()
    );
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .nth(site.line - 1)
        .unwrap_or("")
        .to_owned()
}

#[test]
fn a4d_claim_sites_match_the_ratified_cleanup() {
    assert_eq!(ABSENT.len(), EXPECTED_ABSENT);
    assert_eq!(PRESENT.len(), EXPECTED_PRESENT);
    for site in ABSENT {
        assert!(
            !line(*site).contains(site.text),
            "stale claim at {}:{}",
            site.path,
            site.line
        );
    }
    for site in PRESENT {
        assert!(
            line(*site).contains(site.text),
            "missing claim at {}:{}",
            site.path,
            site.line
        );
    }
}
