# fresh-reader brief — §16.53 transcription review

## What to review

`docs/PIPELINE_SPEC_V1.md:9577` onward — the new `## 16.53` section (ends right before
`## 16. Summary of what v1 is NOT`). This is a ratification transcription of two Fable
rulings (the original vacuous-cross-crate-tests tie-break, plus a narrow follow-up
correction) and the implementation's real measurements. Per this project's step-1a rule,
the *transcription* needs a reader before commit — not only the rulings it records.

## Ground truth to check the transcription against (read these, don't trust the section's
own summary of them)

1. `docs/BRIEF_vacuous_crosscrate_tests_plan.md` — original task framing, §16.42 item 10
   quoted there.
2. `docs/BRIEF_vacuous_tests_architect_position.md` and
   `docs/BRIEF_vacuous_tests_rust_engineer_position.md` — both full joint-planning
   positions, including their real probe matrices.
3. `docs/BRIEF_fable_vacuous_tests_tiebreak.md` — the framing dispatched to Fable, and
   what the two positions agreed/disagreed on.
4. `docs/HANDOVER.md`'s "What just happened" section — the fullest surviving record of
   Fable's original ruling text (received in an earlier session, not saved as a separate
   transcript file).
5. `docs/BRIEF_fable_group_d_reexam.md` — the follow-up correction brief (the escalated
   `pc_export_alpha_composite_over_...` test).
6. `docs/PIPELINE_SPEC_V1.md:8395-8413` — §16.42 item 10's actual text, to check the
   verbatim quote in §16.53's second paragraph is exact, character for character.
7. `crates/pc-pipeline/tests/l1_morph_equivalence.rs` and
   `crates/pc-pipeline/tests/l4_composite_equivalence.rs` — the actual final test files,
   to check every test name, disposition, and count claimed in §16.53 against what is
   really there.
8. `crates/pc-mask/tests/m5_alpha_composite_value_lock.rs` (T1) and
   `crates/pc-testkit/tests/composite_morph_source_sites.rs` (T2) — the two new files, to
   check the gate's actual constants (`EXPECTED_DEFINITION_SITES`,
   `EXPECTED_REEXPORT_SITES`, the 7-row lists) against what §16.53 item 4 claims.
9. `docs/COOKBOOK.md` rule 8 (the "tests are frozen" section) — check the new exit 4 text
   this task added is consistent with what §16.53 item 6 claims to have codified there,
   and that the "exactly three" → "exactly four" count was updated consistently (also
   check `CLAUDE.md` and `.claude/agents/rust-engineer.md`'s references to the exit count
   were updated, or flag if they weren't).

## What to check specifically

- **The §16.42 item 10 quote is character-for-character exact** — this project's own
  transcription rule requires quoting scope verbatim, not paraphrasing.
- **§16.53 explicitly does NOT mark item 10 SUPERSEDED/NARROWED/QUALIFIED** — confirm no
  such marker was accidentally introduced (would trigger `spec_supersession.rs`'s gate
  and require a back-pointer at item 10's site, which this entry deliberately does not
  add). Search for the trigger tokens (`SUPERSEDED`, `NARROWED`, `QUALIFIED`, `ERRATUM`,
  `amended`, `withdrawn`, `RE-GROUNDED`) inside §16.53's own text and confirm none of them
  appear as a supersession-claim usage (the word "amended" or similar appearing in
  ordinary prose, not as a marker, is fine — check the actual gate's parsing rules in
  `crates/pc-testkit/tests/spec_supersession.rs` if unsure what counts).
- **Every test name in items 2, 3, 5, 8 matches a real test that exists (for KEEP/renamed
  tests) or no longer exists (for deletions) in the actual files.**
- **The gate-demonstration matrix in item 7 is presented as what was actually measured**,
  not restated more strongly than the implementation's report supports. Cross-check
  against the implementation agent's own report if available in the conversation
  transcript, or against the test files' own doc comments (both new test files carry
  extensive doc-comment records of what was measured).
- **The "no rebuild" ruling for the escalated test (item 3) is grounded** — check that
  `composite_value_lock.rs` and `m5_alpha_composite_value_lock.rs` really do carry the
  clip-behavior coverage the entry claims, by reading those files directly.
- **Item 9's "what this entry does NOT decide" list is honest** — nothing claimed as
  settled elsewhere in the entry is quietly re-opened by omission, and nothing left open
  is mis-stated as resolved.
- **Numbers**: 8 → 5 dispositions accounted for (test file diffs), the 1498→1499 passing
  count claim, the 6→7 re-export-site correction, the definition-site count of 7. Re-derive
  what's cheaply re-derivable (e.g. actually count the test functions in the two files)
  rather than trusting the prose.

## What NOT to do

Do not re-litigate Fable's rulings themselves, or re-run the mutation probes — that
work is done and is not this review's job. This is specifically a check of whether the
*transcription* accurately and honestly represents what was decided and measured, per
this project's step-1a rule. Report findings; do not edit the spec file yourself.
