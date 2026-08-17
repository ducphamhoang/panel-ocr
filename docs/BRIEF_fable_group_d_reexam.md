# Fable re-examination — one escalated test's disposition

## Context

You (Fable) previously ruled on the vacuous cross-crate tests
(`docs/BRIEF_fable_vacuous_tests_tiebreak.md`), placing three `pc_export_*` tests in
"Group D: STRIP the now-identity cross-crate loop, KEEP + rename the hand-derived
assertions," architect's position winning over the audit's "duplicates
`composite_value_lock.rs`" claim.

The implementer (a fresh `rust-engineer` dispatch, executing your ruling) applied that
disposition to two of the three Group D members cleanly (`pc_export_blend_channel_...`,
`pc_export_resize_nearest_rgba_...` — both stripped-and-renamed, hand-derived tables
kept) and then stopped on the third, escalating rather than guessing. Quoting their
report:

> "`pc_export_alpha_composite_over_agrees_with_pc_denoise_including_the_offset_clip` is
> left byte-identical. Fable's Group D says 'STRIP the now-identity comparison, KEEP +
> RENAME the hand-derived assertions.' **This test has no hand-derived assertions.** Its
> entire body is the `pc_export` vs `pc_denoise` loop over four offsets (now an identity
> comparison) plus one `assert_ne!(changed, make_base())` control. Stripping as
> instructed would leave a test asserting only 'something changed' — precisely the
> cookbook rule 1 / rule 6 defect this whole task exists to remove.
>
> Note also that Fable's own justification for Group D cites measurements for **only**
> the blend member (2/8 rows) and the resize member (`(3,2)` vs `(3,1)`); it offers none
> for this one, and the **architect's own position had this test in Group C (delete
> outright)**, not Group D. It appears to have been absorbed into Group D's membership
> list without evidence."

This matches what's in `docs/BRIEF_vacuous_tests_architect_position.md`: architect's
"Group C — DELETE (with a blocking precondition)" explicitly lists
`pc_export_alpha_composite_over_agrees_...` alongside the two sole-detector inpaint tests
("measured 0-red under a full §16.45 revert; `composite_value_lock.rs` measured to catch
it — delete outright"), and architect's Group D (strip-and-rename) names only the blend
and resize tests. rust-engineer's own position doc's Q2 keeps all three as "genuinely
redundant, but still KEEP" on cost grounds — never assigns it a strip-and-rename either.

**So: no position anywhere in the record (not the audit, not either Opus side's original
position, not your own stated grounds) actually argues this specific test should be
stripped-and-renamed. Its inclusion in your Group D list appears to be a transcription
slip, carrying over the "three `pc_export_*` tests" framing from the brief without the
one you'd need to re-derive coverage against being different in kind (no hand-derived
half to preserve).**

## What's asked of you

Rule on this one test only — nothing else from the original ruling is reopened:

1. Given the implementer's finding that stripping it would leave only a vacuous
   `assert_ne!`, and that architect's own original evidence placed it in Group C
   (delete, on measured grounds: 0-red under a full §16.45 alpha-formula revert,
   `composite_value_lock.rs` measured to catch the same regression) — does Group C's
   disposition (DELETE outright) govern for this test instead, correcting what looks
   like an oversight in the Group D list?
2. Or is there a synthesis you intend that neither original position captured (e.g.
   rebuild it with a genuine hand-derived table before stripping the cross-crate loop,
   rather than deleting outright)?

Whichever you choose, state which cookbook-8 exit applies (exit 1 additive if rebuilt
with new hand-derived content first, or the "fourth exit in substance" you already
approved elsewhere in this same ruling if deleted outright per Group C).
