# Adjudicator rulings — source record

`PIPELINE_SPEC_V1.md` is normative. This file is not. It exists for one reason: when a ruling is
transcribed into a `§16.x` entry, `CLAUDE.md` step 1a sends a fresh reader at the *transcription* —
and until now that reader had nothing to check the transcription **against**. The first step-1a
review said so explicitly:

> I could not check item 3(d) against the Fable ruling itself. `docs/` holds only
> `ARCHITECTURE_DECISIONS.md`, `COOKBOOK.md`, `GOLDEN_CALIBRATION.md`, `PIPELINE_SPEC_V1.md` — no
> ruling log. I verified the STATUS block's quote is verbatim **against item 3(d)**, i.e. against the
> transcription, not against the source. If 3(d) widened the ruling, quoting 3(d) verbatim cannot
> detect it.

## READ THIS BEFORE TREATING ANY ENTRY BELOW AS A SOURCE

**This file is the Orchestrator's record of what the adjudicator ruled. It is NOT a verbatim
transcript.** It was written from the Orchestrator's working notes, which were themselves condensed
from the adjudicator's replies. So:

- Text in `> quotes` is a phrase captured as the adjudicator's own wording. Treat those as quotable.
- Everything else is **the Orchestrator's paraphrase**, including the numbers.
- A reader checking a `§16.x` transcription against this file is therefore comparing one
  Orchestrator record against another. That catches *drift between the two*, which is worth having —
  it would have caught several defects this project has actually shipped — but it **does not**
  establish fidelity to the adjudicator. Do not describe it as if it does.

**For future rulings: capture the adjudicator's reply verbatim into this file before condensing it
anywhere else.** That is the only version of this file that closes the gap properly. Entries below
predate the practice and are marked accordingly.

---

## Fable ratification package — 2026-07-29 (record quality: PARAPHRASE, not verbatim)

Convened as tie-breaker on the F1 sequence after the two Opus subagents disagreed. Fable re-ran the
probe itself under the pinned checkout and cross-validated against §16.20 item 9's independently
ratified measurement of the same coverage-filter firing. **Fable wrote no code and edited no files**;
implementation goes through the normal pipeline.

Transcription status: **NOT YET TRANSCRIBED** (task #23). It blocks the recording run (#12, #13).

### R1 — Derivation is recorded

Engineer's position wins. The architect's prose qualifier was refused —
> not a competing design; it is a sentence

The impossibility holds in its decisive form: a split-derived block is observationally
`xyxy == bbox(lines)` with `ours ⊄ xyxy`, which is *exactly* what a genuine leg-1 defect looks like,
so any pair-local predicate exempting one exempts the other.

Fragility objection answered: the oracle already reaches inside upstream by ratified decision three
times (§16.24 items 8, 9, 18(e)); the oracle is the frozen artifact, not the recipe (§16.20 item 7);
an upstream bump is already a re-record plus full re-sign event.

**Binding condition.** The recorder MUST run pristine and instrumented `group_output` on deep-copied
identical inputs and hard-fail unless outputs are field-by-field identical (`xyxy`, `lines`,
`language`, `vertical`, `font_size`). `PROVENANCE.json` records that the assertion ran.

**Schema.** `derivation: YoloUnioned | YoloSynthesizedCorners | YoloSplit | DbnetScattered` — a
CLOSED enum; the recorder fails on any unclassifiable block; there is no `other` bucket.
`rect_yolo: Option<[i32;4]>`. `eng_expanded: bool`; when set, `lines_pre_expand` is REQUIRED plus
`expand_size`. One exact epsilon-free law per derivation. Measured 24/24, 2/2, 7/7, 11/11, and 6/6
eng-expanded under `lines_pre_expand`.

`YoloSplit`'s law turns `Mechanism::DocumentedSplitMerge` from an unfalsifiable citation into
verified evidence: sub-block `xyxy == bbox(lines)`; `rect_yolo == parent`; the unmatched ours-side
parent's `rect == rect_yolo`.

### R2 — Discriminator rejected

Recomputed TP=1, FP=4, FN=1 — precision 20%, with a false negative **on the ratified page**. It also
manufactures false reds: locke block 0 under `LineLess` asserts `[586,163,606,236] == [588,164,605,235]`.

### R3 — F-4 blocks, and the page is not re-selected

F-4 blocks on 18(h)(ii)+(iv) and 3(e), **not** on 18(f) — no port defect exists, because the
mechanism is upstream's own post-`adjust_bbox` mutation.

The oracle page is NOT re-selected: under corrected recording every p01 block closes exactly (3
paired at `[0,0,0,0]` over `lines_pre_expand`, 1 `ClassDuplicateOf`, 1 `CoverageFilteredUpstream`).
Re-selection cannot dodge F-4 anyway — any page with an eng-classified horizontal block has it.

**Two corrections against the engineer.** (a) "3 of 4 gated blocks escape" **overcounts**: block 1 is
the §14.13 class-duplicate of block 2 and never a gated pair, so the correct figure is **2 of 3
paired**. (b) The 11/49 no-legal-mechanism ground does **not** bite p01 (zero scattered, zero split
there); it blocks the *vocabulary* and the *alternative pages* (p02 has 2 scattered, p03 has 1).

**§16.25 item 10 pre-committed this outcome.** It named the fact unverified, presupposed "no", and
ruled that if the check came back otherwise then "the finding is that item 18(b)'s leg-2 law needs an
erratum". It came back otherwise — 11/49 measured. The erratum is that ruling **firing**, not a fresh
judgment. **Neither architect cited it**, which Fable flagged as how scope drift starts.

Coverage: `Partial` does not fire on scattered blocks, because §16.25 item 6(a) computes over the
PAIRED set. Item 10's obligation is re-scoped to one score per gated yolo-derived block.

### R4 — F-5: item 3(b) alone is no longer fit as the sole geometry gate

65 of 100 edges are unconstrained; 8 of 25 blocks are free on all four; `spikey` tolerates
`[19,60,23,17]` px.

**New gating row, exact and epsilon-free: `ours.rect == rect_yolo` on every paired block.** 18(h) is
NOT reopened — its amendment made `upstream.xyxy - ours.rect` *diagnostic* precisely because that is
non-zero when healthy; the new row gates `ours.rect - rect_yolo`, which is zero when healthy.

Contingency: a straddle pixel can fail this row at recording time. Exits are 18(h)(ii)'s;
classification is 18(j)'s (a §16.x shared sensitivity, **never** §14); evidence is pre-truncation
floats on BOTH sides.

**18(d) resolved:** branch (2), the truncation straddle (darkrays `110.9988` → `110`). Branch (1) is
REFUTED, and §16.20 item 12's transcription was correct as written. Branch (2) does not gate
recording.

### R5 — Clamp

The engineer's disposition is adopted **whole**. An erratum withdraws §16.6 item 4's false
`clip_coords` citation (grep exits 1; the only clamps are IoU arithmetic at `yolov5_utils.py:166,169`
— the same family as `PAD_VALUE=114`). The clamp is KEPT, re-grounded as §14 register entry 18 plus
`DEVIATION(18)` at `yolo.rs:164`; the false citation in the doc comment at `yolo.rs:144-148` is
rewritten in the same change. No relaxation, no `validate()` change, no deletion.
`UpstreamBoxOutsideFrame` is adopted as a GATING row (0/38 today).

### R6 — Both collateral errata confirmed

(a) **21(f) NARROWED.** Nightmare block 4 gives `residual_leg1 [45,74,-1,-1]` — all four
union-impossible directions — with leg 1 EXACT. Its per-edge attribution is true of `YoloUnioned`
pairs and silent otherwise. This is cookbook rule 15 applied to the rule that cites rule 15.

(b) **18(i)'s message re-worded derivation-conditional**, plus a new structural guard: a pair whose
upstream derivation is `YoloSplit` or `DbnetScattered` is ITSELF a gating row. Cookbook rule 15's
final bullet carries the same unconditional sentence and is corrected too.

### R7 — Test corpus

Cookbook rule 8, **exit 1**: additive. Replacement refused — the frozen tests assert correct
behaviour over the artifact GRAMMAR, which admits `lines: []`.

Add reachable-shape controls: a synthesized-corners pair, a dominating-lines pair, a multi-line pair,
and a split-shaped unmatched structure.

Plus errata to the two spec sites that CAUSED the weighting: §16.20 item 3's "this case is not
hypothetical" and 18(k)'s "rare-but-real" both conflate upstream's line-less **filter** branch (real
— it fired on `[438,1407,498,1446]`, `mask_score 0.0359`, `conf 0.621`) with a line-less **output**
block (impossible: 0/49, and provable from control flow).

### R8 — The supersession span rule

Ruled 2026-07-29, and the only ruling in this package already transcribed — as §16.26 item 3(d),
landed in `6983b81`.

> A three-component sub-section anchor resolves to its two-component parent's span, but ONLY IF the
> literal sub-section token appears as a bold heading inside that span

Teaching the outline parser a bold-heading section class is **REFUSED**: there is exactly one such
subsection in the file, it has no reliable terminator, and §8.3's `**N — Title.**` item markers are
near-misses waiting to be misparsed into spans. `§7.2.999` must still fail.

Ruled after the enforcing test failed on the pinned row `("8.7", "7.2.1")` and an agent tried to pass
it by truncating every dotted anchor to two components — which turned the suite green and would have
resolved `§7.2.999` too. That truncation was reverted.

---

## Sequencing Fable set for the package

Must land BEFORE the recording run, as ONE ratification package (a new §16.x entry plus the errata),
each erratum carrying its §16.26 marker AND its back-pointer at the target, with the whole
transcription going through step 1a's fresh-reader review:

1. **Derivation recording (R1)**, including the authorized mechanical adaptation of the frozen struct
   literals in `f1_oracle_comparator.rs`, with a §16.24 item 19(e)-style condition-by-condition table
   — **not** the sentence "nothing was weakened".
2. **Leg-1 gating row (R4)** plus the mis-paired-derivation guard (R6b).
3. **`DbnetScattered` mechanism/register entry (R3)** — without it, p02/p03-class pages have only
   `Open`, and stretching `DocumentedSplitMerge` over §14.12 would widen a register entry past its
   recorded scope.
4. **All six errata**: 18(b) leg-2 scope, 21(f), 18(i), §16.20 item 3, 18(k), §16.6 item 4 + §14
   entry 18, and 18(d) recorded as resolved.

May accompany but does NOT gate: `UpstreamBoxOutsideFrame` (0/38); the additive reachable-shape
controls (owed before the atomic recording commit regardless); the cookbook rule 15 correction; the
ours-side pre-truncation probe (owed at recording time for diagnostics).
