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

Transcription status: **NOT YET TRANSCRIBED** (task #23), **except R8**, already landed as §16.26
item 3(d) in `6983b81` — see R8 below. It blocks the recording run (#12, #13).

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

*Open provenance gap, found 2026-07-30, NOT resolved.* The counts above (24+2+7+11 = 44, plus 6/6
eng-expanded) do not describe the three recorded oracle-page candidates alone. P01/P02/P03 are now
recovered (see R3) — P01 byte-for-byte verified against §16.24 item 17's recorded **sha256**; P02/P03
verified only by **size** against item 17(a)'s recorded byte counts, since the spec records no hash
for either — then re-run through upstream's real `group_output`: **P01 = 4 blocks, P02 = 6, P03 = 4 —
14 total**, nowhere
near 44 or 49 (R3/R7's denominator). Whatever corpus produced 44/49 therefore includes pages beyond
P01–P03 (plausibly `demo_bubbles` and/or others not named here), and its exact composition is not
recorded anywhere retrievable — the probe scripts that produced it lived only in the deleted
scratchpad. **Do not transcribe the specific historical counts as a reproducible fact.** What R1
actually needs in the spec is the **schema** — the closed enum, the binding condition, one exact
law per derivation — which does not depend on which corpus validated it. Treat "24/24, 2/2, 7/7,
11/11" as historical supporting evidence of unconfirmed provenance, not as a spec-level assertion.

### R2 — Discriminator rejected

Recomputed TP=1, FP=4, FN=1 — precision 20%, with a false negative **on the ratified page**. It also
manufactures false reds: locke block 0 under `LineLess` asserts `[586,163,606,236] == [588,164,605,235]`.

*Unresolved, found 2026-07-30.* No fixture named `locke` exists anywhere — not in this repo, not in
upstream's `media/` tree, not as an identifier in this project's own Rust source (the one repo hit
for the string is an unrelated CLI-test path comment naming the manga title "Choujin Locke",
coincidental). The cited coordinates (`x≈586–606`) are categorically impossible on any of the 7
committed `demo_bubbles` crops (max width 354px) and do not appear on any block detected on the
real, now-recovered P01/P02/P03 pages either (checked directly against this session's own
`group_output` run). This worked example cannot currently be verified against any available asset
and should be treated as **unverifiable**, not merely mistyped, pending recovery of whatever page it
was actually computed against. It does not block transcription: R2 ("discriminator rejected") is
supporting evidence for preferring R4's gating row, not itself part of the required #23 transcription
list.

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

*Found 2026-07-30: "p01/p02/p03" identified and recovered.* These are §16.24 item 17's Pepper&Carrot
episode 1 pages — P01 the ratified oracle page, P02/P03 its rejected siblings (item 17(a)). None
were actually committed to `tests/fixtures/upstream/` despite item 17 saying they would be (a
ratified step that was never executed — separate from anything in this ratification package).
Re-fetched from `peppercarrot.com`'s `0_sources/ep01_Potion-of-Flight/low-res/` directory. P01 is
441,914 B, sha256 `3bef9922e09cea66ab12271da0070025768ae9bc5d286f41ced617468131267e` — an **exact
match** to item 17's recorded hash, so P01 alone is byte-for-byte verified. Item 17(a) records no
hash for P02/P03, only sizes; those sizes (448,475 B / 302,541 B) match exactly, which is weaker
evidence of identity than a hash match but is what the spec actually lets us check. Running
upstream's real `group_output` on all three: P01 → 4 blocks, P02 → 6, P03 → 4. **This is a
downstream measurement, not additional identity evidence** — it does not itself confirm P02/P03 are
the right files (their identity rests only on the size match above, weaker than P01's hash match),
and it does **not** reconcile the 44/49 counts either way (see the gap noted on R1): 14 total blocks
across these three pages is far short of either figure, so the broader corpus R1/R7 measured against
includes more than just these three.

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

*Re-verified 2026-07-30 against the rebuilt oracle.* Instrumented the exact truncation site
(`inference.py:114-124`, `postprocess_yolo`'s `.astype(np.int32)`) on both fixtures with the real
weights: `darkrays` box 0 gives pre-truncation x2 = `110.9987564086914` → truncates to `110`,
matching this line to four decimals; `nightmare`'s sole YOLO box is nowhere near 110/111. **This
line's fixture attribution is correct as written.** §16.24 item 18(d)(2)'s prose is not thereby
wrong either — its cited `ratio_x = 219/654` and `111.02/110.98` figures are `nightmare`'s own
geometry, used as an illustrative computation of the same `r≈3.0–3.8` regime both fixtures sit in
(darkrays: 208×320, r=3.2). The two were never the same measurement; reading them as one shared
worked example is the only thing that looked like a contradiction. Not an erratum — nothing here
was wrong, so it does not belong in the errata count below (see the corrected count in the
Sequencing section).

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

*Re-verified 2026-07-30 against the pin.* `group_output` lives at
`pcleaner/comic_text_detector/utils/textblock.py:447-534` (nested under `pcleaner/`, not repo root —
corrects a path HANDOVER.md's placeholder implied). Read directly: every path to `final_blk_list`
is line-guaranteed — a `blk_list` block either already carries lines, or (lines 485-492) is
`continue`d out on failing the coverage filter, or has synthetic lines appended before being kept;
scattered-line blocks (line 474) are constructed with at least one line by definition. No branch
appends a line-less block. **R7's control-flow claim holds**, confirmed against the pin rather than
assumed (the `0/49` count itself is a measurement and inherits R1's open provenance gap above). The
English-expansion loop HANDOVER.md flagged with an unverified `~519-535` placeholder is at **lines
518-532** (`for blk in final_blk_list: if blk.language == "eng" and not blk.vertical:` through
`blk.font_size += expand_size`); that placeholder is now grounded and can be retired.

### R8 — The supersession span rule

Ruled 2026-07-29, and the only ruling in this package already transcribed — as §16.26 item 3(d),
landed in `6983b81`.

> A three-component sub-section anchor resolves to its two-component parent's span, but ONLY IF the
> literal sub-section token appears as a bold heading inside that span

Teaching the outline parser a bold-heading section class is **REFUSED**: there is exactly one such
subsection in the file, it has no reliable terminator, and the **32 file-wide** `**N — Title.**` item
markers (§8.3 has only 7 of them; the rest are in §9.3/§11.3/§12.3 — the file-wide 32, not §8.3's 7,
is the figure that bears on the parser-class decision) are near-misses waiting to be misparsed into
spans. `§7.2.999` must still fail.

Ruled after the enforcing test failed on the pinned row `("8.7", "7.2.1")` and an agent tried to pass
it by truncating every dotted anchor to two components — which turned green all seven assertions
then present, and would have resolved `§7.2.999` too. That truncation was reverted.

*Corrected 2026-07-30:* this entry previously said "§8.3's item markers" and "turned the suite
green", both narrower/looser than what §16.26 item 3(d) — the actual ratified transcription — says.
Reworded above to match 3(d) exactly; this file's own purpose is catching exactly this kind of
drift, so leaving it uncorrected here would have been the wrong example to set.

---

## Sequencing Fable set for the package

Must land BEFORE the recording run, as ONE ratification package (a new §16.x entry plus the errata),
each erratum carrying its §16.26 marker AND its back-pointer at the target, with the whole
transcription going through step 1a's fresh-reader review:

1. **Derivation recording (R1)** — transcribe the **schema** (closed enum, binding condition, one
   exact law per derivation) with a §16.24 item 19(e)-style condition-by-condition table — **not**
   the sentence "nothing was weakened", and **not** the specific historical counts (see R1's open
   provenance gap above). This item also claims a mechanical adaptation of the frozen struct
   literals in `f1_oracle_comparator.rs` is "authorized" — **found 2026-07-30: no R-item above
   actually grants that authorization.** Confirm it explicitly, quoting the specific ruling's scope,
   before treating the frozen-test edit as licensed; do not proceed on this sentence alone.
2. **Leg-1 gating row (R4)** plus the mis-paired-derivation guard (R6b).
3. **`DbnetScattered` mechanism/register entry (R3)** — without it, p02/p03-class pages have only
   `Open`, and stretching `DocumentedSplitMerge` over §14.12 would widen a register entry past its
   recorded scope. **Found 2026-07-30, not previously flagged here: this amends §16.24 item 4's
   closed mechanism enumeration and item 20(c)'s table**, so the transcription needs its own §16.26
   marker and back-pointer at item 4/20(c), same as any other amendment to a ratified enumeration
   (§16.25 item 8's own precedent) — it is not a bare addition. It also **re-scopes §16.25 item 10**
   (the 11/49 measurement falsifies item 10's presupposition that every gated block descends from a
   yolo box): item 10 needs its own back-pointer too, which the errata list below previously
   omitted.
4. **The errata — six.** `18(b)` leg-2 scope, `21(f)`, `18(i)`,
   §16.20 item 3, `18(k)`, §16.6 item 4 + §14 entry 18. `§16.25 item 10`'s re-scope (step 3 above) is
   a **seventh** transcription obligation, tracked separately since it rides on the `DbnetScattered`
   entry rather than being its own erratum. `18(d)` is **not** an erratum — R4 above found nothing
   there was wrong (item 12 was already correct); it needs a closure note, not a §16.26 marker, and
   an earlier draft of this list miscounted by including it.

May accompany but does NOT gate: `UpstreamBoxOutsideFrame` (0/38); the additive reachable-shape
controls (owed before the atomic recording commit regardless); the cookbook rule 15 correction; the
ours-side pre-truncation probe (owed at recording time for diagnostics).

---

## §16.27 item 1(g) frozen-literal adaptation — Fable tie-break, 2026-07-30 (record quality: this entry captures the ruling closer to verbatim than any prior one; long paraphrase passages are marked as such)

§16.27 item 1(g) deferred authorization for adapting the frozen struct literals in
`crates/pc-detect/tests/f1_oracle_comparator.rs` to the new §16.27 item 1(a) recording fields,
pending "a joint architect-plus-engineer or Fable ruling". The joint architect + rust-engineer
planning pass (spawned together, each blind to the other) **agreed** on the bulk of the scope but
**disagreed** on one mechanism, so per `CLAUDE.md`'s tie-break rule Fable was convened to decide
between the two positions, not to design from scratch.

**What both agents independently found and agreed on** (paraphrase): the closed 4-value
`derivation` enum admits no value for a line-less block, so 24 of the 28 block instances the file
constructs cannot honestly carry any derivation value — null-completion (`None`/`false`, never a
fabricated value) is the only assignment that asserts nothing false at all 9 `OracleBlock`
construction sites. `derivation` must be `Option<Derivation>` with `#[serde(default)]` on
`OracleBlock`, forced by two existing frozen assertions (`f1_oracle_comparator.rs:1739-1747`'s
`minimal` literal and the `"eng"`-rejection probe at `:1783-1786`) that a required field would
either break outright or silently pass for the wrong reason. Both refused enriching the two
sites where the union arithmetic technically forces a `rect_yolo` value (`upstream_truth()` blocks
A and C) — one on grounds of a fixture collision with the in-place `xyxy` mutation the
fault-injection tests already perform (`:171`, `:352`), the other on grounds of subject drift —
and both agree `Totals` and its ~13 literals need no edit at all (§16.27 item 3(b) already rules
those authored anti-vacuity literals "gain nothing here"). Both agree the comparator does not
re-check the four derivation laws (they are recorder-side, item 1(b)/(c)) and that item 1(a)'s
"REQUIRED" binds the recorder, not the Rust type.

**Where they disagreed, and what Fable decided.** Both identified the same residual hole: with
`derivation: Option<_>`, a real recorded artifact that silently dropped the field would pass
`compare()` unnoticed. The Rust Engineer proposed closing it by signing `derivation` on
`ExpectedPair` itself (mirroring the existing `IdentityBranch` signed-field mechanism) and gating a
mismatch, including a signed-vs-absent mismatch, as a new `Divergence` variant. The Architect
proposed routing it through the existing per-field `OracleCoverage`/`FieldCoverage` channel instead
(the mechanism §16.25 already uses for `confidence`/`language`), reusing the existing
`InconsistentOracleCoverage` variant and deferring the "uniform absence" gate to the real-page
atomic-recording-commit's own assertion — which the Architect itself flagged, unprompted, as
needing its own ratification it was not granting.

Fable's decisive finding, quoted because it is the ruling's own reasoning and not a paraphrase of
it:

> The decisive ground is that a ratified clause already answers the Architect's proposal, and the
> Architect did not cite it. §16.25 item 5 carries a two-condition scope test for exactly this
> question ... `derivation` is none of those — it is an upstream-only recording probe, exactly the
> shape of `base_xyxy_pretruncation`, which §16.25 item 5 names as the worked exclusion.

and on why the Architect's design does not actually close the hole it was built for:

> The Architect's design leaves the named hole open. Its uniform-absence case does not gate inside
> `compare()` at all; the gate is deferred to a real-page authored assertion the Architect itself
> flags as "extending §16.24 item 6's ratified two-part CI contract", "needing its own
> ratification", and "not granted here". So as submitted, the design closes the silent-pass hole
> only after a future ratification that does not exist.

**Ruling: the Rust Engineer's signed-per-pair mechanism wins**, with two amendments Fable added
after re-deriving the arithmetic and re-reading the ratified precedent itself (not merely picking a
side):

1. **Amendment 1 — bidirectional exact equality.** The Engineer's original text covered only
   "signed `Some(d)`, artifact doesn't match". Fable requires the comparison to fire in the third
   direction too — signed `None`, artifact `Some(b)` — closing a hole where a future control could
   carry a real derivation while signing `None` and silently skip the gate.
2. **Amendment 2 — the anti-vacuity literal's exact VALUE is not ratified, only its form.** The
   Engineer's proposed `report.leg1_rows_checked == 14` conflates §16.27 item 1(f)'s three-page
   14-block *census total* with the ratified oracle page P01's own paired-block count, which is
   **3** (§16.27 item 4 / this file's R3: "3 paired at `[0,0,0,0]` ... 1 `ClassDuplicateOf`, 1
   `CoverageFilteredUpstream`"). Fable authorizes the counter *mechanism* and requires it be
   exercised non-vacuously in the same commit and asserted against an authored (hand-derived, never
   artifact-derived) literal at the real-page gate — but does not itself ratify "14" or any other
   number as that literal's value, since neither agent's ruling actually derived one for P01.

**Final scope, as Fable stated it:**

- `ExpectedPair` gains `derivation: Option<Derivation>`; all 21 frozen literals get `derivation:
  None`.
- Final ratified `Divergence` variant count is **19**: the existing 16, plus §16.27 item 2's leg-1
  row, plus item 2(c)'s structural guard, plus a new signed-derivation-mismatch gating variant.
  Partition test moves to `samples.len() == 19`, `(gating, diagnostic) == (17, 2)`.
  `UpstreamBoxOutsideFrame` is not among the 19 (stays untranscribed per item 11).
- A `derivation == None` artifact block disables the derivation-conditional machinery for its pair
  (item 2's row, item 2(c)'s guard, item 6's derivation-conditional message) — must be stated in
  `oracle.rs`'s doc comment, not merely implemented. This is safe only because Amendment 1's
  bidirectional signature check is unconditional.
- Everything the two agents agreed on stands, plus two things grafted from the Architect's losing
  position because Fable found no defect in them: §16.24 item 1(f)'s two conditions carried
  verbatim (every existing value retained verbatim; new values are literals, never derived from a
  run), and the `bare_block` comment recording the asymmetry against the populated full-shape JSON
  exemplar.
- The full-shape JSON literal (`f1_oracle_comparator.rs:1749-1772`) is populated with real values,
  not left `None` — the Rust Engineer's own proposal (`derivation: Some(Derivation::YoloUnioned)`,
  `rect_yolo: Some([29,105,86,261])`, `eng_expanded: false`, the last two figures being the
  `ours`/expected geometry of the block that literal already models), which Fable independently
  re-solved the union arithmetic for and confirmed satisfies the `YoloUnioned` law. Both agents'
  rulings required this literal be kept in sync with its own "a dropped optional field is the quiet
  direction of drift" comment; this is that requirement applied to the three new fields, plus the
  three assertions needed to make the population checked rather than merely present.
- New fire-case tests required in the same commit: the leg-1 row firing, the leg-1 row passing
  non-trivially, item 2(c)'s guard firing, and — the case this whole dispute exists to catch — the
  signed-derivation mismatch firing in its silent-drop direction (signed `Some`, artifact `None`).
  R7's four additive controls may supply pass-side coverage but may not substitute for these — and
  to supply that pass-side coverage at all, at least one of R7's controls must sign a real
  `derivation` value, since a signed `None` (per the `derivation == None` disabling rule above)
  cannot exercise any of the new machinery.

**Explicitly left open, not decided here:** whether an *unmatched* upstream entry's signed
`DbnetScattered` mechanism gets verified against that block's recorded `derivation` inside
`compare()` — a residual gap in *both* designs, named but not closed; F1-C may propose a fix, but
it needs its own ruling. The serde spelling of `Derivation`'s enum values is left to F1-C, with the
condition that it is pinned in the full-shape JSON literal in the same commit (that literal is the
only Rust-side gate on the not-yet-written Python recording script). The Architect's ruling also
left the Rust encoding of the `eng_expanded`/`lines_pre_expand`/`expand_size` trio free — a flat
three fields, or a grouped `Option<EngExpansion { .. }>` that makes the "REQUIRED together"
conditional unrepresentable-if-violated — noting a preference for the grouped form without ruling
it, since "the authorization's scope is identical either way; only the token count differs." Fable
did not revisit this; it stands as the Architect left it.
