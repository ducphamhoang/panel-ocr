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
predating that practice are marked accordingly; the two entries below are the source record for
§16.32's two consultations.

---

## Entry 1 — original CPU-EP design tie-break — 2026-08-03 (BEST-EFFORT RECONSTRUCTION, not verbatim)

The original Fable adjudication was not captured into this file at the time it happened. The only
surviving verifiable artifact is commit `3801b32`'s message, checked with `git log -p 3801b32 -1`.
That message contains one Fable-related sentence, quoted verbatim here:

> A Fable adjudicator ruling (independently reproduced by the ruling itself) required the flag ship only bundled with thread confinement.

This is a **BEST-EFFORT RECONSTRUCTION**, not a verbatim transcript. Which position was held by the
architect and which by the rust-engineer is not recorded anywhere in the surviving material and is
not being invented here.

---

## Entry 2 — task-#29-planning tie-break — 2026-08-04 (VERBATIM SOURCE TRANSCRIPTION)

Fable was convened during task #29's planning phase, separately from the original CPU-EP tie-break,
to resolve two disagreements in the joint architect+rust-engineer plan. The ruling text below is
transcribed from the verbatim source supplied for that planning consultation. The commit named in
Ruling 1 was independently verified with `git log --oneline -- tests/fixtures/recorded/detector/PROVENANCE.json`.
Entry 2 follows this supplied verbatim source text convention, distinct from the paraphrase default
described in the preamble for earlier and other entries.

### Ruling 1 — flush_denormals in PROVENANCE.json: the rust-engineer's position wins, with one graft from the architect.

Decision. #29 does not add the field. The new §16.32 entry records, in prose: (a) the committed detector recording was produced at commit `f6212eb`, which recorded `tests/fixtures/recorded/detector/PROVENANCE.json`, before SessionTuning existed, so its session ran without denormal flushing — the fact a flush_denormals field would carry is false; (b) xtask/src/record.rs writes intra_threads/inter_threads as hardcoded literals rather than observed values, an open defect against §16.21 item 5's "records the thread count actually used... facts, not inferences"; (c) a binding requirement that the next real re-record (the one §16.23 item 5 already budgets) lands the recorder fix — observed values for all three thread/tuning settings — and the flush_denormals schema field in the same change. Graft from the architect's position: the §16.24 item 19(c) table gains the xtask/src/ocr_model_signature.rs writer row in #29, since this gap was verified directly and is independent of this dispute.

Grounds. Two, one measured and one structural. Measured: the three existing "resolved session-construction" fields the architect appeals to are not a class to join — two of the three are literals that merely coincide with TextDetectorConfig::default(), so the precedent being cited is itself the defect. Structural: under the architect's proposal, the committed file's false cannot be written by any recorder in the tree (the recorder at HEAD runs with flushing true), so the value would enter the file by hand-edit or by a one-off migration branch — a self-attested fact in a document whose entire purpose is that a recorder wrote it. The engineer's route keeps "provenance records facts" literally true: every field in the committed file was emitted by the recorder that ran. Deferral costs nothing the spec prose doesn't cover, and the re-record that makes the field honest is already budgeted.

Scope. This ruling covers the flush_denormals field of OursPins in the detector group's PROVENANCE.json, and the two thread-count literals in xtask/src/record.rs's detector ours block. It does NOT decide: whether the future schema change needs a PROVENANCE_SCHEMA_VERSION bump, the shape of SessionTuning fields beyond flush_denormals, or anything about the other five recorded groups.

### Ruling 2 — bench-tuning containment gate: the rust-engineer's position wins on "gate now", with the architect's unification finding converted into binding conditions on the gate's design.

Decision. #29 adds the containment test. But the architect's technical objection is measured fact, not speculation — cargo test --workspace really does compile pc-detect with bench-tuning active and link that rlib into pc-cli's units — so the gate lands under three binding conditions:

1. The traversal covers only [dependencies]/[build-dependencies] edges and [features] implication chains reachable from pc-cli. It must never read a [dev-dependencies] section, or pc-detect's self-dev-dep makes it permanently red.
2. The test's name and doc claim only what it checks: that no manifest on pc-cli's normal/build dependency graph enables pc-detect/bench-tuning — i.e. the production binary's graph. It must state that workspace test builds DO carry bench-tuning into the linked pc-detect via dev-dep unification and that this is by design and out of the gate's scope.
3. The anti-vacuity negative control (feature still declared where expected) stays.

Grounds. Three production-graph leaks with zero automation is exactly the consensus-shaped gap this pipeline keeps paying for. The architect's dilemma ("vacuously green or wrongly red") is a false dichotomy for a manifest reader: unification is a build-time phenomenon that edits no manifest, so a normal/build-edge manifest walk is neither tainted by it nor vacuous — it is a regression gate on precisely the mechanism class (manifest edits) that caused all three historical leaks in the production build.

Scope. This ruling covers one gate: pc-detect/bench-tuning reachability from pc-cli's normal/build manifest graph. It does NOT decide: whether the workspace-test-build taint needs its own mitigation someday; whether an additional resolver-backed check should complement the manifest reader; or CI tiering for either.

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

## F1 Phase 2 plan: identity operand, oracle-page location, DETECTOR_ORACLE.md sequencing (Fable tie-break, 2026-07-30; record quality: captures Fable's reply near-verbatim, both rulings' reasoning paragraphs are the adjudicator's own text condensed only by dropping repeated setup)

The joint architect + rust-engineer planning pass for #12/#13 Phase 2 (the Python detector-oracle
recording script, the real-page comparison run, the atomic recording commit) was spawned per
CLAUDE.md's Plan step, each blind to the other. Both independently identified and converged on the
same fix for the one question that actually blocks writing code: `crates/pc-detect/src/oracle.rs`'s
identity check unions the upstream-SERVED `lines` field, but the real oracle page needs
`lines_pre_expand` (the pre-expansion polygons) to pass, per §16.27 item 4. Both proposed: read
`lines_pre_expand` when the artifact recorded it, else fall back to `lines`; zero frozen
`ExpectedPair` literals move; no new `Divergence` variant is needed, since the identity already
self-gates a dropped `lines_pre_expand` (the expansion is strictly outward-monotone). The architect
additionally rebuilt the real upstream checkout (pinned commit
`0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`) and ran the actual instrumentation, empirically
reproducing this exact failure mode and confirming the fix on the real P01 page. **This premise was
not re-litigated by Fable and is not part of either ruling below** — it is agreed, not adjudicated.

Two real disagreements were referred to Fable.

### Ruling 1 — where the committed oracle page lives: the Architect wins. Move, do not duplicate.

The Rust Engineer read `crates/pc-testkit/src/provenance.rs:334-347`'s comment (citing "§16.24 item
2 requires the page to live inside the declaring group") as an over-claim on ratified text that only
requires single declaration, and proposed committing a second, byte-identical copy inside
`tests/fixtures/recorded/detector/` alongside the existing `tests/fixtures/upstream/oracle_pages/`
original. The Architect traced the shipped validator's actual code and concluded the group-prefix
requirement is forced by the code itself, comment aside, and proposed moving the file (byte-preserving
`git mv`) with no duplicate.

Fable built a scratch probe crate calling `pc_testkit::provenance::validate` directly on two
otherwise-identical detector groups differing only in `input_page`, and ran it. Verbatim output:

```
input_page = tests/fixtures/recorded/detector/ja_...E01P01.jpg
  violations: []
input_page = tests/fixtures/upstream/oracle_pages/ja_...E01P01.jpg
  violations: [CommittedPathOutsideGroup { at: "detector.input_page", .. }]
```

Fable's ruling, quoted:

> So the group-prefix requirement is forced by shipped code, and by two independent frozen gates,
> not one: (1) `crates/pc-testkit/src/provenance.rs:334-347` (`validate_detector`) — the behavior
> the probe measured, and it is pinned by a frozen test: `provenance_schema.rs::
> r8_reaches_the_source_and_input_page_path_slots_too`'s last case asserts exactly
> `CommittedPathOutsideGroup` for a relative page in the wrong group. (2)
> `crates/pc-testkit/tests/recorded_provenance.rs:315-321` — a declared path that does not strip
> `tests/fixtures/recorded` falls into `assert_known_non_committed_form` (`:184-208`), which panics
> unless the path is scratch or a bare `*.pt.onnx`. An `oracle_pages/` `input_page` is neither.
>
> The engineer's factual premise — "the prefix is only a wrong comment on an unenforced constraint"
> — is therefore wrong. Given a copy inside the group is mandatory, the `oracle_pages/` original
> would be a second 441,914-byte file that nothing consumes (verified: `grep -rn oracle_pages` over
> the tree hits only `ATTRIBUTION.md` and `docs/HANDOVER.md`, both prose) and that no test pins —
> pure drift surface and repo weight. §16.24 item 17 ratifies the page's identity (441,914 B, sha256
> `3bef9922…`, "unmodified") and an ATTRIBUTION entry; it ratifies no filesystem path, so the move
> violates nothing.

Fable grafted two things from the losing position: the comment at `provenance.rs:336-341` does
over-attribute (item 2's text is single-declaration; the lives-inside-the-group requirement actually
flows from item 1's preserved frozen walker plus item 2's walker-visibility mandate) — fix the
citation, comment text only, zero behavior change; and the engineer's checked-redundancy instinct
survives as: at the atomic commit, assert the recorded `input_page_sha256` equals item 17's ratified
literal `3bef9922e09cea66ab12271da0070025768ae9bc5d286f41ced617468131267e`, so an accidental re-encode
during the move fails loudly.

**Decision:** P01 moves (`git mv`, byte-preserving) from `tests/fixtures/upstream/oracle_pages/` to
`tests/fixtures/recorded/detector/`, inside the atomic recording commit (the frozen
`recorded_provenance.rs::discover_groups`/`read_provenance` gates panic on an early, PROVENANCE-less
move — measured, not asserted), with `tests/fixtures/upstream/ATTRIBUTION.md`'s row updated to the
new path and `docs/HANDOVER.md:51`'s stale reference corrected. No second copy is committed anywhere.
P02/P03 stay in `oracle_pages/` (used only for the `DbnetScattered` census, never as `input_page`).
**Scope, per Fable:** this ruling covers P01 only; it does not decide the destination filename
(bytes and digest are pinned, the name is the planners' choice), does not move P02/P03, and
authorizes no behavior change to any validator — only the comment-citation fix.

### Ruling 2 — when `docs/DETECTOR_ORACLE.md` lands: the Architect wins, with two binding conditions.

The Rust Engineer read §16.24 item 6's atomic-commit list literally, bundling the doc and its
verdicts into the single atomic commit. The Architect proposed landing the doc as an early skeleton
— `OPEN` rows, no verdicts, no signatures — together with item 11's four doc-shape gates in
"dormant but verified" form, by analogy with `xtask/tests/provenance_digests.rs`'s existing
`EXPECTED_DETECTOR_PIN_GROUPS = 0` pattern.

Fable's ruling, quoted:

> The ratified sentence, verbatim (spec lines 3970-3972): "The recording commit is atomic: fixtures
> + provenance + item 1(f)'s constant edits + the real-page gate test with its literal
> `Expectations` + the doc verdicts + the three signatures land in one commit. Until then no
> real-page gate test exists." It names "the doc verdicts", not the doc file. The engineer's
> literal-reading position is literal about a paraphrase: the brief's quotation of item 6 inserted
> "`docs/DETECTOR_ORACLE.md`" and "in one commit or none" — neither string appears in the spec
> (`grep -n "one commit or none"` hits nothing). This is the project's own widening-by-paraphrase
> defect, operating on the adjudication input itself.
>
> Item 11's ratified gate reads "no `OPEN` row once the fixture is present" (spec line 4059). That
> conditional is vacuous unless a doc with `OPEN` rows may exist while the fixture is absent — the
> ratifiers contemplated exactly the pre-fixture state the architect proposes.
>
> The "dormant but verified" precedent is real and already accepted practice:
> `EXPECTED_DETECTOR_PIN_GROUPS: usize = 0` at `xtask/tests/provenance_digests.rs:64`, with
> `detector_pins()` asserting the discovered count against that literal before any early return, and
> a comment (`:208-211`) explicitly distinguishing this from the conditional-return gate item 6
> rejects — citing item 6 by name. I ran it: 6 tests green, the dormant legs printing their
> `DORMANT:` disclosure.

**Decision:** `docs/DETECTOR_ORACLE.md` may land early as a skeleton (rows `OPEN`, no verdicts, no
signatures) together with item 11's four doc-shape gates in dormant-but-verified form. The verdicts,
the three signatures, the fixtures, the provenance, the item 1(f) constant edits, and the real-page
gate test with its literal `Expectations` land in the atomic commit, exactly per item 6.
**Two binding conditions, both from ratified text:** (1) every dormant doc gate must follow the
`detector_pins()` pattern — the dormancy premise asserted against a pinned literal (e.g.
`EXPECTED_SIGNATURES = 0`, raised to 3 in the atomic commit) before any early return; a bare
`if !fixture_present { return }` is forbidden, since item 6 expressly rejects "a conditional-return
gate" as "the shape cookbook rule 1 exists to resist." (2) "Until then no real-page gate test
exists" stands unqualified — the early task may not contain the real-page comparison test in any
form, not `#[ignore]`d, not dormant, not skeletal. Only item 11's four doc-shape gates land early.
**Scope, per Fable:** licenses landing early exactly two things — the doc file's skeleton and item
11's four gates in dormant-but-verified form; does not license landing any other item-6 member early;
does not touch the maintainer's still-open signature-scheduling decision; does not reinterpret "the
doc verdicts" at any other site the phrase appears.

**What Fable ran, all real:** the scratch probe crate against `pc_testkit::provenance::validate`
(Ruling 1's load-bearing measurement); `cargo test -p pc-testkit --test recorded_provenance -q` (1
passed) and `cargo test -p xtask --test provenance_digests -q` (6 passed) at HEAD; `grep -rn
oracle_pages` and `grep -n "one commit or none"` over the tree/spec. **Not settled empirically:**
Ruling 2's reading of "once the fixture is present" as contemplating the pre-fixture doc state is
interpretation of ratified prose, not measurement — Fable flagged this itself, while calling it
forced by the conditional's grammar.

---

## P7 manga-ocr artifact, vocab, task split — Fable tie-break, 2026-08-02 (record quality: MIXED — see the scope note immediately below; do not read the "VERBATIM" claim as covering this whole entry)

**Scope of the verbatim claim, stated precisely so this entry cannot be over-cited.** Only the
`>`-blockquoted passage under "Fable's reply, verbatim" below is the adjudicator's own text,
captured directly rather than paraphrased (this file's own "for future rulings" instruction,
followed here for the first time) — the one edit made to it is removal of a closing
usage/token-count line, which is run infrastructure metadata, not part of the ruling. Everything
outside that blockquote — the "Question put to Fable" framing immediately below, and the "Note
added by the Orchestrator, not Fable" paragraph at the end of this entry — is the Orchestrator's
own prose, written to give the blockquote context, and carries the same paraphrase-not-source
status as every other entry in this file per the header note above. A citation of this entry that
quotes outside the blockquote is quoting the Orchestrator, not the adjudicator.

**Question put to Fable** *(Orchestrator's framing, not Fable's text)*: `architect` and
`rust-engineer` jointly planned P7 (manga-ocr ONNX backend) and disagreed on which third-party ONNX
artifact to pin — `architect` chose `onnx-community/manga-ocr-base-ONNX` (rejecting
`mayocream/manga-ocr-onnx` as "a finetune," citing its vocab.txt as 30,216 B against kha-white's
24,072 B); `rust-engineer` chose `mayocream/manga-ocr-onnx` and did not rebut the finetune concern.
Fable was asked to make the final call on the artifact, plus two secondary questions (vocab
delivery mechanism; whether P7 should include CLI/pipeline wiring or split into P7+P8).

**Fable's reply, verbatim** *(the blockquote below only)*:

> **Mode statement:** I am answering in **Mode A** (tie-break, final call), per the brief.
> Advisory-only status suspended for these three decisions; I wrote no files.
>
> **Prior-ratification check (ran, clean):** `grep -in "mayocream|onnx-community|kha-white|koharu|vocab" docs/PIPELINE_SPEC_V1.md` — no ratified clause pins an OCR artifact. The question is genuinely open.
>
> ### What I actually ran, and what it showed
>
> Network access worked; every claim below is measured, not read off a model card.
>
> 1. **The vocab files are the same vocabulary.** Fetched both at the exact pinned commits. `kha-white/manga-ocr-base` vocab.txt @ `aa6573bd`: 24,072 B, sha256 `344fbb6b…` (architect's pin verified), **6144 lines**. `mayocream/manga-ocr-onnx` vocab.txt @ `24b12778`: 30,216 B, sha256 `5cb5c558…` (engineer's pin verified), **also 6144 lines**. `xxd` shows mayocream's file is CRLF; after `tr -d '\r'` its sha256 is **byte-identical to kha-white's** (`344fbb6b…`). 30,216 − 24,072 = 6,144 = exactly one `\r` per line. The size discrepancy the architect read as "different vocab" is line endings, nothing else.
> 2. **The ONNX weights are the same weights.** Downloaded all four fp32 files; all four sha256s match the respective agents' pins exactly (mayo enc `15fa8155…`, mayo dec `ef776526…`, oc enc `df35f64c…`, oc dec `31ca14d6…`). Parsed initializers with `onnx` 1.22.0: decoders — 63/63 tensors equal (41 name-matched bit-exact, 22 exporter-renamed matched as full-byte multisets); encoders — 198/198 equal (126 name-matched bit-exact, 72 by full bytes). The exports differ only in exporter metadata: torch 2.7.0/opset 14/IR 7 (mayocream) vs torch 2.11.0/opset 18/IR 8 (onnx-community). Graph I/O signatures are identical (`input_ids` + `encoder_hidden_states [b, s, 768]` → `logits [b, s, 6144]`).
> 3. **Both anchor to the true base model, not merely to each other.** Downloaded `kha-white/manga-ocr-base`'s original `pytorch_model.bin` (444,135,475 B, sha256 `c63e0bb5…`) at the pinned commit and compared tensors by name: **mayocream decoder 41/41 bit-exact** (including the full 6144×768 `word_embeddings.weight` — the single most finetune-sensitive tensor — and `cls.predictions.bias`), **mayocream encoder 126/126 bit-exact**, **onnx-community encoder 126/126 bit-exact**, worst diff 0.0 everywhere. The onnx-community decoder anchors transitively through its byte-identity with mayocream's.
> 4. **Configs are functionally identical.** `generation_config.json` differs only in the `transformers_version` string (4.51.1 vs 4.57.6); all generation parameters (beam 4, length_penalty 2.0, max_length 300, no_repeat_ngram 3, tokens 2/3/0) identical. `preprocessor_config.json` differs only in the deprecated alias `ViTFeatureExtractor` vs `ViTImageProcessor`.
> 5. **Repo facts.** `onnx-community/manga-ocr-base-ONNX` @ `f9023406` contains **no vocab.txt and no tokenizer files at all** (full tree fetched via HF API) — the architect's cross-repo vendoring was forced, not a preference. mayocream's README states the export command verbatim (`optimum-cli export onnx --model kha-white/manga-ocr-base --task vision2seq-lm`); its 3-commit history shows upload + README edit, no training activity.
>
> ### Ruling 1 (primary): pin `mayocream/manga-ocr-onnx` @ `24b12778d85800835e2ca409236de281b8ab7b9f` — the rust-engineer's artifact — for the encoder and decoder ONNX files. The architect's disqualifying argument is refuted by measurement.
>
> **Grounds.** The architect's stated and only reason for rejecting mayocream — "a finetune … would put us permanently off upstream with no oracle at all" — is empirically false: every weight tensor in mayocream's export is bit-exact against `kha-white/manga-ocr-base`'s original checkpoint (evidence items 2–3), and its "different" vocab is the same 6144 tokens with CRLF endings (item 1). With the disqualifier gone, mayocream wins on the remaining merits: (a) P7's ratified scope text says "port from koharu `manga_ocr`", and mayocream's export is the artifact koharu itself runs — making koharu a byte-identical secondary oracle during implementation, with zero artifact-difference confounds; (b) the repo is self-contained (tokenizer_config, special_tokens_map, generation_config all present) where onnx-community's carries no tokenizer files and nests the fp32 files among 16 quantized variants someone can grab by mistake; (c) opset 14 is the more conservative target for `ort =2.0.0-rc.12` (opset 18 would very likely also work — unmeasured, and with mayocream it doesn't need to).
>
> **Graft from the losing position (mandatory):** the vocab.txt is **not** taken from mayocream's repo — it is vendored from `kha-white/manga-ocr-base` per the architect's approach (Ruling 2). Mayocream's CRLF vocab is a live defect vector: a naive line-by-line loader would append `\r` to every decoded token. Also graft the architect's habit of pinning the true-upstream commit: the §16.x transcription should record the weight-identity verification (the sha256s and tensor counts above) so the oracle chain to kha-white is in the record, not in a README's claim.
>
> **Scope:** this ruling holds for the two fp32 files `encoder_model.onnx` and `decoder_model.onnx` at commit `24b12778…` only, with the sha256s `15fa8155…`/`ef776526…`. It certifies nothing about any quantized variant, any other commit of either repo, or any other file in mayocream's repo — in particular **not** its vocab.txt, which this ruling explicitly declines to use. It also does not re-verify the unanimous beam-4 finding (out of scope per the brief). What rests on reasoning rather than measurement: the availability-longevity comparison between the two hosting accounts (unknowable either way; sha256 verification in `pc-models` protects integrity, and `--model-path` is the existing mitigation for URL death), and I ran no actual inference on either export — behavioral identity is inferred from total weight identity plus identical graph I/O and identical generation configs, not from an executed OCR run.
>
> ### Ruling 2 (vocab handling): vendor vocab.txt as a repo-tracked file — the architect's approach — sourced from `kha-white/manga-ocr-base` @ `aa6573bd10b0d446cbf622e29c3e084914df9741`, 24,072 B LF, sha256 `344fbb6b…`. Do not make it a `ModelSpec` registry entry.
>
> **Grounds.** (a) The spec's binding general rule (the §16.x item at `docs/PIPELINE_SPEC_V1.md:2866`): "no test-reachable code path may perform network I/O." Vocab parsing, token→text decode and post-processing are pure logic that belongs in the default test tier; a registry-fetched vocab would push those tests behind provisioning, or force a duplicated in-repo fixture — and a gate on a duplicate is cookbook rule 12's inversion (guarding a copy). (b) 24 KB of Apache-2.0 text, with an existing repo convention for vendored, hash-verifiable third-party fixtures plus attribution (`tests/fixtures/upstream/ATTRIBUTION.md`; spec line 4634). (c) It permanently eliminates the CRLF hazard from Ruling 1. Two conditions travel with this ruling: the vendored file gets a sha256 gate in-repo (the `model_signature.rs` recorded-fixture pattern), and a `.gitattributes` entry marking it `-text` so eol normalization can never silently alter the bytes the gate pins — a CRLF-vs-LF sha256 mismatch is precisely the failure this file has already demonstrated once.
>
> **Scope:** this ruling covers vocab.txt only — one 24 KB text file. It is not a precedent for vendoring model binaries (the two .onnx files stay registry entries with sha256 pins, per both plans), and it does not decide the loading mechanism (`include_str!` vs path-based read) — that stays with the joint plan.
>
> ### Ruling 3 (task scope): split — P7 stays pc-ocr-only (heavy, isolated), and a new P8 carries the CLI/pipeline wiring. The rust-engineer's structure wins, with binding conditions.
>
> **Grounds.** The engineer's C.3 observation is correct and I verified its textual basis: §16.12 item 4 (spec line ~2310) reads "P7 (manga-ocr) is unstarted, **so** … pc-cli passes `None` … inactive until P7", and the shipped WARN at `crates/pc-cli/src/setup.rs:94-96` says "v1 ships no OCR engine (task P7)". A pc-ocr-only P7 with no successor leaves a shipped binary emitting a message that has become false (an engine exists; the CLI just doesn't pass it) and a spec promise unfulfilled. But the cure is not welding the wiring into P7: the repo's own precedent separates the detector's ONNX session (D4, heavy) from `run()` wiring (D7) and CLI exposure (X1); §9.5's batching rule isolates heavy tasks; and the two halves have disjoint test surfaces — P7's parity tests need real models (onnx-tier/provisioned), while wiring tests run against the existing P6 mock in the default tier. Conditions, which the transcription must carry: (i) **v1.0 is complete only when P7 and P8 have both landed** — P8 is not optional follow-up, otherwise this ruling would just re-create the gap C.3 found; (ii) when the split is ratified, §16.12 item 4's "until P7" sentence and the `setup.rs` WARN-text expectation get **dual-end supersession markers** naming the ratifying §16.x entry, per §16.19's convention and the supersession cross-check gate; (iii) P8 enters the §9.5 and §13 task tables as its own row (deps: P7, plus the existing pipeline/CLI tasks it touches).
>
> **Scope:** this rules on task *structure* only. It does not design P8 (factory construction, `onnx` feature gating in pc-cli, WARN replacement text) — that remains the joint architects' plan work — and it does not rule on P8's simple/heavy classification beyond noting the table's own taxonomy would call mock-testable cross-crate wiring "simple".
>
> ### What none of this decides
>
> The beam-search supersession of "greedy decode loop" (spec lines 836/1330) is unanimous background per the brief — not adjudicated here, and my Ruling 3's table amendments will land in the same §16.x sequence but are a separate transcription. I also did not review `p7_engine.rs`'s drafted assertions; its hardcoded expected strings should be re-checked against the mayocream-pinned artifacts once the plan is updated, since I did not verify which artifact produced them.

**Note added by the Orchestrator, not Fable:** Fable's closing line above is explicit that the
beam-4-vs-greedy question is *not* something it adjudicated — it was relayed as unanimous
background in the brief, not measured or ruled on by Fable itself. The actual beam-4 finding
transcribed into §16.30 item 4 is a **joint-architect finding**: both `architect` and
`rust-engineer`, working independently in separate fresh sessions with no shared context, each
re-derived it from scratch (reading `kha-white/manga-ocr-base`'s `config.json`, the pinned
mayocream export's `generation_config.json`, and `manga_ocr/ocr.py`'s `generate()` call; running a
beam-4 and a greedy decode over the model's own 12 published labels: beam 12/12, greedy 10/12,
both agents' numbers agreeing exactly). Because this is agreement between the two Opus subagents,
not a disagreement, it does not fall under CLAUDE.md's Fable-escalation rule — it goes through the
step-1a fresh-reader path like any other joint-architect conclusion, which is why it is transcribed
into the same §16.30 entry as Fable's three rulings but is labeled separately within it.

---

## P7c infra — signature schema, pin home, frozen provenance predicate — Fable tie-break, 2026-08-02 (record quality: MIXED — see the scope note immediately below; do not read the phrase "captured directly" as covering this whole entry)

**Scope of the verbatim claim, stated precisely so this entry cannot be over-cited (same convention as this file's prior P7 entry above).** Only the `>`-blockquoted passage under "Fable's reply, verbatim" below is the adjudicator's own text, captured directly rather than paraphrased, per this file's own "for future rulings" instruction — the one edit made to it is removal of a closing usage/token-count line, which is run infrastructure metadata, not part of the ruling. The "Question put to Fable" framing immediately below is the Orchestrator's own prose, carrying the same paraphrase-not-source status as every other passage in this file per the header note at the top. A citation of this entry that quotes outside the blockquote is quoting the Orchestrator, not the adjudicator.

**Question put to Fable** *(Orchestrator's framing, not Fable's text)*: task P7c needs to record the ONNX graph signatures of the two pinned manga-ocr artifacts (encoder/decoder), extending `xtask`'s existing detector-only signature-recording infrastructure — which cannot represent symbolic ONNX dimensions, and both models' graphs are mostly-to-entirely symbolic. `architect` and `rust-engineer`, each blind to the other, converged on two points (a new `ocr_model_signature` provenance group rather than widening the existing one; rejecting an in-place `Vec<i64>` → `Vec<Dim>` mutation of `TensorSignature`, both having compiled that mutation against the real frozen test suite and gotten real `rustc` errors) but diverged on three others: (A) how to restructure the schema given that rejection — `architect` proposed a wholly separate parallel type set (`OcrModelSignature`/`OcrTensorSignature`), `rust-engineer` proposed refactoring the existing types into generics with type aliases (`TensorSig<D>`/`ModelSig<D>`, with `TensorSignature = TensorSig<i64>` etc.); (B) where the two artifact pins (sha256/size) live — `architect` wanted one canonical `pc-testkit` constant imported by both the recorder and the test, `rust-engineer` wanted the pins duplicated as independent literals in both places; (C) how to fix a frozen predicate (`assert_known_non_committed_form` in `crates/pc-testkit/tests/recorded_provenance.rs`) that rejects the two new artifacts' bare filenames as "not committed" references, since neither ends in `.pt.onnx` — `rust-engineer` proposed relaxing the suffix check to `.onnx` plus a new compensating digest table; `architect` proposed an enumerated 3-name allow-list. Fable was asked to rule on A, B, and C, verifying independently rather than only adjudicating prose.

**Fable's reply, verbatim** *(the blockquote below only — the "Question put to Fable" text above is the Orchestrator's framing, not Fable's)*:

> **Mode statement:** Mode A (tie-break, final call). Advisory-only status suspended for these three decisions. I wrote no files in the repository; all probes ran in a scratch clone and a scratch crate, and `git status --porcelain` on `/home/ducph/duc/panel-ocr` is empty.
>
> **Prior-ratification check (ran, relevant):** §16.30's own text reserves this ground rather than deciding it — *"The artifact digests in item 1 have no in-repo gate yet; giving them one is P7's job, not this entry's."* §16.16 item 2's reject-symbolic clause binds the `model_signature` **group's recorder**, not the Rust representation of signatures generally. §16.24 items 1(e), 1(f), 2, and cookbook rule 13's corollary bear directly on C (details there). No ratified clause decides A or B.
>
> ## What I actually ran
>
> 1. **Downloaded both pinned artifacts** from HF at commit `24b12778…`. Measured: decoder `ef7765261e9d…` / 117,480,262 B; encoder `15fa8155fe9b…` / 343,454,249 B — **both match §16.30 item 1's table exactly.**
> 2. **Parsed both graphs myself** with a standalone re-implementation of `xtask/src/model_signature.rs`'s protobuf walk, `parse_dim` widened to concrete-or-symbolic. Measured: encoder — 198 initializers, 1 true input `pixel_values` f32 `[batch_size, num_channels, height, width]`, output `last_hidden_state` f32 `[batch_size, Addlast_hidden_state_dim_1, Addlast_hidden_state_dim_2]`, **7 symbolic dims, zero concrete**; decoder — 2 true inputs (`input_ids` i64 both-symbolic; `encoder_hidden_states` f32 `[sym, sym, 768]`), output `logits` f32 `[sym, sym, 6144]`; **opset 14 and ir_version 7 on both**. The frozen `parse_dim` would bail on **both** files. Both agents' cross-confirmed graph data is confirmed a third time, independently.
> 3. **Applied the rust-engineer's generic refactor verbatim** to a scratch clone at HEAD (`1158809`): `cargo test -p pc-testkit --test model_signature -p pc-detect --test d4_signature` green with **zero edits to either frozen file**; full `cargo test --workspace` → **881 passed / 0 failed / 6 ignored**; `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean. Reverted the probe and re-ran: baseline is **identical, 881/0/6**. The engineer's headline claim is measured true, by me, not accepted from the transcript.
> 4. **Serde probe** on the generic types: untagged `Dim` round-trips (`768` stays a number, `"batch_size"` stays a string, re-parse equals original); the `ModelSignature = ModelSig<i64>` alias still **rejects** a JSON shape containing a symbolic string, exactly as the old `Vec<i64>` did; a JSON float is rejected; number→`Fixed`, string→`Symbolic`.
> 5. **Read the pin plumbing end-to-end** (`/home/ducph/duc/panel-ocr/crates/pc-models/src/lib.rs:26-31`, `xtask/src/model_signature.rs:76,220-226,279-284`, `xtask/tests/provenance_digests.rs` H2, `crates/pc-detect/tests/d4_signature.rs:62-84`, §16.16 item 5) to settle B's factual question.
>
> ## Ruling A — the schema: **the rust-engineer's generic refactor wins.**
>
> **Decision.** Restructure `crates/pc-testkit/src/model_signature.rs` as `TensorSig<D>` / `ModelSig<D>` with aliases `TensorSignature = TensorSig<i64>`, `ModelSignature = ModelSig<i64>`, `SymbolicTensorSignature = TensorSig<Dim>`, `SymbolicModelSignature = ModelSig<Dim>`, and `#[serde(untagged)] enum Dim { Fixed(i64), Symbolic(String) }`. Do **not** create a parallel `OcrModelSignature`/`OcrTensorSignature` type set.
>
> **Grounds.**
> - Measured, not argued: the refactor is behavior-preserving at zero test-edit cost — 881/881 workspace, clippy `-D warnings` clean, both frozen consumers (`d4_signature.rs`, `tests/model_signature.rs`) compile and pass unmodified. That is precisely §16.24 item 1's own acceptance standard for touching this territory ("green **with the test file unmodified**"), met here by measurement.
> - The architect's justification defeats itself. "Field-for-field parallel so a future unification is a rename, not a merge" prepays permanent duplication to make a future unification cheap — but the unification is available **now**, and I measured its cost at zero. Deferring a free merge and buying drift risk in the meantime is strictly worse.
> - The frozen test file's own header (`crates/pc-testkit/tests/model_signature.rs:4-6`) states the anti-duplication ground: *"Duplicating the shape in either consumer would let the two drift."* Two field-for-field-parallel type sets inside `pc-testkit` are the same drift vector one module over, and nothing would enforce the "parallel" property the architect's future-rename plan depends on. Cookbook rule 14: one typed definition, shared.
> - The safety property the architect's design buys (the detector schema cannot silently change meaning) is already held by the frozen tests: I verified `the_named_convenience_loader_matches_the_general_one` asserts `shape == vec![1, 3, 1024, 1024]` against the alias — flipping `ModelSignature`'s alias target away from `i64` is a compile error in a frozen file, so the alias is self-guarding.
>
> **Grafts (binding conditions on the win).**
> 1. The refactor must land **with additive tests** pinning what my probe verified, since none of the existing frozen tests covers the new surface: (i) `SymbolicModelSignature` round-trips a mixed symbolic/fixed shape; (ii) the `i64` alias still **rejects** a symbolic shape element — the old `Vec<i64>` rejected `"batch_size"` by type and the alias must observably keep doing so; (iii) `Dim` rejects a JSON float. All three pass under the probe; the implementer writes their own test text.
> 2. This ruling does **not** relax §16.16 item 2: the `model_signature` group's recorder keeps rejecting symbolic/dynamic/zero/missing dims. If `xtask`'s low-level walk is shared between the detector and OCR recorders, the detector path's symbolic-bail must remain and be provable (unit tests in the bin crate are established practice — `xtask/src/record.rs` has ~10).
>
> **Scope.** This ruling holds for the type definitions in `crates/pc-testkit/src/model_signature.rs` and the four aliases above. It does **not** decide: `xtask`-side code organization (shared walk vs. new parse path — either is fine so long as graft 2 holds); which module hosts the OCR group's path constants and loader (a new `ocr_model_signature.rs` module holding constants/loader/pins built **on the shared generic types** is compatible and keeps the architect's module boundary without the type duplication); any new schema fields (e.g., recording `ir_version` — neither position proposed it; adding one needs its own argument).
>
> ## Ruling B — pin home: **the architect wins. One `OcrModelPin` constant per file, two importers.**
>
> **Decision.** The encoder/decoder pins (file name, sha256, size_bytes) live once, as constants in `pc-testkit` (the architect's `MANGA_OCR_ENCODER`/`MANGA_OCR_DECODER`), imported by the `xtask` recorder (which verifies the real file against the pin **before** parsing, as `xtask/src/model_signature.rs:76` does today) and by `crates/pc-ocr/tests/p7_signature.rs`. Dependency edges verified: `xtask` already depends on `pc-testkit`; `pc-testkit` is already a dev-dependency of `pc-ocr` (`crates/pc-ocr/Cargo.toml`).
>
> **The load-bearing factual question, settled by reading the code:** the existing pattern is **one constant**. `pc_models::COMIC_TEXT_DETECTOR` is defined once (`crates/pc-models/src/lib.rs:26-31`) and imported by the recorder (`verify_sha256(&source, COMIC_TEXT_DETECTOR.sha256)`, plus the file-name check at `:279-284`), by the H2 gate (`xtask/tests/provenance_digests.rs::the_model_signature_source_digest_equals_the_pc_models_constant`), and by `pc-models`' own keystone. `d4_signature.rs:62-69` **deliberately does not** carry the digest literal, citing §16.16 item 5's "single source of truth" clause. The rust-engineer's characterization — two independently-typed literal copies as "the existing pattern" — is **factually wrong about the digest**, and with it goes the main ground for that position.
>
> **Graft from the losing position (binding).** The existing pattern is not constants-only: `d4_signature.rs:73-74` pins `size_bytes == 94_669_756` and `opset == 11` as **frozen-test literals**. Without that leg, a single edit to the pin constant plus a re-record against a different real file keeps every test green — nothing frozen anchors *which* artifact is pinned. So `p7_signature.rs` must assert the pin constants' identity fields (sha256, size_bytes) against the §16.30 item 1 literals, in the frozen test. That is duplication **with a comparator between the copies**, which is the H2 pattern — unlike the engineer's proposal, where the `xtask`-local and test-local copies were compared by nothing and a drift would surface only at the next re-record.
>
> **Scope.** This ruling holds for the two P7 pins (encoder/decoder sha256 + size) and their two named consumers (the `xtask` OCR recorder; `crates/pc-ocr/tests/p7_signature.rs`). It does **not** decide P8a's `ModelSpec` design; when P8a moves model identity into `pc_models`, it must either retire the `pc-testkit` pin or add an identity assertion binding the two — leaving both unbound would recreate exactly the drift this ruling exists to prevent. Record that obligation with P7c's transcription.
>
> ## Ruling C — the frozen predicate: **synthesis; neither option as stated is complete. Enumerated allow-list (architect's membership) + compensating digest leg bound to the pins (engineer's second half).**
>
> **Decision.** Ratify (joint architects, as both agents agree is required) an edit to `assert_known_non_committed_form` (`crates/pc-testkit/tests/recorded_provenance.rs:184-208`) replacing the `.ends_with(".pt.onnx")` suffix pattern with an **enumerated allow-list of exactly three bare filenames**: `comictextdetector.pt.onnx`, `encoder_model.onnx`, `decoder_model.onnx`. **And** add, additively in `xtask/tests/provenance_digests.rs`, a bidirectional gate asserting set equality between the bare-model-filename sources declared anywhere in the tree and that allow-list, with each one's `source_sha256` equal to its **pin constant** (`pc_models::COMIC_TEXT_DETECTOR.sha256` for the first — already H2 — and the Ruling-B `OcrModelPin` constants for the other two), not to a second independently-typed literal table.
>
> **Grounds.**
> - Against option (a)'s suffix widening: cookbook rule 13's corollary is near-verbatim on point — *"`ignore dotfiles` or an extension allowlist would have re-opened it silently… the exempt set is itself asserted: a stray file must not pass by merely looking exemption-shaped."* Widening `.pt.onnx`→`.onnx` converts a narrow accident of the detector's filename into a genuine extension allowlist. §16.24 item 1(e) already called loosening this exemption "the bypass wearing different clothes"; an enumerated set is the *opposite* move — it is **stricter than today's predicate**, not laxer, which is the direction every ratified precedent here prefers.
> - Against option (b) as the architect left it: the rust-engineer's objection is correct — membership without a digest binding means `encoder_model.onnx` declared with a garbage `source_sha256` passes the frozen gate (it never hashes non-committed sources). But the engineer drew a false dichotomy: the compensating leg is not intrinsic to suffix-widening; it attaches to the allow-list just as well. Take it.
> - One correction to the engineer's compensating leg: a **literal table** in the xtask test would re-introduce B's rejected two-copies-no-comparator shape. Bind to the pin constants instead; the frozen-test literal leg from Ruling B is what anchors the constants themselves.
> - Housekeeping the ratification must authorize explicitly: the comment at `crates/pc-testkit/tests/provenance_schema.rs:587-590` describes the exemption as "keyed on a single path component ending `.pt.onnx`" and goes stale under this edit — a comment-only correction, same class as §16.29 item 2's precedent. The `EXPECTED_GROUPS`/`EXPECTED_DECLARED_PATHS` additions for the new group are **already authorized** additive edits under §16.24 item 1(f) (retain every existing entry verbatim; constants stay literal) and need no new ratification.
>
> **Scope.** This ruling holds for `assert_known_non_committed_form`'s bare-model-filename branch and the three named files. It does **not** widen the scratch-prefix branch, does not touch the key-name walker (§16.24 item 1(b): neither deleted nor patched), and does not pre-authorize any fourth filename — a future model file needs another explicit allow-list edit plus its own pin, by design.
>
> **Verified vs. accepted.** Verified by execution: the artifact digests/sizes; both graphs' full I/O signatures and the frozen `parse_dim` bailing on them; the generic refactor's 881/0/6-with-zero-test-edits and clean clippy, against a same-tree 881/0/6 baseline; the `Dim` serde semantics; the single-constant pin pattern. Accepted without re-execution: that the naive in-place `Vec<i64>`→`Vec<Dim>` mutation fails to compile in the frozen files (both agents compiled it; consistent with my reading of `d4_signature.rs:198`'s `Copy` deref and `:206-208`'s `columns - 5`, and it is settled per the brief); that "pins must not live in `pc_models` before P8a" is agreed ground. Rulings A and B's decisions rest on measurements above; Ruling C rests on ratified-text reasoning (rule 13's corollary, §16.24 item 1(e)) rather than an experiment — there is no runnable oracle for a governance-shape choice, so mark C as the reasoning-based ruling of the three. Upstream PanelCleaner was not consulted for these three questions: none is a ported-behavior ambiguity; the tiebreak oracle for artifact facts here is the pinned files themselves, which I ran.

**Note added by the Orchestrator, not Fable:** all three rulings are transcribed into a new §16.x spec entry as their own ratification, separate from §16.30 (which explicitly reserved this ground: "giving them [the artifact digests] one [an in-repo gate] is P7's job, not this entry's").
