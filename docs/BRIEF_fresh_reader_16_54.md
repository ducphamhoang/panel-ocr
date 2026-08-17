# fresh-reader brief — §16.54 transcription review

## What to review

`docs/PIPELINE_SPEC_V1.md:9889` onward — the new `## 16.54` section (ends right before
`## 16. Summary of what v1 is NOT`). This is a ratification transcription of a Fable
tie-break between two independent Opus (architect + rust-engineer) planning passes on a
staged OCR‖LaMa pipeline design. Per this project's step-1a rule, the *transcription*
needs a reader before commit — not only the ruling it records.

## Ground truth to check the transcription against (read these, don't trust the
section's own summary of them)

1. `docs/BRIEF_staged_ocr_inpaint_pipeline.md` — the original design brief both Opus
   passes were dispatched against.
2. `docs/BRIEF_fable_stagegate_tiebreak.md` — the framing dispatched to Fable, and what
   the two positions converged on / disagreed on, as the Orchestrator summarized it.
3. `docs/PIPELINE_SPEC_V1.md:440-442` (§4.5) — the "semaphore-free bound" clause, to
   check the verbatim quotes in item 4 are exact, character for character, and that the
   Orchestrator's/Fable's reading is represented accurately (narrow: describes only the
   image-level pool-sizing bound; does not forbid a pipeline-level admission gate).
4. `docs/PIPELINE_SPEC_V1.md`'s §16.32 section — the detect worker-thread pattern the
   rejected design ("Design B") wanted to generalize; check item 3's grounds (the
   confinement is a ratified remedy for a detect-specific DAZ/FTZ hazard, not a house
   style) accurately reflect what §16.32 actually says.
5. `crates/pc-pipeline/src/batch.rs`, `ctx.rs` — check the "already approaches
   `max(stage_cost)`" claim's supporting facts (rayon `par_iter` over whole images,
   `DEVIATION(11)`) and `PipelineCtx`'s `#[derive(Clone, Copy)]` (item 5's graft about
   not silently dropping `Copy`) are both real, at the line numbers/facts claimed.
6. `crates/pc-ocr/src/onnx.rs`, `manga.rs`, `decode.rs` — check item 3's claim that OCR's
   `decode_step_batch` loop (not a single request/response call) is why generalizing the
   detect worker-thread pattern to OCR is costlier than the rejected design's "zero
   pipeline lines" framing implies.
7. `crates/pc-inpaint/src/onnx.rs` — check the single-`Mutex<Session>` claim.

## What to check specifically

- **Every Fable quote is character-for-character exact** against the task-notification
  text the Orchestrator received (available in conversation if you need to cross-check
  further than this brief allows) — this project's transcription rule requires quoting
  scope verbatim, not paraphrasing.
- **No accidental supersession marker was introduced.** Search §16.54's own text for the
  trigger tokens (`SUPERSEDED`, `NARROWED`, `QUALIFIED`, `ERRATUM`, `amended`,
  `withdrawn`, `RE-GROUNDED`) and confirm none appear as a supersession-claim usage. Note:
  the Orchestrator already caught and fixed one real hit here — "amended" landing on the
  same physical line as "§4.5" tripped `spec_supersession.rs`'s Layer-B scanner and was
  reworded to "revised" (confirm this reads correctly and no other instance was missed).
- **Item 2 (Ruling 0)'s "unconditionally committed task is a measurement" framing is
  honest** — i.e. the entry does not quietly commit to building `StageGate` regardless of
  what the measurement shows. Check the closing item 6 ("what this entry does NOT
  decide") doesn't contradict item 2's conditionality.
- **Item 3 (Ruling 1)'s scope quote is exact and item 6 doesn't silently re-open it** —
  e.g. confirm detect's §16.32 mechanism is genuinely untouched by this entry, and that
  "capacity >1" is correctly named as NOT authorized here.
- **Item 4 (Ruling 2) doesn't actually amend/revise §4.5's text** — check the entry
  itself, correctly, defers that revision to a future implementation-landing entry,
  and does not perform it now (the Orchestrator deliberately left §4.5's own text
  untouched pending that future entry — confirm this is what's written, not assumed).
- **Item 5's grafts are stated as binding, not optional** — check the wording doesn't
  soften Fable's "not optional cleanup" framing into something skippable.
- **Provenance paragraph honesty**: confirm it doesn't claim a benchmark was run (none
  was, by both Opus passes and by Fable, all self-disclosed) and doesn't overstate what
  "no implementation exists yet" means.
- **Numbers**: the a4d re-pin (9902→10086, 184 net lines) — recompute independently via
  `grep -n` for the exact pinned claim text and confirm it's a single occurrence, not
  assumed from the Orchestrator's own arithmetic.

## What NOT to do

Do not re-litigate Fable's rulings themselves, and do not re-run any measurement (P0/S0
hasn't happened yet — that's explicitly future work, not this review's job). This is
specifically a check of whether the *transcription* accurately and honestly represents
what was decided, per this project's step-1a rule. Report findings; do not edit the spec
file yourself.
