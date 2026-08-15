# Brief: joint plan — OCR beam-batching (Spec-sensitive tier)

Not temporary — this is the planning brief for the task itself; keep it until the task
is implemented and reviewed, then it can be retired like other consumed briefs.

## Context (read the sources, don't take this brief's word for it)

- `docs/PERFORMANCE_BACKLOG.md` (top section) and `docs/WORKSTATE.md`'s sixth 2026-08-15
  entry — the Fable advisory consult that surfaced this.
- `crates/pc-ocr/src/decode.rs` — the beam search implementation (`beam_search`,
  `LogitsSource` trait, `MANGA_OCR_BEAM_CONFIG` with `num_beams: 4`).
- `crates/pc-ocr/tests/p7_decode.rs:220-260` — **the frozen test that pins the current
  behavior**, `the_loop_reseeds_with_the_configured_start_token_and_reruns_the_full_prefix_each_step`.
  Its own comment states the constraint's source: *"The pinned decoder ONNX export has NO
  KV-cache input (P7c's recorded signature: only `input_ids` + `encoder_hidden_states`), so
  every step must present the FULL prefix seen so far."*
- `crates/pc-ocr/src/onnx.rs:189-277` — the decoder's actual ONNX call site.
  **Line 221 hardcodes the input tensor shape to `[1, input_ids.len()]`** — batch size 1.
  Whether the pinned decoder.onnx graph would accept batch > 1 is currently unverified —
  nobody has tested it.

## What's actually being asked, and what is explicitly NOT being asked

**In scope**: determine whether batching the 4 beams into one decoder call per step
(instead of 4 sequential single-prefix calls) is (a) technically possible against the
*pinned* decoder artifact, and if so, (b) whether it's worth doing, and (c) what the
correct implementation shape is.

**Explicitly NOT in scope for this plan**: adding a KV cache to the decoder. That would
require a different ONNX export of the model entirely (the pinned artifact has no
`past_key_values` input), which is a much larger undertaking — re-exporting from the
original manga-ocr HuggingFace checkpoint, re-verifying against upstream, re-pinning a
digest, re-running the golden/divergence measurements. If the joint plan concludes this
is worth pursuing anyway, say so explicitly as a separate, much heavier follow-on item —
do not fold it into this task's scope.

**Also explicitly NOT in scope**: int8 quantization (Fable's second suggestion). Separate
lever, separate plan, not this task.

## Required first step: a spike, before any implementation plan

Mirror `§16.38`'s L0 spike for LaMa's own batch-dimension question exactly (same
methodology, cited in `docs/PIPELINE_SPEC_V1.md` line 7571: fed 256×256/640×512/1024×1024
inputs, measured real pass/fail and real timing, before any plan was written). For the
OCR decoder:

1. Using the real cached decoder model (`PANEL_OCR_MANGA_OCR_DECODER` env var convention,
   same as `crates/pc-ocr/tests/p7_session.rs`), attempt a real `ort::Session::run()` call
   with `input_ids` shaped `[2, seq_len]` or `[4, seq_len]` instead of `[1, seq_len]`.
   Does it succeed or fail? If it fails, what's the exact error (mirroring how LaMa's
   spike recorded the exact `"Got invalid dimensions..."` failure text)?
2. If it succeeds: measure real wall-clock time for batch=1 vs batch=4 (same shape as
   LaMa's 1.58s/3.03s/6.55s table) to determine whether batching actually helps or
   scales linearly like LaMa's did. **Do not assume it helps just because it's a
   transformer and LaMa is a CNN** — that's a plausible hypothesis, not evidence, per
   this session's own established discipline of measuring before concluding.
3. If it fails or doesn't help: the plan should say so plainly and either propose the
   correct alternative (if any) or conclude this specific fix is not viable, rather than
   forcing an implementation that measurement doesn't support.

## If the spike supports proceeding

Draft the actual task breakdown and the actual test code (per this project's pipeline —
the Rust Engineer drafts real test code, not just descriptions), including:

- What happens to the frozen `p7_decode.rs` test — per CLAUDE.md, a test contradicting
  the spec goes back to a joint architect ruling, never a unilateral edit. If the
  current sequential-per-beam test needs to change to reflect batched calls, that
  decision and its replacement test content are this plan's job to produce, not
  Codex's/cmdc's.
- Whether this is upstream-parity-restoring or a new deviation needing its own
  `DEVIATION(n)` register entry — check against real upstream manga-ocr behavior
  (this project's own convention: consult the upstream implementation by running it,
  not just reading it, per `CLAUDE.md`'s Notes section and cookbook rule 3).
- Whether `docs/OCR_DEVICE_DIVERGENCE.md`'s recorded CPU/CUDA divergence numbers need
  re-measurement after this change (they were measured against the current, unbatched
  call shape).
- Task sequencing and heavy/simple classification per CLAUDE.md's batching rule.

## If the two of you disagree

Standard pipeline rule: the Orchestrator does not pick a winner. Report the disagreement
plainly (both positions, where they diverge and why) and it goes to Fable to adjudicate.
