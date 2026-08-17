# Fable tie-break — where should stage-concurrency control live?

## Context, so this reads standalone

Two independent Opus passes (architect + rust-engineer) planned a staged OCR‖LaMa
pipeline for `panel-ocr` (brief: `docs/BRIEF_staged_ocr_inpaint_pipeline.md`, both full
reports available in the Orchestrator's conversation if you need them — this brief
extracts only the disagreement and the material each side needs to have it decided).

**Strong convergence, not in dispute — state this so you don't re-derive it:**

- Today's architecture (`pc-pipeline`'s `run_batch` = `rayon::par_iter` over whole
  images, `DEVIATION(11)`; each model-backed stage serialized by its own lock — detect
  on a dedicated worker thread per §16.32, OCR on two `Mutex<Session>`, LaMa on one
  `Mutex<Session>`) **already achieves close to `max(stage_cost)` throughput today**,
  under default config (`threads > 1`, `images > 1`), via lock contention acting as an
  incidental queue. Both sides derived this independently by reading the code, not by
  measuring — **neither side has run a real benchmark**, and both explicitly gate their
  own proposal on a measurement task (architect's "P0", rust-engineer's "S0") that is
  allowed to cancel the entire redesign if real headroom turns out to be negligible or
  a cheaper `--threads` increase already captures it.
- Neither proposes reversing `DEVIATION(11)` (per-image rayon parallelism stays).
- Both want the mechanism device-agnostic: CPU-measured, defaults to unbounded/no-op,
  so CUDA behavior is unchanged by default and no capacity is asserted for CUDA.
- Both want a new additive determinism gate (varying scheduling/capacity at fixed
  thread count, not just thread count itself, per §5.7/§16.12 item 17's existing gate
  which doesn't reach this axis).

## The actual disagreement — where does the concurrency cap live?

**Architect's "Design B": generalize §16.32's already-ratified pattern.**
`pc-detect` already has exactly this shape in production: a named worker thread
(`pc-detect-onnx`) owns the only `Session`, fed by `mpsc::channel`, with a
catch-forward-resume panic protocol. Architect proposes the same shape for `pc-ocr` and
`pc-inpaint`: replace each crate's internal `Mutex<Session>` with a `StageServer`
(worker thread + channel) **entirely inside that crate**. `Inpainter`/`OcrEngine` trait
signatures are unchanged; `pc-pipeline` is not touched at all — zero lines, per
architect's own accounting, and this is presented as the main argument for the design
(every frozen `pc-pipeline` test, `BatchSummary`'s guarantee, the panic-isolation
boundary, checkpoint interleaving, long-strip splitting — all untouched because nothing
in `pc-pipeline` changes).

**Rust-engineer's "Option A": an explicit, pipeline-owned admission gate.**
Add `pc_pipeline::gate::StageGate` — a counting permit (RAII guard) — as a new field on
`PipelineCtx` (`StageGates { detect, ocr, inpaint }`), acquired around each model-backed
call inside `run_stages`. The existing internal `Mutex<Session>` in `pc-ocr`/`pc-inpaint`
is left exactly as-is, untouched — this adds a *second*, independently-configurable cap
in front of it, at the pipeline layer where it can be tested, tuned via config, and
observed (`in_flight()`), rather than living as an invisible implementation detail of
another crate. Rust-engineer's stated reason for preferring this over touching the
engine crates: "the locks are `pc-ocr`/`pc-inpaint`-internal and invisible to
`pc-pipeline`: nothing at pipeline level can observe, test, configure, or schedule
against them" — i.e. architect's Design B doesn't actually fix the thing rust-engineer
thinks is broken (capacity is unexpressible, hard-coded at exactly 1 forever; no pipeline-
level test can pin "at most one LaMa inference at a time" because the property is
private to another crate).

**These are not compatible as written — pick one, or state explicitly if a real hybrid
resolves it** (e.g., does architect's `StageServer` need to also expose its own capacity/
`in_flight()` as a public, pipeline-testable property, which would make it equivalent to
rust-engineer's requirement without adding a second layer? Neither side considered this
combination — if it's the right answer, say so, but verify it actually satisfies both
sides' stated requirements rather than asserting it as an easy synthesis.)

## The spec-text sub-question, self-escalated by rust-engineer (do not let it go unresolved)

`docs/PIPELINE_SPEC_V1.md` §4.5 (line 440-442), quoted in full:

> `rayon` `par_iter` over images at the *whole-pipeline* granularity (not per stage),
> with a semaphore-free bound: `min(config.max_threads_or_cpus, images.len())`.
> Rationale: upstream parallelizes per-stage with a process pool because Python needs
> it; per-image parallelism in Rust gives better cache locality and makes per-image
> error isolation trivial. The detect stage is the exception: one `ort` session is
> created per run, but it is owned by a single dedicated worker thread
> (`pc-detect-onnx`, §16.32) and no rayon worker ever touches it directly. What the
> rayon workers share is the `Arc<dyn TextDetector>`: each sends its input tensor to
> the worker over a channel and decodes the reply on its own thread. `ort::Session`
> being `Send + Sync` for the CPU EP is still true and is no longer the reason anything
> works. `text_detector.concurrent_models` still controls how many sessions are
> created (default 1, shared), and a value greater than 1 is warned-and-ignored in v1.

Rust-engineer flagged this rather than deciding it themselves (correct call, per this
project's standing rule that an implementer/planner who thinks a spec question needs a
ruling stops and reports rather than picking the reading that favors their own design):
does "a semaphore-free bound" describe *only* the specific image-level pool-sizing bound
named right after it (`min(threads, images.len())`) — the Orchestrator's own reading, on
plain grammar, since the clause's referent is "the bound" of that one sentence — or does
it state a broader constraint against any semaphore-shaped construct anywhere in the
batch execution path, which would bear directly on rust-engineer's `StageGate` proposal
(a semaphore, deliberately placed inside the batch path) and would need either an
amendment or a scoping clarification before that design could proceed as written?

Note for your own calibration: architect's Design B never touches `pc-pipeline`, so this
clause is not in tension with it either way — this sub-question only matters if you rule
for rust-engineer's Option A, or some hybrid that puts a gate in `pc-pipeline`.

## What to rule on

1. **Design B (StageServer inside `pc-ocr`/`pc-inpaint`) vs. Option A (`StageGate` in
   `pc-pipeline`) vs. a real hybrid** — pick one, with reasoning, the same standard this
   project's tie-breaks are held to (build/verify something concrete where you can,
   per this project's own conventions for adjudication, rather than paper-reasoning
   alone, if a cheap check is available to you).
2. **§4.5's "semaphore-free bound" scope** — narrow (image-level pool sizing only) or
   broad (no semaphore anywhere in the batch path)? State which reading you're ruling
   under and why, since it's load-bearing for whichever design wins.
3. Anything else in the two positions that should be grafted from the losing side into
   the winning one (architect explicitly names "the two good ideas" pattern used in this
   project's prior tie-breaks — do the same here if applicable).

Full agent reports (verbatim) are available from the Orchestrator on request if you need
more than this brief's extraction — ask if the summary above is insufficient to rule.
