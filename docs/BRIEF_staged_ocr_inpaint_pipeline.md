# Brief — staged OCR‖LaMa pipeline (joint architect + rust-engineer design pass)

## Origin and framing (read this before the "current facts" section)

User-proposed optimization, refined over several rounds of investigation this session.
Original framing ("overlap OCR of image N+1 with LaMa-inpaint of image N because LaMa is
slower") was **corrected by the user**: this is not about which stage is faster. The
correct framing is a genuine **staged/pipelined execution model** — a dedicated OCR-stage
worker and a dedicated LaMa-stage worker, each continuously fed one image/region at a
time from its own queue, running concurrently. Once both stages are kept continuously
busy, batch throughput is bound by `max(stage_cost)` across the run, not by which stage
is faster or by their sum — so the earlier "is OCR or LaMa dominant" debate is irrelevant
to whether this is worth doing. It only matters for estimating the *size* of the win.

CPU-only in scope: GPU-2/3/4 already shipped real, measured CUDA paths for every stage
(§16.49) and are explicitly out of scope here — the user confirmed "gpu thing is done
here," and the whole `PERFORMANCE_BACKLOG.md` investigation this session's numbers come
from is scoped CPU-only ("focus on CPU for now"). Do not extend this design to CUDA
without a separate measurement — GPU compute has none of the idle-core headroom the CPU
case relies on (a single CUDA kernel typically saturates the device), so a CPU-proven
overlap may buy nothing on GPU. State explicitly in the plan whether the design should be
CPU-only-gated at the type/config level, or left device-agnostic pending a future
measurement — don't leave it undecided by omission.

## What's actually being asked

Is a real producer-consumer staged pipeline (OCR-stage and LaMa-stage each running as
independent, continuously-fed workers across a batch of images) structurally feasible as
an evolution of `pc-pipeline`, and if so, what does the minimal viable design look like?

## Current architecture — verified directly by reading the code, not assumed

- `crates/pc-pipeline/src/single.rs:274` (`run_stages`) is **one straight-line
  synchronous function per image**: detect → preprocess (OCR happens *inside* this
  stage, gated by `performing_ocr`/`ocr_enabled`) → mask → denoise → inpaint. Each stage
  is a blocking call feeding the next. No queue, no channel, no stage-owning thread
  exists anywhere in this crate today.
- The **only** concurrency primitive in the whole pipeline is
  `crates/pc-pipeline/src/batch.rs:66-119` (`run_batch`)'s `rayon::par_iter` over
  **whole images**, on a pipeline-owned `rayon::ThreadPoolBuilder` sized by
  `options.threads` — `DEVIATION(11)`, explicitly documented as a deliberate departure
  from upstream PanelCleaner's own per-*stage* Python multiprocessing pools (upstream
  needs process pools because Python's GIL makes threads useless for CPU-bound work;
  this Rust project chose per-image thread parallelism instead when it was designed).
  Each rayon worker runs one image's entire `run_stages()` alone, start to finish.
  Today's "OCR‖LaMa overlap" only happens incidentally when two different images'
  workers land on different stages at the same moment — nothing coordinates or
  guarantees it, and nothing keeps either stage continuously busy.
- The graphify AST graph (`graphify-out/graph.json`, `GRAPH_REPORT.md`, rebuilt this
  session, scoped to `crates/`+`xtask/`) independently confirms this: `run_stages`,
  `process_image`, `PipelineCtx`, `run_inpaint`, `Step`, `single.rs`, `options.rs` all
  cluster into **one community** — a tightly-coupled blob, not five separable
  stage-modules with clean seams. Use this graph as supporting material; it's already
  built and current as of this session.
- OCR (`crates/pc-ocr/src/onnx.rs:60-61`) and LaMa
  (`crates/pc-inpaint/src/onnx.rs:253`) each already sit behind their own **independent**
  `Mutex<Session>` — so "at most 1 concurrent `Run()` per model" is already mechanically
  enforced today, coincidentally rather than by design. This is a real asset for a
  staged design (the safety cap you'd want already exists) — confirm in the plan whether
  it's sufficient on its own or whether the new architecture needs its own explicit
  concurrency cap independent of this incidental one.
- `pc_inpaint::inpaint_page` is **already callable fully standalone** — proof:
  `panel-ocr inpaint` (`crates/pc-cli/src/standalone_inpaint.rs`) derives regions from a
  user-painted mask and calls it directly, with no detector, no OCR, no `pc-pipeline`
  involvement at all. `pc_ocr` is *not* fully standalone the same way: `panel-ocr ocr`
  (`crates/pc-cli/src/lib.rs:130`, `run_ocr`) still runs through
  `pc_pipeline::process_image` and requires the Detect stage to run first, since OCR
  consumes the text boxes Detect produces — a real data dependency, not an artificial
  one. Neither of these two standalone CLI paths runs concurrently with anything; they
  are separate single-shot invocations, cited here only to establish that the two
  crates (`pc_ocr`, `pc_inpaint`) are not hard-wired to each other or to the rest of the
  chain at the function level — the missing piece is purely the batch-level orchestrator.

## Constraints a staged design must account for (ratified/frozen, not optional)

- **`BatchSummary.outcomes` is in input order, independent of `options.threads`**
  (§5.7 / §16.12 item 17, frozen test). A staged design must preserve this even though
  images may now *finish* their stages in a different order than they started.
- **Per-image panic isolation** currently happens at the whole-image boundary
  (`process_image_isolated`, `batch.rs:31-51`, `DEVIATION(10)`). If an image's stages
  run on different worker threads, define what "isolate one bad image" means when a
  panic happens mid-flight in, say, the OCR-stage worker while that image's LaMa work is
  still queued or already dispatched elsewhere.
- **Checkpoint writes are currently interleaved inline** between stage calls inside
  `run_stages` (disk mode only — see `checkpoint::write_page_raw`,
  `write_page`, `write_mask_data` calls in `single.rs`). A staged design must still
  write checkpoints at the right stage boundaries without serializing on them.
- **`process_image_with_splitting` sits above `run_stages`** (long-strip segmentation,
  §16.14 item 1): one image can become multiple segments, each running the full chain,
  stitched once at export. Decide what "one work item" means for a stage queue when an
  input image isn't 1:1 with a pipeline run.
- **`fail_fast`/`run_fatal` semantics** (`batch.rs`'s `AtomicBool` gates): a run-fatal
  failure (e.g. detector/inpainter provisioning failure) currently stops *new* images
  from starting while letting in-flight ones finish. Define the equivalent for a staged
  pipeline — does a run-fatal LaMa failure drain the OCR-stage queue, stop it
  immediately, or something else?
- **Reversing `DEVIATION(11)`** (or partially reversing it — e.g., stage-pipelining
  layered *on top of* per-image rayon parallelism, vs. replacing it outright) is itself
  a deviation-registry-level architecture decision. If the joint plan lands on reversing
  or narrowing it, that needs its own ratification (§16.x), not a quiet implementation
  change — flag this explicitly rather than treating it as a normal code change.

## Measured data already available — reuse, don't re-derive

- `docs/PERFORMANCE_BACKLOG.md`'s "NEW LEVER, 2026-08-17" entry: a throwaway 2-call test
  found OCR‖LaMa concurrent execution "nearly free" (27.579s vs theoretical best
  27.253s, ~1.2% overhead) — **but the box/tile count behind `ocr_alone=3.919s` and
  `inpaint_alone=27.253s` is unstated and likely unrepresentative** (implies ~2 OCR boxes
  vs ~2 LaMa tiles; a real page can have 10+ boxes). Re-run this with a real page's
  actual box/tile distribution before trusting the "nearly free" conclusion generalizes.
- Already ruled out, don't re-try: **3 concurrent same-model OCR sessions measured
  1.76x slower** (memory-bandwidth/cache contention, not thread-pool scheduling — the
  oversubscription theory was checked directly and rejected). Capping `intra_threads=1`
  on those 3 sessions measured **worse** (0.37x vs 0.44x uncapped) — this rules out
  thread-capping as a fix for the *same-model* concurrency case specifically.
- **Not yet tested, and directly relevant to a "run OCR at 1 thread so it doesn't fight
  LaMa for cores" idea**: nobody has measured capping OCR's own `intra_threads` to 1
  while a real LaMa call runs concurrently (a *different* model, not another OCR
  session). The same-model capped result above does not necessarily generalize to this
  pairing — this is the cheapest, most direct next experiment and should ground the
  plan rather than be assumed either way.
- All three model-backed stages already build their session once per process and reuse
  it for the whole batch (`OnceLock`-latched detector/inpainter, eager up-front OCR) —
  no rebuild-per-image concern for a staged design to worry about.

## What the joint plan needs to produce

1. Whether a producer-consumer staged pipeline (dedicated OCR-stage worker(s) + dedicated
   LaMa-stage worker(s), each fed by its own queue) is achievable as an evolution of
   `pc-pipeline`, or requires a substantially different structure. Sketch the minimal
   viable shape (e.g., channel-based stage workers, one thread/pool per model-backed
   stage — detect, OCR, inpaint — with CPU-only glue stages like mask geometry/denoise
   placed wherever they naturally fit best).
2. Whether this replaces `DEVIATION(11)`'s per-image rayon model outright, or composes
   with it (e.g. a shared stage-pipeline server the whole batch feeds into, vs.
   per-rayon-worker mini-pipelines).
3. How `BatchSummary`'s determinism guarantee and the panic-isolation boundary need to
   be redefined for the staged shape chosen.
4. Whether a new frozen determinism gate is needed (byte-identical output regardless of
   stage-queue scheduling/timing), analogous to Annotation's A5 gate (§16.37 item 8).
5. An explicit CPU-only-vs-device-agnostic scoping decision (see "Origin and framing"
   above), stated as a decision, not left implicit.
6. Task breakdown per this project's standard planning format (scope/interfaces touched,
   draft test code per task tied to a concrete requirement, simple-vs-heavy
   classification) — per CLAUDE.md's Spec-sensitive tier, since this touches
   `BatchSummary`'s frozen guarantee and a registered deviation.

If the two of you disagree on approach, per standing process the Orchestrator will
convene `fable-adjudicator` to tie-break — do not compromise on a blended answer
yourselves.
