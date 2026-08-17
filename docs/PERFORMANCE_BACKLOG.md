# CPU performance backlog

## STATUS, 2026-08-17 — denormal-flush fix (§16.52 D0-D2) shipped, bit-exact; D3's residual-growth diagnostic found a real but different, unexplained pattern; D4 stays deferred

The detector's denormal-flush regression (batch-mode CPU, 1.66x-5.3x slower than separate
single-page processes) is **root-caused, fixed, and verified bit-exact**: §16.52 ratified
the design (a scoped RAII MXCSR guard, `pc_ort::denormal`, reapplied on every single
inference rather than relying on ONNX Runtime's process-wide once-flag). D0 (`4f4f36b`),
D1 (`1be086f`), D2 (`1bf7b52`) are implemented and committed. D2's real-weights re-
measurement found **bit-identical output across every thread count (0/1/8) and both
flush states**, against the actual committed fixture, not just self-consistency — closing
the exact evidence gap §16.52 item 1 flagged (that the pre-fix tests may have compared
FTZ against FTZ due to test-execution ordering).

**D3 (residual ~8x slowdown diagnosis, 2026-08-17): the original theory doesn't survive,
and a real but different, unexplained growth pattern is now confirmed instead.** With
D1's fix genuinely reapplying the flush guard on every call (not just once at
construction, and confirmed leak-free by D2), the caller-thread-denormal-leak explanation
for any residual growth no longer applies at all — so if growth still appears, it isn't
that. A throwaway diagnostic (10 consecutive real `detect()` calls, same detector
instance, real cached weights) measured:
  - `intra_threads=0`: `[0.57, 0.48, 0.45, 0.47, 0.46, 0.63, 1.68, 1.55, 1.55, 1.62]`
    (seconds) — a genuine step-function jump after ~5-6 calls, **2.85x** growth
    (last/first), not noise (the jump is consistent across calls 7-10, not a single
    outlier).
  - `intra_threads=1`: `[4.68, 4.45, 4.60, 4.44, 4.40, 4.40, 4.47, 4.46, 4.40, 5.89]` —
    essentially flat except one outlier at the very last call, **1.26x** nominal growth
    but not a real trend.

  **This is the opposite of what the original "residual is unexplained, possibly the
  intra-op pool" framing predicted ruling out**: growth is real specifically where a
  thread pool exists (`intra_threads=0`) and largely absent where it doesn't
  (`intra_threads=1`) — but this does NOT revive the pool-contention theory as originally
  stated (architect already showed, by reading ONNX Runtime's pinned source, that the
  pool's own denormal flag is unconditioned by the once-flag and was never the leak
  mechanism). What's growing here is unrelated to denormal correctness (D2 already proved
  zero bits move) — candidate causes not yet distinguished: ONNX Runtime's intra-op pool
  warming up its spin-then-block behavior differently over repeated calls, CPU frequency
  scaling/turbo-boost throttling down under sustained multi-core load, or some other
  session-lifetime state that only accumulates when multiple pool threads are active.
  **Left open, not closed** — this is a maintainer-local measurement (no gate, no frozen
  test), not required to close §16.52, and the throwaway test file was deleted after use
  per this project's convention.

**D3 follow-up (2026-08-17, same day): tested whether forcing `SessionTuning::
intra_op_spinning = Some(true)` (never let ONNX Runtime's intra-op pool block/sleep)
eliminates the growth. Result: inconclusive in the hoped-for direction, and the tradeoff
is bad enough that it argues against pursuing this lever further.** `SessionTuning`
already exposes `intra_op_spinning: Option<bool>` (currently `None`, deferring to ORT's
own default), so this was a config-only experiment, no code change. Same 10-call
methodology, same real cached weights, `intra_threads=0`:
  - Baseline (`intra_op_spinning=None`): `[0.48, 0.43, 0.42, 0.42, 0.43, 0.44, 1.23, 1.69,
    1.55, 1.54]` — reproduces the step-function growth, **3.21x**.
  - Forced spin (`intra_op_spinning=Some(true)`): `[3.26, 3.15, 3.10, 3.16, 3.14, 3.09,
    3.47, 3.90, 4.31, 4.10]` — **every single call is 7-10x slower in absolute terms**
    (3.1-4.3s vs. 0.4-1.7s), and growth is NOT eliminated — it's smoother (no sharp
    mid-run step) but still real (3.09s → 4.30s across the run), **1.26x**.

  **This does not confirm the pool's spin-vs-block transition is the (sole) cause, and it
  rules out `intra_op_spinning = Some(true)` as a viable fix regardless of cause** — the
  absolute-time cost is far larger than any growth-smoothing benefit. The fact that growth
  persists even under forced spinning (just smoother) is if anything mildly more
  consistent with the CPU-frequency-throttling candidate than the thread-pool-warmup one
  (sustained spin-waiting keeps the CPU busier for longer, which would produce more
  thermal throttling, not less) — but this is not a confirmed mechanism, only a
  plausibility note. **Not pursuing this lever further**: the config knob that could have
  been a cheap fix turned out to be a bad trade, and distinguishing CPU throttling from
  ORT-internal state now needs real-time frequency/thermal tooling (Intel Power Gadget,
  HWiNFO, or `Get-Counter '\Processor Information(_Total)\% Processor Performance'`) that
  this investigation has twice now deferred rather than attempted. Throwaway test file
  deleted after use, no code changes, no commits.

**D4 (pc-ocr/pc-inpaint denormal exposure) is NOT newly motivated by D3's finding.**
Fable's ruling 3(b) deferred D4 specifically to its own future denormal-exposure
measurement; D3's finding is a different phenomenon (a thread-pool/thermal growth
pattern, not a denormal-flag leak), so it provides no new evidence either for or against
whether `pc-ocr`/`pc-inpaint` are exposed to the *denormal* issue specifically. D4 stays
deferred exactly as ruled, on its own merits, not accelerated or further delayed by this
finding.

## STATUS, 2026-08-16 — OCR beam-batching (B1-B3) shipped; B4 deferred

The decode-loop fix this backlog's headline finding pointed at is **implemented and
committed**: §16.51 ratified the design, B1 (`db64059`) and B2+B3 (`de99274`) shipped it,
verified against the real pinned weights (bit-identical batched-vs-single logits, and an
end-to-end match against real upstream `manga_ocr`'s own decoded text). Real measured
speedup: ~1.1x-1.6x, degrading with sequence length — not the ~4x this doc originally
guessed from call-count alone (see §16.51 item 4 for the corrected, measured numbers).

**B4** (the `xtask`/`OCR_DEVICE_DIVERGENCE.md` diagnostic-tooling half of §16.51 item 6/7)
is **explicitly deferred, not forgotten**, by direct user decision: it only affects the
accuracy of a non-gating CPU-vs-CUDA comparison document, has zero effect on CPU behavior
or the shipped production code path, and needs real GPU hardware + cuDNN back on `PATH`
to attempt — none of which serves the standing "focus on CPU for now" priority. Pick it
up whenever GPU-side diagnostic accuracy matters again; nothing about B1-B3 depends on it.

**Naive OCR concurrency remains NOT shipped, and its root cause is now RESOLVED (2026-08-16): memory-bandwidth/cache contention, not thread-pool scheduling.** The discriminating experiment this doc named as the next step was run: 3 concurrent decoder sessions with `intra_threads` explicitly capped to 1 each (vs. the default/auto 0) measured **0.37x** — *worse* than the uncapped concurrent control's **0.44x**, both against a 1x sequential baseline. If burst-level ONNX-Runtime thread-pool contention were the cause, capping threads should have helped; it didn't, ruling that hypothesis out directly (not by elimination). **Conclusion: concurrent OCR sessions contend for memory bandwidth/cache, not CPU scheduling — this is not fixable by tuning thread counts, and further concurrency-based OCR speedup attempts on this hardware should not be pursued via that lever.** The speedup that WAS shipped (batching, B1-B3) is a different mechanism entirely — fewer, larger sequential calls, not concurrent execution.

## Fable advisory consult, 2026-08-15 — READ THIS FIRST, it reorders everything below

Consulted per this project's >5-iteration escalation rule (advisory only, no code — see
`CLAUDE.md`). Full findings in `docs/WORKSTATE.md`'s sixth 2026-08-15 entry; the headline:

**The biggest lever isn't concurrency at all — it's `crates/pc-ocr/src/decode.rs`'s beam
search calling the decoder once per beam per step, with no KV cache and the 4 beams never
batched into one call** (verified by reading `decode.rs:74-75`, `24-25`, `manga.rs:60`,
not assumed). Each OCR box costs roughly 4× seq_len full decoder forward passes,
re-processing the whole prefix every step. This is very likely the actual first-order
reason a box costs ~2.1s on CPU — independent of every machine/thread/concurrency
question below. Two concrete, bounded fixes, neither implemented: **(1)** batch the 4
beams into one decoder call per step (~4x fewer `Run()` calls — and arguably closer to
upstream manga-ocr's own HF `generate()` shape, not a deviation from it, though that
parity claim needs confirming against upstream, not assumed); **(2)** int8 dynamic
quantization of the OCR weights (helps regardless of which concurrency hypothesis below
turns out true). **Recommended priority: put this in front of the architects as its own
OCR-cost workstream before spending more effort on the concurrency mystery.**

On the two questions actually asked:
- **Q1 (historical ~9x gap)**: refined, not confirmed. Pure subtraction methodology
  can't shrink a whole-run wall-clock total by 9x, so "the diff method understated cost"
  (as this doc originally framed it) can't be the full story. Stronger candidate:
  **the 2026-08-11 "CPU" run may not actually have run on CPU** — its own ratios
  (GPU 2.3x/1.9x faster than "CPU") are physically implausible for a 512×512 CNN and a
  batch-1 transformer decode (normal range 10-50x; today's measured ~12.5x is the normal
  shape). One bounded forensic step, not yet done: check that revision's device-policy
  code for whether a CPU-labeled profile could still register CUDA, optionally rebuild
  and rerun. If confirmed, mark the historical CPU rows unreliable and stop chasing
  further — don't escalate to thermal/frequency profiling regardless of outcome.
- **Q2 (concurrency slowdown cause)**: hypothesis (a), a process-shared ONNX Runtime
  thread pool, is **settled FALSE by reading the pinned `ort` 2.0.0-rc.12 source directly**
  (`environment.rs:471,528`, `impl_commit.rs:155-156`): a global pool only exists if
  `with_global_thread_pool` is called, and nothing in this workspace calls it — each
  session owns its own pool. **But the "oversubscription is ruled out" conclusion from
  the fifth follow-up is narrower than stated**: the ~400ms `Get-Process` sampling
  measures *sustained average* utilization and can miss millisecond-scale burst
  saturation when three ~12-thread pools (36 threads on 20 logical processors, some
  landing on the i7-12700KF's slower E-cores) fire at once. Burst-level contention is
  NOT ruled out. Cheapest next experiment: rerun the same 3-engine comparison with
  `with_intra_threads(1)` per session — near-linear speedup at 3×1-thread points to
  burst/scheduling contention (tractable via thread caps + core affinity); still flat
  points to memory-bandwidth/cache contention (thread capping is a dead end there;
  the lever becomes fewer concurrent sessions or smaller weights).


Not started. This is a parking lot for a 2026-08-15 investigation into why CPU-executed
`clean` runs look far slower than `docs/WORKSTATE.md`'s 2026-08-11 per-stage timing entry
claimed, and what's actually worth trying next. See `docs/WORKSTATE.md`'s three
"same day, Nth follow-up" entries from 2026-08-15 for the full blow-by-blow; this document
is the distilled action list, not a replacement for that record.

**Scope note, so this doesn't get re-litigated from scratch:** GPU is fine, fast, and
matches history (§16.49/GPU-4's ratified CUDA path works correctly once cuDNN is on `PATH`
— see the `panel-ocr-cudnn-path` local Claude memory). Everything below is CPU-only, per
the user's explicit "focus on CPU for now."

## What's already established (measured, not assumed — don't re-derive)

1. **OCR is the dominant cost, not LaMa.** On a real 10-page batch (104 detected boxes),
   detect+mask cost ~8.6s; adding OCR brought the total to ~229s — OCR alone is ~220s,
   ~30x the historical `7.387s` figure for the same stage. Isolated via the same
   cumulative-diff method the historical entry used, on the *exact same* 10 pages.
2. **LaMa's per-call cost is fixed regardless of region size** — the model's ONNX input
   is hard-pinned to `[1, 3, 512, 512]` (`crates/pc-inpaint/src/onnx.rs:40-81`). Verified
   directly: a 20×20 mask and a full-image mask on the same page took 12.9s vs 13.2s,
   effectively identical. "Detect only the bubble, clean only the bubble" saves tile
   *count*, not per-tile *cost*.
3. **All three model-backed stages already build their session once per process and
   reuse it for the whole batch** — verified by reading the code and its tests, not
   assumed: detector (`OnceLock`-latched, single worker thread), OCR (eager, up front,
   before the image loop), inpainter (`OnceLock`-latched, a test asserts literal `Arc`
   pointer identity across 8 calls). There is no rebuild-per-image bug anywhere.
4. **Every model session serializes inference, regardless of thread count.**
   Inpainter and OCR encoder/decoder are `Mutex<Session>`; the detector uses a single
   dedicated worker thread processing requests one at a time
   (`crates/pc-detect/src/onnx.rs` `DEVIATION(15)`: *"a configured value greater than 1
   is warned-and-ignored"*). `pc-pipeline`'s `rayon` batch parallelism (`max_threads`)
   only helps the non-model parts of the pipeline (image I/O, mask geometry, NLM
   denoise) — it does nothing for OCR/inpaint/detect inference cost.

## Ruled out — don't re-try these without new evidence

- **LaMa batching** (stack multiple tiles into one call): already measured by this
  project, b=1/2/4 → 1.58s/3.03s/6.55s, **linear, no throughput gain**
  (`docs/PIPELINE_SPEC_V1.md` line 7571, task L5, 2026-08-07). Closed.
- **Naive OCR concurrency** (N independent engine instances, N threads, default/auto
  per-session thread count): measured 2026-08-15 with a throwaway test (3 engines/threads
  vs 1, same 6 demo_bubbles crops) — **0.57x, i.e. ~1.76x SLOWER than sequential.**

  **The obvious explanation for this (core oversubscription, "one call already uses all
  cores") was checked directly and is WRONG — corrected same day.** Sampled the process's
  actual CPU time during a real single sequential call on this 20-logical-core machine
  (`Get-Process`/cumulative `TotalProcessorTime` deltas, not inference): a single call
  uses **roughly 2-4 logical cores at once, not anywhere near all 20.** So three
  concurrent calls needing ~9-12 cores total should have had headroom on paper — the
  measured 1.76x slowdown is real but its cause is still open. Two untested candidates,
  neither confirmed: **(a)** ONNX Runtime may use one shared default thread pool per
  *process* rather than per *session*, so concurrent `Run()` calls contend for pool
  scheduling regardless of idle raw cores; **(b)** memory-bandwidth/cache contention
  between large session weights (343MB + 117MB each) — sequential execution keeps one
  session's working set warm in cache, concurrent execution thrashes a shared L3 against
  itself. Do not re-explain this as "oversubscription" without new evidence; that
  specific explanation was checked and ruled out.
- **Crop size, thread-count *default*, debug-vs-release build, code/dependency
  regression since GPU-4, per-image session rebuild, OS power-plan cap** — all
  independently tested and ruled out as explanations for the historical-vs-measured gap.
  See `docs/WORKSTATE.md`'s 2026-08-15 entries for exactly what was tried.

## Untried, most promising next step

**Diagnose WHY concurrency lost, before trying to fix it.** The original plan here was
"cap per-session thread count to avoid oversubscription" — but that premise (a single
call uses ~all cores) was checked directly 2026-08-15 and found false (measured ~2-4 of
20 logical cores per call). Capping threads may still help, but it's no longer the
obviously-right fix; it's one candidate among at least two. The real next step is
figuring out which of the two live hypotheses above is actually responsible:
- Check whether `ort`/ONNX Runtime creates one thread pool per `Session` or shares a
  process-global default — read `ort` 2.0.0-rc.12's source directly (same "read the
  pinned source, don't guess" standard the GPU-3/GPU-4 work already used for `ort`'s CUDA
  behavior) rather than assuming either way.
- If it's genuinely thread-pool contention, try explicit *distinct* thread pool
  configuration per session (not just capping the count) and re-measure.
- If it's cache/memory-bandwidth contention, capping thread count won't help much;
  the lever there would be running fewer concurrent sessions with larger per-session
  batches of *sequential* work each (reducing context-switch/cache-thrash frequency)
  rather than pure N-way concurrency.

Not attempted yet — this is the actual next experiment.

## Untried, unknown — no evidence either way

- **OCR batching** (stack multiple boxes' encoder/decode into one call): never measured
  in this codebase, unlike LaMa's. A transformer encoder/decoder may batch differently
  than LaMa's CNN — plausible either way, no real number exists yet.
- **True multi-session pooled concurrency for the detector**: same oversubscription
  caution as OCR applies; `[text_detector] concurrent_models` config field exists but is
  currently ignored above 1 (`DEVIATION(15)`). Same "cap per-session threads first"
  precondition as the OCR item above.

## Open, unresolved, lower priority

- **Why the 2026-08-11 historical OCR number (`7.387s`/10 pages, ~71ms/box) looks
  implausible against today's measured ~2.1s/box for the same real page set.** Current
  working theory: the historical number's cumulative-diff methodology understated real
  cost, rather than today's CPU being genuinely regressed — but this isn't confirmed.
  Confirming it would mean re-deriving the historical measurement with a direct
  per-stage timer instead of run-total subtraction. Not attempted.
- **Real hardware frequency/thermal profiling** (Intel Power Gadget, HWiNFO,
  `Get-Counter '\Processor Information(_Total)\% Processor Performance'`): only worth
  doing if the capped-thread OCR experiment above still leaves an unexplained gap. The
  OS power-plan check already done (2026-08-15) found no throttle-percentage cap, but
  didn't reach real-time frequency data.

## Also noted, not scoped here

- `panel-ocr models path` registers `lama-manga-inpainter` → `lama-manga.onnx` sourced
  from `mayocream/koharu` (not a separate `mayocream/lama-manga` fetch) — confirmed this
  is already the manga-finetuned weights upstream PanelCleaner itself uses
  (`docs/PIPELINE_SPEC_V1.md` DEVIATION(25)). No further action implied; recorded here
  only so a future session doesn't re-open the "should we swap to manga-tuned LaMa"
  question from the original research pass — it's already the shipped model.
