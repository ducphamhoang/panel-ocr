# CPU performance backlog

## §16.54 P0/S0 MEASUREMENT, 2026-08-17 — inconclusive: real headroom is comparable to run-to-run noise; `StageGate` implementation NOT started

Per §16.54 Ruling 0's binding requirement, the three cancel criteria were pre-registered
(`docs/BRIEF_p0_stagegate_measurement.md`) before any timed run. Result: **the
measurement cannot confidently clear or fail its own criteria — the signal and the noise
floor are the same size.** This is reported as the honest outcome, per Ruling 0's own
instruction to report even "cancel" or "inconclusive" rather than a forced verdict.

**Setup**: 4 real Choujin Locke v02 pages (010-013, smaller than the architect's
suggested "≥8 images," disclosed as a deviation for wall-clock cost — the first thing to
redo if this measurement is picked back up), real cached ONNX weights (all four models
confirmed present via `panel-ocr models path`), CPU device, 20 logical cores, cumulative-
diff progressive-stage-disabling (the same trusted, no-new-code method used 2026-08-11),
one warm-up run discarded per that entry's cold-start lesson.

**Hazard check (Ruling 3 graft), clean**: `grep -rn "MXCSR|flush_denormals|_MM_SET_FLUSH_ZERO|_MM_SET_DENORMALS_ZERO|denormal" crates/pc-ocr/src crates/pc-inpaint/src`
— zero hits. Neither engine crate touches any denormal/MXCSR flag the way the detector's
`flush_denormals` does; there is no analogous hazard for §16.32's confinement pattern to
guard against here.

**Timed results, single run each (repeat below shows why these are not to be trusted to
more than ~1 significant figure)**:

| Stage (cumulative) | Wall-clock, 4 pages, default threads |
|---|---:|
| Detect + mask | 2.609s |
| + OCR | 8.978s (OCR alone ≈ 6.4s) |
| + Denoise | 6.879s (denoise alone ≈ **-2.1s** — noise, not a real negative cost; swamped by OCR run-to-run variance, same shape as the 2026-08-11 denoise finding) |
| + Inpaint (= full) | 30.495s (LaMa alone ≈ 23.6s) |
| Full, repeated | **35.801s** — 17% higher than the first full run, same config, same input, same process |
| Full, `--threads 1` | 50.698s (1.66x slower than default-threads' first sample) |

**Why this is inconclusive against the pre-registered criteria, not a clean pass or
fail**: criterion (a)'s cancel bar was "actual wall-clock within ~15% of the analytic
`max(stage_cost)` bound." The first full-run sample (30.5s) against the LaMa-dominated
bound (23.6s) is ~23% over — above the cancel threshold, i.e. "don't cancel" by the
letter of the pre-registered rule. **But the very next identical run (35.8s) shows 17%
run-to-run noise on its own**, comparable in size to the 23% gap the first sample showed
against the bound. A measurement whose noise floor is the same size as the effect it's
trying to detect cannot honestly be reported as clearing or failing a 15% bar — the
15% figure itself is now known to be inside the noise band, which was not known when it
was written into the pre-registered brief (a real limitation of pre-registering a
threshold before the first sample, disclosed rather than quietly using a different
threshold after the fact).

**One real, useful byproduct**: this measurement's OCR aggregate (≈6.4s / 4 pages ≈
1.6s/page) sits much closer to the "high" historical figure (~2.1s/box,
`docs/PERFORMANCE_BACKLOG.md`'s "OCR is the dominant cost" entry) than to the "low"
2026-08-11 per-stage table's 7.387s-for-a-whole-10-page-batch figure — informally
supporting, though not proving with a controlled re-derivation, that document's own
standing working theory that the low number understated real cost rather than today's
CPU being regressed. Not a substitute for the controlled re-derivation that entry itself
says is the real way to close the question.

**Also not completed this pass, disclosed rather than silently skipped**: the OCR
`intra_threads=1`-while-LaMa-runs experiment both Opus passes named. `intra_threads` is
hard-coded in `pc-ocr`'s construction code, not profile-configurable (confirmed: only
`[text_detector]` exposes `intra_threads`/`inter_threads` in
`crates/pc-config/src/default_profile.toml`) — running it needs a temporary source patch
plus rebuild plus revert, a real but larger time cost than this pass budgeted for.

**Recommendation, not a unilateral decision**: given (1) the observed gap and the noise
floor are the same order of magnitude, (2) Fable's own prior (reasoning-only) expectation
was that this measurement would likely cancel the redesign, and (3) building `StageGate`
is real implementation effort for a benefit this measurement cannot confidently show
clears the bar — **`StageGate` implementation is not started.** This is not the same as
"cancel per Ruling 0" — that requires actually clearing/failing the criteria, which this
measurement's noise floor prevented. The honest state is: **undecided, pending a properly
powered re-measurement** (larger batch, ≥3-5 repeats per configuration, a real confidence
interval instead of single samples, the intra_threads=1 experiment, and per-stage timing
if `pc-pipeline` ever grows one instead of relying on cumulative-diff). Whoever picks
this up next should not treat this pass's single-sample numbers as evidence either way —
only as a demonstration that the noise floor must be characterized before the pre-
registered criteria can be evaluated at all.

### FOLLOW-UP, same day — larger batch (10 real pages, 3 repeats): the noise floor drops and a real gap now clearly exceeds it, but the CAUSE of the gap is still open — this is an update, not a revised final verdict

Re-ran the same cumulative-diff method on 10 real Choujin Locke v02 pages (010-019,
matching the architect's original "≥8 images" suggestion this pass had shortened), with
the full-pipeline stage repeated **3 times** instead of 2, specifically to characterize
the noise floor the 4-page run couldn't.

| Stage (cumulative) | Wall-clock, 10 pages, default threads |
|---|---:|
| Detect + mask | 5.014s |
| + OCR | 7.621s (OCR alone ≈ 2.6s) |
| + Denoise | 7.282s (denoise alone ≈ **-0.3s** — noise, same shape as every prior denoise measurement) |
| + Inpaint (= full), run 1 | 23.027s |
| + Inpaint (= full), run 2 | 24.510s |
| + Inpaint (= full), run 3 | 24.562s |
| `--threads 1` | 93.796s |

**The noise floor is now much tighter**: the 3 full-pipeline repeats span 23.03-24.56s,
a **~6.4%** spread around their mean (24.03s) — well under the 4-page run's 17%. This
confirms the expected effect of a larger batch: fixed per-process overhead and
run-to-run variance amortize over more work, shrinking *relative* noise. This alone is a
useful, disclosed correction to the earlier entry's implicit assumption that batch size
wouldn't matter to the noise question.

**The gap now clearly exceeds the noise.** LaMa's aggregate cost (mean full − stage-C
value) ≈ 24.03 − 7.282 ≈ 16.75s. The analytic bound (perfect staging, bounded only by
the single largest stage) is therefore ≈16.75s. Actual mean wall-clock (24.03s) exceeds
it by **≈30%** — roughly 5× the ≈6.4% noise band. Unlike the 4-page run, this signal is
not plausibly just noise. **Criterion (a) (real headroom against the analytic bound) is
therefore not cancelled by this data — a real gap this batch's own repeats can't explain
away.** Raising `--threads` further cannot close it either (default threads already
equals `min(20 cores, 10 images) = 10`, the image-count ceiling, not a core ceiling) —
**criterion (b) is also not cancelled.**

**But criterion (c) — that the gap is attributable to per-stage admission/scheduling,
the specific thing `StageGate` (§16.54 Ruling 1's approved mechanism) would fix — is
NOT yet confirmed, and this matters more than it might look.** Two distinct, untested
explanations for the same 30% number, neither ruled out:

- **Fill/drain.** The first image's LaMa call can't start until that image's own
  detect+OCR+denoise are done, and the last image's LaMa call has no later image's
  earlier-stage work left to hide behind it. This is a real cost on *any* finite batch,
  and it is specifically the cost the rejected/deferred full staged-executor design
  ("S4" in the joint plans, explicitly left unauthorized by §16.54) would address — a
  plain admission gate (`StageGate`) does **not** touch it, because it doesn't change
  *when* the pipeline starts or ends, only how many calls are admitted concurrently.
- **CPU oversubscription across different models.** Detect, OCR, and LaMa are each
  configured `intra_threads = 0` (use all logical cores) per call
  (`crates/pc-detect/src/onnx.rs`, `pc-ocr/src/onnx.rs:454`,
  `pc-inpaint/src/onnx.rs:420`). With 10 rayon workers in flight, several
  all-core-hungry calls from *different* models can genuinely be running at once,
  competing for the same 20 logical cores — the same mechanism already confirmed to cost
  ~1.76x for 3 concurrent *same-model* OCR sessions, here potentially recurring across
  different models. `StageGate` does not fix this either: it only bounds how many calls
  of *one* model run concurrently (already 1, via the existing mutex) — it says nothing
  about calls from *different* models competing for cores at the same instant.

**Distinguishing test, not yet run** (the natural next step, not yet executed — time
budget for this pass went to the batch-size/repeat-count re-measurement instead):
re-run this same 10-page cumulative-diff at a **larger** batch size (e.g. 20-30 pages).
If the percentage gap **shrinks** as batch size grows, that points to fill/drain (a
roughly fixed one-time cost, amortizing over more images) — and would mean `StageGate`
is the wrong fix regardless of what P0/S0's other criteria say, since fill/drain needs
the deferred S4 design instead. If the percentage gap stays **roughly constant**
regardless of batch size, that points to CPU oversubscription (a per-call, not
one-time, cost) — which `StageGate` also does not fix, but which might respond to
per-model `intra_threads` tuning instead (a config change, not an architecture change,
and not something either Opus plan proposed touching). Either outcome argues *against*
proceeding straight to `StageGate` on the current evidence; **this update sharpens what
"real headroom" means without yet identifying a fix `StageGate` would actually deliver.**

**Status, restated so it isn't over-read**: this is an **update to the P0/S0 record**,
not a revised final verdict superseding the "not started" recommendation above. The
noise-vs-signal question this entry's own worry was about is now resolved (signal
exceeds noise at 10 pages) — but a *new*, more specific open question replaced it
(which mechanism, if any, the gap actually calls for), and `StageGate` implementation
remains not started pending it.

### PARKED, 2026-08-17 — investigation stops here, by explicit user decision, not silent abandonment

The natural next step (repeat the cumulative-diff at 20-30 pages to distinguish
fill/drain from cross-model CPU oversubscription) is **not being pursued**, per the
user's explicit reasoning, recorded so a future session finds a decision rather than an
abandoned thread: **any conclusion from further measurement on this one machine risks
being hardware-topology-specific, not general.** This machine's own CPU
(i7-12700KF) already has a documented heterogeneous P-core/E-core split that this
project independently flagged as a real confound (§16.21's "oversubscription is real...
some landing on the i7-12700KF's slower E-cores"). Chasing a precise root-cause
attribution across batch sizes on this one machine, only to then need to ask whether it
generalizes to a 4-core laptop, a shared cloud vCPU, or a different vendor's
core-scheduling behavior, is an open-ended cost with no natural stopping point — and per
CLAUDE.md's right-sizing principle, effort should be proportional to expected benefit,
not open-ended "to be sure." That benefit was never solid to begin with: Fable's own
Ruling 0 (§16.54) predicted, from reasoning alone, that a real measurement would likely
cancel this redesign; nothing measured since has raised that expectation, only
sharpened what an eventual "yes" would need to show.

**Disposition: `StageGate` implementation stays not started. This entire investigation
(§16.54's ratification, both P0/S0 measurement passes, and this parking decision) is
closed for now, not deleted or hidden** — a future session with a concrete reason to
revisit (a specific deployment target's hardware, a user report that batch throughput is
the actual bottleneck, or interest in the fill/drain-vs-oversubscription question for its
own sake) should start from this record rather than re-deriving it, and should treat the
20-30-page distinguishing experiment above as the first thing to run, not this session's
single-machine numbers as an answer.

## NEW LEVER, 2026-08-17 — cross-image OCR/inpaint overlap: measured promising, not yet designed

**User-proposed optimization: instead of running each image's whole pipeline strictly
sequentially, overlap "part A" (detect+mask+OCR) of image N+1 with "part B" (LaMa-inpaint)
of image N, since they're independent once part A's output for image N is already
consumed.** Before any architecture work, the real prerequisite question was measured
directly: does running OCR and LaMa-inpaint CONCURRENTLY (two different real ONNX
sessions, two different models) contend for memory bandwidth the way three concurrent
*same-model* OCR sessions were already measured to (2026-08-15: 0.57x, ~1.76x slower) —
or is that finding specific to identical sessions competing for the same memory
footprint?

**Measured (throwaway test, real cached weights, deleted after use): overlap is nearly
free — the opposite of the same-model finding.**
```
SEQUENTIAL: ocr_alone=3.919s inpaint_alone=27.253s total=31.172s
CONCURRENT: total=27.579s (ideal overlap = max(ocr_alone, inpaint_alone) = 27.253s)
speedup_vs_sequential = 1.13x
```
Concurrent wall-clock (27.579s) is within ~1.2% of the theoretical best case (27.253s,
running OCR entirely "for free" inside inpaint's larger cost) — essentially no
contention penalty. This is the opposite of the OCR-vs-OCR case: two *different* models
(different weight sets, different memory access patterns) apparently don't compete for
the same cache lines/bandwidth the way three copies of the *same* model do.

**Why this could matter in practice**: LaMa-inpaint dominates per-image cost (27.25s vs.
3.9s OCR here, consistent with the 2026-08-11 per-stage breakdown finding inpaint is the
single largest pipeline stage on both CPU and GPU). If a real batch pipeline could
genuinely overlap image N+1's detect+mask+OCR inside image N's inpaint window, the OCR
(and likely mask) cost could be nearly hidden for every image except the first/last in a
batch — a real, additive lever on top of the already-shipped denormal-flush fix and OCR
beam-batching.

**Not yet designed, not yet scoped.** This is a real architecture change to
`crates/pc-pipeline` (currently whole-image, data-parallel-across-images via
`rayon::par_iter` per DEVIATION(11) in `batch.rs`, not a staged/overlapped pipeline) —
would need a producer-consumer handoff between stages, touches `BatchSummary`'s frozen
input-order guarantee, and needs the same joint architect+rust-engineer planning this
project's Spec-sensitive tier requires for anything touching core pipeline architecture
and frozen tests. This entry records the measured prerequisite only; the design itself is
the next step if picked up.

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

**D3 follow-up #2 (2026-08-17, same day): extended to 30 consecutive calls to answer
directly whether a realistic 20-30 image batch keeps getting progressively slower, or
plateaus. It plateaus — real, measured, not extrapolated.** Same detector instance, same
methodology, `intra_threads=0`, real cached weights, 30 calls instead of 10:
```
0.50, 0.44, 0.43, 0.44, 0.43, 0.44, 1.19, 1.59, 1.54, 1.72, 1.70, 1.59, 1.73, 1.62, 1.92,
1.93, 1.81, 1.80, 1.78, 1.55, 1.50, 1.51, 1.52, 1.50, 1.51, 1.52, 1.50, 1.52, 1.52, 1.52
```
The same step-up appears at call ~6-7. Calls 6-18 wobble (1.19s-1.93s, noisier than the
first 10-call measurement suggested, peaking at call 15). **From call ~19 onward it
settles into a tight, stable plateau (1.50-1.52s for the last 10 consecutive calls)** —
not a continued climb, and not the earlier measurement's apparent stability either
(that was itself mid-wobble, just sampled at a quieter point). Growth against the first
call: 3.05x by the last call, 3.88x at the noisy peak (call 15) — but the peak is a
transient, not the steady state.

**Direct answer to the practical question this follow-up exists for: no, a 20-30 image
batch does NOT keep getting progressively slower.** There is one step-up early (after
~6 calls), a noisy transition period for the next ~12 calls, then a stable, flat
steady-state that holds for the remainder of the batch. A large batch pays the same
one-time "warm-up" cost a 10-image batch does and then stabilizes — it does not
compound further as the batch grows longer. This resolves the open question the
original 10-call measurement couldn't answer on its own (whether the plateau seen there
was real steady-state or just a shorter window that hadn't found a second climb yet).

**D3, final assessment (2026-08-17): every code-level explanation is now ruled out by
direct evidence; CPU-level behavior (most likely thermal/frequency throttling) is the
leading candidate, but not yet confirmed by direct measurement — stated at that
confidence level, not higher.** What's actually been eliminated, each by a real
experiment rather than by reasoning: a denormal-flag leak (D1 reapplies the guard every
call, D2 confirms zero bits move); the ORT thread-pool spin-vs-block transition as a fix
(forcing `intra_op_spinning=true` made things worse, not better); unbounded/compounding
growth (30-call run plateaus, doesn't climb further). What points toward CPU-level
causes specifically, all from real experiments already run: the pattern appears only
with multiple cores active (`intra_threads=0`) and is largely absent on a single core
(`intra_threads=1`); forcing more sustained core activity (spin=true) made the pattern
worse, the opposite of what a software-only cause would predict; and it settles into a
stable plateau rather than growing without bound, consistent with a CPU finding a lower
sustained clock and holding there. **What would close this out, not yet done**:
real-time CPU frequency/thermal measurement during a live repeated-call run (Intel Power
Gadget, HWiNFO, or `Get-Counter '\Processor Information(_Total)\% Processor
Performance'` on Windows) — every prior attempt in this investigation to reach for that
tooling stopped short of it. Until that measurement exists, "CPU throttling" is the
best-supported working hypothesis, not a proven conclusion — and it does not change
anything about §16.52's correctness or the real ~3.38x speedup already measured against
upstream, which already reflects whatever this pattern costs in practice.

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
