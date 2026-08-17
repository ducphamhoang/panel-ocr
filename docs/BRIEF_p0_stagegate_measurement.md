# P0/S0 measurement — pre-registered before running (§16.54 Ruling 0)

Written before any timed run, per §16.54's binding consequence: "whoever picks up the
measurement task must write the three cancel criteria into its own brief *before running
it*, and must report the result against them honestly even if the result is 'cancel the
redesign'."

## Setup

4 real manga pages (Choujin Locke v02, pages 010-013 — a smaller batch than the
architect's suggested "≥8 images," disclosed as a deviation for wall-clock cost reasons;
if this measurement is inconclusive, a larger batch is the first thing to redo, not a
reason to trust a marginal result), real cached ONNX weights (all four models confirmed
present via `panel-ocr models path`), CPU device (default, matches this ruling's
CPU-only scope), 20 logical cores on this machine, default profile except where a
variant below overrides `ocr_enabled` or uses `--skip-*`.

Method: the same cumulative-diff, progressive-stage-disabling methodology this project
already used and trusted on 2026-08-11 (no per-stage timing exists in the shipped tool,
so this is measured by re-running the batch with one more stage enabled each time and
subtracting totals) — reused deliberately rather than building new instrumentation,
since it needs no code change and this project's own convention favors reusing an
established, trusted method over inventing a new one for a one-off measurement.

## Pre-registered cancel criteria (§16.54 Ruling 0)

The redesign (Option A, `StageGate`) proceeds past this measurement only if ALL three
hold. Any one failing is enough to cancel:

- **(a) Real headroom against the analytic bound.** Compare actual wall-clock of a full
  run (all stages, default threads) against the analytic `max(stage_cost)` bound
  (the largest single stage's aggregate cost across the batch — the best any staged
  pipeline could achieve with these exact per-call costs). If actual wall-clock is
  within ~15% of that bound, there is no material headroom to capture — **cancel**.
- **(b) Not already capturable by raising `--threads`.** Compare wall-clock at
  `--threads 1` vs `--threads 0` (all 20 cores). If the *default* configuration already
  lands close to the analytic bound, headroom is already captured by today's rayon
  parallelism and no new mechanism is needed — **cancel**, regardless of what (a) shows
  at threads=1.
- **(c) Attributable to per-stage admission/scheduling at fixed thread count.** If (a)
  shows a gap, confirm the gap tracks stage serialization (e.g., correlates with the
  bottleneck stage's aggregate cost) rather than something a `StageGate` couldn't fix
  anyway (I/O, decode, export overhead unrelated to model-stage contention) — **cancel**
  if the gap is dominated by non-model-stage cost, since `StageGate` only gates
  model-backed stages.

## Also measured, per Fable's Ruling 3 graft and both Opus passes' request

- **Hazard trigger** (Ruling 3, graft 1): a cheap static check for thread-state side
  effects analogous to detect's DAZ/FTZ hazard (§16.32) in the OCR/LaMa sessions —
  before any live concurrency probe, since a clean static result may make one
  unnecessary.
- **OCR `intra_threads=1` while a real LaMa call runs concurrently** — the specific
  experiment both Opus passes named as untried and directly relevant to "would pinning
  OCR to one thread to stop it fighting LaMa for cores help or hurt," using this batch's
  real box count rather than the earlier session's unrepresentative ~2-box test.
- **Resolve, if possible, the standing 7.4s-vs-220s OCR aggregate discrepancy**
  (`docs/PERFORMANCE_BACKLOG.md`, "Open, unresolved, lower priority") — this measurement
  uses the same cumulative-diff method on a real batch, so it's a natural byproduct.

## What this measurement will NOT do

Re-litigate Fable's ruling on mechanism placement, build any part of `StageGate`, or
touch `pc-pipeline`/`pc-ocr`/`pc-inpaint` source. Throwaway profile TOML files and
copied test pages live under the session scratchpad, not the repo; nothing here is
committed except this brief and the eventual result write-up.
