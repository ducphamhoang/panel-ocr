# Brief: diagnose the batch-mode CPU regression

Not temporary in the usual sense — this is the working brief for an active diagnosis;
delete once the root cause is confirmed and either fixed or formally deferred.

## The finding (already measured, real, reproduced twice — treat as given, don't re-derive)

CPU-only, release build, 10 real Choujin Locke pages (`rockéQè¬010.png`-`019.png`,
~104 total detected boxes), identical output (byte-identical `_clean.png`/`_mask.png`
confirmed across modes):

| | 10 separate single-page processes | 1 batch process (10 paths, one invocation) |
|---|---:|---:|
| Full pipeline | 159.3s | **265s (1.66x slower)** |
| `--skip-inpaint` | 43.7s | **230.8s (5.3x slower)** |

**Batch-size sweep** (`--no-cache --skip-inpaint`, default threads, pages in prefix order)
shows a step function, not a slope: pages 1-2 cost ~3s each; **every page from the 3rd
onward costs ~25-30s**. Confirmed independently via `-vv` log timestamp gaps: the 10-page
run shows exactly 9 non-overlapping "bursts," not smooth per-page progress.

**Real CPU-time burn confirmed** (not idle waiting): `Get-Process` CPU-time polling shows
~450 CPU-seconds consumed over 265s wall for work that costs ~26s CPU-equivalent when
run as separate processes — 4-10x more CPU cycles for identical output.

## What's already ruled out, by measurement — don't re-check these

- Session-construction overhead: ~1.3s for the detector (measured), negligible against
  265s. Rayon pool rebuild: confirmed built ONCE per batch (`crates/pc-pipeline/src/batch.rs:79-83`,
  read directly, before the `pool.install(...)` call). Cache/disk I/O: 0 MB read/write
  during the slow run. Denoise: `--skip-denoise` changes nothing (233.5s vs 230.8s). Box
  count/content: per-page batch cost (~28s) doesn't correlate with per-page box count
  (5-15 boxes). Thread count: `--threads 1` through default(10) all land in 230-283s —
  not classic lock-contention-scales-with-worker-count behavior.

## What's confirmed but NOT sufficient to explain the magnitude

All three model stages serialize every inference call through a single point:
- Detector: `crates/pc-detect/src/onnx.rs` — one dedicated worker thread, `run_worker`
  processes requests via a serial `recv()` loop (`DEVIATION(15)`).
- OCR: `crates/pc-ocr/src/onnx.rs` — `encoder: Mutex<Session>`, `decoder: Mutex<Session>`,
  the decoder lock taken PER BEAM STEP (~104 boxes × N tokens of tiny, globally-serialized
  calls).
- LaMa: `crates/pc-inpaint/src/onnx.rs` — `session: Mutex<Session>`.

`resolve_threads` (`crates/pc-pipeline/src/options.rs:114-123`) sizes the rayon pool to
`min(cores, image_count)` — 10 workers here, hitting 3 global serialization points, ~1.7
of 20 cores actually used.

**This predicts batch time ≈ sum-of-parts (i.e. ~flat per-page cost). It does NOT predict
batch being 1.66x-5.3x WORSE than ten separate sequential processes, and it does not
predict a sharp cliff specifically at N=3 pages that barely responds to thread count.**
That gap — roughly 190 seconds of unexplained real CPU burn in the `--skip-inpaint`
10-page case — is what this task exists to find.

## Leading unconfirmed hypotheses, neither verified

1. **ONNX Runtime intra-op thread-pool thrashing.** `intra_threads = 0` / `inter_threads
   = 0` in the profile means each session's own internal thread pool sizes itself to all
   20 cores. With ≥3 rayon workers alternately entering the SAME locked session (detector
   worker thread, OCR mutex, LaMa mutex), ORT's spin-then-park behavior inside each
   session's pool could burn real cycles without making progress, once enough concurrent
   contenders exist to create a spin/wake storm — possibly explaining both the magnitude
   and the cliff-at-N=3 shape (2 concurrent contenders might not trigger it; 3+ might).
2. **Something in image-splitting** (`process_image_with_splitting` / `strip.rs`)
   behaving differently once more than ~2 images are in flight concurrently.

Neither is confirmed. This is what needs real instrumentation to settle.

## What's needed now

**Add temporary tracing instrumentation** (spans or explicit `Instant::now()` timing,
your judgment) around:
- Each of the three session-lock acquisition points (detector's request-send/response-recv,
  OCR's encoder/decoder mutex lock, LaMa's mutex lock) — capture time spent WAITING for
  the lock vs. time spent actually running inference, per call, per rayon worker thread.
- The rayon `per-image` closure in `crates/pc-pipeline/src/batch.rs`, to see real wall-time
  per page from the worker's own perspective (not just log-timestamp-gap inference).

Run the same 10-page `--no-cache --skip-inpaint` batch with this instrumentation, and the
same workload as 3, 5, and 10 separate concurrent-but-independent single-page invocations
(if easy) to see whether the thrash appears even without the shared-lock architecture (which
would point away from hypothesis 1 and toward something else, e.g. hypothesis 2 or a third
mechanism neither of us has named yet).

**This is diagnostic only.** Revert the instrumentation once you have real data — do not
leave permanent tracing code in the tree, and do not attempt an actual fix yet. Report:
the real measured lock-wait vs. inference-time split, whether the N=3 cliff reproduces
under instrumentation, and a clear verdict on which hypothesis (or a third one) the data
actually supports.
