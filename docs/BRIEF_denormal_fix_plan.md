# Brief: joint plan — fix the detector's ineffective denormal-flush

## Confirmed finding (real, measured via temporary instrumentation, already reverted — tree is clean)

`crates/pc-detect/src/onnx.rs:589-590` calls `builder.with_flush_to_zero()` (already
existed, from §16.21's prior attempt at this exact class of problem) — but the flag
never reaches the `pc-detect-onnx` worker thread (`onnx.rs:642`, `run_worker`) that
actually executes `Session::run`. Measured MXCSR on that thread: `ftz=0 daz=0` — unset.

From the 3rd inference in a process onward, detector calls hit denormal floats and cost
~30x more (0.9s → 27-30s per call). Manually setting MXCSR FTZ+DAZ bits on that one
thread, nothing else changed: 10-page batch wall dropped 240.3s → 31.0s (9.9x),
detector-only inference time 234.2s → 23.6s. This is the dominant cause of the whole
batch-mode regression investigated today — not thread contention, not ORT memory
arenas, not OCR.

**Residual, unexplained**: even with the calling thread fixed, cost still grows ~8x
across the 10 calls (0.455s → 3.722s). Leading theory, NOT confirmed: ORT's own internal
intra-op thread pool (separate threads from the caller) never gets the flag either, since
MXCSR is per-thread and the probe only touched the calling thread.

## What the plan must cover

1. **Why `with_flush_to_zero()` doesn't work today.** Read the `ort` 2.0.0-rc.12 pinned
   source for what that builder method actually configures (likely an ONNX Runtime
   `SessionOptions` flag that's real inside ORT's own graph execution, but doesn't affect
   how *this* worker thread's CPU state is set before calling into it — or possibly it's
   simply not wired up correctly on this platform/version). Don't assume; read the source.
2. **A real fix that reaches every thread that runs float ops**, not just the caller —
   including ORT's own intra-op thread pool if that's confirmed as the residual cause.
   Options to weigh: setting MXCSR via a custom thread-creation hook if `ort`/ONNX Runtime
   exposes one; setting it process-wide before any session/thread pool exists; or
   confirming whether `with_flush_to_zero()` needs a different call site/order to actually
   take effect.
3. **Same check for `pc-ocr` and `pc-inpaint`** — neither currently requests flush-to-zero
   at all (`crates/pc-ocr/src/onnx.rs:454-459`, `crates/pc-inpaint/src/onnx.rs`
   `build_session`). Determine whether they're exposed to the same risk and should get
   the same fix, or whether it's provably detector-specific (fixed-size-input stages may
   not be equally exposed — investigate, don't assume).
4. **§16.21's own prior ratification** claimed this was already handled — this plan needs
   to address that directly: was §16.21 wrong, incomplete, or did something regress it?
   Check the existing frozen tests tied to that section (grep `flush_to_zero`/`FTZ`/`DAZ`/
   `denormal` across `crates/pc-detect/tests/`) — a real fix must not silently break
   whatever those already assert.
5. **Test plan**: how do you verify the fix WITHOUT a fragile, timing-based test (wall-clock
   assertions are flaky) — e.g. asserting the actual MXCSR/thread state directly, or
   asserting on a synthetic denormal-heavy input's correctness+performance in a bounded,
   deterministic way. Real test code, per this project's pipeline, not just a description.

## Scope fence

This is Spec-sensitive tier (reopens/corrects an existing ratified deviation, §16.21;
core production inference code, all three ONNX-backed crates potentially affected).
Standard pipeline: joint plan here, ratify into `docs/PIPELINE_SPEC_V1.md` if the two of
you converge (or escalate to Fable if you disagree), then implement task-by-task with
independent review.
