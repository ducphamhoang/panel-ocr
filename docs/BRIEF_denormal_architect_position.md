# architect's denormal-fix position (2026-08-16, joint planning pass vs. `docs/BRIEF_denormal_fix_plan.md`)

# Technical Architecture position — denormal-flush fix (brief: `docs/BRIEF_denormal_fix_plan.md`)

## 1. Root cause — verified in source at the pinned versions, not inferred

`SessionBuilder::with_flush_to_zero()` in `ort` 2.0.0-rc.12 (`src/session/builder/impl_config_keys.rs:36-40`) does exactly one thing:

```rust
pub fn with_flush_to_zero(mut self) -> BuilderResult {
    match self.add_config_entry("session.set_denormal_as_zero", "1") { ... }
}
```

It sets a **session config string**. It touches no CPU register. What ONNX Runtime then does with that string, at the pinned binary version **1.24.2** (`ort-sys-2.0.0-rc.12/build/download/dist.txt` line 8 → `pyke:ort-rs/ms@1.24.2/x86_64-pc-windows-msvc`), is two separate things in `onnxruntime/core/session/inference_session.cc` (verified at tag `v1.24.2`, lines 404-441; byte-identical to `v1.22.0`):

```cpp
bool set_denormal_as_zero =
    session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigSetDenormalAsZero, "0") == "1";

// The only first session option for flush-to-zero and denormal-as-zero is effective to main thread and OpenMP threads.
{
  static std::once_flag once;
  std::call_once(once, [&] { SetDenormalAsZero(set_denormal_as_zero); ... });
}
...
to.set_denormal_as_zero = set_denormal_as_zero;     // per-session intra-op pool params (line 441)
```

**(a) The calling thread is covered by a process-wide `std::call_once`.** `SetDenormalAsZero` (`core/common/denormal.cc`) sets DAZ+FTZ on *the thread that calls `InferenceSession::Initialize`* — i.e. our `pc-detect-onnx` worker, since it owns `commit_from_file`. But it runs **only for the first `InferenceSession` initialized in the process**, and it runs with *that* session's flag value. A first session with the flag off calls `SetDenormalAsZero(false)` and burns the flag for the process.

**(b) ORT's own intra-op pool threads are covered per-session, not once.** `EigenNonBlockingThreadPool.h:775` captures `thread_options.set_denormal_as_zero`, and `:1542` calls `SetDenormalAsZero(set_denormal_as_zero_)` at the top of every `WorkerLoop`. This is read from the config entry directly (line 441), independently of the once-flag.

**Which session is first in a real `panel-ocr clean` run: the OCR encoder, not the detector.** `crates/pc-cli/src/lib.rs:96-108` builds the detector *provider* (lazy `OnceLock`, `crates/pc-cli/src/detector.rs:94`) but then calls `ocr::build_factory_for_device(...)` **eagerly**, and that chain is eager all the way down — `MangaOcrEngine::from_paths_for_device` → `MangaOcrSessions::from_paths_with_policy` (`crates/pc-ocr/src/onnx.rs:99-100`) calls `build_session` twice, on the **main thread**, and `pc-ocr`'s `build_session` (`crates/pc-ocr/src/onnx.rs:444-466`) never sets the flag. `ocr_enabled` defaults to `true` (`crates/pc-config/src/profile.rs:279`).

So in the shipped binary: OCR encoder initializes first → `SetDenormalAsZero(false)` on the main thread → once-flag consumed → the detector's later `Initialize` on `pc-detect-onnx` never reaches the branch → **worker MXCSR stays `0x1f80`, `ftz=0 daz=0`**, exactly as the brief measured. The caller thread participates in ORT's intra-op parallel work, so a third of the work runs without flushing while the pool threads flush — which is why setting the bit on that one thread bought 9.9x.

**Answer to brief item 4 (was §16.21 wrong / incomplete / regressed?): incomplete, never regressed.** §16.21 item 6's sentence — *"Setting ONNX Runtime's `session.set_denormal_as_zero` (`SessionBuilder::with_flush_to_zero()`) closes the gap"* — is **true under a condition it does not state**: that the detector session is the first `InferenceSession` initialized in the process. That held in the `xtask bench` harness where it was measured. It does not hold in `run_clean`. This is CLAUDE.md's "a claim's scope travels with it" defect, one level down: a measurement correct about one process shape, transcribed as a property of the flag.

**Why the frozen gate never caught it — cookbook rule 13 (what does the gate *enumerate*?).** `crates/pc-detect/tests/perf2_mxcsr_hygiene.rs:87-90` constructs `flush_detector` **before** `baseline_detector`. In its own fresh process the flush session is first, the once-flag fires with `true`, and the ≥3.0x timing assertion passes honestly. The gate's enumerated scenario is precisely the one production never takes. Same for `perf2_bit_exactness.rs:194-221`, which runs the flush arm (`:199`) before the flush-off arm.

**A second, latent defect in the same family, worth reporting on its own.** `perf2_bit_exactness.rs` has three tests in one binary; cargo runs them concurrently. If `the_real_detector_reproduces_the_committed_fixture_at_every_thread_count` (which uses `SessionTuning::default()`, flush=**true**) wins the race, the once-flag fires `true`, and then `flushing_denormals_does_not_change_one_raw_output_float`'s "flush off" arm *also* runs with FTZ set on its worker — the test compares FTZ-vs-FTZ and asserts equality vacuously. Its contrast is order-dependent and unasserted. That means §16.21 item 6's *"no recorded float moved"* evidence is only as good as an ordering nobody pinned, and re-measuring it is a required task below, not an optional one.

**The brief's leading theory for the residual ~8x is refuted by (b) above.** ORT's intra-op pool threads *do* get DAZ/FTZ today, per session, unconditioned by the once-flag. The residual has some other cause. The plan must not encode the refuted theory as a premise; it gets a measurement task with a decisive cheap experiment (`intra_threads = 1` removes the pool entirely).

## 2. Design

### 2.1 The mechanism must be ours, not ORT's

Any fix routed through `with_flush_to_zero()` alone is order-dependent by construction — it depends on which crate happens to build a session first, which is a property of CLI argument parsing, not of the detector. **We set the CPU state ourselves, scoped, at the call site.** `with_flush_to_zero()` stays (it is the only way pool threads get the bit); our control is additive and covers the participating caller.

### 2.2 New module: `crates/pc-ort/src/denormal.rs`

`pc-ort` is already an unconditional path dependency of all three ONNX stage crates (`crates/pc-{detect,ocr,inpaint}/Cargo.toml`), and the module needs **no `ort` and no `onnx` feature** — so its tests run in the **default tier**, with no model, no ONNX Runtime, and no timing.

```rust
/// Observed denormal-handling state of the CURRENT thread. `None` where uncontrollable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushState { pub flush_to_zero: bool, pub denormals_are_zero: bool }

pub fn denormal_control_available() -> bool;          // x86_64 + SSE3 (mirrors ORT's own CPUID gate)
pub fn flush_state() -> Option<FlushState>;           // read-only probe of this thread
pub fn raw_control_word() -> Option<u32>;             // for exact-restoration assertions

/// RAII scope. Restores the EXACT prior control word on Drop, including during unwind.
pub struct MxcsrScope { /* saved: Option<u32> */ }
impl MxcsrScope {
    pub fn flushing() -> Self;   // sets FTZ|DAZ for this thread, then restores on drop
    pub fn fencing() -> Self;    // changes nothing; restores on drop — fences third-party mutation
}
```

Implementation note, **measured, because it changes the task's shape**: `std::arch::x86_64::_mm_getcsr`/`_mm_setcsr` are `#[deprecated]` on the pinned toolchain (`rustc 1.94.1`) with the message *"use inline assembly instead"* — under `cargo clippy -- -D warnings` they are a hard error. The implementation must use `core::arch::asm!("stmxcsr …")` / `asm!("ldmxcsr …")`. I compiled and ran both variants on this machine (see §4).

**This introduces the workspace's first `unsafe`** — `grep -rln "unsafe " --include=*.rs crates` returns nothing today. That is a decision, not an implementation detail, and it belongs in the ratification. I propose containing it: add a workspace lint `unsafe_code = "deny"` with a single `#[allow(unsafe_code)]` on this module, so the boundary is enforced rather than conventional.

### 2.3 Wiring — `pc-detect`

- Thread `tuning.flush_denormals` through to the worker loop (today it dies in `build_session`). `run_worker(&mut session, rx, flush_denormals)`.
- In `infer()` (`crates/pc-detect/src/onnx.rs:663`), immediately around `session.run(...)`: `let _flush = flush_denormals.then(MxcsrScope::flushing);`
- Wrap `build_session`'s `commit_from_file` in `MxcsrScope::fencing()`. Invariant, uniform across all three crates: **no `ort` call may leave the calling thread's MXCSR changed.** This turns §16.32's stated hazard ("ONNX Runtime sets DAZ/FTZ on the constructing thread and never restores it") from a thing we dodge by confinement into a thing we prevent.
- **Observability aimed at the artifact carrying the risk** (cookbook rule 12): inside the scope, immediately before `session.run`, record `flush_state()` into a relaxed `AtomicU32` on the detector; expose `OnnxDetector::last_inference_flush_state() -> Option<FlushState>`.

  **I want this ungated, and I expect `rust-engineer` to prefer `#[cfg(any(test, feature = "bench-tuning"))]` for consistency with `panic_next_infer_for_test`.** My grounds: those are *injection* hooks that change behaviour and must not ship; this is a *read-only observation of the shipped path*. If it is cfg-gated, the assertion is made against a binary the user never receives, and the production path's flush state is verified by nothing — which is the copy-vs-artifact inversion rule 12 exists for. Cost is one relaxed store per ~0.5 s inference. If we disagree, this goes to Fable rather than to a compromise.

### 2.4 `pc-ocr` / `pc-inpaint` — brief item 3

Structurally they are exposed **identically**: `crates/pc-ocr/src/onnx.rs:444` and `crates/pc-inpaint/src/onnx.rs:406` build sessions on the caller with no flag, and run on caller threads with no confinement. "Fixed-size input" is **not** a reason to exclude them — denormal stalls are a function of *activation values*, not tensor shape.

But I will **not** ship a flush in OCR/inpaint in this change, for a reason that is about evidence, not caution:

- §16.21 item 6's bit-exactness evidence is the detector's, on the detector's model, and (per §1 above) is itself now suspect. There is no equivalent evidence for OCR. Flushing denormals in a beam search can flip an argmax tie and change emitted text — a user-visible output change with no measured basis. Applying the detector's ruling to OCR is exactly cookbook rule 4's "a clause's subject is not the thing you are applying it to".
- What they *do* get in this change, with zero float-semantics risk by construction, is `MxcsrScope::fencing()` around `commit_from_file` — it restores whatever we had, so it cannot move a bit.
- The flush decision for OCR/inpaint is gated on its own measurement task (D4/D5), producing the table, and its own §16.x entry.

### 2.5 Test plan — no timing assertions anywhere in the gate

**T1 — `crates/pc-ort/tests/denormal_scope.rs`, default tier, no model, no ONNX, deterministic.** These can all be written before a line of implementation exists and will fail (fail-to-compile, then fail-to-assert). Nine assertions; the two that matter most:

- **FTZ and DAZ are asserted by *independent* probes.** `MIN_POSITIVE / 2.0` detects FTZ only; `f32::from_bits(1) * 1e30` detects DAZ only. I verified on this machine that each isolates the other (§4). A single-probe "FTZ+DAZ" assertion is satisfiable with DAZ off — cookbook rule 1's dominant defect class, avoided by construction rather than by naming.
- **Exact restoration is asserted on the whole control word**, not on the two bits — a guard that clobbers the rounding mode or an exception mask must turn this red.

Plus: raw-word equality after `catch_unwind` of a panic inside the scope; nesting; and thread-locality (a thread spawned inside an active scope must report `ftz=false`).

**T2 — `crates/pc-detect/tests/perf3_denormal_reaches_inference.rs`, opt-in, real weights. The test that is red today.** It must be **the only test in its file**, because it is process-ordering-sensitive and cargo runs tests in threads of one process — this is the exact property whose absence hid the bug.

1. Construct a **non-flushing** `OnnxDetector` (`flush_denormals: false`) **first**, reproducing what the OCR encoder does to the once-flag in production.
2. Then construct the shipped-default detector, run `detect`, assert `last_inference_flush_state() == Some(FlushState { flush_to_zero: true, denormals_are_zero: true })`.
3. **Negative control**, so the observation is shown capable of reporting the wrong value: the `flush_denormals: false` detector's own run reports both `false`.

**T3 — bit-exactness, re-measured rather than inherited.** Re-assert the raw-float digest against the committed recorded fixture under the fixed path, and re-run `flushing_denormals_does_not_change_one_raw_output_float` with the flush state of each arm now *asserted* rather than assumed. This is required work, not verification theatre: §1's ordering finding means the existing evidence may have compared FTZ against FTZ.

**T4/T5 — maintainer-local measurement, not gates.** Producing the numbers runs models; comparing anything frozen must not. The residual experiment is decisive and cheap: with the detector fixed, measure the per-call curve at `intra_threads = 1` (no pool exists at all) against `0`. If the growth persists at 1, the pool is definitively excluded and the brief's theory is closed by measurement as well as by source.

### 2.6 Task sequencing

| # | Task | Depends on | Interfaces touched | Class |
|---|---|---|---|---|
| **D0** | `pc-ort::denormal` module + T1 | — | new `pc_ort::denormal` public API; workspace `unsafe_code` lint | **heavy** (first `unsafe` in the workspace; inline asm; unwind-safe Drop) |
| **D1** | Wire into `pc-detect`: thread the flag to the worker, scope around `run`, fence around `commit_from_file`, `last_inference_flush_state()` + T2 | D0 | `OnnxDetector` (+1 pub method), private `run_worker`/`infer` signatures | **heavy** |
| **D2** | `MxcsrScope::fencing()` around `commit_from_file` in `pc-ocr` and `pc-inpaint` | D0 | `build_session` in both, internal only | **simple** — batchable with D1 in one Codex call |
| **D3** | Re-measure bit-exactness + freeze T3 | D1 | test files only | **heavy** (real weights, fixture-adjacent) |
| **D4** | Residual diagnosis (`intra_threads` 1 vs 0, arena/memory-pattern variants) | D1 | `xtask bench` only, no gate | **heavy**, maintainer-local |
| **D5** | OCR/inpaint exposure measurement → its own §16.x if positive | D2, D4 | measurement harness only | **heavy**, deferred |

D0 must land alone and green before D1/D2 are dispatched: every later task's test depends on `flush_state()` being trustworthy, and a broken probe would report success from a gate that cannot reach its target (cookbook rule 6a).

## 3. Ambiguities I am naming rather than silently resolving

1. **Does §16.32's "the flag ships only bundled with thread confinement" admit *scope* confinement?** The RAII scope is strictly stronger than the ratified thread confinement (the bit exists only for the duration of the ORT call, on any thread, rather than permanently on one dedicated thread), and it satisfies the hazard the ruling names verbatim. But the ruling's bundle is ratified in ORT-behaviour terms, and substituting the mechanism is a modification of a ratified deviation. **Design assumption, made visible: scope confinement satisfies the bundle.** Needs the §16.52 ruling to say so explicitly; I will not treat "strictly stronger" as self-executing.
2. **§16.21 item 6 and §16.32 need qualification, not reversal — and the supersession cross-check is a test here, not a habit.** Sites carrying the now-conditional claim, enumerated (cookbook rule 14, counted not `head`ed): `docs/PIPELINE_SPEC_V1.md:3700` (§16.21 item 6), §16.32 item 1's quoted commit message and item 2(a) (`:6553-6600`), `docs/COOKBOOK.md:239`, `docs/RULINGS.md` Entries 1-2, and the three frozen test files `perf2_mxcsr_hygiene.rs` / `perf2_bit_exactness.rs` / `perf2_tuning.rs`. Each needs a back-pointer at the site, or `crates/pc-testkit/tests/spec_supersession.rs` will be satisfied only at the claiming end — the end a future reader does not land on.
3. **Frozen-test status of `perf2_bit_exactness.rs`'s ordering vacuity.** Under cookbook rule 8 this is not a drafting slip (the corrected version claims *more*, not less), so it routes to exit 1 — **additive strengthening**: add per-arm flush-state assertions alongside the existing digest comparison; change no existing assertion. I want this stated in the ratification so it is not later mistaken for licence to amend.
4. **`last_inference_flush_state()` gating** — stated disagreement in §2.3; my position is ungated, and I would rather it go to Fable than be split.
5. **Not decided by me, flagged for the maintainer:** whether the workspace accepts its first `unsafe`. There is no MXCSR access without it. If the answer is no, this plan has no fix and the item becomes "document that the flag is order-dependent and force detector-first construction" — which I consider markedly worse (it makes correctness depend on CLI call order), but it is the maintainer's call, not mine.

## 4. What I checked, and the actual output — including the clean results

**Verified by reading pinned source:**
- `ort` 2.0.0-rc.12 `with_flush_to_zero` sets only `session.set_denormal_as_zero=1` — quoted in §1 from `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ort-2.0.0-rc.12/src/session/builder/impl_config_keys.rs:36-40`. There is a *second*, different `with_flush_to_zero` on `GlobalThreadPoolOptions` (`src/environment.rs:328`, `SetGlobalDenormalAsZero`) which we do **not** call and which only affects a global thread pool we do not use.
- Pinned ORT binary is **1.24.2** (`dist.txt`), not a guess: `https://cdn.pyke.io/0/pyke:ort-rs/ms@1.24.2/x86_64-pc-windows-msvc.tar.lzma2`.
- The `std::call_once` block and `to.set_denormal_as_zero` assignments exist at `v1.24.2` lines 404-441, fetched and grepped, and are identical to `v1.22.0` — I checked both so the finding does not rest on one tag.
- `EigenNonBlockingThreadPool.h:775/1498/1542` at `v1.24.2` — pool threads set DAZ/FTZ in `WorkerLoop`.

**Verified by reading this repo:** the eager-OCR/lazy-detector ordering (`pc-cli/src/lib.rs:96-108`, `pc-cli/src/ocr.rs:44-60`, `pc-ocr/src/onnx.rs:99-100`, `pc-cli/src/detector.rs:94`); `ocr_enabled` default `true`; `pc-ocr`/`pc-inpaint` `build_session` set no flag; the flush-first ordering in both existing test files; `pc-ort` is an unconditional dependency of all three stage crates; **no `unsafe` and no `unsafe_code` lint exists anywhere in `crates/`** (grep returned empty — a clean result worth stating).

**Verified by compiling and running probes on this machine** (scratchpad, nothing written to the repo; `git status` unchanged):
- `_mm_getcsr`/`_mm_setcsr` under `#![deny(warnings)]` on rustc 1.94.1: `error: use of deprecated function … use inline assembly instead`. Two errors, no binary produced.
- The `stmxcsr`/`ldmxcsr` inline-asm version compiles clean under `deny(warnings)` and prints:
  ```
  prev=0x1f80  ftz=5.877472e-39  daz=1.4012985e-15
  ftz-only=0x9f80  ftz=0e0            daz=1.4012985e-15
  daz-only=0x1fc0  ftz=5.877472e-39  daz=0e0
  both=0x9fc0      ftz=0e0            daz=0e0
  restored=0x1f80  ftz=5.877472e-39  daz=1.4012985e-15
  child mxcsr=0x1f80 ftz=5.877472e-39 daz=1.4012985e-15
  ```
  This is what grounds four separate design claims: default MXCSR is `0x1f80`; FTZ=`0x8000`, DAZ=`0x0040`; the two probes discriminate the two bits **independently**; MXCSR is per-thread (the child sees the default while the parent has both bits set).

**Assumed, labelled as such, not measured:**
- That the once-flag consumption is what the brief's `ftz=0 daz=0` measurement observed. The mechanism is proven in source and the ordering is proven in our code, but I did not re-run the instrumented binary — T2 is precisely the test that converts this from an assumption into an asserted property, and it is designed to be red today.
- That the residual ~8x is not the intra-op pool. Source says the pool is already covered; I have not measured a pool thread's MXCSR. D4's `intra_threads = 1` run settles it empirically.
- That the fixed path is bit-identical to the committed fixture. §16.21 item 6 says so; §1 shows that evidence may have been vacuous. D3 re-measures rather than inherits.

**Not consulted, deliberately:** upstream PanelCleaner. `docs/WORKSTATE.md:519` already records the verified reason — upstream's OCR path is `transformers.VisionEncoderDecoderModel.generate()` in PyTorch and never touches ONNX Runtime, so the substrate of this failure is absent from its stack. It is not an oracle for a question about ORT's `std::once_flag`. Running it here would produce a comparison with no bearing on the decision.
