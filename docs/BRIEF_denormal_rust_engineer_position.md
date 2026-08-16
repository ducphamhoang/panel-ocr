# rust-engineer's denormal-fix position (2026-08-16, joint planning pass vs. `docs/BRIEF_denormal_fix_plan.md`)

## Independent design position — the ineffective denormal flush (`docs/BRIEF_denormal_fix_plan.md`)

I settled the mechanism by **running** it, not by reading `ort`. All numbers below are real output from a throwaway probe I built against the pinned `ort = "=2.0.0-rc.12"` with the same feature set as the workspace, driving the real `comictextdetector.pt.onnx` at `C:\Users\ducph\AppData\Local\panel-ocr\models\comictextdetector.pt.onnx`. Probe source: `C:\Users\ducph\AppData\Local\Temp\claude\E--game-prototypes-panel-ocr\25bef249-a7e4-41e0-bd25-ede6b3a81c97\scratchpad\mxcsr-probe\src\main.rs` (scratchpad only — `git diff --stat -- crates/ xtask/ Cargo.toml docs/PIPELINE_SPEC_V1.md` returns empty; I changed nothing in the tree).

---

## 1. Why `with_flush_to_zero()` doesn't work today — measured, and it is *not* what the brief assumed

### What the source says

Two distinct methods share the name in rc.12:

- `SessionBuilder::with_flush_to_zero()` — `ort-2.0.0-rc.12/src/session/builder/impl_config_keys.rs:36-40`, which is only `add_config_entry("session.set_denormal_as_zero", "1")`. This is the one `crates/pc-detect/src/onnx.rs:590` calls.
- `GlobalThreadPoolOptions::with_flush_to_zero()` — `environment.rs:328-331`, `SetGlobalDenormalAsZero`. Applies to an *environment-level* global pool. We never configure one, so it is irrelevant here.

### What ONNX Runtime actually does with that key — measured

`session.set_denormal_as_zero` has **two effects with different scopes**, and only one of them is reliable:

| effect | scope | order-dependent? |
|---|---|---|
| (i) FTZ+DAZ on the session's **own intra-op pool threads** | per-session | **No** — always works |
| (ii) FTZ+DAZ on the **thread that initializes the session** (and therefore the thread that later calls `Session::run`) | **process-wide, once** | **Yes** — the *first* session to initialize in the process decides, permanently |

Evidence, verbatim probe output:

**CASE A — flush-requesting session is the first in the process:**
```
caller AFTER  build: mxcsr=0x9fe0 ftz=1 daz=1
  pool[0] AT ENTRY (before work()): mxcsr=0x1f80 ftz=0 daz=0     <- ORT sets it inside work(), not at spawn
  ...11 pool threads...
  RUN 1: elapsed=426.7006ms   RUN 2: elapsed=367.9153ms   RUN 3: elapsed=357.2668ms
  pool[0] AT EXIT (after work()): mxcsr=0x9fe0 ftz=1 daz=1
```

**CASE G — a *non*-flushing session initializes first, then a flush-requesting session on a separate thread (this is our pipeline):**
```
  worker: MXCSR after  session#2 build: mxcsr=0x1fa0 ftz=0 daz=0   <- the flag never reaches the run-caller
  worker RUN 1: elapsed=896.9826ms
  worker RUN 2: elapsed=873.2801ms
  session#2 pool[0] AT EXIT: mxcsr=0x9fe0 ftz=1 daz=1              <- but the pool threads DID get it
```

**So: `with_flush_to_zero()` is not broken and is not mis-wired. It is *stolen*.** There are exactly three production session-construction sites (enumerated with `grep -rn "Session::builder" --include=*.rs crates/ xtask/`, output read to the end, counted per file — 3 in `src/`, 1 in a test):

- `crates/pc-detect/src/onnx.rs:569` — the **only** one that requests flushing (`:589-593`)
- `crates/pc-ocr/src/onnx.rs:445` — does not
- `crates/pc-inpaint/src/onnx.rs:407` — does not

All three are lazily constructed behind independent `OnceLock`s (`crates/pc-cli/src/detector.rs:60`, `crates/pc-cli/src/inpainter.rs:110`) under a rayon pool. **Which one initializes first is a scheduling outcome, not a program property.** Whenever it isn't the detector, the detector's `pc-detect-onnx` worker runs `Session::run` with `ftz=0 daz=0` — precisely the state the brief measured.

### Cost, measured, same model, same synthetic 1×3×1024×1024 input

| configuration | per-inference |
|---|---|
| no flush requested anywhere | **335.5 s** |
| flush requested, process-once **stolen** (pool flushed, run-caller not) | 0.873 – 0.897 s |
| flush requested, process-once **won** | 0.357 – 0.427 s |
| process-once stolen **+ we set MXCSR on the run-caller ourselves** | 0.383 – 0.438 s |

---

## 2. The brief's preferred option is measurably wrong — reject it

> *"setting MXCSR via a custom thread-creation hook if `ort`/ONNX Runtime exposes one"*

The hook exists (`ort-2.0.0-rc.12/src/session/builder/impl_options.rs:283`, `SessionBuilder::with_thread_manager`). **It does not work for this.** CASE C set FTZ+DAZ on all 11 pool threads before calling `work()`, with the ORT key off:

```
  pool[0] AT ENTRY (before work()): mxcsr=0x1f80 ftz=0 daz=0
caller AFTER  our set_ftz_daz: mxcsr=0x9fe0 ftz=1 daz=1
  RUN 1: elapsed=2.806472s   RUN 2: elapsed=91.9720044s   RUN 3: elapsed=94.4250229s
  pool[0] AT EXIT (after work()): mxcsr=0x1fb2 ftz=0 daz=0        <- ORT CLOBBERED our value
```

ORT writes MXCSR from the session's own option inside `work()`, overwriting anything the thread manager set. **The ThreadManager hook is a dead end for pool threads; only ORT's config key reaches them.**

The brief's other option — *"setting it process-wide before any session/thread pool exists"* — must also be rejected, for two independent reasons: it makes `SessionTuning { flush_denormals: false }` unrepresentable (which the frozen `flush_denormals_does_not_escape_the_detector_worker_thread` baseline arm and `flushing_denormals_does_not_change_one_raw_output_float`'s off-arm both depend on), and it silently changes float semantics for every pure-Rust stage — the exact hazard §16.32 was ratified to prevent.

---

## 3. Proposed fix: hybrid, because the two thread populations need two different mechanisms

1. **Keep `with_flush_to_zero()`.** It is the only thing that reaches ORT's intra-op pool threads, and that half is order-independent (CASE G).
2. **Add an explicit RAII MXCSR guard in `pc-ort`** (`DenormalFlushGuard`): save MXCSR, `|= FTZ|DAZ`, restore the exact saved word on drop. `x86_64` only; a no-op elsewhere.
3. **Apply it on the thread that calls `Session::run`** — in `pc-detect`'s `infer()` (`crates/pc-detect/src/onnx.rs:667-672`), on the confined worker — **conditionally on `tuning.flush_denormals`**, which requires threading that flag onto the worker.

This is strictly stronger than today's arrangement: it is order-independent, it is scoped to a *dynamic extent* rather than a thread lifetime, and CASE H confirms it recovers full speed under the poisoned order (0.383–0.438 s vs. the unfixed 0.873–0.897 s).

### The risk I de-risked before proposing it

Enabling FTZ/DAZ on the run-caller could have perturbed floats and turned the **frozen** `perf2_bit_exactness.rs` fixtures red. Measured, two processes, identical input, FNV-1a over every raw output f32:

```
J (once WON, caller ftz=1 daz=1): elapsed=419.2051ms floats=3597312 fnv1a64=0xb4f9c5acfd873185
I (once STOLEN, caller ftz=0 daz=0): elapsed=923.4546ms floats=3597312 fnv1a64=0xb4f9c5acfd873185
```

Identical digest. `floats=3597312` also matches `EXPECTED_RAW_FLOATS = 3_597_312` in `perf2_bit_exactness.rs` — an independent cross-check that the probe hit the same model and shape. **Caveat, stated because it is a real limit: one synthetic input, one machine.** The frozen opt-in tests on the *real* recorded page must be re-run against the fix; that is a gate condition, not an assumption.

---

## 4. §16.21 item 6 / §16.32 — was it wrong, incomplete, or regressed?

**Incomplete. Not wrong, and not regressed by a later change.**

§16.21 item 6's closure and §16.32's numbers were honestly measured — in a **detector-only benchmark process**, where the detector's session is necessarily the first to initialize, i.e. exactly CASE A. Neither section established, or could have established from that setup, that effect (ii) is a process-wide once that any earlier non-flushing session consumes. §16.32 item 2(b)'s worker-thread confinement is correct and still needed; what is missing is that the flag's coverage of the *inference-calling* thread was never order-robust.

Two consequences for the transcription, which I am flagging rather than resolving:

- §16.32 item 2(b) ratifies a **bundle** (flag + thread confinement). Adding a scoped MXCSR guard changes that bundle and needs a §16.x entry. It should quote §16.32's own scope verbatim beside the correction, per `CLAUDE.md`'s scope-travels rule.
- §16.21 item 6's "**CLOSED, 2026-08-03**" paragraph asserts *"this machine now faster than the recorded cv2.dnn baseline on the same page"* unconditionally. That claim is true only when the detector wins the process-once. It needs qualifying, not deleting.

**Existing frozen tests: none break.** I checked each:

- `perf2_mxcsr_hygiene.rs:70` `flush_denormals_does_not_escape_the_detector_worker_thread` constructs the **flush** detector first (`:87-90`), so it already wins the once; its `speedup >= 3.0` and both hygiene assertions stay satisfied, and the guard's restore-on-drop makes the hygiene half stronger, not weaker.
- `perf2_bit_exactness.rs`'s `the_real_detector_reproduces_the_committed_fixture_at_every_thread_count` iterates `off` **then** `on`, so its "on" arm has *always* been running with a stolen once and an unflushed caller. After the fix that arm genuinely flushes on the caller. Bit-identical per the I/J measurement above — but this is the exact test that must be re-run on real weights before the change is believed.
- `perf2_tuning.rs`'s default test and `perf2_from_path_smoke.rs` are unaffected (the latter gets faster).

---

## 5. `pc-ocr` and `pc-inpaint` — my position is *measure*, do not extend the fix blind

**It is not provably detector-specific.** Denormal exposure is a property of weight/activation magnitudes, not of input-size fixedness, so "fixed-size-input stages may not be equally exposed" is a hypothesis with nothing behind it. LaMa is a Conv/FFC net of the same family as the detector; manga-ocr is ViT + GPT-2. But I have measured neither, and I will not draft tests for a change whose need is unestablished.

One structural fact that *is* established and that shapes whatever comes next: both run `Session::run` **on the caller thread under a `Mutex<Session>`** (`crates/pc-ocr/src/onnx.rs:60-61`, `crates/pc-inpaint/src/onnx.rs:253`) — a rayon worker that also runs pure-Rust masking/denoise/compositing float code. §16.32's thread-confinement design is therefore *unavailable* to them. If they ever adopt flushing:

- the RAII guard's **scoped** extent is the only safe mechanism there, and
- calling `with_flush_to_zero()` in either of them introduces a *new* hazard: if that session wins the process-once, ORT sets FTZ+DAZ **permanently and unrestorably** on a rayon worker thread. That would need session *construction* to be wrapped in the same guard so ORT's leak is undone. I deliberately keep construction-guarding out of the detector task and attach it to this one, where it is actually load-bearing.

---

## 6. Drafted tests

### T1 — `crates/pc-ort/tests/denormal_guard.rs` (default tier, no model, no ONNX Runtime)

Verifies §16.x item (1): the guard sets FTZ+DAZ for its scope and restores the exact prior MXCSR word.

```rust
//! FROZEN. Gate for the scoped denormal-flush guard (§16.x item 1).
//!
//! The guard exists because ONNX Runtime's `session.set_denormal_as_zero` reaches the
//! inference-CALLING thread only through a process-wide once, which the first session to
//! initialize in the process consumes (measured 2026-08-16, ort =2.0.0-rc.12).
#![cfg(target_arch = "x86_64")]

use pc_ort::denormal::DenormalFlushGuard;

// Hard-coded from Intel SDM Vol.1 §10.2.3, deliberately NOT imported from the crate under
// test: an expectation derived from its own subject passes vacuously (cookbook rule 7/13).
const FTZ: u32 = 1 << 15;
const DAZ: u32 = 1 << 6;
const ABI_DEFAULT_MXCSR: u32 = 0x1f80;
const ROUND_TOWARD_NEG_INF: u32 = 0x2000; // MXCSR.RC = 01; neither FTZ nor DAZ

/// A denormal-flush oracle that reads no MXCSR bit at all — it does arithmetic and looks
/// at the result. Independent of the bits the guard writes, so the bit-level and
/// behavioural assertions below cannot confirm each other.
fn denormal_survives_on_this_thread() -> bool {
    let smallest_normal = std::hint::black_box(f32::MIN_POSITIVE);
    let divisor = std::hint::black_box(2.0_f32);
    std::hint::black_box(smallest_normal / divisor) != 0.0
}

#[test]
fn a_subnormal_f32_flushes_to_zero_inside_the_guards_scope_and_survives_after_its_drop() {
    assert!(
        denormal_survives_on_this_thread(),
        "precondition: this test thread must start with denormals intact"
    );

    let inside = {
        let _guard = DenormalFlushGuard::enable();
        denormal_survives_on_this_thread()
    };
    let after = denormal_survives_on_this_thread();

    // Asserted as a pair, so a guard that never enables flushing (false, true is
    // impossible for it) and a guard that never restores are each rejected by a
    // DIFFERENT element. A single-sided assertion could not tell them apart.
    assert_eq!(
        (inside, after),
        (false, true),
        "(denormal survives inside guard, denormal survives after drop)"
    );
}

#[test]
fn the_guard_restores_the_exact_prior_mxcsr_word_rather_than_a_hard_coded_default() {
    let original = pc_ort::denormal::read_mxcsr_for_test();
    // A prior state that is NOT the ABI default and carries neither FTZ nor DAZ, so a
    // restore that writes 0x1f80 fails here while passing a test that started at 0x1f80.
    let prior = (original | ROUND_TOWARD_NEG_INF) & !(FTZ | DAZ);
    assert_ne!(
        prior, ABI_DEFAULT_MXCSR,
        "the probe state must differ from the ABI default or this test cannot fail"
    );

    pc_ort::denormal::write_mxcsr_for_test(prior);
    let inside = pc_ort::denormal::read_mxcsr_for_test_inside(|| {
        let _guard = DenormalFlushGuard::enable();
        pc_ort::denormal::read_mxcsr_for_test()
    });
    let after = pc_ort::denormal::read_mxcsr_for_test();
    pc_ort::denormal::write_mxcsr_for_test(original); // leave the thread as we found it

    assert_eq!(
        (inside, after),
        (prior | FTZ | DAZ, prior),
        "(mxcsr inside guard, mxcsr after drop) — the guard must set exactly FTZ|DAZ and \
         restore the exact saved word, touching no other field (e.g. the RC rounding mode)"
    );
}
```

*(`read_mxcsr_for_test_inside` is only a closure-runner so the read happens before the guard drops; if the implementer prefers a plain block, that is a mechanical detail, not an assertion change.)*

**What turns each red:** test 1 — omitting the `_mm_setcsr` on construction gives `(true, true)`; omitting restore gives `(false, false)`. Test 2 — restoring `0x1f80` instead of the saved word; setting FTZ without DAZ; clobbering the rounding-mode field. **Anti-vacuity literals:** `0x1f80`, `0x2000`, `1<<15`, `1<<6`, and the exact expected word `prior | FTZ | DAZ`.

**Requirement traced:** the guard's contract — "sets flushing for a bounded extent and leaves the thread exactly as found" — which is what makes it compatible with §16.32's hygiene requirement instead of a second leak.

### T2 — `crates/pc-detect/tests/perf3_denormal_order.rs` (onnx feature, opt-in real weights)

Verifies §16.x item (2): the detector's denormal flushing must not depend on which ONNX Runtime session happened to initialize first in the process. **This is the test that is red today and green after the fix.**

Requires one new hook in `crates/pc-detect/src/onnx.rs`, under `#[cfg(any(test, feature = "testkit"))]`. Its shape is load-bearing, so I state it rather than leave it to the implementer: `run_worker` must route **both** inference and the probe through **one** shared policy wrapper —

```rust
fn with_denormal_policy<R>(flush: bool, body: impl FnOnce() -> R) -> R {
    let _guard = flush.then(DenormalFlushGuard::enable);
    body()
}
```

— so the probe exercises the production wrapper rather than a parallel copy of it (cookbook rule 12: a gate aimed at a copy proves nothing about the original).

```rust
#![cfg(all(feature = "onnx", target_arch = "x86_64"))]

use pc_detect::onnx::{OnnxDetector, SessionTuning};
use std::path::PathBuf;
use std::sync::Mutex;

static REAL_MODEL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn model() -> PathBuf {
    PathBuf::from(
        std::env::var_os("PANEL_OCR_ONNX_MODEL").expect("PANEL_OCR_ONNX_MODEL is required"),
    )
}

/// §16.x item 2. Measured root cause this pins (ort =2.0.0-rc.12 + its bundled ONNX
/// Runtime, 2026-08-16): `session.set_denormal_as_zero` reaches a session's own intra-op
/// POOL threads per-session, but reaches the thread that CALLS `Session::run` only through
/// a process-wide once that the first session to initialize consumes. `pc-detect` is the
/// only one of the three production session sites that requests flushing
/// (`crates/pc-ocr/src/onnx.rs:445` and `crates/pc-inpaint/src/onnx.rs:407` do not), and
/// all three initialize lazily under a rayon pool — so before this gate, whether the
/// detector flushed at all was a thread-scheduling outcome.
#[test]
#[ignore = "opt-in: needs PANEL_OCR_ONNX_MODEL and ONNX Runtime; run with \
            `PANEL_OCR_ONNX_MODEL=/path/comictextdetector.pt.onnx cargo test -p pc-detect \
             --features onnx,testkit,bench-tuning --test perf3_denormal_order -- --ignored --nocapture`"]
fn the_detector_worker_flushes_denormals_even_when_a_non_flushing_session_initialized_first() {
    let _test_guard = REAL_MODEL_TEST_LOCK
        .lock()
        .expect("real-model test lock must not be poisoned");

    // Deliberately consume ONNX Runtime's process-wide denormal once with a session that
    // does NOT request flushing. This ordering IS the regression.
    let non_flushing = OnnxDetector::from_path_with_tuning(
        &model(),
        0,
        0,
        &SessionTuning { flush_denormals: false, ..SessionTuning::default() },
    )
    .expect("non-flushing session must open");

    let flushing =
        OnnxDetector::from_path_with_tuning(&model(), 0, 0, &SessionTuning::default())
            .expect("flushing session must open");

    let observed = (
        non_flushing
            .worker_flushes_denormals_during_inference()
            .expect("non-flushing worker must answer the probe"),
        flushing
            .worker_flushes_denormals_during_inference()
            .expect("flushing worker must answer the probe"),
    );

    // A PAIR, not a single boolean. `(false, false)` is today's defect. `(true, true)` is
    // the tempting wrong fix — a process-wide or unconditional set — which would make
    // `flush_denormals: false` unrepresentable and silently change float semantics for
    // every pure-Rust stage (the hazard §16.32 exists to prevent). Only `(false, true)`
    // is the ratified behaviour, and only an assertion over both arms can tell the three
    // apart; a one-sided "the flushing detector flushes" passes for all of them.
    assert_eq!(
        observed,
        (false, true),
        "(non-flushing worker flushes denormals, flushing worker flushes denormals)"
    );
}
```

**What turns it red:** no guard at all → `(false, false)`; guard applied unconditionally, or a process-wide `_mm_setcsr` at startup → `(true, true)`; guard placed on the caller thread instead of the worker → `(false, false)`.

**Honest limit, stated so the name does not over-claim:** this pins that the *policy wrapper is active on the worker with the correct per-session flag*. It does not itself prove `session.run` is textually inside the wrapper. That placement is pinned by the existing opt-in wall-clock tests (`perf2_from_path_smoke.rs:28`'s `< 5s`, `perf2_mxcsr_hygiene.rs:116`'s `speedup >= 3.0`) plus the fact that `infer` is the wrapper's only other call site. I am deliberately **not** adding a new timing assertion — the brief asked for a non-fragile gate and the pair-assertion above is it.

### Tests I deliberately did **not** draft, and why

- **A bit-exactness test for the fix.** `perf2_bit_exactness.rs`'s `flushing_denormals_does_not_change_one_raw_output_float` and `the_real_detector_reproduces_the_committed_fixture_at_every_thread_count` already do exactly this, against hard-coded independent oracles (`EXPECTED_RAW_FLOATS`, `expected_blocks()`, the committed mask and JSON), and already in the poisoned order. Duplicating them would pad coverage. **Re-running them on real weights is a gate condition on T2**, not a new test.
- **A "every session site declares a denormal policy" enumeration gate** (cookbook rule 13/14 shape). After the fix the detector no longer depends on the process-once, so a fourth non-flushing site could no longer steal anything from it — the gate would guard a hazard the fix removes. It becomes worth having *only if* `pc-ocr`/`pc-inpaint` adopt flushing; I would attach it to that task, not this one.
- **Anything for `pc-ocr`/`pc-inpaint`.** Their need is unmeasured. Drafting tests for an unestablished requirement is the "test that cannot name its requirement" case.

---

## 7. Task breakdown and classification

| # | task | scope | class |
|---|---|---|---|
| **T-R** | Ratify §16.x: the process-once finding, the correction to §16.32 item 2(b)'s bundle, the qualification of §16.21 item 6's CLOSED paragraph. Quote both sections' scope verbatim beside the correction. | `docs/PIPELINE_SPEC_V1.md` only | must precede T2 |
| **T1** | `pc-ort::denormal` — `DenormalFlushGuard` + the two test-only MXCSR accessors, ungated by `onnx`, `x86_64`-conditional. T1 tests first. | `crates/pc-ort/src/` | **simple** |
| **T2** | Thread `flush_denormals` to the worker; add `with_denormal_policy`; wrap `infer`; add `WorkerRequest::ProbeDenormalState` + `worker_flushes_denormals_during_inference` under `cfg(any(test, feature="testkit"))`. T2 test first. | `crates/pc-detect/src/onnx.rs` | **heavy** — own call; touches the §16.32 bundle and needs the opt-in real-weights re-run |
| **T3** | Measure `encoder_model.onnx`, `decoder_model.onnx`, `lama-manga.onnx` for denormal exposure using the CASE A/E methodology; then decide (and only then draft tests). Includes construction-guarding if either adopts flushing. | `pc-ocr`, `pc-inpaint` | **heavy** — measurement-first, separate call |
| **T0** | *Optional diagnostic:* identify which session actually steals the once in a real 10-page batch. | none | **simple**, optional — the fix is order-independent, so this is explanatory only |

**Gate condition on T2 (beyond the four-command bar):** with real weights, `perf2_bit_exactness.rs`'s two flush-comparing tests and `perf2_mxcsr_hygiene.rs`'s hygiene test must be re-run `--ignored` and pass. If any raw-float digest moves, that is a frozen-test contradiction and goes to both architects jointly (cookbook rule 8, exit 2) — not a unilateral anything.

---

## 8. Open item I could not close, reported as unclosed

The brief's **residual ~8x growth across 10 calls (0.455 s → 3.722 s)** with the caller manually fixed. My CASE C shows growth of exactly that shape — `2.806 s → 91.97 s → 94.43 s` across three identical runs — but only in a configuration where the **pool** threads were unflushed, which is not the brief's residual configuration (there the ORT key was on, so the pool threads should have been flushed). CASE A, everything flushed, shows no growth at all (`0.427 → 0.368 → 0.357 s`). **I therefore cannot attribute the residual, and I am not going to guess.** The next measurement I would run is CASE A extended to ~10 consecutive runs on the *real recorded page* rather than my synthetic input, to establish whether the residual survives a fully-flushed configuration at all before anyone designs against it.

## Checks I ran

- `git status --short --branch`, `git diff --stat -- crates/ xtask/ Cargo.toml docs/PIPELINE_SPEC_V1.md` — **empty**; I changed nothing in the tree.
- `grep -rn "Session::builder" --include=*.rs crates/ xtask/` with a per-file count (`| awk -F: '{print $1}' | sort | uniq -c`) — 3 `src/` sites + 1 test site, output read to the end, not truncated.
- Probe: `cargo build --release` green; cases A, C, E, G, H, I, J executed against real weights, output quoted above verbatim.
- **Not run:** the four-command verification bar. There is nothing to verify — this was a planning pass with no tree changes. The drafted tests have not been compiled; they reference APIs that do not exist yet, by design.
