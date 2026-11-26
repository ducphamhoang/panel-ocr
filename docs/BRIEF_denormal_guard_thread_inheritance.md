# BRIEF (TEMPORARY, delete when consumed): frozen test `denormal_guard` contradicts Linux reality

Written 2026-11-26 by the Orchestrator for a joint `architect` + `rust-engineer` decision.
Both of you read this; you do **not** see each other's answer. Each reports an independent position.

## What is being decided

One frozen test fails deterministically on this container (Linux 6.18, x86_64, glibc 2.39):

`crates/pc-ort/tests/denormal_guard.rs` → `a_thread_spawned_inside_an_active_guard_scope_does_not_inherit_the_flush_state`
(lines ~129-139). Run: `cargo test -p pc-ort --test denormal_guard` (5 pass, 1 fail, 8/8 runs; also fails at the parent commit, so it is pre-existing).

The test's assertion message says: *"MXCSR is per-thread on x86_64 -- a freshly spawned thread must start with the ABI default state regardless of what the spawning thread's guard set"*.

## Evidence the Orchestrator has (verify it yourself, do not take it on trust)

- Out-of-repo probe `/tmp/claude-0/-home-user-panel-ocr/98ceae0c-a98c-561e-89d2-1d5a63456c1b/scratchpad/probe/probe.rs`
  (built binary `.../probe/probe`): sets `MXCSR |= FTZ|DAZ` on the main thread, then `std::thread::spawn`s a thread that reads
  MXCSR. Output: child reads `0x9fc0` (FTZ and DAZ set), i.e. the child **inherited** the parent's word.
  Main-thread initial word was `0x1f80`.
- The other 5 tests in the file (restore on drop, nesting, panic-safe restore, FTZ/DAZ independent bits, subnormal flush inside scope) pass.
- With real weights present, the opt-in tests pass: `perf2_bit_exactness` (3/3), `perf2_mxcsr_hygiene` (3/3), `perf3_denormal_order`, `perf2_from_path_smoke`.
  Run them with `PANEL_OCR_ONNX_MODEL=/root/.cache/panel-ocr/models/comictextdetector.pt.onnx cargo test -p pc-detect --features onnx,testkit,bench-tuning --test <name> -- --ignored --nocapture` (slow: `perf2_bit_exactness` ~8 min; you do not need to rerun it).
- The Orchestrator believes the over-claim lives **only in that test's message**, not in spec §16.52 or in `pc_ort::denormal`'s docs
  (they claim "MXCSR is a per-thread register", which is true). **That is a belief, not a finding: check it.**

## What the Orchestrator does NOT know

- Whether Windows / Intel-macOS new threads start from the default MXCSR (the test presumably passed on the maintainer's Windows machine when committed). You cannot run those; say what is documented vs assumed.
- Whether `.github/workflows/ci.yml`'s `ubuntu-latest` x86_64 leg is red for this reason (CI status not inspectable from here).
- Whether anything in production depends on a spawned thread **not** inheriting FTZ/DAZ (e.g. threads the detector worker or ONNX Runtime spawn while a `DenormalFlushGuard` is live), or conversely on it inheriting. Trace this; the measured bit-exactness tests pass, but say why/whether that is sufficient.

## Files to read first

- `crates/pc-ort/tests/denormal_guard.rs` (whole), `crates/pc-ort/src/denormal.rs` (module docs + guard)
- `docs/PIPELINE_SPEC_V1.md` §16.52 (starts ~line 9551; item 3 is the guard ruling), and grep it for MXCSR/thread claims
- `docs/COOKBOOK.md` — **"the four legitimate exits from a frozen test"** and the over-claims list; CLAUDE.md "Pipeline" step 2 exception: a test that contradicts the spec goes to the two architects jointly, never a unilateral edit
- `crates/pc-detect/src/onnx.rs` and `crates/pc-detect/tests/perf2_mxcsr_hygiene.rs` (how the guard is actually used and what "hygiene" is gated)
- `.github/workflows/ci.yml` (OS matrix)
- `docs/WORKSTATE.md`, section "2026-11-26 (cloud session…)" at the end, for the full record

## Hard constraints

- **Do not edit, write, stage or commit anything in the repo.** This is a decision, not an implementation. (`rust-engineer` has write tools; do not use them here. Shell writes are also off-limits: your read-only status is by instruction, not enforcement.)
- The frozen test must not be changed by anyone but through the process; propose text, do not apply it.
- Do not run the full workspace suite; scoped `cargo test -p pc-ort --test denormal_guard` and reading are enough. Scratch work goes under the session scratchpad dir only.

## Report (keep it tight; findings, not narration)

1. **Fact check**: which of the Orchestrator's claims above you confirmed, which you refuted, and how (command + output).
2. **Is the test's premise wrong, or is the guard/spec wrong?** Cite lines. Include whether any other site (test, doc, spec, comment) carries the same premise.
3. **Production impact** of child inheritance, with the code path you traced.
4. **Options** (at least: drop/weaken the assertion to what holds everywhere; replace with a documented per-OS expectation; `cfg`-gate; anything else you think is better) with the cost/risk of each, including what each would stop the suite from catching.
5. **One recommendation** with the *exact* proposed replacement text/`cfg`, which cookbook exit it is, and whether the spec needs a §16.x note (if so, say what it must quote verbatim, per the transcription rule).
6. **What you could not verify** and what would settle it.
7. Anything in this brief you think is wrong or leading.
