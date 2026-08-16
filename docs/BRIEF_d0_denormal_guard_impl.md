# BRIEF — D0: `pc_ort::denormal` module implementation

Implements §16.52 item 5's D0 task (`docs/PIPELINE_SPEC_V1.md`, search "## 16.52"). Read
that section in full for context before starting — this brief is deliberately surgical,
not a re-explanation.

## Scope

Create `crates/pc-ort/src/denormal.rs`, a new module in the `pc-ort` crate, and wire it
into `crates/pc-ort/src/lib.rs`. This module needs **no `ort` dependency and no `onnx`
feature** — it is a standalone x86_64 MXCSR primitive. It must compile and its tests must
run in the **default feature tier**.

**The test file already exists and is genuinely red — do not edit it.**
`crates/pc-ort/tests/denormal_guard.rs` was written first, confirmed to fail to compile
(`could not find 'denormal' in 'pc_ort'`) before this brief was written. Your job is to
make it compile and pass, not to change what it asserts. If you believe an assertion in
it is wrong, stop and report that — do not silently edit the test.

## Required public API (inferred from the test file — match it exactly)

```rust
/// Observed denormal-handling state of the CURRENT thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushState {
    pub flush_to_zero: bool,
    pub denormals_are_zero: bool,
}

/// RAII scope. On construction, saves the calling thread's current MXCSR word, then sets
/// FTZ (bit 15) and DAZ (bit 6). On Drop — including during an unwind — restores the
/// EXACT saved word, not a hard-coded default. x86_64 only.
pub struct DenormalFlushGuard { /* private */ }

impl DenormalFlushGuard {
    pub fn enable() -> Self;
}

impl Drop for DenormalFlushGuard {
    fn drop(&mut self);
}

// Test-only accessors. Read the raw MXCSR control word / write an arbitrary one. These
// exist so the test file can construct precise prior-state scenarios (see
// `the_guard_restores_the_exact_prior_mxcsr_word_rather_than_a_hard_coded_default` and
// `ftz_and_daz_are_independently_verifiable_bits_neither_probe_satisfied_by_the_other`).
pub fn read_mxcsr_for_test() -> u32;
pub fn write_mxcsr_for_test(value: u32);
```

Also expose, for later tasks (D1 needs these, not this brief, but keep the surface in
mind so you don't paint yourself into a corner):

```rust
pub fn denormal_control_available() -> bool; // x86_64 + SSE3 check; not exercised by
                                              // this brief's tests, but should exist so
                                              // D1 doesn't need to touch this file again
pub fn flush_state() -> Option<FlushState>;  // read-only probe of the CURRENT thread's
                                              // FTZ/DAZ bits, via read_mxcsr's FTZ/DAZ bit
                                              // masks — None on non-x86_64
```

## Implementation constraints, load-bearing, not optional

- **Use inline assembly, not the `_mm_getcsr`/`_mm_setcsr` intrinsics.** Those intrinsics
  are `#[deprecated]` on this project's pinned toolchain (`rustc 1.94.1`), with the
  message "use inline assembly instead" — and this project's clippy gate runs with
  `-D warnings`, which turns that deprecation into a hard compile error. Use
  `core::arch::asm!("stmxcsr [{}]", ...)` / `core::arch::asm!("ldmxcsr [{}]", ...)`
  (or the equivalent `in(reg)`/`out(reg)` form) to read/write the 32-bit MXCSR word.
- **This is the workspace's first `unsafe` code.** `Cargo.toml` now has
  `[workspace.lints.rust] unsafe_code = "deny"`, and `crates/pc-ort/Cargo.toml` opts in
  via `[lints] workspace = true` (both already added, do not touch). You need exactly one
  `#[allow(unsafe_code)]` at the top of `denormal.rs` — do not scatter `#[allow]`s
  elsewhere in the crate, and do not weaken the workspace lint.
- **FTZ = bit 15 (`0x8000`), DAZ = bit 6 (`0x0040`)**, per Intel SDM Vol.1 §10.2.3. The
  test file hard-codes these independently of your implementation (`FTZ: u32 = 1 << 15`,
  `DAZ: u32 = 1 << 6`) — deliberately, so your implementation can't accidentally validate
  itself against its own wrong constant. Match the real bit positions.
- **The guard must restore the *exact* saved word**, not `0x1f80` or any other hard-coded
  default. Save the full 32-bit MXCSR on construction (`stmxcsr`), OR the FTZ/DAZ bits
  with it — on drop, write the saved value back verbatim (`ldmxcsr`). Do not clear/set
  only the two bits on drop — the nesting test (`nested_guards_restore_each_layers_own_
  saved_word_in_order`) and the panic test both depend on exact-word restoration, not
  bit-twiddling restoration.
- **MXCSR is per-thread on x86_64.** No shared/global state — each `DenormalFlushGuard`
  reads/writes only the calling thread's own control register. Do not use a `static`/
  `thread_local!` to fake this; the actual CPU instruction is already per-thread, so a
  correct `stmxcsr`/`ldmxcsr` implementation gets this for free.
- **`#[cfg(target_arch = "x86_64")]` the whole module** (the test file itself is
  `#![cfg(target_arch = "x86_64")]`-gated, matching this). Do not attempt a non-x86_64
  fallback in this task — that's out of scope.

## What NOT to do

- Do not touch `crates/pc-detect/`, `crates/pc-ocr/`, or `crates/pc-inpaint/` — that's
  D1/D2/D4, separate future tasks. This brief is `pc-ort` only.
- Do not add a timing/benchmark assertion anywhere in this test file or a new one — §16.52
  item 5 is explicit that D0's gate has zero model, zero ONNX Runtime, zero timing.
- Do not edit `crates/pc-ort/tests/denormal_guard.rs`. If a test in it seems wrong, stop
  and report why — don't fix it yourself.

## Verification bar for this task

Since this is Spec-sensitive tier work (§16.52), you are the implementer, so per the
standing cmdc preamble in `CLAUDE.md`, run the full four-command bar yourself and report
the actual output, not just "tests pass":

1. `cargo test -p pc-ort --test denormal_guard` — must go from red (compile error) to
   green (6 tests passing). Quote the actual passing output.
2. `cargo test --workspace` — confirm nothing else broke.
3. `cargo clippy --workspace --all-targets --all-features -- -D warnings` — must be
   clean. This is the check that would have caught `_mm_getcsr`/`_mm_setcsr` deprecation
   as an error, so don't skip it.
4. `cargo fmt --all --check` — must be clean.

Report `git diff --stat` for exactly what you touched, and confirm you did not edit
`crates/pc-ort/tests/denormal_guard.rs`, `Cargo.toml`, or `crates/pc-ort/Cargo.toml`
(all three were prepared ahead of this dispatch and are out of your scope to change).
