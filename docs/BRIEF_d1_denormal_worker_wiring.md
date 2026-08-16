# BRIEF — D1: wire `pc_ort::denormal` into `pc-detect`'s worker

Implements §16.52 item 5's D1 task (`docs/PIPELINE_SPEC_V1.md`, search "## 16.52"). D0
(`crates/pc-ort/src/denormal.rs`) is already implemented and committed (`4f4f36b`) — read
it before starting, it's the only primitive this task uses.

## The test file already exists and is genuinely red — do not edit it

`crates/pc-detect/tests/perf3_denormal_order.rs` was written first and confirmed to fail
to compile (`no method named 'last_inference_flush_state' found for struct 'OnnxDetector'`)
before this brief was written. It is `#[ignore]`d (needs `PANEL_OCR_ONNX_MODEL` + real
ONNX weights) — you do NOT need real weights to make it compile; you only need real
weights if you want to actually run it (optional, but recommended if the managed model
cache is available on this machine: run `panel-ocr models path` to check, and the file's
own `#[ignore = "..."]` message has the exact env-var-prefixed command). If you cannot run
it for real, that's fine — report that explicitly rather than claiming you did.

## Scope

Modify `crates/pc-detect/src/onnx.rs` only. No other file.

1. **Add a field to `OnnxDetector`**, gated `#[cfg(target_arch = "x86_64")]` (the concept
   doesn't exist on other architectures — the field, its accessor, and its wiring are all
   x86_64-only; this matches `pc_ort::denormal`'s own arch gate and the test file's own
   `#![cfg(all(feature = "onnx", target_arch = "x86_64"))]`):

   ```rust
   #[cfg(target_arch = "x86_64")]
   last_flush_state: std::sync::Arc<std::sync::atomic::AtomicU8>,
   ```

   Encode `Option<pc_ort::denormal::FlushState>` into a single `u8` so it can be an atomic
   (one relaxed store per inference, per §16.52 item 3(a) — "cost is one relaxed atomic
   store per ~0.5s inference, not a real cost"):
   - `0` = never recorded (the `None` state, e.g. before the first inference).
   - Bit 2 (`0b100`) set = "recorded"; bit 0 = `flush_to_zero`; bit 1 = `denormals_are_zero`.
   - So: `encode(state) = 0b100 | (state.flush_to_zero as u8) | ((state.denormals_are_zero as u8) << 1)`.
   - Decode: `if raw & 0b100 == 0 { None } else { Some(FlushState { flush_to_zero: raw & 1 != 0, denormals_are_zero: raw & 2 != 0 }) }`.

2. **Initialize it in `from_path_with_tuning_impl`**: `Arc::new(AtomicU8::new(0))`, cloned
   into the worker closure the same way `worker_model_path`/`worker_tuning`/`worker_policy`
   already are. Add it to the `Self { ... }` struct literal (gated the same way as the
   field).

3. **Thread `tuning.flush_denormals` and the atomic through to the worker.** `run_worker`'s
   signature currently is `fn run_worker(session: &mut Session, requests: mpsc::Receiver<WorkerRequest>)`.
   Add two parameters: `flush_denormals: bool` and (x86_64-gated)
   `last_flush_state: &std::sync::atomic::AtomicU8`. Thread both down into `infer()`, whose
   signature currently is `fn infer(session: &mut Session, input: Tensor<f32>) -> Result<WorkerOutputs, StageError>` —
   add the same two parameters.

4. **Inside `infer()`, immediately before `session.run(...)`** (x86_64-gated block):
   ```rust
   let _guard = flush_denormals.then(pc_ort::denormal::DenormalFlushGuard::enable);
   if let Some(state) = pc_ort::denormal::flush_state() {
       last_flush_state.store(encode(state), std::sync::atomic::Ordering::Relaxed);
   }
   ```
   The guard must stay alive across the `session.run(...)` call (i.e. don't drop it before
   the call — bind it to `_guard`, not `_`, and let it drop naturally at the end of
   `infer`'s scope, which is after `session.run` completes). On non-x86_64, `flush_denormals`
   is simply unused by this function (silence any unused-parameter warning with
   `let _ = flush_denormals;` inside a `#[cfg(not(target_arch = "x86_64"))]` block, or an
   equivalent — your call on the exact mechanism, as long as it compiles clean under
   `-D warnings` on x86_64; non-x86_64 correctness isn't independently verifiable in this
   session, so don't over-engineer it, just don't leave an unused-variable warning path
   that would only show up on a different target).

5. **Add the ungated-in-name-but-x86_64-gated-in-existence public accessor**:
   ```rust
   #[cfg(target_arch = "x86_64")]
   pub fn last_inference_flush_state(&self) -> Option<pc_ort::denormal::FlushState> {
       // decode self.last_flush_state.load(Ordering::Relaxed) per the encoding above
   }
   ```
   This is deliberately **not** gated behind `#[cfg(any(test, feature = "testkit"))]` or
   similar — §16.52 item 3(a) ratified it as an ungated, always-shipped observation of the
   real inference path (unlike the injection hooks like `panic_next_infer_for_test`, which
   change behavior and correctly stay test-only). The only gate is architecture, because
   the concept is meaningless off x86_64.

## What NOT to do

- Do not touch `crates/pc-ocr/` or `crates/pc-inpaint/` — that's D4, a separate, currently
  deferred future task (§16.52 ruling 3(b): no defensive changes there in this pass).
- Do not add a `WorkerRequest::ProbeDenormalState` variant or any gated probe-request
  machinery — §16.52 item 3(a) explicitly dissolved that design in favor of the plain
  ungated accessor plus a real `detect()` call. Keep it simple: one atomic, one accessor.
- Do not edit `crates/pc-detect/tests/perf3_denormal_order.rs`. If an assertion in it
  seems wrong, stop and report why — don't fix it yourself.
- Do not add a timing assertion anywhere. This task's gate is behavioral (does the flag
  actually apply, pair-assertion `(false, false, true, true)`), not a speed proxy.

## Verification bar for this task

Spec-sensitive tier — you are the implementer, so per the standing cmdc preamble in
`CLAUDE.md`, run the full four-command bar yourself and report the actual output:

1. `cargo test -p pc-detect --features onnx,testkit,bench-tuning --test perf3_denormal_order -- --ignored`
   — first confirm it now COMPILES (the two `E0599` errors from before this brief must be
   gone). If `PANEL_OCR_ONNX_MODEL` is set to a real cached `comictextdetector.pt.onnx`
   on this machine, actually run it and report the real pass/fail. If not, report clearly
   that you verified compilation only, not a real run — do not claim a run you didn't do.
2. `cargo test --workspace` — confirm nothing else broke (196 binaries as of the last
   commit before this task).
3. `cargo test --workspace --all-targets --features pc-cli/onnx` — the onnx tier; also
   compiles `crates/pc-detect/tests/perf2_mxcsr_hygiene.rs` and friends, which must still
   pass unchanged (they don't touch `last_inference_flush_state` at all, so they should
   be unaffected — if any of them regress, that's a real problem, stop and report it).
4. `cargo clippy --workspace --all-targets --all-features -- -D warnings` — must be
   clean. Since this touches `pc-detect`, which is NOT currently opted into the
   `[workspace.lints.rust] unsafe_code = "deny"` lint (only `pc-ort` is), you should not
   need any `unsafe` in this file at all — if you find yourself needing `unsafe` here,
   stop and report why, because the design (per this brief) shouldn't require it.
5. `cargo fmt --all --check` — must be clean.

Report `git diff --stat` for exactly what you touched (should be
`crates/pc-detect/src/onnx.rs` only), and confirm you did not edit
`crates/pc-detect/tests/perf3_denormal_order.rs` or any file under `crates/pc-ort/`.
