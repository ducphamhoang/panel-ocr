# Fable tie-break — denormal-flush fix joint plan (2026-08-16)

## Context

Both `architect` and `rust-engineer` were dispatched jointly against
`docs/BRIEF_denormal_fix_plan.md` to plan the fix for the batch-mode CPU regression
recorded in `docs/WORKSTATE.md`'s 2026-08-16 "Discovered a SEPARATE, more serious real
batch-mode CPU regression" entry. Their full independent positions are at:

- `docs/BRIEF_denormal_architect_position.md`
- `docs/BRIEF_denormal_rust_engineer_position.md`

**Read both in full before ruling.** They converge on the core mechanism (ONNX Runtime's
`session.set_denormal_as_zero` reaches a session's own intra-op pool threads reliably,
per-session, but reaches the thread that *calls* `Session::run` only through a
process-wide `std::call_once` that the first session to initialize in the process
consumes) and on the general fix shape (a new RAII MXCSR guard, owned by `pc-ort`,
scoped around the detector's `Session::run` call). Per this project's pipeline
(`CLAUDE.md`), where the two Opus planners disagree the Orchestrator does not pick a
side — that call is yours, and only yours, for this decision.

## The three points of actual disagreement, extracted so you don't have to re-diff the whole documents

### 1. Is the production root-cause deterministic or a scheduling race?

- **architect**: reads `crates/pc-cli/src/lib.rs:96-108` and traces that OCR's session
  construction is **eager** (`ocr::build_factory_for_device(...)` runs before pipeline
  execution) while the detector's provider is **lazy** (`OnceLock`, only built on first
  use). Concludes the production ordering is a deterministic consequence of that
  eager/lazy split — OCR always wins the once-flag in a real `run_clean` invocation, not
  "sometimes."
- **rust-engineer**: states "All three are lazily constructed behind independent
  `OnceLock`s (`crates/pc-cli/src/detector.rs:60`, `crates/pc-cli/src/inpainter.rs:110`)
  under a rayon pool. Which one initializes first is a scheduling outcome, not a program
  property." This treats the ordering as racy/nondeterministic, not as a deterministic
  consequence of an eager-vs-lazy split.
- **Why this matters for the fix, not just for accuracy**: if the ordering is
  deterministic (architect's reading), a much cheaper alternative fix exists in
  principle — force detector-first construction — though architect itself doesn't
  propose relying on that (both agree the fix should be order-independent regardless).
  If it's genuinely racy (rust-engineer's reading), that's a stronger, more urgent
  argument for why the CPU-state-owned-by-us design is the *only* correct answer, not
  merely the safer one. This is a factual question about this codebase's actual
  construction order under real CLI usage (`ocr_enabled` default, `pc-cli/src/detector.rs`
  vs `pc-cli/src/ocr.rs` call order in `run_clean`/`run_pipeline`) — it may be resolvable
  by you re-reading the same call sites, not merely a stylistic difference to average
  over.

### 2. Should the flush-state observability method ship gated or ungated?

- **architect** (§2.3): wants `OnnxDetector::last_inference_flush_state()` **ungated**,
  shipping in the production binary unconditionally. Argument: this is a read-only
  observation of the shipped path (not a behavior-injection hook like
  `panic_next_infer_for_test`), and gating it behind `#[cfg(test)]`/a feature means the
  production binary's actual flush state is verified by nothing — the exact
  copy-vs-artifact inversion cookbook rule 12 warns against. Cost: one relaxed atomic
  store per ~0.5s inference.
- **rust-engineer** (§6, T2 draft): designs the equivalent method
  (`worker_flushes_denormals_during_inference()`) **gated**
  `#[cfg(any(test, feature = "testkit"))]` — consistent with the existing
  `panic_next_infer_for_test`-style injection-hook convention in this codebase.
- Architect explicitly flagged this exact disagreement as expected and said "if we
  disagree, this goes to Fable rather than to a compromise" — so this one is squarely
  yours to call, not to split.

### 3. Should `pc-ocr`/`pc-inpaint`'s session-construction sites get defensive
   "fencing" now, before any decision to flush there?

- **architect** (§2.4): proposes wrapping `commit_from_file` in both `pc-ocr` and
  `pc-inpaint` with `MxcsrScope::fencing()` **now**, in this same change (task D2,
  classified "simple," batchable with D1) — on the grounds that fencing changes no
  float semantics (it only restores whatever state was already present) and closes
  §16.32's leak hazard defensively, ahead of any measurement about whether OCR/inpaint
  should ever flush.
- **rust-engineer** (§5): takes the position that nothing about `pc-ocr`/`pc-inpaint`'s
  construction sites should change in this task at all — "I will not draft tests for a
  change whose need is unestablished" — deferring *any* touch to those sites, including
  defensive fencing, to its own measurement-first task (T3), which only guards
  construction "if either adopts flushing."
- Note this is a narrower disagreement than "should OCR/inpaint flush" — both agree
  that decision needs its own measurement task and is out of scope now. The
  disagreement is specifically whether the *defensive, zero-semantics-risk* fencing
  wrapper is worth adding preemptively in this change or should wait.

## Convergent findings, for your context (not in dispute, no need to re-litigate)

- Both independently reject the brief's original leading theory for the residual ~8x
  slowdown (ORT's internal thread pool) — architect via reading ORT's pinned source
  (pool threads are covered per-session, not once-gated), rust-engineer via measurement
  (a configuration with pool threads deliberately unflushed reproduces the residual
  growth shape, but that isn't the brief's actual residual configuration). Both leave
  the residual formally unattributed and defer to a follow-up measurement task
  (architect's D4, rust-engineer's open item in §8) rather than guessing.
- Both agree §16.21 item 6 and §16.32 were honestly measured but incomplete — correct
  in a detector-only benchmark process where the detector necessarily initializes
  first, not established (and now shown false) in the real `run_clean` process shape.
  Both want qualification, not reversal, with the source's scope quoted verbatim
  alongside the correction, per `CLAUDE.md`'s "a claim's scope travels with it" rule.
- Both agree no *existing* frozen test breaks under the proposed fix, and both agree
  the bit-exactness fixtures must be re-run against real weights before the fix is
  trusted (not merely assumed unchanged from the synthetic-input probe evidence).
- Both propose an RAII guard in `pc-ort` as the mechanism, scoped (not
  thread-lifetime-permanent), and both agree `with_flush_to_zero()` must be kept
  because it's the only way ORT's pool threads get the bit.
- Both flag that scope-confinement (the new RAII mechanism) modifying §16.32's ratified
  "flag + thread-confinement" bundle needs its own explicit §16.x ratification entry
  quoting §16.32's scope verbatim — neither treats this as self-executing.
- Neither treats the "first `unsafe` in this workspace" question as theirs to decide
  unilaterally — architect proposes containing it via a workspace `unsafe_code = "deny"`
  lint plus one `#[allow(unsafe_code)]` on the new module; rust-engineer's drafted code
  needs the same primitive (MXCSR read/write via inline asm) but doesn't independently
  propose the lint. Not a contradiction — silence, not disagreement. You may rule on
  whether the lint-containment approach is adopted if you think it's in scope for this
  tie-break, or leave it to ratification.

## What's asked of you

1. Rule on disagreements 1-3 above, explicitly, the way this project's Fable
   tie-break has ruled before (§16.51's precedent: state which side's conclusion wins,
   and whether the losing side's underlying premise was still sound even where its
   conclusion lost — see WORKSTATE.md's account of the OCR-batching tie-break for the
   shape expected).
2. State whether there's a fourth option, or synthesis, either planner missed — as
   happened in the §16.51 tie-break, where you found the decisive piece neither plan
   had reached on its own.
3. You are **not** being asked to write any code, and not being asked to resolve the
   ambiguities both planners already agreed are open (residual ~8x cause, whether
   OCR/inpaint should ever flush) — those are follow-up measurement tasks, not tie-break
   material.
