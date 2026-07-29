# Working Pipeline

This project follows a fixed multi-agent pipeline. Follow it for every feature/spec
unless the user explicitly overrides it for a given task.

## Roles

- **Orchestrator (Sonnet, me)**: drives the whole pipeline, never writes plan or
  implementation code directly. Delegates, tracks state, escalates.
- **Technical Architecture (Opus subagent)**: designs the high-level approach for a
  spec — module boundaries, interfaces, data flow, sequencing of tasks.
- **Senior Rust Engineer (Opus subagent)**: co-authors the plan with the architect,
  and is responsible for drafting the actual *test code* (not just descriptions) for
  each planned task, mapped explicitly back to the spec requirement it verifies.
  Also performs the final post-implementation review.
- **Codex (via the `codex:rescue` skill / Codex CLI)**: implements tasks, task by
  task, strictly test-driven. Does not invent scope beyond the plan.
- **Fable (Senior Rust Engineer)**: normally advisory-only — consulted when a
  task/bug resists a fix after more than 5 iterations, giving advice only, never
  writing or touching code. Exception: Fable also acts as the tie-breaker when the
  two Opus subagents disagree on the plan (see below) — in that specific case Fable
  makes the final call, still without writing code.

## Pipeline

1. **Plan**: Orchestrator spawns the Opus Technical Architecture and Opus Senior
   Rust Engineer subagents together to produce a plan for the spec. The plan must
   include, per task:
   - Scope and interfaces touched
   - The actual test code to write first (drafted by the Rust Engineer), tied
     explicitly back to the spec requirement it verifies
   - Whether the task is "simple" (safe to batch sequentially with related tasks in
     one Codex call) or "heavy" (needs its own isolated call)
   - If the two Opus subagents disagree on approach, the Orchestrator does not
     pick one itself — it spawns a Fable Senior Rust Engineer subagent to review
     both positions and make the final call (advisory role suspended for this one
     decision; Fable still does not write code).

2. **TDD implementation loop**, per task (or batch of related simple tasks):
   - Before Codex writes any implementation code, the Orchestrator checks the
     planned tests against the spec for relevance/correctness.
   - Codex implements against the tests. **Tests are not to be changed** once
     written — only the implementation is iterated until tests pass.
   - Exception: if a test is later found to contradict the spec, that goes back to
     the two Opus architects jointly to decide — never a unilateral test edit by
     Codex or the Orchestrator.
   - Batching rule: related + simple tasks may be handed to Codex sequentially in a
     single call. Heavy/complex tasks get their own separate call.
   - Escalation rule: if a task/bug isn't resolved within ~5 iterations, stop
     iterating blindly and consult Fable for advice (diagnosis/approach only, no
     code from Fable). Apply Fable's advice via Codex as usual.

3. **Review**: once Codex reports a task/batch done and all tests are green, the
   Orchestrator spawns the Opus Senior Rust Engineer subagent again to review the
   implementation against the original spec (not just "tests pass").

4. **Progress watch**: the Orchestrator checks in on any long-running subagent
   roughly every 5 minutes (via scheduled wake-ups, not busy polling) to confirm
   it's still making progress, separate from whatever task-completion notifications
   already fire. To catch a dead/stuck agent early rather than waiting out the full
   interval:
   - Treat any background-task or agent completion event carrying a failure/error
     status (crash, non-zero exit, tool error) as an immediate check, not something
     to defer to the next scheduled wake-up.
   - On each scheduled check-in, verify there was actual observable progress since
     the last one (new output, state change, partial result) — silence or an
     unchanged state for a full interval is treated as a possible hang, not
     assumed to be normal slow work, and gets investigated right away (e.g. check
     task status/output directly) instead of waiting another 5 minutes.
   - If an agent is confirmed dead or hung, restart/resume it rather than silently
     waiting further.

## Notes

- Specs are the source of truth; the plan and tests must trace back to them.
- **Read [`docs/COOKBOOK.md`](docs/COOKBOOK.md) before an audit, before ratifying a
  deviation, and before trusting a green test suite.** It records this project's recurring
  process failures and the decisions that resolved them — the dominant defect class (a test
  whose name claims more than its assertion verifies), how to classify per-image vs
  run-fatal failures from a function signature, why upstream PanelCleaner is the tiebreak
  oracle, the three legitimate exits from a frozen test, and a running list of over-claims
  to check yourself against. Add to it whenever a pattern recurs or a costly mistake is
  resolved; it is process memory, not a changelog.
- When the spec is ambiguous or two readings conflict, consult the upstream implementation
  this project ports — https://github.com/VoxelCubes/PanelCleaner — by running it, not only
  by reading it. See cookbook rule 3 for the no-sudo install recipe.
- **Committing is pre-authorized once work is ready** — no need to ask first. "Ready"
  means all of the following have been verified by actually running them, not assumed:
  `cargo test --workspace` green, the `onnx` feature tier green
  (`cargo test --workspace --all-targets --features pc-cli/onnx`),
  `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean,
  `cargo fmt --all --check` clean, and any open review-gate finding either fixed or
  explicitly recorded as a ratified decision (§16.x) rather than left silently open.
  Commit along task/spec boundaries so each commit is reviewable on its own; state in
  the message what was verified.
- **Pushing still requires being asked**, per standing repo conventions.
- One exception to pre-authorized committing: an `insta` snapshot may not be committed
  until §15.10(a)'s hand-traced review is recorded in `docs/GOLDEN_CALIBRATION.md` with
  reviewer/date/method. That review needs a human reviewer independent of whoever
  produced the snapshot — the same self-reference rule as §16.13 item 4. Derive and
  present the expected values; do not self-attest them.
