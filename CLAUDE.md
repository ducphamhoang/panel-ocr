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

1a. **Ratification transcription is reviewed before it is committed.** When a ruling
   (joint-architect or Fable) is transcribed into `docs/PIPELINE_SPEC_V1.md` as a
   §16.x entry, the *transcription* gets a reader before the commit — not only the
   ruling it records.

   **Why this step exists, stated so it is not dropped as ceremony.** The pipeline
   has an adjudicator for *disagreement* and had **no adversary for consensus**.
   Fable is convened only when the two architects disagree and is otherwise
   advisory-only, so it sees disputes and never sees ordinary work. Every defect
   that reached the repo in the F1 sequence was the opposite of a dispute — a place
   everyone agreed because nobody checked: §16.24 item 1(a)'s single-consumer
   framing (missed by architect, engineer, Fable *and* Orchestrator), item
   18(h)(ii)'s "residuals must be zero" (written by the Orchestrator, re-read by
   nobody before commit), item 20(b)'s "NO count fields" (contradicting a ratified
   clause), item 5's derivation applied past its scope, and three supersession
   markers asserted in a new entry while the old sites stayed unqualified.
   **Rulings got two architects; the transcription of them got none.**

   Two binding consequences:

   - **A fresh reader is called for a ratification, not only for a dispute.** Cost
     is reading one section. This is closer to Fable's proper role than waiting for
     disagreement, and it is the only step aimed at consensus rather than conflict.
   - **When transcribing a narrow conclusion, quote the source's scope verbatim
     beside it.** All three over-generalisations happened while paraphrasing rather
     than quoting: a conclusion that was correct about one field, one consumer or
     one residual form was restated in wider terms than its evidence allowed. If
     the source says "confidence", the transcription says "confidence" and not
     "the field"; widening is a separate, argued step.

   The mechanical half of this is a test rather than a habit — see the supersession
   cross-check gate under Notes.

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
   - **A status field is not progress.** Two Codex jobs in the F1 sequence returned
     `completed` having written nothing, and one `resume` failed at 0s from a
     collision with another job. What distinguished real work from a no-op was file
     mtime plus a build-error or test count — so check an artifact that changes, not
     a field that claims. A fresh task is also more reliable than a resume: resumes
     are what collided.

## Notes

- Specs are the source of truth; the plan and tests must trace back to them.
- **The supersession cross-check is a TEST, not a habit.** For every supersession claim
  in `docs/PIPELINE_SPEC_V1.md` — `SUPERSEDED`, `amended`, `ERRATUM`, `QUALIFIED`,
  `withdrawn`, `NARROWED`, `RE-GROUNDED` — the site it names must carry a pointer back
  to the claiming section. §16.19's convention already required the marker *at* the
  superseded text; what it lacked was enforcement, so the marker was repeatedly written
  only at the claiming end, which is the end a future reader does **not** land on.
  This is cookbook rule 14 applied to the spec rather than to code, and it is the one
  safeguard here that does not depend on anyone remembering it. Deleting a marker must
  turn a test red and name which claim and which target.
- **A claim's scope travels with it.** When transcribing a ruling, quote the source's
  scope verbatim beside the conclusion. Every over-generalisation in the F1 sequence
  happened while paraphrasing: correct about one field, one consumer or one residual
  form, restated in wider terms than the evidence allowed. Widening is a separate step
  and needs its own argument.
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
