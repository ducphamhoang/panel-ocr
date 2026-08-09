# Working Pipeline

This project follows a fixed multi-agent pipeline. Follow it for every feature/spec
unless the user explicitly overrides it for a given task.

## Roles

Four of these roles have definitions in `.claude/agents/`. The *intent* is that a role's
prohibitions live in frontmatter the harness reads, instead of being sentences in a prompt
that an agent can ignore.

**Status first, because it changes how to read everything below: the mechanism has now
been observed working, once, from a fresh session.** The first attempt — spawning
`rust-engineer` in the same session that created the files — failed (Caveat 1, first
half). A later attempt from a separate, fresh session — spawning `fresh-reader` on a
trivial read task — succeeded: the type resolved and the agent executed and returned
real output. Only `fresh-reader` was tested this way; `architect`, `rust-engineer` and
`fable-adjudicator` were not separately confirmed, though there is no reason to expect
the loader to treat them differently. This settles hot-reload-from-a-fresh-session; it
does **not** settle whether the harness honours `tools` or `model` (see the gate
enumeration below and Caveat 1's second half) — so keep treating every statement about
what a role "cannot" do as **what the definition asks for, not as something measured**,
and keep stating the prohibition in the brief as well.

What the gate in `crates/pc-testkit/tests/agent_definitions.rs` establishes, enumerated
rather than summarised because summarising it has already overstated it twice:

- the directory holds exactly the four expected names, and each `name` matches its filename;
- each frontmatter carries all four required keys, and `model` is an allowed value and the
  one pinned for that role;
- the three read-only definitions list **none** of `Edit`/`Write`/`NotebookEdit`, and
  `rust-engineer` lists `Edit` and `Write`;
- each frontmatter parses **under `yaml-rust2` 0.11.0 plus a printable-characters rule** —
  which is a *proxy*, not "valid YAML": that crate is measurably laxer than PyYAML (it accepts
  control bytes PyYAML rejects), and the harness's own loader is neither of them;
- the `Bash` caveat is still present in all three read-only definitions and in this file.

It establishes **nothing** about whether the harness reads these files, whether it honours
`tools` or `model`, or even whether `Read`/`Grep`/`Glob`/`Bash` are present — **no test
asserts that a tool is present**, apart from the two `rust-engineer` rows above.

Passing this gate means the files satisfy the checks listed above, and nothing beyond them.
Resist restating that as a property of the files — "well-formed", "valid", "correct" all
smuggle back the generality bullet four is careful to deny. A summary sentence placed under
an enumeration re-inflates the enumeration; that has happened here three times, twice inside
the very edits written to fix it.

This next paragraph states what the mechanism is *for*, not what it does — loading (the
file parses, the agent type resolves) is now observed for `fresh-reader`, but whether the
harness additionally *honours* `tools`/`model` once it reads them is a separate premise
that nothing here establishes either way. **If** both hold — the definitions load, and the
harness enforces `tools`/`model` from them — then `fresh-reader`, `architect` and
`fable-adjudicator` listing no `Edit`/`Write`/`NotebookEdit` would mean the harness
withholds those three tools regardless of what the agent decides; and `model` being part
of the definition would stop a review silently running on the wrong tier because a spawn
forgot to pass one. Caveat 2 covers what "read-only" would still not cover even then.

Both caveats are load-bearing. Neither is a footnote.

> **CAVEAT 1, MEASURED 2026-07-30 (same-session) and 2026-07-30 (fresh-session): the
> definitions are NOT hot-reloaded within a session, but a fresh session DOES pick them
> up.** Spawning `rust-engineer` in the same session that created the files failed with
> *"Agent type 'rust-engineer' not found"*, listing only the built-ins — a session that
> adds or edits a definition still cannot use it there, and must fall back to a generic
> subagent with the role preamble written out by hand. In a later, separate session,
> spawning `fresh-reader` on a trivial read task succeeded: the type resolved and the
> agent executed and returned real output. Only `fresh-reader` was tested this way —
> `architect`, `rust-engineer` and `fable-adjudicator` were not separately confirmed,
> though there is no reason to expect the loader to treat them differently. This does
> **not** establish that the harness honours `tools` or `model`: no test asserts a tool
> is present, and this probe neither attempted a write nor checked which model ran. So
> treat "the harness enforces the prohibition" as the intended design, not a confirmed
> behaviour, and keep stating the prohibition in the brief as well.
>
> **CAVEAT 2: "read-only" names three tools a definition withholds — it does NOT mean the
> agent cannot write.** All three read-only agents carry `Bash`, so a determined one can
> write via `>`, `sed -i`, or `git commit`. **Do not describe them as unable to edit;** the
> accurate statement is that their definitions withhold `Edit`/`Write`/`NotebookEdit` —
> which the harness would honour if it reads them at all, per Caveat 1 — while shell writes
> are prevented by instruction alone, in every case.
>
> Note the asymmetry, because it decides how much Caveat 1 costs you: the tool restriction
> depends on two things — the definitions loading (now observed, for `fresh-reader`, in a
> fresh session) **and** the harness honouring `tools` from them (still unverified either
> way) — but the shell-write gap depends on neither. That half is instruction-only whether
> or not the definitions load or their fields are enforced.
>
> Keeping `Bash` was a deliberate decision (maintainer, 2026-07-30). Every high-value
> finding these reviewers produced came from *running* something — a constructed probe
> appended to a real spec line, a re-implementation validated against the 28 known rows
> before being trusted — and a reviewer who cannot run the suite cannot check whether a
> gate is capable of failing, which is the check that has mattered most here. So the
> enforcement raises the bar from casual to deliberate; it does not make violation
> impossible.
>
> Each definition states this, and `read_only_agents_disclose_the_bash_limitation` keeps
> the disclosure from being deleted while the tool stays. That test gates the
> *disclosure*, not the behaviour — nothing in this repo can gate the behaviour.

The definitions carry only the boilerplate that was retyped every time. **They do not
replace the per-invocation brief**, which is where a review's value actually comes
from: the findings worth having came from "re-derive this number", "check this quote
character-for-character against that clause", "is this blind spot reachable?" — none
of which a template can produce. Each definition says so, and instructs the agent to
report a brief that names nothing concrete rather than reviewing generically.

- **Orchestrator (me)**: drives the whole pipeline, never writes plan or
  implementation code directly. Delegates, tracks state, escalates.
- **Technical Architecture** → agent `architect` (read-only): designs the high-level
  approach for a spec — module boundaries, interfaces, data flow, sequencing of tasks.
- **Senior Rust Engineer** → agent `rust-engineer` (the only one with write access):
  co-authors the plan with the architect, and is responsible for drafting the actual
  *test code* (not just descriptions) for each planned task, mapped explicitly back to
  the spec requirement it verifies. Also performs the final post-implementation review.
  Because it can write, it must never review its own earlier output (§16.13 item 4).
- **Codex (via the `codex:rescue` skill / Codex CLI)**: implements tasks, task by
  task, strictly test-driven. Does not invent scope beyond the plan.
- **Fable** → agent `fable-adjudicator` (read-only, `model: fable`): normally
  advisory-only — consulted when a task/bug resists a fix after more than 5
  iterations, giving advice only, never writing or touching code. Exception: Fable
  also acts as the tie-breaker when the two Opus subagents disagree on the plan (see
  below) — in that specific case Fable makes the final call, still without writing
  code.
- **Step-1a reader** → agent `fresh-reader` (read-only): the reader a ratification's
  *transcription* gets before it is committed. **Always a new spawn — never resume or
  re-message a previous `fresh-reader`**, because resuming destroys the freshness the
  gate depends on, and never assign it to whoever produced the artifact.

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

   Three binding consequences:

   - **A fresh reader is called for a ratification, not only for a dispute.** Cost
     is reading one section. This is closer to Fable's proper role than waiting for
     disagreement, and it is the only step aimed at consensus rather than conflict.
   - **When transcribing a narrow conclusion, quote the source's scope verbatim
     beside it.** All three over-generalisations happened while paraphrasing rather
     than quoting: a conclusion that was correct about one field, one consumer or
     one residual form was restated in wider terms than its evidence allowed. If
     the source says "confidence", the transcription says "confidence" and not
     "the field"; widening is a separate, argued step.
   - **A provenance claim is part of the transcription and is checked like one.** No
     section may attribute itself to a ruling, session, or reviewer that did not
     occur. The header's provenance (who ruled, when, in what mode) is a factual
     claim the fresh reader verifies against what actually happened, not framing.
     An implementation agent that believes a ratification is needed stops and
     reports that to the Orchestrator; it never writes the section itself, and never
     invents or back-dates a session to satisfy a gate — whatever the merit of the
     section's content. Real instance: the original §16.41 (2026-08-09), whose
     six items were substantively accurate and whose claimed joint ruling never
     happened; accuracy of content does not launder fabricated provenance.

   The scope-quoting bullet's mechanical half is a test rather than a habit — see the
   supersession cross-check gate under Notes. The fresh-reader and provenance bullets
   above have no equivalent automated gate; they are checked by the fresh reader
   actually doing the check, not by a test that would fail if they didn't.

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
