# Working Pipeline

## Durable orchestrator work state — read and update this first

`docs/WORKSTATE.md` is the canonical live continuation index for the multi-worktree
orchestrator. At the start of every session, read it before choosing a task. Then verify
its dated observations against the actual worktrees (`git status --short --branch`,
`git rev-parse HEAD`, changed-file mtimes/diffs), the ratified sections of
`docs/PIPELINE_SPEC_V1.md`, and the frozen tests. A task status or agent self-report is
not evidence of progress.

After every meaningful transition — task start, test freeze, implementation commit,
review result, verification result, blocker, or merge — update `docs/WORKSTATE.md` with
the observed HEAD and actual command results. Re-read it before updating so a newer
state is not silently overwritten. If a future session needs one-off continuation
scaffolding (e.g. `docs/HANDOVER.md`), mark it "TEMPORARY, delete when consumed" and
actually delete it once consumed — do not let it accumulate as a second, silently
stale source of truth alongside this file.

**Unified-spec rule for feature worktrees:** `docs/PIPELINE_SPEC_V1.md` is maintained as
one canonical document on the main/integration line. Do not add ratification sections,
supersession claims, or other normative-spec edits only to an independent feature
worktree. Feature worktrees may carry implementation and test changes plus
`docs/WORKSTATE.md` evidence, but spec transcription and reconciliation happen on the
clean integration branch after the ordered merges. When a feature branch has a
ratified documentation decision, record the decision and blocker in `WORKSTATE.md` and
carry the transcription to the unified integration spec.

(Historical: the mask-parity/lama-inpaint integration sequence this paragraph used to
describe — A4-d, LaMa L6-4/L6-5, the clean integration branch, the ordered merge, spec
reconciliation, and the Simple/Annotation/LaMa benchmark — completed 2026-08-10 and is
now on `main`. See `docs/WORKSTATE.md`'s Mission section for the closure record. The
general rule stands for any future multi-worktree sequence: trace the ordered queue
`docs/WORKSTATE.md` actually declares for that work, and do not reorder or merge early
unless the user explicitly changes the decision.)

## Model routing

Pipeline roles use native Claude model tiers, pinned in each agent's frontmatter under
`.claude/agents/`: `model: opus` for the architect/rust-engineer/fresh-reader roles,
`model: fable` for `fable-adjudicator`. Pass the tier alias through the Agent tool's
`model` param the same way (`opus`, `sonnet`, `haiku`, `fable`); nothing else is a valid
value here — `crates/pc-testkit/tests/agent_definitions.rs` gates the allowed set.

**Retired: the `codex-vbg/gpt-5.6-sol` / `codex-vbg/gpt-5.6-luna` proxy routing.** For a
stretch this project addressed a local Anthropic-compatible proxy (`codex-vbg/<model>`,
see `setup-ccx-proxy.ps1`, listening on `127.0.0.1:8080`) directly by model string
instead of using the tier aliases above, with Sol standing in for the former
Opus/Fable tiers and Luna for the former Sonnet tier. That routing depended on the
proxy process being up; when it wasn't (confirmed 2026-08-09: nothing listening on
`127.0.0.1:8080`, every agent spawn failing with "model may not exist or you may not
have access"), the entire pipeline stalled with no visible task-level cause. Reverted
back to the native tiers above for reliability. If a future session wants to re-route
through a local proxy, verify the proxy is actually listening before repointing agent
frontmatter at it, and don't assume a `model:` string that isn't one of `opus`/`sonnet`/
`haiku`/`fable` will resolve — check the Agent tool's own accepted values first.

This project follows a fixed multi-agent pipeline. Follow it for every feature/spec
unless the user explicitly overrides it for a given task.

**Mechanical verification does not need the pinned tier.** Re-deriving a byte count,
checking a citation/line number against source, counting occurrences, confirming a
test's real output matches a claimed number — none of this is judgment work. Route it
to a fast/cheap tier (`haiku`) instead of spinning up an Opus-tier `architect`/
`rust-engineer` call for it. Reserve the pinned tiers for the things that actually need
that reasoning: design, disagreement, and the specific correctness judgment a
step exists to provide. (Cost lesson from the 2026-08-10 defaults-flip task: verifying
a Fable-quoted byte count or a spec section number does not need the same tier that
adjudicated the underlying disagreement.)

## Right-sizing the pipeline

**Not every task needs everything below.** Classify before invoking anything, and
default to the lightest tier that fits — escalate only when a step's own output
surfaces a real disagreement or spec conflict, never pre-emptively "to be safe":

- **Direct.** No ratified deviation being reversed, no frozen-fixture-producing path
  touched, no cross-cutting effect, the approach isn't contestable (doc fixes, log/CLI
  wording, a config default with nothing ratified attached, a one-file bugfix). The
  Orchestrator implements directly, runs the relevant tests, reports. No architect, no
  `rust-engineer`, no Fable, no ratification.
- **Simple, one review.** Touches real code but the approach isn't contested and the
  blast radius is contained. One `rust-engineer` implementation pass, one independent
  review pass. No dual plan, no Fable, no multi-round review — if the first review
  finds something, fix it and re-run tests; only spin a second review round if that fix
  itself is contestable, not to re-confirm a mechanical correction.
- **Spec-sensitive.** Reverses or creates a ratified deviation (`DEVIATION(n)`), touches
  fixture/provenance-producing machinery, or two reasonable people could actually
  disagree on the approach. This is the only tier where the full pipeline below (joint
  plan, Fable on disagreement, ratification + fresh-reader, task-by-task review) is
  worth its cost — and even here, prefer one thorough, explicitly-adversarial review
  pass over open-ended review→fix→review loops (see step 3).

If genuinely unsure which tier a task needs, say so to the user and ask, rather than
defaulting to the heaviest tier "to be safe" — that default is itself the expensive
mistake (see the 2026-08-10 defaults-flip task, ~5.5M subagent tokens for what turned
out to be two contested points and a handful of mechanical bugs a single careful pass
would have caught).

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

## Agent dispatch briefs

**Every subagent dispatch — the four named roles above, `codex:codex-rescue`, `cmdc`,
or any other external agent call — gets its full task context written to a markdown
brief file first, and the dispatch prompt tells the agent to read that file rather
than inlining the context in the prompt itself.** (Decided 2026-08-09, prompted by the
`codex-rescue` model-routing incident: iterating on a subagent's instructions mid-task
meant re-typing the whole brief into a fresh prompt each retry, with no single place
recording what the *current* instructions actually are.)

Why a file instead of a prompt string:

- **The brief can be revised in place and the agent told to re-read it**, instead of
  re-explaining from scratch on every retry. If an agent misunderstands a step, comes
  back with a wrong assumption, or needs a correction mid-task, edit the brief file and
  send a follow-up telling it to re-read the file — this works whether the agent is
  paused/resumed (Agent tool) or a fresh dispatch (`codex-rescue`/`cmdc`, which don't
  preserve conversation state the same way).
- **One artifact, not scattered prompt text**, to point a reviewer at when checking
  whether an implementation actually followed its brief — "did it do what the brief
  said" is a diffable question when the brief is a file.
- Applies uniformly across dispatch mechanisms that don't share a common prompt
  interface: an `Agent` tool subagent gets `Read <path>` naturally (all four roles
  already carry the `Read` tool); `codex:codex-rescue` is a thin forwarder, so the
  forwarded task text itself becomes "read the brief at `<path>` and follow it exactly"
  — the underlying Codex CLI task reads the file with its own tools; `cmdc -p` works
  the same way.

Placement: write the brief under the **target worktree's own path** when the task is
scoped to one worktree (so a worktree-relative mention inside the brief resolves
correctly and the brief travels if the worktree is inspected later), otherwise under
the session scratchpad. Either way, pass the **absolute path** in the dispatch prompt.
Keep the dispatch prompt itself short: point at the brief, state anything that changed
since the brief was written (if this is a retry), and nothing else duplicated from the
brief's own content — duplicating defeats the point of having a single revisable place.

This does not relax any other rule in this file — a brief-file dispatch to
`rust-engineer` still needs an independent reviewer per §16.13 item 4, a ratification
transcription still needs `fresh-reader`, and an implementer's own self-report is still
not verification (re-run the actual checks).

### Standing cmdc dispatch preamble

**Every `cmdc` dispatch prompt includes this preamble** (verbatim or paraphrased, but
covering all five points), instead of re-typing it into each brief. Codified 2026-08-11
from a real, recurring pattern across GPU-2's and GPU-3's implementation calls: cmdc's
self-reports were consistently honest about what they *did* check, but twice under-
scoped the verification bar (G2-A/B and G2-C both reported "clippy clean" having checked
only the touched crates, and skipped `cargo fmt --all` entirely) — caught only because
the Orchestrator re-ran the full bar independently afterward. The preamble moves that
catch earlier, at the source, rather than relying on a second pass to find it every time.

1. **Exactly one party runs the full four-command verification bar per task — never
   both.** For Spec-sensitive tier, that's still you (the implementer): `cargo test
   --workspace`, `cargo test --workspace --all-targets --features pc-cli/onnx`, `cargo
   clippy --workspace --all-targets --all-features -- -D warnings`, `cargo fmt --all
   --check` (+ the `cuda`-tier command if the brief names one). For Direct/Simple tier,
   the Orchestrator owns the one full-bar run at the gate instead — run your scoped
   tests plus `cargo fmt --all --check` during iteration, and say explicitly in your
   report "scoped only, full bar deferred to the Orchestrator" rather than claiming the
   full bar. Scoped tests during iteration are fine either way; what changes is who runs
   the full bar once, not whether it runs. (Decided 2026-08-15, Fable-adjudicated: the
   G2-A/B and G2-C incidents behind this preamble were a *misreporting* problem, already
   fixed by points 2–6 below — running the full bar twice on a Simple task was pure
   duplication, not an added safeguard. Never applies to Spec-sensitive work, where the
   duplication is the point.)
2. **Disclose every deviation from the brief explicitly**, however small (a manifest
   edit the brief didn't spell out, a clippy-driven rewording of a drafted test) — never
   silently absorb one, even when the fix is obviously correct.
3. **For every frozen/untouchable file the brief names, confirm byte-for-byte identity
   via `git diff <path>` and report that it returned empty** — not just "I didn't edit
   it." An empty diff is checkable; a claim isn't.
4. **Respect scope fences literally.** If the brief says a file/module is out of scope
   for this call, do not touch it even if a fix looks trivially applicable there — name
   the gap in your report instead.
5. **Quote the actual red output you observed before the green**, the same way you quote
   the green output — a real failing-test transcript is harder to fake than a claim of
   "observed red."
6. **Report `git diff --stat` output directly** as part of "files touched," not a
   prose recollection of what you edited.

**This does not relax the Orchestrator's own re-verification duty for Spec-sensitive
work**, where both runs still happen: a more complete cmdc self-report there is not a
substitute for independently re-running the bar and re-checking the diff before
committing — it only reduces how often that independent check finds something the
report missed. For Direct/Simple tier, per point 1 above, the Orchestrator's single
gate-time run **is** the re-verification duty, not an addition to a second implementer
run — do not also demand the implementer run the full bar there.

**Keep briefs surgical, not exhaustive.** A brief should state what changed and what's
needed for *this* step — pointing at an existing artifact (a prior brief, a spec
section, a committed file) is cheaper and just as effective as re-deriving or
re-including context that step already established. Every extra paragraph is input
tokens on every subsequent read of that brief, including every resume.

**Cap resume chains.** Resuming the same agent (`SendMessage` to an existing agent)
re-sends its *entire* accumulated transcript every time — a long fix→review→fix loop on
one instance compounds cost turn over turn, not linearly. Resuming is fine for a short,
continuous chain (implement → fix one review's findings). Once a task has gone through
more than ~2-3 resume rounds, prefer a fresh spawn with a distilled brief (state the
current file state and what's left, not the history of how it got there) over continuing
to grow the same instance's context indefinitely.

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
   - **Targeted runs during iteration, full bar only at the gate.** While actively
     iterating on one crate/test, run the scoped command (`cargo test -p <crate>
     --test <name>`), not `cargo test --workspace`. The full workspace run, the `onnx`
     feature tier, clippy, and fmt are the *final* verification gate for a task/batch —
     run them once when the task is believed done, not after every intermediate edit.
     A full-workspace run's output is large; reading it into context repeatedly for
     work that hasn't converged yet is pure waste.

3. **Review**: once Codex reports a task/batch done and all tests are green, the
   Orchestrator spawns the Opus Senior Rust Engineer subagent again to review the
   implementation against the original spec (not just "tests pass"). **Ask for one
   thorough, explicitly-adversarial pass, not an open-ended loop.** Tell the reviewer
   this is likely its only pass — falsify claims, re-derive numbers, check the whole
   diff, not just the parts flagged as risky. Batch every finding from that one pass
   into a single fix round. Only spin a second review round if a fix itself introduces
   something contestable (a design call, a spec-text change) — never merely to
   re-confirm a mechanical correction (a typo, a wrong citation, a stale comment) that
   the fix report already demonstrates was applied and re-verified.

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

### Work-state maintenance

- `docs/WORKSTATE.md` is the canonical live continuation index; update it after each meaningful transition and record observed HEADs plus actual command results.
- The work-state file is a control document, not the normative source: reconcile it against git, frozen tests, and the ratified spec before acting.
- Do not silently merge stale `HANDOVER.md` state into the current plan; preserve stale-state corrections in the work-state update log.
- **Carve-out (2026-08-15):** a Direct-tier task that starts and finishes within one
  session, ending in a clean commit with no open blocker, may skip the per-transition
  updates and record a single entry at completion instead — the commit itself is the
  observed-state record for that kind of task, and the next session reconciles
  WORKSTATE against git regardless. Anything spanning sessions, worktrees, or leaving a
  blocker open still gets the full per-transition discipline above.

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
  - **Carve-out for diffs touching only `.md` files** (no `.rs`, no manifest, no
    `tests/fixtures/`): run `cargo test --workspace` only — this repo's prose is a real
    test input (`crates/pc-testkit/tests/agent_definitions.rs`, `spec_supersession.rs`,
    `a4d_claim_sites.rs` and others parse `CLAUDE.md`/`docs/PIPELINE_SPEC_V1.md` at
    runtime, and cookbook rule 6a records a paragraph rewrap turning one of these gates
    red) — and skip the onnx tier, clippy, and fmt, since those three are functions of
    Rust source and manifests that a pure-doc diff cannot move. Do not substitute a
    crate-scoped run (e.g. `-p pc-testkit`) for this — measured slower than the full
    workspace run on this repo, with no offsetting safety benefit. Any diff that touches
    even one non-`.md` file gets the full bar, no partial credit.
  - **Measured cost, so "skip it to save time" isn't the right frame for code diffs**:
    the full four-command bar runs in ~90s wall-clock on a warm `target/` (test ~27s,
    onnx ~35s, clippy ~23s, fmt ~6s); the multi-minute numbers people remember are a
    cold-cache/fresh-worktree cost paid once, not a per-task cost. Keeping a warm
    `target/` per worktree is the actual lever, not narrowing which commands run.
  - **Who runs it**: for Spec-sensitive tier, both the implementer and the Orchestrator
    still run the full bar independently (see the standing cmdc preamble). For
    Direct/Simple tier, exactly one run is required — the Orchestrator's, at the gate —
    per the cmdc preamble's point 1; do not also demand a duplicate implementer run
    there.
- **Pushing still requires being asked**, per standing repo conventions.
- **The `insta` snapshot exception below this line was retired 2026-08-15 (see the
  entry immediately after) — kept struck through, not deleted, per this file's own
  supersession convention** (never silently delete a superseded rule; the next reader
  needs to know it changed and why): ~~an `insta` snapshot may not be committed until
  §15.10(a)'s hand-traced review is recorded in `docs/GOLDEN_CALIBRATION.md` with
  reviewer/date/method, by a reviewer independent of whoever produced the snapshot.~~
  **SUPERSEDED 2026-08-15**: snapshots were dropped project-wide by §16.20 (Fable
  tie-break) — `insta` is in no crate manifest and `assert_json_snapshot!` in no test,
  confirmed by direct search, not assumed. Per cookbook rule 12's own ruling, the
  §15.10(a) gate was aimed at a copy (the snapshot transcription), not the artifact that
  carries the risk (recorded model/detector output) — the obligation already moved
  there. **The live version of this rule is: a committed fixture under
  `tests/fixtures/recorded/` may not be added or changed without an independent human
  review of the derivation**, same independence requirement as §16.13 item 4, tracked
  via the recorded-fixture digest gate (§16.13 item 6) and `docs/DETECTOR_ORACLE.md`'s
  provenance, not via `docs/GOLDEN_CALIBRATION.md`. If `insta` or per-field snapshotting
  is ever reintroduced, restore the original hand-traced-review requirement rather than
  assuming this carve-out still applies to it.
