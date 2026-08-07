# Handover — TEMPORARY, delete when consumed

Written 2026-08-07, replacing the previous version (2026-08-03, which covered v1.0's F1
detector-oracle landing and F2/CPU-EP/v1.5-sequencing open items — all of that is now
superseded by the mask-parity and lama-inpaint work described below). **This file is
scaffolding, not a record.** Anything in it worth keeping permanently belongs in
`PIPELINE_SPEC_V1.md`, `COOKBOOK.md`, `RULINGS.md`, or `ARCHITECTURE_DECISIONS.md` instead.

## Read this first: the standing blocker

**Codex cannot currently write to either of the two active feature worktrees.** This blocks
every remaining code-implementation task below. Diagnosis, so the next session doesn't have
to re-discover it:

- Confirmed via a completed `codex-companion.mjs` job that made **zero edits** and reported
  `"Blocked by the managed Windows sandbox before any edits could be made."` after trying
  every patch mechanism available to it (embedded `apply_patch`, the direct `apply_patch`
  wrapper, git's own patch applicator) — all failed with `Access is denied`.
- `icacls` on the main repo (`D:\Duc\panel-ocr`) shows two extra sandbox-session SIDs granted
  `Modify`, on top of the generic `CodexSandboxUsers` group. The two worktree directories
  (`D:\Duc\panel-ocr-mask-parity`, `D:\Duc\panel-ocr-lama-inpaint`) have **only** the generic
  `CodexSandboxUsers` entry — no session-specific grant — which is evidently insufficient.
  This means Codex's Windows sandbox was never onboarded/provisioned for these two paths.
- `--cwd <worktree>` on the `task` subcommand **does** correctly scope `workspaceRoot` (fixed
  a *different*, earlier bug where jobs defaulted to the main repo root and silently treated
  worktrees as read-only) — but it does not fix the ACL gap itself.
- `codex-windows-sandbox-setup.exe` (in `~/.codex/.sandbox-bin/`) exists and is presumably
  the tool that provisions this, but it takes an internal encoded payload, not plain CLI
  flags — not safely hand-invokable, and this session deliberately did not attempt it (a
  system-level sandbox/ACL change needs human authorization, not agent guesswork).

**What would unblock it** (untried this session, in rough order of likely-least-effort):
1. Run `codex` interactively once inside each worktree directory — this often triggers a
   one-time per-path sandbox onboarding.
2. Check whether `/codex:setup` (or an equivalent onboarding command) accepts a target
   directory and can be pointed at the two worktrees.
3. Manually grant the two worktree directories the same session-SID ACL entries the main
   repo has (needs to know which SID(s) to add — `icacls D:\Duc\panel-ocr` shows the pattern).

**Once unblocked**, dispatch Codex with the corrected pattern (bypass the `codex-rescue`
subagent for this, since its fixed forwarding logic doesn't know about `--cwd`):

```
node "<codex plugin path>/scripts/codex-companion.mjs" task --background --write --fresh \
  --cwd "<worktree-path>" --prompt-file "<path-to-prompt-file>"
```
Check status/result with the **same** `--cwd`:
```
node ".../codex-companion.mjs" status <job-id> --cwd "<worktree-path>"
```
Prompt text is easier to manage as a file (`--prompt-file`) than as a shell-escaped
positional argument, especially since these prompts embed full Rust source for frozen test
files.

## Repo / worktree state

Three worktrees, same repo, independent working directories:

- **Main** — `D:\Duc\panel-ocr`, branch `claude/codex-plugin-install-jxirxa`, HEAD `78fecb7`,
  clean. Not touched this session beyond this handover file and an earlier backlog note.
- **`mask-parity`** — `D:\Duc\panel-ocr-mask-parity`, HEAD `aa6f3c7`, clean.
  Commits: `1b4e12f` (R0) → `e3a9d70` (A1) → `70a617c` (A2) → `dacd666` (A3) →
  `0a41904` (A3b) → `aa6f3c7` (A4 ratification, §16.39).
- **`lama-inpaint`** — `D:\Duc\panel-ocr-lama-inpaint`, HEAD `66cf7e4`, clean.
  Commits: `ee7cb60` (L-R) → `41b90b3` (L1) → `534734e` (L2+L3) → `f517aa8` (L4) →
  `ead8bd1` (L5) → `66cf7e4` (L6 ratification, §16.38 items 22-25).

Verification bar last actually run (not quoted from an older doc): `cargo test --workspace`
green on both branches at their respective ratification commits (mask-parity: 1194 passed /
0 failed / 3 ignored, re-verified after A4's ratification landed; lama-inpaint: 1218 passed /
0 failed / 3 ignored, re-verified after L6's ratification landed). **Re-run yourself before
trusting these numbers** — cookbook rule 6 — especially since neither branch has had the
`onnx` tier / clippy / fmt re-run since the *ratification* commits specifically (they were
last confirmed clean earlier in the review cycle, on the pre-ratification code).

`export PATH="/c/Users/ducph/.cargo/bin:$PATH"` if `cargo` is not found in a fresh Bash
session (same issue as before, cookbook rule 11).

## What's committed: mask-parity (Annotation mask-refine port)

Full A1→A3b algorithm port is implemented and unit-tested (85 tests). `aa6f3c7` ratifies
the two remaining design decisions needed to wire it in:
- **Coverage operand under Annotation mode**: score against the *unrefined* detector mask
  (upstream's own operand), not the refined one. Simple mode is unchanged. Both the
  architect and rust-engineer planning passes independently reached this by reading/running
  upstream at the pinned commit — no Fable escalation needed, genuine convergence.
- **`DEVIATION(12)`**: narrowed, not retired. Upstream has no mode switch; panel-ocr's
  default stays `Simple`, so a default run still diverges from upstream's unconditional
  refinement. The primary comment relocates from `crates/pc-detect/src/mask.rs:150` to
  `MaskRefineMode`'s `#[default] Simple` variant in `crates/pc-config/src/profile.rs`.

**One flagged-but-undecided item, left open for the two architects rather than resolved
unilaterally**: whether §16.39 item 2's "contradicts the word 'retirement' in §16.37 items
8/11(d)" claim needs its own back-pointer at §16.37, or whether the marker at §14 item 12
(the register entry that carries the actual proposition) is sufficient as scoped. The
fourth fresh-reader's own adjudication leaned "acceptable as scoped" but recorded both
sides — worth a quick architect confirmation before it's forgotten, not urgent.

### Next: A4-a (heavy, needs Codex)

Wire the `Annotation` branch into `pc_detect::run` per §16.39's ruling. Test code is fully
drafted (captured in this session's transcript, not yet written to disk) — six oracle-backed
tests in a new `crates/pc-detect/tests/a4_run_annotation.rs`, covering: mode acceptance +
mask emission, the Simple-stays-default quarantine, the undetected-mask pass reachability
with zero detected blocks, per-image error classification for an empty expanded window, disk
artifact writing, and end-to-end reproduction on the recorded page. A separate
`a4_coverage_operand.rs` (four more tests) locks in the operand ruling specifically,
including one synthetic fixture that's the *only* thing in the repo that can actually
discriminate the two candidate operands (the committed real page can't — both operands agree
on it). A4-b (the operand-specific implementation) is sequenced after A4-a lands.

## What's committed: lama-inpaint (LaMa neural-inpainting fallback)

Full `pc-inpaint` crate exists and works standalone: real ONNX LaMa session (verified against
the actual downloaded 512×512 model), tile geometry, Gaussian-fade blending, eligibility
checks, config, model registry partition. `66cf7e4` ratifies wiring it into
`pc-pipeline`'s (currently nonexistent) `Step::Inpaint` stage:
- **Mask-precedence correction**: §16.38 item 12(a) as originally written ("mask =
  inpainted_mask.or(...)") mis-transcribed upstream. The real inpainted-mask export is a
  three-layer alpha composite (combined_mask, then noise_mask if denoising ran, then
  inpainting output) — verified character-for-character against `image_export.py:241-263`
  at the pinned upstream commit. Again, independent convergence between architect and
  rust-engineer, no Fable needed.
- **Eligibility-first design**: the pipeline must call `pc_inpaint::select_regions()` before
  ever asking the `InpainterProvider` for a session, so a run with zero eligible regions
  never provisions the 207MB model / ONNX session.

### Next: L6-1 through L6-5 (needs Codex)

Five tasks, four Codex calls: L6-1+L6-2 batched (simple — `Step::Inpaint` enum insertion,
cache suffix constants), L6-3 (heavy — `pc-export` precedence/composite), L6-4 (heavy —
the actual pipeline stage wiring, `PipelineCtx.inpainters`, eligibility-first call site),
L6-5 (simple, depends on L6-4 — `--skip-inpaint` CLI flag + provider handoff). Full test
code for all five is drafted (captured in transcript) — 20 tests across five new test files.
**L6-1+L6-2 was attempted via Codex this session and hit the sandbox blocker described
above before writing anything** — safe to retry verbatim once Codex can write here; nothing
was left in a partial state (confirmed: `git status` in this worktree is clean).

One thing worth reading before dispatching L6-3: the drafted test file
`l6_inpaint_precedence.rs` includes one test marked `#[ignore] // BLOCKED — DO NOT FREEZE`
in the original draft — that block is now resolved by `66cf7e4`'s ratification, so the
`#[ignore]` should come off before handing it to Codex (the drafted content already
reflects the three-layer composite design; just confirm the attribute is removed).

## What's NOT started

- **Task #22**: consolidate the 4-copy `blend_channel`/`alpha_composite_over`/
  `resize_nearest_rgba`/`composite_rgb` duplication (`pc-mask`, `pc-denoise`, `pc-export`,
  `pc-inpaint`) into `pc_imageops`. Existing but unfulfilled ticket at spec §16.11 item 10.
  Should follow the L1/Gaussian-hoist pattern exactly (body byte-identical, frozen tests
  unedited). Needs Codex like everything else above.
- **Task #16 / #6**: the real-page benchmark (Simple vs Annotation vs LaMa). Blocked on both
  A4 and L6 landing — neither alternate mode can run end-to-end yet.
- Eventual merge reconciliation of `docs/PIPELINE_SPEC_V1.md` between the two branches
  (`mask-parity` used §16.39, `lama-inpaint` used §16.38, chosen specifically to avoid
  colliding — but the two branches' spec files have now diverged independently and will
  need a real merge, not just a fast-forward, whenever they're combined).

## Process notes for whoever picks this back up

- Both ratifications this session went through **3-4 fresh-reader review rounds each**
  before landing clean — this is normal for this project's pipeline, not a sign anything
  was unusually wrong. The recurring defect class was almost entirely: a number or count in
  corrective prose that was itself miscounted, or a claim whose grounding array/data source
  was misidentified. Every one of these was caught by an independent fresh-reader re-deriving
  the number from source rather than trusting the prior draft — keep doing that.
- Two genuinely useful debugging techniques from this session, worth reusing: (1) when a
  background agent's status looks stuck, check `git diff --stat` in the actual worktree for
  real file-level progress rather than trusting a status field alone (this project's own
  cookbook rule 6-adjacent principle, applied to agent monitoring); (2) when Codex's sandbox
  behaves unexpectedly, check `icacls` on the target directory against a known-working one
  before assuming it's a code/prompt problem.
