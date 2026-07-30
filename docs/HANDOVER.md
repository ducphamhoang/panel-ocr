# Handover — TEMPORARY, delete when consumed

Written 2026-07-30, after #23 landed. **This file is scaffolding, not a record.** Delete it once
the next session has read it and #12/#13 (the F1 recording run) is under way; anything in it worth
keeping permanently belongs in `PIPELINE_SPEC_V1.md`, `COOKBOOK.md`, or `RULINGS.md` instead. If
you are reading this and #12/#13 is already done, it is stale — delete it.

Everything from the #23 session that was worth keeping has already been migrated: the agent-load
result is in `CLAUDE.md`, the corrected ratification record is in `RULINGS.md`, the transcription
is `PIPELINE_SPEC_V1.md` §16.27, and two process lessons are in `COOKBOOK.md` rules 11 and 16. This
file does not repeat any of that — read those directly.

## Repo state

- Branch `claude/codex-plugin-install-jxirxa`, `HEAD` = `7477491`, working tree clean, **nothing
  pushed** (pushing needs asking, per standing convention).
- Verification bar, actually run (not assumed): **839 default / 850 onnx**, 6 ignored per tier,
  `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean, `cargo fmt --all
  --check` clean.
- `export PATH="$HOME/.cargo/bin:$PATH"` or cargo is not found (`COOKBOOK.md` rule 11).
- Git has no configured identity; commits used `GIT_AUTHOR_NAME=Claude
  GIT_AUTHOR_EMAIL=noreply@anthropic.com` (+ `GIT_COMMITTER_*`) as env vars.

## Do this first

**Step 0 — get the frozen-test authorization §16.27 item 1(g) deliberately withheld.** The F1
recording run needs a mechanical adaptation of the frozen struct literals in
`crates/pc-detect/tests/f1_oracle_comparator.rs`. No ruling grants this yet. Spawn the joint
`architect` + `rust-engineer` planning pass (per `CLAUDE.md`'s pipeline) to produce and quote a
citable ruling scoping exactly what "mechanical adaptation" covers, before touching that file. If
they disagree, that's what `fable-adjudicator` as tie-breaker is for — not a unilateral call.

**Step 1 — rebuild the oracle.** Gone again (scratchpad is session-scoped). `COOKBOOK.md` rule 3
has the recipe: ~1.6 GB, no sudo, ~10 minutes. Don't skip its step 5 import check. Model weights:
`comictextdetector.pt.onnx`, sha256 `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`
— re-verify after download.

**Step 2 — vendor the oracle page.** §16.24 item 17 ratified committing P01 (and its rejected
siblings P02/P03) into `tests/fixtures/upstream/`, with an `ATTRIBUTION.md` entry — never actually
done, across two sessions now. Source, recovered and verified this session:
`peppercarrot.com`'s `0_sources/ep01_Potion-of-Flight/low-res/ja_Pepper-and-Carrot_by-David-Revoy_E01P0{1,2,3,4}.jpg`.
P01 sha256 `3bef9922e09cea66ab12271da0070025768ae9bc5d286f41ced617468131267e` (441,914 B) — exact
match to item 17's recorded hash. P02 (448,475 B) / P03 (302,541 B) match item 17(a)'s recorded
sizes; no hash was ever recorded for those two. Do this as its own commit, not folded into the
recording commit.

## Then the actual work

**#12/#13 — the F1 recording run.** §16.27 is the schema to record against:

- Item 1: the derivation-recording schema (closed 4-value enum, the pristine+instrumented
  deep-copy binding condition, one exact law per derivation). Item 1's own text already disclaims
  the historical validation counts (44 vs. a 49 used elsewhere never reconciled) as unreproducible
  — record fresh counts from this run; do not carry the old numbers forward as if they were this
  run's result.
- Item 2: the new leg-1 gating row, `ours.rect == rect_yolo`, exact and epsilon-free, plus the
  mis-paired-derivation structural guard.
- Item 3: `DbnetScattered` as a mechanism — amends §16.24 item 4's enum and item 20(c)'s table,
  re-scopes §16.25 item 10. Note the identifier spelling and its lack of a `§14` register entry
  number were the transcribing engineer's own judgment call, not separately ratified by name —
  worth a second look if anything downstream depends on the exact spelling.
- Items 4–9: the six errata, already landed in the spec text (nothing further to transcribe,
  just implement/record against the corrected text).
- Item 10: 18(d) needs no action — closure note only, nothing was wrong there.

**Supersession gate**: `crates/pc-testkit/tests/spec_supersession.rs` now enforces every marker
gets a back-pointer, with per-target multiplicity AND per-sub-item distinctness both checked
(`EXPECTED_PARSED_CLAIMS = 12`, `RATIFIED_SUPERSESSIONS` has 12 rows). Adding a new claim during
#12/#13 means raising the count and extending the constant in the same commit — the gate fails
loud and names the exact site if you don't.

**Step 1a still applies**: any new `§16.x` entry recording this run's outcome gets a fresh
`fresh-reader` review before commit — spawned new, never resumed, never the author.

Then #14, then §16.24 item 5's ratified order: GPU-1 → GPU-2 → CPU-EP investigation → INI + Lab
NLM → LaMa → PSD → DBNet lines → Annotation.

## Open risks worth carrying

- **Two agents editing the same file concurrently is now a proven-survivable pattern, not just a
  risk to avoid** (`COOKBOOK.md` rule 11) — but it cost real time working out reconciliation both
  times it happened this session. Prefer sequencing write-agents on the same file when the task
  allows it; only run them concurrently when the work is genuinely separable.
- **Stop-time review findings were 5-for-5 real this session** (`COOKBOOK.md` rule 16) — every one
  was a claim about what a file contains, not a runtime-semantics guess. Keep trusting file-content
  claims from that channel; still verify anything that claims what code *does* by actually running
  it.
- **The still-owed list in `COOKBOOK.md` rule 11** (`provenance_is_current`'s redundant condition,
  the R1–R23 doc-comment numbering, `UpstreamBoxOutsideFrame`'s missing register entry) is
  unrelated to #12/#13 and can be picked up whenever convenient — not blocking.
