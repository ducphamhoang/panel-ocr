# Handover — TEMPORARY, delete when consumed

Written 2026-07-30, rewritten same day after F1 Phase 2's implementation landed. **This file is
scaffolding, not a record.** Delete it once #12/#13 (the F1 recording run) is fully done; anything
in it worth keeping permanently belongs in `PIPELINE_SPEC_V1.md`, `COOKBOOK.md`, or `RULINGS.md`
instead — and as of this rewrite, essentially everything from the previous version of this file
has already moved there. This version is deliberately short.

## Repo state

- Branch `claude/codex-plugin-install-jxirxa`, `HEAD` = `31bea40`, working tree clean, **nothing
  pushed** (pushing needs asking, per standing convention).
- Verification bar, actually run: `cargo test --workspace` (860 passed), the onnx tier
  (`cargo test --workspace --all-targets --features pc-cli/onnx`, 871 passed), `cargo clippy
  --workspace --all-targets --all-features -- -D warnings` clean, `cargo fmt --all --check` clean,
  `cargo test -p pc-testkit --test spec_supersession` (12/12). **Do not cite older figures** —
  847/858 and 738/749 both appeared in this repo's own docs earlier the same session and were
  stale by the time anyone read them; re-run the count in `COOKBOOK.md`'s rule 6, never quote it.
- `export PATH="$HOME/.cargo/bin:$PATH"` or cargo is not found (`COOKBOOK.md` rule 11).
- Git has no configured identity; commits used `GIT_AUTHOR_NAME=Claude
  GIT_AUTHOR_EMAIL=noreply@anthropic.com` (+ `GIT_COMMITTER_*`) as env vars.

## #12/#13 (F1 Phase 2) — code and tooling DONE, the maintainer-gated recording commit is NOT

Ten commits landed this session: the §16.29 ratification (identity operand, oracle-page location,
`DETECTOR_ORACLE.md` sequencing — Fable tie-break, full source in `docs/RULINGS.md`); a correction
to it caught by Codex mid-implementation; the identity-operand fix + reachable-shape controls;
nine cookbook cleanup items (one caught wrong and reverted — `UpstreamBoxOutsideFrame`, see
`COOKBOOK.md`); the Python detector-oracle recorder (`xtask/scripts/record_detector_oracle.py`);
un-stubbing `xtask`'s `Group::Detector`; two real bugs in `provenance_is_current_under` caught by
the stop-time review gate and fixed with regression tests (the detector group could never be
recognized as current, and its input page's digest was never checked); and a batch of fixes from an
independent rust-engineer review (stale doc comments, a write-ordering bug, stale cookbook
bookkeeping). See each commit message for its own account — they're detailed and current.

**What's left for #12/#13 to be *fully* done is exactly the atomic recording commit**, per §16.24
item 6 and the sequencing `docs/RULINGS.md`/the architect's plan laid out:

1. Rebuild the upstream oracle environment fresh (`COOKBOOK.md` rule 3's recipe) — the checkout,
   venv and ONNX weights used this session were session-scoped scratchpad state and are gone in a
   new session. (A full rehearsal run's output currently sits in
   `/tmp/claude-1000/.../scratchpad/detector_recording_run/detector/` from this session, but that
   scratchpad won't survive either — treat it as a reference, not a source, for the real run.)
2. Run `cargo xtask record-fixtures --only detector` for real, then `git mv` the oracle page from
   `tests/fixtures/upstream/oracle_pages/...E01P01.jpg` into the recorded output directory (§16.29
   item 2 — the recorder already copies it there as a plain file write; the atomic commit is where
   that becomes a tracked move) and update `tests/fixtures/upstream/ATTRIBUTION.md`'s row.
3. Hand-derive and author `leg1_rows_checked`'s real value and the real-page gate's literal
   `Expectations` — never from the artifact or a re-run (§16.28 Amendment 2 / §16.24 item 20(b)).
4. Write `docs/DETECTOR_ORACLE.md`'s verdicts (its skeleton + item 11's four doc-shape gates may
   land earlier in dormant-but-verified form per §16.29 item 3, but the verdicts themselves are
   atomic-commit content).
5. Update the frozen `EXPECTED_GROUPS`/`EXPECTED_DECLARED_PATHS`/`EXPECTED_COMMITTED_PATHS`/
   `EXPECTED_DETECTOR_PIN_GROUPS` constants (`crates/pc-testkit/tests/recorded_provenance.rs`,
   `xtask/tests/provenance_digests.rs`).
6. Get the three §16.20 item 3(f) signatures — non-self-reference (§16.13 item 4): none of them
   may be whoever produced the artifact.
7. Land all of the above in one commit (or none) — §16.24 item 6's atomicity rule.
8. Separately, immediately after: un-ignore the four `#[ignore]`d readers of the new fixture
   (`d7_run.rs` a6/b9, `p5_run.rs` b11, `n4_run.rs` b13), each with its own hand-derived expected
   values.

**One open item needs a ratification before or alongside that commit, not a unilateral fix**: the
independent review found §16.27 item 1(c)'s `YoloSplit` third conjunct is verified nowhere but
tautologically — see `COOKBOOK.md`'s "still owed" list (HIGH item) for the full account and why
closing it needs `Mechanism::DocumentedSplitMerge` to carry an upstream index, a design question
for the joint architects. Smaller MEDIUM/LOW items are tracked in the same place.

Then #14, then §16.24 item 5's ratified order: GPU-1 → GPU-2 → CPU-EP investigation → INI + Lab
NLM → LaMa → PSD → DBNet lines → Annotation (all v1.5 — see §16 "Summary of what v1 is NOT"; none
of this gates v1.0).

## The other remaining v1.0 task: P7 (manga-ocr ONNX backend)

Per `docs/PIPELINE_SPEC_V1.md` §13's consolidated task table, P7 is the only other incomplete
v1.0 task — `crates/pc-ocr` is currently just a trait + mock (212 lines), no real ONNX backend.
Once #12/#13's atomic commit lands, P7 is the last thing standing between here and v1.0 done.
