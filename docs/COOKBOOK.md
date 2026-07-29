# Cookbook — recurring failure patterns and how we decided

Working notes for this project's *process*, not its algorithms. The spec
(`PIPELINE_SPEC_V1.md`) says what to build; this file says how we have been getting it
wrong and what we do about it. Each entry is a pattern we hit more than once, or one whose
cost was high enough to be worth never repeating.

Read this before an audit, before ratifying a deviation, and before trusting a green test
suite. Numbered §-references are to `PIPELINE_SPEC_V1.md`.

---

## 1. A test whose name claims more than its assertion verifies

**The dominant defect class of this project.** Recorded as the general rule in §16.19
item 8. Six instances found in one audit pass; the suite was green throughout.

Worst case: `pc-detect/tests/d7_run.rs` had a helper documented as comparing "every field
(handles by `path`), which is strictly stronger than string equality". It wasn't.
`ImageHandle`'s `PartialEq` (`pc-core/src/image_handle.rs:107-113`) compares `path` only,
and in memory mode both sides are `None` — so the field is skipped entirely. An auditor
proved that two runs over **different images** (masks of 460 vs 623 non-zero pixels)
compared EQUAL. The test could not fail.

Other shapes of the same bug:

- `assert!(tensor.is_f32() == (tensor.element_type == ELEMENT_TYPE_F32))` — where
  `is_f32()` *is literally that comparison*. Asserts `x == x`.
- `assert_eq!(f(x), f(x))` — same function, same input, both sides.
- A CI step named "frozen snapshot guard" that counted `.snap` files instead of checking
  their paths. A snapshot in the wrong place passed.
- Comments asserting properties nobody verified: `noisy_gray`'s "bit-identical across
  platforms" claim, `padding_tiers_compose`'s per-step-vs-end-clamp note.

**The rule:** for every assertion, ask *what concrete wrong value would make this line
fail?* If you cannot name one, the line is decoration. For any `PartialEq`-based
assertion, go read the `impl` — a derived-looking `PartialEq` may compare almost nothing.
Prefer asserting on the specific bytes/fields you care about over comparing two whole
structs.

**Corollary:** the same scepticism applies to a doc comment. A false comment is worse
than no comment, because the next reader will build on it. Four doc-vs-code
contradictions were found in the same pass.

---

## 2. Never claim a contract doesn't exist based on a grep

The most expensive mistake of the session, and the reason rule 3 exists.

A review gate flagged that our change violated a "do not cache failures" contract. I
grepped for `retry|retries|re-attempt|transient`, found nothing, and told the user
emphatically the finding was "factually wrong" and "no such contract exists."

§16.12 item 2 said, in words none of those terms match:

> *"Only successful ONNX construction is cached; failures are not cached."*

Both Opus architects found it independently, on the same spec I had just searched.

**The rule:** a grep proves a *string* is absent, never that a *concept* is absent. Before
denying a clause exists, read the relevant section — the whole section — or ask an agent
whose context is the spec rather than a search index. And when the tooling flags
something, the prior probability that a stop-gate hallucinated a spec clause is much lower
than the probability that my search terms were wrong.

**How it was fixed:** the sentence was quoted verbatim in §16.19 and marked
**SUPERSEDED**, with item 2 rewritten to "neither outcome is retried within a run." A
superseded clause is never silently deleted — the next reader needs to know the rule
changed and why, otherwise they will reintroduce it.

---

## 3. Upstream PanelCleaner is the tiebreak oracle

When the spec is ambiguous or two readings conflict, the answer is in
[VoxelCubes/PanelCleaner](https://github.com/VoxelCubes/PanelCleaner) — the implementation
this project is a port of. This is now standing policy, and it has already earned its
keep twice.

It is cheap to consult properly (a full local run, not just reading Python). Verified
end-to-end against upstream commit `0afa21f`; run from a scratch directory:

```bash
# 1. uv, user-local. `python3 -m venv` needs the python3-venv package, which needs sudo.
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$PWD/uvbin" sh
export PATH="$PWD/uvbin:$PATH"

# 2. upstream at a pinned commit -- an oracle nobody can re-run is not an oracle
git clone https://github.com/VoxelCubes/PanelCleaner.git
git -C PanelCleaner checkout 0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3

# 3. venv, and point uv at it for every subsequent `uv pip`
uv venv pcvenv
export VIRTUAL_ENV="$PWD/pcvenv"

# 4. the DETECTOR-PATH SUBSET of requirements.txt. Do not use `-r requirements.txt`:
#    it drags in PySide6 (Qt GUI), simple_lama_inpainting, pytesseract, psd-tools and
#    dbus-python, none of which the detector path touches. This filter reproduces
#    exactly the set that works; PyYAML is icon-cache-only per upstream's own comment.
grep -vE '^(PySide6|simple_lama_inpainting|pytesseract|psd-tools|strenum|dbus-python|pywin32|win10toast|pyuac|PyYAML)' \
  PanelCleaner/requirements.txt | grep -vE '^\s*$' > detector-reqs.txt
uv pip install --torch-backend=cpu -r detector-reqs.txt

# 5. prove the detector entry point actually imports before trusting any output
PYTHONPATH="$PWD/PanelCleaner" ./pcvenv/bin/python -c \
  'import pcleaner.ctd_interface as c; print("OK", hasattr(c, "model2annotations"))'
```

~1.6 GB, no sudo, ~10 minutes. `--torch-backend=cpu` resolves `torch==2.13.0+cpu` /
`torchvision==0.28.0+cpu`; the default CUDA build is several GB larger for no benefit here.
Add `--dry-run` to step 4 to re-verify the recipe resolves without downloading anything.

**Install the real dependency rather than stubbing it.** `pcleaner.ctd_interface` failed on
`No module named 'manga_ocr'` — it is in `requirements.txt`, and the filter above keeps it.
Stubbing it would have produced an output that looks like upstream's but isn't: the exact
failure mode an oracle exists to prevent. Step 5 exists for the same reason — an import
error surfaced at run time is easy to paper over with a mock.

**Pin what you compared against.** Record the upstream commit SHA, the resolved dependency
versions (`uv pip freeze`), the weights' sha256, and the input page's sha256 alongside any
result. Without those, a divergence found later cannot be attributed to us or to them.

### What this found

Upstream's `group_output` (`comic_text_detector/utils/textblock.py` ~:483-491):

```python
for blk in blk_list:
    if len(blk.lines) == 0:                      # <-- guards BOTH branches below
        bx1, by1, bx2, by2 = blk.xyxy
        if mask is not None:
            mask_score = mask[by1:by2, bx1:bx2].mean() / 255
            if mask_score < mask_score_thresh:
                continue                          # the false-positive filter
        blk.lines = xywh2xyxypoly(xywh)...        # line synthesis
```

The `if len(blk.lines) == 0:` guard wraps the filter **and** the synthesis. Upstream
therefore applies its false-positive filter **only to line-less blocks**. Our §8.3 step 6
applies `mask_coverage < 0.1` to **every** block, justified at spec line 686 as *"exactly
the code path upstream takes when the line map yields nothing."*

It isn't. On the test page upstream's blocks had 3 and 2 line polygons, so upstream never
ran the filter on them at all. **Consequence: a block with DBNet lines and low mask
coverage is kept by upstream and dropped by us — a silently missing text box.** It did not
materialise on that page (2 boxes = 2 boxes), but it is structural. It also explains the
1–3 px top-left inset in our coordinates: upstream's `xyxy` is line-informed, ours is the
raw YOLO box.

*Status: awaiting architect ratification — either narrow our filter's scope to match, or
correct the provenance sentence to say the parity is conditional. Do not leave it as a
sentence that claims unconditional parity.*

**The lesson beyond the bug:** a parity claim verified on one page is a claim about that
page. This one was written as a claim about the algorithm. When you write "this matches
upstream", say *on what input you checked*, and check whether the code path you compared
was even taken.

---

## 4. Classify a failure by what the function takes, not by how it feels

Three separate bugs came from getting per-image vs run-fatal wrong, and the resolution is
a single mechanical test.

**Ask: does the failing function receive the image?**

| Function | Takes image? | Classification |
|---|---|---|
| `DetectorProvider::initialize_detector(&self)` | no | image-independent *by construction* → provider may declare run-fatal (§16.19 item 4) |
| `TextDetector::detect(&self, image)` | yes | image-dependent *by construction* → per-image `Failed { step: Detect }`, exit 2 (§5.2) |

This is decidable by reading a signature, which is why it beats intuition.

Two things that are **not** valid reasons to promote a failure to run-fatal:

- *"It will obviously fail for every image too."* Deterministic ≠ run-fatal. A profile that
  makes every page fail in masking, or a replay dir with every fixture missing, is still
  §5.2's business. Reading it otherwise swallows §5.2 whole.
- *"The `StageError` variant looks fatal."* Both a provisioning refusal and a missing
  replay fixture arrive as `StageError::Model`. **Fatality is declared by the provider**
  (`DetectorProvider::failures_are_run_fatal`, defaulting to `false`), never inferred at
  the consumer. That is §16.19 item 5, and it exists because inferring it from the variant
  is what caused the live defect below.

**The live defect this fixed:** a missing replay fixture for page 7 aborted the whole
batch, killing pages 1–6 — a direct violation of §16.12 item 3, sitting in committed code.

**A related trap — §16.19 item 5(b) is a predicate, not a law.** I once cited it to argue
session poisoning was run-fatal. It is the criterion a *provider* uses when declaring its
own failures fatal; it is not a universal statement about all failures everywhere. When
quoting a clause, check whether its subject is the thing you are applying it to.

---

## 5. Don't manufacture an image-independent failure out of a shared resource

`OnnxDetector` holds one `Mutex<Session>` for the entire run (`DEVIATION(15)`;
`concurrent_models > 1` is warned-and-ignored in v1), and every image clones the same
`Arc`. The original `detect` held the guard to the end of the function — so the critical
section covered `bind_outputs`, `validate_output_shapes`, `decode_blocks`, `decode_mask`:
all pure arithmetic on owned data, in our own code.

One panic in that arithmetic poisoned the session, and every remaining page got
`Inference("ONNX session mutex was poisoned")` — **a failure named after our lock rather
than its cause, which no spec clause asked for.** Note the shape: correctly classified per
§16.12, and still wrong, because the failure itself was our invention.

**The fix (§16.19 item 10):** hold the lock for exactly `Session::run` plus the copy into
owned `Vec<f32>`; decode outside it.

```rust
let (metas, values) = {
    let mut session = self.session.lock().unwrap_or_else(/* recover, see below */);
    let runtime_outputs = session.run(ort::inputs![input])?;
    extract_outputs(&runtime_outputs)?      // SessionOutputs borrows the session
};                                          // guard dies here
decode_outputs(&metas, &values, &boxed.geometry)   // ungated, no lock held
```

**Two things that made this correct rather than merely narrower:**

1. **Don't add a panic site while fixing a panic-poisoning bug.** `decode_outputs` became
   public over two independent slices, so `values[binding.blks]` would be a fresh panic
   surface. It validates `values.len() == metas.len()` and every bound index, returning
   `StageError::InvalidInput`. Self-defeating otherwise.
2. **Poison recovery needs a per-site justification, not a precedent.** We recover poison
   in two places for two *different* reasons: `pc-cli/src/detector.rs` because its mutex
   guards `()` (no invariant to corrupt); `pc-detect/src/onnx.rs` because `ort` rc.12's
   `Session::run(&mut self)` (`session/mod.rs:212`) delegates to `run_inner(&self)`
   (`:272`), which only *reads* `Session`'s three fields — the `&mut` is an aliasing
   device so `SessionOutputs<'s>` can't coexist with a second run. Cost of a mid-run panic
   is a leak, not corruption.

   **"We recovered poison elsewhere" is never a reason.** Cite the sources in the comment
   so an `ort` bump has a concrete thing to re-check.

---

## 6. Prove the test *ran*, not that it compiles

The `onnx` feature's tests were type-checked by `cargo clippy --all-features` for weeks
and **never executed**. Eleven tests. `--all-features` on a check-only pass is not
coverage; clippy never links.

`.github/workflows/ci.yml` now has a separate `test-onnx` job running
`cargo test --workspace --all-targets --features pc-cli/onnx`, with a deliberate note that
`ORT_SKIP_DOWNLOAD` must **not** be set there — it is fine for clippy (which never links)
and fatal for a tier that links test binaries.

**Related trap:** `#[ignore]` is invisible in a green summary. I claimed no onnx test was
ignored; `d4_real_weights_smoke_test` is, and my grep anchored on `ignored` in a way its
reason string ran past. Count ignored tests explicitly:

```bash
cargo test --workspace 2>&1 | grep -E "^test result" \
  | awk '{p+=$4; f+=$6; i+=$8} END {print "passed="p" failed="f" ignored="i}'
```

Current bar, both tiers: **735** default / **746** onnx, 6 ignored, clippy
`--all-features` 0, fmt 0.

**Also:** a gate that is `#[ignore]` + `unimplemented!()` enforces *nothing*. Four of them
(§8.7(A)6, §8.7(B)9, §9.7(B)11, §11.7(B)13) are placeholders blocked on F1. That is a
deliberate, documented state — but never count them as coverage.

---

## 7. Never accept a produced output as its own expected value

Accepting a snapshot declares *"whatever the code emitted is the truth."* If the code was
wrong that day, the bug is frozen as the expected value and the test defends it forever —
it then proves only that the code still does what it did, never that it matches the spec.

**Snapshots were dropped project-wide** by §16.20 (Fable tie-break); `insta` is out of the
manifest and both former sites use hand-written assertions. The principle did not go away
with them — it moved to **F1's recorded detector fixture** plus a committed upstream oracle
run (§16.20 item 3, `docs/DETECTOR_ORACLE.md`). Read rule 12 for *why* it moved, which is
the more useful lesson.

The break in the circle must come from **outside the code under test**:

- A hand-trace derived from the spec text.
- A recorded run of upstream PanelCleaner (rule 3) — but **per field, not per site.**
  Upstream is a real oracle for `scale`, `image_size` and box geometry, and *not* for
  `raw_mask` (a different algorithm — measured IoU 0.258, since v1 ships
  `MaskRefineMode::Simple` and upstream runs `refine_mask`), `mask_coverage` (a different
  operand — upstream's `group_output` gets the *unrefined* mask, ≈2× apart), or
  `confidence` (not persisted at all). For the §9.7(B)11 tiers it is *worse* than a
  hand-trace, because §14.2, §14.3 and §14.13 all perturb that stage's input. "Upstream is
  a stronger oracle" is true only where upstream computes the same quantity.

**And it must not be self-attested.** §16.13 item 4: the reviewer has to be independent of
whoever produced the artifact. If I derive the expected value using the same reading of the
spec that produced the code, and generate the output by running that code, a misreading
confirms itself. Same reason F3 needed a *recorded model signature*
(`tests/fixtures/recorded/model_signature/`, sha256-verified against the real artifact by
`xtask/src/model_signature.rs:91`) instead of asserting `ROW_STRIDE == ROW_STRIDE`.

**Running the oracle is not the same act as judging it.** Producing the upstream run may be
done by whoever produced the artifact — every degree of freedom (version, commit, profile,
model sha256 *and backend*, page sha256, command line) is enumerated in committed
provenance and reproducible by a third party, so nothing can be smuggled in. **Signing the
verdicts may not.** That asymmetry is what lets an agent do the mechanical work while the
human's job shrinks to a checkable list.

**Prefer an exact identity to a tolerance.** Where a divergence has a known mechanism,
state it as an equation and assert it. Concrete: `upstream.xyxy == bbox(ours.rect ∪
bbox(upstream.lines))` held 8/8 coordinates on one page and, when deliberately attacked on a
12-box page it had never seen, **10/10 blocks and 40/40 coordinates**. A tolerance would
have bought nothing and cost real detection power — ±3 px per edge is ±12–19% of area on
small boxes, and would absorb exactly the off-by-one letterbox error §14.14 exists for.

**Know which quantities are unmeasurable before promising to measure them.** Two independent
probes found that a **1-LSB** input difference (our f32-exact bilinear vs OpenCV's
fixed-point weights: max Δ 1, 77% of pixels identical) moves a detection confidence by
**0.079–0.089**, while §8.3 step 4's objectness and class gates both sit at 0.4. So no
defensible confidence tolerance exists — one spanning the noise also spans "kept" and
"dropped". Geometry, measured under the same perturbation, stayed exact. Gate geometry;
record confidence as a diagnostic and say so.

---

## 8. Tests are frozen — but know the three exits

Once written, a test is not edited to make it pass. Only the implementation is iterated.
There are exactly three legitimate ways out, and picking the wrong one is how a suite
rots:

1. **Additive strengthening** — adding a *new* assertion or test alongside the existing
   ones. Always allowed. Precedent language: *"a frozen-test addition, not an amendment"*
   (§16.14 item 4).
2. **Contradiction with the spec** — the test asserts something the spec forbids. Goes to
   **both** Opus architects jointly, never a unilateral edit by Codex or the orchestrator.
   Recorded as a §16.x ratification when resolved. Real instance: §11 item 22, where a
   test built a region with σ = 0.1 while asserting it got selected under a strictly-greater
   cutoff at 0.25 — unsatisfiable, and the slip was in a test whose actual subject was
   something else.
3. **A deliberate deviation from upstream** — gets a `DEVIATION(n)` comment at the
   implementation site plus a §14 register entry. §16-only deviations use the qualified
   form `DEVIATION(§16.11 item 5)`.

Keep the §14 register and the code in sync **in both directions** — it has desynced each
way: entries present in the register with no comment at the implementation site, and
comments in code with no register entry.

---

## 9. Verify your monitor before you trust it

Three monitors reported progress that wasn't there. All three failed silently, which is
the whole problem.

- `pgrep -c -f 'cargo|rustc'` **matched its own shell**, because the pattern string was in
  that shell's command line. Reported "cargo/rustc=2" when zero were running. Use a
  bracketed class: `[r]ustc`.
- **Agent output-file size is not progress.** Those files sit at 131 bytes until the agent
  completes, then jump to full size. A monitor watching `wc -c` sees a hang that isn't one
  and a finish it can't distinguish from a crash.
- A monitor **guessed the status JSON's shape**. Use the companion `status --json` API and
  read the real key (`running`), don't infer it.

**The rule:** pick a signal that provably changes when work advances, and *test the monitor
against a known-idle state first.* A monitor that always says "fine" is worse than none —
it converts a hang into a silent wait. Per `CLAUDE.md`, an unchanged state for a full
interval is a suspected hang to investigate, not slow work to wait out.

---

## 10. Report the thing you actually verified

A running list of my own over-claims, kept because the pattern repeats and the correction
is always the same: state what was measured, on what, and what remains unmeasured.

| Claimed | Actually |
|---|---|
| A new concurrency test was "the proof the existing suite couldn't give" | It passed **299/300** runs against a deliberately gate-less implementation. Fixed with `Barrier::new(4)` + 25 fresh-provider iterations |
| Two `assert_json_snapshot!` sites "exist" | Both are comments inside `#[ignore]`d `unimplemented!()` bodies; `insta` isn't a dependency of any crate |
| A CLI failure prints one `fatal:` line | Empirically `error: model error: …`, exit 1. `pc-cli`'s fatal path returns `Err(anyhow!(..))` from `run_pipeline` *before* `BatchSummary::render` is consulted. I had passed an advisor's description into the spec verbatim without running the binary |
| No onnx test was `#[ignore]`d | `d4_real_weights_smoke_test` is |
| Caps lead-ins weren't house style | Precedent exists at spec `:2243`. Normalised anyway, but as a judgment call, said so |
| Doubted a keystone test as circular | It *was* externally grounded, via the sha256-verified recorded signature |

Codex is subject to the same rule: it judged `validate_output_shapes` "required no
extension" when its messages named a failing shape but not the output's name or siblings —
leaving three of four failure paths diagnosable and one not. **A subagent's self-report is
a claim, not a verification.** Re-run the suite yourself; read the diff.

---

## 11. Small operational notes

- **`cargo` is not on the default PATH** for dispatched agents. Every Codex prompt needs
  `export PATH="$HOME/.cargo/bin:$PATH"`.
- **Git has no configured identity** in this environment (`fatal: empty ident name`). Match
  the existing commits via env vars rather than writing `git config`:
  `GIT_AUTHOR_NAME=Claude GIT_AUTHOR_EMAIL=noreply@anthropic.com` (+ `GIT_COMMITTER_*`).
- **`python3 -m venv` fails** (`python3-venv` absent, installing it needs sudo). Use
  user-local `uv`.
- **Committing is pre-authorized** once `CLAUDE.md`'s verification bar is met — all of it
  actually run, not assumed. **Pushing still requires being asked.**
- **`git commit -a` does not stage untracked files.** It silently committed a `CLAUDE.md`
  edit while leaving out the new file the commit was *about*. `git status --porcelain` after
  every commit; `??` lines mean the commit is incomplete.

---

## 12. Check that a review gate points at the artifact that carries the risk

The most expensive kind of wrong safeguard is not a missing one — it is one that exists,
looks rigorous, and guards a copy.

§15.10(a) mandated a hand-traced human review before the first `insta` snapshot could be
committed. Sound-sounding, and aimed at nothing. Under `ReplayDetector` (§7.2.1) the
snapshot's `rect`, `class_index` and `confidence` are **copied out of
`_detector_blocks.json` untouched** (`crates/pc-detect/src/lib.rs`); the only values it
computes are `mask_coverage`, the survivor set, `scale` and `image_size`. So the gate
reviewed a transcription, while the recorded fixture — the artifact that actually accepts
model output as truth, and the one a wrong `PAD_VALUE` or a mis-cropped mask would poison —
had **no safeguard at all.**

Both Opus reviewers converged on this independently, from opposite directions, and Fable's
ruling was that an amendment strengthening the snapshot review *"does not cure that
inversion — it decorates it."* The fix was to delete the snapshots and re-aim the obligation
at the recording step.

**The test to run on any gate you write or inherit:** trace the value backwards from the
assertion to where it was *produced*. If the answer is "another committed file", the gate is
checking a copy, and the risk lives one hop upstream. A gate on derived data can only catch
derivation bugs — never the input it derived from.

**And ask what the gate *enumerates*, not only what it asserts** — see rule 13, which is the
same family and cost three more iterations to find.

Two related traps in the same family:

- **A dependency graph that hides the same inversion.** `insta` was declared in the
  workspace manifest and used by zero crates, while a CI job policed snapshot *paths* with
  an allowlist derived from insta's naming convention rather than from any real macro call —
  the job's own comment admitted the paths were UNVERIFIED. A guard whose expectations were
  never confronted with reality is rule 9 in a different costume.
- **A constant change that turns a passing test vacuous — or worse, silently valid.** When
  `PAD_VALUE` moved 114 → 0, six assertions comparing pixels against `Rgb([PAD_VALUE; 3])`
  stayed sound only because no test's content colour was black. A seventh
  (`to_nchw_divides_by_255_and_not_by_256`) used `PAD_VALUE` as its sample byte, where
  `0/255 == 0/256` — that one *failed*, which is the lucky outcome. **After changing any
  constant a test compares against, re-derive whether each assertion can still fail.** And
  check the non-test direction too: the change was safe only because nothing identifies the
  pad region *by colour* (`crop_letterbox` uses `dw`/`dh` geometry), so a black-bordered
  page cannot be mistaken for padding. That was the real risk, and it needed a grep, not an
  assumption.

---

## 13. Ask what a gate *enumerates*, not only what it asserts

Rule 1 asks whether an assertion can fail. That is necessary and not sufficient: an
assertion can be perfectly falsifiable and still be **blind to everything outside the set it
iterates**. A gate is only as complete as its input set, and the input set is usually
invisible in the assertion.

The recorded-fixture digest gate (§16.13 item 6) took **five** iterations. Each version was
green, each looked thorough, and only the last one was both:

| # | what it bound | how it stayed satisfiable while broken |
|---|---|---|
| 1 | three hand-named artifacts | never visited `tests/fixtures/recorded/model_signature/` at all — it declares digests as `output_sha256`/`source_sha256`, so even a generic walk keyed on `sha256` found nothing there, leaving a committed file verified by nothing |
| 2 | counts: 6 declarations, 4 committed | **cardinality is not identity** — duplicate one declared path, drop another, and the totals still read 6/4 while the dropped fixture goes unverified |
| 3 | the exact sorted *sets* of paths | a declaration could sit in the wrong group's `PROVENANCE.json`: path set identical, every byte verified, but no provenance file described its own group |
| 4 | + group locality | **coverage ran one direction only** |
| 5 | + every file is declared | — |

Iterations 1–4 are rule 1's family: the check was satisfiable without doing the work.
**Iteration 4→5 is a different failure and the reason this rule exists.** The gate verified
that every *declaration* had a matching file with the right bytes in the right group — and
never that every *file* had a declaration. It enumerated group **directories**, never the
files inside them. An orphan fixture could sit in the tree with no digest and no provenance,
trusted by every consuming golden test. Proven by copying a fixture to
`nlm/undeclared_orphan.png`: the gate passed.

**The question that finds this class:** for each thing the gate protects, *what does it
iterate to find them?* If the answer is "a list I wrote", "a set derived from the artifact
under test", or "the manifest", then anything absent from that source is unguarded — and
absence is exactly what you are trying to detect. Bidirectional coverage means enumerating
**both** populations independently and asserting they match.

**Corollary — a broad exemption is the bypass wearing different clothes.** Closing this
needed exactly two exemptions (a group's own `PROVENANCE.json`; the root `.gitkeep`), each
named. `ignore dotfiles` or an extension allowlist would have re-opened it silently. So the
exempt set is itself asserted: a stray file must not pass by merely *looking*
exemption-shaped.

**Two related traps in the same family:**

- **A hard-coded expected count is a ratchet in the wrong direction.** It fails when
  something is added — the safe case, since a human then looks — and passes when something
  is quietly swapped. Prefer asserting the expected *set*; the count comes free from it.
- **Diagnostics are part of the gate.** A count mismatch says "6 != 7" and sends the reader
  hunting. A set mismatch can say which paths appeared unexpectedly and which expected ones
  went missing. The second costs a few lines and turns a failure into a fix.

*Honest note on how these were found:* the review gate caught iterations 1, 2 and 4, not me,
and iterations 1 and 2 were my own specification errors — I chose the count as the safeguard
and named only three artifacts. What I did do right was falsify each fix myself rather than
accept a report (rule 10), which is how the locality and orphan proofs got run at all. Every
row above was demonstrated by deliberately breaking the tree and watching the gate fail, then
restoring it — and `git status --porcelain -- tests/fixtures/` being empty is a hard
precondition before any commit, because a leaked probe is precisely the corruption this gate
exists to detect.
