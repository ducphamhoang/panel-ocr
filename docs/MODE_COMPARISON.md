# Mode comparison benchmark (spec §16.43)

**Generated in full by `cargo xtask mode-bench` — do not hand-edit.** Every value here
is measured at generation time. This is a **non-gating** report: it gates no CI job and
blocks no merge, the same status as `cargo xtask calibrate-goldens` (§7.3) and
`cargo xtask mask-sweep` (§16.35).

## 1. Method and provenance

- **Source:** --demo-bubbles (7 vendored crops)
- **Source detail:** The 7 vendored crops — the only source carrying a reference. Every crop is under 512px on both axes, so each LaMa cell here is a single edge-replicated tile (§16.38 items 5(c), 5(d)) and this source cannot evidence `DEVIATION(24)`'s tiling behavior.
- **Cells run:** simple, annotation, simple+lama, annotation+lama
- **Stage sequence:** `pc_detect::run` → `pc_preprocess::run` → `pc_mask::run` → `pc_pipeline::run_inpaint`, invoked directly (not `pc-cli`'s `run_clean`).
- **Metric:** `pc_testkit::golden::GoldenReport` (`compare_gray_with_shape` / `compare_gray`) — no new metric type.
- **Eligibility:** `pc_inpaint::select_regions`, computed and reported for every cell (it reads only `MaskRegionStats`), LaMa or not.

## 2. Device

device: requested cpu; no execution provider registered explicitly (ONNX Runtime's built-in CPU provider is implicit); per-node operator placement is not claimed

That is the **one** device statement for this whole invocation: one `--device` flag, resolved once through `pc_core::device::resolve`, one resolved policy. The per-stage rows below carry only per-stage facts and cross-reference the statement above; they do not restate it, and there is no per-session device heading.

| Stage | Model path | Expected sha256 | Digest verified? | Session constructed? | Constructing function |
|---|---|---|---|---|---|
| detector | C:\Users\ducph\AppData\Local/panel-ocr/models/comictextdetector.pt.onnx | `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f` | yes | yes | `pc_detect::onnx::OnnxDetector::from_path_with_config` |
| inpainter | C:\Users\ducph\AppData\Local\panel-ocr\models\lama-manga.onnx | `50a1abae0d73bd46d08eae36c8590cd59ad09029494c9698702b050ef00b0100` | yes | yes | `pc_inpaint::onnx::OnnxInpainter::from_path_for_device` |

Mechanism disclosure per stage — not a second policy statement:

- **detector:** its session constructor takes the resolved device policy directly and attempts real registration when a provider is requested — refusal now happens only when the stage has no ratified path for the requested device (§16.47 item 4), not as a substitute for registration
- **inpainter:** its construction route (`from_path_for_device`) re-resolves the same requested device through the same resolver

## 3. Per-page, per-cell measurements

| Page | Cell | Mask mode | Detected boxes | Masking regions | Succeeded | Failed | Dropped | Eligible | Inpainting ran | Tiles inferred | Reference agreement (not quality) | Outcome |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| black | simple | Simple | 2 | 2 | 1 | 1 | 0 | 1 | no | — | exact 0.842624 / mad 7.841926 / ssim 0.786630 | measured |
| black | annotation | Annotation | 2 | 2 | 1 | 1 | 0 | 1 | no | — | exact 0.843446 / mad 7.839722 / ssim 0.787030 | measured |
| black | simple+lama | Simple | 2 | 2 | 1 | 1 | 0 | 1 | yes | 1 | exact 0.729399 / mad 4.352432 / ssim 0.836432 | measured |
| black | annotation+lama | Annotation | 2 | 2 | 1 | 1 | 0 | 1 | yes | 1 | exact 0.764254 / mad 4.435287 / ssim 0.848105 | measured |
| darkrays | simple | Simple | 2 | 1 | 0 | 1 | 0 | 1 | no | — | exact 0.812109 / mad 13.989694 / ssim 0.729663 | measured |
| darkrays | annotation | Annotation | 2 | 1 | 0 | 1 | 0 | 1 | no | — | exact 0.812109 / mad 13.989694 / ssim 0.729663 | measured |
| darkrays | simple+lama | Simple | 2 | 1 | 0 | 1 | 0 | 1 | yes | 1 | exact 0.780123 / mad 8.020478 / ssim 0.889442 | measured |
| darkrays | annotation+lama | Annotation | 2 | 1 | 0 | 1 | 0 | 1 | yes | 1 | exact 0.851773 / mad 3.974639 / ssim 0.937458 | measured |
| handwritten | simple | Simple | 0 | 0 | 0 | 0 | 0 | 0 | no | — | exact 0.661616 / mad 20.541982 / ssim 0.621282 | measured |
| handwritten | annotation | Annotation | 0 | 0 | 0 | 0 | 0 | 0 | no | — | exact 0.661616 / mad 20.541982 / ssim 0.621282 | measured |
| handwritten | simple+lama | Simple | 0 | 0 | 0 | 0 | 0 | 0 | no | — | exact 0.661616 / mad 20.541982 / ssim 0.621282 | measured; inpainting was reachable but nothing was eligible on this page — `pc_pipeline::run_inpaint` returned `Ok(None)` before any model was sought |
| handwritten | annotation+lama | Annotation | 0 | 0 | 0 | 0 | 0 | 0 | no | — | exact 0.661616 / mad 20.541982 / ssim 0.621282 | measured; inpainting was reachable but nothing was eligible on this page — `pc_pipeline::run_inpaint` returned `Ok(None)` before any model was sought |
| nightmare | simple | Simple | 1 | 1 | 0 | 1 | 0 | 1 | no | — | exact 0.775377 / mad 7.396528 / ssim 0.826528 | measured |
| nightmare | annotation | Annotation | 1 | 1 | 0 | 1 | 0 | 1 | no | — | exact 0.775377 / mad 7.396528 / ssim 0.826528 | measured |
| nightmare | simple+lama | Simple | 1 | 1 | 0 | 1 | 0 | 1 | yes | 1 | exact 0.751601 / mad 4.549902 / ssim 0.880403 | measured |
| nightmare | annotation+lama | Annotation | 1 | 1 | 0 | 1 | 0 | 1 | yes | 1 | exact 0.774152 / mad 4.584981 / ssim 0.883350 | measured |
| ray | simple | Simple | 1 | 1 | 0 | 1 | 0 | 1 | no | — | exact 0.511208 / mad 50.733995 / ssim 0.373859 | measured |
| ray | annotation | Annotation | 1 | 1 | 0 | 1 | 0 | 1 | no | — | exact 0.511208 / mad 50.733995 / ssim 0.373859 | measured |
| ray | simple+lama | Simple | 1 | 1 | 0 | 1 | 0 | 1 | yes | 1 | exact 0.560672 / mad 44.471564 / ssim 0.474731 | measured |
| ray | annotation+lama | Annotation | 1 | 1 | 0 | 1 | 0 | 1 | yes | 1 | exact 0.560030 / mad 44.522333 / ssim 0.474612 | measured |
| spikey | simple | Simple | 2 | 2 | 1 | 1 | 0 | 1 | no | — | exact 0.873488 / mad 10.659174 / ssim 0.832276 | measured |
| spikey | annotation | Annotation | 2 | 2 | 1 | 1 | 0 | 1 | no | — | exact 0.873360 / mad 10.659477 / ssim 0.832261 | measured |
| spikey | simple+lama | Simple | 2 | 2 | 1 | 1 | 0 | 1 | yes | 1 | exact 0.901282 / mad 4.634125 / ssim 0.925114 | measured |
| spikey | annotation+lama | Annotation | 2 | 2 | 1 | 1 | 0 | 1 | yes | 1 | exact 0.905136 / mad 4.511132 / ssim 0.926212 | measured |
| square | simple | Simple | 1 | 1 | 0 | 1 | 0 | 1 | no | — | exact 0.785082 / mad 14.689841 / ssim 0.680407 | measured |
| square | annotation | Annotation | 1 | 1 | 0 | 1 | 0 | 1 | no | — | exact 0.785082 / mad 14.689841 / ssim 0.680407 | measured |
| square | simple+lama | Simple | 1 | 1 | 0 | 1 | 0 | 1 | yes | 1 | exact 0.775077 / mad 5.828164 / ssim 0.871795 | measured |
| square | annotation+lama | Annotation | 1 | 1 | 0 | 1 | 0 | 1 | yes | 1 | exact 0.768236 / mad 5.998817 / ssim 0.868097 | measured |

## 4. Eligibility segments

### 4.1 Full population — every page, every cell

No eligibility conditioning applies to this segment, so a **cross-cell mean is permitted** here: every cell reports on every page, whether or not that page was eligible for that cell.

| Cell | Pages in segment | Mean failed regions | Mean eligible regions | Mean reference mean-abs-diff (lower = closer agreement) |
|---|---:|---:|---:|---:|
| simple | 7 | 0.857143 | 0.857143 | 17.979020 |
| annotation | 7 | 0.857143 | 0.857143 | 17.978748 |
| simple+lama | 7 | 0.857143 | 0.857143 | 13.199807 |
| annotation+lama | 7 | 0.857143 | 0.857143 | 12.652739 |

### 4.2 Per-cell eligible subsets

Each row is that cell's own `{page : eligible_regions > 0}`. These subsets are **not cross-comparable**: the eligibility predicate is mode-dependent, so two cells' rows here describe different populations and averaging across them would compare inpainted output against bare masking output under a caption claiming comparability.

| Cell | Pages in segment | Mean failed regions | Mean eligible regions | Mean reference mean-abs-diff (lower = closer agreement) |
|---|---:|---:|---:|---:|
| simple | 6 | 1.000000 | 1.000000 | 17.551859 |
| annotation | 6 | 1.000000 | 1.000000 | 17.551543 |
| simple+lama | 6 | 1.000000 | 1.000000 | 11.976111 |
| annotation+lama | 6 | 1.000000 | 1.000000 | 11.337865 |

### 4.3 Common-eligible intersection over (simple, annotation, simple+lama, annotation+lama)

The intersection of the eligible sets over the cells actually run in this invocation (simple, annotation, simple+lama, annotation+lama). This is the only *eligibility-restricted* segment a cross-cell mean may be computed over. A later run with a different cell set re-segments this table visibly, because the cell set is named in the heading above.

Pages in the intersection: black, darkrays, nightmare, ray, spikey, square

| Cell | Pages in segment | Mean failed regions | Mean eligible regions | Mean reference mean-abs-diff (lower = closer agreement) |
|---|---:|---:|---:|---:|
| simple | 6 | 1.000000 | 1.000000 | 17.551859 |
| annotation | 6 | 1.000000 | 1.000000 | 17.551543 |
| simple+lama | 6 | 1.000000 | 1.000000 | 11.976111 |
| annotation+lama | 6 | 1.000000 | 1.000000 | 11.337865 |

## 5. Reference comparison

The `Reference agreement (not quality)` column in section 3 reports agreement with the vendored reference, under all three of D5's binding conditions (§16.43 item 7):

- The reference is the vendored `demo_bubbles/*_clean.png`. Its producing PanelCleaner
  version, profile and environment are **unrecorded**, so §16.37 item 3's pins are
  unsatisfiable for the reference side; only the ours-side numbers here are reproducible.
- The reference column is headed **agreement with reference**, never "quality": a LaMa
  cell can be visually better while further from a non-inpainted reference.
- No pass/fail assertion is made against `_clean.png`, ever — §15.2 stays fully in force.

## 6. Verdict

**Non-gating.** This report ranks nothing and asserts nothing. It records what each cell measured on the same inputs so a maintainer can read the differences directly. Cells are not ordered by their distance to any reference image, and no cell is declared better or worse than another here.

This report does **not** close §16.38 item 17(b) decision point D2. That decision needs a
side-by-side run of upstream PanelCleaner and this port on a real page with a human
verdict under §15.10(a)'s independence rule; mode-bench compares this port's own cells to
each other and runs no upstream.
