# Vendored upstream test fixtures — attribution

The files in this directory are **copied verbatim from PanelCleaner** and are not
original works of panel-ocr. They are vendored so that panel-ocr's parity and
calibration tests (spec §7.1) can run without a network fetch or a submodule.

## Source

| | |
|---|---|
| Project | PanelCleaner |
| Author | VoxelCubes and PanelCleaner contributors |
| Repository | https://github.com/VoxelCubes/PanelCleaner |
| Commit | `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3` |
| Commit date | 2026-07-16 |
| Commit subject | "Update flatpak metadata" |
| Upstream version at that commit | `pcleaner 2.11.11` (`pcleaner/__init__.py`) |
| License | GNU General Public License v3.0 (see the upstream `LICENSE`) |

## License

PanelCleaner is licensed under the **GNU General Public License, version 3**.
panel-ocr is likewise licensed under **GPL-3.0**, so these files are redistributed
here under the same terms. The full licence text is in this repository's root
`LICENSE`. No copyright notice, licence text, or attribution has been removed from
any copied file, and none of these files have been modified — they are byte-for-byte
copies of the upstream originals.

## Files and their upstream paths

| Vendored path (relative to this directory) | Upstream path |
|---|---|
| `demo_bubbles/black_bubble_raw.png` | `media/demo_bubbles/black_bubble_raw.png` |
| `demo_bubbles/black_bubble_clean.png` | `media/demo_bubbles/black_bubble_clean.png` |
| `demo_bubbles/darkrays_bubble_raw.png` | `media/demo_bubbles/darkrays_bubble_raw.png` |
| `demo_bubbles/darkrays_bubble_clean.png` | `media/demo_bubbles/darkrays_bubble_clean.png` |
| `demo_bubbles/handwritten_bubble_raw.png` | `media/demo_bubbles/handwritten_bubble_raw.png` |
| `demo_bubbles/handwritten_bubble_clean.png` | `media/demo_bubbles/handwritten_bubble_clean.png` |
| `demo_bubbles/nightmare_bubble_raw.png` | `media/demo_bubbles/nightmare_bubble_raw.png` |
| `demo_bubbles/nightmare_bubble_clean.png` | `media/demo_bubbles/nightmare_bubble_clean.png` |
| `demo_bubbles/ray_bubble_raw.png` | `media/demo_bubbles/ray_bubble_raw.png` |
| `demo_bubbles/ray_bubble_clean.png` | `media/demo_bubbles/ray_bubble_clean.png` |
| `demo_bubbles/spikey_bubble_raw.png` | `media/demo_bubbles/spikey_bubble_raw.png` |
| `demo_bubbles/spikey_bubble_clean.png` | `media/demo_bubbles/spikey_bubble_clean.png` |
| `demo_bubbles/square_bubble_raw.png` | `media/demo_bubbles/square_bubble_raw.png` |
| `demo_bubbles/square_bubble_clean.png` | `media/demo_bubbles/square_bubble_clean.png` |
| `long_strip.jpg` | `tests/mock_files/long_strip.jpg` |
| `ocr_output/good_detected_text.csv` | `tests/mock_files/ocr_output/good_detected_text.csv` |
| `ocr_output/good_detected_text.txt` | `tests/mock_files/ocr_output/good_detected_text.txt` |

## Measured properties (spec §7.1 — use as test constants)

All `demo_bubbles` images are 8-bit **grayscale** PNGs, and each `raw`/`clean` pair
shares one size:

| fixture | size |
|---|---|
| `black` | 202 x 319 |
| `darkrays` | 208 x 320 |
| `handwritten` | 72 x 132 |
| `nightmare` | 219 x 343 |
| `ray` | 256 x 329 |
| `spikey` | 354 x 354 |
| `square` | 144 x 270 |

`long_strip.jpg` is 1000 x 8000 RGB, progressive JPEG, 300 dpi.

## Status of the `*_clean.png` images — read before writing a test against them

The `demo_bubbles` files live in upstream's `media/` directory: they are **README
demo assets**, not part of upstream's own test suite, and the PanelCleaner version
and profile that produced each `*_clean.png` are not recorded anywhere upstream.

Per spec §15.2 (decided by Fable) they are therefore **calibration fixtures, not
frozen gates**: §10.7(B) item 15 compares against them and records the numbers in
`docs/GOLDEN_CALIBRATION.md` without gating CI. Do not add a pass/fail assertion
against a `*_clean.png` image. The one frozen assertion that touches this data is
§10.7(B) item 16 (`black_bubble`'s chosen fill colour must be dark and must not hit
the off-white snap), which is independent of mask-refinement mode.

## Re-vendoring

If these fixtures are ever refreshed from a newer upstream commit, update the commit
hash, date, and version in the table above in the same change, and re-run
`cargo xtask calibrate-goldens` (task F2) so `docs/GOLDEN_CALIBRATION.md` still
describes the files actually present.

---

## `oracle_pages/` — Pepper&Carrot (F1 oracle candidate pages, §16.24 item 17)

These three files are **not** from PanelCleaner. They are the real manga-style pages
the F1 recording run (task #12/#13) compares our detector output against upstream's,
per §7.2's "one or two full manga pages the maintainer supplies, license-clean" and
§16.24 item 17's ratification.

| | |
|---|---|
| Work | *Pepper&Carrot*, episode 1, "Potion of Flight", Japanese translation |
| Author | David Revoy and the Pepper&Carrot translation contributors |
| Source | https://www.peppercarrot.com/0_sources/ep01_Potion-of-Flight/low-res/ |
| License | **CC-BY 4.0** — redistribution permitted with attribution |
| Retrieved | 2026-07-30 |
| Modified | No — byte-for-byte as served |

| Vendored file | Size | sha256 | Status |
|---|---|---|---|
| `oracle_pages/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg` | 441,914 B | `3bef9922e09cea66ab12271da0070025768ae9bc5d286f41ced617468131267e` | **P01 — the ratified oracle page** (§16.24 item 17; exact hash match) |
| `oracle_pages/ja_Pepper-and-Carrot_by-David-Revoy_E01P02.jpg` | 448,475 B | `d7ba528d2ab96c7fb0723197d2349d2981a0b0dd853b16179e1a06cbdd5064d3` | P02 — rejected sibling (item 17(a)); size matches the recorded figure, no sha256 was ever ratified for it |
| `oracle_pages/ja_Pepper-and-Carrot_by-David-Revoy_E01P03.jpg` | 302,541 B | `c62ff81bdac3c979959c55d4cd4dea01659a35097d7cc5723c57c54531694e23` | P03 — rejected sibling (item 17(a)); size matches the recorded figure, no sha256 was ever ratified for it |

P02/P03 are vendored (not just P01) because §16.27 item 1(f)'s `DbnetScattered` census is
measured over all three pages, not P01 alone (see `docs/PIPELINE_SPEC_V1.md` §16.27 item 1(f)
and item 3). P01 is the only one of the three with a sha256 ratified anywhere in the spec;
P02/P03's identity rests on the (weaker) size match recorded at item 17(a) — this table does
not claim their sha256 values are ratified, only that they are what this session fetched and
hashed on 2026-07-30, recorded here per this file's own convention of hash-verifiable
vendored fixtures. P04 (a 1200x24 footer strip, not a page — item 17(a)) is not vendored.

---

## `crates/pc-ocr/assets/vocab.txt` (manga-ocr vocabulary, §16.30 item 2)

Not from PanelCleaner or this directory — recorded here anyway per §16.30 item 2's own
instruction: "Whatever path is used, it should follow the existing
`tests/fixtures/upstream/ATTRIBUTION.md` convention... attribution is owed regardless of
which directory the file lands in." This entry is that attribution; the file itself lives
under `crates/pc-ocr/assets/`, not under this directory, because it is source the `pc-ocr`
crate embeds (`include_str!`), not a test fixture read at test time.

| | |
|---|---|
| Project | manga-ocr |
| Author | Maciej Budyś (`kha-white`) |
| Repository | https://huggingface.co/kha-white/manga-ocr-base |
| Commit | `aa6573bd10b0d446cbf622e29c3e084914df9741` |
| License | Apache License 2.0 — full text at `crates/pc-ocr/assets/LICENSE-APACHE-2.0.txt` |
| Vendored file | `crates/pc-ocr/assets/vocab.txt` |
| Upstream path | `vocab.txt` (repository root) |
| Size | 24,072 bytes |
| sha256 | `344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d` |
| Retrieved | 2026-08-02 |
| Modified | No — byte-for-byte as served (LF line endings in the original; a `.gitattributes`
  entry marks the vendored copy `-text` so eol normalization cannot alter it on checkout) |

**Note on the license file's own provenance.** `kha-white/manga-ocr-base`'s HuggingFace repo
declares `license: apache-2.0` in its README front matter but does not itself vendor a
`LICENSE` file (confirmed via the HF API's file listing at the pinned commit — the repo
holds only `.gitattributes`, `README.md`, `config.json`, `preprocessor_config.json`,
`pytorch_model.bin`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt`). So
`crates/pc-ocr/assets/LICENSE-APACHE-2.0.txt` is the canonical Apache License 2.0 text
fetched from `https://www.apache.org/licenses/LICENSE-2.0.txt` (sha256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`, 11,358 bytes) — the
standard, project-independent text the license declaration refers to — not a file copied
from upstream's own repository, because no such file exists there to copy.

Chosen over `mayocream/manga-ocr-onnx`'s copy of the same file (30,216 bytes, CRLF-terminated)
per Fable's ruling — see `docs/PIPELINE_SPEC_V1.md` §16.30 item 2 and `docs/RULINGS.md`'s
"P7 manga-ocr artifact, vocab, task split" entry for the full reasoning and the CRLF-hazard
arithmetic (30,216 = 24,072 + 6,144 line endings).
