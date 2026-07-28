#!/usr/bin/env python3
"""Empirically verify spec §10.3 step 2's `FIND_EDGES` reduction against real PIL.

§10.3 step 2 replaces PIL's `ImageFilter.FIND_EDGES` on a mode-`1` mask with the
closed form

    edge(p) = mask[p] == 1 && (p is on the 1-pixel image border
                               || any of the 8 neighbours of p is 0)

and §16.11-era item §16.9/21 corrected §10.7(A)4's "all 9" to "8" for a fully-set
3x3 mask **by hand reasoning**. This script checks both claims against the real
library rather than against the reasoning, over exhaustive small cases plus random
masks. It writes no fixture — `pc_mask::border` has no recorded-fixture dependency
— it only reports agreement, which is what makes the hand proof auditable.

Usage: verify_find_edges.py [iterations]
Exit code 0 iff PIL and the closed form agree on every case.
"""

import itertools
import json
import sys

import numpy as np
from PIL import Image, ImageFilter


def pil_edges(mask: np.ndarray) -> np.ndarray:
    """Truthy pixels of FIND_EDGES applied to the mode-`1` mask, as upstream does."""
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("1")
    filtered = np.asarray(img.filter(ImageFilter.FIND_EDGES).convert("L"))
    return filtered != 0


def closed_form_edges(mask: np.ndarray) -> np.ndarray:
    """spec §10.3 step 2, transcribed literally."""
    height, width = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for y in range(height):
        for x in range(width):
            if mask[y, x] != 1:
                continue
            on_border = x == 0 or y == 0 or x == width - 1 or y == height - 1
            if on_border:
                out[y, x] = True
                continue
            neighbourhood = mask[y - 1 : y + 2, x - 1 : x + 2]
            out[y, x] = bool((neighbourhood == 0).any())
    return out


def main(argv):
    iterations = int(argv[1]) if len(argv) > 1 else 500
    rng = np.random.default_rng(20260728)
    mismatches = []
    checked = 0

    # Exhaustive over every 3x3 mask (2**9 = 512) -- this is where §16.9 item 21's
    # correction lives, so it gets a complete rather than a sampled check.
    for bits in itertools.product([0, 1], repeat=9):
        mask = np.array(bits, dtype=np.uint8).reshape(3, 3)
        checked += 1
        if not np.array_equal(pil_edges(mask), closed_form_edges(mask)):
            mismatches.append(mask.tolist())

    full_3x3 = np.ones((3, 3), dtype=np.uint8)
    full_3x3_count = int(pil_edges(full_3x3).sum())

    # Random larger masks at several densities.
    for _ in range(iterations):
        height = int(rng.integers(1, 24))
        width = int(rng.integers(1, 24))
        density = float(rng.uniform(0.05, 0.95))
        mask = (rng.random((height, width)) < density).astype(np.uint8)
        checked += 1
        if not np.array_equal(pil_edges(mask), closed_form_edges(mask)):
            mismatches.append(mask.tolist())

    from PIL import __version__ as pil_version

    print(
        json.dumps(
            {
                "tool": "PIL.ImageFilter.FIND_EDGES",
                "pillow_version": pil_version,
                "cases_checked": checked,
                "mismatches": len(mismatches),
                "full_3x3_edge_count": full_3x3_count,
                "spec_10_7_A_4_expected": 8,
                "first_mismatch": mismatches[0] if mismatches else None,
            },
            indent=2,
        )
    )
    return 0 if not mismatches and full_3x3_count == 8 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
