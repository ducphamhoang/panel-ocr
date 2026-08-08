//! A4-d site gate for the ratified Annotation claim cleanup.
//!
//! Every row is a named `(path, line, substring)` site from §16.39 items 5(c),
//! 5(d), 5(d.i), or 5(g).  The line coordinate is intentional: a file-wide
//! `contains()` check can pass after a claim moves to an unrelated paragraph.

use pc_testkit::paths;
use std::fs;

#[derive(Clone, Copy)]
struct Site {
    path: &'static str,
    line: usize,
    text: &'static str,
}

const EXPECTED_ABSENT: usize = 32;
const EXPECTED_PRESENT: usize = 34;

const ABSENT: &[Site] = &[
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 499,
        text: "annotation\" rejected in v1",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 699,
        text: "only `Simple` is implemented in v1",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1475,
        text: "rejected by `pc-detect`",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1527,
        text: "can reject `MaskRefineMode::Annotation`",
    },
    Site {
        path: "crates/pc-config/src/default_profile.toml",
        line: 20,
        text: "annotation is rejected",
    },
    Site {
        path: "crates/pc-config/src/profile.rs",
        line: 208,
        text: "Only `Simple` is implemented",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 11,
        text: "still rejects",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 14,
        text: "still rejects",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 17,
        text: "still rejects",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 86,
        text: "Rejects `MaskRefineMode::Annotation`",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 90,
        text: "return Err(StageError::InvalidInput",
    },
    Site {
        path: "crates/pc-detect/src/detector.rs",
        line: 29,
        text: "so the stage can reject",
    },
    Site {
        path: "crates/pc-config/tests/defaults.rs",
        line: 113,
        text: "literal-§6-block",
    },
    Site {
        path: "crates/pc-config/tests/defaults.rs",
        line: 356,
        text: "pc-detect is what rejects",
    },
    Site {
        path: "crates/pc-config/tests/defaults.rs",
        line: 361,
        text: "pc-detect is what rejects",
    },
    Site {
        path: "crates/pc-detect/tests/d7_run.rs",
        line: 196,
        text: "run_rejects_annotation_refine_mode",
    },
    Site {
        path: "crates/pc-detect/src/mask.rs",
        line: 150,
        text: "DEVIATION(12): v1 ships",
    },
    Site {
        path: "README.md",
        line: 102,
        text: "[ ] `Mask RefineMode::Annotation`",
    },
    Site {
        path: "crates/pc-detect/src/annotate.rs",
        line: 7,
        text: "not wired into anything",
    },
    Site {
        path: "crates/pc-detect/src/annotate_merge.rs",
        line: 14,
        text: "not wired into anything",
    },
    Site {
        path: "crates/pc-detect/src/annotate_refine.rs",
        line: 7,
        text: "not wired into anything",
    },
    Site {
        path: "crates/pc-detect/tests/a1_annotate_topk.rs",
        line: 19,
        text: "is *not* Annotation mode",
    },
    Site {
        path: "crates/pc-detect/tests/a2_annotate_otsu.rs",
        line: 65,
        text: "is *not* Annotation mode",
    },
    Site {
        path: "crates/pc-detect/tests/a3_annotate_merge.rs",
        line: 60,
        text: "is *not* Annotation mode",
    },
    Site {
        path: "crates/pc-detect/src/annotate_merge.rs",
        line: 22,
        text: "DEVIATION(12) retirement",
    },
    Site {
        path: "crates/pc-detect/src/annotate_refine.rs",
        line: 10,
        text: "DEVIATION(12) retirement",
    },
    Site {
        path: "crates/pc-detect/tests/a3_annotate_merge.rs",
        line: 68,
        text: "DEVIATION(12) retirement",
    },
    Site {
        path: "docs/HANDOVER.md",
        line: 39,
        text: "still pending",
    },
    Site {
        path: "docs/DETECTOR_ORACLE.md",
        line: 52,
        text: "crates/pc-detect/src/mask.rs:150",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 2779,
        text: "pc-detect's `refine_simple`",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3798,
        text: "`MaskRefineMode::Annotation`; and",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3874,
        text: "**Annotation** last",
    },
];

const PRESENT: &[Site] = &[
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 11,
        text: "path wired through [`run`]",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 14,
        text: "used by the path wired through [`run`]",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 17,
        text: "used by the path wired through",
    },
    Site {
        path: "crates/pc-detect/src/annotate.rs",
        line: 7,
        text: "supplies Annotation primitives",
    },
    Site {
        path: "crates/pc-detect/src/annotate_merge.rs",
        line: 14,
        text: "supplies Annotation merge primitives",
    },
    Site {
        path: "crates/pc-detect/src/annotate_refine.rs",
        line: 7,
        text: "supplies Annotation page-level refinement",
    },
    Site {
        path: "crates/pc-detect/tests/a1_annotate_topk.rs",
        line: 19,
        text: "A1 covers Annotation primitives",
    },
    Site {
        path: "crates/pc-detect/tests/a2_annotate_otsu.rs",
        line: 65,
        text: "A2 covers Annotation primitives",
    },
    Site {
        path: "crates/pc-detect/tests/a3_annotate_merge.rs",
        line: 60,
        text: "A3 covers Annotation primitives",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 499,
        text: "Annotation is opt-in; Simple remains the default",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 699,
        text: "`Annotation` is the opt-in upstream refinement path",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1475,
        text: "accepted by config and selects the upstream refinement path",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1527,
        text: "stage selects the `MaskRefineMode::Annotation` refinement path",
    },
    Site {
        path: "crates/pc-config/src/default_profile.toml",
        line: 20,
        text: "Annotation is opt-in; Simple remains the default",
    },
    Site {
        path: "crates/pc-config/src/profile.rs",
        line: 208,
        text: "Simple remains the shipped default",
    },
    Site {
        path: "crates/pc-detect/src/lib.rs",
        line: 86,
        text: "Supports the `MaskRefineMode::Annotation` path",
    },
    Site {
        path: "crates/pc-detect/src/detector.rs",
        line: 29,
        text: "because the stage selects the refinement path",
    },
    Site {
        path: "crates/pc-config/tests/defaults.rs",
        line: 361,
        text: "config accepts annotation as an opt-in refinement mode",
    },
    Site {
        path: "crates/pc-detect/src/mask.rs",
        line: 172,
        text: "default-value divergence from",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 1442,
        text: "The door is open: A4 ships `Annotation`",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3780,
        text: "§16.39 later ratified and landed that opt-in path",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3792,
        text: "The 2026-07-29 plan recorded these as v1.5 scope",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3810,
        text: "Historical sequencing (2026-07-29)",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3829,
        text: "§16.39 subsequently landed the path",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 3874,
        text: "historical final step; §16.39 subsequently landed it",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 7666,
        text: "Upstream `refine_mask`/`refine_undetected_mask`",
    },
    Site {
        path: "README.md",
        line: 103,
        text: "### v2",
    },
    Site {
        path: "docs/HANDOVER.md",
        line: 39,
        text: "was removed under §16.39 item 3(d)",
    },
    Site {
        path: "docs/DETECTOR_ORACLE.md",
        line: 52,
        text: "DEVIATION(12)` at the default `Simple` variant in `crates/pc-config/src/profile.rs`",
    },
    Site {
        path: "docs/PIPELINE_SPEC_V1.md",
        line: 2779,
        text: "DEVIATION(12)` on the default `Simple` variant of `MaskRefineMode`",
    },
    Site {
        path: "crates/pc-config/src/profile.rs",
        line: 214,
        text: "DEVIATION(12): the shipped default is `Simple`",
    },
    Site {
        path: "crates/pc-detect/src/annotate_merge.rs",
        line: 21,
        text: "DEVIATION(12)` narrowing",
    },
    Site {
        path: "crates/pc-detect/src/annotate_refine.rs",
        line: 10,
        text: "DEVIATION(12)` narrowing",
    },
    Site {
        path: "crates/pc-detect/tests/a3_annotate_merge.rs",
        line: 67,
        text: "DEVIATION(12)` narrowing",
    },
];

fn line(site: Site) -> String {
    let path = paths::workspace_root().join(site.path);
    assert!(
        path.is_file(),
        "A4-d site path does not exist: {}",
        path.display()
    );
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .nth(site.line - 1)
        .unwrap_or("")
        .to_owned()
}

#[test]
fn a4d_claim_sites_match_the_ratified_cleanup() {
    assert_eq!(ABSENT.len(), EXPECTED_ABSENT);
    assert_eq!(PRESENT.len(), EXPECTED_PRESENT);
    for site in ABSENT {
        assert!(
            !line(*site).contains(site.text),
            "stale claim at {}:{}",
            site.path,
            site.line
        );
    }
    for site in PRESENT {
        assert!(
            line(*site).contains(site.text),
            "missing claim at {}:{}",
            site.path,
            site.line
        );
    }
}
