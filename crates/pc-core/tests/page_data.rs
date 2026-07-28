//! C2 tests — spec §2.4 `PageDataRaw`, §2.5 `PageData` + invariants, §2.6 `MaskData`,
//! §2.7 analytics.

use image::{DynamicImage, RgbImage};
use pc_core::{
    DenoiseAnalytic, DetectAnalytic, DetectedBlock, ImageHandle, Language, MaskData,
    MaskFittingAnalytic, MaskRegionStats, MaskingRegion, OcrAnalytic, PageData, PageDataRaw, Rect,
    RemovedBox, StageError, TextBox, SCHEMA_VERSION,
};
use std::path::PathBuf;

const CANVAS: (u32, u32) = (100, 200);

fn handle(name: &str) -> ImageHandle {
    ImageHandle::from_path(format!("/cache/{name}"))
}

/// A minimal, invariant-satisfying `PageData`: one tight box, one extended box, one
/// masking region whose reference contains its masking rect. Everything inside CANVAS.
fn valid_page_data() -> PageData {
    PageData {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from("/in/page.png"),
        base_image: handle("u_page_base.png"),
        raw_mask: handle("u_page_raw_mask.png"),
        scale: 1.0,
        image_size: CANVAS,
        page_language: Some(Language::Japanese),
        text_boxes: vec![TextBox {
            rect: Rect::new(20, 20, 40, 60),
            language: Some(Language::Japanese),
        }],
        extended_boxes: vec![Rect::new(15, 15, 45, 65)],
        masking_regions: vec![MaskingRegion {
            masking: Rect::new(15, 15, 45, 65),
            reference: Rect::new(0, 0, 65, 85),
        }],
    }
}

// ---------------------------------------------------------------- §2.5 invariants

#[test]
// spec §2.5: a well-formed PageData satisfies all three invariants
fn valid_page_data_passes_validation() {
    valid_page_data()
        .validate()
        .expect("baseline fixture must be valid");
}

#[test]
// spec §2.5 invariant 1: extended_boxes.len() == text_boxes.len() — one extended
// box per tight box, which is what makes the index pairing meaningful
fn extended_boxes_must_be_one_per_text_box() {
    let mut p = valid_page_data();
    p.extended_boxes.push(Rect::new(50, 50, 60, 60));
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));

    let mut p = valid_page_data();
    p.extended_boxes.clear();
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));

    // zero boxes on both sides is legal (a page with no detections)
    let mut p = valid_page_data();
    p.text_boxes.clear();
    p.extended_boxes.clear();
    p.masking_regions.clear();
    p.validate().expect("an empty page is valid");
}

#[test]
// spec §2.5 invariant 2: every MaskingRegion's `reference` contains its `masking` —
// §10.3 step 1 computes non-negative offsets from that containment, so a violation
// would produce a negative crop offset downstream
fn reference_must_contain_masking() {
    // reference smaller than masking on the right
    let mut p = valid_page_data();
    p.masking_regions[0].reference = Rect::new(0, 0, 44, 85);
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));

    // reference offset such that x1 is inside masking's x1
    let mut p = valid_page_data();
    p.masking_regions[0].reference = Rect::new(20, 0, 65, 85);
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));

    // coincident rects are contained (padding of 0 is legal)
    let mut p = valid_page_data();
    p.masking_regions[0].reference = p.masking_regions[0].masking;
    p.validate().expect("reference == masking is contained");
}

#[test]
// spec §2.5 invariant 3: all rects lie within image_size. "Within" means
// x1 >= 0, y1 >= 0, x2 <= width, y2 <= height — x2/y2 are exclusive for cropping,
// and pad() clamps them to exactly the canvas extent, so flush is legal.
fn all_rects_must_lie_within_image_size() {
    // flush against the canvas is fine
    let mut p = valid_page_data();
    p.text_boxes[0].rect = Rect::new(0, 0, 100, 200);
    p.extended_boxes[0] = Rect::new(0, 0, 100, 200);
    p.masking_regions[0] = MaskingRegion {
        masking: Rect::new(0, 0, 100, 200),
        reference: Rect::new(0, 0, 100, 200),
    };
    p.validate()
        .expect("canvas-flush rects are within image_size");

    // one pixel past the right edge is not
    let mut p = valid_page_data();
    p.text_boxes[0].rect = Rect::new(0, 0, 101, 10);
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));
    assert!(p
        .validate()
        .expect_err("a right-edge overflow must be rejected")
        .to_string()
        .contains("all page rects must lie within image_size"));

    // negative origin is not
    let mut p = valid_page_data();
    p.text_boxes[0].rect = Rect::new(-1, 0, 10, 10);
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));
    assert!(p
        .validate()
        .expect_err("a negative origin must be rejected")
        .to_string()
        .contains("all page rects must lie within image_size"));

    // the check covers extended_boxes too, not only text_boxes
    let mut p = valid_page_data();
    p.extended_boxes[0] = Rect::new(0, 0, 10, 201);
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));
    assert!(p
        .validate()
        .expect_err("an extended-box overflow must be rejected")
        .to_string()
        .contains("all page rects must lie within image_size"));

    // ...and both rects of every masking region
    let mut p = valid_page_data();
    p.masking_regions[0].masking = Rect::new(0, 0, 10, 201);
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));
    assert!(p
        .validate()
        .expect_err("a masking-rect overflow must be rejected")
        .to_string()
        .contains("every masking region reference must contain its masking rect"));
    let mut p = valid_page_data();
    p.masking_regions[0].reference = Rect::new(0, 0, 10, 201);
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));
    assert!(p
        .validate()
        .expect_err("a reference-rect overflow must be rejected")
        .to_string()
        .contains("every masking region reference must contain its masking rect"));

    // The masking rect is contained by its reference, so invariant 3 is the
    // first failing invariant even though both masking rects exceed image_size.
    let mut p = valid_page_data();
    p.masking_regions[0] = MaskingRegion {
        masking: Rect::new(0, 0, 10, 201),
        reference: Rect::new(0, 0, 10, 205),
    };
    assert!(matches!(p.validate(), Err(StageError::InvalidInput(_))));
    assert!(p
        .validate()
        .expect_err("contained masking rects beyond image_size must be rejected")
        .to_string()
        .contains("all page rects must lie within image_size"));
}

#[test]
// spec §2.4: PageDataRaw's blocks are in base_image coordinates and must lie within
// image_size
fn page_data_raw_validates_block_rects() {
    let mut raw = valid_page_data_raw();
    raw.validate().expect("baseline raw fixture must be valid");

    raw.blocks[0].rect = Rect::new(0, 0, 101, 10);
    assert!(matches!(raw.validate(), Err(StageError::InvalidInput(_))));
}

// ---------------------------------------------------------------- serde contracts

fn valid_page_data_raw() -> PageDataRaw {
    PageDataRaw {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from("/in/page.png"),
        base_image: handle("u_page_base.png"),
        raw_mask: handle("u_page_raw_mask.png"),
        scale: 0.5,
        image_size: CANVAS,
        blocks: vec![DetectedBlock {
            rect: Rect::new(10, 10, 30, 30),
            language: Some(Language::Japanese),
            confidence: 0.875,
            mask_coverage: 0.5,
        }],
    }
}

#[test]
// spec §2: schema_version is the FIRST field of every persisted struct and starts
// at 1, so a future format change is detectable by reading the head of the JSON
fn schema_version_is_first_and_is_one() {
    assert_eq!(SCHEMA_VERSION, 1);
    for json in [
        serde_json::to_string(&valid_page_data_raw()).unwrap(),
        serde_json::to_string(&valid_page_data()).unwrap(),
        serde_json::to_string(&valid_mask_data()).unwrap(),
    ] {
        assert!(
            json.starts_with(r#"{"schema_version":1,"#),
            "schema_version must be first: {json}"
        );
    }
}

#[test]
// spec §2.4: PageDataRaw is the #raw.json contract — round-tripping it must be
// lossless. Compared as canonical JSON because ImageHandle has no derived PartialEq
// (it compares path only, see §2.3).
fn page_data_raw_round_trips_losslessly() {
    let raw = valid_page_data_raw();
    let json = serde_json::to_string(&raw).unwrap();
    let back: PageDataRaw = serde_json::from_str(&json).unwrap();
    assert_eq!(serde_json::to_string(&back).unwrap(), json);
    assert_eq!(back.blocks, raw.blocks);
    assert_eq!(back.scale, raw.scale);
    assert_eq!(back.image_size, raw.image_size);
    assert_eq!(back.original_path, raw.original_path);
    assert_eq!(back.base_image.path, raw.base_image.path);
    assert_eq!(back.raw_mask.path, raw.raw_mask.path);
}

#[test]
// spec §2.5: PageData is the #clean.json contract, and a deserialized checkpoint
// must still satisfy the §2.5 invariants
fn page_data_round_trips_and_stays_valid() {
    let p = valid_page_data();
    let json = serde_json::to_string(&p).unwrap();
    let back: PageData = serde_json::from_str(&json).unwrap();
    assert_eq!(serde_json::to_string(&back).unwrap(), json);
    assert_eq!(back.text_boxes, p.text_boxes);
    assert_eq!(back.extended_boxes, p.extended_boxes);
    assert_eq!(back.masking_regions, p.masking_regions);
    assert_eq!(back.page_language, Some(Language::Japanese));
    back.validate()
        .expect("a round-tripped checkpoint must still be valid");
}

#[test]
// spec §2.5: an unknown page_language is null, not a sentinel string
fn page_language_none_serializes_as_null() {
    let mut p = valid_page_data();
    p.page_language = None;
    p.text_boxes[0].language = None;
    let json = serde_json::to_string(&p).unwrap();
    assert!(json.contains(r#""page_language":null"#));
    let back: PageData = serde_json::from_str(&json).unwrap();
    assert_eq!(back.page_language, None);
    assert_eq!(back.text_boxes[0].language, None);
}

#[test]
// spec §2.5: image_size is a (u32,u32) pair, serialized as a two-element array
fn image_size_is_a_json_pair() {
    let json = serde_json::to_string(&valid_page_data()).unwrap();
    assert!(json.contains(r#""image_size":[100,200]"#), "{json}");
}

#[test]
// spec §2.3 + §2.5: serializing a PageData that still holds an in-memory-only
// handle must fail rather than write an unusable checkpoint
fn page_data_with_unmaterialized_handle_cannot_be_checkpointed() {
    let mut p = valid_page_data();
    p.base_image = ImageHandle::from_memory(DynamicImage::ImageRgb8(RgbImage::new(2, 2)));
    assert!(
        serde_json::to_string(&p).is_err(),
        "a checkpoint must not reference a path-less handle (§2.3)"
    );
}

// ---------------------------------------------------------------- §2.6 MaskData

fn valid_mask_data() -> MaskData {
    MaskData {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from("/in/page.png"),
        base_image: handle("u_page_base.png"),
        combined_mask: handle("u_page_combined_mask.png"),
        scale: 1.0,
        regions: vec![
            MaskRegionStats {
                rect: Rect::new(15, 15, 45, 65),
                std_deviation: 3.25,
                failed: false,
                thickness: Some(8),
            },
            MaskRegionStats {
                rect: Rect::new(50, 50, 70, 90),
                std_deviation: 18.0,
                failed: true,
                thickness: None,
            },
        ],
    }
}

#[test]
// spec §2.6: MaskData is the #mask_data.json contract; `thickness: None` (the box
// mask was chosen) serializes as null, and failed regions are RETAINED in the list
fn mask_data_round_trips_including_failures_and_null_thickness() {
    let m = valid_mask_data();
    let json = serde_json::to_string(&m).unwrap();
    assert!(json.contains(r#""thickness":null"#), "{json}");
    let back: MaskData = serde_json::from_str(&json).unwrap();
    assert_eq!(serde_json::to_string(&back).unwrap(), json);
    assert_eq!(back.regions, m.regions);
    assert_eq!(
        back.regions.len(),
        2,
        "failed regions stay in MaskData (§2.6)"
    );
    assert!(back.regions[1].failed);
    assert_eq!(back.regions[1].thickness, None);
}

#[test]
// spec §2.6: std_deviation is f64 and must survive a JSON round-trip bit-exactly —
// §11.3 step 3 compares it against noise_min_standard_deviation with strict >, so a
// lost low-order bit could change which regions get denoised
fn mask_data_std_deviation_survives_json_bit_exactly() {
    let mut m = valid_mask_data();
    m.regions[0].std_deviation = 0.2500000000000001;
    let back: MaskData = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back.regions[0].std_deviation, 0.2500000000000001);
    assert!(back.regions[0].std_deviation > 0.25);
}

// ---------------------------------------------------------------- §2.7 analytics

#[test]
// spec §2.7 + §9.7(A)9: OcrAnalytic records removed boxes in ORIGINAL image
// coordinates; areas are i64 to match Rect::area()
fn ocr_analytic_round_trips() {
    let a = OcrAnalytic {
        path: PathBuf::from("/in/img1.jpg"),
        num_boxes: 4,
        box_areas_ocred: vec![2_800, 20_000],
        box_areas_removed: vec![2_800],
        removed: vec![RemovedBox {
            text: "！？".into(),
            // (10,10,30,30) at page scale 0.5 -> original coordinates
            rect: Rect::new(10, 10, 30, 30).scale(1.0 / 0.5),
        }],
    };
    assert_eq!(a.removed[0].rect, Rect::new(20, 20, 60, 60));
    let json = serde_json::to_string(&a).unwrap();
    let back: OcrAnalytic = serde_json::from_str(&json).unwrap();
    assert_eq!(back, a);
    // non-ASCII engine output must survive verbatim
    assert_eq!(back.removed[0].text, "！？");
}

#[test]
// spec §2.7: an OcrAnalytic with no small-enough candidates still reports the
// pre-removal box count with empty vectors (§9.3 step 7)
fn ocr_analytic_empty_case() {
    let a = OcrAnalytic {
        path: PathBuf::from("/in/img1.jpg"),
        num_boxes: 7,
        box_areas_ocred: vec![],
        box_areas_removed: vec![],
        removed: vec![],
    };
    let back: OcrAnalytic = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
    assert_eq!(back, a);
    assert_eq!(back.num_boxes, 7);
}

#[test]
// spec §2.7: the remaining analytics records round-trip, including the
// "no fit found" shape (§10.7(A)9) and the all-regions std_deviations list (§11.3 step 6)
fn other_analytics_round_trip() {
    let m = MaskFittingAnalytic {
        path: PathBuf::from("/in/page.png"),
        fit_found: false,
        candidate_index: 1,
        std_deviation: 18.0,
        thickness: None,
    };
    assert_eq!(
        serde_json::from_str::<MaskFittingAnalytic>(&serde_json::to_string(&m).unwrap()).unwrap(),
        m
    );

    let d = DenoiseAnalytic {
        path: PathBuf::from("/in/page.png"),
        std_deviations: vec![0.25, 0.26, 20.0],
        boxes_denoised: 1,
    };
    assert_eq!(
        serde_json::from_str::<DenoiseAnalytic>(&serde_json::to_string(&d).unwrap()).unwrap(),
        d
    );

    let t = DetectAnalytic {
        path: PathBuf::from("/in/page.png"),
        blocks_detected: 9,
        blocks_kept: 7,
    };
    assert_eq!(
        serde_json::from_str::<DetectAnalytic>(&serde_json::to_string(&t).unwrap()).unwrap(),
        t
    );
}
