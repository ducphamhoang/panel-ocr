//! Task D5 -- spec §8.3 step 4, §8.7(A)3, §14.13/§15.1, §16.6 item 4. FROZEN.
//!
//! Every fixture value is a **dyadic rational** (1/2, 3/4, 7/8, 15/16, 13/32, 1.0) so
//! products and comparisons are exact in f32 and no test can go flaky on rounding. The
//! two thresholds (0.4) are the only non-dyadic numbers, and no fixture score is ever
//! placed within 1/64 of them.

use pc_core::Language;
use pc_detect::yolo::{
    class_to_language, filter_candidates, iou, nms, postprocess, rescale, round3, Candidate,
    LetterboxGeometry, CLASS_SCORE_THRESHOLD, MAX_DET, NMS_IOU_THRESHOLD, N_CLASSES,
    OBJECTNESS_THRESHOLD, ROW_STRIDE,
};

/// One `[cx, cy, w, h, objectness, c0, c1]` row.
fn row(cx: f32, cy: f32, w: f32, h: f32, objectness: f32, classes: [f32; N_CLASSES]) -> Vec<f32> {
    let mut values = vec![cx, cy, w, h, objectness];
    values.extend_from_slice(&classes);
    values
}

fn rows(items: &[Vec<f32>]) -> Vec<f32> {
    items.iter().flatten().copied().collect()
}

/// spec §8.7(A)3 / §14.13, as a single 6-candidate fixture:
///   A  keep      obj 7/8  * c0 15/16 = 0.8203125
///   B  suppress  obj 3/4  * c1 15/16 = 0.703125    (IoU(A,B) = 0.8824 -- different class!)
///   E  keep      obj 3/4  * c1 7/8   = 0.65625     (IoU(A,E) = 0.1428, below 0.35)
///   F  suppress  obj 1/2  * c1 15/16 = 0.46875     (IoU(E,F) = 0.8824)
///   C  dropped   obj 1/2  * c1 3/4   = 0.375       (fails the class-score gate)
///   D  dropped   obj 3/8  = 0.375                  (fails the objectness gate)
fn a3_fixture() -> Vec<f32> {
    rows(&[
        row(100.0, 100.0, 64.0, 64.0, 0.875, [0.9375, 0.0]), // A -> (68,68,132,132)
        row(104.0, 100.0, 64.0, 64.0, 0.75, [0.0, 0.9375]),  // B -> (72,68,136,132)
        row(532.0, 532.0, 64.0, 64.0, 0.5, [0.0, 0.75]),     // C: score gate
        row(632.0, 632.0, 64.0, 64.0, 0.375, [0.0, 1.0]),    // D: objectness gate
        row(148.0, 100.0, 64.0, 64.0, 0.75, [0.0, 0.875]),   // E -> (116,68,180,132)
        row(152.0, 100.0, 64.0, 64.0, 0.5, [0.0, 0.9375]),   // F -> (120,68,184,132)
    ])
}

// ------------------------------------------------------------ constants

#[test]
fn constants_match_the_spec() {
    assert_eq!(OBJECTNESS_THRESHOLD, 0.4);
    assert_eq!(CLASS_SCORE_THRESHOLD, 0.4);
    assert_eq!(NMS_IOU_THRESHOLD, 0.35);
    assert_eq!(MAX_DET, 300);
    assert_eq!(N_CLASSES, 2);
    assert_eq!(ROW_STRIDE, 7);
}

// ------------------------------------------------------------ gates + decode

#[test]
fn filter_candidates_decodes_xywh_to_xyxy() {
    let candidates = filter_candidates(&row(100.0, 100.0, 64.0, 64.0, 0.875, [0.9375, 0.0]));

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].xyxy, [68.0, 68.0, 132.0, 132.0]);
    assert_eq!(candidates[0].class_index, 0);
    assert_eq!(candidates[0].score, 0.8203125);
}

#[test]
fn objectness_gate_is_strictly_greater() {
    // 0.375 < 0.4 -> dropped; 0.40625 > 0.4 -> kept (its class score also clears 0.4).
    let dropped = filter_candidates(&row(10.0, 10.0, 4.0, 4.0, 0.375, [1.0, 0.0]));
    let kept = filter_candidates(&row(10.0, 10.0, 4.0, 4.0, 0.40625, [1.0, 0.0]));

    assert!(dropped.is_empty());
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].score, 0.40625);
}

#[test]
fn class_score_gate_is_a_second_separate_strictly_greater_filter() {
    // spec §8.3 step 4: objectness clears 0.4 but objectness*class does not.
    let dropped = filter_candidates(&row(10.0, 10.0, 4.0, 4.0, 0.5, [0.0, 0.75]));
    let kept = filter_candidates(&row(10.0, 10.0, 4.0, 4.0, 0.5, [0.0, 0.875]));

    assert!(dropped.is_empty(), "0.5 * 0.75 = 0.375 <= 0.4");
    assert_eq!(kept.len(), 1, "0.5 * 0.875 = 0.4375 > 0.4");
    assert_eq!(kept[0].class_index, 1);
}

#[test]
fn best_class_wins_and_sets_the_class_index() {
    let candidates = filter_candidates(&row(10.0, 10.0, 4.0, 4.0, 1.0, [0.5, 0.9375]));

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].class_index, 1);
    assert_eq!(candidates[0].score, 0.9375);
}

#[test]
fn filter_candidates_preserves_input_order() {
    let candidates = filter_candidates(&rows(&[
        row(10.0, 10.0, 4.0, 4.0, 1.0, [0.5, 0.0]),
        row(20.0, 20.0, 4.0, 4.0, 1.0, [0.0, 0.9375]),
    ]));

    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].xyxy[0], 8.0);
    assert_eq!(candidates[1].xyxy[0], 18.0);
}

// ------------------------------------------------------------ IoU

#[test]
fn iou_of_identical_boxes_is_one_and_of_disjoint_boxes_is_zero() {
    let a = [0.0, 0.0, 4.0, 4.0];
    let b = [8.0, 8.0, 12.0, 12.0];

    assert_eq!(iou(&a, &a), 1.0);
    assert_eq!(iou(&a, &b), 0.0);
}

#[test]
fn iou_is_exact_for_a_dyadic_half_overlap() {
    // inter = 4*2 = 8, union = 16 + 8 - 8 = 16 -> 0.5
    let a = [0.0, 0.0, 4.0, 4.0];
    let b = [0.0, 0.0, 4.0, 2.0];

    assert_eq!(iou(&a, &b), 0.5);
    assert_eq!(iou(&b, &a), 0.5);
}

#[test]
fn iou_of_degenerate_boxes_is_zero_not_nan() {
    let empty = [1.0, 1.0, 1.0, 1.0];

    assert_eq!(iou(&empty, &empty), 0.0);
}

// ------------------------------------------------------------ NMS

#[test]
fn a3_class_agnostic_nms_with_both_confidence_gates() {
    // spec §8.7(A)3 + DEVIATION(13)/§14.13: B is suppressed by A even though they carry
    // *different* class indices -- that is precisely what class-agnostic means, and is
    // the duplicate-balloon problem this deviation exists to remove.
    let candidates = filter_candidates(&a3_fixture());

    assert_eq!(candidates.len(), 4, "C and D fail the two confidence gates");

    let survivors = nms(candidates);

    assert_eq!(survivors.len(), 2);
    assert_eq!(survivors[0].xyxy, [68.0, 68.0, 132.0, 132.0]);
    assert_eq!(survivors[0].class_index, 0);
    assert_eq!(survivors[0].score, 0.8203125);
    assert_eq!(survivors[1].xyxy, [116.0, 68.0, 180.0, 132.0]);
    assert_eq!(survivors[1].class_index, 1);
    assert_eq!(survivors[1].score, 0.65625);
}

#[test]
fn nms_emits_survivors_in_descending_score_order() {
    let survivors = nms(vec![
        Candidate {
            xyxy: [0.0, 0.0, 4.0, 4.0],
            class_index: 0,
            score: 0.5,
        },
        Candidate {
            xyxy: [100.0, 100.0, 104.0, 104.0],
            class_index: 1,
            score: 0.9375,
        },
        Candidate {
            xyxy: [200.0, 200.0, 204.0, 204.0],
            class_index: 2,
            score: 0.75,
        },
    ]);

    assert_eq!(
        survivors.iter().map(|c| c.score).collect::<Vec<_>>(),
        vec![0.9375, 0.75, 0.5]
    );
}

#[test]
fn nms_keeps_overlaps_at_or_below_the_iou_threshold() {
    // inter = 2*8 = 16, union = 64 + 64 - 16 = 112 -> IoU = 0.142857 < 0.35 -> both kept.
    let survivors = nms(vec![
        Candidate {
            xyxy: [0.0, 0.0, 8.0, 8.0],
            class_index: 0,
            score: 0.9375,
        },
        Candidate {
            xyxy: [6.0, 0.0, 14.0, 8.0],
            class_index: 0,
            score: 0.75,
        },
    ]);

    assert_eq!(survivors.len(), 2);
}

#[test]
fn nms_suppresses_overlaps_above_the_iou_threshold() {
    // inter = 6*8 = 48, union = 64 + 64 - 48 = 80 -> IoU = 0.6 > 0.35.
    let survivors = nms(vec![
        Candidate {
            xyxy: [0.0, 0.0, 8.0, 8.0],
            class_index: 0,
            score: 0.9375,
        },
        Candidate {
            xyxy: [2.0, 0.0, 10.0, 8.0],
            class_index: 1,
            score: 0.75,
        },
    ]);

    assert_eq!(survivors.len(), 1);
    assert_eq!(survivors[0].score, 0.9375);
}

#[test]
fn nms_caps_survivors_at_max_det() {
    // 400 pairwise-disjoint boxes: nothing suppresses anything, so only the cap can
    // reduce the count (spec §8.3 step 4, upstream's post-NMS `max_det`).
    let candidates = (0..400_u32)
        .map(|index| {
            let x = (index % 20) as f32 * 32.0;
            let y = (index / 20) as f32 * 32.0;
            Candidate {
                xyxy: [x, y, x + 16.0, y + 16.0],
                class_index: 0,
                // Distinct dyadic scores, all above both gates, descending with index.
                score: 0.5 + (400 - index) as f32 / 2048.0,
            }
        })
        .collect::<Vec<_>>();

    assert_eq!(nms(candidates).len(), MAX_DET);
}

// ------------------------------------------------------------ rescale + clip

/// A 512x256 image letterboxed into 1024x1024: r = min(1024/512, 1024/256) = 2, so the
/// resized content is 1024x512 and `dh = 512`, `dw = 0`. `resize_ratio` is 1/r = 0.5.
fn geometry() -> LetterboxGeometry {
    LetterboxGeometry {
        net_size: 1024,
        dw: 0.0,
        dh: 512.0,
        image_size: (512, 256),
    }
}

#[test]
fn resize_ratio_is_the_inverse_letterbox_scale_on_both_axes() {
    assert_eq!(geometry().resize_ratio(), 0.5);
}

#[test]
fn rescale_maps_letterboxed_coordinates_back_to_the_base_image() {
    let blocks = rescale(
        &[Candidate {
            xyxy: [100.0, 100.0, 200.0, 200.0],
            class_index: 1,
            score: 0.8203125,
        }],
        &geometry(),
    );

    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].rect, pc_core::Rect::new(50, 50, 100, 100));
    assert_eq!(blocks[0].class_index, 1);
}

#[test]
fn rescale_truncates_toward_zero_before_clipping() {
    // 101 * 0.5 = 50.5 -> 50 (upstream `astype(np.int32)`), not 51.
    let blocks = rescale(
        &[Candidate {
            xyxy: [101.0, 101.0, 203.0, 203.0],
            class_index: 0,
            score: 0.5,
        }],
        &geometry(),
    );

    assert_eq!(blocks[0].rect, pc_core::Rect::new(50, 50, 101, 101));
}

#[test]
fn rescale_clips_boxes_that_overhang_image_bounds() {
    // spec §16.6 item 4 (resolved: clip TO bounds). NOT a port of an upstream mechanism: this
    // comment cited `clip_coords`, which does not exist in the pinned upstream checkout; upstream's
    // only clamping is the IoU-intersection arithmetic at `yolov5_utils.py:166,169`. The clamp is a
    // deliberate v1 divergence — §14 register entry 18 / §16.27 item 9.
    // Raw: (-20, -8, 2100, 1100) * 0.5 = (-10, -4, 1050, 550) -> clipped to the 512x256
    // base image.
    let blocks = rescale(
        &[Candidate {
            xyxy: [-20.0, -8.0, 2100.0, 1100.0],
            class_index: 0,
            score: 0.75,
        }],
        &geometry(),
    );

    assert_eq!(blocks[0].rect, pc_core::Rect::new(0, 0, 512, 256));
}

#[test]
fn rescale_rounds_confidence_to_three_decimals() {
    let blocks = rescale(
        &[Candidate {
            xyxy: [0.0, 0.0, 8.0, 8.0],
            class_index: 0,
            score: 0.8203125,
        }],
        &geometry(),
    );

    assert_eq!(blocks[0].confidence, 0.82);
}

#[test]
fn round3_rounds_to_three_decimals() {
    assert_eq!(round3(0.8203125), 0.82);
    assert_eq!(round3(0.5), 0.5);
    assert_eq!(round3(0.0), 0.0);
    assert_eq!(round3(1.0), 1.0);
}

// ------------------------------------------------------------ class -> language

#[test]
fn class_index_maps_to_language_per_spec() {
    assert_eq!(class_to_language(0), Some(Language::English));
    assert_eq!(class_to_language(1), Some(Language::Japanese));
    assert_eq!(class_to_language(2), None);
}

#[test]
fn unknown_class_indices_map_to_none() {
    // spec §8.3 step 4: any other index -> None (plus a WARN, which is a logging
    // side effect and therefore not asserted here).
    for class_index in [3_u8, 7, 255] {
        assert_eq!(class_to_language(class_index), None);
    }
}

// ------------------------------------------------------------ composition

#[test]
fn postprocess_is_filter_then_nms_then_rescale() {
    let fixture = a3_fixture();
    let geometry = geometry();

    let composed = postprocess(&fixture, &geometry);
    let manual = rescale(&nms(filter_candidates(&fixture)), &geometry);

    assert_eq!(composed, manual);
    assert_eq!(composed.len(), 2);
}

#[test]
fn postprocess_of_an_empty_tensor_is_empty() {
    assert!(postprocess(&[], &geometry()).is_empty());
}
