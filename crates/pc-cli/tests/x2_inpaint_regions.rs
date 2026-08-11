use image::{Rgba, RgbaImage};
use pc_cli::standalone_inpaint::regions_from_brush_mask;

fn painted(w: u32, h: u32, blobs: &[(u32, u32, u32, u32)]) -> RgbaImage {
    let mut mask = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    for &(x1, y1, x2, y2) in blobs {
        for y in y1..y2 {
            for x in x1..x2 {
                mask.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
    }
    mask
}

#[test]
fn two_disjoint_blobs_become_two_failed_regions_with_matching_bounding_boxes() {
    let mask = painted(100, 100, &[(10, 10, 30, 25), (60, 60, 90, 80)]);
    let (mut regions, raw_mask) = regions_from_brush_mask(&mask);
    regions.sort_by_key(|r| r.rect.x1);

    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].rect, pc_core::Rect::new(10, 10, 30, 25));
    assert_eq!(regions[1].rect, pc_core::Rect::new(60, 60, 90, 80));
    assert!(regions.iter().all(|r| r.failed && r.thickness.is_none()));

    // raw_mask carries exactly the painted silhouette, not just the bounding boxes.
    assert_eq!(raw_mask.get_pixel(15, 15).0[0], 255);
    assert_eq!(raw_mask.get_pixel(45, 45).0[0], 0); // between the two blobs
}

#[test]
fn an_unpainted_mask_yields_no_regions() {
    let mask = painted(50, 50, &[]);
    let (regions, _raw_mask) = regions_from_brush_mask(&mask);
    assert!(regions.is_empty());
}

#[test]
fn an_opaque_black_paint_stroke_is_still_detected_via_alpha_not_luma() {
    // Luma of `Rgba([0, 0, 0, 255])` is 0 — a luma-keyed reader would see nothing painted.
    // `regions_from_brush_mask` must key on alpha (spec §14 item 5 / DEVIATION(28)'s ground:
    // "a mask's 'is this pixel covered' signal must not depend on the brightness of its fill
    // color"), so this must still find a region.
    let mut mask = RgbaImage::from_pixel(50, 50, Rgba([0, 0, 0, 0]));
    for y in 10..20 {
        for x in 10..20 {
            mask.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
    }
    let (regions, raw_mask) = regions_from_brush_mask(&mask);
    assert_eq!(regions.len(), 1);
    assert_eq!(raw_mask.get_pixel(15, 15).0[0], 255);
}
