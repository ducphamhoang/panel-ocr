//! spec §10.7(A)16 -- the frozen `black_bubble` gate, plus §16.9 items 12 and 18.
//! FROZEN.
//!
//! This is the one fixture-backed gate in Stage 3. Per §16.9 item 17 `pc-mask` reads no
//! `*_clean.png` at all (that comparison is F2's, in `cargo xtask calibrate-goldens`),
//! and per §16.9 item 18 this gate is driven from the always-present upstream *raw*
//! image rather than from a recorded fixture: it checks only the **selected fill
//! colour**, which is independent of where the precise mask came from.

use pc_config::MaskerConfig;
use pc_core::Rect;
use pc_imageops::BinaryMask;
use pc_mask::border::BaseCanvas;
use pc_mask::fit_region;

/// A region verified to lie inside the balloon: 11 200 px, mean luma 24.3, 989 light
/// text pixels (§16.9 item 18).
const MASKING: Rect = Rect {
    x1: 60,
    y1: 100,
    x2: 140,
    y2: 240,
};

#[test]
fn a16_black_bubble_fill_colour_is_dark_and_the_off_white_snap_does_not_fire() {
    // spec §10.7(A)16: the chosen median_color must be dark (max channel <= 40) and the
    // off-white snap must NOT fire. This catches the class of bug where a naive
    // implementation always fills white -- upstream's own black_bubble_clean.png fills
    // these pixels with exactly 0, so a white fill would be maximally wrong.
    let raw = pc_testkit::images::load_luma8(pc_testkit::paths::upstream(
        "demo_bubbles/black_bubble_raw.png",
    ));
    let size = raw.dimensions();
    assert_eq!(
        size,
        (202, 319),
        "spec §7.1's measured size for this fixture"
    );

    let reference = MASKING.pad(20, size);
    let precise = BinaryMask::from_gray_threshold(&raw, pc_imageops::PIL_BINARY_THRESHOLD);
    let box_mask = pc_imageops::rasterize_boxes(&[MASKING], size);
    let cut = precise.and(&box_mask);
    assert!(
        !cut.is_blank(),
        "the balloon interior does contain light text"
    );

    let fitment = fit_region(
        &BaseCanvas::Gray(raw),
        &cut,
        &box_mask,
        MASKING,
        reference,
        &MaskerConfig::default(),
    )
    .expect("a non-blank region is always reported");

    let maximum = *fitment.median_color.iter().max().expect("three channels");
    assert!(
        maximum <= 40,
        "black_bubble fill colour {:?} is not dark",
        fitment.median_color
    );
    assert_ne!(
        fitment.median_color,
        [255, 255, 255],
        "the off-white snap must not fire on a dark bubble"
    );

    // §16.9 item 12 is load-bearing here: on this thresholded precise mask the border
    // deviation exceeds the default mask_max_standard_deviation, so the fit is reported
    // as a failure -- and the fill colour must still be present to be checked.
    assert!(fitment.std_deviation > 0.0);
    assert_eq!(fitment.masking_rect, MASKING);
    assert_eq!(fitment.coords, (reference.x1, reference.y1));
}
