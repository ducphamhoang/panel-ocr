//! Task N3 -- spec §11.3 steps 3-5, §11.7(A)6/8/9/10, §16.10 items 2, 14-16. FROZEN.

mod common;

use common::{combined_mask, region, shared_nlm, PAGE_SIZE, REGION};
use image::{GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use pc_config::DenoiserConfig;
use pc_core::Rect;
use pc_denoise::{morph, noise_mask};
use pc_imageops::BinaryMask;

// ---------------------------------------------------------------- §11.3 step 3

#[test]
fn a6_region_selection_is_strictly_greater_and_excludes_failures() {
    // spec §11.7(A)6, verbatim: with noise_min_standard_deviation = 0.25 and regions
    // [sigma=0.25 ok, sigma=0.26 ok, sigma=20 failed], exactly ONE region (sigma=0.26)
    // is denoised. Both the strict `>` (0.25 is NOT selected) and the `failed`
    // exclusion (sigma=20 is not selected despite being far over the cutoff) are locked
    // here -- §11.3 step 3's rationale is that a perfect fit has no noise ring to hide
    // and a failed fit painted nothing to blend.
    let regions = vec![
        region(Rect::new(0, 0, 10, 10), 0.25, false),
        region(Rect::new(20, 0, 30, 10), 0.26, false),
        region(Rect::new(40, 0, 50, 10), 20.0, true),
    ];
    let selected = noise_mask::select_regions(&regions, 0.25);
    assert_eq!(selected.len(), 1, "selected: {selected:?}");
    assert_eq!(selected[0].std_deviation, 0.26);
    assert_eq!(selected[0].rect, Rect::new(20, 0, 30, 10));
}

#[test]
fn a6_selection_preserves_region_order() {
    // §16.10 item 16: order is preserved, because layers composite source-over in that
    // order and a later region must win an overlap (the same tie-break as §16.9's
    // `build_combined_mask`).
    let regions = vec![
        region(Rect::new(0, 0, 10, 10), 5.0, false),
        region(Rect::new(1, 1, 11, 11), 0.1, false),
        region(Rect::new(2, 2, 12, 12), 7.0, false),
        region(Rect::new(3, 3, 13, 13), 6.0, true),
        region(Rect::new(4, 4, 14, 14), 1.0, false),
    ];
    let rects: Vec<Rect> = noise_mask::select_regions(&regions, 0.25)
        .iter()
        .map(|region| region.rect)
        .collect();
    assert_eq!(
        rects,
        vec![
            Rect::new(0, 0, 10, 10),
            Rect::new(2, 2, 12, 12),
            Rect::new(4, 4, 14, 14),
        ]
    );
}

// ---------------------------------------------------------------- §11.3 step 4.3

#[test]
fn a9_a_black_filled_mask_still_produces_a_non_empty_noise_mask() {
    // spec §11.7(A)9 / §14.5 / §15.6 -- the regression test for the upstream luma bug.
    // Upstream's `grow_mask` does `mask.convert("L")` on the RGBA cutout, which takes
    // RGB luma and DISCARDS alpha; for a `black_bubble` fill of (0,0,0,255) the luma is
    // 0 everywhere and the noise mask silently vanishes. v1 uses ALPHA.
    let black_fill = combined_mask(PAGE_SIZE, &[(REGION, [0, 0, 0, 255])]);
    let crop = REGION.to_crop(PAGE_SIZE).expect("an interior rect");
    let cutout = noise_mask::crop_rgba(&black_fill, crop);

    // The upstream path, spelled out so the bug is visible in the test, not just the
    // comment: every RGB luma in the cutout is 0, so a luma-derived mask is blank.
    assert!(
        cutout
            .pixels()
            .all(|pixel| pixel.0[0] == 0 && pixel.0[1] == 0 && pixel.0[2] == 0),
        "the fixture must be the black-fill case"
    );

    let binary = noise_mask::alpha_binary(&cutout);
    assert!(
        !binary.is_blank(),
        "the alpha-derived mask must be non-empty"
    );
    assert_eq!(
        binary.count_set(),
        (REGION.width() * REGION.height()) as usize,
        "every opaque pixel of the fill is covered"
    );

    // And the grown+faded mask that actually gets attached is non-empty too.
    let faded = noise_mask::fade_mask(&binary, 5, 1);
    assert!(faded.pixels().any(|pixel| pixel.0[0] > 0));
}

#[test]
fn alpha_binary_is_strictly_greater_than_zero() {
    // A partially transparent pixel still counts as covered; only alpha == 0 does not.
    let mask = RgbaImage::from_fn(4, 1, |x, _| {
        Rgba([255, 255, 255, [0, 1, 128, 255][x as usize]])
    });
    let binary = noise_mask::alpha_binary(&mask);
    assert_eq!(
        (0..4).map(|x| binary.get(x, 0)).collect::<Vec<_>>(),
        vec![false, true, true, true]
    );
}

// ---------------------------------------------------------------- §11.3 step 4.4

#[test]
fn the_growth_kernel_matches_the_masking_stage_cell_for_cell() {
    // §16.10 item 2: `pc_denoise::morph::kernel` is a duplicate of `pc_mask::grow::kernel`
    // (§1 rule 2 forbids the dependency that would let us share it). This test pins the
    // full matrices by hand so the two copies cannot drift silently.
    //
    // `noise_outline_size = 5` -> an 11x11 OpenCV MORPH_ELLIPSE. Row half-widths, from
    // `dx = round(sqrt(25 - dy^2))` for dy = -5..=5:
    //   0, 3, 4, 5, 5, 5, 5, 5, 4, 3, 0
    // giving row widths 1, 7, 9, 11, 11, 11, 11, 11, 9, 7, 1 and 89 set cells.
    let ellipse = morph::kernel(5);
    assert_eq!(ellipse.diameter(), 11);
    assert_eq!(ellipse.count(), 89);
    let widths: Vec<usize> = (0..11)
        .map(|i| (0..11).filter(|j| ellipse.get(*j, i)).count())
        .collect();
    assert_eq!(widths, vec![1, 7, 9, 11, 11, 11, 11, 11, 9, 7, 1]);
    for i in 0..11_u32 {
        let set: Vec<u32> = (0..11).filter(|j| ellipse.get(*j, i)).collect();
        if let (Some(first), Some(last)) = (set.first(), set.last()) {
            assert_eq!(
                (last - first + 1) as usize,
                set.len(),
                "row {i} must be contiguous"
            );
            assert_eq!(first + last, 10, "row {i} must be centred");
        }
    }

    // §16.9 item 5 / §11.3 step 4: `size == 0` is the single centre pixel, so dilation
    // is the identity. Zeroing the "corners" of a 1x1 kernel would erase the mask.
    let identity = morph::kernel(0);
    assert_eq!(identity.diameter(), 1);
    assert_eq!(identity.as_cells(), &[1]);

    // The small branch: a full square with the four corners zeroed.
    assert_eq!(morph::kernel(1).as_cells(), &[0, 1, 0, 1, 1, 1, 0, 1, 0]);
    assert_eq!(morph::kernel(2).count(), 21);
}

#[test]
fn dilation_with_a_zero_kernel_is_the_identity() {
    let mut dot = BinaryMask::new(7, 7);
    dot.set(3, 3, true);
    assert_eq!(morph::dilate(&dot, &morph::kernel(0)), dot);
}

#[test]
fn fade_mask_grows_then_fades_a_single_dot() {
    // spec §11.3 steps 4.4-4.5, the docstring intent quoted in §14.5: "grow the mask
    // and fade its edges". With outline 5 / fade 0 the result is exactly the kernel
    // footprint at 255; with fade 1 the centre stays 255 and a soft ring appears
    // OUTSIDE the footprint, which is what the noise mask blends through.
    let mut dot = BinaryMask::new(31, 31);
    dot.set(15, 15, true);

    let hard = noise_mask::fade_mask(&dot, 5, 0);
    assert_eq!(
        hard.pixels().filter(|pixel| pixel.0[0] == 255).count(),
        89,
        "outline 5 stamps the 11x11 ellipse, unblurred"
    );
    assert!(hard
        .pixels()
        .all(|pixel| pixel.0[0] == 0 || pixel.0[0] == 255));

    let faded = noise_mask::fade_mask(&dot, 5, 1);
    assert_eq!(faded.get_pixel(15, 15).0[0], 255, "the centre stays opaque");
    assert!(
        faded
            .pixels()
            .any(|pixel| pixel.0[0] > 0 && pixel.0[0] < 255),
        "the fade must produce intermediate alphas"
    );
    let hard_extent = extent(&hard);
    let faded_extent = extent(&faded);
    assert!(
        faded_extent > hard_extent,
        "fading widens the mask: {faded_extent} vs {hard_extent}"
    );
    assert!(
        faded_extent <= hard_extent + 2 * 3,
        "and no further than 3*sigma: {faded_extent} vs {hard_extent}"
    );
}

/// Width of the bounding box of the non-zero pixels.
fn extent(image: &GrayImage) -> u32 {
    let columns: Vec<u32> = (0..image.width())
        .filter(|x| (0..image.height()).any(|y| image.get_pixel(*x, y).0[0] > 0))
        .collect();
    match (columns.first(), columns.last()) {
        (Some(first), Some(last)) => last - first + 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------- §11.3 step 4.7

#[test]
fn attach_alpha_keeps_the_rgb_and_takes_the_mask_as_alpha() {
    let rgb = RgbImage::from_fn(3, 2, |x, y| Rgb([x as u8, y as u8, 9]));
    let alpha = GrayImage::from_fn(3, 2, |x, y| Luma([(x * 10 + y) as u8]));
    let layer = noise_mask::attach_alpha(&rgb, &alpha);
    for (x, y, pixel) in layer.enumerate_pixels() {
        assert_eq!(pixel.0, [x as u8, y as u8, 9, (x * 10 + y) as u8]);
    }
}

// ---------------------------------------------------------------- §16.10 item 14

#[test]
fn a_degenerate_scaled_rect_has_no_crop() {
    // §16.10 item 14: a rect that scales to nothing, or lands entirely outside the
    // canvas, yields `None` -- which `build_noise_mask` must treat as a skip with a
    // WARN, never as a `StageError` (§5.6).
    assert!(noise_mask::region_crop(Rect::new(5, 5, 5, 9), 1.0, PAGE_SIZE).is_none());
    assert!(noise_mask::region_crop(Rect::new(500, 500, 520, 520), 1.0, PAGE_SIZE).is_none());
    // `Rect::scale` truncates toward zero (§2.1), so a 1-px-wide rect can collapse.
    assert!(noise_mask::region_crop(Rect::new(10, 10, 11, 20), 0.25, PAGE_SIZE).is_none());

    let (scaled, crop) =
        noise_mask::region_crop(REGION, 2.0, (PAGE_SIZE.0 * 2, PAGE_SIZE.1 * 2)).expect("in range");
    assert_eq!(scaled, Rect::new(80, 60, 140, 120));
    assert_eq!(crop, (80, 60, 60, 60));
}

// ---------------------------------------------------------------- §11.3 step 5

#[test]
fn a10_no_selected_regions_yields_a_fully_transparent_noise_mask() {
    // spec §11.7(A)10, at the `build_noise_mask` level: zero qualifying regions means a
    // fully transparent RGBA mask of the canvas size and `boxes_denoised == 0` -- NOT
    // an error and NOT a zero-sized image.
    let _nlm = shared_nlm();
    let cleaned = RgbImage::from_pixel(PAGE_SIZE.0, PAGE_SIZE.1, Rgb([200, 200, 200]));
    let mask = combined_mask(PAGE_SIZE, &[]);
    let (noise, denoised_count) =
        noise_mask::build_noise_mask(&cleaned, &mask, &[], 1.0, &DenoiserConfig::default());

    assert_eq!(denoised_count, 0);
    assert_eq!(noise.dimensions(), PAGE_SIZE);
    assert!(
        noise.pixels().all(|pixel| pixel.0 == [0, 0, 0, 0]),
        "the blank noise mask must be fully transparent"
    );
}

#[test]
fn a8_build_noise_mask_touches_only_the_grown_and_faded_region() {
    // spec §11.7(A)8: the noise mask is opaque ONLY inside
    // `region.rect.pad(noise_outline_size + 3 * noise_fade_radius)`. Everything outside
    // must stay fully transparent, so compositing it can never alter a pixel the masker
    // did not already own. This is the containment guarantee §11.3's "Explicitly out of
    // scope" paragraph promises ("this stage only ever touches pixels inside the grown
    // +faded per-region masks").
    let _nlm = shared_nlm();
    let config = DenoiserConfig {
        template_window_size: 3,
        search_window_size: 5,
        ..DenoiserConfig::default()
    };
    let cleaned = RgbImage::from_fn(PAGE_SIZE.0, PAGE_SIZE.1, |x, y| {
        Rgb([((x * 7 + y * 13) % 256) as u8; 3])
    });
    let mask = combined_mask(PAGE_SIZE, &[(REGION, [255, 255, 255, 255])]);
    let regions = vec![region(REGION, 3.0, false)];
    let selected = noise_mask::select_regions(&regions, config.noise_min_standard_deviation);
    assert_eq!(selected.len(), 1);

    let (noise, denoised_count) =
        noise_mask::build_noise_mask(&cleaned, &mask, &selected, 1.0, &config);
    assert_eq!(denoised_count, 1, "§16.10 item 15 counts produced layers");
    assert_eq!(noise.dimensions(), PAGE_SIZE);

    let reach = (config.noise_outline_size + 3 * config.noise_fade_radius) as i32;
    let allowed = REGION.pad(reach, PAGE_SIZE);
    for (x, y, pixel) in noise.enumerate_pixels() {
        if pixel.0[3] == 0 {
            continue;
        }
        assert!(
            allowed.contains((x as i32, y as i32)),
            "the noise mask is opaque at ({x}, {y}), outside {allowed:?}"
        );
    }
    assert!(
        noise.pixels().any(|pixel| pixel.0[3] > 0),
        "a selected region must actually produce a layer"
    );
}

#[test]
fn a8_layers_land_at_the_region_origin() {
    // The placement half of §11.7(A)8: the layer is composited at the SCALED rect's
    // top-left (§11.3 step 4.8), so a region in the lower-right quadrant must not leave
    // any opaque pixel in the upper-left one.
    let _nlm = shared_nlm();
    let config = DenoiserConfig {
        template_window_size: 3,
        search_window_size: 5,
        noise_outline_size: 0,
        noise_fade_radius: 0,
        ..DenoiserConfig::default()
    };
    let cleaned = RgbImage::from_pixel(PAGE_SIZE.0, PAGE_SIZE.1, Rgb([120, 120, 120]));
    let mask = combined_mask(PAGE_SIZE, &[(REGION, [10, 20, 30, 255])]);
    let regions = vec![region(REGION, 3.0, false)];
    let selected = noise_mask::select_regions(&regions, config.noise_min_standard_deviation);

    let (noise, _) = noise_mask::build_noise_mask(&cleaned, &mask, &selected, 1.0, &config);
    // With no growth and no fade the opaque set is exactly the region rect.
    for (x, y, pixel) in noise.enumerate_pixels() {
        let inside = REGION.contains((x as i32, y as i32))
            && (x as i32) < REGION.x2
            && (y as i32) < REGION.y2;
        assert_eq!(
            pixel.0[3] > 0,
            inside,
            "alpha at ({x}, {y}) disagrees with the region rect"
        );
    }
}
