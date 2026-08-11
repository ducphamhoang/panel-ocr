use image::{Rgb, RgbImage, Rgba, RgbaImage};
use pc_cli::standalone_inpaint::regions_from_brush_mask;
use pc_config::Profile;
use pc_inpaint::stub::StubInpainter;
use pc_inpaint::PageInput;

fn base_image(w: u32, h: u32) -> RgbImage {
    RgbImage::from_pixel(w, h, Rgb([200, 200, 200]))
}

fn painted_mask(w: u32, h: u32, rect: (u32, u32, u32, u32)) -> RgbaImage {
    let mut mask = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    let (x1, y1, x2, y2) = rect;
    for y in y1..y2 {
        for x in x1..x2 {
            mask.put_pixel(x, y, Rgba([255, 255, 255, 255]));
        }
    }
    mask
}

/// The correctness point Task 3's design comment argues for: feeding the user's own paint
/// into `combined_mask` instead of a transparent no-op layer must VISIBLY diverge from the
/// no-op in the fade band — or the no-op design decision is unguarded. Pixels far outside
/// the painted region agree under both, so the divergence must be asserted inside the fade
/// band, on the silhouette's own edge.
#[test]
fn a_leaked_combined_mask_visibly_diverges_from_the_transparent_one_in_the_fade_band() {
    let image = base_image(200, 200);
    let mask = painted_mask(200, 200, (80, 80, 120, 120));
    let (regions, raw_mask) = regions_from_brush_mask(&mask);
    let transparent = RgbaImage::new(200, 200);
    let config = Profile::default().inpainter;
    let min_mask_thickness = Profile::default().masker.min_mask_thickness;

    let run = |combined_mask: &RgbaImage| {
        pc_inpaint::inpaint_page(
            PageInput {
                original: &image,
                raw_mask: &raw_mask,
                combined_mask,
                noise_mask: None,
                regions: &regions,
                min_mask_thickness,
                config: &config,
            },
            &StubInpainter::flat(10, 20, 30),
        )
        .expect("stub inpainter never fails")
    };

    // What `standalone_inpaint::run` actually does: an empty transparent canvas.
    let correct = run(&transparent);
    // The bug this whole design decision guards against: feeding the user's own paint in.
    let leaked = run(&mask);

    // Well outside any growth/fade/isolation radius — both variants must agree here.
    let corner_correct = correct.clean_inpaint.get_pixel(5, 5);
    let corner_leaked = leaked.clean_inpaint.get_pixel(5, 5);
    assert_eq!(
        [corner_correct[0], corner_correct[1], corner_correct[2]],
        [200, 200, 200]
    );
    assert_eq!(corner_correct, corner_leaked);

    // Inside the fade band, on the silhouette's own edge (x = 80, the painted rect's left
    // edge) — this is where a leaked opaque combined_mask visibly diverges (measured: the
    // review's counterfactual differs here, alpha ≈ 248 either side of the true edge).
    assert_ne!(
        correct.clean_inpaint.get_pixel(80, 100),
        leaked.clean_inpaint.get_pixel(80, 100),
        "a combined_mask carrying the user's paint must diverge from the transparent \
         no-op in the fade band, or the no-op design decision is unguarded"
    );

    // M2: the success path actually fills — the painted region's centre matches the stub's
    // flat fill, alpha flattened to 255.
    let centre = correct.clean_inpaint.get_pixel(100, 100);
    assert_eq!(*centre, image::Rgba([10, 20, 30, 255]));
}

#[test]
fn an_all_transparent_mask_is_rejected_with_no_output_written() {
    // regions_from_brush_mask's empty case is asserted directly in x2_inpaint_regions.rs;
    // this test is the CLI-level contract: run() must bail out, not silently write an
    // unchanged image. Exercised through `pc_cli::standalone_inpaint::run` with a real
    // temp dir + real files.
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let image_path = dir.path().join("page.png");
    let mask_path = dir.path().join("mask.png");
    let output_path = dir.path().join("out.png");
    base_image(50, 50)
        .save_with_format(&image_path, image::ImageFormat::Png)
        .unwrap();
    RgbaImage::from_pixel(50, 50, Rgba([0, 0, 0, 0]))
        .save_with_format(&mask_path, image::ImageFormat::Png)
        .unwrap();

    let args = pc_cli::args::InpaintArgs {
        image: image_path,
        mask: mask_path,
        output: output_path.clone(),
        profile: None,
        profile_path: None,
        model_path: None,
        cache_dir: Some(dir.path().to_path_buf()),
    };
    let error = pc_cli::standalone_inpaint::run(args).expect_err("empty mask must be rejected");
    assert!(error.to_string().contains("no painted pixels"));
    assert!(!output_path.exists());
}

#[test]
fn mismatched_mask_and_image_dimensions_are_rejected() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let image_path = dir.path().join("page.png");
    let mask_path = dir.path().join("mask.png");
    base_image(50, 50)
        .save_with_format(&image_path, image::ImageFormat::Png)
        .unwrap();
    painted_mask(30, 30, (5, 5, 10, 10))
        .save_with_format(&mask_path, image::ImageFormat::Png)
        .unwrap();

    let args = pc_cli::args::InpaintArgs {
        image: image_path,
        mask: mask_path,
        output: dir.path().join("out.png"),
        profile: None,
        profile_path: None,
        model_path: None,
        cache_dir: Some(dir.path().to_path_buf()),
    };
    let error =
        pc_cli::standalone_inpaint::run(args).expect_err("dimension mismatch must be rejected");
    assert!(error.to_string().contains("same pixel dimensions"));
}
