//! PIL-compatible preprocessing for the manga-ocr image encoder.

pub const INPUT_SIZE: u32 = 224;

const PRECISION_BITS: u32 = 22;
const PRECISION_SCALE: i64 = 1i64 << PRECISION_BITS;
const ROUNDING_BIAS: i64 = 1i64 << (PRECISION_BITS - 1);

/// Pillow's fixed-point ITU-R 601-2 conversion used by `Image.convert("L")`.
pub fn pil_luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * 19595 + u32::from(g) * 38470 + u32::from(b) * 7471 + 0x8000) >> 16) as u8
}

/// Convert an image to grayscale using Pillow's RGB-to-L conversion.
pub fn to_luma_pil(image: &image::DynamicImage) -> image::GrayImage {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb
        .pixels()
        .map(|pixel| pil_luma(pixel[0], pixel[1], pixel[2]))
        .collect();

    image::GrayImage::from_raw(width, height, pixels).expect("RGB and grayscale dimensions match")
}

#[derive(Debug)]
struct Coefficients {
    xmin: i32,
    weights: Vec<i32>,
}

fn bilinear_filter(value: f64) -> f64 {
    if value.abs() < 1.0 {
        1.0 - value.abs()
    } else {
        0.0
    }
}

fn coefficients(in_size: u32, out_size: u32) -> Vec<Coefficients> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filterscale;
    let ksize = (support.ceil() as i32) * 2 + 1;
    let ss = 1.0 / filterscale;

    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor() as i32).max(0);
            let xmax = ((center + support + 0.5).floor() as i32).min(in_size as i32) - xmin;

            let mut weights = Vec::with_capacity(ksize as usize);
            let mut normalized = Vec::with_capacity(xmax as usize);
            let mut total = 0.0;
            for x in 0..xmax {
                let weight = bilinear_filter((x as f64 + xmin as f64 - center + 0.5) * ss);
                total += weight;
                normalized.push(weight);
            }

            if total != 0.0 {
                for weight in &mut normalized {
                    *weight /= total;
                }
            }

            for weight in normalized {
                let scaled = weight * PRECISION_SCALE as f64;
                let fixed = if weight < 0.0 {
                    (scaled - 0.5) as i32
                } else {
                    (scaled + 0.5) as i32
                };
                weights.push(fixed);
            }

            Coefficients { xmin, weights }
        })
        .collect()
}

fn horizontal_pass(
    image: &image::GrayImage,
    coefficients: &[Coefficients],
    width: u32,
) -> image::GrayImage {
    let source_height = image.height();
    let mut output = vec![0u8; (width * source_height) as usize];

    for y in 0..source_height {
        for x in 0..width {
            let coefficient = &coefficients[x as usize];
            let mut sum = ROUNDING_BIAS;
            for (tap, &weight) in coefficient.weights.iter().enumerate() {
                let source_x = (coefficient.xmin as u32) + tap as u32;
                sum += i64::from(image.get_pixel(source_x, y)[0]) * i64::from(weight);
            }
            output[(y * width + x) as usize] = (sum >> PRECISION_BITS).clamp(0, 255) as u8;
        }
    }

    image::GrayImage::from_raw(width, source_height, output)
        .expect("horizontal pass dimensions match the buffer")
}

fn vertical_pass(
    image: &image::GrayImage,
    coefficients: &[Coefficients],
    height: u32,
) -> image::GrayImage {
    let width = image.width();
    let mut output = vec![0u8; (width * height) as usize];

    for y in 0..height {
        let coefficient = &coefficients[y as usize];
        for x in 0..width {
            let mut sum = ROUNDING_BIAS;
            for (tap, &weight) in coefficient.weights.iter().enumerate() {
                let source_y = (coefficient.xmin as u32) + tap as u32;
                sum += i64::from(image.get_pixel(x, source_y)[0]) * i64::from(weight);
            }
            output[(y * width + x) as usize] = (sum >> PRECISION_BITS).clamp(0, 255) as u8;
        }
    }

    image::GrayImage::from_raw(width, height, output)
        .expect("vertical pass dimensions match the buffer")
}

/// Resize a grayscale image with Pillow's fixed-point bilinear resampler.
pub fn pillow_resize_bilinear(
    image: &image::GrayImage,
    width: u32,
    height: u32,
) -> image::GrayImage {
    let horizontal = coefficients(image.width(), width);
    let vertical = coefficients(image.height(), height);
    let intermediate = horizontal_pass(image, &horizontal, width);
    vertical_pass(&intermediate, &vertical, height)
}

/// Convert, resize, duplicate into three NCHW planes, and normalize to `[-1, 1]`.
pub fn pixel_values(crop: &image::DynamicImage) -> Vec<f32> {
    let luma = to_luma_pil(crop);
    let resized = pillow_resize_bilinear(&luma, INPUT_SIZE, INPUT_SIZE);
    let plane = resized
        .as_raw()
        .iter()
        .map(|&pixel| (f32::from(pixel) / 255.0 - 0.5) / 0.5)
        .collect::<Vec<_>>();

    let mut values = Vec::with_capacity(3 * plane.len());
    values.extend_from_slice(&plane);
    values.extend_from_slice(&plane);
    values.extend_from_slice(&plane);
    values
}
