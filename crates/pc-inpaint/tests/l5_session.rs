//! Task **L5** — [`OnnxInpainter`] construction, in the `onnx` tier but with **no weights**.
//!
//! §16.38 item 9 declares session construction **run-fatal**: *"construction failures render
//! as `StageError::Model`"* (item 9(g)). Every test here is about that half of the split; the
//! per-image half lives in `l5_tensors.rs` (and, through the stub, in L4's
//! `l4_inpainter_trait.rs`).
#![cfg(feature = "onnx")]

use pc_core::device::{Device, DeviceSupport};
use pc_core::StageError;
use pc_inpaint::onnx::OnnxInpainter;
use pc_inpaint::Inpainter;
use tempfile::TempDir;

fn assert_send_sync<T: Send + Sync>() {}

/// [`pc_inpaint::Inpainter`] is `Send + Sync` because one session serves every rayon worker
/// (§4.5), and §16.38 item 8(d)'s latch hands the same `Arc` to all of them. If `Session` ever
/// stops being holdable behind a `Mutex` this stops compiling, which is the point.
#[test]
fn the_onnx_inpainter_is_send_and_sync_and_usable_as_a_dyn_inpainter() {
    assert_send_sync::<OnnxInpainter>();
    // Object safety, asserted by construction rather than assumed: `inpaint_page` takes
    // `&dyn Inpainter`.
    fn _takes_dyn(_: &dyn Inpainter) {}
}

/// §16.38 item 9: a missing model is run-fatal. The path must be named, because §16.19 item
/// 3's remedy — *"the user re-running `panel-ocr models download`"* — is not actionable
/// otherwise.
///
/// The message must **not** mention ONNX Runtime: the pre-flight runs before any `ort` call,
/// so a missing file cannot be misreported as a runtime problem. That ordering is the same
/// one `pc_ocr::onnx`'s `both_missing_paths_fail_on_the_encoder_preflight` pins.
#[test]
fn a_missing_model_file_is_a_model_error_naming_its_own_path_before_any_ort_call() {
    let root = TempDir::new().expect("temp dir");
    let absent = root.path().join("lama-manga.onnx");

    let error = OnnxInpainter::from_path(&absent).expect_err("a missing model is refused");

    assert!(
        matches!(error, StageError::Model(_)),
        "construction failures are run-fatal `Model`, never per-image `Inference` \
         (§16.38 item 9(g)); got {error:?}"
    );
    let message = error.to_string();
    assert!(message.contains(&absent.display().to_string()), "{message}");
    assert!(!message.contains("ONNX Runtime"), "{message}");
}

/// A path that exists but is not a directory-free regular file, and a path that is a
/// directory, both fail the same pre-flight.
#[test]
fn a_directory_in_place_of_the_model_is_a_model_error() {
    let root = TempDir::new().expect("temp dir");
    let as_dir = root.path().join("lama-manga.onnx");
    std::fs::create_dir(&as_dir).expect("create dir");

    let error = OnnxInpainter::from_path(&as_dir).expect_err("a directory is not a model");

    assert!(matches!(error, StageError::Model(_)), "got {error:?}");
}

/// §16.38 item 9: *"a missing or **uninitializable** inpainting model is RUN-FATAL"*.
///
/// A present file that `ort` cannot parse is the uninitializable case, and it must classify
/// the same way a missing one does — not as an `Inference` error, which would let the
/// pipeline continue to the next page and re-attempt (§16.38 item 8(d)'s latch exists to stop
/// exactly that).
#[test]
fn a_present_file_that_is_not_an_onnx_graph_is_also_a_run_fatal_model_error() {
    let root = TempDir::new().expect("temp dir");
    let garbage = root.path().join("lama-manga.onnx");
    std::fs::write(&garbage, b"this is not an ONNX graph").expect("write garbage");

    let error = OnnxInpainter::from_path(&garbage).expect_err("garbage bytes are refused");

    assert!(
        matches!(error, StageError::Model(_)),
        "an unloadable artifact is run-fatal, not per-image; got {error:?}"
    );
    assert!(
        error.to_string().contains(&garbage.display().to_string()),
        "{error}"
    );
}

/// §16.38 item 16(d): *"`ort` session construction through `pc_core::device::resolve` (§16.36
/// item 2, **no second policy surface**)"*.
///
/// The expected message is taken from `pc_core::device::resolve` itself — an **independent
/// oracle**, since that is a different crate with its own frozen tests
/// (`crates/pc-core/tests/device_policy.rs`) — and not from any copy inside `pc-inpaint`. If
/// `pc-inpaint` grew its own refusal wording, this would go red.
///
/// The model path is a **nonexistent** file on purpose: the device refusal must arrive before
/// the file pre-flight, or a user on a CUDA-less build gets told to download a model instead
/// of to rebuild.
#[test]
fn a_device_this_build_cannot_provide_is_refused_by_pc_core_policy_before_the_model_pre_flight() {
    let root = TempDir::new().expect("temp dir");
    let absent = root.path().join("lama-manga.onnx");

    match pc_core::device::resolve(Device::Cuda, DeviceSupport::compiled()) {
        Err(refusal) => {
            let error = OnnxInpainter::from_path_for_device(&absent, Device::Cuda)
                .expect_err("a device this build cannot provide is refused");
            let StageError::Model(message) = &error else {
                panic!("a device refusal is run-fatal `Model`; got {error:?}");
            };
            // The payload, not the rendered `Display` (which adds a "model error: " prefix),
            // so the comparison is against `pc_core`'s own string and nothing else.
            assert_eq!(
                message,
                &refusal.message(),
                "the stage must carry `pc_core`'s refusal verbatim, not its own wording"
            );
            assert!(
                !message.contains(&absent.display().to_string()),
                "the device refusal must precede the model pre-flight; got {message}"
            );
        }
        Ok(_) => {
            // A build that CAN provide CUDA must fall through to the model pre-flight, so
            // the same call fails on the absent file instead. Asserted rather than skipped,
            // so this test is never vacuous in either build.
            let error = OnnxInpainter::from_path_for_device(&absent, Device::Cuda)
                .expect_err("the absent model is still refused");
            assert!(
                error.to_string().contains(&absent.display().to_string()),
                "got {error}"
            );
        }
    }
}

/// The CPU device resolves on every build, so this call must get **past** the device gate and
/// fail on the model instead. Without it, the test above could pass on a stage that refuses
/// every device.
#[test]
fn the_cpu_device_always_resolves_so_construction_fails_on_the_model_rather_than_the_device() {
    let root = TempDir::new().expect("temp dir");
    let absent = root.path().join("lama-manga.onnx");

    let error = OnnxInpainter::from_path_for_device(&absent, Device::Cpu)
        .expect_err("the model is still absent");

    assert!(
        error.to_string().contains(&absent.display().to_string()),
        "the CPU path must reach the model pre-flight; got {error}"
    );
    assert!(
        !error
            .to_string()
            .contains("execution provider is not available"),
        "got {error}"
    );
}
