//! Task D4b -- spec §8.3 step 3's `ort` session (CPU EP, `intra_threads = 1`, one session
//! per run). FROZEN per CLAUDE.md.
//!
//! `#![cfg(feature = "onnx")]`: without the non-default `onnx` feature this file compiles
//! to an empty test binary, so a plain `cargo test --workspace` neither builds `ort` nor
//! needs an ONNX Runtime shared library.
//!
//! **Known state at freeze time: nothing in this file can run in this checkout, and that
//! is expected.** The workspace pins `ort` with `default-features = false`, which drops
//! `download-binaries`, so the moment D4b references an `ort` symbol the test binary needs
//! a native ONNX Runtime to link against -- and there is none here. This file is therefore
//! blocked as a whole, not merely its `#[ignore]`d test. Every D4 assertion that can run
//! today lives in `d4_onnx.rs` (task D4a), which is ungated and `ort`-free by design.
//!
//! **Concurrency note, and why no test here shares a session across threads naively.**
//! `ort` rc.12's `Session::run` takes `&mut self` (`session/mod.rs:212`), so §4.5's
//! "one session created once and shared" cannot mean concurrent inference on one session.
//! v1 wraps the session in a `Mutex` -- `DEVIATION(15)` -- which keeps
//! `OnnxDetector: Send + Sync` and keeps `TextDetector::detect(&self)` unchanged, at the
//! cost of serialising inference across images. `text_detector.concurrent_models > 1` is
//! warned-and-ignored; a session pool was rejected because §8.3 lists multi-model
//! concurrency as out of scope for v1. The smoke test below therefore asserts that shared
//! concurrent use is *correct and deterministic*, not that it is parallel.
#![cfg(feature = "onnx")]

use pc_core::StageError;
use pc_detect::onnx::{describe_outputs, runtime_available, OnnxDetector};
use pc_detect::yolo::ROW_STRIDE;
use pc_detect::TextDetector;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn onnx_detector_is_shareable_across_threads() {
    // spec §4.5: the parallelism is at the image level and every image borrows the same
    // `&dyn TextDetector`, so the detector type must be `Send + Sync`. With `Session::run`
    // taking `&mut self`, that property is what the `Mutex` (DEVIATION(15)) exists to
    // preserve -- if the wrapper is ever removed this stops compiling, which is the point.
    // Compile-time only: no session is constructed, so no weights are needed.
    assert_send_sync::<OnnxDetector>();
}

#[test]
fn from_path_reports_a_missing_model_file_as_a_model_error() {
    // spec §5.3: a missing model is fatal and must say which path was tried. The
    // path-shaped behaviour is separately tested, runtime-free, by
    // `d4_onnx.rs::ensure_model_file_reports_a_missing_file_by_name`; this test pins that
    // `from_path` actually performs that pre-flight before reaching `ort`.
    //
    // `from_path` takes an already-resolved, already-verified path: resolution and
    // download live in `pc-models`, which `pc-detect` deliberately does NOT depend on
    // (§1 rule 1 -- a stage crate must not acquire a `reqwest`/rustls dependency).
    let root = tempfile::TempDir::new().expect("temp dir");
    let missing = root.path().join("comictextdetector.pt.onnx");

    let error = OnnxDetector::from_path(&missing).expect_err("no such file");

    assert!(matches!(error, StageError::Model(_)));
    assert!(error.to_string().contains("comictextdetector.pt.onnx"));
}

#[test]
#[ignore = "opt-in: needs BOTH the real weights (PANEL_OCR_ONNX_MODEL=<file>) and an ONNX \
            Runtime shared library. Run with `PANEL_OCR_ONNX_MODEL=/path/comictextdetector.pt.onnx \
            cargo test -p pc-detect --features onnx --test d4_session -- --ignored --nocapture`"]
fn d4_real_weights_smoke_test() {
    // TWO independent preconditions, deliberately separated:
    //   1. the weights -- spec §7.2 / §16.13 item 8: maintainer-supplied, not in the repo,
    //      and §16.13 item 4 forbids fabricating a stand-in `.onnx`;
    //   2. the ONNX Runtime shared library -- a *different* gap from the weights, because
    //      `ort` is pinned with `default-features = false`.
    // Either one missing is a reported skip, never a failure (§16.13 item 4's principle).
    let Some(model) = std::env::var_os("PANEL_OCR_ONNX_MODEL") else {
        eprintln!(
            "skipping: set PANEL_OCR_ONNX_MODEL to the sha256-verified \
             comictextdetector.pt.onnx"
        );
        return;
    };
    if !runtime_available() {
        eprintln!("skipping: no ONNX Runtime shared library could be loaded");
        return;
    }
    let model = std::path::PathBuf::from(model);
    let detector = OnnxDetector::from_path(&model).expect("the real session opens");

    // spec §8.3 step 3 requires the ACTUAL output names and shapes to be observable
    // (logged once at DEBUG) so a model swap is diagnosable rather than mysterious.
    let outputs = detector.outputs();
    assert_eq!(outputs.len(), 3, "{}", describe_outputs(outputs));
    for name in ["blk", "seg", "det"] {
        assert!(
            outputs.iter().any(|output| output.name == name),
            "expected an output named `{name}`: {}",
            describe_outputs(outputs)
        );
    }

    // ---- the empirical confirmation of spec §16.15's `N_CLASSES` correction ----
    // If this fails, the shipped model's `blk` last dimension disagrees with §8.3 step 3's
    // `5 + n_classes` and therefore with the frozen `yolo::ROW_STRIDE`. That is a
    // spec/frozen-test conflict: escalate to BOTH architects per CLAUDE.md. Do NOT edit
    // `ROW_STRIDE`, `N_CLASSES`, or this assertion to make it pass.
    // (`bind_outputs` enforces the same invariant on every session construction; this is
    // the observation that validates the constant against the real weights.)
    let blk = &outputs[0];
    assert_eq!(
        *blk.shape.last().expect("blk has a last dimension"),
        ROW_STRIDE as i64,
        "blk shape {:?} disagrees with the frozen yolo::ROW_STRIDE = {ROW_STRIDE}",
        blk.shape
    );

    // ---- the standing `ort`-vs-protobuf-walk cross-check (task F3, spec §16.16) ----
    // The committed model signature that `d4_signature.rs` asserts against is produced by
    // a dependency-free protobuf walk in `xtask`, chosen because the authority for a
    // *structural* fact is the bytes of the sha256-verified file, not a third-party
    // library (contrast §16.13 item 4, which governs *behavioural* references). The cost
    // of that choice is that the walk is new code whose failure mode is plausible rather
    // than obvious; this comparison is the structural answer to it. Whenever anyone runs
    // this smoke test with real weights, two independent deserializers of the same file
    // are compared: `ort`'s graph reader and our walk. A disagreement means one of them is
    // wrong and escalates -- it is never patched by re-recording.
    //
    // Names and shapes only: `OutputMeta` does not carry element types, so the dtype half
    // of the signature is covered by `d4_signature.rs` against the walk alone.
    let recorded = pc_testkit::model_signature::comic_text_detector_signature();
    assert_eq!(
        outputs.len(),
        recorded.outputs.len(),
        "live session: {}",
        describe_outputs(outputs)
    );
    for (live, walked) in outputs.iter().zip(recorded.outputs.iter()) {
        assert_eq!(
            live.name,
            walked.name,
            "`ort` and the recorded protobuf walk disagree on output order/names: {}",
            describe_outputs(outputs)
        );
        assert_eq!(
            live.shape, walked.shape,
            "`ort` and the recorded protobuf walk disagree on `{}`'s shape",
            live.name
        );
    }

    let page = image::RgbImage::from_fn(700, 1000, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 200])
    });

    let first = detector.detect(&page).expect("inference succeeds");

    // spec §8.2's contract: the returned mask is 8-bit and the SAME SIZE as the input
    // image, i.e. the backend applied §8.3 step 5's crop and resize itself.
    assert_eq!(first.mask.dimensions(), page.dimensions());
    // spec §16.6 item 4: every rescaled box is clipped into the base image.
    for block in &first.blocks {
        assert!(
            block.rect.x1 >= 0 && block.rect.y1 >= 0,
            "unclipped box {:?}",
            block.rect
        );
        assert!(
            block.rect.x2 <= 700 && block.rect.y2 <= 1000,
            "unclipped box {:?}",
            block.rect
        );
    }

    // spec §5.7: the same session must give the same answer call after call.
    let second = detector.detect(&page).expect("inference succeeds");
    assert_eq!(second.blocks, first.blocks);
    assert_eq!(second.mask.as_raw(), first.mask.as_raw());

    // spec §4.5 + DEVIATION(15): four threads sharing ONE `&dyn TextDetector` must all
    // succeed and agree. This is the property the `Mutex` buys -- serialised, but correct
    // and deterministic; it is what makes the detector usable from the rayon batch runner.
    let shared: &dyn TextDetector = &detector;
    let results: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| shared.detect(&page).expect("inference succeeds")))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker did not panic"))
            .collect()
    });

    assert_eq!(results.len(), 4);
    for detection in &results {
        assert_eq!(detection.blocks, first.blocks);
        assert_eq!(detection.mask.as_raw(), first.mask.as_raw());
    }
}
