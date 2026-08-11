#![cfg(feature = "onnx")]

use ort::ep::CUDA;
use pc_core::device::{resolve, Device, DevicePolicy, DeviceSupport};
use pc_detect::onnx::execution_provider_dispatches;

/// `WITH_CUDA` explicitly, not `compiled()`: reachable in EVERY build, per §16.36 item 2
/// making `DeviceSupport` a value rather than a `cfg!` read.
fn cuda_policy() -> DevicePolicy {
    resolve(Device::Cuda, DeviceSupport::WITH_CUDA)
        .expect("the explicit WITH_CUDA capability resolves cuda in any build")
}

/// spec §16.22 item 3 + §16.36 item 7 bullet 4: our session options must pin Heuristic or
/// Default, never `ort`'s own `#[default]` Exhaustive. Oracle is `ort`'s own mapping
/// (`"HEURISTIC"` string), not anything this codebase wrote.
#[test]
fn a_cuda_policy_pins_heuristic_conv_algorithm_search_on_the_ort_provider() {
    let dispatches = execution_provider_dispatches(&cuda_policy());
    assert_eq!(
        dispatches.len(),
        1,
        "one explicit provider, per §16.22 item 5(c)"
    );

    let cuda = dispatches[0]
        .downcast_ref::<CUDA>()
        .expect("the single dispatch must be ort's CUDA provider");
    let configured = format!("{cuda:?}");
    assert!(
        configured.contains(r#""cudnn_conv_algo_search": "HEURISTIC""#),
        "ort must receive HEURISTIC; got {configured}"
    );
    assert!(
        !configured.contains("EXHAUSTIVE"),
        "ConvAlgorithmSearch::Exhaustive is forbidden by §16.22 item 3; got {configured}"
    );

    // Anti-vacuity: `ort` sets NO such option key by default — the key can only be
    // present because our mapping put it there.
    let untouched = format!("{:?}", CUDA::default());
    assert!(
        !untouched.contains("cudnn_conv_algo_search"),
        "ort's default no longer leaves the key unset; this test's premise changed: {untouched}"
    );
}

/// spec §16.22 item 5(c): EP registration uses error-on-failure. `ort`'s constructor
/// defaults to `error_on_failure: false` ("silently fail and fall back to ... the CPU
/// provider") — countermanding that is the single most important line in the feature.
#[test]
fn the_cuda_dispatch_carries_error_on_failure_rather_than_orts_silent_default() {
    let dispatches = execution_provider_dispatches(&cuda_policy());
    assert_eq!(
        format!("{:?}", dispatches[0]),
        "CUDAExecutionProvider { error_on_failure: true }"
    );
}

/// spec §16.22 item 5(c), behaviourally: a registration that cannot succeed must produce
/// a loud run-fatal `StageError::Model`, never a silent CPU session.
///
/// Gated to non-`cuda` builds: without `ort/cuda`, `CUDA::register` is compiled to
/// `Err(RegisterError::MissingFeature)` unconditionally, independent of hardware/driver/
/// cuDNN — the only deterministic-everywhere case. In a `cuda` build the outcome depends
/// on the machine (see §16.47 item 8's manual real-model verification instead).
#[cfg(not(feature = "cuda"))]
#[test]
fn a_cuda_registration_that_cannot_succeed_is_a_loud_model_error() {
    use pc_core::StageError;

    let builder = ort::session::Session::builder().expect("a session builder needs no model");
    let error = pc_detect::onnx::apply_device_policy(builder, &cuda_policy())
        .err()
        .expect("registration must fail loudly, never fall back to the CPU provider");

    match error {
        StageError::Model(_) => {}
        other => {
            panic!("a registration failure is run-fatal `Model` (§16.36 item 4); got {other:?}")
        }
    }
}
