#![cfg(feature = "onnx")]

use ort::ep::{ExecutionProvider, CUDA};

/// spec §16.22 item 5(d): `ort-sys`'s build-time resolver can silently fall back to its
/// CPU-only `none` distribution. `CUDA::is_available()` enumerates providers COMPILED
/// INTO the linked runtime (no GPU/driver/cuDNN needed) — measured on a machine with
/// neither: `Ok(false)` without `ort/cuda`, `Ok(true)` with it.
#[cfg(feature = "cuda")]
#[test]
fn the_linked_runtime_carries_the_cuda_provider_and_not_the_none_distribution() {
    assert!(
        CUDA::default()
            .is_available()
            .expect("provider enumeration"),
        "ort-sys downgraded to its CPU-only `none` distribution: the linked ONNX Runtime \
         has no CUDAExecutionProvider despite the `cuda` feature being enabled"
    );
}

/// Anti-vacuity control: if `is_available()` reported `true` regardless of which
/// distribution was linked, the guard above would pass on a downgraded build too.
#[cfg(not(feature = "cuda"))]
#[test]
fn the_downgrade_guards_probe_reports_false_when_no_cuda_distribution_is_linked() {
    assert!(
        !CUDA::default()
            .is_available()
            .expect("provider enumeration"),
        "the `none` distribution must not advertise a CUDAExecutionProvider"
    );
}
