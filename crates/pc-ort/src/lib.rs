//! `pc-ort` -- the shared `ort` execution-provider boundary (§16.47 item 3).
//!
//! GPU-2 built this inside `pc-detect`; G3-A extracts it into this leaf crate so
//! `pc-ocr` (G3-B/C) and later `pc-inpaint` (GPU-4) can reuse it without duplicating
//! the single most important line in the feature (`.error_on_failure()`) or creating a
//! forbidden stage-crate-to-stage-crate dependency.
/// Scoped MXCSR denormal-flush control (§16.52 item 5's D0). Default tier, no `ort`, no
/// `onnx` feature; `x86_64`-only, matching `crates/pc-ort/tests/denormal_guard.rs`.
#[cfg(target_arch = "x86_64")]
pub mod denormal;

/// The explicit execution-provider registrations a resolved [`DevicePolicy`] calls for
/// (§16.47 item 3).
///
/// Each `ProviderRequest::Cuda` becomes an `ort::ep::CUDA` configured with its pinned
/// convolution-algorithm search mode, then `.error_on_failure()` — countermanding `ort`'s
/// own default (`ExecutionProviderDispatch::new` sets `error_on_failure: false`,
/// documented as "silently fail and fall back to ... the CPU provider"), which §16.22
/// item 5(c) calls "the single most important line in the feature".
///
/// A CPU policy is structurally "no explicit registration" (§16.36 item 2), so an empty
/// `provider_requests()` yields an empty list — a caller registering that list would be
/// an explicit no-op registration, which `apply_device_policy` deliberately avoids.
#[cfg(feature = "onnx")]
pub fn execution_provider_dispatches(policy: &DevicePolicy) -> Vec<ExecutionProviderDispatch> {
    policy
        .provider_requests()
        .iter()
        .map(|request| match request {
            ProviderRequest::Cuda {
                conv_algorithm_search,
            } => CUDA::default()
                .with_conv_algorithm_search(match conv_algorithm_search {
                    pc_core::device::ConvAlgorithmSearch::Heuristic => {
                        ort::ep::cuda::ConvAlgorithmSearch::Heuristic
                    }
                    pc_core::device::ConvAlgorithmSearch::Default => {
                        ort::ep::cuda::ConvAlgorithmSearch::Default
                    }
                })
                .build()
                .error_on_failure(),
        })
        .collect()
}

/// Apply a resolved [`DevicePolicy`]'s execution-provider registrations to a session
/// builder (§16.47 item 3).
///
/// An empty dispatch list returns the builder **unchanged** — `with_execution_providers`
/// is never called with `[]`. This is load-bearing for §16.32's bit-exact CPU floats
/// (§16.36 item 2: a CPU policy is structurally "no explicit registration"). A
/// registration error is run-fatal `StageError::Model`, consistent with every other
/// builder step in `build_session`.
#[cfg(feature = "onnx")]
pub fn apply_device_policy(
    builder: SessionBuilder,
    policy: &DevicePolicy,
) -> Result<SessionBuilder, StageError> {
    let dispatches = execution_provider_dispatches(policy);
    if dispatches.is_empty() {
        return Ok(builder);
    }
    builder
        .with_execution_providers(dispatches)
        .map_err(|error| {
            StageError::Model(format!("failed to configure execution providers: {error}"))
        })
}

#[cfg(feature = "onnx")]
use ort::{
    ep::{ExecutionProviderDispatch, CUDA},
    session::builder::SessionBuilder,
};
#[cfg(feature = "onnx")]
use pc_core::{
    device::{DevicePolicy, ProviderRequest},
    StageError,
};
