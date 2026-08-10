//! GPU-2 (§16.47 item 7, row 1) — §16.36 item 7 bullet 1: "`DeviceSupport::compiled()`
//! becomes conditional on GPU-2's `cuda` feature." GPU-1 deliberately left this
//! unfrozen; this is the assertion that hand-off asked for, in a NEW file so the frozen
//! G1-A gate (`device_policy.rs`) is not touched.
//!
//! Both cfg branches assert. Neither is a skip, so this test can never pass by being
//! compiled out of the arm that matters.

use pc_core::device::{Device, DeviceSupport};

#[test]
fn compiled_support_includes_cuda_exactly_when_the_cuda_feature_is_on() {
    #[cfg(feature = "cuda")]
    {
        assert_eq!(DeviceSupport::compiled(), DeviceSupport::WITH_CUDA);
        assert!(DeviceSupport::compiled().supports(Device::Cuda));
    }
    #[cfg(not(feature = "cuda"))]
    {
        assert_eq!(DeviceSupport::compiled(), DeviceSupport::CPU_ONLY);
        assert!(!DeviceSupport::compiled().supports(Device::Cuda));
    }
    // True in every build, in both arms — restated here so neither arm above can be
    // reduced to a tautology by deleting the other.
    assert!(DeviceSupport::compiled().supports(Device::Cpu));
}
