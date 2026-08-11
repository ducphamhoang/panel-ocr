//! Device selection and execution-provider policy (§16.22, §16.36).

use serde::{Deserialize, Serialize};

/// A device explicitly requested by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    #[default]
    Cpu,
    Cuda,
}

impl Device {
    pub const ALL: &'static [Device] = &[Device::Cpu, Device::Cuda];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

/// The execution providers this build can register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceSupport {
    cpu: bool,
    cuda: bool,
}

impl DeviceSupport {
    pub const CPU_ONLY: Self = Self {
        cpu: true,
        cuda: false,
    };

    pub const WITH_CUDA: Self = Self {
        cpu: true,
        cuda: true,
    };

    pub const fn compiled() -> Self {
        if cfg!(feature = "cuda") {
            Self::WITH_CUDA
        } else {
            Self::CPU_ONLY
        }
    }

    pub const fn supports(self, device: Device) -> bool {
        match device {
            Device::Cpu => self.cpu,
            Device::Cuda => self.cuda,
        }
    }
}

/// The convolution algorithm search policy for an explicit CUDA provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvAlgorithmSearch {
    Heuristic,
    Default,
}

/// An explicit execution-provider registration request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderRequest {
    Cuda {
        conv_algorithm_search: ConvAlgorithmSearch,
    },
}

/// The resolved, model-agnostic device policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePolicy {
    requested: Device,
    provider_requests: Vec<ProviderRequest>,
}

impl DevicePolicy {
    pub fn cpu() -> Self {
        Self {
            requested: Device::Cpu,
            provider_requests: Vec::new(),
        }
    }

    pub const fn requested(&self) -> Device {
        self.requested
    }

    pub fn provider_requests(&self) -> &[ProviderRequest] {
        &self.provider_requests
    }

    pub fn registration_must_error_on_failure(&self) -> bool {
        !self.provider_requests().is_empty()
    }

    pub const fn provenance_execution_provider(&self) -> &'static str {
        self.requested.as_str()
    }

    pub fn report(&self) -> String {
        let requested = self.requested.as_str();
        if self.provider_requests.is_empty() {
            return format!(
                "device: requested {requested}; no execution provider registered explicitly \
                 (ONNX Runtime's built-in CPU provider is implicit); per-node operator placement \
                 is not claimed"
            );
        }

        let registered = self
            .provider_requests
            .iter()
            .map(|request| match request {
                ProviderRequest::Cuda {
                    conv_algorithm_search,
                } => {
                    let search = match conv_algorithm_search {
                        ConvAlgorithmSearch::Heuristic => "heuristic",
                        ConvAlgorithmSearch::Default => "default",
                    };
                    format!("cuda (conv algorithm search: {search}, error on registration failure)")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "device: requested {requested}; execution providers registered explicitly: \
             {registered}; per-node operator placement is not claimed"
        )
    }
}

/// Why a requested device could not be resolved for this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRefusal {
    NotCompiledIn { requested: Device },
}

impl DeviceRefusal {
    pub fn message(&self) -> String {
        match self {
            Self::NotCompiledIn {
                requested: Device::Cuda,
            } => "the cuda execution provider is not available in this build (panel-ocr was compiled without the `cuda` feature, so no CUDA execution provider is linked in). Rebuild with `--features cuda`, or set `device = \"cpu\"` under `[general]` in your profile to run on the CPU execution provider.".to_owned(),
            Self::NotCompiledIn {
                requested: Device::Cpu,
            } => "the cpu execution provider is not available in this build".to_owned(),
        }
    }
}

pub fn resolve(requested: Device, support: DeviceSupport) -> Result<DevicePolicy, DeviceRefusal> {
    if !support.supports(requested) {
        return Err(DeviceRefusal::NotCompiledIn { requested });
    }

    Ok(match requested {
        Device::Cpu => DevicePolicy::cpu(),
        Device::Cuda => DevicePolicy {
            requested,
            provider_requests: vec![ProviderRequest::Cuda {
                conv_algorithm_search: ConvAlgorithmSearch::Heuristic,
            }],
        },
    })
}
