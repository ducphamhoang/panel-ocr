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

/// A pipeline stage with a device policy that may or may not have a ratified path for a
/// requested device (§16.47 item 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    Ocr,
    Inpaint,
}

impl Stage {
    /// The lowercase wire spelling used in refusal messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ocr => "ocr",
            Self::Inpaint => "inpaint",
        }
    }
}

/// Why a requested device could not be resolved for this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRefusal {
    NotCompiledIn { requested: Device },
    NoRatifiedStagePath { requested: Device, stage: Stage },
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
            Self::NoRatifiedStagePath {
                requested: Device::Cuda,
                stage,
            } => {
                // Per-stage citation (§16.47 item 4's ratification-review correction: OCR
                // and LaMa do NOT share one citation) and enabled-flag path, keyed on
                // `stage` rather than duplicated as a second hard-coded name —
                // `Stage::as_str()` is the single source for the wire spelling.
                let (citation, flag, section) = match stage {
                    Stage::Ocr => ("§16.36 item 3", "ocr_enabled", "[preprocessor]"),
                    Stage::Inpaint => ("§16.38 item 15(d)", "inpainting_enabled", "[inpainter]"),
                };
                format!(
                    "the cuda execution provider is compiled into this build, but no CUDA path is ratified for the {} stage ({citation} / §16.47 item 1: CUDA has only been measured for the text detector). Set `{flag} = false` under `{section}` to run the detector on CUDA, or set device = \"cpu\" to run every stage on the CPU execution provider.",
                    stage.as_str()
                )
            }
            // `ensure_stage_supports` never constructs this arm (a Cpu request always has
            // a ratified path — see its match below) — but `DeviceRefusal`'s fields are
            // public, so a caller could construct one directly. No live panic risk: a
            // plain, honest message rather than `unreachable!()`.
            Self::NoRatifiedStagePath {
                requested: Device::Cpu,
                stage,
            } => format!(
                "the cpu execution provider always has a ratified path for the {} stage; this refusal should never have been constructed",
                stage.as_str()
            ),
        }
    }
}

/// Refuse a stage whose requested device resolves in this build but has no ratified path
/// for the stage (§16.47 item 4). Called at a stage seam AFTER `resolve(...)` — in a
/// non-`cuda` build `resolve` already refuses with the frozen `NotCompiledIn`, so this
/// only fires in a build that CAN register the requested device.
pub fn ensure_stage_supports(policy: &DevicePolicy, stage: Stage) -> Result<(), DeviceRefusal> {
    match policy.requested() {
        Device::Cpu => Ok(()),
        Device::Cuda => Err(DeviceRefusal::NoRatifiedStagePath {
            requested: Device::Cuda,
            stage,
        }),
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
