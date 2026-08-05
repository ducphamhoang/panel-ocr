//! G1-A tests — spec §16.36 item 2 (`pc_core::device`), §16.22 items 3 and 5(a)–(f).
//! Frozen gates.
//!
//! Everything here is pure: no model, no ONNX Runtime, no `ort` types, no `cuda`
//! feature. These tests must compile and pass in the default (non-`onnx`) tier, which
//! is the whole point of §16.36 item 2's "pure, testable with no model and no ONNX
//! Runtime" placement ruling.
//!
//! Deliberately NOT asserted, per §16.36 item 7: `DeviceSupport::compiled() ==
//! CPU_ONLY`. GPU-2 makes `compiled()` conditional on its `cuda` feature, and freezing
//! that equality here would hand GPU-2 a frozen-test conflict. Only the
//! capability-explicit `CPU_ONLY` / `WITH_CUDA` constructors are frozen, plus the one
//! property of `compiled()` that holds in every build forever (CPU is always available).
//!
//! Shapes these tests assume, matching the crate's existing idioms:
//!   * `Device::ALL: &'static [Device]` (as `Output::ALL` in `src/output.rs`)
//!   * `DevicePolicy::provider_requests(&self) -> &[ProviderRequest]` -- the resolved
//!     list of EXPLICIT registrations, empty for cpu. `ProviderRequest` has exactly one
//!     variant, `Cuda { conv_algorithm_search }`; there is no `Cpu` variant, by §16.36
//!     item 2. This accessor is the only addition to item 2's sketch, and it exists
//!     because §16.22 item 3's "our session options must pin `Heuristic`" is otherwise
//!     unobservable from a test.
//!   * `Device`, `DeviceSupport`, `DevicePolicy`, `DeviceRefusal`, `ConvAlgorithmSearch`
//!     and `ProviderRequest` are all `Debug + PartialEq`; the first four are `Copy`
//!     except `DevicePolicy`, which these tests only ever borrow or clone-by-value.

use pc_core::device::{
    resolve, ConvAlgorithmSearch, Device, DevicePolicy, DeviceRefusal, DeviceSupport,
    ProviderRequest,
};

// --------------------------------------------------------------------- Device

#[test]
// spec §16.22 item 5(a) / §16.36 item 1: `device = "cpu"` is the default; `"cuda"` must
// be explicit. The typed default is the other half of the shipped TOML default.
fn device_defaults_to_cpu() {
    assert_eq!(Device::default(), Device::Cpu);
}

#[test]
// spec §16.36 item 2: `Device::ALL = [Cpu, Cuda]`. Both the identity (which variants,
// in which order) and the cardinality are asserted — a set assertion catches a swap, a
// count catches an addition, and neither catches what the other does. The `2` is a
// hard-coded literal that cannot be computed from the enum.
fn device_all_is_exactly_cpu_then_cuda() {
    assert_eq!(Device::ALL, &[Device::Cpu, Device::Cuda]);
    assert_eq!(Device::ALL.len(), 2);
}

#[test]
// spec §16.36 item 2 ("serde snake_case") + item 5, which replaces `pc-testkit`'s
// independently-typed `"cpu"` literal with `Device::Cpu.as_str()`: the wire spelling and
// `as_str` must be the same two strings, so the provenance field and the config text
// cannot drift apart. Both are hard-coded here, not derived from each other.
fn device_wire_spelling_and_as_str_agree_on_frozen_strings() {
    assert_eq!(serde_json::to_string(&Device::Cpu).unwrap(), r#""cpu""#);
    assert_eq!(serde_json::to_string(&Device::Cuda).unwrap(), r#""cuda""#);
    assert_eq!(Device::Cpu.as_str(), "cpu");
    assert_eq!(Device::Cuda.as_str(), "cuda");

    assert_eq!(
        serde_json::from_str::<Device>(r#""cpu""#).unwrap(),
        Device::Cpu
    );
    assert_eq!(
        serde_json::from_str::<Device>(r#""cuda""#).unwrap(),
        Device::Cuda
    );
}

#[test]
// spec §16.36 item 2: the documented set is closed — a spelling outside it is a
// deserialization error, not a silent fallback to `Cpu` (which would make the opt-in
// invisible, §14 item 7).
fn undocumented_device_spellings_are_rejected_by_serde() {
    for bad in [
        r#""mps""#,
        r#""directml""#,
        r#""coreml""#,
        r#""CPU""#,
        r#""gpu""#,
    ] {
        assert!(
            serde_json::from_str::<Device>(bad).is_err(),
            "{bad} must not deserialize into a Device"
        );
    }
}

// -------------------------------------------------------------- DeviceSupport

#[test]
// spec §16.36 item 2: `DeviceSupport` is a VALUE describing which devices this build can
// register, not a `cfg!` read — which is what makes the CUDA-available branch testable in
// a build with no `cuda` feature at all.
fn capability_constants_report_exactly_what_they_name() {
    assert!(DeviceSupport::CPU_ONLY.supports(Device::Cpu));
    assert!(!DeviceSupport::CPU_ONLY.supports(Device::Cuda));
    assert!(DeviceSupport::WITH_CUDA.supports(Device::Cpu));
    assert!(DeviceSupport::WITH_CUDA.supports(Device::Cuda));
    assert_ne!(DeviceSupport::CPU_ONLY, DeviceSupport::WITH_CUDA);
}

#[test]
// spec §16.36 item 2 + §16.22 item 5(e): ONNX Runtime's CPU provider is built in, so
// every build can always run on CPU. This is the one property of `compiled()` that is
// true in GPU-1 and stays true after GPU-2 flips it to be `cuda`-conditional (§16.36
// item 7), so it is safe to freeze while `compiled() == CPU_ONLY` is not.
fn the_compiled_support_always_includes_cpu() {
    assert!(DeviceSupport::compiled().supports(Device::Cpu));
}

// ------------------------------------------------------------------- resolve

#[test]
// spec §16.22 item 5(a) "Opt-in only … Never auto-detected": a `cpu` REQUEST resolves to
// a cpu policy even in a build where CUDA IS available. This is the test that fails if
// the resolver ever "upgrades" a request from build capability — the exact behaviour
// §16.36 item 1 records upstream doing (`ctd_interface.py:64`) and DEVIATION(22)
// deliberately diverges from.
fn a_cpu_request_stays_on_cpu_even_when_cuda_is_available() {
    let policy =
        resolve(Device::Cpu, DeviceSupport::WITH_CUDA).expect("a cpu request is always resolvable");

    assert_eq!(policy.requested(), Device::Cpu);
    assert_eq!(policy.provenance_execution_provider(), "cpu");
    // exact-slice equality, not a count: it pins BOTH that nothing is registered and
    // (with the cuda test below) that the two policies are not the same list.
    assert_eq!(policy.provider_requests(), &[] as &[ProviderRequest]);
    assert!(!policy.registration_must_error_on_failure());
    assert_eq!(policy, DevicePolicy::cpu());
}

#[test]
// spec §16.36 item 2: the same request in a CPU-only build resolves identically — the
// resolved policy is a function of the REQUEST for `cpu`, not of the capability value.
fn a_cpu_request_resolves_on_a_cpu_only_build() {
    let policy =
        resolve(Device::Cpu, DeviceSupport::CPU_ONLY).expect("a cpu request is always resolvable");

    assert_eq!(policy.requested(), Device::Cpu);
    assert_eq!(policy.provenance_execution_provider(), "cpu");
    assert_eq!(policy.provider_requests(), &[] as &[ProviderRequest]);
    assert!(!policy.registration_must_error_on_failure());
    assert_eq!(policy, DevicePolicy::cpu());
}

#[test]
// spec §16.36 item 2 + §16.22 item 5(c): requesting a device this build cannot register
// is a refusal that names the requested device — not a silent fall back to CPU, which is
// `ort`'s own default behaviour and the single thing §16.22 item 5(c) exists to countermand.
fn a_cuda_request_on_a_cpu_only_build_is_refused_naming_the_device() {
    let refusal = resolve(Device::Cuda, DeviceSupport::CPU_ONLY)
        .expect_err("cuda is not registrable in a cpu-only build");

    assert_eq!(
        refusal,
        DeviceRefusal::NotCompiledIn {
            requested: Device::Cuda
        }
    );
}

#[test]
// spec §16.22 items 5(c) and 3: a resolvable cuda request produces a policy that (1)
// requests exactly one explicit provider, (2) demands error-on-failure so a failed
// registration cannot degrade into a silent ~150 s/page CPU run, and (3) pins
// `ConvAlgorithmSearch::Heuristic` specifically. `Exhaustive` is unrepresentable in the
// type, so the live risk this asserts is the OTHER in-range choice, `Default`.
fn a_cuda_request_resolves_to_one_error_on_failure_heuristic_registration() {
    let policy = resolve(Device::Cuda, DeviceSupport::WITH_CUDA)
        .expect("cuda is registrable when the build supports it");

    assert_eq!(policy.requested(), Device::Cuda);
    assert_eq!(policy.provenance_execution_provider(), "cuda");
    assert!(policy.registration_must_error_on_failure());
    assert_eq!(
        policy.provider_requests(),
        &[ProviderRequest::Cuda {
            conv_algorithm_search: ConvAlgorithmSearch::Heuristic
        }]
    );
    assert_eq!(policy.provider_requests().len(), 1);
    assert_ne!(policy, DevicePolicy::cpu());
}

#[test]
// spec §16.36 item 2: "a CPU policy is structurally 'no explicit registration'" — there
// is deliberately no `ProviderRequest::Cpu`, because registering an explicit CPU provider
// could perturb §16.32's bit-exact recorded floats. Asserted as a property of every
// resolvable request rather than of one hand-picked policy, so a future `Cpu` variant
// cannot sneak in behind a passing suite.
fn only_a_non_cpu_request_asks_for_an_explicit_provider() {
    for device in Device::ALL {
        let Ok(policy) = resolve(*device, DeviceSupport::WITH_CUDA) else {
            panic!("WITH_CUDA must resolve every Device::ALL variant; {device:?} did not");
        };
        assert_eq!(
            policy.provider_requests().is_empty(),
            *device == Device::Cpu,
            "{device:?}: explicit provider registration must happen for non-cpu devices only"
        );
        // §16.36 item 2: error-on-failure is true exactly when something is registered.
        assert_eq!(
            policy.registration_must_error_on_failure(),
            !policy.provider_requests().is_empty(),
            "{device:?}: error-on-failure must track explicit registration"
        );
    }
}

#[test]
// spec §16.22 item 6 / §16.36 item 5: the provenance string is DERIVED from the resolved
// policy, never a second independently-typed literal. The literals above ("cpu"/"cuda")
// are the frozen expectations; this loop additionally proves the derivation holds for
// every variant, so a variant added later cannot get a hand-written provenance string.
fn provenance_execution_provider_is_derived_for_every_device() {
    for device in Device::ALL {
        let policy = resolve(*device, DeviceSupport::WITH_CUDA).expect("resolvable");
        assert_eq!(
            policy.provenance_execution_provider(),
            device.as_str(),
            "{device:?}"
        );
    }
}

// -------------------------------------------------------------- user-facing text

#[test]
// spec §16.36 items 2 and 4: this exact string is what a user reads on stderr as
// `error: model error: <message>` (exit 1) when they set `device = "cuda"` on a build
// without CUDA. Frozen verbatim, matching the shipped `ONNX_UNAVAILABLE` idiom in
// `crates/pc-cli/src/{detector,ocr}.rs`: what is unavailable, why, and both ways out
// (switch the key, or rebuild with the feature).
fn the_not_compiled_in_refusal_message_is_frozen() {
    let refusal = DeviceRefusal::NotCompiledIn {
        requested: Device::Cuda,
    };
    assert_eq!(
        refusal.message(),
        "the cuda execution provider is not available in this build (panel-ocr was \
compiled without the `cuda` feature, so no CUDA execution provider is linked in). \
Rebuild with `--features cuda`, or set `device = \"cpu\"` under `[general]` in your \
profile to run on the CPU execution provider."
    );
}

#[test]
// spec §16.22 item 5(e): "Report what was requested and what registered — never per-node
// placement." `ort` rc.12 exposes no session provider list and no placement
// introspection, so a report claiming "we are on GPU" would be unfounded. Both strings
// are frozen verbatim because §16.36 item 5 quotes `xtask bench`'s report into spec
// sections, where a reworded line silently changes the record.
fn policy_reports_name_the_request_and_disclaim_per_node_placement() {
    assert_eq!(
        DevicePolicy::cpu().report(),
        "device: requested cpu; no execution provider registered explicitly (ONNX \
Runtime's built-in CPU provider is implicit); per-node operator placement is not claimed"
    );
    assert_eq!(
        resolve(Device::Cuda, DeviceSupport::WITH_CUDA)
            .unwrap()
            .report(),
        "device: requested cuda; execution providers registered explicitly: cuda (conv \
algorithm search: heuristic, error on registration failure); per-node operator placement \
is not claimed"
    );
}

#[test]
// spec §16.22 item 5(e), stated as a property rather than as prose: no report may claim
// per-node placement, and every report must name the device that was requested.
fn every_report_names_its_device_and_carries_the_per_node_disclaimer() {
    for device in Device::ALL {
        let report = resolve(*device, DeviceSupport::WITH_CUDA).unwrap().report();
        assert!(
            report.contains(device.as_str()),
            "{device:?}: report must name the requested device: {report}"
        );
        assert!(
            report.contains("per-node"),
            "{device:?}: report must disclaim per-node placement: {report}"
        );
    }
}
