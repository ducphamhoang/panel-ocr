//! Task T2 (§16.35 items 7-8) -- masking diagnostics are tracing events only. FROZEN.
//!
//! This captures `tracing` directly rather than asserting pc-cli's stderr formatting: T2
//! belongs to the M6 `pc_mask::run` loop. The structured names below are the frozen interface.

mod common;

use common::{
    box_mask_of, canvas_of, cut_of, memory_input, page, simple_page, text_raw_mask, PAGE_SIZE,
};
use image::{DynamicImage, GrayImage, Luma};
use pc_config::MaskerConfig;
use pc_core::Rect;
use pc_mask::fit::fit_region_scored;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, Once};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Record};
use tracing::{Event, Id, Level, Metadata, Subscriber};

static GLOBAL_SUBSCRIBER: Once = Once::new();

struct EnabledNoop;

impl Subscriber for EnabledNoop {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _span: &Id, _values: &Record<'_>) {}
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    fn event(&self, _event: &Event<'_>) {}
    fn enter(&self, _span: &Id) {}
    fn exit(&self, _span: &Id) {}
    fn max_level_hint(&self) -> Option<tracing::metadata::LevelFilter> {
        Some(tracing::metadata::LevelFilter::TRACE)
    }
}

fn install_global_subscriber() {
    GLOBAL_SUBSCRIBER.call_once(|| {
        let _ = tracing::subscriber::set_global_default(EnabledNoop);
    });
}

const MASKING: Rect = Rect {
    x1: 70,
    y1: 50,
    x2: 130,
    y2: 110,
};
const REFERENCE: Rect = Rect {
    x1: 50,
    y1: 30,
    x2: 150,
    y2: 130,
};

fn ramp_page() -> pc_core::PageData {
    let text = [Rect::new(80, 60, 120, 100)];
    let ramp = GrayImage::from_fn(PAGE_SIZE.0, PAGE_SIZE.1, |x, _| Luma([(x * 5 % 256) as u8]));
    page(
        DynamicImage::ImageLuma8(ramp),
        text_raw_mask(PAGE_SIZE, &text),
        &[MASKING],
        20,
        1.0,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedEvent {
    level: Level,
    fields: BTreeMap<String, String>,
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<CapturedEvent>>>);

impl Capture {
    fn events(&self) -> Vec<CapturedEvent> {
        self.0.lock().expect("capture lock").clone()
    }
}

#[derive(Default)]
struct FieldVisitor(BTreeMap<String, String>);

impl Visit for FieldVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.0.insert(field.name().into(), value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
}

impl Subscriber for Capture {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _span: &Id, _values: &Record<'_>) {}
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        self.0.lock().expect("capture lock").push(CapturedEvent {
            level: *event.metadata().level(),
            fields: visitor.0,
        });
    }
    fn enter(&self, _span: &Id) {}
    fn exit(&self, _span: &Id) {}
    fn max_level_hint(&self) -> Option<tracing::metadata::LevelFilter> {
        Some(tracing::metadata::LevelFilter::TRACE)
    }
}

#[test]
fn a_region_that_fails_even_with_the_rescue_warns_with_greedy_and_lowest_deviations() {
    install_global_subscriber();
    let page = ramp_page();
    let config = MaskerConfig::default();
    let report = fit_region_scored(
        &canvas_of(&page),
        &cut_of(&page),
        &box_mask_of(&page),
        MASKING,
        REFERENCE,
        &config,
    )
    .expect("the ramp reaches fitting");
    let greedy = report.candidate_deviations[report.greedy_index];
    let lowest = report
        .candidate_deviations
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .expect("non-empty ladder");
    assert_ne!(greedy, lowest, "the fixture must distinguish both fields");
    assert!(greedy > config.mask_max_standard_deviation);
    assert!(lowest > config.mask_max_standard_deviation);

    let capture = Capture::default();
    let output = tracing::subscriber::with_default(capture.clone(), || {
        pc_mask::run(memory_input(page, config)).expect("mask stage succeeds")
    });
    assert!(output.mask_data.regions[0].failed);

    let warnings: Vec<_> = capture
        .events()
        .into_iter()
        .filter(|event| event.level == Level::WARN)
        .collect();
    assert_eq!(warnings.len(), 1, "one failed region emits one WARN");
    assert_eq!(
        warnings[0].fields.get("greedy_deviation"),
        Some(&greedy.to_string())
    );
    assert_eq!(
        warnings[0].fields.get("lowest_deviation"),
        Some(&lowest.to_string())
    );
}

#[test]
fn a_successful_region_debugs_the_full_deviation_vector() {
    install_global_subscriber();
    let capture = Capture::default();
    let output = tracing::subscriber::with_default(capture.clone(), || {
        pc_mask::run(memory_input(simple_page(), MaskerConfig::default())).unwrap()
    });
    assert!(!output.mask_data.regions[0].failed);

    let events = capture.events();
    assert!(events.iter().all(|event| event.level != Level::WARN));
    let debug: Vec<_> = events
        .iter()
        .filter(|event| event.level == Level::DEBUG)
        .collect();
    assert_eq!(debug.len(), 1, "one successful region emits one DEBUG");
    assert_eq!(
        debug[0].fields.get("candidate_deviations"),
        Some(&format!("{:?}", vec![0.0_f64; 12]))
    );
}

fn json_keys(value: &serde_json::Value) -> BTreeSet<&str> {
    value
        .as_object()
        .expect("serialized struct is an object")
        .keys()
        .map(String::as_str)
        .collect()
}

#[test]
fn diagnostics_do_not_expand_mask_data_or_analytics_json() {
    install_global_subscriber();
    let output = pc_mask::run(memory_input(simple_page(), MaskerConfig::default())).unwrap();
    let region = serde_json::to_value(&output.mask_data.regions[0]).unwrap();
    let analytic = serde_json::to_value(&output.analytics[0]).unwrap();
    assert_eq!(
        json_keys(&region),
        BTreeSet::from(["failed", "rect", "std_deviation", "thickness"])
    );
    assert_eq!(
        json_keys(&analytic),
        BTreeSet::from([
            "candidate_index",
            "fit_found",
            "path",
            "std_deviation",
            "thickness",
        ])
    );
}
