//! Task P6 (pc-ocr half) — spec §9.2, §16.8 items 1 and 4. FROZEN.
//!
//! These lock the *mock* contract, because every other Stage-2 test is written against
//! it: if the mock's scripting or routing semantics drift, the P6 filter tests in
//! `pc-preprocess` silently start testing something else.

use image::DynamicImage;
use pc_core::{Language, StageError};
use pc_ocr::{MockOcrEngine, MockOcrFactory, OcrEngine, OcrEngineFactory};

fn crop(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageLuma8(image::GrayImage::new(width, height))
}

// ------------------------------------------------------------ engine contract (§9.2)

#[test]
fn a_default_engine_returns_empty_text_for_every_crop() {
    // spec §9.7(B)11 runs the whole page with a mock that returns "" everywhere.
    let engine = MockOcrEngine::new();

    assert_eq!(engine.recognize(&crop(4, 4)).expect("mock succeeds"), "");
    assert_eq!(engine.recognize(&crop(8, 2)).expect("mock succeeds"), "");
    assert_eq!(engine.calls(), 2);
}

#[test]
fn a_scripted_responses_are_returned_in_call_order_then_fall_back() {
    let engine = MockOcrEngine::new()
        .with_script(["first", "second"])
        .with_response("fallback");

    assert_eq!(engine.recognize(&crop(1, 1)).expect("ok"), "first");
    assert_eq!(engine.recognize(&crop(1, 1)).expect("ok"), "second");
    assert_eq!(engine.recognize(&crop(1, 1)).expect("ok"), "fallback");
    assert_eq!(engine.calls(), 3);
}

#[test]
fn a_engine_records_the_dimensions_of_every_crop_it_saw() {
    let engine = MockOcrEngine::new();

    engine.recognize(&crop(7, 3)).expect("ok");
    engine.recognize(&crop(2, 9)).expect("ok");

    assert_eq!(engine.crops(), vec![(7, 3), (2, 9)]);
}

#[test]
fn a_failing_engine_yields_an_inference_error() {
    // §9.3 step 7 / §14.9: the *stage* turns this into "keep the box"; the engine
    // itself simply reports the failure.
    let engine = MockOcrEngine::new().failing();

    let error = engine.recognize(&crop(4, 4)).expect_err("mock fails");

    assert!(matches!(error, StageError::Inference(_)), "got {error:?}");
    assert_eq!(engine.calls(), 1, "a failing call still counts as a call");
}

#[test]
fn a_languages_are_reported_as_configured() {
    let engine = MockOcrEngine::new().with_languages(vec![Language::Japanese]);

    assert_eq!(engine.languages(), &[Language::Japanese]);
}

// ------------------------------------------------------------ factory routing (§16.8 item 4)

#[test]
fn a_default_factory_offers_its_engine_for_every_language_including_unknown() {
    let factory = MockOcrFactory::default();

    assert!(factory.engine_for(Some(Language::Japanese)).is_some());
    assert!(factory.engine_for(Some(Language::English)).is_some());
    assert!(factory.engine_for(None).is_some());
}

#[test]
fn a_factory_routes_only_the_languages_it_accepts() {
    // spec §16.8 item 4: `None` in the accept list means the *unknown* language, and a
    // language outside the list gets no engine at all.
    let factory =
        MockOcrFactory::new(MockOcrEngine::new()).accepting(vec![Some(Language::Japanese)]);

    assert!(factory.engine_for(Some(Language::Japanese)).is_some());
    assert!(factory.engine_for(Some(Language::English)).is_none());
    assert!(factory.engine_for(None).is_none());

    let unknown_only = MockOcrFactory::new(MockOcrEngine::new()).accepting(vec![None]);

    assert!(unknown_only.engine_for(None).is_some());
    assert!(unknown_only.engine_for(Some(Language::Japanese)).is_none());
}

#[test]
fn a_denying_factory_never_returns_an_engine() {
    let factory = MockOcrFactory::default().denying_all();

    assert!(factory.engine_for(Some(Language::Japanese)).is_none());
    assert!(factory.engine_for(None).is_none());
    assert_eq!(factory.engine().calls(), 0);
}

#[test]
fn a_calls_made_through_the_factory_are_visible_on_the_owned_engine() {
    // The Stage-2 tests assert on `factory.engine().calls()`; that is only meaningful
    // if the factory hands out the engine it owns rather than a clone.
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_response("text"));

    let engine = factory.engine_for(None).expect("engine is offered");
    assert_eq!(engine.recognize(&crop(3, 3)).expect("ok"), "text");

    assert_eq!(factory.engine().calls(), 1);
}

// ------------------------------------------------------------ thread-safety (§4.5)

#[test]
fn a_traits_are_object_safe_and_shareable_across_threads() {
    // §9.2: both traits are `Send + Sync` so §4.5 can share one factory across the
    // rayon pool. This is a compile-time assertion with a runtime smoke test attached.
    fn assert_shared<T: Send + Sync + ?Sized>() {}
    assert_shared::<dyn OcrEngine>();
    assert_shared::<dyn OcrEngineFactory>();

    let factory = MockOcrFactory::new(MockOcrEngine::new().with_response("shared"));
    let shared: &dyn OcrEngineFactory = &factory;

    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(move || {
                let engine = shared.engine_for(Some(Language::Japanese)).expect("engine");
                assert_eq!(engine.recognize(&crop(2, 2)).expect("ok"), "shared");
            });
        }
    });

    assert_eq!(factory.engine().calls(), 4);
}
