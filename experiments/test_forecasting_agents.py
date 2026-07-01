from experiments.forecasting_agents.fixtures import SAMPLE_EVIDENCE, SAMPLE_QUESTIONS
from experiments.forecasting_agents.forecasters import (
    AIABaselineForecaster,
    AdvancedForecastLab,
    EvidenceGraphForecaster,
    MarketOnlyForecaster,
    StructuredEvidenceGraphForecaster,
    source_prior,
)
from experiments.forecasting_agents.forecastbench_adapter import (
    horizon_adjusted_prior,
    horizon_bucket,
    load_forecaster_medians,
    parse_market_probability,
    source_base_prior,
    structured_time_series_evidence,
    target_key,
)
from experiments.forecasting_agents.metrics import brier_score, summarize_results
from experiments.forecasting_agents.historical_timeseries_forecastbench import historical_probability, parse_dbnomics_id, parse_yfinance_chart
from experiments.forecasting_agents.search_llm_forecaster import (
    HeuristicReasoner,
    HybridAnalogSearchForecaster,
    LocalEvidenceSearchProvider,
    SearchEnabledLLMForecaster,
    plan_search_queries,
    _parse_reasoner_response,
    retrieved_historical_probability,
)
from experiments.forecasting_agents.synthetic_benchmark import generate_synthetic_dataset
from datetime import datetime, timedelta


def test_brier_score():
    assert brier_score(0.75, 1) == 0.0625
    assert brier_score(0.25, 0) == 0.0625


def test_all_forecasters_return_valid_probabilities():
    forecasters = [MarketOnlyForecaster(), AIABaselineForecaster(), EvidenceGraphForecaster(), StructuredEvidenceGraphForecaster(), AdvancedForecastLab()]
    for question in SAMPLE_QUESTIONS:
        evidence = SAMPLE_EVIDENCE[question.id]
        for forecaster in forecasters:
            result = forecaster.forecast(question, evidence)
            assert 0.01 <= result.scored_probability() <= 0.99
            assert 0.0 < result.confidence <= 0.95
            assert result.question_id == question.id
            assert result.system == forecaster.name


def test_summary_contains_each_system():
    results = []
    for forecaster in [AIABaselineForecaster(), EvidenceGraphForecaster(), AdvancedForecastLab()]:
        for question in SAMPLE_QUESTIONS:
            results.append(forecaster.forecast(question, SAMPLE_EVIDENCE[question.id]))
    outcomes = {question.id: int(question.outcome) for question in SAMPLE_QUESTIONS}
    summary = summarize_results(results, outcomes)
    assert set(summary) == {"aia_baseline", "evidence_graph_v1", "advanced_v1"}
    for row in summary.values():
        assert row["n"] == 3.0
        assert row["brier"] >= 0.0


def test_synthetic_generator_is_deterministic():
    questions_a, evidence_a = generate_synthetic_dataset(n=8, seed=123)
    questions_b, evidence_b = generate_synthetic_dataset(n=8, seed=123)
    assert [question.outcome for question in questions_a] == [question.outcome for question in questions_b]
    assert [question.market_probability for question in questions_a] == [question.market_probability for question in questions_b]
    assert [item.weight for item in evidence_a[questions_a[0].id]] == [item.weight for item in evidence_b[questions_b[0].id]]


def test_forecastbench_market_probability_parser():
    assert parse_market_probability({"source": "manifold", "freeze_datetime_value": "0.73", "freeze_datetime_value_explanation": "The market value."}) == 0.73
    assert parse_market_probability({"source": "fred", "freeze_datetime_value": "5.23", "freeze_datetime_value_explanation": "The latest value."}) is None
    assert target_key("abc", None) == "abc::NA"
    assert target_key("abc", "2025-01-01") == "abc::2025-01-01"


def test_forecastbench_horizon_prior_adjustment():
    assert horizon_bucket("2024-07-21", "2024-08-15") == "near"
    assert horizon_bucket("2024-07-21", "2024-11-01") == "short"
    assert horizon_bucket("2024-07-21", "2025-06-01") == "mid"
    assert horizon_bucket("2024-07-21", "2026-01-01") == "long"
    assert horizon_bucket("bad-date", "2025-01-01") == "unknown"

    yfinance_question = {"source": "yfinance", "question": "", "background": ""}
    dbnomics_question = {"source": "dbnomics", "question": "", "background": ""}
    assert horizon_adjusted_prior(yfinance_question, "2024-07-21", "2024-11-01") > source_base_prior(yfinance_question)
    assert horizon_adjusted_prior(dbnomics_question, "2024-07-21", "2024-11-01") < source_base_prior(dbnomics_question)


def test_source_prior_prefers_source_horizon_key():
    question = SAMPLE_QUESTIONS[0]
    question = type(question)(**{**question.__dict__, "metadata": {"forecastbench_source": "fred", "horizon_bucket": "mid"}})
    assert source_prior(question, {"fred": 0.4, "fred::mid": 0.7}) == 0.7
    assert source_prior(question, {"fred": 0.4}) == 0.4
    assert source_prior(question, {}) is None


def test_structured_timeseries_evidence_for_forecastbench_sources():
    db_question = type(SAMPLE_QUESTIONS[0])(
        **{
            **SAMPLE_QUESTIONS[0].__dict__,
            "id": "db::2026-01-01",
            "resolution_date": "2026-01-01",
            "metadata": {"forecastbench_source": "dbnomics", "horizon_bucket": "short"},
        }
    )
    db_evidence = structured_time_series_evidence(
        db_question,
        {"source": "dbnomics", "question": "Will daily average temperature be higher?", "freeze_datetime_value": "14.0"},
    )
    assert db_evidence
    assert db_evidence[0].stance == "opposes"
    assert "recent" in db_evidence[0].tags

    yf_question = type(SAMPLE_QUESTIONS[0])(
        **{
            **SAMPLE_QUESTIONS[0].__dict__,
            "id": "yf::2026-01-01",
            "resolution_date": "2026-01-01",
            "metadata": {"forecastbench_source": "yfinance", "horizon_bucket": "short"},
        }
    )
    yf_evidence = structured_time_series_evidence(
        yf_question,
        {"source": "yfinance", "question": "Will the market close price be higher?", "freeze_datetime_value": "100.0"},
    )
    assert yf_evidence
    assert yf_evidence[0].stance == "supports"


def test_historical_analog_probability_uses_past_horizon_pairs():
    history = [(datetime(2020, 1, 1) + timedelta(days=day), float(day)) for day in range(80)]
    probability = historical_probability(history, "2020-03-01", "2020-03-11", 60.0)
    assert probability is not None
    assert probability > 0.7
    assert parse_dbnomics_id("meteofrance_TEMPERATURE_celsius.07607.D") == (
        "meteofrance",
        "TEMPERATURE",
        "celsius.07607.D",
    )


def test_parse_yfinance_chart_payload():
    payload = '{"chart":{"result":[{"timestamp":[1609459200,1609545600],"indicators":{"quote":[{"close":[10.0,null]}]}}]}}'
    rows = parse_yfinance_chart(payload)
    assert len(rows) == 1
    assert rows[0][1] == 10.0


def test_search_llm_loop_retrieves_and_forecasts():
    question = SAMPLE_QUESTIONS[0]
    question = type(question)(
        **{
            **question.__dict__,
            "metadata": {
                **question.metadata,
                "forecast_due_date": "2026-01-01",
                "base_prior": 0.45,
            },
        }
    )
    queries = plan_search_queries(question)
    assert {query.purpose for query in queries} == {"base_rate", "recent", "resolution", "counterevidence"}

    provider = LocalEvidenceSearchProvider({question.id: 0.72})
    docs = provider.search(question, SAMPLE_EVIDENCE[question.id], queries[0])
    assert any(doc.source == "historical_analog" for doc in docs)
    assert retrieved_historical_probability(docs) == 0.72

    forecaster = SearchEnabledLLMForecaster(search_provider=provider, reasoner=HeuristicReasoner())
    result = forecaster.forecast(question, SAMPLE_EVIDENCE[question.id])
    assert result.system == "search_llm_loop"
    assert 0.01 <= result.scored_probability() <= 0.99
    assert result.diagnostics["query_count"] == 4
    assert result.diagnostics["document_count"] >= 1
    assert result.diagnostics["historical_analog_probability"] == 0.72
    assert result.diagnostics["analog_gate_fired"] is True
    assert result.scored_probability() == 0.72

    ungated = SearchEnabledLLMForecaster(search_provider=provider, reasoner=HeuristicReasoner(), analog_gate=False)
    ungated_result = ungated.forecast(question, SAMPLE_EVIDENCE[question.id])
    assert ungated_result.diagnostics["analog_gate_fired"] is False
    assert ungated_result.scored_probability() != 0.72

    hybrid = HybridAnalogSearchForecaster(search_provider=provider, reasoner=HeuristicReasoner())
    hybrid_result = hybrid.forecast(question, SAMPLE_EVIDENCE[question.id])
    assert hybrid_result.system == "hybrid_analog_search"
    assert hybrid_result.scored_probability() == 0.72
    assert hybrid_result.diagnostics["gate_decision"] == "historical_analog"


def test_reasoner_response_parser_recovers_probability_from_malformed_json():
    parsed = _parse_reasoner_response('{"probability": 0.42, "confidence": 0.6, "rationale": "unterminated}')
    assert parsed["probability"] == 0.42
    assert parsed["confidence"] == 0.6


def test_load_forecaster_medians(tmp_path):
    path = tmp_path / "forecasts.json"
    path.write_text(
        '{"forecasts":[{"id":"a","forecast":0.2,"resolution_date":"2025-01-01"},{"id":"a","forecast":0.8,"resolution_date":"2025-01-01"},{"id":"b","forecast":0.4}]}',
        encoding="utf-8",
    )
    medians = load_forecaster_medians(path)
    assert medians["a::2025-01-01"] == 0.5
    assert medians["b::NA"] == 0.4
