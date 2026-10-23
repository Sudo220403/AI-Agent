"""Fault-injection and reliability test suite for the LangGraph Financial Agent."""

from __future__ import annotations

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.agent import run_financial_agent
from src.observability import TelemetryManager
from src.schemas import DataSourceStatus, FinancialFilingData
from src.tools import CircuitBreaker, CircuitBreakerOpenError
from src.tools.market_api import AlphaVantageClient
from src.tools.pdf_parser import generate_sample_10k_pdf


@pytest.fixture(scope="module")
def sample_ibm_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pdf_file = tmp_path_factory.mktemp("filings") / "ibm_10k_2025.pdf"
    return generate_sample_10k_pdf(pdf_file, ticker="IBM")


def test_pydantic_accounting_invariants_reject_hallucinated_numbers() -> None:
    """
    Verify that `FinancialFilingData` strictly rejects LLM outputs that violate
    GAAP Income Statement, Balance Sheet, Cash Flow, or EPS accounting identities.
    """
    with pytest.raises(ValidationError) as exc_info:
        FinancialFilingData(
            company_name="Hallucinated Corp",
            ticker="HALC",
            filing_type="10-K",
            fiscal_year=2025,
            fiscal_period="FY2025",
            revenue_musd=1000.0,
            cost_of_revenue_musd=400.0,
            gross_profit_musd=850.0,  # INVALID: 1000 - 400 = 600, not 850
            operating_income_musd=250.0,
            net_income_musd=200.0,
            diluted_eps=2.00,
            cash_and_equivalents_musd=300.0,
            total_assets_musd=5000.0,  # INVALID: Liabilities (1000) + Equity (1500) = 2500, not 5000
            total_debt_musd=600.0,
            total_liabilities_musd=1000.0,
            total_Shareholders_equity_musd=1500.0,
            operating_cash_flow_musd=280.0,
            capital_expenditures_musd=80.0,
            free_cash_flow_musd=200.0,
            diluted_shares_millions=100.0,
            key_risks=[
                {
                    "category": "Competition",
                    "severity": "HIGH",
                    "summary": "Intense pricing pressure in enterprise cloud segment.",
                }
            ],
            management_guidance_notes="Management expects 8% revenue growth in FY2026.",
        )

    err_text = str(exc_info.value)
    assert "Income Statement Invariant Violation" in err_text
    assert "Balance Sheet Identity Violation" in err_text


def test_langgraph_self_healing_extraction_loop(sample_ibm_pdf: Path) -> None:
    """
    Verify that when Attempt #1 produces corrupted financial numbers,
    `validate_extraction` catches the GAAP violation, routes back to `extract_filing_data`,
    and repairs the extraction on Attempt #2.
    """
    final_state = run_financial_agent(
        pdf_path=str(sample_ibm_pdf),
        ticker_override="IBM",
        inject_corrupted_first_extraction=True,
    )

    assert final_state.get("fatal_error") is None
    report = final_state["final_report"]
    assert report is not None
    assert report.telemetry.extraction_attempts == 2
    assert report.telemetry.self_healing_repairs_triggered == 1
    assert any("Attempt 1 failed validation" in evt for evt in report.telemetry.validation_events)
    assert any("Attempt 2: Self-healing repair succeeded" in evt for evt in report.telemetry.validation_events)


def test_market_api_exponential_backoff_recovery_on_http_429() -> None:
    """
    Verify that `AlphaVantageClient` retries automatically via Tenacity when
    an initial HTTP 429 rate-limit occurs, succeeding on Attempt #2.
    """
    client = AlphaVantageClient(api_key="demo")
    snapshot, retries_used, cb_state = client.fetch_market_snapshot(
        ticker="IBM",
        inject_failure_mode="rate_limit_once",
    )

    assert snapshot.ticker == "IBM"
    assert snapshot.current_price > 0
    assert retries_used >= 1
    assert cb_state == "CLOSED"


def test_circuit_breaker_opens_and_degrades_to_fallback_cache_on_total_outage() -> None:
    """
    Verify that 3 consecutive HTTP 503 failures trip the Circuit Breaker to OPEN
    and gracefully degrade to the verified fallback cache with an audit warning banner.
    """
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=60.0)
    client = AlphaVantageClient(api_key="demo", circuit_breaker=breaker)

    snapshot, retries_used, cb_state = client.fetch_market_snapshot(
        ticker="IBM",
        inject_failure_mode="total_outage",
    )

    assert cb_state == "OPEN"
    assert retries_used == 2  # 3 total attempts = 1 initial + 2 retries
    assert snapshot.source_status == DataSourceStatus.FALLBACK_CACHE
    assert snapshot.warning_banner is not None
    assert "Degraded Mode Active" in snapshot.warning_banner

    # Subsequent call while OPEN should fast-fail with 0 HTTP retries
    with pytest.raises(CircuitBreakerOpenError):
        breaker.before_call()


def test_missing_or_corrupted_pdf_fails_cleanly_with_telemetry() -> None:
    """Verify that a non-existent PDF terminates cleanly at `ingest_pdf` without crashing."""
    final_state = run_financial_agent(pdf_path="data/non_existent_10k.pdf")
    assert final_state.get("fatal_error") is not None
    assert "PDF Ingestion Failure" in final_state["fatal_error"]
    assert final_state.get("final_report") is None


def test_opentelemetry_spans_recorded(sample_ibm_pdf: Path) -> None:
    """Verify that every LangGraph node emits structured OpenTelemetry spans."""
    telemetry = TelemetryManager.get_instance()
    telemetry.jsonl_exporter.clear_memory()

    final_state = run_financial_agent(
        pdf_path=str(sample_ibm_pdf),
        ticker_override="IBM",
        inject_corrupted_first_extraction=True,
        inject_api_failure_mode="rate_limit_once",
    )
    assert final_state.get("final_report") is not None

    span_names = [s["name"] for s in telemetry.jsonl_exporter.exported_spans]
    assert "langgraph.node.ingest_pdf" in span_names
    assert "langgraph.node.extract_filing_data" in span_names
    assert "langgraph.node.validate_extraction" in span_names
    assert "langgraph.node.fetch_market_data" in span_names
    assert "langgraph.node.run_financial_calculations" in span_names
    assert "langgraph.node.synthesize_report" in span_names
    assert "langgraph.node.financial_agent_workflow" in span_names
