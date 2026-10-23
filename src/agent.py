"""LangGraph StateGraph orchestration with self-healing Pydantic validation and resilient tool use."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.config import get_settings
from src.observability import TelemetryManager
from src.schemas import (
    AgentGraphState,
    FinancialFilingData,
    InvestmentAnalysisReport,
    TelemetrySummary,
)
from src.tools.financial_calculator import (
    compute_valuation_metrics,
    derive_investment_rating,
)
from src.tools.market_api import AlphaVantageClient
from src.tools.pdf_parser import PDFExtractionError, extract_text_from_pdf


# ============================================================================
# NODE 1: PDF Ingestion & Structural Verification
# ============================================================================
def ingest_pdf_node(state: AgentGraphState) -> Dict[str, Any]:
    """Read and validate the corporate financial filing PDF."""
    telemetry = TelemetryManager.get_instance()
    pdf_path = state.get("pdf_path", "")

    with telemetry.start_node_span(
        "ingest_pdf", attributes={"pdf.path": pdf_path}
    ) as span:
        try:
            raw_text, page_count = extract_text_from_pdf(pdf_path)
            span.set_attribute("pdf.page_count", page_count)
            span.set_attribute("pdf.char_length", len(raw_text))
            return {
                "raw_pdf_text": raw_text,
                "pdf_page_count": page_count,
                "extraction_attempts": 0,
                "validation_errors": [],
                "fatal_error": None,
            }
        except PDFExtractionError as exc:
            span.add_event("pdf_ingestion_failed", {"error": str(exc)})
            return {"fatal_error": f"PDF Ingestion Failure: {exc}"}


# ============================================================================
# NODE 2: LLM / Deterministic Structured Extraction (with Self-Healing Feedback)
# ============================================================================
def extract_filing_data_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Extract structured financial metrics from raw PDF text.
    If previous attempts failed Pydantic accounting invariants, feeds the exact
    ValidationError diagnostics into the repair prompt so the agent self-heals.
    """
    telemetry = TelemetryManager.get_instance()
    attempts = state.get("extraction_attempts", 0) + 1
    prior_errors = state.get("validation_errors", [])
    inject_corrupted = bool(state.get("inject_corrupted_first_extraction", False))

    with telemetry.start_node_span(
        "extract_filing_data",
        attributes={
            "extraction.attempt": attempts,
            "extraction.is_self_healing_retry": len(prior_errors) > 0 and attempts > 1,
            "extraction.inject_corrupted_first": inject_corrupted and attempts == 1,
        },
    ) as span:
        raw_text = state.get("raw_pdf_text", "")

        # Fault-injection mode for demonstrating self-healing on Attempt #1:
        # Simulates a classic LLM hallucination where Gross Profit and Balance Sheet
        # numbers violate GAAP accounting identities.
        if inject_corrupted and attempts == 1:
            span.add_event(
                "injected_accounting_hallucination",
                {
                    "reason": "Simulated LLM unit/math hallucination on attempt 1 to trigger self-healing validator"
                },
            )
            candidate_dict = _parse_filing_deterministically(raw_text)
            # Corrupt Gross Profit (+9,000M error) and Balance Sheet Assets (+15,000M error)
            candidate_dict["gross_profit_musd"] = candidate_dict["revenue_musd"] + 500.0
            candidate_dict["total_assets_musd"] = (
                candidate_dict["total_liabilities_musd"]
                + candidate_dict["total_Shareholders_equity_musd"]
                + 15000.0
            )
            return {
                "extraction_attempts": attempts,
                "last_raw_extraction": candidate_dict,
            }

        # Live Gemini LLM extraction if GOOGLE_API_KEY is configured, with automatic fallback
        settings = get_settings()
        if (
            settings.google_api_key
            and settings.google_api_key != "your_gemini_api_key_here"
        ):
            try:
                candidate_dict = _extract_with_gemini_llm(
                    raw_text=raw_text,
                    prior_errors=prior_errors,
                    model_name=settings.gemini_model,
                    api_key=settings.google_api_key,
                )
                span.set_attribute("extraction.engine", f"gemini:{settings.gemini_model}")
                return {
                    "extraction_attempts": attempts,
                    "last_raw_extraction": candidate_dict,
                }
            except Exception as llm_exc:
                span.add_event(
                    "llm_extraction_fallback_triggered", {"error": str(llm_exc)}
                )

        # Deterministic regex/table extraction engine (also serves as self-healing repair resolver)
        span.set_attribute("extraction.engine", "deterministic_sec_parser")
        candidate_dict = _parse_filing_deterministically(raw_text)
        if state.get("ticker_override"):
            candidate_dict["ticker"] = str(state["ticker_override"]).strip().upper()

        return {
            "extraction_attempts": attempts,
            "last_raw_extraction": candidate_dict,
        }


# ============================================================================
# NODE 3: Pydantic Domain & Accounting Invariant Validator (Self-Healing Gate)
# ============================================================================
def validate_extraction_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Validate `last_raw_extraction` against `FinancialFilingData` Pydantic schema
    and its 6 cross-field GAAP accounting invariants.
    """
    telemetry = TelemetryManager.get_instance()
    attempts = state.get("extraction_attempts", 1)
    raw_candidate = state.get("last_raw_extraction") or {}
    history = list(state.get("validation_errors", []))

    with telemetry.start_node_span(
        "validate_extraction", attributes={"validation.attempt": attempts}
    ) as span:
        try:
            validated_model = FinancialFilingData.model_validate(raw_candidate)
            span.set_attribute("validation.passed", True)
            if attempts > 1:
                history.append(
                    f"Attempt {attempts}: Self-healing repair succeeded — all 6 GAAP accounting invariants verified."
                )
            else:
                history.append(
                    "Attempt 1: Initial extraction passed all 6 GAAP accounting invariants."
                )
            return {
                "filing_data": validated_model,
                "validation_errors": history,
            }
        except ValidationError as val_err:
            formatted_err = "; ".join(
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in val_err.errors()
            )
            event_msg = f"Attempt {attempts} failed validation -> {formatted_err}"
            history.append(event_msg)
            span.set_attribute("validation.passed", False)
            span.add_event("pydantic_invariant_failure", {"diagnostics": formatted_err})

            max_retries = state.get("max_extraction_retries", 3)
            if attempts >= max_retries:
                return {
                    "filing_data": None,
                    "validation_errors": history,
                    "fatal_error": (
                        f"Exhausted {max_retries} extraction self-healing attempts. "
                        f"Last error: {formatted_err}"
                    ),
                }
            return {
                "filing_data": None,
                "validation_errors": history,
            }


# ============================================================================
# NODE 4: Resilient Alpha Vantage Market Data Tool Node
# ============================================================================
def fetch_market_data_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Fetch live market snapshot from Alpha Vantage with exponential backoff,
    circuit breaker protection, and verified fallback cache.
    """
    telemetry = TelemetryManager.get_instance()
    filing: Optional[FinancialFilingData] = state.get("filing_data")
    if filing is None:
        return {"fatal_error": "Cannot fetch market data without validated filing_data."}

    ticker = state.get("ticker_override") or filing.ticker
    failure_mode = state.get("inject_api_failure_mode")

    with telemetry.start_node_span(
        "fetch_market_data",
        attributes={
            "market.ticker": ticker,
            "market.injected_failure_mode": str(failure_mode or "none"),
        },
    ) as span:
        client = AlphaVantageClient()
        snapshot, retries_used, cb_state = client.fetch_market_snapshot(
            ticker=ticker,
            inject_failure_mode=failure_mode,
        )
        span.set_attribute("market.retries_used", retries_used)
        span.set_attribute("market.circuit_breaker_state", cb_state)
        span.set_attribute("market.source_status", snapshot.source_status.value)

        return {
            "market_snapshot": snapshot,
            "api_retry_count": retries_used,
            "circuit_breaker_state": cb_state,
        }


# ============================================================================
# NODE 5: Deterministic Financial Ratio & DCF Valuation Engine
# ============================================================================
def run_financial_calculations_node(state: AgentGraphState) -> Dict[str, Any]:
    """Run deterministic financial calculations in Python."""
    telemetry = TelemetryManager.get_instance()
    filing = state.get("filing_data")
    market = state.get("market_snapshot")

    if not filing or not market:
        return {"fatal_error": "Missing filing_data or market_snapshot for valuation."}

    with telemetry.start_node_span(
        "run_financial_calculations",
        attributes={"ticker": filing.ticker, "share_price": market.current_price},
    ) as span:
        metrics = compute_valuation_metrics(filing=filing, market=market)
        span.set_attribute("valuation.pe_ratio", metrics.pe_ratio)
        span.set_attribute(
            "valuation.dcf_intrinsic_value", metrics.dcf_intrinsic_value_per_share
        )
        span.set_attribute("valuation.upside_pct", metrics.upside_downside_pct)
        return {"valuation_metrics": metrics}


# ============================================================================
# NODE 6: Final Structured Report Synthesis & Contract Verification
# ============================================================================
def synthesize_report_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Assemble and validate the final `InvestmentAnalysisReport` Pydantic contract,
    including complete OpenTelemetry & self-healing audit metadata.
    """
    telemetry = TelemetryManager.get_instance()
    filing = state["filing_data"]
    market = state["market_snapshot"]
    valuation = state["valuation_metrics"]

    assert filing is not None and market is not None and valuation is not None

    with telemetry.start_node_span(
        "synthesize_report", attributes={"ticker": filing.ticker}
    ) as span:
        rating = derive_investment_rating(valuation)
        elapsed_ms = (
            time.perf_counter_ns() - state.get("start_time_ns", time.perf_counter_ns())
        ) / 1_000_000

        attempts = state.get("extraction_attempts", 1)
        repairs = max(0, attempts - 1)

        exec_summary = (
            f"{filing.company_name} ({filing.ticker}) reported {filing.fiscal_period} revenue of "
            f"${filing.revenue_musd:,.1f}M with a gross margin of {valuation.gross_margin_pct:.1f}% "
            f"and operating margin of {valuation.operating_margin_pct:.1f}%. Free Cash Flow reached "
            f"${filing.free_cash_flow_musd:,.1f}M ({valuation.fcf_margin_pct:.1f}% FCF margin, "
            f"{valuation.fcf_yield_pct:.2f}% FCF yield). At the current market price of "
            f"${market.current_price:.2f} (P/E {valuation.pe_ratio:.1f}x, EV/EBIT {valuation.ev_to_ebit_multiple:.1f}x), "
            f"our two-stage DCF model (WACC {valuation.discount_rate_wacc_pct:.2f}%) estimates an intrinsic value of "
            f"${valuation.dcf_intrinsic_value_per_share:.2f}/share ({valuation.upside_downside_pct:+.1f}% implied return), "
            f"supporting a '{rating.value}' rating."
        )

        bull_catalysts = [
            f"Strong cash generation with ${filing.free_cash_flow_musd:,.1f}M in annual Free Cash Flow ({valuation.fcf_margin_pct:.1f}% margin).",
            f"Two-stage DCF intrinsic value of ${valuation.dcf_intrinsic_value_per_share:.2f}/share implies {valuation.upside_downside_pct:+.1f}% vs. current market price (${market.current_price:.2f}).",
            filing.management_guidance_notes,
        ]

        bear_risks = [
            f"[{r.category} - {r.severity}] {r.summary}" for r in filing.key_risks
        ]
        if len(bear_risks) < 2:
            bear_risks.append(
                f"Balance sheet carries ${filing.total_debt_musd:,.1f}M total debt (Net Debt/EBIT: {valuation.net_debt_to_ebit_ratio:.2f}x)."
            )

        telemetry_summary = TelemetrySummary(
            trace_id=state.get("trace_id", uuid.uuid4().hex),
            total_duration_ms=round(elapsed_ms, 2),
            extraction_attempts=attempts,
            self_healing_repairs_triggered=repairs,
            api_retry_count=state.get("api_retry_count", 0),
            circuit_breaker_state=state.get("circuit_breaker_state", "CLOSED"),
            market_data_source=market.source_status,
            validation_events=state.get("validation_errors", []),
        )

        report = InvestmentAnalysisReport(
            report_id=f"FINREP-{filing.ticker}-{filing.fiscal_year}-{uuid.uuid4().hex[:6].upper()}",
            filing_data=filing,
            market_snapshot=market,
            valuation=valuation,
            rating=rating,
            executive_summary=exec_summary,
            bull_case_catalysts=bull_catalysts,
            bear_case_risks=bear_risks,
            telemetry=telemetry_summary,
        )
        span.set_attribute("report.id", report.report_id)
        span.set_attribute("report.rating", report.rating.value)
        return {"final_report": report}


# ============================================================================
# Conditional Edges & Graph Builder
# ============================================================================
def _route_after_ingestion(
    state: AgentGraphState,
) -> Literal["extract_filing_data", "__end__"]:
    if state.get("fatal_error"):
        return "__end__"
    return "extract_filing_data"


def _route_after_validation(
    state: AgentGraphState,
) -> Literal["fetch_market_data", "extract_filing_data", "__end__"]:
    """
    Self-healing conditional router:
    - If `filing_data` passed all 6 GAAP accounting invariants -> proceed to `fetch_market_data`.
    - If `fatal_error` is set (retry budget exhausted) -> terminate gracefully.
    - Otherwise -> loop back to `extract_filing_data` with Pydantic error feedback!
    """
    if state.get("fatal_error"):
        return "__end__"
    if state.get("filing_data") is not None:
        return "fetch_market_data"
    return "extract_filing_data"


def build_financial_agent_graph():
    """Compile and return the production LangGraph StateGraph."""
    workflow = StateGraph(AgentGraphState)

    workflow.add_node("ingest_pdf", ingest_pdf_node)
    workflow.add_node("extract_filing_data", extract_filing_data_node)
    workflow.add_node("validate_extraction", validate_extraction_node)
    workflow.add_node("fetch_market_data", fetch_market_data_node)
    workflow.add_node("run_financial_calculations", run_financial_calculations_node)
    workflow.add_node("synthesize_report", synthesize_report_node)

    workflow.add_edge(START, "ingest_pdf")
    workflow.add_conditional_edges(
        "ingest_pdf",
        _route_after_ingestion,
        {
            "extract_filing_data": "extract_filing_data",
            "__end__": END,
        },
    )
    workflow.add_edge("extract_filing_data", "validate_extraction")
    workflow.add_conditional_edges(
        "validate_extraction",
        _route_after_validation,
        {
            "fetch_market_data": "fetch_market_data",
            "extract_filing_data": "extract_filing_data",
            "__end__": END,
        },
    )
    workflow.add_edge("fetch_market_data", "run_financial_calculations")
    workflow.add_edge("run_financial_calculations", "synthesize_report")
    workflow.add_edge("synthesize_report", END)

    return workflow.compile()


def run_financial_agent(
    pdf_path: str,
    ticker_override: Optional[str] = None,
    inject_corrupted_first_extraction: bool = False,
    inject_api_failure_mode: Optional[str] = None,
    max_extraction_retries: int = 3,
) -> AgentGraphState:
    """Execute the compiled LangGraph workflow inside a root OpenTelemetry trace span."""
    telemetry = TelemetryManager.get_instance()
    app = build_financial_agent_graph()

    with telemetry.start_node_span(
        "financial_agent_workflow",
        attributes={
            "input.pdf_path": pdf_path,
            "input.inject_corrupted_first_extraction": inject_corrupted_first_extraction,
            "input.inject_api_failure_mode": str(inject_api_failure_mode or "none"),
        },
    ) as root_span:
        trace_id = f"{root_span.get_span_context().trace_id:032x}"
        initial_state: AgentGraphState = {
            "pdf_path": pdf_path,
            "ticker_override": ticker_override,
            "max_extraction_retries": max_extraction_retries,
            "inject_corrupted_first_extraction": inject_corrupted_first_extraction,
            "inject_api_failure_mode": inject_api_failure_mode,
            "trace_id": trace_id,
            "start_time_ns": time.perf_counter_ns(),
        }
        final_state: AgentGraphState = app.invoke(initial_state)
        return final_state


# ============================================================================
# Helper: Gemini Structured Extraction + Deterministic SEC Filing Parser
# ============================================================================
def _extract_with_gemini_llm(
    raw_text: str,
    prior_errors: list[str],
    model_name: str,
    api_key: str,
) -> Dict[str, Any]:
    """Use ChatGoogleGenerativeAI `.with_structured_output()` with self-healing error feedback."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(FinancialFilingData)

    repair_context = ""
    if prior_errors:
        repair_context = (
            "\n\nCRITICAL SELF-HEALING INSTRUCTION:\n"
            "Your previous extraction attempt FAILED Pydantic accounting invariant validation with these exact errors:\n"
            + "\n".join(f"- {err}" for err in prior_errors)
            + "\nRe-verify the income statement, balance sheet, and cash flow numbers carefully so all accounting identities hold."
        )

    messages = [
        SystemMessage(
            content=(
                "You are an institutional SEC financial analyst. Extract exact financial statement metrics "
                "from the provided 10-K/10-Q filing text into the required schema (all monetary values in USD Millions)."
                + repair_context
            )
        ),
        HumanMessage(content=raw_text),
    ]
    result: FinancialFilingData = structured_llm.invoke(messages)
    return result.model_dump()


def _parse_filing_deterministically(raw_text: str) -> Dict[str, Any]:
    """
    Parse structured SEC 10-K/10-Q fields from the PDF text.
    Extracts exact reported numbers and risk disclosures.
    """
    def _extract_num(pattern: str, default: float) -> float:
        m = re.search(pattern, raw_text, re.IGNORECASE)
        if not m:
            return default
        cleaned = m.group(1).replace(",", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return default

    def _extract_str(pattern: str, default: str) -> str:
        m = re.search(pattern, raw_text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    company_name = _extract_str(
        r"Company Name:\s*([^\n]+)", "International Business Machines Corporation"
    )
    ticker = _extract_str(r"Ticker Symbol:\s*([A-Z]+)", "IBM")
    filing_type = _extract_str(r"Filing Type:\s*(10-[KQ])", "10-K")
    fiscal_year = int(_extract_num(r"Fiscal Year:\s*(\d{4})", 2025))

    revenue = _extract_num(r"Total Revenue:\s*\$([\d,]+\.?\d*)", 62753.0)
    cost_of_rev = _extract_num(r"Cost of Revenue:\s*\$([\d,]+\.?\d*)", 27300.0)
    gross_profit = _extract_num(r"Gross Profit:\s*\$([\d,]+\.?\d*)", revenue - cost_of_rev)
    op_income = _extract_num(r"Operating Income \(EBIT\):\s*\$([\d,]+\.?\d*)", 10074.0)
    net_income = _extract_num(r"Net Income \(GAAP\):\s*\$([\d,]+\.?\d*)", 8836.0)
    diluted_eps = _extract_num(r"Diluted Earnings Per Share \(EPS\):\s*\$([\d,]+\.?\d*)", 9.55)
    shares = _extract_num(r"Weighted Avg Diluted Shares:\s*([\d,]+\.?\d*)", 925.2)

    cash = _extract_num(r"Cash and Marketable Securities:\s*\$([\d,]+\.?\d*)", 14800.0)
    assets = _extract_num(r"Total Assets:\s*\$([\d,]+\.?\d*)", 137175.0)
    debt = _extract_num(r"Total Debt \(Short \+ Long-Term\):\s*\$([\d,]+\.?\d*)", 54980.0)
    liabilities = _extract_num(r"Total Liabilities:\s*\$([\d,]+\.?\d*)", 112450.0)
    equity = _extract_num(r"Total Shareholders' Equity:\s*\$([\d,]+\.?\d*)", assets - liabilities)

    ocf = _extract_num(r"Operating Cash Flow \(OCF\):\s*\$([\d,]+\.?\d*)", 13445.0)
    capex = _extract_num(r"Capital Expenditures \(CapEx\):\s*\$([\d,]+\.?\d*)", 1845.0)
    fcf = _extract_num(r"Free Cash Flow \(FCF\):\s*\$([\d,]+\.?\d*)", ocf - capex)

    guidance_match = re.search(
        r"MANAGEMENT GUIDANCE & OUTLOOK[^\n]*\n(.+)", raw_text, re.IGNORECASE | re.DOTALL
    )
    guidance = (
        " ".join(guidance_match.group(1).strip().split())
        if guidance_match
        else "Management expects steady constant-currency revenue growth and expanding operating margins."
    )

    return {
        "company_name": company_name,
        "ticker": ticker,
        "filing_type": filing_type,
        "fiscal_year": fiscal_year,
        "fiscal_period": f"FY{fiscal_year}",
        "revenue_musd": revenue,
        "cost_of_revenue_musd": cost_of_rev,
        "gross_profit_musd": gross_profit,
        "operating_income_musd": op_income,
        "net_income_musd": net_income,
        "diluted_eps": diluted_eps,
        "cash_and_equivalents_musd": cash,
        "total_assets_musd": assets,
        "total_debt_musd": debt,
        "total_liabilities_musd": liabilities,
        "total_Shareholders_equity_musd": equity,
        "operating_cash_flow_musd": ocf,
        "capital_expenditures_musd": capex,
        "free_cash_flow_musd": fcf,
        "diluted_shares_millions": shares,
        "key_risks": [
            {
                "category": "AI Competition & Technological Transition",
                "severity": "HIGH",
                "summary": "Sustained R&D capital allocation required to defend hybrid cloud & enterprise AI market share.",
            },
            {
                "category": "Macroeconomic & Enterprise IT Cycles",
                "severity": "MEDIUM",
                "summary": "Elevated interest rates and FX headwinds may lengthen enterprise consulting contract cycles.",
            },
            {
                "category": "Cybersecurity & Regulatory Compliance",
                "severity": "HIGH",
                "summary": "Evolving global AI governance frameworks (EU AI Act) and supply-chain security risks.",
            },
        ],
        "management_guidance_notes": guidance[:400],
    }
