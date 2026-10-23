"""Pydantic v2 domain schemas with cross-field financial invariants and self-healing diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator, model_validator


class FilingType(str, Enum):
    """Supported SEC corporate filing types."""

    FORM_10K = "10-K"
    FORM_10Q = "10-Q"


class DataSourceStatus(str, Enum):
    """Provenance status of external tool calls for transparency & auditability."""

    LIVE_API = "LIVE_ALPHAVANTAGE_API"
    DEMO_ENDPOINT = "ALPHAVANTAGE_DEMO_ENDPOINT"
    FALLBACK_CACHE = "DEGRADED_FALLBACK_CACHE"


class InvestmentRating(str, Enum):
    """Standardized equity research rating."""

    UNDERVALUED_ACCUMULATE = "UNDERVALUED / ACCUMULATE"
    FAIRLY_VALUED_HOLD = "FAIRLY VALUED / HOLD"
    OVERVALUED_CAUTIOUS = "OVERVALUED / CAUTIOUS"
    DISTRESSED_HIGH_RISK = "DISTRESSED / HIGH RISK"


class RiskItem(BaseModel):
    """Structured risk factor extracted from Item 1A of the filing."""

    category: str = Field(
        ..., description="Risk category (e.g., Supply Chain, Regulatory, Macroeconomic, AI Competition)"
    )
    severity: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    summary: str = Field(..., min_length=10, description="Concise explanation of the risk factor")


class FinancialFilingData(BaseModel):
    """
    Structured financial metrics extracted from a 10-K or 10-Q PDF filing.
    All monetary values are normalized to USD Millions ($M), except per-share figures.
    Includes strict cross-field accounting invariants to detect LLM hallucinations.
    """

    company_name: str = Field(..., min_length=2, description="Full legal name of the corporation")
    ticker: str = Field(..., min_length=1, max_length=6, description="Uppercase stock ticker symbol")
    filing_type: FilingType = Field(..., description="SEC filing form type (10-K or 10-Q)")
    fiscal_year: int = Field(..., ge=2000, le=2030, description="Fiscal year of the report")
    fiscal_period: str = Field(..., description="e.g., FY2025, Q4 2025")

    # Income Statement ($ Millions)
    revenue_musd: float = Field(..., gt=0, description="Total Revenue in USD Millions")
    cost_of_revenue_musd: float = Field(..., ge=0, description="Cost of Goods Sold / Cost of Revenue in USD Millions")
    gross_profit_musd: float = Field(..., description="Gross Profit in USD Millions")
    operating_income_musd: float = Field(..., description="Operating Income (EBIT) in USD Millions")
    net_income_musd: float = Field(..., description="Net Income (GAAP) in USD Millions")
    diluted_eps: float = Field(..., description="Diluted Earnings Per Share in USD")

    # Balance Sheet ($ Millions)
    cash_and_equivalents_musd: float = Field(
        ..., ge=0, description="Cash, cash equivalents, and marketable securities in USD Millions"
    )
    total_assets_musd: float = Field(..., gt=0, description="Total Assets in USD Millions")
    total_debt_musd: float = Field(..., ge=0, description="Short-term + Long-term Debt in USD Millions")
    total_liabilities_musd: float = Field(..., ge=0, description="Total Liabilities in USD Millions")
    total_Shareholders_equity_musd: float = Field(
        ..., description="Total Stockholders' Equity in USD Millions"
    )

    # Cash Flow & Share Count ($ Millions / Millions of shares)
    operating_cash_flow_musd: float = Field(..., description="Cash Flow from Operating Activities in USD Millions")
    capital_expenditures_musd: float = Field(
        ..., ge=0, description="Capital Expenditures (CapEx, positive number) in USD Millions"
    )
    free_cash_flow_musd: float = Field(..., description="Free Cash Flow (OCF - CapEx) in USD Millions")
    diluted_shares_millions: float = Field(
        ..., gt=0, description="Weighted average diluted shares outstanding in Millions"
    )

    # Qualitative Disclosures
    key_risks: List[RiskItem] = Field(
        ..., min_length=1, description="Key risk factors disclosed in the filing"
    )
    management_guidance_notes: str = Field(
        ..., min_length=10, description="Summary of management's forward-looking commentary"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.isalpha():
            raise ValueError(f"Ticker '{v}' must contain only alphabetic characters.")
        return cleaned

    @model_validator(mode="after")
    def verify_accounting_invariants(self) -> "FinancialFilingData":
        """
        Cross-field financial domain invariants.
        If the LLM hallucinates inconsistent numbers or mixes units ($B vs $M),
        this validator raises a detailed error used by the self-healing retry node.
        """
        errors: List[str] = []

        # 1. Gross Profit = Revenue - Cost of Revenue (tolerance 2% or $5M for rounding)
        expected_gp = self.revenue_musd - self.cost_of_revenue_musd
        gp_diff = abs(self.gross_profit_musd - expected_gp)
        if gp_diff > max(5.0, 0.02 * self.revenue_musd):
            errors.append(
                f"Income Statement Invariant Violation: gross_profit_musd ({self.gross_profit_musd}) "
                f"does not match revenue_musd ({self.revenue_musd}) - cost_of_revenue_musd "
                f"({self.cost_of_revenue_musd}) = {expected_gp:.2f} (diff={gp_diff:.2f})."
            )

        # 2. Gross Profit cannot exceed Revenue
        if self.gross_profit_musd > self.revenue_musd * 1.005:
            errors.append(
                f"Gross profit ({self.gross_profit_musd}) cannot exceed Total Revenue ({self.revenue_musd})."
            )

        # 3. Balance Sheet Accounting Identity: Assets = Liabilities + Shareholders' Equity (tolerance 2% or $10M)
        expected_assets = self.total_liabilities_musd + self.total_Shareholders_equity_musd
        bs_diff = abs(self.total_assets_musd - expected_assets)
        if bs_diff > max(10.0, 0.02 * self.total_assets_musd):
            errors.append(
                f"Balance Sheet Identity Violation: total_assets_musd ({self.total_assets_musd}) "
                f"!= total_liabilities_musd ({self.total_liabilities_musd}) + "
                f"total_Shareholders_equity_musd ({self.total_Shareholders_equity_musd}) = {expected_assets:.2f}."
            )

        # 4. Total Debt cannot exceed Total Liabilities
        if self.total_debt_musd > self.total_liabilities_musd * 1.01:
            errors.append(
                f"Debt Invariant Violation: total_debt_musd ({self.total_debt_musd}) cannot exceed "
                f"total_liabilities_musd ({self.total_liabilities_musd})."
            )

        # 5. Free Cash Flow = Operating Cash Flow - Capital Expenditures (tolerance $5M or 2%)
        expected_fcf = self.operating_cash_flow_musd - self.capital_expenditures_musd
        fcf_diff = abs(self.free_cash_flow_musd - expected_fcf)
        if fcf_diff > max(5.0, 0.02 * abs(self.operating_cash_flow_musd)):
            errors.append(
                f"Cash Flow Identity Violation: free_cash_flow_musd ({self.free_cash_flow_musd}) "
                f"!= operating_cash_flow_musd ({self.operating_cash_flow_musd}) - "
                f"capital_expenditures_musd ({self.capital_expenditures_musd}) = {expected_fcf:.2f}."
            )

        # 6. Diluted EPS consistency with Net Income / Diluted Shares (tolerance $0.15/share)
        implied_eps = self.net_income_musd / self.diluted_shares_millions
        if abs(self.diluted_eps - implied_eps) > 0.15:
            errors.append(
                f"EPS Consistency Violation: diluted_eps ({self.diluted_eps}) is inconsistent with "
                f"net_income_musd ({self.net_income_musd}) / diluted_shares_millions "
                f"({self.diluted_shares_millions}) = {implied_eps:.2f}."
            )

        if errors:
            raise ValueError(" | ".join(errors))

        return self


class MarketSnapshot(BaseModel):
    """Live or fallback market quote and fundamental overview from Alpha Vantage."""

    ticker: str = Field(..., description="Stock ticker symbol")
    current_price: float = Field(..., gt=0, description="Latest share price in USD")
    previous_close: float = Field(..., gt=0, description="Previous trading day close price in USD")
    daily_change_pct: float = Field(..., description="Daily percentage price change")
    volume: int = Field(..., ge=0, description="Trading volume")
    week_52_high: float = Field(..., gt=0, description="52-week high price in USD")
    week_52_low: float = Field(..., gt=0, description="52-week low price in USD")
    beta: float = Field(default=1.0, description="Equity Beta vs S&P 500")
    sector: str = Field(default="Technology", description="GICS Sector")
    industry: str = Field(default="Software & IT Services", description="Industry classification")
    source_status: DataSourceStatus = Field(
        ..., description="Whether market data came from live Alpha Vantage API or fallback cache"
    )
    retrieved_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    warning_banner: Optional[str] = Field(
        default=None, description="Warning if degraded fallback cache was used"
    )

    @model_validator(mode="after")
    def validate_price_range(self) -> "MarketSnapshot":
        if self.week_52_low > self.week_52_high:
            raise ValueError(
                f"52-week low ({self.week_52_low}) cannot exceed 52-week high ({self.week_52_high})."
            )
        return self


class ValuationMetrics(BaseModel):
    """
    Deterministic valuation and credit health metrics computed in Python
    (never delegated to LLM mental math).
    """

    implied_market_cap_musd: float = Field(..., gt=0)
    enterprise_value_musd: float = Field(..., gt=0)

    # Profitability Margins (%)
    gross_margin_pct: float
    operating_margin_pct: float
    net_margin_pct: float
    fcf_margin_pct: float
    return_on_equity_pct: float
    return_on_assets_pct: float

    # Valuation Multiples
    pe_ratio: float = Field(..., description="Price-to-Earnings (Diluted) multiple")
    ev_to_ebit_multiple: float = Field(..., description="Enterprise Value / Operating Income (EBIT)")
    price_to_book_multiple: float = Field(..., description="Market Cap / Shareholders' Equity")
    fcf_yield_pct: float = Field(..., description="Free Cash Flow / Market Cap (%)")
    ev_to_fcf_multiple: float = Field(..., description="Enterprise Value / Free Cash Flow")

    # Leverage & Solvency
    net_debt_musd: float = Field(..., description="Total Debt - Cash & Equivalents ($M)")
    debt_to_equity_ratio: float = Field(..., ge=0)
    net_debt_to_ebit_ratio: float = Field(..., description="Net Debt / Operating Income")

    # Simplified Two-Stage DCF Intrinsic Value Estimate
    dcf_intrinsic_value_per_share: float = Field(
        ..., gt=0, description="5-year DCF + Terminal Value intrinsic value per share (USD)"
    )
    upside_downside_pct: float = Field(
        ..., description="Percentage upside (+) or downside (-) to DCF intrinsic value"
    )
    discount_rate_wacc_pct: float = Field(..., description="CAPM-derived WACC used in DCF (%)")


class TelemetrySummary(BaseModel):
    """Observability and fault-recovery audit trail embedded in the final report."""

    trace_id: str
    total_duration_ms: float
    extraction_attempts: int = Field(..., ge=1)
    self_healing_repairs_triggered: int = Field(..., ge=0)
    api_retry_count: int = Field(..., ge=0)
    circuit_breaker_state: str
    market_data_source: DataSourceStatus
    validation_events: List[str] = Field(default_factory=list)


class InvestmentAnalysisReport(BaseModel):
    """Final validated output contract produced by the LangGraph workflow."""

    report_id: str
    generated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    filing_data: FinancialFilingData
    market_snapshot: MarketSnapshot
    valuation: ValuationMetrics
    rating: InvestmentRating
    executive_summary: str = Field(..., min_length=40)
    bull_case_catalysts: List[str] = Field(..., min_length=2)
    bear_case_risks: List[str] = Field(..., min_length=2)
    telemetry: TelemetrySummary


class AgentGraphState(TypedDict, total=False):
    """
    LangGraph state dictionary passed across workflow nodes.
    Tracks inputs, intermediate extractions, validation feedback loops, and telemetry.
    """

    pdf_path: str
    ticker_override: Optional[str]
    raw_pdf_text: str
    pdf_page_count: int

    # Extraction & Self-Healing Loop State
    extraction_attempts: int
    max_extraction_retries: int
    validation_errors: List[str]
    last_raw_extraction: Optional[dict[str, Any]]
    filing_data: Optional[FinancialFilingData]

    # Fault-Injection Hooks for Testing & Demonstration
    inject_corrupted_first_extraction: bool
    inject_api_failure_mode: Optional[str]  # e.g., "rate_limit_once", "total_outage"

    # Market Tool & Calculation State
    market_snapshot: Optional[MarketSnapshot]
    api_retry_count: int
    circuit_breaker_state: str
    valuation_metrics: Optional[ValuationMetrics]

    # Final Output
    final_report: Optional[InvestmentAnalysisReport]
    fatal_error: Optional[str]
    trace_id: str
    start_time_ns: int
