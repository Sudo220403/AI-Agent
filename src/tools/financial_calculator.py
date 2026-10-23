"""Deterministic financial ratio, credit leverage, and two-stage DCF valuation engine."""

from __future__ import annotations

from src.schemas import (
    FinancialFilingData,
    InvestmentRating,
    MarketSnapshot,
    ValuationMetrics,
)


def compute_valuation_metrics(
    filing: FinancialFilingData, market: MarketSnapshot
) -> ValuationMetrics:
    """
    Deterministically compute valuation multiples, profitability margins,
    leverage ratios, and a 2-stage Free Cash Flow DCF intrinsic value per share.
    Never delegates arithmetic to an LLM.
    """
    # 1. Market Capitalization & Enterprise Value ($M)
    market_cap_musd = market.current_price * filing.diluted_shares_millions
    net_debt_musd = filing.total_debt_musd - filing.cash_and_equivalents_musd
    enterprise_value_musd = max(market_cap_musd + net_debt_musd, 1.0)

    # 2. Profitability Margins (%)
    gross_margin_pct = (filing.gross_profit_musd / filing.revenue_musd) * 100.0
    operating_margin_pct = (filing.operating_income_musd / filing.revenue_musd) * 100.0
    net_margin_pct = (filing.net_income_musd / filing.revenue_musd) * 100.0
    fcf_margin_pct = (filing.free_cash_flow_musd / filing.revenue_musd) * 100.0

    roe_pct = (
        (filing.net_income_musd / filing.total_Shareholders_equity_musd) * 100.0
        if filing.total_Shareholders_equity_musd > 0
        else 0.0
    )
    roa_pct = (filing.net_income_musd / filing.total_assets_musd) * 100.0

    # 3. Valuation Multiples
    pe_ratio = (
        market.current_price / filing.diluted_eps if filing.diluted_eps > 0 else 999.0
    )
    ev_to_ebit = (
        enterprise_value_musd / filing.operating_income_musd
        if filing.operating_income_musd > 0
        else 999.0
    )
    price_to_book = (
        market_cap_musd / filing.total_Shareholders_equity_musd
        if filing.total_Shareholders_equity_musd > 0
        else 999.0
    )
    fcf_yield_pct = (filing.free_cash_flow_musd / market_cap_musd) * 100.0
    ev_to_fcf = (
        enterprise_value_musd / filing.free_cash_flow_musd
        if filing.free_cash_flow_musd > 0
        else 999.0
    )

    # 4. Solvency & Leverage Ratios
    debt_to_equity = (
        filing.total_debt_musd / filing.total_Shareholders_equity_musd
        if filing.total_Shareholders_equity_musd > 0
        else 99.0
    )
    net_debt_to_ebit = (
        net_debt_musd / filing.operating_income_musd
        if filing.operating_income_musd > 0
        else 99.0
    )

    # 5. Simplified Two-Stage DCF Valuation Model
    # CAPM Cost of Equity: Risk-free rate (4.25%) + Beta * Equity Risk Premium (5.0%)
    risk_free_rate = 0.0425
    equity_risk_premium = 0.050
    cost_of_equity = risk_free_rate + max(0.5, min(market.beta, 2.0)) * equity_risk_premium
    wacc = max(0.075, min(cost_of_equity, 0.135))  # Bounded 7.5% to 13.5%

    # Stage 1: 5-year explicit FCF projection (6.5% annual growth for quality compounders)
    stage1_growth = 0.065
    terminal_growth = 0.025
    base_fcf = max(filing.free_cash_flow_musd, filing.net_income_musd * 0.8)

    pv_explicit_fcf = 0.0
    projected_fcf = base_fcf
    for year in range(1, 6):
        projected_fcf *= 1.0 + stage1_growth
        pv_explicit_fcf += projected_fcf / ((1.0 + wacc) ** year)

    # Stage 2: Gordon Growth Terminal Value at Year 5
    terminal_fcf = projected_fcf * (1.0 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** 5)

    implied_enterprise_value_musd = pv_explicit_fcf + pv_terminal_value
    implied_equity_value_musd = max(
        implied_enterprise_value_musd - net_debt_musd, market_cap_musd * 0.25
    )
    intrinsic_value_per_share = implied_equity_value_musd / filing.diluted_shares_millions
    upside_downside_pct = (
        (intrinsic_value_per_share - market.current_price) / market.current_price
    ) * 100.0

    return ValuationMetrics(
        implied_market_cap_musd=round(market_cap_musd, 2),
        enterprise_value_musd=round(enterprise_value_musd, 2),
        gross_margin_pct=round(gross_margin_pct, 2),
        operating_margin_pct=round(operating_margin_pct, 2),
        net_margin_pct=round(net_margin_pct, 2),
        fcf_margin_pct=round(fcf_margin_pct, 2),
        return_on_equity_pct=round(roe_pct, 2),
        return_on_assets_pct=round(roa_pct, 2),
        pe_ratio=round(pe_ratio, 2),
        ev_to_ebit_multiple=round(ev_to_ebit, 2),
        price_to_book_multiple=round(price_to_book, 2),
        fcf_yield_pct=round(fcf_yield_pct, 2),
        ev_to_fcf_multiple=round(ev_to_fcf, 2),
        net_debt_musd=round(net_debt_musd, 2),
        debt_to_equity_ratio=round(debt_to_equity, 2),
        net_debt_to_ebit_ratio=round(net_debt_to_ebit, 2),
        dcf_intrinsic_value_per_share=round(intrinsic_value_per_share, 2),
        upside_downside_pct=round(upside_downside_pct, 2),
        discount_rate_wacc_pct=round(wacc * 100.0, 2),
    )


def derive_investment_rating(valuation: ValuationMetrics) -> InvestmentRating:
    """Assign a standardized rating based on DCF upside, FCF yield, and leverage."""
    if valuation.net_debt_to_ebit_ratio > 6.5 and valuation.fcf_yield_pct < 2.0:
        return InvestmentRating.DISTRESSED_HIGH_RISK
    if valuation.upside_downside_pct >= 12.0 or (
        valuation.fcf_yield_pct >= 5.5 and valuation.pe_ratio < 24.0
    ):
        return InvestmentRating.UNDERVALUED_ACCUMULATE
    if valuation.upside_downside_pct <= -15.0 or valuation.pe_ratio > 40.0:
        return InvestmentRating.OVERVALUED_CAUTIOUS
    return InvestmentRating.FAIRLY_VALUED_HOLD
