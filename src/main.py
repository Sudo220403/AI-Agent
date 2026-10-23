"""Rich CLI Runner, Markdown/JSON Report Exporter, and Fault-Injection Chaos Demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent import run_financial_agent
from src.observability import TelemetryManager
from src.schemas import InvestmentAnalysisReport
from src.tools.pdf_parser import generate_sample_10k_pdf


console = Console()


def render_report_to_console(report: InvestmentAnalysisReport) -> None:
    """Render the validated InvestmentAnalysisReport using Rich panels and tables."""
    f = report.filing_data
    m = report.market_snapshot
    v = report.valuation
    t = report.telemetry

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{f.company_name} ({f.ticker})[/bold cyan] — [bold yellow]{f.filing_type} ({f.fiscal_period})[/bold yellow]\n"
            f"Report ID: [dim]{report.report_id}[/dim] | Rating: [bold green]{report.rating.value}[/bold green]\n"
            f"Trace ID: [magenta]{t.trace_id}[/magenta] | Duration: [bold]{t.total_duration_ms:.1f} ms[/bold]",
            title="[bold white]Institutional Financial Filing & Market Intelligence Report[/bold white]",
            border_style="bright_blue",
        )
    )

    if m.warning_banner:
        console.print(
            Panel(
                f"[bold yellow]{m.warning_banner}[/bold yellow]",
                title="Resilience & Fallback Notice",
                border_style="yellow",
            )
        )

    # Table 1: Extracted GAAP Financial Statements & Live Market Data
    fin_table = Table(
        title="1. Validated SEC Filing Metrics ($M) & Live Market Snapshot",
        show_header=True,
        header_style="bold cyan",
    )
    fin_table.add_column("Filing Metric (GAAP)", style="white")
    fin_table.add_column("Reported Value", justify="right", style="green")
    fin_table.add_column("Market / Valuation Metric", style="white")
    fin_table.add_column("Computed Value", justify="right", style="bright_yellow")

    fin_table.add_row(
        "Total Revenue",
        f"${f.revenue_musd:,.1f}M",
        "Current Share Price",
        f"${m.current_price:,.2f} ({m.daily_change_pct:+.2f}%)",
    )
    fin_table.add_row(
        "Gross Profit (Margin)",
        f"${f.gross_profit_musd:,.1f}M ({v.gross_margin_pct:.1f}%)",
        "52-Week Price Range",
        f"${m.week_52_low:,.2f} – ${m.week_52_high:,.2f}",
    )
    fin_table.add_row(
        "Operating Income / EBIT",
        f"${f.operating_income_musd:,.1f}M ({v.operating_margin_pct:.1f}%)",
        "Implied Market Cap",
        f"${v.implied_market_cap_musd:,.1f}M",
    )
    fin_table.add_row(
        "Net Income (GAAP)",
        f"${f.net_income_musd:,.1f}M ({v.net_margin_pct:.1f}%)",
        "Enterprise Value (EV)",
        f"${v.enterprise_value_musd:,.1f}M",
    )
    fin_table.add_row(
        "Diluted EPS",
        f"${f.diluted_eps:,.2f}",
        "P/E Ratio | EV/EBIT",
        f"{v.pe_ratio:.1f}x | {v.ev_to_ebit_multiple:.1f}x",
    )
    fin_table.add_row(
        "Free Cash Flow (FCF)",
        f"${f.free_cash_flow_musd:,.1f}M ({v.fcf_margin_pct:.1f}%)",
        "FCF Yield | EV/FCF",
        f"{v.fcf_yield_pct:.2f}% | {v.ev_to_fcf_multiple:.1f}x",
    )
    fin_table.add_row(
        "Total Assets | Equity",
        f"${f.total_assets_musd:,.1f}M | ${f.total_Shareholders_equity_musd:,.1f}M",
        "Net Debt | Debt/Equity",
        f"${v.net_debt_musd:,.1f}M | {v.debt_to_equity_ratio:.2f}x",
    )
    fin_table.add_row(
        "Total Debt | Cash",
        f"${f.total_debt_musd:,.1f}M | ${f.cash_and_equivalents_musd:,.1f}M",
        f"2-Stage DCF Value (WACC {v.discount_rate_wacc_pct:.1f}%)",
        f"${v.dcf_intrinsic_value_per_share:,.2f} ({v.upside_downside_pct:+.1f}%)",
    )
    console.print(fin_table)

    # Table 2: Reliability, Self-Healing & OpenTelemetry Audit Trail
    rel_table = Table(
        title="2. Production Reliability, Self-Healing & OpenTelemetry Audit",
        show_header=True,
        header_style="bold magenta",
    )
    rel_table.add_column("Subsystem", style="cyan")
    rel_table.add_column("Status / Count", style="bold white")
    rel_table.add_column("Diagnostic Details", style="dim white")

    rel_table.add_row(
        "Pydantic Accounting Gate",
        f"{t.extraction_attempts} Attempt(s) ({t.self_healing_repairs_triggered} Repair[s])",
        " | ".join(t.validation_events),
    )
    rel_table.add_row(
        "Alpha Vantage API Client",
        f"{t.api_retry_count} Retry(s) | Source: {t.market_data_source.value}",
        f"Circuit Breaker State: {t.circuit_breaker_state}",
    )
    console.print(rel_table)

    console.print(
        Panel(
            f"[bold]Executive Summary:[/bold]\n{report.executive_summary}\n\n"
            f"[bold green]Bull Case Catalysts:[/bold green]\n"
            + "\n".join(f"  * {c}" for c in report.bull_case_catalysts)
            + "\n\n[bold red]Bear Case & Filing Risks (Item 1A):[/bold red]\n"
            + "\n".join(f"  * {r}" for r in report.bear_case_risks),
            title="3. Equity Research Synthesis",
            border_style="green",
        )
    )


def export_report_artifacts(
    report: InvestmentAnalysisReport, output_dir: Path
) -> tuple[Path, Path]:
    """Save validated report as both structured JSON and executive Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker = report.filing_data.ticker.lower()
    json_path = output_dir / f"{ticker}_analysis_report.json"
    md_path = output_dir / f"{ticker}_analysis_report.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    f = report.filing_data
    m = report.market_snapshot
    v = report.valuation
    t = report.telemetry

    md_content = f"""# Institutional Equity & Filing Intelligence Report: {f.company_name} ({f.ticker})

- **Report ID:** `{report.report_id}`
- **Generated At (UTC):** `{report.generated_at_utc}`
- **Filing Analyzed:** `{f.filing_type}` ({f.fiscal_period})
- **Investment Rating:** **{report.rating.value}**
- **OpenTelemetry Trace ID:** `{t.trace_id}`

---

## 1. Executive Summary

{report.executive_summary}

---

## 2. Validated GAAP Financial Statement Metrics ($ Millions)

| Metric | Value ($M) | Valuation / Profitability Ratio | Value |
| :--- | :---: | :--- | :---: |
| **Total Revenue** | ${f.revenue_musd:,.1f}M | **Gross Margin** | {v.gross_margin_pct:.2f}% |
| **Gross Profit** | ${f.gross_profit_musd:,.1f}M | **Operating Margin (EBIT)** | {v.operating_margin_pct:.2f}% |
| **Operating Income (EBIT)** | ${f.operating_income_musd:,.1f}M | **Net Profit Margin** | {v.net_margin_pct:.2f}% |
| **Net Income (GAAP)** | ${f.net_income_musd:,.1f}M | **Free Cash Flow Margin** | {v.fcf_margin_pct:.2f}% |
| **Free Cash Flow (FCF)** | ${f.free_cash_flow_musd:,.1f}M | **Return on Equity (ROE)** | {v.return_on_equity_pct:.2f}% |
| **Total Assets** | ${f.total_assets_musd:,.1f}M | **Price / Earnings (P/E)** | {v.pe_ratio:.2f}x |
| **Total Debt** | ${f.total_debt_musd:,.1f}M | **EV / EBIT** | {v.ev_to_ebit_multiple:.2f}x |
| **Shareholders' Equity** | ${f.total_Shareholders_equity_musd:,.1f}M | **FCF Yield** | {v.fcf_yield_pct:.2f}% |
| **Diluted EPS** | ${f.diluted_eps:.2f} | **2-Stage DCF Intrinsic Value** | **${v.dcf_intrinsic_value_per_share:.2f}** ({v.upside_downside_pct:+.1f}%) |

---

## 3. Bull Case Catalysts & Bear Case Risk Factors

### Bull Case Catalysts
{chr(10).join(f"- {item}" for item in report.bull_case_catalysts)}

### Disclosed Filing Risks (Item 1A)
{chr(10).join(f"- {item}" for item in report.bear_case_risks)}

---

## 4. Reliability, Self-Healing & Telemetry Audit

- **Market Data Provenance:** `{t.market_data_source.value}` (Share Price: `${m.current_price:.2f}`)
- **Extraction Attempts:** `{t.extraction_attempts}` (`{t.self_healing_repairs_triggered}` Self-Healing Repair[s] Triggered)
- **Market API Retries:** `{t.api_retry_count}` (Circuit Breaker: `{t.circuit_breaker_state}`)
- **Validation Events:**
{chr(10).join(f"  - `{evt}`" for evt in t.validation_events)}
"""
    md_path.write_text(md_content, encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production LangGraph Financial Filing & Market Intelligence Agent"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to the corporate 10-K/10-Q PDF filing (auto-generated for --ticker if omitted or missing).",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="IBM",
        help="Ticker symbol to generate/analyze (default: IBM).",
    )
    parser.add_argument(
        "--simulate-failures",
        action="store_true",
        help="Inject both an initial LLM accounting hallucination AND an Alpha Vantage HTTP 429 rate-limit to demonstrate self-healing + tenacity retry recovery.",
    )
    parser.add_argument(
        "--simulate-outage",
        action="store_true",
        help="Simulate a complete Alpha Vantage 503 outage to demonstrate Circuit Breaker opening and fallback cache degradation.",
    )
    args = parser.parse_args()

    ticker_clean = args.ticker.strip().upper()
    pdf_path = (
        Path(args.pdf)
        if args.pdf
        else Path(f"data/sample_{ticker_clean.lower()}_10k_fy2025.pdf")
    )
    if not pdf_path.exists():
        console.print(
            f"[cyan]Generating realistic SEC Form 10-K PDF at [bold]{pdf_path}[/bold] for ticker [bold]{ticker_clean}[/bold]...[/cyan]"
        )
        generate_sample_10k_pdf(pdf_path, ticker=ticker_clean)

    api_failure_mode = None
    if args.simulate_outage:
        api_failure_mode = "total_outage"
    elif args.simulate_failures:
        api_failure_mode = "rate_limit_once"

    if args.simulate_failures or args.simulate_outage:
        console.print(
            Panel(
                f"[bold yellow]Fault-Injection Mode Enabled[/bold yellow]\n"
                f"* Corrupt Attempt #1 Extraction (triggers Pydantic GAAP invariant error + LangGraph self-healing loop): [bold]{args.simulate_failures}[/bold]\n"
                f"* Market API Fault Mode: [bold]{api_failure_mode}[/bold]",
                title="Chaos & Resilience Verification",
                border_style="yellow",
            )
        )

    final_state = run_financial_agent(
        pdf_path=str(pdf_path),
        ticker_override=args.ticker,
        inject_corrupted_first_extraction=args.simulate_failures,
        inject_api_failure_mode=api_failure_mode,
    )

    if final_state.get("fatal_error"):
        console.print(
            Panel(
                f"[bold red]{final_state['fatal_error']}[/bold red]",
                title="Fatal Workflow Error",
                border_style="red",
            )
        )
        return 1

    report: InvestmentAnalysisReport = final_state["final_report"]
    render_report_to_console(report)

    json_out, md_out = export_report_artifacts(report, Path("outputs"))
    otel_path = TelemetryManager.get_instance().jsonl_exporter.file_path
    console.print(
        f"\n[bold green][OK] Validated artifacts exported:[/bold green]\n"
        f"  * JSON Contract:    [underline]{json_out}[/underline]\n"
        f"  * Markdown Report:  [underline]{md_out}[/underline]\n"
        f"  * OTel JSONL Spans: [underline]{otel_path}[/underline]\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
