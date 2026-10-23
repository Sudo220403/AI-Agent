"""Resilient PDF filing parser and sample SEC 10-K PDF generator."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from fpdf import FPDF
from pypdf import PdfReader


class PDFExtractionError(Exception):
    """Raised when a PDF filing is missing, encrypted, empty, or corrupted."""


def extract_text_from_pdf(pdf_path: str | Path) -> Tuple[str, int]:
    """
    Extract full text and page count from a corporate financial filing PDF.
    Validates file existence, readability, and minimum content density.
    """
    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        raise PDFExtractionError(f"Filing PDF not found at path: {path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PDFExtractionError(f"Failed to open or parse PDF header '{path}': {exc}") from exc

    if reader.is_encrypted:
        raise PDFExtractionError(f"Filing PDF '{path}' is encrypted and cannot be parsed.")

    page_count = len(reader.pages)
    if page_count == 0:
        raise PDFExtractionError(f"Filing PDF '{path}' contains 0 pages.")

    extracted_pages: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            extracted_pages.append(f"--- [PAGE {idx}] ---\n{page_text}")

    full_text = "\n\n".join(extracted_pages).strip()
    if len(full_text) < 100:
        raise PDFExtractionError(
            f"Filing PDF '{path}' yielded insufficient text ({len(full_text)} chars). "
            "It may be an un-OCRed scan or empty document."
        )

    return full_text, page_count


def generate_sample_10k_pdf(output_path: str | Path, ticker: str = "IBM") -> Path:
    """
    Generate a realistic, multi-page SEC Form 10-K PDF filing for testing and evaluation.
    Supports 'IBM' (works with Alpha Vantage free demo key) or 'ACME'.
    All financial figures satisfy strict GAAP accounting identities.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ticker = ticker.strip().upper()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1: Cover & Consolidated Statements of Income & Balance Sheet
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    profiles = {
        "IBM": (
            "International Business Machines Corporation",
            (62753.0, 27300.0, 35453.0, 10074.0, 8836.0, 9.55),
            (14800.0, 137175.0, 54980.0, 112450.0, 24725.0),
            (13445.0, 1845.0, 11600.0, 925.2),
        ),
        "MSFT": (
            "Microsoft Corporation",
            (245122.0, 74114.0, 171008.0, 109433.0, 88136.0, 11.80),
            (75543.0, 512163.0, 51990.0, 243686.0, 268477.0),
            (118548.0, 44477.0, 74071.0, 7469.0),
        ),
        "AAPL": (
            "Apple Inc.",
            (391035.0, 210352.0, 180683.0, 123216.0, 93736.0, 6.08),
            (65171.0, 364980.0, 106629.0, 308030.0, 56950.0),
            (118254.0, 9447.0, 108807.0, 15408.0),
        ),
        "NVDA": (
            "NVIDIA Corporation",
            (130497.0, 32624.0, 97873.0, 81453.0, 72880.0, 2.94),
            (43210.0, 111601.0, 8463.0, 32274.0, 79327.0),
            (64089.0, 3236.0, 60853.0, 24800.0),
        ),
        "GOOGL": (
            "Alphabet Inc.",
            (350018.0, 146306.0, 203712.0, 112390.0, 100118.0, 8.04),
            (95657.0, 450256.0, 10883.0, 125172.0, 325084.0),
            (125299.0, 52535.0, 72764.0, 12450.0),
        ),
    }
    if ticker in profiles:
        company_name, (rev, cor, gp, opinc, netinc, eps), (cash, assets, debt, liab, equity), (ocf, capex, fcf, shares) = profiles[ticker]
    else:
        company_name = "Acme Cloud & AI Systems Inc."
        rev, cor, gp, opinc, netinc, eps = 12500.0, 4500.0, 8000.0, 3200.0, 2480.0, 6.20
        cash, assets, debt, liab, equity = 4200.0, 22000.0, 3100.0, 8500.0, 13500.0
        ocf, capex, fcf, shares = 3450.0, 650.0, 2800.0, 400.0

    pdf.cell(0, 10, "UNITED STATES SECURITIES AND EXCHANGE COMMISSION", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"FORM 10-K - ANNUAL REPORT (FY2025) - {company_name} ({ticker})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    body_page_1 = (
        f"Company Name: {company_name}\n"
        f"Ticker Symbol: {ticker}\n"
        f"Filing Type: 10-K\n"
        f"Fiscal Year: 2025 (Period: FY2025)\n\n"
        f"ITEM 8. CONSOLIDATED FINANCIAL STATEMENTS (All dollar figures in USD Millions, except per-share data)\n"
        f"---------------------------------------------------------------------------------------------\n"
        f"CONSOLIDATED STATEMENT OF EARNINGS (FY2025):\n"
        f"  Total Revenue:                       ${rev:,.1f} Million\n"
        f"  Cost of Revenue:                     ${cor:,.1f} Million\n"
        f"  Gross Profit:                        ${gp:,.1f} Million\n"
        f"  Operating Income (EBIT):             ${opinc:,.1f} Million\n"
        f"  Net Income (GAAP):                   ${netinc:,.1f} Million\n"
        f"  Diluted Earnings Per Share (EPS):    ${eps:.2f}\n"
        f"  Weighted Avg Diluted Shares:         {shares:.1f} Million\n\n"
        f"CONSOLIDATED BALANCE SHEET (As of December 31, 2025):\n"
        f"  Cash and Marketable Securities:      ${cash:,.1f} Million\n"
        f"  Total Assets:                        ${assets:,.1f} Million\n"
        f"  Total Debt (Short + Long-Term):      ${debt:,.1f} Million\n"
        f"  Total Liabilities:                   ${liab:,.1f} Million\n"
        f"  Total Shareholders' Equity:          ${equity:,.1f} Million\n\n"
        f"CONSOLIDATED STATEMENT OF CASH FLOWS (FY2025):\n"
        f"  Operating Cash Flow (OCF):           ${ocf:,.1f} Million\n"
        f"  Capital Expenditures (CapEx):        ${capex:,.1f} Million\n"
        f"  Free Cash Flow (FCF):                ${fcf:,.1f} Million\n"
    )
    pdf.multi_cell(0, 6, body_page_1)

    # Page 2: Item 1A Risk Factors & Item 7 MD&A Guidance
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "ITEM 1A. RISK FACTORS & ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    body_page_2 = (
        "KEY RISK DISCLOSURES:\n"
        "1. [AI Competition & Technological Transition - Severity: HIGH] Rapid enterprise shift toward "
        "generative AI and sovereign hybrid cloud architectures requires sustained R&D capital allocation and "
        "specialized talent retention to defend market share against hyperscale cloud competitors.\n"
        "2. [Macroeconomic & Enterprise IT Spending Cycles - Severity: MEDIUM] Prolonged elevated interest rates "
        "and foreign exchange volatility could lengthen enterprise consulting contract cycles and defer discretionary IT spend.\n"
        "3. [Cybersecurity & Regulatory Compliance - Severity: HIGH] Evolving global AI governance frameworks "
        "(EU AI Act, US federal compliance) and sophisticated supply-chain cyber threats create ongoing legal and operational exposure.\n\n"
        "MANAGEMENT GUIDANCE & OUTLOOK (FY2026):\n"
        "Management projects constant-currency revenue growth of 5% to 7% in FY2026 driven by enterprise generative AI "
        "bookings and Red Hat hybrid cloud software expansion, with full-year Free Cash Flow expected to exceed "
        f"${fcf * 1.06:,.0f} Million alongside expanding operating margins."
    )
    pdf.multi_cell(0, 6, body_page_2)
    pdf.output(str(dest))
    return dest
