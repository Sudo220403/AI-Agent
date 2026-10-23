"""Resilient Alpha Vantage Market API client with Tenacity backoff, Circuit Breaker, and Fallback Cache."""

from __future__ import annotations

from typing import Optional, Tuple
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import get_settings
from src.observability import TelemetryManager
from src.schemas import DataSourceStatus, MarketSnapshot
from src.tools import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    TransientAPIError,
    get_fallback_market_snapshot,
)


class AlphaVantageClient:
    """
    Production HTTP client for Alpha Vantage (`GLOBAL_QUOTE` and `OVERVIEW`).
    Features:
    - Exponential backoff + random jitter via `tenacity` on HTTP 429/502/503, network timeouts,
      and Alpha Vantage's JSON-level rate-limit payloads (`"Note"` / `"Information"`).
    - Stateful CircuitBreaker (`CLOSED` -> `OPEN` -> `HALF_OPEN`).
    - Automatic graceful degradation to verified fallback snapshot if retries are exhausted
      or the circuit breaker is open.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.alphavantage_api_key
        self.base_url = settings.alphavantage_base_url
        self.timeout = settings.api_timeout_seconds
        self.max_retries = settings.max_api_retries
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_timeout_seconds=settings.circuit_breaker_recovery_timeout,
        )
        self._Transient_failure_injected_once = False

    def fetch_market_snapshot(
        self,
        ticker: str,
        inject_failure_mode: Optional[str] = None,
    ) -> Tuple[MarketSnapshot, int, str]:
        """
        Query live market data for `ticker`.
        Returns `(MarketSnapshot, retry_attempts_used, circuit_breaker_state)`.
        """
        normalized_ticker = ticker.strip().upper()
        telemetry = TelemetryManager.get_instance()
        attempt_counter = {"count": 0}

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=0.35, max=2.5, jitter=0.25),
            retry=retry_if_exception_type(TransientAPIError),
        )
        def _execute_with_retry() -> MarketSnapshot:
            self.circuit_breaker.before_call()
            attempt_counter["count"] += 1
            current_attempt = attempt_counter["count"]

            with telemetry.start_node_span(
                "alphavantage_http_request",
                attributes={
                    "market.ticker": normalized_ticker,
                    "http.attempt": current_attempt,
                    "circuit_breaker.state": self.circuit_breaker.state,
                },
            ) as span:
                try:
                    # Fault-injection hooks for testing and resilience demonstrations
                    if inject_failure_mode == "rate_limit_once" and not self._Transient_failure_injected_once:
                        self._Transient_failure_injected_once = True
                        span.add_event(
                            "injected_http_429_rate_limit",
                            {"detail": "Simulated Alpha Vantage 429 burst limit on attempt 1"},
                        )
                        raise TransientAPIError(
                            "Simulated HTTP 429 Too Many Requests from Alpha Vantage API"
                        )

                    if inject_failure_mode == "total_outage":
                        span.add_event(
                            "injected_http_503_outage",
                            {"detail": f"Simulated upstream 503 outage on attempt {current_attempt}"},
                        )
                        raise TransientAPIError(
                            f"Simulated HTTP 503 Service Unavailable on attempt {current_attempt}"
                        )

                    snapshot = self._call_alphavantage_endpoints(normalized_ticker)
                    self.circuit_breaker.record_success()
                    span.set_attribute("market.source_status", snapshot.source_status.value)
                    return snapshot

                except TransientAPIError:
                    self.circuit_breaker.record_failure()
                    span.set_attribute(
                        "circuit_breaker.state_after_failure", self.circuit_breaker.state
                    )
                    raise

        try:
            snapshot = _execute_with_retry()
            retries_used = max(0, attempt_counter["count"] - 1)
            return snapshot, retries_used, self.circuit_breaker.state
        except (TransientAPIError, CircuitBreakerOpenError) as exc:
            retries_used = max(0, attempt_counter["count"] - 1)
            fallback = get_fallback_market_snapshot(
                normalized_ticker, reason=f"{type(exc).__name__}: {exc}"
            )
            return fallback, retries_used, self.circuit_breaker.state

    def _call_alphavantage_endpoints(self, ticker: str) -> MarketSnapshot:
        """Call Alpha Vantage GLOBAL_QUOTE and OVERVIEW endpoints via httpx."""
        # Note: Alpha Vantage's free "demo" key officially supports IBM for GLOBAL_QUOTE & OVERVIEW
        effective_key = self.api_key
        is_demo = effective_key.lower() == "demo"

        if is_demo and ticker != "IBM":
            raise TransientAPIError(
                f"Alpha Vantage 'demo' key only supports ticker 'IBM' (requested '{ticker}'). "
                "Provide ALPHAVANTAGE_API_KEY in .env for arbitrary live tickers."
            )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                quote_resp = client.get(
                    self.base_url,
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": ticker,
                        "apikey": effective_key,
                    },
                )
                if quote_resp.status_code in (429, 500, 502, 503, 504):
                    raise TransientAPIError(
                        f"Alpha Vantage returned HTTP {quote_resp.status_code}"
                    )
                quote_resp.raise_for_status()
                quote_payload = quote_resp.json()

                # Check for Alpha Vantage JSON-level rate limiting or informational block
                if "Note" in quote_payload or "Information" in quote_payload:
                    msg = quote_payload.get("Note") or quote_payload.get("Information")
                    raise TransientAPIError(f"Alpha Vantage API rate limit message: {msg}")

                global_quote = quote_payload.get("Global Quote") or {}
                if not global_quote or "05. price" not in global_quote:
                    raise TransientAPIError(
                        f"Empty or malformed Global Quote payload for {ticker}: {quote_payload}"
                    )

                # Fetch company overview for 52-week range, beta, and sector
                overview_resp = client.get(
                    self.base_url,
                    params={
                        "function": "OVERVIEW",
                        "symbol": ticker,
                        "apikey": effective_key,
                    },
                )
                overview_payload = (
                    overview_resp.json() if overview_resp.status_code == 200 else {}
                )

        except httpx.RequestError as exc:
            raise TransientAPIError(f"Network transport error contacting Alpha Vantage: {exc}") from exc

        price = float(global_quote["05. price"])
        prev_close = float(global_quote.get("08. previous close") or price)
        change_pct_raw = str(global_quote.get("10. change percent", "0%")).replace("%", "").strip()
        daily_change_pct = float(change_pct_raw) if change_pct_raw else 0.0
        volume = int(float(global_quote.get("06. volume", 0)))

        high_52 = _safe_float(overview_payload.get("52WeekHigh"), default=round(price * 1.15, 2))
        low_52 = _safe_float(overview_payload.get("52WeekLow"), default=round(price * 0.80, 2))
        if low_52 > high_52:
            low_52, high_52 = min(low_52, price), max(high_52, price)

        beta = _safe_float(overview_payload.get("Beta"), default=0.85)
        sector = str(overview_payload.get("Sector") or "Information Technology")
        industry = str(overview_payload.get("Industry") or "Enterprise Software & Services")

        return MarketSnapshot(
            ticker=ticker,
            current_price=round(price, 2),
            previous_close=round(prev_close, 2),
            daily_change_pct=round(daily_change_pct, 2),
            volume=volume,
            week_52_high=round(max(high_52, price), 2),
            week_52_low=round(min(low_52, price), 2),
            beta=round(beta, 2),
            sector=sector,
            industry=industry,
            source_status=(
                DataSourceStatus.DEMO_ENDPOINT
                if is_demo
                else DataSourceStatus.LIVE_API
            ),
        )


def _safe_float(val: object, default: float) -> float:
    try:
        if val is None or val in ("None", "-", ""):
            return default
        return float(str(val))
    except (ValueError, TypeError):
        return default
