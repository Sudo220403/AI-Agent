"""OpenTelemetry distributed tracing, JSONL span exporting, and LangSmith integration."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

from src.config import get_settings


class JSONLSpanExporter(SpanExporter):
    """
    Exports OpenTelemetry spans to a structured local JSONL file (`logs/otel_traces.jsonl`)
    and keeps an in-memory buffer so CLI dashboards and unit tests can inspect exact spans.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.exported_spans: List[Dict[str, Any]] = []

    def export(self, spans: tuple[ReadableSpan, ...]) -> SpanExportResult:
        try:
            with self.file_path.open("a", encoding="utf-8") as f:
                for span in spans:
                    record = {
                        "trace_id": f"{span.context.trace_id:032x}",
                        "span_id": f"{span.context.span_id:016x}",
                        "parent_span_id": (
                            f"{span.parent.span_id:016x}" if span.parent else None
                        ),
                        "name": span.name,
                        "status": span.status.status_code.name,
                        "status_description": span.status.description,
                        "start_time_ns": span.start_time,
                        "end_time_ns": span.end_time,
                        "duration_ms": round(
                            ((span.end_time or 0) - (span.start_time or 0)) / 1_000_000,
                            3,
                        ),
                        "attributes": dict(span.attributes or {}),
                        "events": [
                            {
                                "name": evt.name,
                                "timestamp_ns": evt.timestamp,
                                "attributes": dict(evt.attributes or {}),
                            }
                            for evt in span.events
                        ],
                        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    self.exported_spans.append(record)
                    f.write(json.dumps(record) + "\n")
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def clear_memory(self) -> None:
        self.exported_spans.clear()


class TelemetryManager:
    """Singleton manager for OpenTelemetry Tracer and JSONL Span Exporter."""

    _instance: Optional["TelemetryManager"] = None

    def __init__(self) -> None:
        settings = get_settings()
        self.langsmith_enabled = settings.configure_langsmith()
        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.version": "0.1.0",
                "langsmith.enabled": self.langsmith_enabled,
            }
        )
        self.provider = TracerProvider(resource=resource)
        self.jsonl_exporter = JSONLSpanExporter(settings.otel_traces_export_path)
        self.provider.add_span_processor(SimpleSpanProcessor(self.jsonl_exporter))
        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer("finagent.langgraph")

    @classmethod
    def get_instance(cls) -> "TelemetryManager":
        if cls._instance is None:
            cls._instance = TelemetryManager()
        return cls._instance

    @contextmanager
    def start_node_span(
        self, node_name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Generator[trace.Span, None, None]:
        """Context manager that wraps a LangGraph node or tool execution in an OTel span."""
        with self.tracer.start_as_current_span(f"langgraph.node.{node_name}") as span:
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        span.set_attribute(k, str(v) if not isinstance(v, (bool, int, float)) else v)
            start_ns = time.perf_counter_ns()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                span.set_attribute("node.duration_ms", round(elapsed_ms, 2))
