"""In-process metrics registry with Prometheus text exposition.

Dependency-free and thread-safe. Recording happens at two chokepoints:
``upstream._log_request`` (every upstream call) and the server route handler
(every client-facing request). Exposed at ``GET /metrics``.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


def _labels(pairs: dict[str, str]) -> str:
    if not pairs:
        return ""
    body = ",".join(f'{key}="{value}"' for key, value in sorted(pairs.items()))
    return "{" + body + "}"


class Metrics:
    """Cumulative counters, latency sums, and token/cost totals.

    Latency is tracked as a Prometheus-style summary (``_count`` + ``_sum``)
    rather than a histogram to stay lightweight and free of bucket config.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> labels-tuple -> value
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._latency_count: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._latency_sum: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._upstream_latency_count: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._upstream_latency_sum: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def _add(self, name: str, labels: dict[str, str], value: float) -> None:
        key = tuple(sorted(labels.items()))
        bucket = self._counters[name]
        bucket[key] = bucket.get(key, 0.0) + value

    def record_request(self, *, route: str, outcome: str, latency_ms: float) -> None:
        """Record a client-facing request (fusion / pass_through) outcome."""
        with self._lock:
            self._add("openfusion_requests_total", {"route": route, "outcome": outcome}, 1)
            key = (("route", route),)
            self._latency_count[key] += 1
            self._latency_sum[key] += latency_ms

    def record_upstream(
        self,
        *,
        phase: str,
        outcome: str,
        latency_ms: float,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Record a single upstream call: outcome, latency, tokens, cost."""
        with self._lock:
            self._add(
                "openfusion_upstream_requests_total",
                {"phase": phase, "outcome": outcome},
                1,
            )
            key = (("phase", phase),)
            self._upstream_latency_count[key] += 1
            self._upstream_latency_sum[key] += latency_ms
            if usage:
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if isinstance(prompt, (int, float)):
                    self._add(
                        "openfusion_tokens_total", {"phase": phase, "kind": "prompt"}, prompt
                    )
                if isinstance(completion, (int, float)):
                    self._add(
                        "openfusion_tokens_total",
                        {"phase": phase, "kind": "completion"},
                        completion,
                    )
                cost = usage.get("cost")
                if isinstance(cost, (int, float)):
                    self._add("openfusion_cost_usd_total", {"phase": phase}, cost)

    def record_panel(self, *, succeeded: int, failed: int) -> None:
        """Record panel-member outcomes for one fusion request."""
        with self._lock:
            if succeeded:
                self._add("openfusion_panel_members_total", {"outcome": "success"}, succeeded)
            if failed:
                self._add("openfusion_panel_members_total", {"outcome": "failure"}, failed)

    def value(self, name: str, **labels: str) -> float:
        """Return a single counter value for the given labels (0 if unset)."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._counters.get(name, {}).get(key, 0.0)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly view of all series (used by tests/JSON callers)."""
        with self._lock:
            counters = {
                name: [{"labels": dict(key), "value": value} for key, value in series.items()]
                for name, series in self._counters.items()
            }
            request_latency = {
                dict(key)["route"]: {
                    "count": self._latency_count[key],
                    "sum_ms": self._latency_sum[key],
                }
                for key in self._latency_count
            }
            upstream_latency = {
                dict(key)["phase"]: {
                    "count": self._upstream_latency_count[key],
                    "sum_ms": self._upstream_latency_sum[key],
                }
                for key in self._upstream_latency_count
            }
        return {
            "counters": counters,
            "request_latency_ms": request_latency,
            "upstream_latency_ms": upstream_latency,
        }

    def render_prometheus(self) -> str:
        """Render the registry in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            counter_help = {
                "openfusion_requests_total": "Client-facing requests by route and outcome.",
                "openfusion_upstream_requests_total": "Upstream calls by phase and outcome.",
                "openfusion_panel_members_total": "Panel-member outcomes across fusion requests.",
                "openfusion_tokens_total": "Tokens consumed by phase and kind.",
                "openfusion_cost_usd_total": "Reported upstream cost (USD) by phase.",
            }
            for name, series in sorted(self._counters.items()):
                lines.append(f"# HELP {name} {counter_help.get(name, name)}")
                lines.append(f"# TYPE {name} counter")
                for key, value in sorted(series.items()):
                    lines.append(f"{name}{_labels(dict(key))} {_fmt(value)}")

            self._render_summary(
                lines,
                name="openfusion_request_latency_ms",
                help_text="Client-facing request latency in milliseconds.",
                counts=self._latency_count,
                sums=self._latency_sum,
            )
            self._render_summary(
                lines,
                name="openfusion_upstream_latency_ms",
                help_text="Upstream call latency in milliseconds.",
                counts=self._upstream_latency_count,
                sums=self._upstream_latency_sum,
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_summary(
        lines: list[str],
        *,
        name: str,
        help_text: str,
        counts: dict[tuple[tuple[str, str], ...], int],
        sums: dict[tuple[tuple[str, str], ...], float],
    ) -> None:
        if not counts:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} summary")
        for key in sorted(counts):
            label_map = dict(key)
            lines.append(f"{name}_count{_labels(label_map)} {counts[key]}")
            lines.append(f"{name}_sum{_labels(label_map)} {_fmt(sums[key])}")

    def reset(self) -> None:
        """Clear all series (test helper)."""
        with self._lock:
            self._counters.clear()
            self._latency_count.clear()
            self._latency_sum.clear()
            self._upstream_latency_count.clear()
            self._upstream_latency_sum.clear()


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(value)


METRICS = Metrics()
