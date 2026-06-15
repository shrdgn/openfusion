"""Cost-control policy for bounded upstream requests."""

from __future__ import annotations

import copy
from enum import StrEnum
from typing import Any

from openfusion.config import CostControlsConfig
from openfusion.errors import InvalidRequestError


class RequestPhase(StrEnum):
    PASS_THROUGH = "pass_through"
    PANEL = "panel"
    JUDGE = "judge"


class CostPolicy:
    """Apply configured token ceilings without logging prompt content or secrets."""

    def __init__(self, config: CostControlsConfig) -> None:
        self._config = config

    def validate_max_tokens(self, body: dict[str, Any]) -> None:
        requested = body.get("max_tokens")
        if requested is None:
            return
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise InvalidRequestError("max_tokens must be an integer", code="invalid_max_tokens")
        if requested < 1:
            raise InvalidRequestError(
                "max_tokens must be greater than 0",
                code="invalid_max_tokens",
            )

    def apply_token_limit(
        self,
        body: dict[str, Any],
        phase: RequestPhase,
        *,
        reject_over_limit: bool,
    ) -> dict[str, Any]:
        limit = self._limit_for(phase)
        if limit is None:
            return copy.deepcopy(body)

        capped = copy.deepcopy(body)
        requested = capped.get("max_tokens")
        if requested is None:
            capped["max_tokens"] = limit
            return capped
        self.validate_max_tokens(capped)
        if requested <= limit:
            return capped
        if reject_over_limit:
            raise InvalidRequestError(
                f"max_tokens {requested} exceeds configured {phase.value} limit {limit}",
                code="max_tokens_exceeds_limit",
            )
        capped["max_tokens"] = limit
        return capped

    def _limit_for(self, phase: RequestPhase) -> int | None:
        if phase == RequestPhase.PASS_THROUGH:
            return self._config.pass_through_max_tokens
        if phase == RequestPhase.PANEL:
            return self._config.panel_max_tokens
        if phase == RequestPhase.JUDGE:
            return self._config.judge_max_tokens
        raise ValueError(f"Unknown request phase: {phase}")
