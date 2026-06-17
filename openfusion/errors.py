"""OpenAI-compatible error types and response helpers."""

from __future__ import annotations

from typing import Any


class OpenFusionError(Exception):
    """Base error with OpenAI-compatible fields."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "api_error",
        code: str | None = None,
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.code = code
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "message": self.message,
                "type": self.error_type,
            }
        }
        if self.code is not None:
            payload["error"]["code"] = self.code
        return payload


class InvalidRequestError(OpenFusionError):
    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(
            message,
            error_type="invalid_request_error",
            code=code,
            status_code=400,
        )


class UpstreamError(OpenFusionError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "upstream_error",
        status_code: int = 502,
    ) -> None:
        super().__init__(
            message,
            error_type="upstream_error",
            code=code,
            status_code=status_code,
        )
        self.upstream_status_code = status_code


class AuthenticationError(OpenFusionError):
    def __init__(self, message: str = "Invalid API key") -> None:
        super().__init__(
            message,
            error_type="invalid_request_error",
            code="invalid_api_key",
            status_code=401,
        )


class RateLimitError(OpenFusionError):
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(
            message,
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429,
        )


class OverloadedError(OpenFusionError):
    def __init__(self, message: str = "Server busy: too many concurrent requests") -> None:
        super().__init__(
            message,
            error_type="overloaded_error",
            code="server_overloaded",
            status_code=503,
        )
