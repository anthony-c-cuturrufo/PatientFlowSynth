"""Exact response-level cost accounting for OpenRouter non-streaming responses."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any


class OpenRouterCostError(ValueError):
    """Raised when an OpenRouter response cannot support exact cost accounting."""


class OpenRouterResponseCostLimitError(RuntimeError):
    """Raised before a request that would start at or above the approved ceiling."""


class OpenRouterResponseCostGuard:
    """Accumulate service-reported response costs and stop before the next request."""

    def __init__(
        self,
        *,
        prior_cost_usd: Decimal | str,
        hard_cap_usd: Decimal | str,
    ) -> None:
        self.prior_cost_usd = _nonnegative_decimal(prior_cost_usd, "prior_cost_usd")
        self.hard_cap_usd = _positive_decimal(hard_cap_usd, "hard_cap_usd")
        if self.prior_cost_usd > self.hard_cap_usd:
            raise OpenRouterCostError("prior_cost_usd exceeds hard_cap_usd")
        self.observed_cost_usd = Decimal(0)
        self.response_count = 0
        self._response_ids: set[str] = set()

    @property
    def cumulative_cost_usd(self) -> Decimal:
        return self.prior_cost_usd + self.observed_cost_usd

    def before_request(self) -> None:
        if self.cumulative_cost_usd >= self.hard_cap_usd:
            raise OpenRouterResponseCostLimitError(
                "The approved OpenRouter cost ceiling was reached before the next request."
            )

    def after_response(
        self,
        response_id: str | None,
        response_metadata: Mapping[str, Any],
    ) -> None:
        if response_metadata.get("gateway") != "openrouter":
            raise OpenRouterCostError("response did not identify the OpenRouter gateway")
        if not isinstance(response_id, str) or not response_id:
            raise OpenRouterCostError("response has no response_id")
        if response_id in self._response_ids:
            raise OpenRouterCostError("duplicate response_id in guarded execution")
        cost = response_cost_usd(response_metadata)
        self._response_ids.add(response_id)
        self.observed_cost_usd += cost
        self.response_count += 1


def response_cost_usd(response_metadata: Mapping[str, Any]) -> Decimal:
    """Return one service-reported OpenRouter response cost as an exact Decimal."""

    if response_metadata.get("openrouter_cost_source") != "response_usage.cost":
        raise OpenRouterCostError("response cost does not come from OpenRouter usage.cost")
    return _nonnegative_decimal(
        response_metadata.get("openrouter_cost_usd"),
        "openrouter_cost_usd",
    )


def summarize_openrouter_response_costs(
    responses: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Audit a response sequence, rejecting missing or repeated generation IDs."""

    response_ids: set[str] = set()
    total = Decimal(0)
    by_provider: dict[str, Decimal] = {}
    by_model: dict[str, Decimal] = {}
    for response_id, metadata in responses:
        if not response_id:
            raise OpenRouterCostError("response has no response_id")
        if response_id in response_ids:
            raise OpenRouterCostError("duplicate response_id in OpenRouter cost audit")
        response_ids.add(response_id)
        cost = response_cost_usd(metadata)
        total += cost
        provider = _required_text(metadata, "returned_provider_identifier")
        model = _required_text(metadata, "returned_model_identifier")
        by_provider[provider] = by_provider.get(provider, Decimal(0)) + cost
        by_model[model] = by_model.get(model, Decimal(0)) + cost
    return {
        "response_count": len(responses),
        "cost_usd": _decimal_text(total),
        "response_ids": sorted(response_ids),
        "cost_by_returned_provider_usd": {
            key: _decimal_text(value) for key, value in sorted(by_provider.items())
        },
        "cost_by_returned_model_usd": {
            key: _decimal_text(value) for key, value in sorted(by_model.items())
        },
    }


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise OpenRouterCostError(f"{key} must be a nonempty string")
    return item


def _nonnegative_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise OpenRouterCostError(f"{label} must be a nonnegative decimal")
    if isinstance(value, float) and not math.isfinite(value):
        raise OpenRouterCostError(f"{label} must be a nonnegative decimal")
    try:
        item = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise OpenRouterCostError(f"{label} must be a nonnegative decimal") from exc
    if not item.is_finite() or item < 0:
        raise OpenRouterCostError(f"{label} must be a nonnegative decimal")
    return item


def _positive_decimal(value: Any, label: str) -> Decimal:
    item = _nonnegative_decimal(value, label)
    if item == 0:
        raise OpenRouterCostError(f"{label} must be a positive decimal")
    return item


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")
