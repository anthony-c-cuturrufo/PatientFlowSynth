"""Exact, response-level OpenAI text-token cost accounting.

The helpers in this module deliberately accept an explicit sealed pricing mapping.
They do not fetch prices, infer a model, or hide a mutable rate card in module globals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any


class OpenAICostError(ValueError):
    """Raised when response usage cannot support exact sealed-price accounting."""


class OpenAIResponseCostLimitError(RuntimeError):
    """Raised before a request that would start at or above the approved ceiling."""


class OpenAIResponseCostGuard:
    """Exact response-level accounting with a before-request circuit breaker.

    The guard intentionally checks the ceiling immediately before each request.
    A response already in flight can therefore take the total over the ceiling, but
    no subsequent request is allowed. This is the narrow overshoot policy sealed by
    the paid-study protocol.
    """

    def __init__(
        self,
        pricing: Mapping[str, Any],
        *,
        prior_cost_usd: Decimal | str,
        hard_cap_usd: Decimal | str,
    ) -> None:
        self.pricing = dict(pricing)
        self.prior_cost_usd = _nonnegative_decimal(prior_cost_usd, "prior_cost_usd")
        self.hard_cap_usd = _positive_decimal(hard_cap_usd, "hard_cap_usd")
        if self.prior_cost_usd > self.hard_cap_usd:
            raise OpenAICostError("prior_cost_usd exceeds hard_cap_usd")
        service_tier = self.pricing.get("service_tier")
        if not isinstance(service_tier, str) or not service_tier:
            raise OpenAICostError("service_tier must be a nonempty string")
        self.service_tier = service_tier
        self.observed_cost_usd = Decimal(0)
        self.response_count = 0
        self._response_ids: set[str] = set()

    @property
    def cumulative_cost_usd(self) -> Decimal:
        return self.prior_cost_usd + self.observed_cost_usd

    def before_request(self) -> None:
        if self.cumulative_cost_usd >= self.hard_cap_usd:
            raise OpenAIResponseCostLimitError(
                "The approved OpenAI cost ceiling was reached before the next request."
            )

    def after_response(
        self,
        response_id: str | None,
        response_metadata: Mapping[str, Any],
    ) -> None:
        service_tier = response_metadata.get("service_tier")
        if service_tier != self.service_tier:
            raise OpenAICostError("response service tier differs from the sealed pricing contract")
        if not isinstance(response_id, str) or not response_id:
            raise OpenAICostError("response has no response_id")
        if response_id in self._response_ids:
            raise OpenAICostError("duplicate response_id in guarded execution")
        usage = response_metadata.get("token_usage")
        if not isinstance(usage, Mapping):
            raise OpenAICostError("response metadata has no exact token_usage")
        priced = price_standard_text_response(usage, self.pricing)
        self._response_ids.add(response_id)
        self.observed_cost_usd += Decimal(str(priced["cost_usd"]))
        self.response_count += 1


def price_standard_text_response(
    usage: Mapping[str, Any],
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    """Price one response under an explicit standard-tier text-token schedule."""

    input_tokens = _required_nonnegative_int(usage, "input_tokens")
    cached_tokens = _required_nonnegative_int(usage, "cached_input_tokens")
    cache_write_tokens = _required_nonnegative_int(usage, "cache_write_input_tokens")
    output_tokens = _required_nonnegative_int(usage, "output_tokens")
    total_tokens = _required_nonnegative_int(usage, "total_tokens")
    if input_tokens + output_tokens != total_tokens:
        raise OpenAICostError("response token total differs from input plus output")
    if cached_tokens + cache_write_tokens > input_tokens:
        raise OpenAICostError("cached and cache-write tokens exceed response input tokens")

    threshold = _required_positive_int(pricing, "long_context_input_threshold_tokens")
    input_multiplier = _required_positive_decimal(pricing, "long_context_input_multiplier")
    output_multiplier = _required_positive_decimal(pricing, "long_context_output_multiplier")
    long_context = input_tokens > threshold
    effective_input_multiplier = input_multiplier if long_context else Decimal(1)
    effective_output_multiplier = output_multiplier if long_context else Decimal(1)

    input_rate = _required_positive_decimal(pricing, "input_usd_per_million")
    cached_rate = _required_positive_decimal(pricing, "cached_input_usd_per_million")
    cache_write_rate = _required_positive_decimal(pricing, "cache_write_input_usd_per_million")
    output_rate = _required_positive_decimal(pricing, "output_usd_per_million")
    uncached_tokens = input_tokens - cached_tokens - cache_write_tokens
    million = Decimal(1_000_000)
    components = {
        "uncached_input_usd": (
            Decimal(uncached_tokens) * input_rate * effective_input_multiplier / million
        ),
        "cached_input_usd": (
            Decimal(cached_tokens) * cached_rate * effective_input_multiplier / million
        ),
        "cache_write_input_usd": (
            Decimal(cache_write_tokens) * cache_write_rate * effective_input_multiplier / million
        ),
        "output_usd": (
            Decimal(output_tokens) * output_rate * effective_output_multiplier / million
        ),
    }
    cost = sum(components.values(), Decimal(0))
    return {
        "input_tokens": input_tokens,
        "uncached_input_tokens": uncached_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_write_input_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "long_context_pricing_applied": long_context,
        "effective_input_multiplier": _decimal_text(effective_input_multiplier),
        "effective_output_multiplier": _decimal_text(effective_output_multiplier),
        "cost_components_usd": {key: _decimal_text(value) for key, value in components.items()},
        "cost_usd": _decimal_text(cost),
    }


def summarize_standard_text_costs(
    usages: Sequence[Mapping[str, Any]],
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    """Price a response sequence and return exact additive audit totals."""

    priced = [price_standard_text_response(usage, pricing) for usage in usages]
    totals = {
        key: sum(int(row[key]) for row in priced)
        for key in (
            "input_tokens",
            "uncached_input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "total_tokens",
        )
    }
    total_cost = sum((Decimal(str(row["cost_usd"])) for row in priced), Decimal(0))
    return {
        "response_count": len(priced),
        **totals,
        "long_context_response_count": sum(
            bool(row["long_context_pricing_applied"]) for row in priced
        ),
        "cost_usd": _decimal_text(total_cost),
        "responses": priced,
    }


def _required_nonnegative_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int or item < 0:
        raise OpenAICostError(f"{key} must be a nonnegative integer")
    return item


def _required_positive_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int or item <= 0:
        raise OpenAICostError(f"{key} must be a positive integer")
    return item


def _required_positive_decimal(value: Mapping[str, Any], key: str) -> Decimal:
    try:
        item = Decimal(str(value[key]))
    except (KeyError, ValueError) as exc:
        raise OpenAICostError(f"{key} must be a positive decimal") from exc
    if not item.is_finite() or item <= 0:
        raise OpenAICostError(f"{key} must be a positive decimal")
    return item


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _nonnegative_decimal(value: Decimal | str, label: str) -> Decimal:
    try:
        item = Decimal(str(value))
    except ValueError as exc:
        raise OpenAICostError(f"{label} must be a nonnegative decimal") from exc
    if not item.is_finite() or item < 0:
        raise OpenAICostError(f"{label} must be a nonnegative decimal")
    return item


def _positive_decimal(value: Decimal | str, label: str) -> Decimal:
    item = _nonnegative_decimal(value, label)
    if item == 0:
        raise OpenAICostError(f"{label} must be a positive decimal")
    return item
