from __future__ import annotations

from typing import Literal

TruthPolarity = Literal["supported", "refuted", "insufficient_evidence"]

POSITIVE_STATUSES = frozenset({"active", "completed", "final", "amended", "corrected"})
NEGATIVE_STATUSES = frozenset({"cancelled", "not-done", "refuted"})
UNKNOWN_STATUSES = frozenset(
    {"draft", "entered-in-error", "unknown", "preliminary", "registered", "on-hold"}
)


def lifecycle_polarity(
    resource_type: str,
    status: str | None,
    *,
    has_value: bool = True,
) -> TruthPolarity:
    """Return the shared coarse lifecycle polarity used by diagnostics and reports.

    Task programs retain resource-specific decision rules; this helper prevents
    diagnostic adapters and secondary analyses from silently drifting apart.
    """

    if resource_type == "Condition":
        if status == "confirmed":
            return "supported"
        if status == "refuted":
            return "refuted"
        return "insufficient_evidence"
    if resource_type == "Observation":
        if status in {"final", "amended", "corrected"} and has_value:
            return "supported"
        if status == "cancelled":
            return "refuted"
        return "insufficient_evidence"
    if resource_type in {"Procedure", "Immunization"}:
        if status == "completed":
            return "supported"
        if status == "not-done":
            return "refuted"
        return "insufficient_evidence"
    if resource_type == "MedicationAdministration":
        # The endpoint is whether any administration occurred, not whether the
        # administration ran to completion. FHIR R4 ``stopped`` records real-world
        # activity that began and was then halted; ``not-done`` records no activity.
        if status in {"completed", "stopped"}:
            return "supported"
        if status == "not-done":
            return "refuted"
        return "insufficient_evidence"
    if resource_type == "MedicationRequest":
        # This is current-actionability semantics, not the historical claim
        # that a request was ever authored.
        if status == "active":
            return "supported"
        if status in {"cancelled", "completed", "stopped"}:
            return "refuted"
        return "insufficient_evidence"
    if status in POSITIVE_STATUSES:
        return "supported"
    if status in NEGATIVE_STATUSES:
        return "refuted"
    return "insufficient_evidence"
