from __future__ import annotations

import base64
import copy
import hashlib
import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from patient_tool_gym.evidence import (
    EvidenceRecord,
    PatientSnapshot,
    clinical_instant_sort_key,
)

TOOL_MANIFEST_VERSION = "patient-tool-gym.tools.v5"
_CONDITION_STATUS_CODES = frozenset(
    {"active", "inactive", "recurrence", "relapse", "remission", "resolved"}
)
_MEDICATION_STATUS_CODES = frozenset(
    {
        "active",
        "on-hold",
        "cancelled",
        "completed",
        "entered-in-error",
        "stopped",
        "draft",
        "unknown",
    }
)
_PROCEDURE_STATUS_CODES = frozenset(
    {
        "preparation",
        "in-progress",
        "not-done",
        "on-hold",
        "stopped",
        "completed",
        "entered-in-error",
        "unknown",
    }
)
_SERVICE_REQUEST_STATUS_CODES = frozenset(
    {
        "draft",
        "active",
        "on-hold",
        "revoked",
        "completed",
        "entered-in-error",
        "unknown",
    }
)
FHIR_R4_ENCOUNTER_STATUS_CODES = (
    "planned",
    "arrived",
    "triaged",
    "in-progress",
    "onleave",
    "finished",
    "cancelled",
    "entered-in-error",
    "unknown",
)


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetMyProfileArguments(ToolArguments):
    pass


class SearchClinicalRecordsArguments(ToolArguments):
    category: str | None = None
    code: str | None = None
    system: str | None = None
    encounter_id: str | None = None
    cursor: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    limit: int = Field(default=10, ge=1, le=100)
    text: str | None = None


class SearchObservationsArguments(SearchClinicalRecordsArguments):
    pass


class SearchEncountersArguments(ToolArguments):
    code: str | None = None
    system: str | None = None
    status: str | None = None
    cursor: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    limit: int = Field(default=10, ge=1, le=100)
    text: str | None = None


class SearchConditionsArguments(SearchClinicalRecordsArguments):
    clinical_status: str | None = None
    verification_status: str | None = None


class SearchMedicationsArguments(SearchClinicalRecordsArguments):
    record_kind: Literal["request", "administration", "any"]
    status: str | None = None
    intent: str | None = None


class SearchProceduresArguments(SearchClinicalRecordsArguments):
    status: str | None = None


class SearchClinicalDocumentsArguments(SearchClinicalRecordsArguments):
    document_kind: Literal["document_reference", "diagnostic_report", "any"] = "document_reference"
    status: str | None = None
    doc_status: str | None = None


class SearchServiceRequestsArguments(SearchClinicalRecordsArguments):
    status: str | None = None
    intent: str | None = None
    priority: str | None = None


class SearchCarePlansArguments(SearchClinicalRecordsArguments):
    status: (
        Literal[
            "draft",
            "active",
            "on-hold",
            "revoked",
            "completed",
            "entered-in-error",
            "unknown",
        ]
        | None
    ) = None
    intent: Literal["proposal", "plan", "order", "option", "directive"] | None = None


class SearchImmunizationsArguments(SearchClinicalRecordsArguments):
    status: Literal["completed", "entered-in-error", "not-done"] | None = None


class FindRelatedRecordsArguments(ToolArguments):
    anchor_evidence_id: str
    relation: str | None = None
    direction: Literal["outgoing", "incoming", "both"] = "both"
    resource_type: str | None = None
    code: str | None = None
    system: str | None = None
    cursor: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class ReadChartRecordArguments(ToolArguments):
    evidence_id: str


class SubmitAnswerArguments(ToolArguments):
    answer: Any
    evidence_ids: list[str] = Field(default_factory=list)


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    details: dict[str, Any]


class ToolResultSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    records: list[dict[str, Any]]
    searched_scope: dict[str, Any]
    total_matches: int
    complete: bool
    next_cursor: str | None
    snapshot_id: str | None
    error: ToolError | None


class SubmissionResultSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    submission: dict[str, Any] | None
    snapshot_id: str
    error: ToolError | None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments_model: type[ToolArguments]
    output_model: type[BaseModel]
    handler_name: str

    def public_contract(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "manifest_version": TOOL_MANIFEST_VERSION,
            "input_schema": self.arguments_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
        }


_TOOL_SPECS = (
    ToolSpec(
        name="get_my_profile",
        description="Read demographics for the environment-bound patient.",
        arguments_model=GetMyProfileArguments,
        output_model=ToolResultSchema,
        handler_name="_get_my_profile",
    ),
    ToolSpec(
        name="search_encounters",
        description="Search the environment-bound patient's encounters.",
        arguments_model=SearchEncountersArguments,
        output_model=ToolResultSchema,
        handler_name="_search_encounters",
    ),
    ToolSpec(
        name="search_conditions",
        description="Search the environment-bound patient's documented conditions.",
        arguments_model=SearchConditionsArguments,
        output_model=ToolResultSchema,
        handler_name="_search_conditions",
    ),
    ToolSpec(
        name="search_observations",
        description="Search the environment-bound patient's observations.",
        arguments_model=SearchObservationsArguments,
        output_model=ToolResultSchema,
        handler_name="_search_observations",
    ),
    ToolSpec(
        name="search_medications",
        description=(
            "Search the patient's medication requests or administrations without "
            "treating orders as evidence of adherence."
        ),
        arguments_model=SearchMedicationsArguments,
        output_model=ToolResultSchema,
        handler_name="_search_medications",
    ),
    ToolSpec(
        name="search_procedures",
        description="Search the environment-bound patient's documented procedures.",
        arguments_model=SearchProceduresArguments,
        output_model=ToolResultSchema,
        handler_name="_search_procedures",
    ),
    ToolSpec(
        name="search_clinical_documents",
        description=(
            "Search snapshot-visible clinical-document metadata; use read_chart_record for "
            "one selected document."
        ),
        arguments_model=SearchClinicalDocumentsArguments,
        output_model=ToolResultSchema,
        handler_name="_search_clinical_documents",
    ),
    ToolSpec(
        name="search_service_requests",
        description=("Search existing diagnostic, referral, and other service requests read-only."),
        arguments_model=SearchServiceRequestsArguments,
        output_model=ToolResultSchema,
        handler_name="_search_service_requests",
    ),
    ToolSpec(
        name="search_care_plans",
        description="Search the environment-bound patient's care plans read-only.",
        arguments_model=SearchCarePlansArguments,
        output_model=ToolResultSchema,
        handler_name="_search_care_plans",
    ),
    ToolSpec(
        name="search_immunizations",
        description="Search the environment-bound patient's immunization history read-only.",
        arguments_model=SearchImmunizationsArguments,
        output_model=ToolResultSchema,
        handler_name="_search_immunizations",
    ),
    ToolSpec(
        name="find_related_records",
        description="Traverse explicit record references from one evidence record.",
        arguments_model=FindRelatedRecordsArguments,
        output_model=ToolResultSchema,
        handler_name="_find_related_records",
    ),
    ToolSpec(
        name="read_chart_record",
        description="Read one selected normalized chart record by evidence ID.",
        arguments_model=ReadChartRecordArguments,
        output_model=ToolResultSchema,
        handler_name="_read_chart_record",
    ),
    ToolSpec(
        name="submit_answer",
        description="Submit a final answer with supporting evidence IDs.",
        arguments_model=SubmitAnswerArguments,
        output_model=SubmissionResultSchema,
        handler_name="_submit_answer",
    ),
)
_TOOL_SPEC_BY_NAME = {spec.name: spec for spec in _TOOL_SPECS}


class InvalidToolInput(ValueError):
    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__("Invalid tool input")


class ToolRegistry:
    def __init__(self, snapshot: PatientSnapshot) -> None:
        self._snapshot = snapshot

    @property
    def patient_id(self) -> str:
        return self._snapshot.patient_id

    @property
    def snapshot_id(self) -> str:
        return self._snapshot.snapshot_id

    @property
    def snapshot(self) -> PatientSnapshot:
        return self._snapshot

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = _TOOL_SPEC_BY_NAME.get(name)
        if spec is None:
            return _error_result("UNKNOWN_TOOL", {"tool": name}, snapshot_id=self.snapshot_id)
        if not isinstance(arguments, dict):
            return _error_result(
                "INVALID_INPUT",
                {"reason": "arguments_must_be_object"},
                snapshot_id=self.snapshot_id,
            )
        try:
            parsed = spec.arguments_model.model_validate(arguments)
            handler = getattr(self, spec.handler_name)
            result = handler(parsed)
            spec.output_model.model_validate(result)
            return result
        except ValidationError as exc:
            return _error_result(
                "INVALID_INPUT",
                _validation_details(exc),
                snapshot_id=self.snapshot_id,
                submission=name == "submit_answer",
            )
        except InvalidToolInput as exc:
            return _error_result(
                "INVALID_INPUT",
                exc.details,
                snapshot_id=self.snapshot_id,
                submission=name == "submit_answer",
            )

    @staticmethod
    def list_tools() -> list[dict[str, Any]]:
        return [spec.public_contract() for spec in _TOOL_SPECS]

    def _get_my_profile(self, arguments: GetMyProfileArguments) -> dict[str, Any]:
        records = list(self._snapshot.records_by_resource_type("Patient"))
        if len(records) != 1:
            raise InvalidToolInput({"reason": "missing_or_ambiguous_environment_bound_patient"})
        return self._search_result(
            tool_name="get_my_profile",
            records=records,
            normalized={"cursor": None, "limit": 100},
            resource_types=["Patient"],
        )

    def _search_observations(self, arguments: SearchObservationsArguments) -> dict[str, Any]:
        normalized = arguments.model_dump(mode="json")
        _validate_date_range(normalized)
        _validate_search_text(normalized)
        candidates = self._snapshot.records_by_resource_type("Observation")

        encounter_id = normalized.get("encounter_id")
        if encounter_id is not None:
            encounter = self._resolve_encounter(encounter_id)
            related = self._snapshot.related_records(
                encounter.evidence_id, relation="encounter", direction="incoming"
            )
            related_ids = {record.evidence_id for record in related}
            candidates = tuple(record for record in candidates if record.evidence_id in related_ids)

        matches = [record for record in candidates if _matches_common(record, normalized)]
        return self._search_result(
            tool_name="search_observations",
            records=matches,
            normalized=normalized,
            resource_types=["Observation"],
        )

    def _search_encounters(self, arguments: SearchEncountersArguments) -> dict[str, Any]:
        normalized = arguments.model_dump(mode="json")
        _validate_date_range(normalized)
        _validate_search_text(normalized)
        matches = []
        for record in self._snapshot.records_by_resource_type("Encounter"):
            if not _matches_common(record, normalized):
                continue
            status = normalized.get("status")
            record_status = record.payload.get("status", record.value)
            if status is not None and record_status != status:
                continue
            matches.append(record)
        return self._search_result(
            tool_name="search_encounters",
            records=matches,
            normalized=normalized,
            resource_types=["Encounter"],
        )

    def _search_conditions(self, arguments: SearchConditionsArguments) -> dict[str, Any]:
        return self._search_typed_records(
            tool_name="search_conditions",
            arguments=arguments,
            resource_types=("Condition",),
            predicate=lambda record, query: (
                _matches_record_field(
                    record,
                    query.get("clinical_status"),
                    "clinicalStatus",
                    "clinical_status",
                    fallback_values=_CONDITION_STATUS_CODES,
                )
                and _matches_record_field(
                    record,
                    query.get("verification_status"),
                    "verificationStatus",
                    "verification_status",
                )
            ),
        )

    def _search_medications(self, arguments: SearchMedicationsArguments) -> dict[str, Any]:
        resource_types = {
            "request": ("MedicationRequest",),
            "administration": ("MedicationAdministration",),
            "any": ("MedicationRequest", "MedicationAdministration"),
        }[arguments.record_kind]
        return self._search_typed_records(
            tool_name="search_medications",
            arguments=arguments,
            resource_types=resource_types,
            predicate=lambda record, query: (
                _matches_record_field(
                    record,
                    query.get("status"),
                    "status",
                    fallback_values=_MEDICATION_STATUS_CODES,
                )
                and _matches_record_field(record, query.get("intent"), "intent")
            ),
        )

    def _search_procedures(self, arguments: SearchProceduresArguments) -> dict[str, Any]:
        return self._search_typed_records(
            tool_name="search_procedures",
            arguments=arguments,
            resource_types=("Procedure",),
            predicate=lambda record, query: _matches_record_field(
                record,
                query.get("status"),
                "status",
                fallback_values=_PROCEDURE_STATUS_CODES,
            ),
        )

    def _search_clinical_documents(
        self, arguments: SearchClinicalDocumentsArguments
    ) -> dict[str, Any]:
        resource_types = {
            "document_reference": ("DocumentReference",),
            "diagnostic_report": ("DiagnosticReport",),
            "any": ("DocumentReference", "DiagnosticReport"),
        }[arguments.document_kind]
        return self._search_typed_records(
            tool_name="search_clinical_documents",
            arguments=arguments,
            resource_types=resource_types,
            search_record_text=False,
            predicate=lambda record, query: (
                _matches_record_field(record, query.get("status"), "status")
                and _matches_record_field(
                    record, query.get("doc_status"), "docStatus", "doc_status"
                )
            ),
        )

    def _search_service_requests(self, arguments: SearchServiceRequestsArguments) -> dict[str, Any]:
        return self._search_typed_records(
            tool_name="search_service_requests",
            arguments=arguments,
            resource_types=("ServiceRequest",),
            predicate=lambda record, query: (
                _matches_record_field(
                    record,
                    query.get("status"),
                    "status",
                    fallback_values=_SERVICE_REQUEST_STATUS_CODES,
                )
                and _matches_record_field(record, query.get("intent"), "intent")
                and _matches_record_field(record, query.get("priority"), "priority")
            ),
        )

    def _search_care_plans(self, arguments: SearchCarePlansArguments) -> dict[str, Any]:
        return self._search_typed_records(
            tool_name="search_care_plans",
            arguments=arguments,
            resource_types=("CarePlan",),
            predicate=lambda record, query: (
                _matches_record_field(record, query.get("status"), "status")
                and _matches_record_field(record, query.get("intent"), "intent")
            ),
        )

    def _search_immunizations(self, arguments: SearchImmunizationsArguments) -> dict[str, Any]:
        return self._search_typed_records(
            tool_name="search_immunizations",
            arguments=arguments,
            resource_types=("Immunization",),
            predicate=lambda record, query: _matches_record_field(
                record,
                query.get("status"),
                "status",
            ),
        )

    def _search_typed_records(
        self,
        *,
        tool_name: str,
        arguments: SearchClinicalRecordsArguments,
        resource_types: tuple[str, ...],
        predicate: Callable[[EvidenceRecord, dict[str, Any]], bool],
        search_record_text: bool = True,
    ) -> dict[str, Any]:
        normalized = arguments.model_dump(mode="json")
        _validate_date_range(normalized)
        _validate_search_text(normalized)
        candidates = [
            record
            for resource_type in resource_types
            for record in self._snapshot.records_by_resource_type(resource_type)
        ]
        encounter_id = normalized.get("encounter_id")
        if encounter_id is not None:
            encounter = self._resolve_encounter(encounter_id)
            candidates = [
                record
                for record in candidates
                if self._record_is_linked_to_encounter(record, encounter)
            ]
        matches = [
            record
            for record in candidates
            if _matches_common(
                record,
                normalized,
                search_record_text=search_record_text,
            )
            and predicate(record, normalized)
        ]
        return self._search_result(
            tool_name=tool_name,
            records=matches,
            normalized=normalized,
            resource_types=list(resource_types),
        )

    def _find_related_records(self, arguments: FindRelatedRecordsArguments) -> dict[str, Any]:
        normalized = arguments.model_dump(mode="json")
        anchor_id = normalized["anchor_evidence_id"]
        if self._snapshot.record_by_id(anchor_id) is None:
            raise InvalidToolInput({"field": "anchor_evidence_id", "reason": "unknown_evidence_id"})
        matches = list(
            self._snapshot.related_records(
                anchor_id,
                relation=normalized.get("relation"),
                direction=normalized["direction"],
            )
        )
        resource_type = normalized.get("resource_type")
        if resource_type is not None:
            matches = [record for record in matches if record.resource_type == resource_type]
        code = normalized.get("code")
        system = normalized.get("system")
        if code is not None or system is not None:
            matches = [
                record for record in matches if _record_has_coding(record, code=code, system=system)
            ]
        return self._search_result(
            tool_name="find_related_records",
            records=matches,
            normalized=normalized,
            resource_types=sorted({record.resource_type for record in matches}),
        )

    def _read_chart_record(self, arguments: ReadChartRecordArguments) -> dict[str, Any]:
        record = self._snapshot.record_by_id(arguments.evidence_id)
        if record is None:
            raise InvalidToolInput({"field": "evidence_id", "reason": "unknown_evidence_id"})
        return {
            "ok": True,
            "records": [self._record_view(record, include_text=True)],
            "searched_scope": {
                "patient_id": self.patient_id,
                "evidence_id": record.evidence_id,
            },
            "total_matches": 1,
            "complete": True,
            "next_cursor": None,
            "snapshot_id": self.snapshot_id,
            "error": None,
        }

    def _submit_answer(self, arguments: SubmitAnswerArguments) -> dict[str, Any]:
        unknown = sorted(
            evidence_id
            for evidence_id in arguments.evidence_ids
            if self._snapshot.record_by_id(evidence_id) is None
        )
        if unknown:
            raise InvalidToolInput(
                {"field": "evidence_ids", "reason": "unknown_evidence_ids", "ids": unknown}
            )
        return {
            "ok": True,
            "submission": {
                "answer": arguments.answer,
                "evidence_ids": sorted(set(arguments.evidence_ids)),
            },
            "snapshot_id": self.snapshot_id,
            "error": None,
        }

    def _resolve_encounter(self, encounter_id: str) -> EvidenceRecord:
        direct = self._snapshot.record_by_id(encounter_id)
        if direct is not None and direct.resource_type == "Encounter":
            return direct
        matches = [
            record
            for record in self._snapshot.records_by_resource_type("Encounter")
            if record.resource_id == encounter_id
        ]
        if len(matches) != 1:
            raise InvalidToolInput(
                {"field": "encounter_id", "reason": "unknown_or_ambiguous_encounter"}
            )
        return matches[0]

    def _record_is_linked_to_encounter(
        self, record: EvidenceRecord, encounter: EvidenceRecord
    ) -> bool:
        return any(
            link.target_evidence_id == encounter.evidence_id
            and link.relation.split(".")[-1] in {"context", "encounter"}
            for link in self._snapshot.outgoing_links(record.evidence_id)
        )

    def _search_result(
        self,
        *,
        tool_name: str,
        records: list[EvidenceRecord],
        normalized: dict[str, Any],
        resource_types: list[str],
    ) -> dict[str, Any]:
        records.sort(key=lambda record: record.evidence_id)
        records.sort(
            key=lambda record: clinical_instant_sort_key(record.clinical_start),
            reverse=True,
        )
        query = {key: value for key, value in normalized.items() if key != "cursor"}
        query_hash = _query_hash(
            query,
            snapshot_id=self.snapshot_id,
            tool_name=tool_name,
        )
        offset = _decode_cursor(normalized.get("cursor"), query_hash)
        limit = normalized["limit"]
        page = records[offset : offset + limit]
        next_offset = offset + len(page)
        has_more = next_offset < len(records)
        next_cursor = _encode_cursor(next_offset, query_hash) if has_more else None
        searched_scope = {
            "patient_id": self.patient_id,
            "resource_types": resource_types,
            **{key: value for key, value in query.items() if key != "limit"},
        }
        return {
            "ok": True,
            "records": [self._record_view(record) for record in page],
            "searched_scope": searched_scope,
            "total_matches": len(records),
            "complete": not has_more,
            "next_cursor": next_cursor,
            "snapshot_id": self.snapshot_id,
            "error": None,
        }

    def _record_view(self, record: EvidenceRecord, *, include_text: bool = False) -> dict[str, Any]:
        links = [
            {
                "relation": link.relation,
                "target_evidence_id": link.target_evidence_id,
            }
            for link in self._snapshot.outgoing_links(record.evidence_id)
        ]
        is_document = record.resource_type in {"DocumentReference", "DiagnosticReport"}
        value = record.value
        if is_document and not include_text:
            value = {"content_available": value is not None and value != ""}
        view = {
            "evidence_id": record.evidence_id,
            "logical_evidence_group": record.logical_evidence_group,
            "resource_type": record.resource_type,
            "category": record.category,
            "code": copy.deepcopy(record.code),
            "display": record.display,
            "time": copy.deepcopy(record.time),
            "value": copy.deepcopy(value),
            "links": links,
        }
        status = record.payload.get("status")
        if isinstance(status, str) and status:
            view["status"] = status
        attributes = record.payload.get("attributes")
        if isinstance(attributes, dict) and attributes:
            view["attributes"] = copy.deepcopy(attributes)
        if include_text:
            view["text"] = record.text
        return view


class FhirEnumToolRegistry(ToolRegistry):
    """Expose and enforce the complete FHIR R4 Encounter.status value set."""

    @staticmethod
    def list_tools() -> list[dict[str, Any]]:
        contracts = copy.deepcopy(ToolRegistry.list_tools())
        encounter = next(item for item in contracts if item["name"] == "search_encounters")
        encounter["manifest_version"] = f"{TOOL_MANIFEST_VERSION}.encounter-status-enum-v1"
        encounter["description"] = (
            "Search the environment-bound patient's encounters using FHIR R4 Encounter fields."
        )
        encounter["input_schema"]["properties"]["status"] = {
            "anyOf": [
                {"enum": list(FHIR_R4_ENCOUNTER_STATUS_CODES), "type": "string"},
                {"type": "null"},
            ],
            "default": None,
            "description": "FHIR R4 Encounter.status code.",
            "title": "Status",
        }
        return contracts

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_encounters" and isinstance(arguments, dict):
            status = arguments.get("status")
            if status is not None and status not in FHIR_R4_ENCOUNTER_STATUS_CODES:
                return _error_result(
                    "INVALID_INPUT",
                    {
                        "field": "status",
                        "reason": "not_a_fhir_r4_encounter_status",
                        "allowed_values": list(FHIR_R4_ENCOUNTER_STATUS_CODES),
                    },
                    snapshot_id=self.snapshot_id,
                )
        return super().call(name, arguments)


class EncounterStatusAliasToolRegistry(ToolRegistry):
    """Accept a common clinical synonym without disclosing the canonical literal in the prompt."""

    @staticmethod
    def list_tools() -> list[dict[str, Any]]:
        contracts = copy.deepcopy(ToolRegistry.list_tools())
        encounter = next(item for item in contracts if item["name"] == "search_encounters")
        encounter["manifest_version"] = f"{TOOL_MANIFEST_VERSION}.encounter-status-alias-v1"
        encounter["description"] = (
            "Search the environment-bound patient's encounters; common lifecycle-status synonyms "
            "are normalized to FHIR R4 codes."
        )
        encounter["input_schema"]["properties"]["status"]["description"] = (
            "Encounter lifecycle status; common clinical synonyms are accepted."
        )
        return contracts

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized_arguments = copy.deepcopy(arguments)
        if name == "search_encounters" and isinstance(normalized_arguments, dict):
            status = normalized_arguments.get("status")
            if isinstance(status, str) and status.casefold() in {"complete", "completed"}:
                normalized_arguments["status"] = "finished"
        return super().call(name, normalized_arguments)


def registry_for_encounter_status_interface(
    snapshot: PatientSnapshot,
    interface: Literal["legacy_string", "fhir_enum", "normalized_aliases"],
) -> ToolRegistry:
    registry_types = {
        "legacy_string": ToolRegistry,
        "fhir_enum": FhirEnumToolRegistry,
        "normalized_aliases": EncounterStatusAliasToolRegistry,
    }
    return registry_types[interface](snapshot)


def _validation_details(exc: ValidationError) -> dict[str, Any]:
    errors = []
    for error in exc.errors(include_input=False, include_url=False):
        errors.append(
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "reason": error["type"],
            }
        )
    return {"validation_errors": errors}


def _validate_date_range(arguments: dict[str, Any]) -> None:
    start = arguments.get("date_from")
    end = arguments.get("date_to")
    if start is not None and end is not None and start > end:
        raise InvalidToolInput({"field": "date_range", "reason": "date_from_after_date_to"})


def _validate_search_text(arguments: dict[str, Any]) -> None:
    text = arguments.get("text")
    if text is not None and not _normalize_search_text(text):
        raise InvalidToolInput({"field": "text", "reason": "no_searchable_characters"})


def _normalize_search_text(value: str) -> str:
    """Normalize text without allowing punctuation to fuse adjacent tokens."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = unicodedata.normalize("NFKC", normalized)
    punctuation_separated = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(punctuation_separated.split())


def _contains_contiguous_token_sequence(haystack: str, needle: str) -> bool:
    haystack_tokens = _normalize_search_text(haystack).split()
    needle_tokens = _normalize_search_text(needle).split()
    width = len(needle_tokens)
    return width > 0 and any(
        haystack_tokens[offset : offset + width] == needle_tokens
        for offset in range(len(haystack_tokens) - width + 1)
    )


def _searchable_text_fields(
    record: EvidenceRecord,
    *,
    search_record_text: bool,
) -> tuple[str, ...]:
    fields = [record.display]
    if search_record_text:
        fields.append(record.text)
    attributes = record.payload.get("attributes")
    if isinstance(attributes, dict):
        fields.append(json.dumps(attributes, ensure_ascii=False, sort_keys=True))
    if search_record_text and isinstance(record.value, str):
        fields.append(record.value)
    return tuple(fields)


def _matches_common(
    record: EvidenceRecord,
    arguments: dict[str, Any],
    *,
    search_record_text: bool = True,
) -> bool:
    code = arguments.get("code")
    system = arguments.get("system")
    if (code is not None or system is not None) and not _record_has_coding(
        record, code=code, system=system
    ):
        return False
    category = arguments.get("category")
    if category is not None and record.category != category:
        return False
    text = arguments.get("text")
    if text is not None:
        if not any(
            _contains_contiguous_token_sequence(field, text)
            for field in _searchable_text_fields(
                record,
                search_record_text=search_record_text,
            )
        ):
            return False
    date_from = arguments.get("date_from")
    date_to = arguments.get("date_to")
    if date_from is not None or date_to is not None:
        clinical_date = record.local_date
        if not clinical_date:
            return False
        if date_from is not None and clinical_date < date_from:
            return False
        if date_to is not None and clinical_date > date_to:
            return False
    return True


def _matches_record_field(
    record: EvidenceRecord,
    expected: str | None,
    *field_names: str,
    fallback_values: frozenset[str] | None = None,
) -> bool:
    if expected is None:
        return True
    attributes = record.payload.get("attributes")
    for field_name in field_names:
        candidate = record.payload.get(field_name)
        if candidate is None and isinstance(attributes, dict):
            candidate = attributes.get(field_name)
        normalized = _filter_scalar(candidate)
        if normalized is not None:
            return normalized.casefold() == expected.casefold()
    if fallback_values is not None and isinstance(record.value, str):
        normalized_value = record.value.casefold()
        return normalized_value in fallback_values and normalized_value == expected.casefold()
    return False


def _filter_scalar(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("code", "display", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return None


def _record_has_code(record: EvidenceRecord, code: str) -> bool:
    return _record_has_coding(record, code=code, system=None)


def _record_has_coding(record: EvidenceRecord, *, code: str | None, system: str | None) -> bool:
    if (code is None or record.code.get("code") == code) and (
        system is None or record.code.get("system") == system
    ):
        return True
    codings = record.code.get("codings")
    return isinstance(codings, list) and any(
        isinstance(coding, dict)
        and (code is None or coding.get("code") == code)
        and (system is None or coding.get("system") == system)
        for coding in codings
    )


def _query_hash(query: dict[str, Any], *, snapshot_id: str, tool_name: str) -> str:
    canonical = json.dumps(
        {
            "manifest_version": TOOL_MANIFEST_VERSION,
            "snapshot_id": snapshot_id,
            "tool": tool_name,
            "query": query,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _encode_cursor(offset: int, query_hash: str) -> str:
    payload = json.dumps(
        {"offset": offset, "query_hash": query_hash},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None, query_hash: str) -> int:
    if cursor is None:
        return 0
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
        if not isinstance(payload, dict):
            raise ValueError
        if payload.get("query_hash") != query_hash:
            raise ValueError
        offset = payload.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError
        return offset
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidToolInput({"field": "cursor", "reason": "invalid_for_query"}) from exc


def _error_result(
    code: str,
    details: dict[str, Any],
    *,
    snapshot_id: str | None = None,
    submission: bool = False,
) -> dict[str, Any]:
    error = {"code": code, "details": details}
    if submission:
        return {
            "ok": False,
            "submission": None,
            "snapshot_id": snapshot_id or "",
            "error": error,
        }
    return {
        "ok": False,
        "records": [],
        "searched_scope": {},
        "total_matches": 0,
        "complete": True,
        "next_cursor": None,
        "snapshot_id": snapshot_id,
        "error": error,
    }
