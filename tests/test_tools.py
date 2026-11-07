import json
from pathlib import Path

import pytest

from patient_tool_gym.evidence import load_snapshot, snapshot_from_payloads
from patient_tool_gym.tools import (
    FHIR_R4_ENCOUNTER_STATUS_CODES,
    EncounterStatusAliasToolRegistry,
    FhirEnumToolRegistry,
    ToolRegistry,
)
from patient_tool_gym.verifier import _tool_manifest_hash

FIXTURE = Path(__file__).parent / "fixtures" / "patient_a.jsonl"
V1_FIXTURE = Path(__file__).parent / "fixtures" / "patient_v1.jsonl"


def test_tool_manifest_hash_cache_is_immutable_and_registry_class_specific():
    calls = {"a": 0, "b": 0}

    class RegistryA(ToolRegistry):
        @staticmethod
        def list_tools():
            calls["a"] += 1
            return [{"name": "a", "description": "A", "input_schema": {}}]

    class RegistryB(ToolRegistry):
        @staticmethod
        def list_tools():
            calls["b"] += 1
            return [{"name": "b", "description": "B", "input_schema": {}}]

    _tool_manifest_hash.cache_clear()
    try:
        first_a = _tool_manifest_hash(RegistryA)
        assert _tool_manifest_hash(RegistryA) == first_a
        first_b = _tool_manifest_hash(RegistryB)
        assert _tool_manifest_hash(RegistryB) == first_b
        assert first_a != first_b
        assert calls == {"a": 1, "b": 1}
    finally:
        _tool_manifest_hash.cache_clear()


@pytest.fixture
def registry():
    return ToolRegistry(load_snapshot(FIXTURE))


@pytest.fixture
def v1_registry():
    return ToolRegistry(load_snapshot(V1_FIXTURE))


@pytest.fixture
def p2_registry():
    payloads = [
        json.loads(line)
        for line in V1_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    encounter_reference = {"encounter": {"reference": "urn:uuid:enc-2026"}}
    payloads.extend(
        (
            {
                "evidence_id": "synthea:v1-patient:CarePlan:plan-1",
                "patient_id": "v1-patient",
                "resource_type": "CarePlan",
                "source_dataset": "fabricated-v1",
                "category": "assess-plan",
                "code": {
                    "system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
                    "code": "assess-plan",
                    "display": "Assessment and plan of treatment",
                },
                "display": "Assessment and plan of treatment",
                "status": "active",
                "attributes": {"intent": "order"},
                "references": encounter_reference,
                "time": {"start": "2026-05-10T09:40:00Z"},
                "value": "active",
            },
            {
                "evidence_id": "synthea:v1-patient:CarePlan:plan-2",
                "patient_id": "v1-patient",
                "resource_type": "CarePlan",
                "source_dataset": "fabricated-v1",
                "category": "assess-plan",
                "code": {
                    "system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
                    "code": "assess-plan",
                    "display": "Assessment and plan of treatment",
                },
                "display": "Assessment and plan of treatment",
                "status": "completed",
                "attributes": {"intent": "order"},
                "references": encounter_reference,
                "time": {"start": "2025-05-10T09:40:00Z"},
                "value": "completed",
            },
            {
                "evidence_id": "synthea:v1-patient:Immunization:flu-1",
                "patient_id": "v1-patient",
                "resource_type": "Immunization",
                "source_dataset": "fabricated-v1",
                "category": "immunization",
                "code": {
                    "system": "http://hl7.org/fhir/sid/cvx",
                    "code": "140",
                    "display": "Influenza, split virus, trivalent, PF",
                },
                "display": "Influenza, split virus, trivalent, PF",
                "status": "completed",
                "references": encounter_reference,
                "time": {"start": "2026-05-10T09:45:00Z"},
                "value": "completed",
            },
        )
    )
    return ToolRegistry(snapshot_from_payloads(payloads))


def test_search_observations_is_replayable_sorted_and_provenance_preserving(registry):
    arguments = {"code": "29463-7", "limit": 1}

    first = registry.call("search_observations", arguments)
    replay = registry.call("search_observations", arguments)

    assert json.dumps(first, sort_keys=True) == json.dumps(replay, sort_keys=True)
    assert first["ok"] is True
    assert first["total_matches"] == 2
    assert first["complete"] is False
    assert first["records"][0]["evidence_id"].endswith("weight-new")
    assert "source" not in first["records"][0]
    assert first["next_cursor"] is not None

    second_page = registry.call(
        "search_observations",
        {"code": "29463-7", "limit": 1, "cursor": first["next_cursor"]},
    )

    assert second_page["records"][0]["evidence_id"].endswith("weight-old")
    assert second_page["complete"] is True
    assert second_page["next_cursor"] is None
    assert {
        first["records"][0]["evidence_id"],
        second_page["records"][0]["evidence_id"],
    } == {
        "synthea:patient-a:Observation:weight-new",
        "synthea:patient-a:Observation:weight-old",
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"unknown": "field"},
        {"limit": 0},
        {"date_from": "2026-02-01", "date_to": "2026-01-01"},
    ],
)
def test_tool_dispatch_rejects_invalid_arguments_structurally(registry, arguments):
    result = registry.call("search_observations", arguments)

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_INPUT"
    assert result["records"] == []


def test_complete_empty_search_supports_chart_absence(registry):
    result = registry.call("search_observations", {"code": "not-present"})

    assert result["ok"] is True
    assert result["records"] == []
    assert result["total_matches"] == 0
    assert result["complete"] is True
    assert result["searched_scope"]["resource_types"] == ["Observation"]
    assert result["searched_scope"]["patient_id"] == "patient-a"


def test_cursor_cannot_be_reused_with_another_query(registry):
    first = registry.call("search_observations", {"code": "29463-7", "limit": 1})

    result = registry.call(
        "search_observations",
        {"code": "85354-9", "limit": 1, "cursor": first["next_cursor"]},
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_INPUT"
    assert result["error"]["details"]["field"] == "cursor"


def test_cursor_payload_must_be_an_object(registry):
    result = registry.call("search_observations", {"cursor": "W10"})

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_INPUT"


def test_unknown_tool_fails_closed(registry):
    result = registry.call("get_all_patient_data", {})

    assert result["ok"] is False
    assert result["error"] == {
        "code": "UNKNOWN_TOOL",
        "details": {"tool": "get_all_patient_data"},
    }


def test_manifest_exposes_strict_machine_readable_schemas():
    contracts = ToolRegistry.list_tools()

    assert [contract["name"] for contract in contracts] == [
        "get_my_profile",
        "search_encounters",
        "search_conditions",
        "search_observations",
        "search_medications",
        "search_procedures",
        "search_clinical_documents",
        "search_service_requests",
        "search_care_plans",
        "search_immunizations",
        "find_related_records",
        "read_chart_record",
        "submit_answer",
    ]
    observation = next(
        contract for contract in contracts if contract["name"] == "search_observations"
    )
    assert observation["input_schema"]["additionalProperties"] is False
    assert observation["input_schema"]["properties"]["limit"]["maximum"] == 100
    assert all(
        "patient" not in contract["input_schema"].get("properties", {}) for contract in contracts
    )
    assert not any(
        action in contract["name"]
        for contract in contracts
        for action in ("create", "delete", "prescribe", "write_file")
    )
    assert {contract["manifest_version"] for contract in contracts} == {"patient-tool-gym.tools.v5"}


def test_fhir_enum_interface_exposes_and_enforces_encounter_status_codes():
    registry = FhirEnumToolRegistry(load_snapshot(FIXTURE))
    contract = next(
        item for item in registry.list_tools() if item["name"] == "search_encounters"
    )
    status_schema = contract["input_schema"]["properties"]["status"]
    assert status_schema["anyOf"][0]["enum"] == list(FHIR_R4_ENCOUNTER_STATUS_CODES)

    valid = registry.call("search_encounters", {"status": "finished"})
    assert valid["ok"] is True
    assert valid["total_matches"] == 1

    invalid = registry.call("search_encounters", {"status": "completed"})
    assert invalid["ok"] is False
    assert invalid["error"]["details"]["reason"] == "not_a_fhir_r4_encounter_status"


def test_encounter_alias_interface_normalizes_completed_to_finished():
    registry = EncounterStatusAliasToolRegistry(load_snapshot(FIXTURE))
    result = registry.call("search_encounters", {"status": "completed"})

    assert result["ok"] is True
    assert result["total_matches"] == 1
    assert result["searched_scope"]["status"] == "finished"


def test_text_search_matches_contiguous_tokens_across_punctuation_boundaries():
    def observation(
        resource_id: str,
        display: str,
        *,
        text: str | None = None,
    ) -> dict[str, object]:
        return {
            "evidence_id": f"synthea:text-search:Observation:{resource_id}",
            "patient_id": "text-search",
            "resource_type": "Observation",
            "source_dataset": "fabricated-text-search",
            "category": "survey",
            "code": {},
            "display": display,
            "references": {},
            "text": text if text is not None else display,
            "time": {"start": "2026-07-10T12:00:00Z"},
            "value": 1,
        }

    text_registry = ToolRegistry(
        snapshot_from_payloads(
            [
                observation("target", "Fall risk level [Morse Fall Scale]"),
                observation("wrong-order", "Morse Fall Scale — fall risk level"),
                observation("fused-token", "Fall risk level MorseFall Scale"),
                observation(
                    "cross-field",
                    "Fall risk level",
                    text="Morse Fall Scale",
                ),
                observation("unicode", "ＦＡＬＬ risk STRASSE"),
            ]
        )
    )

    result = text_registry.call(
        "search_observations",
        {"text": "Fall risk level Morse Fall Scale"},
    )

    assert result["ok"] is True
    assert [record["evidence_id"] for record in result["records"]] == [
        "synthea:text-search:Observation:target"
    ]

    unicode_result = text_registry.call(
        "search_observations",
        {"text": "fall RISK straße"},
    )
    assert [record["evidence_id"] for record in unicode_result["records"]] == [
        "synthea:text-search:Observation:unicode"
    ]


def test_text_search_does_not_match_query_inside_a_larger_token():
    text_registry = ToolRegistry(
        snapshot_from_payloads(
            [
                {
                    "evidence_id": "synthea:text-search:Observation:brisk",
                    "patient_id": "text-search",
                    "resource_type": "Observation",
                    "category": "survey",
                    "code": {},
                    "display": "Brisk gait assessment",
                    "references": {},
                    "text": "Brisk gait assessment",
                    "time": {"start": "2026-07-10T12:00:00Z"},
                    "value": 1,
                }
            ]
        )
    )

    result = text_registry.call("search_observations", {"text": "risk"})

    assert result["ok"] is True
    assert result["records"] == []
    assert result["total_matches"] == 0


@pytest.mark.parametrize("text", ["", " \t\n", "[]—…"])
def test_text_search_rejects_queries_without_searchable_characters(registry, text):
    result = registry.call("search_observations", {"text": text})

    assert result["ok"] is False
    assert result["error"] == {
        "code": "INVALID_INPUT",
        "details": {"field": "text", "reason": "no_searchable_characters"},
    }


def test_semantically_equivalent_text_queries_cannot_share_a_cursor(registry):
    first = registry.call(
        "search_observations",
        {"text": "Body weight", "limit": 1},
    )
    reused = registry.call(
        "search_observations",
        {"text": "Body-weight", "limit": 1, "cursor": first["next_cursor"]},
    )

    assert first["next_cursor"] is not None
    assert reused["ok"] is False
    assert reused["error"]["details"] == {
        "field": "cursor",
        "reason": "invalid_for_query",
    }


def test_care_plan_search_is_typed_paginated_and_encounter_bound(p2_registry):
    encounter_id = "synthea:v1-patient:Encounter:enc-2026"
    first = p2_registry.call(
        "search_care_plans",
        {
            "system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
            "code": "assess-plan",
            "intent": "order",
            "encounter_id": encounter_id,
            "limit": 1,
        },
    )
    second = p2_registry.call(
        "search_care_plans",
        {
            "system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
            "code": "assess-plan",
            "intent": "order",
            "encounter_id": encounter_id,
            "limit": 1,
            "cursor": first["next_cursor"],
        },
    )

    assert first["ok"] is True
    assert first["total_matches"] == 2
    assert first["complete"] is False
    assert first["records"][0]["evidence_id"].endswith("CarePlan:plan-1")
    assert second["complete"] is True
    assert second["records"][0]["evidence_id"].endswith("CarePlan:plan-2")

    active = p2_registry.call("search_care_plans", {"status": "active"})
    assert [record["evidence_id"] for record in active["records"]] == [
        "synthea:v1-patient:CarePlan:plan-1"
    ]


def test_immunization_search_is_patient_scoped_and_rejects_invalid_filters(p2_registry):
    result = p2_registry.call(
        "search_immunizations",
        {
            "system": "http://hl7.org/fhir/sid/cvx",
            "code": "140",
            "status": "completed",
            "date_from": "2026-05-10",
            "date_to": "2026-05-10",
        },
    )
    invalid_status = p2_registry.call(
        "search_immunizations",
        {"status": "active"},
    )
    injected_patient = p2_registry.call(
        "search_care_plans",
        {"patient_id": "another-patient"},
    )

    assert result["ok"] is True
    assert result["searched_scope"]["patient_id"] == "v1-patient"
    assert [record["resource_type"] for record in result["records"]] == ["Immunization"]
    assert invalid_status["ok"] is False
    assert invalid_status["error"]["code"] == "INVALID_INPUT"
    assert injected_patient["ok"] is False


def test_profile_is_environment_bound_and_omits_source_metadata(v1_registry):
    result = v1_registry.call("get_my_profile", {})
    invalid = v1_registry.call("get_my_profile", {"patient": "someone-else"})

    assert result["ok"] is True
    assert result["total_matches"] == 1
    assert result["records"][0]["resource_type"] == "Patient"
    assert result["records"][0]["display"] == "Alex Patient"
    assert "source" not in result["records"][0]
    assert "patient_id" not in result["records"][0]
    assert invalid["ok"] is False


def test_profile_fails_closed_when_the_snapshot_has_no_patient_record(registry):
    result = registry.call("get_my_profile", {})

    assert result["ok"] is False
    assert result["error"] == {
        "code": "INVALID_INPUT",
        "details": {"reason": "missing_or_ambiguous_environment_bound_patient"},
    }


def test_conditions_filter_by_code_system_status_and_date(v1_registry):
    result = v1_registry.call(
        "search_conditions",
        {
            "system": "http://snomed.info/sct",
            "code": "59621000",
            "clinical_status": "active",
            "date_from": "2026-05-10",
            "date_to": "2026-05-10",
        },
    )
    wrong_system = v1_registry.call(
        "search_conditions",
        {"system": "http://loinc.org", "code": "59621000"},
    )

    assert [record["display"] for record in result["records"]] == ["Essential hypertension"]
    assert wrong_system["records"] == []
    assert wrong_system["complete"] is True


def test_medication_requests_and_administrations_are_never_conflated(v1_registry):
    request = v1_registry.call(
        "search_medications",
        {"record_kind": "request", "status": "active", "code": "29046"},
    )
    administration = v1_registry.call(
        "search_medications",
        {
            "record_kind": "administration",
            "status": "completed",
            "code": "29046",
        },
    )
    missing_kind = v1_registry.call("search_medications", {"code": "29046"})
    status_is_not_intent = v1_registry.call(
        "search_medications",
        {"record_kind": "administration", "intent": "completed"},
    )

    assert len(request["records"]) == 2
    assert {record["resource_type"] for record in request["records"]} == {"MedicationRequest"}
    assert [record["resource_type"] for record in administration["records"]] == [
        "MedicationAdministration"
    ]
    assert missing_kind["ok"] is False
    assert status_is_not_intent["records"] == []


def test_procedure_document_and_service_request_searches_are_read_only(v1_registry):
    encounter_id = "synthea:v1-patient:Encounter:enc-2026"
    procedure = v1_registry.call(
        "search_procedures",
        {"status": "completed", "code": "73761001", "encounter_id": encounter_id},
    )
    document = v1_registry.call(
        "search_clinical_documents",
        {
            "document_kind": "document_reference",
            "text": "After-visit summary",
            "encounter_id": encounter_id,
        },
    )
    diagnostic_report = v1_registry.call(
        "search_clinical_documents",
        {"document_kind": "diagnostic_report", "code": "58410-2"},
    )
    referral = v1_registry.call(
        "search_service_requests",
        {
            "category": "referral",
            "status": "active",
            "intent": "order",
            "priority": "routine",
            "encounter_id": encounter_id,
        },
    )
    status_is_not_priority = v1_registry.call(
        "search_service_requests",
        {"priority": "active"},
    )

    assert [record["resource_type"] for record in procedure["records"]] == ["Procedure"]
    assert document["records"][0]["value"] == {"content_available": True}
    assert document["records"][0]["status"] == "current"
    assert "routine follow-up" not in json.dumps(document["records"][0])
    assert diagnostic_report["records"][0]["resource_type"] == "DiagnosticReport"
    assert referral["records"][0]["resource_type"] == "ServiceRequest"
    assert status_is_not_priority["records"] == []

    read = v1_registry.call(
        "read_chart_record",
        {"evidence_id": document["records"][0]["evidence_id"]},
    )
    assert "routine follow-up" in read["records"][0]["value"]
    assert "routine follow-up" in read["records"][0]["text"]

    hidden_body_search = v1_registry.call(
        "search_clinical_documents",
        {"document_kind": "document_reference", "text": "routine follow-up"},
    )
    assert hidden_body_search["records"] == []


def test_clinician_only_mutation_tools_do_not_exist(v1_registry):
    for name in (
        "fhir_medication_request_create",
        "fhir_service_request_create",
        "fhir_appointment_create",
        "fhir_communication_create_message",
        "write_file",
    ):
        result = v1_registry.call(name, {})
        assert result["error"] == {"code": "UNKNOWN_TOOL", "details": {"tool": name}}


def test_search_encounters_and_encounter_bound_observations(registry):
    encounters = registry.call("search_encounters", {"text": "wellness"})
    encounter_id = encounters["records"][0]["evidence_id"]

    observations = registry.call(
        "search_observations",
        {"encounter_id": encounter_id, "code": "29463-7"},
    )

    assert encounters["ok"] is True
    assert encounters["records"][0]["resource_type"] == "Encounter"
    assert [record["evidence_id"] for record in observations["records"]] == [
        "synthea:patient-a:Observation:weight-new"
    ]


def test_find_related_records_traverses_only_explicit_links(registry):
    encounter_id = "synthea:patient-a:Encounter:enc-1"

    result = registry.call(
        "find_related_records",
        {
            "anchor_evidence_id": encounter_id,
            "relation": "encounter",
            "direction": "incoming",
            "resource_type": "Observation",
        },
    )

    assert {record["evidence_id"] for record in result["records"]} == {
        "synthea:patient-a:Observation:weight-new",
        "synthea:patient-a:Observation:bp-1",
    }
    assert "synthea:patient-a:Observation:weight-old" not in {
        record["evidence_id"] for record in result["records"]
    }


def test_read_record_and_submit_answer_never_expose_source(registry):
    evidence_id = "synthea:patient-a:Observation:weight-new"

    read_result = registry.call("read_chart_record", {"evidence_id": evidence_id})
    submit_result = registry.call(
        "submit_answer", {"answer": "62 kg", "evidence_ids": [evidence_id]}
    )

    assert read_result["ok"] is True
    assert "source" not in read_result["records"][0]
    assert "payload" not in read_result["records"][0]
    assert submit_result["submission"] == {
        "answer": "62 kg",
        "evidence_ids": [evidence_id],
    }


def test_submit_answer_rejects_unknown_evidence(registry):
    result = registry.call("submit_answer", {"answer": "anything", "evidence_ids": ["not-present"]})

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_INPUT"
    assert result["error"]["details"]["reason"] == "unknown_evidence_ids"


def test_cursor_is_bound_to_snapshot_as_well_as_query(registry):
    first = registry.call("search_observations", {"code": "29463-7", "limit": 1})
    payloads = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for payload in payloads:
        payload["patient_id"] = "patient-other"
        payload["evidence_id"] = payload["evidence_id"].replace("patient-a", "patient-other")
    other_registry = ToolRegistry(snapshot_from_payloads(payloads))

    result = other_registry.call(
        "search_observations",
        {"code": "29463-7", "limit": 1, "cursor": first["next_cursor"]},
    )

    assert result["ok"] is False
    assert result["error"]["details"]["field"] == "cursor"


def test_tool_results_cannot_mutate_hash_bound_snapshot(registry):
    result = registry.call("search_observations", {"limit": 1})
    evidence_id = result["records"][0]["evidence_id"]
    record = registry.snapshot.record_by_id(evidence_id)
    assert record is not None
    snapshot_id = registry.snapshot_id
    before = registry.snapshot.canonical_json()

    result["records"][0]["code"]["code"] = "tampered"
    result["records"][0]["time"]["start"] = "1900-01-01"
    if "attributes" in result["records"][0]:
        result["records"][0]["attributes"]["tampered"] = True

    assert registry.snapshot_id == snapshot_id
    assert registry.snapshot.canonical_json() == before
    assert record.code.get("code") != "tampered"
    with pytest.raises(TypeError, match="immutable"):
        record.payload["tampered"] = True
