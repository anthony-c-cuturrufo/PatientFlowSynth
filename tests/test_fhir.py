import base64
import copy
import json
from pathlib import Path

import pytest

from patient_tool_gym.evidence import (
    FHIR_NORMALIZER_VERSION,
    LOGICAL_GROUP_RULES_VERSION,
    PatientBoundaryError,
    SnapshotIntegrityError,
    UnknownEvidenceIdError,
    dump_snapshot,
    load_snapshot,
)
from patient_tool_gym.fhir import FHIRNormalizationError, normalize_fhir_bundle
from patient_tool_gym.tools import ToolRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "fhir_bundle.json"


def _bundle():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _concept(system: str, code: str, display: str) -> dict:
    return {
        "coding": [{"system": system, "code": code, "display": display}],
        "text": display,
    }


def test_normalizes_typed_quantity_and_selected_resources():
    snapshot = normalize_fhir_bundle(_bundle())

    assert snapshot.normalizer_version == FHIR_NORMALIZER_VERSION
    assert [record.resource_type for record in snapshot.records] == [
        "DiagnosticReport",
        "Encounter",
        "Observation",
        "Patient",
    ]
    weight = snapshot.records_by_code("29463-7")[0]
    assert weight.value == {
        "value": 70.5,
        "unit": "kg",
        "system": "http://unitsofmeasure.org",
        "code": "kg",
        "comparator": None,
    }
    assert weight.local_date == "2026-03-14"
    assert weight.logical_evidence_group == f"record:{weight.evidence_id}"
    report = snapshot.records_by_resource_type("DiagnosticReport")[0]
    assert report.value is None
    report_search = ToolRegistry(snapshot).call(
        "search_clinical_documents",
        {"document_kind": "diagnostic_report", "code": "58410-2"},
    )
    assert report_search["records"][0]["status"] == "final"
    assert report_search["records"][0]["value"] == {"content_available": False}


def test_preserves_observation_missingness_versions_and_provenance():
    bundle = _bundle()
    bundle["entry"].extend(
        [
            {
                "fullUrl": "urn:uuid:missing-result",
                "resource": {
                    "resourceType": "Observation",
                    "id": "missing-result",
                    "meta": {
                        "versionId": "7",
                        "lastUpdated": "2026-03-15T10:00:00Z",
                        "profile": ["http://hl7.org/fhir/StructureDefinition/vitalsigns"],
                    },
                    "status": "final",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "effectiveDateTime": "2026-03-15T09:00:00Z",
                    "code": _concept("http://loinc.org", "94531-1", "Missing test result"),
                    "dataAbsentReason": _concept(
                        "http://terminology.hl7.org/CodeSystem/data-absent-reason",
                        "unknown",
                        "Unknown",
                    ),
                },
            },
            {
                "fullUrl": "urn:uuid:provenance-1",
                "resource": {
                    "resourceType": "Provenance",
                    "id": "provenance-1",
                    "target": [{"reference": "urn:uuid:missing-result"}],
                    "recorded": "2026-03-15T10:01:00Z",
                    "activity": _concept(
                        "http://terminology.hl7.org/CodeSystem/v3-DataOperation",
                        "UPDATE",
                        "revise",
                    ),
                    "agent": [{"who": {"reference": "urn:uuid:patient-1"}}],
                },
            },
        ]
    )

    snapshot = normalize_fhir_bundle(bundle)
    observation = next(
        record
        for record in snapshot.records_by_resource_type("Observation")
        if record.resource_id == "missing-result"
    )
    provenance = snapshot.records_by_resource_type("Provenance")[0]

    assert observation.value is None
    assert observation.payload["attributes"]["data_absent_reason"]["code"] == "unknown"
    assert observation.payload["attributes"]["resource_meta"] == {
        "versionId": "7",
        "lastUpdated": "2026-03-15T10:00:00Z",
        "profile": ["http://hl7.org/fhir/StructureDefinition/vitalsigns"],
    }
    assert provenance.time == {"recorded": "2026-03-15T10:01:00Z"}
    assert snapshot.related_records(provenance.evidence_id, relation="target") == (observation,)


def test_builds_only_explicit_bidirectional_graph_traversal():
    snapshot = normalize_fhir_bundle(_bundle())
    encounter = snapshot.records_by_resource_type("Encounter")[0]
    report = snapshot.records_by_resource_type("DiagnosticReport")[0]
    weight = snapshot.records_by_resource_type("Observation")[0]

    encounter_records = snapshot.related_records(
        encounter.evidence_id, relation="encounter", direction="incoming"
    )
    assert encounter_records == (weight,)
    assert snapshot.related_records(report.evidence_id, relation="result") == (weight,)
    assert all("Practitioner?identifier" not in link.target_evidence_id for link in snapshot.links)


def test_snapshot_hash_ignores_bundle_order_and_transport_metadata():
    original = _bundle()
    changed = copy.deepcopy(original)
    changed["entry"] = list(reversed(changed["entry"]))
    changed["id"] = "another-transport-id"
    changed["timestamp"] = "2030-01-01T00:00:00Z"

    assert normalize_fhir_bundle(original).snapshot_id == normalize_fhir_bundle(changed).snapshot_id


def test_snapshot_dump_load_round_trip_and_integrity(tmp_path):
    snapshot = normalize_fhir_bundle(_bundle())
    output = tmp_path / "snapshot.json"
    dump_snapshot(snapshot, output)

    assert load_snapshot(output).snapshot_id == snapshot.snapshot_id

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["records"][0]["display"] = "tampered"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError):
        load_snapshot(output)


def test_unknown_graph_anchor_fails_closed():
    snapshot = normalize_fhir_bundle(_bundle())
    with pytest.raises(UnknownEvidenceIdError):
        snapshot.related_records("missing")


def test_rejects_multiple_patients_and_cross_patient_ownership():
    multiple = _bundle()
    multiple["entry"].append(
        {
            "fullUrl": "urn:uuid:patient-2",
            "resource": {"resourceType": "Patient", "id": "patient-2"},
        }
    )
    with pytest.raises(FHIRNormalizationError, match="exactly one"):
        normalize_fhir_bundle(multiple)

    crossed = _bundle()
    crossed["entry"][2]["resource"]["subject"]["reference"] = "Patient/patient-2"
    with pytest.raises(PatientBoundaryError):
        normalize_fhir_bundle(crossed)

    wrong_reference_type = _bundle()
    wrong_reference_type["entry"][2]["resource"]["subject"]["reference"] = "Group/patient-1"
    with pytest.raises(PatientBoundaryError):
        normalize_fhir_bundle(wrong_reference_type)


def test_normalizes_service_requests_and_gates_inline_document_text():
    bundle = _bundle()
    bundle["entry"].extend(
        [
            {
                "fullUrl": "urn:uuid:referral-1",
                "resource": {
                    "resourceType": "ServiceRequest",
                    "id": "referral-1",
                    "status": "active",
                    "intent": "order",
                    "priority": "routine",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "authoredOn": "2026-03-14T12:00:00Z",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "referral",
                                    "display": "Referral",
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "306206005",
                                "display": "Referral to cardiology service",
                            }
                        ]
                    },
                },
            },
            {
                "fullUrl": "urn:uuid:document-1",
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "document-1",
                    "status": "current",
                    "docStatus": "final",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "context": {"encounter": [{"reference": "urn:uuid:encounter-1"}]},
                    "date": "2026-03-14T12:30:00Z",
                    "description": "Patient after-visit summary",
                    "type": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "34117-2",
                                "display": "After-visit summary",
                            }
                        ]
                    },
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "title": "After-visit summary",
                                "url": "https://example.invalid/hidden",
                                "data": base64.b64encode(
                                    b"Private inline follow-up instructions."
                                ).decode("ascii"),
                            }
                        }
                    ],
                },
            },
            {
                "fullUrl": "urn:uuid:condition-1",
                "resource": {
                    "resourceType": "Condition",
                    "id": "condition-1",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "recordedDate": "2026-03-14T09:05:00-08:00",
                    "onsetDateTime": "2020-01-01T00:00:00-08:00",
                    "clinicalStatus": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                "code": "active",
                            }
                        ]
                    },
                    "verificationStatus": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                                "code": "confirmed",
                            }
                        ]
                    },
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "59621000",
                                "display": "Essential hypertension",
                            }
                        ]
                    },
                },
            },
        ]
    )

    snapshot = normalize_fhir_bundle(bundle)
    registry = ToolRegistry(snapshot)
    request = snapshot.records_by_resource_type("ServiceRequest")[0]
    document = snapshot.records_by_resource_type("DocumentReference")[0]
    condition = snapshot.records_by_resource_type("Condition")[0]

    assert request.category == "referral"
    assert request.payload["attributes"] == {"intent": "order", "priority": "routine"}
    assert request.local_date == "2026-03-14"
    encounter = snapshot.records_by_resource_type("Encounter")[0]
    assert request in snapshot.related_records(
        encounter.evidence_id, relation="encounter", direction="incoming"
    )

    assert condition.time == {
        "recorded": "2026-03-14T09:05:00-08:00",
        "onset": "2020-01-01T00:00:00-08:00",
    }
    assert condition.local_date == "2026-03-14"
    assert condition.payload["attributes"]["clinicalStatus"]["code"] == "active"
    assert condition.payload["attributes"]["verificationStatus"]["code"] == ("confirmed")

    assert document.value == "Private inline follow-up instructions."
    assert document.payload["attributes"] == {
        "docStatus": "final",
        "description": "Patient after-visit summary",
        "attachments": [{"contentType": "text/plain", "title": "After-visit summary"}],
    }
    assert "url" not in json.dumps(document.payload["attributes"])
    assert "data" not in json.dumps(document.payload["attributes"])

    search = registry.call(
        "search_clinical_documents",
        {"document_kind": "document_reference", "code": "34117-2"},
    )
    assert search["records"][0]["value"] == {"content_available": True}
    assert "Private inline" not in json.dumps(search)
    assert (
        registry.call(
            "search_clinical_documents",
            {"document_kind": "document_reference", "text": "Private inline"},
        )["records"]
        == []
    )
    read = registry.call("read_chart_record", {"evidence_id": document.evidence_id})
    assert read["records"][0]["value"] == "Private inline follow-up instructions."

    onset_only_bundle = copy.deepcopy(bundle)
    onset_only_resource = next(
        entry["resource"]
        for entry in onset_only_bundle["entry"]
        if entry["resource"].get("resourceType") == "Condition"
    )
    onset_only_resource.pop("recordedDate")
    onset_only_condition = normalize_fhir_bundle(onset_only_bundle).records_by_resource_type(
        "Condition"
    )[0]
    assert onset_only_condition.time == {"onset": "2020-01-01T00:00:00-08:00"}
    assert onset_only_condition.local_date == "2020-01-01"

    onset_only_resource.pop("onsetDateTime")
    onset_only_resource["onsetPeriod"] = {
        "start": "2019-02-03T00:00:00-08:00",
        "end": "2019-03-04T00:00:00-08:00",
    }
    onset_period_condition = normalize_fhir_bundle(onset_only_bundle).records_by_resource_type(
        "Condition"
    )[0]
    assert onset_period_condition.time == {
        "onset_start": "2019-02-03T00:00:00-08:00",
        "onset_end": "2019-03-04T00:00:00-08:00",
    }
    assert onset_period_condition.local_date == "2019-02-03"


def test_normalizes_real_synthea_indirections_components_and_duplicate_documents():
    bundle = _bundle()
    patient = bundle["entry"][0]["resource"]
    patient["deceasedDateTime"] = "2025-12-01T08:00:00-08:00"
    note = b"Synthetic history and physical note."
    bundle["entry"].extend(
        [
            {
                "fullUrl": "urn:uuid:medication-1",
                "resource": {
                    "resourceType": "Medication",
                    "id": "medication-1",
                    "status": "active",
                    "code": {
                        "coding": [
                            {
                                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                "code": "1535362",
                                "display": "sodium fluoride oral gel",
                            }
                        ],
                        "text": "sodium fluoride oral gel",
                    },
                },
            },
            {
                "fullUrl": "urn:uuid:medication-request-1",
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "medication-request-1",
                    "status": "completed",
                    "intent": "order",
                    "medicationReference": {"reference": "urn:uuid:medication-1"},
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "authoredOn": "2026-03-14T09:20:00-08:00",
                },
            },
            {
                "fullUrl": "urn:uuid:prapare-panel-1",
                "resource": {
                    "resourceType": "Observation",
                    "id": "prapare-panel-1",
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "survey",
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "93025-5",
                                "display": "PRAPARE panel",
                            }
                        ]
                    },
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "effectiveDateTime": "2026-03-14T09:15:00-08:00",
                    "component": [
                        {
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://loinc.org",
                                        "code": "71802-3",
                                        "display": "Housing status",
                                    }
                                ]
                            },
                            "valueCodeableConcept": {
                                "coding": [
                                    {
                                        "system": "http://snomed.info/sct",
                                        "code": "160903007",
                                        "display": "Lives in a house",
                                    }
                                ],
                                "text": "Stable housing",
                            },
                        },
                        {
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://loinc.org",
                                        "code": "93033-9",
                                        "display": "Stress score",
                                    }
                                ]
                            },
                            "valueQuantity": {
                                "value": 2,
                                "unit": "score",
                                "system": "http://unitsofmeasure.org",
                                "code": "{score}",
                            },
                        },
                    ],
                },
            },
            {
                "fullUrl": "urn:uuid:note-report-1",
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "note-report-1",
                    "status": "final",
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "34117-2",
                                "display": "History and physical note",
                            }
                        ]
                    },
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "effectiveDateTime": "2026-03-14T09:30:00-08:00",
                    "presentedForm": [
                        {
                            "contentType": "text/plain; charset=utf-8",
                            "data": base64.b64encode(note).decode("ascii"),
                        }
                    ],
                },
            },
            {
                "fullUrl": "urn:uuid:document-reference-1",
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "document-reference-1",
                    "identifier": [
                        {
                            "system": "urn:ietf:rfc:3986",
                            "value": "urn:uuid:note-report-1",
                        }
                    ],
                    "status": "current",
                    "type": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "34117-2",
                                "display": "History and physical note",
                            }
                        ]
                    },
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "date": "2026-03-14T09:30:00-08:00",
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain; charset=utf-8",
                                "data": base64.b64encode(note).decode("ascii"),
                            }
                        }
                    ],
                    "context": {"encounter": [{"reference": "urn:uuid:encounter-1"}]},
                },
            },
        ]
    )

    snapshot = normalize_fhir_bundle(bundle)
    registry = ToolRegistry(snapshot)

    assert snapshot.normalizer_version == FHIR_NORMALIZER_VERSION
    assert snapshot.logical_group_rules_version == LOGICAL_GROUP_RULES_VERSION
    assert snapshot.records_by_resource_type("Medication") == ()
    medication_request = snapshot.records_by_code("1535362")[0]
    assert medication_request.resource_type == "MedicationRequest"
    assert medication_request.references["medicationReference"] == {
        "reference": "urn:uuid:medication-1"
    }
    assert medication_request.payload["source"]["resolved_medication"] == {
        "reference": "urn:uuid:medication-1",
        "resource_id": "medication-1",
        "resource_type": "Medication",
    }
    assert (
        registry.call(
            "search_medications",
            {"record_kind": "request", "code": "1535362"},
        )["total_matches"]
        == 1
    )

    panel = snapshot.records_by_code("93025-5")[0]
    housing = snapshot.records_by_code("71802-3")[0]
    stress = snapshot.records_by_code("93033-9")[0]
    assert panel.payload["attributes"]["components"][0]["value"]["code"] == ("160903007")
    assert housing.value["display"] == "Stable housing"
    assert stress.value == {
        "value": 2,
        "unit": "score",
        "system": "http://unitsofmeasure.org",
        "code": "{score}",
    }
    assert housing.logical_evidence_group == panel.logical_evidence_group
    assert stress.logical_evidence_group == panel.logical_evidence_group
    assert housing.time == panel.time
    assert housing.payload["status"] == panel.payload["status"] == "final"
    assert housing.payload["source"] == {
        "format": "fhir.Bundle.entry.resource.component",
        "resource_id": "prapare-panel-1.component.0000",
        "resource_type": "Observation",
        "parent_evidence_id": panel.evidence_id,
        "parent_resource_id": "prapare-panel-1",
        "component_index": 0,
    }
    assert snapshot.related_records(
        housing.evidence_id, relation="component_of", direction="outgoing"
    ) == (panel,)
    assert (
        registry.call(
            "search_observations",
            {"code": "71802-3", "encounter_id": "encounter-1"},
        )["records"][0]["value"]["code"]
        == "160903007"
    )

    report = next(
        record
        for record in snapshot.records_by_resource_type("DiagnosticReport")
        if record.resource_id == "note-report-1"
    )
    document = snapshot.records_by_resource_type("DocumentReference")[0]
    assert report.value == note.decode("utf-8")
    assert document.value == report.value
    assert document.logical_evidence_group == report.logical_evidence_group
    report_search = registry.call(
        "search_clinical_documents",
        {"document_kind": "diagnostic_report", "code": "34117-2"},
    )
    assert report_search["records"][0]["value"] == {"content_available": True}
    assert registry.call("read_chart_record", {"evidence_id": report.evidence_id})["records"][0][
        "value"
    ] == note.decode("utf-8")

    normalized_patient = snapshot.records_by_resource_type("Patient")[0]
    assert normalized_patient.value["deceased_boolean"] is None
    assert normalized_patient.value["deceased_date_time"] == ("2025-12-01T08:00:00-08:00")


def test_preserves_patient_deceased_boolean_and_condition_abatement_choices():
    bundle = _bundle()
    patient = bundle["entry"][0]["resource"]
    patient["deceasedBoolean"] = False
    bundle["entry"].extend(
        [
            {
                "fullUrl": "urn:uuid:resolved-condition",
                "resource": {
                    "resourceType": "Condition",
                    "id": "resolved-condition",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "code": {"text": "Resolved condition"},
                    "recordedDate": "2020-01-01T00:00:00Z",
                    "onsetDateTime": "2019-01-01T00:00:00Z",
                    "abatementDateTime": "2021-01-01T00:00:00Z",
                },
            },
            {
                "fullUrl": "urn:uuid:period-condition",
                "resource": {
                    "resourceType": "Condition",
                    "id": "period-condition",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "code": {"text": "Period condition"},
                    "recordedDate": "2022-01-01T00:00:00Z",
                    "abatementPeriod": {
                        "start": "2023-01-01T00:00:00Z",
                        "end": "2023-02-01T00:00:00Z",
                    },
                },
            },
        ]
    )

    snapshot = normalize_fhir_bundle(bundle)
    normalized_patient = snapshot.records_by_resource_type("Patient")[0]
    conditions = {
        record.resource_id: record for record in snapshot.records_by_resource_type("Condition")
    }

    assert normalized_patient.value["deceased_boolean"] is False
    assert normalized_patient.value["deceased_date_time"] is None
    assert conditions["resolved-condition"].time == {
        "recorded": "2020-01-01T00:00:00Z",
        "onset": "2019-01-01T00:00:00Z",
        "abatement": "2021-01-01T00:00:00Z",
    }
    assert conditions["period-condition"].time == {
        "recorded": "2022-01-01T00:00:00Z",
        "abatement_start": "2023-01-01T00:00:00Z",
        "abatement_end": "2023-02-01T00:00:00Z",
    }


def test_preserves_synthea_task_fields_and_explicit_care_plan_relations():
    bundle = _bundle()
    encounter_resource = bundle["entry"][1]["resource"]
    encounter_resource.update(
        {
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "IMP",
                "display": "inpatient encounter",
            },
            "reasonCode": [_concept("http://snomed.info/sct", "16114001", "Fracture of ankle")],
            "hospitalization": {
                "dischargeDisposition": _concept(
                    "http://www.nubc.org/patient-discharge",
                    "01",
                    "Discharged to home or self care",
                )
            },
        }
    )
    bundle["entry"][2]["resource"]["issued"] = "2026-03-14T09:10:00.250-08:00"
    bundle["entry"].extend(
        [
            {
                "fullUrl": "urn:uuid:condition-task-fields",
                "resource": {
                    "resourceType": "Condition",
                    "id": "condition-task-fields",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "code": _concept("http://snomed.info/sct", "16114001", "Fracture of ankle"),
                    "recordedDate": "2026-03-14T09:01:00-08:00",
                },
            },
            {
                "fullUrl": "urn:uuid:procedure-task-fields",
                "resource": {
                    "resourceType": "Procedure",
                    "id": "procedure-task-fields",
                    "status": "completed",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "code": _concept(
                        "http://snomed.info/sct", "183051005", "Recommendation to rest"
                    ),
                    "performedPeriod": {
                        "start": "2026-03-14T09:05:00-08:00",
                        "end": "2026-03-14T09:06:00-08:00",
                    },
                    "reasonCode": [
                        _concept("http://snomed.info/sct", "16114001", "Fracture of ankle")
                    ],
                    "reasonReference": [{"reference": "urn:uuid:condition-task-fields"}],
                },
            },
            {
                "fullUrl": "urn:uuid:care-plan-task-fields",
                "resource": {
                    "resourceType": "CarePlan",
                    "id": "care-plan-task-fields",
                    "status": "completed",
                    "intent": "order",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "period": {
                        "start": "2026-03-14T09:07:00-08:00",
                        "end": "2026-04-14T09:07:00-07:00",
                    },
                    "category": [
                        _concept(
                            "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
                            "assess-plan",
                            "Assessment and plan",
                        ),
                        _concept("http://snomed.info/sct", "385691007", "Fracture care"),
                    ],
                    "addresses": [{"reference": "urn:uuid:condition-task-fields"}],
                    "activity": [
                        {
                            "detail": {
                                "code": _concept(
                                    "http://snomed.info/sct",
                                    "183051005",
                                    "Recommendation to rest",
                                ),
                                "status": "completed",
                                "location": {"display": "Example Orthopedics"},
                                "reasonReference": [
                                    {"reference": "urn:uuid:condition-task-fields"}
                                ],
                            }
                        },
                        {
                            "detail": {
                                "code": _concept(
                                    "http://snomed.info/sct",
                                    "408580007",
                                    "Light exercise target",
                                ),
                                "status": "completed",
                            }
                        },
                    ],
                },
            },
            {
                "fullUrl": "urn:uuid:immunization-task-fields",
                "resource": {
                    "resourceType": "Immunization",
                    "id": "immunization-task-fields",
                    "status": "completed",
                    "primarySource": True,
                    "patient": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "occurrenceDateTime": "2026-03-14T09:08:00-08:00",
                    "vaccineCode": _concept(
                        "http://hl7.org/fhir/sid/cvx", "140", "Influenza vaccine"
                    ),
                },
            },
        ]
    )

    snapshot = normalize_fhir_bundle(bundle)
    registry = ToolRegistry(snapshot)
    encounter = snapshot.records_by_resource_type("Encounter")[0]
    observation = snapshot.records_by_code("29463-7")[0]
    condition = snapshot.records_by_code("16114001")[0]
    procedure = snapshot.records_by_code("183051005")[0]
    care_plan = snapshot.records_by_code("385691007")[0]
    immunization = snapshot.records_by_code("140")[0]

    assert observation.time == {
        "start": "2026-03-14T09:10:00-08:00",
        "issued": "2026-03-14T09:10:00.250-08:00",
    }
    assert encounter.payload["attributes"]["encounter_class"]["code"] == "IMP"
    assert encounter.payload["attributes"]["reason_codes"][0]["code"] == "16114001"
    assert encounter.payload["attributes"]["discharge_disposition"]["code"] == "01"
    assert procedure.payload["attributes"]["reason_codes"][0]["code"] == "16114001"
    assert care_plan.category == "assess-plan"
    assert care_plan.display == "Fracture care"
    activities = care_plan.payload["attributes"]["activities"]
    assert [activity["code"]["code"] for activity in activities] == [
        "183051005",
        "408580007",
    ]
    assert [activity["status"] for activity in activities] == ["completed", "completed"]
    assert activities[0]["location_display"] == "Example Orthopedics"
    assert care_plan in snapshot.related_records(
        encounter.evidence_id,
        relation="encounter",
        direction="incoming",
    )
    assert immunization.code["system"] == "http://hl7.org/fhir/sid/cvx"
    assert immunization.payload["status"] == "completed"
    assert immunization.time == {"start": "2026-03-14T09:08:00-08:00"}
    assert immunization.payload["attributes"]["primary_source"] is True

    care_plan_result = registry.call(
        "find_related_records",
        {
            "anchor_evidence_id": condition.evidence_id,
            "relation": "addresses",
            "direction": "incoming",
            "resource_type": "CarePlan",
        },
    )
    assert care_plan_result["records"][0]["evidence_id"] == care_plan.evidence_id
    assert (
        care_plan_result["records"][0]["attributes"]["activities"][0]["code"]["code"] == "183051005"
    )
    assert snapshot.related_records(
        condition.evidence_id,
        relation="reasonReference",
        direction="incoming",
    ) == (procedure,)
    assert immunization in snapshot.related_records(
        encounter.evidence_id,
        relation="encounter",
        direction="incoming",
    )
