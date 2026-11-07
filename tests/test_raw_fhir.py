import json
import uuid

import pytest

from patient_tool_gym.fhir import normalize_fhir_bundle
from patient_tool_gym.medical_interventions import (
    apply_raw_fhir_patch_chain,
    build_compositional_factorial_patch_chain,
    build_compositional_patch_chain,
    compile_raw_intervention_family,
)
from patient_tool_gym.program import (
    ClinicalRecordSelector,
    EvidenceStateNode,
    TaskProgram,
    ToolNode,
    ValueRef,
    content_hash,
)
from patient_tool_gym.raw_fhir import (
    RawFHIRDistractorUnavailableError,
    RawFHIRInsertedReferenceRoute,
    RawFHIRPatch,
    RawFHIRPatchError,
    apply_raw_fhir_patch,
    build_distractor_patch,
    build_truth_state_patch,
    validate_reference_closure,
    validate_reference_closure_strict,
)
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import compile_task


def _concept(system: str, code: str):
    return {"coding": [{"system": system, "code": code, "display": code}]}


def _observation_bundle():
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": "2026-01-02T00:00:00Z",
        "entry": [
            {
                "fullUrl": "urn:uuid:patient-1",
                "resource": {"resourceType": "Patient", "id": "patient-1"},
            },
            {
                "fullUrl": "urn:uuid:encounter-1",
                "resource": {
                    "resourceType": "Encounter",
                    "id": "encounter-1",
                    "status": "finished",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "period": {"start": "2026-01-01T09:00:00Z"},
                },
            },
            {
                "fullUrl": "urn:uuid:result-1",
                "resource": {
                    "resourceType": "Observation",
                    "id": "result-1",
                    "status": "final",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "encounter": {"reference": "urn:uuid:encounter-1"},
                    "effectiveDateTime": "2026-01-01T10:00:00Z",
                    "code": _concept("http://loinc.org", "1234-5"),
                    "valueQuantity": {"value": 12, "unit": "mg", "code": "mg"},
                },
            },
        ],
    }


def _observation_program():
    return TaskProgram(
        schema_version="patient-tool-gym.program.v3",
        nodes=(
            ToolNode(
                id="search",
                tool="search_observations",
                arguments={"system": "http://loinc.org", "limit": 100},
            ),
            EvidenceStateNode(
                id="answer",
                source=ValueRef(node_id="search"),
                decision_rule_id="observation_final_result.v2",
                selector=ClinicalRecordSelector(
                    system="http://loinc.org",
                    code="1234-5",
                    date_from="2026-01-01",
                    date_to="2026-01-01",
                ),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def _medication_bundle():
    medication = _concept("http://www.nlm.nih.gov/research/umls/rxnorm", "860975")
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": "2026-01-02T00:00:00Z",
        "entry": [
            {
                "fullUrl": "urn:uuid:patient-1",
                "resource": {"resourceType": "Patient", "id": "patient-1"},
            },
            {
                "fullUrl": "urn:uuid:request-1",
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "request-1",
                    "status": "active",
                    "intent": "order",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "authoredOn": "2026-01-01T08:00:00Z",
                    "medicationCodeableConcept": medication,
                },
            },
            {
                "fullUrl": "urn:uuid:administration-1",
                "resource": {
                    "resourceType": "MedicationAdministration",
                    "id": "administration-1",
                    "status": "completed",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "effectiveDateTime": "2026-01-01T09:00:00Z",
                    "medicationCodeableConcept": medication,
                    "request": {"reference": "urn:uuid:request-1"},
                },
            },
        ],
    }


def _medication_program():
    return TaskProgram(
        schema_version="patient-tool-gym.program.v3",
        nodes=(
            ToolNode(
                id="search",
                tool="search_medications",
                arguments={
                    "record_kind": "any",
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "limit": 100,
                },
            ),
            EvidenceStateNode(
                id="answer",
                source=ValueRef(node_id="search"),
                decision_rule_id="medication_administered.v2",
                selector=ClinicalRecordSelector(
                    system="http://www.nlm.nih.gov/research/umls/rxnorm",
                    code="860975",
                ),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def test_raw_patch_preserves_reference_closure_and_normalizes_provenance():
    bundle = _observation_bundle()
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="insufficient_evidence",
    )

    derived = apply_raw_fhir_patch(bundle, patch)
    report = validate_reference_closure(derived)
    snapshot = normalize_fhir_bundle(derived)
    observation = next(
        record
        for record in snapshot.records_by_resource_type("Observation")
        if record.resource_id == "result-1"
    )

    assert report.passed is True
    assert observation.value is None
    assert observation.payload["attributes"]["data_absent_reason"]["code"] == "unknown"
    assert len(snapshot.records_by_resource_type("Provenance")) == 1


def test_supported_baseline_is_a_byte_identical_noop_without_provenance():
    bundle = _observation_bundle()
    bundle.pop("timestamp")
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="supported",
    )
    derived = apply_raw_fhir_patch(bundle, patch)

    assert patch.operations == ()
    assert derived == bundle

    invalid = patch.model_dump(mode="json")
    invalid["intervention_kind"] = "refuted"
    with pytest.raises(ValueError, match="only a supported intervention"):
        RawFHIRPatch.model_validate(invalid)


def test_conflict_clone_is_added_to_the_explicit_intermediary_route():
    bundle = _observation_bundle()
    bundle["entry"].append(
        {
            "fullUrl": "urn:uuid:report-1",
            "resource": {
                "resourceType": "DiagnosticReport",
                "id": "report-1",
                "status": "final",
                "subject": {"reference": "urn:uuid:patient-1"},
                "encounter": {"reference": "urn:uuid:encounter-1"},
                "code": _concept("http://loinc.org", "58410-2"),
                "result": [{"reference": "urn:uuid:result-1"}],
            },
        }
    )
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="conflicting_evidence",
        inserted_reference_route=RawFHIRInsertedReferenceRoute(
            parent_resource_type="DiagnosticReport",
            parent_resource_id="report-1",
            reference_field="result",
        ),
    )

    derived = apply_raw_fhir_patch(bundle, patch)
    report = next(
        entry["resource"] for entry in derived["entry"] if entry["resource"].get("id") == "report-1"
    )
    references = [item["reference"] for item in report["result"]]

    assert references[0] == "urn:uuid:result-1"
    assert len(references) == 2
    assert references[1].startswith("urn:uuid:")
    assert validate_reference_closure_strict(derived).passed is True


def test_supported_condition_preserves_existing_coding_display_as_a_noop():
    bundle = _observation_bundle()
    condition = bundle["entry"][2]["resource"]
    condition.clear()
    condition.update(
        {
            "resourceType": "Condition",
            "id": "condition-1",
            "subject": {"reference": "urn:uuid:patient-1"},
            "code": _concept("http://snomed.info/sct", "59621000"),
            "verificationStatus": {
                "coding": [
                    {
                        "system": ("http://terminology.hl7.org/CodeSystem/condition-ver-status"),
                        "code": "confirmed",
                        "display": "Confirmed",
                    }
                ],
                "text": "Confirmed",
            },
            "recordedDate": "2026-01-01T10:00:00Z",
        }
    )
    bundle["entry"][2]["fullUrl"] = "urn:uuid:condition-1"

    patch = build_truth_state_patch(
        bundle,
        resource_type="Condition",
        resource_id="condition-1",
        target_state="supported",
    )

    assert patch.operations == ()
    assert apply_raw_fhir_patch(bundle, patch) == bundle


def test_provenance_time_is_explicit_and_not_before_bundle_timestamp():
    bundle = _observation_bundle()
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="refuted",
        provenance_recorded_at="2026-01-02T00:00:00Z",
    )
    derived = apply_raw_fhir_patch(bundle, patch)
    provenance = next(
        entry["resource"]
        for entry in derived["entry"]
        if entry["resource"].get("resourceType") == "Provenance"
    )

    assert provenance["recorded"] == "2026-01-02T00:00:00Z"
    with pytest.raises(RawFHIRPatchError, match="cannot be earlier"):
        build_truth_state_patch(
            bundle,
            resource_type="Observation",
            resource_id="result-1",
            target_state="refuted",
            provenance_recorded_at="2026-01-01T00:00:00Z",
        )


def test_same_resource_distractor_tier_is_recorded_and_not_an_exact_duplicate():
    bundle = _observation_bundle()
    second = json.loads(json.dumps(bundle["entry"][2]))
    second["fullUrl"] = "urn:uuid:result-2"
    second["resource"]["id"] = "result-2"
    second["resource"]["code"] = _concept("http://loinc.org", "7777-8")
    bundle["entry"].append(second)
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="answer_preserving_distractor",
        distractor_tier="same_resource_different_code",
    )
    derived = apply_raw_fhir_patch(bundle, patch)
    clone = next(
        entry["resource"]
        for entry in derived["entry"]
        if entry["resource"].get("resourceType") == "Observation"
        and entry["resource"].get("id") not in {"result-1", "result-2"}
    )

    assert patch.distractor_tier == "same_resource_different_code"
    assert clone["effectiveDateTime"].startswith("2026-01-01T")
    assert clone["effectiveDateTime"] not in {
        bundle["entry"][2]["resource"]["effectiveDateTime"],
        second["resource"]["effectiveDateTime"],
    }
    assert "distractor" not in clone["id"] and "conflict" not in clone["id"]
    assert str(uuid.UUID(clone["id"])) == clone["id"]


def test_exact_distractor_tier_raises_typed_error_without_an_admissible_source():
    bundle = _observation_bundle()

    with pytest.raises(RawFHIRDistractorUnavailableError) as captured:
        build_distractor_patch(
            bundle,
            resource_type="Observation",
            resource_id="result-1",
            distractor_tier="same_resource_different_code",
            require_exact_tier=True,
        )

    assert captured.value.requested_tier == "same_resource_different_code"
    assert captured.value.target_resource_type == "Observation"
    assert captured.value.target_resource_id == "result-1"

    legacy = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="answer_preserving_distractor",
        distractor_tier="same_resource_different_code",
    )
    assert legacy.distractor_tier == "cross_resource_routing"


def test_distractor_rejects_a_forbidden_code_in_any_coding_not_just_the_first():
    bundle = _observation_bundle()
    second = json.loads(json.dumps(bundle["entry"][2]))
    second["fullUrl"] = "urn:uuid:result-2"
    second["resource"]["id"] = "result-2"
    second["resource"]["code"] = {
        "coding": [
            {"system": "http://loinc.org", "code": "7777-8"},
            {"system": "http://loinc.org", "code": "1234-5"},
        ]
    }
    bundle["entry"].append(second)

    with pytest.raises(RawFHIRDistractorUnavailableError):
        build_distractor_patch(
            bundle,
            resource_type="Observation",
            resource_id="result-1",
            distractor_tier="same_resource_different_code",
            forbidden_codes=frozenset({("http://loinc.org", "1234-5")}),
            require_exact_tier=True,
        )


@pytest.mark.parametrize(
    "state",
    ["supported", "refuted", "insufficient_evidence", "conflicting_evidence"],
)
def test_compositional_patch_chain_crosses_state_with_independent_distractor(state):
    bundle = _observation_bundle()
    second = json.loads(json.dumps(bundle["entry"][2]))
    second["fullUrl"] = "urn:uuid:result-2"
    second["resource"]["id"] = "result-2"
    second["resource"]["code"] = _concept("http://loinc.org", "7777-8")
    second["resource"]["effectiveDateTime"] = "2025-12-15T10:00:00Z"
    bundle["entry"].append(second)
    answers = []

    for distractor_present in (False, True):
        chain = build_compositional_patch_chain(
            bundle,
            resource_type="Observation",
            resource_id="result-1",
            evidence_state=state,
            distractor_present=distractor_present,
            forbidden_codes=frozenset({("http://loinc.org", "1234-5")}),
        )
        derived = apply_raw_fhir_patch_chain(bundle, chain)
        registry = ToolRegistry(normalize_fhir_bundle(derived))
        task = compile_task(
            registry=registry,
            program=_observation_program(),
            family_id=f"chain-{state}-{distractor_present}",
            construction_type="atomic",
            question="Classify the target observation evidence.",
        )
        answers.append(task.gold_answer.value)
        if distractor_present:
            clone = next(
                entry["resource"]
                for entry in derived["entry"]
                if entry["resource"].get("resourceType") == "Observation"
                and entry["resource"].get("id") not in {"result-1", "result-2"}
                and entry["resource"].get("code") == second["resource"]["code"]
            )
            assert clone["effectiveDateTime"].startswith("2026-01-01")
            assert len(chain.patches) == int(state != "supported") + 1
        else:
            assert len(chain.patches) == int(state != "supported")

    assert answers == [state, state]


def test_conditional_references_require_an_explicit_external_type_allowlist():
    bundle = _observation_bundle()
    bundle["entry"][2]["resource"]["performer"] = [
        {"reference": "Practitioner?identifier=http://example.test/npi|123"}
    ]

    closed = validate_reference_closure(bundle)
    allowed = validate_reference_closure(
        bundle,
        allowed_external_conditional_reference_types=frozenset({"Practitioner"}),
    )

    assert closed.passed is False
    assert allowed.passed is True
    assert allowed.external_conditional_reference_count == 1


def test_strict_closure_resolves_contained_references_in_the_owning_resource():
    bundle = _observation_bundle()
    observation = bundle["entry"][2]["resource"]
    observation["contained"] = [
        {
            "resourceType": "Specimen",
            "id": "specimen-1",
            "subject": {"reference": "urn:uuid:patient-1"},
        }
    ]
    observation["specimen"] = {"reference": "#specimen-1"}

    assert validate_reference_closure_strict(bundle).passed is True

    observation["specimen"] = {"reference": "#missing"}
    report = validate_reference_closure_strict(bundle)
    assert report.passed is False
    assert report.dangling[0].reference == "#missing"


def test_strict_closure_checks_references_inside_contained_resources():
    bundle = _observation_bundle()
    observation = bundle["entry"][2]["resource"]
    observation["contained"] = [
        {
            "resourceType": "Specimen",
            "id": "specimen-1",
            "subject": {"reference": "Patient/missing"},
        }
    ]

    report = validate_reference_closure_strict(bundle)

    assert report.passed is False
    assert report.dangling[0].path == "contained.0.subject.reference"


def test_strict_closure_requires_an_exact_history_version_and_rejects_bare_ids():
    bundle = _observation_bundle()
    patient = bundle["entry"][0]["resource"]
    patient["meta"] = {"versionId": "1"}
    observation = bundle["entry"][2]["resource"]
    observation["subject"] = {"reference": "Patient/patient-1/_history/1"}

    assert validate_reference_closure_strict(bundle).passed is True

    observation["subject"] = {"reference": "Patient/patient-1/_history/999"}
    assert validate_reference_closure_strict(bundle).passed is False

    observation["subject"] = {"reference": "patient-1"}
    assert validate_reference_closure(bundle).passed is True
    assert validate_reference_closure_strict(bundle).passed is False


def test_raw_inserts_are_valid_transaction_entries():
    bundle = _observation_bundle()
    bundle["type"] = "transaction"
    reference_map = {}
    for entry in bundle["entry"]:
        resource = entry["resource"]
        old_full_url = entry["fullUrl"]
        entry["fullUrl"] = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, old_full_url)}"
        reference_map[old_full_url] = entry["fullUrl"]
        entry["request"] = {
            "method": "PUT",
            "url": f"{resource['resourceType']}/{resource['id']}",
        }
    bundle["entry"][1]["resource"]["subject"]["reference"] = reference_map["urn:uuid:patient-1"]
    bundle["entry"][2]["resource"]["subject"]["reference"] = reference_map["urn:uuid:patient-1"]
    bundle["entry"][2]["resource"]["encounter"]["reference"] = reference_map["urn:uuid:encounter-1"]
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="conflicting_evidence",
    )

    derived = apply_raw_fhir_patch(bundle, patch)
    inserted = derived["entry"][len(bundle["entry"]) :]

    assert all(entry["request"]["method"] == "PUT" for entry in inserted)
    assert all(entry["request"]["url"] for entry in inserted)
    assert all(
        str(uuid.UUID(entry["fullUrl"].removeprefix("urn:uuid:")))
        == entry["fullUrl"].removeprefix("urn:uuid:")
        for entry in inserted
    )
    provenances = [
        entry["resource"]
        for entry in inserted
        if entry["resource"].get("resourceType") == "Provenance"
    ]
    assert {item["activity"]["coding"][0]["code"] for item in provenances} == {
        "CREATE",
        "UPDATE",
    }
    assert all(item["agent"][0]["who"]["reference"].startswith("urn:uuid:") for item in provenances)
    assert all(
        "conflict" not in json.dumps(item).casefold()
        and "distractor" not in json.dumps(item).casefold()
        for item in provenances
    )
    assert validate_reference_closure_strict(derived).passed is True


def test_conflicting_observation_shares_event_identity_without_invented_performer():
    bundle = _observation_bundle()
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="conflicting_evidence",
    )

    derived = apply_raw_fhir_patch(bundle, patch)
    clone = next(
        entry["resource"]
        for entry in derived["entry"]
        if entry["resource"].get("resourceType") == "Observation"
        and entry["resource"].get("id") != "result-1"
    )
    original = next(
        entry["resource"] for entry in derived["entry"] if entry["resource"].get("id") == "result-1"
    )

    assert clone["identifier"] == original["identifier"]
    assert "performer" not in clone
    records = normalize_fhir_bundle(derived).records_by_resource_type("Observation")
    assert len({record.logical_evidence_group for record in records}) == 1
    assert validate_reference_closure(derived).passed is True


def test_conflicting_lab_observation_does_not_add_a_polarity_marking_performer():
    bundle = _observation_bundle()
    bundle["entry"][2]["resource"]["meta"] = {
        "profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-observation-lab"]
    }
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="conflicting_evidence",
    )

    derived = apply_raw_fhir_patch(bundle, patch)
    clone = next(
        entry["resource"]
        for entry in derived["entry"]
        if entry["resource"].get("resourceType") == "Observation"
        and entry["resource"].get("id") != "result-1"
    )

    assert "performer" not in clone
    assert "Synthetic laboratory" not in json.dumps(derived)
    assert validate_reference_closure(derived).passed is True


def test_conflicting_medication_request_does_not_repeat_false_profile_assertion():
    bundle = _medication_bundle()
    bundle["entry"][1]["resource"]["meta"] = {
        "versionId": "3",
        "profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-medicationrequest"],
    }
    patch = build_truth_state_patch(
        bundle,
        resource_type="MedicationRequest",
        resource_id="request-1",
        target_state="conflicting_evidence",
    )

    derived = apply_raw_fhir_patch(bundle, patch)
    clone = next(
        entry["resource"]
        for entry in derived["entry"]
        if entry["resource"].get("resourceType") == "MedicationRequest"
        and entry["resource"].get("id") != "request-1"
    )

    assert "meta" not in clone


def test_refuted_observation_and_provenance_use_standard_codes():
    bundle = _observation_bundle()
    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="refuted",
    )

    derived = apply_raw_fhir_patch(bundle, patch)
    observation = next(
        entry["resource"] for entry in derived["entry"] if entry["resource"].get("id") == "result-1"
    )
    provenance = next(
        entry["resource"]
        for entry in derived["entry"]
        if entry["resource"].get("resourceType") == "Provenance"
    )

    assert observation["status"] == "cancelled"
    assert observation["dataAbsentReason"]["coding"][0]["code"] == "not-performed"
    assert provenance["activity"]["coding"][0] == {
        "system": "http://terminology.hl7.org/CodeSystem/v3-DataOperation",
        "code": "UPDATE",
    }
    assert provenance["activity"]["text"] == "FHIR resource update"
    assert provenance["entity"][0]["what"]["identifier"]["value"] == content_hash(
        _observation_bundle()["entry"][2]["resource"]
    )
    assert provenance["agent"][0]["who"]["reference"].startswith("urn:uuid:")
    assert "refuted" not in json.dumps(provenance).casefold()


def test_raw_intervention_family_executes_truth_states_and_distractor():
    family = compile_raw_intervention_family(
        _observation_bundle(),
        program=_observation_program(),
        resource_type="Observation",
        resource_id="result-1",
    )

    assert [member.state for member in family.members] == [
        "supported",
        "refuted",
        "insufficient_evidence",
        "conflicting_evidence",
        "answer_preserving_distractor",
    ]
    assert [member.answer.value for member in family.members] == [
        "supported",
        "refuted",
        "insufficient_evidence",
        "conflicting_evidence",
        "supported",
    ]
    assert all(member.patch.patch_hash.startswith("sha256:") for member in family.members)

    distractor = apply_raw_fhir_patch(_observation_bundle(), family.members[-1].patch)
    inserted = next(
        entry["resource"]
        for entry in distractor["entry"][len(_observation_bundle()["entry"]) :]
        if entry["resource"].get("resourceType") not in {"Device", "Provenance"}
    )
    assert inserted["resourceType"] != "Observation"
    assert str(uuid.UUID(inserted["id"])) == inserted["id"]
    assert "distractor" not in inserted["id"] and "conflict" not in inserted["id"]


def test_medication_lifecycle_distinguishes_order_from_administration():
    family = compile_raw_intervention_family(
        _medication_bundle(),
        program=_medication_program(),
        resource_type="MedicationAdministration",
        resource_id="administration-1",
    )

    assert [member.answer.value for member in family.members] == [
        "supported",
        "refuted",
        "insufficient_evidence",
        "conflicting_evidence",
        "supported",
    ]
    supported = apply_raw_fhir_patch(_medication_bundle(), family.members[0].patch)
    snapshot = normalize_fhir_bundle(supported)
    assert len(snapshot.records_by_resource_type("MedicationRequest")) == 1
    assert len(snapshot.records_by_resource_type("MedicationAdministration")) == 1


def test_reference_closure_rejects_dangling_local_reference():
    bundle = _observation_bundle()
    bundle["entry"][2]["resource"]["encounter"] = {"reference": "Encounter/missing"}

    report = validate_reference_closure(bundle)

    assert report.passed is False
    assert report.dangling[0].reference == "Encounter/missing"
    try:
        build_truth_state_patch(
            bundle,
            resource_type="Observation",
            resource_id="result-1",
            target_state="supported",
        )
    except RawFHIRPatchError:
        pass
    else:
        raise AssertionError("dangling input must fail patch compilation")


def test_reference_closure_rejects_unresolved_absolute_reference():
    bundle = _observation_bundle()
    bundle["entry"][2]["resource"]["encounter"] = {
        "reference": "https://patient-tool-gym.dev/fhir/Encounter/missing"
    }

    report = validate_reference_closure(bundle)

    assert report.passed is False
    assert report.dangling[0].reference.endswith("/Encounter/missing")


def test_reference_closure_accepts_allowlisted_fhir_conditional_external_reference():
    bundle = _observation_bundle()
    bundle["entry"][2]["resource"]["performer"] = [
        {"reference": ("Practitioner?identifier=http://hl7.org/fhir/sid/us-npi|9999988295")}
    ]

    report = validate_reference_closure(
        bundle,
        allowed_external_conditional_reference_types=frozenset({"Practitioner"}),
    )

    assert report.passed is True
    assert report.reference_count == 4


def test_mutating_patch_without_bundle_timestamp_requires_explicit_recorded_time():
    bundle = _observation_bundle()
    bundle.pop("timestamp")

    with pytest.raises(RawFHIRPatchError, match="requires Bundle.timestamp"):
        build_truth_state_patch(
            bundle,
            resource_type="Observation",
            resource_id="result-1",
            target_state="refuted",
        )

    patch = build_truth_state_patch(
        bundle,
        resource_type="Observation",
        resource_id="result-1",
        target_state="refuted",
        provenance_recorded_at="2026-01-01T11:00:00Z",
    )
    assert patch.operations


def test_generated_ids_are_deterministic_opaque_uuids_without_role_labels():
    first = build_truth_state_patch(
        _observation_bundle(),
        resource_type="Observation",
        resource_id="result-1",
        target_state="conflicting_evidence",
    )
    second = build_truth_state_patch(
        _observation_bundle(),
        resource_type="Observation",
        resource_id="result-1",
        target_state="conflicting_evidence",
    )
    first_insert_ids = [
        operation.resource["id"] for operation in first.operations if operation.op == "insert"
    ]
    second_insert_ids = [
        operation.resource["id"] for operation in second.operations if operation.op == "insert"
    ]

    assert first_insert_ids == second_insert_ids
    assert all(str(uuid.UUID(resource_id)) == resource_id for resource_id in first_insert_ids)
    assert all(
        "conflict" not in resource_id and "distractor" not in resource_id
        for resource_id in first_insert_ids
    )


def test_component_observation_intervention_fails_closed():
    bundle = _observation_bundle()
    observation = bundle["entry"][2]["resource"]
    observation.pop("valueQuantity")
    observation["component"] = [
        {
            "code": _concept("http://loinc.org", "7777-8"),
            "valueQuantity": {"value": 4.2, "unit": "mg"},
        }
    ]

    with pytest.raises(RawFHIRPatchError, match="component panels"):
        build_truth_state_patch(
            bundle,
            resource_type="Observation",
            resource_id="result-1",
            target_state="refuted",
        )


def test_refuted_procedure_removes_every_performed_choice_and_rejects_invalid_multiplicity():
    bundle = _observation_bundle()
    procedure = bundle["entry"][2]["resource"]
    procedure.clear()
    procedure.update(
        {
            "resourceType": "Procedure",
            "id": "procedure-1",
            "status": "completed",
            "subject": {"reference": "urn:uuid:patient-1"},
            "code": _concept("http://snomed.info/sct", "80146002"),
            "performedTiming": {"event": ["2026-01-01T10:00:00Z"]},
        }
    )
    bundle["entry"][2]["fullUrl"] = "urn:uuid:procedure-1"
    patch = build_truth_state_patch(
        bundle,
        resource_type="Procedure",
        resource_id="procedure-1",
        target_state="refuted",
    )
    derived = apply_raw_fhir_patch(bundle, patch)
    refuted = next(
        entry["resource"]
        for entry in derived["entry"]
        if entry["resource"].get("id") == "procedure-1"
    )

    assert refuted["status"] == "not-done"
    assert not any(key.startswith("performed") for key in refuted)

    bundle["entry"][2]["resource"]["performedDateTime"] = "2026-01-01T10:00:00Z"
    with pytest.raises(RawFHIRPatchError, match="multiple performed"):
        build_truth_state_patch(
            bundle,
            resource_type="Procedure",
            resource_id="procedure-1",
            target_state="supported",
        )


def test_factorial_supported_plain_is_exact_base_and_two_changed_targets_replay():
    bundle = _observation_bundle()
    second = json.loads(json.dumps(bundle["entry"][2]))
    second["fullUrl"] = "urn:uuid:result-2"
    second["resource"]["id"] = "result-2"
    second["resource"]["code"] = _concept("http://loinc.org", "7777-8")
    bundle["entry"].append(second)

    baseline = build_compositional_factorial_patch_chain(
        bundle,
        primary_resource_type="Observation",
        primary_resource_id="result-1",
        primary_state="supported",
        companion_resource_type="Observation",
        companion_resource_id="result-2",
        companion_state="supported",
        distractor_present=False,
    )
    assert baseline.patches == ()
    assert apply_raw_fhir_patch_chain(bundle, baseline) == bundle

    changed = build_compositional_factorial_patch_chain(
        bundle,
        primary_resource_type="Observation",
        primary_resource_id="result-1",
        primary_state="refuted",
        companion_resource_type="Observation",
        companion_resource_id="result-2",
        companion_state="insufficient_evidence",
        distractor_present=False,
    )
    derived = apply_raw_fhir_patch_chain(bundle, changed)
    observations = {
        entry["resource"]["id"]: entry["resource"]
        for entry in derived["entry"]
        if entry["resource"].get("resourceType") == "Observation"
    }
    assert len(changed.patches) == 2
    assert observations["result-1"]["status"] == "cancelled"
    assert observations["result-2"]["dataAbsentReason"]["coding"][0]["code"] == "unknown"


def test_factorial_procedure_non_supported_state_is_explicitly_unrepresentable():
    bundle = _observation_bundle()
    procedure = bundle["entry"][2]["resource"]
    procedure.clear()
    procedure.update(
        {
            "resourceType": "Procedure",
            "id": "procedure-1",
            "status": "completed",
            "subject": {"reference": "urn:uuid:patient-1"},
            "code": _concept("http://snomed.info/sct", "80146002"),
            "performedDateTime": "2026-01-01T10:00:00Z",
        }
    )
    bundle["entry"][2]["fullUrl"] = "urn:uuid:procedure-1"
    second = json.loads(json.dumps(bundle["entry"][2]))
    second["fullUrl"] = "urn:uuid:procedure-2"
    second["resource"]["id"] = "procedure-2"
    second["resource"]["code"] = _concept("http://snomed.info/sct", "73761001")
    bundle["entry"].append(second)

    with pytest.raises(ValueError, match="exact-date Procedure"):
        build_compositional_factorial_patch_chain(
            bundle,
            primary_resource_type="Procedure",
            primary_resource_id="procedure-1",
            primary_state="supported",
            companion_resource_type="Procedure",
            companion_resource_id="procedure-2",
            companion_state="refuted",
            distractor_present=False,
        )


def test_strict_transaction_closure_requires_urn_for_post_created_resource():
    bundle = _observation_bundle()
    bundle["type"] = "transaction"
    for entry in bundle["entry"]:
        old = entry["fullUrl"]
        entry["fullUrl"] = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, old)}"
        entry["request"] = {
            "method": "POST",
            "url": entry["resource"]["resourceType"],
        }
    patient_full_url = bundle["entry"][0]["fullUrl"]
    encounter_full_url = bundle["entry"][1]["fullUrl"]
    bundle["entry"][1]["resource"]["subject"]["reference"] = patient_full_url
    observation = bundle["entry"][2]["resource"]
    observation["subject"]["reference"] = "Patient/patient-1"
    observation["encounter"]["reference"] = encounter_full_url

    assert validate_reference_closure_strict(bundle).passed is False
    observation["subject"]["reference"] = patient_full_url
    assert validate_reference_closure_strict(bundle).passed is True
