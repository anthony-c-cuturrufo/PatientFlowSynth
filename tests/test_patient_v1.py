import copy
import json
from pathlib import Path

import pytest

from patient_tool_gym.evidence import load_snapshot, snapshot_from_payloads
from patient_tool_gym.patient_v1 import (
    PATIENT_V1_FAMILY_SPECS,
    PresenceQuery,
    SemanticPredicate,
    _complete_records,
    depth_presence_program,
    generate_patient_v1_families,
    get_patient_v1_family_spec,
    summarize_patient_v1_yield,
)
from patient_tool_gym.program import ProgramExecutor, ToolNode, program_signature
from patient_tool_gym.task import TaskArtifact, deterministic_task_id
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import verify_task_artifact

FIXTURE = Path(__file__).parent / "fixtures" / "patient_v1.jsonl"


def _payloads() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _families(
    tasks: tuple[TaskArtifact, ...],
) -> dict[str, dict[str, TaskArtifact]]:
    families: dict[str, dict[str, TaskArtifact]] = {}
    for task in tasks:
        spec_id = task.metadata["lineage"]["spec_id"]
        families.setdefault(spec_id, {})[task.construction_type] = task
    return families


def test_v1_emits_five_complete_verified_matched_families():
    result = generate_patient_v1_families((load_snapshot(FIXTURE),))

    assert result.rejected == ()
    assert len(result.tasks) == 15
    assert set(_families(result.tasks)) == {spec.spec_id for spec in PATIENT_V1_FAMILY_SPECS}
    assert all(task.verification.passed for task in result.tasks)
    assert all(task.gold_answer.kind == "boolean" for task in result.tasks)
    assert all(task.gold_answer.value is True for task in result.tasks)
    assert all(task.provenance.generator_version == "3" for task in result.tasks)
    assert all(task.metadata["lineage"]["template_version"] == "3" for task in result.tasks)

    for family in _families(result.tasks).values():
        assert set(family) == {"atomic", "depth", "width"}
        atomic = family["atomic"]
        depth = family["depth"]
        width = family["width"]
        assert atomic.parent_task_ids == ()
        assert depth.parent_task_ids == width.parent_task_ids == (atomic.task_id,)
        assert atomic.difficulty.dependency_depth == 1
        assert atomic.difficulty.independent_width == 1
        assert depth.difficulty.dependency_depth == 2
        assert depth.difficulty.independent_width == 1
        assert width.difficulty.dependency_depth == 1
        assert width.difficulty.independent_width == 2
        assert len(atomic.gold_evidence_ids) == 1
        assert len(depth.gold_evidence_ids) == 2
        assert len(width.gold_evidence_ids) == 2


def test_every_answer_changing_filter_has_a_patient_facing_realization():
    spec_ids = [
        *(spec.spec_id for spec in PATIENT_V1_FAMILY_SPECS),
        "synthea_employment_status_presence",
        "synthea_insulin_order_presence",
        "synthea_dental_referral_presence",
    ]

    for spec_id in spec_ids:
        spec = get_patient_v1_family_spec(spec_id)
        assert spec is not None
        for query in (spec.primary, spec.companion):
            semantic_arguments = tuple(
                (argument, value)
                for argument, value in query.arguments
                if argument not in {"system", "code"}
            )
            predicates = tuple(
                (predicate.argument, predicate.value) for predicate in query.semantic_predicates
            )
            assert predicates == semantic_arguments
            assert all(
                predicate.patient_phrase.casefold() in query.patient_phrase.casefold()
                for predicate in query.semantic_predicates
            )
            assert query.concept_phrase.casefold() in query.patient_phrase.casefold()


def test_presence_query_rejects_an_unsurfaced_or_misworded_filter():
    with pytest.raises(ValueError, match="every non-concept query argument"):
        PresenceQuery(
            tool="search_procedures",
            resource_type="Procedure",
            arguments=(
                ("system", "http://snomed.info/sct"),
                ("code", "73761001"),
                ("status", "completed"),
            ),
            concept_phrase="colonoscopy",
            patient_phrase="a colonoscopy",
            semantic_predicates=(),
        )

    with pytest.raises(ValueError, match="must appear in the patient phrase"):
        PresenceQuery(
            tool="search_procedures",
            resource_type="Procedure",
            arguments=(
                ("system", "http://snomed.info/sct"),
                ("code", "73761001"),
                ("status", "completed"),
            ),
            concept_phrase="colonoscopy",
            patient_phrase="a colonoscopy",
            semantic_predicates=(SemanticPredicate("status", "completed", "marked completed"),),
        )

    with pytest.raises(ValueError, match="must appear in the patient phrase"):
        PresenceQuery(
            tool="search_conditions",
            resource_type="Condition",
            arguments=(
                ("system", "http://snomed.info/sct"),
                ("code", "59621000"),
                ("clinical_status", "active"),
            ),
            concept_phrase="condition entry",
            patient_phrase="an inactive condition entry",
            semantic_predicates=(SemanticPredicate("clinical_status", "active", "active"),),
        )


@pytest.mark.parametrize(
    ("spec_id", "evidence_suffix", "payload_path", "replacement", "relaxed_argument"),
    [
        (
            "condition_documented_presence",
            "Condition:hypertension",
            ("attributes", "clinicalStatus"),
            "inactive",
            "clinical_status",
        ),
        (
            "condition_documented_presence",
            "Condition:hypertension",
            ("attributes", "verificationStatus"),
            "provisional",
            "verification_status",
        ),
        (
            "medication_order_presence",
            "MedicationRequest:lisinopril-order",
            ("status",),
            "completed",
            "status",
        ),
        (
            "medication_order_presence",
            "MedicationRequest:lisinopril-order",
            ("attributes", "intent"),
            "plan",
            "intent",
        ),
        (
            "procedure_completed_presence",
            "Procedure:colonoscopy",
            ("status",),
            "in-progress",
            "status",
        ),
        (
            "referral_request_presence",
            "ServiceRequest:cardiology-referral",
            ("category",),
            "diagnostic",
            "category",
        ),
        (
            "referral_request_presence",
            "ServiceRequest:cardiology-referral",
            ("status",),
            "completed",
            "status",
        ),
        (
            "referral_request_presence",
            "ServiceRequest:cardiology-referral",
            ("attributes", "intent"),
            "plan",
            "intent",
        ),
        (
            "referral_request_presence",
            "ServiceRequest:cardiology-referral",
            ("attributes", "priority"),
            "urgent",
            "priority",
        ),
    ],
)
def test_semantic_filter_counterfactual_changes_answer_but_not_concept_presence(
    spec_id,
    evidence_suffix,
    payload_path,
    replacement,
    relaxed_argument,
):
    original = load_snapshot(FIXTURE)
    task = _families(generate_patient_v1_families((original,)).tasks)[spec_id]["atomic"]
    payloads = _payloads()
    target = next(
        payload for payload in payloads if payload["evidence_id"].endswith(evidence_suffix)
    )
    container = target
    for key in payload_path[:-1]:
        container = container[key]
    container[payload_path[-1]] = replacement

    registry = ToolRegistry(snapshot_from_payloads(payloads))
    replay = ProgramExecutor(registry).execute(task.program)
    assert replay.answer.value is False

    spec = get_patient_v1_family_spec(spec_id)
    assert spec is not None
    relaxed_arguments = {
        **spec.primary.tool_arguments(),
        "date_from": "2026-05-10",
        "date_to": "2026-05-10",
    }
    relaxed_arguments.pop(relaxed_argument)
    relaxed = registry.call(spec.primary.tool, relaxed_arguments)
    assert relaxed["ok"] is True and relaxed["complete"] is True
    assert [record["evidence_id"] for record in relaxed["records"]] == [target["evidence_id"]]


@pytest.mark.parametrize(
    ("spec_id", "evidence_suffix", "new_resource_type", "new_category", "kind_argument"),
    [
        (
            "medication_order_presence",
            "MedicationRequest:lisinopril-order",
            "MedicationAdministration",
            "medication_administration",
            "record_kind",
        ),
        (
            "clinical_document_presence",
            "DocumentReference:after-visit-summary",
            "DiagnosticReport",
            "laboratory-report",
            "document_kind",
        ),
    ],
)
def test_record_class_counterfactual_changes_answer_but_preserves_coded_concept(
    spec_id,
    evidence_suffix,
    new_resource_type,
    new_category,
    kind_argument,
):
    original = load_snapshot(FIXTURE)
    task = _families(generate_patient_v1_families((original,)).tasks)[spec_id]["atomic"]
    payloads = _payloads()
    target = next(
        payload for payload in payloads if payload["evidence_id"].endswith(evidence_suffix)
    )
    target["resource_type"] = new_resource_type
    target["category"] = new_category
    target["evidence_id"] = f"synthea:v1-patient:{new_resource_type}:kind-counterfactual"
    target["logical_evidence_group"] = f"record:{target['evidence_id']}"
    target["source"] = {
        "resource_id": "kind-counterfactual",
        "resource_type": new_resource_type,
    }

    registry = ToolRegistry(snapshot_from_payloads(payloads))
    assert ProgramExecutor(registry).execute(task.program).answer.value is False

    spec = get_patient_v1_family_spec(spec_id)
    assert spec is not None
    relaxed_arguments = {
        **spec.primary.tool_arguments(),
        "date_from": "2026-05-10",
        "date_to": "2026-05-10",
        kind_argument: "any",
    }
    relaxed = registry.call(spec.primary.tool, relaxed_arguments)
    assert relaxed["ok"] is True and relaxed["complete"] is True
    assert [record["evidence_id"] for record in relaxed["records"]] == [target["evidence_id"]]


def test_depth_scope_says_completed_because_finished_filter_changes_the_answer():
    payloads = _payloads()
    planned = copy.deepcopy(
        next(
            payload for payload in payloads if payload["evidence_id"].endswith("Encounter:enc-2026")
        )
    )
    planned.update(
        {
            "evidence_id": "synthea:v1-patient:Encounter:planned-wellness",
            "logical_evidence_group": "record:synthea:v1-patient:Encounter:planned-wellness",
            "status": "planned",
            "time": {
                "start": "2026-07-10T09:00:00Z",
                "end": "2026-07-10T10:00:00Z",
            },
            "value": "planned",
        }
    )
    planned["source"] = {"resource_id": "planned-wellness"}
    payloads.append(planned)
    snapshot = snapshot_from_payloads(payloads)

    task = _families(generate_patient_v1_families((snapshot,)).tasks)[
        "condition_documented_presence"
    ]["depth"]
    anchor = next(
        node
        for node in task.program.nodes
        if isinstance(node, ToolNode) and node.id == "anchor_encounter"
    )
    assert anchor.arguments["status"] == "finished"
    assert "most recent completed wellness visit" in task.question.casefold()
    assert ProgramExecutor(ToolRegistry(snapshot)).execute(task.program).answer.value is True

    spec = get_patient_v1_family_spec("condition_documented_presence")
    assert spec is not None
    relaxed_encounter_arguments = dict(anchor.arguments)
    relaxed_encounter_arguments.pop("status")
    counterfactual = depth_presence_program(spec.primary, relaxed_encounter_arguments)
    assert ProgramExecutor(ToolRegistry(snapshot)).execute(counterfactual).answer.value is False


def test_each_domain_uses_encounter_bound_search_and_a_real_distractor():
    snapshot = load_snapshot(FIXTURE)
    result = generate_patient_v1_families((snapshot,))
    registry = ToolRegistry(snapshot)

    for spec in PATIENT_V1_FAMILY_SPECS:
        family = _families(result.tasks)[spec.spec_id]
        atomic = family["atomic"]
        depth = family["depth"]
        atomic_id = atomic.gold_evidence_ids[0]

        assert [call.tool for call in depth.reference_trace] == [
            "search_encounters",
            spec.primary.tool,
        ]
        encounter_call, domain_call = depth.reference_trace
        encounter_id = encounter_call.evidence_ids[0]
        assert domain_call.depends_on == ("anchor_encounter",)
        assert domain_call.resolved_arguments["encounter_id"] == encounter_id
        assert set(depth.gold_evidence_ids) == {encounter_id, atomic_id}

        unbound = registry.call(spec.primary.tool, spec.primary.tool_arguments())
        assert unbound["ok"] is True and unbound["complete"] is True
        assert unbound["records"][0]["evidence_id"] != atomic_id


def test_v1_generation_is_invariant_to_record_order():
    original = load_snapshot(FIXTURE)
    reordered = snapshot_from_payloads(reversed(_payloads()))

    assert original.snapshot_id == reordered.snapshot_id
    assert generate_patient_v1_families((original,)) == generate_patient_v1_families((reordered,))


def test_v1_verifier_rejects_question_lineage_parent_and_program_tampering():
    snapshot = load_snapshot(FIXTURE)
    registry = ToolRegistry(snapshot)
    tasks = generate_patient_v1_families((snapshot,)).tasks
    medication = _families(tasks)["medication_order_presence"]

    changed_question = medication["atomic"].model_copy(
        update={"question": "Am I taking lisinopril?"}
    )
    assert not verify_task_artifact(changed_question, registry).passed

    changed_lineage = copy.deepcopy(medication["atomic"].metadata)
    changed_lineage["lineage"]["spec_id"] = "unknown"
    forged_lineage = medication["atomic"].model_copy(update={"metadata": changed_lineage})
    assert not verify_task_artifact(forged_lineage, registry).passed

    changed_parent = medication["depth"].model_copy(
        update={"parent_task_ids": ("patient-task-forged",)}
    )
    assert not verify_task_artifact(changed_parent, registry).passed

    program_payload = medication["atomic"].program.model_dump(mode="json")
    primary = next(node for node in program_payload["nodes"] if node["op"] == "tool")
    primary["arguments"]["record_kind"] = "administration"
    changed_program = medication["atomic"].program.model_validate(program_payload)
    forged_program = medication["atomic"].model_copy(update={"program": changed_program})
    report = verify_task_artifact(forged_program, registry)
    assert not report.passed
    assert "patient_v1_contract_invalid" in report.errors

    stripped_contract = medication["atomic"].model_copy(
        update={
            "metadata": {},
            "provenance": medication["atomic"].provenance.model_copy(
                update={"generator": "hand_authored"}
            ),
        }
    )
    report = verify_task_artifact(stripped_contract, registry)
    assert not report.passed
    assert "patient_v1_contract_invalid" in report.errors

    forged_family_id = "forged-family"
    forged_identity = medication["atomic"].model_copy(
        update={
            "family_id": forged_family_id,
            "task_id": deterministic_task_id(
                medication["atomic"].patient_snapshot_id,
                medication["atomic"].program,
                forged_family_id,
                "atomic",
            ),
            "metadata": {},
        }
    )
    report = verify_task_artifact(forged_identity, registry)
    assert not report.passed
    assert "patient_v1_contract_invalid" in report.errors


def test_program_signatures_preserve_medication_and_document_selectors():
    tasks = generate_patient_v1_families((load_snapshot(FIXTURE),)).tasks
    medication = _families(tasks)["medication_order_presence"]["atomic"]
    document = _families(tasks)["clinical_document_presence"]["atomic"]

    medication_payload = medication.program.model_dump(mode="json")
    medication_node = next(node for node in medication_payload["nodes"] if node["op"] == "tool")
    medication_node["arguments"]["record_kind"] = "administration"
    medication_changed = medication.program.model_validate(medication_payload)

    document_payload = document.program.model_dump(mode="json")
    document_node = next(node for node in document_payload["nodes"] if node["op"] == "tool")
    document_node["arguments"]["document_kind"] = "diagnostic_report"
    document_changed = document.program.model_validate(document_payload)

    assert program_signature(medication.program) != program_signature(medication_changed)
    assert program_signature(document.program) != program_signature(document_changed)


def test_missing_depth_distractor_rejects_only_its_family():
    payloads = _payloads()
    payloads = [
        payload
        for payload in payloads
        if not payload["evidence_id"].endswith("hypertension-follow-up")
    ]

    result = generate_patient_v1_families((snapshot_from_payloads(payloads),))

    assert len(result.tasks) == 12
    assert len(result.rejected) == 2
    assert all(
        rejection.domain_spec_id == "condition_documented_presence" for rejection in result.rejected
    )
    assert [rejection.attempt_index for rejection in result.rejected] == [1, 2]
    assert {rejection.reason for rejection in result.rejected} == {"NO_DEPTH_DISAMBIGUATOR"}


def test_allowlisted_synthea_specs_replace_missing_demo_concepts():
    payloads = _payloads()
    code_map = {
        "59621000": "160903007",
        "44054006": "423315002",
        "29046": "106892",
        "860975": "1664463",
        "73761001": "103697008",
        "86273004": "428211000124100",
    }
    for payload in payloads:
        code = payload.get("code", {}).get("code")
        if code in code_map:
            payload["code"]["code"] = code_map[code]
        if payload.get("resource_type") == "Condition":
            payload["attributes"]["clinicalStatus"] = "resolved"
            payload["value"] = "resolved"

    result = generate_patient_v1_families((snapshot_from_payloads(payloads),))

    assert len(result.tasks) == 15
    assert result.rejected == ()
    assert all(task.verification.passed for task in result.tasks)
    assert {task.metadata["lineage"]["spec_id"] for task in result.tasks} == {
        "synthea_employment_status_presence",
        "synthea_insulin_order_presence",
        "synthea_dental_referral_presence",
        "clinical_document_presence",
        "referral_request_presence",
    }


def test_document_semantics_are_h_and_p_without_a_lifecycle_filter():
    spec = get_patient_v1_family_spec("clinical_document_presence")

    assert spec is not None
    assert dict(spec.primary.arguments) == {
        "document_kind": "document_reference",
        "system": "http://loinc.org",
        "code": "34117-2",
    }
    assert spec.primary.concept_phrase == "history and physical note"
    task = _families(generate_patient_v1_families((load_snapshot(FIXTURE),)).tasks)[
        "clinical_document_presence"
    ]["atomic"]
    assert "history and physical note" in task.question.casefold()
    assert "after-visit summary" not in task.question.casefold()
    assert "current" not in task.question.casefold()


def test_document_width_never_counts_one_logical_evidence_group_twice():
    payloads = _payloads()
    primary = next(
        payload
        for payload in payloads
        if payload["evidence_id"].endswith("DocumentReference:after-visit-summary")
    )
    companion = next(
        payload
        for payload in payloads
        if payload["evidence_id"].endswith("DocumentReference:progress-note")
    )
    companion["logical_evidence_group"] = primary["logical_evidence_group"]

    result = generate_patient_v1_families((snapshot_from_payloads(payloads),))

    document_rejections = [
        item for item in result.rejected if item.domain_spec_id == "clinical_document_presence"
    ]
    assert [(item.spec_id, item.reason) for item in document_rejections] == [
        ("clinical_document_presence", "NO_EXPLICIT_RELATION")
    ]
    coverage = next(
        item for item in result.coverage if item.domain_spec_id == "clinical_document_presence"
    )
    assert coverage.selected_spec_id == (
        "reviewed_clinical_document__history_physical_note__cbc_panel_report"
    )
    width = _families(result.tasks)[coverage.selected_spec_id]["width"]
    evidence = [
        snapshot_from_payloads(payloads).record_by_id(evidence_id)
        for evidence_id in width.gold_evidence_ids
    ]
    assert all(record is not None for record in evidence)
    assert len({record.logical_evidence_group for record in evidence if record}) == 2


def test_absent_service_requests_are_capability_ineligible_not_task_failures():
    payloads = [payload for payload in _payloads() if payload["resource_type"] != "ServiceRequest"]

    result = generate_patient_v1_families((snapshot_from_payloads(payloads),))

    referral_rejections = [
        item for item in result.rejected if item.domain_spec_id == "referral_request_presence"
    ]
    assert len(referral_rejections) == 1
    assert referral_rejections[0].reason == "UNSUPPORTED_CAPABILITY"
    assert referral_rejections[0].capability_supported is False
    referral_coverage = next(
        item for item in result.coverage if item.domain_spec_id == "referral_request_presence"
    )
    assert referral_coverage.resource_count == 0
    assert referral_coverage.attempted_candidate_count == 0
    assert referral_coverage.generated_family_count == 0

    summary = summarize_patient_v1_yield(result)
    assert summary.patient_count == 1
    assert summary.domain_count == 5
    assert summary.supported_domain_count == 4
    assert summary.generated_family_count == 4
    assert summary.rejection_counts == {"UNSUPPORTED_CAPABILITY": 1}


def test_reviewed_medication_allowlist_distinguishes_ingredient_and_product_wording():
    ingredient_pair = get_patient_v1_family_spec("medication_order_presence")
    product_pair = get_patient_v1_family_spec(
        "reviewed_medication__lisinopril_10_mg_tablet__humulin_70_30"
    )

    assert ingredient_pair is not None and product_pair is not None
    assert dict(ingredient_pair.primary.arguments)["code"] == "29046"
    assert ingredient_pair.primary.concept_phrase == "lisinopril order"
    assert "10 mg" not in ingredient_pair.primary.concept_phrase
    assert dict(product_pair.primary.arguments)["code"] == "314076"
    assert product_pair.primary.concept_phrase == "lisinopril 10 mg oral tablet order"
    humulin_pair = get_patient_v1_family_spec("synthea_insulin_order_presence")
    oral_health_pair = get_patient_v1_family_spec(
        "reviewed_procedure__oral_health_education__depression_screening"
    )
    assert humulin_pair is not None and oral_health_pair is not None
    assert humulin_pair.primary.patient_phrase.startswith("a Humulin")
    assert oral_health_pair.primary.patient_phrase.startswith("an oral-health")


@pytest.mark.parametrize(
    ("display", "expected", "forbidden"),
    [
        (
            "Urgent care clinic (environment)",
            "most recent completed urgent-care visit",
            "urgent care clinic",
        ),
        (
            "Follow-up encounter (environment)",
            "most recent completed follow-up visit",
            "follow-up encounter visit",
        ),
    ],
)
def test_encounter_wording_uses_natural_visit_nouns(display, expected, forbidden):
    payloads = _payloads()
    encounter = next(
        payload for payload in payloads if payload["evidence_id"].endswith("Encounter:enc-2026")
    )
    encounter["display"] = display
    encounter["text"] = display

    tasks = generate_patient_v1_families((snapshot_from_payloads(payloads),)).tasks
    depth = _families(tasks)["condition_documented_presence"]["depth"]

    assert expected in depth.question.casefold()
    assert forbidden not in depth.question.casefold()


def test_discovery_collects_every_page_before_declaring_scope_complete():
    template = next(
        payload
        for payload in _payloads()
        if payload["evidence_id"].endswith("Condition:hypertension")
    )
    payloads = []
    for index in range(105):
        payload = copy.deepcopy(template)
        resource_id = f"paged-condition-{index:03d}"
        payload["evidence_id"] = f"synthea:patient-v1:Condition:{resource_id}"
        payload["resource_id"] = resource_id
        payload["logical_evidence_group"] = f"record:{payload['evidence_id']}"
        payloads.append(payload)

    registry = ToolRegistry(snapshot_from_payloads(payloads))
    records = _complete_records(
        registry,
        "search_conditions",
        {"code": "59621000", "limit": 25},
    )

    assert len(records) == 105
    assert len({record["evidence_id"] for record in records}) == 105


def test_v1_questions_are_chart_presence_questions_not_clinical_claims():
    tasks = generate_patient_v1_families((load_snapshot(FIXTURE),)).tasks

    for task in tasks:
        question = task.question.casefold()
        assert "do i have" not in question
        assert "am i taking" not in question
        assert "scheduled" not in question
        assert "my chart show" in question

    assert not any(
        node.tool.startswith(("create", "update", "delete", "write"))
        for task in tasks
        for node in task.program.nodes
        if isinstance(node, ToolNode)
    )
