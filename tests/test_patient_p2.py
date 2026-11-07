import copy
import json
from pathlib import Path

import patient_tool_gym.patient_p2 as p2
from patient_tool_gym.evidence import snapshot_from_payloads
from patient_tool_gym.patient_p2 import (
    P2_DOMAIN_SPECS,
    generate_patient_p2_tasks,
    summarize_patient_p2_yield,
)
from patient_tool_gym.program import ToolNode
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import verify_task_artifact

FIXTURE = Path(__file__).parent / "fixtures" / "patient_v1.jsonl"
SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
CVX = "http://hl7.org/fhir/sid/cvx"


def _base_payloads() -> list[dict]:
    payloads = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for payload in payloads:
        if payload["evidence_id"].endswith("Encounter:enc-2026"):
            payload["code"] = {
                "system": SNOMED,
                "code": "185349003",
                "display": "Encounter for check up",
            }
            payload["display"] = "Encounter for check up (procedure)"
            payload["text"] = payload["display"]
        elif payload["evidence_id"].endswith("Encounter:enc-2026-follow-up"):
            payload["code"] = {
                "system": SNOMED,
                "code": "185349003",
                "display": "Encounter for check up",
            }
            payload["display"] = "Encounter for check up (procedure)"
            payload["text"] = payload["display"]
    biopsy = copy.deepcopy(
        next(payload for payload in payloads if payload["evidence_id"].endswith("Procedure:biopsy"))
    )
    biopsy.update(
        {
            "evidence_id": "synthea:v1-patient:Procedure:biopsy-follow-up",
            "logical_evidence_group": ("record:synthea:v1-patient:Procedure:biopsy-follow-up"),
            "references": {"encounter": {"reference": "urn:uuid:enc-2026-follow-up"}},
            "time": {"start": "2026-06-15T10:21:00Z"},
        }
    )
    biopsy["source"] = {"resource_id": "biopsy-follow-up"}
    payloads.append(biopsy)
    payloads.extend(
        (
            _record(
                "Observation",
                "a1c-initial",
                LOINC,
                "4548-4",
                "Hemoglobin A1c",
                "2026-05-10T09:15:00Z",
                "enc-2026",
                value={"value": 6.1, "unit": "%"},
                status="final",
            ),
            _record(
                "Observation",
                "a1c-follow-up",
                LOINC,
                "4548-4",
                "Hemoglobin A1c",
                "2026-06-15T10:15:00Z",
                "enc-2026-follow-up",
                value={"value": 6.2, "unit": "%"},
                status="final",
            ),
            _record(
                "Observation",
                "glucose-follow-up",
                LOINC,
                "2339-0",
                "Blood glucose",
                "2026-06-15T10:16:00Z",
                "enc-2026-follow-up",
                value={"value": 105, "unit": "mg/dL"},
                status="final",
            ),
            _record(
                "CarePlan",
                "hypertension-plan",
                SNOMED,
                "443402002",
                "Lifestyle education regarding hypertension",
                "2026-05-10T09:35:00Z",
                "enc-2026",
                status="active",
                attributes={"intent": "order"},
                extra_references={"addresses": {"reference": "Condition/hypertension"}},
            ),
            _record(
                "Immunization",
                "flu-initial",
                CVX,
                "140",
                "Influenza, split virus, trivalent, PF",
                "2026-05-10T09:45:00Z",
                "enc-2026",
                status="completed",
            ),
            _record(
                "Immunization",
                "flu-follow-up",
                CVX,
                "140",
                "Influenza, split virus, trivalent, PF",
                "2026-06-15T10:45:00Z",
                "enc-2026-follow-up",
                status="completed",
            ),
            _record(
                "Immunization",
                "td-follow-up",
                CVX,
                "113",
                "Td adult immunization",
                "2026-06-15T10:46:00Z",
                "enc-2026-follow-up",
                status="completed",
            ),
        )
    )
    return payloads


def _record(
    resource_type: str,
    resource_id: str,
    system: str,
    code: str,
    display: str,
    start: str,
    encounter_id: str,
    *,
    value=None,
    status: str | None = None,
    attributes: dict | None = None,
    extra_references: dict | None = None,
) -> dict:
    evidence_id = f"synthea:v1-patient:{resource_type}:{resource_id}"
    return {
        "evidence_id": evidence_id,
        "logical_evidence_group": f"record:{evidence_id}",
        "patient_id": "v1-patient",
        "resource_type": resource_type,
        "source_dataset": "fabricated-v1",
        "category": resource_type.casefold(),
        "code": {"system": system, "code": code, "display": display},
        "display": display,
        "status": status,
        "attributes": attributes or {},
        "references": {
            "encounter": {"reference": f"urn:uuid:{encounter_id}"},
            **(extra_references or {}),
        },
        "time": {"start": start},
        "value": value if value is not None else status,
        "source": {"resource_id": resource_id},
    }


def _result(payloads: list[dict] | None = None):
    snapshot = snapshot_from_payloads(payloads or _base_payloads())
    return snapshot, generate_patient_p2_tasks((snapshot,))


def _tasks_by_domain(result):
    grouped = {}
    for task in result.tasks:
        domain = task.metadata["lineage"]["domain"]
        grouped.setdefault(domain, {})[task.construction_type] = task
    return grouped


def test_p2_emits_one_verified_atomic_family_for_every_required_domain():
    _, result = _result()
    grouped = _tasks_by_domain(result)

    assert set(grouped) == {spec.domain for spec in P2_DOMAIN_SPECS}
    assert result.group_count == 6
    assert all(set(tasks).issuperset({"atomic"}) for tasks in grouped.values())
    assert all(task.verification.passed for task in result.tasks)
    assert all(task.gold_answer.value is True for task in result.tasks)
    assert all(
        task.difficulty.dependency_depth == 1
        for task in (tasks["atomic"] for tasks in grouped.values())
    )
    assert all(item.generated_group_count == 1 for item in result.coverage)
    assert result.partial_group_count >= 1


def test_p2_compositions_are_relation_bound_independent_and_auditable():
    snapshot, result = _result()
    grouped = _tasks_by_domain(result)

    lab_depth = grouped["lab_observation"]["depth"]
    assert [call.tool for call in lab_depth.reference_trace] == [
        "search_observations",
        "find_related_records",
        "find_related_records",
    ]
    assert lab_depth.difficulty.dependency_depth == 3
    assert lab_depth.difficulty.evidence_width == 3
    assert "encounter that also documents essential hypertension" in lab_depth.question

    care_depth = grouped["care_plan"]["depth"]
    assert [call.tool for call in care_depth.reference_trace] == [
        "search_care_plans",
        "find_related_records",
    ]
    assert care_depth.reference_trace[1].resolved_arguments["relation"] == "addresses"
    assert care_depth.difficulty.dependency_depth == 2

    lab_width = grouped["lab_observation"]["width"]
    assert lab_width.difficulty.independent_width == 2
    assert len(lab_width.gold_evidence_ids) == 2
    records = [snapshot.record_by_id(item) for item in lab_width.gold_evidence_ids]
    assert len({record.logical_evidence_group for record in records if record}) == 2
    lab_coverage = next(item for item in result.coverage if item.domain == "lab_observation")
    assert lab_coverage.width_scope == "cross_domain"
    assert lab_coverage.width_companion_domain == "condition"
    assert lab_coverage.explicit_relation == "observation→encounter→condition"


def test_p2_is_invariant_to_input_order_and_contract_tampering():
    payloads = _base_payloads()
    original, result = _result(payloads)
    reordered, reordered_result = _result(list(reversed(payloads)))

    assert original.snapshot_id == reordered.snapshot_id
    assert result == reordered_result

    atomic = _tasks_by_domain(result)["immunization"]["atomic"]
    changed = atomic.model_copy(update={"question": "Did I receive a vaccine?"})
    report = verify_task_artifact(changed, ToolRegistry(original))
    assert report.passed is False
    assert "patient_p2_contract_invalid" in report.errors


def test_p2_skips_ambiguous_dates_and_rejects_unsafe_encounter_depth():
    payloads = _base_payloads()
    duplicate = copy.deepcopy(
        next(item for item in payloads if item["evidence_id"].endswith("a1c-follow-up"))
    )
    duplicate["evidence_id"] = "synthea:v1-patient:Observation:a1c-duplicate"
    duplicate["logical_evidence_group"] = "record:synthea:v1-patient:Observation:a1c-duplicate"
    duplicate["source"] = {"resource_id": "a1c-duplicate"}
    payloads.append(duplicate)

    cross_date = copy.deepcopy(
        next(item for item in payloads if item["evidence_id"].endswith("Procedure:colonoscopy"))
    )
    cross_date["evidence_id"] = "synthea:v1-patient:Procedure:cross-date-colonoscopy"
    cross_date["logical_evidence_group"] = (
        "record:synthea:v1-patient:Procedure:cross-date-colonoscopy"
    )
    cross_date["time"] = {"start": "2024-01-01T00:00:00Z"}
    cross_date["references"] = {"encounter": {"reference": "urn:uuid:enc-2026-follow-up"}}
    cross_date["source"] = {"resource_id": "cross-date-colonoscopy"}
    payloads.append(cross_date)

    _, result = _result(payloads)
    grouped = _tasks_by_domain(result)

    lab_atomic = grouped["lab_observation"]["atomic"]
    assert "blood glucose result" in lab_atomic.question
    assert "hemoglobin A1c result" not in lab_atomic.question
    assert "depth" not in grouped["procedure"]
    procedure_rejections = [
        item for item in result.rejected if item.domain == "procedure" and item.stage == "depth"
    ]
    assert [item.reason for item in procedure_rejections] == ["NO_SAFE_DEPTH"]


def test_p2_reports_unsupported_capability_and_pilot_yield_by_domain():
    payloads = [item for item in _base_payloads() if item["resource_type"] != "Immunization"]
    _, result = _result(payloads)
    summary = summarize_patient_p2_yield(result)

    immunization = next(item for item in result.coverage if item.domain == "immunization")
    assert immunization.capability_supported is False
    assert immunization.generated_group_count == 0
    unsupported = [
        item
        for item in result.rejected
        if item.domain == "immunization" and item.reason == "UNSUPPORTED_CAPABILITY"
    ]
    assert len(unsupported) == 1

    assert summary.patient_count == 1
    assert summary.generated_group_count == 5
    assert summary.complete_family_count < summary.generated_group_count
    assert summary.core_depth2_family_count <= summary.complete_family_count
    assert summary.partial_group_count >= 1
    assert set(summary.construction_counts).issuperset({"atomic", "depth", "width"})
    assert any(key.startswith("depth=3") for key in summary.difficulty_counts)
    domain_yields = {item.domain: item for item in summary.domain_yields}
    assert set(domain_yields) == {spec.domain for spec in P2_DOMAIN_SPECS}
    assert domain_yields["immunization"].supported_patients == 0
    assert summary.rejection_counts["UNSUPPORTED_CAPABILITY"] == 1


def test_p2_programs_use_only_patient_scoped_read_tools():
    _, result = _result()

    assert not any(
        node.tool.startswith(("create", "update", "delete", "write"))
        for task in result.tasks
        for node in task.program.nodes
        if isinstance(node, ToolNode)
    )


def test_p2_reviewed_language_uses_natural_articles():
    ldl = p2.get_p2_query("lab_observation:ldl_cholesterol")
    inpatient = p2.get_p2_query("care_plan:inpatient")
    antenatal = p2.get_p2_query("care_plan:antenatal")
    dental_report = p2.get_p2_query("procedure:dental_consultation")

    assert ldl is not None
    assert inpatient is not None
    assert antenatal is not None
    assert dental_report is not None
    assert ldl.query.patient_phrase == "an LDL cholesterol result"
    assert inpatient.query.patient_phrase == "an inpatient care plan"
    assert antenatal.query.patient_phrase == "an antenatal care plan"
    assert dental_report.query.patient_phrase == "a completed dental consultation report"


def test_p2_accounting_distinguishes_groups_complete_families_and_depth3():
    _, result = _result()
    summary = summarize_patient_p2_yield(result)

    assert summary.generated_group_count == 6
    assert summary.complete_family_count == 1
    assert summary.core_depth2_family_count == 0
    assert summary.partial_group_count == 5
    assert (
        summary.complete_family_count + summary.partial_group_count == summary.generated_group_count
    )
    lab = next(item for item in result.coverage if item.domain == "lab_observation")
    assert lab.group_shape == "complete"
    assert lab.complete_family_count == 1
    assert lab.core_depth2_family_count == 0
    assert lab.depth_mode == "lab_condition"


def test_p2_prefers_complete_depth2_candidate_over_newer_depth3_candidate():
    payloads = _base_payloads()
    for payload in payloads:
        if payload["evidence_id"].endswith("Encounter:enc-2026"):
            payload["code"] = {
                "system": SNOMED,
                "code": "162673000",
                "display": "General examination of patient",
            }
            payload["display"] = "General examination of patient (procedure)"
            payload["text"] = payload["display"]
        if (
            payload["resource_type"] == "Condition"
            and payload.get("references", {}).get("encounter", {}).get("reference")
            == "urn:uuid:enc-2026"
        ):
            payload["code"] = {
                "system": SNOMED,
                "code": f"not-reviewed-{payload['evidence_id'].rsplit(':', 1)[-1]}",
                "display": "Not reviewed",
            }
    payloads.append(
        _record(
            "Observation",
            "glucose-initial",
            LOINC,
            "2339-0",
            "Blood glucose",
            "2026-05-10T09:16:00Z",
            "enc-2026",
            value={"value": 100, "unit": "mg/dL"},
            status="final",
        )
    )

    _, result = _result(payloads)
    lab = next(item for item in result.coverage if item.domain == "lab_observation")
    atomic = _tasks_by_domain(result)["lab_observation"]["atomic"]

    assert lab.selected_query_id == "lab_observation:blood_glucose"
    assert "May 10, 2026" in atomic.question
    assert lab.depth_mode == "encounter"
    assert lab.core_depth2_family_count == 1


def test_p2_invalid_top_candidate_is_audited_then_next_candidate_runs(monkeypatch):
    p2._DISCOVERY_CACHE.clear()
    p2._SEARCH_CACHE.clear()
    real_compile = p2.compile_patient_p2_family
    calls = 0

    def fail_first(snapshot, selected):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic candidate failure")
        return real_compile(snapshot, selected)

    monkeypatch.setattr(p2, "compile_patient_p2_family", fail_first)
    snapshot = snapshot_from_payloads(_base_payloads())
    result = generate_patient_p2_tasks((snapshot,))

    assert result.group_count == 6
    assert any(item.reason == "INCOMPLETE_SCOPE" for item in result.rejected)
    assert all(task.verification.passed for task in result.tasks)
    p2._DISCOVERY_CACHE.clear()
    p2._SEARCH_CACHE.clear()
