"""Encounter-anchored depth-three tasks over explicit FHIR reference paths."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from patient_tool_gym.program import (
    ClinicalRecordSelector,
    EvidenceStateNode,
    ProgramComplexity,
    ProgramOutputSchema,
    StrictModel,
    TaskProgram,
    ToolNode,
    TruthStateAggregateNode,
    ValueRef,
    analyze_complexity,
    binding,
)

Depth3Topology = Literal["d3w1", "d3w2"]
Depth3PathKind = Literal[
    "diagnostic_report_result",
    "procedure_reason",
    "medication_reason",
]

_COMPANION_CONTRACTS = {
    "condition": ("search_conditions", "condition_documented.v2"),
    "lab_observation": ("search_observations", "observation_final_result.v2"),
    "immunization": ("search_immunizations", "immunization_completed.v2"),
}

_RESULT_PHRASES = {
    "2571-8": "a triglyceride result",
    "2746-6": "a venous blood pH result",
    "55758-7": "a PHQ-2 score",
    "5811-5": "a urine specific-gravity result",
    "6299-2": "a blood urea nitrogen result",
    "70274-6": "a GAD-7 score",
    "75626-2": "an AUDIT-C score",
    "76504-0": "a HARK score",
    "789-8": "a red blood cell count",
    "79819-9": "a left-retina OCT result",
    "8247-9": "a urine-mucus microscopy result",
    "82667-7": "a DAST-10 score",
    "89204-2": "a modified PHQ-9 teen score",
}


class Depth3Candidate(StrictModel):
    schema_version: Literal["patient-tool-gym.depth3-candidate.v1"] = (
        "patient-tool-gym.depth3-candidate.v1"
    )
    superfamily_id: str
    patient_id: str
    base_bundle_path: str
    path_kind: Depth3PathKind
    encounter_scope: Literal["most_recent", "exact_date"]
    encounter_resource_id: str
    encounter_system: str
    encounter_code: str
    encounter_display: str
    encounter_status: Literal["finished"]
    encounter_instant: str
    intermediary_resource_type: Literal[
        "DiagnosticReport",
        "Procedure",
        "MedicationRequest",
    ]
    intermediary_resource_id: str
    intermediary_system: str
    intermediary_code: str
    intermediary_display: str
    intermediary_encounter_relation: Literal["encounter"]
    target_resource_type: Literal["Observation", "Condition"]
    target_resource_id: str
    target_system: str
    target_code: str
    target_display: str
    target_relation: Literal["result", "reasonReference"]
    target_direction: Literal["outgoing"]
    decision_rule_id: Literal[
        "observation_final_result.v2",
        "condition_documented.v2",
    ]
    patient_phrase: str
    companion_domain: Literal[
        "condition",
        "lab_observation",
        "immunization",
    ]
    companion_resource_type: Literal["Condition", "Observation", "Immunization"]
    companion_resource_id: str
    companion_phrase: str
    companion_selector: ClinicalRecordSelector


def build_depth3_program(
    candidate: Depth3Candidate,
    topology: Depth3Topology,
) -> TaskProgram:
    encounter_arguments: dict[str, object] = {
        "text": candidate.encounter_display,
        "status": candidate.encounter_status,
        "system": candidate.encounter_system,
        "code": candidate.encounter_code,
        "limit": 100,
    }
    if candidate.encounter_scope == "exact_date":
        encounter_date = candidate.encounter_instant[:10]
        date.fromisoformat(encounter_date)
        encounter_arguments.update(
            {
                "date_from": encounter_date,
                "date_to": encounter_date,
            }
        )
    nodes: list[object] = [
        ToolNode(
            id="encounter_search",
            tool="search_encounters",
            arguments=encounter_arguments,
        ),
        ToolNode(
            id="intermediary_search",
            tool="find_related_records",
            arguments={
                "anchor_evidence_id": binding(
                    "encounter_search",
                    "records",
                    0,
                    "evidence_id",
                ),
                "relation": candidate.intermediary_encounter_relation,
                "direction": "incoming",
                "resource_type": candidate.intermediary_resource_type,
                "system": candidate.intermediary_system,
                "code": candidate.intermediary_code,
                "limit": 100,
            },
        ),
        ToolNode(
            id="target_search",
            tool="find_related_records",
            arguments={
                "anchor_evidence_id": binding(
                    "intermediary_search",
                    "records",
                    0,
                    "evidence_id",
                ),
                "relation": candidate.target_relation,
                "direction": candidate.target_direction,
                "resource_type": candidate.target_resource_type,
                "limit": 100,
            },
        ),
        EvidenceStateNode(
            id="primary_state",
            source=ValueRef(node_id="target_search"),
            decision_rule_id=candidate.decision_rule_id,
            selector=ClinicalRecordSelector(
                system=candidate.target_system,
                code=candidate.target_code,
            ),
        ),
    ]
    branches = {"primary": ValueRef(node_id="primary_state")}
    if topology == "d3w2":
        companion_tool, companion_rule = _COMPANION_CONTRACTS[candidate.companion_domain]
        selector = candidate.companion_selector
        nodes.extend(
            (
                ToolNode(
                    id="companion_search",
                    tool=companion_tool,
                    arguments={
                        "system": selector.system,
                        "date_from": (
                            selector.date_from.isoformat()
                            if selector.date_from is not None
                            else None
                        ),
                        "date_to": (
                            selector.date_to.isoformat() if selector.date_to is not None else None
                        ),
                        "limit": 100,
                    },
                ),
                EvidenceStateNode(
                    id="companion_state",
                    source=ValueRef(node_id="companion_search"),
                    decision_rule_id=companion_rule,
                    selector=selector,
                ),
            )
        )
        branches["companion"] = ValueRef(node_id="companion_state")
    nodes.append(TruthStateAggregateNode(id="answer", branches=branches))
    program = TaskProgram(
        schema_version="patient-tool-gym.program.v4",
        nodes=tuple(nodes),
        output=ValueRef(node_id="answer"),
        output_schema=_aggregate_output_schema(tuple(branches)),
    )
    verify_depth3_complexity(program, topology)
    return program


def build_depth3_question(
    candidate: Depth3Candidate,
    topology: Depth3Topology,
) -> str:
    encounter = _encounter_phrase(candidate.encounter_display)
    scope = (
        f"my completed {encounter} on {_render_date(candidate.encounter_instant[:10])}"
        if candidate.encounter_scope == "exact_date"
        else f"my most recent completed {encounter}"
    )
    target = _target_phrase(candidate)
    intermediary = _clean_display(candidate.intermediary_display)
    if candidate.path_kind == "diagnostic_report_result":
        primary = f"the {intermediary} from {scope} includes {target}"
    elif candidate.path_kind == "procedure_reason":
        primary = (
            f"the {intermediary} at {scope} lists a confirmed diagnosis of {target} as a reason"
        )
    else:
        primary = (
            f"the {intermediary} order at {scope} lists a confirmed diagnosis of "
            f"{target} as a reason"
        )
    claims = f"First claim (return this as branches.primary): my chart shows {primary}."
    if topology == "d3w2":
        selector = candidate.companion_selector
        claims += (
            " Second claim (return this as branches.companion): my chart shows "
            f"{candidate.companion_phrase} "
            f"({_vocabulary_label(selector.system)} {selector.code}) on "
            f"{_render_date(selector.date_from.isoformat())}."
        )
    fields = (
        "branches.primary and overall"
        if topology == "d3w1"
        else "branches.primary, branches.companion, and overall"
    )
    return (
        f"Using my chart, classify these claims. {claims} Return a JSON object with the "
        f"evidence states for {fields}. For each state, choose supported, refuted, "
        "insufficient evidence, or conflicting evidence."
    )


def verify_depth3_complexity(
    program: TaskProgram,
    topology: Depth3Topology,
) -> ProgramComplexity:
    complexity = analyze_complexity(program, frozenset(), None)
    expected = (3, 1, 1) if topology == "d3w1" else (3, 2, 2)
    observed = (
        complexity.dependency_depth,
        complexity.independent_width,
        complexity.answer_width,
    )
    if observed != expected:
        raise ValueError(f"{topology} expected complexity {expected}, got {observed}")
    return complexity


def _aggregate_output_schema(branch_names: tuple[str, ...]) -> ProgramOutputSchema:
    return ProgramOutputSchema(
        kind="object",
        fields={
            "branches": ProgramOutputSchema(
                kind="object",
                fields={
                    name: ProgramOutputSchema(kind="truth_state") for name in sorted(branch_names)
                },
            ),
            "overall": ProgramOutputSchema(kind="truth_state"),
        },
    )


def _target_phrase(candidate: Depth3Candidate) -> str:
    phrase = _RESULT_PHRASES.get(candidate.target_code)
    if phrase is None:
        phrase = candidate.patient_phrase.strip()
        if phrase and phrase[0].isupper():
            phrase = phrase[0].lower() + phrase[1:]
        if not phrase.startswith(("a ", "an ", "the ")):
            phrase = f"{phrase}"
    return f"{phrase} ({_vocabulary_label(candidate.target_system)} {candidate.target_code})"


def _encounter_phrase(display: str) -> str:
    cleaned = _clean_display(display).casefold()
    replacements = {
        "encounter for check up": "check-up visit",
        "encounter for problem": "problem visit",
        "encounter for symptom": "symptom visit",
        "follow-up encounter": "follow-up visit",
        "general examination of patient": "general examination visit",
        "urgent care clinic": "urgent-care visit",
    }
    cleaned = replacements.get(cleaned, cleaned)
    if not cleaned.endswith(("visit", "admission", "stay", "consultation", "encounter")):
        cleaned = f"{cleaned} visit"
    return cleaned


def _clean_display(display: str) -> str:
    return re.sub(
        r"\s+\((?:procedure|disorder|finding|regime/therapy|environment)\)$",
        "",
        display.strip(),
        flags=re.IGNORECASE,
    )


def _render_date(value: str) -> str:
    parsed = date.fromisoformat(value)
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"


def _vocabulary_label(system: str) -> str:
    return {
        "http://loinc.org": "LOINC",
        "http://snomed.info/sct": "SNOMED CT",
        "http://hl7.org/fhir/sid/cvx": "CVX",
    }.get(system, system)
