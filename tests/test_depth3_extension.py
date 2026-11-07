from patient_tool_gym.depth3_extension import (
    Depth3Candidate,
    build_depth3_program,
    build_depth3_question,
    verify_depth3_complexity,
)
from patient_tool_gym.program import ClinicalRecordSelector


def _candidate() -> Depth3Candidate:
    return Depth3Candidate(
        superfamily_id="family-1",
        patient_id="patient-1",
        base_bundle_path="bundles/patient-1.json",
        path_kind="diagnostic_report_result",
        encounter_scope="most_recent",
        encounter_resource_id="encounter-1",
        encounter_system="http://snomed.info/sct",
        encounter_code="162673000",
        encounter_display="General examination of patient",
        encounter_status="finished",
        encounter_instant="2026-01-01T09:00:00Z",
        intermediary_resource_type="DiagnosticReport",
        intermediary_resource_id="report-1",
        intermediary_system="http://loinc.org",
        intermediary_code="58410-2",
        intermediary_display="Complete blood count panel",
        intermediary_encounter_relation="encounter",
        target_resource_type="Observation",
        target_resource_id="result-1",
        target_system="http://loinc.org",
        target_code="789-8",
        target_display="Erythrocytes [#/volume] in Blood by Automated count",
        target_relation="result",
        target_direction="outgoing",
        decision_rule_id="observation_final_result.v2",
        patient_phrase="a red blood cell count",
        companion_domain="condition",
        companion_resource_type="Condition",
        companion_resource_id="condition-1",
        companion_phrase="hypertension",
        companion_selector=ClinicalRecordSelector(
            system="http://snomed.info/sct",
            code="38341003",
            date_from="2026-01-02",
            date_to="2026-01-02",
        ),
    )


def test_depth3_width_one_is_a_three_call_dependency_chain():
    program = build_depth3_program(_candidate(), "d3w1")
    complexity = verify_depth3_complexity(program, "d3w1")
    tools = [node.tool for node in program.nodes if hasattr(node, "tool")]

    assert tools == [
        "search_encounters",
        "find_related_records",
        "find_related_records",
    ]
    assert complexity.dependency_depth == 3
    assert complexity.independent_width == 1


def test_depth3_width_two_adds_one_independent_claim():
    program = build_depth3_program(_candidate(), "d3w2")
    complexity = verify_depth3_complexity(program, "d3w2")
    question = build_depth3_question(_candidate(), "d3w2")

    assert complexity.dependency_depth == 3
    assert complexity.independent_width == 2
    assert "most recent completed general examination visit" in question
    assert "branches.companion" in question
