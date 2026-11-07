from pathlib import Path

import pytest
from pydantic import ValidationError

import patient_tool_gym.verifier as verifier_module
from patient_tool_gym.evidence import (
    LEGACY_LOGICAL_GROUP_RULES_VERSION,
    load_snapshot,
    snapshot_from_payloads,
)
from patient_tool_gym.lifecycle import lifecycle_polarity
from patient_tool_gym.program import (
    MEDICATION_DECISION_RULE_SEMANTICS,
    AggregateNode,
    BuildObjectNode,
    ClinicalRecordSelector,
    EvidenceStateNode,
    ProgramExecutionError,
    ProgramExecutor,
    ProgramOutputSchema,
    SelectNode,
    TaskProgram,
    ToolNode,
    TruthState,
    TruthStateAggregateNode,
    TypedValue,
    ValueRef,
    VersionTrajectoryNode,
    _classify_evidence_record,
    content_hash,
    infer_program_output_schema,
    program_signature,
    program_submission_answer_schema,
    select_clinical_records,
    truth_state_all,
)
from patient_tool_gym.task import TaskArtifact, deterministic_task_id
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import compile_task, verify_task_artifact

FIXTURE = Path(__file__).parent / "fixtures" / "patient_a.jsonl"
SNOMED = "http://snomed.info/sct"
CVX = "http://hl7.org/fhir/sid/cvx"


def latest_weight_program() -> TaskProgram:
    return TaskProgram(
        nodes=(
            ToolNode(
                id="weights",
                tool="search_observations",
                arguments={"code": "29463-7", "limit": 100},
            ),
            SelectNode(
                id="latest_record",
                source=ValueRef(node_id="weights", path=("records", 0)),
            ),
            SelectNode(
                id="answer",
                source=ValueRef(node_id="latest_record", path=("value",)),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def _concept(system: str, code: str, display: str) -> dict[str, object]:
    coding = {"system": system, "code": code, "display": display}
    return {**coding, "codings": [coding]}


def _evidence_state_registry(
    resource_type: str,
    state: str,
    *,
    duplicate: bool = False,
) -> tuple[ToolRegistry, str, str, str]:
    tool, rule, system, code = {
        "Condition": (
            "search_conditions",
            "condition_documented.v1",
            SNOMED,
            "59621000",
        ),
        "Procedure": (
            "search_procedures",
            "procedure_completed.v1",
            SNOMED,
            "73761001",
        ),
        "Immunization": (
            "search_immunizations",
            "immunization_completed.v1",
            CVX,
            "140",
        ),
    }[resource_type]
    payloads = [
        {
            "category": "demographics",
            "code": {},
            "display": "State Patient",
            "evidence_id": "fixture:state-patient:Patient:state-patient",
            "patient_id": "state-patient",
            "references": {},
            "resource_type": "Patient",
            "source": {"resource_id": "state-patient"},
        }
    ]
    for index in range(2 if duplicate else 1):
        resource_id = f"target-{index}"
        evidence_id = f"fixture:state-patient:{resource_type}:{resource_id}"
        payload = {
            "category": resource_type.casefold(),
            "code": _concept(system, code, "Target concept"),
            "display": "Target concept",
            "evidence_id": evidence_id,
            "logical_evidence_group": f"record:{evidence_id}",
            "patient_id": "state-patient",
            "references": {},
            "resource_type": resource_type,
            "source": {"resource_id": resource_id},
            "text": f"Target concept {state}",
            "time": {"start": f"2026-01-0{index + 1}T09:00:00Z"},
            "value": state,
        }
        if resource_type == "Condition":
            payload["attributes"] = {
                "verificationStatus": _concept(
                    "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    state,
                    state.replace("-", " ").title(),
                )
            }
        else:
            payload["status"] = state
        payloads.append(payload)
    return ToolRegistry(snapshot_from_payloads(payloads)), tool, rule, code


def _evidence_state_program(tool: str, rule: str, code: str) -> TaskProgram:
    system = CVX if tool == "search_immunizations" else SNOMED
    return TaskProgram(
        schema_version="patient-tool-gym.program.v2",
        nodes=(
            ToolNode(
                id="search",
                tool=tool,
                arguments={"system": system, "code": code, "limit": 100},
            ),
            EvidenceStateNode(
                id="answer",
                source=ValueRef(node_id="search"),
                decision_rule_id=rule,
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def _selected_evidence_state_program(
    tool: str,
    rule: str,
    code: str,
    *,
    date_from: str = "2026-01-01",
    date_to: str = "2026-01-01",
    search_limit: int = 100,
) -> TaskProgram:
    system = CVX if tool == "search_immunizations" else SNOMED
    return TaskProgram(
        schema_version="patient-tool-gym.program.v3",
        nodes=(
            ToolNode(
                id="broad_search",
                tool=tool,
                arguments={"system": system, "limit": search_limit},
            ),
            EvidenceStateNode(
                id="answer",
                source=ValueRef(node_id="broad_search"),
                decision_rule_id=rule.replace(".v1", ".v2"),
                selector=ClinicalRecordSelector(
                    system=system,
                    code=code,
                    date_from=date_from,
                    date_to=date_to,
                ),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


_BELNAP_ALL = {
    "supported": {
        "supported": "supported",
        "refuted": "refuted",
        "insufficient_evidence": "insufficient_evidence",
        "conflicting_evidence": "conflicting_evidence",
    },
    "refuted": {
        "supported": "refuted",
        "refuted": "refuted",
        "insufficient_evidence": "refuted",
        "conflicting_evidence": "refuted",
    },
    "insufficient_evidence": {
        "supported": "insufficient_evidence",
        "refuted": "refuted",
        "insufficient_evidence": "insufficient_evidence",
        "conflicting_evidence": "refuted",
    },
    "conflicting_evidence": {
        "supported": "conflicting_evidence",
        "refuted": "refuted",
        "insufficient_evidence": "refuted",
        "conflicting_evidence": "conflicting_evidence",
    },
}


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry(load_snapshot(FIXTURE))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (left, right, expected)
        for left, row in _BELNAP_ALL.items()
        for right, expected in row.items()
    ],
)
def test_truth_state_all_implements_symmetric_belnap_conjunction(left, right, expected):
    result = truth_state_all((TruthState(left), TruthState(right)))

    assert result == TruthState(expected)
    assert result == truth_state_all((TruthState(right), TruthState(left)))


def test_truth_state_aggregate_has_structured_branch_schema_and_width():
    registry, tool, rule, code = _evidence_state_registry("Condition", "confirmed")
    selector = ClinicalRecordSelector(system=SNOMED, code=code)
    program = TaskProgram(
        schema_version="patient-tool-gym.program.v4",
        nodes=(
            ToolNode(
                id="left_search",
                tool=tool,
                arguments={"system": SNOMED, "code": code, "limit": 100},
            ),
            EvidenceStateNode(
                id="left_state",
                source=ValueRef(node_id="left_search"),
                decision_rule_id=rule.replace(".v1", ".v2"),
                selector=selector,
            ),
            ToolNode(
                id="right_search",
                tool=tool,
                arguments={"system": SNOMED, "code": code, "limit": 100},
            ),
            EvidenceStateNode(
                id="right_state",
                source=ValueRef(node_id="right_search"),
                decision_rule_id=rule.replace(".v1", ".v2"),
                selector=selector,
            ),
            TruthStateAggregateNode(
                id="answer",
                branches={
                    "primary": ValueRef(node_id="left_state"),
                    "companion": ValueRef(node_id="right_state"),
                },
            ),
        ),
        output=ValueRef(node_id="answer"),
    )

    result = ProgramExecutor(registry).execute(program)
    schema = infer_program_output_schema(program)
    artifact = compile_task(
        registry=registry,
        program=program,
        family_id="typed-state-width-family",
        construction_type="width",
        question=(
            "For both Target concept branches, report supported, refuted, insufficient "
            "evidence, or conflicting evidence."
        ),
        evidence_quality_policy="exact_witness_v1",
    )

    assert result.answer == TypedValue(
        kind="object",
        value={
            "branches": {
                "companion": TruthState.SUPPORTED,
                "primary": TruthState.SUPPORTED,
            },
            "overall": TruthState.SUPPORTED,
        },
    )
    assert result.complexity.dependency_depth == 1
    assert result.complexity.independent_width == 2
    assert result.complexity.answer_width == 2
    assert schema.fields["overall"].kind == "truth_state"
    assert schema.fields["branches"].fields == {
        "companion": ProgramOutputSchema(kind="truth_state"),
        "primary": ProgramOutputSchema(kind="truth_state"),
    }
    assert artifact.verification.passed is True
    assert len(artifact.evidence_quality_contract.complete_search_requirements) == 2


def test_atomic_program_executes_with_typed_answer_evidence_and_trace(registry):
    result = ProgramExecutor(registry).execute(latest_weight_program())

    assert result.answer.kind == "string"
    assert result.answer.value == "62 kg"
    assert result.evidence_ids == ("synthea:patient-a:Observation:weight-new",)
    assert [call.tool for call in result.reference_trace] == ["search_observations"]
    assert result.reference_trace[0].resolved_arguments["patient_id"] == "patient-a"
    assert result.complexity.dependency_depth == 1
    assert result.complexity.independent_width == 1


@pytest.mark.parametrize(
    ("resource_type", "record_state", "expected"),
    [
        ("Condition", "confirmed", "supported"),
        ("Condition", "refuted", "refuted"),
        ("Condition", "unconfirmed", "insufficient_evidence"),
        ("Procedure", "completed", "supported"),
        ("Procedure", "not-done", "refuted"),
        ("Procedure", "entered-in-error", "insufficient_evidence"),
        ("Immunization", "completed", "supported"),
        ("Immunization", "not-done", "refuted"),
        ("Immunization", "entered-in-error", "insufficient_evidence"),
    ],
)
def test_evidence_state_node_classifies_resource_specific_states(
    resource_type,
    record_state,
    expected,
):
    state_registry, tool, rule, code = _evidence_state_registry(resource_type, record_state)

    result = ProgramExecutor(state_registry).execute(_evidence_state_program(tool, rule, code))

    assert result.answer == TypedValue(kind="truth_state", value=expected)
    assert len(result.evidence_ids) == 1
    assert result.reference_trace[0].complete is True


@pytest.mark.parametrize(
    ("rule_id", "resource_type", "status", "has_value"),
    [
        ("condition_documented.v2", "Condition", "confirmed", True),
        ("condition_documented.v2", "Condition", "refuted", True),
        ("condition_documented.v2", "Condition", "unconfirmed", True),
        ("observation_final_result.v2", "Observation", "final", True),
        ("observation_final_result.v2", "Observation", "amended", True),
        ("observation_final_result.v2", "Observation", "corrected", True),
        ("observation_final_result.v2", "Observation", "final", False),
        ("observation_final_result.v2", "Observation", "cancelled", True),
        ("observation_final_result.v2", "Observation", "registered", True),
        ("observation_final_result.v2", "Observation", "preliminary", True),
        ("observation_final_result.v2", "Observation", "entered-in-error", True),
        ("observation_final_result.v2", "Observation", "unknown", True),
        ("procedure_completed.v2", "Procedure", "completed", True),
        ("procedure_completed.v2", "Procedure", "not-done", True),
        ("procedure_completed.v2", "Procedure", "stopped", True),
        ("procedure_completed.v2", "Procedure", "entered-in-error", True),
        ("immunization_completed.v2", "Immunization", "completed", True),
        ("immunization_completed.v2", "Immunization", "not-done", True),
        ("immunization_completed.v2", "Immunization", "stopped", True),
        ("immunization_completed.v2", "Immunization", "entered-in-error", True),
        ("medication_order_actionable.v3", "MedicationRequest", "active", True),
        ("medication_order_actionable.v3", "MedicationRequest", "cancelled", True),
        ("medication_order_actionable.v3", "MedicationRequest", "completed", True),
        ("medication_order_actionable.v3", "MedicationRequest", "stopped", True),
        ("medication_order_actionable.v3", "MedicationRequest", "draft", True),
        ("medication_order_actionable.v3", "MedicationRequest", "on-hold", True),
        ("medication_order_actionable.v3", "MedicationRequest", "revoked", True),
        ("medication_administered.v2", "MedicationAdministration", "completed", True),
        ("medication_administered.v2", "MedicationAdministration", "stopped", True),
        ("medication_administered.v2", "MedicationAdministration", "not-done", True),
        ("medication_administered.v2", "MedicationAdministration", "on-hold", True),
        (
            "medication_administered.v2",
            "MedicationAdministration",
            "entered-in-error",
            True,
        ),
    ],
)
def test_diagnostic_lifecycle_polarity_matches_executable_rules(
    rule_id,
    resource_type,
    status,
    has_value,
):
    record = {
        "resource_type": resource_type,
        "status": status,
        "value": "present" if has_value else None,
        "attributes": (
            {"verificationStatus": {"code": status}}
            if resource_type == "Condition"
            else {}
        ),
    }

    executable = _classify_evidence_record(rule_id, record).value
    diagnostic = lifecycle_polarity(resource_type, status, has_value=has_value)

    assert diagnostic == executable


def test_legacy_medication_order_rule_is_replay_only_not_historical_authorship(registry):
    semantics = MEDICATION_DECISION_RULE_SEMANTICS["medication_ordered.v2"]
    assert semantics.endpoint == "legacy_current_lifecycle_not_historical_order_authorship"
    assert semantics.deprecated_for_new_tasks is True
    assert semantics.replacement_rule_id == "medication_order_actionable.v3"
    program = TaskProgram(
        schema_version="patient-tool-gym.program.v3",
        nodes=(
            ToolNode(id="records", tool="search_medications", arguments={"limit": 100}),
            EvidenceStateNode(
                id="answer",
                source=ValueRef(node_id="records"),
                decision_rule_id="medication_ordered.v2",
                selector=ClinicalRecordSelector(
                    system="http://www.nlm.nih.gov/research/umls/rxnorm",
                    code="123",
                ),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )

    with pytest.raises(ValueError, match="replay-only and cannot compile new tasks"):
        compile_task(
            registry=registry,
            program=program,
            family_id="deprecated-medication-order-rule",
            construction_type="atomic",
            question="Was medication 123 historically ordered?",
        )


@pytest.mark.parametrize(
    ("resource_type", "record_state", "expected"),
    [
        ("Condition", "confirmed", "supported"),
        ("Condition", "refuted", "refuted"),
        ("Condition", "unconfirmed", "insufficient_evidence"),
        ("Procedure", "completed", "supported"),
        ("Procedure", "not-done", "refuted"),
        ("Procedure", "entered-in-error", "insufficient_evidence"),
        ("Immunization", "completed", "supported"),
        ("Immunization", "not-done", "refuted"),
        ("Immunization", "entered-in-error", "insufficient_evidence"),
    ],
)
def test_v2_selector_classifies_resource_specific_states(
    resource_type,
    record_state,
    expected,
):
    state_registry, tool, rule, code = _evidence_state_registry(resource_type, record_state)

    result = ProgramExecutor(state_registry).execute(
        _selected_evidence_state_program(tool, rule, code)
    )

    assert result.answer == TypedValue(kind="truth_state", value=expected)
    assert len(result.evidence_ids) == 1


def test_evidence_state_empty_complete_search_is_insufficient_evidence():
    state_registry, tool, rule, _ = _evidence_state_registry("Condition", "confirmed")

    result = ProgramExecutor(state_registry).execute(
        _evidence_state_program(tool, rule, "missing-code")
    )

    assert result.answer == TypedValue(kind="truth_state", value="insufficient_evidence")
    assert result.evidence_ids == ()


def test_evidence_state_aggregates_multiple_supporting_records():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Procedure", "completed", duplicate=True
    )

    result = ProgramExecutor(state_registry).execute(_evidence_state_program(tool, rule, code))

    assert result.answer == TypedValue(kind="truth_state", value="supported")
    assert len(result.evidence_ids) == 2


def test_v2_selector_filters_broad_retrieval_and_narrows_gold_evidence():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Procedure", "completed", duplicate=True
    )

    result = ProgramExecutor(state_registry).execute(
        _selected_evidence_state_program(tool, rule, code)
    )

    assert result.answer == TypedValue(kind="truth_state", value="supported")
    assert result.evidence_ids == ("fixture:state-patient:Procedure:target-0",)
    assert result.reference_trace[0].evidence_ids == (
        "fixture:state-patient:Procedure:target-0",
        "fixture:state-patient:Procedure:target-1",
    )
    assert result.reference_trace[0].complete is True


def test_v2_selector_no_match_is_insufficient_without_broad_evidence():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Condition", "confirmed", duplicate=True
    )

    result = ProgramExecutor(state_registry).execute(
        _selected_evidence_state_program(
            tool,
            rule,
            code,
            date_from="2027-01-01",
            date_to="2027-01-01",
        )
    )

    assert result.answer == TypedValue(kind="truth_state", value="insufficient_evidence")
    assert result.evidence_ids == ()
    assert len(result.reference_trace[0].evidence_ids) == 2


def test_v2_selector_aggregates_multiple_selected_records():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Immunization", "completed", duplicate=True
    )

    result = ProgramExecutor(state_registry).execute(
        _selected_evidence_state_program(
            tool,
            rule,
            code,
            date_from="2026-01-01",
            date_to="2026-01-02",
        )
    )

    assert result.answer == TypedValue(kind="truth_state", value="supported")
    assert len(result.evidence_ids) == 2


def test_plural_evidence_reports_conflict_when_support_and_refutation_coexist():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Procedure", "completed", duplicate=True
    )
    payloads = [record.payload.copy() for record in state_registry.snapshot.records]
    payloads[1] = {
        **payloads[1],
        "status": "not-done",
        "value": "not-done",
        "text": "Target concept not-done",
        "logical_evidence_group": "clinical-identifier:procedure-event-1",
    }
    payloads[2] = {
        **payloads[2],
        "logical_evidence_group": "clinical-identifier:procedure-event-1",
    }
    conflict_registry = ToolRegistry(snapshot_from_payloads(payloads))

    result = ProgramExecutor(conflict_registry).execute(
        _selected_evidence_state_program(
            tool,
            rule,
            code,
            date_from="2026-01-01",
            date_to="2026-01-02",
        )
    )

    assert result.answer == TypedValue(kind="truth_state", value="conflicting_evidence")
    assert len(result.evidence_ids) == 2


def test_plural_evidence_does_not_call_distinct_attempts_conflicting():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Procedure", "completed", duplicate=True
    )
    payloads = [record.payload.copy() for record in state_registry.snapshot.records]
    payloads[1] = {
        **payloads[1],
        "status": "not-done",
        "value": "not-done",
        "text": "Separate procedure attempt not-done",
    }
    distinct_attempt_registry = ToolRegistry(snapshot_from_payloads(payloads))

    result = ProgramExecutor(distinct_attempt_registry).execute(
        _selected_evidence_state_program(
            tool,
            rule,
            code,
            date_from="2026-01-01",
            date_to="2026-01-02",
        )
    )

    assert result.answer == TypedValue(kind="truth_state", value="supported")


def test_frozen_v2_logical_group_contract_replays_legacy_global_conflict():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Procedure", "completed", duplicate=True
    )
    payloads = [record.payload.copy() for record in state_registry.snapshot.records]
    payloads[1] = {
        **payloads[1],
        "status": "not-done",
        "value": "not-done",
        "text": "Legacy separate record interpreted under v2",
    }
    legacy_registry = ToolRegistry(
        snapshot_from_payloads(
            payloads,
            logical_group_rules_version=LEGACY_LOGICAL_GROUP_RULES_VERSION,
        )
    )

    result = ProgramExecutor(legacy_registry).execute(
        _selected_evidence_state_program(
            tool,
            rule,
            code,
            date_from="2026-01-01",
            date_to="2026-01-02",
        )
    )

    assert result.answer == TypedValue(kind="truth_state", value="conflicting_evidence")


def test_v2_selector_requires_complete_broad_search():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Procedure", "completed", duplicate=True
    )

    with pytest.raises(ProgramExecutionError, match="complete successful search"):
        ProgramExecutor(state_registry).execute(
            _selected_evidence_state_program(
                tool,
                rule,
                code,
                search_limit=1,
            )
        )


def test_select_clinical_records_uses_exact_coding_and_inclusive_dates():
    state_registry, tool, _, code = _evidence_state_registry(
        "Procedure", "completed", duplicate=True
    )
    result = state_registry.call(tool, {"system": SNOMED, "limit": 100})
    selector = ClinicalRecordSelector(
        system=SNOMED,
        code=code,
        date_from="2026-01-02",
        date_to="2026-01-02",
    )

    selected = select_clinical_records(result["records"], selector)

    assert [record["evidence_id"] for record in selected] == [
        "fixture:state-patient:Procedure:target-1"
    ]
    assert ClinicalRecordSelector.model_validate_json(selector.model_dump_json()) == selector
    with pytest.raises(ValidationError, match="date_from must not be after"):
        ClinicalRecordSelector(
            system=SNOMED,
            code=code,
            date_from="2026-01-03",
            date_to="2026-01-02",
        )


def test_evidence_state_rule_and_selector_versions_are_fail_closed():
    selector = ClinicalRecordSelector(system=SNOMED, code="73761001")
    with pytest.raises(ValidationError, match="v2 evidence-state.*require a selector"):
        EvidenceStateNode(
            id="answer",
            source=ValueRef(node_id="search"),
            decision_rule_id="procedure_completed.v2",
        )
    with pytest.raises(ValidationError, match="selectors require a v2"):
        EvidenceStateNode(
            id="answer",
            source=ValueRef(node_id="search"),
            decision_rule_id="procedure_completed.v1",
            selector=selector,
        )

    state_registry, tool, rule, code = _evidence_state_registry("Procedure", "completed")
    program = _selected_evidence_state_program(tool, rule, code)
    with pytest.raises(ValidationError, match="v2/v3 evidence-state.*program.v3"):
        TaskProgram.model_validate(
            {
                **program.model_dump(mode="json"),
                "schema_version": "patient-tool-gym.program.v2",
            }
        )


def test_v3_selector_literals_are_masked_in_program_signature():
    _, tool, rule, code = _evidence_state_registry("Condition", "confirmed")
    first = _selected_evidence_state_program(tool, rule, code)
    second = _selected_evidence_state_program(
        tool,
        rule,
        "444814009",
        date_from="2025-07-12",
        date_to="2025-07-12",
    )

    assert first != second
    assert program_signature(first) == program_signature(second)


def test_selector_identifiability_rejects_single_generic_token_and_requires_trusted_policy():
    state_registry, tool, rule, code = _evidence_state_registry("Procedure", "completed")
    program = _selected_evidence_state_program(tool, rule, code)
    task = compile_task(
        registry=state_registry,
        program=program,
        family_id="selector-policy",
        construction_type="atomic",
        question="Was a procedure recorded?",
    )

    assert "question_selector_not_identifiable" in task.verification.errors
    hidden = task.model_copy(
        update={
            "metadata": {
                "interface_ablation": {
                    "selector_visibility": "hidden",
                }
            }
        }
    )
    standard = verify_task_artifact(hidden, state_registry)
    trusted = verify_task_artifact(
        hidden,
        state_registry,
        policy="selector_hidden_ablation",
    )

    assert standard.passed is False
    assert "question_selector_not_identifiable" in standard.errors
    assert trusted.passed is True
    assert trusted.policy == "selector_hidden_ablation"


def test_evidence_state_and_truth_state_are_versioned_as_v2():
    state_registry, tool, rule, code = _evidence_state_registry("Condition", "confirmed")
    program = _evidence_state_program(tool, rule, code)
    artifact = compile_task(
        registry=state_registry,
        program=program,
        family_id="truth-state-schema-family",
        construction_type="atomic",
        question=(
            f"Classify the chart evidence for code {code} as supported, refuted, or "
            "insufficient evidence."
        ),
    )

    assert program.schema_version == "patient-tool-gym.program.v2"
    assert artifact.schema_version == "patient-tool-gym.task.v2"
    assert artifact.verification.verifier_version == "3"
    assert artifact.verification.passed is True
    leaking = compile_task(
        registry=state_registry,
        program=program,
        family_id="truth-state-leak-family",
        construction_type="atomic",
        question="The evidence state is supported.",
    )
    assert "question_leakage" in leaking.verification.errors
    with pytest.raises(ValidationError, match="evidence_state nodes require"):
        TaskProgram.model_validate(
            {
                **program.model_dump(mode="json"),
                "schema_version": "patient-tool-gym.program.v1",
            }
        )
    with pytest.raises(ValidationError, match="task.v2 requires program.v2 or program.v3"):
        TaskArtifact.model_validate(
            {
                **artifact.model_dump(mode="json"),
                "schema_version": "patient-tool-gym.task.v1",
            }
        )


def test_program_v2_rejects_v3_decision_rules():
    with pytest.raises(
        ValidationError,
        match="v2/v3 evidence-state decision rules require patient-tool-gym.program.v3",
    ):
        TaskProgram(
            schema_version="patient-tool-gym.program.v2",
            nodes=(
                ToolNode(id="records", tool="search_medication_requests"),
                EvidenceStateNode(
                    id="state",
                    source=ValueRef(node_id="records"),
                    decision_rule_id="medication_order_actionable.v3",
                    selector=ClinicalRecordSelector(
                        system="http://www.nlm.nih.gov/research/umls/rxnorm",
                        code="123",
                    ),
                ),
            ),
            output=ValueRef(node_id="state"),
        )


def test_program_v3_compiles_to_task_v2_with_verifier_v5():
    state_registry, tool, rule, code = _evidence_state_registry(
        "Condition", "confirmed", duplicate=True
    )
    program = _selected_evidence_state_program(tool, rule, code)

    artifact = compile_task(
        registry=state_registry,
        program=program,
        family_id="selected-truth-state-family",
        construction_type="atomic",
        question=(
            f"Classify the chart evidence for code {code} as supported, refuted, or "
            "insufficient evidence."
        ),
    )

    assert artifact.schema_version == "patient-tool-gym.task.v2"
    assert artifact.program.schema_version == "patient-tool-gym.program.v3"
    assert artifact.verification.verifier_version == "5"
    assert artifact.verification.passed is True
    assert verify_task_artifact(artifact, state_registry) == artifact.verification
    assert TaskArtifact.model_validate_json(artifact.model_dump_json()) == artifact


def test_truth_state_submission_requires_explicit_typed_envelope():
    typed = TypedValue.from_submission({"kind": "truth_state", "value": "supported"})

    assert typed == TypedValue(kind="truth_state", value="supported")
    assert typed.submission_value() == {
        "kind": "truth_state",
        "value": "supported",
    }
    assert TypedValue.from_submission("supported") == TypedValue(kind="string", value="supported")
    with pytest.raises(ValidationError, match="value does not match declared kind"):
        TypedValue.from_submission({"kind": "truth_state", "value": "unknown"})


def test_clarification_submission_is_a_tightly_validated_typed_envelope():
    typed = TypedValue.from_submission(
        {"kind": "clarification", "value": "missing_clinical_selector"}
    )

    assert typed == TypedValue(kind="clarification", value="missing_clinical_selector")
    assert typed.submission_value() == {
        "kind": "clarification",
        "value": "missing_clinical_selector",
    }
    with pytest.raises(ValidationError, match="value does not match declared kind"):
        TypedValue.from_submission({"kind": "clarification", "value": "other"})


def test_quantity_submission_value_matches_its_declared_object_schema():
    quantity = TypedValue(kind="quantity", value=24.2, unit="kg/m2")

    assert quantity.submission_value() == {"value": 24.2, "unit": "kg/m2"}
    assert TypedValue.from_submission(quantity.submission_value()) == quantity


def test_program_output_schema_is_value_free_explicit_or_semantically_inferred():
    _, tool, rule, code = _evidence_state_registry("Condition", "confirmed")
    truth_program = _evidence_state_program(tool, rule, code)
    interface_program = TaskProgram(
        schema_version="patient-tool-gym.program.v2",
        nodes=(
            ToolNode(id="records", tool=tool, arguments={"code": code, "limit": 100}),
            AggregateNode(
                id="record_count",
                function="count",
                source=ValueRef(node_id="records", path=("records",)),
            ),
            EvidenceStateNode(
                id="evidence_state",
                source=ValueRef(node_id="records"),
                decision_rule_id=rule,
            ),
            BuildObjectNode(
                id="answer",
                fields={
                    "record_count": ValueRef(node_id="record_count"),
                    "evidence_state": ValueRef(node_id="evidence_state"),
                },
            ),
        ),
        output=ValueRef(node_id="answer"),
    )

    assert infer_program_output_schema(truth_program).kind == "truth_state"
    assert program_submission_answer_schema(interface_program) == {
        "type": "object",
        "properties": {
            "record_count": {"type": "integer"},
            "evidence_state": {
                "type": "string",
                "enum": [
                    "supported",
                    "refuted",
                    "insufficient_evidence",
                    "conflicting_evidence",
                ],
            },
        },
        "required": ["evidence_state", "record_count"],
        "additionalProperties": False,
    }

    inferred = infer_program_output_schema(latest_weight_program())
    explicit = latest_weight_program().model_copy(update={"output_schema": inferred})
    assert program_submission_answer_schema(explicit) == {}

    arbitrary = latest_weight_program().model_copy(
        update={"output_schema": ProgramOutputSchema(kind="string")}
    )
    with pytest.raises(ValueError, match="must equal the schema inferred"):
        program_submission_answer_schema(arbitrary)


def test_version_trajectory_schema_is_inferred_without_executing_patient_data():
    program = TaskProgram(
        schema_version="patient-tool-gym.program.v3",
        nodes=(
            ToolNode(id="versions", tool="search_observations"),
            VersionTrajectoryNode(id="trajectory", source=ValueRef(node_id="versions")),
        ),
        output=ValueRef(node_id="trajectory"),
    )

    schema = program_submission_answer_schema(program)

    assert schema["properties"]["version_count"] == {"type": "integer"}
    assert schema["properties"]["answer_changed"] == {"type": "boolean"}
    assert schema["properties"]["earliest"]["properties"]["version_id"] == {"type": "string"}
    assert schema["properties"]["earliest"]["properties"]["value"] == {}
    assert schema["required"] == [
        "answer_changed",
        "earliest",
        "latest",
        "logical_resource_id",
        "version_count",
    ]


def test_response_contract_can_add_the_clarification_alternative():
    _, tool, rule, code = _evidence_state_registry("Condition", "confirmed")
    program = _evidence_state_program(tool, rule, code)
    schema = program_submission_answer_schema(program, allow_clarification=True)

    assert [option["properties"]["kind"]["const"] for option in schema["anyOf"]] == [
        "truth_state",
        "clarification",
    ]
    assert schema["anyOf"][1]["properties"]["value"]["enum"] == ["missing_clinical_selector"]
    assert program_submission_answer_schema(program)["properties"]["kind"]["const"] == (
        "truth_state"
    )


def test_omitted_output_schema_preserves_legacy_program_and_task_identity():
    program = latest_weight_program()
    serialized = program.model_dump(mode="json")
    assert "output_schema" not in serialized
    expected_digest = content_hash(
        {
            "snapshot_id": "legacy-snapshot",
            "program": serialized,
            "family_id": "legacy-family",
            "construction_type": "atomic",
        }
    ).removeprefix("sha256:")

    round_tripped = TaskProgram.model_validate(serialized)

    assert round_tripped.output_schema is None
    assert round_tripped.model_dump(mode="json") == serialized
    assert (
        deterministic_task_id("legacy-snapshot", round_tripped, "legacy-family", "atomic")
        == f"patient-task-{expected_digest[:24]}"
    )


def test_program_rejects_forward_or_missing_references():
    with pytest.raises(ValidationError, match="unknown or forward"):
        TaskProgram(
            nodes=(
                SelectNode(id="answer", source=ValueRef(node_id="not-yet")),
                ToolNode(id="not-yet", tool="search_observations", arguments={}),
            ),
            output=ValueRef(node_id="answer"),
        )


def test_program_rejects_duplicate_node_ids():
    with pytest.raises(ValidationError, match="duplicate program node"):
        TaskProgram(
            nodes=(
                ToolNode(id="same", tool="search_observations", arguments={}),
                ToolNode(id="same", tool="search_observations", arguments={}),
            ),
            output=ValueRef(node_id="same"),
        )


def test_compile_task_is_deterministic_and_replay_verified(registry):
    first = compile_task(
        registry=registry,
        program=latest_weight_program(),
        family_id="latest-weight",
        construction_type="atomic",
        question="What was my latest recorded weight?",
    )
    second = compile_task(
        registry=registry,
        program=latest_weight_program(),
        family_id="latest-weight",
        construction_type="atomic",
        question="What was my latest recorded weight?",
    )

    assert first == second
    assert first.verification.passed is True
    assert first.task_id.startswith("patient-task-")
    assert first.gold_evidence_ids == ("synthea:patient-a:Observation:weight-new",)
    assert TaskArtifact.model_validate_json(first.model_dump_json()) == first


def test_compile_task_executes_before_calling_question_renderer(registry, monkeypatch):
    events = []
    original_execute = verifier_module.ProgramExecutor.execute

    def recorded_execute(executor, program):
        events.append("execute")
        return original_execute(executor, program)

    def render_question():
        events.append("render")
        assert events[0] == "execute"
        return "What was my latest recorded weight?"

    monkeypatch.setattr(verifier_module.ProgramExecutor, "execute", recorded_execute)

    artifact = compile_task(
        registry=registry,
        program=latest_weight_program(),
        family_id="latest-weight-rendered-after-execution",
        construction_type="atomic",
        question=None,
        question_renderer=render_question,
    )

    assert artifact.question == "What was my latest recorded weight?"
    assert events[:2] == ["execute", "render"]


def test_compile_task_does_not_render_when_reference_execution_fails(registry, monkeypatch):
    rendered = False

    def fail_execution(executor, program):
        raise ProgramExecutionError("reference execution failed")

    def render_question():
        nonlocal rendered
        rendered = True
        return "This must not be rendered."

    monkeypatch.setattr(verifier_module.ProgramExecutor, "execute", fail_execution)

    with pytest.raises(ProgramExecutionError, match="reference execution failed"):
        compile_task(
            registry=registry,
            program=latest_weight_program(),
            family_id="failed-reference-execution",
            construction_type="atomic",
            question=None,
            question_renderer=render_question,
        )

    assert rendered is False


def test_version_trajectory_orders_timezone_offsets_as_instants():
    patient_id = "timezone-history-patient"
    snapshot = snapshot_from_payloads(
        [
            {
                "evidence_id": f"synthea:{patient_id}:Observation:result:version:1",
                "patient_id": patient_id,
                "resource_type": "Observation",
                "resource_id": "result",
                "code": {"system": "http://loinc.org", "code": "1234-5"},
                "status": "preliminary",
                "value": 1,
                "attributes": {
                    "resource_meta": {
                        "versionId": "1",
                        "lastUpdated": "2026-01-01T00:30:00+01:00",
                    }
                },
            },
            {
                "evidence_id": f"synthea:{patient_id}:Observation:result:version:2",
                "patient_id": patient_id,
                "resource_type": "Observation",
                "resource_id": "result",
                "code": {"system": "http://loinc.org", "code": "1234-5"},
                "status": "final",
                "value": 2,
                "attributes": {
                    "resource_meta": {
                        "versionId": "2",
                        "lastUpdated": "2025-12-31T23:45:00Z",
                    }
                },
            },
        ]
    )
    program = TaskProgram(
        schema_version="patient-tool-gym.program.v3",
        nodes=(
            ToolNode(
                id="versions",
                tool="search_observations",
                arguments={"system": "http://loinc.org", "code": "1234-5", "limit": 100},
            ),
            VersionTrajectoryNode(id="trajectory", source=ValueRef(node_id="versions")),
        ),
        output=ValueRef(node_id="trajectory"),
    )

    result = ProgramExecutor(ToolRegistry(snapshot)).execute(program)

    assert result.answer.value["earliest"]["version_id"] == "1"
    assert result.answer.value["latest"]["version_id"] == "2"


def test_task_schema_rejects_unknown_fields(registry):
    artifact = compile_task(
        registry=registry,
        program=latest_weight_program(),
        family_id="latest-weight",
        construction_type="atomic",
    )
    payload = artifact.model_dump(mode="json")
    payload["surprise"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TaskArtifact.model_validate(payload)


def test_replay_detects_tampered_answer(registry):
    artifact = compile_task(
        registry=registry,
        program=latest_weight_program(),
        family_id="latest-weight",
        construction_type="atomic",
    )
    tampered = artifact.model_copy(
        update={"gold_answer": artifact.gold_answer.model_copy(update={"value": "999 kg"})}
    )

    report = verify_task_artifact(tampered, registry)

    assert report.passed is False
    assert "typed_answer_mismatch" in report.errors
