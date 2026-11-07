import pytest
from pydantic import ValidationError

from patient_tool_gym.agent_adapters import OPENAI_AGENT_INSTRUCTIONS
from patient_tool_gym.evidence import snapshot_from_payloads
from patient_tool_gym.program import (
    ClinicalRecordSelector,
    EvidenceStateNode,
    TaskProgram,
    ToolNode,
    TypedValue,
    ValueRef,
    content_hash,
)
from patient_tool_gym.task import QuestionContract, TaskArtifact
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import (
    attach_evidence_quality_contract,
    compile_task,
    score_submission,
    verify_task_artifact,
)

SNOMED = "http://snomed.info/sct"
TARGET_CODE = "73761001"
CLARIFICATION = TypedValue(
    kind="clarification",
    value="missing_clinical_selector",
)


def _selector_task() -> tuple[TaskArtifact, ToolRegistry]:
    registry = ToolRegistry(
        snapshot_from_payloads(
            (
                {
                    "category": "demographics",
                    "code": {},
                    "display": "Question-contract patient",
                    "evidence_id": "fixture:question-contract:Patient:patient",
                    "patient_id": "question-contract",
                    "references": {},
                    "resource_type": "Patient",
                    "source": {"resource_id": "patient"},
                },
                {
                    "category": "procedure",
                    "code": {
                        "system": SNOMED,
                        "code": TARGET_CODE,
                        "display": "Target procedure",
                    },
                    "display": "Target procedure",
                    "evidence_id": "fixture:question-contract:Procedure:target",
                    "patient_id": "question-contract",
                    "references": {},
                    "resource_type": "Procedure",
                    "source": {"resource_id": "target"},
                    "status": "completed",
                    "text": "Target procedure completed",
                    "time": {"start": "2026-01-01T09:00:00Z"},
                    "value": "completed",
                },
            )
        )
    )
    program = TaskProgram(
        schema_version="patient-tool-gym.program.v3",
        nodes=(
            ToolNode(
                id="broad_search",
                tool="search_procedures",
                arguments={"system": SNOMED, "limit": 100},
            ),
            EvidenceStateNode(
                id="answer",
                source=ValueRef(node_id="broad_search"),
                decision_rule_id="procedure_completed.v2",
                selector=ClinicalRecordSelector(system=SNOMED, code=TARGET_CODE),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )
    task = compile_task(
        registry=registry,
        program=program,
        family_id="question-contract-family",
        construction_type="atomic",
        question=(
            "Classify Target procedure as supported, refuted, insufficient evidence, "
            "or conflicting evidence."
        ),
    )
    assert task.verification.passed is True
    return task, registry


def _hidden_task(
    task: TaskArtifact,
    *,
    declare_contract: bool = True,
) -> TaskArtifact:
    updates = {
        "question": (
            "Classify the matching chart record as supported, refuted, insufficient evidence, "
            "or conflicting evidence."
        ),
        "metadata": {
            **task.metadata,
            "interface_ablation": {"selector_visibility": "hidden"},
        },
    }
    if declare_contract:
        updates["question_contract"] = QuestionContract(
            answerability="clarification_required",
            reason="missing_clinical_selector",
            allow_clarification=True,
        )
    return task.model_copy(update=updates)


def test_hidden_question_accepts_exact_evidence_free_clarification():
    task, registry = _selector_task()
    hidden = _hidden_task(task)

    report = verify_task_artifact(
        hidden,
        registry,
        policy="selector_hidden_ablation",
    )
    score = score_submission(
        hidden,
        answer=CLARIFICATION,
        evidence_ids=(),
        registry=registry,
    )

    assert report.passed is True
    assert "question_selector_hidden_proved" in report.checks
    assert "hidden_question_clarification_contract_declared" in report.checks
    assert score.answer_correct is False
    assert score.response_correct is True
    assert score.clarification_appropriate is True
    assert score.clarification_pass is True
    assert score.answer_pass is True
    assert score.audit_pass is True
    assert score.passed is True


def test_openai_instructions_disclose_compositional_answer_and_search_contract():
    instructions = OPENAI_AGENT_INSTRUCTIONS

    # Legacy scalar evidence-state tasks retain their typed envelope.
    assert '{"kind":"truth_state","value":STATE}' in instructions
    # Program.v4 tasks instead follow the public, task-specific submit schema.
    assert (
        'raw JSON object shaped as {"branches":{"<required_branch_name>":"<state>"},'
        '"overall":"<state>"}' in instructions
    )
    assert "Each state in this object is a bare string" in instructions
    # The scored complete-search and distractor protocol is public, not a hidden rubric.
    assert (
        "date_from and date_to both equal to that date in YYYY-MM-DD format, and limit 100"
        in instructions
    )
    assert "http://hl7.org/fhir/sid/cvx for CVX" in instructions
    assert "omit code, text, status, clinical_status, verification_status, and cursor" in (
        instructions
    )
    assert "number of returned records equals total_matches" in instructions
    assert "plus the selected encounter record for an encounter-relative branch" in instructions
    # Universal instructions must not contain chart-derived identifiers or answers.
    assert "synthea:" not in instructions
    assert "urn:uuid:" not in instructions


def test_submit_answer_round_trips_raw_compositional_object():
    _, registry = _selector_task()
    answer = {
        "branches": {
            "primary": "conflicting_evidence",
            "companion": "supported",
        },
        "overall": "conflicting_evidence",
    }

    result = registry.call(
        "submit_answer",
        {"answer": answer, "evidence_ids": []},
    )
    parsed = TypedValue.from_submission(result["submission"]["answer"])

    assert result["ok"] is True
    assert parsed == TypedValue(kind="object", value=answer)
    assert parsed.submission_value() == answer


def test_clarification_required_contract_must_expose_the_response():
    with pytest.raises(ValidationError, match="must allow the clarification response"):
        QuestionContract(
            answerability="clarification_required",
            reason="missing_clinical_selector",
        )

    visible_union = QuestionContract(
        answerability="answerable",
        allow_clarification=True,
    )
    assert visible_union.allow_clarification is True


def test_hidden_question_rejects_a_correct_guess_of_the_undisclosed_clinical_answer():
    task, registry = _selector_task()
    hidden = _hidden_task(task)

    score = score_submission(
        hidden,
        answer=hidden.gold_answer,
        evidence_ids=hidden.gold_evidence_ids,
        registry=registry,
    )

    assert score.answer_correct is True
    assert score.response_correct is False
    assert score.clarification_appropriate is False
    assert score.clarification_pass is False
    assert score.answer_pass is False
    assert score.audit_pass is False
    assert score.passed is False


def test_answerable_question_rejects_clarification_even_without_evidence():
    task, registry = _selector_task()

    score = score_submission(
        task,
        answer=CLARIFICATION,
        evidence_ids=(),
        registry=registry,
    )

    assert score.answer_correct is False
    assert score.response_correct is False
    assert score.clarification_appropriate is False
    assert score.clarification_pass is False
    assert score.passed is False


def test_hidden_policy_proves_selector_absence_but_accepts_frozen_legacy_artifact():
    task, registry = _selector_task()
    legacy_hidden = _hidden_task(task, declare_contract=False)
    falsely_declared_hidden = task.model_copy(
        update={
            "metadata": {
                **task.metadata,
                "interface_ablation": {"selector_visibility": "hidden"},
            }
        }
    )

    legacy_report = verify_task_artifact(
        legacy_hidden,
        registry,
        policy="selector_hidden_ablation",
    )
    visible_report = verify_task_artifact(
        falsely_declared_hidden,
        registry,
        policy="selector_hidden_ablation",
    )

    assert legacy_report.passed is True
    assert "legacy_hidden_question_contract_accepted" in legacy_report.checks
    assert visible_report.passed is False
    assert "selector_hidden_policy_selector_still_identifiable" in visible_report.errors


def test_absent_question_contract_preserves_legacy_serialization_and_hash():
    task, _ = _selector_task()
    payload = task.model_dump(mode="json")

    assert task.question_contract is None
    assert "question_contract" not in payload
    assert TaskArtifact.model_validate(payload).model_dump(mode="json") == payload
    assert content_hash(task) == content_hash(payload)


def test_quality_attachment_preserves_selector_hidden_verification_policy():
    task, registry = _selector_task()
    hidden = _hidden_task(task)
    hidden_report = verify_task_artifact(
        hidden,
        registry,
        policy="selector_hidden_ablation",
    )
    hidden = hidden.model_copy(update={"verification": hidden_report})

    attached = attach_evidence_quality_contract(hidden, registry)

    assert attached.verification.passed is True
    assert attached.verification.verifier_version == "6"
    assert attached.verification.policy == "selector_hidden_ablation"
    score = score_submission(
        attached,
        answer=CLARIFICATION,
        evidence_ids=(),
        registry=registry,
    )
    assert score.clarification_pass is True
    assert score.answer_evidence_quality_pass is None
    assert score.audit_evidence_quality_pass is None
    assert score.complete_search_certificate_valid is None


def test_agent_instructions_disclose_the_exact_clarification_envelope():
    assert '{"kind":"clarification","value":"missing_clinical_selector"}' in (
        OPENAI_AGENT_INSTRUCTIONS
    )
    assert "empty evidence_ids list" in OPENAI_AGENT_INSTRUCTIONS
