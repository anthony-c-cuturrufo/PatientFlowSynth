from pathlib import Path

import pytest

import patient_tool_gym.verifier as verifier_module
from patient_tool_gym.agent_adapters import ReferenceAgent
from patient_tool_gym.episodes import _validate_completed_episode_replay, run_episode
from patient_tool_gym.evidence import load_snapshot, snapshot_from_payloads
from patient_tool_gym.evidence_contracts import (
    CompleteSearchCertificate,
    ObservedToolCall,
    build_complete_search_certificates,
    validate_complete_search_certificate,
)
from patient_tool_gym.program import (
    ClinicalRecordSelector,
    EvidenceStateNode,
    ExistsNode,
    ProgramExecutor,
    ProgramOutputSchema,
    SelectNode,
    TaskProgram,
    ToolNode,
    TruthState,
    ValueRef,
    content_hash,
)
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import (
    SubmissionScore,
    attach_evidence_quality_contract,
    compile_task,
    score_submission,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _positive_task(registry: ToolRegistry):
    """Compile the fabricated positive case without the legacy E0 batch helper."""
    program = TaskProgram(
        nodes=(
            ToolNode(
                id="primary_search",
                tool="search_observations",
                arguments={
                    "code": "29463-7",
                    "system": "http://loinc.org",
                    "date_from": "2026-01-25",
                    "date_to": "2026-01-25",
                    "limit": 100,
                },
            ),
            ExistsNode(
                id="answer",
                source=ValueRef(node_id="primary_search", path=("records",)),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )
    return compile_task(
        registry=registry,
        program=program,
        family_id="quality-positive",
        construction_type="atomic",
        question="Was body weight recorded on January 25, 2026?",
    )


def _absence_task() -> tuple[object, ToolRegistry]:
    registry = ToolRegistry(load_snapshot(FIXTURES / "e0_patient_a.jsonl"))
    program = TaskProgram(
        nodes=(
            ToolNode(
                id="search",
                tool="search_observations",
                arguments={"code": "does-not-exist", "limit": 100},
            ),
            ExistsNode(
                id="exists",
                source=ValueRef(node_id="search", path=("records",)),
            ),
        ),
        output=ValueRef(node_id="exists"),
        output_schema=ProgramOutputSchema(kind="boolean"),
    )
    task = compile_task(
        registry=registry,
        program=program,
        family_id="quality-absence",
        construction_type="atomic",
        question="Was the nonexistent laboratory test recorded?",
        evidence_quality_policy="exact_witness_v1",
    )
    return task, registry


def _observation(
    registry: ToolRegistry,
    *,
    call_id: str,
    arguments: dict,
) -> ObservedToolCall:
    return ObservedToolCall(
        call_id=call_id,
        tool="search_observations",
        arguments=arguments,
        result=registry.call("search_observations", arguments),
    )


def _program_observations(
    registry: ToolRegistry,
    *programs: TaskProgram,
) -> tuple[ObservedToolCall, ...]:
    observations = []
    for program_index, program in enumerate(programs, start=1):
        execution = ProgramExecutor(registry).execute(program)
        observations.extend(
            ObservedToolCall(
                call_id=f"program-{program_index}-{call_index}",
                tool=call.tool,
                arguments=call.model_arguments,
                result=registry.call(call.tool, call.model_arguments),
            )
            for call_index, call in enumerate(execution.reference_trace, start=1)
        )
    return tuple(observations)


def test_complete_search_certificate_accepts_equivalent_default_limit_and_scores_separately():
    task, registry = _absence_task()
    assert task.evidence_quality_contract is not None
    requirement = task.evidence_quality_contract.complete_search_requirements[0]
    assert requirement.arguments == {"code": "does-not-exist", "limit": 100}

    # Omitting limit uses the tool default. Because the result fits on one page, its
    # normalized clinical scope and deterministic result are identical to the reference.
    observation = _observation(
        registry,
        call_id="actual-search",
        arguments={"code": "does-not-exist"},
    )
    certificates = build_complete_search_certificates(task, registry, (observation,))

    assert len(certificates) == 1
    assert certificates[0].arguments == {"code": "does-not-exist"}
    assert (
        validate_complete_search_certificate(
            task,
            registry,
            certificates[0],
            observation,
        )
        is True
    )

    without_certificate = score_submission(
        task,
        answer=task.gold_answer,
        evidence_ids=(),
        registry=registry,
    )
    with_certificate = score_submission(
        task,
        answer=task.gold_answer,
        evidence_ids=(),
        registry=registry,
        complete_search_certificates=certificates,
        observed_tool_calls=(observation,),
    )

    assert without_certificate.passed is True
    assert without_certificate.answer_evidence_sufficient is True
    assert without_certificate.answer_evidence_quality_pass is False
    assert without_certificate.complete_search_certificate_valid is False
    assert with_certificate.passed is True
    assert with_certificate.answer_evidence_quality_pass is True
    assert with_certificate.audit_evidence_quality_pass is True
    assert with_certificate.complete_search_certificate_valid is True


def test_episode_derives_and_replays_certificate_from_actual_tool_transcript():
    task, registry = _absence_task()

    record = run_episode(task, registry, ReferenceAgent.from_task(task))

    assert record.status == "completed"
    assert record.score is not None
    assert record.score.passed is True
    assert record.score.answer_evidence_quality_pass is True
    assert record.score.complete_search_certificate_valid is True
    _validate_completed_episode_replay(record, task=task, registry=registry)

def test_complete_search_certificate_rejects_random_complete_search_and_forgery():
    task, registry = _absence_task()
    unrelated = _observation(
        registry,
        call_id="wrong-search",
        arguments={"code": "another-absent-code", "limit": 100},
    )

    assert unrelated.result["complete"] is True
    assert build_complete_search_certificates(task, registry, (unrelated,)) == ()

    requirement = task.evidence_quality_contract.complete_search_requirements[0]
    forged = CompleteSearchCertificate(
        task_id=task.task_id,
        requirement_index=0,
        claim_node_id=requirement.claim_node_id,
        search_node_id=requirement.search_node_id,
        call_id="forged",
        snapshot_id=registry.snapshot_id,
        tool=requirement.tool,
        arguments=unrelated.arguments,
        result_hash=requirement.result_hash,
    )
    assert validate_complete_search_certificate(task, registry, forged, unrelated) is False
    stale_snapshot = forged.model_copy(update={"snapshot_id": "sha256:" + ("0" * 64)})
    assert (
        validate_complete_search_certificate(task, registry, stale_snapshot, unrelated)
        is False
    )

    incomplete = CompleteSearchCertificate(
        task_id=task.task_id,
        requirement_index=0,
        claim_node_id=requirement.claim_node_id,
        search_node_id=requirement.search_node_id,
        call_id="incomplete",
        snapshot_id=registry.snapshot_id,
        tool=requirement.tool,
        arguments={"limit": 1},
        result_hash=requirement.result_hash,
    )
    assert (
        validate_complete_search_certificate(task, registry, incomplete, unrelated)
        is False
    )


def test_score_rejects_replayable_certificate_not_bound_to_observed_call():
    task, registry = _absence_task()
    requirement = task.evidence_quality_contract.complete_search_requirements[0]
    actual = _observation(
        registry,
        call_id="actual-search",
        arguments=requirement.arguments,
    )
    forged = CompleteSearchCertificate(
        task_id=task.task_id,
        requirement_index=0,
        claim_node_id=requirement.claim_node_id,
        search_node_id=requirement.search_node_id,
        call_id="never-executed",
        snapshot_id=registry.snapshot_id,
        tool=requirement.tool,
        arguments=requirement.arguments,
        result_hash=requirement.result_hash,
    )

    score = score_submission(
        task,
        answer=task.gold_answer,
        evidence_ids=(),
        registry=registry,
        complete_search_certificates=(forged,),
        observed_tool_calls=(actual,),
    )

    assert validate_complete_search_certificate(task, registry, forged, actual) is False
    assert score.complete_search_certificate_valid is False
    assert score.answer_evidence_quality_pass is False
    assert "complete_search_certificate_invalid" in score.evidence_quality_failure_codes


def test_closed_world_derivation_resolves_select_paths_to_the_reference_search():
    registry = ToolRegistry(load_snapshot(FIXTURES / "e0_patient_a.jsonl"))
    program = TaskProgram(
        nodes=(
            ToolNode(
                id="search",
                tool="search_observations",
                arguments={"code": "does-not-exist", "limit": 100},
            ),
            SelectNode(
                id="records",
                source=ValueRef(node_id="search", path=("records",)),
            ),
            ExistsNode(id="exists", source=ValueRef(node_id="records")),
        ),
        output=ValueRef(node_id="exists"),
    )

    task = compile_task(
        registry=registry,
        program=program,
        family_id="quality-select-absence",
        construction_type="atomic",
        question="Was the nonexistent laboratory test recorded?",
        evidence_quality_policy="exact_witness_v1",
    )

    assert task.verification.passed is True
    requirement = task.evidence_quality_contract.complete_search_requirements[0]
    assert requirement.search_node_id == "search"
    assert requirement.claim_node_id == "exists"


def test_exact_witness_quality_rejects_supersets_and_the_full_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = ToolRegistry(load_snapshot(FIXTURES / "e0_patient_a.jsonl"))
    legacy = _positive_task(registry)
    task = attach_evidence_quality_contract(legacy, registry)
    assert task.task_id == legacy.task_id
    assert content_hash(task) != content_hash(legacy)
    witness = task.gold_evidence_ids
    extra = next(
        record.evidence_id
        for record in registry.snapshot.records
        if record.evidence_id not in witness
    )
    full_snapshot = tuple(sorted(record.evidence_id for record in registry.snapshot.records))
    observations = _program_observations(registry, task.program)

    guessed_without_retrieval = score_submission(
        task,
        answer=task.gold_answer,
        evidence_ids=witness,
        registry=registry,
    )

    exact = score_submission(
        task,
        answer=task.gold_answer,
        evidence_ids=witness,
        registry=registry,
        observed_tool_calls=observations,
    )
    wrong_answer = score_submission(
        task,
        answer=task.gold_answer.model_copy(update={"value": not task.gold_answer.value}),
        evidence_ids=witness,
        registry=registry,
        observed_tool_calls=observations,
    )
    original_minimizer = verifier_module.minimal_sufficient_subset
    legacy_minimizer_calls = 0

    def counting_minimizer(*args, **kwargs):
        nonlocal legacy_minimizer_calls
        legacy_minimizer_calls += 1
        return original_minimizer(*args, **kwargs)

    monkeypatch.setattr(verifier_module, "minimal_sufficient_subset", counting_minimizer)
    legacy_superset = score_submission(
        legacy,
        answer=legacy.gold_answer,
        evidence_ids=tuple(sorted((*witness, extra))),
        registry=registry,
        observed_tool_calls=observations,
    )
    assert legacy_superset.answer_evidence_sufficient is True
    assert legacy_minimizer_calls == 1

    def forbidden_minimizer(*_args, **_kwargs):
        raise AssertionError("quality supersets must not invoke executable minimization")

    monkeypatch.setattr(verifier_module, "minimal_sufficient_subset", forbidden_minimizer)
    superset = score_submission(
        task,
        answer=task.gold_answer,
        evidence_ids=tuple(sorted((*witness, extra))),
        registry=registry,
    )
    full = score_submission(
        task,
        answer=task.gold_answer,
        evidence_ids=full_snapshot,
        registry=registry,
        observed_tool_calls=observations,
    )

    assert guessed_without_retrieval.answer_evidence_quality_pass is False
    assert "unobserved_evidence_ids" in (
        guessed_without_retrieval.evidence_quality_failure_codes
    )
    assert exact.answer_evidence_quality_pass is True
    assert wrong_answer.answer_correct is False
    assert wrong_answer.answer_evidence_quality_pass is False
    assert "response_incorrect" in wrong_answer.evidence_quality_failure_codes
    assert superset.answer_evidence_sufficient is True
    assert superset.answer_evidence_quality_pass is False
    assert "answer_not_exact_declared_witness" in superset.evidence_quality_failure_codes
    assert full.answer_evidence_sufficient is True
    assert full.full_snapshot_citation is True
    assert full.answer_evidence_quality_pass is False
    assert full.audit_evidence_quality_pass is False


def test_each_alternative_declared_witness_can_pass_exact_quality():
    patient_id = "alternative-quality"
    payloads = [
        {
            "evidence_id": f"synthea:{patient_id}:Observation:weight-{index}",
            "patient_id": patient_id,
            "resource_type": "Observation",
            "resource_id": f"weight-{index}",
            "code": {"system": "http://loinc.org", "code": "29463-7"},
            "display": "Body weight",
            "value": 70 + index,
        }
        for index in (1, 2)
    ]
    registry = ToolRegistry(snapshot_from_payloads(payloads))
    program = TaskProgram(
        nodes=(
            ToolNode(
                id="weights",
                tool="search_observations",
                arguments={
                    "system": "http://loinc.org",
                    "code": "29463-7",
                    "limit": 100,
                },
            ),
            ExistsNode(
                id="exists",
                source=ValueRef(node_id="weights", path=("records",)),
            ),
        ),
        output=ValueRef(node_id="exists"),
    )
    base = compile_task(
        registry=registry,
        program=program,
        family_id="alternative-quality",
        construction_type="atomic",
        question="Was my body weight recorded?",
    )
    alternatives = tuple((evidence_id,) for evidence_id in base.gold_evidence_ids)
    declared = base.model_copy(
        update={"evidence_witnesses": (base.gold_evidence_ids, *alternatives)}
    )
    task = attach_evidence_quality_contract(declared, registry)
    observations = _program_observations(registry, task.program)

    scores = [
        score_submission(
            task,
            answer=task.gold_answer,
            evidence_ids=witness,
            registry=registry,
            observed_tool_calls=observations,
        )
        for witness in alternatives
    ]

    assert task.verification.passed is True
    assert all(score.answer_evidence_quality_pass is True for score in scores)


@pytest.mark.parametrize(
    ("status", "expected", "semantics", "opposing_status"),
    [
        ("completed", TruthState.SUPPORTED, "exclusive_supported", "not-done"),
        ("not-done", TruthState.REFUTED, "exclusive_refuted", "completed"),
    ],
)
def test_exclusive_evidence_state_quality_requires_search_for_opposing_records(
    status: str,
    expected: TruthState,
    semantics: str,
    opposing_status: str,
):
    patient_id = f"exclusive-{status}"

    def procedure(identifier: str, selected_status: str) -> dict:
        return {
            "evidence_id": f"synthea:{patient_id}:Procedure:{identifier}",
            "patient_id": patient_id,
                "resource_type": "Procedure",
                "resource_id": identifier,
                "logical_evidence_group": f"clinical-identifier:{patient_id}:colonoscopy",
                "status": selected_status,
            "code": {"system": "http://snomed.info/sct", "code": "73761001"},
            "display": "Colonoscopy",
        }

    registry = ToolRegistry(snapshot_from_payloads([procedure("one", status)]))
    program = TaskProgram(
        schema_version="patient-tool-gym.program.v3",
        nodes=(
            ToolNode(
                id="procedures",
                tool="search_procedures",
                arguments={"limit": 100},
            ),
            EvidenceStateNode(
                id="state",
                source=ValueRef(node_id="procedures"),
                decision_rule_id="procedure_completed.v2",
                selector=ClinicalRecordSelector(
                    system="http://snomed.info/sct",
                    code="73761001",
                ),
            ),
        ),
        output=ValueRef(node_id="state"),
        output_schema=ProgramOutputSchema(kind="truth_state"),
    )
    task = compile_task(
        registry=registry,
        program=program,
        family_id=f"exclusive-quality-{status}",
        construction_type="atomic",
        question=(
            "Does my chart show the colonoscopy as supported, refuted, insufficient evidence, "
            "or conflicting evidence?"
        ),
        evidence_quality_policy="exact_witness_v1",
    )

    assert task.gold_answer.value == expected.value
    requirement = task.evidence_quality_contract.complete_search_requirements[0]
    assert requirement.claim_semantics == semantics

    opposing_registry = ToolRegistry(
        snapshot_from_payloads(
            [procedure("one", status), procedure("opposing", opposing_status)]
        )
    )
    opposing_result = ProgramExecutor(opposing_registry).execute(program)
    assert opposing_result.answer.value == TruthState.CONFLICTING_EVIDENCE.value


def test_legacy_task_and_score_remain_quality_unscored_and_hash_shape_compatible():
    task, registry = _absence_task()
    legacy = task.model_copy(update={"evidence_quality_contract": None})
    score = score_submission(
        legacy,
        answer=legacy.gold_answer,
        evidence_ids=(),
        registry=registry,
    )
    frozen_score = SubmissionScore.model_validate(
        {
            "passed": True,
            "answer_correct": True,
            "evidence_precision": 1.0,
            "evidence_recall": 1.0,
            "unsupported_evidence_ids": [],
        }
    )

    assert score.evidence_quality_applicable is False
    assert score.answer_evidence_quality_pass is None
    serialized = frozen_score.model_dump(mode="json")
    assert "evidence_quality_applicable" not in serialized
    assert "answer_evidence_quality_pass" not in serialized
    assert content_hash(serialized).startswith("sha256:")


def test_certificate_requirement_distinguishes_positive_and_absence_tasks():
    absence_task, registry = _absence_task()
    positive_legacy = _positive_task(registry)
    positive_task = attach_evidence_quality_contract(positive_legacy, registry)
    assert positive_task.evidence_quality_contract is not None
    assert positive_task.evidence_quality_contract.complete_search_requirements == ()

    absence_record = run_episode(
        absence_task,
        registry,
        ReferenceAgent.from_task(absence_task),
    )
    positive_record = run_episode(
        positive_task,
        registry,
        ReferenceAgent.from_task(positive_task),
    )
    assert absence_record.score.complete_search_certificate_valid is True
    assert positive_record.score.complete_search_certificate_required is False
    assert positive_record.score.complete_search_certificate_valid is None
