from __future__ import annotations

import pytest

from patient_tool_gym.agent_adapters import AdapterDescriptor
from patient_tool_gym.compositional_analysis import (
    EVIDENCE_STATES,
    TOPOLOGIES,
    _split_bindings,
    analyze_compositional_results,
    render_compositional_markdown,
)
from patient_tool_gym.episodes import (
    EpisodeLimits,
    EpisodeRecord,
    EpisodeTokenUsage,
    source_task_hash,
)
from patient_tool_gym.program import (
    ClinicalRecordSelector,
    EvidenceStateNode,
    ProgramComplexity,
    TaskProgram,
    ToolNode,
    TruthState,
    TruthStateAggregateNode,
    TypedValue,
    ValueRef,
    truth_state_all,
)
from patient_tool_gym.task import (
    ArtifactProvenance,
    CompleteSearchRequirement,
    EvidenceQualityContract,
    TaskArtifact,
    VerificationReport,
)
from patient_tool_gym.verifier import SubmissionScore

SNOMED = "http://snomed.info/sct"
SUPERFAMILY = "compositional-family-test"
CONFIG_HASH = "sha256:" + ("c" * 64)


def _program(topology: str) -> TaskProgram:
    nodes = [
        ToolNode(
            id="focal_search",
            tool="search_conditions",
            arguments={"system": SNOMED, "limit": 100},
        ),
        EvidenceStateNode(
            id="focal_state",
            source=ValueRef(node_id="focal_search"),
            decision_rule_id="condition_documented.v2",
            selector=ClinicalRecordSelector(system=SNOMED, code="123"),
        ),
    ]
    branches = {"primary": ValueRef(node_id="focal_state")}
    if topology.endswith("w2"):
        nodes.extend(
            [
                ToolNode(
                    id="companion_search",
                    tool="search_immunizations",
                    arguments={"system": "http://hl7.org/fhir/sid/cvx", "limit": 100},
                ),
                EvidenceStateNode(
                    id="companion_state",
                    source=ValueRef(node_id="companion_search"),
                    decision_rule_id="immunization_completed.v2",
                    selector=ClinicalRecordSelector(
                        system="http://hl7.org/fhir/sid/cvx", code="140"
                    ),
                ),
            ]
        )
        branches["companion"] = ValueRef(node_id="companion_state")
    nodes.append(TruthStateAggregateNode(id="answer", branches=branches))
    return TaskProgram(
        schema_version="patient-tool-gym.program.v4",
        nodes=tuple(nodes),
        output=ValueRef(node_id="answer"),
    )


def _task(
    topology: str,
    state: str,
    distractor: bool,
    *,
    companion_state: str = "supported",
    metadata_schema: str = "v1",
    require_search: bool = False,
) -> TaskArtifact:
    if metadata_schema not in {"v1", "v2"}:
        raise ValueError("unsupported test metadata schema")
    if metadata_schema == "v1" and companion_state != "supported":
        raise ValueError("v1 test tasks have a fixed supported companion")
    companion_suffix = f"-{companion_state}" if metadata_schema == "v2" else ""
    suffix = (
        f"{topology}-{state}{companion_suffix}-"
        f"{'injected' if distractor else 'not-injected'}"
    )
    task_id = f"task-{suffix}"
    snapshot_id = f"snapshot-{suffix}"
    branches = {"primary": state}
    if topology.endswith("w2"):
        branches["companion"] = companion_state
    overall = truth_state_all(tuple(TruthState(value) for value in branches.values())).value
    evidence_id = f"synthea:test:Condition:evidence-{state}"
    requirements = ()
    if require_search:
        requirements = (
            CompleteSearchRequirement(
                claim_semantics="exclusive_supported",
                claim_node_id="focal_state",
                search_node_id="focal_search",
                snapshot_id=snapshot_id,
                tool="search_conditions",
                arguments={"system": SNOMED, "limit": 100},
                result_hash="sha256:" + ("1" * 64),
            ),
        )
    compositional_metadata = {
        "schema_version": (
            "patient-tool-gym.compositional-task-metadata.v2"
            if metadata_schema == "v2"
            else "patient-tool-gym.compositional-task-metadata.v1"
        ),
        "topology": topology,
        "evidence_state": state,
        "superfamily_id": SUPERFAMILY,
        "patient_id": "patient-test",
        "cell_id": suffix,
        "expected_branch_states": branches,
        "expected_overall_state": overall,
    }
    if metadata_schema == "v2":
        compositional_metadata.update(
            {
                "companion_evidence_state": companion_state,
                "distractor_injected": distractor,
            }
        )
    else:
        compositional_metadata["distractor_present"] = distractor

    return TaskArtifact(
        schema_version="patient-tool-gym.task.v2",
        task_id=task_id,
        family_id=f"{SUPERFAMILY}-{topology}",
        construction_type={
            "d1w1": "atomic",
            "d2w1": "depth",
            "d1w2": "width",
            "d2w2": "mixed",
        }[topology],
        patient_snapshot_id=snapshot_id,
        question=f"Question for {suffix}",
        program=_program(topology),
        gold_answer=TypedValue(
            kind="object", value={"branches": branches, "overall": overall}
        ),
        gold_evidence_ids=(evidence_id,),
        evidence_witnesses=((evidence_id,),),
        evidence_quality_contract=EvidenceQualityContract(
            complete_search_requirements=requirements
        ),
        reference_trace=(),
        allowed_tools=("search_conditions", "search_immunizations", "submit_answer"),
        difficulty=ProgramComplexity(
            dependency_depth=2 if topology.startswith("d2") else 1,
            independent_width=2 if topology.endswith("w2") else 1,
            evidence_width=len(branches),
            answer_width=len(branches),
        ),
        provenance=ArtifactProvenance(
            dataset="synthea",
            normalizer_version="test",
            tool_manifest_hash="sha256:" + ("2" * 64),
            generator="test",
            generator_version="1",
            seed=1,
        ),
        verification=VerificationReport(passed=True),
        metadata={"compositional_release": compositional_metadata},
    )


def _score(
    *,
    correct: bool,
    quality: bool,
    require_search: bool,
    certificate_valid: bool | None,
) -> SubmissionScore:
    return SubmissionScore(
        passed=correct and quality,
        answer_pass=correct,
        audit_pass=correct,
        answer_correct=correct,
        response_correct=correct,
        evidence_valid=True,
        evidence_sufficient=True,
        answer_evidence_sufficient=True,
        audit_evidence_complete=True,
        evidence_minimal=True,
        evidence_precision=1.0,
        evidence_recall=1.0,
        matched_witness_index=0,
        matched_answer_witness_index=0,
        matched_audit_witness_index=0,
        evidence_quality_applicable=True,
        answer_evidence_quality_pass=quality,
        audit_evidence_quality_pass=quality,
        complete_search_certificate_required=require_search,
        complete_search_certificate_valid=certificate_valid,
        full_snapshot_citation=False,
    )


def _record(
    task: TaskArtifact,
    *,
    answer: TypedValue | None = None,
    correct: bool = True,
    quality: bool = True,
    require_search: bool = False,
    certificate_valid: bool | None = None,
) -> EpisodeRecord:
    final_answer = task.gold_answer if answer is None else answer
    descriptor = AdapterDescriptor(
        adapter_id="openai-responses",
        model="gpt-test",
        configuration={"reasoning_effort": "medium"},
    )
    return EpisodeRecord(
        schema_version="patient-tool-gym.episode.v3",
        record_hash="sha256:" + ("3" * 64),
        episode_id=f"episode-{task.task_id}",
        task_id=task.task_id,
        source_task_hash=source_task_hash(task),
        patient_snapshot_id=task.patient_snapshot_id,
        question=task.question,
        status="completed",
        adapter_id=descriptor.adapter_id,
        adapter_descriptor=descriptor,
        model=descriptor.model,
        diagnostic_only=False,
        config_hash=CONFIG_HASH,
        seed=1,
        limits=EpisodeLimits(),
        turns=3,
        tool_call_count=3,
        latency_ms=100.0,
        tokens=EpisodeTokenUsage(
            input_tokens=100, output_tokens=10, total_tokens=110, responses=2
        ),
        messages=(),
        tool_calls=(),
        tool_results=(),
        final_answer=final_answer,
        evidence_ids=task.gold_evidence_ids,
        score=_score(
            correct=correct,
            quality=quality,
            require_search=require_search,
            certificate_valid=certificate_valid,
        ),
        errors=(),
    )


def _matrix(*, require_search: bool = False):
    tasks = {}
    records = []
    for topology in TOPOLOGIES:
        for state in EVIDENCE_STATES:
            for distractor in (False, True):
                task = _task(
                    topology,
                    state,
                    distractor,
                    require_search=require_search,
                )
                tasks[task.task_id] = task
                records.append(
                    _record(
                        task,
                        require_search=require_search,
                        quality=not require_search,
                        certificate_valid=False if require_search else None,
                    )
                )
    return tasks, records


def _matrix_v2():
    tasks = {}
    records = []
    for topology in TOPOLOGIES:
        companion_states = EVIDENCE_STATES if topology.endswith("w2") else ("supported",)
        for state in EVIDENCE_STATES:
            for companion_state in companion_states:
                for distractor in (False, True):
                    task = _task(
                        topology,
                        state,
                        distractor,
                        companion_state=companion_state,
                        metadata_schema="v2",
                    )
                    tasks[task.task_id] = task
                    records.append(_record(task))
    return tasks, records


def _analyze(tasks, records, **kwargs):
    return analyze_compositional_results(
        tasks,
        records,
        release_integrity={"passed": True, "release_hash": "sha256:" + ("4" * 64)},
        task_splits={task_id: "train" for task_id in tasks},
        family_dimensions={
            SUPERFAMILY: {
                "focal_domain": "condition",
                "focal_resource_type": "Condition",
                "companion_domain": "immunization",
                "companion_resource_type": "Immunization",
            }
        },
        **kwargs,
    )


def test_clean_canary_reports_branch_evidence_pair_and_family_metrics():
    tasks, records = _matrix()
    report = _analyze(tasks, records, require_clean_canary=True)

    assert report["gate"]["passed"] is True
    assert report["canary_cell_coverage"]["balanced_one_task_per_cell"] is True
    assert report["overall"]["exact_object_accuracy"]["rate"] == 1.0
    assert report["overall"]["companion_branch_accuracy"]["observed"] == 16
    assert report["overall"]["strict_primary_endpoint"]["rate"] == 1.0
    assert report["distractor_pairs"]["descriptive_pair_summary"]["pair_count"] == 16
    strict = report["family_level_factorial_contrasts"]["outcomes"][
        "strict_primary_endpoint_pass"
    ]
    assert strict["complete_family_primary_state_injection_blocks"] == 8
    assert strict["overall"]["depth_main_effect"]["family_count"] == 1
    assert strict["overall"]["depth_main_effect"]["mean"] == 0.0
    assert (
        strict["state_interactions_vs_supported"]["interactions"]["refuted"]
        ["width_main_effect"]["mean"]
        == 0.0
    )
    assert "Strict primary endpoint" in render_compositional_markdown(report)


def test_schema_valid_answer_mistake_is_scientific_outcome_not_gate_failure():
    tasks, records = _matrix()
    task = tasks[records[0].task_id]
    wrong_value = {
        "branches": {"primary": "refuted"},
        "overall": "refuted",
    }
    wrong = _record(
        task,
        answer=TypedValue(kind="object", value=wrong_value),
        correct=False,
        quality=False,
    )
    records[0] = wrong

    report = _analyze(tasks, records, require_clean_canary=True)

    assert report["gate"]["passed"] is True
    assert report["overall"]["answer_schema_valid"]["rate"] == 1.0
    assert report["overall"]["exact_object_accuracy"]["pass_count"] == 31
    assert report["gate"]["performance_not_used_as_gate"][
        "strict_primary_endpoint"
    ]["pass_count"] == 31


def test_invalid_answer_schema_and_missing_cell_fail_harness_gate():
    tasks, records = _matrix()
    task = tasks[records[0].task_id]
    records[0] = _record(
        task,
        answer=TypedValue(kind="object", value={"branches": {"primary": "supported"}}),
        correct=False,
        quality=False,
    )
    schema_report = _analyze(tasks, records, require_clean_canary=True)
    assert schema_report["gate"]["passed"] is False
    assert "valid_structured_answer_schema" in schema_report["gate"]["failure_reasons"]

    records = records[1:]
    coverage_report = _analyze(tasks, records, require_clean_canary=True)
    assert coverage_report["gate"]["passed"] is False
    assert "factorial_cell_coverage" in coverage_report["gate"]["failure_reasons"]
    assert coverage_report["canary_cell_coverage"]["covered_cell_count"] == 31


def test_systematic_complete_search_failure_is_manual_review_not_harness_failure():
    tasks, records = _matrix(require_search=True)
    report = _analyze(tasks, records, require_clean_canary=True)

    assert report["gate"]["passed"] is True
    assert report["gate"]["manual_review"]["required"] is True
    assert report["gate"]["manual_review"]["systematic_complete_search_failure"] is True
    assert report["overall"]["strict_primary_endpoint"]["rate"] == 0.0


def test_task_hash_and_snapshot_identity_mismatches_fail_closed():
    tasks, records = _matrix()
    bad_hash = records[0].model_copy(update={"source_task_hash": "sha256:wrong"})
    with pytest.raises(ValueError, match="source task hash mismatch"):
        _analyze(tasks, [bad_hash, *records[1:]])

    bad_snapshot = records[0].model_copy(update={"patient_snapshot_id": "wrong-snapshot"})
    with pytest.raises(ValueError, match="snapshot binding mismatch"):
        _analyze(tasks, [bad_snapshot, *records[1:]])


def test_compositional_v1_nested_split_assignments_bind_task_hashes():
    task = _task("d1w1", "supported", False)
    splits = {
        "schema_version": "patient-tool-gym.compositional-splits.v1",
        "assignments": {
            "task_assignments": [
                {
                    "task_id": task.task_id,
                    "task_hash": source_task_hash(task),
                    "split": "test",
                }
            ]
        },
    }

    assert _split_bindings(splits, {task.task_id: task}) == {task.task_id: "test"}
    splits["assignments"]["task_assignments"][0]["task_hash"] = "sha256:wrong"
    with pytest.raises(ValueError, match="task hash mismatch"):
        _split_bindings(splits, {task.task_id: task})


def test_v2_factorial_analysis_covers_companion_states_and_injection_pairs():
    tasks, records = _matrix_v2()

    report = _analyze(tasks, records, require_clean_canary=True)

    assert report["schema_version"] == "patient-tool-gym.compositional-results.v2"
    assert report["gate"]["passed"] is True
    coverage = report["canary_cell_coverage"]
    assert coverage["factorial_schema"] == "v2"
    assert coverage["covered_cell_count"] == 80
    assert coverage["expected_cell_count"] == 80
    assert coverage["balanced_one_task_per_cell"] is True
    assert report["distractor_pairs"]["descriptive_pair_summary"]["pair_count"] == 40
    assert set(report["by_companion_evidence_state"]) == set(EVIDENCE_STATES)

    topology = report["family_level_factorial_contrasts"]["outcomes"][
        "strict_primary_endpoint_pass"
    ]
    assert topology["complete_family_primary_state_injection_blocks"] == 8
    assert topology["overall"]["depth_main_effect"]["mean"] == 0.0

    companion = report["family_level_companion_state_contrasts"]["outcomes"][
        "strict_primary_endpoint_pass"
    ]
    assert companion["complete_matched_blocks"] == 16
    assert companion["overall"]["mean"] == 0.0
    assert all(
        summary["mean"] == 0.0
        for summary in companion["by_companion_evidence_state"].values()
    )


def test_v2_metadata_rejects_retired_or_missing_injection_factor():
    task = _task("d1w1", "supported", False, metadata_schema="v2")
    metadata = dict(task.metadata["compositional_release"])
    metadata.pop("distractor_injected")
    metadata["distractor_present"] = False
    malformed = task.model_copy(
        update={"metadata": {"compositional_release": metadata}}
    )

    with pytest.raises(ValueError, match="retired distractor field"):
        _analyze({malformed.task_id: malformed}, [_record(malformed)])


def test_unrecognized_metadata_schema_does_not_receive_v1_compatibility():
    task = _task("d1w1", "supported", False)
    metadata = dict(task.metadata["compositional_release"])
    metadata["schema_version"] = "patient-tool-gym.compositional-task-metadata.v3"
    malformed = task.model_copy(
        update={"metadata": {"compositional_release": metadata}}
    )

    with pytest.raises(ValueError, match="unsupported compositional metadata schema"):
        _analyze({malformed.task_id: malformed}, [_record(malformed)])
