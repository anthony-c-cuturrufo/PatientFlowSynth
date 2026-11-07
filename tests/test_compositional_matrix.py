from pathlib import Path

import pytest

import patient_tool_gym.compositional_matrix as compositional_matrix
from patient_tool_gym.compositional_matrix import (
    TOPOLOGIES,
    build_compositional_program,
    build_compositional_question,
    build_compositional_topology,
    discover_compositional_candidates,
    verify_topology_complexity,
)
from patient_tool_gym.evidence import load_snapshot
from patient_tool_gym.patient_p2 import generate_patient_p2_tasks
from patient_tool_gym.program import (
    EvidenceStateNode,
    ProgramExecutor,
    ToolNode,
    TruthState,
    TruthStateAggregateNode,
    ValueRef,
    infer_program_output_schema,
)
from patient_tool_gym.tools import ToolRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "patient_v1.jsonl"


def _source():
    snapshot = load_snapshot(FIXTURE)
    tasks = generate_patient_p2_tasks((snapshot,)).tasks
    return snapshot, tasks


def test_discovers_deterministic_encounter_depth_candidates_with_disjoint_companions():
    _, tasks = _source()

    first = discover_compositional_candidates(tasks, seed=20260711)
    reordered = discover_compositional_candidates(tuple(reversed(tasks)), seed=20260711)

    assert first == reordered
    assert first.candidate_count == 1
    assert [item.reason for item in first.rejections] == ["REFUTED_DATE_NOT_REPRESENTABLE"]
    assert {candidate.focal_domain for candidate in first.candidates} == {"condition"}
    # Exact-date Procedure claims lose performed[x] under valid nonperformance states, so
    # the four-state width factorial deterministically selects a stateful companion.
    assert first.candidates[0].companion_domain == "immunization"
    task_index = {task.task_id: task for task in tasks}
    for candidate in first.candidates:
        atomic = task_index[candidate.focal_atomic_task_id]
        depth = task_index[candidate.focal_depth_task_id]
        companion = task_index[candidate.companion_task_id]
        assert depth.parent_task_ids == (atomic.task_id,)
        assert candidate.routing_evidence_id in depth.gold_evidence_ids
        assert set(companion.gold_evidence_ids).isdisjoint(depth.gold_evidence_ids)
        assert companion.patient_snapshot_id == atomic.patient_snapshot_id
        assert candidate.companion_family_id != candidate.focal_family_id


def test_builds_and_executes_the_exact_four_topologies_with_structured_branch_outputs():
    snapshot, tasks = _source()
    candidate = discover_compositional_candidates(tasks).candidates[0]
    registry = ToolRegistry(snapshot)
    expected = {
        "d1w1": (1, 1, 1),
        "d2w1": (2, 1, 1),
        "d1w2": (1, 2, 2),
        "d2w2": (2, 2, 2),
    }
    answers = set()

    for topology in TOPOLOGIES:
        built = build_compositional_topology(
            candidate,
            tasks,
            topology,
            registry=registry,
        )
        result = ProgramExecutor(registry).execute(built.program)
        answer = result.answer.value
        answers.add(answer["overall"])

        assert built.program.schema_version == "patient-tool-gym.program.v4"
        assert built.program.output == ValueRef(node_id="answer")
        assert built.program.output_schema == infer_program_output_schema(built.program)
        assert built.program.output_schema.kind == "object"
        assert result.answer.kind == "object"
        assert answer["overall"] in set(TruthState)
        assert set(answer["branches"]) == (
            {"primary", "companion"} if topology.endswith("w2") else {"primary"}
        )
        assert all(state in set(TruthState) for state in answer["branches"].values())
        assert (
            result.complexity.dependency_depth,
            result.complexity.independent_width,
            result.complexity.answer_width,
        ) == expected[topology]
        assert built.structural_complexity.model_copy(update={"evidence_width": 0}) == (
            result.complexity.model_copy(update={"evidence_width": 0})
        )
        if topology.startswith("d2"):
            assert candidate.routing_evidence_id in result.evidence_ids

    # All four programs ask the same focal claim on the same unchanged chart. The width
    # companion is fixed, so topology alone must not silently change the answer.
    assert len(answers) == 1


def test_release_topology_executes_before_masked_question_rendering(monkeypatch):
    snapshot, tasks = _source()
    candidate = discover_compositional_candidates(tasks).candidates[0]
    events = []
    original_execute = compositional_matrix.ProgramExecutor.execute
    original_render = compositional_matrix.render_compositional_question

    def recorded_execute(executor, program):
        events.append("execute")
        return original_execute(executor, program)

    def recorded_render(intent):
        events.append("render")
        assert events[0] == "execute"
        serialized = str(intent.model_dump(mode="json"))
        assert candidate.patient_id not in serialized
        assert candidate.focal_evidence_id not in serialized
        assert candidate.routing_evidence_id not in serialized
        return original_render(intent)

    monkeypatch.setattr(compositional_matrix.ProgramExecutor, "execute", recorded_execute)
    monkeypatch.setattr(
        compositional_matrix,
        "render_compositional_question",
        recorded_render,
    )

    compositional_matrix.build_compositional_topology(
        candidate,
        tasks,
        "d2w2",
        registry=ToolRegistry(snapshot),
    )

    assert events == ["execute", "render"]


def test_failed_reference_execution_cannot_reach_question_renderer(monkeypatch):
    snapshot, tasks = _source()
    candidate = discover_compositional_candidates(tasks).candidates[0]
    rendered = False

    def fail_execution(executor, program):
        raise RuntimeError("reference execution failed")

    def record_render(intent):
        nonlocal rendered
        rendered = True
        return "should not render"

    monkeypatch.setattr(compositional_matrix.ProgramExecutor, "execute", fail_execution)
    monkeypatch.setattr(
        compositional_matrix,
        "render_compositional_question",
        record_render,
    )

    with pytest.raises(RuntimeError, match="reference execution failed"):
        compositional_matrix.build_compositional_topology(
            candidate,
            tasks,
            "d2w2",
            registry=ToolRegistry(snapshot),
        )

    assert rendered is False


def test_broad_clinical_searches_hide_answer_code_and_status_but_keep_typed_selectors():
    _, tasks = _source()
    candidate = discover_compositional_candidates(tasks).candidates[0]

    for topology in TOPOLOGIES:
        program = build_compositional_program(candidate, tasks, topology)
        clinical_searches = [
            node
            for node in program.nodes
            if isinstance(node, ToolNode) and node.tool != "search_encounters"
        ]
        state_nodes = [node for node in program.nodes if isinstance(node, EvidenceStateNode)]
        assert len(clinical_searches) == (2 if topology.endswith("w2") else 1)
        assert len(state_nodes) == len(clinical_searches)
        for search in clinical_searches:
            assert "system" in search.arguments
            assert search.arguments["date_from"] == search.arguments["date_to"]
            assert "code" not in search.arguments
            assert "status" not in search.arguments
            assert "clinical_status" not in search.arguments
            assert "verification_status" not in search.arguments
        assert all(node.selector is not None for node in state_nodes)


def test_depth_variants_bind_the_focal_search_to_the_exact_encounter_result():
    snapshot, tasks = _source()
    candidate = discover_compositional_candidates(tasks).candidates[0]

    for topology in ("d2w1", "d2w2"):
        program = build_compositional_program(candidate, tasks, topology)
        encounter = program.nodes[0]
        focal = program.nodes[1]
        assert isinstance(encounter, ToolNode)
        assert encounter.tool == "search_encounters"
        assert isinstance(focal, ToolNode)
        assert focal.arguments["encounter_id"] == {
            "$ref": "focal_encounter_search",
            "path": ["records", 0, "evidence_id"],
        }

        result = ProgramExecutor(ToolRegistry(snapshot)).execute(program)
        assert [call.tool for call in result.reference_trace[:2]] == [
            "search_encounters",
            focal.tool,
        ]
        assert result.reference_trace[1].depends_on == ("focal_encounter_search",)
        assert (
            result.reference_trace[1].model_arguments["encounter_id"]
            == candidate.routing_evidence_id
        )
        assert candidate.routing_evidence_id in result.evidence_ids


def test_questions_preserve_four_state_vocabulary_and_expose_only_clinical_language():
    _, tasks = _source()
    candidate = discover_compositional_candidates(tasks).candidates[0]

    for topology in TOPOLOGIES:
        question = build_compositional_question(candidate, topology)
        assert all(
            label in question
            for label in (
                "supported",
                "refuted",
                "insufficient evidence",
                "conflicting evidence",
            )
        )
        assert candidate.focal_evidence_id not in question
        assert "urn:uuid:" not in question
        assert candidate.focal_selector.code in question
        if topology.startswith("d2"):
            assert f"most recent {candidate.encounter_name}" in question
        if topology.endswith("w2"):
            assert candidate.companion_phrase in question
            assert candidate.companion_selector.code in question
            assert "First claim" in question
            assert "Second claim" in question
            assert "branches.companion" in question
        else:
            assert "branches.companion" not in question
        assert "branches.primary" in question
        assert "overall" in question


def test_no_disjoint_companion_is_reported_instead_of_reusing_focal_evidence():
    _, tasks = _source()
    focal_family = next(
        task.family_id
        for task in tasks
        if task.construction_type == "depth" and task.metadata["lineage"]["domain"] == "condition"
    )
    only_focal = tuple(task for task in tasks if task.family_id == focal_family)

    result = discover_compositional_candidates(only_focal)

    assert result.candidates == ()
    assert [item.reason for item in result.rejections] == ["NO_DISJOINT_COMPANION"]


def test_topology_verification_rejects_labels_that_do_not_match_the_program_graph():
    _, tasks = _source()
    candidate = discover_compositional_candidates(tasks).candidates[0]
    atomic_program = build_compositional_program(candidate, tasks, "d1w1")

    with pytest.raises(ValueError, match="topology d2w2 requires"):
        verify_topology_complexity(atomic_program, "d2w2")

    aggregate = atomic_program.nodes[-1]
    assert isinstance(aggregate, TruthStateAggregateNode)
    assert aggregate.branches == {"primary": ValueRef(node_id="focal_state")}
