from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any, Literal

from pydantic import Field

from patient_tool_gym.program import (
    ClinicalRecordSelector,
    EvidenceStateNode,
    ExistsNode,
    ProgramComplexity,
    ProgramExecutor,
    ProgramOutputSchema,
    StrictModel,
    TaskProgram,
    ToolNode,
    TruthStateAggregateNode,
    TypedValue,
    ValueRef,
    analyze_complexity,
    binding,
    content_hash,
    node_references,
)
from patient_tool_gym.task import TaskArtifact
from patient_tool_gym.tools import ToolRegistry

COMPOSITIONAL_MATRIX_SCHEMA_VERSION = "patient-tool-gym.compositional-matrix.v2"
COMPOSITIONAL_MATRIX_BUILDER_VERSION = "patient-tool-gym.compositional-matrix-builder.v2"

CompositionalTopology = Literal["d1w1", "d2w1", "d1w2", "d2w2"]
CompositionalConstruction = Literal["atomic", "depth", "width", "mixed"]

TOPOLOGIES: tuple[CompositionalTopology, ...] = (
    "d1w1",
    "d2w1",
    "d1w2",
    "d2w2",
)

_DOMAIN_CONTRACTS: dict[str, tuple[str, str, str]] = {
    "condition": ("Condition", "search_conditions", "condition_documented.v2"),
    "lab_observation": (
        "Observation",
        "search_observations",
        "observation_final_result.v2",
    ),
    "procedure": ("Procedure", "search_procedures", "procedure_completed.v2"),
    "immunization": (
        "Immunization",
        "search_immunizations",
        "immunization_completed.v2",
    ),
}

_TOPOLOGY_SHAPES: dict[CompositionalTopology, tuple[int, int, int]] = {
    "d1w1": (1, 1, 1),
    "d2w1": (2, 1, 1),
    "d1w2": (1, 2, 2),
    "d2w2": (2, 2, 2),
}

_TOPOLOGY_CONSTRUCTIONS: dict[CompositionalTopology, CompositionalConstruction] = {
    "d1w1": "atomic",
    "d2w1": "depth",
    "d1w2": "width",
    "d2w2": "mixed",
}


class CompositionalCandidate(StrictModel):
    """One focal P2 family plus a disjoint same-patient atomic companion."""

    schema_version: Literal["patient-tool-gym.compositional-candidate.v1"] = (
        "patient-tool-gym.compositional-candidate.v1"
    )
    superfamily_id: str
    patient_id: str
    patient_snapshot_id: str
    focal_family_id: str
    focal_domain: str
    focal_resource_type: str
    focal_atomic_task_id: str
    focal_depth_task_id: str
    companion_family_id: str
    companion_domain: str
    companion_resource_type: str
    companion_task_id: str
    focal_evidence_id: str
    routing_evidence_id: str
    companion_evidence_ids: tuple[str, ...]
    focal_selector: ClinicalRecordSelector
    companion_selector: ClinicalRecordSelector
    focal_phrase: str
    companion_phrase: str
    encounter_name: str
    source_task_hashes: dict[str, str]


class CompositionalCandidateRejection(StrictModel):
    schema_version: Literal["patient-tool-gym.compositional-candidate-rejection.v1"] = (
        "patient-tool-gym.compositional-candidate-rejection.v1"
    )
    focal_family_id: str
    reason: Literal[
        "MISSING_ATOMIC_OR_DEPTH",
        "UNSUPPORTED_DOMAIN",
        "UNSUPPORTED_DEPTH_MODE",
        "UNVERIFIED_SOURCE",
        "SOURCE_NOT_POSITIVE",
        "INVALID_SOURCE_LINEAGE",
        "INVALID_PROGRAM_SHAPE",
        "REFUTED_DATE_NOT_REPRESENTABLE",
        "NO_DISJOINT_COMPANION",
        "NO_STATEFUL_COMPANION",
    ]
    details: dict[str, Any] = Field(default_factory=dict)


class CompositionalCandidateDiscovery(StrictModel):
    schema_version: Literal["patient-tool-gym.compositional-candidate-discovery.v2"] = (
        "patient-tool-gym.compositional-candidate-discovery.v2"
    )
    builder_version: Literal["patient-tool-gym.compositional-matrix-builder.v2"] = (
        COMPOSITIONAL_MATRIX_BUILDER_VERSION
    )
    seed: int
    input_task_count: int
    focal_family_count: int
    candidate_count: int
    candidates: tuple[CompositionalCandidate, ...]
    rejections: tuple[CompositionalCandidateRejection, ...]


class CompositionalTopologyBuild(StrictModel):
    schema_version: Literal["patient-tool-gym.compositional-topology-build.v2"] = (
        "patient-tool-gym.compositional-topology-build.v2"
    )
    superfamily_id: str
    topology: CompositionalTopology
    construction_type: CompositionalConstruction
    question_intent: CompositionalQuestionIntent
    question: str
    program: TaskProgram
    structural_complexity: ProgramComplexity


class CompositionalQuestionIntent(StrictModel):
    """Only the non-answer-bearing fields visible to the deterministic renderer."""

    schema_version: Literal["patient-tool-gym.compositional-question-intent.v1"] = (
        "patient-tool-gym.compositional-question-intent.v1"
    )
    topology: CompositionalTopology
    focal_phrase: str
    focal_vocabulary_label: str
    focal_date: str | None = None
    encounter_name: str | None = None
    companion_phrase: str | None = None
    companion_vocabulary_label: str | None = None
    companion_date: str | None = None


def discover_compositional_candidates(
    source_tasks: Sequence[TaskArtifact] | Iterable[TaskArtifact],
    *,
    seed: int = 20260711,
) -> CompositionalCandidateDiscovery:
    """Discover verified encounter-depth P2 families and deterministic companions.

    P2's encounter-depth admission rule already requires another record with the same
    clinical query in a different encounter. This function preserves that compiler contract
    by accepting only the exact three-node, runtime-bound P2 encounter program shape.
    """

    tasks = tuple(source_tasks)
    tasks_by_family: dict[str, list[TaskArtifact]] = {}
    for task in tasks:
        tasks_by_family.setdefault(task.family_id, []).append(task)

    atomic_pool = tuple(
        task
        for task in tasks
        if task.construction_type == "atomic"
        and _task_domain(task) in _DOMAIN_CONTRACTS
        and task.verification.passed
        and task.gold_answer == TypedValue(kind="boolean", value=True)
        and task.difficulty.dependency_depth == 1
        and task.difficulty.independent_width == 1
    )

    candidates: list[CompositionalCandidate] = []
    rejections: list[CompositionalCandidateRejection] = []
    focal_family_count = 0
    for family_id, family_tasks in sorted(tasks_by_family.items()):
        atomic = _only_construction(family_tasks, "atomic")
        depth = _only_construction(family_tasks, "depth")
        if depth is None:
            continue
        focal_family_count += 1
        if atomic is None:
            rejections.append(_rejection(family_id, "MISSING_ATOMIC_OR_DEPTH"))
            continue

        try:
            source_error = _source_family_error(atomic, depth)
        except ValueError as exc:
            source_error = ("INVALID_SOURCE_LINEAGE", {"reason": str(exc)})
        if source_error is not None:
            reason, details = source_error
            rejections.append(_rejection(family_id, reason, details))
            continue

        disjoint_companions = [
            task
            for task in atomic_pool
            if task.patient_snapshot_id == atomic.patient_snapshot_id
            and task.family_id != family_id
            and set(task.gold_evidence_ids).isdisjoint(depth.gold_evidence_ids)
            and _valid_companion(task)
        ]
        if not disjoint_companions:
            rejections.append(
                _rejection(
                    family_id,
                    "NO_DISJOINT_COMPANION",
                    {"patient_snapshot_id": atomic.patient_snapshot_id},
                )
            )
            continue
        # Exact-date Procedure claims cannot retain their selector date after a valid
        # not-done/unknown intervention removes performed[x].  The v2 width factorial
        # therefore admits only companions for which all four evidence states remain
        # executable under the unchanged selector.  This is an explicit eligibility
        # rule, not a silent fallback to a fixed-supported branch.
        companions = [
            task for task in disjoint_companions if _task_domain(task) != "procedure"
        ]
        if not companions:
            rejections.append(
                _rejection(
                    family_id,
                    "NO_STATEFUL_COMPANION",
                    {
                        "patient_snapshot_id": atomic.patient_snapshot_id,
                        "excluded_domains": sorted(
                            {_task_domain(task) for task in disjoint_companions}
                        ),
                        "reason": (
                            "every disjoint companion loses its exact-date selector under "
                            "at least one required evidence-state intervention"
                        ),
                    },
                )
            )
            continue
        companion = min(
            companions,
            key=lambda task: (
                _seeded_order(seed, family_id, task.task_id),
                task.task_id,
            ),
        )
        candidates.append(_candidate(atomic, depth, companion))

    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.superfamily_id))
    ordered_rejections = tuple(
        sorted(rejections, key=lambda item: (item.focal_family_id, item.reason))
    )
    return CompositionalCandidateDiscovery(
        seed=seed,
        input_task_count=len(tasks),
        focal_family_count=focal_family_count,
        candidate_count=len(ordered_candidates),
        candidates=ordered_candidates,
        rejections=ordered_rejections,
    )


def build_compositional_program(
    candidate: CompositionalCandidate,
    source_tasks: Mapping[str, TaskArtifact] | Sequence[TaskArtifact],
    topology: CompositionalTopology,
) -> TaskProgram:
    """Compile one typed program while keeping the clinical selectors fixed.

    Clinical searches deliberately omit the answer-bearing code and lifecycle status. The
    complete broad result is narrowed by an explicit ``ClinicalRecordSelector``. Depth variants
    first resolve the exact encounter and bind its evidence ID into the focal clinical search.
    """

    task_index = _task_index(source_tasks)
    focal_atomic = _bound_source_task(
        candidate, task_index, "focal_atomic", candidate.focal_atomic_task_id
    )
    focal_depth = _bound_source_task(
        candidate, task_index, "focal_depth", candidate.focal_depth_task_id
    )
    companion = _bound_source_task(candidate, task_index, "companion", candidate.companion_task_id)
    focal_contract = _DOMAIN_CONTRACTS[candidate.focal_domain]
    companion_contract = _DOMAIN_CONTRACTS[candidate.companion_domain]

    nodes: list[Any] = []
    if topology in {"d2w1", "d2w2"}:
        source_encounter = focal_depth.program.nodes[0]
        if not isinstance(source_encounter, ToolNode):  # defensive after discovery validation
            raise ValueError("focal depth source has no encounter tool node")
        nodes.append(
            ToolNode(
                id="focal_encounter_search",
                tool="search_encounters",
                arguments=dict(source_encounter.arguments),
            )
        )
        focal_arguments = _broad_arguments(
            focal_atomic,
            candidate.focal_selector,
            encounter_binding=binding("focal_encounter_search", "records", 0, "evidence_id"),
        )
    else:
        focal_arguments = _broad_arguments(focal_atomic, candidate.focal_selector)

    nodes.extend(
        (
            ToolNode(
                id="focal_search",
                tool=focal_contract[1],
                arguments=focal_arguments,
            ),
            EvidenceStateNode(
                id="focal_state",
                source=ValueRef(node_id="focal_search"),
                decision_rule_id=focal_contract[2],
                selector=candidate.focal_selector,
            ),
        )
    )

    branches = {"primary": ValueRef(node_id="focal_state")}
    if topology in {"d1w2", "d2w2"}:
        nodes.extend(
            (
                ToolNode(
                    id="companion_search",
                    tool=companion_contract[1],
                    arguments=_broad_arguments(companion, candidate.companion_selector),
                ),
                EvidenceStateNode(
                    id="companion_state",
                    source=ValueRef(node_id="companion_search"),
                    decision_rule_id=companion_contract[2],
                    selector=candidate.companion_selector,
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
    verify_topology_complexity(program, topology)
    return program


def build_compositional_question(
    candidate: CompositionalCandidate,
    topology: CompositionalTopology,
) -> str:
    """Render through the masked intent projection.

    Release compilation does not call this convenience wrapper.  It executes the
    typed program first, then creates the intent and calls
    :func:`render_compositional_question`.
    """

    return render_compositional_question(_compositional_question_intent(candidate, topology))


def render_compositional_question(intent: CompositionalQuestionIntent) -> str:
    """Render a question from clinical surface fields, without chart or answer-key access."""

    focal_phrase = f"{intent.focal_phrase} ({intent.focal_vocabulary_label})"
    focal_claim = (
        f"{focal_phrase} at my most recent {intent.encounter_name}"
        if intent.topology in {"d2w1", "d2w2"}
        else f"{focal_phrase} on {intent.focal_date}"
    )
    if intent.topology in {"d1w2", "d2w2"}:
        companion_phrase = (
            f"{intent.companion_phrase} ({intent.companion_vocabulary_label})"
        )
        companion_claim = f"{companion_phrase} on {intent.companion_date}"
        claims = (
            f"First claim (return this as branches.primary): my chart shows {focal_claim}. "
            f"Second claim (return this as branches.companion): my chart shows "
            f"{companion_claim}."
        )
    else:
        claims = (
            f"First claim (return this as branches.primary): my chart shows {focal_claim}."
        )
    branch_instruction = (
        "Return a JSON object with the evidence state for branches.primary and overall"
        if intent.topology in {"d1w1", "d2w1"}
        else (
            "Return a JSON object with the evidence states for branches.primary, "
            "branches.companion, and overall"
        )
    )
    return (
        f"Using my chart, classify these claims. {claims} {branch_instruction}. For each state, "
        "choose supported, refuted, insufficient evidence, or conflicting evidence."
    )


def _compositional_question_intent(
    candidate: CompositionalCandidate,
    topology: CompositionalTopology,
) -> CompositionalQuestionIntent:
    """Project a candidate to the renderer's non-identifying clinical surface."""

    is_depth = topology in {"d2w1", "d2w2"}
    has_companion = topology in {"d1w2", "d2w2"}
    return CompositionalQuestionIntent(
        topology=topology,
        focal_phrase=candidate.focal_phrase,
        focal_vocabulary_label=_selector_vocabulary_label(candidate.focal_selector),
        focal_date=(
            None if is_depth else _render_date(candidate.focal_selector.date_from)
        ),
        encounter_name=candidate.encounter_name if is_depth else None,
        companion_phrase=candidate.companion_phrase if has_companion else None,
        companion_vocabulary_label=(
            _selector_vocabulary_label(candidate.companion_selector)
            if has_companion
            else None
        ),
        companion_date=(
            _render_date(candidate.companion_selector.date_from)
            if has_companion
            else None
        ),
    )


def build_compositional_topology(
    candidate: CompositionalCandidate,
    source_tasks: Mapping[str, TaskArtifact] | Sequence[TaskArtifact],
    topology: CompositionalTopology,
    *,
    registry: ToolRegistry,
) -> CompositionalTopologyBuild:
    """Execute the typed program before projecting or rendering its question."""

    program = build_compositional_program(candidate, source_tasks, topology)
    ProgramExecutor(registry).execute(program)
    intent = _compositional_question_intent(candidate, topology)
    return CompositionalTopologyBuild(
        superfamily_id=candidate.superfamily_id,
        topology=topology,
        construction_type=_TOPOLOGY_CONSTRUCTIONS[topology],
        question_intent=intent,
        question=render_compositional_question(intent),
        program=program,
        structural_complexity=verify_topology_complexity(program, topology),
    )


def verify_topology_complexity(
    program: TaskProgram,
    topology: CompositionalTopology,
) -> ProgramComplexity:
    """Reject assigned topology labels that disagree with the executable graph."""

    complexity = analyze_complexity(program, frozenset(), None)
    expected_depth, expected_width, expected_answer_width = _TOPOLOGY_SHAPES[topology]
    observed = (
        complexity.dependency_depth,
        complexity.independent_width,
        complexity.answer_width,
    )
    expected = (expected_depth, expected_width, expected_answer_width)
    if observed != expected:
        raise ValueError(
            f"topology {topology} requires depth/width/answer-width {expected}, got {observed}"
        )
    return complexity


def _candidate(
    atomic: TaskArtifact,
    depth: TaskArtifact,
    companion: TaskArtifact,
) -> CompositionalCandidate:
    selected = _selected_family(atomic)
    companion_selected = _selected_family(companion)
    focal_domain = _task_domain(atomic)
    companion_domain = _task_domain(companion)
    focal_selector = _selector(atomic, selected)
    companion_selector = _selector(companion, companion_selected)
    focal_query = _primary_tool(atomic)
    companion_query = _primary_tool(companion)
    focal_contract = _DOMAIN_CONTRACTS[focal_domain]
    companion_contract = _DOMAIN_CONTRACTS[companion_domain]
    if focal_query.tool != focal_contract[1] or companion_query.tool != companion_contract[1]:
        raise ValueError("source task domain and primary search tool disagree")

    source_hashes = {
        "focal_atomic": content_hash(atomic.model_dump(mode="json")),
        "focal_depth": content_hash(depth.model_dump(mode="json")),
        "companion": content_hash(companion.model_dump(mode="json")),
    }
    identity = {
        "schema_version": COMPOSITIONAL_MATRIX_SCHEMA_VERSION,
        "focal_atomic_task_id": atomic.task_id,
        "focal_depth_task_id": depth.task_id,
        "companion_task_id": companion.task_id,
        "source_task_hashes": source_hashes,
    }
    digest = content_hash(identity).removeprefix("sha256:")
    return CompositionalCandidate(
        superfamily_id=f"compositional-family-{digest[:20]}",
        patient_id=_patient_id(str(selected["primary_evidence_id"])),
        patient_snapshot_id=atomic.patient_snapshot_id,
        focal_family_id=atomic.family_id,
        focal_domain=focal_domain,
        focal_resource_type=focal_contract[0],
        focal_atomic_task_id=atomic.task_id,
        focal_depth_task_id=depth.task_id,
        companion_family_id=companion.family_id,
        companion_domain=companion_domain,
        companion_resource_type=companion_contract[0],
        companion_task_id=companion.task_id,
        focal_evidence_id=str(selected["primary_evidence_id"]),
        routing_evidence_id=str(selected["depth_encounter_evidence_id"]),
        companion_evidence_ids=tuple(sorted(companion.gold_evidence_ids)),
        focal_selector=focal_selector,
        companion_selector=companion_selector,
        focal_phrase=_patient_phrase(selected),
        companion_phrase=_patient_phrase(companion_selected),
        encounter_name=str(selected["depth_encounter_name"]),
        source_task_hashes=source_hashes,
    )


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


def _source_family_error(
    atomic: TaskArtifact,
    depth: TaskArtifact,
) -> tuple[str, dict[str, Any]] | None:
    domain = _task_domain(atomic)
    if domain not in _DOMAIN_CONTRACTS:
        return "UNSUPPORTED_DOMAIN", {"domain": domain}
    if domain == "procedure":
        return "REFUTED_DATE_NOT_REPRESENTABLE", {
            "domain": domain,
            "reason": (
                "FHIR Procedure refutation removes performed[x], so an exact-date selector "
                "cannot distinguish refutation from missing dated evidence"
            ),
        }
    if not atomic.verification.passed or not depth.verification.passed:
        return "UNVERIFIED_SOURCE", {
            "atomic_verified": atomic.verification.passed,
            "depth_verified": depth.verification.passed,
        }
    positive = TypedValue(kind="boolean", value=True)
    if atomic.gold_answer != positive or depth.gold_answer != positive:
        return "SOURCE_NOT_POSITIVE", {}
    selected = _selected_family(atomic)
    depth_selected = _selected_family(depth)
    if selected != depth_selected or selected.get("depth_mode") != "encounter":
        return "UNSUPPORTED_DEPTH_MODE", {"depth_mode": selected.get("depth_mode")}
    if (
        atomic.patient_snapshot_id != depth.patient_snapshot_id
        or depth.parent_task_ids != (atomic.task_id,)
        or depth.metadata.get("lineage", {}).get("root_task_id") != atomic.task_id
        or atomic.provenance.generator != "patient_p2_compiler"
        or depth.provenance.generator != "patient_p2_compiler"
        or atomic.metadata.get("lineage", {}).get("template_id") != "patient_chart_presence_p2"
        or depth.metadata.get("lineage", {}).get("template_id") != "patient_chart_presence_p2"
    ):
        return "INVALID_SOURCE_LINEAGE", {}
    focal_id = selected.get("primary_evidence_id")
    encounter_id = selected.get("depth_encounter_evidence_id")
    if (
        not isinstance(focal_id, str)
        or not isinstance(encounter_id, str)
        or set(atomic.gold_evidence_ids) != {focal_id}
        or not {focal_id, encounter_id}.issubset(depth.gold_evidence_ids)
    ):
        return "INVALID_SOURCE_LINEAGE", {"reason": "evidence lineage mismatch"}
    try:
        _validate_atomic_shape(atomic)
        _validate_depth_shape(atomic, depth)
        _selector(atomic, selected)
    except ValueError as exc:
        return "INVALID_PROGRAM_SHAPE", {"reason": str(exc)}
    return None


def _valid_companion(task: TaskArtifact) -> bool:
    try:
        selected = _selected_family(task)
        _validate_atomic_shape(task)
        _selector(task, selected)
        _patient_phrase(selected)
    except ValueError:
        return False
    return (
        task.provenance.generator == "patient_p2_compiler"
        and task.metadata.get("lineage", {}).get("template_id") == "patient_chart_presence_p2"
        and len(task.gold_evidence_ids) == 1
    )


def _validate_atomic_shape(task: TaskArtifact) -> None:
    if len(task.program.nodes) != 2:
        raise ValueError("atomic source must contain one search and one exists node")
    search, reducer = task.program.nodes
    if not isinstance(search, ToolNode) or not isinstance(reducer, ExistsNode):
        raise ValueError("atomic source must contain one search and one exists node")
    if reducer.source != ValueRef(node_id=search.id, path=("records",)):
        raise ValueError("atomic exists node is not bound to its search records")
    if not isinstance(search.arguments.get("system"), str) or not isinstance(
        search.arguments.get("code"), str
    ):
        raise ValueError("atomic source requires an exact system and code")


def _validate_depth_shape(atomic: TaskArtifact, depth: TaskArtifact) -> None:
    if len(depth.program.nodes) != 3:
        raise ValueError("encounter depth source must contain two searches and one exists node")
    encounter, clinical, reducer = depth.program.nodes
    if (
        not isinstance(encounter, ToolNode)
        or encounter.tool != "search_encounters"
        or not isinstance(clinical, ToolNode)
        or clinical.tool != _primary_tool(atomic).tool
        or not isinstance(reducer, ExistsNode)
    ):
        raise ValueError("encounter depth source has an unexpected operation sequence")
    if node_references(clinical) != (
        ValueRef(node_id=encounter.id, path=("records", 0, "evidence_id")),
    ):
        raise ValueError("depth clinical search lacks the exact encounter evidence binding")
    if reducer.source != ValueRef(node_id=clinical.id, path=("records",)):
        raise ValueError("depth exists node is not bound to its clinical search")
    if depth.difficulty.dependency_depth != 2 or depth.difficulty.independent_width != 1:
        raise ValueError("depth source difficulty is not d2w1")


def _selector(task: TaskArtifact, selected: Mapping[str, Any]) -> ClinicalRecordSelector:
    search = _primary_tool(task)
    system = search.arguments.get("system")
    code = search.arguments.get("code")
    anchor_date = selected.get("anchor_date")
    if not all(isinstance(item, str) and item for item in (system, code, anchor_date)):
        raise ValueError("source selector requires system, code, and anchor_date")
    try:
        clinical_date = date.fromisoformat(anchor_date)
    except ValueError as exc:
        raise ValueError("source anchor_date is not an ISO date") from exc
    return ClinicalRecordSelector(
        system=system,
        code=code,
        date_from=clinical_date,
        date_to=clinical_date,
    )


def _broad_arguments(
    source_atomic: TaskArtifact,
    selector: ClinicalRecordSelector,
    *,
    encounter_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    search = _primary_tool(source_atomic)
    if search.tool not in {contract[1] for contract in _DOMAIN_CONTRACTS.values()}:
        raise ValueError("unsupported broad clinical search tool")
    arguments: dict[str, Any] = {
        "system": selector.system,
        "date_from": selector.date_from.isoformat() if selector.date_from else None,
        "date_to": selector.date_to.isoformat() if selector.date_to else None,
        "limit": 100,
    }
    if encounter_binding is not None:
        arguments["encounter_id"] = encounter_binding
    if any(
        key in arguments for key in ("code", "status", "clinical_status", "verification_status")
    ):
        raise ValueError("broad clinical searches must omit code and lifecycle status")
    return arguments


def _primary_tool(task: TaskArtifact) -> ToolNode:
    search = task.program.nodes[0] if task.program.nodes else None
    if not isinstance(search, ToolNode):
        raise ValueError("source atomic task has no primary tool")
    return search


def _selected_family(task: TaskArtifact) -> dict[str, Any]:
    selected = task.metadata.get("lineage", {}).get("selected_family")
    if not isinstance(selected, dict):
        raise ValueError("source task has no P2 selected-family metadata")
    return selected


def _task_domain(task: TaskArtifact) -> str:
    domain = task.metadata.get("lineage", {}).get("domain")
    return domain if isinstance(domain, str) else ""


def _patient_phrase(selected: Mapping[str, Any]) -> str:
    query_id = selected.get("primary_query_id")
    if not isinstance(query_id, str) or ":" not in query_id:
        raise ValueError("source task has no reviewed P2 query ID")
    # Patient-facing phrases are stable reviewed compiler vocabulary. Keeping the small
    # mapping local avoids importing P2's private renderer or parsing a rendered question.
    from patient_tool_gym.patient_p2 import get_p2_query

    reviewed = get_p2_query(query_id)
    if reviewed is None:
        raise ValueError(f"unknown reviewed P2 query: {query_id}")
    return reviewed.query.patient_phrase


def _patient_id(evidence_id: str) -> str:
    parts = evidence_id.split(":", 3)
    if len(parts) != 4 or not parts[1]:
        raise ValueError("P2 evidence ID does not identify a patient")
    return parts[1]


def _only_construction(
    tasks: Sequence[TaskArtifact],
    construction_type: str,
) -> TaskArtifact | None:
    selected = [task for task in tasks if task.construction_type == construction_type]
    return selected[0] if len(selected) == 1 else None


def _task_index(
    source_tasks: Mapping[str, TaskArtifact] | Sequence[TaskArtifact],
) -> dict[str, TaskArtifact]:
    if isinstance(source_tasks, Mapping):
        index = dict(source_tasks)
    else:
        index = {task.task_id: task for task in source_tasks}
        if len(index) != len(source_tasks):
            raise ValueError("source task collection contains duplicate task IDs")
    if any(task_id != task.task_id for task_id, task in index.items()):
        raise ValueError("source task index key disagrees with task_id")
    return index


def _bound_source_task(
    candidate: CompositionalCandidate,
    task_index: Mapping[str, TaskArtifact],
    role: str,
    task_id: str,
) -> TaskArtifact:
    task = task_index.get(task_id)
    if task is None:
        raise ValueError(f"candidate source task is missing: {task_id}")
    expected_hash = candidate.source_task_hashes.get(role)
    actual_hash = content_hash(task.model_dump(mode="json"))
    if expected_hash != actual_hash:
        raise ValueError(f"candidate {role} source task hash mismatch")
    return task


def _seeded_order(seed: int, focal_family_id: str, companion_task_id: str) -> str:
    value = f"{seed}:{focal_family_id}:{companion_task_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _rejection(
    family_id: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> CompositionalCandidateRejection:
    return CompositionalCandidateRejection(
        focal_family_id=family_id,
        reason=reason,
        details=details or {},
    )


def _render_date(value: date | None) -> str:
    if value is None:
        raise ValueError("compositional question requires an exact clinical date")
    return f"{value.strftime('%B')} {value.day}, {value.year}"


def _selector_vocabulary_label(selector: ClinicalRecordSelector) -> str:
    vocabulary = {
        "http://hl7.org/fhir/sid/cvx": "CVX",
        "http://loinc.org": "LOINC",
        "http://snomed.info/sct": "SNOMED CT",
    }.get(selector.system, selector.system)
    return f"{vocabulary} {selector.code}"
