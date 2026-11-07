from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

from patient_tool_gym.evidence import EvidenceRecord, PatientSnapshot
from patient_tool_gym.patient_v1 import (
    LOINC,
    SNOMED,
    PresenceQuery,
    SemanticPredicate,
    _complete_records,
    _dataset,
    _encounter_selector,
    _indefinite_article,
    _linked_encounter_ids,
    _patient_encounter_name,
    _render_date,
    atomic_presence_program,
    depth_presence_program,
    width_presence_program,
)
from patient_tool_gym.program import (
    ExistsNode,
    ProgramExecutionError,
    ProgramExecutor,
    StrictModel,
    TaskProgram,
    ToolNode,
    TypedValue,
    ValueRef,
    binding,
    content_hash,
    program_signature,
)
from patient_tool_gym.task import TaskArtifact, deterministic_task_id
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import compile_task, verify_task_artifact

TEMPLATE_ID = "patient_chart_presence_p2"
TEMPLATE_VERSION = "1"
GENERATOR = "patient_p2_compiler"

CVX = "http://hl7.org/fhir/sid/cvx"

P2Domain = Literal[
    "condition",
    "lab_observation",
    "procedure",
    "encounter",
    "care_plan",
    "immunization",
]
P2Stage = Literal["atomic", "depth", "width"]
P2RejectionReason = Literal[
    "UNSUPPORTED_CAPABILITY",
    "NO_REVIEWED_MATCH",
    "AMBIGUOUS_DATE_SCOPE",
    "NO_EXPLICIT_RELATION",
    "NO_SAFE_DEPTH",
    "NO_SAFE_WIDTH",
    "INCOMPLETE_SCOPE",
]


@dataclass(frozen=True)
class P2ReviewedQuery:
    query_id: str
    domain: P2Domain
    query: PresenceQuery


@dataclass(frozen=True)
class P2DomainSpec:
    domain: P2Domain
    resource_types: tuple[str, ...]
    queries: tuple[P2ReviewedQuery, ...]


class P2SelectedFamily(StrictModel):
    domain: P2Domain
    primary_query_id: str
    primary_evidence_id: str
    anchor_date: str
    depth_mode: Literal["encounter", "related", "lab_condition"] | None = None
    depth_encounter_evidence_id: str | None = None
    depth_encounter_name: str | None = None
    depth_encounter_arguments: dict[str, Any] | None = None
    related_query_id: str | None = None
    related_evidence_id: str | None = None
    related_relation: str | None = None
    width_query_id: str | None = None
    width_evidence_id: str | None = None
    width_domain: P2Domain | None = None
    width_relation: Literal["same_encounter"] | None = None
    width_encounter_evidence_id: str | None = None


class P2Rejection(StrictModel):
    patient_id: str
    snapshot_id: str
    domain: P2Domain
    stage: P2Stage
    query_id: str
    reason: P2RejectionReason
    attempt_index: int
    capability_supported: bool
    companion_query_id: str | None = None
    companion_domain: P2Domain | None = None


class P2Coverage(StrictModel):
    patient_id: str
    snapshot_id: str
    domain: P2Domain
    resource_types: tuple[str, ...]
    resource_count: int
    capability_supported: bool
    reviewed_query_count: int
    matched_query_count: int
    attempted_query_count: int
    selected_query_id: str | None
    generated_group_count: int
    complete_family_count: int
    core_depth2_family_count: int
    partial_group_count: int
    group_shape: Literal["atomic_only", "atomic_depth", "atomic_width", "complete"] | None
    task_count: int
    construction_types: tuple[Literal["atomic", "depth", "width"], ...]
    depth_mode: Literal["encounter", "related", "lab_condition"] | None
    width_scope: Literal["same_domain", "cross_domain"] | None
    width_companion_domain: P2Domain | None
    explicit_relation: str | None
    rejection_count: int


class P2GenerationResult(StrictModel):
    tasks: tuple[TaskArtifact, ...]
    rejected: tuple[P2Rejection, ...]
    coverage: tuple[P2Coverage, ...]

    @property
    def group_count(self) -> int:
        return sum(item.generated_group_count for item in self.coverage)

    @property
    def complete_family_count(self) -> int:
        """Return only complete atomic+depth+width matched families."""

        return sum(item.complete_family_count for item in self.coverage)

    @property
    def core_depth2_family_count(self) -> int:
        return sum(item.core_depth2_family_count for item in self.coverage)

    @property
    def partial_group_count(self) -> int:
        return sum(item.partial_group_count for item in self.coverage)


class P2DomainYield(StrictModel):
    domain: P2Domain
    supported_patients: int
    matched_queries: int
    attempted_queries: int
    generated_groups: int
    complete_families: int
    core_depth2_families: int
    partial_groups: int
    task_count: int
    construction_counts: dict[str, int]
    rejection_counts: dict[str, int]


class P2YieldSummary(StrictModel):
    patient_count: int
    generated_group_count: int
    complete_family_count: int
    core_depth2_family_count: int
    partial_group_count: int
    task_count: int
    domain_yields: tuple[P2DomainYield, ...]
    construction_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    cross_domain_composition_count: int
    rejection_counts: dict[str, int]


class _P2Ineligible(ValueError):
    def __init__(self, reason: P2RejectionReason) -> None:
        self.reason = reason
        super().__init__(reason)


def _query(
    *,
    query_id: str,
    domain: P2Domain,
    tool: str,
    resource_type: str,
    system: str,
    code: str,
    concept_phrase: str,
    patient_phrase: str,
    filters: tuple[tuple[str, str, str], ...] = (),
) -> P2ReviewedQuery:
    return P2ReviewedQuery(
        query_id=f"{domain}:{query_id}",
        domain=domain,
        query=PresenceQuery(
            tool=tool,
            resource_type=resource_type,
            arguments=(
                *(("system", system), ("code", code)),
                *((argument, value) for argument, value, _ in filters),
            ),
            concept_phrase=concept_phrase,
            patient_phrase=patient_phrase,
            semantic_predicates=tuple(
                SemanticPredicate(argument, value, phrase) for argument, value, phrase in filters
            ),
        ),
    )


def _condition(query_id: str, code: str, phrase: str) -> P2ReviewedQuery:
    return _query(
        query_id=query_id,
        domain="condition",
        tool="search_conditions",
        resource_type="Condition",
        system=SNOMED,
        code=code,
        concept_phrase=phrase,
        patient_phrase=phrase,
    )


def _lab(
    query_id: str,
    code: str,
    phrase: str,
    *,
    article: Literal["a", "an"] = "a",
) -> P2ReviewedQuery:
    return _query(
        query_id=query_id,
        domain="lab_observation",
        tool="search_observations",
        resource_type="Observation",
        system=LOINC,
        code=code,
        concept_phrase=phrase,
        patient_phrase=f"{article} {phrase}",
    )


def _procedure(query_id: str, code: str, phrase: str) -> P2ReviewedQuery:
    return _query(
        query_id=query_id,
        domain="procedure",
        tool="search_procedures",
        resource_type="Procedure",
        system=SNOMED,
        code=code,
        concept_phrase=phrase,
        patient_phrase=f"a completed {phrase}",
        filters=(("status", "completed", "completed"),),
    )


def _encounter(query_id: str, code: str, phrase: str) -> P2ReviewedQuery:
    return _query(
        query_id=query_id,
        domain="encounter",
        tool="search_encounters",
        resource_type="Encounter",
        system=SNOMED,
        code=code,
        concept_phrase=phrase,
        patient_phrase=f"a completed {phrase}",
        filters=(("status", "finished", "completed"),),
    )


def _care_plan(query_id: str, code: str, phrase: str) -> P2ReviewedQuery:
    return _query(
        query_id=query_id,
        domain="care_plan",
        tool="search_care_plans",
        resource_type="CarePlan",
        system=SNOMED,
        code=code,
        concept_phrase=phrase,
        patient_phrase=f"{_indefinite_article(phrase)} {phrase}",
    )


def _immunization(query_id: str, code: str, phrase: str) -> P2ReviewedQuery:
    return _query(
        query_id=query_id,
        domain="immunization",
        tool="search_immunizations",
        resource_type="Immunization",
        system=CVX,
        code=code,
        concept_phrase=phrase,
        patient_phrase=f"a completed {phrase}",
        filters=(("status", "completed", "completed"),),
    )


P2_DOMAIN_SPECS = (
    P2DomainSpec(
        domain="condition",
        resource_types=("Condition",),
        queries=(
            _condition("essential_hypertension", "59621000", "essential hypertension"),
            _condition("type_2_diabetes", "44054006", "type 2 diabetes"),
            _condition("prediabetes", "714628002", "prediabetes"),
            _condition("hyperlipidemia", "55822004", "hyperlipidemia"),
            _condition("heart_failure", "88805009", "heart failure"),
            _condition("full_thickness_burn", "403192003", "full-thickness burn"),
            _condition("ankle_fracture", "16114001", "ankle fracture"),
            _condition("stress", "73595000", "stress"),
            _condition("gingivitis", "66383009", "gingivitis"),
            _condition("viral_sinusitis", "444814009", "viral sinusitis"),
        ),
    ),
    P2DomainSpec(
        domain="lab_observation",
        resource_types=("Observation",),
        queries=(
            _lab("hemoglobin_a1c", "4548-4", "hemoglobin A1c result"),
            _lab("blood_glucose", "2339-0", "blood glucose result"),
            _lab("blood_creatinine", "38483-4", "blood creatinine result"),
            _lab("ldl_cholesterol", "18262-6", "LDL cholesterol result", article="an"),
            _lab("hemoglobin", "718-7", "hemoglobin result"),
            _lab("troponin_i", "89579-7", "high-sensitivity troponin I result"),
        ),
    ),
    P2DomainSpec(
        domain="procedure",
        resource_types=("Procedure",),
        queries=(
            _procedure("colonoscopy", "73761001", "colonoscopy"),
            _procedure("biopsy", "86273004", "biopsy"),
            _procedure("rehabilitation_therapy", "52052004", "rehabilitation therapy"),
            _procedure("hospice_care", "385763009", "hospice care"),
            _procedure("dental_care_referral", "103697008", "dental care referral"),
            _procedure("depression_screening", "171207006", "depression screening"),
            _procedure("medication_reconciliation", "430193006", "medication reconciliation"),
            _procedure("oral_health_education", "243085009", "oral health education"),
            _procedure("dental_consultation", "34043003", "dental consultation report"),
            _procedure("health_needs_assessment", "710824005", "health needs assessment"),
        ),
    ),
    P2DomainSpec(
        domain="encounter",
        resource_types=("Encounter",),
        queries=(
            _encounter("problem_visit", "185347001", "problem visit"),
            _encounter("general_examination", "162673000", "general examination visit"),
            _encounter("check_up", "185349003", "check-up visit"),
            _encounter("follow_up", "390906007", "follow-up visit"),
            _encounter("symptom_visit", "185345009", "symptom visit"),
            _encounter("well_child_visit", "410620009", "well-child visit"),
            _encounter("urgent_care", "702927004", "urgent-care visit"),
            _encounter("emergency_room", "50849002", "emergency-room admission"),
        ),
    ),
    P2DomainSpec(
        domain="care_plan",
        resource_types=("CarePlan",),
        queries=(
            _care_plan("hypertension_lifestyle", "443402002", "hypertension lifestyle plan"),
            _care_plan("diabetes_self_management", "735985000", "diabetes self-management plan"),
            _care_plan(
                "heart_failure_self_management",
                "735984001",
                "heart failure self-management plan",
            ),
            _care_plan("dialysis", "736690008", "dialysis care plan"),
            _care_plan("hyperlipidemia", "736285004", "hyperlipidemia management plan"),
            _care_plan("fracture", "385691007", "fracture care plan"),
            _care_plan("musculoskeletal", "408869004", "musculoskeletal care plan"),
            _care_plan("burn", "133901003", "burn care plan"),
            _care_plan("respiratory", "53950000", "respiratory therapy plan"),
            _care_plan("self_care", "384758001", "self-care plan"),
            _care_plan("wound_care", "225358003", "wound care plan"),
            _care_plan("inpatient", "736353004", "inpatient care plan"),
            _care_plan("discharge", "736372004", "discharge care plan"),
            _care_plan("general", "734163000", "care plan"),
            _care_plan("antenatal", "134435003", "antenatal care plan"),
        ),
    ),
    P2DomainSpec(
        domain="immunization",
        resource_types=("Immunization",),
        queries=(
            _immunization("influenza_trivalent", "140", "trivalent influenza immunization"),
            _immunization("covid_mrna", "207", "mRNA COVID-19 immunization"),
            _immunization("covid_pfizer", "208", "Pfizer COVID-19 immunization"),
            _immunization("dtap", "20", "DTaP immunization"),
            _immunization("pneumococcal_conjugate", "133", "PCV13 immunization"),
            _immunization("inactivated_polio", "10", "inactivated polio immunization"),
            _immunization("hpv_quadrivalent", "62", "quadrivalent HPV immunization"),
            _immunization("td_adult", "113", "adult Td immunization"),
            _immunization("tdap", "115", "Tdap immunization"),
            _immunization("meningococcal", "114", "meningococcal immunization"),
        ),
    ),
)

_DOMAIN_BY_ID = {spec.domain: spec for spec in P2_DOMAIN_SPECS}
_QUERY_BY_ID = {
    reviewed.query_id: reviewed for spec in P2_DOMAIN_SPECS for reviewed in spec.queries
}

_DISCOVERY_CACHE_MAX = 2048
_SEARCH_CACHE_MAX = 20000
_DISCOVERY_CACHE: OrderedDict[tuple[str, str, P2Domain], _DiscoveryOutcome] = OrderedDict()
_SEARCH_CACHE: OrderedDict[tuple[str, str, str, str], tuple[dict[str, Any], ...]] = OrderedDict()
_CACHE_LOCK = RLock()


def get_p2_query(query_id: str) -> P2ReviewedQuery | None:
    return _QUERY_BY_ID.get(query_id)


@dataclass(frozen=True)
class _DiscoveryOutcome:
    family: P2SelectedFamily | None
    matched_query_count: int
    attempted_query_count: int
    rejections: tuple[tuple[P2Stage, str, P2RejectionReason, str | None], ...]
    candidate_families: tuple[P2SelectedFamily, ...] = ()


@dataclass(frozen=True)
class _CandidateEvaluation:
    primary: P2ReviewedQuery
    record: EvidenceRecord
    depth: tuple[Any, ...] | None
    width: tuple[P2ReviewedQuery, EvidenceRecord, str] | None
    width_rejections: tuple[str, ...]

    @property
    def sibling_count(self) -> int:
        return int(self.depth is not None) + int(self.width is not None)

    @property
    def depth2_preference(self) -> int:
        return int(self.depth is not None and self.depth[0] != "lab_condition")


def related_presence_program(
    primary: PresenceQuery,
    related: PresenceQuery,
    clinical_date: str,
    relation: str,
) -> TaskProgram:
    primary_arguments = {
        **primary.tool_arguments(),
        "date_from": clinical_date,
        "date_to": clinical_date,
    }
    related_arguments = {
        "anchor_evidence_id": binding("primary_search", "records", 0, "evidence_id"),
        "relation": relation,
        "direction": "outgoing",
        "resource_type": related.resource_type,
        "system": dict(related.arguments).get("system"),
        "code": dict(related.arguments).get("code"),
        "limit": 100,
    }
    return TaskProgram(
        nodes=(
            ToolNode(id="primary_search", tool=primary.tool, arguments=primary_arguments),
            ToolNode(
                id="related_records",
                tool="find_related_records",
                arguments=related_arguments,
            ),
            ExistsNode(
                id="answer",
                source=ValueRef(node_id="related_records", path=("records",)),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def lab_condition_presence_program(
    primary: PresenceQuery,
    condition: PresenceQuery,
    clinical_date: str,
) -> TaskProgram:
    return TaskProgram(
        nodes=(
            ToolNode(
                id="primary_search",
                tool=primary.tool,
                arguments={
                    **primary.tool_arguments(),
                    "date_from": clinical_date,
                    "date_to": clinical_date,
                },
            ),
            ToolNode(
                id="anchor_encounter",
                tool="find_related_records",
                arguments={
                    "anchor_evidence_id": binding("primary_search", "records", 0, "evidence_id"),
                    "relation": "encounter",
                    "direction": "outgoing",
                    "resource_type": "Encounter",
                    "limit": 100,
                },
            ),
            ToolNode(
                id="encounter_conditions",
                tool="find_related_records",
                arguments={
                    "anchor_evidence_id": binding("anchor_encounter", "records", 0, "evidence_id"),
                    "relation": "encounter",
                    "direction": "incoming",
                    "resource_type": "Condition",
                    "system": dict(condition.arguments)["system"],
                    "code": dict(condition.arguments)["code"],
                    "limit": 100,
                },
            ),
            ExistsNode(
                id="answer",
                source=ValueRef(node_id="encounter_conditions", path=("records",)),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def generate_patient_p2_tasks(
    snapshots: tuple[PatientSnapshot, ...],
) -> P2GenerationResult:
    patient_ids = [snapshot.patient_id for snapshot in snapshots]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("duplicate patient snapshots are not allowed")

    tasks: list[TaskArtifact] = []
    rejected: list[P2Rejection] = []
    coverage: list[P2Coverage] = []
    for snapshot in sorted(snapshots, key=lambda item: (item.patient_id, item.snapshot_id)):
        registry = ToolRegistry(snapshot)
        for domain_spec in P2_DOMAIN_SPECS:
            resource_count = sum(
                len(snapshot.records_by_resource_type(resource_type))
                for resource_type in domain_spec.resource_types
            )
            outcome = _discover_domain(registry, domain_spec)
            for attempt_index, (stage, query_id, reason, companion_id) in enumerate(
                outcome.rejections,
                start=1,
            ):
                companion = get_p2_query(companion_id) if companion_id else None
                rejected.append(
                    P2Rejection(
                        patient_id=snapshot.patient_id,
                        snapshot_id=snapshot.snapshot_id,
                        domain=domain_spec.domain,
                        stage=stage,
                        query_id=query_id,
                        reason=reason,
                        attempt_index=attempt_index,
                        capability_supported=resource_count > 0,
                        companion_query_id=companion_id,
                        companion_domain=companion.domain if companion else None,
                    )
                )

            family_tasks: tuple[TaskArtifact, ...] = ()
            family: P2SelectedFamily | None = None
            compile_rejection_count = 0
            for candidate_index, candidate_family in enumerate(
                outcome.candidate_families,
                start=1,
            ):
                try:
                    family_tasks = compile_patient_p2_family(
                        snapshot,
                        candidate_family,
                    )
                except RuntimeError:
                    compile_rejection_count += 1
                    rejected.append(
                        P2Rejection(
                            patient_id=snapshot.patient_id,
                            snapshot_id=snapshot.snapshot_id,
                            domain=domain_spec.domain,
                            stage="atomic",
                            query_id=candidate_family.primary_query_id,
                            reason="INCOMPLETE_SCOPE",
                            attempt_index=len(outcome.rejections) + candidate_index,
                            capability_supported=True,
                        )
                    )
                    continue
                family = candidate_family
                tasks.extend(family_tasks)
                break
            constructions = tuple(
                task.construction_type
                for task in family_tasks
                if task.construction_type in {"atomic", "depth", "width"}
            )
            construction_set = set(constructions)
            complete_family = construction_set == {"atomic", "depth", "width"}
            depth_task = next(
                (task for task in family_tasks if task.construction_type == "depth"),
                None,
            )
            core_depth2_family = bool(
                complete_family
                and depth_task is not None
                and depth_task.difficulty.dependency_depth == 2
            )
            coverage.append(
                P2Coverage(
                    patient_id=snapshot.patient_id,
                    snapshot_id=snapshot.snapshot_id,
                    domain=domain_spec.domain,
                    resource_types=domain_spec.resource_types,
                    resource_count=resource_count,
                    capability_supported=resource_count > 0,
                    reviewed_query_count=len(domain_spec.queries),
                    matched_query_count=outcome.matched_query_count,
                    attempted_query_count=outcome.attempted_query_count,
                    selected_query_id=family.primary_query_id if family else None,
                    generated_group_count=int(family is not None),
                    complete_family_count=int(complete_family),
                    core_depth2_family_count=int(core_depth2_family),
                    partial_group_count=int(family is not None and not complete_family),
                    group_shape=_group_shape(construction_set),
                    task_count=len(family_tasks),
                    construction_types=constructions,
                    depth_mode=family.depth_mode if family else None,
                    width_scope=(
                        "cross_domain"
                        if family
                        and family.width_domain is not None
                        and family.width_domain != family.domain
                        else "same_domain"
                        if family and family.width_domain is not None
                        else None
                    ),
                    width_companion_domain=family.width_domain if family else None,
                    explicit_relation=(
                        "observation→encounter→condition"
                        if family and family.depth_mode == "lab_condition"
                        else family.related_relation
                        if family and family.depth_mode == "related"
                        else "encounter"
                        if family and family.depth_mode == "encounter"
                        else None
                    ),
                    rejection_count=len(outcome.rejections) + compile_rejection_count,
                )
            )
    return P2GenerationResult(
        tasks=tuple(tasks),
        rejected=tuple(rejected),
        coverage=tuple(coverage),
    )


def summarize_patient_p2_yield(result: P2GenerationResult) -> P2YieldSummary:
    construction_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for task in result.tasks:
        construction_counts[task.construction_type] = (
            construction_counts.get(task.construction_type, 0) + 1
        )
        difficulty_key = (
            f"depth={task.difficulty.dependency_depth},"
            f"width={task.difficulty.independent_width},"
            f"evidence={task.difficulty.evidence_width}"
        )
        difficulty_counts[difficulty_key] = difficulty_counts.get(difficulty_key, 0) + 1

    overall_rejections: dict[str, int] = {}
    for item in result.rejected:
        overall_rejections[item.reason] = overall_rejections.get(item.reason, 0) + 1

    domain_yields: list[P2DomainYield] = []
    for domain_spec in P2_DOMAIN_SPECS:
        domain = domain_spec.domain
        domain_coverage = [item for item in result.coverage if item.domain == domain]
        domain_tasks = [
            task
            for task in result.tasks
            if task.metadata.get("lineage", {}).get("domain") == domain
        ]
        domain_rejections = [item for item in result.rejected if item.domain == domain]
        domain_constructions: dict[str, int] = {}
        domain_rejection_counts: dict[str, int] = {}
        for task in domain_tasks:
            domain_constructions[task.construction_type] = (
                domain_constructions.get(task.construction_type, 0) + 1
            )
        for item in domain_rejections:
            domain_rejection_counts[item.reason] = domain_rejection_counts.get(item.reason, 0) + 1
        domain_yields.append(
            P2DomainYield(
                domain=domain,
                supported_patients=sum(item.capability_supported for item in domain_coverage),
                matched_queries=sum(item.matched_query_count for item in domain_coverage),
                attempted_queries=sum(item.attempted_query_count for item in domain_coverage),
                generated_groups=sum(item.generated_group_count for item in domain_coverage),
                complete_families=sum(item.complete_family_count for item in domain_coverage),
                core_depth2_families=sum(item.core_depth2_family_count for item in domain_coverage),
                partial_groups=sum(item.partial_group_count for item in domain_coverage),
                task_count=len(domain_tasks),
                construction_counts=dict(sorted(domain_constructions.items())),
                rejection_counts=dict(sorted(domain_rejection_counts.items())),
            )
        )

    return P2YieldSummary(
        patient_count=len({item.patient_id for item in result.coverage}),
        generated_group_count=result.group_count,
        complete_family_count=result.complete_family_count,
        core_depth2_family_count=result.core_depth2_family_count,
        partial_group_count=result.partial_group_count,
        task_count=len(result.tasks),
        domain_yields=tuple(domain_yields),
        construction_counts=dict(sorted(construction_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        cross_domain_composition_count=sum(
            item.width_scope == "cross_domain" or item.depth_mode in {"related", "lab_condition"}
            for item in result.coverage
        ),
        rejection_counts=dict(sorted(overall_rejections.items())),
    )


def _group_shape(
    construction_types: set[str],
) -> Literal["atomic_only", "atomic_depth", "atomic_width", "complete"] | None:
    if construction_types == {"atomic", "depth", "width"}:
        return "complete"
    if construction_types == {"atomic", "depth"}:
        return "atomic_depth"
    if construction_types == {"atomic", "width"}:
        return "atomic_width"
    if construction_types == {"atomic"}:
        return "atomic_only"
    return None


def _discover_domain(
    registry: ToolRegistry,
    domain_spec: P2DomainSpec,
) -> _DiscoveryOutcome:
    key = (TEMPLATE_VERSION, registry.snapshot_id, domain_spec.domain)
    with _CACHE_LOCK:
        cached = _DISCOVERY_CACHE.get(key)
        if cached is not None:
            _DISCOVERY_CACHE.move_to_end(key)
            return cached
    outcome = _discover_domain_uncached(registry, domain_spec)
    with _CACHE_LOCK:
        _DISCOVERY_CACHE[key] = outcome
        _DISCOVERY_CACHE.move_to_end(key)
        while len(_DISCOVERY_CACHE) > _DISCOVERY_CACHE_MAX:
            _DISCOVERY_CACHE.popitem(last=False)
    return outcome


def _discover_domain_uncached(
    registry: ToolRegistry,
    domain_spec: P2DomainSpec,
) -> _DiscoveryOutcome:
    snapshot = registry.snapshot
    resource_count = sum(
        len(snapshot.records_by_resource_type(resource_type))
        for resource_type in domain_spec.resource_types
    )
    if resource_count == 0:
        return _DiscoveryOutcome(
            family=None,
            matched_query_count=0,
            attempted_query_count=0,
            rejections=(
                (
                    "atomic",
                    domain_spec.queries[0].query_id,
                    "UNSUPPORTED_CAPABILITY",
                    None,
                ),
            ),
        )

    matched: list[tuple[P2ReviewedQuery, tuple[EvidenceRecord, ...]]] = []
    for reviewed in domain_spec.queries:
        try:
            records = _records_for_query(registry, reviewed)
        except ValueError:
            return _DiscoveryOutcome(
                family=None,
                matched_query_count=0,
                attempted_query_count=0,
                rejections=(("atomic", reviewed.query_id, "INCOMPLETE_SCOPE", None),),
            )
        if records:
            matched.append((reviewed, records))
    if not matched:
        return _DiscoveryOutcome(
            family=None,
            matched_query_count=0,
            attempted_query_count=0,
            rejections=(("atomic", domain_spec.queries[0].query_id, "NO_REVIEWED_MATCH", None),),
        )

    audit: list[tuple[P2Stage, str, P2RejectionReason, str | None]] = []
    candidates: list[_CandidateEvaluation] = []
    for reviewed, records in matched:
        anchors = _unambiguous_records(registry, reviewed, records)
        if not anchors:
            audit.append(("atomic", reviewed.query_id, "AMBIGUOUS_DATE_SCOPE", None))
            continue
        candidates.extend(_evaluate_candidate(registry, reviewed, anchor) for anchor in anchors)
    if not candidates:
        return _DiscoveryOutcome(
            family=None,
            matched_query_count=len(matched),
            attempted_query_count=len(matched),
            rejections=tuple(audit),
        )

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.sibling_count,
            -candidate.depth2_preference,
            -int(candidate.record.local_date.replace("-", "")),
            candidate.primary.query_id,
            candidate.record.evidence_id,
        ),
    )
    valid_candidates: list[tuple[_CandidateEvaluation, P2SelectedFamily]] = []
    for candidate in ordered_candidates:
        candidate_family = _candidate_family(domain_spec.domain, candidate)
        if _candidate_replays(registry, candidate_family):
            valid_candidates.append((candidate, candidate_family))
        else:
            audit.append(("atomic", candidate.primary.query_id, "INCOMPLETE_SCOPE", None))
    if not valid_candidates:
        return _DiscoveryOutcome(
            family=None,
            matched_query_count=len(matched),
            attempted_query_count=len(matched),
            rejections=tuple(audit),
        )
    selected, family = valid_candidates[0]
    primary = selected.primary
    depth = selected.depth
    if depth is None:
        audit.append(("depth", primary.query_id, "NO_SAFE_DEPTH", None))
    audit.extend(
        ("width", primary.query_id, "NO_SAFE_WIDTH", companion_id)
        for companion_id in selected.width_rejections
    )

    return _DiscoveryOutcome(
        family=family,
        matched_query_count=len(matched),
        attempted_query_count=len(matched),
        rejections=tuple(audit),
        candidate_families=tuple(item[1] for item in valid_candidates),
    )


def _records_for_query(
    registry: ToolRegistry,
    reviewed: P2ReviewedQuery,
) -> tuple[EvidenceRecord, ...]:
    views = _cached_complete_records(
        registry,
        reviewed.query.tool,
        reviewed.query.tool_arguments(),
    )
    records: list[EvidenceRecord] = []
    for view in views:
        evidence_id = view.get("evidence_id")
        record = (
            registry.snapshot.record_by_id(evidence_id) if isinstance(evidence_id, str) else None
        )
        if record is None or record.resource_type != reviewed.query.resource_type:
            raise _P2Ineligible("INCOMPLETE_SCOPE")
        records.append(record)
    return tuple(records)


def _cached_complete_records(
    registry: ToolRegistry,
    tool: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    key = (
        TEMPLATE_VERSION,
        registry.snapshot_id,
        tool,
        content_hash(arguments),
    )
    with _CACHE_LOCK:
        cached = _SEARCH_CACHE.get(key)
        if cached is not None:
            _SEARCH_CACHE.move_to_end(key)
            return copy.deepcopy(cached)
    records = _complete_records(registry, tool, arguments)
    with _CACHE_LOCK:
        _SEARCH_CACHE[key] = copy.deepcopy(records)
        _SEARCH_CACHE.move_to_end(key)
        while len(_SEARCH_CACHE) > _SEARCH_CACHE_MAX:
            _SEARCH_CACHE.popitem(last=False)
    return copy.deepcopy(records)


def _unambiguous_records(
    registry: ToolRegistry,
    reviewed: P2ReviewedQuery,
    records: tuple[EvidenceRecord, ...],
) -> tuple[EvidenceRecord, ...]:
    selected: list[EvidenceRecord] = []
    seen_dates: set[str] = set()
    for record in sorted(
        (item for item in records if item.local_date),
        key=lambda item: (item.local_date, item.evidence_id),
        reverse=True,
    ):
        if record.local_date in seen_dates:
            continue
        seen_dates.add(record.local_date)
        exact = _exact_date_records(registry, reviewed, record.local_date)
        if len(exact) == 1 and exact[0].evidence_id == record.evidence_id:
            selected.append(record)
    return tuple(selected)


def _evaluate_candidate(
    registry: ToolRegistry,
    primary: P2ReviewedQuery,
    primary_record: EvidenceRecord,
) -> _CandidateEvaluation:
    depth = _discover_lab_condition_depth(registry, primary, primary_record)
    if depth is None:
        depth = _discover_related_depth(registry, primary, primary_record)
    if depth is None:
        depth = _discover_encounter_depth(registry, primary, primary_record)

    width: tuple[P2ReviewedQuery, EvidenceRecord, str] | None = None
    width_rejections: list[str] = []
    for companion in _width_candidates(primary):
        found = _discover_width(registry, primary_record, companion)
        if found is None:
            width_rejections.append(companion.query_id)
            continue
        companion_record, encounter_id = found
        width = (companion, companion_record, encounter_id)
        break
    return _CandidateEvaluation(
        primary=primary,
        record=primary_record,
        depth=depth,
        width=width,
        width_rejections=tuple(width_rejections),
    )


def _candidate_family(
    domain: P2Domain,
    candidate: _CandidateEvaluation,
) -> P2SelectedFamily:
    depth = candidate.depth
    width = candidate.width
    return P2SelectedFamily(
        domain=domain,
        primary_query_id=candidate.primary.query_id,
        primary_evidence_id=candidate.record.evidence_id,
        anchor_date=candidate.record.local_date,
        depth_mode=depth[0] if depth else None,
        depth_encounter_evidence_id=(
            depth[1].evidence_id if depth and depth[0] in {"encounter", "lab_condition"} else None
        ),
        depth_encounter_name=(
            _patient_encounter_name(depth[1]) if depth and depth[0] == "encounter" else None
        ),
        depth_encounter_arguments=(
            _encounter_selector(depth[1]) if depth and depth[0] == "encounter" else None
        ),
        related_query_id=(
            depth[2].query_id if depth and depth[0] in {"related", "lab_condition"} else None
        ),
        related_evidence_id=(
            depth[1].evidence_id
            if depth and depth[0] == "related"
            else depth[3].evidence_id
            if depth and depth[0] == "lab_condition"
            else None
        ),
        related_relation=(
            "addresses"
            if depth and depth[0] == "related"
            else "encounter"
            if depth and depth[0] == "lab_condition"
            else None
        ),
        width_query_id=width[0].query_id if width else None,
        width_evidence_id=width[1].evidence_id if width else None,
        width_domain=width[0].domain if width else None,
        width_relation="same_encounter" if width else None,
        width_encounter_evidence_id=width[2] if width else None,
    )


def _candidate_replays(
    registry: ToolRegistry,
    selected: P2SelectedFamily,
) -> bool:
    construction_types = ["atomic"]
    if selected.depth_mode is not None:
        construction_types.append("depth")
    if selected.width_query_id is not None:
        construction_types.append("width")
    try:
        for construction_type in construction_types:
            replay = ProgramExecutor(registry).execute(_program(selected, construction_type))
            if replay.answer != TypedValue(kind="boolean", value=True):
                return False
            if set(replay.evidence_ids) != _expected_evidence(
                selected,
                construction_type,
            ):
                return False
            if not all(call.complete for call in replay.reference_trace):
                return False
    except (ProgramExecutionError, ValueError):
        return False
    return True


def _exact_date_records(
    registry: ToolRegistry,
    reviewed: P2ReviewedQuery,
    clinical_date: str,
) -> tuple[EvidenceRecord, ...]:
    views = _cached_complete_records(
        registry,
        reviewed.query.tool,
        {
            **reviewed.query.tool_arguments(),
            "date_from": clinical_date,
            "date_to": clinical_date,
        },
    )
    records: list[EvidenceRecord] = []
    for view in views:
        evidence_id = view.get("evidence_id")
        record = registry.snapshot.record_by_id(evidence_id) if evidence_id else None
        if record is None or record.resource_type != reviewed.query.resource_type:
            raise _P2Ineligible("INCOMPLETE_SCOPE")
        records.append(record)
    return tuple(records)


def _discover_encounter_depth(
    registry: ToolRegistry,
    primary: P2ReviewedQuery,
    primary_record: EvidenceRecord,
) -> tuple[Literal["encounter"], EvidenceRecord, None] | None:
    if primary.domain == "encounter":
        return None
    encounter_ids = _linked_encounter_ids(registry.snapshot, primary_record.evidence_id)
    if len(encounter_ids) != 1:
        return None
    encounter = registry.snapshot.record_by_id(encounter_ids[0])
    if (
        encounter is None
        or encounter.resource_type != "Encounter"
        or encounter.payload.get("status") != "finished"
        or encounter.local_date != primary_record.local_date
    ):
        return None
    encounter_arguments = _encounter_selector(encounter)
    selected_encounters = _cached_complete_records(
        registry,
        "search_encounters",
        encounter_arguments,
    )
    if tuple(view.get("evidence_id") for view in selected_encounters) != (encounter.evidence_id,):
        return None
    bound_views = _cached_complete_records(
        registry,
        primary.query.tool,
        {
            **primary.query.tool_arguments(),
            "encounter_id": encounter.evidence_id,
        },
    )
    if tuple(view.get("evidence_id") for view in bound_views) != (primary_record.evidence_id,):
        return None
    all_records = _records_for_query(registry, primary)
    if not any(
        record.evidence_id != primary_record.evidence_id
        and record.logical_evidence_group != primary_record.logical_evidence_group
        and _linked_encounter_ids(registry.snapshot, record.evidence_id) != encounter_ids
        for record in all_records
    ):
        return None
    return ("encounter", encounter, None)


def _discover_lab_condition_depth(
    registry: ToolRegistry,
    primary: P2ReviewedQuery,
    primary_record: EvidenceRecord,
) -> (
    tuple[
        Literal["lab_condition"],
        EvidenceRecord,
        P2ReviewedQuery,
        EvidenceRecord,
    ]
    | None
):
    if primary.domain != "lab_observation":
        return None
    encounter_ids = _linked_encounter_ids(registry.snapshot, primary_record.evidence_id)
    if len(encounter_ids) != 1:
        return None
    encounter = registry.snapshot.record_by_id(encounter_ids[0])
    if (
        encounter is None
        or encounter.resource_type != "Encounter"
        or encounter.payload.get("status") != "finished"
    ):
        return None
    direct = registry.call(
        "find_related_records",
        {
            "anchor_evidence_id": primary_record.evidence_id,
            "relation": "encounter",
            "direction": "outgoing",
            "resource_type": "Encounter",
            "limit": 100,
        },
    )
    if (
        direct.get("ok") is not True
        or direct.get("complete") is not True
        or tuple(item.get("evidence_id") for item in direct.get("records", []))
        != (encounter.evidence_id,)
    ):
        return None

    for condition in _DOMAIN_BY_ID["condition"].queries:
        arguments = dict(condition.query.arguments)
        related = registry.call(
            "find_related_records",
            {
                "anchor_evidence_id": encounter.evidence_id,
                "relation": "encounter",
                "direction": "incoming",
                "resource_type": "Condition",
                "system": arguments["system"],
                "code": arguments["code"],
                "limit": 100,
            },
        )
        records = related.get("records")
        if (
            related.get("ok") is not True
            or related.get("complete") is not True
            or not isinstance(records, list)
            or len(records) != 1
        ):
            continue
        condition_id = records[0].get("evidence_id")
        condition_record = (
            registry.snapshot.record_by_id(condition_id) if isinstance(condition_id, str) else None
        )
        if (
            condition_record is not None
            and condition_record.logical_evidence_group != primary_record.logical_evidence_group
        ):
            return ("lab_condition", encounter, condition, condition_record)
    return None


def _discover_related_depth(
    registry: ToolRegistry,
    primary: P2ReviewedQuery,
    primary_record: EvidenceRecord,
) -> tuple[Literal["related"], EvidenceRecord, P2ReviewedQuery] | None:
    if primary.domain != "care_plan":
        return None
    targets = {
        link.target_evidence_id
        for link in registry.snapshot.outgoing_links(primary_record.evidence_id)
        if link.relation == "addresses"
    }
    for related in _DOMAIN_BY_ID["condition"].queries:
        candidates = [
            registry.snapshot.record_by_id(evidence_id) for evidence_id in sorted(targets)
        ]
        candidates = [
            record
            for record in candidates
            if record is not None
            and record.resource_type == related.query.resource_type
            and _record_matches_query_code(record, related.query)
            and record.logical_evidence_group != primary_record.logical_evidence_group
        ]
        if len(candidates) != 1:
            continue
        result = registry.call(
            "find_related_records",
            {
                "anchor_evidence_id": primary_record.evidence_id,
                "relation": "addresses",
                "direction": "outgoing",
                "resource_type": related.query.resource_type,
                "system": dict(related.query.arguments)["system"],
                "code": dict(related.query.arguments)["code"],
                "limit": 100,
            },
        )
        if (
            result.get("ok") is True
            and result.get("complete") is True
            and tuple(item.get("evidence_id") for item in result.get("records", []))
            == (candidates[0].evidence_id,)
        ):
            return ("related", candidates[0], related)
    return None


def _width_candidates(primary: P2ReviewedQuery) -> tuple[P2ReviewedQuery, ...]:
    same_domain = tuple(
        reviewed
        for reviewed in _DOMAIN_BY_ID[primary.domain].queries
        if reviewed.query_id != primary.query_id
    )
    if primary.domain == "lab_observation":
        return (*_DOMAIN_BY_ID["condition"].queries, *same_domain)
    if primary.domain == "condition":
        return (*_DOMAIN_BY_ID["lab_observation"].queries, *same_domain)
    if primary.domain == "encounter":
        return (
            *_DOMAIN_BY_ID["lab_observation"].queries,
            *_DOMAIN_BY_ID["procedure"].queries,
            *_DOMAIN_BY_ID["immunization"].queries,
            *_DOMAIN_BY_ID["care_plan"].queries,
        )
    if primary.domain == "care_plan":
        return ()
    return same_domain


def _discover_width(
    registry: ToolRegistry,
    primary_record: EvidenceRecord,
    companion: P2ReviewedQuery,
) -> tuple[EvidenceRecord, str] | None:
    try:
        companion_records = _exact_date_records(
            registry,
            companion,
            primary_record.local_date,
        )
    except ValueError:
        return None
    if len(companion_records) != 1:
        return None
    companion_record = companion_records[0]
    if (
        companion_record.evidence_id == primary_record.evidence_id
        or companion_record.logical_evidence_group == primary_record.logical_evidence_group
    ):
        return None
    if primary_record.resource_type == "Encounter":
        primary_encounters = {primary_record.evidence_id}
    else:
        primary_encounters = set(
            _linked_encounter_ids(registry.snapshot, primary_record.evidence_id)
        )
    if companion_record.resource_type == "Encounter":
        companion_encounters = {companion_record.evidence_id}
    else:
        companion_encounters = set(
            _linked_encounter_ids(registry.snapshot, companion_record.evidence_id)
        )
    shared_encounters = primary_encounters & companion_encounters
    if len(shared_encounters) != 1:
        return None
    return companion_record, next(iter(shared_encounters))


def _record_matches_query_code(record: EvidenceRecord, query: PresenceQuery) -> bool:
    arguments = dict(query.arguments)
    code = arguments.get("code")
    system = arguments.get("system")
    if record.code.get("code") == code and record.code.get("system") == system:
        return True
    codings = record.code.get("codings")
    return isinstance(codings, list) and any(
        isinstance(coding, dict) and coding.get("code") == code and coding.get("system") == system
        for coding in codings
    )


def compile_patient_p2_family(
    snapshot: PatientSnapshot,
    selected: P2SelectedFamily,
) -> tuple[TaskArtifact, ...]:
    expected = _discover_domain(ToolRegistry(snapshot), _DOMAIN_BY_ID[selected.domain])
    if selected not in expected.candidate_families:
        raise ValueError("P2 compilation accepts only reproducibly discovered families")
    primary = _require_query(selected.primary_query_id)
    registry = ToolRegistry(snapshot)
    family_id = _family_id(snapshot, selected)
    dataset = _dataset(snapshot)

    atomic_program = atomic_presence_program(primary.query, selected.anchor_date)
    atomic = compile_task(
        registry=registry,
        program=atomic_program,
        family_id=family_id,
        construction_type="atomic",
        question=_question(selected, "atomic"),
        dataset=dataset,
        generator=GENERATOR,
        generator_version=TEMPLATE_VERSION,
        seed=0,
    )
    atomic = _attach_contract(
        atomic,
        registry,
        selected,
        sibling_role="atomic",
        root_task_id=atomic.task_id,
        derived_from_task_id=None,
    )
    tasks = [atomic]

    if selected.depth_mode is not None:
        depth_program = _program(selected, "depth")
        depth = compile_task(
            registry=registry,
            program=depth_program,
            family_id=family_id,
            construction_type="depth",
            question=_question(selected, "depth"),
            parent_task_ids=(atomic.task_id,),
            dataset=dataset,
            generator=GENERATOR,
            generator_version=TEMPLATE_VERSION,
            seed=0,
        )
        tasks.append(
            _attach_contract(
                depth,
                registry,
                selected,
                sibling_role="depth_2",
                root_task_id=atomic.task_id,
                derived_from_task_id=atomic.task_id,
            )
        )

    if selected.width_query_id is not None:
        width_program = _program(selected, "width")
        width = compile_task(
            registry=registry,
            program=width_program,
            family_id=family_id,
            construction_type="width",
            question=_question(selected, "width"),
            parent_task_ids=(atomic.task_id,),
            dataset=dataset,
            generator=GENERATOR,
            generator_version=TEMPLATE_VERSION,
            seed=0,
        )
        tasks.append(
            _attach_contract(
                width,
                registry,
                selected,
                sibling_role="width_2",
                root_task_id=atomic.task_id,
                derived_from_task_id=atomic.task_id,
            )
        )

    if not all(task.verification.passed for task in tasks):
        raise RuntimeError(f"P2 compiler emitted an invalid family for {selected.domain}")
    return tuple(tasks)


def _program(selected: P2SelectedFamily, construction_type: str) -> TaskProgram:
    primary = _require_query(selected.primary_query_id)
    if construction_type == "atomic":
        return atomic_presence_program(primary.query, selected.anchor_date)
    if construction_type == "depth" and selected.depth_mode == "encounter":
        if selected.depth_encounter_arguments is None:
            raise ValueError("encounter depth is missing selector arguments")
        return depth_presence_program(primary.query, selected.depth_encounter_arguments)
    if construction_type == "depth" and selected.depth_mode == "related":
        related = _require_query(selected.related_query_id)
        if selected.related_relation is None:
            raise ValueError("related depth is missing a relation")
        return related_presence_program(
            primary.query,
            related.query,
            selected.anchor_date,
            selected.related_relation,
        )
    if construction_type == "depth" and selected.depth_mode == "lab_condition":
        condition = _require_query(selected.related_query_id)
        return lab_condition_presence_program(
            primary.query,
            condition.query,
            selected.anchor_date,
        )
    if construction_type == "width":
        companion = _require_query(selected.width_query_id)
        return width_presence_program(
            primary.query,
            companion.query,
            selected.anchor_date,
        )
    raise ValueError("unsupported P2 construction")


def _question(selected: P2SelectedFamily, construction_type: str) -> str:
    primary = _require_query(selected.primary_query_id)
    rendered_date = _render_date(selected.anchor_date)
    if construction_type == "atomic":
        return f"Does my chart show {primary.query.patient_phrase} on {rendered_date}?"
    if construction_type == "depth" and selected.depth_mode == "encounter":
        if selected.depth_encounter_name is None:
            raise ValueError("encounter depth is missing a patient-facing name")
        return (
            f"Does my chart show {primary.query.patient_phrase} at my most recent "
            f"{selected.depth_encounter_name}?"
        )
    if construction_type == "depth" and selected.depth_mode == "related":
        related = _require_query(selected.related_query_id)
        return (
            f"Does my chart show {primary.query.patient_phrase} from {rendered_date} "
            f"that lists {related.query.patient_phrase} as a condition it addresses?"
        )
    if construction_type == "depth" and selected.depth_mode == "lab_condition":
        condition = _require_query(selected.related_query_id)
        return (
            f"Does my chart show {primary.query.patient_phrase} on {rendered_date} from "
            f"an encounter that also documents {condition.query.patient_phrase}?"
        )
    if construction_type == "width":
        companion = _require_query(selected.width_query_id)
        return (
            f"On {rendered_date}, does my chart show both {primary.query.patient_phrase} "
            f"and {companion.query.patient_phrase}?"
        )
    raise ValueError("unsupported P2 construction")


def _require_query(query_id: str | None) -> P2ReviewedQuery:
    query = get_p2_query(query_id) if isinstance(query_id, str) else None
    if query is None:
        raise ValueError("unknown P2 reviewed query")
    return query


def verify_patient_p2_contract(artifact: TaskArtifact, registry: ToolRegistry) -> bool:
    try:
        return _verify_patient_p2_contract(artifact, registry)
    except (KeyError, TypeError, ValueError, _P2Ineligible):
        return False


def _verify_patient_p2_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
) -> bool:
    if set(artifact.metadata) != {"lineage", "program_signature"}:
        return False
    lineage = artifact.metadata["lineage"]
    if not isinstance(lineage, dict):
        return False
    if lineage.get("template_id") != TEMPLATE_ID:
        return False
    domain = lineage.get("domain")
    if domain not in _DOMAIN_BY_ID:
        return False
    selected_payload = lineage.get("selected_family")
    if not isinstance(selected_payload, dict):
        return False
    selected = P2SelectedFamily.model_validate(selected_payload)
    expected = _discover_domain(registry, _DOMAIN_BY_ID[domain])
    if selected not in expected.candidate_families:
        return False

    expected_family_id = _family_id(registry.snapshot, selected)
    expected_program = _program(selected, artifact.construction_type)
    atomic_program = _program(selected, "atomic")
    expected_root_task_id = deterministic_task_id(
        registry.snapshot_id,
        atomic_program,
        expected_family_id,
        "atomic",
    )
    expected_parent_ids = () if artifact.construction_type == "atomic" else (expected_root_task_id,)
    expected_role = {
        "atomic": "atomic",
        "depth": "depth_2",
        "width": "width_2",
    }.get(artifact.construction_type)
    if expected_role is None:
        return False
    expected_lineage = _lineage(
        selected=selected,
        family_id=expected_family_id,
        sibling_role=expected_role,
        root_task_id=expected_root_task_id,
        derived_from_task_id=(
            None if artifact.construction_type == "atomic" else expected_root_task_id
        ),
    )
    expected_evidence = _expected_evidence(selected, artifact.construction_type)
    return all(
        (
            artifact.provenance.generator == GENERATOR,
            artifact.provenance.generator_version == TEMPLATE_VERSION,
            artifact.provenance.seed == 0,
            artifact.provenance.dataset == _dataset(registry.snapshot),
            artifact.family_id == expected_family_id,
            artifact.parent_task_ids == expected_parent_ids,
            artifact.program == expected_program,
            artifact.question == _question(selected, artifact.construction_type),
            artifact.gold_answer == TypedValue(kind="boolean", value=True),
            set(artifact.gold_evidence_ids) == expected_evidence,
            lineage == expected_lineage,
            artifact.metadata["program_signature"] == program_signature(expected_program),
        )
    )


def _expected_evidence(
    selected: P2SelectedFamily,
    construction_type: str,
) -> set[str]:
    if construction_type == "atomic":
        return {selected.primary_evidence_id}
    if construction_type == "depth" and selected.depth_mode == "encounter":
        if selected.depth_encounter_evidence_id is None:
            raise ValueError("missing depth encounter evidence")
        return {selected.primary_evidence_id, selected.depth_encounter_evidence_id}
    if construction_type == "depth" and selected.depth_mode == "related":
        if selected.related_evidence_id is None:
            raise ValueError("missing related evidence")
        return {selected.primary_evidence_id, selected.related_evidence_id}
    if construction_type == "depth" and selected.depth_mode == "lab_condition":
        if selected.depth_encounter_evidence_id is None or selected.related_evidence_id is None:
            raise ValueError("missing lab-condition path evidence")
        return {
            selected.primary_evidence_id,
            selected.depth_encounter_evidence_id,
            selected.related_evidence_id,
        }
    if construction_type == "width" and selected.width_evidence_id is not None:
        return {selected.primary_evidence_id, selected.width_evidence_id}
    raise ValueError("unsupported P2 evidence contract")


def _family_id(snapshot: PatientSnapshot, selected: P2SelectedFamily) -> str:
    digest = content_hash(
        {
            "snapshot_id": snapshot.snapshot_id,
            "template_id": TEMPLATE_ID,
            "template_version": TEMPLATE_VERSION,
            "selected_family": selected.model_dump(mode="json"),
        }
    ).removeprefix("sha256:")
    return f"patient-p2-family-{digest[:24]}"


def _lineage(
    *,
    selected: P2SelectedFamily,
    family_id: str,
    sibling_role: str,
    root_task_id: str,
    derived_from_task_id: str | None,
) -> dict[str, Any]:
    evidence_ids = {
        selected.primary_evidence_id,
        *((selected.depth_encounter_evidence_id,) if selected.depth_encounter_evidence_id else ()),
        *((selected.related_evidence_id,) if selected.related_evidence_id else ()),
        *((selected.width_evidence_id,) if selected.width_evidence_id else ()),
    }
    return {
        "family_id": family_id,
        "sibling_role": sibling_role,
        "root_task_id": root_task_id,
        "derived_from_task_id": derived_from_task_id,
        "template_id": TEMPLATE_ID,
        "template_version": TEMPLATE_VERSION,
        "domain": selected.domain,
        "selected_family": selected.model_dump(mode="json"),
        "anchor_evidence_ids": sorted(evidence_ids),
    }


def _attach_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    selected: P2SelectedFamily,
    *,
    sibling_role: str,
    root_task_id: str,
    derived_from_task_id: str | None,
) -> TaskArtifact:
    enriched = artifact.model_copy(
        update={
            "metadata": {
                "lineage": _lineage(
                    selected=selected,
                    family_id=artifact.family_id,
                    sibling_role=sibling_role,
                    root_task_id=root_task_id,
                    derived_from_task_id=derived_from_task_id,
                ),
                "program_signature": program_signature(artifact.program),
            }
        }
    )
    return enriched.model_copy(update={"verification": verify_task_artifact(enriched, registry)})
