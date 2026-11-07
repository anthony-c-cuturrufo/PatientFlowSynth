from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from patient_tool_gym.evidence import EvidenceRecord, PatientSnapshot
from patient_tool_gym.program import (
    AllNode,
    ExistsNode,
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

TEMPLATE_ID = "patient_chart_presence"
TEMPLATE_VERSION = "3"
GENERATOR = "patient_v1_compiler"

SearchTool = Literal[
    "search_encounters",
    "search_conditions",
    "search_observations",
    "search_medications",
    "search_procedures",
    "search_clinical_documents",
    "search_service_requests",
    "search_care_plans",
    "search_immunizations",
]

_CONCEPT_SELECTOR_ARGUMENTS = frozenset({"system", "code"})


def _normalize_patient_phrase(value: str) -> str:
    return " ".join(value.casefold().split())


def _contains_patient_phrase(container: str, phrase: str) -> bool:
    normalized_container = _normalize_patient_phrase(container)
    normalized_phrase = _normalize_patient_phrase(phrase)
    return (
        re.search(
            rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
            normalized_container,
        )
        is not None
    )


RejectionReason = Literal[
    "MISSING_ENCOUNTER",
    "AMBIGUOUS_ENCOUNTER",
    "MISSING_PRIMARY",
    "AMBIGUOUS_PRIMARY",
    "MISSING_COMPANION",
    "AMBIGUOUS_COMPANION",
    "NO_EXPLICIT_RELATION",
    "NO_DEPTH_DISAMBIGUATOR",
    "INCOMPLETE_SCOPE",
    "UNSUPPORTED_CAPABILITY",
    "NO_REVIEWED_PAIR",
]


@dataclass(frozen=True)
class SemanticPredicate:
    """One answer-changing query argument and its patient-facing realization."""

    argument: str
    value: str
    patient_phrase: str


@dataclass(frozen=True)
class PresenceQuery:
    tool: SearchTool
    resource_type: str
    arguments: tuple[tuple[str, str], ...]
    concept_phrase: str
    patient_phrase: str
    semantic_predicates: tuple[SemanticPredicate, ...]

    def __post_init__(self) -> None:
        argument_map = dict(self.arguments)
        if len(argument_map) != len(self.arguments):
            raise ValueError("presence-query arguments must be unique")
        if any(not key or not value for key, value in self.arguments):
            raise ValueError("presence-query arguments must be non-empty strings")
        if not self.concept_phrase.strip() or not self.patient_phrase.strip():
            raise ValueError("concept and patient phrases must not be empty")
        if not _contains_patient_phrase(self.patient_phrase, self.concept_phrase):
            raise ValueError("the concept phrase must appear in the patient phrase")

        expected = tuple(
            (argument, value)
            for argument, value in self.arguments
            if argument not in _CONCEPT_SELECTOR_ARGUMENTS
        )
        actual = tuple(
            (predicate.argument, predicate.value) for predicate in self.semantic_predicates
        )
        if actual != expected:
            raise ValueError(
                "every non-concept query argument must have one ordered semantic predicate"
            )

        normalized_patient_phrase = _normalize_patient_phrase(self.patient_phrase)
        for predicate in self.semantic_predicates:
            if (
                not predicate.argument
                or not predicate.value
                or not predicate.patient_phrase.strip()
            ):
                raise ValueError("semantic predicates must use non-empty strings")
            if not _contains_patient_phrase(
                normalized_patient_phrase,
                predicate.patient_phrase,
            ):
                raise ValueError(
                    "every semantic predicate phrase must appear in the patient phrase"
                )

    def tool_arguments(self) -> dict[str, Any]:
        return {**dict(self.arguments), "limit": 100}

    @property
    def patient_qualifiers(self) -> tuple[str, ...]:
        """Return patient-safe predicate phrases in their surface-text order."""

        unique = tuple(dict.fromkeys(item.patient_phrase for item in self.semantic_predicates))
        normalized_phrase = _normalize_patient_phrase(self.patient_phrase)
        return tuple(
            sorted(
                unique,
                key=lambda phrase: normalized_phrase.index(_normalize_patient_phrase(phrase)),
            )
        )


@dataclass(frozen=True)
class PresenceFamilySpec:
    spec_id: str
    primary: PresenceQuery
    companion: PresenceQuery


@dataclass(frozen=True)
class ReviewedPresenceQuery:
    """A clinician-readable, terminology-reviewed query available to V1."""

    query_id: str
    query: PresenceQuery


class PatientV1Anchor(StrictModel):
    encounter_evidence_id: str
    primary_evidence_id: str
    companion_evidence_id: str
    anchor_date: str
    encounter_name: str
    encounter_arguments: dict[str, Any]


class PatientV1Rejection(StrictModel):
    patient_id: str
    snapshot_id: str
    domain_spec_id: str
    spec_id: str
    reason: RejectionReason
    resource_types: tuple[str, ...]
    attempt_index: int
    capability_supported: bool


class PatientV1Coverage(StrictModel):
    """Per-patient, per-domain preflight and generation yield."""

    patient_id: str
    snapshot_id: str
    domain_spec_id: str
    resource_types: tuple[str, ...]
    resource_count: int
    capability_supported: bool
    reviewed_candidate_count: int
    matched_candidate_count: int
    attempted_candidate_count: int
    generated_family_count: int
    selected_spec_id: str | None
    rejection_count: int


class PatientV1GenerationResult(StrictModel):
    tasks: tuple[TaskArtifact, ...]
    rejected: tuple[PatientV1Rejection, ...]
    coverage: tuple[PatientV1Coverage, ...]

    @property
    def family_count(self) -> int:
        return len(self.tasks) // 3


class PatientV1YieldSummary(StrictModel):
    patient_count: int
    domain_count: int
    supported_domain_count: int
    generated_family_count: int
    task_count: int
    rejection_count: int
    rejection_counts: dict[str, int]


class _Ineligible(ValueError):
    def __init__(self, reason: RejectionReason) -> None:
        self.reason = reason
        super().__init__(reason)


SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
LOINC = "http://loinc.org"
_TERMINOLOGY_SUFFIXES = (
    " (environment)",
    " (finding)",
    " (observable entity)",
    " (procedure)",
    " (regime/therapy)",
    " (record artifact)",
    " (situation)",
)


def _indefinite_article(phrase: str) -> str:
    normalized = phrase.lstrip().casefold()
    return "an" if normalized and normalized[0] in "aeiou" else "a"


def _condition_query(
    query_id: str,
    code: str,
    concept_phrase: str,
    *,
    clinical_status: str | None = None,
    verification_status: str | None = None,
) -> ReviewedPresenceQuery:
    arguments: list[tuple[str, str]] = [("system", SNOMED), ("code", code)]
    predicates: list[SemanticPredicate] = []
    patient_phrase = concept_phrase
    if clinical_status:
        arguments.append(("clinical_status", clinical_status))
        status_phrase = f"marked {clinical_status}"
        predicates.append(SemanticPredicate("clinical_status", clinical_status, status_phrase))
        patient_phrase = f"{patient_phrase} {status_phrase}"
    if verification_status:
        arguments.append(("verification_status", verification_status))
        predicates.append(
            SemanticPredicate("verification_status", verification_status, verification_status)
        )
        patient_phrase = f"{patient_phrase} and {verification_status}"
    return ReviewedPresenceQuery(
        query_id=query_id,
        query=PresenceQuery(
            tool="search_conditions",
            resource_type="Condition",
            arguments=tuple(arguments),
            concept_phrase=concept_phrase,
            patient_phrase=patient_phrase,
            semantic_predicates=tuple(predicates),
        ),
    )


def _medication_query(
    query_id: str,
    code: str,
    concept_phrase: str,
    *,
    active_order: bool = False,
) -> ReviewedPresenceQuery:
    arguments: list[tuple[str, str]] = [
        ("record_kind", "request"),
        ("system", RXNORM),
        ("code", code),
    ]
    predicates = [SemanticPredicate("record_kind", "request", "order")]
    patient_phrase = f"{_indefinite_article(concept_phrase)} {concept_phrase}"
    if active_order:
        arguments.extend((("status", "active"), ("intent", "order")))
        predicates.extend(
            (
                SemanticPredicate("status", "active", "active"),
                SemanticPredicate("intent", "order", "order"),
            )
        )
        patient_phrase = f"an active {concept_phrase}"
    return ReviewedPresenceQuery(
        query_id=query_id,
        query=PresenceQuery(
            tool="search_medications",
            resource_type="MedicationRequest",
            arguments=tuple(arguments),
            concept_phrase=concept_phrase,
            patient_phrase=patient_phrase,
            semantic_predicates=tuple(predicates),
        ),
    )


def _procedure_query(
    query_id: str,
    code: str,
    concept_phrase: str,
) -> ReviewedPresenceQuery:
    return ReviewedPresenceQuery(
        query_id=query_id,
        query=PresenceQuery(
            tool="search_procedures",
            resource_type="Procedure",
            arguments=(("system", SNOMED), ("code", code), ("status", "completed")),
            concept_phrase=concept_phrase,
            patient_phrase=(
                f"{_indefinite_article(concept_phrase)} {concept_phrase} marked completed"
            ),
            semantic_predicates=(SemanticPredicate("status", "completed", "marked completed"),),
        ),
    )


def _document_query(
    query_id: str,
    resource_type: Literal["DocumentReference", "DiagnosticReport"],
    code: str,
    concept_phrase: str,
) -> ReviewedPresenceQuery:
    document_kind = (
        "document_reference" if resource_type == "DocumentReference" else "diagnostic_report"
    )
    article = _indefinite_article(concept_phrase)
    return ReviewedPresenceQuery(
        query_id=query_id,
        query=PresenceQuery(
            tool="search_clinical_documents",
            resource_type=resource_type,
            arguments=(
                ("document_kind", document_kind),
                ("system", LOINC),
                ("code", code),
            ),
            concept_phrase=concept_phrase,
            patient_phrase=f"{article} {concept_phrase}",
            semantic_predicates=(
                SemanticPredicate("document_kind", document_kind, concept_phrase),
            ),
        ),
    )


def _service_request_query(
    query_id: str,
    code: str,
    concept_phrase: str,
) -> ReviewedPresenceQuery:
    return ReviewedPresenceQuery(
        query_id=query_id,
        query=PresenceQuery(
            tool="search_service_requests",
            resource_type="ServiceRequest",
            arguments=(
                ("category", "referral"),
                ("system", SNOMED),
                ("code", code),
                ("status", "active"),
                ("intent", "order"),
                ("priority", "routine"),
            ),
            concept_phrase=concept_phrase,
            patient_phrase=f"an active {concept_phrase} with routine priority",
            semantic_predicates=(
                SemanticPredicate("category", "referral", "referral"),
                SemanticPredicate("status", "active", "active"),
                SemanticPredicate("intent", "order", "order"),
                SemanticPredicate("priority", "routine", "routine priority"),
            ),
        ),
    )


_REVIEWED_QUERIES_BY_DOMAIN: dict[str, tuple[ReviewedPresenceQuery, ...]] = {
    "condition": (
        _condition_query(
            "essential_hypertension_active_confirmed",
            "59621000",
            "essential hypertension",
            clinical_status="active",
            verification_status="confirmed",
        ),
        _condition_query(
            "type_2_diabetes_active_confirmed",
            "44054006",
            "type 2 diabetes",
            clinical_status="active",
            verification_status="confirmed",
        ),
        _condition_query(
            "full_time_employment_resolved",
            "160903007",
            "full-time employment status",
            clinical_status="resolved",
        ),
        _condition_query(
            "limited_social_contact_resolved",
            "423315002",
            "limited social contact",
            clinical_status="resolved",
        ),
        _condition_query("medication_review_due", "314529007", "medication review due"),
        _condition_query("stress", "73595000", "stress"),
        _condition_query("gingivitis", "66383009", "gingivitis"),
        _condition_query("viral_sinusitis", "444814009", "viral sinusitis"),
    ),
    "medication": (
        # RxNorm 29046 is the lisinopril ingredient, so its wording intentionally
        # makes no product-strength claim. Product concepts below use exact wording.
        _medication_query(
            "lisinopril_ingredient_active_order",
            "29046",
            "lisinopril order",
            active_order=True,
        ),
        _medication_query("metformin_order", "860975", "metformin order", active_order=True),
        _medication_query(
            "lisinopril_10_mg_tablet",
            "314076",
            "lisinopril 10 mg oral tablet order",
        ),
        _medication_query("humulin_70_30", "106892", "Humulin 70/30 injectable suspension order"),
        _medication_query(
            "envarsus_1_mg",
            "1664463",
            "Envarsus 1 mg extended-release tablet order",
        ),
        _medication_query("amlodipine_2_5_mg", "308136", "amlodipine 2.5 mg oral tablet order"),
        _medication_query("simvastatin_10_mg", "314231", "simvastatin 10 mg oral tablet order"),
        _medication_query("ibuprofen_200_mg", "310965", "ibuprofen 200 mg oral tablet order"),
        _medication_query("amoxicillin_500_mg", "308192", "amoxicillin 500 mg oral tablet order"),
    ),
    "procedure": (
        _procedure_query("colonoscopy", "73761001", "colonoscopy"),
        _procedure_query("biopsy", "86273004", "biopsy"),
        _procedure_query("dental_care_referral", "103697008", "dental-care referral"),
        _procedure_query(
            "substance_use_assessment",
            "428211000124100",
            "substance-use assessment",
        ),
        _procedure_query("depression_screening", "171207006", "depression screening"),
        _procedure_query("medication_reconciliation", "430193006", "medication reconciliation"),
        _procedure_query("oral_health_education", "243085009", "oral-health education"),
        _procedure_query(
            "dental_consultation_report",
            "34043003",
            "dental consultation and report",
        ),
        _procedure_query(
            "health_social_care_assessment",
            "710824005",
            "health and social-care needs assessment",
        ),
    ),
    "clinical_document": (
        # LOINC 34117-2 is History and physical note, not an after-visit summary.
        # No current/superseded filter is used: status is lifecycle metadata here,
        # not part of the patient question or evidence identity.
        _document_query(
            "history_physical_note",
            "DocumentReference",
            "34117-2",
            "history and physical note",
        ),
        _document_query("progress_note", "DocumentReference", "11506-3", "progress note"),
        _document_query(
            "basic_metabolic_panel_report",
            "DiagnosticReport",
            "51990-0",
            "basic metabolic panel report",
        ),
        _document_query(
            "lipid_panel_report",
            "DiagnosticReport",
            "57698-3",
            "lipid panel report",
        ),
        _document_query(
            "cbc_panel_report",
            "DiagnosticReport",
            "58410-2",
            "complete blood count panel report",
        ),
        _document_query(
            "phq_2_report",
            "DiagnosticReport",
            "55757-9",
            "PHQ-2 questionnaire report",
        ),
        _document_query(
            "morse_fall_scale_report",
            "DiagnosticReport",
            "59453-1",
            "Morse Fall Scale report",
        ),
        _document_query(
            "gad_7_report",
            "DiagnosticReport",
            "69737-5",
            "GAD-7 questionnaire report",
        ),
    ),
    "service_request": (
        _service_request_query(
            "cardiology_referral",
            "306206005",
            "cardiology referral order",
        ),
        _service_request_query(
            "dietitian_referral",
            "306188006",
            "dietitian referral order",
        ),
    ),
}

_CANONICAL_PAIR_IDS = {
    (
        "condition",
        "essential_hypertension_active_confirmed",
        "type_2_diabetes_active_confirmed",
    ): "condition_documented_presence",
    (
        "medication",
        "lisinopril_ingredient_active_order",
        "metformin_order",
    ): "medication_order_presence",
    ("procedure", "colonoscopy", "biopsy"): "procedure_completed_presence",
    (
        "clinical_document",
        "history_physical_note",
        "progress_note",
    ): "clinical_document_presence",
    (
        "service_request",
        "cardiology_referral",
        "dietitian_referral",
    ): "referral_request_presence",
    (
        "condition",
        "full_time_employment_resolved",
        "limited_social_contact_resolved",
    ): "synthea_employment_status_presence",
    ("medication", "humulin_70_30", "envarsus_1_mg"): "synthea_insulin_order_presence",
    (
        "procedure",
        "dental_care_referral",
        "substance_use_assessment",
    ): "synthea_dental_referral_presence",
}


def _build_allowlisted_specs() -> tuple[PresenceFamilySpec, ...]:
    specs: list[PresenceFamilySpec] = []
    for domain, reviewed_queries in _REVIEWED_QUERIES_BY_DOMAIN.items():
        for primary in reviewed_queries:
            for companion in reviewed_queries:
                if primary.query_id == companion.query_id:
                    continue
                spec_id = _CANONICAL_PAIR_IDS.get(
                    (domain, primary.query_id, companion.query_id),
                    f"reviewed_{domain}__{primary.query_id}__{companion.query_id}",
                )
                specs.append(
                    PresenceFamilySpec(
                        spec_id=spec_id,
                        primary=primary.query,
                        companion=companion.query,
                    )
                )
    return tuple(sorted(specs, key=lambda spec: spec.spec_id))


_ALL_FIXED_SPECS = _build_allowlisted_specs()
_FIXED_SPECS_BY_ID = {spec.spec_id: spec for spec in _ALL_FIXED_SPECS}

_PUBLIC_SPEC_IDS = (
    "condition_documented_presence",
    "medication_order_presence",
    "procedure_completed_presence",
    "clinical_document_presence",
    "referral_request_presence",
)
PATIENT_V1_FAMILY_SPECS = tuple(_FIXED_SPECS_BY_ID[spec_id] for spec_id in _PUBLIC_SPEC_IDS)

_SYNTHEA_SAMPLE_SPEC_IDS = (
    "synthea_employment_status_presence",
    "synthea_insulin_order_presence",
    "synthea_dental_referral_presence",
)
_SYNTHEA_SAMPLE_SPECS = tuple(_FIXED_SPECS_BY_ID[spec_id] for spec_id in _SYNTHEA_SAMPLE_SPEC_IDS)

_DOMAIN_RESOURCE_TYPES = {
    "condition": ("Condition",),
    "medication": ("MedicationRequest",),
    "procedure": ("Procedure",),
    "clinical_document": ("DocumentReference", "DiagnosticReport"),
    "service_request": ("ServiceRequest",),
}
_DOMAIN_DEFAULT_SPEC_ID = {
    "condition": "condition_documented_presence",
    "medication": "medication_order_presence",
    "procedure": "procedure_completed_presence",
    "clinical_document": "clinical_document_presence",
    "service_request": "referral_request_presence",
}
_SPEC_CANDIDATES_BY_DOMAIN = {
    domain: tuple(
        spec
        for spec in _ALL_FIXED_SPECS
        if spec.primary in {reviewed.query for reviewed in _REVIEWED_QUERIES_BY_DOMAIN[domain]}
        and spec.companion in {reviewed.query for reviewed in _REVIEWED_QUERIES_BY_DOMAIN[domain]}
    )
    for domain in _REVIEWED_QUERIES_BY_DOMAIN
}


def get_patient_v1_family_spec(spec_id: str) -> PresenceFamilySpec | None:
    """Return one immutable, versioned V1 specification for safe local projection."""

    return _FIXED_SPECS_BY_ID.get(spec_id)


def atomic_presence_program(query: PresenceQuery, clinical_date: str) -> TaskProgram:
    arguments = {
        **query.tool_arguments(),
        "date_from": clinical_date,
        "date_to": clinical_date,
    }
    return TaskProgram(
        nodes=(
            ToolNode(id="primary_search", tool=query.tool, arguments=arguments),
            ExistsNode(
                id="answer",
                source=ValueRef(node_id="primary_search", path=("records",)),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def depth_presence_program(
    query: PresenceQuery,
    encounter_arguments: dict[str, Any],
) -> TaskProgram:
    arguments = {
        **query.tool_arguments(),
        "encounter_id": binding("anchor_encounter", "records", 0, "evidence_id"),
    }
    return TaskProgram(
        nodes=(
            ToolNode(
                id="anchor_encounter",
                tool="search_encounters",
                arguments=encounter_arguments,
            ),
            ToolNode(id="encounter_records", tool=query.tool, arguments=arguments),
            ExistsNode(
                id="answer",
                source=ValueRef(node_id="encounter_records", path=("records",)),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def width_presence_program(
    primary: PresenceQuery,
    companion: PresenceQuery,
    clinical_date: str,
) -> TaskProgram:
    primary_arguments = {
        **primary.tool_arguments(),
        "date_from": clinical_date,
        "date_to": clinical_date,
    }
    companion_arguments = {
        **companion.tool_arguments(),
        "date_from": clinical_date,
        "date_to": clinical_date,
    }
    return TaskProgram(
        nodes=(
            ToolNode(
                id="primary_search",
                tool=primary.tool,
                arguments=primary_arguments,
            ),
            ExistsNode(
                id="primary_exists",
                source=ValueRef(node_id="primary_search", path=("records",)),
            ),
            ToolNode(
                id="companion_search",
                tool=companion.tool,
                arguments=companion_arguments,
            ),
            ExistsNode(
                id="companion_exists",
                source=ValueRef(node_id="companion_search", path=("records",)),
            ),
            AllNode(
                id="answer",
                sources=(
                    ValueRef(node_id="primary_exists"),
                    ValueRef(node_id="companion_exists"),
                ),
            ),
        ),
        output=ValueRef(node_id="answer"),
    )


def compile_patient_v1_family(
    snapshot: PatientSnapshot,
    spec: PresenceFamilySpec,
) -> tuple[TaskArtifact, TaskArtifact, TaskArtifact]:
    if _FIXED_SPECS_BY_ID.get(spec.spec_id) != spec:
        raise ValueError("V1 compilation accepts only versioned allowlisted specifications")
    registry = ToolRegistry(snapshot)
    anchor = _discover_anchor(registry, spec)
    family_id = _family_id(snapshot, spec, anchor)
    dataset = _dataset(snapshot)

    atomic = compile_task(
        registry=registry,
        program=atomic_presence_program(spec.primary, anchor.anchor_date),
        family_id=family_id,
        construction_type="atomic",
        question=_question(spec, anchor, "atomic"),
        dataset=dataset,
        generator=GENERATOR,
        generator_version=TEMPLATE_VERSION,
        seed=0,
    )
    atomic = _attach_contract(
        atomic,
        registry,
        spec,
        anchor,
        sibling_role="atomic",
        root_task_id=atomic.task_id,
        derived_from_task_id=None,
    )

    depth = compile_task(
        registry=registry,
        program=depth_presence_program(spec.primary, anchor.encounter_arguments),
        family_id=family_id,
        construction_type="depth",
        question=_question(spec, anchor, "depth"),
        parent_task_ids=(atomic.task_id,),
        dataset=dataset,
        generator=GENERATOR,
        generator_version=TEMPLATE_VERSION,
        seed=0,
    )
    depth = _attach_contract(
        depth,
        registry,
        spec,
        anchor,
        sibling_role="depth_2",
        root_task_id=atomic.task_id,
        derived_from_task_id=atomic.task_id,
    )

    width = compile_task(
        registry=registry,
        program=width_presence_program(
            spec.primary,
            spec.companion,
            anchor.anchor_date,
        ),
        family_id=family_id,
        construction_type="width",
        question=_question(spec, anchor, "width"),
        parent_task_ids=(atomic.task_id,),
        dataset=dataset,
        generator=GENERATOR,
        generator_version=TEMPLATE_VERSION,
        seed=0,
    )
    width = _attach_contract(
        width,
        registry,
        spec,
        anchor,
        sibling_role="width_2",
        root_task_id=atomic.task_id,
        derived_from_task_id=atomic.task_id,
    )

    family = (atomic, depth, width)
    if not all(task.verification.passed for task in family):
        raise RuntimeError(f"V1 compiler emitted an invalid family for {spec.spec_id}")
    return family


def generate_patient_v1_families(
    snapshots: tuple[PatientSnapshot, ...],
) -> PatientV1GenerationResult:
    patient_ids = [snapshot.patient_id for snapshot in snapshots]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("duplicate patient snapshots are not allowed")

    tasks: list[TaskArtifact] = []
    rejections: list[PatientV1Rejection] = []
    coverage: list[PatientV1Coverage] = []
    for snapshot in sorted(snapshots, key=lambda item: (item.patient_id, item.snapshot_id)):
        registry = ToolRegistry(snapshot)
        for domain in _REVIEWED_QUERIES_BY_DOMAIN:
            resource_types = _DOMAIN_RESOURCE_TYPES[domain]
            default_spec_id = _DOMAIN_DEFAULT_SPEC_ID[domain]
            resource_count = sum(
                len(snapshot.records_by_resource_type(resource_type))
                for resource_type in resource_types
            )
            domain_rejections: list[PatientV1Rejection] = []
            if resource_count == 0:
                rejection = PatientV1Rejection(
                    patient_id=snapshot.patient_id,
                    snapshot_id=snapshot.snapshot_id,
                    domain_spec_id=default_spec_id,
                    spec_id=default_spec_id,
                    reason="UNSUPPORTED_CAPABILITY",
                    resource_types=resource_types,
                    attempt_index=0,
                    capability_supported=False,
                )
                rejections.append(rejection)
                coverage.append(
                    PatientV1Coverage(
                        patient_id=snapshot.patient_id,
                        snapshot_id=snapshot.snapshot_id,
                        domain_spec_id=default_spec_id,
                        resource_types=resource_types,
                        resource_count=0,
                        capability_supported=False,
                        reviewed_candidate_count=len(_SPEC_CANDIDATES_BY_DOMAIN[domain]),
                        matched_candidate_count=0,
                        attempted_candidate_count=0,
                        generated_family_count=0,
                        selected_spec_id=None,
                        rejection_count=1,
                    )
                )
                continue

            candidates = _matched_candidate_specs(registry, domain)
            if not candidates:
                rejection = PatientV1Rejection(
                    patient_id=snapshot.patient_id,
                    snapshot_id=snapshot.snapshot_id,
                    domain_spec_id=default_spec_id,
                    spec_id=default_spec_id,
                    reason="NO_REVIEWED_PAIR",
                    resource_types=resource_types,
                    attempt_index=0,
                    capability_supported=True,
                )
                rejections.append(rejection)
                coverage.append(
                    PatientV1Coverage(
                        patient_id=snapshot.patient_id,
                        snapshot_id=snapshot.snapshot_id,
                        domain_spec_id=default_spec_id,
                        resource_types=resource_types,
                        resource_count=resource_count,
                        capability_supported=True,
                        reviewed_candidate_count=len(_SPEC_CANDIDATES_BY_DOMAIN[domain]),
                        matched_candidate_count=0,
                        attempted_candidate_count=0,
                        generated_family_count=0,
                        selected_spec_id=None,
                        rejection_count=1,
                    )
                )
                continue

            selected_spec_id: str | None = None
            attempted_count = 0
            for attempted_count, candidate in enumerate(candidates, start=1):
                try:
                    tasks.extend(compile_patient_v1_family(snapshot, candidate))
                    selected_spec_id = candidate.spec_id
                    break
                except _Ineligible as exc:
                    domain_rejections.append(
                        PatientV1Rejection(
                            patient_id=snapshot.patient_id,
                            snapshot_id=snapshot.snapshot_id,
                            domain_spec_id=default_spec_id,
                            spec_id=candidate.spec_id,
                            reason=exc.reason,
                            resource_types=resource_types,
                            attempt_index=attempted_count,
                            capability_supported=True,
                        )
                    )
            rejections.extend(domain_rejections)
            coverage.append(
                PatientV1Coverage(
                    patient_id=snapshot.patient_id,
                    snapshot_id=snapshot.snapshot_id,
                    domain_spec_id=default_spec_id,
                    resource_types=resource_types,
                    resource_count=resource_count,
                    capability_supported=True,
                    reviewed_candidate_count=len(_SPEC_CANDIDATES_BY_DOMAIN[domain]),
                    matched_candidate_count=len(candidates),
                    attempted_candidate_count=attempted_count,
                    generated_family_count=int(selected_spec_id is not None),
                    selected_spec_id=selected_spec_id,
                    rejection_count=len(domain_rejections),
                )
            )
    return PatientV1GenerationResult(
        tasks=tuple(tasks),
        rejected=tuple(rejections),
        coverage=tuple(coverage),
    )


def summarize_patient_v1_yield(
    result: PatientV1GenerationResult,
) -> PatientV1YieldSummary:
    """Aggregate generation output without discarding candidate-level failures."""

    rejection_counts: dict[str, int] = {}
    for rejection in result.rejected:
        rejection_counts[rejection.reason] = rejection_counts.get(rejection.reason, 0) + 1
    return PatientV1YieldSummary(
        patient_count=len({item.patient_id for item in result.coverage}),
        domain_count=len(result.coverage),
        supported_domain_count=sum(item.capability_supported for item in result.coverage),
        generated_family_count=result.family_count,
        task_count=len(result.tasks),
        rejection_count=len(result.rejected),
        rejection_counts=dict(sorted(rejection_counts.items())),
    )


def _matched_candidate_specs(
    registry: ToolRegistry,
    domain: str,
) -> tuple[PresenceFamilySpec, ...]:
    """Select only pairs whose reviewed concepts occur, ordered deterministically by yield."""

    query_counts: dict[PresenceQuery, int] = {}
    for reviewed in _REVIEWED_QUERIES_BY_DOMAIN[domain]:
        result = registry.call(reviewed.query.tool, reviewed.query.tool_arguments())
        count = result.get("total_matches")
        query_counts[reviewed.query] = count if result.get("ok") and isinstance(count, int) else 0
    candidates = [
        spec
        for spec in _SPEC_CANDIDATES_BY_DOMAIN[domain]
        if query_counts[spec.primary] > 0 and query_counts[spec.companion] > 0
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda spec: (
                -query_counts[spec.primary],
                -query_counts[spec.companion],
                spec.spec_id,
            ),
        )
    )


def verify_patient_v1_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
) -> bool:
    try:
        return _verify_patient_v1_contract(artifact, registry)
    except (KeyError, TypeError, ValueError, _Ineligible):
        return False


def _verify_patient_v1_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
) -> bool:
    if set(artifact.metadata) != {"lineage", "program_signature"}:
        return False
    lineage = artifact.metadata["lineage"]
    if not isinstance(lineage, dict):
        return False
    spec_id = lineage.get("spec_id")
    if not isinstance(spec_id, str):
        return False
    spec = _FIXED_SPECS_BY_ID.get(spec_id)
    if spec is None:
        return False
    anchor = _discover_anchor(registry, spec)
    expected_family_id = _family_id(registry.snapshot, spec, anchor)
    expected_program = _program(spec, anchor, artifact.construction_type)
    expected_root_task_id = deterministic_task_id(
        registry.snapshot_id,
        atomic_presence_program(spec.primary, anchor.anchor_date),
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
        family_id=expected_family_id,
        spec=spec,
        anchor=anchor,
        sibling_role=expected_role,
        root_task_id=expected_root_task_id,
        derived_from_task_id=(
            None if artifact.construction_type == "atomic" else expected_root_task_id
        ),
    )
    expected_evidence = {
        "atomic": {anchor.primary_evidence_id},
        "depth": {anchor.encounter_evidence_id, anchor.primary_evidence_id},
        "width": {anchor.primary_evidence_id, anchor.companion_evidence_id},
    }[artifact.construction_type]
    return all(
        (
            artifact.provenance.generator == GENERATOR,
            artifact.provenance.generator_version == TEMPLATE_VERSION,
            artifact.provenance.seed == 0,
            artifact.provenance.dataset == _dataset(registry.snapshot),
            artifact.family_id == expected_family_id,
            artifact.parent_task_ids == expected_parent_ids,
            artifact.program == expected_program,
            artifact.question == _question(spec, anchor, artifact.construction_type),
            artifact.gold_answer == TypedValue(kind="boolean", value=True),
            set(artifact.gold_evidence_ids) == expected_evidence,
            lineage == expected_lineage,
            artifact.metadata["program_signature"] == program_signature(expected_program),
        )
    )


def _discover_anchor(
    registry: ToolRegistry,
    spec: PresenceFamilySpec,
) -> PatientV1Anchor:
    _dataset(registry.snapshot)
    primary_views = _complete_records(
        registry,
        spec.primary.tool,
        spec.primary.tool_arguments(),
    )
    if not primary_views:
        raise _Ineligible("MISSING_PRIMARY")
    companion_views = _complete_records(
        registry,
        spec.companion.tool,
        spec.companion.tool_arguments(),
    )
    if not companion_views:
        raise _Ineligible("MISSING_COMPANION")
    primary_records = _records_from_views(registry.snapshot, primary_views, spec.primary)
    companion_records = _records_from_views(
        registry.snapshot,
        companion_views,
        spec.companion,
    )
    if spec.primary.resource_type == "Condition":
        primary_records = tuple(
            record for record in primary_records if isinstance(record.time.get("recorded"), str)
        )
    if spec.companion.resource_type == "Condition":
        companion_records = tuple(
            record for record in companion_records if isinstance(record.time.get("recorded"), str)
        )
    if not primary_records or not companion_records:
        raise _Ineligible("INCOMPLETE_SCOPE")

    linked_pairs: list[tuple[EvidenceRecord, EvidenceRecord, EvidenceRecord]] = []
    for primary in primary_records:
        encounter_ids = _linked_encounter_ids(registry.snapshot, primary.evidence_id)
        if len(encounter_ids) != 1:
            continue
        encounter = registry.snapshot.record_by_id(encounter_ids[0])
        if (
            encounter is None
            or encounter.resource_type != "Encounter"
            or encounter.payload.get("status") != "finished"
            or not encounter.local_date
            or primary.local_date != encounter.local_date
        ):
            continue
        companions = [
            companion
            for companion in companion_records
            if companion.local_date == encounter.local_date
            and companion.evidence_id != primary.evidence_id
            and companion.logical_evidence_group != primary.logical_evidence_group
            and encounter.evidence_id
            in _linked_encounter_ids(registry.snapshot, companion.evidence_id)
        ]
        if len(companions) != 1:
            continue
        date_primary = _date_records(registry, spec.primary, encounter.local_date)
        date_companion = _date_records(registry, spec.companion, encounter.local_date)
        if tuple(view["evidence_id"] for view in date_primary) != (primary.evidence_id,):
            continue
        if tuple(view["evidence_id"] for view in date_companion) != (companions[0].evidence_id,):
            continue
        linked_pairs.append((encounter, primary, companions[0]))

    if not linked_pairs:
        raise _Ineligible("NO_EXPLICIT_RELATION")

    candidates: list[PatientV1Anchor] = []
    has_depth_disambiguator = False
    for encounter, primary, companion in linked_pairs:
        distractors = [
            record
            for record in primary_records
            if record.evidence_id != primary.evidence_id
            and record.logical_evidence_group != primary.logical_evidence_group
            and record.local_date > encounter.local_date
            and _linked_to_another_encounter(
                registry.snapshot,
                record.evidence_id,
                encounter.evidence_id,
            )
        ]
        if not distractors:
            continue
        has_depth_disambiguator = True
        encounter_arguments = _encounter_selector(encounter)
        selected_encounters = _complete_records(
            registry,
            "search_encounters",
            encounter_arguments,
        )
        if tuple(view.get("evidence_id") for view in selected_encounters) != (
            encounter.evidence_id,
        ):
            continue
        encounter_bound_primary = _complete_records(
            registry,
            spec.primary.tool,
            {
                **spec.primary.tool_arguments(),
                "encounter_id": encounter.evidence_id,
            },
        )
        if tuple(view.get("evidence_id") for view in encounter_bound_primary) != (
            primary.evidence_id,
        ):
            continue
        candidates.append(
            PatientV1Anchor(
                encounter_evidence_id=encounter.evidence_id,
                primary_evidence_id=primary.evidence_id,
                companion_evidence_id=companion.evidence_id,
                anchor_date=encounter.local_date,
                encounter_name=_patient_encounter_name(encounter),
                encounter_arguments=encounter_arguments,
            )
        )

    if not candidates:
        raise _Ineligible(
            "INCOMPLETE_SCOPE" if has_depth_disambiguator else "NO_DEPTH_DISAMBIGUATOR"
        )
    return max(
        candidates,
        key=lambda anchor: (
            anchor.anchor_date,
            anchor.encounter_evidence_id,
            anchor.primary_evidence_id,
        ),
    )


def _complete_records(
    registry: ToolRegistry,
    tool: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    base_arguments = {key: value for key, value in arguments.items() if key != "cursor"}
    cursor = arguments.get("cursor")
    seen_cursors: set[str] = set()
    seen_evidence_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    expected_total: int | None = None

    while True:
        page_arguments = dict(base_arguments)
        if cursor is not None:
            page_arguments["cursor"] = cursor
        result = registry.call(tool, page_arguments)
        if not result.get("ok"):
            raise _Ineligible("INCOMPLETE_SCOPE")
        page = result.get("records")
        total_matches = result.get("total_matches")
        if (
            not isinstance(page, list)
            or not all(isinstance(record, dict) for record in page)
            or not isinstance(total_matches, int)
        ):
            raise _Ineligible("INCOMPLETE_SCOPE")
        if expected_total is None:
            expected_total = total_matches
        elif total_matches != expected_total:
            raise _Ineligible("INCOMPLETE_SCOPE")
        for record in page:
            evidence_id = record.get("evidence_id")
            if not isinstance(evidence_id, str) or evidence_id in seen_evidence_ids:
                raise _Ineligible("INCOMPLETE_SCOPE")
            seen_evidence_ids.add(evidence_id)
            records.append(record)

        if result.get("complete") is True:
            if result.get("next_cursor") is not None or len(records) != expected_total:
                raise _Ineligible("INCOMPLETE_SCOPE")
            return tuple(records)

        next_cursor = result.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
            raise _Ineligible("INCOMPLETE_SCOPE")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _date_records(
    registry: ToolRegistry,
    query: PresenceQuery,
    clinical_date: str,
) -> tuple[dict[str, Any], ...]:
    return _complete_records(
        registry,
        query.tool,
        {
            **query.tool_arguments(),
            "date_from": clinical_date,
            "date_to": clinical_date,
        },
    )


def _records_from_views(
    snapshot: PatientSnapshot,
    views: tuple[dict[str, Any], ...],
    query: PresenceQuery,
) -> tuple[EvidenceRecord, ...]:
    records: list[EvidenceRecord] = []
    for view in views:
        evidence_id = view.get("evidence_id")
        record = snapshot.record_by_id(evidence_id) if isinstance(evidence_id, str) else None
        if record is None or record.resource_type != query.resource_type:
            raise _Ineligible("INCOMPLETE_SCOPE")
        records.append(record)
    return tuple(records)


def _linked_encounter_ids(
    snapshot: PatientSnapshot,
    evidence_id: str,
) -> tuple[str, ...]:
    encounter_ids = {
        link.target_evidence_id
        for link in snapshot.outgoing_links(evidence_id)
        if link.relation.split(".")[-1] in {"context", "encounter"}
        and (target := snapshot.record_by_id(link.target_evidence_id)) is not None
        and target.resource_type == "Encounter"
    }
    return tuple(sorted(encounter_ids))


def _encounter_selector(encounter: EvidenceRecord) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "text": encounter.display,
        "limit": 100,
    }
    status = encounter.payload.get("status")
    if isinstance(status, str) and status:
        arguments["status"] = status
    code = encounter.code.get("code")
    system = encounter.code.get("system")
    if isinstance(code, str) and code:
        arguments["code"] = code
    if isinstance(system, str) and system:
        arguments["system"] = system
    return arguments


def _patient_encounter_name(encounter: EvidenceRecord) -> str:
    name = _strip_terminology_suffix(encounter.display.strip()).casefold()
    if not name:
        return "completed visit"
    normalized_names = {
        "encounter for problem": "problem visit",
        "encounter for check up": "check-up visit",
        "general examination of patient": "general examination visit",
        "follow-up encounter": "follow-up visit",
        "encounter for symptom": "symptom visit",
        "urgent care clinic": "urgent-care visit",
        "administration of vaccine to produce active immunity": "vaccination visit",
    }
    if name in normalized_names:
        name = normalized_names[name]
    elif name.endswith(" clinic"):
        name = f"{name.removesuffix(' clinic')} visit"
    elif not name.endswith((" visit", " encounter", " admission", " stay", " consultation")):
        name = f"{name} visit"
    return f"completed {name}"


def _strip_terminology_suffix(value: str) -> str:
    for suffix in _TERMINOLOGY_SUFFIXES:
        if value.casefold().endswith(suffix):
            return value[: -len(suffix)].rstrip()
    return value


def _linked_to_another_encounter(
    snapshot: PatientSnapshot,
    evidence_id: str,
    anchor_encounter_id: str,
) -> bool:
    return any(
        encounter_id != anchor_encounter_id
        for encounter_id in _linked_encounter_ids(snapshot, evidence_id)
    )


def _dataset(snapshot: PatientSnapshot) -> str:
    datasets = {record.source_dataset for record in snapshot.records}
    if len(datasets) != 1:
        raise _Ineligible("INCOMPLETE_SCOPE")
    return next(iter(datasets))


def _program(
    spec: PresenceFamilySpec,
    anchor: PatientV1Anchor,
    construction_type: str,
) -> TaskProgram:
    if construction_type == "atomic":
        return atomic_presence_program(spec.primary, anchor.anchor_date)
    if construction_type == "depth":
        return depth_presence_program(spec.primary, anchor.encounter_arguments)
    if construction_type == "width":
        return width_presence_program(spec.primary, spec.companion, anchor.anchor_date)
    raise ValueError("unsupported V1 construction type")


def _question(
    spec: PresenceFamilySpec,
    anchor: PatientV1Anchor,
    construction_type: str,
) -> str:
    rendered_date = _render_date(anchor.anchor_date)
    if construction_type == "atomic":
        return f"Does my chart show {spec.primary.patient_phrase} on {rendered_date}?"
    if construction_type == "depth":
        return (
            f"Does my chart show {spec.primary.patient_phrase} at my "
            f"most recent {anchor.encounter_name}?"
        )
    if construction_type == "width":
        return (
            f"On {rendered_date}, does my chart show both {spec.primary.patient_phrase} and "
            f"{spec.companion.patient_phrase}?"
        )
    raise ValueError("unsupported V1 construction type")


def _family_id(
    snapshot: PatientSnapshot,
    spec: PresenceFamilySpec,
    anchor: PatientV1Anchor,
) -> str:
    digest = content_hash(
        {
            "snapshot_id": snapshot.snapshot_id,
            "template_id": TEMPLATE_ID,
            "template_version": TEMPLATE_VERSION,
            "spec_id": spec.spec_id,
            "anchor_evidence_ids": sorted(
                {
                    anchor.encounter_evidence_id,
                    anchor.primary_evidence_id,
                    anchor.companion_evidence_id,
                }
            ),
        }
    ).removeprefix("sha256:")
    return f"patient-v1-family-{digest[:24]}"


def _lineage(
    *,
    family_id: str,
    spec: PresenceFamilySpec,
    anchor: PatientV1Anchor,
    sibling_role: str,
    root_task_id: str,
    derived_from_task_id: str | None,
) -> dict[str, Any]:
    return {
        "family_id": family_id,
        "sibling_role": sibling_role,
        "root_task_id": root_task_id,
        "derived_from_task_id": derived_from_task_id,
        "template_id": TEMPLATE_ID,
        "template_version": TEMPLATE_VERSION,
        "spec_id": spec.spec_id,
        "anchor_encounter_evidence_id": anchor.encounter_evidence_id,
        "anchor_primary_evidence_id": anchor.primary_evidence_id,
        "anchor_companion_evidence_id": anchor.companion_evidence_id,
        "anchor_encounter_arguments": anchor.encounter_arguments,
        "anchor_evidence_ids": sorted(
            {
                anchor.encounter_evidence_id,
                anchor.primary_evidence_id,
                anchor.companion_evidence_id,
            }
        ),
    }


def _attach_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    spec: PresenceFamilySpec,
    anchor: PatientV1Anchor,
    *,
    sibling_role: str,
    root_task_id: str,
    derived_from_task_id: str | None,
) -> TaskArtifact:
    enriched = artifact.model_copy(
        update={
            "metadata": {
                "lineage": _lineage(
                    family_id=artifact.family_id,
                    spec=spec,
                    anchor=anchor,
                    sibling_role=sibling_role,
                    root_task_id=root_task_id,
                    derived_from_task_id=derived_from_task_id,
                ),
                "program_signature": program_signature(artifact.program),
            }
        }
    )
    return enriched.model_copy(update={"verification": verify_task_artifact(enriched, registry)})


def _render_date(value: str) -> str:
    year, month, day = value.split("-")
    months = (
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    return f"{months[int(month)]} {int(day)}, {year}"
