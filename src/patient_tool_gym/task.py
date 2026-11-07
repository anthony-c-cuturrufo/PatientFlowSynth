from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_serializer, model_validator

from patient_tool_gym.program import (
    ProgramComplexity,
    ReferenceCall,
    StrictModel,
    TaskProgram,
    TypedValue,
    content_hash,
)


class ArtifactProvenance(StrictModel):
    dataset: str
    normalizer_version: str
    tool_manifest_hash: str
    generator: str
    generator_version: str
    seed: int | None = None


class VerificationReport(StrictModel):
    verifier_version: str = "2"
    policy: Literal["standard", "selector_hidden_ablation"] = "standard"
    passed: bool
    checks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def serialize_legacy_policy(self, handler: Any) -> dict[str, Any]:
        """Preserve whether a frozen pre-policy report actually serialized the field."""

        payload = handler(self)
        if "policy" not in self.model_fields_set:
            payload.pop("policy", None)
        return payload


class EvidenceContract(StrictModel):
    """Declare separately what supports an answer and what completes an audit."""

    required_level: Literal["answer_sufficient", "audit_complete"] = "answer_sufficient"
    answer_sufficient_witnesses: tuple[tuple[str, ...], ...]
    audit_complete_witnesses: tuple[tuple[str, ...], ...]
    answer_program: TaskProgram | None = None
    audit_program: TaskProgram | None = None

    @model_serializer(mode="wrap")
    def serialize_legacy_programs(self, handler: Any) -> dict[str, Any]:
        """Keep same-layer contracts written before executable programs hash-stable."""

        payload = handler(self)
        if self.answer_program is None:
            payload.pop("answer_program", None)
        if self.audit_program is None:
            payload.pop("audit_program", None)
        return payload

    @model_validator(mode="after")
    def validate_witness_layers(self) -> EvidenceContract:
        for label, witnesses in (
            ("answer-sufficient", self.answer_sufficient_witnesses),
            ("audit-complete", self.audit_complete_witnesses),
        ):
            if not witnesses:
                raise ValueError(f"{label} evidence requires at least one witness")
            for witness in witnesses:
                if witness != tuple(sorted(set(witness))):
                    raise ValueError(
                        f"{label} evidence witnesses must contain sorted unique evidence IDs"
                    )
            if len(witnesses) != len(set(witnesses)):
                raise ValueError(f"{label} evidence witnesses must be unique")

        answer_sets = tuple(map(set, self.answer_sufficient_witnesses))
        audit_sets = tuple(map(set, self.audit_complete_witnesses))
        audit_without_answer = any(
            not any(answer_witness.issubset(audit_witness) for answer_witness in answer_sets)
            for audit_witness in audit_sets
        )
        answer_without_audit = any(
            not any(answer_witness.issubset(audit_witness) for audit_witness in audit_sets)
            for answer_witness in answer_sets
        )
        if audit_without_answer or answer_without_audit:
            raise ValueError(
                "answer-sufficient and audit-complete witnesses require bidirectional "
                "subset coverage"
            )
        layers_differ = set(self.answer_sufficient_witnesses) != set(
            self.audit_complete_witnesses
        )
        if layers_differ and (self.answer_program is None or self.audit_program is None):
            raise ValueError(
                "different evidence layers require executable answer_program and audit_program"
            )
        return self


class CompleteSearchRequirement(StrictModel):
    """Bind one closed-world claim to its exact executable reference search."""

    claim_semantics: Literal[
        "chart_absence",
        "empty_search_boolean",
        "exclusive_supported",
        "exclusive_refuted",
        "insufficient_evidence",
        "conflicting_evidence",
    ]
    claim_node_id: str = Field(min_length=1)
    search_node_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any]
    result_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    completion_semantics: Literal["single_complete_reference_result"] = (
        "single_complete_reference_result"
    )

    @model_validator(mode="after")
    def validate_unpaginated_reference_search(self) -> CompleteSearchRequirement:
        if self.arguments.get("cursor") is not None:
            raise ValueError("complete-search requirements must begin without a cursor")
        return self


class EvidenceQualityContract(StrictModel):
    """Declare exact, threshold-free evidence-quality semantics for one task."""

    schema_version: Literal["patient-tool-gym.evidence-quality-contract.v1"] = (
        "patient-tool-gym.evidence-quality-contract.v1"
    )
    answer_evidence_policy: Literal["exact_declared_witness"] = "exact_declared_witness"
    audit_evidence_policy: Literal["exact_declared_witness"] = "exact_declared_witness"
    full_snapshot_policy: Literal["reject"] = "reject"
    complete_search_requirements: tuple[CompleteSearchRequirement, ...] = ()

    @model_validator(mode="after")
    def validate_unique_requirements(self) -> EvidenceQualityContract:
        identities = tuple(
            (requirement.claim_node_id, requirement.search_node_id)
            for requirement in self.complete_search_requirements
        )
        if len(identities) != len(set(identities)):
            raise ValueError("complete-search requirements must be unique")
        return self


class QuestionContract(StrictModel):
    """Declare whether the visible question identifies a clinical answer target."""

    answerability: Literal["answerable", "clarification_required"] = "answerable"
    reason: Literal["missing_clinical_selector"] | None = None
    allow_clarification: bool = False

    @model_validator(mode="after")
    def validate_answerability(self) -> QuestionContract:
        if self.answerability == "clarification_required":
            if self.reason != "missing_clinical_selector":
                raise ValueError(
                    "clarification-required questions must declare missing_clinical_selector"
                )
            if not self.allow_clarification:
                raise ValueError(
                    "clarification-required questions must allow the clarification response"
                )
        elif self.reason is not None:
            raise ValueError("answerable questions must not declare a clarification reason")
        return self


class TaskArtifact(StrictModel):
    schema_version: Literal["patient-tool-gym.task.v1", "patient-tool-gym.task.v2"] = (
        "patient-tool-gym.task.v1"
    )
    task_id: str
    family_id: str
    parent_task_ids: tuple[str, ...] = ()
    construction_type: Literal["atomic", "depth", "width", "mixed", "missing_evidence"]
    patient_snapshot_id: str
    question: str = ""
    program: TaskProgram
    gold_answer: TypedValue
    gold_evidence_ids: tuple[str, ...]
    evidence_witnesses: tuple[tuple[str, ...], ...] = ()
    evidence_contract: EvidenceContract | None = None
    evidence_quality_contract: EvidenceQualityContract | None = None
    question_contract: QuestionContract | None = None
    reference_trace: tuple[ReferenceCall, ...]
    allowed_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...] = ()
    difficulty: ProgramComplexity
    provenance: ArtifactProvenance
    verification: VerificationReport
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def serialize_legacy_compatible(self, handler: Any) -> dict[str, Any]:
        """Do not change persisted hashes for tasks created before explicit contracts."""

        payload = handler(self)
        if self.evidence_contract is None:
            payload.pop("evidence_contract", None)
        if self.evidence_quality_contract is None:
            payload.pop("evidence_quality_contract", None)
        if self.question_contract is None:
            payload.pop("question_contract", None)
        return payload

    @model_validator(mode="after")
    def validate_schema_generation(self) -> TaskArtifact:
        task_is_v2 = self.schema_version == "patient-tool-gym.task.v2"
        program_is_v2 = self.program.schema_version in {
            "patient-tool-gym.program.v2",
            "patient-tool-gym.program.v3",
            "patient-tool-gym.program.v4",
        }
        answer_is_v2 = self.gold_answer.kind == "truth_state"
        if task_is_v2 != program_is_v2:
            raise ValueError("task.v2 requires program.v2 or program.v3 (or program.v4)")
        if answer_is_v2 and not task_is_v2:
            raise ValueError("truth_state answers require patient-tool-gym.task.v2")
        for witness in self.evidence_witnesses:
            if witness != tuple(sorted(set(witness))):
                raise ValueError("evidence witnesses must contain sorted unique evidence IDs")
        if len(self.evidence_witnesses) != len(set(self.evidence_witnesses)):
            raise ValueError("evidence witnesses must be unique")
        return self

    def valid_evidence_witnesses(self) -> tuple[tuple[str, ...], ...]:
        """Return explicit witnesses, or the legacy canonical evidence set."""

        canonical = tuple(sorted(set(self.gold_evidence_ids)))
        if not self.evidence_witnesses:
            return (canonical,)
        if canonical in self.evidence_witnesses:
            return self.evidence_witnesses
        return (canonical, *self.evidence_witnesses)

    def answer_sufficient_evidence_witnesses(self) -> tuple[tuple[str, ...], ...]:
        """Return answer witnesses, falling back to the legacy evidence contract."""

        if self.evidence_contract is not None:
            return self.evidence_contract.answer_sufficient_witnesses
        return self.valid_evidence_witnesses()

    def audit_complete_evidence_witnesses(self) -> tuple[tuple[str, ...], ...]:
        """Return audit witnesses, falling back to the legacy evidence contract."""

        if self.evidence_contract is not None:
            return self.evidence_contract.audit_complete_witnesses
        return self.valid_evidence_witnesses()

    def required_evidence_level(self) -> Literal["answer_sufficient", "audit_complete"]:
        """Return which evidence layer gates the task's primary pass endpoint."""

        if self.evidence_contract is None:
            return "answer_sufficient"
        return self.evidence_contract.required_level

    def requires_clarification(self) -> bool:
        """Return whether a clarification is the only appropriate visible response."""

        return (
            self.question_contract is not None
            and self.question_contract.answerability == "clarification_required"
        )


def deterministic_task_id(
    snapshot_id: str,
    program: TaskProgram,
    family_id: str,
    construction_type: str,
) -> str:
    digest = content_hash(
        {
            "snapshot_id": snapshot_id,
            "program": program.model_dump(mode="json"),
            "family_id": family_id,
            "construction_type": construction_type,
        }
    ).removeprefix("sha256:")
    return f"patient-task-{digest[:24]}"
