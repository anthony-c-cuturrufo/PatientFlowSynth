from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import model_validator

from patient_tool_gym.fhir import normalize_fhir_bundle
from patient_tool_gym.program import (
    ProgramExecutor,
    StrictModel,
    TaskProgram,
    TypedValue,
    content_hash,
)
from patient_tool_gym.raw_fhir import (
    SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES,
    RawFHIRInsertedReferenceRoute,
    RawFHIRPatch,
    apply_raw_fhir_patch,
    build_distractor_patch,
    build_truth_state_patch,
    validate_reference_closure_strict,
)
from patient_tool_gym.tools import ToolRegistry

InterventionState = Literal[
    "supported",
    "refuted",
    "insufficient_evidence",
    "conflicting_evidence",
    "answer_preserving_distractor",
]
CompositionalEvidenceState = Literal[
    "supported",
    "refuted",
    "insufficient_evidence",
    "conflicting_evidence",
]


class RawInterventionMember(StrictModel):
    state: InterventionState
    patch: RawFHIRPatch
    derived_bundle_hash: str
    snapshot_id: str
    answer: TypedValue
    evidence_ids: tuple[str, ...]
    reference_trace_hash: str


class RawInterventionFamily(StrictModel):
    schema_version: Literal["patient-tool-gym.raw-intervention-family.v1"] = (
        "patient-tool-gym.raw-intervention-family.v1"
    )
    base_bundle_hash: str
    program_hash: str
    resource_type: str
    resource_id: str
    members: tuple[RawInterventionMember, ...]
    family_hash: str


class RawFHIRDeclaredTarget(StrictModel):
    role: Literal["primary", "companion"]
    resource_type: str
    resource_id: str
    evidence_state: CompositionalEvidenceState


class RawFHIRPatchChain(StrictModel):
    """A hash-bound state patch followed by an optional independent distractor."""

    schema_version: Literal["patient-tool-gym.raw-fhir-patch-chain.v1"] = (
        "patient-tool-gym.raw-fhir-patch-chain.v1"
    )
    base_bundle_hash: str
    target_resource_type: str
    target_resource_id: str
    evidence_state: CompositionalEvidenceState
    distractor_present: bool
    declared_targets: tuple[RawFHIRDeclaredTarget, ...] = ()
    patches: tuple[RawFHIRPatch, ...] = ()
    derived_bundle_hash: str
    chain_hash: str

    @model_validator(mode="after")
    def validate_chain_shape(self) -> RawFHIRPatchChain:
        if self.declared_targets:
            primary = self.declared_targets[0]
            if primary.role != "primary":
                raise ValueError("first declared target must be primary")
            if (
                primary.resource_type != self.target_resource_type
                or primary.resource_id != self.target_resource_id
                or primary.evidence_state != self.evidence_state
            ):
                raise ValueError("legacy primary fields must match the primary declaration")
            state_patches = tuple(
                patch
                for patch in self.patches
                if patch.intervention_kind != "answer_preserving_distractor"
            )
            cursor = 0
            for declaration in self.declared_targets:
                if cursor < len(state_patches):
                    patch = state_patches[cursor]
                    if (
                        patch.target_resource_type == declaration.resource_type
                        and patch.target_resource_id == declaration.resource_id
                        and patch.intervention_kind == declaration.evidence_state
                    ):
                        cursor += 1
                        continue
                if declaration.evidence_state != "supported":
                    raise ValueError("only a supported declaration may omit a no-op patch")
            if cursor != len(state_patches):
                raise ValueError("state patch does not match an ordered target declaration")
        else:
            # Preserve deserialization of frozen v1 artifacts.
            if len(self.patches) != 1 + int(self.distractor_present):
                raise ValueError("legacy patch count must equal state patch plus distractor")
            if self.patches[0].intervention_kind != self.evidence_state:
                raise ValueError("first patch must implement the declared evidence state")
        distractor_patches = tuple(
            patch
            for patch in self.patches
            if patch.intervention_kind == "answer_preserving_distractor"
        )
        if len(distractor_patches) != int(self.distractor_present):
            raise ValueError("distractor patch count must match distractor_present")
        if self.distractor_present and self.patches[-1] != distractor_patches[-1]:
            raise ValueError("distractor patch must be final")
        return self


def build_compositional_patch_chain(
    bundle: Mapping[str, Any],
    *,
    resource_type: str,
    resource_id: str,
    evidence_state: CompositionalEvidenceState,
    distractor_present: bool,
    forbidden_codes: frozenset[tuple[str, str]] = frozenset(),
    target_date: str | None = None,
    provenance_recorded_at: str | None = None,
) -> RawFHIRPatchChain:
    """Compile orthogonal medical-state and distractor interventions."""

    base = dict(bundle)
    if not validate_reference_closure_strict(
        base,
        allowed_external_conditional_reference_types=(SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES),
    ).passed:
        raise ValueError("compositional patch source is not strictly reference-closed")
    _assert_release_state_representable(
        base,
        resource_type=resource_type,
        resource_id=resource_id,
        evidence_state=evidence_state,
    )
    state_patch = build_truth_state_patch(
        base,
        resource_type=resource_type,
        resource_id=resource_id,
        target_state=evidence_state,
        provenance_recorded_at=provenance_recorded_at,
    )
    patches: list[RawFHIRPatch] = []
    derived = base
    if state_patch.operations:
        patches.append(state_patch)
        derived = apply_raw_fhir_patch(base, state_patch)
    if distractor_present:
        distractor_patch = build_distractor_patch(
            derived,
            resource_type=resource_type,
            resource_id=resource_id,
            distractor_tier="same_resource_different_code",
            forbidden_codes=forbidden_codes,
            target_date=target_date,
            require_exact_tier=True,
            provenance_recorded_at=provenance_recorded_at,
        )
        patches.append(distractor_patch)
        derived = apply_raw_fhir_patch(derived, distractor_patch)
    closure = validate_reference_closure_strict(
        derived,
        allowed_external_conditional_reference_types=(SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES),
    )
    if not closure.passed:
        raise ValueError("compositional patch chain introduced an unresolved reference")
    draft = RawFHIRPatchChain(
        base_bundle_hash=content_hash(base),
        target_resource_type=resource_type,
        target_resource_id=resource_id,
        evidence_state=evidence_state,
        distractor_present=distractor_present,
        declared_targets=(
            RawFHIRDeclaredTarget(
                role="primary",
                resource_type=resource_type,
                resource_id=resource_id,
                evidence_state=evidence_state,
            ),
        ),
        patches=tuple(patches),
        derived_bundle_hash=content_hash(derived),
        chain_hash="pending",
    )
    return draft.model_copy(
        update={"chain_hash": content_hash(draft.model_dump(mode="json", exclude={"chain_hash"}))}
    )


def build_compositional_factorial_patch_chain(
    bundle: Mapping[str, Any],
    *,
    primary_resource_type: str,
    primary_resource_id: str,
    primary_state: CompositionalEvidenceState,
    companion_resource_type: str,
    companion_resource_id: str,
    companion_state: CompositionalEvidenceState,
    distractor_present: bool,
    forbidden_codes: frozenset[tuple[str, str]] = frozenset(),
    target_date: str | None = None,
    provenance_recorded_at: str | None = None,
    primary_inserted_reference_route: RawFHIRInsertedReferenceRoute | None = None,
) -> RawFHIRPatchChain:
    """Build two independently declared state interventions plus a primary distractor."""

    if (primary_resource_type, primary_resource_id) == (
        companion_resource_type,
        companion_resource_id,
    ):
        raise ValueError("primary and companion intervention targets must be distinct")
    base = dict(bundle)
    if not validate_reference_closure_strict(
        base,
        allowed_external_conditional_reference_types=(SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES),
    ).passed:
        raise ValueError("factorial patch source is not strictly reference-closed")
    declarations = (
        RawFHIRDeclaredTarget(
            role="primary",
            resource_type=primary_resource_type,
            resource_id=primary_resource_id,
            evidence_state=primary_state,
        ),
        RawFHIRDeclaredTarget(
            role="companion",
            resource_type=companion_resource_type,
            resource_id=companion_resource_id,
            evidence_state=companion_state,
        ),
    )
    current = base
    patches: list[RawFHIRPatch] = []
    for declaration in declarations:
        _assert_release_state_representable(
            current,
            resource_type=declaration.resource_type,
            resource_id=declaration.resource_id,
            evidence_state=declaration.evidence_state,
        )
        patch = build_truth_state_patch(
            current,
            resource_type=declaration.resource_type,
            resource_id=declaration.resource_id,
            target_state=declaration.evidence_state,
            provenance_recorded_at=provenance_recorded_at,
            inserted_reference_route=(
                primary_inserted_reference_route if declaration.role == "primary" else None
            ),
        )
        if patch.operations:
            patches.append(patch)
            current = apply_raw_fhir_patch(current, patch)
    if distractor_present:
        distractor_patch = build_distractor_patch(
            current,
            resource_type=primary_resource_type,
            resource_id=primary_resource_id,
            distractor_tier="same_resource_different_code",
            forbidden_codes=forbidden_codes,
            target_date=target_date,
            require_exact_tier=True,
            provenance_recorded_at=provenance_recorded_at,
            inserted_reference_route=primary_inserted_reference_route,
        )
        patches.append(distractor_patch)
        current = apply_raw_fhir_patch(current, distractor_patch)
    if not validate_reference_closure_strict(
        current,
        allowed_external_conditional_reference_types=(SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES),
    ).passed:
        raise ValueError("factorial patch chain introduced an unresolved reference")
    draft = RawFHIRPatchChain(
        base_bundle_hash=content_hash(base),
        target_resource_type=primary_resource_type,
        target_resource_id=primary_resource_id,
        evidence_state=primary_state,
        distractor_present=distractor_present,
        declared_targets=declarations,
        patches=tuple(patches),
        derived_bundle_hash=content_hash(current),
        chain_hash="pending",
    )
    return draft.model_copy(
        update={"chain_hash": content_hash(draft.model_dump(mode="json", exclude={"chain_hash"}))}
    )


def _assert_release_state_representable(
    bundle: Mapping[str, Any],
    *,
    resource_type: str,
    resource_id: str,
    evidence_state: CompositionalEvidenceState,
) -> None:
    target = next(
        (
            resource
            for entry in bundle.get("entry", [])
            if isinstance(entry, Mapping)
            and isinstance((resource := entry.get("resource")), Mapping)
            and resource.get("resourceType") == resource_type
            and resource.get("id") == resource_id
        ),
        None,
    )
    if target is None:
        raise ValueError(f"FHIR resource not found: {resource_type}/{resource_id}")
    if resource_type == "Procedure" and evidence_state != "supported":
        raise ValueError(
            "exact-date Procedure branches cannot represent refuted, insufficient, or "
            "conflicting states: FHIR not-done/entered-in-error records cannot retain "
            "performed[x]; use a linked ServiceRequest occurrence or another companion"
        )


def apply_raw_fhir_patch_chain(
    bundle: Mapping[str, Any],
    chain: RawFHIRPatchChain,
) -> dict[str, Any]:
    """Replay and verify every hash-bound patch in a compositional chain."""

    current = dict(bundle)
    if content_hash(current) != chain.base_bundle_hash:
        raise ValueError("compositional patch-chain base hash mismatch")
    expected_chain_hash = content_hash(chain.model_dump(mode="json", exclude={"chain_hash"}))
    if chain.chain_hash != expected_chain_hash:
        raise ValueError("compositional patch-chain hash mismatch")
    for patch in chain.patches:
        current = apply_raw_fhir_patch(current, patch)
    if content_hash(current) != chain.derived_bundle_hash:
        raise ValueError("compositional patch-chain derived hash mismatch")
    if not validate_reference_closure_strict(
        current,
        allowed_external_conditional_reference_types=(SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES),
    ).passed:
        raise ValueError("compositional patch-chain result is not reference-closed")
    return current


def compile_raw_intervention_family(
    bundle: Mapping[str, Any],
    *,
    program: TaskProgram,
    resource_type: str,
    resource_id: str,
    dataset: str = "synthea",
    provenance_recorded_at: str | None = None,
) -> RawInterventionFamily:
    members: list[RawInterventionMember] = []
    for state in (
        "supported",
        "refuted",
        "insufficient_evidence",
        "conflicting_evidence",
        "answer_preserving_distractor",
    ):
        patch = build_truth_state_patch(
            bundle,
            resource_type=resource_type,
            resource_id=resource_id,
            target_state=state,
            provenance_recorded_at=provenance_recorded_at,
        )
        derived = apply_raw_fhir_patch(bundle, patch)
        snapshot = normalize_fhir_bundle(derived, dataset=dataset)
        result = ProgramExecutor(ToolRegistry(snapshot)).execute(program)
        expected_state = "supported" if state == "answer_preserving_distractor" else state
        expected = TypedValue(kind="truth_state", value=expected_state)
        if result.answer != expected:
            raise ValueError(f"raw FHIR intervention {state} replayed as {result.answer.value!r}")
        members.append(
            RawInterventionMember(
                state=state,
                patch=patch,
                derived_bundle_hash=content_hash(derived),
                snapshot_id=snapshot.snapshot_id,
                answer=result.answer,
                evidence_ids=result.evidence_ids,
                reference_trace_hash=content_hash(
                    [call.model_dump(mode="json") for call in result.reference_trace]
                ),
            )
        )
    draft = RawInterventionFamily(
        base_bundle_hash=content_hash(dict(bundle)),
        program_hash=content_hash(program.model_dump(mode="json")),
        resource_type=resource_type,
        resource_id=resource_id,
        members=tuple(members),
        family_hash="pending",
    )
    return draft.model_copy(
        update={"family_hash": content_hash(draft.model_dump(mode="json", exclude={"family_hash"}))}
    )
