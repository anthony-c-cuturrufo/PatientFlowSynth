from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field

from patient_tool_gym.evidence import EvidenceRecord, PatientSnapshot, snapshot_from_payloads
from patient_tool_gym.program import (
    ProgramExecutionError,
    ProgramExecutor,
    StrictModel,
    content_hash,
)
from patient_tool_gym.task import TaskArtifact
from patient_tool_gym.tools import ToolRegistry


class ObservedToolCall(StrictModel):
    """One actually executed tool call and its stored deterministic result."""

    call_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any]
    result: dict[str, Any]


class CompleteSearchCertificate(StrictModel):
    """A compact claim that one observed call discharged one exact search requirement."""

    schema_version: Literal["patient-tool-gym.complete-search-certificate.v1"] = (
        "patient-tool-gym.complete-search-certificate.v1"
    )
    task_id: str = Field(min_length=1)
    requirement_index: int = Field(ge=0)
    claim_node_id: str = Field(min_length=1)
    search_node_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any]
    result_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def replayed_observed_evidence_ids(
    registry: ToolRegistry,
    observations: Sequence[ObservedToolCall],
) -> frozenset[str]:
    """Return IDs from successful retrieval results that replay exactly.

    Submission results are excluded because they merely echo the model's citations;
    they are not evidence that the cited records were previously observed.
    """

    def collect(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            evidence_id = value.get("evidence_id")
            if isinstance(evidence_id, str):
                found.add(evidence_id)
            for child in value.values():
                found.update(collect(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                found.update(collect(child))
        return found

    observed: set[str] = set()
    for observation in observations:
        if observation.tool == "submit_answer":
            continue
        replayed = registry.call(observation.tool, observation.arguments)
        if replayed != observation.result or replayed.get("ok") is not True:
            continue
        observed.update(collect(replayed))
    return frozenset(observed)


def build_complete_search_certificates(
    task: TaskArtifact,
    registry: ToolRegistry,
    observations: Sequence[ObservedToolCall],
) -> tuple[CompleteSearchCertificate, ...]:
    """Derive certificates only from exact, deterministically replayable observed calls."""

    contract = task.evidence_quality_contract
    if contract is None:
        return ()
    certificates = []
    for requirement_index, requirement in enumerate(contract.complete_search_requirements):
        for observation in observations:
            if (
                observation.tool != requirement.tool
                or observation.arguments.get("cursor") is not None
            ):
                continue
            replayed = registry.call(observation.tool, observation.arguments)
            if (
                replayed != observation.result
                or content_hash(replayed) != requirement.result_hash
            ):
                continue
            certificate = CompleteSearchCertificate(
                task_id=task.task_id,
                requirement_index=requirement_index,
                claim_node_id=requirement.claim_node_id,
                search_node_id=requirement.search_node_id,
                call_id=observation.call_id,
                snapshot_id=registry.snapshot_id,
                tool=observation.tool,
                arguments=observation.arguments,
                result_hash=content_hash(observation.result),
            )
            if validate_complete_search_certificate(
                task,
                registry,
                certificate,
                observation,
            ):
                certificates.append(certificate)
                break
    return tuple(certificates)


def validate_complete_search_certificate(
    task: TaskArtifact,
    registry: ToolRegistry,
    certificate: CompleteSearchCertificate,
    observation: ObservedToolCall,
) -> bool:
    """Validate a certificate against one bound, replayable transcript observation."""

    contract = task.evidence_quality_contract
    if contract is None or certificate.task_id != task.task_id:
        return False
    if certificate.requirement_index >= len(contract.complete_search_requirements):
        return False
    requirement = contract.complete_search_requirements[certificate.requirement_index]
    if (
        certificate.claim_node_id != requirement.claim_node_id
        or certificate.search_node_id != requirement.search_node_id
        or certificate.snapshot_id != task.patient_snapshot_id
        or certificate.snapshot_id != registry.snapshot_id
        or certificate.snapshot_id != requirement.snapshot_id
        or certificate.tool != requirement.tool
        or certificate.call_id != observation.call_id
        or certificate.tool != observation.tool
        or certificate.arguments != observation.arguments
        or certificate.arguments.get("cursor") is not None
        or certificate.result_hash != requirement.result_hash
        or content_hash(observation.result) != certificate.result_hash
    ):
        return False
    replayed = registry.call(observation.tool, observation.arguments)
    reference_result = registry.call(requirement.tool, requirement.arguments)
    records = replayed.get("records")
    total_matches = replayed.get("total_matches")
    return bool(
        replayed.get("ok") is True
        and replayed == observation.result
        and replayed.get("complete") is True
        and replayed.get("next_cursor") is None
        and replayed.get("snapshot_id") == certificate.snapshot_id
        and isinstance(records, list)
        and type(total_matches) is int
        and total_matches >= 0
        and len(records) == total_matches
        and content_hash(replayed) == certificate.result_hash
        and replayed.get("searched_scope") == reference_result.get("searched_scope")
        and content_hash(reference_result) == requirement.result_hash
    )


def complete_search_certificates_valid(
    task: TaskArtifact,
    registry: ToolRegistry,
    certificates: Sequence[CompleteSearchCertificate],
    observations: Sequence[ObservedToolCall],
) -> bool:
    """Require each certificate to bind to an actually observed, replayable call."""

    contract = task.evidence_quality_contract
    if contract is None:
        return not certificates
    observed_by_id: dict[str, ObservedToolCall] = {}
    for observation in observations:
        if observation.call_id in observed_by_id:
            return False
        observed_by_id[observation.call_id] = observation
    expected_indexes = set(range(len(contract.complete_search_requirements)))
    actual_indexes = [certificate.requirement_index for certificate in certificates]

    def bound_to_observation(certificate: CompleteSearchCertificate) -> bool:
        observation = observed_by_id.get(certificate.call_id)
        return bool(
            observation is not None
            and validate_complete_search_certificate(
                task,
                registry,
                certificate,
                observation,
            )
        )

    return bool(
        len(actual_indexes) == len(set(actual_indexes))
        and set(actual_indexes) == expected_indexes
        and all(bound_to_observation(certificate) for certificate in certificates)
    )


def certify_evidence_contract(
    task: TaskArtifact,
    snapshot: PatientSnapshot,
) -> dict[str, Any]:
    witnesses = tuple(tuple(sorted(set(witness))) for witness in task.valid_evidence_witnesses())
    sufficient = [
        witness for witness in witnesses if evidence_satisfies_task(task, snapshot, witness)
    ]
    minimal_candidates = {
        shrunk
        for witness in sufficient
        for shrunk in (
            _shrink_witness(task, snapshot, witness, reverse=False),
            _shrink_witness(task, snapshot, witness, reverse=True),
        )
    }
    minimal = {
        witness
        for witness in minimal_candidates
        if all(
            not evidence_satisfies_task(
                task,
                snapshot,
                tuple(item for item in witness if item != evidence_id),
            )
            for evidence_id in witness
        )
    }
    if () in sufficient:
        minimal.add(())
    canonical = tuple(sorted(set(task.gold_evidence_ids)))
    redundant = [
        evidence_id
        for evidence_id in canonical
        if len(canonical) > 1
        and evidence_satisfies_task(
            task, snapshot, tuple(item for item in canonical if item != evidence_id)
        )
    ]
    necessary = sorted(set.intersection(*(set(witness) for witness in minimal))) if minimal else []
    witnessed_ids = {evidence_id for witness in sufficient for evidence_id in witness}
    candidate_non_witness_ids = [
        record.evidence_id for record in snapshot.records if record.evidence_id not in witnessed_ids
    ]
    invariance_checks = [
        {
            "evidence_id": evidence_id,
            "passed_for_every_declared_minimal_witness": all(
                _same_answer(task, snapshot, tuple(sorted(set(witness) | {evidence_id})))
                for witness in minimal
            ),
        }
        for evidence_id in candidate_non_witness_ids
    ]
    individually_invariant = sorted(
        row["evidence_id"]
        for row in invariance_checks
        if row["passed_for_every_declared_minimal_witness"]
    )
    return {
        "schema_version": "patient-tool-gym.evidence-contract-certificate.v2",
        "task_id": task.task_id,
        "declared_witness_count": len(witnesses),
        "sufficient_witnesses": [list(item) for item in sufficient],
        "derived_deletion_minimal_sufficient_witnesses": [list(item) for item in sorted(minimal)],
        "minimal_witness_derivation": "two_order_greedy_nonexhaustive",
        "declared_witnesses_are_exhaustive": False,
        "necessary_across_declared_minimal_witnesses": necessary,
        "redundant_canonical_evidence_ids": redundant,
        "candidate_non_witness_evidence_ids": candidate_non_witness_ids,
        "individually_answer_invariant_evidence_ids": individually_invariant,
        "individual_invariance_checks": invariance_checks,
        "passed": (
            len(sufficient) == len(witnesses)
            and bool(minimal or () in sufficient)
        ),
    }


def _shrink_witness(
    task: TaskArtifact,
    snapshot: PatientSnapshot,
    witness: tuple[str, ...],
    *,
    reverse: bool,
) -> tuple[str, ...]:
    current = list(witness)
    for evidence_id in sorted(witness, reverse=reverse):
        candidate = tuple(item for item in current if item != evidence_id)
        if candidate and evidence_satisfies_task(task, snapshot, candidate):
            current = list(candidate)
    return tuple(sorted(current))


def evidence_satisfies_task(
    task: TaskArtifact,
    snapshot: PatientSnapshot,
    evidence_ids: tuple[str, ...],
) -> bool:
    """Return whether cited task evidence is sufficient under executable program replay.

    The submitted records are replayed independently of the compiler's one reference
    trajectory. Relevance and deletion-minimality are scored separately by the verifier.
    """

    submitted = set(evidence_ids)
    if not submitted:
        return (
            not task.gold_evidence_ids
            and all(call.complete for call in task.reference_trace)
            and ProgramExecutor(ToolRegistry(snapshot)).execute(task.program).answer
            == task.gold_answer
        )
    return _same_answer(task, snapshot, tuple(sorted(submitted)))


def minimal_sufficient_subset(
    task: TaskArtifact,
    snapshot: PatientSnapshot,
    evidence_ids: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Return one deterministic deletion-minimal sufficient subset, if one exists."""

    submitted = tuple(sorted(set(evidence_ids)))
    if not evidence_satisfies_task(task, snapshot, submitted):
        return None
    candidates = {
        _shrink_witness(task, snapshot, submitted, reverse=False),
        _shrink_witness(task, snapshot, submitted, reverse=True),
    }
    return min(candidates, key=lambda item: (len(item), item))


def _same_answer(
    task: TaskArtifact,
    snapshot: PatientSnapshot,
    evidence_ids: tuple[str, ...],
) -> bool:
    records: list[EvidenceRecord] = []
    for evidence_id in evidence_ids:
        record = snapshot.record_by_id(evidence_id)
        if record is None:
            return False
        records.append(record)
    if not records:
        return evidence_satisfies_task(task, snapshot, ())
    restricted = snapshot_from_payloads(
        [record.payload for record in records],
        normalizer_version=snapshot.normalizer_version,
        relationship_rules_version=snapshot.relationship_rules_version,
        terminology_map_version=snapshot.terminology_map_version,
        logical_group_rules_version=snapshot.logical_group_rules_version,
    )
    try:
        result = ProgramExecutor(ToolRegistry(restricted)).execute(task.program)
    except (ProgramExecutionError, ValueError):
        return False
    return result.answer == task.gold_answer
