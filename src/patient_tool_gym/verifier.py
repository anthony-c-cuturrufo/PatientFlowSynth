from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache
from typing import Any, Literal

from pydantic import model_serializer, model_validator

from patient_tool_gym.evidence import EvidenceRecord, snapshot_from_payloads
from patient_tool_gym.evidence_contracts import (
    CompleteSearchCertificate,
    ObservedToolCall,
    complete_search_certificates_valid,
    evidence_satisfies_task,
    minimal_sufficient_subset,
    replayed_observed_evidence_ids,
)
from patient_tool_gym.program import (
    MEDICATION_DECISION_RULE_SEMANTICS,
    AllNode,
    ChartAbsenceNode,
    EvidenceStateNode,
    ExistsNode,
    ProgramExecutionError,
    ProgramExecutor,
    ReferenceCall,
    SelectNode,
    StrictModel,
    TaskProgram,
    ToolNode,
    TruthState,
    TruthStateAggregateNode,
    TypedValue,
    ValueRef,
    content_hash,
    node_references,
    program_output_schema,
    program_signature,
)
from patient_tool_gym.task import (
    ArtifactProvenance,
    CompleteSearchRequirement,
    EvidenceContract,
    EvidenceQualityContract,
    QuestionContract,
    TaskArtifact,
    VerificationReport,
    deterministic_task_id,
)
from patient_tool_gym.tools import ToolRegistry


@cache
def _tool_manifest_hash(registry_type: type[ToolRegistry]) -> str:
    """Cache only the immutable manifest hash, independently for each registry class."""

    return content_hash(registry_type.list_tools())


class SubmissionScore(StrictModel):
    passed: bool
    required_evidence_level: Literal["answer_sufficient", "audit_complete"] = "answer_sufficient"
    answer_pass: bool | None = None
    audit_pass: bool | None = None
    answer_correct: bool
    response_correct: bool | None = None
    clarification_appropriate: bool | None = None
    clarification_pass: bool | None = None
    evidence_valid: bool
    evidence_sufficient: bool | None = None
    answer_evidence_sufficient: bool | None = None
    audit_evidence_complete: bool | None = None
    evidence_minimal: bool | None = None
    evidence_precision: float
    evidence_recall: float
    matched_witness_index: int | None = None
    matched_answer_witness_index: int | None = None
    matched_audit_witness_index: int | None = None
    unsupported_evidence_ids: tuple[str, ...] = ()
    evidence_quality_applicable: bool = False
    answer_evidence_quality_pass: bool | None = None
    audit_evidence_quality_pass: bool | None = None
    complete_search_certificate_required: bool = False
    complete_search_certificate_valid: bool | None = None
    full_snapshot_citation: bool | None = None
    evidence_quality_failure_codes: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def serialize_legacy_quality_fields(self, handler: Any) -> dict[str, Any]:
        """Keep frozen scores written before evidence-quality endpoints byte-compatible."""

        payload = handler(self)
        quality_fields = {
            "evidence_quality_applicable",
            "answer_evidence_quality_pass",
            "audit_evidence_quality_pass",
            "complete_search_certificate_required",
            "complete_search_certificate_valid",
            "full_snapshot_citation",
            "evidence_quality_failure_codes",
        }
        if not quality_fields.intersection(self.model_fields_set):
            for field in quality_fields:
                payload.pop(field, None)
        return payload

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_exact_set_score(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        is_explicit_dual_layer_score = any(
            field in value
            for field in (
                "answer_evidence_sufficient",
                "audit_evidence_complete",
                "answer_pass",
                "audit_pass",
                "required_evidence_level",
            )
        )
        if not is_explicit_dual_layer_score:
            legacy_sufficient = migrated.get("evidence_sufficient")
            if legacy_sufficient is None:
                legacy_sufficient = migrated.get("evidence_valid")
            if legacy_sufficient is None:
                legacy_sufficient = (
                    migrated.get("evidence_precision") == 1.0
                    and migrated.get("evidence_recall") == 1.0
                    and not migrated.get("unsupported_evidence_ids")
                )
            migrated["evidence_valid"] = not migrated.get("unsupported_evidence_ids")
            migrated["evidence_sufficient"] = bool(legacy_sufficient)
            migrated["answer_evidence_sufficient"] = bool(legacy_sufficient)
            migrated["audit_evidence_complete"] = bool(legacy_sufficient)
            migrated["evidence_minimal"] = migrated.get("evidence_minimal", bool(legacy_sufficient))
            matched_index = migrated.get("matched_witness_index")
            if "matched_witness_index" not in migrated and legacy_sufficient:
                matched_index = 0
                migrated["matched_witness_index"] = matched_index
            migrated["matched_answer_witness_index"] = matched_index
            migrated["matched_audit_witness_index"] = matched_index

        answer_evidence_sufficient = migrated.get("answer_evidence_sufficient")
        if answer_evidence_sufficient is None:
            answer_evidence_sufficient = (
                migrated["evidence_sufficient"]
                if migrated.get("evidence_sufficient") is not None
                else migrated["evidence_valid"]
            )
            migrated["answer_evidence_sufficient"] = answer_evidence_sufficient
        if migrated.get("evidence_sufficient") is None:
            migrated["evidence_sufficient"] = answer_evidence_sufficient
        audit_evidence_complete = migrated.get("audit_evidence_complete")
        if audit_evidence_complete is None:
            audit_evidence_complete = answer_evidence_sufficient
            migrated["audit_evidence_complete"] = audit_evidence_complete
        if migrated.get("evidence_minimal") is None:
            migrated["evidence_minimal"] = answer_evidence_sufficient
        answer_correct = bool(migrated.get("answer_correct"))
        response_correct = migrated.get("response_correct")
        if response_correct is None:
            response_correct = answer_correct
            migrated["response_correct"] = response_correct
        if migrated.get("clarification_appropriate") is None:
            migrated["clarification_appropriate"] = False
        if migrated.get("clarification_pass") is None:
            migrated["clarification_pass"] = False
        if migrated.get("answer_pass") is None:
            migrated["answer_pass"] = response_correct and answer_evidence_sufficient
        if migrated.get("audit_pass") is None:
            migrated["audit_pass"] = response_correct and audit_evidence_complete
        matched_index = migrated.get("matched_witness_index")
        if migrated.get("matched_answer_witness_index") is None and matched_index is not None:
            migrated["matched_answer_witness_index"] = matched_index
        if migrated.get("matched_audit_witness_index") is None and matched_index is not None:
            migrated["matched_audit_witness_index"] = matched_index
        return migrated


def verify_task_artifact(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    *,
    policy: Literal["standard", "selector_hidden_ablation"] = "standard",
) -> VerificationReport:
    errors: list[str] = []
    checks: list[str] = []
    verifier_version = _verifier_version(
        artifact.program,
        artifact.schema_version,
        evidence_quality_contract=artifact.evidence_quality_contract,
    )

    if artifact.patient_snapshot_id != registry.snapshot_id:
        errors.append("snapshot_id_mismatch")
    else:
        checks.append("snapshot_bound")

    expected_task_id = deterministic_task_id(
        artifact.patient_snapshot_id,
        artifact.program,
        artifact.family_id,
        artifact.construction_type,
    )
    if artifact.task_id == expected_task_id:
        checks.append("task_identity_recomputed")
    else:
        errors.append("task_id_mismatch")

    expected_manifest_hash = _tool_manifest_hash(type(registry))
    if artifact.provenance.tool_manifest_hash == expected_manifest_hash:
        checks.append("tool_manifest_bound")
    else:
        errors.append("tool_manifest_mismatch")

    if artifact.provenance.normalizer_version == registry.snapshot.normalizer_version:
        checks.append("normalizer_version_bound")
    else:
        errors.append("normalizer_version_mismatch")

    programs_to_validate = [artifact.program]
    if artifact.evidence_contract is not None:
        programs_to_validate.extend(
            program
            for program in (
                artifact.evidence_contract.answer_program,
                artifact.evidence_contract.audit_program,
            )
            if program is not None
        )
    try:
        for program in programs_to_validate:
            program_output_schema(program)
    except ValueError:
        errors.append("program_output_schema_invalid")
    else:
        checks.append("program_output_schema_valid")

    try:
        replay = ProgramExecutor(registry).execute(artifact.program)
    except ProgramExecutionError:
        errors.append("execution_failed")
        return VerificationReport(
            verifier_version=verifier_version,
            policy=policy,
            passed=False,
            checks=tuple(checks),
            errors=tuple(errors),
        )
    if replay.answer == artifact.gold_answer:
        checks.append("typed_answer_replayed")
    else:
        errors.append("typed_answer_mismatch")

    if replay.evidence_ids == artifact.gold_evidence_ids:
        checks.append("evidence_replayed")
    else:
        errors.append("evidence_mismatch")

    if replay.reference_trace == artifact.reference_trace:
        checks.append("reference_trace_replayed")
    else:
        errors.append("reference_trace_mismatch")

    if replay.complexity == artifact.difficulty:
        checks.append("complexity_recomputed")
    else:
        errors.append("complexity_mismatch")

    if all(call.complete for call in replay.reference_trace):
        checks.append("reference_searches_complete")
    else:
        errors.append("incomplete_reference_search")

    contract_tools = _evidence_contract_tool_names(artifact.evidence_contract)
    used_tools = {call.tool for call in replay.reference_trace} | contract_tools
    expected_allowed_tools = tuple(sorted(used_tools | {"submit_answer"}))
    forbidden_used = set(artifact.forbidden_tools) & used_tools
    template_bound = _is_template_bound(artifact)
    forbidden_policy_valid = not forbidden_used and (
        not template_bound or artifact.forbidden_tools == ()
    )
    if artifact.allowed_tools == expected_allowed_tools and forbidden_policy_valid:
        checks.append("tool_policy_recomputed")
    else:
        errors.append("tool_policy_mismatch")

    _verify_structure(artifact, checks, errors)
    _verify_question(artifact, registry, checks, errors, policy=policy)
    _verify_lineage(artifact, registry, checks, errors)
    _verify_evidence_witnesses(artifact, registry, checks, errors)
    _verify_evidence_contract(artifact, registry, checks, errors)
    _verify_evidence_quality_contract(artifact, registry, checks, errors)

    return VerificationReport(
        verifier_version=verifier_version,
        policy=policy,
        passed=not errors,
        checks=tuple(checks),
        errors=tuple(errors),
    )


def score_submission(
    artifact: TaskArtifact,
    *,
    answer: TypedValue,
    evidence_ids: tuple[str, ...],
    registry: ToolRegistry,
    complete_search_certificates: tuple[CompleteSearchCertificate, ...] = (),
    observed_tool_calls: tuple[ObservedToolCall, ...] = (),
) -> SubmissionScore:
    submitted = set(evidence_ids)
    answer_witnesses = tuple(
        set(witness) for witness in artifact.answer_sufficient_evidence_witnesses()
    )
    audit_witnesses = tuple(
        set(witness) for witness in artifact.audit_complete_evidence_witnesses()
    )
    unknown = tuple(
        sorted(
            evidence_id
            for evidence_id in submitted
            if registry.snapshot.record_by_id(evidence_id) is None
        )
    )

    def witness_metrics(witness: set[str]) -> tuple[float, float, float]:
        true_positive = len(submitted & witness)
        precision = true_positive / len(submitted) if submitted else (1.0 if not witness else 0.0)
        recall = true_positive / len(witness) if witness else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, precision, recall

    ranked = [(*witness_metrics(witness), index) for index, witness in enumerate(answer_witnesses)]
    _, precision, recall, _ = max(ranked, key=lambda item: (item[0], -item[3]))
    matched_answer_index = next(
        (index for index, witness in enumerate(answer_witnesses) if submitted == witness),
        None,
    )
    matched_audit_index = next(
        (index for index, witness in enumerate(audit_witnesses) if submitted == witness),
        None,
    )
    contained_answer_witnesses = tuple(
        witness for witness in answer_witnesses if witness.issubset(submitted)
    )
    contained_audit_witnesses = tuple(
        witness for witness in audit_witnesses if witness.issubset(submitted)
    )
    answer_correct = answer == artifact.gold_answer
    submitted_clarification = answer.kind == "clarification"
    clarification_required = artifact.requires_clarification()
    clarification_appropriate = bool(
        submitted_clarification
        and clarification_required
        and artifact.question_contract is not None
        and answer.value == artifact.question_contract.reason
    )
    clarification_pass = clarification_appropriate and not evidence_ids
    response_correct = (
        clarification_appropriate
        if submitted_clarification or clarification_required
        else answer_correct
    )
    quality_contract = artifact.evidence_quality_contract
    snapshot_evidence_ids = {record.evidence_id for record in registry.snapshot.records}
    full_snapshot_citation = bool(snapshot_evidence_ids) and submitted == snapshot_evidence_ids
    answer_evidence_sufficient = bool(contained_answer_witnesses) and not unknown
    minimal_subset: tuple[str, ...] | None = None
    if not unknown:
        declared_minimal = (
            min(
                (tuple(sorted(witness)) for witness in contained_answer_witnesses),
                key=lambda witness: (len(witness), witness),
            )
            if contained_answer_witnesses
            else None
        )
        skip_executable_minimization = bool(
            quality_contract is not None
            and contained_answer_witnesses
            and (full_snapshot_citation or matched_answer_index is None)
        )
        if skip_executable_minimization:
            executable_minimal = None
        else:
            program_sufficient = evidence_satisfies_task(
                artifact,
                registry.snapshot,
                tuple(sorted(submitted)),
            )
            answer_evidence_sufficient = answer_evidence_sufficient or program_sufficient
            executable_minimal = (
                minimal_sufficient_subset(
                    artifact,
                    registry.snapshot,
                    tuple(sorted(submitted)),
                )
                if program_sufficient and submitted
                else None
            )
        minimal_candidates = tuple(
            witness for witness in (declared_minimal, executable_minimal) if witness is not None
        )
        if minimal_candidates:
            minimal_subset = min(
                minimal_candidates,
                key=lambda witness: (len(witness), witness),
            )
            if submitted:
                precision = len(minimal_subset) / len(submitted)
                recall = 1.0
    audit_evidence_complete = bool(contained_audit_witnesses) and not unknown
    if artifact.evidence_contract is None:
        # A legacy task declared only one evidence layer. Preserve its historical
        # pass semantics instead of inventing a stronger retroactive audit contract.
        audit_evidence_complete = answer_evidence_sufficient
        matched_audit_index = matched_answer_index
    evidence_minimal = bool(
        answer_evidence_sufficient
        and (not submitted or (minimal_subset is not None and submitted == set(minimal_subset)))
    )
    required_level = artifact.required_evidence_level()
    if clarification_required:
        answer_pass = clarification_pass
        audit_pass = clarification_pass
    else:
        answer_pass = response_correct and answer_evidence_sufficient
        audit_pass = response_correct and audit_evidence_complete
    quality_fields: dict[str, Any] = {}
    if quality_contract is not None:
        replayed_observed_ids = replayed_observed_evidence_ids(
            registry,
            observed_tool_calls,
        )
        submitted_evidence_observed = submitted.issubset(replayed_observed_ids)
        certificate_required = bool(quality_contract.complete_search_requirements)
        certificate_satisfied = complete_search_certificates_valid(
            artifact,
            registry,
            complete_search_certificates,
            observed_tool_calls,
        )
        if clarification_required:
            answer_quality = audit_quality = None
            certificate_endpoint = None
            full_snapshot_endpoint = None
            failures = []
        else:
            answer_quality = bool(
                not unknown
                and response_correct
                and submitted_evidence_observed
                and matched_answer_index is not None
                and certificate_satisfied
                and not full_snapshot_citation
            )
            audit_quality = bool(
                not unknown
                and response_correct
                and submitted_evidence_observed
                and matched_audit_index is not None
                and certificate_satisfied
                and not full_snapshot_citation
            )
            certificate_endpoint = certificate_satisfied if certificate_required else None
            full_snapshot_endpoint = full_snapshot_citation
            failures = []
            if unknown:
                failures.append("unsupported_evidence_ids")
            if not response_correct:
                failures.append("response_incorrect")
            if not submitted_evidence_observed:
                failures.append("unobserved_evidence_ids")
            if matched_answer_index is None:
                failures.append("answer_not_exact_declared_witness")
            if matched_audit_index is None:
                failures.append("audit_not_exact_declared_witness")
            if not certificate_satisfied:
                failures.append("complete_search_certificate_invalid")
            if full_snapshot_citation:
                failures.append("full_snapshot_citation")
        quality_fields = {
            "evidence_quality_applicable": True,
            "answer_evidence_quality_pass": answer_quality,
            "audit_evidence_quality_pass": audit_quality,
            "complete_search_certificate_required": certificate_required,
            "complete_search_certificate_valid": certificate_endpoint,
            "full_snapshot_citation": full_snapshot_endpoint,
            "evidence_quality_failure_codes": tuple(failures),
        }
    return SubmissionScore(
        passed=answer_pass if required_level == "answer_sufficient" else audit_pass,
        required_evidence_level=required_level,
        answer_pass=answer_pass,
        audit_pass=audit_pass,
        answer_correct=answer_correct,
        response_correct=response_correct,
        clarification_appropriate=clarification_appropriate,
        clarification_pass=clarification_pass,
        evidence_valid=not unknown,
        evidence_sufficient=answer_evidence_sufficient,
        answer_evidence_sufficient=answer_evidence_sufficient,
        audit_evidence_complete=audit_evidence_complete,
        evidence_minimal=evidence_minimal,
        evidence_precision=round(precision, 6),
        evidence_recall=round(recall, 6),
        matched_witness_index=matched_answer_index,
        matched_answer_witness_index=matched_answer_index,
        matched_audit_witness_index=matched_audit_index,
        unsupported_evidence_ids=unknown,
        **quality_fields,
    )


def _verify_evidence_witnesses(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    checks: list[str],
    errors: list[str],
) -> None:
    witnesses = artifact.valid_evidence_witnesses()
    unknown = sorted(
        {
            evidence_id
            for witness in witnesses
            for evidence_id in witness
            if registry.snapshot.record_by_id(evidence_id) is None
        }
    )
    if unknown:
        errors.append("evidence_witness_unknown_ids")
        return
    if artifact.evidence_witnesses and tuple(sorted(set(artifact.gold_evidence_ids))) not in (
        artifact.evidence_witnesses
    ):
        errors.append("canonical_evidence_witness_missing")
        return
    checks.append("evidence_witnesses_valid")
    if len(witnesses) > 1:
        for witness in witnesses:
            records = [registry.snapshot.record_by_id(evidence_id) for evidence_id in witness]
            if not records or any(record is None for record in records):
                errors.append("alternative_evidence_witness_not_executable")
                return
            restricted = snapshot_from_payloads(
                [record.payload for record in records if record is not None],
                normalizer_version=registry.snapshot.normalizer_version,
                relationship_rules_version=registry.snapshot.relationship_rules_version,
                terminology_map_version=registry.snapshot.terminology_map_version,
                logical_group_rules_version=registry.snapshot.logical_group_rules_version,
            )
            try:
                replay = ProgramExecutor(ToolRegistry(restricted)).execute(artifact.program)
            except ProgramExecutionError:
                errors.append("alternative_evidence_witness_not_executable")
                return
            if replay.answer != artifact.gold_answer or not set(replay.evidence_ids).issubset(
                witness
            ):
                errors.append("alternative_evidence_witness_not_sufficient")
                return
        checks.append("alternative_evidence_witnesses_executable")


def _verify_evidence_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    checks: list[str],
    errors: list[str],
) -> None:
    contract = artifact.evidence_contract
    if contract is None:
        checks.append("legacy_single_layer_evidence_contract")
        return

    witnesses = (
        *contract.answer_sufficient_witnesses,
        *contract.audit_complete_witnesses,
    )
    if any(
        registry.snapshot.record_by_id(evidence_id) is None
        for witness in witnesses
        for evidence_id in witness
    ):
        errors.append("evidence_contract_unknown_ids")
        return

    answer_program = contract.answer_program or artifact.program
    audit_program = contract.audit_program or artifact.program
    required_program = (
        answer_program if contract.required_level == "answer_sufficient" else audit_program
    )
    if artifact.program != required_program:
        errors.append("evidence_contract_gating_program_mismatch")
        return
    if not all(
        _evidence_program_exactly_consumes_witness(
            artifact,
            registry,
            program=answer_program,
            witness=witness,
        )
        for witness in contract.answer_sufficient_witnesses
    ):
        errors.append("answer_sufficient_witness_not_executable")
        return
    if not all(
        _evidence_program_exactly_consumes_witness(
            artifact,
            registry,
            program=audit_program,
            witness=witness,
        )
        for witness in contract.audit_complete_witnesses
    ):
        errors.append("audit_complete_witness_not_executable")
        return
    checks.append("dual_layer_evidence_contract_valid")


def derive_evidence_quality_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
) -> EvidenceQualityContract:
    """Derive the threshold-free quality policy and closed-world search bindings."""

    replay = ProgramExecutor(registry).execute(artifact.program)
    if replay.answer != artifact.gold_answer:
        raise ValueError("cannot derive evidence quality from a task whose answer does not replay")
    return _derive_evidence_quality_contract(
        program=artifact.program,
        reference_trace=replay.reference_trace,
        gold_answer=replay.answer,
        snapshot_id=registry.snapshot_id,
        registry=registry,
    )


def attach_evidence_quality_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
) -> TaskArtifact:
    """Return a newly hash-distinct artifact with v1 evidence-quality semantics attached."""

    contract = derive_evidence_quality_contract(artifact, registry)
    candidate = artifact.model_copy(
        update={
            "evidence_quality_contract": contract,
            "verification": VerificationReport(
                verifier_version=_verifier_version(
                    artifact.program,
                    artifact.schema_version,
                    evidence_quality_contract=contract,
                ),
                policy=artifact.verification.policy,
                passed=False,
            ),
        }
    )
    report = verify_task_artifact(
        candidate,
        registry,
        policy=artifact.verification.policy,
    )
    return candidate.model_copy(update={"verification": report})


def _verify_evidence_quality_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    checks: list[str],
    errors: list[str],
) -> None:
    if artifact.evidence_quality_contract is None:
        checks.append("legacy_evidence_quality_unscored")
        return
    try:
        expected = derive_evidence_quality_contract(artifact, registry)
    except (ProgramExecutionError, ValueError):
        errors.append("evidence_quality_contract_not_executable")
        return
    if artifact.evidence_quality_contract != expected:
        errors.append("evidence_quality_contract_mismatch")
        return
    checks.append("evidence_quality_contract_executable")


def _derive_evidence_quality_contract(
    *,
    program: TaskProgram,
    reference_trace: tuple[ReferenceCall, ...],
    gold_answer: TypedValue,
    snapshot_id: str,
    registry: ToolRegistry,
) -> EvidenceQualityContract:
    nodes = {node.id: node for node in program.nodes}
    ancestor_ids: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in ancestor_ids:
            return
        ancestor_ids.add(node_id)
        node = nodes[node_id]
        for reference in node_references(node):
            visit(reference.node_id)

    visit(program.output.node_id)
    claims: list[tuple[Any, str]] = [
        (node, "chart_absence")
        for node in program.nodes
        if node.id in ancestor_ids and isinstance(node, ChartAbsenceNode)
    ]
    claims.extend(
        (node, "empty_search_boolean")
        for node in program.nodes
        if node.id in ancestor_ids and isinstance(node, ExistsNode)
    )
    state_semantics = {
        TruthState.SUPPORTED.value: "exclusive_supported",
        TruthState.REFUTED.value: "exclusive_refuted",
        TruthState.INSUFFICIENT_EVIDENCE.value: "insufficient_evidence",
    }
    if program.schema_version == "patient-tool-gym.program.v4":
        state_semantics[TruthState.CONFLICTING_EVIDENCE.value] = "conflicting_evidence"
    for index, node in enumerate(program.nodes):
        if node.id not in ancestor_ids or not isinstance(node, EvidenceStateNode):
            continue
        prefix = TaskProgram(
            schema_version=program.schema_version,
            nodes=program.nodes[: index + 1],
            output=ValueRef(node_id=node.id),
        )
        branch_answer = ProgramExecutor(registry).execute(prefix).answer
        semantics = (
            state_semantics.get(branch_answer.value)
            if branch_answer.kind == "truth_state"
            else None
        )
        if semantics is None and branch_answer == TypedValue(
            kind="truth_state",
            value=TruthState.CONFLICTING_EVIDENCE.value,
        ):
            # Frozen v2/v3 contracts did not declare a closed-search obligation
            # for conflict. Preserve those artifact counts exactly; program.v4
            # adds the explicit branch-level certificate.
            continue
        if semantics is None:
            raise ValueError("evidence-state branch did not produce a known truth state")
        claims.append((node, semantics))

    trace_by_node = {call.node_id: call for call in reference_trace}
    requirements = []
    seen_claims: set[str] = set()
    for claim, claim_semantics in claims:
        if claim.id in seen_claims:
            continue
        seen_claims.add(claim.id)
        source_node = nodes[claim.source.node_id]
        source_path = list(claim.source.path)
        while isinstance(source_node, SelectNode):
            source_path = [*source_node.source.path, *source_path]
            source_node = nodes[source_node.source.node_id]
        if not isinstance(source_node, ToolNode):
            raise ValueError("closed-world claim must resolve to one executable tool search")
        reference_call = trace_by_node.get(source_node.id)
        if reference_call is None or reference_call.tool != source_node.tool:
            raise ValueError("closed-world search has no matching reference call")
        result = registry.call(reference_call.tool, reference_call.model_arguments)
        records = result.get("records")
        total_matches = result.get("total_matches")
        if isinstance(claim, ExistsNode):
            selected: Any = result
            try:
                for path_part in source_path:
                    selected = selected[path_part]
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError("closed-world exists path cannot be replayed") from exc
            if bool(selected):
                continue
        if isinstance(claim, ChartAbsenceNode) and isinstance(records, list) and records:
            continue
        if not (
            reference_call.complete
            and result.get("ok") is True
            and result.get("complete") is True
            and result.get("next_cursor") is None
            and result.get("snapshot_id") == snapshot_id
            and isinstance(records, list)
            and type(total_matches) is int
            and total_matches >= 0
            and len(records) == total_matches
            and content_hash(result) == reference_call.result_hash
        ):
            raise ValueError("closed-world reference search is not complete and snapshot-bound")
        requirements.append(
            CompleteSearchRequirement(
                claim_semantics=claim_semantics,
                claim_node_id=claim.id,
                search_node_id=source_node.id,
                snapshot_id=snapshot_id,
                tool=reference_call.tool,
                arguments=reference_call.model_arguments,
                result_hash=reference_call.result_hash,
            )
        )
    return EvidenceQualityContract(complete_search_requirements=tuple(requirements))


def _evidence_program_exactly_consumes_witness(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    *,
    program: TaskProgram,
    witness: tuple[str, ...],
) -> bool:
    if witness:
        records = [registry.snapshot.record_by_id(evidence_id) for evidence_id in witness]
        if any(record is None for record in records):
            return False
        snapshot = snapshot_from_payloads(
            [record.payload for record in records if record is not None],
            normalizer_version=registry.snapshot.normalizer_version,
            relationship_rules_version=registry.snapshot.relationship_rules_version,
            terminology_map_version=registry.snapshot.terminology_map_version,
            logical_group_rules_version=registry.snapshot.logical_group_rules_version,
        )
    else:
        # An empty witness represents a chart-level absence claim. It must replay
        # against the complete bound snapshot and consume no positive record.
        snapshot = registry.snapshot
    try:
        replay = ProgramExecutor(ToolRegistry(snapshot)).execute(program)
    except (ProgramExecutionError, ValueError):
        return False
    return replay.answer == artifact.gold_answer and set(replay.evidence_ids) == set(witness)


def _evidence_contract_tool_names(contract: EvidenceContract | None) -> set[str]:
    if contract is None:
        return set()
    return {
        node.tool
        for program in (contract.answer_program, contract.audit_program)
        if program is not None
        for node in program.nodes
        if isinstance(node, ToolNode)
    }


def _verify_structure(
    artifact: TaskArtifact,
    checks: list[str],
    errors: list[str],
) -> None:
    if artifact.construction_type in {"depth", "mixed"}:
        bound_tool_nodes = [
            node
            for node in artifact.program.nodes
            if isinstance(node, ToolNode) and node_references(node)
        ]
        if bound_tool_nodes and artifact.difficulty.dependency_depth >= 2:
            checks.append("dependency_binding_real")
        else:
            errors.append("dependency_not_real")

    if artifact.construction_type in {"width", "mixed"}:
        output_node = next(
            node for node in artifact.program.nodes if node.id == artifact.program.output.node_id
        )
        legacy_boolean_width = isinstance(output_node, AllNode)
        typed_state_width = (
            isinstance(output_node, TruthStateAggregateNode) and len(output_node.branches) >= 2
        )
        if (legacy_boolean_width or typed_state_width) and (
            artifact.difficulty.independent_width >= 2
        ):
            checks.append("width_branches_independent")
            checks.append(
                "width_truth_state_reducer_structural"
                if typed_state_width
                else "width_all_reducer_structural"
            )
        else:
            errors.append("branch_not_independent_or_necessary")


def _verify_question(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    checks: list[str],
    errors: list[str],
    *,
    policy: Literal["standard", "selector_hidden_ablation"],
) -> None:
    question = artifact.question.casefold()
    leaked = any(evidence_id.casefold() in question for evidence_id in artifact.gold_evidence_ids)
    leaked = leaked or "urn:uuid:" in question or "source.file" in question
    if artifact.gold_answer.kind == "truth_state":
        # Naming the complete closed vocabulary does not reveal which member is
        # gold. Naming only the gold member still does.
        truth_labels = {"supported", "refuted", "insufficient evidence"}
        complete_labels = truth_labels | {"conflicting evidence"}
        if not all(label in question for label in truth_labels) and not all(
            label in question for label in complete_labels
        ):
            leaked = leaked or str(artifact.gold_answer.value).replace("_", " ") in question
    elif artifact.gold_answer.kind not in {"boolean", "null"}:
        rendered_answer = str(artifact.gold_answer.value).casefold()
        leaked = leaked or (len(rendered_answer) >= 3 and rendered_answer in question)
    if leaked:
        errors.append("question_leakage")
    else:
        checks.append("question_answer_masked")

    if artifact.required_evidence_level() == "audit_complete":
        if "provenance" in question:
            checks.append("audit_evidence_requirement_disclosed")
        else:
            errors.append("audit_evidence_requirement_not_disclosed")

    if not question.strip() and policy == "standard" and artifact.question_contract is None:
        checks.append("question_identifiability_not_evaluated_for_empty_compiler_artifact")
        return

    selectors = {
        (node.selector.system, node.selector.code)
        for node in artifact.program.nodes
        if isinstance(node, EvidenceStateNode) and node.selector is not None
    }
    if not selectors:
        if policy == "selector_hidden_ablation":
            errors.append("selector_hidden_policy_without_program_selector")
        elif artifact.requires_clarification():
            errors.append("clarification_contract_without_program_selector")
        return

    missing_selectors: list[tuple[str, str]] = []
    for system, code in sorted(selectors):
        displays = {
            record.display.casefold()
            for record in registry.snapshot.records
            if (
                record.code.get("system") == system
                and record.code.get("code") == code
                and record.display
            )
        }
        if code.casefold() not in question and not any(
            _selector_display_identified(display, question) for display in displays
        ):
            missing_selectors.append((system, code))

    if policy == "selector_hidden_ablation":
        ablation = artifact.metadata.get("interface_ablation")
        declared_hidden = (
            isinstance(ablation, dict) and ablation.get("selector_visibility") == "hidden"
        )
        if not declared_hidden:
            errors.append("selector_hidden_policy_without_hidden_ablation")
            return
        if not missing_selectors:
            errors.append("selector_hidden_policy_selector_still_identifiable")
            return
        checks.append("question_selector_hidden_proved")
        if artifact.question_contract is None:
            # Frozen hidden-selector ablations predate the explicit response contract.
            checks.append("legacy_hidden_question_contract_accepted")
        elif artifact.requires_clarification():
            checks.append("hidden_question_clarification_contract_declared")
        else:
            errors.append("selector_hidden_policy_without_clarification_contract")
            return
        checks.append("question_selector_hidden_by_trusted_ablation_policy")
        return
    if missing_selectors:
        errors.append("question_selector_not_identifiable")
    elif artifact.requires_clarification():
        errors.append("clarification_contract_for_identifiable_question")
    else:
        checks.append("question_selector_identifiable")


def _selector_display_identified(display: str, question: str) -> bool:
    if display in question:
        return True

    def token(value: str) -> str:
        if value.startswith("vaccin"):
            return "vaccin"
        return value.removesuffix("s") if len(value) > 4 else value

    stop = {
        "and",
        "disease",
        "disorder",
        "finding",
        "item",
        "mg",
        "mellitus",
        "observation",
        "oral",
        "procedure",
        "reported",
        "score",
        "tablet",
        "the",
        "total",
    }
    display_tokens = {
        token(item)
        for item in re.findall(r"[a-z0-9]+", display)
        if item not in stop and len(item) > 1
    }
    question_tokens = {token(item) for item in re.findall(r"[a-z0-9]+", question)}
    overlap = display_tokens & question_tokens
    # One generic shared word is not enough to establish a unique selector.
    return len(overlap) >= 2 or (len(display_tokens) == 1 and overlap == display_tokens)


def _verify_lineage(
    artifact: TaskArtifact,
    registry: ToolRegistry,
    checks: list[str],
    errors: list[str],
) -> None:
    requires_lineage = _is_template_bound(artifact)
    if not requires_lineage:
        return

    if "compositional_release" in artifact.metadata:
        from patient_tool_gym.compositional_release import (
            verify_compositional_task_contract,
        )

        if verify_compositional_task_contract(artifact, registry):
            checks.append("compositional_release_contract_recomputed")
        else:
            errors.append("compositional_release_contract_invalid")
        return

    if artifact.provenance.generator == "patient_v1_compiler" or artifact.family_id.startswith(
        "patient-v1-family-"
    ):
        # Imported lazily because the V1 compiler uses compile_task from this module.
        from patient_tool_gym.patient_v1 import verify_patient_v1_contract

        if verify_patient_v1_contract(artifact, registry):
            checks.append("patient_v1_contract_recomputed")
        else:
            errors.append("patient_v1_contract_invalid")
        return

    if artifact.provenance.generator == "patient_p2_compiler" or artifact.family_id.startswith(
        "patient-p2-family-"
    ):
        from patient_tool_gym.patient_p2 import verify_patient_p2_contract

        if verify_patient_p2_contract(artifact, registry):
            checks.append("patient_p2_contract_recomputed")
        else:
            errors.append("patient_p2_contract_invalid")
        return

    lineage = artifact.metadata.get("lineage")
    signature = artifact.metadata.get("program_signature")
    valid = isinstance(lineage, dict)
    if not valid:
        errors.append("lineage_invalid")
        return

    expected_role = {
        "atomic": "atomic",
        "depth": "depth_2",
        "width": "width_2",
    }.get(artifact.construction_type)
    source_datasets = {record.source_dataset for record in registry.snapshot.records}
    valid = valid and artifact.provenance.generator == "deterministic_template"
    valid = valid and artifact.provenance.seed == 0
    valid = valid and len(source_datasets) == 1
    valid = valid and artifact.provenance.dataset in source_datasets
    valid = valid and lineage.get("family_id") == artifact.family_id
    valid = valid and lineage.get("sibling_role") == expected_role
    valid = valid and lineage.get("template_id") == "wellness_observation_presence"
    valid = valid and lineage.get("template_version") == "1"
    valid = valid and artifact.provenance.generator_version == "1"
    valid = valid and signature == program_signature(artifact.program)

    root_task_id = lineage.get("root_task_id")
    derived_from = lineage.get("derived_from_task_id")
    if artifact.construction_type == "atomic":
        valid = valid and root_task_id == artifact.task_id
        valid = valid and derived_from is None and not artifact.parent_task_ids
    else:
        valid = valid and isinstance(root_task_id, str)
        valid = valid and derived_from == root_task_id
        valid = valid and artifact.parent_task_ids == (root_task_id,)

    anchor_ids = lineage.get("anchor_evidence_ids")
    encounter_id = lineage.get("anchor_encounter_evidence_id")
    valid = valid and isinstance(anchor_ids, list)
    if not isinstance(anchor_ids, list):
        errors.append("lineage_invalid")
        return
    if not all(isinstance(value, str) for value in anchor_ids):
        errors.append("lineage_invalid")
        return
    valid = valid and len(anchor_ids) == 3
    valid = valid and anchor_ids == sorted(set(anchor_ids))
    valid = valid and isinstance(encounter_id, str) and encounter_id in anchor_ids

    records = [registry.snapshot.record_by_id(value) for value in anchor_ids]
    valid = valid and all(record is not None for record in records)
    encounter = (
        registry.snapshot.record_by_id(encounter_id) if isinstance(encounter_id, str) else None
    )
    valid = valid and encounter is not None and encounter.resource_type == "Encounter"

    observation_ids: set[str] = set()
    primary_ids: set[str] = set()
    companion_ids: set[str] = set()
    if encounter is not None:
        explicitly_related = {
            record.evidence_id
            for record in registry.snapshot.related_records(
                encounter.evidence_id,
                relation="encounter",
                direction="incoming",
            )
        }
        for record in records:
            if record is None or record.resource_type != "Observation":
                continue
            observation_ids.add(record.evidence_id)
            if _record_has_coding(record, "http://loinc.org", "29463-7"):
                primary_ids.add(record.evidence_id)
            if any(
                _record_has_coding(record, "http://loinc.org", code)
                for code in ("8302-2", "85354-9")
            ):
                companion_ids.add(record.evidence_id)
        valid = valid and len(observation_ids) == 2
        valid = valid and len(primary_ids) == 1 and len(companion_ids) == 1
        valid = valid and observation_ids.issubset(explicitly_related)

    gold_ids = set(artifact.gold_evidence_ids)
    if artifact.construction_type == "atomic":
        valid = valid and gold_ids == primary_ids
    elif artifact.construction_type == "depth":
        valid = valid and gold_ids == primary_ids | {encounter_id}
    elif artifact.construction_type == "width":
        valid = valid and gold_ids == primary_ids | companion_ids

    expected_question = _expected_template_question(
        artifact,
        encounter=encounter,
        companion_records=[
            record
            for record in records
            if record is not None and record.evidence_id in companion_ids
        ],
    )
    valid = valid and artifact.question == expected_question

    expected_family_digest = content_hash(
        {
            "snapshot_id": artifact.patient_snapshot_id,
            "template_id": "wellness_observation_presence",
            "template_version": "1",
            "anchor_evidence_ids": anchor_ids,
        }
    ).removeprefix("sha256:")
    valid = valid and artifact.family_id == f"family-{expected_family_digest[:24]}"

    if valid:
        checks.append("lineage_recomputed")
    else:
        errors.append("lineage_invalid")


def _record_has_coding(record: EvidenceRecord, system: str, code: str) -> bool:
    coding = record.code
    if coding.get("system") == system and coding.get("code") == code:
        return True
    codings = coding.get("codings")
    return isinstance(codings, list) and any(
        isinstance(item, dict) and item.get("system") == system and item.get("code") == code
        for item in codings
    )


def _is_template_bound(artifact: TaskArtifact) -> bool:
    return (
        artifact.provenance.generator == "deterministic_template"
        or artifact.provenance.generator == "patient_v1_compiler"
        or artifact.provenance.generator == "patient_p2_compiler"
        or artifact.family_id.startswith("family-")
        or artifact.family_id.startswith("patient-v1-family-")
        or artifact.family_id.startswith("patient-p2-family-")
        or "compositional_release" in artifact.metadata
        or "lineage" in artifact.metadata
        or "program_signature" in artifact.metadata
    )


def _expected_template_question(
    artifact: TaskArtifact,
    *,
    encounter: EvidenceRecord | None,
    companion_records: list[EvidenceRecord],
) -> str | None:
    if encounter is None or not encounter.local_date:
        return None
    try:
        rendered_date = _render_date(encounter.local_date)
    except (IndexError, ValueError):
        return None
    if artifact.construction_type == "atomic":
        return f"Was my body weight recorded on {rendered_date}?"
    if artifact.construction_type == "width" and len(companion_records) == 1:
        companion = companion_records[0]
        if _record_has_coding(companion, "http://loinc.org", "8302-2"):
            companion_name = "height"
        elif _record_has_coding(companion, "http://loinc.org", "85354-9"):
            companion_name = "blood pressure"
        else:
            return None
        return f"On {rendered_date}, were both my body weight and {companion_name} recorded?"
    if artifact.construction_type == "depth":
        search_node = next(
            (
                node
                for node in artifact.program.nodes
                if isinstance(node, ToolNode) and node.tool == "search_encounters"
            ),
            None,
        )
        query = search_node.arguments.get("text") if search_node is not None else None
        if not isinstance(query, str):
            return None
        encounter_name = {
            "wellness": "wellness visit",
            "Encounter for check up": "check-up",
            "General examination of patient": "general examination",
        }.get(query)
        if encounter_name is not None:
            return f"Was my body weight recorded at my most recent {encounter_name}?"
    return None


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


def compile_task(
    *,
    registry: ToolRegistry,
    program: TaskProgram,
    family_id: str,
    construction_type: str,
    question: str | None = "",
    question_renderer: Callable[[], str] | None = None,
    parent_task_ids: tuple[str, ...] = (),
    forbidden_tools: tuple[str, ...] = (),
    dataset: str = "fixture",
    normalizer_version: str | None = None,
    generator: str = "hand_authored",
    generator_version: str = "1",
    seed: int | None = None,
    evidence_contract: EvidenceContract | None = None,
    evidence_quality_policy: Literal["exact_witness_v1"] | None = None,
    question_contract: QuestionContract | None = None,
) -> TaskArtifact:
    if question_renderer is not None and question not in {None, ""}:
        raise ValueError("question and question_renderer are mutually exclusive")
    if evidence_quality_policy not in {None, "exact_witness_v1"}:
        raise ValueError(f"unsupported evidence quality policy: {evidence_quality_policy}")
    deprecated_rules = sorted(
        {
            node.decision_rule_id
            for node in program.nodes
            if isinstance(node, EvidenceStateNode)
            and node.decision_rule_id in MEDICATION_DECISION_RULE_SEMANTICS
            and MEDICATION_DECISION_RULE_SEMANTICS[node.decision_rule_id].deprecated_for_new_tasks
        }
    )
    if deprecated_rules:
        replacements = sorted(
            {
                MEDICATION_DECISION_RULE_SEMANTICS[rule].replacement_rule_id
                for rule in deprecated_rules
                if MEDICATION_DECISION_RULE_SEMANTICS[rule].replacement_rule_id is not None
            }
        )
        raise ValueError(
            "deprecated decision rule(s) are replay-only and cannot compile new tasks: "
            f"{deprecated_rules}; use {replacements} and an explicitly matching endpoint"
        )
    reference = ProgramExecutor(registry).execute(program)
    rendered_question = (
        question_renderer()
        if question_renderer is not None
        else (question or "")
    )
    allowed_tools = tuple(
        sorted(
            {call.tool for call in reference.reference_trace}
            | _evidence_contract_tool_names(evidence_contract)
            | {"submit_answer"}
        )
    )
    provenance = ArtifactProvenance(
        dataset=dataset,
        normalizer_version=normalizer_version or registry.snapshot.normalizer_version,
        tool_manifest_hash=_tool_manifest_hash(type(registry)),
        generator=generator,
        generator_version=generator_version,
        seed=seed,
    )
    evidence_quality_contract = (
        _derive_evidence_quality_contract(
            program=program,
            reference_trace=reference.reference_trace,
            gold_answer=reference.answer,
            snapshot_id=registry.snapshot_id,
            registry=registry,
        )
        if evidence_quality_policy == "exact_witness_v1"
        else None
    )
    task_schema_version = (
        "patient-tool-gym.task.v2"
        if program.schema_version
        in {
            "patient-tool-gym.program.v2",
            "patient-tool-gym.program.v3",
            "patient-tool-gym.program.v4",
        }
        else "patient-tool-gym.task.v1"
    )
    artifact = TaskArtifact(
        schema_version=task_schema_version,
        task_id=deterministic_task_id(registry.snapshot_id, program, family_id, construction_type),
        family_id=family_id,
        parent_task_ids=parent_task_ids,
        construction_type=construction_type,
        patient_snapshot_id=registry.snapshot_id,
        question=rendered_question,
        program=program,
        gold_answer=reference.answer,
        gold_evidence_ids=reference.evidence_ids,
        evidence_witnesses=(reference.evidence_ids,),
        evidence_contract=evidence_contract,
        evidence_quality_contract=evidence_quality_contract,
        question_contract=question_contract,
        reference_trace=reference.reference_trace,
        allowed_tools=allowed_tools,
        forbidden_tools=forbidden_tools,
        difficulty=reference.complexity,
        provenance=provenance,
        verification=VerificationReport(
            verifier_version=_verifier_version(
                program,
                task_schema_version,
                evidence_quality_contract=evidence_quality_contract,
            ),
            policy="standard",
            passed=False,
        ),
    )
    report = verify_task_artifact(artifact, registry)
    return artifact.model_copy(update={"verification": report})


def _verifier_version(
    program: TaskProgram,
    task_schema_version: str,
    *,
    evidence_quality_contract: EvidenceQualityContract | None,
) -> str:
    if evidence_quality_contract is not None:
        return "7" if program.schema_version == "patient-tool-gym.program.v4" else "6"
    if program.schema_version == "patient-tool-gym.program.v4":
        return "7"
    if program.schema_version == "patient-tool-gym.program.v3":
        return "5"
    if task_schema_version == "patient-tool-gym.task.v2":
        return "3"
    return "2"
