from __future__ import annotations

import hashlib
import json
import re
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote

from patient_tool_gym.compositional_matrix import (
    COMPOSITIONAL_MATRIX_SCHEMA_VERSION,
    TOPOLOGIES,
    CompositionalCandidate,
    CompositionalTopology,
    build_compositional_topology,
    discover_compositional_candidates,
    render_compositional_question,
    verify_topology_complexity,
)
from patient_tool_gym.fhir import load_fhir_bundle, normalize_fhir_bundle
from patient_tool_gym.medical_interventions import (
    CompositionalEvidenceState,
    RawFHIRPatchChain,
    build_compositional_factorial_patch_chain,
)
from patient_tool_gym.program import (
    ProgramExecutor,
    ProgramOutputSchema,
    TaskProgram,
    ToolNode,
    TruthState,
    TruthStateAggregateNode,
    TypedValue,
    analyze_complexity,
    content_hash,
    program_signature,
    truth_state_all,
)
from patient_tool_gym.raw_fhir import (
    SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES,
    RawFHIRDistractorUnavailableError,
    RawFHIRInsert,
    RawFHIRPatch,
    RawFHIRRemove,
    RawFHIRSet,
    ReferenceClosureReport,
    apply_raw_fhir_patch,
    build_distractor_patch,
    validate_reference_closure_strict,
)
from patient_tool_gym.task import TaskArtifact
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import compile_task, verify_task_artifact

COMPOSITIONAL_RELEASE_SCHEMA_VERSION = "patient-tool-gym.compositional-release.v2"
COMPOSITIONAL_RELEASE_BUILDER_VERSION = "patient-tool-gym.compositional-release-builder.v2"
COMPOSITIONAL_RELEASE_GENERATOR = "compositional_intervention_matrix_compiler"
COMPOSITIONAL_RELEASE_GENERATOR_VERSION = "2"
COMPOSITIONAL_PARTICIPANT_SCHEMA_VERSION = "patient-tool-gym.compositional-participant-package.v1"

_VERIFIED_RELEASE_CONTEXT_NONCE = object()


@dataclass(frozen=True)
class VerifiedCompositionalRelease:
    """Process-local proof that a closed release passed one full semantic replay."""

    root: Path
    manifest_sha256: str
    release_hash: str
    _report_json: str = dataclass_field(repr=False)
    _nonce: object = dataclass_field(repr=False, compare=False)

    @property
    def report(self) -> dict[str, Any]:
        value = json.loads(self._report_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise CompositionalReleaseError("verified release context report is malformed")
        return value

    def assert_current(self, release_directory: Path) -> dict[str, Any]:
        """Rehash the closed inventory before reusing the semantic replay result."""

        if self._nonce is not _VERIFIED_RELEASE_CONTEXT_NONCE:
            raise CompositionalReleaseError("verified release context was not issued by verifier")
        root = Path(release_directory).resolve()
        if root != self.root:
            raise CompositionalReleaseError("verified release context path mismatch")
        manifest_path = _safe_release_file(root, "manifest.json")
        if _sha256_file(manifest_path) != self.manifest_sha256:
            raise CompositionalReleaseError("verified release manifest changed")
        manifest = _read_release_json(root, "manifest.json")
        if manifest.get("schema_version") != COMPOSITIONAL_RELEASE_SCHEMA_VERSION:
            raise CompositionalReleaseError("verified release manifest schema changed")
        if (
            manifest.get("release_hash") != self.release_hash
            or content_hash(
                {key: value for key, value in manifest.items() if key != "release_hash"}
            )
            != self.release_hash
        ):
            raise CompositionalReleaseError("verified release hash changed")
        declared = _verify_closed_release_file_inventory(root, manifest)
        report = self.report
        counts = manifest.get("counts")
        required_true = (
            "passed",
            "closed_file_inventory",
            "patch_chains_replayed",
            "task_artifacts_replayed",
            "candidate_discovery_recomputed",
            "topology_builds_recomputed",
            "transaction_dependencies_verified",
            "split_and_validation_hashes_recomputed",
        )
        if (
            report.get("schema_version") != "patient-tool-gym.compositional-release-verification.v1"
            or report.get("release_hash") != self.release_hash
            or report.get("file_count") != len(declared) + 1
            or not isinstance(counts, Mapping)
            or report.get("patient_count") != counts.get("patient_count")
            or report.get("superfamily_count") != counts.get("superfamily_count")
            or report.get("cell_count") != counts.get("cell_count")
            or report.get("task_count") != counts.get("task_count")
            or any(report.get(key) is not True for key in required_true)
        ):
            raise CompositionalReleaseError("verified release context report is inconsistent")
        return report


EVIDENCE_STATES: tuple[CompositionalEvidenceState, ...] = (
    "supported",
    "refuted",
    "insufficient_evidence",
    "conflicting_evidence",
)
WIDTH_TOPOLOGIES: tuple[CompositionalTopology, ...] = ("d1w2", "d2w2")
WIDTH_ONE_TOPOLOGIES: tuple[CompositionalTopology, ...] = ("d1w1", "d2w1")
FACTORIAL_CELL_COUNT = len(EVIDENCE_STATES) * len(EVIDENCE_STATES) * 2
TASKS_PER_SUPERFAMILY = (
    len(WIDTH_ONE_TOPOLOGIES) * len(EVIDENCE_STATES) * 2
    + len(WIDTH_TOPOLOGIES) * FACTORIAL_CELL_COUNT
)
SplitName = Literal["train", "dev", "test"]

_COMPOSITIONAL_METADATA_SCHEMA_VERSION = "patient-tool-gym.compositional-task-metadata.v2"
_FHIR_PROVENANCE_ACTIVITY_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-DataOperation"
_FHIR_PROVENANCE_AGENT_SYSTEM = "http://terminology.hl7.org/CodeSystem/provenance-participant-type"
_CLINICAL_EVENT_IDENTIFIER_SYSTEM = "https://patient-tool-gym.dev/fhir/identifier/clinical-event"
_OBSERVATION_VALUE_FIELDS = frozenset(
    {
        "valueQuantity",
        "valueCodeableConcept",
        "valueString",
        "valueBoolean",
        "valueInteger",
        "valueDecimal",
        "valueRange",
        "valueRatio",
        "valueSampledData",
        "valueTime",
        "valueDateTime",
        "valuePeriod",
    }
)
_INTERVENTION_TIME_FIELDS = frozenset(
    {
        "timestamp",
        "lastUpdated",
        "recorded",
        "issued",
        "authoredOn",
        "occurrenceDateTime",
        "performedDateTime",
        "effectiveDateTime",
        "onsetDateTime",
        "abatementDateTime",
        "start",
        "end",
    }
)
_ROLE_TOKEN_PATTERN = re.compile(
    r"(?:^|[-_:./\\s])(?:conflict(?:ing)?|distractor|supported|refuted|"
    r"insufficient(?:[-_ ]evidence)?|answer[-_ ]preserving)(?:$|[-_:./\\s])",
    re.IGNORECASE,
)


class CompositionalReleaseError(ValueError):
    """Raised before an incomplete or unverifiable release can be published."""


def build_compositional_release(
    source_tasks: Sequence[TaskArtifact] | Path,
    source_bundles: Mapping[str, Mapping[str, Any]] | Path,
    output_directory: Path,
    *,
    seed: int = 20260711,
    materialize_derived_bundles: bool = False,
    synthea_generation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete topology-by-intervention benchmark release.

    Candidate discovery is exhaustive: there is deliberately no target count or quota.
    The output is staged in a sibling temporary directory and published by one rename.
    Derived Bundles are reconstructable from the content-bound base Bundle and patch
    chains, so they are omitted unless explicitly requested.
    """

    output_root = Path(output_directory)
    if output_root.exists():
        raise CompositionalReleaseError(
            "compositional release output already exists; use a fresh versioned path"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CompositionalReleaseError("seed must be an integer")
    generation_provenance = _build_generation_provenance(
        synthea_generation,
        seed=seed,
        materialize_derived_bundles=materialize_derived_bundles,
    )

    tasks, source_task_record, source_task_bytes = _load_source_tasks(source_tasks)
    matrix_discovery = discover_compositional_candidates(tasks, seed=seed)
    task_index = {task.task_id: task for task in tasks}
    bundle_catalog, source_bundle_record = _load_bundle_catalog(source_bundles)
    release_candidates, release_discovery = _release_candidate_discovery(
        matrix_discovery.model_dump(mode="json"),
        bundle_catalog,
        generation_provenance=generation_provenance,
    )
    if not release_candidates:
        raise CompositionalReleaseError("no eligible compositional source families were discovered")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}.",
        suffix=".tmp",
    ) as temporary_name:
        staging = Path(temporary_name)
        for relative in ("bundles/base", "dependencies", "patch-chains", "sources"):
            (staging / relative).mkdir(parents=True, exist_ok=True)
        if materialize_derived_bundles:
            (staging / "bundles/derived").mkdir(parents=True, exist_ok=True)

        source_task_path = staging / "sources/p2-tasks.jsonl"
        source_task_path.write_bytes(source_task_bytes)
        source_task_record = {
            **source_task_record,
            "release_path": "sources/p2-tasks.jsonl",
            "release_sha256": _sha256_file(source_task_path),
        }
        dependency_payloads = source_bundle_record.pop("_dependency_payloads", [])
        released_base_bundles: dict[str, Mapping[str, Any]] = {}
        for candidate in release_candidates:
            candidate_bundle = _bundle_for_candidate(candidate, bundle_catalog)["bundle"]
            released_base_bundles.setdefault(
                content_hash(candidate_bundle),
                candidate_bundle,
            )
        released_dependency_audit = _conditional_dependency_resolution_audit(
            [released_base_bundles[key] for key in sorted(released_base_bundles)],
            [row["bundle"] for row in dependency_payloads],
            scope="unique_released_patient_base_bundles",
        )
        if not released_dependency_audit["passed"]:
            raise CompositionalReleaseError(
                "released conditional references are not uniquely resolved by dependencies"
            )
        dependency_manifest = _materialize_dependency_bundles(
            staging,
            dependency_payloads,
            resolution_audit=released_dependency_audit,
        )
        _write_json(staging / "sources/dependency-manifest.json", dependency_manifest)
        source_bundle_record["dependencies"] = {
            "release_path": "sources/dependency-manifest.json",
            "release_sha256": _sha256_file(staging / "sources/dependency-manifest.json"),
            "dependency_bundle_count": dependency_manifest["dependency_bundle_count"],
            "resolution_audit_hash": dependency_manifest["resolution_audit"]["audit_hash"],
        }
        _write_json(
            staging / "candidate-discovery.json",
            release_discovery,
        )
        compiled_tasks: list[TaskArtifact] = []
        family_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        written_base_paths: set[str] = set()

        for candidate in release_candidates:
            catalog_entry = _bundle_for_candidate(candidate, bundle_catalog)
            bundle = catalog_entry["bundle"]
            base_hash = content_hash(bundle)
            base_closure = _strict_closure(bundle)
            if not base_closure.passed:
                raise CompositionalReleaseError(
                    f"source Bundle for {candidate.superfamily_id} is not strictly closed"
                )

            base_relative = _base_bundle_path(candidate.patient_id, base_hash)
            if base_relative not in written_base_paths:
                _write_json(staging / base_relative, bundle)
                written_base_paths.add(base_relative)

            base_snapshot = normalize_fhir_bundle(bundle, dataset="synthea")
            base_registry = ToolRegistry(base_snapshot)
            topology_builds = {
                topology: build_compositional_topology(
                    candidate,
                    task_index,
                    topology,
                    registry=base_registry,
                )
                for topology in TOPOLOGIES
            }
            topology_contracts = _topology_contracts(candidate, topology_builds)
            _verify_base_support(candidate, bundle, topology_builds)

            focal_id = _resource_id(candidate.focal_evidence_id, candidate.focal_resource_type)
            routing_identity = _resource_identity(candidate.routing_evidence_id, "Encounter")
            if len(candidate.companion_evidence_ids) != 1:
                raise CompositionalReleaseError(
                    "factorial companion interventions require exactly one source record"
                )
            companion_identities = tuple(
                _resource_identity(evidence_id) for evidence_id in candidate.companion_evidence_ids
            )
            companion_resource_type, companion_id = companion_identities[0]
            routing_base_hash = _raw_resource_hash(bundle, *routing_identity)
            companion_base_hashes = {
                evidence_id: _raw_resource_hash(bundle, *identity)
                for evidence_id, identity in zip(
                    candidate.companion_evidence_ids,
                    companion_identities,
                    strict=True,
                )
            }
            forbidden_codes = frozenset(
                {
                    (
                        candidate.focal_selector.system,
                        candidate.focal_selector.code,
                    ),
                    (
                        candidate.companion_selector.system,
                        candidate.companion_selector.code,
                    ),
                }
            )

            cell_rows: list[dict[str, Any]] = []
            canonical_width_one_ids: dict[tuple[str, bool], dict[str, str]] = {}
            provenance_recorded_at = _release_provenance_recorded_at(
                bundle,
                generation_provenance,
            )
            for evidence_state in EVIDENCE_STATES:
                for companion_evidence_state in EVIDENCE_STATES:
                    for distractor_injected in (False, True):
                        chain = build_compositional_factorial_patch_chain(
                            bundle,
                            primary_resource_type=candidate.focal_resource_type,
                            primary_resource_id=focal_id,
                            primary_state=evidence_state,
                            companion_resource_type=companion_resource_type,
                            companion_resource_id=companion_id,
                            companion_state=companion_evidence_state,
                            distractor_present=distractor_injected,
                            forbidden_codes=forbidden_codes,
                            target_date=(
                                candidate.focal_selector.date_from.isoformat()
                                if candidate.focal_selector.date_from is not None
                                else None
                            ),
                            provenance_recorded_at=provenance_recorded_at,
                        )
                        (
                            patch_contract,
                            derived,
                            validated_distractor_baselines,
                        ) = _validate_compositional_patch_chain_contract(
                            bundle,
                            chain,
                            expected_provenance_recorded_at=provenance_recorded_at,
                        )
                        distractor_evidence_id = _validate_distractor_contract(
                            candidate,
                            chain,
                            forbidden_codes=forbidden_codes,
                        )
                        original_cell = (
                            evidence_state == "supported"
                            and companion_evidence_state == "supported"
                            and not distractor_injected
                        )
                        if original_cell and (derived != bundle or chain.patches):
                            raise CompositionalReleaseError(
                                "supported/supported/no-injection cell is not the original Bundle"
                            )
                        if not original_cell and derived == bundle:
                            raise CompositionalReleaseError(
                                "a non-original factorial cell did not change the Bundle"
                            )
                        cell_id = _cell_id(
                            candidate,
                            chain,
                            evidence_state=evidence_state,
                            companion_evidence_state=companion_evidence_state,
                            distractor_injected=distractor_injected,
                        )
                        chain_relative = (
                            Path("patch-chains") / candidate.superfamily_id / f"{cell_id}.json"
                        ).as_posix()
                        _write_json(staging / chain_relative, chain.model_dump(mode="json"))

                        if materialize_derived_bundles:
                            derived_relative = (
                                Path("bundles/derived")
                                / candidate.superfamily_id
                                / f"{cell_id}.json"
                            ).as_posix()
                            _write_json(staging / derived_relative, derived)
                        else:
                            derived_relative = None

                        closure = _strict_closure(derived)
                        if not closure.passed:
                            raise CompositionalReleaseError(
                                f"derived Bundle {cell_id} is not strictly closed"
                            )
                        routing_derived_hash = _raw_resource_hash(derived, *routing_identity)
                        companion_derived_hashes = {
                            evidence_id: _raw_resource_hash(derived, *identity)
                            for evidence_id, identity in zip(
                                candidate.companion_evidence_ids,
                                companion_identities,
                                strict=True,
                            )
                        }
                        if routing_derived_hash != routing_base_hash:
                            raise CompositionalReleaseError(
                                f"chart cell {cell_id} changed routing evidence"
                            )
                        if (
                            companion_evidence_state == "supported"
                            and companion_derived_hashes != companion_base_hashes
                        ):
                            raise CompositionalReleaseError(
                                f"chart cell {cell_id} changed a supported companion"
                            )

                        snapshot = normalize_fhir_bundle(derived, dataset="synthea")
                        leakage_scan = validate_agent_visible_leakage_contract(
                            bundle,
                            chain,
                            normalized_snapshot=snapshot,
                            _validated_distractor_baselines=validated_distractor_baselines,
                        )
                        registry = ToolRegistry(snapshot)
                        task_ids: dict[str, str] = {}
                        canonical_key = (evidence_state, distractor_injected)
                        lineage_task_ids = (
                            {}
                            if companion_evidence_state == "supported"
                            else dict(canonical_width_one_ids.get(canonical_key, {}))
                        )
                        if companion_evidence_state != "supported" and set(lineage_task_ids) != set(
                            WIDTH_ONE_TOPOLOGIES
                        ):
                            raise CompositionalReleaseError(
                                "non-supported companion cell lacks canonical width-one parents"
                            )
                        cell_topologies = (
                            TOPOLOGIES
                            if companion_evidence_state == "supported"
                            else WIDTH_TOPOLOGIES
                        )
                        for topology in cell_topologies:
                            built = topology_builds[topology]
                            expected_branches = _expected_branch_states(
                                topology,
                                evidence_state,
                                companion_evidence_state,
                            )
                            expected_overall = truth_state_all(tuple(expected_branches.values()))
                            observed_branches = _execute_branch_states(registry, built.program)
                            if observed_branches != expected_branches:
                                raise CompositionalReleaseError(
                                    f"{cell_id}/{topology} branch states differ: "
                                    f"expected {expected_branches}, got {observed_branches}"
                                )

                            task = compile_task(
                                registry=registry,
                                program=built.program,
                                family_id=_topology_family_id(candidate, topology),
                                construction_type=built.construction_type,
                                question=None,
                                question_renderer=(
                                    lambda intent=built.question_intent: (
                                        render_compositional_question(intent)
                                    )
                                ),
                                parent_task_ids=_intra_cell_parent_ids(
                                    topology,
                                    lineage_task_ids,
                                ),
                                dataset="synthea",
                                generator=COMPOSITIONAL_RELEASE_GENERATOR,
                                generator_version=(COMPOSITIONAL_RELEASE_GENERATOR_VERSION),
                                seed=seed,
                                evidence_quality_policy="exact_witness_v1",
                            )
                            expected_answer = TypedValue(
                                kind="object",
                                value={
                                    "branches": expected_branches,
                                    "overall": expected_overall,
                                },
                            )
                            if task.gold_answer != expected_answer:
                                raise CompositionalReleaseError(
                                    f"{cell_id}/{topology} expected {expected_overall.value}, "
                                    f"got {task.gold_answer.value}"
                                )
                            _verify_distractor_retrieval(
                                task,
                                distractor_evidence_id=distractor_evidence_id,
                                cell_id=cell_id,
                                topology=topology,
                            )
                            exact_complexity = analyze_complexity(
                                built.program,
                                frozenset(task.gold_evidence_ids),
                                task.gold_answer.value,
                            )
                            if task.difficulty != exact_complexity:
                                raise CompositionalReleaseError(
                                    f"{cell_id}/{topology} stored complexity is not recomputed"
                                )
                            verify_topology_complexity(built.program, topology)
                            if not task.verification.passed or any(
                                not call.complete for call in task.reference_trace
                            ):
                                incomplete_nodes = [
                                    call.node_id
                                    for call in task.reference_trace
                                    if not call.complete
                                ]
                                raise CompositionalReleaseError(
                                    f"{cell_id}/{topology} did not compile with complete replay; "
                                    f"verification_errors={list(task.verification.errors)}; "
                                    f"incomplete_nodes={incomplete_nodes}"
                                )

                            release_metadata = {
                                "schema_version": _COMPOSITIONAL_METADATA_SCHEMA_VERSION,
                                "superfamily_id": candidate.superfamily_id,
                                "cell_id": cell_id,
                                "patient_id": candidate.patient_id,
                                "topology": topology,
                                "evidence_state": evidence_state,
                                "companion_evidence_state": companion_evidence_state,
                                "distractor_injected": distractor_injected,
                                "original_cell": original_cell,
                                "expected_branch_states": {
                                    name: state.value for name, state in expected_branches.items()
                                },
                                "expected_overall_state": expected_overall.value,
                                "base_bundle_hash": base_hash,
                                "patch_chain_hash": chain.chain_hash,
                                "derived_bundle_hash": chain.derived_bundle_hash,
                                "program_hash": topology_contracts[topology]["program_hash"],
                                "question_hash": topology_contracts[topology]["question_hash"],
                                "program_signature": topology_contracts[topology][
                                    "program_signature"
                                ],
                                "routing_raw_hash": routing_base_hash,
                                "routing_evidence_id": candidate.routing_evidence_id,
                                "companion_raw_hashes": companion_base_hashes,
                                "companion_derived_raw_hashes": companion_derived_hashes,
                                "companion_evidence_ids": list(candidate.companion_evidence_ids),
                                "distractor_evidence_id": distractor_evidence_id,
                                "source_task_hashes": candidate.source_task_hashes,
                                "source_task_ids": {
                                    "focal_atomic": candidate.focal_atomic_task_id,
                                    "focal_depth": candidate.focal_depth_task_id,
                                    "companion": candidate.companion_task_id,
                                },
                            }
                            release_metadata["metadata_hash"] = _compositional_metadata_hash(
                                release_metadata
                            )
                            metadata = {"compositional_release": release_metadata}
                            task = task.model_copy(update={"metadata": metadata})
                            replay_report = verify_task_artifact(task, registry)
                            if not replay_report.passed:
                                raise CompositionalReleaseError(
                                    f"{cell_id}/{topology} failed final artifact replay: "
                                    f"{replay_report.errors}"
                                )
                            task = task.model_copy(update={"verification": replay_report})
                            task_ids[topology] = task.task_id
                            lineage_task_ids[topology] = task.task_id
                            compiled_tasks.append(task)
                            validation_rows.append(
                                {
                                    "schema_version": (
                                        "patient-tool-gym.compositional-validation-row.v2"
                                    ),
                                    "superfamily_id": candidate.superfamily_id,
                                    "cell_id": cell_id,
                                    "task_id": task.task_id,
                                    "topology": topology,
                                    "evidence_state": evidence_state,
                                    "companion_evidence_state": companion_evidence_state,
                                    "distractor_injected": distractor_injected,
                                    "original_cell": original_cell,
                                    "expected_branch_states": {
                                        name: state.value
                                        for name, state in expected_branches.items()
                                    },
                                    "observed_branch_states": {
                                        name: state.value
                                        for name, state in observed_branches.items()
                                    },
                                    "expected_overall_state": expected_overall.value,
                                    "observed_overall_state": task.gold_answer.value["overall"],
                                    "complexity": task.difficulty.model_dump(mode="json"),
                                    "strict_reference_closure": closure.model_dump(mode="json"),
                                    "routing_raw_hash_preserved": True,
                                    "companion_change_matches_declared_state": True,
                                    "patch_chain_replayed": True,
                                    "changed_and_inserted_resource_allowlist": patch_contract[
                                        "changed_and_inserted_resource_allowlist"
                                    ],
                                    "provenance_targets_exact": patch_contract[
                                        "provenance_targets_exact"
                                    ],
                                    "provenance_activity_exact": patch_contract[
                                        "provenance_activity_exact"
                                    ],
                                    "provenance_recorded_exact": patch_contract[
                                        "provenance_recorded_exact"
                                    ],
                                    "provenance_agent_exact": patch_contract[
                                        "provenance_agent_exact"
                                    ],
                                    "task_replayed": True,
                                    "all_tool_calls_complete": True,
                                    "intra_cell_parent_lineage": True,
                                    "agent_visible_role_tokens_absent": leakage_scan[
                                        "generated_identifier_role_tokens_absent"
                                    ]
                                    and leakage_scan["generated_narrative_role_tokens_absent"],
                                    "semantic_duplicate_distractors_absent": leakage_scan[
                                        "semantic_duplicate_distractors_absent"
                                    ],
                                    "distractor_retrieved_but_not_gold": (
                                        None if distractor_evidence_id is None else True
                                    ),
                                    "passed": True,
                                }
                            )

                        if companion_evidence_state == "supported":
                            canonical_width_one_ids[canonical_key] = {
                                topology: task_ids[topology] for topology in WIDTH_ONE_TOPOLOGIES
                            }

                        cell_rows.append(
                            {
                                "cell_id": cell_id,
                                "evidence_state": evidence_state,
                                "companion_evidence_state": companion_evidence_state,
                                "distractor_injected": distractor_injected,
                                "original_cell": original_cell,
                                "patch_chain_path": chain_relative,
                                "patch_chain_hash": chain.chain_hash,
                                "patch_chain_file_sha256": _sha256_file(staging / chain_relative),
                                "provenance_recorded_at": provenance_recorded_at,
                                "derived_bundle_hash": chain.derived_bundle_hash,
                                "derived_bundle_path": derived_relative,
                                "snapshot_id": snapshot.snapshot_id,
                                "routing_raw_hash": routing_derived_hash,
                                "companion_raw_hashes": companion_derived_hashes,
                                "companion_base_raw_hashes": companion_base_hashes,
                                "distractor_evidence_id": distractor_evidence_id,
                                "raw_patch_contract": patch_contract,
                                "agent_visible_leakage_scan": leakage_scan,
                                "task_ids": task_ids,
                            }
                        )

            family_row = {
                "schema_version": "patient-tool-gym.compositional-family.v2",
                "superfamily_id": candidate.superfamily_id,
                "patient_id": candidate.patient_id,
                "source_candidate": candidate.model_dump(mode="json"),
                "source_bundle": {
                    "catalog_name": catalog_entry["name"],
                    "source_file_sha256": catalog_entry["file_sha256"],
                    "base_bundle_path": base_relative,
                    "base_bundle_hash": base_hash,
                    "base_bundle_file_sha256": _sha256_file(staging / base_relative),
                    "base_snapshot_id": catalog_entry["snapshot_id"],
                    "strict_reference_closure": base_closure.model_dump(mode="json"),
                },
                "routing_raw_hash": routing_base_hash,
                "companion_raw_hashes": companion_base_hashes,
                "topologies": topology_contracts,
                "cells": cell_rows,
            }
            family_row["family_hash"] = content_hash(family_row)
            family_rows.append(family_row)

        compiled_tasks.sort(key=lambda task: task.task_id)
        family_rows.sort(key=lambda row: row["superfamily_id"])
        validation_rows.sort(key=lambda row: row["task_id"])
        _write_jsonl(
            staging / "tasks.jsonl",
            [task.model_dump(mode="json") for task in compiled_tasks],
        )
        _write_jsonl(staging / "families.jsonl", family_rows)
        source_bundle_catalog = _source_bundle_catalog(
            source_bundle_record,
            family_rows,
        )
        _write_json(staging / "sources/bundle-catalog.json", source_bundle_catalog)
        source_bundle_record = {
            "release_path": "sources/bundle-catalog.json",
            "release_sha256": _sha256_file(staging / "sources/bundle-catalog.json"),
            "bound_bundle_count": source_bundle_catalog["bound_bundle_count"],
            "input": source_bundle_record,
        }
        validation = _validation_report(
            validation_rows,
            candidate_count=len(release_candidates),
        )
        _write_json(staging / "validation-report.json", validation)
        splits = _build_leakage_safe_splits(compiled_tasks, seed=seed)
        _write_json(staging / "splits.json", splits)

        manifest = _manifest(
            staging=staging,
            source_tasks=source_task_record,
            source_bundles=source_bundle_record,
            discovery=release_discovery,
            tasks=compiled_tasks,
            families=family_rows,
            validation=validation,
            splits=splits,
            seed=seed,
            materialize_derived_bundles=materialize_derived_bundles,
            generation_provenance=generation_provenance,
        )
        _write_json(staging / "manifest.json", manifest)
        staging.replace(output_root)
    return manifest


def verify_compositional_release(release_directory: Path) -> dict[str, Any]:
    """Reconstruct and independently verify a hosted compositional release directory."""

    root = Path(release_directory)
    if not root.is_dir():
        raise CompositionalReleaseError("compositional release directory does not exist")
    manifest = _read_release_json(root, "manifest.json")
    if manifest.get("schema_version") != COMPOSITIONAL_RELEASE_SCHEMA_VERSION:
        raise CompositionalReleaseError("compositional release manifest schema is unsupported")
    expected_release_hash = content_hash(
        {key: value for key, value in manifest.items() if key != "release_hash"}
    )
    if manifest.get("release_hash") != expected_release_hash:
        raise CompositionalReleaseError("compositional release manifest hash mismatch")
    declared_files = _verify_closed_release_file_inventory(root, manifest)

    required_files = {
        "candidate-discovery.json",
        "families.jsonl",
        "sources/bundle-catalog.json",
        "sources/dependency-manifest.json",
        "sources/p2-tasks.jsonl",
        "splits.json",
        "tasks.jsonl",
        "validation-report.json",
    }
    if not required_files.issubset(declared_files):
        raise CompositionalReleaseError("compositional release omits a required artifact")

    task_rows = _read_release_jsonl(root, "tasks.jsonl")
    family_rows = _read_release_jsonl(root, "families.jsonl")
    tasks = [TaskArtifact.model_validate(row) for row in task_rows]
    task_index = {task.task_id: task for task in tasks}
    if len(task_index) != len(tasks):
        raise CompositionalReleaseError("compositional release contains duplicate task IDs")
    family_index = {str(row.get("superfamily_id")): row for row in family_rows}
    if len(family_index) != len(family_rows) or "None" in family_index:
        raise CompositionalReleaseError(
            "compositional release contains duplicate or missing superfamily IDs"
        )

    bound_task_counts: Counter[str] = Counter()
    for family in family_rows:
        bound_task_counts.update(
            _verify_standalone_family(
                root,
                family,
                task_index=task_index,
            )
        )
    if set(bound_task_counts) != set(task_index) or any(
        count != 1 for count in bound_task_counts.values()
    ):
        raise CompositionalReleaseError("compositional release has unbound or multiply-bound tasks")

    validation = _read_release_json(root, "validation-report.json")
    _verify_standalone_validation(validation, tasks=tasks, family_count=len(family_rows))
    splits = _read_release_json(root, "splits.json")
    _verify_standalone_splits(splits, tasks=tasks)
    discovery = _read_release_json(root, "candidate-discovery.json")
    _verify_standalone_discovery(discovery, family_rows=family_rows)
    source_tasks = _verify_standalone_source_task_replay(
        root,
        manifest=manifest,
        discovery=discovery,
        family_rows=family_rows,
    )
    bundle_catalog = _read_release_json(root, "sources/bundle-catalog.json")
    _verify_standalone_source_catalog(bundle_catalog, family_rows=family_rows)
    dependency_manifest = _read_release_json(root, "sources/dependency-manifest.json")
    released_base_paths = sorted(
        {str(family["source_bundle"]["base_bundle_path"]) for family in family_rows}
    )
    _verify_standalone_dependencies(
        root,
        dependency_manifest,
        patient_bundles=[_read_release_json(root, relative) for relative in released_base_paths],
    )
    _verify_standalone_manifest_counts(
        manifest,
        tasks=tasks,
        family_rows=family_rows,
        validation=validation,
        splits=splits,
        declared_files=declared_files,
    )
    _verify_generation_provenance(manifest)

    return {
        "schema_version": "patient-tool-gym.compositional-release-verification.v1",
        "passed": True,
        "release_hash": manifest["release_hash"],
        "file_count": len(declared_files) + 1,
        "patient_count": len(
            {task.metadata["compositional_release"]["patient_id"] for task in tasks}
        ),
        "superfamily_count": len(family_rows),
        "cell_count": sum(len(row["cells"]) for row in family_rows),
        "task_count": len(tasks),
        "closed_file_inventory": True,
        "patch_chains_replayed": True,
        "task_artifacts_replayed": True,
        "source_tasks_reloaded": len(source_tasks),
        "candidate_discovery_recomputed": True,
        "topology_builds_recomputed": True,
        "transaction_dependencies_verified": True,
        "split_and_validation_hashes_recomputed": True,
    }


def verify_compositional_release_context(
    release_directory: Path,
) -> VerifiedCompositionalRelease:
    """Run one full replay and issue a process-local, inventory-rechecked context."""

    root = Path(release_directory).resolve()
    report = verify_compositional_release(root)
    manifest_path = _safe_release_file(root, "manifest.json")
    manifest = _read_release_json(root, "manifest.json")
    declared = _verify_closed_release_file_inventory(root, manifest)
    if report.get("file_count") != len(declared) + 1:
        raise CompositionalReleaseError("release inventory changed while issuing verified context")
    return VerifiedCompositionalRelease(
        root=root,
        manifest_sha256=_sha256_file(manifest_path),
        release_hash=str(report["release_hash"]),
        _report_json=json.dumps(report, sort_keys=True, separators=(",", ":")),
        _nonce=_VERIFIED_RELEASE_CONTEXT_NONCE,
    )


def load_compositional_release_verification_context(
    release_directory: Path,
    verification_package_directory: Path,
) -> VerifiedCompositionalRelease:
    """Issue a context from a sealed prior verification after rehashing the release.

    This avoids repeating the expensive semantic replay while still requiring the current
    release to match every byte in its closed manifest and the verification package to match
    its own content hash.
    """

    root = Path(release_directory).resolve()
    package_root = Path(verification_package_directory).resolve()
    package_manifest = _read_release_json(package_root, "manifest.json")
    if package_manifest.get(
        "schema_version"
    ) != "patient-tool-gym.confirmatory-release-verification-package.v1" or package_manifest.get(
        "package_hash"
    ) != content_hash(
        {key: value for key, value in package_manifest.items() if key != "package_hash"}
    ):
        raise CompositionalReleaseError("verification package manifest is invalid")

    verification_binding = package_manifest.get("verification")
    if not isinstance(verification_binding, Mapping):
        raise CompositionalReleaseError("verification package binding is malformed")
    verification_relative = verification_binding.get("path")
    if (
        not isinstance(verification_relative, str)
        or Path(verification_relative).is_absolute()
        or ".." in Path(verification_relative).parts
    ):
        raise CompositionalReleaseError("verification package path is unsafe")
    verification_path = package_root / verification_relative
    if not verification_path.is_file() or _sha256_file(
        verification_path
    ) != verification_binding.get("sha256"):
        raise CompositionalReleaseError("verification package file hash changed")
    wrapper = _read_release_json(package_root, verification_relative)
    core = wrapper.get("core_verification")
    if (
        not isinstance(core, Mapping)
        or wrapper.get("passed") is not True
        or wrapper.get("verification_hash") != verification_binding.get("verification_hash")
        or wrapper.get("verification_hash")
        != content_hash(
            {key: value for key, value in wrapper.items() if key != "verification_hash"}
        )
    ):
        raise CompositionalReleaseError("verification package report is invalid")

    manifest_path = _safe_release_file(root, "manifest.json")
    manifest = _read_release_json(root, "manifest.json")
    source_binding = package_manifest.get("source_release")
    if (
        not isinstance(source_binding, Mapping)
        or _sha256_file(manifest_path) != source_binding.get("manifest_sha256")
        or manifest.get("release_hash") != source_binding.get("release_hash")
        or core.get("release_hash") != source_binding.get("release_hash")
    ):
        raise CompositionalReleaseError("verification package targets a different release")
    context = VerifiedCompositionalRelease(
        root=root,
        manifest_sha256=_sha256_file(manifest_path),
        release_hash=str(core["release_hash"]),
        _report_json=json.dumps(dict(core), sort_keys=True, separators=(",", ":")),
        _nonce=_VERIFIED_RELEASE_CONTEXT_NONCE,
    )
    context.assert_current(root)
    return context


def build_compositional_participant_package(
    evaluator_release_directory: Path,
    output_directory: Path,
    *,
    evaluator_release_context: VerifiedCompositionalRelease | None = None,
) -> dict[str, Any]:
    """Publish prompts without evaluator state, source charts, or held-out test IDs.

    The evaluator release is intentionally a different directory with a different access
    classification.  The participant package contains only train/dev prompt contracts.
    Test prompts, patient/snapshot identifiers, executable gold programs, reference traces,
    expected states, source tasks, Bundles, and patch chains remain evaluator-only.
    """

    evaluator_root = Path(evaluator_release_directory).resolve()
    output_root = Path(output_directory)
    if output_root.exists():
        raise CompositionalReleaseError(
            "participant package output already exists; use a fresh versioned path"
        )
    context = evaluator_release_context or verify_compositional_release_context(evaluator_root)
    if context.root != evaluator_root:
        raise CompositionalReleaseError("verified release context path mismatch")
    verification = context.report
    release_manifest = _read_release_json(evaluator_root, "manifest.json")
    tasks = [
        TaskArtifact.model_validate(row)
        for row in _read_release_jsonl(evaluator_root, "tasks.jsonl")
    ]
    split_payload = _read_release_json(evaluator_root, "splits.json")
    assignment_rows = split_payload.get("assignments", {}).get("task_assignments")
    if not isinstance(assignment_rows, list):
        raise CompositionalReleaseError("evaluator split assignments are malformed")
    assignments = {
        str(row["task_id"]): str(row["split"])
        for row in assignment_rows
        if isinstance(row, Mapping)
    }
    if set(assignments) != {task.task_id for task in tasks}:
        raise CompositionalReleaseError("evaluator split assignments do not cover all tasks")

    public_rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        split = assignments[task.task_id]
        if split == "test":
            continue
        if split not in {"train", "dev"}:
            raise CompositionalReleaseError("participant package encountered an invalid split")
        tools = sorted({node.tool for node in task.program.nodes if isinstance(node, ToolNode)})
        public_rows.append(
            {
                "schema_version": "patient-tool-gym.compositional-participant-task.v1",
                "task_id": task.task_id,
                "split": split,
                "question": task.question,
                "allowed_tools": tools,
                "response_schema": task.program.output_schema.model_dump(mode="json"),
            }
        )

    dependency_count = int(
        release_manifest.get("sources", {})
        .get("bundles", {})
        .get("input", {})
        .get("dependencies", {})
        .get("dependency_bundle_count", 0)
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}.",
        suffix=".tmp",
    ) as temporary_name:
        staging = Path(temporary_name) / "participant"
        staging.mkdir()
        _write_jsonl(staging / "tasks.jsonl", public_rows)
        file_row = {
            "path": "tasks.jsonl",
            "bytes": (staging / "tasks.jsonl").stat().st_size,
            "sha256": _sha256_file(staging / "tasks.jsonl"),
        }
        split_counts = Counter(assignments.values())
        payload: dict[str, Any] = {
            "schema_version": COMPOSITIONAL_PARTICIPANT_SCHEMA_VERSION,
            "access_classification": "public_participant",
            "source_evaluator_release": {
                "schema_version": release_manifest["schema_version"],
                "release_hash": verification["release_hash"],
            },
            "contents": {
                "included_splits": ["train", "dev"],
                "excluded_split": "test",
                "task_count": len(public_rows),
                "counts_by_split": dict(
                    sorted(Counter(row["split"] for row in public_rows).items())
                ),
                "held_out_test_task_count": split_counts.get("test", 0),
            },
            "privacy_contract": {
                "source_tasks_included": False,
                "charts_or_snapshots_included": False,
                "test_task_ids_included": False,
                "gold_answers_included": False,
                "gold_evidence_included": False,
                "reference_traces_included": False,
                "expected_state_metadata_included": False,
                "runtime_chart_access": "evaluator_managed",
            },
            "runtime_contract": {
                "dependency_preload_required": dependency_count > 0,
                "dependency_bundle_count": dependency_count,
                "dependency_files_distributed_to_participants": False,
                "dependency_resolution": "evaluator_managed_before_patient_chart_load",
            },
            "files": [file_row],
        }
        payload["package_hash"] = content_hash(payload)
        _write_json(staging / "manifest.json", payload)
        verify_compositional_participant_package(
            staging,
            evaluator_release_directory=evaluator_root,
            evaluator_release_context=context,
        )
        staging.replace(output_root)
    return payload


def verify_compositional_participant_package(
    participant_directory: Path,
    *,
    evaluator_release_directory: Path | None = None,
    evaluator_release_context: VerifiedCompositionalRelease | None = None,
) -> dict[str, Any]:
    """Reject any participant package containing evaluator-only or held-out fields."""

    root = Path(participant_directory).resolve()
    manifest = _read_release_json(root, "manifest.json")
    if manifest.get("schema_version") != COMPOSITIONAL_PARTICIPANT_SCHEMA_VERSION:
        raise CompositionalReleaseError("participant manifest schema is unsupported")
    expected_hash = content_hash(
        {key: value for key, value in manifest.items() if key != "package_hash"}
    )
    if manifest.get("package_hash") != expected_hash:
        raise CompositionalReleaseError("participant package hash mismatch")
    rows = _read_release_jsonl(root, "tasks.jsonl")
    allowed_fields = {
        "schema_version",
        "task_id",
        "split",
        "question",
        "allowed_tools",
        "response_schema",
    }
    forbidden_field_fragments = (
        "gold",
        "evidence",
        "reference_trace",
        "expected",
        "patient",
        "snapshot",
        "source_task",
        "patch",
        "bundle",
        "metadata",
        "program",
    )
    task_ids: set[str] = set()
    for row in rows:
        if set(row) != allowed_fields or any(
            fragment in key.lower() for key in row for fragment in forbidden_field_fragments
        ):
            raise CompositionalReleaseError("participant task contains an evaluator-only field")
        if (
            row.get("schema_version") != "patient-tool-gym.compositional-participant-task.v1"
            or row.get("split") not in {"train", "dev"}
            or not isinstance(row.get("task_id"), str)
            or not isinstance(row.get("question"), str)
            or not isinstance(row.get("allowed_tools"), list)
            or not isinstance(row.get("response_schema"), Mapping)
        ):
            raise CompositionalReleaseError("participant task row is malformed")
        if row["task_id"] in task_ids:
            raise CompositionalReleaseError("participant package duplicates a task ID")
        task_ids.add(row["task_id"])
    files = manifest.get("files")
    if files != [
        {
            "path": "tasks.jsonl",
            "bytes": (root / "tasks.jsonl").stat().st_size,
            "sha256": _sha256_file(root / "tasks.jsonl"),
        }
    ]:
        raise CompositionalReleaseError("participant closed file inventory mismatch")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != {"manifest.json", "tasks.jsonl"}:
        raise CompositionalReleaseError("participant package contains undeclared files")

    if evaluator_release_directory is None and evaluator_release_context is not None:
        raise CompositionalReleaseError(
            "an evaluator release context requires an evaluator release directory"
        )
    if evaluator_release_directory is not None:
        evaluator_root = Path(evaluator_release_directory).resolve()
        verification = (
            verify_compositional_release(evaluator_root)
            if evaluator_release_context is None
            else evaluator_release_context.assert_current(evaluator_root)
        )
        source_binding = manifest.get("source_evaluator_release")
        if not isinstance(source_binding, Mapping) or source_binding.get(
            "release_hash"
        ) != verification.get("release_hash"):
            raise CompositionalReleaseError(
                "participant package is bound to a different evaluator release"
            )
        evaluator_tasks = {
            task.task_id: task
            for task in (
                TaskArtifact.model_validate(row)
                for row in _read_release_jsonl(evaluator_root, "tasks.jsonl")
            )
        }
        splits = _read_release_json(evaluator_root, "splits.json")
        assignments = {
            str(row["task_id"]): str(row["split"])
            for row in splits["assignments"]["task_assignments"]
        }
        expected_public_ids = {
            task_id for task_id, split in assignments.items() if split in {"train", "dev"}
        }
        test_ids = {task_id for task_id, split in assignments.items() if split == "test"}
        if task_ids != expected_public_ids or task_ids & test_ids:
            raise CompositionalReleaseError(
                "participant package leaks or omits split-bound task IDs"
            )
        for row in rows:
            task = evaluator_tasks[row["task_id"]]
            expected_tools = sorted(
                {node.tool for node in task.program.nodes if isinstance(node, ToolNode)}
            )
            if (
                row["question"] != task.question
                or row["split"] != assignments[task.task_id]
                or row["allowed_tools"] != expected_tools
                or row["response_schema"] != task.program.output_schema.model_dump(mode="json")
            ):
                raise CompositionalReleaseError(
                    "participant prompt contract differs from the evaluator task"
                )
    return {
        "schema_version": "patient-tool-gym.compositional-participant-verification.v1",
        "passed": True,
        "task_count": len(rows),
        "test_task_ids_excluded": True,
        "evaluator_only_fields_excluded": True,
        "closed_file_inventory": True,
    }


def validate_compositional_patch_chain_contract(
    base_bundle: Mapping[str, Any],
    chain: RawFHIRPatchChain,
    *,
    expected_provenance_recorded_at: str | None = None,
) -> dict[str, Any]:
    """Verify declared targets, operation allowlists, neutral Provenance, and no-op cells."""

    report, _, _ = _validate_compositional_patch_chain_contract(
        base_bundle,
        chain,
        expected_provenance_recorded_at=expected_provenance_recorded_at,
    )
    return report


def _validate_compositional_patch_chain_contract(
    base_bundle: Mapping[str, Any],
    chain: RawFHIRPatchChain,
    *,
    expected_provenance_recorded_at: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[tuple[tuple[str, str, str], ...], ...],
]:
    """Return both the validation report and the already replayed derived Bundle."""

    current = dict(base_bundle)
    if chain.patches and chain.patches[0].base_bundle_hash != chain.base_bundle_hash:
        raise CompositionalReleaseError("first raw patch does not bind the chain base Bundle")
    no_op_bundle_hash = content_hash(current) if not chain.patches else None
    if no_op_bundle_hash is not None and no_op_bundle_hash != chain.base_bundle_hash:
        raise CompositionalReleaseError("raw patch contract base Bundle hash mismatch")
    expected_chain_hash = content_hash(chain.model_dump(mode="json", exclude={"chain_hash"}))
    if chain.chain_hash != expected_chain_hash:
        raise CompositionalReleaseError("raw patch contract chain hash mismatch")
    declarations = chain.declared_targets or (
        {
            "role": "primary",
            "resource_type": chain.target_resource_type,
            "resource_id": chain.target_resource_id,
            "evidence_state": chain.evidence_state,
        },
    )
    declaration_index: dict[tuple[str, str], tuple[str, str]] = {}
    for declaration in declarations:
        role = declaration.role if hasattr(declaration, "role") else str(declaration["role"])
        resource_type = (
            declaration.resource_type
            if hasattr(declaration, "resource_type")
            else str(declaration["resource_type"])
        )
        resource_id = (
            declaration.resource_id
            if hasattr(declaration, "resource_id")
            else str(declaration["resource_id"])
        )
        state = (
            declaration.evidence_state
            if hasattr(declaration, "evidence_state")
            else str(declaration["evidence_state"])
        )
        if (resource_type, resource_id) not in _raw_resource_index(current):
            raise CompositionalReleaseError("raw patch declared target resource is missing")
        if (resource_type, resource_id) in declaration_index:
            raise CompositionalReleaseError("raw patch declares one target more than once")
        declaration_index[(resource_type, resource_id)] = (role, state)

    observed_state_targets: set[tuple[str, str]] = set()

    patch_rows: list[dict[str, Any]] = []
    validated_distractor_baselines: list[tuple[tuple[str, str, str], ...]] = []
    for index, patch in enumerate(chain.patches):
        target_identity = (patch.target_resource_type, patch.target_resource_id)
        distractor_patch = patch.intervention_kind == "answer_preserving_distractor"
        if distractor_patch:
            if target_identity != (chain.target_resource_type, chain.target_resource_id):
                raise CompositionalReleaseError("distractor patch changed the primary target")
        else:
            declaration = declaration_index.get(target_identity)
            if declaration is None or patch.intervention_kind != declaration[1]:
                raise CompositionalReleaseError(
                    "state patch does not match an ordered target declaration"
                )
            if target_identity in observed_state_targets:
                raise CompositionalReleaseError("declared target has more than one state patch")
            observed_state_targets.add(target_identity)
        if tuple(sorted(patch.allowed_external_conditional_reference_types)) != tuple(
            sorted(_present_allowed_conditional_reference_types(current))
        ):
            raise CompositionalReleaseError(
                "raw patch conditional-reference exception list is not exact"
            )

        non_insert_operations = tuple(
            operation
            for operation in patch.operations
            if isinstance(operation, (RawFHIRSet, RawFHIRRemove))
        )
        insert_operations = tuple(
            operation for operation in patch.operations if isinstance(operation, RawFHIRInsert)
        )
        provenance_operations = tuple(
            operation
            for operation in insert_operations
            if operation.resource.get("resourceType") == "Provenance"
        )
        clinical_insert_operations = tuple(
            operation
            for operation in insert_operations
            if operation.resource.get("resourceType") not in {"Provenance", "Device"}
        )
        device_operations = tuple(
            operation
            for operation in insert_operations
            if operation.resource.get("resourceType") == "Device"
        )
        if len(device_operations) > 1 or not provenance_operations:
            raise CompositionalReleaseError("changed patches require exact compiler Provenance")

        target_resource = _raw_resource_index(current)[target_identity]
        expected_kind = patch.intervention_kind
        if not distractor_patch:
            if patch.distractor_tier is not None:
                raise CompositionalReleaseError("a state patch cannot declare a distractor tier")
            _validate_state_operation_allowlist(
                patch,
                non_insert_operations,
                evidence_state=expected_kind,
            )
            if expected_kind == "conflicting_evidence":
                if len(clinical_insert_operations) != 1:
                    raise CompositionalReleaseError(
                        "a conflicting-evidence patch must insert exactly one conflict record"
                    )
                conflict = clinical_insert_operations[0]
                _validate_clinical_insert_identity(
                    conflict,
                    expected_resource_type=patch.target_resource_type,
                )
                _validate_conflict_clone(target_resource, conflict.resource)
                identifier_operations = [
                    operation
                    for operation in non_insert_operations
                    if tuple(operation.path) == ("identifier",)
                    and isinstance(operation, RawFHIRSet)
                ]
                if len(identifier_operations) != 1:
                    raise CompositionalReleaseError(
                        "conflicting evidence must add one shared event identifier"
                    )
                identifiers = identifier_operations[0].value
                shared = (
                    [
                        identifier
                        for identifier in identifiers
                        if isinstance(identifier, Mapping)
                        and identifier.get("system") == _CLINICAL_EVENT_IDENTIFIER_SYSTEM
                        and isinstance(identifier.get("value"), str)
                    ]
                    if isinstance(identifiers, list)
                    else []
                )
                if len(shared) != 1 or shared[0] not in conflict.resource.get("identifier", []):
                    raise CompositionalReleaseError(
                        "opposed records do not share one clinical-event identifier"
                    )
            else:
                if clinical_insert_operations:
                    raise CompositionalReleaseError(
                        "a non-conflict state patch cannot insert a clinical record"
                    )
        else:
            if non_insert_operations:
                raise CompositionalReleaseError(
                    "a distractor patch cannot modify an existing resource"
                )
            if patch.distractor_tier not in {
                "same_resource_different_code",
                "same_resource_near_match",
            }:
                raise CompositionalReleaseError(
                    "a compositional distractor must use an admitted same-resource tier"
                )
            if len(clinical_insert_operations) != 1:
                raise CompositionalReleaseError(
                    "a distractor patch must insert exactly one clinical record"
                )
            distractor = clinical_insert_operations[0]
            _validate_clinical_insert_identity(
                distractor,
                expected_resource_type=patch.target_resource_type,
            )
            _validate_distractor_clone(target_resource, distractor.resource)
            distractor_type, _ = _raw_identity(distractor.resource)
            validated_distractor_baselines.append(
                tuple(
                    (
                        existing_type,
                        existing_id,
                        _semantic_fhir_fingerprint(existing),
                    )
                    for (existing_type, existing_id), existing in _raw_resource_index(
                        current
                    ).items()
                    if existing_type == distractor_type
                )
            )
        if not distractor_patch:
            validated_distractor_baselines.append(())
        provenance_report = _validate_neutral_patch_provenance(
            current,
            non_insert_operations=non_insert_operations,
            clinical_insert_operations=clinical_insert_operations,
            provenance_operations=provenance_operations,
            device_operations=device_operations,
            expected_recorded_at=(
                expected_provenance_recorded_at
                or (
                    str(base_bundle["timestamp"])
                    if isinstance(base_bundle.get("timestamp"), str)
                    else None
                )
            ),
        )
        derived = apply_raw_fhir_patch(current, patch)
        _validate_exact_bundle_delta(
            current,
            derived,
            non_insert_operations=non_insert_operations,
            insert_operations=insert_operations,
        )
        patch_rows.append(
            {
                "patch_index": index,
                "intervention_kind": expected_kind,
                "modified_existing_resource_count": len(
                    {
                        (operation.resource_type, operation.resource_id)
                        for operation in non_insert_operations
                    }
                ),
                "inserted_clinical_resource_count": len(clinical_insert_operations),
                "inserted_provenance_resource_count": len(provenance_operations),
                "provenance_target_count": provenance_report["target_count"],
                "passed": True,
            }
        )
        current = derived

    for identity, (_, state) in declaration_index.items():
        if state != "supported" and identity not in observed_state_targets:
            raise CompositionalReleaseError("non-supported declaration omitted its state patch")
    if len(
        [
            patch
            for patch in chain.patches
            if patch.intervention_kind == "answer_preserving_distractor"
        ]
    ) != int(chain.distractor_present):
        raise CompositionalReleaseError("distractor declaration and patch count differ")

    derived_bundle_hash = no_op_bundle_hash or content_hash(current)
    if derived_bundle_hash != chain.derived_bundle_hash:
        raise CompositionalReleaseError("raw patch contract derived Bundle hash mismatch")
    if not chain.patches and not _strict_closure(current).passed:
        raise CompositionalReleaseError("raw patch contract result is not reference-closed")
    return (
        {
            "schema_version": "patient-tool-gym.compositional-raw-patch-validation.v2",
            "passed": True,
            "patch_count": len(patch_rows),
            "changed_and_inserted_resource_allowlist": True,
            "provenance_targets_exact": True,
            "provenance_activity_exact": True,
            "provenance_recorded_exact": True,
            "provenance_agent_exact": True,
            "original_noop_cell": not chain.patches,
            "patches": patch_rows,
        },
        current,
        tuple(validated_distractor_baselines),
    )


def validate_agent_visible_leakage_contract(
    base_bundle: Mapping[str, Any],
    chain: RawFHIRPatchChain,
    *,
    normalized_snapshot: Any,
    _validated_distractor_baselines: (Sequence[Sequence[tuple[str, str, str]]] | None) = None,
) -> dict[str, Any]:
    """Reject generated role labels and duplicate distractors visible to an agent.

    Hash and FHIR validation cannot detect benchmark shortcuts.  This check follows the
    patch sequence, scans generated identifiers and human-readable labels, confirms that
    normalized evidence IDs are opaque, and compares each distractor against every clinical
    resource already present immediately before insertion.
    """

    if _validated_distractor_baselines is not None and len(_validated_distractor_baselines) != len(
        chain.patches
    ):
        raise CompositionalReleaseError("validated leakage baseline count mismatch")
    current: Mapping[str, Any] = dict(base_bundle)
    inserted_clinical_ids: set[tuple[str, str]] = set()
    identifier_findings: list[str] = []
    narrative_findings: list[str] = []
    duplicate_findings: list[dict[str, str]] = []
    distractor_count = 0
    for patch_index, patch in enumerate(chain.patches):
        insertions = [
            operation for operation in patch.operations if isinstance(operation, RawFHIRInsert)
        ]
        clinical = [
            operation
            for operation in insertions
            if operation.resource.get("resourceType") not in {"Provenance", "Device"}
        ]
        before_index = (
            _raw_resource_index(current) if _validated_distractor_baselines is None else {}
        )
        for operation in insertions:
            resource = operation.resource
            resource_type, resource_id = _raw_identity(resource)
            identifier_values = [resource_id]
            if operation.full_url is not None:
                identifier_values.append(operation.full_url)
            for value in identifier_values:
                if _ROLE_TOKEN_PATTERN.search(value):
                    identifier_findings.append(f"{resource_type}/{resource_id}:{value}")
            for path, value in _human_readable_values(resource):
                lowered = value.lower()
                if (
                    "patient-tool-gym intervention" in lowered
                    or "answer_preserving_distractor" in lowered
                    or "answer-preserving distractor" in lowered
                ):
                    narrative_findings.append(f"{resource_type}/{resource_id}:{path}")
            if resource_type not in {"Provenance", "Device"}:
                inserted_clinical_ids.add((resource_type, resource_id))

        if patch.intervention_kind == "answer_preserving_distractor":
            distractor_count += len(clinical)
            for operation in clinical:
                resource_type, resource_id = _raw_identity(operation.resource)
                fingerprint = _semantic_fhir_fingerprint(operation.resource)
                baselines = (
                    tuple(
                        (
                            existing_type,
                            existing_id,
                            _semantic_fhir_fingerprint(existing),
                        )
                        for (existing_type, existing_id), existing in before_index.items()
                        if existing_type == resource_type
                    )
                    if _validated_distractor_baselines is None
                    else tuple(_validated_distractor_baselines[patch_index])
                )
                for existing_type, existing_id, existing_fingerprint in baselines:
                    if existing_type == resource_type and existing_fingerprint == fingerprint:
                        duplicate_findings.append(
                            {
                                "distractor": f"{resource_type}/{resource_id}",
                                "duplicates": f"{existing_type}/{existing_id}",
                            }
                        )
        if _validated_distractor_baselines is None:
            current = apply_raw_fhir_patch(current, patch)

    snapshot_records = getattr(normalized_snapshot, "records", ())
    for record in snapshot_records:
        source = getattr(record, "source", None)
        source_id = getattr(source, "resource_id", None)
        resource_type = getattr(record, "resource_type", None)
        if (resource_type, source_id) not in inserted_clinical_ids:
            continue
        evidence_id = getattr(record, "evidence_id", "")
        if isinstance(evidence_id, str) and _ROLE_TOKEN_PATTERN.search(evidence_id):
            identifier_findings.append(f"normalized:{evidence_id}")

    if identifier_findings:
        raise CompositionalReleaseError(
            "agent-visible generated identifier leaks an intervention role: "
            + identifier_findings[0]
        )
    if narrative_findings:
        raise CompositionalReleaseError(
            "agent-visible generated narrative leaks an intervention role: " + narrative_findings[0]
        )
    if duplicate_findings:
        finding = duplicate_findings[0]
        raise CompositionalReleaseError(
            "distractor is a semantic duplicate of an existing chart record: "
            f"{finding['distractor']} == {finding['duplicates']}"
        )
    return {
        "schema_version": "patient-tool-gym.agent-visible-leakage-scan.v1",
        "passed": True,
        "generated_identifier_role_tokens_absent": True,
        "generated_narrative_role_tokens_absent": True,
        "semantic_duplicate_distractors_absent": True,
        "inserted_clinical_resource_count": len(inserted_clinical_ids),
        "distractor_resource_count": distractor_count,
        "scan_hash": content_hash(
            {
                "inserted_clinical_ids": sorted(
                    f"{resource_type}/{resource_id}"
                    for resource_type, resource_id in inserted_clinical_ids
                ),
                "distractor_count": distractor_count,
            }
        ),
    }


def _semantic_fhir_fingerprint(resource: Mapping[str, Any]) -> str:
    """Canonical clinical content, excluding generated identity and presentation."""

    def strip(value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {
                str(child_key): strip(child, key=str(child_key))
                for child_key, child in sorted(value.items(), key=lambda item: str(item[0]))
                if child_key not in {"id", "meta", "text", "contained"}
            }
        if isinstance(value, list):
            return [strip(child, key=key) for child in value]
        return value

    return content_hash(strip(resource))


def _human_readable_values(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            if isinstance(child, str) and str(key) in {
                "text",
                "display",
                "title",
                "div",
                "description",
            }:
                result.append((".".join(child_path), child))
            else:
                result.extend(_human_readable_values(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_human_readable_values(child, path=(*path, str(index))))
    return result


def _validate_state_operation_allowlist(
    patch: RawFHIRPatch,
    operations: tuple[RawFHIRSet | RawFHIRRemove, ...],
    *,
    evidence_state: CompositionalEvidenceState,
) -> None:
    desired_state = "supported" if evidence_state == "conflicting_evidence" else evidence_state
    seen_paths: set[tuple[Any, ...]] = set()
    for operation in operations:
        if (
            operation.resource_type != patch.target_resource_type
            or operation.resource_id != patch.target_resource_id
        ):
            raise CompositionalReleaseError(
                "a state patch may modify only its declared target resource"
            )
        path = tuple(operation.path)
        if path in seen_paths:
            raise CompositionalReleaseError("a state patch contains duplicate field operations")
        seen_paths.add(path)
        if len(path) != 1 or not isinstance(path[0], str):
            raise CompositionalReleaseError(
                "state intervention operations must target one top-level FHIR field"
            )
        field = path[0]
        if evidence_state == "conflicting_evidence" and field == "identifier":
            continue
        if not _state_field_is_allowed(
            patch.target_resource_type,
            desired_state,
            field,
        ):
            raise CompositionalReleaseError(
                f"state intervention cannot modify {patch.target_resource_type}.{field}"
            )


def _state_field_is_allowed(resource_type: str, state: str, field: str) -> bool:
    if resource_type == "Condition":
        allowed = {
            "supported": {"verificationStatus"},
            "refuted": {"verificationStatus", "clinicalStatus"},
            "insufficient_evidence": {"verificationStatus"},
        }[state]
        return field in allowed or (state == "refuted" and field.startswith("abatement"))
    if resource_type == "Observation":
        allowed = {
            "supported": {"status", "dataAbsentReason"},
            "refuted": {"status", "dataAbsentReason"} | _OBSERVATION_VALUE_FIELDS,
            "insufficient_evidence": {
                "status",
                "dataAbsentReason",
            }
            | _OBSERVATION_VALUE_FIELDS,
        }[state]
        return field in allowed
    if resource_type == "Procedure":
        allowed = {"status"}
        if state != "supported":
            allowed |= {"performedDateTime", "performedPeriod"}
        return field in allowed
    if resource_type in {
        "Immunization",
        "MedicationAdministration",
        "MedicationRequest",
    }:
        return field == "status"
    return False


def _validate_clinical_insert_identity(
    operation: RawFHIRInsert,
    *,
    expected_resource_type: str,
) -> None:
    resource_type = operation.resource.get("resourceType")
    resource_id = operation.resource.get("id")
    if (
        resource_type != expected_resource_type
        or not isinstance(resource_id, str)
        or not resource_id
    ):
        raise CompositionalReleaseError(
            "inserted clinical evidence must have the target resource type and a stable ID"
        )
    if operation.full_url != _generated_full_url(resource_type, resource_id):
        raise CompositionalReleaseError(
            "inserted clinical evidence fullUrl is not derived from its exact identity"
        )


def _validate_conflict_clone(
    target: Mapping[str, Any],
    clone: Mapping[str, Any],
) -> None:
    if clone.get("id") == target.get("id"):
        raise CompositionalReleaseError("conflict evidence must have a distinct resource ID")
    if _primary_raw_code(clone) != _primary_raw_code(target):
        raise CompositionalReleaseError("conflict evidence must preserve the target clinical code")
    _validate_cloned_context(target, clone)


def _validate_distractor_clone(
    target: Mapping[str, Any],
    clone: Mapping[str, Any],
) -> None:
    if clone.get("id") == target.get("id"):
        raise CompositionalReleaseError("distractor evidence must have a distinct resource ID")
    target_code = _primary_raw_code(target)
    clone_code = _primary_raw_code(clone)
    if target_code is None or clone_code is None or clone_code == target_code:
        raise CompositionalReleaseError(
            "same-resource distractor evidence must use a different clinical code"
        )
    if clone_code[0] != target_code[0]:
        raise CompositionalReleaseError(
            "same-resource distractor evidence must preserve the clinical code system"
        )
    _validate_cloned_context(target, clone)
    target_date = _first_clinical_date(target)
    clone_date = _first_clinical_date(clone)
    if target_date is not None and clone_date != target_date:
        raise CompositionalReleaseError(
            "same-resource distractor evidence must remain in the target date cell"
        )


def _validate_cloned_context(
    target: Mapping[str, Any],
    clone: Mapping[str, Any],
) -> None:
    for field in ("subject", "patient", "encounter"):
        if field in target and clone.get(field) != target.get(field):
            raise CompositionalReleaseError(
                f"inserted clinical evidence changed the target {field} context"
            )


def _first_clinical_date(resource: Mapping[str, Any]) -> str | None:
    for field in (
        "effectiveDateTime",
        "occurrenceDateTime",
        "performedDateTime",
        "authoredOn",
        "recordedDate",
        "onsetDateTime",
    ):
        value = resource.get(field)
        if isinstance(value, str) and len(value) >= 10:
            return value[:10]
    return None


def _validate_neutral_patch_provenance(
    bundle: Mapping[str, Any],
    *,
    non_insert_operations: tuple[RawFHIRSet | RawFHIRRemove, ...],
    clinical_insert_operations: tuple[RawFHIRInsert, ...],
    provenance_operations: tuple[RawFHIRInsert, ...],
    device_operations: tuple[RawFHIRInsert, ...],
    expected_recorded_at: str | None,
) -> dict[str, Any]:
    update_targets = {
        _raw_bundle_reference(bundle, operation.resource_type, operation.resource_id)
        for operation in non_insert_operations
    }
    create_targets = {str(operation.full_url) for operation in clinical_insert_operations}
    expected_by_operation = {
        operation: targets
        for operation, targets in (
            ("UPDATE", update_targets),
            ("CREATE", create_targets),
        )
        if targets
    }
    observed_operations: set[str] = set()
    known_device_references = {
        _raw_bundle_reference(bundle, resource_type, resource_id)
        for resource_type, resource_id in _raw_resource_index(bundle)
        if resource_type == "Device"
    } | {str(operation.full_url) for operation in device_operations}
    for operation in provenance_operations:
        resource = operation.resource
        coding = resource.get("activity", {}).get("coding")
        code = (
            coding[0].get("code")
            if isinstance(coding, list) and len(coding) == 1 and isinstance(coding[0], Mapping)
            else None
        )
        if code not in expected_by_operation or code in observed_operations:
            raise CompositionalReleaseError("Provenance activity does not match actual operations")
        observed_operations.add(str(code))
        targets = resource.get("target")
        observed_targets = {
            str(row.get("reference")) for row in targets or () if isinstance(row, Mapping)
        }
        if observed_targets != expected_by_operation[code] or len(targets or ()) != len(
            observed_targets
        ):
            raise CompositionalReleaseError("Provenance targets are not operation-exact")
        activity_text = resource.get("activity", {}).get("text")
        if activity_text != f"FHIR resource {str(code).casefold()}":
            raise CompositionalReleaseError("Provenance activity text is not role-neutral")
        agents = resource.get("agent")
        who = (
            agents[0].get("who", {}).get("reference")
            if isinstance(agents, list) and len(agents) == 1 and isinstance(agents[0], Mapping)
            else None
        )
        if who not in known_device_references:
            raise CompositionalReleaseError("Provenance agent is not the compiler Device")
        agent_type = agents[0].get("type", {}).get("coding")
        if not (
            isinstance(agent_type, list)
            and len(agent_type) == 1
            and isinstance(agent_type[0], Mapping)
            and agent_type[0].get("system") == _FHIR_PROVENANCE_AGENT_SYSTEM
            and agent_type[0].get("code") == "assembler"
        ):
            raise CompositionalReleaseError("Provenance agent role is not exact")
        recorded = resource.get("recorded")
        if not isinstance(recorded, str) or (
            expected_recorded_at is not None and recorded != expected_recorded_at
        ):
            raise CompositionalReleaseError("Provenance has no bounded recorded instant")
        for entity in resource.get("entity", []):
            if not isinstance(entity, Mapping) or entity.get("role") != "source":
                raise CompositionalReleaseError("Provenance entity is not a source binding")
            what = entity.get("what")
            if not isinstance(what, Mapping):
                raise CompositionalReleaseError("Provenance source binding is malformed")
            reference = what.get("reference")
            identifier = what.get("identifier")
            source = _raw_resource_by_reference(bundle, reference)
            if (
                source is None
                or not isinstance(identifier, Mapping)
                or identifier.get("value") != content_hash(source)
            ):
                raise CompositionalReleaseError("Provenance source hash does not bind chart input")
    if observed_operations != set(expected_by_operation):
        raise CompositionalReleaseError("Provenance omits an actual CREATE or UPDATE operation")
    return {
        "operation_count": len(observed_operations),
        "target_count": len(update_targets) + len(create_targets),
    }


def _raw_resource_by_reference(
    bundle: Mapping[str, Any],
    reference: Any,
) -> Mapping[str, Any] | None:
    if not isinstance(reference, str):
        return None
    for entry in bundle.get("entry", []):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("resource"), Mapping):
            continue
        resource = entry["resource"]
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        if reference in {entry.get("fullUrl"), f"{resource_type}/{resource_id}"}:
            return resource
    return None


def _validate_exact_provenance(
    bundle: Mapping[str, Any],
    operation: RawFHIRInsert,
    *,
    expected_targets: tuple[str, ...],
    intervention_kind: str,
) -> None:
    patient_ids = [
        resource_id
        for (resource_type, resource_id) in _raw_resource_index(bundle)
        if resource_type == "Patient"
    ]
    if len(patient_ids) != 1:
        raise CompositionalReleaseError(
            "exact intervention Provenance requires one top-level Patient"
        )
    digest = content_hash(
        {"targets": list(expected_targets), "state": intervention_kind}
    ).removeprefix("sha256:")
    provenance_id = f"ptg-{digest[:24]}"
    expected = {
        "resourceType": "Provenance",
        "id": provenance_id,
        "text": {
            "status": "generated",
            "div": (
                '<div xmlns="http://www.w3.org/1999/xhtml">'
                "<p>Provenance record for a chart update.</p></div>"
            ),
        },
        "target": [{"reference": target} for target in expected_targets],
        "recorded": _expected_intervention_recorded(bundle),
        "activity": {
            "coding": [
                {
                    "system": _FHIR_PROVENANCE_ACTIVITY_SYSTEM,
                    "code": "UPDATE",
                }
            ],
            "text": f"patient-tool-gym intervention: {intervention_kind}",
        },
        "agent": [
            {
                "type": {
                    "coding": [
                        {
                            "system": _FHIR_PROVENANCE_AGENT_SYSTEM,
                            "code": "assembler",
                        }
                    ]
                },
                "who": {"reference": f"Patient/{patient_ids[0]}"},
            }
        ],
    }
    if operation.resource != expected:
        raise CompositionalReleaseError(
            "intervention Provenance target/activity/time/agent contract is not exact"
        )
    if operation.full_url != _generated_full_url("Provenance", provenance_id):
        raise CompositionalReleaseError(
            "intervention Provenance fullUrl is not derived from its exact identity"
        )


def _expected_intervention_recorded(bundle: Mapping[str, Any]) -> str:
    candidates: list[datetime] = []

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key in _INTERVENTION_TIME_FIELDS:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return
            if parsed.tzinfo is not None:
                candidates.append(parsed.astimezone(UTC))

    visit(bundle)
    if not candidates:
        raise CompositionalReleaseError(
            "exact intervention Provenance requires a visible chart timestamp"
        )
    return (max(candidates) + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")


def _release_provenance_recorded_at(
    bundle: Mapping[str, Any],
    generation_provenance: Mapping[str, Any],
) -> str:
    upstream = generation_provenance.get("upstream")
    recorded_at = (
        upstream.get("source_generation_completed_at") if isinstance(upstream, Mapping) else None
    )
    if isinstance(recorded_at, str):
        return recorded_at
    timestamp = bundle.get("timestamp")
    if isinstance(timestamp, str):
        return timestamp
    raise CompositionalReleaseError(
        "release interventions require source_generation_completed_at or Bundle.timestamp"
    )


def _generated_full_url(resource_type: str, resource_id: str) -> str:
    deterministic = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"https://patient-tool-gym.dev/fhir/{resource_type}/{resource_id}",
    )
    return f"urn:uuid:{deterministic}"


def _validate_exact_bundle_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    non_insert_operations: tuple[RawFHIRSet | RawFHIRRemove, ...],
    insert_operations: tuple[RawFHIRInsert, ...],
) -> None:
    before_top = {key: value for key, value in before.items() if key != "entry"}
    after_top = {key: value for key, value in after.items() if key != "entry"}
    if before_top != after_top:
        raise CompositionalReleaseError("raw patch changed Bundle metadata outside entry")
    before_entries = before.get("entry")
    after_entries = after.get("entry")
    if not isinstance(before_entries, list) or not isinstance(after_entries, list):
        raise CompositionalReleaseError("raw patch contract requires Bundle.entry arrays")
    if len(after_entries) != len(before_entries) + len(insert_operations):
        raise CompositionalReleaseError("raw patch inserted or removed an undeclared entry")

    actual_changed: set[tuple[str, str]] = set()
    for index, before_entry in enumerate(before_entries):
        after_entry = after_entries[index]
        if not isinstance(before_entry, Mapping) or not isinstance(after_entry, Mapping):
            raise CompositionalReleaseError("raw patch encountered a malformed Bundle entry")
        before_wrapper = {key: value for key, value in before_entry.items() if key != "resource"}
        after_wrapper = {key: value for key, value in after_entry.items() if key != "resource"}
        if before_wrapper != after_wrapper:
            raise CompositionalReleaseError("raw patch changed an existing Bundle entry wrapper")
        before_resource = before_entry.get("resource")
        after_resource = after_entry.get("resource")
        if not isinstance(before_resource, Mapping) or not isinstance(after_resource, Mapping):
            raise CompositionalReleaseError("raw patch encountered a malformed FHIR resource")
        before_identity = _raw_identity(before_resource)
        after_identity = _raw_identity(after_resource)
        if before_identity != after_identity:
            raise CompositionalReleaseError("raw patch changed an existing resource identity")
        if before_resource != after_resource:
            actual_changed.add(before_identity)

    expected_changed = {
        (operation.resource_type, operation.resource_id) for operation in non_insert_operations
    }
    if actual_changed != expected_changed:
        raise CompositionalReleaseError(
            "raw patch changed an existing resource outside its exact operation allowlist"
        )
    for offset, operation in enumerate(insert_operations):
        inserted_entry = after_entries[len(before_entries) + offset]
        if not isinstance(inserted_entry, Mapping):
            raise CompositionalReleaseError("raw patch inserted a malformed Bundle entry")
        expected_entry: dict[str, Any] = {"resource": operation.resource}
        if operation.full_url is not None:
            expected_entry["fullUrl"] = operation.full_url
        if before.get("type") in {"batch", "transaction"}:
            resource_type, resource_id = _raw_identity(operation.resource)
            expected_entry["request"] = {
                "method": "PUT",
                "url": f"{resource_type}/{resource_id}",
            }
        if inserted_entry != expected_entry:
            raise CompositionalReleaseError(
                "raw patch inserted an entry outside its exact operation allowlist"
            )


def _raw_resource_index(bundle: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    entries = bundle.get("entry")
    if not isinstance(entries, list):
        raise CompositionalReleaseError("raw patch contract requires a Bundle entry array")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("resource"), Mapping):
            raise CompositionalReleaseError("raw patch contract encountered a malformed entry")
        resource = entry["resource"]
        identity = _raw_identity(resource)
        if identity in result:
            raise CompositionalReleaseError(
                f"raw patch contract found duplicate {identity[0]}/{identity[1]}"
            )
        result[identity] = resource
    return result


def _raw_identity(resource: Mapping[str, Any]) -> tuple[str, str]:
    resource_type = resource.get("resourceType")
    resource_id = resource.get("id")
    if not isinstance(resource_type, str) or not isinstance(resource_id, str):
        raise CompositionalReleaseError("FHIR resources require resourceType and id")
    return resource_type, resource_id


def _raw_bundle_reference(
    bundle: Mapping[str, Any],
    resource_type: str,
    resource_id: str,
) -> str:
    for entry in bundle.get("entry", []):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("resource"), Mapping):
            continue
        resource = entry["resource"]
        if resource.get("resourceType") == resource_type and resource.get("id") == resource_id:
            full_url = entry.get("fullUrl")
            if isinstance(full_url, str) and full_url:
                return full_url
            return f"{resource_type}/{resource_id}"
    raise CompositionalReleaseError("raw patch target reference cannot be resolved")


def _present_allowed_conditional_reference_types(
    bundle: Mapping[str, Any],
) -> frozenset[str]:
    result: set[str] = set()

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key == "reference" and "?" in value:
            resource_type = value.split("?", 1)[0]
            if resource_type in SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES:
                result.add(resource_type)

    visit(bundle)
    return frozenset(result)


def _compositional_metadata_hash(metadata: Mapping[str, Any]) -> str:
    return content_hash({key: value for key, value in metadata.items() if key != "metadata_hash"})


def _is_content_hash(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError:
        return False
    return True


def _is_task_id(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("patient-task-") or len(value) != 37:
        return False
    try:
        int(value.removeprefix("patient-task-"), 16)
    except ValueError:
        return False
    return True


def verify_compositional_task_contract(
    artifact: TaskArtifact,
    registry: ToolRegistry,
) -> bool:
    """Recompute the normalized-task portion of the compositional release contract."""

    try:
        if set(artifact.metadata) != {"compositional_release"}:
            return False
        metadata = artifact.metadata.get("compositional_release")
        if not isinstance(metadata, dict):
            return False
        required_metadata_keys = {
            "schema_version",
            "superfamily_id",
            "cell_id",
            "patient_id",
            "topology",
            "evidence_state",
            "companion_evidence_state",
            "distractor_injected",
            "original_cell",
            "expected_branch_states",
            "expected_overall_state",
            "base_bundle_hash",
            "patch_chain_hash",
            "derived_bundle_hash",
            "program_hash",
            "question_hash",
            "program_signature",
            "routing_raw_hash",
            "routing_evidence_id",
            "companion_raw_hashes",
            "companion_derived_raw_hashes",
            "companion_evidence_ids",
            "distractor_evidence_id",
            "source_task_hashes",
            "source_task_ids",
            "metadata_hash",
        }
        if set(metadata) != required_metadata_keys:
            return False
        if metadata.get("metadata_hash") != _compositional_metadata_hash(metadata):
            return False
        topology = metadata.get("topology")
        evidence_state = metadata.get("evidence_state")
        companion_evidence_state = metadata.get("companion_evidence_state")
        distractor_injected = metadata.get("distractor_injected")
        original_cell = metadata.get("original_cell")
        if (
            topology not in TOPOLOGIES
            or evidence_state not in EVIDENCE_STATES
            or companion_evidence_state not in EVIDENCE_STATES
            or not isinstance(distractor_injected, bool)
            or not isinstance(original_cell, bool)
            or (topology in WIDTH_ONE_TOPOLOGIES and companion_evidence_state != "supported")
        ):
            return False
        construction = {
            "d1w1": "atomic",
            "d2w1": "depth",
            "d1w2": "width",
            "d2w2": "mixed",
        }[topology]
        if artifact.construction_type != construction:
            return False
        if artifact.provenance.generator != COMPOSITIONAL_RELEASE_GENERATOR:
            return False
        if artifact.provenance.generator_version != COMPOSITIONAL_RELEASE_GENERATOR_VERSION:
            return False
        if artifact.provenance.dataset != "synthea" or not isinstance(
            artifact.provenance.seed, int
        ):
            return False
        if metadata.get("schema_version") != _COMPOSITIONAL_METADATA_SCHEMA_VERSION:
            return False
        superfamily_id = metadata.get("superfamily_id")
        if (
            not isinstance(superfamily_id, str)
            or artifact.family_id != _topology_family_id_from_id(superfamily_id, topology)
            or metadata.get("patient_id") != registry.patient_id
        ):
            return False
        raw_hash_fields = (
            "base_bundle_hash",
            "patch_chain_hash",
            "derived_bundle_hash",
            "routing_raw_hash",
            "program_hash",
            "question_hash",
            "metadata_hash",
        )
        if not all(_is_content_hash(metadata.get(field)) for field in raw_hash_fields):
            return False
        expected_original = (
            evidence_state == "supported"
            and companion_evidence_state == "supported"
            and not distractor_injected
        )
        if original_cell != expected_original or (
            (metadata["base_bundle_hash"] == metadata["derived_bundle_hash"]) != expected_original
        ):
            return False
        expected_cell_digest = content_hash(
            {
                "superfamily_id": superfamily_id,
                "evidence_state": evidence_state,
                "companion_evidence_state": companion_evidence_state,
                "distractor_injected": distractor_injected,
                "chain_hash": metadata["patch_chain_hash"],
            }
        ).removeprefix("sha256:")[:16]
        expected_cell_suffix = "injected" if distractor_injected else "original-no-injection"
        if metadata.get("cell_id") != (
            f"primary-{evidence_state}-companion-{companion_evidence_state}-"
            f"{expected_cell_suffix}-{expected_cell_digest}"
        ):
            return False
        if metadata.get("program_hash") != content_hash(artifact.program.model_dump(mode="json")):
            return False
        if metadata.get("question_hash") != content_hash(artifact.question):
            return False
        if metadata.get("program_signature") != program_signature(artifact.program):
            return False
        verify_topology_complexity(artifact.program, topology)

        expected = _expected_branch_states(
            topology,
            evidence_state,
            companion_evidence_state,
        )
        serialized_expected = {name: state.value for name, state in expected.items()}
        overall = truth_state_all(tuple(expected.values()))
        if metadata.get("expected_branch_states") != serialized_expected:
            return False
        if metadata.get("expected_overall_state") != overall.value:
            return False
        if artifact.gold_answer != TypedValue(
            kind="object",
            value={"branches": expected, "overall": overall},
        ):
            return False

        parent_count = len(artifact.parent_task_ids)
        if parent_count != {"d1w1": 0, "d2w1": 1, "d1w2": 1, "d2w2": 2}[topology]:
            return False
        if len(set(artifact.parent_task_ids)) != parent_count or not all(
            _is_task_id(parent) for parent in artifact.parent_task_ids
        ):
            return False

        routing_evidence_id = metadata.get("routing_evidence_id")
        patient_id = str(metadata["patient_id"])
        if (
            not isinstance(routing_evidence_id, str)
            or not routing_evidence_id.startswith(f"synthea:{patient_id}:")
            or _resource_identity(routing_evidence_id, "Encounter")[0] != "Encounter"
        ):
            return False
        if topology.startswith("d2"):
            encounter_calls = [
                call
                for call in artifact.reference_trace
                if call.node_id == "focal_encounter_search"
            ]
            focal_calls = [
                call for call in artifact.reference_trace if call.node_id == "focal_search"
            ]
            if (
                len(encounter_calls) != 1
                or len(focal_calls) != 1
                or routing_evidence_id not in encounter_calls[0].evidence_ids
                or routing_evidence_id not in artifact.gold_evidence_ids
                or focal_calls[0].depends_on != ("focal_encounter_search",)
            ):
                return False
        elif routing_evidence_id in artifact.gold_evidence_ids:
            return False

        companion_evidence_ids = metadata.get("companion_evidence_ids")
        if not isinstance(companion_evidence_ids, list) or not all(
            isinstance(value, str) for value in companion_evidence_ids
        ):
            return False
        if (
            not companion_evidence_ids
            or companion_evidence_ids != sorted(set(companion_evidence_ids))
            or not all(
                value.startswith(f"synthea:{patient_id}:") for value in companion_evidence_ids
            )
        ):
            return False
        companion_raw_hashes = metadata.get("companion_raw_hashes")
        companion_derived_raw_hashes = metadata.get("companion_derived_raw_hashes")
        if (
            not isinstance(companion_raw_hashes, dict)
            or set(companion_raw_hashes) != set(companion_evidence_ids)
            or not all(_is_content_hash(value) for value in companion_raw_hashes.values())
            or not isinstance(companion_derived_raw_hashes, dict)
            or set(companion_derived_raw_hashes) != set(companion_evidence_ids)
            or not all(_is_content_hash(value) for value in companion_derived_raw_hashes.values())
            or (
                companion_evidence_state == "supported"
                and companion_derived_raw_hashes != companion_raw_hashes
            )
        ):
            return False
        companion_overlap = set(companion_evidence_ids) & set(artifact.gold_evidence_ids)
        if topology.endswith("w2"):
            if not companion_overlap:
                return False
        elif companion_overlap:
            return False

        distractor_id = metadata.get("distractor_evidence_id")
        if distractor_injected:
            focal_call = next(
                call for call in artifact.reference_trace if call.node_id == "focal_search"
            )
            if (
                not isinstance(distractor_id, str)
                or not distractor_id.startswith(f"synthea:{patient_id}:")
                or distractor_id not in focal_call.evidence_ids
                or distractor_id in artifact.gold_evidence_ids
            ):
                return False
        elif distractor_id is not None:
            return False
        source_hashes = metadata.get("source_task_hashes")
        source_ids = metadata.get("source_task_ids")
        if not isinstance(source_hashes, dict) or set(source_hashes) != {
            "focal_atomic",
            "focal_depth",
            "companion",
        }:
            return False
        if not isinstance(source_ids, dict) or set(source_ids) != set(source_hashes):
            return False
        if not all(_is_content_hash(value) for value in source_hashes.values()):
            return False
        if not all(_is_task_id(value) for value in source_ids.values()):
            return False
        source_identity = {
            "schema_version": COMPOSITIONAL_MATRIX_SCHEMA_VERSION,
            "focal_atomic_task_id": source_ids["focal_atomic"],
            "focal_depth_task_id": source_ids["focal_depth"],
            "companion_task_id": source_ids["companion"],
            "source_task_hashes": source_hashes,
        }
        source_digest = content_hash(source_identity).removeprefix("sha256:")
        if superfamily_id != f"compositional-family-{source_digest[:20]}":
            return False
    except (KeyError, StopIteration, TypeError, ValueError):
        return False
    return True


def _load_source_tasks(
    source: Sequence[TaskArtifact] | Path,
) -> tuple[tuple[TaskArtifact, ...], dict[str, Any], bytes]:
    if isinstance(source, Path):
        raw = source.read_bytes()
        tasks = tuple(
            TaskArtifact.model_validate_json(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        )
        record = {
            "kind": "jsonl",
            "name": source.name,
            "sha256": _sha256_bytes(raw),
            "task_count": len(tasks),
        }
        release_raw = raw
    else:
        tasks = tuple(source)
        release_raw = b"".join(
            _canonical_bytes(task.model_dump(mode="json")) + b"\n"
            for task in sorted(tasks, key=lambda item: item.task_id)
        )
        record = {
            "kind": "in_memory",
            "content_hash": content_hash(
                [
                    task.model_dump(mode="json")
                    for task in sorted(tasks, key=lambda item: item.task_id)
                ]
            ),
            "task_count": len(tasks),
        }
    if not tasks:
        raise CompositionalReleaseError("source task collection is empty")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise CompositionalReleaseError("source task collection has duplicate task IDs")
    return tasks, record, release_raw


def _release_candidate_discovery(
    matrix_discovery: dict[str, Any],
    bundle_catalog: Mapping[tuple[str, str], list[dict[str, Any]]],
    *,
    generation_provenance: Mapping[str, Any],
) -> tuple[tuple[CompositionalCandidate, ...], dict[str, Any]]:
    """Apply medical and raw-chart representability gates without a quota."""

    eligible: list[CompositionalCandidate] = []
    rejections: list[dict[str, Any]] = []
    for payload in matrix_discovery["candidates"]:
        candidate = CompositionalCandidate.model_validate(payload)
        if candidate.focal_domain == "procedure":
            rejections.append(
                {
                    "superfamily_id": candidate.superfamily_id,
                    "focal_family_id": candidate.focal_family_id,
                    "focal_domain": candidate.focal_domain,
                    "reason": "REFUTED_DATE_NOT_REPRESENTABLE",
                    "details": {
                        "explanation": (
                            "FHIR R4 Procedure.status=not-done must not be represented as an "
                            "event with performed[x], while the exact-date selector requires "
                            "performed[x] to retain the chart cell's date. Procedure is also "
                            "excluded from the four-state companion pool."
                        ),
                        "planned_resolution": (
                            "future encounter-date selection with guarded nonperformance semantics"
                        ),
                    },
                }
            )
            continue
        if candidate.focal_domain not in {
            "condition",
            "lab_observation",
            "immunization",
        }:
            rejections.append(
                {
                    "superfamily_id": candidate.superfamily_id,
                    "focal_family_id": candidate.focal_family_id,
                    "focal_domain": candidate.focal_domain,
                    "reason": "UNSUPPORTED_FOCAL_DOMAIN",
                    "details": {},
                }
            )
            continue
        catalog_entry = _bundle_for_candidate(candidate, bundle_catalog)
        bundle = catalog_entry["bundle"]
        focal_id = _resource_id(candidate.focal_evidence_id, candidate.focal_resource_type)
        forbidden_codes = frozenset(
            {
                (candidate.focal_selector.system, candidate.focal_selector.code),
                (candidate.companion_selector.system, candidate.companion_selector.code),
            }
        )
        try:
            patch = build_distractor_patch(
                bundle,
                resource_type=candidate.focal_resource_type,
                resource_id=focal_id,
                distractor_tier="same_resource_different_code",
                forbidden_codes=forbidden_codes,
                target_date=(
                    candidate.focal_selector.date_from.isoformat()
                    if candidate.focal_selector.date_from is not None
                    else None
                ),
                require_exact_tier=True,
                provenance_recorded_at=_release_provenance_recorded_at(
                    bundle,
                    generation_provenance,
                ),
            )
        except RawFHIRDistractorUnavailableError as exc:
            rejections.append(
                {
                    "superfamily_id": candidate.superfamily_id,
                    "focal_family_id": candidate.focal_family_id,
                    "focal_domain": candidate.focal_domain,
                    "reason": "NO_STRICT_DISTRACTOR",
                    "details": {
                        "requested_tier": exc.requested_tier,
                        "target_resource_type": exc.target_resource_type,
                        "target_resource_id": exc.target_resource_id,
                        "required_code_system": candidate.focal_selector.system,
                        "required_encounter_evidence_id": candidate.routing_evidence_id,
                        "forbidden_codes": [
                            {"system": system, "code": code}
                            for system, code in sorted(forbidden_codes)
                        ],
                        "resolution": (
                            "candidate excluded; the compiler does not synthesize or move a "
                            "source record to satisfy the distractor contract"
                        ),
                    },
                }
            )
            continue
        if patch.distractor_tier != "same_resource_different_code":
            raise CompositionalReleaseError(
                f"{candidate.superfamily_id} strict distractor preflight changed tier"
            )
        eligible.append(candidate)
    eligible.sort(key=lambda item: item.superfamily_id)
    rejections.sort(key=lambda row: (row["superfamily_id"], row["reason"]))
    report = {
        "schema_version": "patient-tool-gym.compositional-release-discovery.v2",
        "selection_policy": "all_medically_representable_candidates",
        "quota": None,
        "raw_bundle_preflight": {
            "required_distractor_tier": "same_resource_different_code",
            "same_resource_type": True,
            "same_code_system": True,
            "same_encounter": True,
            "different_code": True,
            "source_records_synthesized_or_moved": False,
        },
        "matrix_discovery_hash": content_hash(matrix_discovery),
        "matrix_discovery": matrix_discovery,
        "matrix_candidate_count": len(matrix_discovery["candidates"]),
        "eligible_candidate_count": len(eligible),
        "candidates": [candidate.model_dump(mode="json") for candidate in eligible],
        "release_rejections": rejections,
    }
    report["discovery_hash"] = content_hash(report)
    return tuple(eligible), report


def _load_bundle_catalog(
    source: Mapping[str, Mapping[str, Any]] | Path,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    rows: list[tuple[str, dict[str, Any], str]] = []
    if isinstance(source, Path):
        paths = [source] if source.is_file() else sorted(source.rglob("*.json"))
        if not paths:
            raise CompositionalReleaseError("source Bundle path contains no JSON files")
        for path in paths:
            bundle = load_fhir_bundle(path)
            raw = path.read_bytes()
            name = path.name if source.is_file() else path.relative_to(source).as_posix()
            rows.append((name, bundle, _sha256_bytes(raw)))
        source_kind = "path"
    else:
        for name, value in sorted(source.items()):
            bundle = dict(value)
            rows.append(
                (
                    str(name),
                    bundle,
                    _sha256_bytes(_canonical_bytes(bundle)),
                )
            )
        source_kind = "mapping"
    if not rows:
        raise CompositionalReleaseError("source Bundle collection is empty")

    catalog: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    dependency_payloads: list[dict[str, Any]] = []
    patient_bundles: list[dict[str, Any]] = []
    for name, bundle, file_hash in rows:
        patient_count = _top_level_patient_count(bundle, name=name)
        if patient_count == 0:
            resource_counts = Counter(
                str(entry["resource"].get("resourceType"))
                for entry in bundle.get("entry", [])
                if isinstance(entry, Mapping) and isinstance(entry.get("resource"), Mapping)
            )
            dependency_payloads.append(
                {
                    "name": name,
                    "file_sha256": file_hash,
                    "bundle_hash": content_hash(bundle),
                    "resource_counts": dict(sorted(resource_counts.items())),
                    "bundle": bundle,
                }
            )
            skipped.append(
                {
                    "name": name,
                    "file_sha256": file_hash,
                    "reason": "DECLARED_TRANSACTION_DEPENDENCY",
                    "resource_counts": dict(sorted(resource_counts.items())),
                }
            )
            continue
        if patient_count > 1:
            raise CompositionalReleaseError(
                f"source Bundle {name} contains {patient_count} top-level Patient resources"
            )
        closure = _strict_closure(bundle)
        if not closure.passed:
            raise CompositionalReleaseError(f"source Bundle is not strictly closed: {name}")
        snapshot = normalize_fhir_bundle(bundle, dataset="synthea")
        semantic_hash = content_hash(bundle)
        patient_bundles.append(bundle)
        row = {
            "name": name,
            "file_sha256": file_hash,
            "bundle_hash": semantic_hash,
            "patient_id": snapshot.patient_id,
            "snapshot_id": snapshot.snapshot_id,
            "bundle": bundle,
        }
        catalog.setdefault((snapshot.patient_id, snapshot.snapshot_id), []).append(row)
        manifest_rows.append({key: value for key, value in row.items() if key != "bundle"})
    if not manifest_rows:
        raise CompositionalReleaseError(
            "source Bundle collection contains no single-patient Bundles"
        )
    dependency_audit = _conditional_dependency_resolution_audit(
        patient_bundles,
        [row["bundle"] for row in dependency_payloads],
        scope="all_source_patient_bundles",
    )
    if not dependency_audit["passed"]:
        raise CompositionalReleaseError(
            "source conditional references are not uniquely resolved by dependencies"
        )
    return catalog, {
        "kind": source_kind,
        "input_json_count": len(rows),
        "bundle_count": len(manifest_rows),
        "skipped_nonpatient_bundle_count": len(skipped),
        "skipped": sorted(skipped, key=lambda row: row["name"]),
        "bundles": sorted(manifest_rows, key=lambda row: (row["name"], row["bundle_hash"])),
        "conditional_dependency_resolution": dependency_audit,
        "_dependency_payloads": dependency_payloads,
    }


def _top_level_patient_count(bundle: Mapping[str, Any], *, name: str) -> int:
    if bundle.get("resourceType") != "Bundle":
        raise CompositionalReleaseError(f"source JSON is not a FHIR Bundle: {name}")
    entries = bundle.get("entry")
    if not isinstance(entries, list):
        raise CompositionalReleaseError(f"source Bundle has no entry array: {name}")
    return sum(
        1
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(entry.get("resource"), Mapping)
        and entry["resource"].get("resourceType") == "Patient"
    )


def _conditional_dependency_resolution_audit(
    patient_bundles: Sequence[Mapping[str, Any]],
    dependency_bundles: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    dependency_index: dict[tuple[str, str, str], set[str]] = {}
    dependency_summaries: list[dict[str, Any]] = []
    for bundle in dependency_bundles:
        methods: Counter[str] = Counter()
        resource_counts: Counter[str] = Counter()
        for entry in bundle.get("entry", []):
            if not isinstance(entry, Mapping) or not isinstance(entry.get("resource"), Mapping):
                continue
            resource = entry["resource"]
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            if not isinstance(resource_type, str) or not isinstance(resource_id, str):
                continue
            resource_counts[resource_type] += 1
            request = entry.get("request")
            if isinstance(request, Mapping) and isinstance(request.get("method"), str):
                methods[str(request["method"])] += 1
            for identifier in resource.get("identifier", []):
                if not isinstance(identifier, Mapping):
                    continue
                system = identifier.get("system")
                value = identifier.get("value")
                if isinstance(system, str) and isinstance(value, str):
                    dependency_index.setdefault((resource_type, system, value), set()).add(
                        f"{resource_type}/{resource_id}"
                    )
        dependency_summaries.append(
            {
                "bundle_type": bundle.get("type"),
                "entry_count": len(bundle.get("entry", [])),
                "request_methods": dict(sorted(methods.items())),
                "resource_counts": dict(sorted(resource_counts.items())),
            }
        )

    references: list[str] = []

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key == "reference" and "?" in value:
            references.append(value)

    for bundle in patient_bundles:
        visit(bundle)

    unresolved: Counter[str] = Counter()
    ambiguous: Counter[str] = Counter()
    resolved_by_type: Counter[str] = Counter()
    for reference in references:
        resource_type, query = reference.split("?", 1)
        identifier_values = parse_qs(query, keep_blank_values=True).get("identifier", [])
        if len(identifier_values) != 1 or "|" not in identifier_values[0]:
            unresolved[reference] += 1
            continue
        system, value = unquote(identifier_values[0]).split("|", 1)
        matches = dependency_index.get((resource_type, system, value), set())
        if len(matches) == 1:
            resolved_by_type[resource_type] += 1
        elif not matches:
            unresolved[reference] += 1
        else:
            ambiguous[reference] += 1
    payload = {
        "schema_version": "patient-tool-gym.fhir-transaction-dependency-audit.v1",
        "scope": scope,
        "passed": not unresolved and not ambiguous,
        "patient_bundle_count": len(patient_bundles),
        "dependency_bundle_count": len(dependency_bundles),
        "conditional_reference_count": len(references),
        "uniquely_resolved_count": sum(resolved_by_type.values()),
        "unresolved_count": sum(unresolved.values()),
        "ambiguous_count": sum(ambiguous.values()),
        "resolved_counts_by_resource_type": dict(sorted(resolved_by_type.items())),
        "unresolved_examples": sorted(unresolved)[:20],
        "ambiguous_examples": sorted(ambiguous)[:20],
        "dependency_bundles": dependency_summaries,
        "declared_load_order": [
            "dependency_bundles",
            "patient_bundle",
        ],
    }
    payload["audit_hash"] = content_hash(payload)
    return payload


def _materialize_dependency_bundles(
    staging: Path,
    payloads: Sequence[Mapping[str, Any]],
    *,
    resolution_audit: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        bundle = payload.get("bundle")
        resource_counts = payload.get("resource_counts")
        if not isinstance(bundle, Mapping) or not isinstance(resource_counts, Mapping):
            raise CompositionalReleaseError("transaction dependency payload is malformed")
        resource_types = set(map(str, resource_counts))
        if resource_types <= {"Location", "Organization"}:
            stage = 1
        elif resource_types <= {"Practitioner", "PractitionerRole"}:
            stage = 2
        else:
            raise CompositionalReleaseError(
                "nonpatient source Bundle has an undeclared dependency resource type"
            )
        bundle_hash = content_hash(dict(bundle))
        name = str(payload.get("name", "dependency.json"))
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in Path(name).stem
        )
        relative = (
            Path("dependencies")
            / f"{stage:02d}-{safe_name}-{bundle_hash.removeprefix('sha256:')[:16]}.json"
        ).as_posix()
        _write_json(staging / relative, bundle)
        rows.append(
            {
                "load_stage": stage,
                "source_name": name,
                "path": relative,
                "file_sha256": _sha256_file(staging / relative),
                "bundle_hash": bundle_hash,
                "bundle_type": bundle.get("type"),
                "entry_count": len(bundle.get("entry", [])),
                "resource_counts": dict(sorted(resource_counts.items())),
            }
        )
    rows.sort(key=lambda row: (row["load_stage"], row["path"]))
    manifest = {
        "schema_version": "patient-tool-gym.fhir-transaction-dependencies.v1",
        "dependency_bundle_count": len(rows),
        "load_order": [row["path"] for row in rows],
        "patient_bundles_load_after_dependencies": True,
        "dependencies": rows,
        "resolution_audit": dict(resolution_audit),
    }
    manifest["dependency_manifest_hash"] = content_hash(manifest)
    return manifest


def _bundle_for_candidate(
    candidate: CompositionalCandidate,
    catalog: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    matches = catalog.get((candidate.patient_id, candidate.patient_snapshot_id), [])
    by_hash = {str(row["bundle_hash"]): row for row in matches}
    if len(by_hash) != 1:
        raise CompositionalReleaseError(
            f"{candidate.superfamily_id} requires exactly one raw Bundle matching patient "
            f"{candidate.patient_id} and snapshot {candidate.patient_snapshot_id}; "
            f"found {len(by_hash)}"
        )
    return next(iter(by_hash.values()))


def _topology_contracts(
    candidate: CompositionalCandidate,
    builds: Mapping[CompositionalTopology, Any],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        built = builds[topology]
        rows[topology] = {
            "family_id": _topology_family_id(candidate, topology),
            "construction_type": built.construction_type,
            "program": built.program.model_dump(mode="json"),
            "program_hash": content_hash(built.program.model_dump(mode="json")),
            "program_signature": program_signature(built.program),
            "question": built.question,
            "question_hash": content_hash(built.question),
            "structural_complexity": built.structural_complexity.model_dump(mode="json"),
        }
    return rows


def _verify_base_support(
    candidate: CompositionalCandidate,
    bundle: Mapping[str, Any],
    builds: Mapping[CompositionalTopology, Any],
) -> None:
    snapshot = normalize_fhir_bundle(bundle, dataset="synthea")
    if snapshot.snapshot_id != candidate.patient_snapshot_id:
        raise CompositionalReleaseError(
            f"raw Bundle snapshot does not match source tasks for {candidate.superfamily_id}"
        )
    registry = ToolRegistry(snapshot)
    for topology in TOPOLOGIES:
        built = builds[topology]
        branches = _execute_branch_states(registry, built.program)
        if set(branches.values()) != {TruthState.SUPPORTED}:
            raise CompositionalReleaseError(
                f"base chart is not supported for every {topology} branch: {branches}"
            )
        result = ProgramExecutor(registry).execute(built.program)
        expected_answer = TypedValue(
            kind="object",
            value={
                "branches": {name: TruthState.SUPPORTED for name in sorted(branches)},
                "overall": TruthState.SUPPORTED,
            },
        )
        if result.answer != expected_answer:
            raise CompositionalReleaseError(
                f"base chart does not replay as supported for {topology}"
            )


def _execute_branch_states(
    registry: ToolRegistry,
    program: TaskProgram,
) -> dict[str, TruthState]:
    aggregate = next(
        (node for node in reversed(program.nodes) if isinstance(node, TruthStateAggregateNode)),
        None,
    )
    if aggregate is None:
        raise CompositionalReleaseError("compositional program has no truth-state aggregate")
    states: dict[str, TruthState] = {}
    for name, reference in sorted(aggregate.branches.items()):
        branch_program = TaskProgram(
            schema_version=program.schema_version,
            nodes=program.nodes,
            output=reference,
            output_schema=ProgramOutputSchema(kind="truth_state"),
        )
        result = ProgramExecutor(registry).execute(branch_program)
        if result.answer.kind != "truth_state":
            raise CompositionalReleaseError("branch program did not return a truth state")
        states[name] = TruthState(result.answer.value)
    return states


def _expected_branch_states(
    topology: CompositionalTopology,
    evidence_state: CompositionalEvidenceState,
    companion_evidence_state: CompositionalEvidenceState,
) -> dict[str, TruthState]:
    branches = {"primary": TruthState(evidence_state)}
    if topology in {"d1w2", "d2w2"}:
        branches["companion"] = TruthState(companion_evidence_state)
    return dict(sorted(branches.items()))


def _intra_cell_parent_ids(
    topology: CompositionalTopology,
    task_ids: Mapping[str, str],
) -> tuple[str, ...]:
    if topology == "d1w1":
        return ()
    if topology in {"d2w1", "d1w2"}:
        root = task_ids.get("d1w1")
        if root is None:
            raise CompositionalReleaseError(
                f"{topology} requires the same-cell d1w1 task to be compiled first"
            )
        return (root,)
    parents = tuple(sorted(task_ids[parent] for parent in ("d2w1", "d1w2") if parent in task_ids))
    if len(parents) != 2:
        raise CompositionalReleaseError(
            "d2w2 requires the same-cell d2w1 and d1w2 tasks to be compiled first"
        )
    return parents


def _resource_identity(
    evidence_id: str,
    expected_resource_type: str | None = None,
) -> tuple[str, str]:
    parts = evidence_id.split(":", 3)
    if len(parts) != 4 or not parts[2] or not parts[3]:
        raise CompositionalReleaseError(f"invalid evidence ID: {evidence_id}")
    resource_type, resource_id = parts[2], parts[3]
    if ":version:" in resource_id:
        resource_id = resource_id.split(":version:", 1)[0]
    if resource_type == "Observation" and ":component:" in resource_id:
        resource_id = resource_id.split(":component:", 1)[0]
    if expected_resource_type is not None and resource_type != expected_resource_type:
        raise CompositionalReleaseError(f"evidence {evidence_id} is not a {expected_resource_type}")
    return resource_type, resource_id


def _resource_id(evidence_id: str, expected_resource_type: str) -> str:
    return _resource_identity(evidence_id, expected_resource_type)[1]


def _raw_resource_hash(bundle: Mapping[str, Any], resource_type: str, resource_id: str) -> str:
    resources = [
        entry["resource"]
        for entry in bundle.get("entry", [])
        if isinstance(entry, Mapping)
        and isinstance(entry.get("resource"), Mapping)
        and entry["resource"].get("resourceType") == resource_type
        and entry["resource"].get("id") == resource_id
    ]
    if len(resources) != 1:
        raise CompositionalReleaseError(
            f"raw Bundle requires exactly one {resource_type}/{resource_id}; found {len(resources)}"
        )
    return content_hash(dict(resources[0]))


def _validate_distractor_contract(
    candidate: CompositionalCandidate,
    chain: RawFHIRPatchChain,
    *,
    forbidden_codes: frozenset[tuple[str, str]],
) -> str | None:
    """Return the inserted distractor evidence ID after validating its role."""

    if not chain.distractor_present:
        return None
    patch = chain.patches[-1]
    if patch.distractor_tier != "same_resource_different_code":
        raise CompositionalReleaseError(
            f"{candidate.superfamily_id} distractor did not preserve the requested tier: "
            f"{patch.distractor_tier}"
        )
    inserted = [
        operation.resource
        for operation in patch.operations
        if isinstance(operation, RawFHIRInsert)
        and operation.resource.get("resourceType") not in {"Provenance", "Device"}
    ]
    if len(inserted) != 1:
        raise CompositionalReleaseError(
            f"{candidate.superfamily_id} distractor patch must insert one clinical record"
        )
    resource = inserted[0]
    resource_type = resource.get("resourceType")
    resource_id = resource.get("id")
    if (
        resource_type != candidate.focal_resource_type
        or not isinstance(resource_id, str)
        or not resource_id
    ):
        raise CompositionalReleaseError(
            f"{candidate.superfamily_id} distractor is not a same-resource clinical record"
        )
    distractor_code = _primary_raw_code(resource)
    if (
        distractor_code is None
        or distractor_code[0] != candidate.focal_selector.system
        or distractor_code in forbidden_codes
    ):
        raise CompositionalReleaseError(
            f"{candidate.superfamily_id} distractor changed systems or reused a forbidden code"
        )
    return f"synthea:{candidate.patient_id}:{resource_type}:{resource_id}"


def _primary_raw_code(resource: Mapping[str, Any]) -> tuple[str, str] | None:
    resource_type = resource.get("resourceType")
    concept: Any = resource.get("code")
    if resource_type == "Immunization":
        concept = resource.get("vaccineCode")
    if not isinstance(concept, Mapping):
        return None
    coding = concept.get("coding")
    if isinstance(coding, list):
        for item in coding:
            if (
                isinstance(item, Mapping)
                and isinstance(item.get("system"), str)
                and isinstance(item.get("code"), str)
            ):
                return str(item["system"]), str(item["code"])
    if isinstance(concept.get("system"), str) and isinstance(concept.get("code"), str):
        return str(concept["system"]), str(concept["code"])
    return None


def _verify_distractor_retrieval(
    task: TaskArtifact,
    *,
    distractor_evidence_id: str | None,
    cell_id: str,
    topology: CompositionalTopology,
) -> None:
    if distractor_evidence_id is None:
        return
    focal_calls = [call for call in task.reference_trace if call.node_id == "focal_search"]
    if len(focal_calls) != 1:
        raise CompositionalReleaseError(
            f"{cell_id}/{topology} has no unique focal broad-search call"
        )
    if distractor_evidence_id not in focal_calls[0].evidence_ids:
        raise CompositionalReleaseError(
            f"{cell_id}/{topology} distractor was not observed by focal broad search"
        )
    if distractor_evidence_id in task.gold_evidence_ids:
        raise CompositionalReleaseError(
            f"{cell_id}/{topology} distractor leaked into answer evidence"
        )


def _strict_closure(bundle: Mapping[str, Any]) -> ReferenceClosureReport:
    return validate_reference_closure_strict(
        bundle,
        allowed_external_conditional_reference_types=(SYNTHEA_EXTERNAL_CONDITIONAL_REFERENCE_TYPES),
    )


def _base_bundle_path(patient_id: str, bundle_hash: str) -> str:
    digest = bundle_hash.removeprefix("sha256:")[:16]
    safe_patient = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in patient_id
    )
    return (Path("bundles/base") / f"{safe_patient}-{digest}.json").as_posix()


def _cell_id(
    candidate: CompositionalCandidate,
    chain: RawFHIRPatchChain,
    *,
    evidence_state: CompositionalEvidenceState,
    companion_evidence_state: CompositionalEvidenceState,
    distractor_injected: bool,
) -> str:
    digest = content_hash(
        {
            "superfamily_id": candidate.superfamily_id,
            "evidence_state": evidence_state,
            "companion_evidence_state": companion_evidence_state,
            "distractor_injected": distractor_injected,
            "chain_hash": chain.chain_hash,
        }
    ).removeprefix("sha256:")[:16]
    suffix = "injected" if distractor_injected else "original-no-injection"
    return f"primary-{evidence_state}-companion-{companion_evidence_state}-{suffix}-{digest}"


def _topology_family_id(
    candidate: CompositionalCandidate,
    topology: CompositionalTopology,
) -> str:
    return f"{candidate.superfamily_id}-{topology}"


def _topology_family_id_from_id(
    superfamily_id: str,
    topology: CompositionalTopology,
) -> str:
    return f"{superfamily_id}-{topology}"


def _validation_report(
    rows: list[dict[str, Any]],
    *,
    candidate_count: int,
) -> dict[str, Any]:
    expected = candidate_count * TASKS_PER_SUPERFAMILY
    if len(rows) != expected:
        raise CompositionalReleaseError(
            f"validation coverage is incomplete: expected {expected}, got {len(rows)}"
        )
    passed = all(row["passed"] for row in rows)
    if not passed:
        raise CompositionalReleaseError("one or more compositional validation rows failed")
    report = {
        "schema_version": "patient-tool-gym.compositional-validation-report.v2",
        "passed": True,
        "candidate_count": candidate_count,
        "cell_count": candidate_count * FACTORIAL_CELL_COUNT,
        "task_count": len(rows),
        "counts_by_topology": dict(sorted(Counter(row["topology"] for row in rows).items())),
        "counts_by_evidence_state": dict(
            sorted(Counter(row["evidence_state"] for row in rows).items())
        ),
        "counts_by_companion_evidence_state": dict(
            sorted(Counter(row["companion_evidence_state"] for row in rows).items())
        ),
        "counts_by_distractor_injection": {
            str(key).lower(): value
            for key, value in sorted(
                Counter(row["distractor_injected"] for row in rows).items(),
                key=lambda item: item[0],
            )
        },
        "rows": rows,
    }
    report["validation_hash"] = content_hash(report)
    return report


def _build_leakage_safe_splits(
    tasks: list[TaskArtifact],
    *,
    seed: int,
) -> dict[str, Any]:
    by_patient: dict[str, list[TaskArtifact]] = {}
    for task in tasks:
        metadata = task.metadata["compositional_release"]
        by_patient.setdefault(metadata["patient_id"], []).append(task)
    ordered_patients = sorted(
        by_patient,
        key=lambda patient_id: (
            hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest(),
            patient_id,
        ),
    )
    assignments = _balanced_unit_assignments(
        [(patient_id, len(by_patient[patient_id])) for patient_id in ordered_patients]
    )
    rows = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        metadata = task.metadata["compositional_release"]
        rows.append(
            {
                "task_id": task.task_id,
                "task_hash": content_hash(task.model_dump(mode="json")),
                "split": assignments[metadata["patient_id"]],
                "patient_id": metadata["patient_id"],
                "superfamily_id": metadata["superfamily_id"],
                "family_id": task.family_id,
                "cell_id": metadata["cell_id"],
                "topology": metadata["topology"],
            }
        )
    overlap = {
        axis: _split_overlap(rows, axis)
        for axis in ("patient_id", "superfamily_id", "family_id", "cell_id")
    }
    passed = all(not values for axis_report in overlap.values() for values in axis_report.values())
    if not passed:
        raise CompositionalReleaseError("split construction introduced family leakage")
    payload = {
        "unit_key": "patient_id",
        "task_assignments": rows,
        "unit_assignments": [
            {
                "patient_id": patient_id,
                "split": assignments[patient_id],
                "task_count": len(by_patient[patient_id]),
            }
            for patient_id in sorted(by_patient)
        ],
    }
    return {
        "schema_version": "patient-tool-gym.compositional-splits.v2",
        "seed": seed,
        "policy": "patient_and_nested_family_disjoint",
        "ratios": {"train": 0.70, "dev": 0.15, "test": 0.15},
        "three_nonempty_splits_possible": len(by_patient) >= 3,
        "passed": True,
        "counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "overlap_audit": overlap,
        "assignments_hash": content_hash(payload),
        "assignments": payload,
    }


def _source_bundle_catalog(
    input_catalog: dict[str, Any],
    families: list[dict[str, Any]],
) -> dict[str, Any]:
    bindings = [
        {
            "superfamily_id": family["superfamily_id"],
            "patient_id": family["patient_id"],
            "catalog_name": family["source_bundle"]["catalog_name"],
            "base_bundle_path": family["source_bundle"]["base_bundle_path"],
            "base_bundle_hash": family["source_bundle"]["base_bundle_hash"],
            "base_bundle_file_sha256": family["source_bundle"]["base_bundle_file_sha256"],
            "base_snapshot_id": family["source_bundle"]["base_snapshot_id"],
        }
        for family in families
    ]
    bindings.sort(key=lambda row: row["superfamily_id"])
    unique_bundles = {(row["base_bundle_path"], row["base_bundle_hash"]) for row in bindings}
    payload = {
        "schema_version": "patient-tool-gym.compositional-source-bundles.v2",
        "patient_bundle_reconstruction": True,
        "conditional_dependencies_declared_separately": True,
        "input_catalog": input_catalog,
        "bound_bundle_count": len(unique_bundles),
        "candidate_binding_count": len(bindings),
        "candidate_bindings": bindings,
    }
    payload["catalog_hash"] = content_hash(payload)
    return payload


def _balanced_unit_assignments(
    units: list[tuple[str, int]],
) -> dict[str, SplitName]:
    names: tuple[SplitName, ...] = (
        "train",
        "dev",
        "test",
    )
    targets = {"train": 0.70, "dev": 0.15, "test": 0.15}
    counts = {name: 0 for name in names}
    assignments: dict[str, SplitName] = {}
    total = sum(weight for _, weight in units)
    for index, (unit_id, weight) in enumerate(units):
        remaining = len(units) - index
        empty: list[SplitName] = [name for name in names if counts[name] == 0]
        choices: list[SplitName] = empty if remaining == len(empty) else list(names)

        def objective(name: SplitName, weight: int = weight) -> tuple[float, int]:
            projected = dict(counts)
            projected[name] += weight
            error = sum(abs(projected[key] - targets[key] * total) for key in names)
            return error, names.index(name)

        selected = min(choices, key=objective)
        assignments[unit_id] = selected
        counts[selected] += weight
    return assignments


def _split_overlap(rows: list[dict[str, Any]], axis: str) -> dict[str, list[str]]:
    values = {
        split: {str(row[axis]) for row in rows if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    return {
        "train_dev": sorted(values["train"] & values["dev"]),
        "train_test": sorted(values["train"] & values["test"]),
        "dev_test": sorted(values["dev"] & values["test"]),
    }


def _verify_closed_release_file_inventory(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise CompositionalReleaseError("release manifest files must be an array")
    declared: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, dict) or set(raw_row) != {"path", "bytes", "sha256"}:
            raise CompositionalReleaseError("release manifest contains a malformed file row")
        relative = raw_row.get("path")
        byte_count = raw_row.get("bytes")
        digest = raw_row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative == "manifest.json"
        ):
            raise CompositionalReleaseError("release manifest contains an unsafe file path")
        if relative in declared:
            raise CompositionalReleaseError("release manifest contains duplicate file paths")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise CompositionalReleaseError("release manifest contains an invalid file size")
        if not _is_plain_sha256(digest):
            raise CompositionalReleaseError("release manifest contains an invalid file digest")
        declared[relative] = raw_row

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CompositionalReleaseError("release directory cannot contain symbolic links")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(declared) | {"manifest.json"}:
        raise CompositionalReleaseError("release file inventory is not closed")
    for relative, row in declared.items():
        path = root / relative
        if path.stat().st_size != row["bytes"] or _sha256_file(path) != row["sha256"]:
            raise CompositionalReleaseError(f"release file hash or size mismatch: {relative}")
    return declared


def _verify_standalone_family(
    root: Path,
    family: dict[str, Any],
    *,
    task_index: Mapping[str, TaskArtifact],
) -> list[str]:
    if family.get("schema_version") != "patient-tool-gym.compositional-family.v2":
        raise CompositionalReleaseError("release contains an unsupported family schema")
    if family.get("family_hash") != content_hash(
        {key: value for key, value in family.items() if key != "family_hash"}
    ):
        raise CompositionalReleaseError("compositional family hash mismatch")
    candidate = CompositionalCandidate.model_validate(family.get("source_candidate"))
    if family.get("superfamily_id") != candidate.superfamily_id:
        raise CompositionalReleaseError("family source candidate identity mismatch")
    if family.get("patient_id") != candidate.patient_id:
        raise CompositionalReleaseError("family source candidate patient mismatch")

    source = family.get("source_bundle")
    if not isinstance(source, dict):
        raise CompositionalReleaseError("family source Bundle binding is malformed")
    base_path = _safe_release_file(root, source.get("base_bundle_path"))
    if _sha256_file(base_path) != source.get("base_bundle_file_sha256"):
        raise CompositionalReleaseError("family base Bundle file hash mismatch")
    base = _read_release_json(root, str(source["base_bundle_path"]))
    base_bundle_hash = content_hash(base)
    if base_bundle_hash != source.get("base_bundle_hash"):
        raise CompositionalReleaseError("family base Bundle semantic hash mismatch")
    base_closure = _strict_closure(base)
    if not base_closure.passed or source.get("strict_reference_closure") != (
        base_closure.model_dump(mode="json")
    ):
        raise CompositionalReleaseError("family base Bundle closure report mismatch")
    base_snapshot = normalize_fhir_bundle(base, dataset="synthea")
    if (
        base_snapshot.patient_id != candidate.patient_id
        or base_snapshot.snapshot_id != candidate.patient_snapshot_id
        or source.get("base_snapshot_id") != base_snapshot.snapshot_id
    ):
        raise CompositionalReleaseError("family base Bundle snapshot binding mismatch")

    routing_identity = _resource_identity(candidate.routing_evidence_id, "Encounter")
    companion_identities = tuple(
        _resource_identity(evidence_id) for evidence_id in candidate.companion_evidence_ids
    )
    routing_hash = _raw_resource_hash(base, *routing_identity)
    companion_hashes = {
        evidence_id: _raw_resource_hash(base, *identity)
        for evidence_id, identity in zip(
            candidate.companion_evidence_ids,
            companion_identities,
            strict=True,
        )
    }
    if family.get("routing_raw_hash") != routing_hash:
        raise CompositionalReleaseError("family routing raw hash mismatch")
    if family.get("companion_raw_hashes") != companion_hashes:
        raise CompositionalReleaseError("family companion raw hash mismatch")

    topology_rows = family.get("topologies")
    if not isinstance(topology_rows, dict) or set(topology_rows) != set(TOPOLOGIES):
        raise CompositionalReleaseError("family topology inventory is incomplete")
    for topology, row in topology_rows.items():
        if not isinstance(row, dict):
            raise CompositionalReleaseError("family topology contract is malformed")
        if row.get("family_id") != _topology_family_id(candidate, topology):
            raise CompositionalReleaseError("family topology ID mismatch")
        if row.get("program_hash") != content_hash(row.get("program")):
            raise CompositionalReleaseError("family topology program hash mismatch")
        if row.get("question_hash") != content_hash(row.get("question")):
            raise CompositionalReleaseError("family topology question hash mismatch")
        topology_program = TaskProgram.model_validate(row.get("program"))
        if row.get("program_signature") != program_signature(topology_program):
            raise CompositionalReleaseError("family topology program signature mismatch")
        structural_complexity = verify_topology_complexity(topology_program, topology)
        if row.get("structural_complexity") != structural_complexity.model_dump(mode="json"):
            raise CompositionalReleaseError("family topology structural complexity mismatch")

    cells = family.get("cells")
    if not isinstance(cells, list) or len(cells) != FACTORIAL_CELL_COUNT:
        raise CompositionalReleaseError("family intervention cell inventory is incomplete")
    observed_cells = Counter(
        (
            cell.get("evidence_state"),
            cell.get("companion_evidence_state"),
            cell.get("distractor_injected"),
        )
        for cell in cells
        if isinstance(cell, dict)
    )
    expected_cells = Counter(
        (state, companion, distractor)
        for state in EVIDENCE_STATES
        for companion in EVIDENCE_STATES
        for distractor in (False, True)
    )
    if observed_cells != expected_cells:
        raise CompositionalReleaseError("family intervention matrix is not exactly balanced")

    canonical_width_one = {
        (str(cell["evidence_state"]), bool(cell["distractor_injected"])): cell["task_ids"]
        for cell in cells
        if cell.get("companion_evidence_state") == "supported"
    }
    bound_task_ids: list[str] = []
    for cell in cells:
        chain_path = str(cell.get("patch_chain_path"))
        chain_file = _safe_release_file(root, chain_path)
        if _sha256_file(chain_file) != cell.get("patch_chain_file_sha256"):
            raise CompositionalReleaseError("cell patch-chain file hash mismatch")
        chain = RawFHIRPatchChain.model_validate(_read_release_json(root, chain_path))
        evidence_state = cell.get("evidence_state")
        companion_evidence_state = cell.get("companion_evidence_state")
        distractor_injected = cell.get("distractor_injected")
        declarations = chain.declared_targets
        if (
            chain.chain_hash != cell.get("patch_chain_hash")
            or chain.derived_bundle_hash != cell.get("derived_bundle_hash")
            or chain.evidence_state != cell.get("evidence_state")
            or chain.distractor_present != distractor_injected
            or len(declarations) != 2
            or declarations[0].role != "primary"
            or declarations[0].evidence_state != evidence_state
            or declarations[0].resource_type != candidate.focal_resource_type
            or declarations[0].resource_id
            != _resource_id(candidate.focal_evidence_id, candidate.focal_resource_type)
            or declarations[1].role != "companion"
            or declarations[1].evidence_state != companion_evidence_state
            or declarations[1].resource_type != companion_identities[0][0]
            or declarations[1].resource_id != companion_identities[0][1]
            or chain.target_resource_type != candidate.focal_resource_type
            or chain.target_resource_id
            != _resource_id(candidate.focal_evidence_id, candidate.focal_resource_type)
            or cell.get("cell_id")
            != _cell_id(
                candidate,
                chain,
                evidence_state=evidence_state,
                companion_evidence_state=companion_evidence_state,
                distractor_injected=distractor_injected,
            )
        ):
            raise CompositionalReleaseError("cell patch-chain binding mismatch")
        provenance_recorded_at = cell.get("provenance_recorded_at")
        if not isinstance(provenance_recorded_at, str):
            raise CompositionalReleaseError("cell has no bound Provenance recorded instant")
        (
            patch_contract,
            derived,
            validated_distractor_baselines,
        ) = _validate_compositional_patch_chain_contract(
            base,
            chain,
            expected_provenance_recorded_at=provenance_recorded_at,
        )
        if cell.get("raw_patch_contract") != patch_contract:
            raise CompositionalReleaseError("cell raw patch validation record mismatch")
        forbidden_codes = frozenset(
            {
                (candidate.focal_selector.system, candidate.focal_selector.code),
                (candidate.companion_selector.system, candidate.companion_selector.code),
            }
        )
        distractor_evidence_id = _validate_distractor_contract(
            candidate,
            chain,
            forbidden_codes=forbidden_codes,
        )
        if cell.get("distractor_evidence_id") != distractor_evidence_id:
            raise CompositionalReleaseError("cell distractor evidence binding mismatch")
        original_cell = (
            evidence_state == "supported"
            and companion_evidence_state == "supported"
            and distractor_injected is False
        )
        if cell.get("original_cell") != original_cell or (derived == base) != original_cell:
            raise CompositionalReleaseError("cell original/no-op declaration is inconsistent")
        derived_path = cell.get("derived_bundle_path")
        if derived_path is not None:
            materialized_path = _safe_release_file(root, derived_path)
            if materialized_path.read_bytes() != _canonical_bytes(derived) + b"\n":
                raise CompositionalReleaseError("materialized derived Bundle replay mismatch")
        derived_snapshot = normalize_fhir_bundle(derived, dataset="synthea")
        if cell.get("snapshot_id") != derived_snapshot.snapshot_id:
            raise CompositionalReleaseError("cell snapshot binding mismatch")
        leakage_scan = validate_agent_visible_leakage_contract(
            base,
            chain,
            normalized_snapshot=derived_snapshot,
            _validated_distractor_baselines=validated_distractor_baselines,
        )
        if cell.get("agent_visible_leakage_scan") != leakage_scan:
            raise CompositionalReleaseError("cell agent-visible leakage scan mismatch")
        if (
            cell.get("routing_raw_hash") != routing_hash
            or _raw_resource_hash(derived, *routing_identity) != routing_hash
        ):
            raise CompositionalReleaseError("cell changed routing raw evidence")
        derived_companion_hashes = {
            evidence_id: _raw_resource_hash(derived, *identity)
            for evidence_id, identity in zip(
                candidate.companion_evidence_ids,
                companion_identities,
                strict=True,
            )
        }
        if (
            cell.get("companion_base_raw_hashes") != companion_hashes
            or cell.get("companion_raw_hashes") != derived_companion_hashes
            or (
                companion_evidence_state == "supported"
                and derived_companion_hashes != companion_hashes
            )
        ):
            raise CompositionalReleaseError("cell companion state/hash binding is invalid")

        task_ids = cell.get("task_ids")
        expected_topologies = (
            set(TOPOLOGIES) if companion_evidence_state == "supported" else set(WIDTH_TOPOLOGIES)
        )
        if not isinstance(task_ids, dict) or set(task_ids) != expected_topologies:
            raise CompositionalReleaseError("cell task inventory is incomplete")
        registry = ToolRegistry(derived_snapshot)
        for topology in expected_topologies:
            task_id = task_ids[topology]
            task = task_index.get(task_id)
            if task is None:
                raise CompositionalReleaseError("cell references an unknown task")
            metadata = task.metadata.get("compositional_release", {})
            if (
                metadata.get("superfamily_id") != candidate.superfamily_id
                or metadata.get("cell_id") != cell.get("cell_id")
                or metadata.get("topology") != topology
                or metadata.get("evidence_state") != chain.evidence_state
                or metadata.get("companion_evidence_state") != companion_evidence_state
                or metadata.get("distractor_injected") != distractor_injected
                or metadata.get("original_cell") != original_cell
                or metadata.get("base_bundle_hash") != base_bundle_hash
                or metadata.get("patch_chain_hash") != chain.chain_hash
                or metadata.get("derived_bundle_hash") != chain.derived_bundle_hash
                or metadata.get("routing_raw_hash") != routing_hash
                or metadata.get("companion_raw_hashes") != companion_hashes
                or metadata.get("companion_derived_raw_hashes") != derived_companion_hashes
                or metadata.get("distractor_evidence_id") != distractor_evidence_id
                or metadata.get("source_task_hashes") != candidate.source_task_hashes
                or metadata.get("source_task_ids")
                != {
                    "focal_atomic": candidate.focal_atomic_task_id,
                    "focal_depth": candidate.focal_depth_task_id,
                    "companion": candidate.companion_task_id,
                }
                or task.program.model_dump(mode="json") != topology_rows[topology].get("program")
                or task.question != topology_rows[topology].get("question")
            ):
                raise CompositionalReleaseError("cell task-to-family binding mismatch")
            report = verify_task_artifact(task, registry)
            if not report.passed:
                raise CompositionalReleaseError(
                    f"standalone task replay failed for {task_id}: {report.errors}"
                )
            bound_task_ids.append(task_id)

        canonical = canonical_width_one.get((str(evidence_state), bool(distractor_injected)))
        if not isinstance(canonical, Mapping) or set(canonical) < set(WIDTH_ONE_TOPOLOGIES):
            raise CompositionalReleaseError("cell lacks canonical width-one lineage roots")
        if companion_evidence_state == "supported":
            if task_index[task_ids["d1w1"]].parent_task_ids != ():
                raise CompositionalReleaseError("cell atomic lineage is invalid")
            if task_index[task_ids["d2w1"]].parent_task_ids != (task_ids["d1w1"],):
                raise CompositionalReleaseError("cell depth lineage is invalid")
        if task_index[task_ids["d1w2"]].parent_task_ids != (canonical["d1w1"],):
            raise CompositionalReleaseError("cell width lineage is invalid")
        expected_mixed_parents = tuple(sorted((canonical["d2w1"], task_ids["d1w2"])))
        if task_index[task_ids["d2w2"]].parent_task_ids != expected_mixed_parents:
            raise CompositionalReleaseError("cell mixed lineage is invalid")
    return bound_task_ids


def _verify_standalone_validation(
    validation: dict[str, Any],
    *,
    tasks: list[TaskArtifact],
    family_count: int,
) -> None:
    expected_hash = content_hash(
        {key: value for key, value in validation.items() if key != "validation_hash"}
    )
    if validation.get("validation_hash") != expected_hash:
        raise CompositionalReleaseError("validation report hash mismatch")
    if (
        validation.get("passed") is not True
        or validation.get("candidate_count") != family_count
        or validation.get("cell_count") != family_count * FACTORIAL_CELL_COUNT
        or validation.get("task_count") != len(tasks)
    ):
        raise CompositionalReleaseError("validation report counts or status mismatch")
    rows = validation.get("rows")
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise CompositionalReleaseError("validation report task coverage mismatch")
    task_index = {task.task_id: task for task in tasks}
    if {row.get("task_id") for row in rows if isinstance(row, dict)} != set(task_index):
        raise CompositionalReleaseError("validation report task IDs mismatch")
    required_true = {
        "passed",
        "routing_raw_hash_preserved",
        "companion_change_matches_declared_state",
        "patch_chain_replayed",
        "changed_and_inserted_resource_allowlist",
        "provenance_targets_exact",
        "provenance_activity_exact",
        "provenance_recorded_exact",
        "provenance_agent_exact",
        "task_replayed",
        "all_tool_calls_complete",
        "intra_cell_parent_lineage",
        "agent_visible_role_tokens_absent",
        "semantic_duplicate_distractors_absent",
    }
    for row in rows:
        if not isinstance(row, dict) or any(row.get(field) is not True for field in required_true):
            raise CompositionalReleaseError("validation report contains a failed contract")
        task = task_index[row["task_id"]]
        metadata = task.metadata["compositional_release"]
        if any(
            row.get(field) != metadata.get(field)
            for field in (
                "superfamily_id",
                "cell_id",
                "topology",
                "evidence_state",
                "companion_evidence_state",
                "distractor_injected",
                "original_cell",
                "expected_branch_states",
                "expected_overall_state",
            )
        ):
            raise CompositionalReleaseError("validation report task metadata mismatch")
    metadata = [task.metadata["compositional_release"] for task in tasks]
    if (
        validation.get("counts_by_topology")
        != dict(sorted(Counter(row["topology"] for row in metadata).items()))
        or validation.get("counts_by_evidence_state")
        != dict(sorted(Counter(row["evidence_state"] for row in metadata).items()))
        or validation.get("counts_by_companion_evidence_state")
        != dict(sorted(Counter(row["companion_evidence_state"] for row in metadata).items()))
    ):
        raise CompositionalReleaseError("validation report matrix counts mismatch")
    expected_distractor_counts = {
        str(key).lower(): value
        for key, value in sorted(
            Counter(row["distractor_injected"] for row in metadata).items(),
            key=lambda item: item[0],
        )
    }
    if validation.get("counts_by_distractor_injection") != expected_distractor_counts:
        raise CompositionalReleaseError("validation report distractor-injection counts mismatch")


def _verify_standalone_splits(
    splits: dict[str, Any],
    *,
    tasks: list[TaskArtifact],
) -> None:
    assignments = splits.get("assignments")
    if not isinstance(assignments, dict) or splits.get("assignments_hash") != content_hash(
        assignments
    ):
        raise CompositionalReleaseError("split assignment hash mismatch")
    rows = assignments.get("task_assignments")
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise CompositionalReleaseError("split task coverage mismatch")
    task_index = {task.task_id: task for task in tasks}
    if {row.get("task_id") for row in rows if isinstance(row, dict)} != set(task_index):
        raise CompositionalReleaseError("split task IDs mismatch")
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in {"train", "dev", "test"}:
            raise CompositionalReleaseError("split row is malformed")
        task = task_index[row["task_id"]]
        metadata = task.metadata["compositional_release"]
        if (
            row.get("task_hash") != content_hash(task.model_dump(mode="json"))
            or row.get("patient_id") != metadata["patient_id"]
            or row.get("superfamily_id") != metadata["superfamily_id"]
            or row.get("family_id") != task.family_id
            or row.get("cell_id") != metadata["cell_id"]
            or row.get("topology") != metadata["topology"]
        ):
            raise CompositionalReleaseError("split row task binding mismatch")
    overlap = {
        axis: _split_overlap(rows, axis)
        for axis in ("patient_id", "superfamily_id", "family_id", "cell_id")
    }
    if (
        splits.get("passed") is not True
        or splits.get("overlap_audit") != overlap
        or any(values for audit in overlap.values() for values in audit.values())
    ):
        raise CompositionalReleaseError("split leakage audit mismatch")
    patient_assignments: dict[str, str] = {}
    patient_counts: Counter[str] = Counter()
    for row in rows:
        patient_id = str(row["patient_id"])
        split = str(row["split"])
        if patient_id in patient_assignments and patient_assignments[patient_id] != split:
            raise CompositionalReleaseError("split assigns one patient to multiple partitions")
        patient_assignments[patient_id] = split
        patient_counts[patient_id] += 1
    expected_units = [
        {
            "patient_id": patient_id,
            "split": patient_assignments[patient_id],
            "task_count": patient_counts[patient_id],
        }
        for patient_id in sorted(patient_assignments)
    ]
    if (
        assignments.get("unit_key") != "patient_id"
        or assignments.get("unit_assignments") != expected_units
        or splits.get("counts") != dict(sorted(Counter(row["split"] for row in rows).items()))
    ):
        raise CompositionalReleaseError("split unit assignments or counts mismatch")


def _verify_standalone_discovery(
    discovery: dict[str, Any],
    *,
    family_rows: list[dict[str, Any]],
) -> None:
    if discovery.get("discovery_hash") != content_hash(
        {key: value for key, value in discovery.items() if key != "discovery_hash"}
    ):
        raise CompositionalReleaseError("candidate discovery hash mismatch")
    matrix = discovery.get("matrix_discovery")
    if not isinstance(matrix, dict) or discovery.get("matrix_discovery_hash") != content_hash(
        matrix
    ):
        raise CompositionalReleaseError("matrix discovery source hash mismatch")
    expected_preflight = {
        "required_distractor_tier": "same_resource_different_code",
        "same_resource_type": True,
        "same_code_system": True,
        "same_encounter": True,
        "different_code": True,
        "source_records_synthesized_or_moved": False,
    }
    if discovery.get("raw_bundle_preflight") != expected_preflight:
        raise CompositionalReleaseError("candidate discovery strict distractor policy mismatch")
    family_ids = {str(row["superfamily_id"]) for row in family_rows}
    candidates = discovery.get("candidates")
    matrix_candidates = matrix.get("candidates")
    release_rejections = discovery.get("release_rejections")
    if not isinstance(matrix_candidates, list) or not isinstance(release_rejections, list):
        raise CompositionalReleaseError("candidate discovery rows are malformed")
    matrix_ids = [str(row.get("superfamily_id")) for row in matrix_candidates]
    candidate_ids = [str(row.get("superfamily_id")) for row in candidates or ()]
    rejected_ids = [str(row.get("superfamily_id")) for row in release_rejections]
    if (
        len(matrix_ids) != len(set(matrix_ids))
        or len(candidate_ids) != len(set(candidate_ids))
        or len(rejected_ids) != len(set(rejected_ids))
        or set(candidate_ids) & set(rejected_ids)
        or set(candidate_ids) | set(rejected_ids) != set(matrix_ids)
        or discovery.get("matrix_candidate_count") != len(matrix_ids)
    ):
        raise CompositionalReleaseError("candidate discovery partition mismatch")
    family_candidates = {str(row["superfamily_id"]): row["source_candidate"] for row in family_rows}
    if (
        not isinstance(candidates, list)
        or discovery.get("eligible_candidate_count") != len(family_rows)
        or {str(row.get("superfamily_id")) for row in candidates} != family_ids
        or {str(row.get("superfamily_id")): row for row in candidates if isinstance(row, dict)}
        != family_candidates
    ):
        raise CompositionalReleaseError("candidate discovery family coverage mismatch")


def _verify_standalone_source_task_replay(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    discovery: Mapping[str, Any],
    family_rows: Sequence[Mapping[str, Any]],
) -> tuple[TaskArtifact, ...]:
    """Reload the frozen P2 source and independently rerun compiler discovery.

    Hash-checking the copied JSONL proves only that bytes did not change.  This gate also
    proves that those bytes still produce the declared candidate set, source-task bindings,
    programs, questions, and topology complexity under the release compiler.
    """

    source_record = manifest.get("sources", {}).get("tasks")
    if not isinstance(source_record, Mapping):
        raise CompositionalReleaseError("manifest source-task binding is malformed")
    relative = source_record.get("release_path")
    source_path = _safe_release_file(root, relative)
    raw = source_path.read_bytes()
    if source_record.get("release_sha256") != _sha256_bytes(raw) or source_record.get(
        "task_count"
    ) != sum(bool(line.strip()) for line in raw.decode("utf-8").splitlines()):
        raise CompositionalReleaseError("frozen source-task bytes or count mismatch")
    source_tasks = tuple(
        TaskArtifact.model_validate_json(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    )
    if len({task.task_id for task in source_tasks}) != len(source_tasks):
        raise CompositionalReleaseError("frozen source tasks contain duplicate IDs")

    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CompositionalReleaseError("release manifest has an invalid discovery seed")
    recomputed = discover_compositional_candidates(source_tasks, seed=seed).model_dump(mode="json")
    declared = discovery.get("matrix_discovery")
    if recomputed != declared or discovery.get("matrix_discovery_hash") != content_hash(recomputed):
        raise CompositionalReleaseError("frozen source tasks do not reproduce candidate discovery")

    task_index = {task.task_id: task for task in source_tasks}
    for family in family_rows:
        candidate = CompositionalCandidate.model_validate(family.get("source_candidate"))
        source_bundle = family.get("source_bundle")
        if not isinstance(source_bundle, Mapping):
            raise CompositionalReleaseError("family source Bundle binding is malformed")
        base_bundle = _read_release_json(root, str(source_bundle.get("base_bundle_path")))
        base_registry = ToolRegistry(normalize_fhir_bundle(base_bundle, dataset="synthea"))
        topology_rows = family.get("topologies")
        if not isinstance(topology_rows, Mapping):
            raise CompositionalReleaseError("family topology contracts are malformed")
        for topology in TOPOLOGIES:
            rebuilt = build_compositional_topology(
                candidate,
                task_index,
                topology,
                registry=base_registry,
            )
            declared_topology = topology_rows.get(topology)
            if not isinstance(declared_topology, Mapping) or any(
                (
                    declared_topology.get("family_id") != _topology_family_id(candidate, topology),
                    declared_topology.get("program") != rebuilt.program.model_dump(mode="json"),
                    declared_topology.get("question") != rebuilt.question,
                    declared_topology.get("program_hash")
                    != content_hash(rebuilt.program.model_dump(mode="json")),
                    declared_topology.get("question_hash") != content_hash(rebuilt.question),
                    declared_topology.get("program_signature")
                    != program_signature(rebuilt.program),
                    declared_topology.get("structural_complexity")
                    != rebuilt.structural_complexity.model_dump(mode="json"),
                )
            ):
                raise CompositionalReleaseError(
                    "frozen source tasks do not reproduce a declared topology build"
                )
    return source_tasks


def _verify_standalone_source_catalog(
    catalog: dict[str, Any],
    *,
    family_rows: list[dict[str, Any]],
) -> None:
    if catalog.get("catalog_hash") != content_hash(
        {key: value for key, value in catalog.items() if key != "catalog_hash"}
    ):
        raise CompositionalReleaseError("source Bundle catalog hash mismatch")
    bindings = catalog.get("candidate_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(family_rows):
        raise CompositionalReleaseError("source Bundle catalog family coverage mismatch")
    expected = [
        {
            "superfamily_id": family["superfamily_id"],
            "patient_id": family["patient_id"],
            "catalog_name": family["source_bundle"]["catalog_name"],
            "base_bundle_path": family["source_bundle"]["base_bundle_path"],
            "base_bundle_hash": family["source_bundle"]["base_bundle_hash"],
            "base_bundle_file_sha256": family["source_bundle"]["base_bundle_file_sha256"],
            "base_snapshot_id": family["source_bundle"]["base_snapshot_id"],
        }
        for family in family_rows
    ]
    expected.sort(key=lambda row: row["superfamily_id"])
    if bindings != expected:
        raise CompositionalReleaseError("source Bundle catalog binding mismatch")


def _verify_standalone_dependencies(
    root: Path,
    dependency_manifest: Mapping[str, Any],
    *,
    patient_bundles: Sequence[Mapping[str, Any]],
) -> None:
    if dependency_manifest.get("schema_version") != (
        "patient-tool-gym.fhir-transaction-dependencies.v1"
    ) or dependency_manifest.get("dependency_manifest_hash") != content_hash(
        {
            key: value
            for key, value in dependency_manifest.items()
            if key != "dependency_manifest_hash"
        }
    ):
        raise CompositionalReleaseError("transaction dependency manifest hash is invalid")
    rows = dependency_manifest.get("dependencies")
    load_order = dependency_manifest.get("load_order")
    if (
        not isinstance(rows, list)
        or dependency_manifest.get("dependency_bundle_count") != len(rows)
        or load_order != [row.get("path") for row in rows if isinstance(row, Mapping)]
        or dependency_manifest.get("patient_bundles_load_after_dependencies") is not True
    ):
        raise CompositionalReleaseError("transaction dependency load order is malformed")
    previous_stage = 0
    bundles: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CompositionalReleaseError("transaction dependency row is malformed")
        stage = row.get("load_stage")
        relative = row.get("path")
        if not isinstance(stage, int) or stage < previous_stage:
            raise CompositionalReleaseError("transaction dependency stages are out of order")
        previous_stage = stage
        path = _safe_release_file(root, relative)
        bundle = _read_release_json(root, str(relative))
        if (
            _sha256_file(path) != row.get("file_sha256")
            or content_hash(bundle) != row.get("bundle_hash")
            or len(bundle.get("entry", [])) != row.get("entry_count")
        ):
            raise CompositionalReleaseError("transaction dependency Bundle binding mismatch")
        bundles.append(bundle)
    audit = dependency_manifest.get("resolution_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("audit_hash")
        != content_hash({key: value for key, value in audit.items() if key != "audit_hash"})
        or audit.get("passed") is not True
        or audit.get("dependency_bundle_count") != len(bundles)
        or audit.get("conditional_reference_count") != audit.get("uniquely_resolved_count")
        or audit.get("unresolved_count") != 0
        or audit.get("ambiguous_count") != 0
    ):
        raise CompositionalReleaseError("transaction dependency resolution audit is invalid")
    recomputed = _conditional_dependency_resolution_audit(
        patient_bundles,
        bundles,
        scope="unique_released_patient_base_bundles",
    )
    if dict(audit) != recomputed:
        raise CompositionalReleaseError(
            "transaction dependency resolution audit does not replay from released Bundles"
        )


def _verify_standalone_manifest_counts(
    manifest: dict[str, Any],
    *,
    tasks: list[TaskArtifact],
    family_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    splits: dict[str, Any],
    declared_files: Mapping[str, dict[str, Any]],
) -> None:
    metadata = [task.metadata["compositional_release"] for task in tasks]
    counts = manifest.get("counts")
    expected_counts = {
        "patient_count": len({row["patient_id"] for row in metadata}),
        "superfamily_count": len(family_rows),
        "topology_family_count": len(family_rows) * len(TOPOLOGIES),
        "cell_count": len(family_rows) * FACTORIAL_CELL_COUNT,
        "task_count": len(tasks),
        "base_bundle_file_count": sum(path.startswith("bundles/base/") for path in declared_files),
        "patch_chain_file_count": sum(path.startswith("patch-chains/") for path in declared_files),
        "derived_bundle_file_count": sum(
            path.startswith("bundles/derived/") for path in declared_files
        ),
        "dependency_bundle_file_count": sum(
            path.startswith("dependencies/") for path in declared_files
        ),
        "by_topology": dict(sorted(Counter(row["topology"] for row in metadata).items())),
        "by_evidence_state": dict(
            sorted(Counter(row["evidence_state"] for row in metadata).items())
        ),
        "by_companion_evidence_state": dict(
            sorted(Counter(row["companion_evidence_state"] for row in metadata).items())
        ),
        "by_distractor_injection": {
            str(key).lower(): value
            for key, value in sorted(
                Counter(row["distractor_injected"] for row in metadata).items()
            )
        },
    }
    if counts != expected_counts:
        raise CompositionalReleaseError("release manifest counts mismatch")
    if manifest.get("validation", {}).get("validation_hash") != validation.get("validation_hash"):
        raise CompositionalReleaseError("manifest validation hash binding mismatch")
    if manifest.get("splits", {}).get("assignments_hash") != splits.get("assignments_hash"):
        raise CompositionalReleaseError("manifest split hash binding mismatch")
    source_tasks = manifest.get("sources", {}).get("tasks", {})
    source_bundles = manifest.get("sources", {}).get("bundles", {})
    for source in (source_tasks, source_bundles):
        release_path = source.get("release_path")
        if (
            not isinstance(release_path, str)
            or release_path not in declared_files
            or source.get("release_sha256") != declared_files[release_path]["sha256"]
        ):
            raise CompositionalReleaseError("manifest source artifact binding mismatch")


def _safe_release_file(root: Path, relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise CompositionalReleaseError("release artifact path is unsafe")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise CompositionalReleaseError(f"release artifact is missing: {relative}")
    return path


def _read_release_json(root: Path, relative: str) -> dict[str, Any]:
    path = _safe_release_file(root, relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompositionalReleaseError(f"release artifact is not valid JSON: {relative}") from exc
    if not isinstance(value, dict):
        raise CompositionalReleaseError(f"release JSON artifact is not an object: {relative}")
    return value


def _read_release_jsonl(root: Path, relative: str) -> list[dict[str, Any]]:
    path = _safe_release_file(root, relative)
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CompositionalReleaseError(
                    f"release JSONL artifact contains a non-object: {relative}"
                )
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompositionalReleaseError(f"release artifact is not valid JSONL: {relative}") from exc
    return rows


def _is_plain_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _build_generation_provenance(
    supplied: Mapping[str, Any] | None,
    *,
    seed: int,
    materialize_derived_bundles: bool,
) -> dict[str, Any]:
    required = (
        "version",
        "generator_commit",
        "seed",
        "reference_date",
        "end_date",
        "command",
        "source_generation_completed_at",
    )
    upstream = dict(supplied or {})
    unknown = [
        key for key in required if not isinstance(upstream.get(key), str) or not upstream[key]
    ]
    for date_key in ("reference_date", "end_date"):
        value = upstream.get(date_key)
        if isinstance(value, str) and value:
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise CompositionalReleaseError(
                    f"Synthea {date_key} must be an ISO calendar date"
                ) from exc
    recorded_at = upstream.get("source_generation_completed_at")
    if isinstance(recorded_at, str) and recorded_at:
        try:
            parsed_recorded = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CompositionalReleaseError(
                "Synthea source_generation_completed_at must be an ISO instant"
            ) from exc
        if parsed_recorded.tzinfo is None:
            raise CompositionalReleaseError(
                "Synthea source_generation_completed_at must include a timezone"
            )
    if not unknown and date.fromisoformat(upstream["end_date"]) < date.fromisoformat(
        upstream["reference_date"]
    ):
        raise CompositionalReleaseError("Synthea end_date cannot precede reference_date")
    upstream = {
        "name": "Synthea",
        "version": upstream.get("version"),
        "generator_commit": upstream.get("generator_commit"),
        "seed": upstream.get("seed"),
        "reference_date": upstream.get("reference_date"),
        "end_date": upstream.get("end_date"),
        "command": upstream.get("command"),
        "source_generation_completed_at": upstream.get("source_generation_completed_at"),
        "cohort_manifest_sha256": upstream.get("cohort_manifest_sha256"),
    }

    package = Path(__file__).resolve().parent
    source_names = (
        "compositional_matrix.py",
        "compositional_release.py",
        "medical_interventions.py",
        "raw_fhir.py",
        "fhir.py",
        "program.py",
        "tools.py",
        "task.py",
        "verifier.py",
    )
    source_files = [
        {
            "path": f"src/patient_tool_gym/{name}",
            "sha256": _sha256_file(package / name),
        }
        for name in source_names
    ]
    config = {
        "release_seed": seed,
        "materialize_derived_bundles": materialize_derived_bundles,
        "topologies": list(TOPOLOGIES),
        "evidence_states": list(EVIDENCE_STATES),
        "companion_state_factorial": "all_four_states_for_width_topologies",
        "width_one_companion_state": "supported",
        "distractor_injected": [False, True],
        "candidate_selection": "all_medically_representable_candidates",
    }
    payload = {
        "schema_version": "patient-tool-gym.compositional-generation-provenance.v1",
        "complete": not unknown,
        "missing_upstream_fields": unknown,
        "upstream": upstream,
        "compiler": {
            "builder_version": COMPOSITIONAL_RELEASE_BUILDER_VERSION,
            "source_files": source_files,
            "source_tree_hash": content_hash(source_files),
            "config": config,
            "config_hash": content_hash(config),
        },
    }
    payload["provenance_hash"] = content_hash(payload)
    return payload


def _verify_generation_provenance(manifest: Mapping[str, Any]) -> None:
    provenance = manifest.get("generation_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("schema_version") != (
        "patient-tool-gym.compositional-generation-provenance.v1"
    ):
        raise CompositionalReleaseError("generation provenance schema is unsupported")
    if provenance.get("provenance_hash") != content_hash(
        {key: value for key, value in provenance.items() if key != "provenance_hash"}
    ):
        raise CompositionalReleaseError("generation provenance hash mismatch")
    missing = provenance.get("missing_upstream_fields")
    if not isinstance(missing, list) or provenance.get("complete") != (not missing):
        raise CompositionalReleaseError("generation provenance completeness is inconsistent")
    upstream = provenance.get("upstream")
    compiler = provenance.get("compiler")
    if not isinstance(upstream, Mapping) or not isinstance(compiler, Mapping):
        raise CompositionalReleaseError("generation provenance sections are malformed")
    if upstream.get("name") != "Synthea":
        raise CompositionalReleaseError("generation provenance names an unsupported source")
    source_files = compiler.get("source_files")
    config = compiler.get("config")
    if (
        not isinstance(source_files, list)
        or not all(
            isinstance(row, Mapping)
            and isinstance(row.get("path"), str)
            and _is_plain_sha256(row.get("sha256"))
            for row in source_files
        )
        or compiler.get("source_tree_hash") != content_hash(source_files)
        or not isinstance(config, Mapping)
        or compiler.get("config_hash") != content_hash(config)
        or config.get("release_seed") != manifest.get("seed")
        or config.get("topologies") != list(TOPOLOGIES)
        or config.get("evidence_states") != list(EVIDENCE_STATES)
    ):
        raise CompositionalReleaseError("generation compiler provenance is inconsistent")


def _manifest(
    *,
    staging: Path,
    source_tasks: dict[str, Any],
    source_bundles: dict[str, Any],
    discovery: dict[str, Any],
    tasks: list[TaskArtifact],
    families: list[dict[str, Any]],
    validation: dict[str, Any],
    splits: dict[str, Any],
    seed: int,
    materialize_derived_bundles: bool,
    generation_provenance: dict[str, Any],
) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(staging).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in staging.rglob("*") if item.is_file())
    ]
    metadata = [task.metadata["compositional_release"] for task in tasks]
    payload = {
        "schema_version": COMPOSITIONAL_RELEASE_SCHEMA_VERSION,
        "builder_version": COMPOSITIONAL_RELEASE_BUILDER_VERSION,
        "access_classification": "private_evaluator_and_audit",
        "seed": seed,
        "generation_provenance": generation_provenance,
        "selection_policy": {
            "eligible_family_policy": "all_discovered",
            "quota": None,
            "candidate_count": len(families),
            "rejection_count": (
                len(discovery["matrix_discovery"]["rejections"])
                + len(discovery["release_rejections"])
            ),
            "discovery_hash": discovery["discovery_hash"],
        },
        "storage_policy": {
            "base_bundles_content_addressed": True,
            "patch_chains_materialized": True,
            "derived_bundles_materialized": materialize_derived_bundles,
            "redundant_derived_bundles_omitted_by_default": True,
        },
        "sources": {
            "tasks": source_tasks,
            "bundles": source_bundles,
        },
        "counts": {
            "patient_count": len({row["patient_id"] for row in metadata}),
            "superfamily_count": len(families),
            "topology_family_count": len(families) * len(TOPOLOGIES),
            "cell_count": len(families) * FACTORIAL_CELL_COUNT,
            "task_count": len(tasks),
            "base_bundle_file_count": sum(row["path"].startswith("bundles/base/") for row in files),
            "patch_chain_file_count": sum(row["path"].startswith("patch-chains/") for row in files),
            "derived_bundle_file_count": sum(
                row["path"].startswith("bundles/derived/") for row in files
            ),
            "dependency_bundle_file_count": sum(
                row["path"].startswith("dependencies/") for row in files
            ),
            "by_topology": dict(sorted(Counter(row["topology"] for row in metadata).items())),
            "by_evidence_state": dict(
                sorted(Counter(row["evidence_state"] for row in metadata).items())
            ),
            "by_companion_evidence_state": dict(
                sorted(Counter(row["companion_evidence_state"] for row in metadata).items())
            ),
            "by_distractor_injection": {
                str(key).lower(): value
                for key, value in sorted(
                    Counter(row["distractor_injected"] for row in metadata).items()
                )
            },
        },
        "matrix": {
            "topologies": list(TOPOLOGIES),
            "primary_evidence_states": list(EVIDENCE_STATES),
            "companion_evidence_states": list(EVIDENCE_STATES),
            "distractor_injected": [False, True],
            "chart_cells_per_superfamily": FACTORIAL_CELL_COUNT,
            "tasks_per_superfamily": TASKS_PER_SUPERFAMILY,
            "width_one_policy": "compiled_only_when_companion_is_supported",
            "width_two_policy": "full_primary_x_companion_state_factorial",
        },
        "validation": {
            "passed": validation["passed"],
            "task_count": validation["task_count"],
            "validation_hash": validation["validation_hash"],
            "strict_reference_closure": True,
            "patch_chain_replay": True,
            "changed_and_inserted_resource_allowlist": True,
            "provenance_targets_activity_time_and_agent_exact": True,
            "task_metadata_contract_hash_recomputed": True,
            "complete_task_replay": True,
            "routing_raw_hash_preservation": True,
            "companion_change_bound_to_declared_state": True,
            "agent_visible_role_leakage_scan": True,
            "semantic_duplicate_distractor_scan": True,
            "original_supported_cell_byte_identical": True,
        },
        "limitations": {
            "procedure_focal_interventions_excluded": True,
            "procedure_exclusion_reason": "REFUTED_DATE_NOT_REPRESENTABLE",
            "procedure_companion_interventions_excluded": True,
            "planned_future_extension": (
                "encounter-date selection with guarded nonperformance semantics"
            ),
        },
        "splits": {
            "passed": splits["passed"],
            "policy": splits["policy"],
            "assignments_hash": splits["assignments_hash"],
        },
        "files": files,
    }
    payload["release_hash"] = content_hash(payload)
    return payload


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    path.write_bytes(raw)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())
