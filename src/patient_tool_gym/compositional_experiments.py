"""Prepare content-bound runtime inputs for compositional benchmark experiments.

The compositional release intentionally stores raw FHIR Bundles and patch chains as
its source of truth.  ``run-episodes`` intentionally consumes normalized
``PatientSnapshot`` files.  This module bridges those two representations without
modifying the frozen release, and records enough hashes to independently audit the
bridge.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from patient_tool_gym.compositional_matrix import TOPOLOGIES
from patient_tool_gym.compositional_release import (
    EVIDENCE_STATES,
    FACTORIAL_CELL_COUNT,
    TASKS_PER_SUPERFAMILY,
    WIDTH_TOPOLOGIES,
    VerifiedCompositionalRelease,
    verify_compositional_release,
)
from patient_tool_gym.evidence import load_snapshot
from patient_tool_gym.fhir import normalize_fhir_bundle
from patient_tool_gym.medical_interventions import (
    RawFHIRPatchChain,
    apply_raw_fhir_patch_chain,
)
from patient_tool_gym.program import content_hash
from patient_tool_gym.task import TaskArtifact

COMPOSITIONAL_EXPERIMENT_INPUT_SCHEMA_VERSION = (
    "patient-tool-gym.compositional-experiment-inputs.v2"
)
COMPOSITIONAL_EXPERIMENT_BINDING_SCHEMA_VERSION = (
    "patient-tool-gym.compositional-experiment-snapshot-binding.v2"
)
COMPOSITIONAL_EXPERIMENT_BUILDER_VERSION = (
    "patient-tool-gym.compositional-experiment-input-builder.v2"
)


class CompositionalExperimentPreparationError(ValueError):
    """Raised when experiment inputs cannot be prepared or independently verified."""


def prepare_compositional_experiment_inputs(
    release_directory: Path,
    output_directory: Path,
    *,
    seed: int = 20260711,
    release_context: VerifiedCompositionalRelease | None = None,
) -> dict[str, Any]:
    """Normalize every chart cell and prepare full, split, and canary task cohorts.

    The output is built in a temporary sibling directory and published with one
    rename.  The source release is verified before any output is published.  The
    resulting directory is deliberately separate from the frozen release because
    that release has a closed file inventory.
    """

    release_root = Path(release_directory).resolve()
    output_root = Path(output_directory)
    if output_root.exists():
        raise CompositionalExperimentPreparationError(
            "experiment input output already exists; use a fresh versioned path"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CompositionalExperimentPreparationError("seed must be an integer")

    release_verification = (
        verify_compositional_release(release_root)
        if release_context is None
        else release_context.assert_current(release_root)
    )
    source_manifest = _read_json(release_root / "manifest.json")
    source_manifest_sha256 = _sha256_file(release_root / "manifest.json")
    source_release_hash = source_manifest.get("release_hash")
    if source_release_hash != release_verification.get("release_hash"):
        raise CompositionalExperimentPreparationError(
            "source release verification returned a different release hash"
        )

    task_rows = _read_jsonl(release_root / "tasks.jsonl")
    tasks = [TaskArtifact.model_validate(row) for row in task_rows]
    task_index = {task.task_id: task for task in tasks}
    task_row_index = {str(row.get("task_id")): row for row in task_rows}
    if len(task_index) != len(tasks) or len(task_row_index) != len(task_rows):
        raise CompositionalExperimentPreparationError("source tasks contain duplicate IDs")
    family_rows = _read_jsonl(release_root / "families.jsonl")
    split_payload = _read_json(release_root / "splits.json")
    split_assignments = _validated_split_assignments(split_payload, task_index)

    datasets = {task.provenance.dataset for task in tasks}
    if len(datasets) != 1:
        raise CompositionalExperimentPreparationError(
            "compositional experiment preparation requires one source dataset"
        )
    dataset = next(iter(datasets))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}.",
        suffix=".tmp",
    ) as temporary_name:
        staging = Path(temporary_name) / "experiment-inputs"
        staging.mkdir()
        bindings = _materialize_snapshot_bindings(
            release_root=release_root,
            staging=staging,
            family_rows=family_rows,
            task_index=task_index,
            dataset=dataset,
            source_release_hash=str(source_release_hash),
        )
        _write_jsonl(staging / "snapshot-bindings.jsonl", bindings)

        ordered_task_ids = [task.task_id for task in tasks]
        cohort_ids: dict[str, list[str]] = {"all": ordered_task_ids}
        for split in ("train", "dev", "test"):
            cohort_ids[split] = [
                task_id for task_id in ordered_task_ids if split_assignments[task_id] == split
            ]

        canary_assignments = select_balanced_compositional_canary(
            family_rows,
            task_index,
            seed=seed,
        )
        cohort_ids["canary-factorial-80"] = [row["task_id"] for row in canary_assignments]

        cohort_records: dict[str, dict[str, Any]] = {}
        for name, task_ids in cohort_ids.items():
            relative_path = f"tasks/{name}.jsonl"
            rows = [task_row_index[task_id] for task_id in task_ids]
            _write_jsonl(staging / relative_path, rows)
            cohort_records[name] = _cohort_record(
                staging / relative_path,
                relative_path=relative_path,
                task_ids=task_ids,
                task_index=task_index,
            )

        files = _file_inventory(staging)
        payload: dict[str, Any] = {
            "schema_version": COMPOSITIONAL_EXPERIMENT_INPUT_SCHEMA_VERSION,
            "builder_version": COMPOSITIONAL_EXPERIMENT_BUILDER_VERSION,
            "seed": seed,
            "source_release": {
                "schema_version": source_manifest.get("schema_version"),
                "release_hash": source_release_hash,
                "manifest_sha256": source_manifest_sha256,
                "split_assignments_hash": split_payload.get("assignments_hash"),
                "task_jsonl_sha256": _sha256_file(release_root / "tasks.jsonl"),
                "verified": True,
            },
            "counts": {
                "patient_count": len({row["patient_id"] for row in bindings}),
                "resource_domain_count": len({row["focal_resource_type"] for row in bindings}),
                "snapshot_count": len(bindings),
                "task_count": len(tasks),
                "canary_task_count": len(canary_assignments),
                "split_task_counts": {
                    split: len(cohort_ids[split]) for split in ("train", "dev", "test")
                },
            },
            "matrix": {
                "topologies": list(TOPOLOGIES),
                "primary_evidence_states": list(EVIDENCE_STATES),
                "companion_evidence_states": list(EVIDENCE_STATES),
                "distractor_injected": [False, True],
                "chart_cells_per_superfamily": FACTORIAL_CELL_COUNT,
                "tasks_per_superfamily": TASKS_PER_SUPERFAMILY,
                "expected_canary_stratum_count": TASKS_PER_SUPERFAMILY,
            },
            "snapshot_bindings": {
                "path": "snapshot-bindings.jsonl",
                "sha256": _sha256_file(staging / "snapshot-bindings.jsonl"),
                "binding_set_hash": content_hash([row["binding_hash"] for row in bindings]),
                "all_task_snapshot_ids_match": True,
                "all_bundle_hashes_match": True,
                "snapshot_ids_unique": True,
            },
            "cohorts": cohort_records,
            "canary": {
                "design": (
                    "one task per admitted topology x primary_state x companion_state x "
                    "distractor_injected stratum; "
                    "patient use balanced before reuse and resource domains distributed"
                ),
                "assignment_hash": content_hash(canary_assignments),
                "assignments": canary_assignments,
                "patient_counts": dict(
                    sorted(Counter(row["patient_id"] for row in canary_assignments).items())
                ),
                "resource_domain_counts": dict(
                    sorted(
                        Counter(row["focal_resource_type"] for row in canary_assignments).items()
                    )
                ),
            },
            "integrity": {
                "source_release_verified": True,
                "derived_bundle_hashes_recomputed": True,
                "patch_chains_replayed": True,
                "snapshots_reloaded": True,
                "task_bindings_complete": True,
                "split_partition_complete_and_disjoint": True,
                "closed_file_inventory": True,
            },
            "files": files,
        }
        payload["experiment_input_hash"] = content_hash(payload)
        _write_json(staging / "manifest.json", payload)
        staged_verification = verify_compositional_experiment_inputs(staging)
        if not staged_verification["passed"]:
            raise CompositionalExperimentPreparationError(
                "staged experiment inputs failed independent verification"
            )
        staging.replace(output_root)
    return payload


def select_balanced_compositional_canary(
    family_rows: Sequence[Mapping[str, Any]],
    task_index: Mapping[str, TaskArtifact],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Select one task from all 80 admitted topology-factorial strata.

    Selection is deterministic.  Greedy ranking first minimizes patient reuse, then
    minimizes use of a resource domain relative to the number of source families in
    that domain.  A seeded cryptographic digest breaks all remaining ties.  Thus the
    80-task reference cohort exercises all admitted cells while spreading requests across
    patients and the available focal-resource domains.
    """

    if not family_rows:
        raise CompositionalExperimentPreparationError(
            "cannot select a canary from an empty family collection"
        )
    families: list[dict[str, str]] = []
    cell_index: dict[tuple[str, str, str, bool], Mapping[str, Any]] = {}
    seen_patients: set[str] = set()
    for raw_family in family_rows:
        superfamily_id = _required_string(raw_family, "superfamily_id")
        patient_id = _required_string(raw_family, "patient_id")
        if patient_id in seen_patients:
            raise CompositionalExperimentPreparationError(
                "canary balancing requires one superfamily per patient"
            )
        seen_patients.add(patient_id)
        source_candidate = raw_family.get("source_candidate")
        if not isinstance(source_candidate, Mapping):
            raise CompositionalExperimentPreparationError(
                f"family {superfamily_id} has no source candidate"
            )
        resource_type = _required_string(source_candidate, "focal_resource_type")
        families.append(
            {
                "superfamily_id": superfamily_id,
                "patient_id": patient_id,
                "focal_resource_type": resource_type,
            }
        )
        cells = raw_family.get("cells")
        if not isinstance(cells, list):
            raise CompositionalExperimentPreparationError(
                f"family {superfamily_id} has no chart cells"
            )
        for cell in cells:
            if not isinstance(cell, Mapping):
                raise CompositionalExperimentPreparationError("family has a malformed cell")
            state = _required_string(cell, "evidence_state")
            companion_state = _required_string(cell, "companion_evidence_state")
            distractor = cell.get("distractor_injected")
            if not isinstance(distractor, bool):
                raise CompositionalExperimentPreparationError(
                    f"family {superfamily_id} has a malformed distractor flag"
                )
            key = (superfamily_id, state, companion_state, distractor)
            if key in cell_index:
                raise CompositionalExperimentPreparationError(
                    f"family {superfamily_id} duplicates a canary chart cell"
                )
            cell_index[key] = cell

    domain_capacity = Counter(row["focal_resource_type"] for row in families)
    patient_usage: Counter[str] = Counter()
    domain_usage: Counter[str] = Counter()
    patient_topology_usage: Counter[tuple[str, str]] = Counter()
    patient_state_usage: Counter[tuple[str, str]] = Counter()
    patient_companion_state_usage: Counter[tuple[str, str]] = Counter()
    patient_distractor_usage: Counter[tuple[str, bool]] = Counter()

    strata = [
        (topology, state, companion_state, distractor)
        for topology in TOPOLOGIES
        for state in EVIDENCE_STATES
        for companion_state in (EVIDENCE_STATES if topology in WIDTH_TOPOLOGIES else ("supported",))
        for distractor in (False, True)
    ]
    assignment_order = sorted(
        strata,
        key=lambda stratum: _stable_digest(seed, "stratum", *map(str, stratum)),
    )
    selected_by_stratum: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    for topology, state, companion_state, distractor in assignment_order:
        available = [
            family
            for family in families
            if (
                family["superfamily_id"],
                state,
                companion_state,
                distractor,
            )
            in cell_index
        ]
        if not available:
            raise CompositionalExperimentPreparationError(
                "no family supplies canary stratum "
                f"{topology}/{state}/{companion_state}/{distractor}"
            )

        def rank(
            family: Mapping[str, str],
            topology: str = topology,
            state: str = state,
            companion_state: str = companion_state,
            distractor: bool = distractor,
        ) -> tuple[Any, ...]:
            patient_id = family["patient_id"]
            resource_type = family["focal_resource_type"]
            # Cross multiplication avoids floating-point ranking.  All family counts
            # are positive because ``resource_type`` came from ``families``.
            return (
                patient_usage[patient_id],
                Fraction(domain_usage[resource_type], domain_capacity[resource_type]),
                patient_topology_usage[(patient_id, topology)],
                patient_state_usage[(patient_id, state)],
                patient_companion_state_usage[(patient_id, companion_state)],
                patient_distractor_usage[(patient_id, distractor)],
                _stable_digest(
                    seed,
                    "assignment",
                    topology,
                    state,
                    companion_state,
                    str(distractor),
                    family["superfamily_id"],
                ),
                family["superfamily_id"],
            )

        chosen = min(available, key=rank)
        cell = cell_index[(chosen["superfamily_id"], state, companion_state, distractor)]
        task_ids = cell.get("task_ids")
        task_id = task_ids.get(topology) if isinstance(task_ids, Mapping) else None
        if not isinstance(task_id, str) or task_id not in task_index:
            raise CompositionalExperimentPreparationError(
                "canary stratum has no bound task: "
                f"{topology}/{state}/{companion_state}/{distractor}"
            )
        task = task_index[task_id]
        metadata = task.metadata.get("compositional_release")
        if not isinstance(metadata, dict) or (
            metadata.get("topology") != topology
            or metadata.get("evidence_state") != state
            or metadata.get("companion_evidence_state") != companion_state
            or metadata.get("distractor_injected") is not distractor
            or metadata.get("superfamily_id") != chosen["superfamily_id"]
        ):
            raise CompositionalExperimentPreparationError(
                f"task {task_id} does not match its requested canary stratum"
            )
        selected_by_stratum[(topology, state, companion_state, distractor)] = {
            "topology": topology,
            "evidence_state": state,
            "companion_evidence_state": companion_state,
            "distractor_injected": distractor,
            "task_id": task_id,
            "task_hash": content_hash(task.model_dump(mode="json")),
            "patient_id": chosen["patient_id"],
            "superfamily_id": chosen["superfamily_id"],
            "cell_id": _required_string(cell, "cell_id"),
            "patient_snapshot_id": task.patient_snapshot_id,
            "focal_resource_type": chosen["focal_resource_type"],
        }
        patient_usage[chosen["patient_id"]] += 1
        domain_usage[chosen["focal_resource_type"]] += 1
        patient_topology_usage[(chosen["patient_id"], topology)] += 1
        patient_state_usage[(chosen["patient_id"], state)] += 1
        patient_companion_state_usage[(chosen["patient_id"], companion_state)] += 1
        patient_distractor_usage[(chosen["patient_id"], distractor)] += 1

    return [selected_by_stratum[stratum] for stratum in strata]


def verify_compositional_experiment_inputs(
    input_directory: Path,
    *,
    release_directory: Path | None = None,
    release_context: VerifiedCompositionalRelease | None = None,
) -> dict[str, Any]:
    """Independently verify the closed experiment-input artifact."""

    root = Path(input_directory).resolve()
    if not root.is_dir():
        raise CompositionalExperimentPreparationError("experiment input directory does not exist")
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != COMPOSITIONAL_EXPERIMENT_INPUT_SCHEMA_VERSION:
        raise CompositionalExperimentPreparationError(
            "experiment input manifest schema is unsupported"
        )
    expected_manifest_hash = content_hash(
        {key: value for key, value in manifest.items() if key != "experiment_input_hash"}
    )
    if manifest.get("experiment_input_hash") != expected_manifest_hash:
        raise CompositionalExperimentPreparationError("experiment input manifest hash mismatch")
    _verify_file_inventory(root, manifest)

    if release_directory is None and release_context is not None:
        raise CompositionalExperimentPreparationError(
            "a release context requires a release directory"
        )
    if release_directory is not None:
        release_root = Path(release_directory).resolve()
        release_report = (
            verify_compositional_release(release_root)
            if release_context is None
            else release_context.assert_current(release_root)
        )
        source = manifest.get("source_release")
        if not isinstance(source, dict) or (
            source.get("release_hash") != release_report.get("release_hash")
            or source.get("manifest_sha256") != _sha256_file(release_root / "manifest.json")
            or source.get("task_jsonl_sha256") != _sha256_file(release_root / "tasks.jsonl")
        ):
            raise CompositionalExperimentPreparationError(
                "experiment inputs are bound to a different source release"
            )

    binding_rows = _read_jsonl(root / "snapshot-bindings.jsonl")
    snapshot_ids: set[str] = set()
    snapshot_paths: set[str] = set()
    bound_tasks: dict[str, str] = {}
    for row in binding_rows:
        expected_binding_hash = content_hash(
            {key: value for key, value in row.items() if key != "binding_hash"}
        )
        if (
            row.get("schema_version") != COMPOSITIONAL_EXPERIMENT_BINDING_SCHEMA_VERSION
            or row.get("binding_hash") != expected_binding_hash
        ):
            raise CompositionalExperimentPreparationError(
                "snapshot binding hash or schema is invalid"
            )
        snapshot_record = row.get("snapshot")
        if not isinstance(snapshot_record, dict):
            raise CompositionalExperimentPreparationError("snapshot binding has no snapshot")
        relative_path = snapshot_record.get("path")
        snapshot_id = snapshot_record.get("snapshot_id")
        if not isinstance(relative_path, str) or not isinstance(snapshot_id, str):
            raise CompositionalExperimentPreparationError(
                "snapshot binding has an invalid path or ID"
            )
        if relative_path in snapshot_paths or snapshot_id in snapshot_ids:
            raise CompositionalExperimentPreparationError(
                "snapshot bindings contain a duplicate path or snapshot ID"
            )
        snapshot_paths.add(relative_path)
        snapshot_ids.add(snapshot_id)
        snapshot_path = _safe_relative_file(root, relative_path)
        if snapshot_record.get("file_sha256") != _sha256_file(snapshot_path):
            raise CompositionalExperimentPreparationError("snapshot file hash mismatch")
        snapshot = load_snapshot(snapshot_path)
        if (
            snapshot.snapshot_id != snapshot_id
            or snapshot.patient_id != row.get("patient_id")
            or len(snapshot.records) != snapshot_record.get("record_count")
        ):
            raise CompositionalExperimentPreparationError(
                "normalized snapshot does not match its binding"
            )
        task_records = row.get("tasks")
        if not isinstance(task_records, list) or len(task_records) not in {
            len(TOPOLOGIES),
            len(WIDTH_TOPOLOGIES),
        }:
            raise CompositionalExperimentPreparationError(
                "snapshot binding does not contain every topology task"
            )
        for task_record in task_records:
            if not isinstance(task_record, dict) or not isinstance(task_record.get("task_id"), str):
                raise CompositionalExperimentPreparationError(
                    "snapshot binding has a malformed task record"
                )
            task_id = task_record["task_id"]
            if task_id in bound_tasks:
                raise CompositionalExperimentPreparationError(
                    "a task is bound to more than one snapshot"
                )
            bound_tasks[task_id] = snapshot_id

    cohort_manifest = manifest.get("cohorts")
    if not isinstance(cohort_manifest, dict):
        raise CompositionalExperimentPreparationError("manifest has no cohort bindings")
    cohort_tasks: dict[str, list[TaskArtifact]] = {}
    for name in ("all", "train", "dev", "test", "canary-factorial-80"):
        record = cohort_manifest.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise CompositionalExperimentPreparationError(f"manifest omits {name} cohort")
        cohort_path = _safe_relative_file(root, record["path"])
        rows = _read_jsonl(cohort_path)
        cohort = [TaskArtifact.model_validate(row) for row in rows]
        ids = [task.task_id for task in cohort]
        hashes = [content_hash(task.model_dump(mode="json")) for task in cohort]
        if (
            len(set(ids)) != len(ids)
            or record.get("task_count") != len(ids)
            or record.get("file_sha256") != _sha256_file(cohort_path)
            or record.get("task_ids_hash") != content_hash(ids)
            or record.get("task_bindings_hash")
            != content_hash(
                [
                    {"task_id": task_id, "task_hash": task_hash}
                    for task_id, task_hash in zip(ids, hashes, strict=True)
                ]
            )
        ):
            raise CompositionalExperimentPreparationError(
                f"{name} cohort hash or cardinality is invalid"
            )
        cohort_tasks[name] = cohort

    all_by_id = {task.task_id: task for task in cohort_tasks["all"]}
    if len(all_by_id) != len(cohort_tasks["all"]) or set(bound_tasks) != set(all_by_id):
        raise CompositionalExperimentPreparationError(
            "snapshot bindings do not cover the complete task cohort exactly once"
        )
    for task_id, snapshot_id in bound_tasks.items():
        task = all_by_id[task_id]
        if task.patient_snapshot_id != snapshot_id:
            raise CompositionalExperimentPreparationError(
                f"task {task_id} is bound to the wrong normalized snapshot"
            )
    split_sets = {
        name: {task.task_id for task in cohort_tasks[name]} for name in ("train", "dev", "test")
    }
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ) or set().union(*split_sets.values()) != set(all_by_id):
        raise CompositionalExperimentPreparationError(
            "train/dev/test cohorts are not a complete disjoint partition"
        )

    canary = cohort_tasks["canary-factorial-80"]
    observed_strata = {
        (
            task.metadata["compositional_release"]["topology"],
            task.metadata["compositional_release"]["evidence_state"],
            task.metadata["compositional_release"]["companion_evidence_state"],
            task.metadata["compositional_release"]["distractor_injected"],
        )
        for task in canary
    }
    expected_strata = {
        (topology, state, companion_state, distractor)
        for topology in TOPOLOGIES
        for state in EVIDENCE_STATES
        for companion_state in (EVIDENCE_STATES if topology in WIDTH_TOPOLOGIES else ("supported",))
        for distractor in (False, True)
    }
    canary_manifest = manifest.get("canary")
    assignments = canary_manifest.get("assignments") if isinstance(canary_manifest, dict) else None
    if (
        len(canary) != len(expected_strata)
        or observed_strata != expected_strata
        or not isinstance(assignments, list)
        or canary_manifest.get("assignment_hash") != content_hash(assignments)
        or [task.task_id for task in canary]
        != [row.get("task_id") for row in assignments if isinstance(row, dict)]
    ):
        raise CompositionalExperimentPreparationError(
            "canary does not cover the complete factorial matrix exactly once"
        )

    counts = manifest.get("counts")
    if not isinstance(counts, dict) or (
        counts.get("snapshot_count") != len(binding_rows)
        or counts.get("task_count") != len(all_by_id)
        or counts.get("canary_task_count") != len(canary)
    ):
        raise CompositionalExperimentPreparationError("manifest counts are inconsistent")

    return {
        "schema_version": "patient-tool-gym.compositional-experiment-input-verification.v2",
        "passed": True,
        "experiment_input_hash": manifest["experiment_input_hash"],
        "source_release_hash": manifest["source_release"]["release_hash"],
        "snapshot_count": len(binding_rows),
        "task_count": len(all_by_id),
        "canary_task_count": len(canary),
        "split_task_counts": {name: len(tasks) for name, tasks in split_sets.items()},
        "snapshot_ids_unique": True,
        "task_snapshot_bindings_complete": True,
        "closed_file_inventory": True,
    }


def _materialize_snapshot_bindings(
    *,
    release_root: Path,
    staging: Path,
    family_rows: Sequence[Mapping[str, Any]],
    task_index: Mapping[str, TaskArtifact],
    dataset: str,
    source_release_hash: str,
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    seen_snapshot_ids: set[str] = set()
    seen_snapshot_paths: set[str] = set()
    seen_task_ids: set[str] = set()
    for family in sorted(family_rows, key=lambda row: str(row.get("superfamily_id"))):
        superfamily_id = _required_string(family, "superfamily_id")
        patient_id = _required_string(family, "patient_id")
        source_candidate = family.get("source_candidate")
        source_bundle = family.get("source_bundle")
        cells = family.get("cells")
        if (
            not isinstance(source_candidate, Mapping)
            or not isinstance(source_bundle, Mapping)
            or not isinstance(cells, list)
        ):
            raise CompositionalExperimentPreparationError(
                f"family {superfamily_id} is missing release bindings"
            )
        focal_resource_type = _required_string(source_candidate, "focal_resource_type")
        base_relative = _required_string(source_bundle, "base_bundle_path")
        base_path = _safe_relative_file(release_root, base_relative)
        base = _read_json(base_path)
        base_hash = content_hash(base)
        if base_hash != source_bundle.get("base_bundle_hash"):
            raise CompositionalExperimentPreparationError(
                f"family {superfamily_id} base Bundle hash mismatch"
            )

        cell_order = {state: index for index, state in enumerate(EVIDENCE_STATES)}
        ordered_cells = sorted(
            cells,
            key=lambda row: (
                cell_order.get(str(row.get("evidence_state")), len(cell_order)),
                cell_order.get(str(row.get("companion_evidence_state")), len(cell_order)),
                bool(row.get("distractor_injected")),
                str(row.get("cell_id")),
            ),
        )
        for cell in ordered_cells:
            if not isinstance(cell, Mapping):
                raise CompositionalExperimentPreparationError(
                    f"family {superfamily_id} contains a malformed cell"
                )
            cell_id = _required_string(cell, "cell_id")
            evidence_state = _required_string(cell, "evidence_state")
            companion_evidence_state = _required_string(cell, "companion_evidence_state")
            distractor_injected = cell.get("distractor_injected")
            if (
                evidence_state not in EVIDENCE_STATES
                or not isinstance(distractor_injected, bool)
                or companion_evidence_state not in EVIDENCE_STATES
            ):
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} has invalid matrix coordinates"
                )
            chain_relative = _required_string(cell, "patch_chain_path")
            chain_path = _safe_relative_file(release_root, chain_relative)
            chain = RawFHIRPatchChain.model_validate(_read_json(chain_path))
            if chain.base_bundle_hash != base_hash or chain.chain_hash != cell.get(
                "patch_chain_hash"
            ):
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} patch chain is bound to a different base Bundle"
                )
            replayed = apply_raw_fhir_patch_chain(base, chain)
            replayed_hash = content_hash(replayed)
            if replayed_hash != chain.derived_bundle_hash or replayed_hash != cell.get(
                "derived_bundle_hash"
            ):
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} derived Bundle replay hash mismatch"
                )

            derived_relative = cell.get("derived_bundle_path")
            derived_file_sha256: str | None = None
            if derived_relative is not None:
                if not isinstance(derived_relative, str):
                    raise CompositionalExperimentPreparationError(
                        f"cell {cell_id} has an invalid derived Bundle path"
                    )
                derived_path = _safe_relative_file(release_root, derived_relative)
                materialized = _read_json(derived_path)
                if content_hash(materialized) != replayed_hash or materialized != replayed:
                    raise CompositionalExperimentPreparationError(
                        f"cell {cell_id} materialized Bundle differs from patch replay"
                    )
                derived_file_sha256 = _sha256_file(derived_path)

            snapshot = normalize_fhir_bundle(replayed, dataset=dataset)
            if snapshot.snapshot_id != cell.get("snapshot_id"):
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} normalized to an unexpected snapshot ID"
                )
            if snapshot.patient_id != patient_id:
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} normalized across the family patient boundary"
                )
            if snapshot.snapshot_id in seen_snapshot_ids:
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} duplicates normalized snapshot {snapshot.snapshot_id}"
                )
            snapshot_relative = (Path("snapshots") / superfamily_id / f"{cell_id}.json").as_posix()
            if snapshot_relative in seen_snapshot_paths:
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} duplicates a normalized snapshot path"
                )
            seen_snapshot_ids.add(snapshot.snapshot_id)
            seen_snapshot_paths.add(snapshot_relative)
            snapshot_path = staging / snapshot_relative
            _write_json(snapshot_path, snapshot.to_payload())
            reloaded = load_snapshot(snapshot_path)
            if reloaded.snapshot_id != snapshot.snapshot_id:
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} snapshot serialization changed its identity"
                )

            cell_task_ids = cell.get("task_ids")
            expected_topologies = (
                set(TOPOLOGIES)
                if companion_evidence_state == "supported"
                else set(WIDTH_TOPOLOGIES)
            )
            if not isinstance(cell_task_ids, Mapping) or set(cell_task_ids) != expected_topologies:
                raise CompositionalExperimentPreparationError(
                    f"cell {cell_id} does not bind exactly one task per topology"
                )
            task_records: list[dict[str, Any]] = []
            for topology in sorted(expected_topologies, key=TOPOLOGIES.index):
                task_id = cell_task_ids.get(topology)
                task = task_index.get(str(task_id))
                if task is None or task.task_id in seen_task_ids:
                    raise CompositionalExperimentPreparationError(
                        f"cell {cell_id}/{topology} has a missing or duplicate task binding"
                    )
                metadata = task.metadata.get("compositional_release")
                if not isinstance(metadata, dict) or any(
                    (
                        metadata.get("superfamily_id") != superfamily_id,
                        metadata.get("cell_id") != cell_id,
                        metadata.get("patient_id") != patient_id,
                        metadata.get("topology") != topology,
                        metadata.get("evidence_state") != evidence_state,
                        metadata.get("companion_evidence_state") != companion_evidence_state,
                        metadata.get("distractor_injected") is not distractor_injected,
                        metadata.get("base_bundle_hash") != base_hash,
                        metadata.get("patch_chain_hash") != chain.chain_hash,
                        metadata.get("derived_bundle_hash") != replayed_hash,
                        task.patient_snapshot_id != snapshot.snapshot_id,
                    )
                ):
                    raise CompositionalExperimentPreparationError(
                        f"task {task.task_id} does not match cell {cell_id}/{topology}"
                    )
                seen_task_ids.add(task.task_id)
                task_records.append(
                    {
                        "topology": topology,
                        "task_id": task.task_id,
                        "task_hash": content_hash(task.model_dump(mode="json")),
                    }
                )

            binding: dict[str, Any] = {
                "schema_version": COMPOSITIONAL_EXPERIMENT_BINDING_SCHEMA_VERSION,
                "source_release_hash": source_release_hash,
                "superfamily_id": superfamily_id,
                "patient_id": patient_id,
                "cell_id": cell_id,
                "evidence_state": evidence_state,
                "companion_evidence_state": companion_evidence_state,
                "distractor_injected": distractor_injected,
                "focal_resource_type": focal_resource_type,
                "base_bundle": {
                    "path": base_relative,
                    "content_hash": base_hash,
                    "file_sha256": _sha256_file(base_path),
                },
                "patch_chain": {
                    "path": chain_relative,
                    "chain_hash": chain.chain_hash,
                    "file_sha256": _sha256_file(chain_path),
                },
                "derived_bundle": {
                    "path": derived_relative,
                    "content_hash": replayed_hash,
                    "file_sha256": derived_file_sha256,
                    "materialized_in_source_release": derived_relative is not None,
                    "replayed": True,
                },
                "snapshot": {
                    "path": snapshot_relative,
                    "snapshot_id": snapshot.snapshot_id,
                    "file_sha256": _sha256_file(snapshot_path),
                    "record_count": len(snapshot.records),
                    "link_count": len(snapshot.links),
                    "normalizer_version": snapshot.normalizer_version,
                },
                "tasks": task_records,
            }
            binding["binding_hash"] = content_hash(binding)
            bindings.append(binding)

    if seen_task_ids != set(task_index):
        missing = sorted(set(task_index) - seen_task_ids)
        extras = sorted(seen_task_ids - set(task_index))
        raise CompositionalExperimentPreparationError(
            "snapshot bindings do not cover source tasks exactly: "
            f"missing={missing[:3]}, extras={extras[:3]}"
        )
    expected_cell_count = len(family_rows) * FACTORIAL_CELL_COUNT
    if len(bindings) != expected_cell_count:
        raise CompositionalExperimentPreparationError(
            f"expected {expected_cell_count} chart cells, prepared {len(bindings)}"
        )
    return bindings


def _validated_split_assignments(
    payload: Mapping[str, Any],
    task_index: Mapping[str, TaskArtifact],
) -> dict[str, str]:
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or payload.get("assignments_hash") != content_hash(
        assignments
    ):
        raise CompositionalExperimentPreparationError("source split assignment hash is invalid")
    rows = assignments.get("task_assignments")
    if not isinstance(rows, list):
        raise CompositionalExperimentPreparationError("source split has no task assignments")
    result: dict[str, str] = {}
    patients_by_split: dict[str, set[str]] = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }
    for row in rows:
        if not isinstance(row, dict):
            raise CompositionalExperimentPreparationError(
                "source split contains a malformed assignment"
            )
        task_id = row.get("task_id")
        split = row.get("split")
        patient_id = row.get("patient_id")
        task = task_index.get(str(task_id))
        if (
            task is None
            or task.task_id in result
            or split not in patients_by_split
            or not isinstance(patient_id, str)
            or row.get("task_hash") != content_hash(task.model_dump(mode="json"))
        ):
            raise CompositionalExperimentPreparationError(
                "source split contains an invalid or duplicate task assignment"
            )
        result[task.task_id] = split
        patients_by_split[split].add(patient_id)
    if set(result) != set(task_index):
        raise CompositionalExperimentPreparationError(
            "source split does not cover every compositional task"
        )
    if any(
        patients_by_split[left] & patients_by_split[right]
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise CompositionalExperimentPreparationError("source split is not patient-disjoint")
    return result


def _cohort_record(
    path: Path,
    *,
    relative_path: str,
    task_ids: Sequence[str],
    task_index: Mapping[str, TaskArtifact],
) -> dict[str, Any]:
    task_bindings = [
        {
            "task_id": task_id,
            "task_hash": content_hash(task_index[task_id].model_dump(mode="json")),
        }
        for task_id in task_ids
    ]
    return {
        "path": relative_path,
        "file_sha256": _sha256_file(path),
        "task_count": len(task_ids),
        "task_ids_hash": content_hash(list(task_ids)),
        "task_bindings_hash": content_hash(task_bindings),
    }


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _verify_file_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise CompositionalExperimentPreparationError("manifest has no file inventory")
    declared: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise CompositionalExperimentPreparationError(
                "manifest contains a malformed file record"
            )
        relative = row["path"]
        if relative in declared or relative == "manifest.json":
            raise CompositionalExperimentPreparationError(
                "manifest contains a duplicate or recursive file record"
            )
        path = _safe_relative_file(root, relative)
        if path.stat().st_size != row.get("bytes") or _sha256_file(path) != row.get("sha256"):
            raise CompositionalExperimentPreparationError(
                f"experiment input file differs from manifest: {relative}"
            )
        declared[relative] = row
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(declared) | {"manifest.json"}:
        raise CompositionalExperimentPreparationError(
            "experiment input directory does not match its closed file inventory"
        )


def _safe_relative_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CompositionalExperimentPreparationError(f"unsafe relative path: {relative}")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise CompositionalExperimentPreparationError(
            f"bound file does not exist inside its root: {relative}"
        )
    return path


def _required_string(value: Mapping[str, Any], key: str) -> str:
    observed = value.get(key)
    if not isinstance(observed, str) or not observed:
        raise CompositionalExperimentPreparationError(f"missing required string: {key}")
    return observed


def _stable_digest(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompositionalExperimentPreparationError(
            f"artifact is not valid JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CompositionalExperimentPreparationError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CompositionalExperimentPreparationError(
                    f"JSONL row {line_number} is not an object: {path}"
                )
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompositionalExperimentPreparationError(
            f"artifact is not valid JSONL: {path}"
        ) from exc
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical_bytes(row) + b"\n" for row in rows))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
