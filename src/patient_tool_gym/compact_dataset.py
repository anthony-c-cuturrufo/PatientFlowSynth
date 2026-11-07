"""Exact, compact split exports from a sealed synthetic compositional release.

Frozen tasks, family rows, base Bundles and patch chains are copied, not compiled
again. Materialized charts are omitted and reconstructed from the original patches.
The public package has its own closed inventory and identity; it is not the complete
private evaluator release and never copies its source/selection inventories.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from patient_tool_gym.compositional_matrix import CompositionalCandidate
from patient_tool_gym.compositional_release import (
    COMPOSITIONAL_RELEASE_SCHEMA_VERSION,
    EVIDENCE_STATES,
    TOPOLOGIES,
    WIDTH_TOPOLOGIES,
    _conditional_dependency_resolution_audit,
    _validate_compositional_patch_chain_contract,
    _verify_standalone_family,
)
from patient_tool_gym.fhir import normalize_fhir_bundle
from patient_tool_gym.medical_interventions import RawFHIRPatchChain
from patient_tool_gym.program import content_hash
from patient_tool_gym.task import TaskArtifact
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import verify_task_artifact

COMPACT_DATASET_SCHEMA_VERSION = "patientflowsynth.compact-dataset.v1"
_SPLITS_SCHEMA = "patientflowsynth.compact-splits.v1"
_SPLITS = ("train", "dev", "test")
_METADATA_FILES = {"tasks.jsonl", "families.jsonl", "splits.json"}
_DEPENDENCY_TYPES = {"Location", "Organization", "Practitioner"}


class CompactDatasetError(ValueError):
    """A dataset is incomplete, unsafe, or inconsistent with its frozen contracts."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CompactDatasetError("duplicate JSON object key")
        result[key] = value
    return result


def _decode(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise CompactDatasetError("expected a JSON object")
    return value


def _rows(raw: bytes) -> Iterator[tuple[dict[str, Any], bytes]]:
    for line in io.BytesIO(raw):
        if line.strip():
            yield _decode(line), line


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CompactDatasetError("invalid relative artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise CompactDatasetError("artifact path escapes its dataset")
    if ":" in value or path.as_posix() != value:
        raise CompactDatasetError("noncanonical artifact path")
    return value


def _file(root: Path, relative: Any) -> Path:
    relative = _relative(relative)
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise CompactDatasetError("symlinks are not permitted in dataset paths")
    if not current.is_file():
        raise CompactDatasetError(f"missing dataset file: {relative}")
    if root.resolve() not in current.resolve().parents:
        raise CompactDatasetError("artifact path escapes its dataset")
    return current


def _hash(value: Any, *, prefixed: bool = False) -> bool:
    pattern = r"sha256:[0-9a-f]{64}" if prefixed else r"[0-9a-f]{64}"
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _inventory(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise CompactDatasetError("file inventory must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise CompactDatasetError("malformed file inventory entry")
        path = _relative(row["path"])
        if path == "manifest.json" or path in result:
            raise CompactDatasetError("duplicate or recursive manifest inventory entry")
        if type(row["bytes"]) is not int or row["bytes"] < 0 or not _hash(row["sha256"]):
            raise CompactDatasetError("invalid file size or digest")
        result[path] = row
    return result


def _read_bound(root: Path, relative: str, inventory: Mapping[str, Mapping[str, Any]]) -> bytes:
    if relative not in inventory:
        raise CompactDatasetError(f"file is not declared in the inventory: {relative}")
    row = inventory[relative]
    path = _file(root, relative)
    if path.stat().st_size != row["bytes"]:
        raise CompactDatasetError(f"file size mismatch: {relative}")
    raw = path.read_bytes()
    if _sha256_bytes(raw) != row["sha256"]:
        raise CompactDatasetError(f"file hash mismatch: {relative}")
    return raw


def _closed_inventory(root: Path, inventory: Mapping[str, Mapping[str, Any]]) -> None:
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CompactDatasetError("symlinks are not permitted in a dataset")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(inventory) | {"manifest.json"}:
        raise CompactDatasetError("closed file inventory mismatch")
    for relative, row in inventory.items():
        path = _file(root, relative)
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise CompactDatasetError(f"file integrity mismatch: {relative}")


class _BaseBundles(Sequence):
    """Sized input for the original audit, without retaining every decoded chart."""

    def __init__(self, root: Path, paths: list[str], inventory: Mapping) -> None:
        self.root, self.paths, self.inventory = root, paths, inventory

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        return _decode(_read_bound(self.root, self.paths[index], self.inventory))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _included_splits(splits: Sequence[str]) -> tuple[str, ...]:
    if isinstance(splits, str) or not splits or len(set(splits)) != len(splits):
        raise CompactDatasetError("select one or more distinct train/dev/test splits")
    if any(split not in _SPLITS for split in splits):
        raise CompactDatasetError("unknown split")
    return tuple(split for split in _SPLITS if split in splits)


def is_compact_dataset(root: str | Path) -> bool:
    path = Path(root) / "manifest.json"
    if not path.is_file() or path.is_symlink():
        return False
    try:
        return _decode(path.read_bytes()).get("schema_version") == COMPACT_DATASET_SCHEMA_VERSION
    except (OSError, ValueError):
        return False


class CompactDataset:
    """Verified package bindings with a bounded, lazy chart-registry cache."""

    def __init__(self, root: Path, manifest: dict[str, Any], *, cache_size: int) -> None:
        if type(cache_size) is not int or cache_size <= 0:
            raise CompactDatasetError("cache_size must be a positive integer")
        self.root = root
        self.manifest = manifest
        self._inventory = _inventory(manifest["files"])
        self._cache_size = cache_size
        self._cache: OrderedDict[str, ToolRegistry] = OrderedDict()
        self.tasks: dict[str, TaskArtifact] = {}
        for row, _ in _rows(_read_bound(root, "tasks.jsonl", self._inventory)):
            task = TaskArtifact.model_validate(row)
            if task.task_id in self.tasks or not task.verification.passed:
                raise CompactDatasetError("duplicate or unverified task")
            self.tasks[task.task_id] = task
        self.family_rows = [
            row for row, _ in _rows(_read_bound(root, "families.jsonl", self._inventory))
        ]
        split_payload = _decode(_read_bound(root, "splits.json", self._inventory))
        if split_payload.get("schema_version") != _SPLITS_SCHEMA:
            raise CompactDatasetError("unsupported split schema")
        assignments = split_payload.get("task_assignments")
        if not isinstance(assignments, list):
            raise CompactDatasetError("missing task split assignments")
        self.task_splits: dict[str, str] = {}
        patient_splits: dict[str, str] = {}
        for row in assignments:
            if not isinstance(row, dict):
                raise CompactDatasetError("malformed split row")
            task = self.tasks.get(row.get("task_id"))
            if task is None or task.task_id in self.task_splits:
                raise CompactDatasetError("duplicate or unbound split assignment")
            metadata = task.metadata.get("compositional_release", {})
            split = row.get("split")
            if split not in manifest["included_splits"]:
                raise CompactDatasetError("task belongs to an excluded split")
            expected = {
                "task_hash": content_hash(task.model_dump(mode="json")),
                "patient_id": metadata.get("patient_id"),
                "superfamily_id": metadata.get("superfamily_id"),
                "family_id": task.family_id,
                "cell_id": metadata.get("cell_id"),
                "topology": metadata.get("topology"),
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise CompactDatasetError("split task/patient/cell binding mismatch")
            patient = metadata.get("patient_id")
            if not isinstance(patient, str) or (
                patient in patient_splits and patient_splits[patient] != split
            ):
                raise CompactDatasetError("one patient is assigned to multiple splits")
            patient_splits[patient] = split
            self.task_splits[task.task_id] = split
        if not self.tasks or set(self.task_splits) != set(self.tasks):
            raise CompactDatasetError("split assignments do not cover all tasks")
        self.family_dimensions: dict[str, dict[str, str]] = {}
        self._cells: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        expected_files = set(_METADATA_FILES) | set(manifest["dependency_paths"])
        bound = Counter()
        family_patients = set()
        for family in self.family_rows:
            if family.get("family_hash") != content_hash(
                {key: value for key, value in family.items() if key != "family_hash"}
            ):
                raise CompactDatasetError("original family contract hash mismatch")
            candidate = CompositionalCandidate.model_validate(family.get("source_candidate"))
            family_id = family.get("superfamily_id")
            patient = family.get("patient_id")
            if (
                family_id != candidate.superfamily_id
                or patient != candidate.patient_id
                or family_id in self.family_dimensions
                or patient in family_patients
                or patient not in patient_splits
            ):
                raise CompactDatasetError("family patient/identity binding mismatch")
            family_patients.add(patient)
            self.family_dimensions[family_id] = {
                key: str(getattr(candidate, key))
                for key in (
                    "focal_domain",
                    "focal_resource_type",
                    "companion_domain",
                    "companion_resource_type",
                )
            }
            source = family.get("source_bundle", {})
            base_path = _relative(source.get("base_bundle_path"))
            if not base_path.startswith("bundles/base/"):
                raise CompactDatasetError("base Bundle is outside its allowed directory")
            expected_files.add(base_path)
            if self._inventory.get(base_path, {}).get("sha256") != source.get(
                "base_bundle_file_sha256"
            ):
                raise CompactDatasetError("base Bundle inventory binding mismatch")
            topologies = family.get("topologies")
            if not isinstance(topologies, dict) or set(topologies) != set(TOPOLOGIES):
                raise CompactDatasetError("family topology inventory mismatch")
            cells = family.get("cells")
            if not isinstance(cells, list) or len(cells) != 32:
                raise CompactDatasetError("family requires exactly 32 chart cells")
            coordinates = Counter()
            for cell in cells:
                if not isinstance(cell, dict):
                    raise CompactDatasetError("malformed chart cell")
                coordinates[
                    (
                        cell.get("evidence_state"),
                        cell.get("companion_evidence_state"),
                        cell.get("distractor_injected"),
                    )
                ] += 1
                if type(cell.get("distractor_injected")) is not bool:
                    raise CompactDatasetError("cell distractor flag must be Boolean")
                patch_path = _relative(cell.get("patch_chain_path"))
                if not patch_path.startswith("patch-chains/"):
                    raise CompactDatasetError("patch is outside its allowed directory")
                expected_files.add(patch_path)
                if self._inventory.get(patch_path, {}).get("sha256") != cell.get(
                    "patch_chain_file_sha256"
                ):
                    raise CompactDatasetError("patch-chain inventory binding mismatch")
                snapshot_id = cell.get("snapshot_id")
                if not _hash(snapshot_id, prefixed=True) or snapshot_id in self._cells:
                    raise CompactDatasetError("duplicate or invalid snapshot identity")
                self._cells[snapshot_id] = (family, cell)
                task_ids = cell.get("task_ids")
                topology_set = (
                    set(TOPOLOGIES)
                    if cell.get("companion_evidence_state") == "supported"
                    else set(WIDTH_TOPOLOGIES)
                )
                if not isinstance(task_ids, dict) or set(task_ids) != topology_set:
                    raise CompactDatasetError("cell topology/task inventory mismatch")
                for topology, task_id in task_ids.items():
                    task = self.tasks.get(task_id)
                    if task is None:
                        raise CompactDatasetError("cell references an absent task")
                    metadata = task.metadata.get("compositional_release", {})
                    expected = {
                        "patient_id": patient,
                        "superfamily_id": family_id,
                        "cell_id": cell.get("cell_id"),
                        "topology": topology,
                        "evidence_state": cell.get("evidence_state"),
                        "companion_evidence_state": cell.get("companion_evidence_state"),
                        "distractor_injected": cell.get("distractor_injected"),
                        "base_bundle_hash": source.get("base_bundle_hash"),
                        "patch_chain_hash": cell.get("patch_chain_hash"),
                        "derived_bundle_hash": cell.get("derived_bundle_hash"),
                    }
                    if (
                        task.patient_snapshot_id != snapshot_id
                        or task.family_id != f"{family_id}-{topology}"
                        or any(metadata.get(key) != value for key, value in expected.items())
                        or task.question != topologies[topology].get("question")
                        or task.program.model_dump(mode="json")
                        != topologies[topology].get("program")
                    ):
                        raise CompactDatasetError("task/family/patient/cell binding mismatch")
                    bound[task_id] += 1
            expected_coordinates = Counter(
                (primary, companion, distractor)
                for primary in EVIDENCE_STATES
                for companion in EVIDENCE_STATES
                for distractor in (False, True)
            )
            if coordinates != expected_coordinates:
                raise CompactDatasetError("family evidence-state matrix mismatch")
        if set(bound) != set(self.tasks) or any(count != 1 for count in bound.values()):
            raise CompactDatasetError("tasks are unbound or multiply bound to chart cells")
        if set(self._inventory) != expected_files:
            raise CompactDatasetError("inventory includes missing or nonessential/private files")
        self._patient_splits = patient_splits
        self._check_dependencies()
        self._counts = {
            "patient_count": len(patient_splits),
            "superfamily_count": len(self.family_rows),
            "topology_family_count": 4 * len(self.family_rows),
            "cell_count": len(self._cells),
            "task_count": len(self.tasks),
            "base_bundle_file_count": sum(
                path.startswith("bundles/base/") for path in expected_files
            ),
            "patch_chain_file_count": sum(
                path.startswith("patch-chains/") for path in expected_files
            ),
            "dependency_bundle_file_count": len(manifest["dependency_paths"]),
            "task_counts_by_split": {
                split: list(self.task_splits.values()).count(split)
                for split in manifest["included_splits"]
            },
            "patient_counts_by_split": {
                split: list(patient_splits.values()).count(split)
                for split in manifest["included_splits"]
            },
        }
        if manifest["counts"] != self._counts:
            raise CompactDatasetError(
                "manifest counts disagree with original split/family bindings"
            )
        self.report = self._report(False)

    def _check_dependencies(self) -> None:
        for relative in self.manifest["dependency_paths"]:
            bundle = _decode(_read_bound(self.root, relative, self._inventory))
            entries = bundle.get("entry")
            if (
                bundle.get("resourceType") != "Bundle"
                or bundle.get("type") != "transaction"
                or set(bundle) != {"resourceType", "type", "entry"}
                or not isinstance(entries, list)
            ):
                raise CompactDatasetError("invalid projected dependency Bundle")
            for entry in entries:
                resource = entry.get("resource", {}) if isinstance(entry, dict) else {}
                if (
                    not isinstance(resource, dict)
                    or resource.get("resourceType") not in _DEPENDENCY_TYPES
                ):
                    raise CompactDatasetError(
                        "dependency Bundle contains a patient/clinical resource"
                    )
                identifiers = resource.get("identifier")
                if (
                    set(resource) != {"resourceType", "id", "identifier"}
                    or not isinstance(resource.get("id"), str)
                    or not resource["id"]
                    or not isinstance(identifiers, list)
                    or not identifiers
                    or any(
                        not isinstance(identifier, dict)
                        or set(identifier) != {"system", "value"}
                        or any(
                            not isinstance(value, str) or not value for value in identifier.values()
                        )
                        for identifier in identifiers
                    )
                    or set(entry) != {"fullUrl", "resource", "request"}
                    or entry["fullUrl"] != f"urn:uuid:{resource['id']}"
                    or entry["request"]
                    != {
                        "method": "PUT",
                        "url": f"{resource['resourceType']}/{resource['id']}",
                    }
                ):
                    raise CompactDatasetError("dependency exceeds the identity-only projection")

    def _report(self, replay: bool) -> dict[str, Any]:
        return {
            "passed": True,
            "schema_version": COMPACT_DATASET_SCHEMA_VERSION,
            "package_hash": self.manifest["package_hash"],
            "source_release_hash": self.manifest["source_release"]["release_hash"],
            "counts": dict(self._counts),
            "closed_file_inventory": True,
            "original_split_bindings_checked": True,
            "semantic_replay": replay,
            "task_artifacts_replayed": replay,
            "replayed_cell_count": len(self._cells) if replay else 0,
            "replayed_task_count": len(self.tasks) if replay else 0,
        }

    def registry(self, snapshot_id: str) -> ToolRegistry:
        """Reconstruct and replay a chart on first access; retain at most cache_size charts."""
        if snapshot_id not in self._cells:
            raise CompactDatasetError("snapshot is not part of this published subset")
        family, cell = self._cells[snapshot_id]
        base_raw = _read_bound(
            self.root, family["source_bundle"]["base_bundle_path"], self._inventory
        )
        patch_raw = _read_bound(self.root, cell["patch_chain_path"], self._inventory)
        if snapshot_id in self._cache:
            self._cache.move_to_end(snapshot_id)
            return self._cache[snapshot_id]
        base = _decode(base_raw)
        chain = RawFHIRPatchChain.model_validate(_decode(patch_raw))
        if content_hash(base) != family["source_bundle"]["base_bundle_hash"]:
            raise CompactDatasetError("base Bundle semantic hash mismatch")
        if (
            chain.chain_hash != cell["patch_chain_hash"]
            or chain.derived_bundle_hash != cell["derived_bundle_hash"]
        ):
            raise CompactDatasetError("chart patch-chain identity mismatch")
        contract, derived, _ = _validate_compositional_patch_chain_contract(
            base,
            chain,
            expected_provenance_recorded_at=cell["provenance_recorded_at"],
        )
        if contract != cell["raw_patch_contract"]:
            raise CompactDatasetError("chart patch contract differs from the frozen contract")
        snapshot = normalize_fhir_bundle(derived, dataset="synthea")
        if snapshot.snapshot_id != snapshot_id or snapshot.patient_id != family["patient_id"]:
            raise CompactDatasetError(
                "reconstructed chart has a wrong patient or snapshot identity"
            )
        registry = ToolRegistry(snapshot)
        for task_id in cell["task_ids"].values():
            task = self.tasks[task_id]
            report = verify_task_artifact(task, registry)
            if not report.passed or report != task.verification:
                raise CompactDatasetError("task does not retain its original successful replay")
        self._cache[snapshot_id] = registry
        self._cache.move_to_end(snapshot_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return registry

    def verify(self) -> dict[str, Any]:
        """Replay every selected family/cell/task using the unchanged semantic verifier."""
        _closed_inventory(self.root, self._inventory)
        bound = Counter()
        for family in self.family_rows:
            # Stored family rows and their original hashes are never rewritten. The
            # verifier's optional materialized-file check is disabled only in this
            # in-memory transport view; all patch, state, witness and task gates run.
            view = copy.deepcopy(family)
            for cell in view["cells"]:
                cell["derived_bundle_path"] = None
            view["family_hash"] = content_hash(
                {key: value for key, value in view.items() if key != "family_hash"}
            )
            bound.update(_verify_standalone_family(self.root, view, task_index=self.tasks))
        if set(bound) != set(self.tasks) or any(count != 1 for count in bound.values()):
            raise CompactDatasetError(
                "semantic replay did not cover every selected task exactly once"
            )
        dependencies = [
            _decode(_read_bound(self.root, path, self._inventory))
            for path in self.manifest["dependency_paths"]
        ]
        audit = _conditional_dependency_resolution_audit(
            _BaseBundles(
                self.root,
                [family["source_bundle"]["base_bundle_path"] for family in self.family_rows],
                self._inventory,
            ),
            dependencies,
            scope="public_compact_selected_patient_bundles",
        )
        if not audit["passed"]:
            raise CompactDatasetError("selected patient dependencies do not resolve uniquely")
        self.report = self._report(True)
        return self.report


def load_compact_dataset(
    root: str | Path,
    *,
    replay: bool = False,
    cache_size: int = 4,
) -> CompactDataset:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise CompactDatasetError("dataset root must be a real directory")
    root = root.resolve()
    manifest = _decode(_file(root, "manifest.json").read_bytes())
    expected_keys = {
        "schema_version",
        "source_release",
        "included_splits",
        "counts",
        "files",
        "tasks_path",
        "families_path",
        "splits_path",
        "dependency_paths",
        "package_hash",
    }
    if (
        set(manifest) != expected_keys
        or manifest["schema_version"] != COMPACT_DATASET_SCHEMA_VERSION
    ):
        raise CompactDatasetError("unsupported compact dataset manifest")
    if manifest["package_hash"] != content_hash(
        {key: value for key, value in manifest.items() if key != "package_hash"}
    ):
        raise CompactDatasetError("compact manifest hash mismatch")
    if list(_included_splits(manifest["included_splits"])) != manifest["included_splits"]:
        raise CompactDatasetError("split order is not canonical")
    source = manifest["source_release"]
    if (
        not isinstance(source, dict)
        or set(source) != {"schema_version", "release_hash", "manifest_sha256"}
        or source["schema_version"] != COMPOSITIONAL_RELEASE_SCHEMA_VERSION
        or not _hash(source["release_hash"], prefixed=True)
        or not _hash(source["manifest_sha256"])
    ):
        raise CompactDatasetError("invalid source release identity")
    for key, value in (
        ("tasks_path", "tasks.jsonl"),
        ("families_path", "families.jsonl"),
        ("splits_path", "splits.json"),
    ):
        if manifest[key] != value:
            raise CompactDatasetError("unexpected metadata path")
    dependencies = manifest["dependency_paths"]
    if not isinstance(dependencies, list) or dependencies != sorted(set(dependencies)):
        raise CompactDatasetError("invalid dependency path list")
    for path in dependencies:
        if not _relative(path).startswith("dependencies/") or not path.endswith(".json"):
            raise CompactDatasetError("dependency path is outside its allowed directory")
    inventory = _inventory(manifest["files"])
    _closed_inventory(root, inventory)
    dataset = CompactDataset(root, manifest, cache_size=cache_size)
    if replay:
        dataset.verify()
    return dataset


def verify_compact_dataset(root: str | Path) -> dict[str, Any]:
    return load_compact_dataset(root, replay=True).report


def export_compact_dataset(
    source: str | Path,
    destination: str | Path,
    *,
    splits: Sequence[str] = ("train", "dev"),
    expected_source_hash: str | None = None,
) -> dict[str, Any]:
    """Export original split members; default excludes every held-out test patient."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_dir():
        raise CompactDatasetError("source release must be a real directory")
    source = source.resolve()
    if destination.exists() or destination.is_symlink():
        raise CompactDatasetError("destination already exists; use a fresh directory")
    if source == destination.resolve() or source in destination.resolve().parents:
        raise CompactDatasetError("destination must be outside the immutable source release")
    included = _included_splits(splits)
    raw_manifest = _file(source, "manifest.json").read_bytes()
    original = _decode(raw_manifest)
    if original.get("schema_version") != COMPOSITIONAL_RELEASE_SCHEMA_VERSION or original.get(
        "release_hash"
    ) != content_hash({key: value for key, value in original.items() if key != "release_hash"}):
        raise CompactDatasetError("source release manifest identity is invalid")
    if expected_source_hash is not None and original["release_hash"] != expected_source_hash:
        raise CompactDatasetError("source release differs from the expected pinned hash")
    inventory = _inventory(original.get("files"))
    split_payload = _decode(_read_bound(source, "splits.json", inventory))
    assignments = split_payload.get("assignments", {})
    if split_payload.get("assignments_hash") != content_hash(assignments):
        raise CompactDatasetError("source split assignment hash mismatch")
    rows = assignments.get("task_assignments")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise CompactDatasetError("malformed original split assignments")
    assignment_index = {row.get("task_id"): row for row in rows}
    if len(assignment_index) != len(rows) or any(row.get("split") not in _SPLITS for row in rows):
        raise CompactDatasetError("duplicate or invalid original split assignment")
    selected_rows = [row for row in rows if row["split"] in included]
    selected_ids = {row["task_id"] for row in selected_rows}
    selected_patients = {row["patient_id"] for row in selected_rows}
    if not selected_ids:
        raise CompactDatasetError("selected splits contain no tasks")
    patient_split_sets: dict[str, set[str]] = {}
    for row in rows:
        patient_split_sets.setdefault(row["patient_id"], set()).add(row["split"])
    if any(len(values) != 1 for values in patient_split_sets.values()):
        raise CompactDatasetError("source assigns one patient to multiple splits")
    task_lines = []
    seen = set()
    for row, line in _rows(_read_bound(source, "tasks.jsonl", inventory)):
        task_id = row.get("task_id")
        if task_id in seen or task_id not in assignment_index:
            raise CompactDatasetError("source task/split inventory mismatch")
        seen.add(task_id)
        if task_id in selected_ids:
            if content_hash(row) != assignment_index[task_id].get("task_hash"):
                raise CompactDatasetError("original selected task hash mismatch")
            task_lines.append(line if line.endswith(b"\n") else line + b"\n")
    if seen != set(assignment_index):
        raise CompactDatasetError("source split contains absent tasks")
    family_lines, families = [], []
    for row, line in _rows(_read_bound(source, "families.jsonl", inventory)):
        if row.get("patient_id") in selected_patients:
            families.append(row)
            family_lines.append(line if line.endswith(b"\n") else line + b"\n")
    dependency_paths = sorted(path for path in inventory if path.startswith("dependencies/"))
    copied = set(dependency_paths)
    for family in families:
        copied.add(family["source_bundle"]["base_bundle_path"])
        copied.update(cell["patch_chain_path"] for cell in family["cells"])
    counts_by_split = Counter(row["split"] for row in selected_rows)
    patient_counts = Counter(
        next(iter(patient_split_sets[patient])) for patient in selected_patients
    )
    manifest: dict[str, Any] = {
        "schema_version": COMPACT_DATASET_SCHEMA_VERSION,
        "source_release": {
            "schema_version": original["schema_version"],
            "release_hash": original["release_hash"],
            "manifest_sha256": _sha256_bytes(raw_manifest),
        },
        "included_splits": list(included),
        "tasks_path": "tasks.jsonl",
        "families_path": "families.jsonl",
        "splits_path": "splits.json",
        "dependency_paths": dependency_paths,
        "counts": {
            "patient_count": len(selected_patients),
            "superfamily_count": len(families),
            "topology_family_count": len(families) * 4,
            "cell_count": len(families) * 32,
            "task_count": len(selected_ids),
            "base_bundle_file_count": sum(path.startswith("bundles/base/") for path in copied),
            "patch_chain_file_count": sum(path.startswith("patch-chains/") for path in copied),
            "dependency_bundle_file_count": len(dependency_paths),
            "task_counts_by_split": {split: counts_by_split[split] for split in included},
            "patient_counts_by_split": {split: patient_counts[split] for split in included},
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        (staging / "tasks.jsonl").write_bytes(b"".join(task_lines))
        (staging / "families.jsonl").write_bytes(b"".join(family_lines))
        (staging / "splits.json").write_bytes(
            _json_bytes(
                {
                    "schema_version": _SPLITS_SCHEMA,
                    "task_assignments": selected_rows,
                }
            )
        )
        for relative in sorted(copied):
            raw = _read_bound(source, relative, inventory)
            target = staging / _relative(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        manifest["files"] = [
            {
                "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest["package_hash"] = content_hash(manifest)
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        verify_compact_dataset(staging)
        if destination.exists() or destination.is_symlink():
            raise CompactDatasetError("destination appeared during export; refusing to overwrite")
        staging.rename(destination)
    return manifest
