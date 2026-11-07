from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from patient_tool_gym.compact_dataset import (
    CompactDatasetError,
    export_compact_dataset,
    is_compact_dataset,
    load_compact_dataset,
    verify_compact_dataset,
)
from patient_tool_gym.compositional_release import build_compositional_release
from patient_tool_gym.fhir import normalize_fhir_bundle
from patient_tool_gym.patient_p2 import generate_patient_p2_tasks
from patient_tool_gym.program import content_hash
from tests.test_compositional_release import _fixture_bundle


def _read(path: Path):
    return json.loads(path.read_text())


def _lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _write_lines(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _reseal(root: Path) -> None:
    manifest = _read(root / "manifest.json")
    for row in manifest["files"]:
        path = root / row["path"]
        row["bytes"] = path.stat().st_size
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.pop("package_hash")
    manifest["package_hash"] = content_hash(manifest)
    _write(root / "manifest.json", manifest)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("compact-source")
    bundles = {}
    for index in range(3):
        # Fully fabricated fixture; distinct Patient identities with unchanged clinical facts.
        bundle = json.loads(
            json.dumps(_fixture_bundle()).replace(
                "release-patient",
                f"compact-patient-{index}",
            )
        )
        bundles[f"patient-{index}.json"] = bundle
    snapshots = tuple(
        normalize_fhir_bundle(bundle, dataset="synthea") for bundle in bundles.values()
    )
    tasks = generate_patient_p2_tasks(snapshots).tasks
    destination = root / "release"
    build_compositional_release(
        tasks, bundles, destination, seed=31, materialize_derived_bundles=True
    )
    split_rows = _read(destination / "splits.json")["assignments"]["task_assignments"]
    assert {row["split"] for row in split_rows} == {"train", "dev", "test"}
    return destination


@pytest.fixture(scope="module")
def compact(source, tmp_path_factory):
    destination = tmp_path_factory.mktemp("compact-export") / "dataset"
    export_compact_dataset(source, destination)
    return destination


def test_default_export_preserves_original_rows_and_excludes_test(source, compact):
    original = _read(source / "manifest.json")
    before = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    manifest = _read(compact / "manifest.json")
    assert is_compact_dataset(compact)
    assert manifest["source_release"]["release_hash"] == original["release_hash"]
    assert manifest["included_splits"] == ["train", "dev"]
    assert manifest["counts"]["patient_count"] == 2
    assert manifest["counts"]["task_count"] == 160
    assert manifest["counts"]["cell_count"] == 64
    assignments = _read(source / "splits.json")["assignments"]["task_assignments"]
    selected = {row["task_id"] for row in assignments if row["split"] != "test"}
    private_ids = {row["task_id"] for row in assignments if row["split"] == "test"}
    private_patients = {row["patient_id"] for row in assignments if row["split"] == "test"}
    source_lines = {
        json.loads(line)["task_id"]: line
        for line in (source / "tasks.jsonl").read_bytes().splitlines()
    }
    actual_lines = (compact / "tasks.jsonl").read_bytes().splitlines()
    assert {json.loads(line)["task_id"] for line in actual_lines} == selected
    assert all(source_lines[json.loads(line)["task_id"]] == line for line in actual_lines)
    assert _read(compact / "splits.json")["task_assignments"] == [
        row for row in assignments if row["split"] != "test"
    ]
    source_family_lines = set((source / "families.jsonl").read_bytes().splitlines())
    assert set((compact / "families.jsonl").read_bytes().splitlines()) <= source_family_lines
    assert not (compact / "sources").exists()
    assert not (compact / "bundles" / "derived").exists()
    for row in manifest["files"]:
        raw = (compact / row["path"]).read_bytes()
        assert not any(identifier.encode() in raw for identifier in private_ids | private_patients)
        if row["path"].startswith(("bundles/", "patch-chains/", "dependencies/")):
            assert raw == (source / row["path"]).read_bytes()
    assert hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() == before


def test_full_replay_and_lazy_bounded_registry(compact):
    report = verify_compact_dataset(compact)
    assert report["semantic_replay"] is True
    assert report["replayed_cell_count"] == 64
    assert report["replayed_task_count"] == 160
    dataset = load_compact_dataset(compact, cache_size=1)
    assert dataset.report["semantic_replay"] is False
    snapshot_ids = list(dict.fromkeys(task.patient_snapshot_id for task in dataset.tasks.values()))
    first = dataset.registry(snapshot_ids[0])
    assert dataset.registry(snapshot_ids[0]) is first
    dataset.registry(snapshot_ids[1])
    assert len(dataset._cache) == 1
    assert dataset.registry(snapshot_ids[0]) is not first
    with pytest.raises(CompactDatasetError, match="not part"):
        dataset.registry("sha256:" + "0" * 64)


def test_all_splits_requires_explicit_selection(source, tmp_path):
    manifest = export_compact_dataset(source, tmp_path / "all", splits=("train", "dev", "test"))
    assert manifest["counts"]["patient_count"] == 3
    assert manifest["counts"]["task_count"] == 240
    assert manifest["included_splits"] == ["train", "dev", "test"]


def test_export_rejects_pin_overwrite_and_nested_destination(source, compact, tmp_path):
    with pytest.raises(CompactDatasetError, match="pinned hash"):
        export_compact_dataset(
            source, tmp_path / "wrong-pin", expected_source_hash="sha256:" + "0" * 64
        )
    with pytest.raises(CompactDatasetError, match="already exists"):
        export_compact_dataset(source, compact)
    with pytest.raises(CompactDatasetError, match="outside"):
        export_compact_dataset(source, source / "nested")
    with pytest.raises(CompactDatasetError, match="distinct"):
        export_compact_dataset(source, tmp_path / "duplicate", splits=("train", "train"))


@pytest.fixture
def modified(compact, tmp_path):
    root = tmp_path / "modified"
    shutil.copytree(compact, root)
    return root


def test_detects_tampered_bytes_and_extra_files(modified):
    (modified / "unexpected-private-file.json").write_text("{}")
    with pytest.raises(CompactDatasetError, match="inventory"):
        load_compact_dataset(modified)
    (modified / "unexpected-private-file.json").unlink()
    with (modified / "tasks.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(CompactDatasetError, match="integrity"):
        load_compact_dataset(modified)


def test_rejects_parent_traversal_and_symlink(modified, tmp_path):
    manifest = _read(modified / "manifest.json")
    original = copy.deepcopy(manifest)
    manifest["files"][0]["path"] = "../outside.json"
    manifest.pop("package_hash")
    manifest["package_hash"] = content_hash(manifest)
    _write(modified / "manifest.json", manifest)
    with pytest.raises(CompactDatasetError, match="escapes"):
        load_compact_dataset(modified)
    _write(modified / "manifest.json", original)
    (modified / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(CompactDatasetError, match="symlinks"):
        load_compact_dataset(modified)


def test_rejects_resealed_wrong_patient_binding(modified):
    families = _lines(modified / "families.jsonl")
    families[0]["patient_id"] = "wrong-patient"
    families[0].pop("family_hash")
    families[0]["family_hash"] = content_hash(families[0])
    _write_lines(modified / "families.jsonl", families)
    _reseal(modified)
    with pytest.raises(CompactDatasetError, match="patient/identity"):
        load_compact_dataset(modified)


def test_rejects_resealed_cross_patient_split(modified):
    splits = _read(modified / "splits.json")
    row = splits["task_assignments"][0]
    row["split"] = "dev" if row["split"] == "train" else "train"
    _write(modified / "splits.json", splits)
    _reseal(modified)
    with pytest.raises(CompactDatasetError, match="multiple splits"):
        load_compact_dataset(modified)


def test_rejects_resealed_wrong_cell_binding(modified):
    tasks = _lines(modified / "tasks.jsonl")
    tasks[0]["patient_snapshot_id"] = "sha256:" + "0" * 64
    _write_lines(modified / "tasks.jsonl", tasks)
    splits = _read(modified / "splits.json")
    for row in splits["task_assignments"]:
        if row["task_id"] == tasks[0]["task_id"]:
            row["task_hash"] = content_hash(tasks[0])
    _write(modified / "splits.json", splits)
    _reseal(modified)
    with pytest.raises(ValueError, match="binding|snapshot"):
        load_compact_dataset(modified)


def test_semantic_replay_rejects_resealed_wrong_answer(modified):
    tasks = _lines(modified / "tasks.jsonl")
    answer = tasks[0]["gold_answer"]["value"]
    answer["branches"]["primary"] = (
        "refuted" if answer["branches"]["primary"] != "refuted" else "supported"
    )
    _write_lines(modified / "tasks.jsonl", tasks)
    splits = _read(modified / "splits.json")
    for row in splits["task_assignments"]:
        if row["task_id"] == tasks[0]["task_id"]:
            row["task_hash"] = content_hash(tasks[0])
    _write(modified / "splits.json", splits)
    _reseal(modified)
    with pytest.raises(ValueError):
        verify_compact_dataset(modified)


def test_cached_registry_checks_bound_file_changes(modified):
    dataset = load_compact_dataset(modified)
    snapshot_id = next(iter(dataset.tasks.values())).patient_snapshot_id
    dataset.registry(snapshot_id)
    _, cell = dataset._cells[snapshot_id]
    with (modified / cell["patch_chain_path"]).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(CompactDatasetError, match="size mismatch"):
        dataset.registry(snapshot_id)


@pytest.mark.parametrize("extra_field", ["subject", "text", "identifier", "entry"])
def test_rejects_nonprojected_dependency_fields(modified, extra_field):
    resource = {
        "resourceType": "Practitioner",
        "id": "fabricated-provider",
        "identifier": [{"system": "urn:example:provider", "value": "fabricated-provider"}],
    }
    entry = {
        "fullUrl": "urn:uuid:fabricated-provider",
        "resource": resource,
        "request": {"method": "PUT", "url": "Practitioner/fabricated-provider"},
    }
    bundle = {"resourceType": "Bundle", "type": "transaction", "entry": [entry]}
    relative = "dependencies/projected-test.json"
    (modified / "dependencies").mkdir(exist_ok=True)
    _write(modified / relative, bundle)
    manifest = _read(modified / "manifest.json")
    manifest["dependency_paths"].append(relative)
    manifest["dependency_paths"].sort()
    manifest["counts"]["dependency_bundle_file_count"] += 1
    manifest["files"].append({"path": relative, "bytes": 0, "sha256": "0" * 64})
    _write(modified / "manifest.json", manifest)
    _reseal(modified)
    assert load_compact_dataset(modified).report["passed"]
    if extra_field == "identifier":
        resource["identifier"][0]["extension"] = [{"valueString": "private patient detail"}]
    elif extra_field == "entry":
        entry["subject"] = {"reference": "urn:uuid:private-patient"}
    else:
        resource[extra_field] = {"reference": "urn:uuid:private-patient"}
    _write(modified / relative, bundle)
    _reseal(modified)
    with pytest.raises(CompactDatasetError, match="identity-only projection"):
        load_compact_dataset(modified)
