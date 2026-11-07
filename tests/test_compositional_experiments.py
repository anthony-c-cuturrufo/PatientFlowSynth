from __future__ import annotations

import json
from pathlib import Path

import pytest

from patient_tool_gym.compositional_experiments import (
    CompositionalExperimentPreparationError,
    prepare_compositional_experiment_inputs,
    verify_compositional_experiment_inputs,
)
from patient_tool_gym.compositional_matrix import TOPOLOGIES
from patient_tool_gym.compositional_release import (
    EVIDENCE_STATES,
    build_compositional_release,
    verify_compositional_release_context,
)
from patient_tool_gym.evidence import load_snapshot
from tests.test_compositional_release import _source


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _build_source_release(tmp_path: Path) -> Path:
    bundle, tasks = _source()
    release = tmp_path / "source-release"
    # Deliberately omit derived Bundle files.  Preparation must replay the frozen
    # patch chains, so the same code works for compact and materialized releases.
    build_compositional_release(
        tasks,
        {"fixture.json": bundle},
        release,
        seed=31,
        materialize_derived_bundles=False,
    )
    return release


def test_prepares_complete_runtime_inputs_without_mutating_release(tmp_path):
    release = _build_source_release(tmp_path)
    source_files_before = {
        path.relative_to(release).as_posix(): path.read_bytes()
        for path in release.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "experiment-inputs"

    manifest = prepare_compositional_experiment_inputs(release, output, seed=41)

    source_files_after = {
        path.relative_to(release).as_posix(): path.read_bytes()
        for path in release.rglob("*")
        if path.is_file()
    }
    assert source_files_after == source_files_before
    assert manifest["counts"] == {
        "patient_count": 1,
        "resource_domain_count": 1,
        "snapshot_count": 32,
        "task_count": 80,
        "canary_task_count": 80,
        "split_task_counts": {"train": 80, "dev": 0, "test": 0},
    }
    assert manifest["source_release"]["release_hash"].startswith("sha256:")
    assert manifest["source_release"]["task_jsonl_sha256"].startswith("sha256:")

    bindings = _jsonl(output / "snapshot-bindings.jsonl")
    assert len(bindings) == 32
    assert len({row["snapshot"]["snapshot_id"] for row in bindings}) == 32
    assert {len(row["tasks"]) for row in bindings} == {2, len(TOPOLOGIES)}
    assert all(row["derived_bundle"]["path"] is None for row in bindings)
    assert all(row["derived_bundle"]["replayed"] for row in bindings)
    for row in bindings:
        snapshot = load_snapshot(output / row["snapshot"]["path"])
        assert snapshot.snapshot_id == row["snapshot"]["snapshot_id"]
        assert snapshot.patient_id == row["patient_id"]

    all_tasks = _jsonl(output / "tasks/all.jsonl")
    canary = _jsonl(output / "tasks/canary-factorial-80.jsonl")
    assert len(all_tasks) == len(canary) == 80
    assert {
        (
            row["metadata"]["compositional_release"]["topology"],
            row["metadata"]["compositional_release"]["evidence_state"],
            row["metadata"]["compositional_release"]["companion_evidence_state"],
            row["metadata"]["compositional_release"]["distractor_injected"],
        )
        for row in canary
    } == {
        (topology, state, companion_state, distractor)
        for topology in TOPOLOGIES
        for state in EVIDENCE_STATES
        for companion_state in (
            EVIDENCE_STATES if topology.endswith("w2") else ("supported",)
        )
        for distractor in (False, True)
    }
    assert verify_compositional_experiment_inputs(
        output, release_directory=release
    )["passed"]


def test_preparation_is_deterministic_and_rejects_existing_output(tmp_path):
    release = _build_source_release(tmp_path)
    first = tmp_path / "inputs-first"
    second = tmp_path / "inputs-second"

    first_manifest = prepare_compositional_experiment_inputs(release, first, seed=43)
    second_manifest = prepare_compositional_experiment_inputs(release, second, seed=43)

    assert first_manifest == second_manifest
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    with pytest.raises(CompositionalExperimentPreparationError, match="already exists"):
        prepare_compositional_experiment_inputs(release, first, seed=43)


def test_preparation_and_verification_reuse_verified_release_context(tmp_path):
    release = _build_source_release(tmp_path)
    output = tmp_path / "experiment-inputs"
    context = verify_compositional_release_context(release)

    prepare_compositional_experiment_inputs(
        release,
        output,
        seed=45,
        release_context=context,
    )

    assert verify_compositional_experiment_inputs(
        output,
        release_directory=release,
        release_context=context,
    )["passed"]


def test_verifier_rejects_tampered_snapshot(tmp_path):
    release = _build_source_release(tmp_path)
    output = tmp_path / "experiment-inputs"
    prepare_compositional_experiment_inputs(release, output, seed=47)
    binding = _jsonl(output / "snapshot-bindings.jsonl")[0]
    snapshot_path = output / binding["snapshot"]["path"]
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b"\n")

    with pytest.raises(
        CompositionalExperimentPreparationError,
        match="differs from manifest",
    ):
        verify_compositional_experiment_inputs(output)
