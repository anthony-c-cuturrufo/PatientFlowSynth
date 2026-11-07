from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from patient_tool_gym import compositional_release as compositional_release_module
from patient_tool_gym.compositional_release import (
    EVIDENCE_STATES,
    CompositionalReleaseError,
    _validate_compositional_patch_chain_contract,
    _verify_standalone_dependencies,
    build_compositional_participant_package,
    build_compositional_release,
    load_compositional_release_verification_context,
    validate_agent_visible_leakage_contract,
    validate_compositional_patch_chain_contract,
    verify_compositional_participant_package,
    verify_compositional_release_context,
)
from patient_tool_gym.fhir import normalize_fhir_bundle
from patient_tool_gym.medical_interventions import (
    RawFHIRPatchChain,
    apply_raw_fhir_patch_chain,
    build_compositional_patch_chain,
)
from patient_tool_gym.patient_p2 import generate_patient_p2_tasks
from patient_tool_gym.program import content_hash
from patient_tool_gym.raw_fhir import (
    RawFHIRInsert,
    RawFHIRPatch,
    RawFHIRSet,
    apply_raw_fhir_patch,
)
from patient_tool_gym.task import TaskArtifact
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import verify_task_artifact

SNOMED = "http://snomed.info/sct"


def _concept(system: str, code: str, display: str) -> dict:
    return {
        "coding": [{"system": system, "code": code, "display": display}],
        "text": display,
    }


def _entry(resource: dict) -> dict:
    return {"fullUrl": f"urn:uuid:{resource['id']}", "resource": resource}


def _encounter(resource_id: str, code: str, display: str, start: str) -> dict:
    return {
        "resourceType": "Encounter",
        "id": resource_id,
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB"},
        "type": [_concept(SNOMED, code, display)],
        "subject": {"reference": "urn:uuid:release-patient"},
        "period": {"start": start, "end": start.replace(":00Z", ":50Z")},
    }


def _condition(
    resource_id: str,
    code: str,
    display: str,
    encounter_id: str,
    recorded: str,
) -> dict:
    return {
        "resourceType": "Condition",
        "id": resource_id,
        "clinicalStatus": _concept(
            "http://terminology.hl7.org/CodeSystem/condition-clinical", "active", "Active"
        ),
        "verificationStatus": _concept(
            "http://terminology.hl7.org/CodeSystem/condition-ver-status",
            "confirmed",
            "Confirmed",
        ),
        "code": _concept(SNOMED, code, display),
        "subject": {"reference": "urn:uuid:release-patient"},
        "encounter": {"reference": f"urn:uuid:{encounter_id}"},
        "recordedDate": recorded,
    }


def _procedure(
    resource_id: str,
    code: str,
    display: str,
    encounter_id: str,
    performed: str,
) -> dict:
    return {
        "resourceType": "Procedure",
        "id": resource_id,
        "status": "completed",
        "code": _concept(SNOMED, code, display),
        "subject": {"reference": "urn:uuid:release-patient"},
        "encounter": {"reference": f"urn:uuid:{encounter_id}"},
        "performedDateTime": performed,
    }


def _immunization(resource_id: str, encounter_id: str, occurrence: str) -> dict:
    return {
        "resourceType": "Immunization",
        "id": resource_id,
        "status": "completed",
        "vaccineCode": _concept(
            "http://hl7.org/fhir/sid/cvx",
            "140",
            "Influenza, seasonal, injectable, preservative free",
        ),
        "patient": {"reference": "urn:uuid:release-patient"},
        "encounter": {"reference": f"urn:uuid:{encounter_id}"},
        "occurrenceDateTime": occurrence,
    }


def _fixture_bundle() -> dict:
    resources = [
        {
            "resourceType": "Patient",
            "id": "release-patient",
            "active": True,
            "name": [{"given": ["Release"], "family": "Patient"}],
            "gender": "female",
            "birthDate": "1980-06-15",
        },
        _encounter("enc-initial", "185349003", "Encounter for check up", "2026-05-10T09:00:00Z"),
        _encounter("enc-follow-up", "390906007", "Follow-up visit", "2026-06-15T10:00:00Z"),
        _condition(
            "hypertension-initial",
            "59621000",
            "Essential hypertension",
            "enc-initial",
            "2026-05-10T09:05:00Z",
        ),
        _condition(
            "hypertension-follow-up",
            "59621000",
            "Essential hypertension",
            "enc-follow-up",
            "2026-06-15T10:05:00Z",
        ),
        _condition(
            "diabetes",
            "44054006",
            "Type 2 diabetes mellitus",
            "enc-follow-up",
            "2026-06-15T10:06:00Z",
        ),
        _condition(
            "stress",
            "73595000",
            "Stress",
            "enc-follow-up",
            "2026-06-15T10:07:00Z",
        ),
        _condition(
            "hyperlipidemia",
            "55822004",
            "Hyperlipidemia",
            "enc-follow-up",
            "2026-06-15T10:08:00Z",
        ),
        _procedure(
            "colonoscopy-follow-up",
            "73761001",
            "Colonoscopy",
            "enc-follow-up",
            "2026-06-15T10:20:00Z",
        ),
        _immunization(
            "influenza-follow-up",
            "enc-follow-up",
            "2026-06-15T10:30:00Z",
        ),
        _procedure(
            "biopsy",
            "86273004",
            "Biopsy",
            "enc-follow-up",
            "2026-06-15T10:21:00Z",
        ),
        _procedure(
            "rehabilitation",
            "52052004",
            "Rehabilitation therapy",
            "enc-follow-up",
            "2026-06-15T10:22:00Z",
        ),
        _procedure(
            "screening",
            "171207006",
            "Depression screening",
            "enc-follow-up",
            "2026-06-15T10:23:00Z",
        ),
    ]
    return {
        "resourceType": "Bundle",
        "id": "compositional-release-fixture",
        "type": "collection",
        "timestamp": "2026-06-16T00:00:00Z",
        "entry": [_entry(resource) for resource in resources],
    }


def _source() -> tuple[dict, tuple[TaskArtifact, ...]]:
    bundle = _fixture_bundle()
    snapshot = normalize_fhir_bundle(bundle, dataset="synthea")
    tasks = generate_patient_p2_tasks((snapshot,)).tasks
    return bundle, tasks


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _rehash_patch(patch: RawFHIRPatch) -> RawFHIRPatch:
    draft = patch.model_copy(update={"patch_hash": "pending"})
    return draft.model_copy(
        update={"patch_hash": content_hash(draft.model_dump(mode="json", exclude={"patch_hash"}))}
    )


def _rehash_single_patch_chain(
    bundle: dict,
    chain: RawFHIRPatchChain,
    patch: RawFHIRPatch,
) -> RawFHIRPatchChain:
    patch = _rehash_patch(patch)
    derived = apply_raw_fhir_patch(bundle, patch)
    draft = chain.model_copy(
        update={
            "patches": (patch,),
            "derived_bundle_hash": content_hash(derived),
            "chain_hash": "pending",
        }
    )
    return draft.model_copy(
        update={"chain_hash": content_hash(draft.model_dump(mode="json", exclude={"chain_hash"}))}
    )


def _supported_plain_condition_chain(bundle: dict) -> RawFHIRPatchChain:
    return build_compositional_patch_chain(
        bundle,
        resource_type="Condition",
        resource_id="hypertension-follow-up",
        evidence_state="refuted",
        distractor_present=False,
    )


def _release_task_context(tmp_path: Path) -> tuple[TaskArtifact, ToolRegistry]:
    bundle, tasks = _source()
    output = tmp_path / "metadata-contract-release"
    build_compositional_release(tasks, {"fixture.json": bundle}, output, seed=31)
    families = _jsonl(output / "families.jsonl")
    family = families[0]
    cell = next(
        row
        for row in family["cells"]
        if row["evidence_state"] == "supported"
        and row["companion_evidence_state"] == "supported"
        and not row["distractor_injected"]
    )
    task_id = cell["task_ids"]["d1w1"]
    task = next(
        TaskArtifact.model_validate(row)
        for row in _jsonl(output / "tasks.jsonl")
        if row["task_id"] == task_id
    )
    base = json.loads((output / family["source_bundle"]["base_bundle_path"]).read_text())
    chain = RawFHIRPatchChain.model_validate(
        json.loads((output / cell["patch_chain_path"]).read_text())
    )
    derived = apply_raw_fhir_patch_chain(base, chain)
    registry = ToolRegistry(normalize_fhir_bundle(derived, dataset="synthea"))
    return task, registry


def test_builds_complete_patch_first_release_from_all_fixture_candidates(tmp_path):
    bundle, tasks = _source()
    output = tmp_path / "release-v1"

    manifest = build_compositional_release(
        tasks,
        {"fixture.json": bundle},
        output,
        seed=17,
    )

    family_count = manifest["counts"]["superfamily_count"]
    assert family_count >= 1
    assert manifest["selection_policy"]["eligible_family_policy"] == "all_discovered"
    assert manifest["selection_policy"]["quota"] is None
    assert manifest["counts"]["task_count"] == family_count * 80
    assert manifest["counts"]["cell_count"] == family_count * 32
    assert manifest["counts"]["patch_chain_file_count"] == family_count * 32
    assert manifest["counts"]["derived_bundle_file_count"] == 0
    assert manifest["storage_policy"]["derived_bundles_materialized"] is False
    assert not (output / "bundles/derived").exists()
    assert (output / "sources/p2-tasks.jsonl").is_file()
    assert (output / "sources/bundle-catalog.json").is_file()
    assert manifest["sources"]["tasks"]["release_sha256"] == next(
        row["sha256"] for row in manifest["files"] if row["path"] == "sources/p2-tasks.jsonl"
    )

    families = _jsonl(output / "families.jsonl")
    release_tasks = [TaskArtifact.model_validate(row) for row in _jsonl(output / "tasks.jsonl")]
    validation = json.loads((output / "validation-report.json").read_text())
    splits = json.loads((output / "splits.json").read_text())
    assert len(families) == family_count
    assert validation["passed"] is True
    assert validation["task_count"] == len(release_tasks)
    assert splits["passed"] is True
    assert all(
        not overlap for audit in splits["overlap_audit"].values() for overlap in audit.values()
    )

    for family in families:
        assert len(family["cells"]) == 32
        assert {row["evidence_state"] for row in family["cells"]} == set(EVIDENCE_STATES)
        assert {row["companion_evidence_state"] for row in family["cells"]} == set(EVIDENCE_STATES)
        assert {row["distractor_injected"] for row in family["cells"]} == {False, True}
        assert len(family["topologies"]) == 4
        for topology in family["topologies"].values():
            assert topology["program_hash"] == content_hash(topology["program"])
            assert topology["question_hash"] == content_hash(topology["question"])

        base = json.loads((output / family["source_bundle"]["base_bundle_path"]).read_text())
        for cell in family["cells"]:
            chain = RawFHIRPatchChain.model_validate(
                json.loads((output / cell["patch_chain_path"]).read_text())
            )
            _, derived, baselines = _validate_compositional_patch_chain_contract(
                base,
                chain,
                expected_provenance_recorded_at=cell["provenance_recorded_at"],
            )
            snapshot = normalize_fhir_bundle(derived, dataset="synthea")
            fast_scan = validate_agent_visible_leakage_contract(
                base,
                chain,
                normalized_snapshot=snapshot,
                _validated_distractor_baselines=baselines,
            )
            independent_scan = validate_agent_visible_leakage_contract(
                base,
                chain,
                normalized_snapshot=snapshot,
            )
            assert fast_scan == independent_scan == cell["agent_visible_leakage_scan"]
            assert content_hash(derived) == cell["derived_bundle_hash"]

    grouped: dict[tuple[str, str], list[TaskArtifact]] = {}
    for task in release_tasks:
        metadata = task.metadata["compositional_release"]
        grouped.setdefault((metadata["superfamily_id"], metadata["topology"]), []).append(task)
    for siblings in grouped.values():
        assert len({content_hash(task.program.model_dump(mode="json")) for task in siblings}) == 1
        assert len({task.question for task in siblings}) == 1
        for task in siblings:
            metadata = task.metadata["compositional_release"]
            answer = task.gold_answer.value
            assert answer["overall"] == metadata["expected_overall_state"]
            assert answer["branches"] == metadata["expected_branch_states"]
            distractor_id = metadata["distractor_evidence_id"]
            if metadata["distractor_injected"]:
                focal_call = next(
                    call for call in task.reference_trace if call.node_id == "focal_search"
                )
                assert distractor_id in focal_call.evidence_ids
                assert distractor_id not in task.gold_evidence_ids
            else:
                assert distractor_id is None
            if metadata["topology"].endswith("w1"):
                assert metadata["companion_evidence_state"] == "supported"
                assert metadata["expected_overall_state"] == metadata["evidence_state"]

    by_cell: dict[tuple[str, str], dict[str, TaskArtifact]] = {}
    for task in release_tasks:
        metadata = task.metadata["compositional_release"]
        by_cell.setdefault((metadata["superfamily_id"], metadata["cell_id"]), {})[
            metadata["topology"]
        ] = task
    canonical: dict[tuple[str, str, bool], dict[str, TaskArtifact]] = {}
    for cell in by_cell.values():
        sample = next(iter(cell.values())).metadata["compositional_release"]
        if sample["companion_evidence_state"] == "supported":
            canonical[
                (
                    sample["superfamily_id"],
                    sample["evidence_state"],
                    sample["distractor_injected"],
                )
            ] = cell
    for cell in by_cell.values():
        sample = next(iter(cell.values())).metadata["compositional_release"]
        root = canonical[
            (sample["superfamily_id"], sample["evidence_state"], sample["distractor_injected"])
        ]
        if sample["companion_evidence_state"] == "supported":
            assert cell["d1w1"].parent_task_ids == ()
            assert cell["d2w1"].parent_task_ids == (cell["d1w1"].task_id,)
        assert cell["d1w2"].parent_task_ids == (root["d1w1"].task_id,)
        assert cell["d2w2"].parent_task_ids == tuple(
            sorted((root["d2w1"].task_id, cell["d1w2"].task_id))
        )


@pytest.mark.parametrize("tampered_field", ["target", "activity", "recorded", "agent"])
def test_raw_patch_contract_rejects_exact_provenance_tampering(tampered_field):
    bundle = _fixture_bundle()
    chain = _supported_plain_condition_chain(bundle)
    patch = chain.patches[0]
    operations = list(patch.operations)
    provenance_index = next(
        index
        for index, operation in enumerate(operations)
        if isinstance(operation, RawFHIRInsert)
        and operation.resource.get("resourceType") == "Provenance"
    )
    provenance_operation = operations[provenance_index]
    assert isinstance(provenance_operation, RawFHIRInsert)
    provenance = copy.deepcopy(provenance_operation.resource)
    if tampered_field == "target":
        provenance["target"] = [{"reference": "urn:uuid:release-patient"}]
    elif tampered_field == "activity":
        provenance["activity"]["coding"][0]["code"] = "CREATE"
    elif tampered_field == "recorded":
        provenance["recorded"] = "2020-01-01T00:00:00Z"
    else:
        provenance["agent"][0]["type"]["coding"][0]["code"] = "author"
    operations[provenance_index] = provenance_operation.model_copy(update={"resource": provenance})
    bad_patch = patch.model_copy(update={"operations": tuple(operations)})
    bad_chain = _rehash_single_patch_chain(bundle, chain, bad_patch)

    with pytest.raises(
        CompositionalReleaseError,
        match="Provenance",
    ):
        validate_compositional_patch_chain_contract(bundle, bad_chain)


def test_raw_patch_contract_rejects_nonclinical_target_field_change():
    bundle = _fixture_bundle()
    chain = _supported_plain_condition_chain(bundle)
    patch = chain.patches[0]
    target = next(
        entry["resource"]
        for entry in bundle["entry"]
        if entry["resource"].get("id") == "hypertension-follow-up"
    )
    invalid_set = RawFHIRSet(
        resource_type="Condition",
        resource_id="hypertension-follow-up",
        path=("code",),
        prior_present=True,
        prior_hash=content_hash(target["code"]),
        value=_concept(SNOMED, "999999", "Unrelated diagnosis"),
    )
    bad_patch = patch.model_copy(update={"operations": (*patch.operations, invalid_set)})
    bad_chain = _rehash_single_patch_chain(bundle, chain, bad_patch)

    with pytest.raises(
        CompositionalReleaseError,
        match=r"cannot modify Condition\.code",
    ):
        validate_compositional_patch_chain_contract(bundle, bad_chain)


def test_raw_patch_contract_rejects_undeclared_inserted_resource():
    bundle = _fixture_bundle()
    chain = _supported_plain_condition_chain(bundle)
    patch = chain.patches[0]
    extra = RawFHIRInsert(
        full_url="urn:uuid:undeclared-organization",
        resource={
            "resourceType": "Organization",
            "id": "undeclared-organization",
            "name": "Undeclared Organization",
        },
    )
    bad_patch = patch.model_copy(update={"operations": (*patch.operations, extra)})
    bad_chain = _rehash_single_patch_chain(bundle, chain, bad_patch)

    with pytest.raises(
        CompositionalReleaseError,
        match="non-conflict state patch cannot insert a clinical record",
    ):
        validate_compositional_patch_chain_contract(bundle, bad_chain)


def test_verify_task_rejects_tampered_compositional_metadata(tmp_path):
    task, registry = _release_task_context(tmp_path)
    assert verify_task_artifact(task, registry).passed

    mutations = {
        "cell_id": lambda metadata: metadata.__setitem__("cell_id", "tampered-cell"),
        "patch_chain_hash": lambda metadata: metadata.__setitem__(
            "patch_chain_hash", "sha256:" + "1" * 64
        ),
        "expected_overall_state": lambda metadata: metadata.__setitem__(
            "expected_overall_state", "refuted"
        ),
        "source_task_hash": lambda metadata: metadata["source_task_hashes"].__setitem__(
            "focal_atomic", "sha256:" + "2" * 64
        ),
        "unknown_field": lambda metadata: metadata.__setitem__("undeclared", True),
    }
    for label, mutate in mutations.items():
        payload = task.model_dump(mode="json")
        metadata = payload["metadata"]["compositional_release"]
        mutate(metadata)
        metadata["metadata_hash"] = content_hash(
            {key: value for key, value in metadata.items() if key != "metadata_hash"}
        )
        tampered = TaskArtifact.model_validate(payload)
        report = verify_task_artifact(tampered, registry)
        assert not report.passed, label
        assert "compositional_release_contract_invalid" in report.errors, label

    payload = task.model_dump(mode="json")
    payload["metadata"]["compositional_release"]["base_bundle_hash"] = "sha256:" + "3" * 64
    tampered = TaskArtifact.model_validate(payload)
    report = verify_task_artifact(tampered, registry)
    assert not report.passed
    assert "compositional_release_contract_invalid" in report.errors


def test_standalone_release_verifier_replays_once_and_rejects_inventory_tampering(
    tmp_path,
    monkeypatch,
):
    bundle, tasks = _source()
    output = tmp_path / "standalone-release"
    manifest = build_compositional_release(
        tasks,
        {"fixture.json": bundle},
        output,
        seed=47,
    )

    expected_patch_calls = sum(
        len(
            RawFHIRPatchChain.model_validate(
                json.loads((output / cell["patch_chain_path"]).read_text())
            ).patches
        )
        for family in _jsonl(output / "families.jsonl")
        for cell in family["cells"]
    )
    patch_calls = 0
    apply_patch = compositional_release_module.apply_raw_fhir_patch

    def counted_apply(bundle, patch):
        nonlocal patch_calls
        patch_calls += 1
        return apply_patch(bundle, patch)

    monkeypatch.setattr(
        compositional_release_module,
        "apply_raw_fhir_patch",
        counted_apply,
    )
    context = verify_compositional_release_context(output)
    assert patch_calls == expected_patch_calls
    report = context.report
    assert report["passed"] is True
    assert report["release_hash"] == manifest["release_hash"]
    assert report["task_count"] == manifest["counts"]["task_count"]
    assert report["patch_chains_replayed"] is True
    assert report["task_artifacts_replayed"] is True

    unexpected = output / "unexpected.json"
    unexpected.write_text("{}\n")
    with pytest.raises(CompositionalReleaseError, match="inventory is not closed"):
        context.assert_current(output)
    unexpected.unlink()

    with pytest.raises(CompositionalReleaseError, match="path mismatch"):
        context.assert_current(tmp_path / "different-release")

    tasks_path = output / "tasks.jsonl"
    original_tasks = tasks_path.read_bytes()
    tasks_path.write_bytes(original_tasks + b"\n")
    with pytest.raises(CompositionalReleaseError, match="file hash or size mismatch"):
        context.assert_current(output)
    tasks_path.write_bytes(original_tasks)

    manifest_path = output / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text())
    manifest_payload["release_hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest_payload))
    with pytest.raises(CompositionalReleaseError, match="manifest changed"):
        context.assert_current(output)


def test_participant_package_excludes_gold_sources_charts_and_test_ids(tmp_path, monkeypatch):
    bundle, tasks = _source()
    release = tmp_path / "evaluator"
    participant = tmp_path / "participant"
    build_compositional_release(tasks, {"fixture.json": bundle}, release)

    semantic_replays = 0
    original_verify = compositional_release_module.verify_compositional_release

    def counted_verify(path):
        nonlocal semantic_replays
        semantic_replays += 1
        return original_verify(path)

    monkeypatch.setattr(
        compositional_release_module,
        "verify_compositional_release",
        counted_verify,
    )
    manifest = build_compositional_participant_package(release, participant)
    rows = _jsonl(participant / "tasks.jsonl")

    assert manifest["access_classification"] == "public_participant"
    assert {path.name for path in participant.iterdir()} == {"manifest.json", "tasks.jsonl"}
    assert rows
    assert set(rows[0]) == {
        "schema_version",
        "task_id",
        "split",
        "question",
        "allowed_tools",
        "response_schema",
    }
    assert verify_compositional_participant_package(
        participant,
        evaluator_release_directory=release,
    )["passed"]
    assert semantic_replays == 2


def test_sealed_verification_package_can_issue_rechecked_context(tmp_path):
    bundle, tasks = _source()
    release = tmp_path / "evaluator"
    verification_package = tmp_path / "verification"
    build_compositional_release(tasks, {"fixture.json": bundle}, release)
    context = verify_compositional_release_context(release)

    verification_package.mkdir()
    report = {
        "schema_version": "patient-tool-gym.confirmatory-release-verification.v1",
        "passed": True,
        "core_verification": context.report,
    }
    report["verification_hash"] = content_hash(report)
    verification_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    (verification_package / "verification.json").write_bytes(verification_bytes)
    release_manifest_bytes = (release / "manifest.json").read_bytes()
    package = {
        "schema_version": "patient-tool-gym.confirmatory-release-verification-package.v1",
        "source_release": {
            "manifest_sha256": hashlib.sha256(release_manifest_bytes).hexdigest(),
            "path": str(release),
            "release_hash": context.release_hash,
        },
        "verification": {
            "path": "verification.json",
            "sha256": hashlib.sha256(verification_bytes).hexdigest(),
            "verification_hash": report["verification_hash"],
        },
    }
    package["package_hash"] = content_hash(package)
    (verification_package / "manifest.json").write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n"
    )

    loaded = load_compositional_release_verification_context(
        release,
        verification_package,
    )
    assert loaded.report == context.report
    assert loaded.assert_current(release)["passed"] is True


def test_release_is_deterministic_and_publishes_only_after_success(tmp_path):
    bundle, tasks = _source()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_compositional_release(tasks, {"fixture.json": bundle}, first, seed=7)
    second_manifest = build_compositional_release(tasks, {"fixture.json": bundle}, second, seed=7)

    assert first_manifest == second_manifest
    assert (first / "tasks.jsonl").read_bytes() == (second / "tasks.jsonl").read_bytes()
    assert (first / "families.jsonl").read_bytes() == (second / "families.jsonl").read_bytes()
    with pytest.raises(CompositionalReleaseError, match="already exists"):
        build_compositional_release(tasks, {"fixture.json": bundle}, first)


def test_records_procedure_focal_as_medically_unrepresentable_in_v1(tmp_path):
    bundle = _fixture_bundle()
    bundle["entry"].append(
        _entry(
            _procedure(
                "colonoscopy-initial",
                "73761001",
                "Colonoscopy",
                "enc-initial",
                "2026-05-10T09:20:00Z",
            )
        )
    )
    snapshot = normalize_fhir_bundle(bundle, dataset="synthea")
    tasks = generate_patient_p2_tasks((snapshot,)).tasks
    output = tmp_path / "release-with-procedure"

    manifest = build_compositional_release(tasks, {"fixture.json": bundle}, output)
    discovery = json.loads((output / "candidate-discovery.json").read_text())

    assert discovery["matrix_candidate_count"] == 1
    assert discovery["eligible_candidate_count"] == 1
    assert [row["reason"] for row in discovery["matrix_discovery"]["rejections"]] == [
        "REFUTED_DATE_NOT_REPRESENTABLE"
    ]
    assert discovery["release_rejections"] == []
    assert manifest["limitations"]["procedure_focal_interventions_excluded"] is True
    assert manifest["counts"]["superfamily_count"] == 1


def test_preflight_rejects_only_candidates_without_a_strict_raw_distractor(tmp_path):
    complete_bundle, complete_tasks = _source()
    sparse_bundle = json.loads(
        json.dumps(complete_bundle).replace("release-patient", "release-patient-sparse")
    )
    sparse_bundle["id"] = "compositional-release-sparse-fixture"
    sparse_bundle["entry"] = [
        entry
        for entry in sparse_bundle["entry"]
        if entry["resource"].get("resourceType") != "Condition"
        or entry["resource"].get("code", {}).get("coding", [{}])[0].get("code") == "59621000"
    ]
    sparse_tasks = generate_patient_p2_tasks(
        (normalize_fhir_bundle(sparse_bundle, dataset="synthea"),)
    ).tasks
    output = tmp_path / "release-with-sparse-candidate"

    manifest = build_compositional_release(
        (*complete_tasks, *sparse_tasks),
        {
            "complete.json": complete_bundle,
            "sparse.json": sparse_bundle,
        },
        output,
    )
    discovery = json.loads((output / "candidate-discovery.json").read_text())

    assert discovery["matrix_candidate_count"] == 2
    assert discovery["eligible_candidate_count"] == 1
    assert manifest["counts"]["superfamily_count"] == 1
    assert [row["reason"] for row in discovery["release_rejections"]] == ["NO_STRICT_DISTRACTOR"]
    rejection = discovery["release_rejections"][0]
    assert rejection["details"]["requested_tier"] == "same_resource_different_code"
    assert rejection["details"]["required_code_system"] == SNOMED
    assert discovery["raw_bundle_preflight"] == {
        "required_distractor_tier": "same_resource_different_code",
        "same_resource_type": True,
        "same_code_system": True,
        "same_encounter": True,
        "different_code": True,
        "source_records_synthesized_or_moved": False,
    }
    assert discovery["discovery_hash"] == content_hash(
        {key: value for key, value in discovery.items() if key != "discovery_hash"}
    )


def test_path_catalog_skips_nonpatient_synthea_bundles_and_records_them(tmp_path):
    bundle, tasks = _source()
    source = tmp_path / "synthea"
    source.mkdir()
    (source / "patient.json").write_text(json.dumps(bundle))
    (source / "hospitalInformation.json").write_text(
        json.dumps(
            {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [
                    _entry(
                        {
                            "resourceType": "Organization",
                            "id": "hospital-1",
                            "name": "Synthetic Hospital",
                        }
                    )
                ],
            }
        )
    )

    manifest = build_compositional_release(tasks, source, tmp_path / "release")
    catalog = manifest["sources"]["bundles"]["input"]

    assert catalog["input_json_count"] == 2
    assert catalog["bundle_count"] == 1
    assert catalog["skipped_nonpatient_bundle_count"] == 1
    assert [(row["name"], row["reason"]) for row in catalog["skipped"]] == [
        ("hospitalInformation.json", "DECLARED_TRANSACTION_DEPENDENCY")
    ]
    assert len(catalog["skipped"][0]["file_sha256"]) == 64


def test_path_catalog_rejects_a_multi_patient_bundle(tmp_path):
    bundle, tasks = _source()
    bundle["entry"].append(
        _entry(
            {
                "resourceType": "Patient",
                "id": "second-patient",
                "active": True,
            }
        )
    )
    source = tmp_path / "synthea"
    source.mkdir()
    (source / "invalid.json").write_text(json.dumps(bundle))

    with pytest.raises(CompositionalReleaseError, match="2 top-level Patient"):
        build_compositional_release(tasks, source, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_dependency_audit_is_recomputed_from_released_base_bundles(tmp_path):
    bundle, tasks = _source()
    output = tmp_path / "release"
    build_compositional_release(tasks, {"fixture.json": bundle}, output)

    dependency_manifest = json.loads((output / "sources/dependency-manifest.json").read_text())
    assert dependency_manifest["resolution_audit"]["scope"] == (
        "unique_released_patient_base_bundles"
    )
    assert dependency_manifest["resolution_audit"]["patient_bundle_count"] == 1

    tampered = copy.deepcopy(dependency_manifest)
    tampered_audit = tampered["resolution_audit"]
    tampered_audit["patient_bundle_count"] = 999
    tampered_audit["audit_hash"] = content_hash(
        {key: value for key, value in tampered_audit.items() if key != "audit_hash"}
    )
    tampered["dependency_manifest_hash"] = content_hash(
        {key: value for key, value in tampered.items() if key != "dependency_manifest_hash"}
    )

    with pytest.raises(CompositionalReleaseError, match="does not replay"):
        _verify_standalone_dependencies(
            output,
            tampered,
            patient_bundles=[bundle],
        )
