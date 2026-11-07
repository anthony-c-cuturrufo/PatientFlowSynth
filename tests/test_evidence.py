import json
from pathlib import Path

import pytest

from patient_tool_gym.evidence import (
    DuplicateEvidenceIdError,
    PatientBoundaryError,
    load_snapshot,
    snapshot_from_payloads,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_snapshot_hash_and_order_are_independent_of_jsonl_line_order():
    first = load_snapshot(FIXTURES / "patient_a.jsonl")
    reordered = load_snapshot(FIXTURES / "patient_a_reordered.jsonl")

    assert first.snapshot_id == reordered.snapshot_id
    assert first.canonical_json() == reordered.canonical_json()
    assert [record.evidence_id for record in first.records] == sorted(
        record.evidence_id for record in first.records
    )


def test_loader_rejects_duplicate_evidence_ids_without_overwriting():
    with pytest.raises(DuplicateEvidenceIdError) as exc_info:
        load_snapshot(FIXTURES / "duplicate_evidence_id.jsonl")

    assert exc_info.value.evidence_id == "duplicate-id"
    assert exc_info.value.line_numbers == (1, 2)


def test_loader_rejects_records_from_more_than_one_patient():
    with pytest.raises(PatientBoundaryError) as exc_info:
        load_snapshot(FIXTURES / "mixed_patients.jsonl")

    assert exc_info.value.expected_patient_id == "patient-a"
    assert exc_info.value.observed_patient_id == "patient-b"
    assert "Patient B record" not in str(exc_info.value)


def test_semantic_hash_excludes_source_path_but_includes_normalizer_version():
    payloads = [
        json.loads(line)
        for line in (FIXTURES / "patient_a.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    moved = json.loads(json.dumps(payloads))
    for payload in moved:
        payload.setdefault("source", {})["file"] = "/a/different/local/path.json"

    first = snapshot_from_payloads(payloads, normalizer_version="normalizer.v1")
    moved_snapshot = snapshot_from_payloads(moved, normalizer_version="normalizer.v1")
    upgraded = snapshot_from_payloads(payloads, normalizer_version="normalizer.v2")

    assert first.snapshot_id == moved_snapshot.snapshot_id
    assert first.snapshot_id != upgraded.snapshot_id
