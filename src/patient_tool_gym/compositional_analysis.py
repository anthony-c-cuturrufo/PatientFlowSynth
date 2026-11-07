"""Integrity-bound analysis for compositional intervention episodes.

The release contains repeated, matched observations from a small number of patient
superfamilies.  Episode-level rates are useful descriptions, but the factorial
contrasts in this module are first computed within each superfamily and only then
summarized across superfamilies. This prevents repeated task rows from being presented
as independent clinical samples.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from patient_tool_gym.compositional_release import verify_compositional_release
from patient_tool_gym.episodes import EpisodeRecord, load_episode_records, source_task_hash
from patient_tool_gym.evidence_contracts import (
    ObservedToolCall,
    build_complete_search_certificates,
)
from patient_tool_gym.program import TruthState, content_hash
from patient_tool_gym.task import TaskArtifact
from patient_tool_gym.tools import ToolRegistry
from patient_tool_gym.verifier import SubmissionScore, score_submission

ANALYSIS_SCHEMA_VERSION = "patient-tool-gym.compositional-results.v2"
_TASK_METADATA_V1 = "patient-tool-gym.compositional-task-metadata.v1"
_TASK_METADATA_V2 = "patient-tool-gym.compositional-task-metadata.v2"
_SPLIT_SCHEMAS = {
    "patient-tool-gym.compositional-splits.v1",
    "patient-tool-gym.compositional-splits.v2",
}
TOPOLOGIES = ("d1w1", "d2w1", "d1w2", "d2w2")
EVIDENCE_STATES = (
    "supported",
    "refuted",
    "insufficient_evidence",
    "conflicting_evidence",
)
DISTRACTOR_VALUES = (False, True)
TRUTH_STATES = frozenset(state.value for state in TruthState)

_FOCAL_TOOL_RESOURCE_TYPES = {
    "search_conditions": "Condition",
    "search_observations": "Observation",
    "search_procedures": "Procedure",
    "search_immunizations": "Immunization",
    "search_medications": "Medication",
}
_CONTRAST_NAMES = (
    "depth_at_w1",
    "depth_at_w2",
    "depth_main_effect",
    "width_at_d1",
    "width_at_d2",
    "width_main_effect",
    "depth_by_width_interaction",
    "d2w2_minus_d1w1",
)
_FACTORIAL_OUTCOMES = (
    "exact_object_correct",
    "primary_branch_correct",
    "overall_correct",
    "strict_primary_endpoint_pass",
)
_SCORE_COMPARISON_FIELDS = (
    "answer_correct",
    "response_correct",
    "answer_evidence_sufficient",
    "matched_answer_witness_index",
    "answer_evidence_quality_pass",
    "complete_search_certificate_required",
    "complete_search_certificate_valid",
)


def load_verified_release(
    release_directory: str | Path,
) -> tuple[
    dict[str, TaskArtifact],
    dict[str, str],
    dict[str, dict[str, str]],
    dict[str, Any],
]:
    """Verify the closed release and return its analysis bindings.

    ``verify_compositional_release`` checks the manifest hash, every declared file,
    family/task bindings, patch replay, task replay, and split hashes.  This loader then
    independently binds each split assignment to the immutable task content hash.
    """

    root = Path(release_directory)
    verification = verify_compositional_release(root)
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = _load_tasks(root / "tasks.jsonl")
    splits = json.loads((root / "splits.json").read_text())
    task_splits = _split_bindings(splits, tasks)

    family_dimensions: dict[str, dict[str, str]] = {}
    for row in _load_jsonl(root / "families.jsonl"):
        superfamily_id = row.get("superfamily_id")
        candidate = row.get("source_candidate")
        if not isinstance(superfamily_id, str) or not isinstance(candidate, dict):
            raise ValueError("malformed compositional family classification")
        dimensions = {
            "focal_domain": _required_text(candidate, "focal_domain", superfamily_id),
            "focal_resource_type": _required_text(
                candidate, "focal_resource_type", superfamily_id
            ),
            "companion_domain": _required_text(
                candidate, "companion_domain", superfamily_id
            ),
            "companion_resource_type": _required_text(
                candidate, "companion_resource_type", superfamily_id
            ),
        }
        if superfamily_id in family_dimensions:
            raise ValueError(f"duplicate compositional family {superfamily_id}")
        family_dimensions[superfamily_id] = dimensions

    integrity = {
        **verification,
        "manifest_release_hash_recomputed": True,
        "split_task_hashes_recomputed": True,
        "manifest_file_sha256": _sha256_file(root / "manifest.json"),
        "tasks_file_sha256": _sha256_file(root / "tasks.jsonl"),
        "splits_file_sha256": _sha256_file(root / "splits.json"),
    }
    if manifest.get("release_hash") != verification["release_hash"]:
        raise ValueError("verified release hash differs from manifest release hash")
    return tasks, task_splits, family_dimensions, integrity


def _split_bindings(
    splits: Mapping[str, Any], tasks: Mapping[str, TaskArtifact]
) -> dict[str, str]:
    """Bind explicit v1/v2 split assignments to immutable tasks."""

    if splits.get("schema_version") not in _SPLIT_SCHEMAS:
        raise ValueError("unsupported compositional split schema")

    assignments = splits.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("compositional split artifact has no assignments object")
    rows = assignments.get("task_assignments")
    if not isinstance(rows, list):
        raise ValueError("compositional split artifact has no task assignments")
    task_splits: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("compositional split assignment is not an object")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in tasks:
            raise ValueError("split assignment references an absent compositional task")
        if task_id in task_splits:
            raise ValueError(f"duplicate split assignment for {task_id}")
        if row.get("task_hash") != source_task_hash(tasks[task_id]):
            raise ValueError(f"split assignment task hash mismatch for {task_id}")
        split = row.get("split")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"invalid split assignment for {task_id}")
        task_splits[task_id] = str(split)
    if set(task_splits) != set(tasks):
        raise ValueError("split assignments do not exactly cover the release tasks")
    return task_splits


def analyze_compositional_results(
    tasks: Mapping[str, TaskArtifact],
    records: Sequence[EpisodeRecord],
    *,
    release_integrity: Mapping[str, Any],
    task_splits: Mapping[str, str],
    family_dimensions: Mapping[str, Mapping[str, str]],
    registries: Mapping[str, ToolRegistry] | None = None,
    require_complete_release: bool = False,
    require_score_replay: bool = False,
    require_clean_canary: bool = False,
    source_files: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze one model/interface configuration over a release or canary subset."""

    if not records:
        raise ValueError("episode input contains no records")
    task_index = dict(tasks)
    # A compact dataset supplies a lazy mapping backed by a bounded chart cache.
    # Copying it would eagerly reconstruct and retain every chart in the run.
    registry_index = registries if registries is not None else {}
    task_dimensions = {
        task_id: _task_dimensions(
            task,
            split=task_splits.get(task_id),
            family_dimensions=family_dimensions,
        )
        for task_id, task in task_index.items()
    }
    identity = _validate_episode_identity(
        task_index,
        records,
        require_complete_release=require_complete_release,
    )
    configuration = _validate_configuration(records)

    rows = [
        _episode_row(
            task_index[record.task_id],
            record,
            task_dimensions[record.task_id],
            registry=registry_index.get(record.patient_snapshot_id),
            require_score_replay=require_score_replay,
        )
        for record in records
    ]
    score_sources = Counter(str(row["score_source"]) for row in rows)
    if require_score_replay and score_sources.get("full_snapshot_replay", 0) != len(rows):
        raise ValueError("strict score replay was requested but not all episodes were replayed")

    coverage = _canary_cell_coverage(rows)
    gate = _harness_gate(
        rows,
        coverage=coverage,
        release_integrity=release_integrity,
        identity=identity,
        configuration=configuration,
        require_clean_canary=require_clean_canary,
        require_score_replay=require_score_replay,
    )
    report: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "release": dict(release_integrity),
        "identity": identity,
        "configuration": configuration,
        "analysis_policy": {
            "descriptive_episode_unit": "episode",
            "factorial_inference_unit": "patient_superfamily",
            "independent_superfamily_count": len(
                {row["superfamily_id"] for row in rows}
            ),
            "warning": (
                "Episode-level rates are descriptive. Factorial contrasts are computed "
                "within patient superfamilies and summarized across superfamilies; task "
                "rows are not treated as independent clinical samples."
            ),
            "repeat_policy": (
                "Repeated episodes are averaged within task before family-level contrasts."
            ),
            "strict_primary_endpoint_definition": (
                "completed episode AND exact structured response correctness AND exact match "
                "to one acceptable answer-evidence witness AND answer-evidence-quality pass "
                "AND, when required, a valid complete-search certificate"
            ),
            "contrast_scale": (
                "Absolute change in pass probability; positive values favor greater "
                "depth/width."
            ),
        },
        "overall": _core_summary(rows),
        "by_topology": _group_summaries(rows, "topology"),
        "by_evidence_state": _group_summaries(rows, "evidence_state"),
        "by_companion_evidence_state": _group_summaries(
            rows, "companion_evidence_state"
        ),
        "by_distractor_injection": _group_summaries(
            rows, "distractor_injection_label"
        ),
        "by_split": _group_summaries(rows, "split"),
        "by_focal_resource_domain": _group_summaries(rows, "focal_domain"),
        "by_focal_resource_type": _group_summaries(rows, "focal_resource_type"),
        "by_superfamily": _group_summaries(rows, "superfamily_id"),
        "answer_state_confusion": _answer_confusion(rows),
        "distractor_pairs": _distractor_pairs(rows),
        "family_level_factorial_contrasts": _factorial_contrasts(rows),
        "family_level_companion_state_contrasts": _companion_state_contrasts(rows),
        "canary_cell_coverage": coverage,
        "gate": gate,
        "score_sources": dict(sorted(score_sources.items())),
        "resource_use": _resource_use(rows),
        "source_files": dict(source_files or {}),
        "row_count": len(rows),
        "unique_task_count": len({row["task_id"] for row in rows}),
    }
    report["report_payload_content_hash"] = content_hash(report)
    return report


def render_compositional_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to the JSON report."""

    overall = report["overall"]
    coverage = report["canary_cell_coverage"]
    release = report["release"]
    lines = [
        "# Compositional GPT results",
        "",
        f"- Release: `{release['release_hash']}`",
        f"- Harness gate: {'PASS' if report['gate']['passed'] else 'FAIL'}",
        f"- Episodes: {report['row_count']} across {report['unique_task_count']} tasks",
        (
            "- Independent analysis units: "
            f"{report['analysis_policy']['independent_superfamily_count']} patient superfamilies"
        ),
        (
            "- Factorial-cell coverage: "
            f"{coverage['covered_cell_count']}/{coverage['expected_cell_count']}"
        ),
        "",
        "## Main endpoints",
        "",
        "| Endpoint | Passes | Rate |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Completed", "completion"),
        ("Valid answer schema", "answer_schema_valid"),
        ("Exact structured answer", "exact_object_accuracy"),
        ("Primary branch", "primary_branch_accuracy"),
        ("Overall state", "overall_accuracy"),
        ("Exact acceptable evidence witness", "evidence_witness_success"),
        ("Answer evidence-quality pass", "answer_evidence_quality_pass"),
        ("Strict primary endpoint", "strict_primary_endpoint"),
    ):
        metric = overall[key]
        rate = "NA" if metric["rate"] is None else f"{100 * metric['rate']:.1f}%"
        lines.append(f"| {label} | {metric['pass_count']}/{metric['observed']} | {rate} |")

    lines.extend(
        [
            "",
            "## Family-level strict-endpoint contrasts",
            "",
            "Positive values indicate higher pass probability for the more complex topology.",
            "",
            "| Contrast | Mean change | Families |",
            "|---|---:|---:|",
        ]
    )
    strict = report["family_level_factorial_contrasts"]["outcomes"][
        "strict_primary_endpoint_pass"
    ]["overall"]
    for name in _CONTRAST_NAMES:
        summary = strict[name]
        value = "NA" if summary["mean"] is None else f"{summary['mean']:+.3f}"
        lines.append(f"| `{name}` | {value} | {summary['family_count']} |")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            report["analysis_policy"]["warning"],
            "",
        ]
    )
    return "\n".join(lines)


def load_registries(paths: Sequence[str | Path]) -> dict[str, ToolRegistry]:
    """Load normalized snapshot JSON files or directories into a snapshot index."""

    from patient_tool_gym.evidence import load_snapshot

    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.json")))
        else:
            expanded.append(path)
    registries: dict[str, ToolRegistry] = {}
    for path in expanded:
        snapshot = load_snapshot(path)
        if snapshot.snapshot_id in registries:
            raise ValueError(f"duplicate normalized snapshot ID supplied by {path}")
        registries[snapshot.snapshot_id] = ToolRegistry(snapshot)
    return registries


def load_episode_file(path: str | Path) -> list[EpisodeRecord]:
    return list(load_episode_records(path).values())


def _load_tasks(path: Path) -> dict[str, TaskArtifact]:
    tasks: dict[str, TaskArtifact] = {}
    for line_number, row in enumerate(_load_jsonl(path), start=1):
        try:
            task = TaskArtifact.model_validate(row)
        except ValueError as exc:
            raise ValueError(f"invalid task record at line {line_number}") from exc
        if task.task_id in tasks:
            raise ValueError(f"duplicate task ID at line {line_number}: {task.task_id}")
        tasks[task.task_id] = task
    if not tasks:
        raise ValueError("compositional task file contains no tasks")
    return tasks


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _validate_episode_identity(
    tasks: Mapping[str, TaskArtifact],
    records: Sequence[EpisodeRecord],
    *,
    require_complete_release: bool,
) -> dict[str, Any]:
    episode_ids = [record.episode_id for record in records]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode input contains duplicate episode IDs")
    unexpected = sorted({record.task_id for record in records} - set(tasks))
    if unexpected:
        raise ValueError(f"episodes reference absent release tasks: {unexpected[:3]}")

    hash_mismatches = []
    snapshot_mismatches = []
    question_mismatches = []
    for record in records:
        task = tasks[record.task_id]
        if record.source_task_hash != source_task_hash(task):
            hash_mismatches.append(record.task_id)
        if record.patient_snapshot_id != task.patient_snapshot_id:
            snapshot_mismatches.append(record.task_id)
        if record.question != task.question:
            question_mismatches.append(record.task_id)
    if hash_mismatches:
        raise ValueError(f"source task hash mismatch for {sorted(set(hash_mismatches))[:3]}")
    if snapshot_mismatches:
        raise ValueError(f"snapshot binding mismatch for {sorted(set(snapshot_mismatches))[:3]}")
    if question_mismatches:
        raise ValueError(f"question binding mismatch for {sorted(set(question_mismatches))[:3]}")

    counts = Counter(record.task_id for record in records)
    missing = sorted(set(tasks) - set(counts))
    if require_complete_release and missing:
        raise ValueError(f"complete release analysis is missing {len(missing)} tasks")
    repeat_distribution = Counter(counts.values())
    return {
        "passed": True,
        "release_task_count": len(tasks),
        "episode_count": len(records),
        "observed_task_count": len(counts),
        "missing_release_task_count": len(missing),
        "missing_release_task_ids": missing,
        "episodes_per_observed_task": {
            str(count): task_count for count, task_count in sorted(repeat_distribution.items())
        },
        "source_task_hashes_match": True,
        "snapshot_bindings_match": True,
        "question_bindings_match": True,
        "episode_ids_unique": True,
        "complete_release_required": require_complete_release,
        "complete_release_observed": not missing,
    }


def _validate_configuration(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    signatures: dict[str, dict[str, Any]] = {}
    for record in records:
        descriptor = record.adapter_descriptor
        payload: dict[str, Any] = {
            "adapter_id": record.adapter_id,
            "model": record.model,
            "diagnostic_only": record.diagnostic_only,
            "limits": record.limits.model_dump(mode="json"),
        }
        if descriptor is not None:
            payload["adapter_descriptor"] = descriptor.model_dump(mode="json")
        else:
            # Legacy records cannot separate a seed change from an interface change.
            payload["legacy_config_hash"] = record.config_hash
        signatures.setdefault(content_hash(payload), payload)
    if len(signatures) != 1:
        raise ValueError(
            "analysis input mixes model/interface configurations; analyze each configuration "
            "separately"
        )
    signature, payload = next(iter(signatures.items()))
    return {
        "homogeneous": True,
        "analysis_configuration_hash": signature,
        "configuration": payload,
        "seeds": sorted(
            {record.seed for record in records},
            key=lambda value: (value is None, value),
        ),
        "episode_config_hashes": sorted({record.config_hash for record in records}),
    }


def _task_dimensions(
    task: TaskArtifact,
    *,
    split: str | None,
    family_dimensions: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    metadata = task.metadata.get("compositional_release")
    if not isinstance(metadata, dict):
        raise ValueError(f"task {task.task_id} is not a compositional-release task")
    topology = _required_text(metadata, "topology", task.task_id)
    evidence_state = _required_text(metadata, "evidence_state", task.task_id)
    metadata_schema = _required_text(metadata, "schema_version", task.task_id)
    superfamily_id = _required_text(metadata, "superfamily_id", task.task_id)
    patient_id = _required_text(metadata, "patient_id", task.task_id)
    cell_id = _required_text(metadata, "cell_id", task.task_id)
    if metadata_schema == _TASK_METADATA_V2:
        companion_evidence_state = _required_text(
            metadata, "companion_evidence_state", task.task_id
        )
        distractor_injected = metadata.get("distractor_injected")
        if "distractor_present" in metadata:
            raise ValueError(
                f"task {task.task_id} v2 metadata contains the retired distractor field"
            )
    elif metadata_schema == _TASK_METADATA_V1:
        # Frozen v1 releases had a fixed-supported companion and named the injection
        # coordinate as if it described absolute chart presence.
        companion_evidence_state = "supported"
        distractor_injected = metadata.get("distractor_present")
    else:
        raise ValueError(
            f"task {task.task_id} has unsupported compositional metadata schema"
        )
    if topology not in TOPOLOGIES:
        raise ValueError(f"task {task.task_id} has invalid topology {topology!r}")
    if evidence_state not in EVIDENCE_STATES:
        raise ValueError(f"task {task.task_id} has invalid evidence state {evidence_state!r}")
    if companion_evidence_state not in EVIDENCE_STATES:
        raise ValueError(
            f"task {task.task_id} has invalid companion state "
            f"{companion_evidence_state!r}"
        )
    if topology.endswith("w1") and companion_evidence_state != "supported":
        raise ValueError(f"task {task.task_id} width-one companion state is not supported")
    if not isinstance(distractor_injected, bool):
        raise ValueError(f"task {task.task_id} has no Boolean distractor injection")
    if task.family_id != f"{superfamily_id}-{topology}":
        raise ValueError(f"task {task.task_id} family/topology binding is inconsistent")
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"task {task.task_id} has no valid release split")

    expected_branches = metadata.get("expected_branch_states")
    expected_overall = metadata.get("expected_overall_state")
    expected_keys = {"primary"} | ({"companion"} if topology.endswith("w2") else set())
    if (
        not isinstance(expected_branches, dict)
        or set(expected_branches) != expected_keys
        or any(value not in TRUTH_STATES for value in expected_branches.values())
        or expected_overall not in TRUTH_STATES
    ):
        raise ValueError(f"task {task.task_id} has an invalid expected answer classification")
    if expected_branches.get("primary") != evidence_state or (
        topology.endswith("w2")
        and expected_branches.get("companion") != companion_evidence_state
    ):
        raise ValueError(
            f"task {task.task_id} factor coordinates disagree with expected branches"
        )
    expected_answer = {"branches": expected_branches, "overall": expected_overall}
    if task.gold_answer.kind != "object" or task.gold_answer.value != expected_answer:
        raise ValueError(f"task {task.task_id} gold answer disagrees with compositional metadata")

    family = family_dimensions.get(superfamily_id)
    if family is None:
        raise ValueError(f"task {task.task_id} references an absent superfamily")
    inferred_resource_type = _infer_focal_resource_type(task)
    if inferred_resource_type != family.get("focal_resource_type"):
        raise ValueError(f"task {task.task_id} focal resource classification is inconsistent")
    return {
        "topology": topology,
        "evidence_state": evidence_state,
        "companion_evidence_state": companion_evidence_state,
        "distractor_injected": distractor_injected,
        "distractor_injection_label": (
            "injected" if distractor_injected else "not_injected"
        ),
        "factorial_schema": "v2" if metadata_schema == _TASK_METADATA_V2 else "v1",
        "superfamily_id": superfamily_id,
        "patient_id": patient_id,
        "cell_id": cell_id,
        "split": split,
        **dict(family),
    }


def _infer_focal_resource_type(task: TaskArtifact) -> str:
    focal = [
        node
        for node in task.program.nodes
        if getattr(node, "id", None) == "focal_search"
    ]
    if len(focal) != 1:
        raise ValueError(f"task {task.task_id} does not contain one focal search node")
    tool = getattr(focal[0], "tool", None)
    resource_type = _FOCAL_TOOL_RESOURCE_TYPES.get(str(tool))
    if resource_type is None:
        raise ValueError(f"task {task.task_id} uses an unsupported focal search tool")
    return resource_type


def _episode_row(
    task: TaskArtifact,
    record: EpisodeRecord,
    dimensions: Mapping[str, Any],
    *,
    registry: ToolRegistry | None,
    require_score_replay: bool,
) -> dict[str, Any]:
    schema_valid, observed_branches, observed_overall = _structured_answer(record, task)
    expected = task.gold_answer.value
    exact = bool(schema_valid and record.final_answer == task.gold_answer)
    primary_correct = bool(
        schema_valid and observed_branches.get("primary") == expected["branches"]["primary"]
    )
    companion_correct = (
        bool(
            schema_valid
            and observed_branches.get("companion") == expected["branches"]["companion"]
        )
        if "companion" in expected["branches"]
        else None
    )
    overall_correct = bool(schema_valid and observed_overall == expected["overall"])

    witnesses = tuple(set(witness) for witness in task.answer_sufficient_evidence_witnesses())
    submitted = set(record.evidence_ids)
    matched_witness_index = next(
        (index for index, witness in enumerate(witnesses) if submitted == witness), None
    )
    evidence_witness_success = matched_witness_index is not None

    score, score_source = _episode_score(task, record, registry=registry)
    if require_score_replay and score_source != "full_snapshot_replay":
        raise ValueError(f"episode {record.episode_id} has no normalized snapshot for replay")
    if score is not None:
        if bool(score.answer_correct) != exact:
            raise ValueError(f"episode {record.episode_id} score disagrees with exact answer")
        if score.response_correct is not None and bool(score.response_correct) != exact:
            raise ValueError(f"episode {record.episode_id} response score disagrees with answer")
        if score.matched_answer_witness_index != matched_witness_index:
            raise ValueError(f"episode {record.episode_id} score disagrees with evidence witness")

    quality_pass = score.answer_evidence_quality_pass if score is not None else None
    complete_search_required = bool(
        task.evidence_quality_contract is not None
        and task.evidence_quality_contract.complete_search_requirements
    )
    certificate_valid = score.complete_search_certificate_valid if score is not None else None
    complete_search_compliant = (
        certificate_valid is True if complete_search_required else True
    )
    strict = bool(
        record.status == "completed"
        and exact
        and evidence_witness_success
        and quality_pass is True
        and complete_search_compliant
    )

    return {
        "task_id": task.task_id,
        "episode_id": record.episode_id,
        "run_key": record.config_hash,
        **dict(dimensions),
        "status": record.status,
        "answer_present": record.final_answer is not None,
        "answer_schema_valid": schema_valid,
        "exact_object_correct": exact,
        "response_correct": exact,
        "primary_branch_correct": primary_correct,
        "companion_branch_correct": companion_correct,
        "overall_correct": overall_correct,
        "expected_branches": expected["branches"],
        "observed_branches": observed_branches if schema_valid else None,
        "expected_overall": expected["overall"],
        "observed_overall": observed_overall if schema_valid else None,
        "observed_answer": record.final_answer.value if record.final_answer is not None else None,
        "evidence_witness_success": evidence_witness_success,
        "matched_answer_witness_index": matched_witness_index,
        "answer_evidence_quality_pass": quality_pass,
        "complete_search_required": complete_search_required,
        "complete_search_certificate_valid": certificate_valid,
        "complete_search_compliant": complete_search_compliant,
        "strict_primary_endpoint_pass": strict,
        "score_source": score_source,
        "error_codes": sorted({error.code for error in record.errors}),
        "tool_calls": record.tool_call_count,
        "turns": record.turns,
        "latency_ms": record.latency_ms,
        "input_tokens": record.tokens.input_tokens,
        "output_tokens": record.tokens.output_tokens,
        "total_tokens": record.tokens.total_tokens,
        "responses": record.tokens.responses,
    }


def _structured_answer(
    record: EpisodeRecord, task: TaskArtifact
) -> tuple[bool, dict[str, str], str | None]:
    answer = record.final_answer
    if answer is None or answer.kind != "object" or not isinstance(answer.value, dict):
        return False, {}, None
    value = answer.value
    if set(value) != {"branches", "overall"}:
        return False, {}, None
    branches = value.get("branches")
    overall = value.get("overall")
    expected_branches = task.gold_answer.value["branches"]
    if (
        not isinstance(branches, dict)
        or set(branches) != set(expected_branches)
        or any(not isinstance(item, str) or item not in TRUTH_STATES for item in branches.values())
        or not isinstance(overall, str)
        or overall not in TRUTH_STATES
    ):
        return False, {}, None
    return True, dict(branches), overall


def _episode_score(
    task: TaskArtifact,
    record: EpisodeRecord,
    *,
    registry: ToolRegistry | None,
) -> tuple[SubmissionScore | None, str]:
    if record.final_answer is None:
        return None, "submission_missing"
    if registry is None:
        score_source = "stored_score_unreplayed" if record.score is not None else "score_missing"
        return record.score, score_source
    if registry.snapshot_id != task.patient_snapshot_id:
        raise ValueError(f"snapshot registry mismatch for task {task.task_id}")
    observations = _observed_tool_calls(record)
    certificates = build_complete_search_certificates(task, registry, observations)
    replayed = score_submission(
        task,
        answer=record.final_answer,
        evidence_ids=record.evidence_ids,
        registry=registry,
        complete_search_certificates=certificates,
        observed_tool_calls=observations,
    )
    if record.score is None:
        raise ValueError(f"episode {record.episode_id} has a submission but no stored score")
    mismatches = [
        field
        for field in _SCORE_COMPARISON_FIELDS
        if getattr(record.score, field) != getattr(replayed, field)
    ]
    if mismatches:
        raise ValueError(
            f"episode {record.episode_id} stored/replayed score mismatch: {mismatches}"
        )
    return replayed, "full_snapshot_replay"


def _observed_tool_calls(record: EpisodeRecord) -> tuple[ObservedToolCall, ...]:
    executed = tuple(call for call in record.tool_calls if call.executed)
    if len(executed) != len(record.tool_results):
        raise ValueError(f"episode {record.episode_id} has inconsistent tool-call accounting")
    observations = []
    for call, result in zip(executed, record.tool_results, strict=True):
        if call.call_id != result.call_id or call.name != result.name:
            raise ValueError(f"episode {record.episode_id} has misaligned tool results")
        observations.append(
            ObservedToolCall(
                call_id=call.call_id,
                tool=call.name,
                arguments=call.arguments,
                result=result.result,
            )
        )
    return tuple(observations)


def _core_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(rows),
        "unique_task_count": len({row["task_id"] for row in rows}),
        "completion": _binary_summary(row["status"] == "completed" for row in rows),
        "answer_present": _binary_summary(row["answer_present"] for row in rows),
        "answer_schema_valid": _binary_summary(row["answer_schema_valid"] for row in rows),
        "exact_object_accuracy": _binary_summary(
            row["exact_object_correct"] for row in rows
        ),
        "response_accuracy": _binary_summary(row["response_correct"] for row in rows),
        "primary_branch_accuracy": _binary_summary(
            row["primary_branch_correct"] for row in rows
        ),
        "companion_branch_accuracy": _binary_summary(
            row["companion_branch_correct"] for row in rows
        ),
        "overall_accuracy": _binary_summary(row["overall_correct"] for row in rows),
        "evidence_witness_success": _binary_summary(
            row["evidence_witness_success"] for row in rows
        ),
        "answer_evidence_quality_pass": _binary_summary(
            row["answer_evidence_quality_pass"] for row in rows
        ),
        "complete_search_certificate_valid": _binary_summary(
            row["complete_search_certificate_valid"]
            for row in rows
            if row["complete_search_required"]
        ),
        "strict_primary_endpoint": _binary_summary(
            row["strict_primary_endpoint_pass"] for row in rows
        ),
        "episode_error_codes": dict(
            sorted(Counter(code for row in rows for code in row["error_codes"]).items())
        ),
        "resource_use": _resource_use(rows),
    }


def _binary_summary(values: Any) -> dict[str, Any]:
    materialized = list(values)
    observed = [bool(value) for value in materialized if value is not None]
    passed = sum(observed)
    return {
        "pass_count": passed,
        "fail_count": len(observed) - passed,
        "observed": len(observed),
        "missing": len(materialized) - len(observed),
        "rate": _round(passed / len(observed)) if observed else None,
    }


def _group_summaries(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _core_summary(selected) for name, selected in sorted(groups.items())}


def _answer_confusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    branch_confusion: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    overall: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        observed_branches = row["observed_branches"] or {}
        for branch, expected in row["expected_branches"].items():
            observed = observed_branches.get(branch, "__missing_or_invalid__")
            branch_confusion[branch][str(expected)][str(observed)] += 1
        observed_overall = row["observed_overall"] or "__missing_or_invalid__"
        overall[str(row["expected_overall"])][str(observed_overall)] += 1
    return {
        "branches": {
            branch: {
                expected: dict(sorted(counts.items()))
                for expected, counts in sorted(expected_rows.items())
            }
            for branch, expected_rows in sorted(branch_confusion.items())
        },
        "overall": {
            expected: dict(sorted(counts.items()))
            for expected, counts in sorted(overall.items())
        },
    }


def _distractor_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[
        tuple[str, str, str, str, str], dict[bool, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["run_key"]),
            str(row["superfamily_id"]),
            str(row["topology"]),
            str(row["evidence_state"]),
            str(row["companion_evidence_state"]),
        )
        distractor = bool(row["distractor_injected"])
        if distractor in grouped[key]:
            raise ValueError("multiple episodes occupy one distractor-pair cell")
        grouped[key][distractor] = row

    pair_rows = []
    for key, pair in sorted(grouped.items()):
        if set(pair) != {False, True}:
            continue
        absent, present = pair[False], pair[True]
        comparable = bool(absent["answer_schema_valid"] and present["answer_schema_valid"])
        invariant = (
            absent["observed_answer"] == present["observed_answer"] if comparable else None
        )
        pair_rows.append(
            {
                "run_key": key[0],
                "superfamily_id": key[1],
                "topology": key[2],
                "evidence_state": key[3],
                "companion_evidence_state": key[4],
                "answer_comparable": comparable,
                "answer_invariant": invariant,
                "answer_flip": None if invariant is None else not invariant,
                "correct_to_incorrect": bool(
                    absent["exact_object_correct"] and not present["exact_object_correct"]
                ),
                "incorrect_to_correct": bool(
                    not absent["exact_object_correct"] and present["exact_object_correct"]
                ),
                "strict_pass_to_fail": bool(
                    absent["strict_primary_endpoint_pass"]
                    and not present["strict_primary_endpoint_pass"]
                ),
                "exact_accuracy_difference": int(present["exact_object_correct"])
                - int(absent["exact_object_correct"]),
                "strict_endpoint_difference": int(present["strict_primary_endpoint_pass"])
                - int(absent["strict_primary_endpoint_pass"]),
            }
        )

    def summary(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "pair_count": len(selected),
            "answer_comparable": _binary_summary(
                row["answer_comparable"] for row in selected
            ),
            "answer_invariance": _binary_summary(
                row["answer_invariant"] for row in selected
            ),
            "answer_flip": _binary_summary(row["answer_flip"] for row in selected),
            "correct_to_incorrect": _binary_summary(
                row["correct_to_incorrect"] for row in selected
            ),
            "incorrect_to_correct": _binary_summary(
                row["incorrect_to_correct"] for row in selected
            ),
            "strict_pass_to_fail": _binary_summary(
                row["strict_pass_to_fail"] for row in selected
            ),
        }

    by_family_exact: dict[str, list[float]] = defaultdict(list)
    by_family_strict: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        by_family_exact[str(row["superfamily_id"])].append(
            float(row["exact_accuracy_difference"])
        )
        by_family_strict[str(row["superfamily_id"])].append(
            float(row["strict_endpoint_difference"])
        )
    return {
        "pairing_key": (
            "run_config_hash × superfamily × topology × primary_evidence_state × "
            "companion_evidence_state"
        ),
        "descriptive_pair_summary": summary(pair_rows),
        "by_topology": _pair_groups(pair_rows, "topology", summary),
        "by_evidence_state": _pair_groups(pair_rows, "evidence_state", summary),
        "by_companion_evidence_state": _pair_groups(
            pair_rows, "companion_evidence_state", summary
        ),
        "family_level_distractor_effect": {
            "exact_object_accuracy": _family_value_summary(
                {family: mean(values) for family, values in by_family_exact.items()}
            ),
            "strict_primary_endpoint": _family_value_summary(
                {family: mean(values) for family, values in by_family_strict.items()}
            ),
            "definition": (
                "mean(injected - not_injected) within each patient superfamily"
            ),
        },
    }


def _pair_groups(rows: Sequence[Mapping[str, Any]], key: str, summarize: Any) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarize(selected) for name, selected in sorted(groups.items())}


def _factorial_contrasts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_rows = _average_repeats_within_task(rows)
    outcomes: dict[str, Any] = {}
    for outcome in _FACTORIAL_OUTCOMES:
        observations = _contrast_observations(task_rows, outcome)
        outcomes[outcome] = {
            "complete_family_primary_state_injection_blocks": len(observations),
            "overall": _summarize_contrasts(observations, group_fields=()),
            "by_evidence_state": {
                state: _summarize_contrasts(
                    [row for row in observations if row["evidence_state"] == state],
                    group_fields=("superfamily_id",),
                )
                for state in EVIDENCE_STATES
            },
            "by_companion_evidence_state": {
                state: _summarize_contrasts(
                    [
                        row
                        for row in observations
                        if row["companion_evidence_state"] == state
                    ],
                    group_fields=("superfamily_id",),
                )
                for state in EVIDENCE_STATES
            },
            "by_distractor_injection": {
                label: _summarize_contrasts(
                    [
                        row
                        for row in observations
                        if row["distractor_injection_label"] == label
                    ],
                    group_fields=("superfamily_id",),
                )
                for label in ("not_injected", "injected")
            },
            "state_interactions_vs_supported": _state_interactions(observations),
        }
    return {
        "statistical_unit": "patient_superfamily",
        "task_repeat_reduction": "mean within immutable task",
        "cell_contrast_definition": {
            "depth_at_w1": "D2W1 - D1W1",
            "depth_at_w2": "D2W2 - D1W2",
            "depth_main_effect": "mean(depth_at_w1, depth_at_w2)",
            "width_at_d1": "D1W2 - D1W1",
            "width_at_d2": "D2W2 - D2W1",
            "width_main_effect": "mean(width_at_d1, width_at_d2)",
            "depth_by_width_interaction": "D2W2 - D2W1 - D1W2 + D1W1",
            "d2w2_minus_d1w1": "D2W2 - D1W1",
        },
        "outcomes": outcomes,
    }


def _companion_state_contrasts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare each non-supported companion state with its matched supported chart."""

    task_rows = _average_repeats_within_task(rows)
    blocks: dict[
        tuple[str, str, str, bool], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in task_rows:
        if not str(row["topology"]).endswith("w2"):
            continue
        key = (
            str(row["superfamily_id"]),
            str(row["topology"]),
            str(row["evidence_state"]),
            bool(row["distractor_injected"]),
        )
        companion_state = str(row["companion_evidence_state"])
        if companion_state in blocks[key]:
            raise ValueError("multiple tasks occupy one companion-state factorial cell")
        blocks[key][companion_state] = row

    outcome_reports: dict[str, Any] = {}
    for outcome in _FACTORIAL_OUTCOMES:
        observations: list[dict[str, Any]] = []
        for (family, topology, primary_state, injected), state_rows in sorted(
            blocks.items()
        ):
            if set(state_rows) != set(EVIDENCE_STATES):
                continue
            supported = float(state_rows["supported"][outcome])
            for companion_state in EVIDENCE_STATES:
                if companion_state == "supported":
                    continue
                observations.append(
                    {
                        "superfamily_id": family,
                        "topology": topology,
                        "evidence_state": primary_state,
                        "companion_evidence_state": companion_state,
                        "distractor_injected": injected,
                        "difference_vs_supported": (
                            float(state_rows[companion_state][outcome]) - supported
                        ),
                    }
                )

        def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            by_family: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                by_family[str(row["superfamily_id"])].append(
                    float(row["difference_vs_supported"])
                )
            return _family_value_summary(
                {family: mean(values) for family, values in by_family.items()}
            )

        outcome_reports[outcome] = {
            "complete_matched_blocks": len(observations) // 3,
            "overall": summarize(observations),
            "by_companion_evidence_state": {
                state: summarize(
                    [
                        row
                        for row in observations
                        if row["companion_evidence_state"] == state
                    ]
                )
                for state in EVIDENCE_STATES
                if state != "supported"
            },
            "by_topology": {
                topology: summarize(
                    [row for row in observations if row["topology"] == topology]
                )
                for topology in ("d1w2", "d2w2")
            },
        }
    return {
        "statistical_unit": "patient_superfamily",
        "reference_state": "supported",
        "contrast_definition": (
            "outcome in named companion state minus the matched supported-companion "
            "outcome within family, topology, primary state, and injection cell"
        ),
        "outcomes": outcome_reports,
    }


def _average_repeats_within_task(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    averaged = []
    for task_id, selected in sorted(grouped.items()):
        first = selected[0]
        dimensions = (
            "topology",
            "evidence_state",
            "companion_evidence_state",
            "distractor_injected",
            "distractor_injection_label",
            "factorial_schema",
            "superfamily_id",
            "patient_id",
            "cell_id",
            "split",
        )
        if any(any(row[key] != first[key] for key in dimensions) for row in selected[1:]):
            raise ValueError(f"repeated task {task_id} changed factorial classification")
        averaged.append(
            {
                "task_id": task_id,
                **{key: first[key] for key in dimensions},
                **{
                    outcome: mean(float(row[outcome]) for row in selected)
                    for outcome in _FACTORIAL_OUTCOMES
                },
                "repeat_count": len(selected),
            }
        )
    return averaged


def _contrast_observations(
    task_rows: Sequence[Mapping[str, Any]], outcome: str
) -> list[dict[str, Any]]:
    blocks: dict[tuple[str, str, str, bool], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in task_rows:
        key = (
            str(row["superfamily_id"]),
            str(row["evidence_state"]),
            str(row["companion_evidence_state"]),
            bool(row["distractor_injected"]),
        )
        topology = str(row["topology"])
        if topology in blocks[key]:
            raise ValueError("multiple tasks occupy one superfamily factorial cell")
        blocks[key][topology] = row

    observations = []
    for (family, state, companion_state, distractor), topology_rows in sorted(
        blocks.items()
    ):
        if set(topology_rows) != set(TOPOLOGIES):
            continue
        values = {topology: float(topology_rows[topology][outcome]) for topology in TOPOLOGIES}
        contrasts = _contrasts(values)
        observations.append(
            {
                "superfamily_id": family,
                "evidence_state": state,
                "companion_evidence_state": companion_state,
                "distractor_injected": distractor,
                "distractor_injection_label": (
                    "injected" if distractor else "not_injected"
                ),
                "topology_values": values,
                **contrasts,
            }
        )
    return observations


def _contrasts(values: Mapping[str, float]) -> dict[str, float]:
    depth_w1 = values["d2w1"] - values["d1w1"]
    depth_w2 = values["d2w2"] - values["d1w2"]
    width_d1 = values["d1w2"] - values["d1w1"]
    width_d2 = values["d2w2"] - values["d2w1"]
    return {
        "depth_at_w1": depth_w1,
        "depth_at_w2": depth_w2,
        "depth_main_effect": mean((depth_w1, depth_w2)),
        "width_at_d1": width_d1,
        "width_at_d2": width_d2,
        "width_main_effect": mean((width_d1, width_d2)),
        "depth_by_width_interaction": values["d2w2"]
        - values["d2w1"]
        - values["d1w2"]
        + values["d1w1"],
        "d2w2_minus_d1w1": values["d2w2"] - values["d1w1"],
    }


def _summarize_contrasts(
    rows: Sequence[Mapping[str, Any]], *, group_fields: tuple[str, ...]
) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields) or (str(row["superfamily_id"]),)
        grouped[key].append(row)
    # Even an "overall" summary is reduced to one value per family.
    if not group_fields:
        grouped = defaultdict(list)
        for row in rows:
            grouped[(str(row["superfamily_id"]),)].append(row)

    result: dict[str, Any] = {}
    for contrast in _CONTRAST_NAMES:
        family_values = {
            "|".join(key): mean(float(row[contrast]) for row in selected)
            for key, selected in grouped.items()
        }
        result[contrast] = _family_value_summary(family_values)
    return result


def _state_interactions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    family_state: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family_state[(str(row["superfamily_id"]), str(row["evidence_state"]))].append(row)
    state_effects: dict[tuple[str, str], dict[str, float]] = {}
    for key, selected in family_state.items():
        state_effects[key] = {
            contrast: mean(float(row[contrast]) for row in selected)
            for contrast in _CONTRAST_NAMES
        }
    result: dict[str, Any] = {}
    for state in EVIDENCE_STATES:
        if state == "supported":
            continue
        result[state] = {}
        for contrast in _CONTRAST_NAMES:
            values = {
                family: effects[contrast]
                - state_effects[(family, "supported")][contrast]
                for (family, candidate_state), effects in state_effects.items()
                if candidate_state == state and (family, "supported") in state_effects
            }
            result[state][contrast] = _family_value_summary(values)
    return {
        "definition": (
            "topology contrast in named state minus the same-family "
            "supported-state contrast"
        ),
        "interactions": result,
    }


def _family_value_summary(values: Mapping[str, float]) -> dict[str, Any]:
    ordered = sorted((str(family), float(value)) for family, value in values.items())
    numbers = [value for _, value in ordered]
    return {
        "family_count": len(numbers),
        "mean": _round(mean(numbers)) if numbers else None,
        "median": _round(median(numbers)) if numbers else None,
        "minimum": _round(min(numbers)) if numbers else None,
        "maximum": _round(max(numbers)) if numbers else None,
        "family_values": [
            {"superfamily_id": family, "value": _round(value)} for family, value in ordered
        ],
    }


def _harness_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    coverage: Mapping[str, Any],
    release_integrity: Mapping[str, Any],
    identity: Mapping[str, Any],
    configuration: Mapping[str, Any],
    require_clean_canary: bool,
    require_score_replay: bool,
) -> dict[str, Any]:
    """Separate experimental validity failures from scientific model outcomes."""

    unique_task_count = len({row["task_id"] for row in rows})
    clean_canary_shape_required = (
        require_clean_canary
        or unique_task_count <= int(coverage["expected_cell_count"])
    )
    cell_coverage_pass = bool(coverage["all_factorial_cells_covered"])
    if clean_canary_shape_required:
        cell_coverage_pass = cell_coverage_pass and bool(
            coverage["balanced_one_task_per_cell"]
        ) and bool(coverage["balanced_one_episode_per_cell"])

    terminal_pass = all(row["status"] == "completed" for row in rows)
    schema_pass = all(row["answer_schema_valid"] for row in rows)
    error_counts = Counter(code for row in rows for code in row["error_codes"])
    error_free = not error_counts
    score_fields_computable = all(
        row["score_source"] not in {"submission_missing", "score_missing"}
        and row["answer_evidence_quality_pass"] is not None
        and (
            not row["complete_search_required"]
            or row["complete_search_certificate_valid"] is not None
        )
        for row in rows
    )
    replay_pass = not require_score_replay or all(
        row["score_source"] == "full_snapshot_replay" for row in rows
    )
    checks = {
        "verified_release_integrity": bool(release_integrity.get("passed")),
        "task_episode_identity": bool(identity.get("passed")),
        "homogeneous_model_interface_configuration": bool(
            configuration.get("homogeneous")
        ),
        "factorial_cell_coverage": cell_coverage_pass,
        "completed_terminal_submissions": terminal_pass,
        "valid_structured_answer_schema": schema_pass,
        "no_protocol_limit_adapter_or_infrastructure_errors": error_free,
        "score_fields_computable": score_fields_computable,
        "required_score_replay_complete": replay_pass,
    }
    required_search_rows = [row for row in rows if row["complete_search_required"]]
    valid_search_count = sum(
        row["complete_search_certificate_valid"] is True for row in required_search_rows
    )
    systematic_search_failure = bool(required_search_rows and valid_search_count == 0)
    performance = {
        "exact_object_accuracy": _binary_summary(
            row["exact_object_correct"] for row in rows
        ),
        "evidence_witness_success": _binary_summary(
            row["evidence_witness_success"] for row in rows
        ),
        "answer_evidence_quality_pass": _binary_summary(
            row["answer_evidence_quality_pass"] for row in rows
        ),
        "strict_primary_endpoint": _binary_summary(
            row["strict_primary_endpoint_pass"] for row in rows
        ),
        "complete_search_certificate_valid": _binary_summary(
            row["complete_search_certificate_valid"] for row in required_search_rows
        ),
    }
    return {
        "passed": all(checks.values()),
        "scope": "clean_canary" if clean_canary_shape_required else "covered_experiment",
        "checks": checks,
        "failure_reasons": [name for name, passed in checks.items() if not passed],
        "episode_error_codes": dict(sorted(error_counts.items())),
        "performance_not_used_as_gate": performance,
        "manual_review": {
            "required": systematic_search_failure,
            "systematic_complete_search_failure": systematic_search_failure,
            "required_complete_search_episode_count": len(required_search_rows),
            "valid_complete_search_episode_count": valid_search_count,
            "reason": (
                "No episode requiring closed-world search produced a valid complete-search "
                "certificate. Determine whether this reflects model behavior, interface "
                "instructions, or snapshot/scorer wiring before a full paid run."
                if systematic_search_failure
                else None
            ),
        },
        "policy_note": (
            "Wrong but schema-valid answers and evidence-quality failures are scientific model "
            "outcomes, not harness-gate failures. Missing/invalid terminal schemas, protocol "
            "errors, incomplete cells, or uncomputable score fields fail the gate."
        ),
    }


def _canary_cell_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), row)
    schemas = {str(row["factorial_schema"]) for row in by_task.values()}
    if len(schemas) != 1 or not schemas <= {"v1", "v2"}:
        raise ValueError("analysis input mixes or omits factorial schemas")
    factorial_schema = next(iter(schemas))
    counts: Counter[tuple[str, str, str, bool]] = Counter(
        (
            str(row["topology"]),
            str(row["evidence_state"]),
            str(row["companion_evidence_state"]),
            bool(row["distractor_injected"]),
        )
        for row in by_task.values()
    )
    episode_counts: Counter[tuple[str, str, str, bool]] = Counter(
        (
            str(row["topology"]),
            str(row["evidence_state"]),
            str(row["companion_evidence_state"]),
            bool(row["distractor_injected"]),
        )
        for row in rows
    )
    expected = [
        (topology, state, companion_state, distractor)
        for topology in TOPOLOGIES
        for state in EVIDENCE_STATES
        for companion_state in (
            EVIDENCE_STATES
            if factorial_schema == "v2" and topology.endswith("w2")
            else ("supported",)
        )
        for distractor in DISTRACTOR_VALUES
    ]
    missing = [
        _cell_label(*cell)
        for cell in expected
        if counts.get(cell, 0) == 0
    ]
    cell_counts = {
        _cell_label(*cell): counts.get(cell, 0)
        for cell in expected
    }
    episode_cell_counts = {
        _cell_label(*cell): episode_counts.get(cell, 0)
        for cell in expected
    }
    family_cells: dict[str, set[tuple[str, str, str, bool]]] = defaultdict(set)
    for row in by_task.values():
        family_cells[str(row["superfamily_id"])].add(
            (
                str(row["topology"]),
                str(row["evidence_state"]),
                str(row["companion_evidence_state"]),
                bool(row["distractor_injected"]),
            )
        )
    return {
        "expected_cell_count": len(expected),
        "factorial_schema": factorial_schema,
        "covered_cell_count": len(expected) - len(missing),
        "all_factorial_cells_covered": not missing,
        "balanced_one_task_per_cell": not missing
        and all(counts[cell] == 1 for cell in expected),
        "balanced_one_episode_per_cell": not missing
        and all(episode_counts[cell] == 1 for cell in expected),
        "missing_cells": missing,
        "unique_task_counts_by_cell": cell_counts,
        "episode_counts_by_cell": episode_cell_counts,
        "complete_superfamily_count": sum(
            cells == set(expected) for cells in family_cells.values()
        ),
        "observed_superfamily_count": len(family_cells),
    }


def _cell_label(
    topology: str,
    state: str,
    companion_state: str,
    distractor: bool,
) -> str:
    return (
        f"{topology}|primary={state}|companion={companion_state}|"
        f"{'injected' if distractor else 'not_injected'}"
    )


def _resource_use(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "tool_calls": _numeric_summary(row["tool_calls"] for row in rows),
        "turns": _numeric_summary(row["turns"] for row in rows),
        "latency_ms": _numeric_summary(row["latency_ms"] for row in rows),
        "input_tokens": _numeric_summary(
            (row["input_tokens"] for row in rows), total=True
        ),
        "output_tokens": _numeric_summary(
            (row["output_tokens"] for row in rows), total=True
        ),
        "total_tokens": _numeric_summary(
            (row["total_tokens"] for row in rows), total=True
        ),
        "responses": _numeric_summary((row["responses"] for row in rows), total=True),
    }


def _numeric_summary(values: Any, *, total: bool = False) -> dict[str, Any]:
    materialized = list(values)
    observed = [float(value) for value in materialized if value is not None]
    result: dict[str, Any] = {
        "observed": len(observed),
        "missing": len(materialized) - len(observed),
        "mean": _round(mean(observed)) if observed else None,
        "median": _round(median(observed)) if observed else None,
        "minimum": _round(min(observed)) if observed else None,
        "maximum": _round(max(observed)) if observed else None,
    }
    if total:
        result["sum"] = int(sum(observed)) if observed else None
    return result


def _required_text(values: Mapping[str, Any], key: str, identity: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{identity} has no usable {key!r}")
    return value


def _round(value: float) -> float:
    return round(float(value), 6)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
