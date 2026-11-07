"""Small local compiler and evaluator interface for PatientFlowSynth.

Only an explicitly selected custom adapter may call a model. The reference adapter
uses evaluator-only answers and is an oracle diagnostic, never a model baseline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from patient_tool_gym.agent_adapters import AgentAdapter, ReferenceAgent
from patient_tool_gym.compact_dataset import (
    CompactDataset,
    is_compact_dataset,
    load_compact_dataset,
    verify_compact_dataset,
)
from patient_tool_gym.compositional_analysis import (
    analyze_compositional_results,
    load_episode_file,
    load_verified_release,
)
from patient_tool_gym.compositional_experiments import (
    prepare_compositional_experiment_inputs,
    select_balanced_compositional_canary,
    verify_compositional_experiment_inputs,
)
from patient_tool_gym.compositional_release import (
    build_compositional_release,
    verify_compositional_release_context,
)
from patient_tool_gym.episodes import EpisodeLimits, run_episode, run_episodes_file
from patient_tool_gym.evidence import load_snapshot
from patient_tool_gym.fhir import load_fhir_bundle, normalize_fhir_bundle
from patient_tool_gym.patient_p2 import generate_patient_p2_tasks
from patient_tool_gym.program import content_hash
from patient_tool_gym.task import TaskArtifact
from patient_tool_gym.tools import ToolRegistry

COHORTS = ("all", "train", "dev", "test", "canary-factorial-80")
COMPACT_RUNTIME_SCHEMA = "patientflowsynth.compact-runtime.v1"


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patientflowsynth", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_command = commands.add_parser("compile", help="Compile local synthetic FHIR Bundles.")
    compile_command.add_argument("bundles", type=Path, help="Bundle JSON or JSON-only directory.")
    compile_command.add_argument("output", type=Path, help="Fresh evaluator release directory.")
    compile_command.add_argument("--seed", type=int, default=20260711)
    compile_command.add_argument("--materialize-derived", action="store_true")

    verify = commands.add_parser("verify", help="Verify closed inventory and semantic replay.")
    verify.add_argument("release", type=Path)

    prepare = commands.add_parser(
        "prepare", help="Prepare verified runtime charts and task cohorts."
    )
    prepare.add_argument("release", type=Path)
    prepare.add_argument("output", type=Path, help="Fresh directory outside the release.")
    prepare.add_argument("--seed", type=int, default=20260711)

    run = commands.add_parser("run", help="Run a diagnostic oracle or an explicit custom adapter.")
    run.add_argument("release", type=Path)
    run.add_argument("runtime", type=Path)
    run.add_argument(
        "output", type=Path, help="Fresh episode JSONL file; never overwritten/resumed."
    )
    run.add_argument(
        "--adapter",
        default="reference",
        metavar="reference|MODULE:FACTORY",
        help="Reference is a gold-aware diagnostic. Custom trusted code may make paid requests.",
    )
    run.add_argument("--cohort", choices=COHORTS, default="all")
    run.add_argument(
        "--limit", type=_positive, help="Run only the first N tasks; report as a subset."
    )
    run.add_argument("--max-turns", type=_positive, default=12)
    run.add_argument("--max-tool-calls", type=_positive, default=24)

    summarize = commands.add_parser("summarize", help="Replay scores and summarize one run.")
    summarize.add_argument("release", type=Path)
    summarize.add_argument("runtime", type=Path)
    summarize.add_argument("episodes", type=Path)
    summarize.add_argument(
        "--output", type=Path, help="Save the full hash-bound report; console output stays concise."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = _dispatch(arguments)
    except ValidationError as exc:
        # Do not echo patient payloads or plugin credentials from Pydantic's input repr.
        message = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_input=False, include_context=False)
        )
        print(json.dumps({"ok": False, "error": message}), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command == "compile":
        return _compile(arguments)
    if arguments.command == "verify":
        if is_compact_dataset(arguments.release):
            return verify_compact_dataset(arguments.release)
        return verify_compositional_release_context(arguments.release).report
    if arguments.command == "prepare":
        _fresh(arguments.output)
        _outside(arguments.output, arguments.release)
        if is_compact_dataset(arguments.release):
            return _prepare_compact_runtime(arguments.release, arguments.output, arguments.seed)
        context = verify_compositional_release_context(arguments.release)
        manifest = prepare_compositional_experiment_inputs(
            arguments.release,
            arguments.output,
            seed=arguments.seed,
            release_context=context,
        )
        verification = verify_compositional_experiment_inputs(
            arguments.output,
            release_directory=arguments.release,
            release_context=context,
        )
        return {"ok": True, "counts": manifest["counts"], "verification": verification}
    if arguments.command == "run":
        return _run(arguments)
    return _summarize(arguments)


def _fresh(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists; choose a fresh path: {path}")


def _outside(output: Path, *closed_directories: Path) -> None:
    target = output.resolve()
    for directory in closed_directories:
        root = directory.resolve()
        if target == root or root in target.parents:
            raise ValueError("output must be outside the immutable input directories")


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root.resolve() not in path.parents:
        raise ValueError("artifact path escapes its input directory")
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON objects in {path}")
                rows.append(value)
    return rows


def _compile(arguments: argparse.Namespace) -> dict[str, Any]:
    source = arguments.bundles
    _fresh(arguments.output)
    if source.is_dir():
        _outside(arguments.output, source)
        paths = sorted(source.rglob("*.json"))
    elif source.is_file():
        paths = [source]
    else:
        raise ValueError("source Bundle path does not exist")
    if not paths:
        raise ValueError("source Bundle path contains no JSON files")
    snapshots = []
    dependency_count = 0
    for path in paths:
        bundle = load_fhir_bundle(path)
        entries = bundle.get("entry")
        if bundle.get("resourceType") != "Bundle" or not isinstance(entries, list):
            raise ValueError(f"expected a FHIR Bundle: {path}")
        if any(
            not isinstance(entry, dict) or not isinstance(entry.get("resource"), dict)
            for entry in entries
        ):
            raise ValueError(f"malformed FHIR Bundle entry: {path}")
        patients = sum(entry["resource"].get("resourceType") == "Patient" for entry in entries)
        if patients == 0:
            dependency_count += 1
            continue
        snapshots.append(normalize_fhir_bundle(bundle, dataset="synthea"))
    if not snapshots:
        raise ValueError("source contains no single-patient Bundles")
    generated = generate_patient_p2_tasks(tuple(snapshots))
    manifest = build_compositional_release(
        generated.tasks,
        source,
        arguments.output,
        seed=arguments.seed,
        materialize_derived_bundles=arguments.materialize_derived,
    )
    verification = verify_compositional_release_context(arguments.output).report
    return {
        "ok": True,
        "release_hash": manifest["release_hash"],
        "counts": manifest["counts"],
        "input_patient_count": len(snapshots),
        "dependency_bundle_count": dependency_count,
        "source_task_count": len(generated.tasks),
        "verification": verification,
        "scope": "All eligible supplied candidates; not the paper's sealed 225-patient census.",
    }


def _compact_runtime_payload(dataset: CompactDataset, seed: int) -> dict[str, Any]:
    cohorts = {"all": {"task_ids": sorted(dataset.tasks)}}
    for split in ("train", "dev", "test"):
        cohorts[split] = {
            "task_ids": sorted(
                task_id for task_id, assigned in dataset.task_splits.items() if assigned == split
            )
        }
    canary = select_balanced_compositional_canary(dataset.family_rows, dataset.tasks, seed=seed)
    cohorts["canary-factorial-80"] = {"task_ids": [row["task_id"] for row in canary]}
    payload = {
        "schema_version": COMPACT_RUNTIME_SCHEMA,
        "dataset_manifest_hash": content_hash(dataset.manifest),
        "dataset_package_hash": dataset.manifest["package_hash"],
        "seed": seed,
        "cohorts": cohorts,
    }
    payload["runtime_hash"] = content_hash(payload)
    return payload


def _prepare_compact_runtime(release: Path, output: Path, seed: int) -> dict[str, Any]:
    dataset = load_compact_dataset(release)
    manifest = _compact_runtime_payload(dataset, seed)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        "counts": dataset.manifest["counts"],
        "runtime_hash": manifest["runtime_hash"],
        "dataset_verification": dataset.report,
        "chart_loading": "Verified on demand; normalized charts are not duplicated.",
    }


def _runtime(
    release: Path, runtime: Path
) -> tuple[dict[str, Any], dict[str, Path] | CompactDataset]:
    if is_compact_dataset(release):
        dataset = load_compact_dataset(release)
        inventory = list(runtime.rglob("*"))
        if any(path.is_symlink() for path in inventory) or {
            path.relative_to(runtime).as_posix() for path in inventory if path.is_file()
        } != {"manifest.json"}:
            raise ValueError("compact runtime file inventory is not closed")
        manifest = _json(runtime / "manifest.json")
        seed = manifest.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("compact runtime seed must be an integer")
        expected = _compact_runtime_payload(dataset, seed)
        if manifest != expected:
            raise ValueError("compact runtime does not match its dataset or cohort bindings")
        return manifest, dataset
    context = verify_compositional_release_context(release)
    verify_compositional_experiment_inputs(
        runtime,
        release_directory=release,
        release_context=context,
    )
    manifest = _json(runtime / "manifest.json")
    bindings = _jsonl(_contained(runtime, manifest["snapshot_bindings"]["path"]))
    paths = {
        row["snapshot"]["snapshot_id"]: _contained(runtime, row["snapshot"]["path"])
        for row in bindings
    }
    if len(paths) != len(bindings):
        raise ValueError("duplicate runtime snapshot bindings")
    return manifest, paths


class _LazyRegistries(Mapping[str, ToolRegistry]):
    """Keep only the dataset's small chart cache during full-dataset analysis."""

    def __init__(self, dataset: CompactDataset, snapshot_ids: set[str]) -> None:
        self.dataset = dataset
        self.snapshot_ids = frozenset(snapshot_ids)

    def __getitem__(self, key: str) -> ToolRegistry:
        if key not in self.snapshot_ids:
            raise KeyError(key)
        return self.dataset.registry(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.snapshot_ids)

    def __len__(self) -> int:
        return len(self.snapshot_ids)


def _registries(paths: dict[str, Path], snapshot_ids: set[str]) -> dict[str, ToolRegistry]:
    registries = {}
    for snapshot_id in sorted(snapshot_ids):
        snapshot = load_snapshot(paths[snapshot_id])
        if snapshot.snapshot_id != snapshot_id:
            raise ValueError("runtime snapshot identity differs from its binding")
        registries[snapshot_id] = ToolRegistry(snapshot)
    return registries


def _custom_adapter(specification: str) -> AgentAdapter:
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name or ":" in factory_name:
        raise ValueError("adapter must be reference or MODULE:FACTORY")
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
        adapter = factory()  # No task, gold answer, witness, or reference trace is provided.
    except Exception as exc:
        raise ValueError(f"custom adapter factory failed ({type(exc).__name__})") from None
    if not isinstance(adapter, AgentAdapter):
        raise ValueError("custom factory must return an AgentAdapter with descriptor and start")
    return adapter


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    _fresh(arguments.output)
    _outside(arguments.output, arguments.release, arguments.runtime)
    manifest, paths = _runtime(arguments.release, arguments.runtime)
    if isinstance(paths, CompactDataset):
        tasks = [
            paths.tasks[task_id] for task_id in manifest["cohorts"][arguments.cohort]["task_ids"]
        ]
    else:
        task_path = _contained(arguments.runtime, manifest["cohorts"][arguments.cohort]["path"])
        tasks = [TaskArtifact.model_validate(row) for row in _jsonl(task_path)]
    cohort_count = len(tasks)
    if arguments.limit is not None:
        tasks = tasks[: arguments.limit]
    if not tasks:
        raise ValueError("selected cohort contains no tasks")
    registries = (
        _LazyRegistries(paths, {task.patient_snapshot_id for task in tasks})
        if isinstance(paths, CompactDataset)
        else _registries(paths, {task.patient_snapshot_id for task in tasks})
    )
    if arguments.adapter == "reference":
        adapter_factory = ReferenceAgent.from_task
        diagnostic_only = True
    else:
        adapter = _custom_adapter(arguments.adapter)

        def adapter_factory(_task: TaskArtifact) -> AgentAdapter:
            return adapter

        diagnostic_only = adapter.descriptor.diagnostic_only
    # Exclusive creation prevents accidentally resuming or overwriting a prior run.
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8"):
        pass
    limits = EpisodeLimits(max_turns=arguments.max_turns, max_tool_calls=arguments.max_tool_calls)
    if isinstance(paths, CompactDataset):
        # The original batch API retains every chart in its preflight list. Stream
        # compact datasets so a complete run does not load thousands of charts.
        completed = 0
        descriptor = None
        with arguments.output.open("a", encoding="utf-8") as handle:
            for task in tasks:
                current_adapter = adapter_factory(task)
                current_descriptor = current_adapter.descriptor.model_dump(mode="json")
                if descriptor is None:
                    descriptor = current_descriptor
                elif current_descriptor != descriptor:
                    raise ValueError("adapter configuration changed during the run; stopping")
                record = run_episode(
                    task,
                    registries[task.patient_snapshot_id],
                    current_adapter,
                    limits=limits,
                )
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed += record.status == "completed"
        selected = len(tasks)
    else:
        summary = run_episodes_file(
            tasks,
            snapshot_resolver=registries.__getitem__,
            adapter_factory=adapter_factory,
            output_path=arguments.output,
            limits=limits,
        )
        selected, completed = summary.selected, summary.completed
    return {
        "ok": True,
        "episodes": str(arguments.output),
        "selected": selected,
        "cohort_task_count": cohort_count,
        "subset": selected != cohort_count,
        "completed": completed,
        "not_completed": selected - completed,
        "diagnostic_only": diagnostic_only,
        "note": "Use summarize for replayed strict scores; reference runs are not model baselines.",
    }


def _summarize(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.output is not None:
        _fresh(arguments.output)
        _outside(arguments.output, arguments.release, arguments.runtime)
    _, paths = _runtime(arguments.release, arguments.runtime)
    records = load_episode_file(arguments.episodes)
    if isinstance(paths, CompactDataset):
        tasks, splits = paths.tasks, paths.task_splits
        families, integrity = paths.family_dimensions, paths.report
        registries = _LazyRegistries(paths, {record.patient_snapshot_id for record in records})
    else:
        tasks, splits, families, integrity = load_verified_release(arguments.release)
        registries = _registries(paths, {record.patient_snapshot_id for record in records})
    report = analyze_compositional_results(
        tasks,
        records,
        release_integrity=integrity,
        task_splits=splits,
        family_dimensions=families,
        registries=registries,
        # The legacy analysis rejects no-answer failures when this flag is true.
        # Always supply verified registries, replay every submitted answer, and
        # retain missing answers as failures instead of dropping their denominators.
        require_score_replay=all(record.final_answer is not None for record in records),
    )
    if set(report["score_sources"]) - {"full_snapshot_replay", "submission_missing"}:
        raise ValueError("summary contains scores that were not replayed")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    diagnostic_count = sum(record.diagnostic_only for record in records)
    evaluation_kind = (
        "oracle_diagnostic_only"
        if diagnostic_count == len(records)
        else "mixed_diagnostic_and_non_diagnostic"
        if diagnostic_count
        else "non_oracle_agent"
    )
    return {
        "ok": True,
        "evaluation_kind": evaluation_kind,
        "diagnostic_episode_count": diagnostic_count,
        "note": (
            "Gold-aware oracle diagnostic; not a model baseline."
            if diagnostic_count
            else "One model/interface configuration; failures remain in outcome denominators."
        ),
        "observed_task_count": report["identity"]["observed_task_count"],
        "release_task_count": len(tasks),
        "subset": not report["identity"]["complete_release_observed"],
        "overall": {
            key: report["overall"][key]
            for key in (
                "episode_count",
                "completion",
                "exact_object_accuracy",
                "evidence_witness_success",
                "complete_search_certificate_valid",
                "strict_primary_endpoint",
            )
        },
        "score_sources": report["score_sources"],
        "runtime_and_release_verified": True,
        "submitted_answers_replayed": report["score_sources"].get("full_snapshot_replay", 0),
        "missing_answers_counted_as_failures": report["score_sources"].get("submission_missing", 0),
        "full_report": str(arguments.output) if arguments.output is not None else None,
        "full_report_hash": report["report_payload_content_hash"],
    }
