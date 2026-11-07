from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from patient_tool_gym.agent_adapters import AdapterDescriptor, AdapterTurn
from patient_tool_gym.cli import main
from patient_tool_gym.program import content_hash

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo_patient.json"


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli-release")
    release = root / "release"
    runtime = root / "runtime"
    assert main(["compile", str(DEMO), str(release)]) == 0
    assert main(["prepare", str(release), str(runtime)]) == 0
    return release, runtime


def test_demo_compilation_and_verification(compiled, capsys):
    release, _ = compiled
    assert main(["verify", str(release)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    manifest = json.loads((release / "manifest.json").read_text())
    counts = manifest["counts"]
    assert counts["patient_count"] == 1
    assert counts["task_count"] == 80 * counts["superfamily_count"]
    assert counts["cell_count"] == 32 * counts["superfamily_count"]


def test_reference_run_and_replayed_summary(compiled, tmp_path, capsys):
    release, runtime = compiled
    episodes = tmp_path / "episodes.jsonl"
    assert main(["run", str(release), str(runtime), str(episodes), "--limit", "4"]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["diagnostic_only"] is True
    assert run["selected"] == 4
    assert run["subset"] is True
    summary = tmp_path / "summary.json"
    assert (
        main(
            [
                "summarize",
                str(release),
                str(runtime),
                str(episodes),
                "--output",
                str(summary),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["episode_count"] == 4
    assert report["overall"]["strict_primary_endpoint"]["rate"] == 1.0
    assert report["evaluation_kind"] == "oracle_diagnostic_only"
    assert report["diagnostic_episode_count"] == 4
    assert report["submitted_answers_replayed"] == 4
    assert report["subset"] is True
    assert report["observed_task_count"] == 4
    assert "family_level_factorial_contrasts" not in report
    full_report = json.loads(summary.read_text())
    assert "family_level_factorial_contrasts" in full_report
    assert (
        full_report["overall"]["strict_primary_endpoint"]
        == report["overall"]["strict_primary_endpoint"]
    )
    assert full_report.pop("report_payload_content_hash") == report["full_report_hash"]
    assert content_hash(full_report) == report["full_report_hash"]
    before = episodes.read_bytes()
    assert main(["run", str(release), str(runtime), str(episodes)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert episodes.read_bytes() == before
    assert (
        main(
            [
                "summarize",
                str(release),
                str(runtime),
                str(episodes),
                "--output",
                str(summary),
            ]
        )
        == 1
    )
    assert "already exists" in capsys.readouterr().err


def test_compile_refuses_existing_output(compiled, capsys):
    release, _ = compiled
    before = (release / "manifest.json").read_bytes()
    assert main(["compile", str(DEMO), str(release)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert (release / "manifest.json").read_bytes() == before


def test_prepare_refuses_existing_or_nested_output(compiled, capsys):
    release, runtime = compiled
    assert main(["prepare", str(release), str(runtime)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["prepare", str(release), str(release / "runtime")]) == 1
    assert "outside" in capsys.readouterr().err


def test_tampered_runtime_rejected_before_custom_factory(compiled, tmp_path, monkeypatch, capsys):
    release, runtime = compiled
    tampered = tmp_path / "tampered"
    shutil.copytree(runtime, tampered)
    with (tampered / "tasks" / "all.jsonl").open("a") as handle:
        handle.write("{}\n")
    imported = []
    monkeypatch.setattr("patient_tool_gym.cli.importlib.import_module", imported.append)
    output = tmp_path / "episodes.jsonl"
    assert (
        main(
            [
                "run",
                str(release),
                str(tampered),
                str(output),
                "--adapter",
                "plugin:factory",
            ]
        )
        == 1
    )
    assert imported == []
    assert not output.exists()
    assert capsys.readouterr().err


def test_run_rejects_output_inside_runtime(compiled, capsys):
    release, runtime = compiled
    assert main(["run", str(release), str(runtime), str(runtime / "episodes.jsonl")]) == 1
    assert "outside" in capsys.readouterr().err


def test_custom_factory_has_no_gold_inputs(compiled, tmp_path, monkeypatch, capsys):
    release, runtime = compiled
    observed = []

    class Adapter:
        descriptor = AdapterDescriptor(adapter_id="offline-test")

        def start(self, request, *, seed):
            observed.append(request.model_dump())
            return self

        def next(self, tool_results):
            return AdapterTurn(message="No answer.")

    def factory():
        observed.append("factory-called-with-no-arguments")
        return Adapter()

    module = types.ModuleType("patientflowsynth_test_plugin")
    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "custom.jsonl"
    assert (
        main(
            [
                "run",
                str(release),
                str(runtime),
                str(output),
                "--limit",
                "1",
                "--max-turns",
                "1",
                "--adapter",
                f"{module.__name__}:factory",
            ]
        )
        == 0
    )
    assert observed[0] == "factory-called-with-no-arguments"
    assert set(observed[1]) == {"question", "tools"}
    assert json.loads(capsys.readouterr().out)["diagnostic_only"] is False
    record = json.loads(output.read_text())
    assert record["status"] != "completed"
    assert main(["summarize", str(release), str(runtime), str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["strict_primary_endpoint"]["rate"] == 0.0
    assert report["evaluation_kind"] == "non_oracle_agent"
    assert report["missing_answers_counted_as_failures"] == 1
    assert report["submitted_answers_replayed"] == 0


def test_compile_rejects_bad_source_and_nested_output(tmp_path, capsys):
    source = tmp_path / "input"
    source.mkdir()
    (source / "not-a-bundle.json").write_text("{}")
    assert main(["compile", str(source), str(tmp_path / "release")]) == 1
    assert "FHIR Bundle" in capsys.readouterr().err
    assert main(["compile", str(source), str(source / "release")]) == 1
    assert "outside" in capsys.readouterr().err


def test_limit_must_be_positive():
    with pytest.raises(SystemExit) as exc:
        main(["run", "release", "runtime", "output", "--limit", "0"])
    assert exc.value.code == 2
