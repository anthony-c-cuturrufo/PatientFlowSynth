"""Exercise the generation helper with local stubs, never downloaded software."""

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_synthea.sh"
PIN = "2ab21caa03f8c6a0400c06593e53f1c1b7510abe"


def _executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


@pytest.fixture
def generator(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(binaries / "git", f'''#!/usr/bin/env python3
import os
import sys
args = sys.argv[4:]
if args == ["rev-parse", "--show-toplevel"]:
    print(sys.argv[3])
elif args == ["rev-parse", "HEAD"]:
    print(os.environ.get("PFS_TEST_COMMIT", "{PIN}"))
elif args[:1] == ["status"]:
    print(os.environ.get("PFS_TEST_DIRTY", ""))
else:
    raise SystemExit(1)
''')
    _executable(binaries / "java", '''#!/usr/bin/env python3
import sys
print('openjdk version "17.0.18"', file=sys.stderr)
''')
    _executable(source / "run_synthea", '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
output = Path(next(arg.split("=", 1)[1] for arg in sys.argv
                   if arg.startswith("--exporter.baseDirectory=")))
output.mkdir()
(output / "arguments.json").write_text(json.dumps(sys.argv[1:]))
print("Simulated patient-generation log stays in the output directory.")
if os.environ.get("PFS_TEST_FAIL"):
    raise SystemExit(7)
(output / "fhir").mkdir()
(output / "fhir" / "fixture.json").write_text(json.dumps({
    "resourceType": "Bundle", "type": "collection", "entry": []
}))
''')
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("JAVA_HOME", raising=False)
    return source, tmp_path / "new-cohort"


def _run(source, output, *arguments):
    return subprocess.run(
        ["bash", str(SCRIPT), "--source", str(source), "--output", str(output), *arguments],
        capture_output=True, text=True,
    )


def test_dry_run_validates_without_creating_outputs(generator):
    source, output = generator
    result = _run(source, output, "--patients", "2", "--seed", "41", "--dry-run")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["dry_run"] is True
    assert report["synthea_commit"] == PIN
    assert report["parameters"]["patients_requested"] == 2
    assert report["parameters"]["seed"] == 41
    assert not output.exists()
    assert str(source) not in result.stdout


@pytest.mark.parametrize(
    "name,value", [("PFS_TEST_COMMIT", "wrong"), ("PFS_TEST_DIRTY", " M file")]
)
def test_rejects_wrong_or_dirty_source(generator, monkeypatch, name, value):
    source, output = generator
    monkeypatch.setenv(name, value)
    assert _run(source, output, "--dry-run").returncode == 1
    assert not output.exists()


def test_refuses_existing_and_nested_output(generator):
    source, output = generator
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged")
    assert _run(source, output).returncode == 1
    assert sentinel.read_text() == "unchanged"
    nested = source / "generated"
    assert _run(source, nested).returncode == 1
    assert not nested.exists()


@pytest.mark.parametrize("arguments", [
    ["--patients", "0"], ["--seed", "-1"], ["--reference-date", "20260230"],
    ["--end-date", "20200101"], ["--end-date", "20990101"], ["--state", "bad'state"],
])
def test_rejects_invalid_settings(generator, arguments):
    source, output = generator
    assert _run(source, output, *arguments).returncode != 0
    assert not output.exists()


def test_rejects_launcher_metacharacters(generator):
    source, output = generator
    quoted = output.with_name("unsafe'path")
    assert _run(source, quoted).returncode == 1
    assert not quoted.exists()


def test_records_execution_and_captures_log(generator):
    source, output = generator
    result = _run(source, output, "--patients", "1", "--clinician-seed", "9")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "generation.json").read_text())
    assert manifest["parameters"]["clinician_seed"] == 9
    assert str(source) not in json.dumps(manifest)
    assert str(output) not in json.dumps(manifest)
    status = json.loads((output / "result.json").read_text())
    assert status["completed"] is True
    assert status["fhir_json_file_count"] == 1
    arguments = json.loads((output / "output/arguments.json").read_text())
    assert "--exporter.hospital.fhir.export=true" in arguments
    assert "--exporter.practitioner.fhir.export=true" in arguments
    assert "--generate.terminology_service_url=" in arguments
    assert "Simulated patient-generation log" not in result.stdout
    assert "Simulated patient-generation log" in (output / "generation.log").read_text()


def test_failure_retains_evidence_and_cannot_be_overwritten(generator, monkeypatch):
    source, output = generator
    monkeypatch.setenv("PFS_TEST_FAIL", "1")
    result = _run(source, output)
    assert result.returncode == 1
    status = json.loads((output / "result.json").read_text())
    assert status["completed"] is False
    assert status["returncode"] == 7
    assert (output / "generation.log").is_file()
    assert "already exists" in _run(source, output).stderr
