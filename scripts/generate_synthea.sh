#!/usr/bin/env bash
# Generate a new synthetic cohort from the pinned, separately installed Synthea source.
set -euo pipefail

exec python3 - "$@" <<'PY'
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

PIN = "2ab21caa03f8c6a0400c06593e53f1c1b7510abe"


def fail(message):
    raise ValueError(message)


def number(value):
    if not re.fullmatch(r"[0-9]+", value) or int(value) > 2**63 - 1:
        raise argparse.ArgumentTypeError("expected a nonnegative signed-64-bit integer")
    return int(value)


def calendar_date(value):
    if not re.fullmatch(r"[0-9]{8}", value):
        raise argparse.ArgumentTypeError("expected a date as YYYYMMDD")
    try:
        return dt.datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid calendar date") from exc


def read_git(source, *arguments):
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(source), *arguments],
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new cohort, not the paper's selected 225-patient dataset."
    )
    parser.add_argument("--source", required=True, type=Path, help="Clean pinned Synthea checkout.")
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory outside checkout.")
    parser.add_argument("--patients", type=number, default=100)
    parser.add_argument("--seed", type=number, default=31)
    parser.add_argument("--clinician-seed", type=number, default=32)
    parser.add_argument("--reference-date", type=calendar_date, default=calendar_date("20260707"))
    parser.add_argument("--end-date", type=calendar_date, default=calendar_date("20260707"))
    parser.add_argument("--state", default="Massachusetts")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show settings; no writes.")
    args = parser.parse_args()
    if not 1 <= args.patients <= 2**31 - 1:
        fail("--patients must be between 1 and 2147483647")
    if args.end_date < args.reference_date:
        fail("--end-date must not precede --reference-date")
    if args.end_date > dt.datetime.now(dt.timezone.utc).date():
        fail("--end-date must not be in the future")
    if not re.fullmatch(r"[A-Za-z][A-Za-z .-]*", args.state):
        fail("--state must contain only letters, spaces, dots, or hyphens")
    if args.output.exists() or args.output.is_symlink():
        fail("output already exists; choose a fresh directory")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output == source or source in output.parents:
        fail("output must be outside the Synthea checkout")
    if not shutil.which("git"):
        fail("git is required")
    if Path(read_git(source, "rev-parse", "--show-toplevel")).resolve() != source:
        fail("--source must identify the checkout root")
    if read_git(source, "rev-parse", "HEAD") != PIN:
        fail(f"Synthea checkout must be pinned to {PIN}")
    if read_git(source, "status", "--porcelain", "--untracked-files=normal"):
        fail("Synthea checkout has uncommitted changes; use a clean checkout")
    launcher = source / "run_synthea"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        fail("Synthea run_synthea launcher is missing or not executable")

    parameters = {
        "patients_requested": args.patients,
        "seed": args.seed,
        "clinician_seed": args.clinician_seed,
        "reference_date": args.reference_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "state": args.state,
    }
    # The pinned upstream launcher interpolates single-quoted Groovy literals.
    # Refuse unsafe argument characters instead of relying on shell quoting alone.
    options = [
        "-s", str(args.seed), "-cs", str(args.clinician_seed),
        "-r", args.reference_date.strftime("%Y%m%d"),
        "-e", args.end_date.strftime("%Y%m%d"), "-p", str(args.patients),
        "--generate.only_alive_patients=true",
        "--generate.terminology_service_url=",
        f"--exporter.baseDirectory={output / 'output'}",
        "--exporter.use_uuid_filenames=true",
        "--exporter.fhir.export=true",
        "--exporter.fhir.bulk_data=false",
        "--exporter.fhir.transaction_bundle=true",
        "--exporter.fhir.use_us_core_ig=true",
        "--exporter.fhir.us_core_version=6.1.0",
        "--exporter.hospital.fhir.export=true",
        "--exporter.practitioner.fhir.export=true",
        "--exporter.metadata.export=true",
        "--exporter.ccda.export=false",
        "--exporter.csv.export=false",
        "--exporter.json.export=false",
        "--exporter.fhir_stu3.export=false",
        "--exporter.fhir_dstu2.export=false",
        args.state,
    ]
    if any("'" in value or "\\" in value or any(ord(c) < 32 for c in value) for value in options):
        fail("output path contains characters unsafe for the pinned Synthea launcher")
    provenance = {
        "schema_version": "patientflowsynth.new-synthea-cohort.v1",
        "scope": "new cohort; not the paper's selected patient census",
        "synthea_commit": PIN,
        "synthea_version": "v4.0.0-1-g2ab21caa0",
        "parameters": parameters,
        "fhir_directory": "output/fhir",
        "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
        "export_options": [
            value for value in options if value.startswith("--")
            and not value.startswith("--exporter.baseDirectory=")
        ],
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **provenance}, indent=2, sort_keys=True))
        return 0
    java_home = os.environ.get("JAVA_HOME")
    java = str(Path(java_home) / "bin/java") if java_home else shutil.which("java")
    if not java:
        fail("install a Java 17 JDK and make java available")
    version = subprocess.run([java, "-version"], text=True, capture_output=True, check=True)
    version_text = version.stderr + version.stdout
    if not re.search(r'version "17(?:[.\"])', version_text):
        fail("this pinned helper requires Java 17; configure JAVA_HOME or PATH")
    provenance["java_version"] = version_text.splitlines()[0]
    provenance["started_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Atomic refusal if another process claimed this output directory.
    write_json(output / "generation.json", provenance)
    print("Running local Synthea; Gradle may download its build dependencies.", flush=True)
    with (output / "generation.log").open("x", encoding="utf-8") as log:
        result = subprocess.run([str(launcher), *options], cwd=source, stdout=log, stderr=log)
    files = list((output / "output/fhir").glob("*.json"))
    succeeded = result.returncode == 0 and bool(files)
    write_json(output / "result.json", {
        "completed": succeeded,
        "returncode": result.returncode,
        "fhir_json_file_count": len(files),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    if not succeeded:
        fail("generation failed or produced no FHIR JSON; inspect generation.log and use a fresh output")
    print(f"Generated {len(files)} FHIR JSON files under output/fhir (including dependencies).")
    return 0


try:
    raise SystemExit(main())
except (OSError, ValueError, subprocess.SubprocessError) as error:
    print(f"Generation helper: {error}", file=sys.stderr)
    raise SystemExit(1) from None
PY
