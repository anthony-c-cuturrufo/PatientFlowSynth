#!/usr/bin/env python3
"""Maintainer tool: build a train/dev-only archive from a verified frozen release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path


def create_archive(package: Path, destination: Path) -> dict:
    """Create a reproducible tar.gz, rejecting links and unexpected file types."""
    if destination.exists() or destination.is_symlink():
        raise ValueError("archive destination already exists")
    entries = [package, *sorted(package.rglob("*"))]
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries):
        raise ValueError("package must contain regular files and directories only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expanded = 0
    with destination.open("xb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6
    ) as compressed, tarfile.open(fileobj=compressed, mode="w", dereference=True) as archive:
        for path in entries:
            name = path.relative_to(package.parent).as_posix()
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() else 0o644
            expanded += info.size
            if path.is_file():
                with path.open("rb") as content:
                    archive.addfile(info, content)
            else:
                archive.addfile(info)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "archive_root": package.name,
        "filename": destination.name,
        "sha256": digest.hexdigest(),
        "bytes": destination.stat().st_size,
        "uncompressed_bytes": expanded,
        "member_count": len(entries),
    }


def main() -> int:
    from patient_tool_gym.compact_dataset import export_compact_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="Fresh staging directory; never overwritten.")
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--expected-source-hash", required=True)
    parser.add_argument("--synthea-root", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", args.version):
        parser.error("version must have the form v1.0.0")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    source_root = args.source.resolve(strict=True)
    output_root = args.output.resolve()
    if output_root == source_root or source_root in output_root.parents:
        parser.error("output must be outside the frozen source release")
    for name in ("LICENSE", "NOTICE"):
        if not (args.synthea_root / name).is_file():
            parser.error(f"missing upstream {name}")
    package = args.output / f"patientflowsynth-train-dev-{args.version}"
    package.mkdir(parents=True)
    manifest = export_compact_dataset(
        args.source,
        package / "release",
        splits=("train", "dev"),
        expected_source_hash=args.expected_source_hash,
    )
    # Export publishes its output only after full chart reconstruction and task replay.
    print(json.dumps({"stage": "exported_and_replayed", "counts": manifest["counts"]}), flush=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "LICENSE", package / "LICENSE")
    shutil.copyfile(args.synthea_root / "LICENSE", package / "SYNTHEA_LICENSE")
    shutil.copyfile(args.synthea_root / "NOTICE", package / "SYNTHEA_NOTICE")
    (package / "README.md").write_text(
        "# PatientFlowSynth Training and Development Data\n\n"
        "An exact train/dev subset of the paper's frozen synthetic benchmark. "
        "Original task definitions, chart bytes, and split assignments are preserved. "
        "The test set is not included.\n\n"
        "Charts are reconstructed from base FHIR Bundles and replayable edit chains. "
        "Install the PatientFlowSynth code, then run `patientflowsynth verify release` "
        "from this directory. See the repository's runtime guide to run an agent.\n\n"
        "Patients were generated with Synthea. Facility names and clinical terminology "
        "may come from upstream directories. The PatientFlowSynth code license is "
        "Apache-2.0; upstream licensing and terminology notices are preserved in "
        "SYNTHEA_LICENSE and SYNTHEA_NOTICE.\n",
        encoding="utf-8",
    )
    archive = args.output / f"{package.name}.tar.gz"
    descriptor = create_archive(package, archive)
    descriptor.update({
        "schema_version": "patientflowsynth.data-download.v1",
        "url": (
            "https://github.com/anthony-c-cuturrufo/PatientFlowSynth/releases/download/"
            f"data-{args.version}/{archive.name}"
        ),
        "dataset_hash": manifest["package_hash"],
        "source_release_hash": args.expected_source_hash,
        "included_splits": ["train", "dev"],
        "counts": manifest["counts"],
    })
    (args.output / "data-release.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    (args.output / "SHA256SUMS").write_text(f"{descriptor['sha256']}  {archive.name}\n")
    print(json.dumps({"stage": "packaged", **descriptor}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
