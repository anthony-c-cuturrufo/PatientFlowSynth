#!/usr/bin/env python3
"""Download and checksum-verify the versioned PatientFlowSynth data archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def download_data(descriptor: dict, output: Path) -> Path:
    """Extract only regular files below a fresh destination after verifying SHA-256."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    archive_root = descriptor["archive_root"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", archive_root):
        raise ValueError("invalid archive root")
    if not re.fullmatch(r"[a-f0-9]{64}", descriptor["sha256"]):
        raise ValueError("invalid SHA-256")
    url = urlsplit(descriptor["url"])
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("download URL must use HTTPS without embedded credentials")
    for field in ("bytes", "uncompressed_bytes", "member_count"):
        if type(descriptor[field]) is not int or descriptor[field] <= 0:
            raise ValueError(f"invalid archive {field}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".patientflowsynth-download-", dir=output.parent
    ) as tmp:
        temporary = Path(tmp)
        archive_path = temporary / "download.tar.gz"
        digest = hashlib.sha256()
        received = 0
        request = Request(descriptor["url"], headers={"User-Agent": "PatientFlowSynth-data/1.0"})
        with urlopen(request, timeout=120) as response, archive_path.open("xb") as destination:
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("download redirected outside HTTPS")
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > descriptor["bytes"]:
                    raise ValueError("download exceeds the published size")
                digest.update(chunk)
                destination.write(chunk)
        if received != descriptor["bytes"] or digest.hexdigest() != descriptor["sha256"]:
            raise ValueError("download checksum or size does not match the published archive")

        extracted = temporary / "extracted"
        extracted.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != descriptor["member_count"]:
                raise ValueError("archive member count does not match")
            if sum(member.size for member in members) != descriptor["uncompressed_bytes"]:
                raise ValueError("archive expanded size does not match")
            seen: set[str] = set()
            for member in members:
                name = member.name.rstrip("/")
                path = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or ":" in name
                    or path.is_absolute()
                    or path.as_posix() != name
                    or ".." in path.parts
                    or path.parts[0] != archive_root
                    or name in seen
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("archive contains an unsafe or duplicate member")
                seen.add(name)
            for member in members:
                target = extracted.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("archive member has no file content")
                with stream, target.open("xb") as destination:
                    shutil.copyfileobj(stream, destination)
                target.chmod(0o644)
        package = extracted / archive_root
        if not (package / "release" / "manifest.json").is_file():
            raise ValueError("archive contains no compact dataset manifest")
        if output.exists() or output.is_symlink():
            raise ValueError(f"output already exists: {output}")
        package.rename(output)
    return output / "release"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor", type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "data-release.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        descriptor = json.loads(args.descriptor.read_text())
        output = args.output or Path("data") / descriptor["archive_root"]
        release = download_data(descriptor, output)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        parser.exit(1, f"Data download failed: {exc}\n")
    print(f"Downloaded and checksum-verified: {release}")
    print(f"Next: patientflowsynth verify {release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
