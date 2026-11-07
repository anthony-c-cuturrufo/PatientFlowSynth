from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import download_data as downloader
from scripts.package_data import create_archive


def _package(tmp_path: Path):
    package = tmp_path / "patientflowsynth-test-v1.0.0"
    (package / "release").mkdir(parents=True)
    (package / "release" / "manifest.json").write_text(json.dumps({"test": True}))
    archive = tmp_path / "test.tar.gz"
    descriptor = create_archive(package, archive)
    descriptor["url"] = "https://github.com/example/test.tar.gz"
    return package, archive, descriptor


def _response(monkeypatch, contents: bytes):
    class Response(io.BytesIO):
        def geturl(self):
            return "https://release-assets.githubusercontent.com/test.tar.gz"

    monkeypatch.setattr(downloader, "urlopen", lambda *_args, **_kwargs: Response(contents))


def test_archive_is_deterministic_and_download_is_verified(tmp_path, monkeypatch):
    package, archive, descriptor = _package(tmp_path)
    other = tmp_path / "again.tar.gz"
    assert create_archive(package, other)["sha256"] == descriptor["sha256"]
    _response(monkeypatch, archive.read_bytes())
    output = tmp_path / "downloaded"
    result = downloader.download_data(descriptor, output)
    assert result == output / "release"
    assert json.loads((result / "manifest.json").read_text()) == {"test": True}
    with pytest.raises(ValueError, match="already exists"):
        downloader.download_data(descriptor, output)


def test_corrupt_download_is_not_extracted(tmp_path, monkeypatch):
    _, archive, descriptor = _package(tmp_path)
    _response(monkeypatch, archive.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="checksum"):
        downloader.download_data(descriptor, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", ["traversal", "windows_drive", "symlink", "hardlink", "duplicate"])
def test_unsafe_archive_members_are_rejected(tmp_path, monkeypatch, kind):
    _, _, descriptor = _package(tmp_path)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        name = descriptor["archive_root"] + "/file"
        if kind == "traversal":
            name += "/../../escaped"
        elif kind == "windows_drive":
            name += "/C:/escaped"
        info = tarfile.TarInfo(name)
        if kind in {"symlink", "hardlink"}:
            info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            info.linkname = "../../escaped"
        archive.addfile(info)
        if kind == "duplicate":
            archive.addfile(info)
        payload = tarfile.TarInfo(descriptor["archive_root"] + "/payload")
        payload.size = 1
        archive.addfile(payload, io.BytesIO(b"x"))
    data = raw.getvalue()
    descriptor.update({
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        "uncompressed_bytes": 1, "member_count": 3 if kind == "duplicate" else 2,
    })
    _response(monkeypatch, data)
    with pytest.raises(ValueError, match="unsafe or duplicate"):
        downloader.download_data(descriptor, tmp_path / "output")
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "output").exists()


def test_packaging_rejects_symlinks(tmp_path):
    package, _, _ = _package(tmp_path)
    (package / "bad").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="regular files"):
        create_archive(package, tmp_path / "unsafe.tar.gz")


def test_packaging_materializes_hardlinked_files(tmp_path, monkeypatch):
    package, _, _ = _package(tmp_path)
    (package / "copy.json").hardlink_to(package / "release" / "manifest.json")
    archive = tmp_path / "hardlinked.tar.gz"
    descriptor = create_archive(package, archive)
    descriptor["url"] = "https://github.com/example/hardlinked.tar.gz"
    with tarfile.open(archive) as contents:
        assert all(member.isfile() or member.isdir() for member in contents)
    _response(monkeypatch, archive.read_bytes())
    release = downloader.download_data(descriptor, tmp_path / "downloaded")
    assert (release.parent / "copy.json").read_bytes() == (release / "manifest.json").read_bytes()
