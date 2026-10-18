import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from oos_detect.reproduce import backup


@pytest.fixture
def run(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    metadata = {"source": {"root": {"revision": "actual", "dirty": False}},
                "runtime": {"python": "3.12.13", "torch": "actual"},
                "data_hash": "data", "embedding_hash": "embedding"}
    (directory / "manifest.json").write_text(json.dumps(metadata))
    (directory / "config.json").write_text('{"seed": 0}')
    (directory / "phases.jsonl").write_text('{"step": 1}\n')
    (directory / "checkpoints").mkdir()
    (directory / "checkpoints/step-000001.pt").write_bytes(b"tiny checkpoint")
    return directory


def _legacy_backup(run, root, serial=1):
    """Build an original-format fixture, independent of the compressed writer."""
    root.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256(str(run.resolve()).encode()).hexdigest()
    generation = f"oos-backup-{run_id}-{serial:020d}-{'a' * 32}"
    archive = root / (generation + ".tar")
    files = {}
    with tarfile.open(archive, "x:", dereference=True) as target:
        for path in sorted(run.rglob("*")):
            if path.is_file():
                name = path.relative_to(run).as_posix()
                content = path.read_bytes()
                files[name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                target.add(path, arcname=name, recursive=False)
    manifest = {"format": "oos-detect-backup-v1", "generation": generation, "run_id": run_id,
                "archive": archive.name, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": files, "metadata": json.loads((run / "manifest.json").read_text())}
    (root / (generation + ".complete.json")).write_text(json.dumps(manifest))
    return manifest


def test_round_trip_all_retained_files_and_metadata(run, tmp_path):
    (run / "checkpoints/step-000002.pt").write_bytes(b"next checkpoint")
    (run / "phases.jsonl").write_text('{"step": 1}\n{"step": 2}\n')
    (run / "results.json").write_text('{"accuracy": 0.5}')
    root = tmp_path / "backups"
    manifest = backup.backup_run(run, root)
    expected = json.loads((run / "manifest.json").read_text())
    destination = tmp_path / "restored"
    assert backup.restore_run(root, destination, expected) == manifest
    assert set(manifest["files"]) == {
        path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()}
    for name in manifest["files"]:
        assert (destination / name).read_bytes() == (run / name).read_bytes()


@pytest.mark.parametrize("legacy", [False, True])
def test_corrupt_newest_falls_back(run, tmp_path, legacy):
    root = tmp_path / "backups"
    first = _legacy_backup(run, root) if legacy else backup.backup_run(run, root)
    (run / "phases.jsonl").write_text('{"step": 2}\n')
    newest = _legacy_backup(run, root, serial=2) if legacy else backup.backup_run(run, root)
    (root / newest["archive"]).write_bytes(b"corrupt")
    destination = tmp_path / "restore"
    assert backup.restore_run(root, destination) == first
    assert (destination / "phases.jsonl").read_text() == '{"step": 1}\n'


def test_partial_copy_preserves_previous(run, tmp_path, monkeypatch):
    root = tmp_path / "backups"
    first = backup.backup_run(run, root)
    original = backup._copy

    def interrupted(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(backup, "_copy", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        backup.backup_run(run, root)
    assert len(list(root.glob("*.complete.json"))) == 1
    assert len(list(root.glob("*.tar.gz"))) == 2
    monkeypatch.setattr(backup, "_copy", original)
    assert backup.restore_run(root, tmp_path / "restored") == first


def test_retention_protects_unrelated_incomplete_and_corrupt(run, tmp_path):
    root = tmp_path / "backups"
    corrupt = backup.backup_run(run, root)
    (root / corrupt["archive"]).write_bytes(b"corrupt")
    unrelated = root / "family-photo.tar"
    unrelated.write_bytes(b"user data")
    incomplete = root / ("oos-backup-" + "a" * 64 + "-" + "0" * 20 + "-" + "b" * 32 + ".tar")
    incomplete.write_bytes(b"partial")
    malformed = root / incomplete.name.replace(".tar", ".complete.json")
    malformed.write_text("{partial")
    generations = [backup.backup_run(run, root) for _ in range(3)]
    assert not (root / generations[0]["archive"]).exists()
    for manifest in generations[1:]:
        assert (root / manifest["archive"]).exists()
    assert (root / corrupt["archive"]).read_bytes() == b"corrupt"
    assert unrelated.read_bytes() == b"user data"
    assert incomplete.read_bytes() == b"partial"
    assert malformed.read_text() == "{partial"


@pytest.mark.parametrize("field,value", [
    ("data_hash", "different"), ("embedding_hash", "different"),
    ("source", {"root": {"revision": "fake"}}),
    ("runtime", {"python": "3.12.13", "torch": "different"}),
])
def test_metadata_mismatch_is_nonmutating(run, tmp_path, field, value):
    root = tmp_path / "backups"
    manifest = backup.backup_run(run, root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="metadata"):
        backup.restore_run(root, destination, {field: value})
    assert not destination.exists()
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(ValueError, match="conflicts"):
        backup.backup_run(run, root, {field: value})
    assert manifest["metadata"][field] != value


def test_destination_collision_and_no_backup(run, tmp_path):
    root = tmp_path / "backups"
    assert backup.restore_run(root, tmp_path / "absent") is None
    backup.backup_run(run, root)
    with pytest.raises(FileExistsError):
        backup.restore_run(root, run)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        backup.restore_run(root, dangling)
    assert not (tmp_path / "missing").exists()


def test_symlink_source_rejected(run, tmp_path):
    (run / "link").symlink_to(run / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        backup.backup_run(run, tmp_path / "backups")


@pytest.mark.parametrize("kind", ["traversal", "symlink", "hardlink", "file-hash"])
@pytest.mark.parametrize("legacy", [False, True])
def test_unsafe_archive_rejected_even_with_matching_archive_hash(run, tmp_path, kind, legacy):
    root = tmp_path / "backups"
    manifest = _legacy_backup(run, root) if legacy else backup.backup_run(run, root)
    archive = root / manifest["archive"]
    original_files = {name: (run / name).read_bytes() for name in manifest["files"]}
    with tarfile.open(archive, "w:" if legacy else "w:gz") as target:
        if kind == "file-hash":
            for name, content in original_files.items():
                if name == "config.json":
                    content = b"X" * len(content)
                member = tarfile.TarInfo(name)
                member.size = len(content)
                target.addfile(member, io.BytesIO(content))
        else:
            name = "../escaped" if kind == "traversal" else "manifest.json"
            member = tarfile.TarInfo(name)
            if kind in ("symlink", "hardlink"):
                member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                member.linkname = "../escaped"
                target.addfile(member)
            else:
                content = b"altered"
                member.size = len(content)
                target.addfile(member, io.BytesIO(content))
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / (manifest["generation"] + ".complete.json")).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="No valid"):
        backup.restore_run(root, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()
    assert not (tmp_path / "escaped").exists()


def test_free_space_guard_before_copy(run, tmp_path, monkeypatch):
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 0})())
    with pytest.raises(OSError, match="free space"):
        backup.backup_run(run, tmp_path / "backups")
    assert not (tmp_path / "backups").exists()


def test_drive_paths_rejected_without_access():
    with pytest.raises(ValueError, match="local"):
        backup.backup_run("/content/drive/MyDrive/run", "/content/backups")
    with pytest.raises(ValueError, match="local"):
        backup.restore_run("/content/backups", "/content/drive/MyDrive/new-run")


def test_failed_readback_does_not_publish_completion(run, tmp_path, monkeypatch):
    root = tmp_path / "backups"
    first = backup.backup_run(run, root)
    original = backup._copy

    def corrupt_copy(source, destination):
        original(source, destination)
        Path(destination).write_bytes(b"bad copy")

    monkeypatch.setattr(backup, "_copy", corrupt_copy)
    with pytest.raises(ValueError, match="SHA256"):
        backup.backup_run(run, root)
    assert len(list(root.glob("*.complete.json"))) == 1
    monkeypatch.setattr(backup, "_copy", original)
    assert backup.restore_run(root, tmp_path / "restore") == first


def test_corrupt_completion_ignored(run, tmp_path):
    root = tmp_path / "backups"
    first = backup.backup_run(run, root)
    newest = backup.backup_run(run, root)
    (root / (newest["generation"] + ".complete.json")).write_text("{partial")
    assert backup.restore_run(root, tmp_path / "restore") == first


def test_retention_is_per_run(run, tmp_path):
    other = tmp_path / "other-run"
    other.mkdir()
    (other / "manifest.json").write_bytes((run / "manifest.json").read_bytes())
    root = tmp_path / "backups"
    protected = backup.backup_run(other, root)
    for _ in range(3):
        backup.backup_run(run, root)
    assert (root / protected["archive"]).exists()
    assert (root / (protected["generation"] + ".complete.json")).exists()


def test_level_one_compression_preserves_full_phase_log(run, tmp_path, monkeypatch):
    records = [{"phase": "generator_entropy", "boundary_serial": step,
                "epoch": step // 100, "batch": step % 100, "loss": 0.693147,
                "temperature": 0.5, "policy_baseline": 0.123456,
                "checkpoint": None} for step in range(6000)]
    content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode()
    (run / "phases.jsonl").write_bytes(content)
    original = backup.tarfile.open
    levels = []

    def observe_open(name, mode="r", **kwargs):
        if mode == "x:gz":
            levels.append(kwargs.get("compresslevel"))
        return original(name, mode, **kwargs)

    monkeypatch.setattr(backup.tarfile, "open", observe_open)
    root = tmp_path / "backups"
    manifest = backup.backup_run(run, root)
    archive = root / manifest["archive"]
    assert levels == [1]
    assert archive.name.endswith(".tar.gz")
    assert archive.read_bytes()[:2] == b"\x1f\x8b"
    assert archive.stat().st_size < len(content) // 4
    destination = tmp_path / "restore"
    assert backup.restore_run(root, destination) == manifest
    assert (destination / "phases.jsonl").read_bytes() == content
    assert (run / "phases.jsonl").read_bytes() == content


def test_original_uncompressed_fixture_restores_without_changes(run, tmp_path):
    root = tmp_path / "backups"
    manifest = _legacy_backup(run, root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    destination = tmp_path / "restore"
    assert backup.restore_run(root, destination, manifest["metadata"]) == manifest
    for name in manifest["files"]:
        assert (destination / name).read_bytes() == (run / name).read_bytes()
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}


def test_mixed_format_retention_counts_verified_generations_together(run, tmp_path):
    root = tmp_path / "backups"
    legacy = _legacy_backup(run, root)
    original_archive = (root / legacy["archive"]).read_bytes()
    original_completion = (root / (legacy["generation"] + ".complete.json")).read_bytes()
    compressed = backup.backup_run(run, root)
    assert (root / legacy["archive"]).read_bytes() == original_archive
    assert (root / (legacy["generation"] + ".complete.json")).read_bytes() == original_completion
    newest = backup.backup_run(run, root)
    assert not (root / legacy["archive"]).exists()
    assert not (root / (legacy["generation"] + ".complete.json")).exists()
    assert (root / compressed["archive"]).exists()
    assert (root / newest["archive"]).exists()


def test_truncated_gzip_with_matching_whole_hash_falls_back(run, tmp_path):
    root = tmp_path / "backups"
    legacy = _legacy_backup(run, root)
    newest = backup.backup_run(run, root)
    archive = root / newest["archive"]
    content = archive.read_bytes()
    archive.write_bytes(content[:len(content) // 2])
    newest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / (newest["generation"] + ".complete.json")).write_text(json.dumps(newest))
    assert backup.restore_run(root, tmp_path / "restore") == legacy
