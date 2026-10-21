import gzip
import hashlib
import io
import tarfile
from pathlib import Path

from core.io import read_json
from runtime.containers.repairs import environment_overrides
from runtime.images.ownership import normalize_layer


def test_uid_repair_preserves_contents_links_modes_and_whiteouts(tmp_path):
    source = tmp_path / "source"
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w", format=tarfile.PAX_FORMAT) as tar:
        file = tarfile.TarInfo("testbed/space name")
        file.uid, file.gid, file.mode, file.size = 197609, 197121, 0o755, 7
        file.pax_headers = {"uid": "197609", "gid": "197121", "mtime": "12.5"}
        tar.addfile(file, io.BytesIO(b"payload"))
        link = tarfile.TarInfo("testbed/hardlink")
        link.type, link.linkname, link.uid = tarfile.LNKTYPE, file.name, 1000
        tar.addfile(link)
        whiteout = tarfile.TarInfo("testbed/.wh.old")
        tar.addfile(whiteout)
    original = gzip.compress(data.getvalue(), mtime=0)
    source.mkdir()
    path = source / hashlib.sha256(original).hexdigest()
    path.write_bytes(original)
    dest, cache = tmp_path / "blobs", tmp_path / "cache"
    dest.mkdir()
    cache.mkdir()
    result = normalize_layer(path, dest, cache)
    output = dest / result["descriptor"]["digest"][7:]
    assert path.read_bytes() == original
    assert "sha256:" + hashlib.sha256(gzip.decompress(output.read_bytes())).hexdigest() == result["diff_id"]
    with tarfile.open(output) as tar:
        file, link, whiteout = tar.getmembers()
        assert (file.uid, file.gid, file.mode) == (0, 0, 0o755)
        assert tar.extractfile(file).read() == b"payload"
        assert (link.uid, link.type, link.linkname) == (1000, tarfile.LNKTYPE, file.name)
        assert whiteout.name == "testbed/.wh.old"
    assert normalize_layer(path, dest, cache) == result
    assert read_json(next(cache.glob("*.json")))["changed_entries"] == 1


def test_existing_ownership_needs_no_derived_blob(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        file = tarfile.TarInfo("ordinary")
        file.uid, file.gid = 65535, 1000
        tar.addfile(file)
    blob = gzip.compress(data.getvalue(), mtime=0)
    source = tmp_path / hashlib.sha256(blob).hexdigest()
    source.write_bytes(blob)
    dest, cache = tmp_path / "blobs", tmp_path / "cache"
    dest.mkdir()
    cache.mkdir()
    result = normalize_layer(source, dest, cache)
    assert result["changed_entries"] == 0
    assert result["descriptor"]["digest"] == "sha256:" + source.name


def test_repairs_bound_threads_and_use_shared_offline_policy(cfg):
    iid = "scikit-learn__scikit-learn-13933"
    env = environment_overrides(cfg, {"instance_id": iid, "repo": "scikit-learn/scikit-learn"})
    assert env["variables"]["OMP_NUM_THREADS"] == "1"
    assert env["variables"]["OPENBLAS_NUM_THREADS"] == "1"
    network = environment_overrides(
        cfg, {"instance_id": "psf__requests-1713", "repo": "psf/requests"}
    )
    assert network["network_mode"] == "bridge"
    assert network["variables"]["NO_PROXY"] == "*"
    assert network["variables"]["HTTPS_PROXY"] == ""
    from runtime.containers.factory import ContainerFactory

    factory = ContainerFactory(cfg, client=object())
    row = {"instance_id": "psf__requests-2873", "repo": "psf/requests"}
    policy = factory.policy_for(row)
    assert "pytest-httpbin==1.0.2" in policy.specs(row["repo"])
    assert "--no-index" in policy.bootstrap(row["repo"])
    assert Path(policy.environment["wheelhouse"]).name == "gold600-wheels"


def test_wakeup_reports_unresolved_task_ids_without_sorting_dicts(tmp_path, monkeypatch):
    from core.io import write_json
    from runtime.scripts.wakeup_nonverified600_gold import retry_failed

    initial = tmp_path / "initial"
    initial.mkdir()
    (tmp_path / "state/gold-image-audit").mkdir(parents=True)
    write_json(initial / "audit-config.json", {"environment": {"repo_pins": {}}, "execution": {}, "images": {}})
    rows = [{"instance_id": iid, "repo": "ordinary/repo"} for iid in ["task-b", "task-a"]]
    summary = {"results": []}
    result = retry_failed(tmp_path, rows, summary, initial, "retry", 1, 0)
    assert result["failed"] == ["task-a", "task-b"]
    assert result["stage"] == "needs-investigation"
