import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

docker = pytest.importorskip("docker")
from core.io import read_json, write_json
from runtime.images.nas import ImageLeaseManager


class Images:
    def __init__(self):
        self.ids = set()
        self.removed = []

    def get(self, image_id):
        if image_id not in self.ids:
            raise docker.errors.ImageNotFound(image_id)
        return SimpleNamespace(id=image_id)

    def remove(self, image_id, force=False):
        self.ids.discard(image_id)
        self.removed.append(image_id)


class Client:
    def __init__(self, root):
        self.images = Images()
        self.containers = SimpleNamespace(list=lambda **kwargs: [])
        self.root = root

    def info(self):
        return {"ID": "test-daemon", "DockerRootDir": str(self.root)}


def store(i):
    return SimpleNamespace(image_id="sha256:" + f"{i:064x}", record={"compressed_bytes": 1})


def test_preexisting_image_is_never_enrolled_or_removed(cfg):
    client = Client(cfg["project_root"])
    manager = ImageLeaseManager(cfg, client)
    image = store(1)
    client.images.ids.add(image.image_id)
    with manager.lease(image):
        pass
    assert manager.cleanup() == {}
    assert client.images.removed == []


def test_four_independent_loads_overlap(cfg, monkeypatch):
    client = Client(cfg["project_root"])
    manager = ImageLeaseManager(cfg, client)
    lock = threading.Lock()
    active = 0
    peak = 0
    barrier = threading.Barrier(4, timeout=3)

    def load(image):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        barrier.wait()
        time.sleep(0.02)
        client.images.ids.add(image.image_id)
        with lock:
            active -= 1

    monkeypatch.setattr(manager, "_load", load)

    def worker(i):
        with manager.lease(store(i)):
            pass

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(1, 5)))
    assert peak == 4
    assert len(client.images.removed) == 4


def test_same_image_loaded_once_and_lease_blocks_cleanup(cfg, monkeypatch):
    client = Client(cfg["project_root"])
    manager = ImageLeaseManager(cfg, client)
    loads = []

    def load(image):
        loads.append(image.image_id)
        client.images.ids.add(image.image_id)

    monkeypatch.setattr(manager, "_load", load)
    barrier = threading.Barrier(4, timeout=3)

    def worker(i):
        with manager.lease(store(1)):
            barrier.wait()
            assert manager.cleanup(store(1).image_id).get(store(1).image_id) == "leased"

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    assert len(loads) == 1
    assert client.images.removed == [store(1).image_id]


def test_crashed_reservation_recovered(cfg):
    manager = ImageLeaseManager(cfg, Client(cfg["project_root"]))
    write_json(
        manager.state / "journal.json",
        {"images": {}, "reservations": {"dead": {"bytes": 9999999, "pid": 99999999, "owner": "dead"}}},
    )
    with manager._reservation(1):
        assert "dead" not in read_json(manager.state / "journal.json")["reservations"]
    assert read_json(manager.state / "journal.json")["reservations"] == {}
