import copy

import pytest

from runtime.images.webarena_browser import _worker_update_compatible


@pytest.fixture
def inputs():
    return {"browser_source_revision": "revision", "browser_source_sha256": {"env.py": "source"},
            "base_image": "python@sha256:base", "base_image_id": "sha256:base",
            "worker_sha256": "new-worker", "requirements_sha256": "requirements",
            "dockerfile_sha256": "dockerfile"}


def test_worker_update_keeps_validated_browser_dependencies(inputs):
    previous = {**inputs, "worker_sha256": "old-worker", "selftest": {"passed": True}}
    assert _worker_update_compatible(previous, inputs)


@pytest.mark.parametrize("field", ["browser_source_revision", "browser_source_sha256",
    "base_image", "base_image_id", "requirements_sha256", "dockerfile_sha256"])
def test_worker_update_rejects_changed_dependency_identity(inputs, field):
    previous = {**copy.deepcopy(inputs), "selftest": {"passed": True}}
    previous[field] = "changed"
    assert not _worker_update_compatible(previous, inputs)


@pytest.mark.parametrize("selftest", [{}, {"passed": False}, {"passed": "true"}])
def test_worker_update_requires_real_passed_parent_contract(inputs, selftest):
    assert not _worker_update_compatible({**inputs, "selftest": selftest}, inputs)
