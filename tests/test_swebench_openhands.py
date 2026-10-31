import pytest


def test_swe_openhands_profile_is_distinct_from_benchflow():
    from runtime.agents.swebench_openhands import SWEOpenHandsProfile, validate_profile

    profile = SWEOpenHandsProfile(bridge_listen="127.0.0.1", bridge_advertise="127.0.0.1")
    assert profile.workspace == "/testbed"
    assert "skill_mode" not in validate_profile(profile.model_dump())
    with pytest.raises(ValueError, match="under /testbed"):
        SWEOpenHandsProfile(bridge_listen="127.0.0.1", bridge_advertise="127.0.0.1",
                            task_cwd="/tmp")


def test_patch_export_records_empty_and_nonempty(tmp_path):
    from runtime.backends.native.swe_patch import publish_patch, validate_diff

    empty = validate_diff("")
    assert empty["empty"] is True
    diff = "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    patch, receipt = publish_patch(tmp_path, diff, binding={"attempt_id": "a1"}, base_commit="abc")
    assert patch.read_text() == diff
    assert receipt["status"] == "submitted"
    assert receipt["base_commit"] == "abc"


def test_patch_export_rejects_path_escape():
    from runtime.backends.native.swe_patch import validate_diff

    with pytest.raises(ValueError, match="unsafe path"):
        validate_diff("--- a/../secret\n+++ b/../secret\n@@ -1 +1 @@\n-a\n+b\n")


def test_native_swe_openhands_adapter_has_explicit_binding():
    from runtime.backends.registry import adapter_for, resolve

    cfg = {"benchmark": {"backend": "swebench"}, "agent": {"backend": "openhands"}}
    assert resolve(cfg).adapter == "native_swe_openhands"
    assert adapter_for(cfg).__class__.__name__ == "NativeSWEOpenHandsAdapter"
