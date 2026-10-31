import pytest
from application.execution import harness


def test_workspace_subject_never_falls_through_to_swe_patch_harness(monkeypatch,tmp_path):
    monkeypatch.setattr(harness,'load_plan',lambda *args:{'config':{'benchmark':{'backend':'skillsbench'}}})
    with pytest.raises(ValueError,match='live official verifier'):
        harness.prepare(tmp_path,'run')
    with pytest.raises(ValueError,match='complete state snapshot'):
        harness.run(tmp_path,'run','evaluation')
