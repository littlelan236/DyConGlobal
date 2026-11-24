import pytest
import yaml
from RL.dataset.branch_probe import preflight

def test_requires_explicit_dataset_config(tmp_path):
    config = tmp_path / "probe.yaml"
    config.write_text(yaml.safe_dump({}))
    with pytest.raises(KeyError, match="dataset_root"):
        preflight(config)

def test_cli_accepts_managed_execution_output():
    from argparse import ArgumentParser
    from RL.dataset.cli import register
    parser = ArgumentParser()
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["rl-dataset", "branch-probe", "--probe-config", "x.yaml", "--output", "out"])
    assert args.rl_dataset_command == "branch-probe"
    assert args.output == "out"

def test_scoring_consumes_capture_even_when_export_is_disabled():
    from runtime.serving.vllm_plugin import capture_rows_need_processing
    raw = [[object()]]
    assert capture_rows_need_processing(raw, export_enabled=False, scoring_enabled=True)
    assert not capture_rows_need_processing(raw, export_enabled=False, scoring_enabled=False)
