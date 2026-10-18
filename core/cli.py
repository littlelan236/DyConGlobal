"""Light CLI; heavy model, agent and Docker dependencies are lazy imports."""

import argparse
import json
from pathlib import Path

from baseline.catalog import Catalog
from core.config import resolve
from core.io import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dycon")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[])
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ["config-check", "catalog-check", "runtime-check", "recover"]:
        sub.add_parser(cmd)
    sub.add_parser("capture-recover").add_argument("run_id")
    p = sub.add_parser("online-index")
    p.add_argument("run_id")
    p.add_argument("index_id")
    for cmd in ["tasks-import", "rollouts-import", "evaluations-import"]:
        sub.add_parser(cmd).add_argument("list")
    sub.add_parser("model-register").add_argument("checkpoint")
    p = sub.add_parser("rollout")
    p.add_argument("run_id")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--task-id", action="append", default=[])
    p.add_argument("--unrolled", action="store_true")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Run the real tool/capture startup gate without admitting tasks",
    )
    sub.add_parser("prepare-evaluation").add_argument("run_id")
    p = sub.add_parser("evaluate")
    p.add_argument("run_id")
    p.add_argument("eval_id")
    p.add_argument("--gold", action="store_true")
    sub.add_parser("enrich").add_argument("dataset_id")
    sub.add_parser("token-trace").add_argument("dataset_id")
    sub.add_parser("trajectory-token-audit").add_argument("dataset_id")
    p = sub.add_parser("hidden")
    p.add_argument("dataset_id")
    p.add_argument("capture_id")
    p = sub.add_parser("images-check")
    p.add_argument("--deep", action="store_true")
    p = sub.add_parser("baseline-select")
    p.add_argument("profile")
    p.add_argument("task_key")
    p.add_argument("rollout_id")
    p = sub.add_parser("token-join")
    p.add_argument("dataset_id")
    p.add_argument("capture_id")
    p.add_argument("index_id")
    sub.add_parser("hidden-token-join-audit").add_argument("index_id")
    p = sub.add_parser("dataset-index")
    p.add_argument("dataset_id")
    p.add_argument("capture_id")
    p.add_argument("index_id")
    p = sub.add_parser("action-index")
    p.add_argument("shared_index_id")
    p.add_argument("index_id")
    p = sub.add_parser("difficulty-evaluate")
    p.add_argument("index_id")
    p.add_argument("predictions")
    p.add_argument("evaluation_id")
    a = parser.parse_args(argv)
    if a.command == "rollout" and a.set:
        parser.error("rollout rejects --set; edit the unique config.yaml or resume its immutable snapshot")
    if a.command == "rollout" and a.resume and (a.task_id or a.unrolled):
        parser.error("resume uses the immutable task selection")
    cfg = resolve(a.config, a.set)
    catalog = Catalog(cfg["project_root"])
    if a.command == "config-check":
        result = cfg
    elif a.command == "catalog-check":
        result = catalog.verify()
    elif a.command == "runtime-check":
        from baseline.workflow import assert_runtime_ready

        assert_runtime_ready(cfg)
        result = {"ready": True}
    elif a.command == "capture-recover":
        from hidden.recover_online import recover_online

        result = recover_online(cfg, a.run_id)
    elif a.command == "online-index":
        from difficulty.datasets.online import build_published_index

        result = str(build_published_index(cfg, a.run_id, a.index_id))
    elif a.command == "tasks-import":
        result = catalog.register_tasks(read_json(a.list))
    elif a.command == "rollouts-import":
        from baseline.importers.legacy import import_list

        result = import_list(cfg["project_root"], a.list)
    elif a.command == "evaluations-import":
        from baseline.importers.evaluation import import_evaluation

        result = [import_evaluation(cfg["project_root"], spec) for spec in read_json(a.list)]
    elif a.command == "model-register":
        from core.model_identity import register

        result = {"fingerprint": register(cfg["project_root"], a.checkpoint)}
    elif a.command == "rollout":
        if a.resume and a.set:
            raise ValueError("Resume uses immutable snapshot; --set requires a new run")
        from baseline.rollout.runner import run_manifest

        result = run_manifest(
            cfg, a.run_id, a.resume, task_ids=a.task_id, unrolled=a.unrolled, check_only=a.check_only
        )
    elif a.command == "prepare-evaluation":
        from baseline.workflow import prepare_run

        result = str(
            prepare_run(
                cfg, a.run_id, partition=cfg["benchmark"]["partition"], purposes=cfg["benchmark"]["purposes"]
            )
        )
    elif a.command == "evaluate":
        if a.set:
            raise ValueError("Evaluation uses the saved run configuration")
        from baseline.evaluation.harness import evaluate
        from core.config import load_snapshot

        cfg = load_snapshot(cfg["project_root"], a.run_id)
        result = evaluate(cfg, Path(cfg["project_root"]) / "runs" / a.run_id, a.eval_id, a.gold)
    elif a.command == "enrich":
        from transformers import AutoTokenizer

        from core.model_identity import verify_model
        from enrichment.trajectory import enrich

        verify_model(cfg)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["path"], local_files_only=True, trust_remote_code=cfg["model"]["trust_remote_code"]
        )
        result = str(enrich(cfg, a.dataset_id, tokenizer))
    elif a.command == "token-trace":
        from transformers import AutoTokenizer

        from enrichment.token_trace import build_token_traces

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["path"], local_files_only=True, trust_remote_code=cfg["model"]["trust_remote_code"]
        )
        result = [str(path) for path in build_token_traces(cfg, a.dataset_id, tokenizer)]
    elif a.command == "trajectory-token-audit":
        from enrichment.token_trace import audit_token_traces

        result = audit_token_traces(cfg, a.dataset_id)
    elif a.command == "hidden":
        if cfg["hidden"]["adapter"] == "vllm_eager":
            import subprocess

            args = [
                cfg["hidden"]["vllm"]["python"],
                "-m",
                "hidden.scripts.extract_vllm",
                "--config",
                cfg["config_source"],
            ]
            for override in a.set:
                args += ["--set", override]
            subprocess.run([*args, a.dataset_id, a.capture_id], cwd=cfg["project_root"], check=True)
            return
        else:
            from hidden.adapters.hf import load_adapter
            from hidden.extract import extract_hidden

            result = str(extract_hidden(cfg, a.dataset_id, a.capture_id, load_adapter(cfg)))
    elif a.command == "images-check":
        from runtime.images.nas import NasImageStore

        tasks = catalog.select_tasks(partition=cfg["benchmark"]["partition"])["tasks"]
        if not tasks:
            raise ValueError("Task scope is empty")
        result = {
            k: len(NasImageStore(cfg, t["instance"]["nas_image"]).verify(a.deep)) for k, t in tasks.items()
        }
    elif a.command == "baseline-select":
        catalog.select_baseline(a.profile, a.task_key, a.rollout_id)
        result = {"selected": a.rollout_id}
    elif a.command == "token-join":
        from difficulty.datasets.token_join import build_token_hidden_index

        result = str(build_token_hidden_index(cfg, a.dataset_id, a.capture_id, a.index_id))
    elif a.command == "hidden-token-join-audit":
        from difficulty.datasets.token_join import audit_token_hidden_index

        result = audit_token_hidden_index(cfg, a.index_id)
    elif a.command == "dataset-index":
        from difficulty.datasets.index import build_index

        result = str(build_index(cfg, a.dataset_id, a.capture_id, a.index_id))
    elif a.command == "action-index":
        from action.datasets.index import create_action_index

        result = str(create_action_index(cfg, a.shared_index_id, a.index_id))
    elif a.command == "difficulty-evaluate":
        from difficulty.evaluation.run import evaluate_predictions

        result = str(evaluate_predictions(cfg, a.index_id, a.predictions, a.evaluation_id))
    else:
        from runtime.containers.recovery import recover

        result = recover(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


def module_main(command):
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[])
    known, rest = parser.parse_known_args(sys.argv[1:])
    args = ["--config", known.config]
    for override in known.set:
        args += ["--set", override]
    main([*args, command, *rest])
