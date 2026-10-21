"""Run a published predictor on receipt-bound replay hidden; no fitting."""
import importlib
from pathlib import Path
import numpy as np
from core.io import artifact_dir, owned_path, read_json, sha256, write_json
from runtime.serving.parallel import artifact_contract
from difficulty.datasets.paths import dataset_directory, source_directory


def predict_hidden(cfg, index_id, predictor, prediction_id):
    index_path=dataset_directory(cfg,index_id)/"index.json"
    index=read_json(index_path)
    if index.get("schema_version")!=1:
        raise ValueError("Prediction currently requires offline schema 1 index")
    source=source_directory(cfg,index,index_path.parent,"enrichment")
    capture=source_directory(cfg,index,index_path.parent,"hidden")
    receipt=read_json(capture/"receipt.json")
    if sha256(capture/"receipt.json")!=index["capture_receipt_sha256"]:
        raise ValueError("Capture receipt changed")
    if sha256(source/"receipt.json")!=index["source_receipt_sha256"]:
        raise ValueError("Enrichment receipt changed")
    path=Path(predictor)
    if not path.is_absolute():path=Path(cfg["project_root"])/path
    contract=artifact_contract(path,model_fingerprint=cfg["model"]["fingerprint"])
    target=contract["target"]
    targets={"submission_remaining_reasoning_tokens_log1p":("remaining_reasoning_tokens",True),
             "submission_remaining_steps_linear":("remaining_steps",False)}
    if target not in targets or contract["endpoint"]!="submission":
        raise ValueError("Unsupported predictor target/endpoint for offline workload scoring")
    label_target,log1p=targets[target]
    if not set(contract["sites"])<=set(receipt["layers"]):
        raise ValueError("Replay did not capture all predictor sites")
    module,name=contract["loader"].split(":")
    provider=getattr(importlib.import_module(module),name)(path.parent,contract)
    labels={}
    for name in {row["label_shard"] for row in index["samples"]}:
        labels.update({r["sample_id"]:r for r in read_json(owned_path(source,name))})
    rows=[];features=[]
    for row in index["samples"]:
        label=labels[row["sample_id"]]
        # The exported predictor was trained at think_close; open is another distribution.
        if label["anchor"]!="think_close":continue
        tensor=owned_path(capture,row["hidden_path"])
        if sha256(tensor)!=receipt["files"][row["hidden_path"]]:raise ValueError("Hidden tensor changed")
        with np.load(tensor,allow_pickle=False) as data:
            features.append(data["hidden"][[receipt["layers"].index(i) for i in contract["sites"]]])
        rows.append(row)
    if not rows:raise ValueError("No close samples in index")
    scores=np.asarray(provider.predict(np.stack(features)),dtype=float)
    if scores.shape!=(len(rows),) or not np.isfinite(scores).all():raise ValueError("Invalid predictor output")
    native=np.expm1(scores) if log1p else scores
    if not np.isfinite(native).all():raise ValueError("Prediction inverse transform overflowed")
    output=artifact_dir(cfg,"difficulty",prediction_id,"predictions")
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"predictions.json",[{"sample_id":r["sample_id"],"target":label_target,"prediction":float(v)}
                                        for r,v in zip(rows,native,strict=True)])
    write_json(output/"native-scores.json",[{"sample_id":r["sample_id"],"score":float(v)}
                                         for r,v in zip(rows,scores,strict=True)])
    write_json(output/"receipt.json",{"schema_version":1,"state":"complete","predictor_sha256":sha256(path),
               "source_files":{str(index_path):sha256(index_path),str(path):sha256(path)},
               "index_sha256":sha256(index_path),"target":target,"endpoint":"submission",
               "sites":contract["sites"],"count":len(rows),"negative_predictions_preserved":True,
               "files":{n:sha256(output/n) for n in ["predictions.json","native-scores.json"]}})
    return {"output":str(output),"predictions":str(output/"predictions.json"),"count":len(rows),"target":target}
