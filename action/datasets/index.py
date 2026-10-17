"""Reserve unlabeled action samples sharing the same latent artifacts."""

from core.io import artifact_dir, read_json, sha256, write_json


def create_action_index(cfg, difficulty_index_id, action_index_id):
    source = artifact_dir(cfg, "difficulty", difficulty_index_id, "datasets") / "index.json"
    data = read_json(source)
    out = artifact_dir(cfg, "action", action_index_id, "datasets")
    if out.exists():
        raise FileExistsError(out)
    write_json(
        out / "index.json",
        {
            "schema_version": 1,
            "taxonomy_version": None,
            "shared_index": difficulty_index_id,
            "shared_index_sha256": sha256(source),
            "samples": [
                {"sample_id": row["sample_id"], "action_label": None, "label_source": None}
                for row in data["samples"]
            ],
        },
    )
    return out
