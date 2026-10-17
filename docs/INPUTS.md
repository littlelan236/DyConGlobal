# 显式输入格式

`tasks-import` 接受 JSON list。以下是格式示例，不是实际 benchmark 范围；instance 需携带对应 SWE-bench 4.1.0 harness 所需原始字段。

```json
[
  {
    "benchmark": "SWE-bench_Verified",
    "revision": "PINNED_DATASET_REVISION",
    "instance_id": "org__repo-123",
    "repo": "org/repo",
    "base_commit": "BASE_COMMIT",
    "partition": "train",
    "exclusion_reason": null,
    "instance": {
      "instance_id": "org__repo-123",
      "repo": "org/repo",
      "base_commit": "BASE_COMMIT",
      "problem_statement": "...",
      "nas_image": {
        "store": "images/COHORT",
        "manifest": "images/TASK.json",
        "manifest_sha256": "SHA256",
        "image_id": "sha256:MANIFEST_DIGEST"
      }
    }
  }
]
```

NAS 引用的 `store` 相对 config.paths.nas_root；manifest 内包含 manifest_digest、manifest_size、config/layers OCI descriptors 和 compressed_bytes。blob 固定在 `<store>/blobs/sha256/<digest>`。
从旧 NAS 复用时先将静态资产复制/发布到新项目 NAS namespace 并校验 digest，不能直接把旧可变工程目录设成运行依赖。

`rollouts-import` 接受 JSON list，每项为：

```json
{
  "source": "/absolute/path/to/old/raw.traj.json",
  "expected_sha256": "OPTIONAL_SOURCE_SHA256",
  "task": {"...": "完整上述 task 对象"},
  "run_id": "SOURCE_RUN",
  "attempt_id": "SOURCE_ATTEMPT",
  "purposes": ["training", "baseline"],
  "model_fingerprint": "REGISTERED_CHECKPOINT_FINGERPRINT"
}
```

导入器不修改原始轨迹。原始文件的用途或模型指纹冲突会拒绝；缺失评测时保持 resolved=null，不根据提交状态猜测。

`evaluations-import` 接受 JSON list，每项为：

```json
{
  "evaluation_id": "IMPORTED_EVAL",
  "report": "/absolute/path/to/source/report.json",
  "report_sha256": "SOURCE_REPORT_SHA256",
  "bindings": {
    "REGISTERED_ROLLOUT_ID": {
      "evaluated_patch": "/absolute/path/to/the/patch.diff/actually/graded",
      "outcome_path": ["outcomes", "INSTANCE_ID"]
    }
  }
}
```

outcome_path 是到源报告内结果对象的 key list；官方单条 report 常为 `["INSTANCE_ID"]`。结果含 boolean resolved，completed 缺省时仅允许明确的 boolean resolved 作为已分级证据。
必须提供当时实际评测 patch 文件，SHA 必须与注册 rollout patch 一致。

评估器预测文件不预设模型架构，是 JSON list：

```json
[
  {"sample_id": "ROLLOUT:STEP:ANCHOR:OCCURRENCE", "target": "remaining_steps", "prediction": 12.3}
]
```

target 允许 remaining_steps / remaining_reasoning_tokens。真实值从冻结 enrichment 引用读取，不接受预测文件自己提供的 GT。按 anchor 分别评分/画图，缺失成功标签单独计数。
