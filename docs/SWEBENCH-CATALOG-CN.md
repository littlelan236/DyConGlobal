# SWE-bench catalog 分类与 NAS 同步

正式清单为项目 `catalog/manifest.json`。SWE-bench 使用三个 benchmark 名称：

| 类别 | catalog benchmark | 定义 |
|---|---|---|
| verified | `swebench_verified` | 官方 Verified 的 test 子集 |
| dev | `swebench_dev` | 官方 SWE-bench dev |
| nonverified | `swebench_nonverified` | 官方 SWE-bench test 排除 Verified；不含 dev |

三类均由已有 SWE-bench backend 执行。历史名称仍保留 backend 兼容能力，以便加载冻结 run。
`partition` 继续表示项目内部 train/val/test/excluded 用途，不表示本表官方分类。

## 查看与选择

```bash
.venvs/runner/bin/dycon swebench catalog-list
.venvs/runner/bin/dycon swebench catalog-list --group verified --output temp/my-selection/verified.json
.venvs/runner/bin/dycon run create MY_RUN --operation rollout --tasks-file temp/my-selection/verified.json
```

`catalog-list` 输出分组数量、选中数量和 ID 哈希；不会打印全部任务。`--output` 写入兼容正式
`run create --tasks-file` 的 `{tasks: [instance_id, ...]}`，拒绝覆盖已有文件。创建 run 时仍需按
项目配置约定配置 `benchmark.backend: swebench`、agent、模型、环境等。该任务文件记录导出时
catalog 身份，创建 run 将通过现有 run 接口绑定当时正式 catalog，不是历史快照恢复接口。

## 同步现有 NAS 镜像

```bash
.venvs/runner/bin/dycon swebench catalog-sync \
  --source PINNED_OFFICIAL_SOURCE.json \
  --image-plan /VM/wangtianyi/DyConGlobal/VERIFIED_STORE/dataset/plan.json \
  --image-plan /VM/wangtianyi/DyConGlobal/DEV_STORE/dataset/plan.json \
  --output catalog/changes/UNIQUE_PREVIEW_ID
```

默认预览。确认指定来源和范围后，使用新的 output 目录并添加 `--apply` 进行原子同步。
来源文件 schema 1：`sources` 提供 verified/dev/test 的 dataset/revision/split/ids 等溯源；
`rows` 提供完整 verified/dev 官方任务正文。支持的固定版本与 ID 集哈希见
`catalog/swebench_sync.py`。两个 image-plan 为 `swebench images-pull` 使用的固定完整计划，
包含既有镜像引用和缺失 ID。同步只接受对应固定官方数据与镜像名。

同步会对旧 SWE 任务按官方 ID 重新分组、重建 catalog 键；补充 NAS 已有且未登记的 verified/dev。
旧任务正文、revision、partition、环境 overrides 和镜像 pin 保持原样；新任务使用官方完整正文、
已有可验证 NAS 引用和项目 train partition。未知官方归属或多个旧 revision 的重复 ID 会拒绝，
不会任选一个。新任务可选不代表其容器环境已经通过 gold 验收。

NAS `images/ID.json` 必须存在，镜像记录哈希与完整 OCI 图引用大小检查必须通过，才会登记。
下载失败、只有 metadata 或 partial blob 的任务留在报告 `missing_official_ids`，不会进入可选清单。
同步既不发起下载，也不自动改用其他 repair store；已有任务保留原来显式绑定的镜像引用。

catalog 通过原有事务锁更新，只替换 SWE 子集，保留其他 benchmark；提交前检查 catalog 未被
并发修改。`--output` 拒绝复用旧目录。应用产物包含原 catalog 精确备份、更新后 catalog、官方来源、
旧新任务键映射和带前后哈希的报告。重复同步相同输入不增加 catalog revision。

已有 `runs/*/plan.json`、`selection.json`、`config.yaml`、`resolved-config.json` 不被同步修改；
旧 run 继续使用自身冻结任务与旧键。无需移动或复制 NAS 大镜像层。
