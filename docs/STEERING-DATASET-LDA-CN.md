# Consensus Steering Dataset 与 LDA / t-SNE / UMAP

核对日期：2026-11-01。离线模块：steering/；正式入口：core.cli / dycon。

## 数据契约

- actual-consensus-action-v1：候选 exact canonical action 与历史实际 winner action 相同为 positive，否则有效 action 为 negative。标签不是正确性、resolved 或 remaining-work。
- 遍历全部候选，包含 estimator filtered。selected/retained/score/loop exclusion 仅作为来源字段。
- 每个 decision/branch 只选择一个明确绑定的 generated think_close。默认不选 prompt/history、newline、startup/smoke；多 close 歧义计 excluded，不取第一条或平均。
- 从来源 run 冻结配置解析 NAS，只消费登记的 ACKed/finalized publication。核对 trajectory SHA、完整 decision、历史策略、winner、request/branch/token 身份和 receipt。
- 真正缺失计 excluded；SHA、winner/canonical 矛盾、身份错绑和不兼容特征空间属于硬错误，不降级为 retained-only。
- native site ID 来自 shape/dtype 和 model inventory，区分 decoder_output/final_norm。默认拒绝 model/tokenizer/intervention/site/dtype 差异。

## 环境与隔离目录

使用独立 analysis Python，新增科学依赖见 locks/steering-analysis.txt；实际版本冻结到 analysis.json。不安装到共享 serving 环境。
本次工作目录：

~~~text
/data3/wangtianyi/DyConGlobal/temp/steering-isolation-20261101/worktree
~~~

其中 .venvs/analysis/bin/dycon 安装到隔离环境。源码仍留在独立 worktree，未合并回主工程。
离线 CLI 从 --config 读取项目位置和 artifact roots；capture/策略/模型身份取各来源 run 冻结快照，不用当前根配置代替历史配置。

## 最小命令

以下均为新建示例 ID；已有不可变 ID 拒绝覆盖。ROOT 为基础目录，自动追加 steering/category/ID。

~~~bash
cd /data3/wangtianyi/DyConGlobal/temp/steering-isolation-20261101/worktree
CLI=.venvs/analysis/bin/dycon
CONFIG=/home/wangtianyi/DyConGlobal/config.yaml
ROOT=/data3/wangtianyi/DyConGlobal

$CLI --config "$CONFIG" steering-dataset build NEW_DATASET_ID --source-run RUN_ID --all-submitted --site 3 --storage-root "$ROOT"
$CLI --config "$CONFIG" steering-dataset check NEW_DATASET_ID --dataset-root "$ROOT"
$CLI --config "$CONFIG" steering-dataset cache NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --site 3
$CLI --config "$CONFIG" steering-dataset cache NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --site 3 --check-data
$CLI --config "$CONFIG" steering-visualize NEW_ANALYSIS_ID --dataset-id NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --storage-root "$ROOT" --site 3 --fit-scope all --memory-budget-gib 32 --max-display-points 20000 --seed 42 --facet retained --format png --format svg --format pdf
$CLI --config "$CONFIG" artifact verify NEW_DATASET_ID --scope steering-dataset --storage-root "$ROOT"
$CLI --config "$CONFIG" artifact verify NEW_ANALYSIS_ID --scope steering-analysis --storage-root "$ROOT"
~~~

--source-run 可重复。三种选择三选一：单 run 重复 --rollout-id、--rollout-list bindings.json、或显式 --all-submitted。
跨 run 的 binding JSON 每条包含 source_run_id、rollout_id，可选 trajectory_sha256。重复 binding、含糊裸 ID、非法 ID、不同身份同名副本拒绝。
显式选择保留同 task 不同 repetition/attempt，不自动挑最新，不继承 estimator 的 resolved-only 规则。
不传 build --site 时冻结兼容 inventory；--site 可重复，表示 native ID 而非数组位置。cache 可只选冻结 sites 的子集。

## 路径与校验

paths.artifact_roots 支持 steering_dataset、steering_cache、steering_analysis；未设置使用 project root。
显式 --storage-root / --dataset-root / --cache-root 优先；读取还支持 artifact_scan_roots。未登记磁盘可以显式传基础根目录。
不放宽 core.io.owned_path，不将任意外盘路径交给 difficulty dataset resolver。

- steering/datasets/ID：dataset/selection/rollouts JSON，decisions/samples/excluded JSONL，source-evidence、dependencies、summary、receipt。仅冻结元数据和 tensor locator，不复制整份 NAS。
- steering/cache/HASH：cache.json、site-ID.f32.npy、receipt.json。物理 cache root 不进入 hash；缓存缺失不代表 dataset 损坏。
- steering/outputs/ID：analysis/selection/summary、完整 points.csv、points-metadata.jsonl、transform/*.npz、projections/*.npy、plots/*、report.md、index.html、receipt。

dataset check/verify 默认核对 metadata 和绑定来源；--with-tensors 才读大张量。
cache 构建按 shard 分组，校验 shard/member SHA、shape/dtype/finite，按冻结样本次序写 float32 memmap。
缓存有空间预算、排他锁、同盘临时目录和原子发布；半成品、损坏和跨盘冲突拒绝复用，不补零。

## LDA 和分组

使用 sklearn LinearDiscriminantAnalysis(solver=svd, n_components=1)，等先验，不附加 StandardScaler。
各 site 单独拟合，符号固定为 positive 拟合集均值更大，投影始终为 [N,1]；strip 纵轴只有类别和展示 jitter。
SVD 前估算工作内存；--memory-budget-gib 不足报错，不静默截断样本。缺类/退化记录 unavailable，全部 site 不可用时 CLI 返回 2。

--fit-scope all 是探索性监督可视化，不能把同批分离当作泛化或实际 steering 效果证据。
--fit-scope grouped --split groups.json：每个 benchmark/revision/task 只能进入一个 split，跨 run 同 task 与同 decision 分支不泄漏。
只在 train 拟合，再向全部样本应用冻结 transform。分组须完整覆盖选中任务，重复、冲突、遗漏均拒绝。

~~~json
{"groups":[{"task_key":"TASK_KEY_A","split":"train"},{"task_key":"TASK_KEY_B","split":"validation"}]}
~~~

group 也可使用完整 benchmark/revision/instance_id。分析保存拟合样本清单、分组和来源文件 SHA。
points.csv 保存全部坐标与身份；--max-display-points 只限制 strip 显示，seed、ordinal 清单和 jitter 都冻结，统计和直方图仍用全量。

## t-SNE / UMAP 二维可视化

使用同一 dataset/cache 和统一 `dycon steering-visualize` 入口，通过 `--method tsne|umap`
选择方法；未指定 method 的新拟合仍使用 LDA。两种方法均为无监督全 cohort 探索性嵌入，
标签只用于颜色和计数，不传给拟合器。当前仅支持 `--fit-scope all`，拒绝 grouped/split，
避免把联合嵌入当作独立验证。单一标签不阻止拟合；少于 4 条或完全相同特征记录 unavailable。

~~~bash
$CLI --config "$CONFIG" steering-visualize NEW_TSNE_ID --method tsne --dataset-id NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --storage-root "$ROOT" --site 3 --pca-components 50 --perplexity 30 --max-iter 1000 --threads 4 --memory-budget-gib 32 --seed 42 --max-display-points 20000 --facet retained --format png --format svg --format pdf
$CLI --config "$CONFIG" steering-visualize NEW_UMAP_ID --method umap --dataset-id NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --storage-root "$ROOT" --site 3 --pca-components 50 --n-neighbors 15 --min-dist 0.1 --threads 4 --memory-budget-gib 32 --seed 42 --max-display-points 20000 --facet retained --format png --format svg --format pdf
~~~

- 默认先做不白化、无 StandardScaler 的 randomized PCA，维度为 `min(50, width, N-1)`；
  `--pca-components 0` 禁用。全部样本进入 PCA 和嵌入，不根据标签采样。
- t-SNE：sklearn Barnes-Hut，二维、Euclidean、learning_rate=auto、early_exaggeration=12、
  angle=0.5；默认 perplexity=30（必须小于 N）、max_iter=1000（至少 250）。PCA 初始化，
  输入只有一维时改用随机初始化。
- UMAP：umap-learn，二维、Euclidean、spectral 初始化、spread=1、low_memory=True；
  默认 n_neighbors=15（必须满足 2≤值<N）、min_dist=0.1（范围 [0,1]）。
  为保证固定 seed 的可复现性，UMAP 优化使用单线程；`--threads` 控制 BLAS/OpenMP，默认 4。
- 内存估算覆盖输入、PCA、稀疏邻居图和工作区；预算不足拒绝拟合。估算不是实际峰值承诺。
- `points.csv` 保存每个 site 全部 N 行、原身份和 TSNE1/TSNE2 或 UMAP1/UMAP2；
  `projections/site-ID.npy` 为 `[N,2]`。显示上限仅限制散点，不改变拟合或导出。
- `analysis.json` 冻结参数、PCA 信息、版本、seed、线程限制和诊断。
  `transform/*.npz` 对这两种方法仅保存无 pickle 的审计描述和可选 PCA 数组，
  不导出可用于新样本的非线性模型；不能像 LDA 一样调用线性 transform。
- 每种方法和 site 的坐标系独立，方向、点间距离和可见分离不能直接跨方法/seed 比较，
  也不能解释为泛化、action 正确性或实际 steering 效果。

设计参考旧仓库 `littlelan236/TACTrefractored` 的 `src/tact/research/plots.py` 与
`docs/POINT-CLOUDS.md`（commit `dea0e39a820f94586f9ed6a9c8e77f2c589ca874`）：
复用完整点身份、PNG/PDF/SVG/HTML、冻结坐标与校验的设计。该版本没有 t-SNE/UMAP
实现，因此算法通过 sklearn/umap-learn 新增，未复制旧项目的三类轴定义。

## 冻结重绘

~~~bash
$CLI --config "$CONFIG" steering-visualize NEW_REDRAW_ID --redraw-from OLD_ANALYSIS_ID --analysis-root "$ROOT" --storage-root "$ROOT" --site 3 --seed 43 --max-display-points 5000 --format png
~~~

先校验旧 bundle，只读冻结 labels、transform 和坐标，不读 source run/NAS/cache，不调用 fit。
发布到新 analysis ID，不改变原 dataset receipt。投影或 transform 篡改会被拒绝。transform 为无 pickle NPZ，不是 steering payload。
LDA/t-SNE/UMAP 均支持；method 默认继承源 bundle。显式不同 method 或重新传入嵌入拟合参数将被拒绝。

## 本次真实验收

用户指定 run：verified80-consensus32-resolved106plus331-gpu03-tp4-w4-s64-20261027 的全部 submitted rollout。
冻结 68 条轨迹、2,933 个 decision、93,856 个候选；site 3 有 74,704 个有效样本（29,991 positive / 44,713 negative）。
58 条轨迹有 included 样本，另外 10 条保留 excluded 覆盖。19,152 个 excluded 分别为：19,136 缺少匹配的已发布/finalized session、14 缺有效 tool action、2 close 未生成。
publication 未匹配不等价于 NAS 底层文件不存在；不得据此清理或修复原 capture。
保留 27,656 个 nonselected positive、14,561 个 filtered positive、19,865 个 retained negative；226 个 decision 的全部有效 action 同意 winner。
真实验收仅使用该单 run 和 native site 3；多 run、动态 site、固定/熵策略与不同 representative 的覆盖由回归验证。

Dataset：verified80-consensus32-dataset-20261101；analysis：verified80-consensus32-lda-site3-20261101；重绘：verified80-consensus32-lda-site3-redraw-20261101。
三类产物根均为 /data3/wangtianyi/DyConGlobal。正式 artifact verify 核对 7,482 个冻结来源文件；cache --check-data 和真实冻结重绘均通过。
没有重新 rollout、启动 GPU、训练 estimator、修改根配置或原 capture。合并主工程前需审查 core/cli.py、hidden/artifacts.py、pyproject.toml 与其他 agent 的并发修改。

## 2026-11-01 t-SNE / UMAP 拓展验收

复用上述 dataset/cache，site 3 全部 74,704 行均参与拟合和导出。默认 PCA 50 维
保留约 89.01% 方差；seed=42、绘图显示上限 20,000、retained 分面、PNG/SVG/PDF/HTML。
t-SNE 使用 perplexity=30、max_iter=1000；UMAP 使用 n_neighbors=15、min_dist=0.1、n_epochs=200。

| 分析 ID | 正式 artifact verify | 实测拟合及导出耗时 |
|---|---|---|
| verified80-consensus32-tsne-site3-20261101 | verified | 2:55.06 |
| verified80-consensus32-umap-site3-20261101 | verified | 1:32.28 |
| verified80-consensus32-tsne-site3-redraw-20261101 | verified | 冻结坐标重绘 |
| verified80-consensus32-umap-site3-redraw-20261101 | verified | 冻结坐标重绘 |

两项重绘采用 seed=99、显示 5,000 点，projection 和 descriptor 均与原分析逐字节一致。
旧 LDA artifact、原 cache 校验、97 项 steering/共识/分支/capture 相关测试、Ruff 和
git diff --check 均通过。测试还覆盖无来源重绘、固定 seed 重复拟合、单标签、退化输入、
非法参数和内存预算拒绝；未运行整个工程测试套件。
临时日志和拓展增量补丁位于 worktree 内 `temp/steering-manifold-validation/`。

