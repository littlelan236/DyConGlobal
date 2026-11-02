# Consensus 分支 Steering Dataset 与 LDA 可视化：调研及实施计划

日期：2026-11-01（Asia/Shanghai）  
远程工程：/home/wangtianyi/DyConGlobal（10.249.187.9）  
调研版本：main@f12b8e06bc482826cb7e20261f552c275c6077e8  
状态：设计计划；新增目录、schema、CLI 均为拟实现内容，本次只交付文档。

## 1. 范围与确认事项

1. difficulty/datasets/ 和 difficulty/training/ 是 estimator 数据与拟合链路，不是 steering 数据链路。新功能独立建设，只借鉴其不可变数据集、来源校验、缓存和磁盘组织方式。
2. 从用户指定的数个 consensus run、指定的一批 canonical rollout 轨迹，读取已经发布的 capture，构建二分类 branch hidden 数据集。
3. 用户已经确认：对每个 decision 的**全部可验证候选分支**标注，包含被 estimator 过滤者；与实际最终获胜 action 相同者为 positive，其余有效 action 为 negative。
4. 当前工程未保留旧 latent 点云模块，首期仅新增 LDA，接入正式 dycon CLI。
5. Dataset、可重建 feature cache、可视化产物均支持指定磁盘根目录。
6. Payload 提取、方向发布、实际注入、在线控制、estimator 训练不在本期范围。可视化投影参数不作为 steering payload。

首期设计默认：每个 branch 使用一个与 decision 绑定的 think_close hidden，各 site 单独分析；不把 reasoning newline 的所有 token 展开为样本，也不平均整段 reasoning。此粒度冻结到 manifest，扩展须新建 schema/数据集。

## 2. 当前工程调研

### 2.1 可复用基础

| 位置 | 已核实职责 | 本期复用边界 |
|---|---|---|
| rollout/consensus.py:32,240 | 原共识选择及证据验证，含固定比例与动态策略 | 验证原冻结策略，不重新选择 winner |
| core/action_identity.py:23 | exact tool action canonicalization | 分支与 winner 使用同一身份口径 |
| runtime/serving/capture.py:1238-1282 | finalize 全候选 branch-index、coverage、receipt | 消费已有发布物，不改 capture producer |
| hidden/artifacts.py:126 | read_branch_index 校验 receipt 和显式分类修订 | 读取全候选并保存实际修订身份 |
| hidden/artifacts.py:13,33 | loose NPZ / tar member 读取和 SHA 验证 | 复用张量读取，不另写 NAS 解包器 |
| action/analyzers/branches.py:30-159 | full/embedded decision 对齐、候选 action 解析、实际 winner 校验 | 借鉴/抽取中立 helper，不依赖 action 统计输出 |
| difficulty/datasets/selection.py；action/datasets/trajectories.py | 指定轨迹、来源 binding、SHA 冻结 | 仅借鉴纯选择，不继承 resolved-only/remaining-work 规则 |
| difficulty/datasets/submission.py:190-289 | shard 分组读取、分 site memmap、锁/空间检查/原子发布 | 借鉴缓存组织，不调用 estimator 样本或目标计算 |
| core/io.py:62-120,152 | 多磁盘 roots、run resolver、difficulty cache 路径 | 复用通用 roots/run resolver，新增 steering typed resolver |

原始 capture 根必须从每个来源 run 的冻结配置解析：

~~~text
<frozen paths.nas_root>/<frozen capture.export.root>/<RUN_ID>/
  sessions/<SESSION_ID>/
    finalization.json
    parallel/
      branch-receipt.json
      branch-index.json
      branch-coverage.json
      decisions/<DECISION_ID>.json
  index/...
  <tensor files / tar shards>
~~~

常见物理目录为 /VM/wangtianyi/DyConGlobal/hidden/online/<RUN_ID>/。本地 runs/<RUN_ID>/hidden/sessions 和 hidden/index 可以为空，不能据此推断 capture 丢失。Run 自身可通过项目别名放在 /data3/wangtianyi/DyConGlobal/runs/，应走现有 resolver。不得扫描整个 NAS，也不得用当前根配置代替来源快照。

实物抽样：parallel-consensus32-dynamic-w2-s64-lowreserve-t8h-20261025 的 canonical rollout r-ec6db837637430b0bf33afcbc590af51 有 NAS finalized session；branch-index 含 608 条 decision/branch 记录。抽样 decision 有 32 个 candidates、32 份 response、32 个 close_sample_id，而 retained 只有 16 个。这证明仅消费 retained metadata 会漏掉用户要求的候选；它不是全量可用性或类别数统计。

旧 consensus 可能仅为 retained 保存 compact action，filtered 标记 identity_status=not_evaluated；full decision 仍可有 raw response。必须读取 receipt 绑定的 raw response 补齐 action，不能将 estimator_filtered 或未解析 action 直接当 negative。

### 2.2 存储现状的准确边界

当前 difficulty/datasets/<ID>/ 元数据主要保存在项目内；可配置外盘的是 paths.artifact_roots.difficulty_cache 对应的 difficulty/cache/<HASH>/。core/io.artifact_dir 尚无 steering module，owned_path 不默认接受任意外部 dataset 路径。

新 dataset/analysis 的外盘存放是本期需要显式增加的功能，不能宣称现有 difficulty dataset 已原生支持任意 output-dir，也不能为此全局放宽 owned_path。

## 3. 旧工程可视化核查与取舍

用户提供旧仓库：https://github.com/littlelan236/TACTrefractored 。本次读取固定提交 dea0e39a820f94586f9ed6a9c8e77f2c589ca874：

- src/tact/research/axes.py:10-12,64-110：tact_mean_difference、lda_contrast、fisher；标签是 OVERTHINKING / OVERACTING / CALIBRATED。
- src/tact/research/axes.py:209：保存训练集拟合后的投影和逐样本 label/instance/step。
- src/tact/research/plots.py:140：点云、按 site 的 AUC、方法比较、PNG/PDF/SVG、CSV/NPZ、HTML/报告和完整性校验。
- tests/test_plots.py：投影数值、坐标篡改、模型不兼容、历史 bundle 无投影时拒绝等回归测试。

对当前 DyCon 的 core/application/runtime/steering/action/difficulty/docs/tests 检索这些方法和关键符号，未找到对应模块。现有 difficulty 的预测/真值散点不是此类正负 hidden 点云；EasySteer vendor 的 vector extraction/PCA 也不是这套工作流。

**决策：新增二分类 LDA；本期不整套迁移旧三分类 axes/payload 流程，不顺带增加均值差、Fisher、t-SNE、UMAP。** 复用“冻结投影、重画不拟合、导出坐标、来源/文件校验”的设计，不复制旧标签和双轴公式。

旧 repo 的 vendor checkout 遇到 Windows 长路径/LFS 下载问题，但上述研究文本源码已经从固定 Git 对象/已取得文件读取。调研不依赖旧 vendor 运行时，也未运行旧模型。

## 4. 标签与样本契约

### 4.1 actual-consensus-action-v1

对 decision d，读取原历史最终获胜 action a*(d)；对候选 branch b，使用现有 canonical_action 得到 a(b)：

~~~text
positive / y=1 : a(b) == a*(d)
negative / y=0 : a(b) != a*(d)，且 a(b) 是可验证的有效 action
excluded      : action/完整响应/hidden 绑定等证据缺失
~~~

- selected_branch_id 仅是实际执行代表，不是 positive 集合。未选中但 action 相同的分支也必须为 positive。
- 遍历全部候选，不要求 retained=true，不以 estimator score 的优劣或是否有限为标签依据；score、retained、loop-exclusion 仅保留为来源信息。
- 固定比例和 parallel_enthophy 动态共识使用同一“实际最终获胜 action”定义。不能重新按全候选票数选 winner，也不能用今天配置覆盖历史选择。
- exact action 使用现有工具名/参数对象/tool-call 顺序规则；命令字符串保留，不自行按 shell 语义合并，也不只比较 action 类别。
- 标签只表达共识一致性，不是正确/错误、resolved 或 remaining-work。数据集不要求 harness resolved，不使用 estimator 的成功筛选。
- 全体同意、仅有 positive 的 decision 保留并单列统计，不能伪造 negative；LDA 的整体拟合集才要求两类齐全。
- SHA 错误、winner/canonical 矛盾、身份错绑属于硬错误；真实缺失才记 excluded，不能将完整性错误静默计作缺失。

### 4.2 一条 branch-close 对应一个 metadata 样本

样本保存所选 sites；每个 site cache 的同一行对应同一 metadata 样本。复合身份至少为：

~~~text
source_run_id + rollout_id + session_id + decision_id + branch_id + sample_id
~~~

字段至少包含：task/revision/model/tokenizer、trajectory SHA、agent_step/message_index、native_request_id、parent_prefix_sha256、close_sample_id、token/phase/absolute_position、selected/retained、action/hash、consensus action/hash、label/y、tensor locator/member SHA、site inventory/schema、capture_stage/dtype、来源证据 SHA。

以 candidate.close_sample_id 绑定 branch-index 对应 close_event，核对 request/decision/branch/token 坐标。旧记录缺少 close_sample_id 时，只有唯一且身份完全匹配的 close event 可用，并记录选择依据；多 close 有歧义时排除，不取列表第一个或擅自平均。默认不纳入 prompt/history close、newline 和 startup/smoke session。

不同 branch 即使 action 相同也保留不同 hidden。重复输入 binding 拒绝；同一底层 capture 的不同路径别名只计一次并记 alias。不能仅凭 hidden bytes 相同去合并独立采样。

## 5. 提取与冻结流程

1. 冻结数个 source runs 与 binding JSON。每条明确 source_run_id/rollout_id，可提供 expected trajectory SHA。单 run 支持重复 rollout-id；多 run 禁止有歧义的裸 ID。首期要求显式轨迹列表，不默认扩大为全 run。
2. 用 RunEvidence、既有 run resolver、来源冻结配置解析 canonical trajectory。同 task 的 repetition/attempt 独立保留，不自动挑最新；排除 quarantined 和未选择的轨迹。
3. 在 NAS 找到唯一绑定 rollout/trajectory SHA/model 的 finalized session，只消费 ACK 后的 publication。Run 仍执行时只选择已完成且 canonical 身份冻结的 session，不读在途 spool。
4. 验证 branch receipt、完整候选、原 consensus policy、winner→canonical assistant response；从 full response 解析 filtered action，禁止降级成只保留 retained。
5. 对有效 action 标注，再关联 close hidden。为全部候选记录 included/excluded 和原因，不能丢掉负样本的缺失覆盖信息。
6. 保存 selection、rollouts、decisions、精简来源证据快照、samples/excluded/summary/receipt。不得改 source run 的 trajectory、capture、labels 或配置。
7. 独立 cache 阶段按 shard 批量读数值、验证 SHA/shape/有限值，按 site 写 float32 memmap。不重请求模型，不执行 trajectory 工具。

同一数据空间要求 model fingerprint、tokenizer、hidden/site 语义兼容；维度相同不代表同空间。默认拒绝不同 checkpoint、未声明的 site/dtype 差异或不同 steering intervention 来源混合。允许的来源差异必须显式声明、冻结并分组展示。

现有 difficulty.public_features 硬编码 [32,4096]，且最后 public site 来自 native layer_32 而非 decoder layer_31。新模块必须从 model/capture inventory 记录 site ID、decoder_output/final_norm 和维度，不复制这个硬编码或按数组排序猜 site。

## 6. 目录与存储设计

### 6.1 源码

~~~text
steering/                           # 新项目级离线研究模块
  __init__.py
  README.md
  artifacts.py                      # schema/receipt/verify
  datasets/
    __init__.py
    selection.py                    # 多 run / 轨迹 binding
    build.py                        # finalized branch → dataset
    labels.py                       # 标签纯函数
    features.py                     # 动态 site 与张量读取/校验
    cache.py                        # 分 shard 读取、分 site memmap
    paths.py                        # steering typed artifact roots
  visualization/
    __init__.py
    lda.py                          # 只做分析投影
    render.py                       # 分布/strip 图、坐标、HTML
    service.py                      # CLI 生命周期/冻结/receipt
runtime/steering/easysteer/          # 继续仅负责现有 serving adapter
core/cli.py                         # 正式入口注册
tests/test_steering_dataset*.py
tests/test_steering_lda*.py
~~~

不在 difficulty/training 添加 steering，不改 vendor 建另一套 launcher。如选择/feature 解码确需共享，只将中立 helper 小范围抽至 core/hidden，不从 estimator 模型训练模块导入。

### 6.2 三类产物分离

~~~text
<dataset_root>/steering/datasets/<DATASET_ID>/
  dataset.json                      # schema/标签规则/sites/来源身份
  selection.json / rollouts.json
  decisions.jsonl                   # 历史 winner、候选覆盖
  samples.jsonl / excluded.jsonl
  source-evidence/<RUN_ID>/...       # 选定证据快照和父 receipt 身份
  summary.json / receipt.json

<cache_root>/steering/cache/<CACHE_HASH>/
  cache.json
  site-<SITE_ID>.f32.npy             # [N,H]，同一冻结样本顺序

<analysis_root>/steering/outputs/<ANALYSIS_ID>/
  analysis.json / selection.json
  transform/site-<SITE_ID>.npz       # 可视化参数，无 pickle
  projections/site-<SITE_ID>.npy     # [N,1]
  points.csv                        # identity、label、LD1、分组/展示标志
  summary.json / report.md
  plots/site-<SITE_ID>-distribution.png|svg|pdf
  plots/site-<SITE_ID>-strip.png|svg|pdf
  index.html / receipt.json
~~~

Dataset 默认是 immutable metadata + 已发布 tensor locator，不再复制整份 NAS。Cache 是可删后重建的数值副本，不存在不代表 dataset 损坏。Analysis 引用 dataset receipt，新增图不改变 dataset receipt。完全脱离 NAS 的归档 export 不隐含在本期。

### 6.3 指定保存磁盘

拟沿用 paths.artifact_roots 的 base root 语义，增加：

~~~yaml
paths:
  artifact_roots:
    steering_dataset: /data3/wangtianyi/DyConGlobal
    steering_cache: /data3/wangtianyi/DyConGlobal
    steering_analysis: /data3/wangtianyi/DyConGlobal
~~~

这是拟议配置，本次不改根 config。未配置时使用项目根；命令的 storage-root/cache-root 优先，实际位置冻结到 artifact manifest。三个 root 都是基目录，自动追加 steering/category/ID，metadata、cache、图表允许分盘。

按 ID 读取使用配置 roots/scan_roots；未登记外盘显式传 dataset-root。新 typed resolver 只认可显式/配置的绝对根和固定 category/ID，核对 manifest/receipt；同 ID 不同身份的副本冲突报错。不能直接把任意外盘路径交给目前只识别项目内 difficulty 的 dataset_directory，也不全局放宽 owned_path。

## 7. Cache 契约

- Hash 包含 dataset receipt、样本顺序、site 及语义、hidden 来源、dtype、feature-schema；物理磁盘根不改变数据身份。
- 预计算 N×H×sites×4 和临时/保留空间；按 site memmap、按 shard 分组读，避免全量 raw hidden 入内存。
- 排他锁创建，目标盘内临时目录写入，finite/尺寸检查、flush 和 SHA 完成后原子 rename；半成品无有效 cache.json，不复用。
- cache.json 记录 N/H/sites、样本顺序和每文件 SHA。损坏或多盘冲突直接拒绝，不补零、不改标签。
- Dataset check/verify 默认核对元数据/来源；with-tensors 或 cache/check-data 阶段才读数值。缓存损坏与源 dataset 损坏分开报告。
- 借鉴 difficulty 的缓存机制，不继承固定 32 sites、4096 width 或 remaining-work target。

## 8. LDA 可视化

### 8.1 二分类只能产生一个判别维度

标准二分类 LDA 最多只有一个判别轴 LD1。旧 OT/OA 双轴基于三分类及两个对比方向，不能照搬为二分类 LDA 2D，也不能复制同轴冒充二维。

每个 site 独立拟合，坐标保存为 [N,1]，首期提供：

- 两类 LD1 共用分箱的直方图/经验分布，标注样本数与 class prior。
- LD1 strip plot；纵轴是类别/固定 seed 的展示 jitter，明确不是第二 latent 维度。
- 可按 run/retained 分面检查来源差异；不同 site 独立坐标不视作同一空间。

本期不为了二维额外添加 PCA。points.csv 保存完整样本投影和身份；需要限制展示点数时必须显式记录上限/seed/选中清单，统计仍用全量坐标，不静默抽样。

### 8.2 拟合与解释

- 默认 fit_scope=all，明确为探索性监督可视化；同批数据上的分离程度不能当作泛化、action 正确性或 steering 效果证明。
- 如提供 train/validation 分组文件，按 benchmark/revision/task 分组；跨 run 同 task 及同 decision 的分支不得泄漏。只在 train 拟合并向 validation 应用，不在 validation refit。分组/样本清单冻结在 analysis，不复用 estimator training attempt。
- 拟采用 sklearn LinearDiscriminantAnalysis(solver='svd', n_components=1)，两类等 prior，不默认附加 StandardScaler；solver/tol/prior/预处理/版本均冻结。以拟合集均值确定方向符号，使 positive 均值更大。
- 缺类、退化/无有效判别方向明确 unavailable/失败，不伪造坐标。一个 site 一个 site 处理；SVD 分配前估计工作内存，超预算报错或要求显式拟合抽样上限。memmap 不等于 LDA 拟合恒定内存，不静默截断。
- transform 参数、顺序、投影和 SHA 全部保存，重画只读冻结坐标，不从来源 run 重取标签或重拟合。
- 不同 branch 默认等权，保留同 action 的多份 hidden；报告每 decision/task/run 的数量和类别比例，不将相关分支宣称为独立泛化样本。若以后增加 decision-balanced 方法须另存方法/权重。

## 9. 拟议 CLI（尚未实现）

使用 analysis 环境的 dycon，离线读已发布 capture，不启 GPU serving。以下 ID/路径均为占位，不是已创建产物：

~~~bash
# bindings.json 的每项含 source_run_id 和 rollout_id
dycon steering-dataset build NEW_DATASET_ID --source-run RUN_A --source-run RUN_B --rollout-list bindings.json --storage-root /data3/wangtianyi/DyConGlobal

dycon steering-dataset check NEW_DATASET_ID --dataset-root /data3/wangtianyi/DyConGlobal

dycon steering-dataset cache NEW_DATASET_ID --dataset-root /data3/wangtianyi/DyConGlobal --cache-root /data4/wangtianyi/DyConGlobal

dycon steering-visualize NEW_ANALYSIS_ID --dataset-id NEW_DATASET_ID --dataset-root /data3/wangtianyi/DyConGlobal --cache-root /data4/wangtianyi/DyConGlobal --storage-root /data3/wangtianyi/DyConGlobal --method lda --site 3 --fit-scope all

dycon artifact verify NEW_DATASET_ID --scope steering-dataset --storage-root /data3/wangtianyi/DyConGlobal
dycon artifact verify NEW_ANALYSIS_ID --scope steering-analysis --storage-root /data3/wangtianyi/DyConGlobal
~~~

site 可重复指定，代表显式 site ID 而不是数组位置。Dataset 未指定 sites 时冻结选择范围中兼容的 inventory；可视化明确选择 sites。外盘必须贯通 build/check/cache/plot/verify，不能只有创建支持外盘。最终参数命名随现有主 CLI 对齐，不另建生产 shell launcher。

## 10. 实施顺序与验收

| 阶段 | 工作内容 | 必须通过的验收 |
|---|---|---|
| P1 契约与路径 | 新 steering package、标签纯函数、typed roots、schema/receipt、CLI 骨架 | 多 run/重复 binding/非法路径/ID 冲突/外盘解析 |
| P2 Dataset | finalized session/decision/branch join、全候选解析、winner/canonical 校验、样本/excluded 冻结 | retained 外同 action 为 positive；retained 内异 action 为 negative；缺失不当 negative；startup 排除 |
| P3 Hidden/cache | 动态 site、复用 loose/tar reader、SHA/finite、memmap/锁/空间/原子发布 | 小样本数值与源一致、顺序可复核、跨盘命中、半成品/损坏拒绝 |
| P4 LDA/绘图 CLI | LD1、探索/显式分组模式、坐标/分布/strip/HTML、analysis receipt | 只一判别维、缺类/退化/内存预算明确、重画不 refit、篡改拒绝 |
| P5 真实离线 smoke/文档 | 至少两个可用 consensus run 各指定一条轨迹，正式 CLI 构建/缓存/绘图/verify | 全部标签和 excluded 可追溯、产物落到指定盘、source run/estimator 产物不变 |

P2/P3/P5 使用现有 capture，不为验收重新 rollout。选择含两类、retained 外可解析 action、非 selected positive 的真实 decision；单类 dataset 允许成立，但 LDA 明确不可拟合，不伪造对比组。

关键回归覆盖：

1. 全候选/retained、selected/positive 差别；固定/动态共识与不同 representative policy。
2. 全体相同 action、无 winner、failed/cancelled/partial、compact 缺失而 full response 可恢复。
3. close 未生成/已生成缺 hidden、多 close 歧义、request/branch/sample 错绑、SHA 篡改。
4. 多 run 同名 rollout、同 task 多 repetition/attempt、重复 capture alias、模型/tokenizer/site 不兼容。
5. 后续 source run 新增评测不改变已冻结标签；已绑定 trajectory/publication bytes 改变可检测。
6. 外盘各类产物读写、空间不足、中断、同名副本冲突、分组泄漏。

## 11. 本次交付与后续边界

后续实施的完成标准是独立 steering dataset 和 LDA 正式 CLI、可指定磁盘的三类产物、身份/标签/数值校验、真实多 run 小样本离线验收和文档。

**本次仅交付调研和计划，不创建真实 dataset/cache/analysis，不改根配置，不启动 rollout/GPU/训练，不清理或修改原 capture。** 实际源 run、轨迹、磁盘由实施时显式指定，文中的抽样案例不是用户正式 cohort。Payload 计算和实际 steering 另行设计。
