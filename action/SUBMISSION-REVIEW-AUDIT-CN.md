# Submitted patch 与收尾复查分析

任务日期：2026-10-26。工程根目录：`/home/wangtianyi/DyConGlobal`。

## 1. 已确认的统计范围

按用户明确口径：同一组 80 tasks 的 rollout run 中，eligible submitted **严格大于 30**。
以 `verified80-parallel-baseline-20261021/tasks.json` 的 80 个 instance ID 为参照，
检查全部 active run 的冻结 plan/tasks/evidence；不是按目录名包含 `80` 搜索。
共有 17 个候选 rollout run，纳入以下 5 个；完整纳入/排除理由在输出 `scope.json`。

| Run | 已登记 canonical 轨迹 | Submitted | 已评测 submitted |
|---|---:|---:|---:|
| `parallel-consensus32-dynamic-w2-s64-lowreserve-t8h-20261025` | 86 | 40 | 40 |
| `verified80-parallel-baseline-20261021` | 80 | 73 | 73 |
| `verified80-parallel-reasoning-fraction-gpu01-w4-s16-20261025` | 86 | 61 | 61 |
| `verified80-parallel-reasoning-log1p-20261021` | 80 | 69 | 69 |
| `verified80-parallel-step-linear-20261021` | 73 | 59 | 52 |
| 合计 | 405 | 302 | 295 |

观测单位是 source run + canonical rollout ID。包括已登记的全部 eligible submitted attempt，
不是每个 task 只保留一个 final attempt；86 条轨迹并不意味着有 86 道题。
excluded/quarantined 不纳入；归档的迁移前副本不重复计数；未登记的进行中轨迹不纳入本快照。
脚本检查跨来源相同 task/trajectory SHA 是否重复，发现则拒绝计算。

`verified80-baseline-harness-repeat-20261026` 和
`verified80-baseline-harness-supplement-slots8-20261026` 只贡献重新评测证据，不能新增轨迹样本。
其他 harness run 必须通过评测 provenance 明确关联被统计的 source run 才可参与。

## 2. 空/无效 patch 的定义与结果

主口径为**每条 submitted 轨迹最新的非 gold 评测**；附报历史任一次出现 patch 失败。
最新由 SHA 绑定报告的时区明确的 `finished_at` 决定，不使用文件修改时间或
`selected_evaluation_id` 的登记顺序。外部重评还必须匹配 rollout ID、trajectory SHA、
patch SHA、instance/repo/base_commit。

旧报告的 `infra_error + failure_type=invalid_patch`、`infra_error + MissingPatch`，
显式 `invalid_patch`/`invalidpatch`，以及新版 `completed + failure_type=invalid_patch`
按具体原因统一。新版 completed 不代表有效 patch 或 resolved。

- 空 patch：实际 SHA 校验的 patch bytes 为空/全空白，且评测明确为 MissingPatch/patch 失败；
  对旧的无原因 infra_error 可用相同 SHA 的空 artifact 作有限补证。网络/依赖等明确其他原因优先保留。
- 非空无效 patch：评测明确记录 patch 应用失败。不能把普通 unresolved、测试失败、
  网络失败、依赖错误、timeout 全部算进去，也不把“有合法 diff 头”当作成功应用的证明。
- 某些迁移后的 outcome 丢失 failure_type/reason；从相邻 `source-report.json` 恢复诊断，
  必须校验原报告 SHA，并匹配 patch/status/resolved。原 evidence 不改写。
- 无评测仍是 unknown。空 artifact 本身不能制造一条评测失败。

| Run（简称） | 空 patch | 非空无效 patch | 失败数 / submitted | 比例 | 失败数 / 已评测 |
|---|---:|---:|---:|---:|---:|
| consensus32 dynamic lowreserve | 1 | 3 | 4/40 | 10.00% | 10.00% |
| parallel baseline | 1 | 7 | 8/73 | 10.96% | 10.96% |
| reasoning fraction | 3 | 4 | 7/61 | 11.48% | 11.48% |
| reasoning log1p | 2 | 6 | 8/69 | 11.59% | 11.59% |
| step linear | 1 | 5 | 6/59 | 10.17% | 11.54% |
| 合计 | **8** | **25** | **33/302** | **10.93%** | **33/295 = 11.19%** |

历史任一次失败也是 33 条，各 run 的历史任一次数量与最新失败数量一致。
step-linear 有 7 条 submitted 尚无评测，不能解释成成功；总体 10.93% 是“已确认失败占全部 submitted”，
不是未评测样本都已确认无问题后的最终失败率。
这些比例是本批次描述性统计；5 个 run 重复使用相同 80 tasks，不应按 302 个独立题目做二项显著性推断。

## 3. 运行入口与产物

实现为 `action/analyzers/submissions.py`，通过统一 CLI 调用；
共享 `action/datasets/trajectories.py` 的 SHA 绑定 canonical reader，不另实现轨迹加载器。
纯 CPU/只读来源数据，不请求 LLM、不启动 rollout/harness/GPU，不执行轨迹中的命令。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/analysis/bin/dycon action-submission-audit --list
.venvs/analysis/bin/dycon action-submission-audit submitted-patch-80tasks-gt30-final-20261026 \
  --task-count 80 --min-submitted 30
.venvs/analysis/bin/dycon action-submission-audit submitted-patch-80tasks-gt30-final-20261026 --check
```

输出为 `action/outputs/ANALYSIS_ID/`，同 ID 拒绝覆盖：

- `scope.json`：全部 17 个候选及排除理由；`audit.json`：范围、时间、全部输入文件 SHA 和策略。
- `summary.json` / `REPORT.md`：汇总、分 run 分母与未知样本。
- `trajectories.jsonl`：302 条逐轨迹 patch、最新/全部历史评测、原因和来源路径；CSV 为轻量表。
- `inputs/`：冻结 plan/tasks/evidence/评测报告；`code/`：实现快照；`audit-receipt.json`：输出哈希。
- `review-samples.jsonl`：8 条真实 canonical 轨迹的逐 step 抽样阅读材料，覆盖 5 个 run、
  有效/空/非空无效提交和137步长轨迹；不是随机代表性样本，也不是已验证的复查标签。

本审计使用专属 `--check`，不是 action 七分类分析，不传给 `action-check` 或 `artifact --scope action-analysis`。
`--check` 验证产物与输入 SHA、逐轨迹唯一性及汇总重算；来源 run 后续更新会报告快照已过期，
应使用新 analysis ID 重跑，不修改旧报告。

## 4. 实际轨迹观察：为什么不能只识别关键词

所有 step ID 均从 0 开始，只计选中的 canonical assistant messages，tool observation 不另加 step。
以下内容来自输出的 `review-samples.jsonl`，完整身份和 SHA 在相同记录内。

| Run / rollout | 实际观察 | 对原则的影响 |
|---|---|---|
| baseline / `r-d2bd1a35cb0f6baee93935b84b1b7c6d`，xarray-4629，13步 | step 3 修改 `xarray/core/merge.py`；4 读改后代码；5 重跑 reproducer；6–8 新建/修正边界测试；9 跑已有测试；10 清理；11 生成/查看 diff；12 提交 | 复查包含创建和修复临时测试，不能把所有写文件操作视为“再次开发”；清理/打包/提交应独立标注 |
| baseline / `r-fcb0568e4597df8cf1fa20e3aec6e1c8`，pylint-7277，32步 | 17–20 已在验证，但21又改生产文件 docstring；22后重新检查代码/测试；31最终提交的是源文件内容，经评测为无效 patch | 复查可以多次进入/退出；最后一次交付内容修改后的复查才是“最终复查”；验证充分不保证提交有效 |
| reasoning fraction / `r-b5bea03acff6604ef1dc227df049f8b2`，astropy-7336，19步 | 7后反复跑 reproducer、读现有测试、执行单测、看最终代码；16生成 patch；17又重跑 reproducer；18提交 | 打包后仍可继续复查；不能遇到第一次生成 patch 就截断 |
| consensus / `r-ec6db837637430b0bf33afcbc590af51`，astropy-7336，19步 | 7后核查；16尝试 `cat patch.txt` 提交，但之后17才生成 patch，18再次提交 | 必须依据实际终止/observation识别有效提交边界，不能用第一次出现提交字符串 |
| step-linear / `r-3ad340e10f12890f369e33ad854533ae`，pytest-6202，28步 | 12后查找测试、反复调整测试路径和忽略缺失依赖；25仍新建最小测试；26生成 patch；27提交 | 为核查而找测试、处理测试环境、临时验证脚本应计入复查成本；不能只数 pytest 命令 |
| baseline / `r-3ebc34567aaffe8886c2a8403a3fd7a5`，django-13513，40步，空 patch | 后半段运行验证和查看代码；34 `git diff` 为空，35查git历史，37工作区clean，39空提交；模型声称修复已存在 | 没有可辨识的新修改也可能有复查；标注 `no_delivery_change`，不能强行记长度0或宣称修复确实已存在 |
| step-linear / `r-77edf194284e3f8dc003292c0047f73c`，django-15368，137步 | 112–120仍反复检查；121/123修改生产代码，125测试失败，128回退该文件，129后验证、看diff、跑多组测试；136提交源代码片段而非diff | `check`可能是诊断/循环或继续开发；checkout/stash/restore改变候选交付状态，需要重新定最终复查边界 |
| log1p / `r-1ba2c626e15e0f11579184c7a35c14dc`，xarray-4629，11步 | 改动后建边界测试，7修正验证脚本语法，8清理，9生成patch，10提交 | 修复“验证脚本”不是修复“目标实现”，两者对阶段切分不同 |

管道 `pytest ... | head/tail` 可能掩盖测试进程返回码；LLM 的“all tests pass”也不一定可靠。
阶段标签表示执行核查的行为，不等价于验证成功，应另存 observation 中的执行结果和不确定性。

## 5. 建议的评价原则（提案，尚未全量标注）

### 5.1 先判断 step 的目标，再判断是否属于收尾

建议使用独立于现有互斥 action 七分类的阶段标签，保留原分类不改写：

1. `DEVELOP/DIAGNOSE`：定位根因、形成方案、修改拟交付实现或交付测试、继续修复目标行为。
2. `VERIFY`：针对已形成的候选解查看改后代码/diff/status，重跑 reproducer、回归、边界测试，
   核对需求和副作用。读文件/grep/find本身不够，必须由前后上下文、具体对象和意图支持“验证已有候选解”。
3. `VERIFY_SETUP`：直接为上述验证查找/读取测试、创建或修正一次性验证脚本、调整测试运行方式、
   处理仅阻碍验证的依赖/配置。不能把改动目标实现伪装成setup；是否属于交付测试要看实际用途和最终变更。
4. `PACKAGE`、`CLEANUP`、`SUBMIT`：生成提交payload、移除临时文件、提交，各自独立。
   一个step同时包含验证和打包/修改时标 `MIXED`，保存子操作/secondary tags，不重复计数。
5. `UNKNOWN`：缺少reasoning/内容或命令效果不清时保留未知，不能只凭关键词猜测。

判断依据优先顺序：实际命令/代码与目标路径、后续 observation、前后候选解的连续性，
再用 assistant content/reasoning 支持目的；“let me check”不能单独构成证据。
测试失败仍可属于复查；之后修改实现时退出复查，而不是事后把失败的核查全部改成开发。

### 5.2 “进入最终复查”的边界

- 存在候选解，并开始核对其正确性/完整性，是进入一次复查 episode 的积极证据。
- 对完整轨迹作回溯分段：再次修改交付实现/交付测试/docstring，或 checkout/stash/restore 使候选状态改变，
  则关闭当前episode；之后首次 VERIFY/VERIFY_SETUP 可开始新的episode。
- “最终复查”是最终候选状态稳定后、有效终止提交前的最后一个复查episode。
  临时测试的创建/修正不会自动重置，打包和清理可穿插；新的目标修复会重置。
- 同一个step既修改候选解又验证，按MIXED留存。仅按assistant-step时，严格边界设在其后；
  如要把step内后半段纳入，需要另外定义子操作单位，不得悄悄按整步全算。
- 没有目标修改但有收尾核查：单列 `no_delivery_change_review`；明确没有任何复查行为才为0；
  看不清则unknown。未提交/被截断轨迹的收尾长度属于右删失，不与正常submitted混算。
- parallel模式只分析canonical所选路径，不能把未选的32个候选分支累加为轨迹长度。

### 5.3 建议同时报告的长度

主单位先用assistant step，与现有action analysis一致；不要把assistant、tool返回和命令条数混用。

- **最终复查净步数**：最终episode中 VERIFY + VERIFY_SETUP 的step数；每步最多1次。
  PACKAGE/CLEANUP/SUBMIT/MIXED/UNKNOWN单列，提供含MIXED与不含MIXED的敏感性口径。
- **收尾跨度**：最终episode起点到有效SUBMIT之前的assistant步数，允许穿插打包/清理/未知。
  与净步数同时报告，反映收尾总成本和核查本身成本。
- **全程复查成本**：所有episodes的净步数，以及episode数、复查后返工次数。
  避免把“早就复查过，但后来又改代码”的成本丢掉。
- **占比**：净步数 / 全部assistant步数；跨度占比单列。分run报样本数、缺失率、median/IQR/P90、
  mean与macro比例，并按有效提交/空patch/非空无效patch分组。

示例：baseline xarray的step4开始最终复查，12为提交。
按本提案，4–9是6个VERIFY/VERIFY_SETUP步骤，10清理、11生成/查看patch单列PACKAGE，
收尾跨度为8步（4–11）。如果将11的diff查看也计为复查，则净步数为7；这正是需要显式固定的MIXED口径。
这只是**提案下的人工示例**，不是经过双人标注确认的总体统计。

token长度可以作为第二阶段指标：明确reasoning-only或全部assistant输出，优先使用可绑定到
选中request的native usage/token IDs；reasoning_content与provider/extra.response中的副本只算一次，
不能把branch总token、prompt累计token或重复存储的reasoning算进去。
若只能重新tokenize，必须标为estimated并固定tokenizer，不能称exact。墙钟时间另计，不混同思考长度。

### 5.4 建议的标注与验证流程

先对这8条真实样例确定边界和MIXED规则，再按run ×提交有效性×轨迹长度分层抽样，
扩充人工标注集。边界至少保留原始message_index/step_id、前后证据、标签、置信度和不确定原因。
使用两名独立标注者时报告标签一致率/κ、起点绝对误差与±1步命中率，再讨论分歧。
未验证规则前，不将关键词自动匹配输出称作“复查长度真值”。

当前已完成真实轨迹阅读、8条样例归档和以上原则提案；**尚未把302条轨迹自动标注为复查长度**。
下一步需先固定“净步数/收尾跨度/全程成本”主指标以及MIXED边界，再实施分类和总体统计。

## 6. 验证记录

- 新增 `tests/test_action_submissions.py`：16项通过，覆盖三种失败表示、网络/依赖排除、
  未评测分母、空分母、strict >30 cohort、共享reader及SHA拒绝。
- 实际5个run共302条submitted的完整审计运行完成；专属`--check`通过，67个输出文件校验完成。
- 现有action分类/绘图与新增测试合跑：52 passed，1 failed。
  唯一失败为 `test_run_config_freezes_classifier_cwd_and_no_command_execution`，
  原因是该测试仅写environment的resolved-config fixture缺少project_root，
  当前`rollout.retry.default_final_ids → core.config.load_snapshot`抛KeyError。
  用任务开始前原版canonical reader/selector在独立测试进程中复现相同错误，确认不是本次reader提取引入；
  未在本统计任务中修改无关retry逻辑或该旧测试。


## 7. 后续自动化方法探索

已补充30条新轨迹，新增文件角色/副作用追踪与状态机方法、可运行探索原型及方法对比，
详见 [复查长度自动判定](REVIEW-LENGTH-AUTOMATION-CN.md)。原型尚非正式全量复查长度统计。


## 复查长度实现完成

复查规则实现和此前5个run的全量运行已完成，见[复查长度分析](REVIEW-LENGTH-CN.md)。
该文给出302条submitted的逐轨迹结果、unknown分母、27项新增测试和可复现入口。
