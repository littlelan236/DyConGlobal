# B=8 关键 step 续跑数据采集

本功能通过正式 `dycon run` 生命周期采集三组对照数据。母轨迹使用 parallel_consensus，保存全部 8 个候选的原 estimator 分数和实际 sampler 分布熵；选点后，各原候选在独立初始容器中重放前缀、注入一次、单路运行至结束。没有运行时容器快照、commit、导出或可写层克隆。

部署位置：`10.249.187.9:/home/wangtianyi/DyConGlobal`。本文描述采集端；对应的 RL 导入和训练功能部署在 `10.249.189.84`，其本机说明为 `docs/unsorted/KEY-STEP-GRPO-CN.md`。两台服务器各自维护代码，交接的是采集数据与证据文件。

## 支持范围

当前支持 SWE-bench + mini-swe-agent、managed TP=1、非 steering、显式整数 generation.seed、原始固定镜像、开启 native token export。多卡使用已有 per_gpu_tp1 serving pool。TP=4 历史 run 可以作为参数来源，但新 run 必须通过允许的 clone overrides 设为 TP=1，并核对 GPU/端口资源；历史缺失的熵不能补写成新证据。

`RL/dataset/configs/key-step-source.yaml` 为 run-local 配置入口。模型、estimator、资源、预算、任务池由正式基础配置或 `--from-run` 冻结。母轨迹强制 B=8、parallel_consensus、every_step，并关闭按 limits 重新抽样。

## 正式工作流

大写名称是需替换的路径或新 run 名。从项目根目录执行，基础配置必须已有有效 estimator。clone 使用 `--from-run`，并保留源 run 的 repeat_count（大于 1 时须显式传入）。

```bash
.venvs/runner/bin/dycon run create A_RUN --operation rollout --tasks-file TRAIN_TASKS \
  --key-step-config RL/dataset/configs/key-step-source.yaml
.venvs/runner/bin/dycon run start A_RUN --detach
.venvs/runner/bin/dycon run wait A_RUN

.venvs/runner/bin/dycon rl-dataset key-step-plan --source-run A_RUN --output PLAN_DIR
.venvs/runner/bin/dycon run create SCAN_RUN --operation rollout --from-run A_RUN \
  --key-step-config PLAN_DIR/scan.yaml
.venvs/runner/bin/dycon run start SCAN_RUN --detach
.venvs/runner/bin/dycon run wait SCAN_RUN

.venvs/runner/bin/dycon rl-dataset key-step-select --plan PLAN_DIR/plan.json \
  --scan-run SCAN_RUN --output SELECTION_DIR
.venvs/runner/bin/dycon run create CHILD_RUN --operation rollout --from-run A_RUN \
  --key-step-config SELECTION_DIR/continuation.yaml
.venvs/runner/bin/dycon run start CHILD_RUN --detach
.venvs/runner/bin/dycon run wait CHILD_RUN

.venvs/runner/bin/dycon harness prepare CHILD_RUN
.venvs/runner/bin/dycon harness run CHILD_RUN EVAL_ID
.venvs/runner/bin/dycon harness status CHILD_RUN
.venvs/runner/bin/dycon rl-dataset key-step-seal --run CHILD_RUN --output GROUP_DIR
```

scan 是 CPU 重放验证，不启动模型。首次不一致前已经通过的边界仍可选，不一致后保守排除。实际 child 仍独立重放并再次校验至目标边界。

K=0 的轨迹保留审计，不强行补点。全部 K=0 时不创建空 child run。重复 plan/select/seal 必须使用完全相同输入；输入或评测变化后使用新的输出目录。

## 分数、熵与选点

标准差使用所有原分支，ddof=0；被共识 eligibility 排除的有限分数仍参与计算。缺任一原分支分数即不完整，信号缺口不插值。

默认自适应保持此前实现：全轨迹 median 高度门槛；`noise=1.4826*MAD(diff)/sqrt(2)`；`prominence=max(0.10,1.5*noise)`；distance=3；wlen=9。仅原始轨迹首尾使用单侧端点规则，缺口两边不充当端点。不因 B=8 对标准差乘补偿系数或擅自改阈值。

生成熵使用温度、penalty、grammar、top-k/top-p 等变换后的实际 sampler 分布，单位 nats。先对每个候选实际输出 token 取均值，再对 8 个候选等权平均。隐藏的尾部 stop token 不计入保留输出。greedy 分布熵为 0。只保存标量，不落盘词表向量，不消耗 RNG。

先在完整 score 信号上找 adaptive 峰，再与共同可采集范围 U 相交得到 K。U 要求重放通过、8 个原响应/score/熵/native ledger 完整，没有 abort/loop 截断。随机组在 U 中均匀无放回选 K；熵组按全分支平均熵 top-K，同分取较早 step。方法重叠时共享物理 child，仍保留各方法逻辑成员关系。重复动作仍保留原分支多重性。

## 纯重放与恢复

母轨迹 query 前记录历史请求摘要、执行动作数、调用/费用/格式错误/循环状态和已消耗时间。动作额外记录参数与观察摘要，不重复正常原始响应。文件验证清单按边界增量保存。

child 使用同一原镜像、bootstrap、包约束及网络策略，只重放 t 之前实际执行的工具动作。逐动作严格比较输出/返回码/异常，逐边界比较文件与逻辑 Git 状态。通过后恢复 H_t 与 agent 状态，用原 parser 注入原候选，再进入普通 agent 共用的 continue_run。母轨迹 t 之后的消息不进入 child 上下文。

验证覆盖 `/testbed`、`/tmp`、testbed Python 的安装包版本/RECORD 元数据、Git HEAD/逻辑 index/diff。文件≤1MiB 计算内容摘要；较大文件只看大小/mtime，发生变化即排除，不对大文件或镜像做全量 SHA。可写外部挂载和残留后台进程不受支持。状态探针失败的边界不可用，母轨迹继续执行。

这个范围不能证明外部服务状态、进程内存、范围外任意文件、未更新元数据的包内容相同。联网输出变化会导致重放被拒绝；复现相同输出不等于远程服务状态被固定。预算恢复保留母轨迹已消耗的调用数、费用和时间，原候选消耗一次 query，重放耗时单列为基础设施成本。

child 内存保存完整 H_t，磁盘轨迹只保存注入及实际后缀，公共历史使用单一 prefix 引用。

## 混合调度与失败

去重后的所有 child 构成 run 内共享队列，固定 seed shuffle，再按 repo family 轮转。所有 serving replicas 和重试共享调度器：同 task 活跃≤1；family 上限为 `max(1,ceil(workers/当前待执行 family 数))`；同共享服务活跃≤1、启动至少间隔 5 秒；失败重排至少退避 30 秒，时间可配置。

未知任务默认共享 unknown-external 配额，可能让实际并发降到 1。应在 source YAML 的 task_services 中配置已经核实的服务，同一服务必须同名，本地任务可填空列表。shuffle 不能消除公网限流；配额作用域为该 run，不跨独立 run 或外部进程。

重试保持 child identity 和 seed，从初始镜像重放。环境/传输/证据故障保持 null。已接受的 agent 正常终止未提交为 0；已提交仅在正式 harness 有布尔 resolved 时取 0/1，未评测保持 null。同一母轨迹任一所选 parent 缺成员或证据，该母轨迹三组一起排除配对训练比较。

## 封存与训练边界

groups.json 保存 parent 的 8 个原成员、最终 reward、组内标准化 advantage、原分支证据和 child 后缀引用。完整但零方差组仍保留，train_eligible=false。seal 复核 native token、raw T=1 概率向量、逐 turn 身份和完整性；collection_complete 与 data_ready 分开报告。

`RL.dataset.continuation.member_turns/iter_groups` 是 Python reader，保留各 turn 实际 prompt，prompt/tool observation mask=0、completion mask=1。objective 接收在各 turn 实际 prompt 上计算的 token logprob，仅在 child 内拼接 logprob 目标，复用现有 GRPO objective，按 token→child→parent 等权归约。

采集端 `grpo_ready` 固定 false，表示本机旧的单 step estimator dataset 的 build/lora-train 不能直接消费这个多 turn 终局 reward schema。它不表示 189.84 的新导入器尚未实现，也不要求为了训练修改这个冻结标志。raw T=1 概率不自动视为应用温度/top-p 后的历史行为策略概率。

## 向 189.84 交接

交接 `continuation-groups-v1` 的 `groups.json` 及其引用的母轨迹、child 后缀、decision、native token ledger、raw T=1 概率和评测证据。单独复制 groups.json 不足以训练。189.84 当前按封存路径读取引用，因此数据必须能按这些路径访问；不要手改 receipt 或冻结身份。本文不要求同步采集端源码。

在 189.84 项目根目录通过 `dycon rl-prepare-key-steps --groups COLLECTION_GROUPS.json --method adaptive|random|entropy --output DATASET_DIR` 分别导入三种方法，再通过其 `RL/experiments/key-step-grpo.yaml` 和 `dycon rl-train` 训练。此处三个 method 值需逐次使用，不是把竖线表达式直接传给命令。

RL 端重新计算奖励：明确循环 -3、有效 unresolved -1、resolved 为 +1 加组内长度附加分 0～2。长度使用整条轨迹的生成 token 总数，包含公共前缀；不计工具输出或重复 prompt。公共前缀不作监督目标，但后缀损失可通过完整前向图传播梯度。采集端二元奖励和旧 advantage 不作为这套奖励的替代。

## 已验证范围

实现阶段相关回归为 213 passed、2 skipped、3 failed；3 项失败在修改前源码中已复现。另完成真实 Qwen3.5-9B 的 8 分支熵记录验证、真实容器的前缀重放及篡改拒绝验证，以及 1000 组信号与原默认自适应实现的一致性对照。

尚未完成真实母轨迹 A → scan → 三方法 child → harness → 189.84 训练的完整端到端试点。上述工程验证不等同于批量采集成功率、联网重放通过率或 RL 效果验收。
