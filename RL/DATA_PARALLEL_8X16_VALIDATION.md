# 8 卡 RL 8×16：实现与验证记录

日期：2026-11-14。代码已实现；状态为显式实验模式。GPU 验收按用户安排等待 8 张卡空闲。

## 已实现的协议

统一入口仍为 `dycon lora-train --training-config YAML`。完整 32 分支父组只评分一次，
冻结 retained、advantage 和 reference；确定性互补 16 分支分片分两轮执行。
常规窗口为 8 个不同父组各一个 half，两轮尾窗独立保留。
每条 branch 权重为 `1/(16*K)`，K 为本窗口实际父组数；使用 FP32 SUM。
Prepared/streaming 共用冻结计划，按 rank 选择 completion/reference。
新增执行窗口 cursor，区分已消费 half、见过的父组及两半完成的父组。
完整 update 后保存全部 rank RNG、optimizer、scheduler 和 adapter；失败不提交窗口。
Prefix telemetry 在成功和失败时显式 drain；状态和 TensorBoard 区分上述计数。

8 卡必须显式配置 16 分支、distinct-parent、boundary-grad；旧 1/2/4 卡模式保持支持。
Exact resume 要求相同协议、冻结计划和 world size。旧 v10 checkpoint 不能直接转换，
新阶段必须明确绑定真实起点的 anchor；streaming warm-start 和 8×16 append-resume 不支持。
完整字段与限制见 [协议文档](DATA_PARALLEL_8X16.md)。

## CPU/Gloo 结果

最终综合回归：**426 passed、12 skipped、29 warnings，400.06 秒，无失败**。
跳过项为未启用的 GPU/性能 opt-in 用例；warnings 是既有 fixture 的 observability 路径提示。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 timeout 900s \
  .venvs/rl/bin/python -m pytest -q tests/test_grpo*.py \
  tests/test_lora_config.py tests/test_lora_checkpoint.py \
  tests/test_lora_memory.py tests/test_lora_oom_recovery.py \
  tests/test_rl_observability.py tests/test_rl_observability_training.py
```

覆盖 G=8/9/12/13/16/6773、跨进程稳定划分、两轮完整覆盖、计划/cursor 篡改、
冻结评分/advantage/reference 保持、固定参数 loss/gradient 恒等式、非零 K3、变长 mask。
真实 8 进程 Gloo 与独立串行选定分支 oracle 比较梯度、裁剪、AdamW 和 scheduler，
覆盖 1–7 half 尾窗、idle rank、unused parameter、rank RNG 及身份/覆盖/非有限/指纹失败。
真实 tiny Qwen CPU 在 prepared/streaming 两条路径逐 4 个窗口边界 resume，
另有 service 锁、artifact verify、status/TensorBoard 与注入 supervisor 清理测试。

回归发现旧测试错误拒绝基线已支持的 `kl_beta=0`，已改为接受零并拒绝负值。
旧双 rank 非有限测试曾在进程收尾时 SIGABRT，单独复跑通过；仅给该测试增加退出 barrier，
随后两个异常窗口用例连续三轮通过，最终完整回归通过。未放宽数值容差。

## 真实数据 dry-run 与历史校正

正式 runner CLI 在 v10 数据 dry-run 通过：6773 父组、13546 half、1694 更新、216736 分支，
每轮各一个 5-parent/80-branch 尾窗。执行计划 SHA256：
`e687b69eb73f182b14bdd586919817e975d1663455c30cf56f16115574b6920d`。
Dry-run 未加载模型，不验证 GPU 容量。

只读审计确认历史 v10 为 2 ranks、每 rank 2 个完整组。63 步的 4158 条序列化 prefix 行
经 follower 历史精确匹配后还原为 252 条当前记录；rank 0 取当前前两条，follower 取末两条。
校正 prefix/suffix=2.4035077065，prefix 占两者总和的 70.6185475%。
这是跨 rank 同步分段计算时间之和，不是 wall time 或 8 卡提速预测。
原 metrics SHA256 保持 `ae3ec5218fe164758217f4793596e9e097888f5ec7a41ca755154f6fe1f8c972`。

原始日志、归属明细、配置和完整性清单位于服务器
`temp/data-parallel-8x16-acceptance/`，该目录为忽略的本地验收证据，不随源码提交。
任务 29 个交付文件 hash 一致，实施前归档内其他文件无变化；未覆盖已有未提交修改。

## 待验收

- 8 张空闲 GPU、Qwen3.5-9B、正式 CLI：两条数据路径、跨轮/尾窗、暂停恢复、全部 rank 指纹/RNG、artifact verify。
- 真实 v10 代表性长 prompt 和历史 OOM 窗口容量；16 分支不会令 prefix graph/boundary buffer 的形状减半。
- 同数据、同总 branch 预算的 4×32 对 8×16 短/长 prefix 吞吐，至少两次重复；报告 median/P90、branches/s、尾窗、总耗时、critical path、sync、I/O、显存、失败和启动/checkpoint 成本。
- 共同评测集、多 seed、同轨迹预算的训练质量对照。

GPU 用例见 `tests/test_grpo_execution_gpu.py`，尚未执行。短/长合成用例不能替代真实 OOM prefix。
当前没有提速、10% 吞吐门槛达标、不再 OOM 或质量提升的结论。
两半之间参数更新形成新优化轨迹，不宣称与整个 4×32 训练等价。
