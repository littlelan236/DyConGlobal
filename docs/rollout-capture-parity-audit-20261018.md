# DyConGlobal 普通 rollout / capture rollout 一致性审计

本次对本地与远程当前代码、实际运行快照、服务启动记录和 baseline manifest
进行了核查。核心文件的本地/远程 SHA-256 一致。这里只记录审计结果；未重启
暂停的正式任务，也未将已失败的启动结果当作合格的正式 rollout 数据。

用户指出的工程内复用关系成立：

```text
普通 CLI -> resolve(config.yaml, overrides) -> run_manifest
capture 启动器 -> resolve(config.yaml, overrides) -> snapshot -> run_manifest(resume=True)
共同后续路径：ServingManager -> run_task -> DurableAgent + make_model -> LitellmModel
```

上一轮用“旧工程默认值未保留”解释这次工程内差异不准确。这次直接原因是启动时
人为传入了 `serving.tool_parser=hermes`，而非 capture 分支自动替换解析器。
在相同 resolved config 下，ServingManager 对普通/capture 共用同一个解析器字段。
当前唯一 config 的该字段是 null，并不存在一个普通路径自动采用 XML、capture 路径
自动采用 Hermes 的实现。

## 已确认的问题

| 问题 | 证据及影响 | 范围 |
|---|---|---|
| 正式启动额外覆盖解析器 | `runs/capture-verified200-20261018-120720/resolved-config.json` 的 overrides 明确包含 `serving.tool_parser=hermes`；`serving/launch.json` 与进程命令一致。实际 Qwen 输出是 XML 工具调用。 | 启动配置错误，非核心代码分叉。 |
| 速度实验同样硬编码 Hermes 和其他行为参数 | `hidden/evaluation/benchmark_rollout_throughput.py:79` 在 resolve 后直接修改 serving、generation、execution。普通/capture 两组都受影响，且修改后没有重新计算 config_hash。 | 诊断入口存在第二套参数；原 hash 不能证明这些实际参数。 |
| 既往普通速度实验并未证明真实 coding 流程正常 | 两组 baseline 各 5 条、capture 组 5 条，15 条均为 RepeatedFormatError；正式错误启动另有 7 条同样的终止结果。 | 不可将这些速度结果解释为正常 coding rollout 的性能验收。RepeatedFormatError 是真实终止状态，但不能等同于工具链验收通过。 |
| capture 故障可改变 agent 终止行为 | `baseline/rollout/model.py:111` 等待 native receipt 时直接传播异常。设置 record_and_continue，模拟 receipt 超时，实际 query 仍抛 TimeoutError。 | 正常模型响应可能因为采集控制链故障而变成 rollout 失败；失败策略未贯穿全链路。 |
| capture.dtype 未落实 | `runtime/serving/vllm_plugin.py:139` 原生采集固定 float32；controller 原样保存数组。配置 float16、输入 float32 的落盘实测仍为 float32。 | 是采集/存储 dtype 控制失效；模型 serving.dtype=float16 本身没有被改成 FP32。 |
| export.enabled 未落实 | controller 无条件建立异步导出队列。设置 export.enabled=false 的隔离实测仍写出 NPZ。 | 配置无法关闭导出。 |
| newline_policy / flush_interval_ms 存在未接线项 | manager 的 tokenizer inventory 无条件选择所有含换行 token，未按 double_only 过滤；在线 finalizer 也未按该策略过滤。flush_interval_ms 只有 schema 声明，没有运行时读取；插件 drain 间隔固定 0.1 秒。 | 静态确认的非默认设置问题；当前 all/100 的默认组合没有因此发生取值偏差。 |
| 格式错误响应的精确对齐遗漏 | LitellmModel.query 抛 FormatError 时，RecordingModel 在附加 native token 证据前就退出；finalize 只检查 role=assistant，而 mini-swe-agent 把这类响应放在 user 格式错误消息的 extra.response 中。 | 对齐不能完整覆盖真实模型请求。此次错误启动的轨迹可没有 assistant 消息，但仍有 native 请求，不能视为完整采集成功。 |

速度实验结果来自以下实际 run 的 `tasks/*/result.json` 和唯一 manifest：

```text
speed-baseline-20261018-034421 : 5 × RepeatedFormatError
speed-baseline-20261018-035007 : 5 × RepeatedFormatError
speed-capture-20261018-035007  : 5 × RepeatedFormatError
capture-verified200-20261018-120720 : 7 × RepeatedFormatError
```

已有 22 条记录仍存在 baseline manifest 中。本审计没有删除或移动它们；恢复正式
运行前应保留原始证据并明确隔离错误启动/诊断结果，防止混入正式数据选择。

## 已核对一致的部分

对同一 cfg 仅切换 capture.enabled、添加必要 capture provenance 后，实际调用
model_config 并去掉 dycon_capture/capture_select，剩余请求配置完全相等。
温度、top_p、top_k、seed、max_tokens、thinking 模板参数和 steering payload
不是分别在两套模型客户端中实现的。

对正式运行快照逐项比较唯一 config，generation、agent、images、timeouts、steering
均未出现值覆盖；environment 只有相对路径解析为绝对路径的正常变化。GPU、TP、
workers、max_num_seqs、KV 预算、模型身份和 capture 开关/层/路径是显式运行覆盖项。
解析器是此次额外选错的服务设置。

服务参数存在采集所需的明确差别：启用原生采集插件、worker extension、hook 所需
的 EasySteer 支持和图采集尺寸；混合 prefill/capture 的执行方式也不能承诺与普通
服务完全相同。这些区别需独立验收，但不构成把 tool parser 换成 Hermes 的理由。

## 隔离复现实测

复现使用 runner venv、临时产物目录及 mocked 模型响应，不启动 GPU 作业，不改变
线上任务状态：

```text
REQUEST_PARITY True
EXPORT_PROBE export_enabled=false, configured_dtype=float16, persisted=true, actual_dtype=float32
EXPORT_PROBE export_enabled=true,  configured_dtype=float16, persisted=true, actual_dtype=float32
FAILURE_POLICY_PROBE configured=record_and_continue, result=TimeoutError
```

此前的 103 passed / 5 skipped 是已有测试套件结果。其伪造 HTTP endpoint 已直接返回
结构化 tool_calls，不经过实际 Qwen XML 解析，所以不能用它证明真实模型工具链正常。
独立的 exact-token hidden 数值比较仍应按其原测试范围理解；本审计没有据此否定其数值，
也不能用数值比较替代真实 agent 工具调用验收。

恢复前需要统一 parser 的配置来源，清除速度脚本的静默行为覆盖并冻结真实 config hash，
补齐上述采集配置与失败路径，然后用相同配置下 capture 开/关的真实工具调用作为启动门槛。
