# RepeatedFormatError 具体原因核验

结论：此次错误集中出现的直接原因是 XML 工具调用被交给 Hermes 的 JSON 解析器。
已在不重新生成文本、不启用 GPU 的条件下，用当前 pinned vLLM 的两种原生解析器
对完全相同的已保存响应进行对照重放，定位到了确定的解析失败环节。

| 来源 | 轨迹数 | 保存的格式错误响应数 | Hermes 提取有效 bash | qwen3_xml 提取有效 bash |
|---|---:|---:|---:|---:|
| speed-baseline-20261018-034421 | 5 | 15 | 0 | 15 |
| speed-baseline-20261018-035007 | 5 | 15 | 0 | 15 |
| speed-capture-20261018-035007 | 5 | 15 | 0 | 15 |
| capture-verified200-20261018-120720 | 7 | 21 | 0 | 21 |
| 合计 | 22 | 66 | 0 | 66 |

全部 66 个响应都包含 `<function=bash>` 的 XML 工具调用，结束原因均为 stop，
并非输出 token 上限截断。有效调用判定同时要求 parser 返回 tool_calls、函数名为
bash，以及 arguments 可解析为包含字符串 command 的 JSON 对象。
当前运行时的 qwen3_xml 与 qwen3_coder 指向同一个原生解析器类。

失败链路：

```text
模型生成 <tool_call><function=bash><parameter=command>...</parameter>...</tool_call>
 -> Hermes 对 tool_call 内部执行 json.loads，遇到 XML 标签而失败
 -> HTTP 响应保留原始 content，但结构化 tool_calls=null
 -> mini-swe-agent 只从结构化 tool_calls 读取动作，不再次解析 content
 -> No tool calls found -> FormatError
 -> 连续达到当前 max_consecutive_format_errors=3 -> RepeatedFormatError
```

因此用户看到的是“模型明明给出了命令，agent 仍反馈格式错误”。反复提示模型提供
bash 调用无法修复服务端使用错误协议解析器的问题。

历史对照：唯一 manifest 中迁移的 1,104 条历史轨迹，终止结果分别为 Submitted 872、
LimitsExceeded 154、ContextWindowExceededError 76、RepeatedFormatError 2。
历史 RepeatedFormatError 占约 0.18%，与用户“以前很少出现”的观察一致。

两条历史失败的原文核验：

- `r-e4901acdb0794911ae715440f9b79f71`，来自 speed20-20261011-registry_cap。
  轨迹已有 67 条消息；最后的响应反复用 Markdown 的 bash 代码块展示提交命令，
  未实际产生工具调用。这是模型后期没有遵循调用协议，不是此次有效 XML 被错误解析。
- `r-2a26198c0752d9a644ac2fa4c77288b3`，来自 tact-add-easysteer-eval-20261014。
  轨迹有 145 条消息；最后的响应 finish_reason=length，每次 completion 81920 tokens
  全部计作 reasoning，尚未生成工具调用就达到输出上限。这是输出被截断。

已核对的历史常规/EasySteer 运行配置使用 qwen3_xml，连续格式错误阈值为 10；当前
DyConGlobal 阈值为 3。这一差异使错误更早终止，但不是此次系统性失败的根因：同一批
响应只换正确解析器即可全部恢复有效调用，提高阈值只会多做几次失败请求。

本对照证明当前这 66 个已保存响应的直接失败原因，不证明 capture 对所有 prompt、
hidden 数值或采样行为完全没有影响，也未替代修正配置后的真实工具执行验收。
生产主进程在本次核验中保持暂停。

逐响应解析结果、response_id、rollout_id 与消息坐标保存在：
`docs/repeated-format-error-parser-replay-20261018.json`。
