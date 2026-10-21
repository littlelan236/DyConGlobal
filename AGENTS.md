# DyConGlobal 工作约定

先读 `README.md`、`docs/ENTRYPOINTS.md` 和 `docs/MINIMUM-RUN-CN.md`；容器问题另读 `docs/container-environment-repair-cn.md`，再核对实际代码。历史 run 证据不代表当前实现或当前数据规模。

## 统一入口与行为对齐

- 正式任务使用 `core.cli`（安装后的 `dycon`）；各任务命令、环境和产物见 `docs/ENTRYPOINTS.md`。
- 已有统一入口时，不得自行创建临时脚本、复制 launcher、直接调用内部 runner/worker，或另起一套配置、任务选择、模型请求和产物写入逻辑。这会绕过身份校验、锁、不可变快照、恢复和 receipt，造成行为不一致。
- 普通、capture、steering、steering+capture rollout 都通过 `dycon rollout`；只在唯一 `config.yaml` 中切换开关。正式 rollout 拒绝 `--set`，resume 使用原始快照。
- 批量历史 extraction 使用 `enrichment.scripts.launch_extraction`；queue worker 由 launcher 调度。该 launcher 固定 `anchors-20261017` 批次及 GPU 0–3，不是通用批次创建工具。
- 诊断先查找已有 `hidden/evaluation/`、`runtime/scripts/` 工具，核对参数、硬编码范围和副作用。历史 speed-test/验收脚本不能替代生产入口。
- 只有 Python API 的能力应明确说明边界。需要新增正式任务时，应在既有 CLI/模块中实现可维护入口并补充文档；不要用临时训练/提取脚本填补入口空缺。
- action trajectory 分类使用 dycon action-classify，直接读取 manifest 绑定的 canonical trajectory；不得要求 enrichment 作为隐含前置，也不得另写 trajectory 读取脚本。

## 配置与数据

- 唯一可编辑配置为 `config.yaml`；唯一任务分区、用途、轨迹和评测关系来源为 `catalog/manifest.json`。
- `runs/<run>/resolved-config.json`、selection、hash 和 artifact receipt 是不可变证据，禁止为适配新参数而修改旧快照。
- unresolved、未评测和基础设施故障的成功剩余量标签保留 null；重建 token 证据不能冒充 exact IDs；action 分析复用现有 hidden。
- 使用任务对应的 `.venvs/runner`、`.venvs/analysis`、`.venvs/easysteer`、`.venvs/latent-vllm`，不混用推理和提取 runtime。
- 保留已有未提交修改。SSH 密码和 API key 不得写入工程文档或配置。
