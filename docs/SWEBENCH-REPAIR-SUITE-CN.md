# SWE-bench 修复验证入口

使用 `.venvs/runner/bin/dycon harness repair-suite --plan PLAN.json` 执行阻塞阶段；该服务委托已有 run/harness/images/wheelhouse CLI，不实现另一套评测器。

计划 schema_version=1，绑定 source_run、task_count、task_ids_sha256，配置 collection_smoke、timeout_diagnostics、image_repairs、task_settings、experiments 和 negative_source_run。新 suite_id、新 wheelhouse_id 保留历史失败现场。输出位于项目 `temp/<suite_id>/`；gold/test patch 和 required 不允许修改。

依赖发布先于评测和镜像工作。`dycon wheelhouse build ID --source DIR --download-plan PLAN.json` 支持精确的 `{schema_version:1,python_tags:[cp36,cp38,cp39],releases:[{name,version}]}` 计划。先解析整份 PyPI release 分发及兼容标签，再下载；没有 wheel 时拒绝，不能隐式源码安装。实际文件必须符合 PyPI SHA-256。发布 manifest 保存下载计划摘要、解析结果和每个 wheel 的 METADATA。失败 staging 不覆盖，改用新 ID。

如果 wheelhouse 已预先发布，计划可绑定 wheelhouse_manifest_sha256；服务校验发布摘要及所有 wheel 后使用。

`reuse_baseline_batches` 可为 collection-smoke、timeout-diagnostics 配置 `{path,sha256}`。只复用项目 temp 中有内容摘要绑定的完整台账：任务 ID 集合、execution_finished、audit_complete、所属容器清理和完整任务配置必须一致。结果明确标记 reused 和原 run_id；不是新执行的计数。新依赖候选和 gold145 不复用旧评分。

编排用新 run 克隆冻结配置，通过原始 selection.source_run_id 获取真实提交证据。候选代表 gold 不满足完整契约则回退；有适用负例时要求相同 scheme 的 ValidOutcome/reward0。最终 gold145、负例 pair 和未覆盖方案分开统计。audit 非零仍保存 acceptance 台账，不能将不完整证据记作通过。

用独立于 SSH 的受管服务执行长阶段；启动确认一次后结束交互，不循环查询进度。completion.json、batches.json、experiments.json、pair-results.json 是阶段结果。pytest phase/一次超时栈属于执行证据。

失败项续测可指定 `failed_only_baseline: {path, sha256}`、精确 `task_ids` 和 `final_batch`。
基线必须是 source_run 的完整 acceptance 台账，审计和清理完成，选集必须恰为 gold_valid 不为 true 的任务；之前通过的任务拒绝混入。空 collection_smoke/timeout_diagnostics 不执行测试。`run_negative: false` 显式禁止加载、创建和运行负例，只报告 gold 验证结果；默认仍保留原负例流程。

Pydicom 路径参数修复通过任务 environment_overrides.variables 的 `DYCON_SWEBENCH_TEST_IDENTITIES` 声明完整 `{required,raw}` 契约。只允许已知历史数据根到 `/testbed/pydicom/data/test_files/` 的迁移，方法名、每个文件参数、顺序和参数后缀须完全相同。官方评分与外部审计共用适配器，审计另核对绑定的真实 collection 和 setup/call/teardown，歧义及缺证据拒绝验收。冻结 required/gold/test patch 不变。

资源层的 `sklearn-california-v1` 仅支持校验值与原任务base一致的 California housing 压缩包；还绑定 loader 源码SHA，在容器内按官方 loadtxt、列顺序和joblib路径生成原生缓存。未知转换、错误原始摘要、冲突缓存拒绝启动。
