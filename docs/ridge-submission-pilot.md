# 单层 Ridge × alpha 验证实验

正式入口（analysis 环境，工程根目录）：

```bash
.venvs/analysis/bin/dycon difficulty-ridge EXPERIMENT_ID --mode dry-run
.venvs/analysis/bin/dycon difficulty-ridge DATA_CHECK_ID --mode check-data
.venvs/analysis/bin/dycon difficulty-ridge FIT_ID
```

每个命令使用不同的新 ID，目录已存在时拒绝覆盖，包括未完成的目录。默认模式为 fit；dry-run 校验冻结选择、源证据并保存固定 task 划分，不读取全部 hidden 张量或拟合。check-data 另校验实际张量，构建可重建分层缓存。fit 完成网格搜索、保存模型、选择、预测、指标与图。命令不重新 rollout/提取 hidden。最终 receipt.mode 区分三类产物，不能把 dry-run/check-data 宣称为实验完成。

唯一 config.yaml 的 difficulty.ridge_probe 保存来源、receipt SHA、targets、layers、alphas、split_seed、validation_tasks、expected_tasks/expected_samples 和线程上限。当前为 submission-shortest375-20261019、238 task/14,384 行、190 train/48 validation、seed=42、32 layers×13 alphas×2 targets。离线调试可以使用统一 CLI 的全局 --set（放在子命令前）缩小 grid，结果保存精确 resolved config 和 grid 数量；调试产物不能冒充全网格验收。

新配置块是可选的；没有 ridge_probe 的旧 YAML 解析结果不增加新字段，保护旧配置哈希。既有快照、成功标签、训练 API 和 rollout 路径不重写。

## 数据与划分

只消费已发布的 rollouts.json、samples.jsonl 和 receipt。校验所有选择文件 SHA、当前 catalog 提交/隔离状态、task 和 canonical/patch SHA、source_files、sample→step→message 对应关系及两个非负有效标签。来源清单的旧 split=train 不是实验角色；split.json 是本次唯一划分。repo+instance_id 分组，单 task 只保留一条已选最短轨迹；同 trajectory SHA 不得跨区。

repo 配额约20%，singleton repo 全训练；其余 repo 两侧至少各一 task，再按来源最大余数法及 SHA256(seed|repo|instance_id) 确定性选择。当前验证恰好48 task，实际行数由 task 决定。任务/样本ID、repo/source配额、canonical step分布和样本顺序哈希全部保存。

标签为到该次提交（不要求 resolved=true）：reasoning 采用 log1p，step 原始线性。最终 close steps=1，未来 reasoning=0；保守排除 censored 后缀。两个目标用同一14,384行，不改原成功剩余量语义。

输入每次仅4096维单层 hidden；public 0..30 decoder output、31 final norm，在线 native32→public31。三类 locator 为迁移NPZ+row、补提单点NPZ、NAS tar member。原始/成员SHA、native sites、32×4096形状与有限值检查通过后才进入缓存。缓存按 source receipt和sample顺序标识，缓存各层有SHA；损坏时报错，不静默复用。

## 拟合与选择

仅 train 拟合每层 Standardizer，零/近零方差 scale=1；平方误差求和 + alpha||w||²，截距不惩罚，不拼层。全层正 alpha 网格为10的-4至8次方。每层复用一次谱分解（样本少于维度时用不截断的thin SVD），两目标和所有 alpha 独立得到参数；与原RidgeEstimator数值等价。

验证 pooled R²为主选择指标。两个目标分别取最大值，1e-10内并列先取较大alpha再取较小层号。负R²保留，零方差/不足两行/非有限结果为undefined，不能获选。最高值为选择后的验证分数，独立新任务测试未包含在本命令中。选优后不合并训练/验证重拟合。

CLI在导入NumPy前设置OPENBLAS/MKL/OMP/BLIS线程上限，默认4，CPU顺序逐层执行。共享分解没有改变Ridge目标函数。失败不写最终receipt，输入或代码变化要求新ID。

## 产物

位于 difficulty/checkpoints/EXPERIMENT_ID/：experiment.json、split.json、data-check.json（check-data/fit）、trials.jsonl（完整832条）、models/（832个NPZ）、selected.json、validation-predictions.jsonl、metrics.json、report.md、plots/ 和 receipt.json。fit时额外保存每目标验证R²网格、预测散点、按长度确定性选取的短/中/长轨迹图。

NPZ包含coef、intercept、feature_mean/scale、target、public/native site、alpha、fit_scope=train、源receipt及split SHA，可无pickle加载。预测为((h-mean)/scale)@coef+intercept；reasoning的原单位再expm1。每个候选有train/validation指标和模型SHA。冠军报告训练均值基线、原单位指标、source/repo/task分组和task-macro，undefined组及expm1非有限值明确计数。

state/difficulty-cache/中的32个float32分层memmap约7.02 GiB，另预留2 GiB安全余量。缓存可删除重建；权威输入仍为源选择与hidden，不重复发布新数据集。final receipt将模型/代码/划分/来源哈希绑定，并区分调试模式；代码在运行途中发生变化则拒绝完成。

相关测试：tests/test_ridge_probe.py、tests/test_difficulty_estimators.py。数值/小型fixture不替代真实来源核验，check-data不替代fit结果。
