# DyConGlobal 领域术语

## 任务与运行

**Task**：由 benchmark、revision 和 `instance_id` 标识的问题。

**Run**：一次冻结配置和任务选择的项目操作。Run 保存 rollout、attempt、patch、evaluation 及来源证据。

**Rollout / trajectory**：一个独立采样产生的模型回复、工具动作、环境观察和终止结果的有序记录。复制已有轨迹不构成新样本。

**Repetition**：同一 task 的一个独立采样槽位。`repeat_count` 是每个 task 计划的 repetition 数。

**Attempt**：执行一个 sample slot 的物理尝试。基础设施 retry 属于同一个 sample slot；重新采样必须使用新的 repetition 身份。

**Submission**：agent 交付 patch 的行为，不代表 patch 已解决任务。

**Resolved rollout**：被选中的 patch 经绑定的 benchmark evaluation 判定解决任务的轨迹。

**Infrastructure error**：阻止可靠完成执行或评分的环境/服务错误。它不是 benchmark 未解决结果，resolved 保持 null。

## 数据与标签

**Catalog**：版本化 benchmark task 定义及 task partition。它不是 run 证据或训练数据 split。

**Run evidence**：绑定 run 的 task、sample、attempt、canonical rollout、patch、evaluation 及 provenance。

**Trajectory selection**：在明确来源 run 后，冻结供某一操作使用的 rollout 成员集合。

**Boundary anchor**：一条模型回复中固定的信息观察位置，例如 `think_close`。

**Remaining workload**：成功轨迹在指定 anchor 之后到最终成功提交之间的实际步骤数或 reasoning token 数。它是轨迹相关代理，不是最小必要工作量。

**Time to termination**：从 anchor 到轨迹以任意原因终止的剩余长度。失败轨迹也可观测，但它不等于剩余成功工作量。

**Exact native token IDs**：由运行时原生 tokenization 记录的精确 ID。Tokenizer 重建值是不同证据，不能互相冒充。

## Analysis

**Action observation**：一个 assistant message 及其提交的 tool-call 记录，是 action 分类的观测单位。

**Primary action category**：按固定优先级从可观察 action 组成得到的唯一类别；无法安全判定的记录标为 `OTHER`。

**Difficulty dataset / split / attempt**：分别是冻结样本与特征的数据集、该数据集上的固定成员划分、以及使用固定输入和配置的一次训练/评估操作。

**Published predictor**：从训练结果显式发布、供推理使用且绑定来源参数和数据身份的模型版本。

**Capture publication**：带 receipt 的 finalized hidden 及标签发布。它证明所绑定数据已发布，不证明 rollout 成功或 evaluation resolved。
