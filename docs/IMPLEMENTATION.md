# 已实现与验收边界

实现了配置/数据契约、锁与原子写、显式导入、原始轨迹复用、并发 rollout、官方 harness adapter、模型服务生命周期、双锚点 enrichment、HF 分块重放、数据索引、预测评分与轨迹图、NAS load/cleanup/recovery、独立 EasySteer runtime。

baseline manifest 保持 revision=0、tasks=0、rollouts=0。difficulty.estimator、action.taxonomy、model.path/fingerprint/served_name 为空；difficulty/models、training、checkpoints 仅保留目录，不选择或训练估计器。

测试包含：manifest 并发写与分区隔离、patch/evaluation 绑定、失败标签不补零、当前锚点不含未来动作、capture 幂等与共享索引、合成小 Llama 分块/完整 forward 对齐、同镜像去重/保护、4 路加载并发、失效空间预约恢复、并发 rollout/resume、真实 mini-swe-agent→假 HTTP endpoint→提交协议、harness success/infra 区分。

真实 Docker smoke 使用极小合成 OCI fixture：4 worker 从 NAS stream load、创建并移除容器、最后删除本工程导入镜像；不执行真实 shell、SWE-bench 任务或 gold tests。

正式模型的 hidden site/数值校准、真实工具执行环境、SWE-bench gold、模型服务 GPU/TP/图性能仍需在范围和 checkpoint 确定后做专项验证。
