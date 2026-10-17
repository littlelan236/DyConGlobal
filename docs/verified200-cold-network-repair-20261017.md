# 200 tasks 冷加载评测：4 个联网任务失败分析与修复

本轮 200 tasks 首次冷加载评测结果为 196/200 resolved；200/200 均有加载前镜像不存在、NAS 加载成功、评测后镜像已删除的证据。4 个未通过项是测试执行时网络不可达，不是 NAS 镜像加载失败。上一轮将统一网络默认值设为 none，使依赖真实 HTTP 服务的测试出现回归。

| Task ID | 原始日志证据 | 受影响测试 |
| --- | --- | --- |
| psf__requests-1921 | httpbin.org 的 DNS 解析失败；48 failed, 71 passed | Requests 的 HTTP、重定向、认证、cookie 等集成测试；目标新增测试本身通过，但其他必需测试失败 |
| sphinx-doc__sphinx-10435 | 下载 python.org 的 python-logo.png 时 Network is unreachable | test_latex_images；目标新增 test_latex_code_role 通过，但 PASS_TO_PASS 图片测试失败 |
| sphinx-doc__sphinx-7985 | w3.org、google.com 等 URL 的 Network is unreachable | test_defaults、test_defaults_json、test_anchors_ignored |
| sphinx-doc__sphinx-8475 | 外部链接出现 Connection refused / Network is unreachable | test_defaults、test_defaults_json、test_anchors_ignored；目标新增重定向测试通过 |

以上 4 项在之前使用 Docker 默认网络的 gold 评测中均为 resolved。宿主机再次访问 httpbin 的 HTTP/HTTPS URL 和 python.org 图片均得到 HTTP 200；google.com 直连超时，因此恢复网络后仍需以实际 harness 结果为准。

修复位于 runtime/containers/repairs.py，仅对这 4 个 task ID 设置 network_mode=bridge。runtime/containers/factory.py 的 spec 改为读取 policy_for(instance) 产生的有效环境配置，使任务级网络设置真正进入 Docker create；baseline、steering、harness 共用该入口。依赖 bootstrap 仍使用离线 wheels，未修改 gold patch、test patch、测试断言或评分逻辑。

4 个 task ID 与当前 baseline manifest 的 258-task 清单不相交。未停止或重启 258 tasks 的导出、校验及排队进程，未修改其配置、数据或 NAS 产物。重测只使用一个 worker，并复用全局 NAS 加载槽位及磁盘空间预留机制。

验证：Ruff 通过；网络任务匹配、有效容器 spec、普通任务网络保持、共享 bootstrap、harness 等相关测试共 10 项通过。

重测输出：state/gold-image-audit/verified200-network-repair-cold-20261017/。逐镜像生命周期证据在 image-lifecycle.jsonl，最终结果在 summary.json。重测开始时 PID 为 3212302，采用 pidfd 等待退出事件，无定时进度轮询。

重测结果待进程退出后补充；不得将“已启动”视为通过。


重测结束后自动汇总（保留原始评测记录）：

```json
{
  "at": "2026-10-17T17:29:03.632326+08:00",
  "stage": "passed",
  "retry_tasks": 4,
  "retry_resolved": 4,
  "total_tasks": 200,
  "total_resolved": 200,
  "cold_lifecycle_verified": 200,
  "failed": [],
  "remaining_repair_images": [],
  "remaining_repair_containers": []
}
```
