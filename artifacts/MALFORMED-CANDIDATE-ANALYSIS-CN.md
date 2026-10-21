# malformed patch 任务的候选分支核查

核查范围是 6 个官方 Harness 报告 `Only garbage was found in the patch input` 的任务。读取了每个任务最后一次 decision 的四个候选 response、工具调用命令、selected 标记和 score。

| task | selected | 未选分支的最终命令 | 判断 |
|---|---:|---|---|
| `django__django-11099` | 2 | branch 0: `git -C /testbed diff HEAD -- django/contrib/auth/validators.py > patch.txt && cat patch.txt` | **明确存在可生成 unified diff 的未选候选** |
| `django__django-10880` | 0 | branch 1/2/3: `cat patch.txt` | 未创建 `patch.txt` 的证据，不能确认 |
| `pydata__xarray-4075` | 3 | branch 0/1/2: `cat patch.txt` | 未创建 `patch.txt` 的证据，不能确认 |
| `scikit-learn__scikit-learn-26323` | 0 | branch 1: `cat patch.txt`；branch 3: `cat ... | git diff` | 不能确认；branch 3 命令本身不是可靠的 diff 输出方式 |
| `sympy__sympy-12481` | 3 | branch 2: `cat > patch.txt <<'EOF'`，内含完整 `diff --git` unified diff，再 `cat patch.txt` | **明确存在可生成 unified diff 的未选候选** |
| `scikit-learn__scikit-learn-13142` | 3 | branch 0: `cat patch.txt` | 未创建 `patch.txt` 的证据，不能确认 |

## 结论

至少有 **2/6** 个 malformed-patch 任务存在明确的“未选分支会生成 unified diff”的候选：

- `django__django-11099`：branch 0，score `0.9594`；最终选中 branch 2，score `0.2628`，选中分支执行 `cat /testbed/django/contrib/auth/validators.py`，导致提交源码全文而不是 diff。
- `sympy__sympy-12481`：branch 2，score `1.9721`；最终选中 branch 3，score `0.9453`，选中分支执行 `cat /testbed/sympy/combinatorics/permutations.py | head -910 | tail -20`，导致提交源码片段。

这两个未选分支的工具调用是在候选生成阶段记录的，未实际执行，因此不能声称已保存的 patch 字节与理想 diff 完全一致；但命令本身明确是 diff 生成/输出路径。其余 4 个任务的 `cat patch.txt` 没有对应的 patch 文件创建记录，不能把它们算作有效 patch 候选。

这说明当前问题至少部分是：Ridge 只按 `minimum_score` 选择剩余难度最低的 eligible 分支，不检查最终提交命令是否输出 unified diff。未选分支的 patch 质量没有进入评分目标。完整原始 decision 证据见 `malformed-candidates.raw.txt`，逐任务 trajectory/harness 证据见 `resolved-regressions64-evidence.json`。
