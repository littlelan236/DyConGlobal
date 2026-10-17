# Two gold-patch failures: root-cause analysis

Both failures are environment failures. `completed=true` means the harness produced a grading result; it does not establish successful test collection or a healthy environment.

## sympy__sympy-23262

- Gold patch applied successfully. Target FAIL_TO_PASS `test_issue_14941` passed.
- Required PASS_TO_PASS `test_integral` failed. The traceback states that NumPy 2.0 removed `np.PINF`, which this historical SymPy version imports.
- Bootstrap explicitly installed `scipy==1.13.1` and resolved its NumPy dependency to `numpy==2.0.2`. Environment probes before and after evaluation confirm both versions.
- The environment policy pins only SciPy, with no NumPy upper bound. This allows an incompatible dependency for SymPy 1.11.dev0.
- Test file totals: 88 passed, 20 skipped, 9 exceptions. Eight additional exceptions involve conversion of strings such as `np.float64(...)`, consistent with this old SymPy code's incompatibility with NumPy 2 scalar representations. The scoring report specifically records `test_integral` as the required PASS_TO_PASS failure.
- Repair: add a compatible NumPy constraint (candidate `numpy==1.26.4`, alongside SciPy 1.13.1) and provide its wheel through the shared container bootstrap policy. Verify with a fresh gold run; no repair validation has been performed in this analysis.

## pylint-dev__pylint-4661

- Gold patch applied successfully. It adds `import appdirs` and declares `appdirs>=1.4.0` in setup.cfg.
- Existing image lacks appdirs. Policy has no package constraints and wheelhouse=null, so `/opt/dycon-wheels` is not mounted.
- Official evaluation runs `python -m pip install -e .`, which tries to obtain appdirs. With network_mode=none it cannot resolve/download it. Log: `No matching distribution found for appdirs>=1.4.0`.
- Test collection then fails while loading tests/conftest.py, ending in `ModuleNotFoundError: No module named 'appdirs'` from pylint/config/__init__.py.
- `test_pylint_home` was not demonstrated to fail an assertion: it never ran. Harness grades the absent expected FAIL_TO_PASS test as failure.
- Repair: provide `appdirs==1.4.4` as an offline wheel and install it in the shared container bootstrap entry, then rerun gold in a fresh container. Do not add a harness-only pip install or enable external network as a substitute.

## Limits and previous classification

The earlier classification of these two results as real gold-patch failures was incorrect. The combined 198/200 pass count is a grading observation, not evidence that all 200 environments are healthy. These two environments remain unvalidated until dependencies are repaired and gold passes. Rootless Docker namespace warnings are separate from the dependency failures documented here; this analysis does not prove they caused previous ConnectionResetError results.

Evidence includes original report.json, test_output.txt, eval.sh, run_instance.log, bootstrap.log, environment-policy.json, container-spec.json and both environment probes, retained in each task subdirectory. No benchmark tests or gold patches were altered.


## 修复验证（2026-10-17）

两项任务已通过统一容器 bootstrap 修复并重新执行 gold：

- `sympy__sympy-23262`: task override 将 wheelhouse 指向 `runtime/cache/gold-wheels`，并加入 `scipy==1.13.1`、`numpy==1.26.4`；结果 `completed=true, resolved=true`。
- `pylint-dev__pylint-4661`: task override 使用同一 wheelhouse；其中包含离线 `appdirs==1.4.4` wheel；结果 `completed=true, resolved=true`。

容器创建和启动仍走 `ContainerFactory.create()` 与 `ManagedContainer.start()`；bootstrap 由 `EnvironmentPolicy.bootstrap()` 执行，harness 只提交 gold patch 和运行官方评测。
