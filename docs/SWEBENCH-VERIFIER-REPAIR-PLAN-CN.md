# 所有 VerifierError 的原因分析和修复计划

日期：2026-11-17。基线：目标服务器 `10.249.188.230` 的 `swebench-full-gold-20261116`，共1,276项；v2逐项审计145项VerifierError，124项主因已确认，21项因果待查。本文覆盖全部145项，未将历史gold控制结论冒充普通模型提交的自动归因规则。

正式训练结算已在两台服务器接入：新评测先审计再投影，VerifierError/Review的resolved和reward为null；新建RL训练数据排除明确无效轨迹的完整分组。原始official report、历史run、sealed数据集、checkpoint和运行中训练不改写。当前判定策略升级为v3，结算策略为swebench_validity_settlement_v1。两机208项CPU测试通过；本次并未修复下列环境故障。

## 分类总表

| 原因代码 | 数量 | 原因状态 | 优先级 |
|---|---:|---|---|
| `runtime_dependency_incompatible` | 56 | 已确认 | P0 |
| `missing_distribution` | 6 | 已确认 | P0 |
| `test_framework_nose_compatibility` | 21 | 已确认 | P0 |
| `test_resource_unavailable` | 26 | 已确认 | P1 |
| `test_resource_missing` | 2 | 已确认 | P1 |
| `missing_verifier_dependency` | 2 | 已确认 | P0 |
| `container_image_ownership_unmapped` | 8 | 已确认 | P1 |
| `required_test_parameter_identity_mismatch` | 1 | 已确认 | P1 |
| `parser_incomplete` | 1 | 已确认 | P1 |
| `missing_verifier_tool` | 1 | 已确认 | P0 |
| `gold_dependency_api_failure` | 10 | 待因果对照 | P2 |
| `gold_test_collection_failed` | 1 | 待因果对照 | P2 |
| `gold_native_process_crash` | 2 | 待因果对照 | P2 |
| `gold_runtime_recursion_failure` | 2 | 待因果对照 | P2 |
| `gold_runtime_warning_failure` | 1 | 待因果对照 | P2 |
| `gold_evaluation_timeout` | 5 | 待因果对照 | P2 |

原始125个reward0全部为VerifierError；19项基础设施错误同样无有效reward；另有1项官方reward1（Django15521）为无效正例。1,131项有效正例覆盖率为88.64%，不能报告为全量100%恢复。

## 各类修复措施与验收

### runtime_dependency_incompatible（56项，P0）

证据与判断：pvlib 启动导入使用 np.Inf，NumPy 2 已移除该 API，必需测试未运行。

修复：按任务历史 Python/依赖支持范围构建兼容 NumPy（优先验证 <2 的固定版本）与 scipy/pandas 组合，发布不可变 wheelhouse/环境 policy；不改项目源代码或 gold。

验收：导入/collect 成功；全部 required F2P/P2P 实际运行并通过。

任务：`pvlib__pvlib-python-1026`、`pvlib__pvlib-python-1033`、`pvlib__pvlib-python-1048`、`pvlib__pvlib-python-1053`、`pvlib__pvlib-python-1072`、`pvlib__pvlib-python-1073`、`pvlib__pvlib-python-1089`、`pvlib__pvlib-python-1093`、`pvlib__pvlib-python-1138`、`pvlib__pvlib-python-1140`、`pvlib__pvlib-python-1154`、`pvlib__pvlib-python-1160`、`pvlib__pvlib-python-1165`、`pvlib__pvlib-python-1176`、`pvlib__pvlib-python-1181`、`pvlib__pvlib-python-1186`、`pvlib__pvlib-python-1191`、`pvlib__pvlib-python-1193`、`pvlib__pvlib-python-1213`、`pvlib__pvlib-python-1216`、`pvlib__pvlib-python-1224`、`pvlib__pvlib-python-1239`、`pvlib__pvlib-python-1273`、`pvlib__pvlib-python-1296`、`pvlib__pvlib-python-1349`、`pvlib__pvlib-python-1368`、`pvlib__pvlib-python-1395`、`pvlib__pvlib-python-1426`、`pvlib__pvlib-python-1448`、`pvlib__pvlib-python-1458`、`pvlib__pvlib-python-1469`、`pvlib__pvlib-python-1478`、`pvlib__pvlib-python-1480`、`pvlib__pvlib-python-1518`、`pvlib__pvlib-python-1589`、`pvlib__pvlib-python-1606`、`pvlib__pvlib-python-1616`、`pvlib__pvlib-python-1623`、`pvlib__pvlib-python-1653`、`pvlib__pvlib-python-1666`、`pvlib__pvlib-python-1682`、`pvlib__pvlib-python-1707`、`pvlib__pvlib-python-1737`、`pvlib__pvlib-python-1738`、`pvlib__pvlib-python-1739`、`pvlib__pvlib-python-1764`、`pvlib__pvlib-python-1782`、`pvlib__pvlib-python-1822`、`pvlib__pvlib-python-1832`、`pvlib__pvlib-python-1852`、`pvlib__pvlib-python-1854`、`pvlib__pvlib-python-763`、`pvlib__pvlib-python-807`、`pvlib__pvlib-python-823`、`pvlib__pvlib-python-907`、`pvlib__pvlib-python-980`。

### missing_distribution（6项，P0）

证据与判断：离线 wheelhouse 缺指定构建/运行包：Astropy 4项 cython==0.29.34、1项 jinja2==3.0.3；pvlib-718 缺 nrel-pysam。

修复：为各任务 Python/架构补齐精确 wheel 或可验证的离线构建依赖，发布新 wheelhouse receipt。仅换 PyPI 源不能解决离线无包与版本不兼容。

验收：pip 安装阶段、preflight/probe 均通过；原 pin 不被静默替换；gold required 全通过。

任务：`astropy__astropy-13803`、`astropy__astropy-14907`、`astropy__astropy-14938`、`astropy__astropy-14991`、`astropy__astropy-14995`、`pvlib__pvlib-python-718`。

### test_framework_nose_compatibility（21项，P0）

证据与判断：历史测试依赖 nose setup/teardown，当前 pytest 8 移除支持；fixture 未初始化，或 FileSet/handler 全局状态未恢复。

修复：选择任务支持的 pytest/插件组合（先对照 pytest<8 的固定兼容版本），核对 setup 和 teardown 执行及状态恢复；不改 test patch。

验收：fixture 初始化与恢复回执，单测/整模块均通过；Pydicom1428/1598检查 FileSet.__len__，其他任务检查 handler。

任务：`pydicom__pydicom-1031`、`pydicom__pydicom-1048`、`pydicom__pydicom-1069`、`pydicom__pydicom-1076`、`pydicom__pydicom-1090`、`pydicom__pydicom-1139`、`pydicom__pydicom-1194`、`pydicom__pydicom-1236`、`pydicom__pydicom-1334`、`pydicom__pydicom-1365`、`pydicom__pydicom-1375`、`pydicom__pydicom-1413`、`pydicom__pydicom-1416`、`pydicom__pydicom-1428`、`pydicom__pydicom-1539`、`pydicom__pydicom-1555`、`pydicom__pydicom-1598`、`pydicom__pydicom-1608`、`pydicom__pydicom-1633`、`pydicom__pydicom-1674`、`pydicom__pydicom-1720`。

### test_resource_unavailable（26项，P1）

证据与判断：必需测试访问外部资源时出现 DNS、路由、reset、TLS 或 HTTP403；Requests18、Sphinx6、Seaborn1、scikit-learn1。

修复：逐测试记录 URL、预期响应/timeout 与实际失败；按需修 DNS/路由、代理、CA，或发布同内容哈希缓存。Requests负向网络测试必须保留预期超时/SSL语义，bridge 本身不足。

验收：对应请求行为符合测试意图，缓存内容相同，全部 required 通过；Sphinx8509额外 LaTeX 差异仍单独核查。

任务：`mwaskom__seaborn-2457`、`psf__requests-1635`、`psf__requests-1724`、`psf__requests-1733`、`psf__requests-1766`、`psf__requests-1768`、`psf__requests-1776`、`psf__requests-1921`、`psf__requests-1944`、`psf__requests-1962`、`psf__requests-1963`、`psf__requests-2148`、`psf__requests-2153`、`psf__requests-2317`、`psf__requests-2617`、`psf__requests-2674`、`psf__requests-2821`、`psf__requests-2931`、`psf__requests-5414`、`scikit-learn__scikit-learn-13620`、`sphinx-doc__sphinx-10435`、`sphinx-doc__sphinx-11266`、`sphinx-doc__sphinx-7985`、`sphinx-doc__sphinx-8269`、`sphinx-doc__sphinx-8475`、`sphinx-doc__sphinx-8509`。

### test_resource_missing（2项，P1）

证据与判断：Pydicom1241/1439 下载失败造成数据文件缺失；还可能并存 handler、nose、参数ID问题。

修复：固定官方测试数据版本和校验值，预取数据并挂载项目原有资源缓存，验证 get_testdata_file 命中；继续核对 handler/fixture。

验收：数据文件非空且哈希正确，全部 required通过；1439不得仅凭网络修复宣称20个MISSING恢复。

任务：`pydicom__pydicom-1241`、`pydicom__pydicom-1439`。

### missing_verifier_dependency（2项，P0）

证据与判断：PyVista4315/432 动态加载缺 libGL.so.1 或 libXrender.so.1。

修复：通过统一环境定义加入提供该 SONAME 的系统包及必要图形运行时，发布新镜像；核对 ldd 与实际 VTK import。

验收：加载器缺库消失、collect和 required全部通过；不能把后续API异常视为同一问题。

任务：`pyvista__pyvista-4315`、`pyvista__pyvista-432`。

### container_image_ownership_unmapped（8项，P1）

证据与判断：8项 Matplotlib 镜像解包出现 Lchown UID197609/GID197121，rootless subordinate ID 范围不足，尚未测试。

修复：核对容器引擎/rootless配置和subuid/subgid实际映射，准备覆盖镜像UID/GID的宿主机变更方案；维护窗口实施，或生成保留权限语义的合规镜像。

验收：镜像解包、create/start、清理均成功后执行gold。不得直接chown NAS、全局prune或重启其他工作负载。

任务：`matplotlib__matplotlib-20488`、`matplotlib__matplotlib-21568`、`matplotlib__matplotlib-22865`、`matplotlib__matplotlib-23111`、`matplotlib__matplotlib-24627`、`matplotlib__matplotlib-24637`、`matplotlib__matplotlib-26342`、`matplotlib__matplotlib-26466`。

### required_test_parameter_identity_mismatch（1项，P1）

证据与判断：Pydicom995 冻结绝对路径参数ID与官方 parser 的最后路径分量规范化不一致；44项未匹配。

修复：逐项比对原raw nodeid、冻结ID、源码参数及官方规范化；生成唯一无碰撞映射及版本化契约修订，用新run验证；25个skip另查。

验收：所有required身份唯一；skip理由满足契约；不按basename模糊合并或仅凭256passed判全通过。

任务：`pydicom__pydicom-995`。

### parser_incomplete（1项，P1）

证据与判断：Sphinx9711 tox/pytest有1passed与exit0，官方parser状态映射为空。

修复：对照tox实际命令/pytest输出配置，采用官方兼容结果输出或精确runner适配并绑定nodeID；保留原日志/官方结果。

验收：精确required test_needs_extensions取得可验证PASSED；额外普通失败反例仍正确为0。

任务：`sphinx-doc__sphinx-9711`。

### missing_verifier_tool（1项，P0）

证据与判断：Django15521 缺xgettext，63项测试全部skip；官方空分母导致reward1。

修复：统一镜像环境加入GNU gettext及相关工具，记录版本与PATH；新run保持原gold/test patch。

验收：63项不再因缺工具skip；全部required实际执行通过；原reward1仍保留为无效证据。

任务：`django__django-15521`。

### gold_dependency_api_failure（10项，P2）

证据与判断：10项PyVista在必需测试中发生VTK/PyVista接口异常；具体版本因果未证实。

修复：冻结base/test/gold与当前VTK版本，读任务声明支持范围，选择代表3675/4225/4853作只变依赖版本的对照；再扩展10项。

验收：对照证实接口差异消失，记录gold修改范围与版本来源；禁止直接宣称ABI不兼容或盲升VTK。

任务：`pyvista__pyvista-3675`、`pyvista__pyvista-3710`、`pyvista__pyvista-3747`、`pyvista__pyvista-4225`、`pyvista__pyvista-4311`、`pyvista__pyvista-4329`、`pyvista__pyvista-4406`、`pyvista__pyvista-4414`、`pyvista__pyvista-4417`、`pyvista__pyvista-4853`。

### gold_test_collection_failed（1项，P2）

证据与判断：PyVista4808 collection遇 vtkPoints 无dtype，required未开始；原因尚未确认。

修复：定位import/collection创建对象链，检查任务源码与依赖支持版本，用与API组一致的单变量对照；保留失败collection证据。

验收：collect成功且所有required执行通过；不得用清空required或跳过模块解决。

任务：`pyvista__pyvista-4808`。

### gold_native_process_crash（2项，P2）

证据与判断：PyVista3750/4648 原生Segmentation fault/139；环境、驱动、VTK或代码责任尚未证实。

修复：保留faulthandler/native栈、显示后端、VTK和动态库清单；对照受支持VTK及无显示环境策略，必要时同gold最小复现。

验收：原崩溃点通过，完整required运行；139不能当成timeout/OOM证据，不能将未确认原因写为图形库缺失。

任务：`pyvista__pyvista-3750`、`pyvista__pyvista-4648`。

### gold_runtime_recursion_failure（2项，P2）

证据与判断：SymPy13001 test_issue_4785、13146 test_evalf_bugs真实RecursionError；Python/依赖因果未知。

修复：比较该base支持的Python/mpmath与当前版本，记录recursionlimit和测试顺序，做单变量gold对照；不只凭旧代码+Python3.9下结论。

验收：支持版本消除递归且required全通过；若证实gold补丁本身缺陷则进入人工gold_patch_defect例外。

任务：`sympy__sympy-13001`、`sympy__sympy-13146`。

### gold_runtime_warning_failure（1项，P2）

证据与判断：SymPy13865 test_homogeneous_function在collections.Iterable产生DeprecationWarning异常，不是RecursionError。

修复：记录实际Python警告过滤器与测试配置，核对历史支持Python及警告行为，单变量对照；不全局屏蔽warnings。

验收：兼容策略有历史依据且required通过，未遮蔽其他有效警告/测试失败。

任务：`sympy__sympy-13865`。

### gold_evaluation_timeout（5项，P2）

证据与判断：Requests2193/2413、Sphinx9309/9467、Pydicom1458 超时，根因未明。

修复：按最后阶段/测试定位阻塞：外部请求、资源加载、collection或计算；新run增加阶段/内层命令回执，固定单任务对照。

验收：证明阻塞原因并修复；只有确认正常测试时长才调整有界timeout；137不自动视为timeout。

任务：`psf__requests-2193`、`psf__requests-2413`、`pydicom__pydicom-1458`、`sphinx-doc__sphinx-9309`、`sphinx-doc__sphinx-9467`。

## 执行顺序与统一约束

1. P0先补wheelhouse、固定NumPy/pytest兼容组合、系统库和gettext；每类先选代表任务用新run验证，再扩大该类。
2. P1处理外部资源、parser/ID契约及镜像映射。宿主机subuid/subgid或Docker重启须先形成影响范围和维护方案，避开现有训练；缓存、镜像、wheelhouse均发布新版本与receipt。
3. P2完成21项单变量因果对照，再决定环境修复或gold缺陷例外；未知原因状态仍为needs_investigation，不能通过改分类宣称已修复。
4. 每项修复保持frozen base、gold/test patch、required测试集合与评分阈值。新run通过正式dycon run/harness命令创建、准备、运行；不得修改历史run和官方site-packages。
5. 修复主因后重新核查所有F2P/P2P，保留并存故障；网络、nose、handler、ID和LaTeX问题可能同时存在。普通模型有效断言失败仍为0；无效结果永不补成0或1。
6. 新环境通过gold后加至少一个普通失败补丁反例，证明verifier能够区分真实失败；并核对真实内层返回码、身份映射、stdout/stderr回执、无skip掩盖和容器清理。
7. 全量验收报告同时给出官方原始率、有效率、VerifierError/Review、原因确认覆盖率、无效轨迹排除数。不根据筛选后的有效子集计算全量100%。
8. 耗时评测通过正式后台任务与完成触发回执执行。启动确认后不轮询；待后续明确状态请求读取一次完成回执。

## 每任务验收记录模板

记录 task ID、原run/physical attempt、原日志SHA256、reason、cause_confirmed、原环境镜像/依赖摘要、单变量变更、修复资源版本/receipt、新run/evaluation、F2P/P2P身份/数量、真实测试rc、skip、resolved、有效reward、清理回执及剩余故障。只有新评测完整通过才标记repaired；计划/分类通过不等于修复完成。

## 全部145项逐任务台账

证据JSON保留在本任务evidence/all-verifier-errors/，以下每行均链接本地完整审计记录；远程原始文件路径和哈希见记录内evidence字段。

| task | reason | 原始类别 | 原因状态 | 证据 |
|---|---|---|---|
| `matplotlib__matplotlib-20488` | `container_image_ownership_unmapped` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/matplotlib__matplotlib-20488.json) |
| `matplotlib__matplotlib-21568` | `container_image_ownership_unmapped` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/matplotlib__matplotlib-21568.json) |
| `matplotlib__matplotlib-22865` | `container_image_ownership_unmapped` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/matplotlib__matplotlib-22865.json) |
| `matplotlib__matplotlib-23111` | `container_image_ownership_unmapped` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/matplotlib__matplotlib-23111.json) |
| `matplotlib__matplotlib-24627` | `container_image_ownership_unmapped` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/matplotlib__matplotlib-24627.json) |
| `matplotlib__matplotlib-24637` | `container_image_ownership_unmapped` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/matplotlib__matplotlib-24637.json) |
| `matplotlib__matplotlib-26342` | `container_image_ownership_unmapped` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/matplotlib__matplotlib-26342.json) |
| `matplotlib__matplotlib-26466` | `container_image_ownership_unmapped` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/matplotlib__matplotlib-26466.json) |
| `pyvista__pyvista-3675` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-3675.json) |
| `pyvista__pyvista-3710` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-3710.json) |
| `pyvista__pyvista-3747` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-3747.json) |
| `pyvista__pyvista-4225` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4225.json) |
| `pyvista__pyvista-4311` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4311.json) |
| `pyvista__pyvista-4329` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4329.json) |
| `pyvista__pyvista-4406` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4406.json) |
| `pyvista__pyvista-4414` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4414.json) |
| `pyvista__pyvista-4417` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4417.json) |
| `pyvista__pyvista-4853` | `gold_dependency_api_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4853.json) |
| `psf__requests-2193` | `gold_evaluation_timeout` | infra | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2193.json) |
| `psf__requests-2413` | `gold_evaluation_timeout` | infra | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2413.json) |
| `pydicom__pydicom-1458` | `gold_evaluation_timeout` | infra | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1458.json) |
| `sphinx-doc__sphinx-9309` | `gold_evaluation_timeout` | infra | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-9309.json) |
| `sphinx-doc__sphinx-9467` | `gold_evaluation_timeout` | infra | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-9467.json) |
| `pyvista__pyvista-3750` | `gold_native_process_crash` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-3750.json) |
| `pyvista__pyvista-4648` | `gold_native_process_crash` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4648.json) |
| `sympy__sympy-13001` | `gold_runtime_recursion_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sympy__sympy-13001.json) |
| `sympy__sympy-13146` | `gold_runtime_recursion_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sympy__sympy-13146.json) |
| `sympy__sympy-13865` | `gold_runtime_warning_failure` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sympy__sympy-13865.json) |
| `pyvista__pyvista-4808` | `gold_test_collection_failed` | negative | needs_investigation | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4808.json) |
| `astropy__astropy-13803` | `missing_distribution` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/astropy__astropy-13803.json) |
| `astropy__astropy-14907` | `missing_distribution` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/astropy__astropy-14907.json) |
| `astropy__astropy-14938` | `missing_distribution` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/astropy__astropy-14938.json) |
| `astropy__astropy-14991` | `missing_distribution` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/astropy__astropy-14991.json) |
| `astropy__astropy-14995` | `missing_distribution` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/astropy__astropy-14995.json) |
| `pvlib__pvlib-python-718` | `missing_distribution` | infra | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-718.json) |
| `pyvista__pyvista-4315` | `missing_verifier_dependency` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-4315.json) |
| `pyvista__pyvista-432` | `missing_verifier_dependency` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pyvista__pyvista-432.json) |
| `django__django-15521` | `missing_verifier_tool` | positive | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/django__django-15521.json) |
| `sphinx-doc__sphinx-9711` | `parser_incomplete` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-9711.json) |
| `pydicom__pydicom-995` | `required_test_parameter_identity_mismatch` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-995.json) |
| `pvlib__pvlib-python-1026` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1026.json) |
| `pvlib__pvlib-python-1033` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1033.json) |
| `pvlib__pvlib-python-1048` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1048.json) |
| `pvlib__pvlib-python-1053` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1053.json) |
| `pvlib__pvlib-python-1072` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1072.json) |
| `pvlib__pvlib-python-1073` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1073.json) |
| `pvlib__pvlib-python-1089` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1089.json) |
| `pvlib__pvlib-python-1093` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1093.json) |
| `pvlib__pvlib-python-1138` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1138.json) |
| `pvlib__pvlib-python-1140` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1140.json) |
| `pvlib__pvlib-python-1154` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1154.json) |
| `pvlib__pvlib-python-1160` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1160.json) |
| `pvlib__pvlib-python-1165` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1165.json) |
| `pvlib__pvlib-python-1176` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1176.json) |
| `pvlib__pvlib-python-1181` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1181.json) |
| `pvlib__pvlib-python-1186` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1186.json) |
| `pvlib__pvlib-python-1191` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1191.json) |
| `pvlib__pvlib-python-1193` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1193.json) |
| `pvlib__pvlib-python-1213` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1213.json) |
| `pvlib__pvlib-python-1216` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1216.json) |
| `pvlib__pvlib-python-1224` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1224.json) |
| `pvlib__pvlib-python-1239` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1239.json) |
| `pvlib__pvlib-python-1273` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1273.json) |
| `pvlib__pvlib-python-1296` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1296.json) |
| `pvlib__pvlib-python-1349` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1349.json) |
| `pvlib__pvlib-python-1368` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1368.json) |
| `pvlib__pvlib-python-1395` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1395.json) |
| `pvlib__pvlib-python-1426` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1426.json) |
| `pvlib__pvlib-python-1448` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1448.json) |
| `pvlib__pvlib-python-1458` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1458.json) |
| `pvlib__pvlib-python-1469` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1469.json) |
| `pvlib__pvlib-python-1478` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1478.json) |
| `pvlib__pvlib-python-1480` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1480.json) |
| `pvlib__pvlib-python-1518` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1518.json) |
| `pvlib__pvlib-python-1589` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1589.json) |
| `pvlib__pvlib-python-1606` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1606.json) |
| `pvlib__pvlib-python-1616` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1616.json) |
| `pvlib__pvlib-python-1623` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1623.json) |
| `pvlib__pvlib-python-1653` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1653.json) |
| `pvlib__pvlib-python-1666` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1666.json) |
| `pvlib__pvlib-python-1682` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1682.json) |
| `pvlib__pvlib-python-1707` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1707.json) |
| `pvlib__pvlib-python-1737` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1737.json) |
| `pvlib__pvlib-python-1738` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1738.json) |
| `pvlib__pvlib-python-1739` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1739.json) |
| `pvlib__pvlib-python-1764` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1764.json) |
| `pvlib__pvlib-python-1782` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1782.json) |
| `pvlib__pvlib-python-1822` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1822.json) |
| `pvlib__pvlib-python-1832` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1832.json) |
| `pvlib__pvlib-python-1852` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1852.json) |
| `pvlib__pvlib-python-1854` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-1854.json) |
| `pvlib__pvlib-python-763` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-763.json) |
| `pvlib__pvlib-python-807` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-807.json) |
| `pvlib__pvlib-python-823` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-823.json) |
| `pvlib__pvlib-python-907` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-907.json) |
| `pvlib__pvlib-python-980` | `runtime_dependency_incompatible` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pvlib__pvlib-python-980.json) |
| `pydicom__pydicom-1031` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1031.json) |
| `pydicom__pydicom-1048` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1048.json) |
| `pydicom__pydicom-1069` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1069.json) |
| `pydicom__pydicom-1076` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1076.json) |
| `pydicom__pydicom-1090` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1090.json) |
| `pydicom__pydicom-1139` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1139.json) |
| `pydicom__pydicom-1194` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1194.json) |
| `pydicom__pydicom-1236` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1236.json) |
| `pydicom__pydicom-1334` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1334.json) |
| `pydicom__pydicom-1365` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1365.json) |
| `pydicom__pydicom-1375` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1375.json) |
| `pydicom__pydicom-1413` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1413.json) |
| `pydicom__pydicom-1416` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1416.json) |
| `pydicom__pydicom-1428` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1428.json) |
| `pydicom__pydicom-1539` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1539.json) |
| `pydicom__pydicom-1555` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1555.json) |
| `pydicom__pydicom-1598` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1598.json) |
| `pydicom__pydicom-1608` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1608.json) |
| `pydicom__pydicom-1633` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1633.json) |
| `pydicom__pydicom-1674` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1674.json) |
| `pydicom__pydicom-1720` | `test_framework_nose_compatibility` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1720.json) |
| `pydicom__pydicom-1241` | `test_resource_missing` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1241.json) |
| `pydicom__pydicom-1439` | `test_resource_missing` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/pydicom__pydicom-1439.json) |
| `mwaskom__seaborn-2457` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/mwaskom__seaborn-2457.json) |
| `psf__requests-1635` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1635.json) |
| `psf__requests-1724` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1724.json) |
| `psf__requests-1733` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1733.json) |
| `psf__requests-1766` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1766.json) |
| `psf__requests-1768` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1768.json) |
| `psf__requests-1776` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1776.json) |
| `psf__requests-1921` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1921.json) |
| `psf__requests-1944` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1944.json) |
| `psf__requests-1962` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1962.json) |
| `psf__requests-1963` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-1963.json) |
| `psf__requests-2148` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2148.json) |
| `psf__requests-2153` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2153.json) |
| `psf__requests-2317` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2317.json) |
| `psf__requests-2617` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2617.json) |
| `psf__requests-2674` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2674.json) |
| `psf__requests-2821` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2821.json) |
| `psf__requests-2931` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-2931.json) |
| `psf__requests-5414` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/psf__requests-5414.json) |
| `scikit-learn__scikit-learn-13620` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/scikit-learn__scikit-learn-13620.json) |
| `sphinx-doc__sphinx-10435` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-10435.json) |
| `sphinx-doc__sphinx-11266` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-11266.json) |
| `sphinx-doc__sphinx-7985` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-7985.json) |
| `sphinx-doc__sphinx-8269` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-8269.json) |
| `sphinx-doc__sphinx-8475` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-8475.json) |
| `sphinx-doc__sphinx-8509` | `test_resource_unavailable` | negative | confirmed | [记录](../temp/swebench-verifier-settlement-20261117/evidence/all-verifier-errors/sphinx-doc__sphinx-8509.json) |
