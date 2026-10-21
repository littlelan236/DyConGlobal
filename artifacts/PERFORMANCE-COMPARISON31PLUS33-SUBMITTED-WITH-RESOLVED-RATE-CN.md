# Baseline / Parallel 对比：共同 Submitted（31 + 33 结果）

只保留 Baseline 与 Parallel 最终轨迹均 `Submitted` 的 **35 个任务**，并排除 Baseline 存在 reasoning 长循环的任务对。`resolved rate` 以这 35 个共同 Submitted 任务为分母。

| 指标 | Baseline | Parallel（B=4） | 相对变化 |
| --- | ---: | ---: | ---: |
| resolved rate | 18/35 (51.43%) | 14/35 (40.00%) | -22.22% |
| 实际请求数（全分支） | 2,124 | 7,496 | +252.92% |
| prompt（全分支） | 50,644,987 | 176,493,936 | +248.49% |
| completion（全分支） | 457,039 | 2,505,328 | +448.17% |
| total tokens（全分支） | 51,102,026 | 178,999,264 | +250.28% |
| 选择分支 calls | 2,124 | 1,874 | -11.77% |
| 选择分支 prompt | 50,644,987 | 44,123,484 | -12.88% |
| 选择分支 completion | 457,039 | 454,152 | -0.63% |
| 选择分支 total tokens | 51,102,026 | 44,577,636 | -12.77% |


`相对变化=(Parallel-Baseline)/Baseline`；`resolved rate` 的相对变化按 resolved 数量计算。前四个 token 行累计 Parallel B=4 全部候选，后四行只累计最终选择分支；Baseline 为单分支。

## Resolved 状态

| Harness 状态 | Baseline | Parallel |
| --- | ---: | ---: |
| resolved=true | 18/35 (51.43%) | 14/35 (40.00%) |
| resolved=false | 12/35 (34.29%) | 12/35 (34.29%) |
| resolved=null | 5/35 (14.29%) | 9/35 (25.71%) |
| 完成判分 | 30/35 (85.71%) | 26/35 (74.29%) |


Harness 未完成或基础设施失败的结果保留为 `resolved=null`，不当作 resolved=false。后一批 33 个 Parallel 轨迹已全部通过统一 harness 入口处理：20 个完成判分，其中 11 个 resolved；另 13 个为 null。之前缺少有效判分的 5 个 Baseline pending 也已补评，全部因 patch apply failure 保留为 null。

## 分批统计

### 31-task run（12 个共同 Submitted 任务）

| 指标 | Baseline | Parallel（B=4） | 相对变化 |
| --- | ---: | ---: | ---: |
| resolved rate | 4/12 (33.33%) | 5/12 (41.67%) | +25.00% |
| 实际请求数（全分支） | 986 | 3,124 | +216.84% |
| prompt（全分支） | 27,259,247 | 79,601,276 | +192.02% |
| completion（全分支） | 206,344 | 1,140,837 | +452.88% |
| total tokens（全分支） | 27,465,591 | 80,742,113 | +193.98% |
| 选择分支 calls | 986 | 781 | -20.79% |
| 选择分支 prompt | 27,259,247 | 19,900,319 | -27.00% |
| 选择分支 completion | 206,344 | 207,918 | +0.76% |
| 选择分支 total tokens | 27,465,591 | 20,108,237 | -26.79% |

### 后续 33 个结果（23 个共同 Submitted 任务）

| 指标 | Baseline | Parallel（B=4） | 相对变化 |
| --- | ---: | ---: | ---: |
| resolved rate | 14/23 (60.87%) | 9/23 (39.13%) | -35.71% |
| 实际请求数（全分支） | 1,138 | 4,372 | +284.18% |
| prompt（全分支） | 23,385,740 | 96,892,660 | +314.32% |
| completion（全分支） | 250,695 | 1,364,491 | +444.28% |
| total tokens（全分支） | 23,636,435 | 98,257,151 | +315.70% |
| 选择分支 calls | 1,138 | 1,093 | -3.95% |
| 选择分支 prompt | 23,385,740 | 24,223,165 | +3.58% |
| 选择分支 completion | 250,695 | 246,234 | -1.78% |
| 选择分支 total tokens | 23,636,435 | 24,469,399 | +3.52% |


## 逐任务

| 任务 | 批次 | B resolved | P resolved | B calls | P 选中 calls | B 选中 prompt | P 选中 prompt | B 选中 completion | P 选中 completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `astropy__astropy-13977` | 33 | False | None | 39 | 87 | 598,191 | 3,181,954 | 6,861 | 30,501 |
| `astropy__astropy-8707` | 31 | True | True | 63 | 88 | 1,279,355 | 2,011,731 | 12,267 | 15,385 |
| `django__django-10880` | 33 | True | None | 87 | 88 | 2,123,856 | 2,974,147 | 12,884 | 19,147 |
| `django__django-10999` | 31 | False | False | 28 | 20 | 324,654 | 192,078 | 8,652 | 5,699 |
| `django__django-11099` | 33 | True | None | 12 | 22 | 45,809 | 144,931 | 2,825 | 4,018 |
| `django__django-14580` | 33 | True | True | 72 | 59 | 1,457,846 | 1,349,209 | 11,351 | 9,707 |
| `django__django-15368` | 33 | True | True | 42 | 54 | 640,822 | 1,214,506 | 7,841 | 8,222 |
| `django__django-16315` | 33 | False | False | 83 | 111 | 2,353,388 | 3,290,446 | 12,259 | 14,504 |
| `matplotlib__matplotlib-13989` | 33 | True | None | 34 | 35 | 481,885 | 437,016 | 8,045 | 6,902 |
| `matplotlib__matplotlib-14623` | 31 | True | True | 93 | 88 | 2,937,833 | 2,269,673 | 20,915 | 19,437 |
| `matplotlib__matplotlib-24870` | 33 | False | False | 54 | 56 | 1,457,085 | 1,157,456 | 25,481 | 14,299 |
| `matplotlib__matplotlib-25479` | 33 | None | False | 70 | 87 | 2,119,349 | 2,681,080 | 17,157 | 16,882 |
| `mwaskom__seaborn-3187` | 31 | False | False | 136 | 131 | 4,819,112 | 5,546,108 | 39,694 | 44,249 |
| `pydata__xarray-4075` | 33 | True | None | 34 | 26 | 576,747 | 367,627 | 12,539 | 9,797 |
| `pydata__xarray-4629` | 33 | True | True | 16 | 12 | 134,005 | 80,376 | 7,771 | 6,397 |
| `pydata__xarray-6992` | 33 | False | False | 33 | 24 | 340,259 | 197,761 | 8,243 | 6,294 |
| `pylint-dev__pylint-4551` | 31 | None | False | 164 | 98 | 6,594,827 | 3,190,371 | 35,652 | 23,682 |
| `pylint-dev__pylint-4604` | 31 | False | False | 152 | 58 | 4,570,283 | 808,351 | 17,743 | 8,629 |
| `pylint-dev__pylint-6903` | 33 | None | True | 39 | 23 | 689,090 | 215,864 | 17,751 | 6,606 |
| `pylint-dev__pylint-7277` | 33 | True | True | 51 | 17 | 860,020 | 163,089 | 11,902 | 3,968 |
| `pytest-dev__pytest-10081` | 31 | True | True | 49 | 34 | 733,531 | 384,358 | 5,777 | 4,720 |
| `pytest-dev__pytest-10356` | 31 | None | False | 70 | 87 | 1,875,639 | 2,220,991 | 20,691 | 11,179 |
| `pytest-dev__pytest-5631` | 33 | True | True | 30 | 30 | 297,432 | 308,007 | 5,810 | 4,209 |
| `pytest-dev__pytest-7432` | 31 | True | True | 44 | 37 | 752,453 | 611,486 | 10,597 | 9,223 |
| `pytest-dev__pytest-8399` | 33 | True | True | 44 | 64 | 526,809 | 986,847 | 3,916 | 5,366 |
| `scikit-learn__scikit-learn-12682` | 33 | False | False | 106 | 59 | 3,096,835 | 1,152,790 | 15,176 | 11,605 |
| `scikit-learn__scikit-learn-13142` | 33 | True | None | 24 | 46 | 272,640 | 931,740 | 4,208 | 14,532 |
| `scikit-learn__scikit-learn-14496` | 33 | None | True | 41 | 28 | 477,152 | 259,884 | 9,361 | 5,478 |
| `scikit-learn__scikit-learn-14629` | 31 | False | False | 65 | 59 | 1,110,360 | 1,076,808 | 11,385 | 19,411 |
| `scikit-learn__scikit-learn-25102` | 33 | False | False | 79 | 50 | 2,486,372 | 1,573,213 | 26,727 | 29,555 |
| `scikit-learn__scikit-learn-26323` | 33 | True | None | 43 | 30 | 722,737 | 413,032 | 9,676 | 5,340 |
| `sphinx-doc__sphinx-8595` | 33 | True | True | 71 | 50 | 1,234,217 | 698,248 | 7,113 | 6,229 |
| `sympy__sympy-12481` | 33 | True | None | 34 | 35 | 393,194 | 443,942 | 5,798 | 6,676 |
| `sympy__sympy-18199` | 31 | False | None | 71 | 50 | 1,587,990 | 1,287,607 | 13,931 | 41,366 |
| `sympy__sympy-19346` | 31 | False | True | 51 | 31 | 673,210 | 300,757 | 9,040 | 4,938 |


Baseline 长循环排除沿用上一轮同一检测结果。完整逐任务 token、resolved 状态和 harness provenance 见 `comparison31plus33-submitted-no-baseline-loops-with-resolved-rate-20261020.json`。本次未重新 rollout。
