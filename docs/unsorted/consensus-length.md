# Winner-vote length and estimator analysis

From the project root:

```bash
PYTHONPATH=. .venvs/analysis/bin/dycon difficulty-evaluate-consensus-length RUN_ID EVALUATION_ID
```

By default the command uses the project's eligible final-trajectory selection,
without filtering on success. Repeat `--rollout-id` to inspect explicit
trajectories, including intermediate attempts when eligible. The command reads
SHA-bound canonical trajectories and their embedded decisions. It does not replay
generation or read hidden states. It checks the recorded estimator identity and
expects a homogeneous source cohort. The source run is unchanged.

Each step uses only the recorded retained members of its winning action vote.
The estimator rule minimizes `(score, branch_id)`; the length-first rule minimizes
`(completion_tokens, score, branch_id)`. `same_branch` compares these two rules,
and `optima_overlap` checks whether any branch belongs to both sets of optima
before tie-breaking. `token_excess` is the estimator rule's completion length
minus the minimum; `actual_token_regret` uses the actual representative instead.
These are within-step token differences from historical candidates, not rollout
token savings or downstream success estimates. Positive Spearman and Kendall
tau-b indicate that lower scores tend to accompany shorter branches.

The primary summary uses multi-member groups. A metric is undefined when its
required evidence is missing from any member; constant ranks and a zero minimum
length make correlations and relative token excess undefined, respectively. The
step CSV retains singleton and no-winner decisions. Mean metrics weight steps
within a trajectory, trajectories within a task, then tasks equally. Bootstrap
intervals resample tasks or known repositories; `--bootstrap-samples 0` disables
resampling. Token excess median and P90 describe the unweighted step distribution.

The report also includes two out-of-sample R^2 values from length-only linear
regression on complete multi-member groups. The raw model fits an intercept and
predicts scores in their stored units; the within-step model regresses score
deviations from the step mean on length deviations from the step mean. Whole
tasks are assigned to up to five folds using `--seed`. Each task has equal
weight, divided equally across its trajectories, their steps, and their members.
R^2 pools held-out squared errors before comparing them with the weighted mean
score (raw) or zero deviation (within-step). A negative result means the model
has more squared error than its baseline; zero target variance or fewer than
two tasks yields an unavailable result.

## Outputs and reproducibility

Results and source/implementation receipts are written to the configured
`difficulty/evaluations/EVALUATION_ID` directory. Use a new ID for each analysis.

| File | Contents |
|---|---|
| `report.md`, `summary.json` | Coverage, task-weighted metrics and bootstrap intervals, token excess distribution, and held-out R^2 under `r2`. |
| `steps.csv`, `branches.csv` | One row per canonical consensus decision and one row per winning-group member, respectively. |
| `trajectories.json` | Per-trajectory counts of decision statuses and termination reason. |
| `selection.json`, `receipt.json` | Source rollout IDs and trajectory hashes, source config binding, and output and implementation checksums. |

To repeat an exact cohort, take the `sources[].rollout_id` entries from its
`selection.json` and pass each as `--rollout-id`. Keep the original seed
(default `42`) to reproduce the fold assignment. The default selection can
change as new trajectories are accepted. Compare the source trajectory and
frozen configuration hashes when checking that a replay used the same inputs.

## Reference evaluations (2026-11-20)

| Source run | Historical representative rule | Tasks / complete multi-member steps | Same branch | Raw score R^2 | Within-step R^2 | Evaluation ID |
|---|---|---:|---:|---:|---:|---|
| `verified80-consensus32-resolved106plus331-gpu03-tp4-w4-s64-20261027` | Lowest estimator score | 69 / 2,735 | 32.5% | -0.0432 | +0.0058 | `consensus-length-verified80-r2-20261120` |
| `skillsbench-new42-test-consensus32-fixed05-gpu3-w4-s64-capture-20261113` | `min_step_tokens` | 25 / 3,601 | 36.2% | -0.0127 | -0.0019 | `consensus-length-skillsbench-r2-20261120` |

Both evaluations used five task folds. Length alone performed below the mean
baseline for raw scores in both runs. Within a step, it reduced squared error
by about 0.58% versus zero deviation in Verified80 and was slightly worse than
that baseline in SkillsBench. SkillsBench's actual representative achieved
zero token regret on all comparable steps, as expected from its selection
rule. These observations describe historical winning groups; they do not
measure the impact of changing selection rules on eventual task success.

## Visualizations

The two primary plots can be generated from existing evaluation artifacts:

```bash
PYTHONPATH=. .venvs/analysis/bin/dycon difficulty-plot-consensus-length \
  consensus-length-viz-20261120 \
  --evaluation-id consensus-length-verified80-r2-20261120 \
  --evaluation-id consensus-length-skillsbench-r2-20261120
```

The **rank heatmap** puts every member of a multi-branch winning group at
(`completion-token rank`, `estimator-score rank`), where zero is best. The
dashed diagonal is perfect ranking agreement. Tasks are equally weighted, then
rollouts, complete multi-member steps, and members within each group. Both run
panels share the same color scale. Average ranks handle ties; constant columns
sit at 0.5. The raw rank grid is discrete, particularly for small groups. The
**token-excess ECDF** uses `estimator-best tokens - shortest tokens`; its value at
`x` is the task-balanced fraction of comparable decision steps whose excess is
at most `x`. Its zero mass measures equal length, which can exceed the fraction
selecting the same branch when shortest branches are tied. One panel shows
0-100 tokens on a linear scale; another uses symlog to show the entire tail.
Singleton and incomplete groups are excluded from both plots. Inputs are read
from the evaluation's CSV files and checked against its completed receipt; no
rollout or hidden replay is required. PNG, SVG, source matrices, weighted ECDF
values, and a checksum receipt are written under
`difficulty/evaluations/<plot-id>/`. Open `index.html` for the self-contained
Chinese report; it embeds both SVGs and links to the downloadable figures.
