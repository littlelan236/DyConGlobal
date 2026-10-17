# FP16 versus BF16 for actual rollout fidelity

## Scope

This assessment uses Qwen3.5-9B, the model used by 972/1,104 current catalog rollouts. It uses 12 real coding-agent prefixes from the migrated training trajectories (baseline and steering-run histories), with lengths 1,400–8,212 tokens. No steering intervention is applied during the benchmark. It does not execute SWE-bench tools or measure resolution. Each latency round generates exactly 512 tokens per request, so the two dtypes have the same requested output work.

The service is TP=4 with APC, `max_num_seqs=12`, `max_num_batched_tokens=8192`, and native graph selection (`FULL_AND_PIECEWISE`). Three rounds are measured at concurrency 1, 4 and 12 after warm-up. Separate TP=1 eager engines replay the exact tokens emitted by each dtype to compare online capture states with offline replay.

## Latency

| Concurrent requests | BF16 model seconds | FP16 model seconds | FP16 change |
|---:|---:|---:|---:|
| 1 | 3.731 | 3.770 | +1.1% |
| 4 | 5.214 | 5.075 | −2.7% |
| 12 | 6.640 | 6.454 | −2.8% |

The raw model-serving difference is within a few percent and changes sign with concurrency. Historical trajectory timestamps from 48 rollouts in `baseline-qwen35-9b-forward-apc-coord-gpu4-7-w12-20261012` estimate model requests account for 43.6%–51.2% of agent-loop wall time (4,835 requests; integer-second start times bound the estimate). If that fraction scaled exactly like this microbenchmark, the corresponding loop-level effect would be approximately +0.5% at concurrency 1 and −1.2% to −1.4% at concurrency 4/12. This is only a first-order conditional estimate: it excludes container startup and does not model queueing interactions or changes in generated behavior.

Startup was 102.5 s for BF16 and 108.4 s for FP16 in this first-run pair. Compilation/cache state differed, so this is not evidence of a stable startup penalty. It should be amortized across a long rollout cohort and measured with a fresh, repeated, randomized-order benchmark before making scheduling decisions.

The Qwen GDN implementation uses a fused CUDA decode path for BF16 and a Triton fallback for FP16. Despite that kernel difference, the measured steady-state generation throughput was close; FP16 did not incur a material rollout-time penalty in this test.

## Replay accuracy

The online reference contains four `<think>` states, four `</think>` states and twelve selected intermediate generated-token states. The comparison replays the exact token prefix through the existing eager extraction adapter at decoder sites 0, 15 and final norm. Results below use the warm replay; cold replay has the same conclusion.

| Anchor group | BF16 median relative L2 | FP16 median relative L2 | BF16 max | FP16 max |
|---|---:|---:|---:|---:|
| `<think>` | 1.253% | 0.183% | 4.224% | 0.552% |
| `</think>` | 1.100% | 0.198% | 4.845% | 0.742% |
| All selected states | 1.146% | 0.141% | 6.650% | 1.002% |

For the actual `<think>` and `</think>` positions used by the difficulty estimator, FP16 reduced the median replay error by about 5.5–6.8× and the maximum error by about 6.5–7.7× in this sample. This supports using FP16 when the objective is a stable offline replay of an FP16 rollout.

The comparison is conditional on the dtype's own online continuation. BF16 and FP16 do not always sample the same tokens: in this run one of four free generations had a different length (143 versus 190 tokens). Therefore this experiment measures replay fidelity conditional on each dtype's actual trajectory; it does not claim that FP16 preserves task behavior or SWE-bench resolution.

An exact-token-prefix matched subset also confirms the direction of improvement. All four opening-marker prefixes and two closing-marker prefixes match exactly across dtypes. On the matched closing prefixes, the median relative error decreases from 1.089% to 0.157%; on all four matched opening prefixes, it decreases from 1.253% to 0.183%. Their online reference tensors are each evaluated in their own dtype.

As a negative control, FP16 replay of existing BF16 online states did not show the same benefit: at `</think>` the max relative error was 3.73%, versus 3.60% for same-dtype BF16 TP=4 warm replay. Switching only the replay dtype therefore does not recreate an FP16 rollout or repair historical BF16 states.

## Recommendation

For a future **FP16 rollout plus FP16 latent capture** experiment, the expected steady-state rollout cost is approximately neutral at the tested concurrency, while replay fidelity improves substantially at the reasoning markers. FP16 is therefore a reasonable candidate for a controlled new rollout cohort.

It should be a separate model/config identity and separate catalog run. Existing BF16 rollouts and their products remain unchanged. Do not convert existing BF16 trajectories into FP16 products and call them rollout-faithful: the online continuation and recurrent/KV state are already BF16.

Before changing the production rollout default, run a paired A/B cohort with identical tasks, seed policy, TP/APC/graph settings and separate GPU groups. Measure startup amortization, prompt/decode throughput, total agent-loop time, token counts, termination reasons, resolved/submitted rates and tool-call behavior. The current test establishes likely replay precision and a small model-time effect; it does not establish a task-level quality improvement.

Raw results: `hidden/evaluation/outputs/dtype-assessment-20261017/`. Summary JSON: `docs/rollout-fp16-assessment-20261017.json`.
