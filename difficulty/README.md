# Remaining-workload estimators

`difficulty.models.estimators` provides three estimators with input width inferred from the loaded hidden tensor:

- `ridge`: standardized closed-form Ridge regression.
- `ffn_mlp`: LayerNorm -> two Linear-GELU-Dropout blocks -> scalar output.
- `ridge_plus_ffn`: Ridge prediction plus an FFN residual. The residual's final projection is zero-initialized, so its initial output is exactly zero and the combined model initially equals the fitted Ridge model.

Hidden artifacts have shape `[samples, selected_layers, hidden_size]`; the estimator flattens this to `[samples, selected_layers * hidden_size]`. No number or identity of layers is hard-coded.

Supported targets:

- `remaining_reasoning_tokens_log1p`: `log1p(remaining_reasoning_tokens)`
- `remaining_steps_log1p`: `log1p(remaining_steps)`
- `remaining_steps_linear`: raw remaining steps

`difficulty.training.train.load_training_data` joins a `dataset-index` with enrichment labels and hidden tensors, drops unavailable labels, and splits by `rollout_id` to prevent steps from one rollout leaking across train/validation.
