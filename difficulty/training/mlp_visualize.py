"""Visualize per-epoch MLP losses and regression quality metrics."""

import json
from pathlib import Path


def load_history(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def plot_history(history_or_path, output_path, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    history = load_history(history_or_path) if isinstance(history_or_path, (str, Path)) else history_or_path
    if not history:
        raise ValueError('MLP history is empty')
    required = {'epoch', 'train_loss', 'eval_loss', 'test_loss',
                'train_r2', 'validation_r2', 'test_r2'}
    if not required <= set(history[0]):
        raise ValueError(f'MLP history lacks visualization fields: {sorted(required - set(history[0]))}')
    epochs = [row['epoch'] for row in history]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (loss_axis, quality_axis, rate_axis) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for role, key in [('train', 'train_loss'), ('validation', 'eval_loss'), ('test', 'test_loss')]:
        loss_axis.plot(epochs, [row[key] for row in history], label=role)
    loss_axis.set_ylabel('MSE (log1p target)')
    loss_axis.set_title(target)
    loss_axis.legend()
    for role, key in [('train', 'train_r2'), ('validation', 'validation_r2'), ('test', 'test_r2')]:
        quality_axis.plot(epochs, [row[key] for row in history], label=role)
    quality_axis.set_xlabel('Epoch')
    quality_axis.set_ylabel('R2 (regression quality)')
    quality_axis.legend()
    if 'learning_rate' in history[0]:
        rate_axis.plot(epochs, [row['learning_rate'] for row in history], label='learning rate')
        rate_axis.set_yscale('log')
        rate_axis.set_ylabel('Learning rate')
        rate_axis.legend()
    else:
        rate_axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def plot_attempt(attempt_dir):
    attempt_dir = Path(attempt_dir)
    return [plot_history(history, attempt_dir / 'plots' / (history.stem + '-loss-r2.png'), history.stem)
            for history in sorted((attempt_dir / 'history').glob('*.jsonl'))]
