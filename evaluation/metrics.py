from __future__ import annotations

import numpy as np
from scipy.stats import binom


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not isinstance(y_true, np.ndarray) or y_true.ndim != 1:
        raise ValueError(f"Expected y_true to have shape (N,); got {y_true.shape}")
    if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 1:
        raise ValueError(f"Expected y_pred to have shape (N,); got {y_pred.shape}")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred must have the same length; got {y_true.shape[0]} vs {y_pred.shape[0]}")
    return float(np.mean(y_true == y_pred))


def calc_guess_accuracy(y: np.ndarray) -> float:
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError(f"Expected y to have shape (N,); got {y.shape}")
    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("Need at least 2 classes to define guess accuracy.")
    return 1.0 / n_classes


def calc_ucl_accuracy(n_trials: int, alpha: float = 0.05, guess_accuracy: float = 0.5) -> float:
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive int; got {n_trials!r}")
    if not isinstance(alpha, float) or not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be a finite float in (0, 1); got {alpha!r}")
    if not isinstance(guess_accuracy, float) or not np.isfinite(guess_accuracy) or guess_accuracy < 0 or guess_accuracy > 1:
        raise ValueError(f"guess_accuracy must be a finite float in [0, 1]; got {guess_accuracy!r}")
    k_star = int(binom.ppf(1 - alpha, n_trials, guess_accuracy))
    return k_star / n_trials
