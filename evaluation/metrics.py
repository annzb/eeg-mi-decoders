from __future__ import annotations

import numpy as np
from scipy.stats import binom


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def calc_guess_accuracy(y: np.ndarray) -> float:
    classes = np.unique(y); k = len(classes)
    if k < 2: raise ValueError("Need at least 2 classes to define guess accuracy.")
    return 1.0 / k


def calc_ucl_accuracy(n_trials: int, alpha: float = 0.05, guess_accuracy: float = 0.5) -> float:
    if n_trials <= 0: raise ValueError("n_trials must be positive")
    if not (0 < alpha < 1): raise ValueError("alpha must be in (0, 1)")
    if not (0 < guess_accuracy < 1): raise ValueError("guess_accuracy must be in (0, 1)")
    k_star = int(binom.ppf(1 - alpha, n_trials, guess_accuracy))
    return k_star / n_trials
