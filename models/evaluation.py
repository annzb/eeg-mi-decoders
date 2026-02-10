from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import binom

from models import metrics
from models.base import Model


@dataclass
class SubjectEvalResult:
    mean: float
    std: float
    n_repeats: int
    ucl_accuracy: float


def guess_accuracy_uniform(y: np.ndarray) -> float:
    classes = np.unique(y)
    k = len(classes)
    if k < 2:
        raise ValueError("Need at least 2 classes to define guess accuracy.")
    return 1.0 / k


def ucl_accuracy(n_trials: int, alpha: float = 0.05, guess_accuracy: float = 0.5) -> float:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not (0 < guess_accuracy < 1):
        raise ValueError("guess_accuracy must be in (0, 1)")
    k_star = int(binom.ppf(1 - alpha, n_trials, guess_accuracy))
    return k_star / n_trials


def eval_subject_repeated_cv(model: Model, X: np.ndarray, y: np.ndarray, cv, metric: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy) -> SubjectEvalResult:
    accs = []
    n_trials = len(X)
    for train_idx, test_idx in cv.split(y):
        m = model.clone()
        m.fit(X[train_idx], y[train_idx])
        accs.append(metric(y[test_idx], m.predict(X[test_idx])))
    accs = np.asarray(accs, dtype=float)
    guess_accuracy = guess_accuracy_uniform(y)
    ucl = ucl_accuracy(n_trials, alpha=0.05, guess_accuracy=guess_accuracy)
    return SubjectEvalResult(
        mean=float(accs.mean()),
        std=float(accs.std(ddof=0)),
        n_repeats=int(accs.size),
        ucl_accuracy=ucl
    )


def evaluate_all_subjects(model: Model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv) -> Dict[int, SubjectEvalResult]:
    subj_ids = np.unique(groups)
    results: Dict[int, SubjectEvalResult] = {}
    for sid in subj_ids:
        mask = (groups == sid)
        res = eval_subject_repeated_cv(model, X[mask], y[mask], cv=cv)
        results[int(sid)] = res
    return results


def summarize_subject_means(subject_means: Iterable[float]) -> dict:
    a = np.asarray(list(subject_means), dtype=float)
    if a.size == 0:
        raise ValueError("Need at least one subject accuracy.")
    mean = float(a.mean())
    std = float(a.std(ddof=0))          # "± std across subjects" (paper style)
    sem = float(std / np.sqrt(a.size))
    ci95_half = float(1.96 * sem)
    return {"n_subjects": int(a.size), "mean": mean, "std": std, "sem": sem, "ci95_half": ci95_half}



# import numpy as np
# from scipy.stats import binom

# from models import ds1_baseline


# def evaluate_all_subjects(X, Y, groups, n_repeats=120, seed=0):
#     subj_ids = np.unique(groups)
#     results = {}
#     for sid in subj_ids:
#         mask = (groups == sid)
#         Xs, ys = X[mask], Y[mask]
#         mean_acc, std_acc = ds1_baseline.eval_subject_csp_lda(Xs, ys, n_repeats=n_repeats, seed=seed)
#         results[int(sid)] = (mean_acc, std_acc)
#     all_means = np.array([v[0] for v in results.values()], dtype=float)
#     print("\nOverall (mean across subjects): "
#           f"{all_means.mean()*100:.2f}% ± {all_means.std()*100:.2f}% (std across subjects)")
#     return results


# def summarize_eval(acc_per_subject):
#     a = np.asarray(list(acc_per_subject), dtype=float)
#     if a.size == 0:
#         raise ValueError("Need at least one subject accuracy.")
#     mean = float(a.mean())
#     std = float(a.std(ddof=0))       
#     sem = float(std / np.sqrt(a.size))
#     ci95_half = float(1.96 * sem)
#     return {
#         "n_subjects": int(a.size),
#         "mean": mean,
#         "std": std,
#         "sem": sem,
#         "ci95_half": ci95_half,
#     }


# def upper_confidence_limit_accuracy(n_trials: int, alpha: float = 0.05, p: float = 0.5) -> float:
#     if n_trials <= 0:
#         raise ValueError("n_trials must be positive")
#     if not (0 < alpha < 1):
#         raise ValueError("alpha must be in (0, 1)")
#     if not (0 < p < 1):
#         raise ValueError("p must be in (0, 1)")
#     k_star = int(binom.ppf(1 - alpha, n_trials, p))
#     return k_star / n_trials
