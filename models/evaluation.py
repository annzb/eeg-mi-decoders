from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import binom

from models import metrics
from models.model import Model


@dataclass
class SubjectEvalResult:
    mean: float
    std: float
    n_repeats: int
    ucl_accuracy: float


def summarize_subject_means(subject_means: Iterable[float]) -> dict:
    a = np.asarray(list(subject_means), dtype=float)
    if a.size == 0:
        raise ValueError("Need at least one subject accuracy.")
    mean = float(a.mean())
    std = float(a.std(ddof=0))          # "± std across subjects" (paper style)
    sem = float(std / np.sqrt(a.size))
    ci95_half = float(1.96 * sem)
    return {"n_subjects": int(a.size), "mean": mean, "std": std, "sem": sem, "ci95_half": ci95_half}
