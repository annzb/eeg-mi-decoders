from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


class Score:
    def __init__(self, acc_means: Sequence[float]):
        if not hasattr(acc_means, "__len__") or len(acc_means) == 0:
            raise ValueError(f"acc_means must be a non-empty sequence; got {type(acc_means)}")
        if not all(isinstance(x, (float, int)) and np.isfinite(x) and x >= 0 and x <= 1 for x in acc_means):
            raise ValueError(f"acc_means must be a non-empty sequence of finite non-negative floats in [0, 1]; got {acc_means}")
        acc_means = np.asarray(acc_means, dtype=float)
        self._acc_mean = float(np.mean(acc_means))
        self._acc_std = float(np.std(acc_means, ddof=0))
        self._acc_sem = float(np.std(acc_means, ddof=0) / np.sqrt(acc_means.size))
        self._acc_ci95_half = float(1.96 * np.std(acc_means, ddof=0) / np.sqrt(acc_means.size))

    def acc_mean(self) -> float:
        return self._acc_mean

    def acc_std(self) -> float:
        return self._acc_std

    def acc_sem(self) -> float:
        return self._acc_sem

    def acc_ci95_half(self) -> float:
        return self._acc_ci95_half

    def __repr__(self) -> str:
        return f"{self._acc_mean:.3f} ± {self._acc_std:.3f}, sem={self._acc_sem:.3f}, 95% CI={self._acc_ci95_half:.3f}"


@dataclass(frozen=True, slots=True)
class SubjectEvalResult:
    guess_accuracy: float
    ucl_accuracy: float
    train: Optional[Score] = None
    val: Optional[Score] = None
    test: Optional[Score] = None

    def __post_init__(self):
        if not isinstance(self.guess_accuracy, float) or not np.isfinite(self.guess_accuracy) or self.guess_accuracy < 0 or self.guess_accuracy > 1:
            raise ValueError(f"guess_accuracy must be a finite float in [0, 1]; got {self.guess_accuracy!r}")
        if not isinstance(self.ucl_accuracy, float) or not np.isfinite(self.ucl_accuracy) or self.ucl_accuracy < 0 or self.ucl_accuracy > 1:
            raise ValueError(f"ucl_accuracy must be a finite float in [0, 1]; got {self.ucl_accuracy!r}")
        for attr in (self.train, self.val, self.test):
            if attr is not None and not isinstance(attr, Score):
                raise ValueError(f"{attr} must be a Score instance or None; got {type(attr)}")
    
    def __repr__(self) -> str:
        parts = []
        if self.train is not None:
            parts.append(f"train: {self.train!r}")
        if self.val is not None:
            parts.append(f"val: {self.val!r}")
        if self.test is not None:
            parts.append(f"test: {self.test!r}")
        parts.append(f"guess_accuracy: {self.guess_accuracy:.3f}")
        parts.append(f"ucl_accuracy: {self.ucl_accuracy:.3f}")
        return f"{'\n'.join(parts)}"


@dataclass(frozen=True, slots=True)
class DatasetEvalResult:
    per_subject: Optional[Dict[str, SubjectEvalResult]] = None
    train: Optional[Score] = None
    val: Optional[Score] = None
    test: Optional[Score] = None

    def __post_init__(self):
        if self.per_subject is not None and not isinstance(self.per_subject, dict):
            raise ValueError(f"per_subject must be a dict; got {type(self.per_subject)}")
        for attr in (self.train, self.val, self.test):
            if attr is not None and not isinstance(attr, Score):
                raise ValueError(f"{attr} must be a Score instance or None; got {type(attr)}")

    def get_train_scores_per_subject(self) -> Dict[str, Score]:
        if not self.per_subject:
            raise ValueError("Per subject scores are empty")
        if self.train is None:
            raise ValueError("Training scores are empty")
        return {sid: score.train for sid, score in self.per_subject.items()}

    def get_val_scores_per_subject(self) -> Dict[str, Score]:
        if not self.per_subject:
            raise ValueError("Per subject scores are empty")
        if self.val is None:
            raise ValueError("Validation scores are empty")
        return {sid: score.val for sid, score in self.per_subject.items()}

    def get_test_scores_per_subject(self) -> Dict[str, Score]:
        if not self.per_subject:
            raise ValueError("Per subject scores are empty")
        if self.test is None:
            raise ValueError("Test scores are empty")
        return {sid: score.test for sid, score in self.per_subject.items()}

    def get_ucl_per_subject(self) -> Dict[str, float]:
        if not self.per_subject:
            raise ValueError("Per subject scores are empty")
        return {sid: result.ucl_accuracy for sid, result in self.per_subject.items()}

    def __repr__(self) -> str:
        parts = []
        if self.train is not None:
            parts.append(f"train: {self.train!r}")
        if self.val is not None:
            parts.append(f"val: {self.val!r}")
        if self.test is not None:
            parts.append(f"test: {self.test!r}")
        return f"{'\n'.join(parts)}"
