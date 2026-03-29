from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from data import get_dataset
from models import Model
from evaluation.split import Split


@dataclass(frozen=True, slots=True)
class ModelEvalResult:
    model: Model  # unique instance
    split: Split  # unique instance
    guess_accuracy: float  # unique across dataset-subject
    ucl_accuracy: float  # unique across dataset-subject
    ucl_alpha: float = 0.05  # parameter
    dataset_id: Optional[str] = None
    subject_id: Optional[str] = None
    train_acc: Optional[float] = None  # per sample in split, unique across model-split
    val_acc: Optional[float] = None  # per sample in split, unique across model-split
    test_acc: Optional[float] = None  # per sample in split, unique across model-split

    def __post_init__(self):
        if not isinstance(self.model, Model):
            raise ValueError(f"model must be a Model instance; got {type(self.model)}")
        if not isinstance(self.split, Split):
            raise ValueError(f"split must be a Split instance; got {type(self.split)}")
        if not isinstance(self.guess_accuracy, float) or self.guess_accuracy < 0 or self.guess_accuracy > 1:
            raise ValueError(f"guess_accuracy must be a finite float in [0, 1]; got {self.guess_accuracy!r}")
        if not isinstance(self.ucl_accuracy, float) or self.ucl_accuracy < 0 or self.ucl_accuracy > 1:
            raise ValueError(f"ucl_accuracy must be a finite float in [0, 1]; got {self.ucl_accuracy!r}")
        if not isinstance(self.ucl_alpha, float) or self.ucl_alpha < 0 or self.ucl_alpha > 1:
            raise ValueError(f"ucl_alpha must be a finite float in [0, 1]; got {self.ucl_alpha!r}")
        if self.dataset_id is not None and get_dataset(self.dataset_id) is None:
            raise ValueError(f"Dataset {self.dataset_id} not found")
        if self.subject_id is not None and not isinstance(self.subject_id, str):
            raise ValueError(f"subject_id must be a string; got {type(self.subject_id)}")
        if self.train_acc is not None and (not isinstance(self.train_acc, float) or self.train_acc < 0 or self.train_acc > 1):
            raise ValueError(f"train_acc must be a finite float in [0, 1]; got {self.train_acc!r}")
        if self.val_acc is not None and (not isinstance(self.val_acc, float) or self.val_acc < 0 or self.val_acc > 1):
            raise ValueError(f"val_acc must be a finite float in [0, 1]; got {self.val_acc!r}")
        if self.test_acc is not None and (not isinstance(self.test_acc, float) or self.test_acc < 0 or self.test_acc > 1):
            raise ValueError(f"test_acc must be a finite float in [0, 1]; got {self.test_acc!r}")


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
