# classifier.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Type

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class Classifier(Protocol):
    def clone(self) -> "Classifier": ...
    def fit(self, F: np.ndarray, y: np.ndarray) -> "Classifier": ...
    def predict(self, F: np.ndarray) -> np.ndarray: ...


@dataclass
class LDAClassifier(Classifier):
    shrinkage: bool = False

    def clone(self) -> "LDAClassifier":
        return LDAClassifier(shrinkage=self.shrinkage)

    def fit(self, F: np.ndarray, y: np.ndarray) -> "LDAClassifier":
        self.clf_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F, y) if self.shrinkage else LinearDiscriminantAnalysis().fit(F, y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        return self.clf_.predict(F)


@dataclass
class LogRegClassifier(Classifier):
    max_iter: int = 2000

    def __post_init__(self):
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError(f"max_iter must be a positive int; got {self.max_iter!r}")

    def clone(self) -> "LogRegClassifier":
        return LogRegClassifier(max_iter=self.max_iter)

    def fit(self, F: np.ndarray, y: np.ndarray) -> "LogRegClassifier":
        self.clf_ = LogisticRegression(max_iter=self.max_iter).fit(F, y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        return self.clf_.predict(F)


@dataclass
class LinSVMClassifier(Classifier):
    def clone(self) -> "LinSVMClassifier":
        return LinSVMClassifier()

    def fit(self, F: np.ndarray, y: np.ndarray) -> "LinSVMClassifier":
        self.clf_ = LinearSVC(dual="auto").fit(F, y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        return self.clf_.predict(F)


@dataclass
class NearestMeanClassifier(Classifier):
    mu_: dict | None = None
    classes_: np.ndarray | None = None

    def clone(self) -> "NearestMeanClassifier":
        return NearestMeanClassifier()

    def fit(self, F: np.ndarray, y: np.ndarray) -> "NearestMeanClassifier":
        self.classes_ = np.unique(y)
        self.mu_ = {c: F[y == c].mean(axis=0) for c in self.classes_}
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.classes_ is None:
            raise RuntimeError("NearestMeanClassifier is not fitted yet.")
        mus = np.stack([self.mu_[c] for c in self.classes_], axis=0)
        d2 = ((F[:, None, :] - mus[None, :, :]) ** 2).sum(axis=-1)
        return self.classes_[np.argmin(d2, axis=1)]


@dataclass
class ThresholdClassifier(Classifier):
    i0: int = 0
    i1: int = 1
    tau_: float | None = None
    classes_: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.i0, int) or not isinstance(self.i1, int):
            raise ValueError(f"i0 and i1 must be ints; got i0={self.i0!r}, i1={self.i1!r}")
        if self.i0 == self.i1:
            raise ValueError("i0 and i1 must be different feature indices.")

    def clone(self) -> "ThresholdClassifier":
        return ThresholdClassifier(i0=self.i0, i1=self.i1)

    def fit(self, F: np.ndarray, y: np.ndarray) -> "ThresholdClassifier":
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("ThresholdClassifier expects exactly 2 classes.")
        if F.ndim != 2:
            raise ValueError(f"Expected F to have shape (N, D); got {F.shape}")
        D = F.shape[1]
        if min(self.i0, self.i1) < 0 or max(self.i0, self.i1) >= D:
            raise ValueError(f"Feature indices out of range for D={D}: i0={self.i0}, i1={self.i1}")
        s = F[:, self.i0] - F[:, self.i1]
        c0, c1 = self.classes_[0], self.classes_[1]
        m0 = float(np.mean(s[y == c0])); m1 = float(np.mean(s[y == c1]))
        self.tau_ = 0.5 * (m0 + m1)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.tau_ is None or self.classes_ is None:
            raise RuntimeError("ThresholdClassifier is not fitted yet.")
        s = F[:, self.i0] - F[:, self.i1]
        return np.where(s > self.tau_, self.classes_[1], self.classes_[0])


class ClassifierType(Enum):
    LDA = LDAClassifier
    LOGREG = LogRegClassifier
    LINSVM = LinSVMClassifier
    NEAREST_MEAN = NearestMeanClassifier
    THRESHOLD = ThresholdClassifier

    @property
    def cls(self) -> Type[Classifier]:
        return self.value
