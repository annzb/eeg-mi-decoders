from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any, Type

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC


@dataclass(slots=True)
class Classifier(ABC):
    clf_: Any | None = None

    @abstractmethod
    def clone(self) -> "Classifier": ...

    @abstractmethod
    def fit(self, F: np.ndarray, y: np.ndarray): ...

    @abstractmethod
    def predict(self, F: np.ndarray) -> np.ndarray: ...

    def _validate_fit_input(self, F: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(F, np.ndarray):
            raise ValueError(f"Expected F to be a numpy array; got {type(F)}")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError(f"Expected y to have shape (N,); got {y.shape}")
        if F.shape[0] != y.shape[0]:
            raise ValueError(f"F and y must have the same length; got {F.shape[0]} vs {y.shape[0]}")

    def _validate_predict_input(self, F: np.ndarray) -> None:
        if not isinstance(F, np.ndarray):
            raise ValueError(f"Expected F to be a numpy array; got {type(F)}")


@dataclass(slots=True)
class LDAClassifier(Classifier):
    lda_shrinkage: bool = False

    def __post_init__(self):
        if not isinstance(self.lda_shrinkage, bool):
            raise ValueError(f"lda_shrinkage must be a bool; got {self.lda_shrinkage!r}")

    def clone(self) -> "LDAClassifier":
        return LDAClassifier(lda_shrinkage=self.lda_shrinkage)

    def fit(self, F: np.ndarray, y: np.ndarray):
        self._validate_fit_input(F, y)
        self.clf_ = (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F, y)
            if self.lda_shrinkage
            else LinearDiscriminantAnalysis().fit(F, y)
        )

    def predict(self, F: np.ndarray) -> np.ndarray:
        self._validate_predict_input(F)
        return self.clf_.predict(F)


@dataclass(slots=True)
class LogRegClassifier(Classifier):
    logreg_max_iter: int = 2000

    def __post_init__(self):
        if not isinstance(self.logreg_max_iter, int) or self.logreg_max_iter <= 0:
            raise ValueError("logreg_max_iter must be a positive int")

    def clone(self) -> "LogRegClassifier":
        return LogRegClassifier(logreg_max_iter=self.logreg_max_iter)

    def fit(self, F: np.ndarray, y: np.ndarray):
        self._validate_fit_input(F, y)
        self.clf_ = LogisticRegression(max_iter=self.logreg_max_iter).fit(F, y)

    def predict(self, F: np.ndarray) -> np.ndarray:
        self._validate_predict_input(F)
        return self.clf_.predict(F)


@dataclass(slots=True)
class LinSVMClassifier(Classifier):

    def clone(self) -> "LinSVMClassifier":
        return LinSVMClassifier()

    def fit(self, F: np.ndarray, y: np.ndarray):
        self._validate_fit_input(F, y)
        self.clf_ = LinearSVC(dual="auto").fit(F, y)

    def predict(self, F: np.ndarray) -> np.ndarray:
        self._validate_predict_input(F)
        return self.clf_.predict(F)


@dataclass(slots=True)
class NearestMeanClassifier(Classifier):
    mu_: dict | None = None
    classes_: np.ndarray | None = None

    def clone(self) -> "NearestMeanClassifier":
        return NearestMeanClassifier()

    def fit(self, F: np.ndarray, y: np.ndarray):
        self._validate_fit_input(F, y)
        self.classes_ = np.unique(y)
        self.mu_ = {c: F[y == c].mean(axis=0) for c in self.classes_}

    def predict(self, F: np.ndarray) -> np.ndarray:
        self._validate_predict_input(F)
        if self.mu_ is None or self.classes_ is None:
            raise RuntimeError("NearestMeanClassifier is not fitted yet.")
        mus = np.stack([self.mu_[c] for c in self.classes_], axis=0)
        d2 = ((F[:, None, :] - mus[None, :, :]) ** 2).sum(axis=-1)
        return self.classes_[np.argmin(d2, axis=1)]


@dataclass(slots=True)
class ThresholdClassifier(Classifier):
    threshold_i0: int = 0
    threshold_i1: int = 1

    tau_: float | None = None
    classes_: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.threshold_i0, int) or not isinstance(self.threshold_i1, int):
            raise ValueError(f"i0 and i1 must be ints; got i0={self.threshold_i0!r}, i1={self.threshold_i1!r}")
        if self.threshold_i0 == self.threshold_i1:
            raise ValueError("i0 and i1 must be different feature indices.")

    def clone(self) -> "ThresholdClassifier":
        return ThresholdClassifier(i0=self.threshold_i0, i1=self.threshold_i1)

    def fit(self, F: np.ndarray, y: np.ndarray):
        self._validate_fit_input(F, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("ThresholdClassifier expects exactly 2 classes.")
        D = F.shape[1]
        if min(self.threshold_i0, self.threshold_i1) < 0 or max(self.threshold_i0, self.threshold_i1) >= D:
            raise ValueError(f"Feature indices out of range for D={D}: i0={self.threshold_i0}, i1={self.threshold_i1}")
        s = F[:, self.threshold_i0] - F[:, self.threshold_i1]
        c0, c1 = self.classes_[0], self.classes_[1]
        m0 = float(np.mean(s[y == c0])); m1 = float(np.mean(s[y == c1]))
        self.tau_ = 0.5 * (m0 + m1)

    def predict(self, F: np.ndarray) -> np.ndarray:
        self._validate_predict_input(F)
        if self.tau_ is None or self.classes_ is None:
            raise RuntimeError("ThresholdClassifier is not fitted yet.")
        s = F[:, self.threshold_i0] - F[:, self.threshold_i1]
        return np.where(s > self.tau_, self.classes_[1], self.classes_[0])


@dataclass(slots=True)
class VotingSVMClassifier(Classifier):
    svm_C: float = 1.0
    svm_kernel: str = "linear"
    svm_decision_function_shape: str = "ovo"
    svm_gamma: str | float = "scale"
    svm_degree: int = 3

    def __post_init__(self):
        if not isinstance(self.svm_C, (int, float)) or not np.isfinite(self.svm_C) or self.svm_C <= 0:
            raise ValueError(f"svm_C must be finite and > 0; got {self.svm_C!r}")
        if self.svm_kernel not in ("linear", "rbf", "poly", "sigmoid"):
            raise ValueError(f"svm_kernel must be one of linear/rbf/poly/sigmoid; got {self.svm_kernel!r}")
        if self.svm_decision_function_shape not in ("ovo", "ovr"):
            raise ValueError(f"svm_decision_function_shape must be 'ovo' or 'ovr'; got {self.svm_decision_function_shape!r}")
        if not (isinstance(self.svm_gamma, (int, float)) or self.svm_gamma in ("scale", "auto")):
            raise ValueError(f"svm_gamma must be a number or 'scale'/'auto'; got {self.svm_gamma!r}")
        if isinstance(self.svm_gamma, (int, float)) and (not np.isfinite(self.svm_gamma) or self.svm_gamma <= 0):
            raise ValueError(f"svm_gamma must be finite and > 0; got {self.svm_gamma!r}")
        if not isinstance(self.svm_degree, int) or self.svm_degree <= 0:
            raise ValueError(f"svm_degree must be a positive int; got {self.svm_degree!r}")

        self.clf_ = SVC(
            C=float(self.svm_C),
            kernel=self.svm_kernel,
            decision_function_shape=self.svm_decision_function_shape,
            gamma=self.svm_gamma,
            degree=self.svm_degree
        )

    def clone(self) -> "VotingSVMClassifier":
        return VotingSVMClassifier(
            svm_C=self.svm_C,
            svm_kernel=self.svm_kernel,
            svm_decision_function_shape=self.svm_decision_function_shape,
            svm_gamma=self.svm_gamma,
            svm_degree=self.svm_degree,
        )

    def fit(self, F: np.ndarray, y: np.ndarray):
        self._validate_fit_input(F, y)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != F.shape[0]:
            raise ValueError(f"Expected y to have shape (N,); got {y.shape} for N={F.shape[0]}")
        self.clf_.fit(F, y)

    def predict(self, F: np.ndarray) -> np.ndarray:
        self._validate_predict_input(F)
        if self.clf_ is None:
            raise RuntimeError("VotingSVMClassifier is not fitted yet.")
        return self.clf_.predict(F)


# @dataclass(slots=True)
# class RLDAClassifier(Classifier):
#     classes_: np.ndarray | None = None
#     pair_clfs_: list[LinearDiscriminantAnalysis] | None = None
#     pair_labels_: list[tuple[object, object]] | None = None
#     pair_score_ranges_: list[tuple[float, float]] | None = None

#     def clone(self) -> "RLDAClassifier":
#         return RLDAClassifier()

#     def fit(self, F: np.ndarray, y: np.ndarray):
#         self._validate_fit_input(F, y)
#         y = np.asarray(y)
#         classes = np.unique(y)
#         if classes.size < 2:
#             raise ValueError("RLDAClassifier requires at least 2 classes")

#         self.classes_ = classes
#         self.pair_clfs_ = []
#         self.pair_labels_ = []
#         self.pair_score_ranges_ = []

#         for c0, c1 in combinations(classes, 2):
#             mask = (y == c0) | (y == c1)
#             F_pair = F[mask]
#             y_pair = y[mask]
#             clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
#             clf.fit(F_pair, y_pair)
#             raw = clf.decision_function(F_pair)
#             raw = np.asarray(raw, dtype=np.float64).reshape(-1)
#             lo = float(np.min(raw))
#             hi = float(np.max(raw))
#             if not np.isfinite(lo) or not np.isfinite(hi):
#                 raise ValueError(f"Non-finite RLDA decision values for pair {(c0, c1)!r}")

#             self.pair_clfs_.append(clf)
#             self.pair_labels_.append((c0, c1))
#             self.pair_score_ranges_.append((lo, hi))

#         self.clf_ = True

#     @staticmethod
#     def _scale_to_unit_interval(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
#         x = np.asarray(x, dtype=np.float64)
#         if hi <= lo:
#             return np.full_like(x, 0.5, dtype=np.float64)
#         out = (x - lo) / (hi - lo)
#         return np.clip(out, 0.0, 1.0)

#     def decision_scores(self, F: np.ndarray) -> np.ndarray:
#         self._validate_predict_input(F)
#         if self.classes_ is None or self.pair_clfs_ is None or self.pair_labels_ is None or self.pair_score_ranges_ is None:
#             raise RuntimeError("RLDAClassifier is not fitted yet.")

#         class_to_idx = {c: i for i, c in enumerate(self.classes_)}
#         scores = np.zeros((F.shape[0], self.classes_.size), dtype=np.float64)

#         for clf, (c0, c1), (lo, hi) in zip(self.pair_clfs_, self.pair_labels_, self.pair_score_ranges_):
#             raw = clf.decision_function(F)
#             raw = np.asarray(raw, dtype=np.float64).reshape(-1)
#             p_c1 = self._scale_to_unit_interval(raw, lo, hi)
#             p_c0 = 1.0 - p_c1
#             i0 = class_to_idx[c0]
#             i1 = class_to_idx[c1]
#             scores[:, i0] += p_c0
#             scores[:, i1] += p_c1

#         return scores

#     def predict(self, F: np.ndarray) -> np.ndarray:
#         scores = self.decision_scores(F)
#         return self.classes_[np.argmax(scores, axis=1)]


@dataclass(slots=True)
class RLDAClassifier(Classifier):
    classes_: np.ndarray | None = None
    pair_clfs_: list[LinearDiscriminantAnalysis] | None = None
    pair_labels_: list[tuple[object, object]] | None = None
    pair_score_ranges_: list[tuple[float, float]] | None = None

    def clone(self) -> "RLDAClassifier":
        return RLDAClassifier()

    def fit(self, F: np.ndarray, y: np.ndarray):
        self._validate_fit_input(F, y)
        y = np.asarray(y)
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("RLDAClassifier requires at least 2 classes")

        self.classes_ = classes
        self.pair_clfs_ = []
        self.pair_labels_ = []
        self.pair_score_ranges_ = []

        for c0, c1 in combinations(classes, 2):
            mask = (y == c0) | (y == c1)
            F_pair = F[mask]
            y_pair = y[mask]

            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(F_pair, y_pair)

            raw = np.asarray(clf.decision_function(F_pair), dtype=np.float64).reshape(-1)
            lo = float(np.min(raw))
            hi = float(np.max(raw))
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"Non-finite RLDA decision values for pair {(c0, c1)!r}")

            self.pair_clfs_.append(clf)
            self.pair_labels_.append((c0, c1))
            self.pair_score_ranges_.append((lo, hi))

        self.clf_ = True
        return self

    @staticmethod
    def _scale_to_unit_interval(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if hi <= lo:
            return np.full_like(x, 0.5, dtype=np.float64)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    def decision_scores(self, F: np.ndarray) -> np.ndarray:
        self._validate_predict_input(F)
        if self.classes_ is None or self.pair_clfs_ is None or self.pair_labels_ is None or self.pair_score_ranges_ is None:
            raise RuntimeError("RLDAClassifier is not fitted yet.")

        scores = np.zeros((F.shape[0], self.classes_.size), dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for clf, (c0, c1), (lo, hi) in zip(self.pair_clfs_, self.pair_labels_, self.pair_score_ranges_):
            raw = np.asarray(clf.decision_function(F), dtype=np.float64).reshape(-1)
            s1 = self._scale_to_unit_interval(raw, lo, hi)
            s0 = 1.0 - s1
            scores[:, class_to_idx[c0]] += s0
            scores[:, class_to_idx[c1]] += s1

        return scores

    def predict(self, F: np.ndarray) -> np.ndarray:
        self._validate_predict_input(F)
        scores = self.decision_scores(F)
        return self.classes_[np.argmax(scores, axis=1)]


class ClassifierType(Enum):
    LDA = LDAClassifier
    RLDA = RLDAClassifier
    LOGREG = LogRegClassifier
    LINSVM = LinSVMClassifier
    NEAREST_MEAN = NearestMeanClassifier
    THRESHOLD = ThresholdClassifier
    VOTING_SVM = VotingSVMClassifier

    @property
    def cls(self) -> Type[Classifier]:
        return self.value
