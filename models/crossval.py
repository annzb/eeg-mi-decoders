from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple, Literal

import numpy as np

from models import metrics
from models.evaluation import SubjectEvalResult
from models.model import Model


Split = Tuple[np.ndarray, np.ndarray]


class Crossval:
    def split(self, y: np.ndarray, groups: Optional[np.ndarray] = None) -> Iterator[Split]:
        raise NotImplementedError

    def eval_subject(
        self, model: Model, 
        X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        metric: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy
    ) -> SubjectEvalResult:
        accs = []
        for train_idx, test_idx in self.split(y, groups=groups):
            m = model.clone()
            m.fit(X[train_idx], y[train_idx])
            accs.append(metric(y[test_idx], m.predict(X[test_idx])))
        accs = np.asarray(accs, dtype=float)
        p0 = float(metrics.calc_guess_accuracy(y))
        ucl = metrics.calc_ucl_accuracy(int(len(X)), alpha=alpha, guess_accuracy=p0)
        return SubjectEvalResult(
            mean=float(accs.mean()), 
            std=float(accs.std(ddof=0)), 
            n_repeats=int(accs.size), 
            ucl_accuracy=float(ucl)
        )

    def eval_all_subjects(
        self, model: Model,
        X: np.ndarray, y: np.ndarray, groups: np.ndarray,
        alpha: float = 0.05,
        metric: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy
    ) -> Dict[str, SubjectEvalResult]:
        subj_ids = np.unique(groups)
        results: Dict[str, SubjectEvalResult] = {}
        for sid in subj_ids:
            mask = (groups == sid)
            results[sid] = self.eval_subject(model, X[mask], y[mask], groups=None, metric=metric, alpha=alpha)
        return results


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _as_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    _, inv = np.unique(y, return_inverse=True)
    return inv


def _split_into_subsets(idx: np.ndarray, n_subsets: int, rng: np.random.Generator) -> Sequence[np.ndarray]:
    idx = np.asarray(idx, dtype=int).copy()
    rng.shuffle(idx)
    return np.array_split(idx, n_subsets)


@dataclass
class RepeatedSubsetCrossval(Crossval):
    n_subsets: int = 10
    test_k: int = 3
    n_repeats: int = 120
    seed: int = 0
    stratify: bool = True
    require_all_classes_in_train: bool = True

    def split(self, y: np.ndarray, groups: Optional[np.ndarray] = None) -> Iterator[Split]:
        if groups is not None:
            raise ValueError("RepeatedSubsetCrossval does not use groups; pass groups=None.")
        y = _as_int_labels(y)
        n = len(y)
        if n == 0:
            raise ValueError("Empty y.")
        if self.n_subsets < 2:
            raise ValueError("n_subsets must be >= 2")
        if not (1 <= self.test_k < self.n_subsets):
            raise ValueError("test_k must be in [1, n_subsets-1]")
        rng = _rng(self.seed)

        subset_ids = np.arange(self.n_subsets)
        classes = np.unique(y)

        if not self.stratify:
            pooled = _split_into_subsets(np.arange(n), self.n_subsets, rng)
            for _ in range(self.n_repeats):
                test_subset_ids = rng.choice(subset_ids, size=self.test_k, replace=False)
                test_idx = np.concatenate([pooled[k] for k in test_subset_ids])
                train_idx = np.concatenate([pooled[k] for k in subset_ids if k not in test_subset_ids])
                if self.require_all_classes_in_train and len(np.unique(y[train_idx])) != len(classes):
                    continue
                yield train_idx, test_idx
            return

        subsets_by_class = {c: _split_into_subsets(np.where(y == c)[0], self.n_subsets, rng) for c in classes}
        for _ in range(self.n_repeats):
            test_subset_ids = rng.choice(subset_ids, size=self.test_k, replace=False)
            train_idx, test_idx = [], []
            for c in classes:
                cls_subsets = subsets_by_class[c]
                for k in subset_ids:
                    (test_idx if k in test_subset_ids else train_idx).append(cls_subsets[k])
            train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
            test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
            if self.require_all_classes_in_train and len(np.unique(y[train_idx])) != len(classes):
                continue
            yield train_idx, test_idx


@dataclass
class KFoldCrossval(Crossval):
    k: int = 5
    seed: int = 0
    shuffle: bool = True
    stratify: bool = True
    require_all_classes_in_train: bool = True

    def split(self, y: np.ndarray, groups: Optional[np.ndarray] = None) -> Iterator[Split]:
        if groups is not None:
            raise ValueError("KFoldCrossval does not use groups; pass groups=None.")
        y = _as_int_labels(y)
        n = len(y)
        if self.k < 2 or self.k > n:
            raise ValueError("k must be in [2, len(y)]")
        rng = _rng(self.seed)
        classes = np.unique(y)

        if not self.stratify:
            idx = np.arange(n)
            if self.shuffle:
                rng.shuffle(idx)
            folds = np.array_split(idx, self.k)
            for i in range(self.k):
                test_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(self.k) if j != i])
                if self.require_all_classes_in_train and len(np.unique(y[train_idx])) != len(classes):
                    continue
                yield train_idx, test_idx
            return

        folds_by_class = {c: np.array_split((_split := np.where(y == c)[0].copy()), self.k) for c in classes}
        if self.shuffle:
            for c in classes:
                idx = np.where(y == c)[0].copy()
                rng.shuffle(idx)
                folds_by_class[c] = np.array_split(idx, self.k)
        for i in range(self.k):
            test_parts = [folds_by_class[c][i] for c in classes]
            train_parts = [np.concatenate([folds_by_class[c][j] for j in range(self.k) if j != i]) for c in classes]
            test_idx = np.concatenate(test_parts)
            train_idx = np.concatenate(train_parts)
            if self.require_all_classes_in_train and len(np.unique(y[train_idx])) != len(classes):
                continue
            yield train_idx, test_idx


@dataclass
class LeaveOneGroupOut(Crossval):
    def split(self, y: np.ndarray, groups: Optional[np.ndarray] = None) -> Iterator[Split]:
        if groups is None:
            raise ValueError("groups is required for LeaveOneGroupOut.")
        y = _as_int_labels(y)
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups must have same length as y.")
        uniq = np.unique(groups)
        for g in uniq:
            test_idx = np.where(groups == g)[0]
            train_idx = np.where(groups != g)[0]
            yield train_idx, test_idx


def make_crossval(kind: Literal["repeated_subset", "kfold", "logo"], **kwargs) -> Crossval:
    if kind == "repeated_subset":
        return RepeatedSubsetCrossval(**kwargs)
    if kind == "kfold":
        return KFoldCrossval(**kwargs)
    if kind == "logo":
        return LeaveOneGroupOut(**kwargs)
    raise ValueError(f"Unknown crossval kind: {kind}")
