from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple
import uuid

import numpy as np


@dataclass(frozen=True, slots=True)
class Split:
    train_idx: np.ndarray
    val_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None
    split_id: uuid.UUID = uuid.uuid4()

    def __post_init__(self):
        self._validate_split(self.train_idx, self.val_idx, self.test_idx)

    @staticmethod
    def _validate_split(train_idx: np.ndarray, val_idx: Optional[np.ndarray], test_idx: Optional[np.ndarray]) -> None:
        Split._validate_idx(train_idx, "train_idx")
        if val_idx is not None:
            Split._validate_idx(val_idx, "val_idx")
        if test_idx is not None:
            Split._validate_idx(test_idx, "test_idx")
        if val_idx is None and test_idx is None:
            raise ValueError("A split must define at least one of val_idx or test_idx")
        a = np.asarray(train_idx, dtype=int)
        if val_idx is not None:
            b = np.asarray(val_idx, dtype=int)
            if np.intersect1d(a, b).size != 0:
                raise ValueError("train_idx and val_idx must be disjoint")
        if test_idx is not None:
            c = np.asarray(test_idx, dtype=int)
            if np.intersect1d(a, c).size != 0:
                raise ValueError("train_idx and test_idx must be disjoint")
        if val_idx is not None and test_idx is not None:
            if np.intersect1d(np.asarray(val_idx, dtype=int), np.asarray(test_idx, dtype=int)).size != 0:
                raise ValueError("val_idx and test_idx must be disjoint")

    @staticmethod
    def _validate_idx(idx: np.ndarray, name: str) -> None:
        idx = np.asarray(idx)
        if idx.ndim != 1:
            raise ValueError(f"{name} must be 1D; got shape={idx.shape}")
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"{name} must be integer dtype; got dtype={idx.dtype}")
        if idx.size == 0:
            raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class Splitter:
    seed: int = 0
    shuffle: bool = True
    stratify: bool = True
    require_all_classes_in_train: bool = True

    def __call__(self, y: np.ndarray, **kwargs) -> Iterator[Split]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RepeatedSubsetCrossval(Splitter):
    n_subsets: int = 10
    test_k: int = 3
    n_repeats: int = 120

    def __call__(self, y: np.ndarray, **kwargs) -> Iterator[Split]:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape={y.shape}")
        n = int(y.size)
        if n == 0:
            raise ValueError("Empty y")
        if self.n_subsets < 2:
            raise ValueError("n_subsets must be >= 2")
        if not (1 <= self.test_k < self.n_subsets):
            raise ValueError("test_k must be in [1, n_subsets-1]")
        if not isinstance(self.n_repeats, int) or self.n_repeats <= 0:
            raise ValueError("n_repeats must be a positive int")

        rng = np.random.default_rng(self.seed)
        subset_ids = np.arange(self.n_subsets)
        classes = np.unique(y)

        def split_into_subsets(idx: np.ndarray) -> Sequence[np.ndarray]:
            idx = np.asarray(idx, dtype=int).copy()
            rng.shuffle(idx)
            return np.array_split(idx, self.n_subsets)

        if not self.stratify:
            pooled = split_into_subsets(np.arange(n, dtype=int))
            for _ in range(self.n_repeats):
                test_subset_ids = rng.choice(subset_ids, size=self.test_k, replace=False)
                test_idx = np.concatenate([pooled[k] for k in test_subset_ids])
                train_idx = np.concatenate([pooled[k] for k in subset_ids if k not in test_subset_ids])
                if self.require_all_classes_in_train and np.unique(y[train_idx]).size != classes.size:
                    continue
                yield train_idx, None, test_idx
            return

        subsets_by_class = {c: split_into_subsets(np.where(y == c)[0]) for c in classes}
        for _ in range(self.n_repeats):
            test_subset_ids = rng.choice(subset_ids, size=self.test_k, replace=False)
            train_parts, test_parts = [], []
            for c in classes:
                cls_subsets = subsets_by_class[c]
                for k in subset_ids:
                    (test_parts if k in test_subset_ids else train_parts).append(cls_subsets[k])
            train_idx = np.concatenate(train_parts)
            test_idx = np.concatenate(test_parts)
            if self.require_all_classes_in_train and np.unique(y[train_idx]).size != classes.size:
                continue
            yield Split(train_idx, None, test_idx)


@dataclass(frozen=True, slots=True)
class KFoldCrossval(Split):
    k: int = 5

    def __call__(self, y: np.ndarray, **kwargs) -> Iterator[Split]:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape={y.shape}")
        n = int(y.size)
        if self.k < 2 or self.k > n:
            raise ValueError("k must be in [2, len(y)]")

        rng = np.random.default_rng(self.seed)
        classes = np.unique(y)

        if not self.stratify:
            idx = np.arange(n, dtype=int)
            if self.shuffle:
                rng.shuffle(idx)
            folds = np.array_split(idx, self.k)
            for i in range(self.k):
                test_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(self.k) if j != i])
                if self.require_all_classes_in_train and np.unique(y[train_idx]).size != classes.size:
                    continue
                yield Split(train_idx, None, test_idx)
            return

        folds_by_class: Dict[object, Sequence[np.ndarray]] = {}
        for c in classes:
            idx_c = np.where(y == c)[0].astype(int, copy=True)
            if self.shuffle:
                rng.shuffle(idx_c)
            folds_by_class[c] = np.array_split(idx_c, self.k)

        for i in range(self.k):
            test_parts = [folds_by_class[c][i] for c in classes]
            train_parts = [np.concatenate([folds_by_class[c][j] for j in range(self.k) if j != i]) for c in classes]
            test_idx = np.concatenate(test_parts)
            train_idx = np.concatenate(train_parts)
            if self.require_all_classes_in_train and np.unique(y[train_idx]).size != classes.size:
                continue
            yield Split(train_idx, None, test_idx)


@dataclass(frozen=True, slots=True)
class RepeatedHoldoutSplit(Splitter):
    train_frac: float = 0.63
    val_frac: float = 0.27
    test_frac: float = 0.10
    n_repeats: int = 5

    def __post_init__(self):
        for name, v in (("train_frac", self.train_frac), ("val_frac", self.val_frac), ("test_frac", self.test_frac)):
            if not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and >= 0; got {v!r}")
        s = float(self.train_frac + self.val_frac + self.test_frac)
        if abs(s - 1.0) > 1e-12:
            raise ValueError(f"Fractions must sum to 1.0; got {s!r}")
        if self.val_frac == 0.0 and self.test_frac == 0.0:
            raise ValueError("At least one of val_frac or test_frac must be > 0")
        if not isinstance(self.n_repeats, int) or self.n_repeats <= 0:
            raise ValueError(f"n_repeats must be positive int; got {self.n_repeats!r}")

    def __call__(self, y: np.ndarray, **kwargs) -> Iterator[Split]:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape={y.shape}")
        n = int(y.size)
        if n == 0:
            raise ValueError("Empty y")
        rng = np.random.default_rng(self.seed)
        classes = np.unique(y)

        def take_counts(m: int) -> tuple[int, int, int]:
            n_train = int(np.floor(self.train_frac * m))
            n_val = int(np.floor(self.val_frac * m))
            n_test = m - n_train - n_val
            return n_train, n_val, n_test

        for _ in range(self.n_repeats):
            if not self.stratify:
                idx = np.arange(n, dtype=int)
                rng.shuffle(idx)
                n_train, n_val, n_test = take_counts(n)
                if n_train <= 0:
                    continue
                train_idx = idx[:n_train]
                val_idx = idx[n_train : n_train + n_val] if n_val > 0 else None
                test_idx = idx[n_train + n_val :] if n_test > 0 else None
                if self.require_all_classes_in_train and np.unique(y[train_idx]).size != classes.size:
                    continue
                yield Split(train_idx, val_idx, test_idx)
                continue

            train_parts, val_parts, test_parts = [], [], []
            ok = True
            for c in classes:
                idx_c = np.where(y == c)[0].astype(int, copy=True)
                rng.shuffle(idx_c)
                n_train_c, n_val_c, n_test_c = take_counts(int(idx_c.size))
                if n_train_c <= 0:
                    ok = False
                    break
                train_parts.append(idx_c[:n_train_c])
                if n_val_c > 0:
                    val_parts.append(idx_c[n_train_c : n_train_c + n_val_c])
                if n_test_c > 0:
                    test_parts.append(idx_c[n_train_c + n_val_c :])
            if not ok:
                continue
            train_idx = np.concatenate(train_parts)
            val_idx = np.concatenate(val_parts) if len(val_parts) > 0 else None
            test_idx = np.concatenate(test_parts) if len(test_parts) > 0 else None
            if self.require_all_classes_in_train and np.unique(y[train_idx]).size != classes.size:
                continue
            yield Split(train_idx, val_idx, test_idx)
