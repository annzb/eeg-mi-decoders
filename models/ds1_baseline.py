from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from models.base import Model


@dataclass
class CSP_LDA(Model):
    n_components: int = 4
    reg: float = 1e-10
    lda_kwargs: Dict[str, Any] | None = None
    W_: np.ndarray | None = None
    clf_: LinearDiscriminantAnalysis | None = None
    classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP_LDA":
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("CSP_LDA expects exactly 2 classes.")
        self.W_ = fit_csp(X, y, n_components=self.n_components, reg=self.reg)
        F = csp_logvar_features(X, self.W_)
        self.clf_ = LinearDiscriminantAnalysis(**(self.lda_kwargs or {}))
        self.clf_.fit(F, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.clf_ is None:
            raise RuntimeError("Model is not fitted yet.")
        F = csp_logvar_features(X, self.W_)
        return self.clf_.predict(F)


def compute_cov(sample):
    C = sample @ sample.T
    tr = np.trace(C)
    return C / tr if tr > 0 else C


def fit_csp(X_train, y_train, n_components=4, reg=1e-10):
    classes = np.unique(y_train)
    if len(classes) != 2:
        raise ValueError("CSP requires exactly 2 classes.")

    c0, c1 = classes[0], classes[1]
    X0 = X_train[y_train == c0]
    X1 = X_train[y_train == c1]
    C0 = np.mean([compute_cov(x) for x in X0], axis=0)
    C1 = np.mean([compute_cov(x) for x in X1], axis=0)
    n_ch = C0.shape[0]
    C0 = C0 + reg * np.eye(n_ch)
    C1 = C1 + reg * np.eye(n_ch)
    Cc = C0 + C1

    evals, evecs = np.linalg.eig(np.linalg.solve(Cc, C0))
    evals = np.real(evals)
    evecs = np.real(evecs)
    idx = np.argsort(evals)
    evecs = evecs[:, idx]
    k2 = n_components // 2

    W = np.concatenate([evecs[:, :k2], evecs[:, -k2:]], axis=1)
    return W


def csp_logvar_features(X, W):
    Z = np.einsum("tcj,ck->tkj", X, W)
    var = np.var(Z, axis=-1, ddof=0)
    var = var / (np.sum(var, axis=1, keepdims=True) + 1e-12)
    return np.log(var + 1e-12)


# def eval_subject_csp_lda(X_subj, y_subj, n_repeats=120, n_subsets=10, seed=0):
#     rng = np.random.default_rng(seed)
#     classes = np.unique(y_subj)
#     if len(classes) != 2:
#         raise ValueError("Expected exactly 2 classes for MI (left/right).")

#     idx_by_class = {c: np.where(y_subj == c)[0] for c in classes}
#     subsets = {}
#     for c in classes:
#         idx = idx_by_class[c].copy()
#         rng.shuffle(idx)
#         subsets[c] = np.array_split(idx, n_subsets)

#     accs = []
#     subset_ids = np.arange(n_subsets)
#     for _ in range(n_repeats):
#         test_subset_ids = rng.choice(subset_ids, size=3, replace=False)
#         test_idx = []
#         train_idx = []
#         for c in classes:
#             for k in subset_ids:
#                 if k in test_subset_ids:
#                     test_idx.append(subsets[c][k])
#                 else:
#                     train_idx.append(subsets[c][k])
#         test_idx = np.concatenate(test_idx)
#         train_idx = np.concatenate(train_idx)
#         X_train, y_train = X_subj[train_idx], y_subj[train_idx]
#         X_test, y_test = X_subj[test_idx], y_subj[test_idx]
#         W = fit_csp(X_train, y_train, n_components=4)
#         F_train = csp_logvar_features(X_train, W)    
#         F_test = csp_logvar_features(X_test, W)
#         clf = LinearDiscriminantAnalysis()
#         clf.fit(F_train, y_train)
#         accs.append(clf.score(F_test, y_test))

#     accs = np.asarray(accs, dtype=float)
#     return float(accs.mean()), float(accs.std(ddof=0))


@dataclass
class RepeatedSubsetCrossval:
    n_subsets: int = 10
    test_k: int = 3
    n_repeats: int = 120
    seed: int = 0

    def split(self, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Expected exactly 2 classes.")

        idx_by_class = {c: np.where(y == c)[0] for c in classes}
        subsets = {}
        for c in classes:
            idx = idx_by_class[c].copy()
            rng.shuffle(idx)
            subsets[c] = np.array_split(idx, self.n_subsets)

        subset_ids = np.arange(self.n_subsets)
        for _ in range(self.n_repeats):
            test_subset_ids = rng.choice(subset_ids, size=self.test_k, replace=False)

            train_idx, test_idx = [], []
            for c in classes:
                for k in subset_ids:
                    (test_idx if k in test_subset_ids else train_idx).append(subsets[c][k])

            yield np.concatenate(train_idx), np.concatenate(test_idx)
