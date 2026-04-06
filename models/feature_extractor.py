from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Protocol, Type

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data import preprocess


@dataclass(slots=True)
class FeatureExtractor(ABC):
    eps: float = 1e-12
    normalize_var: bool = True
    log_var: bool = True

    def __post_init__(self):
        if not isinstance(self.eps, float) or not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be finite and > 0; got {self.eps!r}")
        if not isinstance(self.normalize_var, bool):
            raise ValueError(f"normalize_var must be a bool, got {self.normalize_var!r}")
        if not isinstance(self.log_var, bool):
            raise ValueError(f"log_var must be a bool, got {self.log_var!r}")
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "FeatureExtractor": ...

    @abstractmethod
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray: ...

    def _validate_fit_input(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError(f"Expected y to have shape (N,); got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same length; got {X.shape[0]} vs {y.shape[0]}")

    def _validate_transform_input(self, X: np.ndarray):
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")

    def clone(self) -> "FeatureExtractor":
        return self.__class__(
            eps=self.eps,
            normalize_var=self.normalize_var,
            log_var=self.log_var,
        )


@dataclass(slots=True)
class CSPLogVar(FeatureExtractor):
    reg: float = 1e-10
    csp_n_components: int = 4
    mode: str = "ovr"  # "ovr", "ovo", "jad"

    csp_filters_: list[np.ndarray] | None = None
    filter_labels_: list[object] | None = None

    def __post_init__(self):
        FeatureExtractor.__post_init__(self)
        if not isinstance(self.reg, float) or not np.isfinite(self.reg) or self.reg < 0:
            raise ValueError(f"reg must be finite and >= 0; got {self.reg!r}")
        if not isinstance(self.csp_n_components, int) or self.csp_n_components <= 0 or self.csp_n_components % 2 != 0:
            raise ValueError(f"csp_n_components must be a positive even int; got {self.csp_n_components!r}")
        if self.mode not in {"ovr", "ovo", "jad"}:
            raise ValueError(f"mode must be one of ('ovr', 'ovo', 'jad'); got {self.mode!r}")

    def clone(self) -> "CSPLogVar":
        return CSPLogVar(
            csp_n_components=self.csp_n_components,
            reg=self.reg,
            mode=self.mode,
            normalize_var=self.normalize_var,
            log_var=self.log_var,
            eps=self.eps,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "CSPLogVar":
        self._validate_fit_input(X, y)
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("CSPLogVar requires at least 2 classes.")

        self.csp_filters_ = []
        self.filter_labels_ = []

        if len(classes) == 2:
            self.csp_filters_.append(self._fit_csp_binary(X, y))
            self.filter_labels_.append((classes[0], classes[1]))
            return self

        if self.mode == "ovr":
            for c in classes:
                y_bin = (y == c).astype(np.int8)
                if np.all(y_bin == 0) or np.all(y_bin == 1):
                    raise ValueError(f"Class {c!r} has no samples; cannot fit CSP.")
                self.csp_filters_.append(self._fit_csp_binary(X, y_bin))
                self.filter_labels_.append((c, "rest"))
            return self

        if self.mode == "ovo":
            for c0, c1 in combinations(classes, 2):
                mask = (y == c0) | (y == c1)
                X_pair = X[mask]
                y_pair = y[mask]
                self.csp_filters_.append(self._fit_csp_binary(X_pair, y_pair))
                self.filter_labels_.append((c0, c1))
            return self

        if self.mode == "jad":
            self.csp_filters_.append(self._fit_csp_jad(X, y))
            self.filter_labels_.append("jad")
            return self

        raise RuntimeError(f"Unsupported mode {self.mode!r}")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._validate_transform_input(X)
        if self.csp_filters_ is None or len(self.csp_filters_) == 0:
            raise RuntimeError("CSPLogVar not fitted")

        feats = []
        for csp_filter in self.csp_filters_:
            Z = np.einsum("nct,ck->nkt", X, csp_filter)  # (N, K, T)
            var = np.var(Z, axis=-1, ddof=0)
            if self.normalize_var:
                var = var / (np.sum(var, axis=1, keepdims=True) + self.eps)
            feats.append(np.log(var + self.eps) if self.log_var else var)
        return np.concatenate(feats, axis=1)

    @staticmethod
    def _compute_cov(x: np.ndarray) -> np.ndarray:
        C = x @ x.T
        tr = np.trace(C)
        return C / tr if tr > 0 else C

    def _fit_csp_binary(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes.")

        c0, c1 = classes[0], classes[1]
        X0 = X[y == c0]
        X1 = X[y == c1]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError("Both classes must have at least one trial for CSP.")

        C0 = np.mean([self._compute_cov(x) for x in X0], axis=0)
        C1 = np.mean([self._compute_cov(x) for x in X1], axis=0)

        n_ch = C0.shape[0]
        if self.reg > 0:
            eye = np.eye(n_ch)
            C0 = C0 + self.reg * eye
            C1 = C1 + self.reg * eye

        Cc = C0 + C1
        evals, evecs = np.linalg.eig(np.linalg.solve(Cc, C0))
        evals = np.real(evals)
        evecs = np.real(evecs)
        order = np.argsort(evals)
        evecs = evecs[:, order]

        k2 = self.csp_n_components // 2
        return np.concatenate([evecs[:, :k2], evecs[:, -k2:]], axis=1)

    def _fit_csp_jad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        if len(classes) < 3:
            return self._fit_csp_binary(X, y)

        class_covs = []
        for c in classes:
            Xc = X[y == c]
            if len(Xc) == 0:
                raise ValueError(f"Class {c!r} has no samples; cannot fit JAD CSP.")
            Cc = np.mean([self._compute_cov(x) for x in Xc], axis=0)
            class_covs.append(Cc)
        class_covs = np.stack(class_covs, axis=0)  # (K, C, C)

        n_ch = class_covs.shape[1]
        if self.reg > 0:
            eye = np.eye(n_ch)
            class_covs = class_covs + self.reg * eye[None, :, :]

        C_mean = np.mean(class_covs, axis=0)
        C_mean = 0.5 * (C_mean + C_mean.T)

        evals, evecs = np.linalg.eigh(C_mean)
        evals = np.clip(evals, self.eps, None)
        P = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T  # whitening

        whitened = np.stack([P @ C @ P.T for C in class_covs], axis=0)
        U = self._approx_joint_diag(whitened)
        diag_covs = np.stack([U.T @ C @ U for C in whitened], axis=0)

        scores = np.var(np.diagonal(diag_covs, axis1=1, axis2=2), axis=0)
        order = np.argsort(scores)
        k2 = self.csp_n_components // 2
        keep = np.concatenate([order[:k2], order[-k2:]])

        W = P.T @ U[:, keep]
        return np.real(W)

    def _approx_joint_diag(self, mats: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
        """
        Orthogonal approximate joint diagonalization for symmetric matrices.
        mats: (M, C, C)
        returns U with shape (C, C) such that U.T @ mats[i] @ U are closer to diagonal.
        """
        mats = np.array(mats, dtype=np.float64, copy=True)
        M, C, _ = mats.shape
        U = np.eye(C)

        for _ in range(max_iter):
            delta = 0.0
            for p in range(C - 1):
                for q in range(p + 1, C):
                    g11 = 0.0
                    g12 = 0.0
                    for m in range(M):
                        app = mats[m, p, p]
                        aqq = mats[m, q, q]
                        apq = mats[m, p, q]
                        g11 += apq * (app - aqq)
                        g12 += (app - aqq) ** 2 - 4.0 * apq ** 2

                    phi = 0.5 * np.arctan2(2.0 * g11, g12 + 1e-20)
                    c = np.cos(phi)
                    s = np.sin(phi)

                    if abs(s) < tol:
                        continue

                    delta = max(delta, abs(s))

                    G = np.array([[c, -s], [s, c]])

                    mats[:, [p, q], :] = np.einsum("ab,mbc->mac", G.T, mats[:, [p, q], :])
                    mats[:, :, [p, q]] = np.einsum("mca,ab->mcb", mats[:, :, [p, q]], G)
                    U[:, [p, q]] = U[:, [p, q]] @ G

            if delta < tol:
                break

        return U

    def transform_blocks(self, X: np.ndarray) -> list[np.ndarray]:
        self._validate_transform_input(X)
        if self.csp_filters_ is None or len(self.csp_filters_) == 0:
            raise RuntimeError("CSPLogVar not fitted")

        blocks = []
        for csp_filter in self.csp_filters_:
            Z = np.einsum("nct,ck->nkt", X, csp_filter)
            var = np.var(Z, axis=-1, ddof=0)
            if self.normalize_var:
                var = var / (np.sum(var, axis=1, keepdims=True) + self.eps)
            blocks.append(np.log(var + self.eps) if self.log_var else var)
        return blocks


@dataclass(slots=True)
class FixedFeatureSelector:
    keep_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.keep_]

    def get_support(self, indices: bool = False):
        if indices:
            return self.keep_
        support = np.zeros(np.max(self.keep_) + 1, dtype=bool)
        support[self.keep_] = True
        return support


# @dataclass(slots=True)
# class FBCSP(FeatureExtractor):
#     sampling_rate: int = 250
#     reg: float = 1e-10
#     csp_n_components: int = 4
#     csp_mode: str = "ovo"
#     band_filters: tuple[tuple[float, float], ...] | None = None
#     bandpass_order: int = 4

#     # stage 1: preselect CSP features inside each band/pair block
#     n_prefilter_features_per_block: int = 2

#     # stage 2: final feature selection on concatenated filter-bank features
#     feature_selection: str = "inner_cv"   # "none", "inner_cv"
#     feature_ranker: str = "anova"         # ranking only
#     n_selected_features: int | None = None
#     feature_selection_cv_folds: int = 5
#     candidate_feature_counts: tuple[int, ...] | None = None

#     # feature_selection: str = "anova_cv"   # "none", "anova", "anova_cv"
#     # n_selected_features: int | None = None
#     # feature_selection_cv_folds: int = 5

#     band_csps_: list[CSPLogVar] | None = None
#     block_keep_indices_: list[np.ndarray] | None = None
#     selector_: FixedFeatureSelector | None = None
#     # selector_: SelectKBest | None = None

#     def __post_init__(self):
#         FeatureExtractor.__post_init__(self)

#         if not isinstance(self.sampling_rate, int) or self.sampling_rate <= 0:
#             raise ValueError(f"sampling_rate must be a positive int; got {self.sampling_rate!r}")
#         if not isinstance(self.reg, float) or not np.isfinite(self.reg) or self.reg < 0:
#             raise ValueError(f"reg must be finite and >= 0; got {self.reg!r}")
#         if not isinstance(self.csp_n_components, int) or self.csp_n_components <= 0 or self.csp_n_components % 2 != 0:
#             raise ValueError(f"csp_n_components must be a positive even int; got {self.csp_n_components!r}")
#         if self.csp_mode not in {"ovr", "ovo", "jad"}:
#             raise ValueError("csp_mode must be one of {'ovr', 'ovo', 'jad'}")
#         if not isinstance(self.bandpass_order, int) or self.bandpass_order <= 0:
#             raise ValueError(f"bandpass_order must be a positive int; got {self.bandpass_order!r}")
#         if not isinstance(self.n_prefilter_features_per_block, int) or self.n_prefilter_features_per_block <= 0:
#             raise ValueError("n_prefilter_features_per_block must be a positive int")
#         if self.feature_selection not in {"none", "anova", "anova_cv"}:
#             raise ValueError("feature_selection must be one of {'none','anova','anova_cv'}")
#         if self.n_selected_features is not None and (not isinstance(self.n_selected_features, int) or self.n_selected_features <= 0):
#             raise ValueError("n_selected_features must be a positive int or None")
#         if not isinstance(self.feature_selection_cv_folds, int) or self.feature_selection_cv_folds < 2:
#             raise ValueError("feature_selection_cv_folds must be an int >= 2")

#         if self.band_filters is None:
#             self.band_filters = make_schirrmeister_hgd_filter_bank(sampling_rate=self.sampling_rate)

#         validated = []
#         nyq = self.sampling_rate / 2.0
#         for band in self.band_filters:
#             if not isinstance(band, tuple) or len(band) != 2:
#                 raise ValueError(f"Each band must be a (lo, hi) tuple; got {band!r}")
#             lo, hi = band
#             if not np.isfinite(lo) or lo < 0:
#                 raise ValueError(f"Invalid lo {lo!r}")
#             if not np.isfinite(hi) or hi <= 0 or hi >= nyq:
#                 raise ValueError(f"Invalid hi {hi!r} for Nyquist {nyq:.6g}")
#             if lo >= hi:
#                 raise ValueError(f"Band low cutoff must be < high cutoff; got {(lo, hi)!r}")
#             validated.append((float(lo), float(hi)))
#         self.band_filters = tuple(validated)

#     def clone(self) -> "FBCSP":
#         return FBCSP(
#             sampling_rate=self.sampling_rate,
#             reg=self.reg,
#             csp_n_components=self.csp_n_components,
#             csp_mode=self.csp_mode,
#             band_filters=self.band_filters,
#             bandpass_order=self.bandpass_order,
#             n_prefilter_features_per_block=self.n_prefilter_features_per_block,
#             feature_selection=self.feature_selection,
#             n_selected_features=self.n_selected_features,
#             feature_selection_cv_folds=self.feature_selection_cv_folds,
#             eps=self.eps,
#             normalize_var=self.normalize_var,
#             log_var=self.log_var,
#         )

#     def _filter_band(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:
#         return preprocess.filter_band_new(
#             X,
#             sampling_rate=self.sampling_rate,
#             lo=lo,
#             hi=hi,
#             order=self.bandpass_order,
#         )

#     def _power_ratio_scores(self, F_block: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
#         pos = F_block[y_bin == 1]
#         neg = F_block[y_bin == 0]
#         if len(pos) == 0 or len(neg) == 0:
#             raise ValueError("Binary split for CSP preselection has an empty class")
#         # F_block is logvar by default, so exponentiate back to variance-like scale
#         V = np.exp(F_block) if self.log_var else F_block
#         pos_mean = np.mean(V[y_bin == 1], axis=0)
#         neg_mean = np.mean(V[y_bin == 0], axis=0)
#         ratio = np.maximum(pos_mean, neg_mean) / np.maximum(np.minimum(pos_mean, neg_mean), self.eps)
#         return ratio

#     def _rank_features(self, X_feats: np.ndarray, y: np.ndarray) -> np.ndarray:
#         scores, _ = f_classif(X_feats, y)
#         scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
#         return np.argsort(scores)[::-1]

#     def _make_final_selector(self, X_feats: np.ndarray, y: np.ndarray):
#         if self.feature_selection == "none":
#             return None

#         cv = StratifiedKFold(
#             n_splits=self.feature_selection_cv_folds,
#             shuffle=True,
#             random_state=0,
#         )
#         candidate_counts = self.candidate_feature_counts
#         if candidate_counts is None:
#             candidate_counts = tuple(range(4, min(X_feats.shape[1], 80) + 1, 4))
#         best_k = None
#         best_score = -np.inf

#         for k in candidate_counts:
#             fold_scores = []
#             for tr_idx, va_idx in cv.split(X_feats, y):
#                 Xtr, ytr = X_feats[tr_idx], y[tr_idx]
#                 Xva, yva = X_feats[va_idx], y[va_idx]
#                 order = self._rank_features(Xtr, ytr)
#                 keep = np.sort(order[:k])
#                 clf = classifier.RLDAClassifier().clone()
#                 clf.fit(Xtr[:, keep], ytr)
#                 yhat = clf.predict(Xva[:, keep])
#                 fold_scores.append(np.mean(yhat == yva))

#             score = float(np.mean(fold_scores))
#             if score > best_score:
#                 best_score = score
#                 best_k = k

#         final_order = self._rank_features(X_feats, y)
#         final_keep = np.sort(final_order[:best_k])
#         return FixedFeatureSelector(keep_=final_keep)

#     def _select_block_features(self, csp: CSPLogVar, X_band: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         blocks = csp.transform_blocks(X_band)
#         if csp.mode != "ovo":
#             raise NotImplementedError(f"FBCSP block preselection is currently implemented only for csp_mode='ovo', got {csp.mode!r}")
#         keep_blocks = []
#         keep_idx = []
#         for block, label in zip(blocks, csp.filter_labels_):
#             c0, c1 = label
#             mask = (y == c0) | (y == c1)
#             y_bin = (y[mask] == c1).astype(np.int8)
#             scores = self._power_ratio_scores(block[mask], y_bin)
#             order = np.argsort(scores)[::-1]
#             k = min(self.n_prefilter_features_per_block, block.shape[1])
#             chosen = np.sort(order[:k])
#             keep_blocks.append(block[:, chosen])
#             keep_idx.append(chosen)
#         return np.concatenate(keep_blocks, axis=1), tuple(keep_idx)

#     def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "FBCSP":
#         self._validate_fit_input(X, y)
#         if len(np.unique(y)) < 2:
#             raise ValueError("FBCSP requires at least 2 classes")

#         self.band_csps_ = []
#         self.block_keep_indices_ = []
#         feat_blocks = []

#         for lo, hi in self.band_filters:
#             X_band = self._filter_band(X, lo, hi)
#             csp = CSPLogVar(
#                 reg=self.reg,
#                 csp_n_components=self.csp_n_components,
#                 eps=self.eps,
#                 normalize_var=self.normalize_var,
#                 log_var=self.log_var,
#                 mode=self.csp_mode
#             )
#             csp.fit(X_band, y)
#             F_band, keep_idx = self._select_block_features(csp, X_band, y)
#             self.band_csps_.append(csp)
#             self.block_keep_indices_.append(keep_idx)
#             feat_blocks.append(F_band)

#         X_feats = np.concatenate(feat_blocks, axis=1)
#         self.selector_ = self._make_final_selector(X_feats, y)
#         return self

#     def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
#         self._validate_transform_input(X)
#         if self.band_csps_ is None or self.block_keep_indices_ is None:
#             raise RuntimeError("FBCSP not fitted")

#         feat_blocks = []
#         for (lo, hi), csp, keep_idx in zip(self.band_filters, self.band_csps_, self.block_keep_indices_):
#             X_band = self._filter_band(X, lo, hi)
#             blocks = csp.transform_blocks(X_band)
#             chosen = [block[:, idx] for block, idx in zip(blocks, keep_idx)]
#             feat_blocks.append(np.concatenate(chosen, axis=1))

#         X_feats = np.concatenate(feat_blocks, axis=1)
#         if self.selector_ is not None:
#             X_feats = self.selector_.transform(X_feats)
#         return X_feats


@dataclass(slots=True)
class FBCSP(FeatureExtractor):
    sampling_rate: int = 250
    reg: float = 1e-10
    csp_n_components: int = 4
    csp_mode: str = "ovo"
    band_filters: tuple[tuple[float, float], ...] | None = None
    bandpass_order: int = 4

    # stage 1
    n_prefilter_pairs_per_block: int = 2

    # exact stage 2 from published code structure
    n_selected_features: int | None = 40
    forward_steps: int = 2
    backward_steps: int = 1
    stop_when_no_improvement: bool = False

    band_csps_: list[CSPLogVar] | None = None
    block_features_: list[list[np.ndarray]] | None = None
    selected_filters_per_band_: list[list[int]] | None = None

    def __post_init__(self):
        FeatureExtractor.__post_init__(self)
        if self.csp_mode != "ovo":
            raise ValueError("Exact Schirrmeister-style FBCSP stage-2 here assumes csp_mode='ovo'")
        if self.n_selected_features is not None:
            if not isinstance(self.n_selected_features, int) or self.n_selected_features <= 0 or self.n_selected_features % 2 != 0:
                raise ValueError("n_selected_features must be a positive even int")
        if not isinstance(self.forward_steps, int) or self.forward_steps <= 0:
            raise ValueError("forward_steps must be a positive int")
        if not isinstance(self.backward_steps, int) or self.backward_steps < 0:
            raise ValueError("backward_steps must be an int >= 0")
        if self.band_filters is None:
            self.band_filters = make_schirrmeister_hgd_filter_bank(
                sampling_rate=self.sampling_rate,
                low_start_hz=0.0,
                high_stop_hz=122.0,
            )
        if not isinstance(self.n_prefilter_pairs_per_block, int) or self.n_prefilter_pairs_per_block <= 0:
            raise ValueError("n_prefilter_pairs_per_block must be a positive int")
        if self.n_prefilter_pairs_per_block > self.csp_n_components // 2:
            raise ValueError("n_prefilter_pairs_per_block cannot exceed csp_n_components // 2")

    def clone(self) -> "FBCSP":
        return FBCSP(
            sampling_rate=self.sampling_rate,
            reg=self.reg,
            csp_n_components=self.csp_n_components,
            csp_mode=self.csp_mode,
            band_filters=self.band_filters,
            bandpass_order=self.bandpass_order,
            n_prefilter_pairs_per_block=self.n_prefilter_pairs_per_block,
            n_selected_features=self.n_selected_features,
            forward_steps=self.forward_steps,
            backward_steps=self.backward_steps,
            stop_when_no_improvement=self.stop_when_no_improvement,
            eps=self.eps,
            normalize_var=self.normalize_var,
            log_var=self.log_var,
        )

    def _filter_band(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return preprocess.filter_band_new(
            X,
            sampling_rate=self.sampling_rate,
            lo=lo,
            hi=hi,
            order=self.bandpass_order,
        )

    def _power_ratio_scores(self, F_block: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
        V = np.exp(F_block) if self.log_var else F_block
        pos_mean = np.mean(V[y_bin == 1], axis=0)
        neg_mean = np.mean(V[y_bin == 0], axis=0)
        return np.maximum(pos_mean, neg_mean) / np.maximum(np.minimum(pos_mean, neg_mean), self.eps)

    def _select_block_features(self, csp: CSPLogVar, X_band: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        blocks = csp.transform_blocks(X_band)
        out = []
        for block, (c0, c1) in zip(blocks, csp.filter_labels_):
            mask = (y == c0) | (y == c1)
            y_bin = (y[mask] == c1).astype(np.int8)
            scores = self._power_ratio_scores(block[mask], y_bin)
            n_feats = block.shape[1]
            if n_feats % 2 != 0:
                raise ValueError(f"Expected even number of CSP features, got {n_feats}")
            n_pairs = n_feats // 2
            pair_scores = np.empty(n_pairs, dtype=np.float64)
            for i in range(n_pairs):
                pair_scores[i] = max(scores[i], scores[-(i + 1)])
            n_keep_pairs = min(self.n_prefilter_pairs_per_block, n_pairs)
            best_pairs = np.sort(np.argsort(pair_scores)[::-1][:n_keep_pairs])
            left_idx = best_pairs
            right_idx = np.array([n_feats - 1 - i for i in best_pairs], dtype=int)
            chosen = np.concatenate([left_idx, right_idx])
            out.append(block[:, chosen])
        return out

    @staticmethod
    def _collect_features_for_filter_selection(features_by_band: list[np.ndarray], filters_per_band: list[int]) -> np.ndarray:
        parts = []
        for F_band, n in zip(features_by_band, filters_per_band):
            if n <= 0:
                continue
            parts.append(np.concatenate([F_band[:, :n], F_band[:, -n:]], axis=1))
        if not parts:
            return np.zeros((features_by_band[0].shape[0], 0), dtype=np.float64)
        return np.concatenate(parts, axis=1)

    @staticmethod
    def _cross_validate_rlda_binary(F: np.ndarray, y: np.ndarray) -> float:
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        accs = []
        classes = np.unique(y)
        c1 = np.max(classes)

        for tr_idx, va_idx in cv.split(F, y):
            Ftr, Fva = F[tr_idx], F[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(Ftr, ytr)
            out = np.asarray(clf.decision_function(Fva)).reshape(-1)
            pred = np.where(out >= 0, c1, np.min(classes))
            accs.append(np.mean(pred == yva))

        return float(np.mean(accs))

    def _select_best_filters_best_filterbands(self, features_by_band: list[np.ndarray], y_pair: np.ndarray) -> list[int]:
        assert self.n_selected_features is not None
        n_bands = len(features_by_band)
        n_pairs_per_band = features_by_band[0].shape[1] // 2
        selected = [0] * n_bands
        last_best = -1.0

        while True:
            for _ in range(self.forward_steps):
                best_acc = -1.0
                best_sel = None
                for band_i in range(n_bands):
                    cand = selected.copy()
                    if cand[band_i] == n_pairs_per_band:
                        continue
                    cand[band_i] += 1
                    F = self._collect_features_for_filter_selection(features_by_band, cand)
                    acc = self._cross_validate_rlda_binary(F, y_pair)
                    if acc > best_acc:
                        best_acc = acc
                        best_sel = cand
                selected = best_sel

            for _ in range(self.backward_steps):
                best_acc = -1.0
                best_sel = None
                for band_i in range(n_bands):
                    cand = selected.copy()
                    if cand[band_i] == 0:
                        continue
                    cand[band_i] -= 1
                    F = self._collect_features_for_filter_selection(features_by_band, cand)
                    acc = self._cross_validate_rlda_binary(F, y_pair)
                    if acc > best_acc:
                        best_acc = acc
                        best_sel = cand
                if best_sel is not None:
                    selected = best_sel

            max_available = 2 * n_bands * n_pairs_per_band
            finished = 2 * np.sum(selected) >= min(self.n_selected_features, max_available)
            if self.stop_when_no_improvement:
                finished = finished or (best_acc <= last_best)
            last_best = best_acc
            if finished:
                return selected

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "FBCSP":
        self._validate_fit_input(X, y)
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("FBCSP requires at least 2 classes")

        self.band_csps_ = []
        self.block_features_ = []
        for lo, hi in self.band_filters:
            X_band = self._filter_band(X, lo, hi)
            csp = CSPLogVar(
                reg=self.reg,
                csp_n_components=self.csp_n_components,
                eps=self.eps,
                normalize_var=self.normalize_var,
                log_var=self.log_var,
                mode=self.csp_mode,
            )
            csp.fit(X_band, y)
            self.band_csps_.append(csp)
            self.block_features_.append(self._select_block_features(csp, X_band, y))

        self.selected_filters_per_band_ = []
        pair_labels = list(combinations(classes, 2))
        for pair_i, (c0, c1) in enumerate(pair_labels):
            mask = (y == c0) | (y == c1)
            y_pair = y[mask]
            per_band_pair_features = [band_blocks[pair_i][mask] for band_blocks in self.block_features_]
            selected = self._select_best_filters_best_filterbands(per_band_pair_features, y_pair)
            self.selected_filters_per_band_.append(selected)

        return self

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._validate_transform_input(X)
        if self.band_csps_ is None or self.selected_filters_per_band_ is None:
            raise RuntimeError("FBCSP not fitted")

        classes = self.band_csps_[0].filter_labels_
        out_blocks = []

        transformed_by_band = []
        for (lo, hi), csp in zip(self.band_filters, self.band_csps_):
            X_band = self._filter_band(X, lo, hi)
            transformed_by_band.append(csp.transform_blocks(X_band))

        for pair_i, selected in enumerate(self.selected_filters_per_band_):
            pair_parts = []
            for band_blocks, n in zip(transformed_by_band, selected):
                if n <= 0:
                    continue
                F_band = band_blocks[pair_i]
                pair_parts.append(np.concatenate([F_band[:, :n], F_band[:, -n:]], axis=1))
            out_blocks.append(np.concatenate(pair_parts, axis=1))

        return np.concatenate(out_blocks, axis=1)


def make_uniform_filter_bank(
    start_hz: float,
    stop_hz: float,
    band_width_hz: float,
    *,
    sampling_rate: int,
    include_low_start: bool = True,
    nyq_margin_hz: float = 0.5,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(sampling_rate, int) or sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be a positive int; got {sampling_rate!r}")
    if not np.isfinite(start_hz) or start_hz < 0:
        raise ValueError(f"start_hz must be finite and >= 0; got {start_hz!r}")
    if not np.isfinite(stop_hz) or stop_hz <= 0:
        raise ValueError(f"stop_hz must be finite and > 0; got {stop_hz!r}")
    if not np.isfinite(band_width_hz) or band_width_hz <= 0:
        raise ValueError(f"band_width_hz must be finite and > 0; got {band_width_hz!r}")
    if start_hz >= stop_hz:
        raise ValueError(f"start_hz must be < stop_hz; got {start_hz!r} >= {stop_hz!r}")

    nyq = sampling_rate / 2.0
    max_hi = nyq - nyq_margin_hz
    if max_hi <= 0:
        raise ValueError(f"Invalid Nyquist margin {nyq_margin_hz!r} for sampling_rate={sampling_rate}")

    stop_hz = min(float(stop_hz), max_hi)

    bands = []
    lo = float(start_hz)
    while lo < stop_hz - 1e-12:
        hi = min(lo + float(band_width_hz), stop_hz)

        if lo == 0.0 and not include_low_start:
            lo = hi
            continue

        bands.append((lo, hi))
        lo = hi

    if len(bands) == 0:
        raise ValueError("No valid bands generated")

    return tuple(bands)


def make_schirrmeister_hgd_filter_bank(
    *,
    sampling_rate: int,
    low_start_hz: float = 4.0,
    high_stop_hz: float = 122.0,
    nyq_margin_hz: float = 0.5,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(sampling_rate, int) or sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be a positive int; got {sampling_rate!r}")
    if not np.isfinite(low_start_hz) or low_start_hz < 0:
        raise ValueError(f"low_start_hz must be finite and >= 0; got {low_start_hz!r}")
    if not np.isfinite(high_stop_hz) or high_stop_hz <= low_start_hz:
        raise ValueError(f"high_stop_hz must be finite and > low_start_hz; got {high_stop_hz!r}")

    nyq = sampling_rate / 2.0
    max_hi = nyq - nyq_margin_hz
    stop = min(float(high_stop_hz), max_hi)

    bands = []

    # up to 13 Hz: width 6, overlap 3 => step 3
    lo = float(low_start_hz)
    while lo < 13.0:
        hi = lo + 6.0
        if hi > stop:
            break
        bands.append((lo, hi))
        lo += 3.0

    # above 10 Hz: width 8, overlap 4 => step 4
    lo = 10.0
    while lo < stop:
        hi = lo + 8.0
        if hi > stop:
            break
        bands.append((lo, hi))
        lo += 4.0

    # de-duplicate while preserving order
    out = []
    seen = set()
    for band in bands:
        key = (round(band[0], 10), round(band[1], 10))
        if key not in seen:
            seen.add(key)
            out.append((float(band[0]), float(band[1])))

    if len(out) == 0:
        raise ValueError("No valid bands generated")

    return tuple(out)


class FeatureExtractorType(Enum):
    CSP_LOGVAR = CSPLogVar
    FBCSP = FBCSP

    @property
    def cls(self) -> Type[FeatureExtractor]:
        return self.value
