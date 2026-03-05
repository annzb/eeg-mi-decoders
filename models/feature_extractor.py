from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Type

import numpy as np


@dataclass(slots=True)
class FeatureExtractor(ABC):
    eps: float = 1e-12
    normalize_var: bool = True
    log_var: bool = True
    
    @abstractmethod
    def clone(self) -> "FeatureExtractor": ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "FeatureExtractor": ...

    @abstractmethod
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray: ...

    def __post_init__(self):
        if not isinstance(self.eps, float) or not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be finite and > 0; got {self.eps!r}")
        if not isinstance(self.normalize_var, bool):
            raise ValueError(f"normalize_var must be a bool, got {self.normalize_var!r}")
        if not isinstance(self.log_var, bool):
            raise ValueError(f"log_var must be a bool, got {self.log_var!r}")

    def _validate_fit_input(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError(f"Expected y to have shape (N,); got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same length; got {X.shape[0]} vs {y.shape[0]}")

    def _validate_transform_input(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")


@dataclass(slots=True)
class ChannelLogVar(FeatureExtractor):
    def clone(self) -> "ChannelLogVar":
        return ChannelLogVar(normalize_var=self.normalize_var, log_var=self.log_var, eps=self.eps)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "ChannelLogVar":
        return self

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._validate_transform_input(X)
        var = np.var(X, axis=-1, ddof=0)
        if self.normalize_var:
            var = var / (np.sum(var, axis=1, keepdims=True) + self.eps)
        return np.log(var + self.eps) if self.log_var else var


@dataclass(slots=True)
class CSPLogVar(FeatureExtractor):
    reg: float = 1e-10
    csp_n_components: int = 4

    csp_filters_: list[np.ndarray] | None = None

    def __post_init__(self):
        FeatureExtractor.__post_init__(self)
        if not isinstance(self.reg, float) or not np.isfinite(self.reg) or self.reg < 0:
            raise ValueError(f"reg must be finite and >= 0; got {self.reg!r}")
        if not isinstance(self.csp_n_components, int) or self.csp_n_components <= 0 or self.csp_n_components % 2 != 0:
            raise ValueError(f"csp_n_components must be a positive even int; got {self.csp_n_components!r}")

    def clone(self) -> "CSPLogVar":
        return CSPLogVar(csp_n_components=self.csp_n_components, reg=self.reg, normalize_var=self.normalize_var, log_var=self.log_var, eps=self.eps)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "CSPLogVar":
        self._validate_fit_input(X, y)
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("CSPLogVar requires at least 2 classes.")

        self.csp_filters_ = []
        if len(classes) == 2:
            self.csp_filters_.append(self._fit_csp_binary(X, y))
            return self
        
        for c in classes:
            y_bin = (y == c).astype(np.int8)
            if np.all(y_bin == 0) or np.all(y_bin == 1):
                raise ValueError(f"Class {c!r} has no samples; cannot fit CSP.")
            self.csp_filters_.append(self._fit_csp_binary(X, y_bin))
        return self

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
        X0 = X[y == c0]; X1 = X[y == c1]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError("Both classes must have at least one trial for CSP.")
        C0 = np.mean([self._compute_cov(x) for x in X0], axis=0)
        C1 = np.mean([self._compute_cov(x) for x in X1], axis=0)
        n_ch = C0.shape[0]
        if self.reg and self.reg > 0:
            C0 = C0 + self.reg * np.eye(n_ch)
            C1 = C1 + self.reg * np.eye(n_ch)
        Cc = C0 + C1
        evals, evecs = np.linalg.eig(np.linalg.solve(Cc, C0))
        evals, evecs = np.real(evals), np.real(evecs)
        evecs = evecs[:, np.argsort(evals)]
        k2 = self.csp_n_components // 2
        return np.concatenate([evecs[:, :k2], evecs[:, -k2:]], axis=1)


@dataclass(slots=True)
class FTACartesian(FeatureExtractor):
    n_bins: int = 5
    max_freq_hz: float = 5.0

    def __post_init__(self):
        FeatureExtractor.__post_init__(self)
        if not isinstance(self.n_bins, int) or self.n_bins <= 0:
            raise ValueError(f"n_bins must be a positive int; got {self.n_bins!r}")
        if not isinstance(self.max_freq_hz, (int, float)) or not np.isfinite(self.max_freq_hz) or self.max_freq_hz <= 0:
            raise ValueError(f"max_freq_hz must be finite and > 0; got {self.max_freq_hz!r}")

    def clone(self) -> "FTACartesian":
        return FTACartesian(
            n_bins=self.n_bins,
            max_freq_hz=self.max_freq_hz,
            normalize_var=self.normalize_var,
            log_var=self.log_var,
            eps=self.eps,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "FTACartesian":
        return self

    def transform(self, X: np.ndarray, srate: float = 1, **kwargs) -> np.ndarray:
        self._validate_transform_input(X)
        if not isinstance(srate, (int, float)) or not np.isfinite(srate) or srate <= 0:
            raise ValueError(f"srate must be finite and > 0; got {srate!r}")

        n_samples, n_channels, n_frames = X.shape
        n_rfft = n_frames // 2 + 1
        if self.n_bins > n_rfft:
            raise ValueError(f"n_bins must be <= n_frames//2+1 ({n_rfft}); got {self.n_bins!r}")

        Z = np.fft.rfft(X, n=n_frames, axis=-1)
        freqs = np.fft.rfftfreq(n_frames, d=1.0 / float(srate))
        if freqs[self.n_bins - 1] > float(self.max_freq_hz) + 1e-9:
            raise ValueError(
                f"n_bins={self.n_bins} exceeds max_freq_hz={self.max_freq_hz!r} for n_frames={n_frames}, srate={srate}. "
                f"Highest kept bin is {freqs[self.n_bins - 1]:.6g} Hz."
            )
        Zk = Z[:, :, : self.n_bins]
        a0 = Zk[:, :, 0].real
        a = Zk[:, :, 1:]
        re = a.real
        im = a.imag
        cart = np.empty((n_samples, n_channels, 1 + 2 * (self.n_bins - 1)), dtype=np.float64)
        cart[:, :, 0] = a0
        if self.n_bins > 1:
            cart[:, :, 1::2] = re
            cart[:, :, 2::2] = im
        return cart.reshape(n_samples, -1)


class FeatureExtractorType(Enum):
    CSP_LOGVAR = CSPLogVar
    CHANNEL_LOGVAR = ChannelLogVar
    FTA_CARTESIAN = FTACartesian

    @property
    def cls(self) -> Type[FeatureExtractor]:
        return self.value
