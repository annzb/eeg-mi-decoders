from typing import Optional, Sequence, Tuple, Callable, Dict, Any

import numpy as np
from scipy.signal import butter, filtfilt


PreprocessPipeline = Tuple[Sequence[Callable], Sequence[Dict[str, Any]]]


def validate_preprocess_pipeline(pipeline: PreprocessPipeline) -> None:
    funcs, kwargs_seq = pipeline
    if not hasattr(funcs, '__len__') or not hasattr(kwargs_seq, '__len__'):
        raise TypeError(f"preprocessing_funcs and preprocessing_kwargs must be sequences, got {type(funcs)} and {type(kwargs_seq)}")
    if len(funcs) != len(kwargs_seq):
        raise ValueError(f"preprocessing_funcs and preprocessing_kwargs must have same length, got {len(funcs)} and {len(kwargs_seq)}")
    for func, kwargs in zip(funcs, kwargs_seq):
        if not callable(func):
            raise TypeError(f"preprocessing func {func.__name__} must be callable, got {type(func)}")
        if not isinstance(kwargs, dict):
            raise TypeError(f"preprocessing kwargs for {func.__name__} must be dict, got {type(kwargs)}")
        for key in kwargs.keys():
            if not isinstance(key, str):
                raise TypeError(f"preprocessing kwargs for {func.__name__} must be dict, got {type(kwargs)}")


def filter_highpass(X, sampling_rate, cutoff=0.5, order=4, **kwargs) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")
    if cutoff <= 0:
        raise ValueError("cutoff frequency must be > 0")
    b, a = butter(order, cutoff, btype="highpass", fs=sampling_rate)
    return filtfilt(b, a, X, axis=-1)


def filter_band(X, sampling_rate, lo=8.0, hi=30.0, order=4, **kwargs) -> np.ndarray:
    b, a = butter(order, [lo, hi], btype="bandpass", fs=sampling_rate)
    return filtfilt(b, a, X, axis=-1)


def common_average_reference(X: np.ndarray, exclude_channels: Optional[Sequence[int]] = None, **kwargs) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")

    if exclude_channels is None or len(exclude_channels) == 0:
        ref = X.mean(axis=1, keepdims=True)
        return X - ref

    exclude_channels = np.asarray(exclude_channels, dtype=int)
    n_ch = X.shape[1]
    if np.any(exclude_channels < 0) or np.any(exclude_channels >= n_ch):
        raise ValueError(f"exclude_channels contains invalid channel indices (n_ch={n_ch})")

    mask = np.ones(n_ch, dtype=bool)
    mask[exclude_channels] = False
    if not np.any(mask):
        raise ValueError("All channels excluded; cannot compute CAR")

    ref = X[:, mask, :].mean(axis=1, keepdims=True)  # (N, 1, Time)
    return X - ref


def channel_log_var(X: np.ndarray, eps: float = 1e-12, normalize_var: bool = True, log_var: bool = True, **kwargs) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")
    if not isinstance(eps, float) or not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and > 0; got {eps!r}")
    if not isinstance(normalize_var, bool):
        raise ValueError(f"normalize_var must be a bool, got {normalize_var!r}")
    if not isinstance(log_var, bool):
        raise ValueError(f"log_var must be a bool, got {log_var!r}")

    var = np.var(X, axis=-1, ddof=0)  # (N, Ch)
    if normalize_var:
        var = var / (np.sum(var, axis=1, keepdims=True) + eps)  # (N, Ch)
    return np.log(var + eps) if log_var else var  # (N, Ch)


def fta_cartesian(X: np.ndarray, srate: float = 1, n_bins: int = 5, max_freq_hz: float = 5.0, **kwargs) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")
    if not isinstance(srate, (int, float)) or not np.isfinite(srate) or srate <= 0:
        raise ValueError(f"srate must be finite and > 0; got {srate!r}")
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError(f"n_bins must be a positive int; got {n_bins!r}")
    if not isinstance(max_freq_hz, (int, float)) or not np.isfinite(max_freq_hz) or max_freq_hz <= 0:
        raise ValueError(f"max_freq_hz must be finite and > 0; got {max_freq_hz!r}")

    n_samples, n_channels, n_frames = X.shape
    n_rfft = n_frames // 2 + 1
    if n_bins > n_rfft:
        raise ValueError(f"n_bins must be <= n_frames//2+1 ({n_rfft}); got {n_bins!r}")
    Z = np.fft.rfft(X, n=n_frames, axis=-1)
    freqs = np.fft.rfftfreq(n_frames, d=1.0 / float(srate))
    if freqs[n_bins - 1] > float(max_freq_hz) + 1e-9:
        raise ValueError(
            f"n_bins={n_bins} exceeds max_freq_hz={max_freq_hz!r} for n_frames={n_frames}, srate={srate}. "
            f"Highest kept bin is {freqs[n_bins - 1]:.6g} Hz."
        )
    Zk = Z[:, :, : n_bins]
    a0 = Zk[:, :, 0].real
    a = Zk[:, :, 1:]
    re = a.real
    im = a.imag
    cart = np.empty((n_samples, n_channels, 1 + 2 * (n_bins - 1)), dtype=np.float64)
    cart[:, :, 0] = a0
    if n_bins > 1:
        cart[:, :, 1::2] = re
        cart[:, :, 2::2] = im
    F = cart.reshape(n_samples, -1)
    np.ascontiguousarray(F, dtype=np.float64) 
    return F
