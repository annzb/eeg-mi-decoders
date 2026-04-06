from math import gcd
from typing import Optional, Sequence, Tuple, Callable, Dict, Any
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


PreprocessPipeline = Tuple[Sequence[Callable], Sequence[Dict[str, Any]]]


@dataclass
class PreprocessResult:
    X: np.ndarray
    Y_mask: np.ndarray | None = None
    sampling_rate: int | None = None
    channel_mask: np.ndarray | None = None


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


def filter_band_new(X, sampling_rate, lo=8.0, hi=30.0, order=4, **kwargs) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")
    if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be > 0; got {sampling_rate!r}")
    if not np.isfinite(lo) or lo < 0:
        raise ValueError(f"lo must be finite and >= 0; got {lo!r}")
    if not np.isfinite(hi) or hi <= 0:
        raise ValueError(f"hi must be finite and > 0; got {hi!r}")
    if lo >= hi:
        raise ValueError(f"lo must be < hi; got {(lo, hi)!r}")

    nyq = sampling_rate / 2.0
    if hi >= nyq:
        raise ValueError(f"hi must be below Nyquist ({nyq:.6g} Hz); got {hi!r}")
    if lo == 0:
        b, a = butter(order, hi, btype="lowpass", fs=sampling_rate)
    else:
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


def resample(X: np.ndarray, sampling_rate: int, target_srate: int, **kwargs) -> PreprocessResult:
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")
    if not isinstance(target_srate, int) or target_srate <= 0:
        raise ValueError(f"target_srate must be a positive int, got {target_srate!r}")
    if sampling_rate == target_srate:
        return PreprocessResult(X=X, sampling_rate=target_srate)
    g = gcd(sampling_rate, target_srate)
    X_resampled = resample_poly(X, target_srate // g, sampling_rate // g, axis=-1)
    return PreprocessResult(X=X_resampled, sampling_rate=target_srate)


def filter_artifacts(
    X: np.ndarray,
    sampling_rate: int,
    threshold_uv: float = 800.0,
    target_interval_s: tuple[float, float] | None = None,
    epoch_start_s: float = 0.0,
    **kwargs
) -> PreprocessResult:
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")
    if not isinstance(sampling_rate, int) or sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be a positive int, got {sampling_rate!r}")
    if not np.isfinite(threshold_uv) or threshold_uv <= 0:
        raise ValueError(f"threshold_uv must be > 0, got {threshold_uv!r}")

    X_target = X
    if target_interval_s is not None:
        if (
            not isinstance(target_interval_s, tuple)
            or len(target_interval_s) != 2
            or not all(isinstance(v, (int, float)) for v in target_interval_s)
        ):
            raise ValueError(f"target_interval_s must be a (start, end) tuple, got {target_interval_s!r}")

        t0, t1 = float(target_interval_s[0]), float(target_interval_s[1])
        if t0 >= t1:
            raise ValueError(f"target_interval_s start must be < end, got {target_interval_s!r}")

        epoch_end_s = epoch_start_s + X.shape[-1] / float(sampling_rate)
        if t0 < epoch_start_s or t1 > epoch_end_s:
            raise ValueError(
                f"target_interval_s={target_interval_s!r} is outside epoch range "
                f"[{epoch_start_s:.6g}, {epoch_end_s:.6g}]"
            )
        start = int(round((t0 - epoch_start_s) * sampling_rate))
        end = int(round((t1 - epoch_start_s) * sampling_rate))
        if start < 0 or end > X.shape[-1] or start >= end:
            raise ValueError(
                f"Computed invalid target slice [{start}:{end}] for X with {X.shape[-1]} samples"
            )
        X_target = X[:, :, start:end]

    Y_mask = np.max(np.abs(X_target), axis=(1, 2)) < threshold_uv
    return PreprocessResult(X=X[Y_mask], Y_mask=Y_mask)

# def filter_artifacts(X: np.ndarray, sampling_rate: int, threshold_uv: float = 800.0, **kwargs) -> PreprocessResult:
#     if X.ndim != 3:
#         raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")
#     if threshold_uv <= 0:
#         raise ValueError(f"threshold_uv must be > 0, got {threshold_uv!r}")
#     mask = np.max(np.abs(X), axis=(1, 2)) < threshold_uv
#     return PreprocessResult(X=X[mask], Y_mask=mask)


def select_channels(
    X: np.ndarray,
    sampling_rate: int,
    channels: Sequence[int | str],
    channel_names: Optional[Sequence[str]] = None,
    **kwargs
) -> PreprocessResult:
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")
    if not hasattr(channels, '__len__') or len(channels) == 0:
        raise ValueError("channels must be a non-empty sequence")

    n_ch = X.shape[1]

    if all(isinstance(ch, (int, np.integer)) for ch in channels):
        idx = np.asarray(channels, dtype=int)
        if np.any(idx < 0) or np.any(idx >= n_ch):
            raise ValueError(f"Channel indices out of range for n_ch={n_ch}: {idx}")
    elif all(isinstance(ch, str) for ch in channels):
        if channel_names is None:
            raise ValueError("channel_names must be provided when selecting channels by name")
        name_to_idx = {name: i for i, name in enumerate(channel_names)}
        missing = [ch for ch in channels if ch not in name_to_idx]
        if missing:
            raise ValueError(f"Unknown channel names: {missing}")
        idx = np.asarray([name_to_idx[ch] for ch in channels], dtype=int)
    else:
        raise ValueError("channels must contain either only ints or only strs")

    channel_mask = np.zeros(n_ch, dtype=bool)
    channel_mask[idx] = True
    return PreprocessResult(X=X[:, idx, :], channel_mask=channel_mask)


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


def exp_running_standardize(
    X: np.ndarray,
    sampling_rate: int,
    factor_new: float = 1e-3,
    init_block_size: int = 1000,
    eps: float = 1e-4,
    **kwargs
) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")
    if not np.isfinite(factor_new) or factor_new <= 0 or factor_new > 1:
        raise ValueError(f"factor_new must be in (0, 1], got {factor_new!r}")
    if not isinstance(init_block_size, int) or init_block_size <= 0:
        raise ValueError(f"init_block_size must be a positive int, got {init_block_size!r}")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps!r}")

    X = np.asarray(X, dtype=np.float64)
    N, C, T = X.shape
    X_std = np.empty_like(X)

    for n in range(N):
        trial = X[n].T  # (T, C)
        out = np.empty_like(trial)

        init_n = min(init_block_size, T)
        init_mean = np.mean(trial[:init_n], axis=0)
        init_std = np.std(trial[:init_n], axis=0)
        out[:init_n] = (trial[:init_n] - init_mean) / np.maximum(init_std, eps)

        if init_n < T:
            mean_t = init_mean.copy()
            var_t = np.var(trial[:init_n], axis=0)

            for t in range(init_n, T):
                x_t = trial[t]
                mean_t = (1.0 - factor_new) * mean_t + factor_new * x_t
                demeaned = x_t - mean_t
                var_t = (1.0 - factor_new) * var_t + factor_new * (demeaned ** 2)
                out[t] = demeaned / np.maximum(np.sqrt(var_t), eps)

        X_std[n] = out.T

    return X_std
