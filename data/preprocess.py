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
