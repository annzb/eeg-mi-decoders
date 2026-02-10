from typing import Optional, Sequence

import numpy as np
from scipy.signal import butter, filtfilt


def filter_highpass(X, sampling_rate, cutoff=0.5, order=4):
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, Ch, Time), got {X.shape}")
    if cutoff <= 0:
        raise ValueError("cutoff frequency must be > 0")
    b, a = butter(order, cutoff, btype="highpass", fs=sampling_rate)
    return filtfilt(b, a, X, axis=-1)


def filter_band(X, sampling_rate, lo=8.0, hi=30.0, order=4):
    b, a = butter(order, [lo, hi], btype="bandpass", fs=sampling_rate)
    return filtfilt(b, a, X, axis=-1)


def common_average_reference(X: np.ndarray, exclude_channels: Optional[Sequence[int]] = None) -> np.ndarray:
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
