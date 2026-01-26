from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

import plots


def filter_band(X, sampling_rate, lo=8.0, hi=30.0, order=4):
    b, a = butter(order, [lo, hi], btype="bandpass", fs=sampling_rate)
    return filtfilt(b, a, X, axis=-1)


class SubjectData(ABC):
    def __init__(
        self, 
        subject_id: str, 
        sampling_rate: int, 
        electrode_locations: np.ndarray,
        electrode_labels: Optional[np.ndarray] = None,
        *args, **kwargs
    ):
        self._subject_id = subject_id
        self._sampling_rate = sampling_rate
        self._electrode_locations = electrode_locations
        self._electrode_labels = electrode_labels
        self._post_init(*args, **kwargs)

    @abstractmethod
    def _post_init(self, X_raw: np.ndarray, *args, **kwargs):
        self._labels = np.array([i for i in range(len(self._label_names))])
        self._n_trials, self._n_channels, self._n_samples = X_raw.shape
        self._X = X_raw
        if self._electrode_locations.shape[0] != self._n_channels or self._electrode_locations.shape[1] < 2:
            raise ValueError(f"electrode_locations shape {self._electrode_locations.shape} incompatible with channels {self._n_channels}")
        if self._electrode_labels is not None and self._electrode_labels.shape[0] != self._n_channels:
            raise ValueError(f"electrode_labels shape {self._electrode_labels.shape} incompatible with channels {self._n_channels}")
    
    def subject_id(self) -> str:
        return self._subject_id
    
    def sampling_rate(self) -> int:
        return self._sampling_rate
    
    def electrode_locations(self) -> np.ndarray:
        return self._electrode_locations
    
    def label_names(self) -> np.ndarray:
        return self._label_names

    def n_trials(self) -> int:
        return self._n_trials
    
    def n_channels(self) -> int:
        return self._n_channels
    
    def n_samples(self) -> int:
        return self._n_samples
    
    def X(self) -> np.ndarray:
        return self._X
    
    def Y(self) -> np.ndarray:
        return self._Y
    
    def labels(self) -> np.ndarray:
        return self._labels


class SubjectDataDs1(SubjectData):
    def __init__(self, *args, **kwargs):
        self._label_names = ('left', 'right')
        super().__init__(*args, **kwargs)

    def _post_init(self, X_left_raw: np.ndarray, X_right_raw: np.ndarray, *args, **kwargs):
        X_raw = np.concatenate([X_left_raw, X_right_raw], axis=0)
        super()._post_init(X_raw=X_raw, *args, **kwargs)
        if X_left_raw.shape != X_right_raw.shape:
            raise ValueError(f"X_left_raw and X_right_raw must have identical shapes, got {X_left_raw.shape} vs {X_right_raw.shape}")
        if X_left_raw.shape[0] != self._n_trials:
            raise ValueError(f"n_trials={self._n_trials}, but X_left_raw has {X_left_raw.shape[0]} trials")
        self._X = filter_band(X_raw, sampling_rate=self._sampling_rate)
        self._Y = np.concatenate([
            np.full(len(X_left_raw), self._labels[0]),
            np.full(len(X_right_raw), self._labels[1])
        ], axis=0)


class SubjectDataDs4(SubjectData):
    def __init__(self, *args, **kwargs):
        self._label_names = ('handL', 'handR', 'passive', 'legL', 'tongue', 'legR')
        super().__init__(*args, **kwargs)

    def _post_init(self, X_raw: np.ndarray, *args, **kwargs):
        super()._post_init(X_raw=X_raw, *args, **kwargs)
        # the rest of the processing from the new dataset
