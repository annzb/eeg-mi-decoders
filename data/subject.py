from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

from data import plots


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
        self._X_raw = X_raw
        self._Y, self._X = np.array([]), np.array([])
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

    def electrode_labels(self) -> np.ndarray:
        return self._electrode_labels

    def labels(self) -> np.ndarray:
        return self._labels

    def n_trials(self) -> int:
        return self._n_trials
    
    def n_channels(self) -> int:
        return self._n_channels
    
    def n_samples(self) -> int:
        return self._n_samples

    def X_raw(self) -> np.ndarray:
        return self._X_raw
    
    def X(self) -> np.ndarray:
        return self._X
    
    def Y(self) -> np.ndarray:
        return self._Y

    def print_info(self):
        print(f'Subject ID: {self._subject_id}')
        print(f'sampling_rate: {self._sampling_rate}, n_trials: {self._n_trials}, n_channels: {self._n_channels}, n_samples: {self._n_samples}')
        print(f'X: {self._X.shape}, Y: {self._Y.shape}')

    def _validate_trial(self, trial: int):
        if trial < 0 or trial >= self._n_trials:
            raise IndexError(f"`trial` out of range: {trial} (valid: 0..{self._n_trials-1})")

    def _validate_channel(self, channel: int):
        if channel < 0 or channel >= self._n_channels:
            raise IndexError(f"`channel` out of range: {channel} (valid: 0..{self._n_channels-1})")

    def _validate_timestamp(self, index=None, timestamp=None):
        if index is not None and timestamp is not None:
            raise ValueError("Specify either `index` or `timestamp`.")
        elif index is None and timestamp is None:
            index = 0
        elif timestamp is not None:
            index = int(round(float(timestamp) * self._sampling_rate))
        if index < 0 or index >= self._n_samples:
            raise IndexError(f"`index` out of range: {index} (valid: 0..{self._n_samples-1})")
        return index

    def _find_joint_trials(self, trial: int):
        self._validate_trial(trial=trial)
        trial_label = int(self._Y[trial])
        label_trials = np.where(self._Y == trial_label)[0]
        if label_trials.size == 0:
            raise ValueError(f"Trial {trial} has no matching labels in Y.")

        trial_index_within_label = int(np.where(label_trials == trial)[0][0])
        joint_trials, joint_labels = [], []

        for other_label in self._labels:
            if other_label == trial_label:
                continue
            other_trials = np.where(self._Y == other_label)[0]
            if trial_index_within_label >= other_trials.size:
                continue
            joint_trial = int(other_trials[trial_index_within_label])
            joint_trials.append(joint_trial)
            joint_labels.append(other_label)

        return np.asarray(joint_trials, dtype=int), np.asarray(joint_labels, dtype=int)

    def plot_head(self, trial: int, index=None, timestamp=None, plot_joint=False):
        self._validate_trial(trial=trial)
        index = self._validate_timestamp(index=index, timestamp=timestamp)
        trial_label = self._Y[trial]
        t = index / self._sampling_rate
        title = f'Subject {self._subject_id}. Trial {trial}. Timestamp {t:.2f}s.'
        plots.plot_scalp(
            signal=self._X[trial, :, index],
            channel_locations=self._electrode_locations[:, :2],
            title=f'{title} Label: {trial_label} ({self._label_names[trial_label]})'
        )
        if plot_joint:
            joint_trials, joint_labels = self._find_joint_trials(trial=trial)
            for joint_trial, joint_label in zip(joint_trials, joint_labels):
                title = f'Subject {self._subject_id}. Trial {joint_trial}. Timestamp {t:.2f}s.'
                plots.plot_scalp(
                    signal=self._X[joint_trial, :, index],
                    channel_locations=self._electrode_locations[:, :2],
                    title=f'{title} Label: {joint_label} ({self._label_names[joint_label]})'
                )

    def plot_channel(self, trial: int, channel: int, index=None, timestamp=None):
        self._validate_trial(trial=trial)
        self._validate_channel(channel=channel)
        index = self._validate_timestamp(index=index, timestamp=timestamp)

        trial_label = self._Y[trial]
        joint_trials, joint_labels = self._find_joint_trials(trial=trial)
        trials = np.concatenate([np.array([trial]), joint_trials], axis=0)
        labels = np.concatenate([np.array([trial_label]), joint_labels], axis=0)
        trial_data = np.concatenate([
            self._X[trial:trial+1, channel, index:], self._X[joint_trials, channel, index:]
        ], axis=0)
        label_names = [f'{self._label_names[l]} (trial {t})' for t, l in zip(trials, labels)]

        title = f"Subject {self._subject_id}. Channel {channel}."
        t = np.arange(trial_data.shape[1], dtype=float) / float(self._sampling_rate)
        plots.plot_eeg_channel_joint(t=t, channel_data=trial_data, label_names=label_names, title=title)

    # def plot_trial(self, trial: int, channel_indices=None, side="left"):
    #     if trial < 0 or trial >= self.n_trials:
    #         raise IndexError(f"`trial` out of range: {trial} (valid: 0..{self.n_trials-1})")
    #     if side not in ("left", "right"):
    #         raise ValueError(f"`side` must be 'left' or 'right'; got {side!r}")

    #     t = np.arange(self.n_samples, dtype=float) / self.sampling_rate
    #     sample = self.X[trial, :, :] if side == "left" else self.X[trial + self.n_trials, :, :]
    #     title = f"Subject {self.subject_id}. Trial {trial}. {side.upper()}"
    #     plots.plot_eeg_heatmap(t, sample, title=title)


class SubjectDataDs1(SubjectData):
    def __init__(self, *args, **kwargs):
        self._label_names = ('handL', 'handR')
        super().__init__(*args, **kwargs)

    def label_names(self) -> tuple:
        return self._label_names

    def _post_init(self, X_left_raw: np.ndarray, X_right_raw: np.ndarray, *args, **kwargs):
        if X_left_raw.shape != X_right_raw.shape:
            raise ValueError(f"X_left_raw and X_right_raw must have identical shapes, got {X_left_raw.shape} vs {X_right_raw.shape}")
        X_raw = np.concatenate([X_left_raw, X_right_raw], axis=0)
        super()._post_init(X_raw=X_raw, *args, **kwargs)
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
