from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Callable, Dict, Sequence

import numpy as np

from data import plots


class SubjectData(ABC):
    def __init__(
        self,
        X_raw: Any,
        subject_id: str, 
        sampling_rate: int, 
        electrode_locations: np.ndarray,
        electrode_labels: Optional[np.ndarray] = None,
        *args, **kwargs
    ):
        if not isinstance(subject_id, str):
            raise ValueError(f"subject_id must be a string, got {type(subject_id)}")
        if not isinstance(sampling_rate, int) or sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be a positive integer, got {type(sampling_rate)}")
        if not isinstance(electrode_locations, np.ndarray) or electrode_locations.shape[1] < 2:
            raise ValueError(f"electrode_locations must be a numpy array with shape (n_channels, 2), got {electrode_locations.shape}")
        if electrode_labels is not None and not isinstance(electrode_labels, np.ndarray) or electrode_labels.shape[0] != electrode_locations.shape[0]:
            raise ValueError(f"electrode_labels must be a numpy array with shape (n_channels,), got {electrode_labels.shape}")
            
        self._subject_id = subject_id
        self._sampling_rate = sampling_rate
        self._electrode_locations = electrode_locations
        self._electrode_labels = electrode_labels

        self._X = self._format_X(X_raw)
        self._n_trials, self._n_channels, self._n_samples = self._X.shape
        self._labels = np.array([i for i in range(len(self.label_names()))])
        self._n_classes = len(self._labels)
        self._Y = self._make_labels()

    def _make_labels(self) -> np.ndarray:
        trials_per_label, leftover_trials = divmod(self._n_trials, self._n_classes)
        Y = np.repeat(
            np.arange(self._n_classes),
            np.full(self._n_classes, trials_per_label) + (np.arange(self._n_classes) < leftover_trials)
        )
        return Y

    @abstractmethod
    def _format_X(self, X_raw: Any) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def channel_names(self) -> tuple:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def label_names(self) -> tuple:
        raise NotImplementedError("Subclasses must implement this method")

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

    def n_classes(self) -> int:
        return self._n_classes
    
    def X(self) -> np.ndarray:
        return self._X
    
    def Y(self) -> np.ndarray:
        return self._Y

    def apply_preprocessing(
        self,
        apply_preprocessing: Sequence[Callable] = (),
        preprocessing_kwargs: Dict[str, Any] = {}
    ) -> None:
        for preprocessing_func in apply_preprocessing:
            self._X = preprocessing_func(self._X, **preprocessing_kwargs)

    def clone(self) -> 'SubjectData':
        return deepcopy(self)

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
        trial_label = int(self.Y()[trial])
        label_trials = np.where(self.Y() == trial_label)[0]
        if label_trials.size == 0:
            raise ValueError(f"Trial {trial} has no matching labels in Y.")

        trial_index_within_label = int(np.where(label_trials == trial)[0][0])
        joint_trials, joint_labels = [], []

        for other_label in self._labels:
            if other_label == trial_label:
                continue
            other_trials = np.where(self.Y() == other_label)[0]
            if trial_index_within_label >= other_trials.size:
                continue
            joint_trial = int(other_trials[trial_index_within_label])
            joint_trials.append(joint_trial)
            joint_labels.append(other_label)

        return np.asarray(joint_trials, dtype=int), np.asarray(joint_labels, dtype=int)

    def plot_head(self, trial: int, index=None, timestamp=None, plot_joint=False):
        self._validate_trial(trial=trial)
        index = self._validate_timestamp(index=index, timestamp=timestamp)
        trial_label = self.Y()[trial]
        t = index / self._sampling_rate
        title = f'Subject {self._subject_id}. Trial {trial}. Timestamp {t:.2f}s'
        plots.plot_scalp(
            signal=self._X[trial, :, index],
            channel_locations=self._electrode_locations[:, :2],
            channel_names=self.channel_names(),
            title=f'{title} [{self.label_names()[trial_label]}]'
        )
        if plot_joint:
            joint_trials, joint_labels = self._find_joint_trials(trial=trial)
            for joint_trial, joint_label in zip(joint_trials, joint_labels):
                title = f'Subject {self._subject_id}. Trial {joint_trial}. Timestamp {t:.2f}s'
                plots.plot_scalp(
                    signal=self._X[joint_trial, :, index],
                    channel_locations=self._electrode_locations[:, :2],
                    channel_names=self.channel_names(),
                    title=f'{title} [{self.label_names()[joint_label]}]'
                )

    def plot_channel(self, trial: int, channel: int, index=None, timestamp=None):
        self._validate_trial(trial=trial)
        self._validate_channel(channel=channel)
        index = self._validate_timestamp(index=index, timestamp=timestamp)

        trial_label = self.Y()[trial]
        joint_trials, joint_labels = self._find_joint_trials(trial=trial)
        trials = np.concatenate([np.array([trial]), joint_trials], axis=0)
        labels = np.concatenate([np.array([trial_label]), joint_labels], axis=0)
        trial_data = np.concatenate([
            self._X[trial:trial+1, channel, index:], self._X[joint_trials, channel, index:]
        ], axis=0)
        label_names = [f'{self.label_names()[l]} (trial {t})' for t, l in zip(trials, labels)]

        title = f"Subject {self._subject_id}. Channel {channel} [{self.channel_names()[channel]}]"
        t = np.arange(trial_data.shape[1], dtype=float) / float(self._sampling_rate)
        plots.plot_eeg_channel_joint(t=t, channel_data=trial_data, label_names=label_names, title=title)

    def plot_trial_heatmap(self, trial: int):
        self._validate_trial(trial=trial)
        label_name = self.label_names()[self.Y()[trial]]
        t = np.arange(self._n_samples, dtype=float) / float(self._sampling_rate)
        sample = self._X[trial, :, :]
        title = f"Subject {self._subject_id}. Trial {trial} [{label_name}]"
        plots.plot_eeg_heatmap(t=t, sample=sample, title=title, channel_names=self.channel_names())

    def print_info(self):
        print(f'Subject ID: {self._subject_id}')
        print(f'sampling_rate: {self._sampling_rate}, n_trials: {self._n_trials}, n_channels: {self._n_channels}, n_samples: {self._n_samples}')
        print(f'X: {self._X.shape}, Y: {self._Y.shape}')
