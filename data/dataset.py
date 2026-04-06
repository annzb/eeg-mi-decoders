from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import uuid

import numpy as np

from data import const
from data.subject import SubjectData
from data.preprocess import PreprocessPipeline, PreprocessResult, validate_preprocess_pipeline


_datasets = {}


class TrialTimeWindow(Enum):
    DEFAULT = (0.0, 1.0)


class Dataset(ABC):

    def __init__(
        self, dataset_path,
        exclude_subject_ids: Sequence[str] = [],
        allow_multifile_subjects: bool = False,
        trial_time_window: TrialTimeWindow = TrialTimeWindow.DEFAULT
    ):
        if not isinstance(allow_multifile_subjects, bool):
            raise ValueError(f"allow_multifile_subjects must be a boolean, got {type(allow_multifile_subjects)}")
        self._validate_time_window(trial_time_window)

        self._dataset_id = str(uuid.uuid4())
        self._file_extension = '.mat'
        self._n_channels = None
        self.dataset_path = Path(dataset_path)
        self._allow_multifile_subjects = allow_multifile_subjects
        self._trial_window = trial_time_window

        self.subject_data = self._load_subject_data(data_path=self.dataset_path, exclude_subject_ids=exclude_subject_ids)
        _datasets[self._dataset_id] = self

    def _validate_time_window(self, time_window: Enum) -> None:
        if not isinstance(time_window, Enum):
            raise TypeError(f"time_window must be an enum, got {type(time_window)}")
        if not isinstance(time_window.value, tuple) or len(time_window.value) != 2 or not all(isinstance(v, (int, float)) for v in time_window.value):
            raise TypeError(f"time_window must be an enum with a (start, end) numeric tuple value, got {time_window!r}")
        if time_window.value[0] >= time_window.value[1]:
            raise ValueError(f"time_window start must be less than end, got {time_window.value}")

    @abstractmethod
    def _read_mat_file(self, mat_file: Path, subject_id: str) -> SubjectData:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    def trial_start_timestamp(self) -> float:
        return self._trial_window.value[0]

    def trial_end_timestamp(self) -> float:
        return self._trial_window.value[1]

    def filename_to_subject_id(self, filename: str) -> str:
        if filename.endswith(self._file_extension):
            return filename.replace(self._file_extension, '')
        return filename

    def subject_ids(self) -> np.ndarray:
        return np.array(sorted(self.subject_data.keys()))

    def get_subject(self, sid: Union[int, str]) -> SubjectData:
        if isinstance(sid, int):
            sid = str(sid)
        if sid in self.subject_data:
            return self.subject_data[sid]
        raise KeyError(f'Subject {sid} not found.')

    def _load_subject_data(self, data_path: Path, exclude_subject_ids: Sequence[str] = []) -> Dict[str, SubjectData]:
        if not data_path.is_dir():
            raise ValueError(f"Could not find directory `{data_path}`")
        if not hasattr(exclude_subject_ids, '__len__'):
            raise ValueError(f"exclude_subject_ids must be a sequence, got {type(exclude_subject_ids)}")

        exclude_subject_ids = set(exclude_subject_ids)
        subject_data = {}
        for p in data_path.iterdir():
            if not p.is_file() or p.suffix.lower() != self._file_extension:
                continue
            subject_id = self.filename_to_subject_id(p.name)
            if (exclude_subject_ids is not None) and (subject_id in exclude_subject_ids):
                continue
            if subject_id in subject_data:
                if self._allow_multifile_subjects:
                    subject_data[subject_id] += self._read_mat_file(mat_file=p, subject_id=subject_id)
                else:
                    raise ValueError(f"Duplicate subject id {subject_id} encountered (file {p.name}).")
            else:
                subject_data[subject_id] = self._read_mat_file(mat_file=p, subject_id=subject_id)
            
        return subject_data

    def crop(self, trial_time_window: Enum):
        if trial_time_window == self._trial_window:
            return
        self._validate_time_window(trial_time_window)
        new_start, new_end = trial_time_window.value
        cur_start, cur_end = self._trial_window.value
        if new_start < cur_start or new_end > cur_end:
            raise ValueError(f"Target window [{new_start}, {new_end}] s is not inside current window [{cur_start}, {cur_end}] s")
        if new_start == cur_start and new_end == cur_end:
            raise ValueError(f"Target window [{new_start}, {new_end}] s must be smaller than current window [{cur_start}, {cur_end}] s")
        for subj in self.subject_data.values():
            sfreq = subj._sampling_rate
            offset_start = int(round((new_start - cur_start) * sfreq))
            offset_end = subj._n_samples - int(round((cur_end - new_end) * sfreq))
            subj._X = subj._X[:, :, offset_start:offset_end]
            subj._n_samples = subj._X.shape[2]
        self._trial_window = trial_time_window

    def apply_preprocessing(self, pipeline: PreprocessPipeline):
        for subject in self.subject_data.values():
            subject.apply_preprocessing(pipeline=pipeline)

    def get_XY(
        self, 
        subject_ids: Optional[Sequence[str]] = None, 
        preprocess_pipeline: Optional[PreprocessPipeline] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_subject_ids = self.subject_ids()
        if subject_ids is None:
            target_ids = all_subject_ids
        elif not hasattr(subject_ids, '__len__'):
            raise ValueError(f"subject_ids must be a sequence, got {type(subject_ids)}")
        else:
            target_ids = []
            for sid in subject_ids:
                try:
                    sid_str = str(sid)
                except Exception as e:
                    raise ValueError(f"Failed to convert subject id {sid} to string: {e}")
                if sid_str not in all_subject_ids:
                    raise KeyError(f"Subject {sid_str} not found. Available: {all_subject_ids}")
                target_ids.append(sid_str)
            target_ids = np.array(target_ids)

        if preprocess_pipeline is not None:
            validate_preprocess_pipeline(preprocess_pipeline)

        X_parts, Y_parts, g_parts = [], [], []
        for sid in target_ids:
            subject = self.subject_data[sid]
            X, Y = subject.X(), subject.Y()
            if preprocess_pipeline:
                for func, kwargs in zip(preprocess_pipeline[0], preprocess_pipeline[1]):
                    result = func(X, sampling_rate=subject._sampling_rate, **kwargs)
                    if isinstance(result, PreprocessResult):
                        X = result.X
                        if result.Y_mask is not None:
                            Y = Y[result.Y_mask]
                    else:
                        X = result
            X_parts.append(X)
            Y_parts.append(Y)
            g_parts.append(np.full(X.shape[0], subject.subject_id()))
        X = np.concatenate(X_parts, axis=0)
        Y = np.concatenate(Y_parts, axis=0)
        groups = np.concatenate(g_parts, axis=0)
        return X, Y, groups

    def get_trial_subsets(self, subject_ids: Optional[Sequence[str]] = None,) -> Dict[str, np.ndarray]:
        all_subject_ids = self.subject_ids()
        if subject_ids is None:
            target_ids = all_subject_ids
        elif not hasattr(subject_ids, "__len__"):
            raise ValueError(f"subject_ids must be a sequence, got {type(subject_ids)}")
        else:
            target_ids = []
            for sid in subject_ids:
                sid = str(sid)
                if sid not in all_subject_ids:
                    raise KeyError(f"Subject {sid} not found. Available: {all_subject_ids}")
                target_ids.append(sid)

        result = {}
        for sid in target_ids:
            subsets = self.subject_data[sid].trial_subsets()
            if subsets is None:
                raise ValueError(f"Subject {sid} has no trial_subsets")
            subsets = np.asarray(subsets)
            if subsets.ndim != 1:
                raise ValueError(f"trial_subsets for subject {sid} must be flat; got shape {subsets.shape}")
            if len(subsets) != self.subject_data[sid].n_trials():
                raise ValueError(
                    f"trial_subsets length mismatch for subject {sid}: "
                    f"{len(subsets)} vs {self.subject_data[sid].n_trials()}"
                )
            result[sid] = subsets
        return result

    def print_info(self):
        print(f"Dataset ID: {self._dataset_id}")
        total_bytes = sum(s.X().nbytes for s in self.subject_data.values())
        print(f"Total subjects: {len(self.subject_data)}")
        print(f"Epoch window: {self.trial_start_timestamp():.3f}–{self.trial_end_timestamp():.3f} s")
        print(f"Total subject data: {total_bytes / 1024**2:.2f} MB")


def get_dataset(dataset_id: str) -> Dataset:
    if not isinstance(dataset_id, str):
        raise ValueError(f"dataset_id must be a string; got {type(dataset_id)}")
    if dataset_id not in _datasets:
        raise ValueError(f"Dataset {dataset_id} not found")
    return _datasets[dataset_id]
