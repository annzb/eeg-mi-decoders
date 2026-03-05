from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from data import const
from data.subject import SubjectData
from data.preprocess import PreprocessPipeline, validate_preprocess_pipeline


class Dataset(ABC):
    def __init__(
        self, dataset_path,
        exclude_subject_ids: Sequence[str] = []
    ):
        self._file_extension = '.mat'
        self._n_channels = None
        self.dataset_path = Path(dataset_path)
        self.subject_data = self._load_subject_data(exclude_subject_ids=exclude_subject_ids)

    @abstractmethod
    def trial_start_timestamp(self) -> float:
        return 0.0

    @abstractmethod
    def trial_end_timestamp(self) -> float:
        return 1.0

    @abstractmethod
    def _read_mat_file(self, mat_file: Path, subject_id: str) -> SubjectData:
        raise NotImplementedError("Subclasses must implement this method")

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

    def _load_subject_data(self, exclude_subject_ids: Sequence[str] = []) -> Dict[str, SubjectData]:
        if not self.dataset_path.is_dir():
            raise ValueError(f"Could not find directory `{self.dataset_path}`")
        if not hasattr(exclude_subject_ids, '__len__'):
            raise ValueError(f"exclude_subject_ids must be a sequence, got {type(exclude_subject_ids)}")

        exclude_subject_ids = set(exclude_subject_ids)
        subject_data = {}
        for p in self.dataset_path.iterdir():
            if not p.is_file() or p.suffix.lower() != self._file_extension:
                continue
            subject_id = self.filename_to_subject_id(p.name)
            if subject_id in subject_data:
                raise ValueError(f"Duplicate subject id {subject_id} encountered (file {p.name}).")
            if (exclude_subject_ids is not None) and (subject_id in exclude_subject_ids):
                continue
            subject_data[subject_id] = self._read_mat_file(mat_file=p, subject_id=subject_id)
            
        return subject_data

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
                    X = func(X, **kwargs)
            X_parts.append(X)
            Y_parts.append(Y)
            g_parts.append(np.full(X.shape[0], subject.subject_id()))
        X = np.concatenate(X_parts, axis=0)
        Y = np.concatenate(Y_parts, axis=0)
        groups = np.concatenate(g_parts, axis=0)
        return X, Y, groups

    def print_info(self):
        total_bytes = sum(s.X().nbytes for s in self.subject_data.values())
        print(f"Total subjects: {len(self.subject_data)}")
        print(f"Epoch window: {self.trial_start_timestamp():.3f}–{self.trial_end_timestamp():.3f} s")
        print(f"Total subject data: {total_bytes / 1024**2:.2f} MB")
