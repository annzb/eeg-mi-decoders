import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np

from data import const
from data.subject import SubjectData


class Dataset(ABC):
    def __init__(
        self, dataset_path,
        exclude_subject_ids: Optional[Sequence[str]] = None,
        apply_preprocessing: Sequence[Callable] = (, ),
        preprocessing_kwargs: Dict[str, Any] = {}
    ):
        self._file_extension = '.mat'
        self._n_channels = None
        self.dataset_path = Path(dataset_path)
        self.subject_data = self._load_subject_data(
            exclude_subject_ids=exclude_subject_ids, 
            apply_preprocessing=apply_preprocessing,
            preprocessing_kwargs=preprocessing_kwargs
        )

    @abstractmethod
    def trial_start_timestamp(self) -> float:
        return 0.0

    @abstractmethod
    def trial_end_timestamp(self) -> float:
        return 1.0

    @abstractmethod
    def _read_mat_file(self, mat_file: Path) -> SubjectData:
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

    def _load_subject_data(
        self, exclude_subject_ids: Optional[Sequence[str]] = None,
        apply_preprocessing: Sequence[Callable] = (, ),
        preprocessing_kwargs: Dict[str, Any] = {}
    ) -> Dict[str, SubjectData]:
        if not self.dataset_path.is_dir():
            raise ValueError(f"Could not find directory `{self.dataset_path}`")
        if exclude_subject_ids is not None and not isinstance(exclude_subject_ids, Sequence):
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
            subject_data[subject_id] = self._read_mat_file(p)
            
        return subject_data

    def get_XY(self, subject_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        subject_ids = self.subject_ids()
        if subject_id is not None:
            if subject_id not in subject_ids:
                raise KeyError(f"Subject {subject_id} not found. Available: {subject_ids}")
            subject_ids = np.array([subject_id])

        X_parts, y_parts, g_parts = [], [], []
        for sid in subject_ids:
            subject = self.subject_data[sid]
            X_parts.append(subject.X())
            y_parts.append(subject.Y())
            g_parts.append(np.full(subject.X().shape[0], subject.subject_id()))
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        groups = np.concatenate(g_parts, axis=0)
        return X, y, groups

        def print_info(self):
            total_bytes = sum(s.X_raw().nbytes for s in self.subject_data.values())
            print(f"Total subjects: {len(self.subject_data)}")
            print(f"Epoch window: {self.trial_start_timestamp():.3f}–{self.trial_end_timestamp():.3f} s")
            print(f"Total subject data: {total_bytes / 1024**2:.2f} MB")
