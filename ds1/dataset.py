import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
from scipy.io import loadmat
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
    def trial_start_timestamp(self):
        return 0.0

    @abstractmethod
    def trial_end_timestamp(self):
        return 1.0

    def filename_to_subject_id(self, filename: str):
        if filename.endswith(self._file_extension):
            return filename.replace(self._file_extension, '')
        return filename

    def subject_ids(self):
        return np.array(sorted(self.subject_data.keys()))

    def get_subject(self, sid: Union[int, str]):
        if isinstance(sid, int):
            sid = str(sid)
        if sid in self.subject_data:
            return self.subject_data[sid]
        raise KeyError(f'Subject {sid} not found.')

    def _load_subject_data(
        self, exclude_subject_ids: Optional[Sequence[str]] = None,
        apply_preprocessing: Sequence[Callable] = (, ),
        preprocessing_kwargs: Dict[str, Any] = {}
    ):
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

            mat = loadmat(p, simplify_cells=True)
            if "eeg" not in mat:
                raise ValueError(f"`{p.name}` missing key 'eeg'")
            eeg = mat["eeg"]
            srate = int(eeg["srate"])
            n_trials = int(eeg["n_imagery_trials"])
            electrode_locations = np.asarray(eeg["psenloc"])
            n_channels = len(electrode_locations)
            onsets = np.where(eeg["imagery_event"] == 1)[0]
            if len(onsets) < n_trials:
                raise ValueError(f"`{p.name}`: found {len(onsets)} imagery onsets, expected at least {n_trials}")

            start = int(round(self.trial_start_timestamp() * srate))
            end = int(round(self.trial_end_timestamp() * srate))
            win = end - start
            if win <= 0:
                raise ValueError("trial_end_timestamp must be > trial_start_timestamp")

            left_stream = np.asarray(eeg["imagery_left"])[: n_channels, :]
            right_stream = np.asarray(eeg["imagery_right"])[: n_channels, :]
            X_left = np.empty(
                (n_trials, n_channels, win),
                dtype=left_stream.dtype
            )
            X_right = np.empty(
                (n_trials, n_channels, win),
                dtype=right_stream.dtype
            )
            stream_len = left_stream.shape[1]
            for i, onset in enumerate(onsets[:n_trials]):
                a = onset + start
                b = onset + end
                if a < 0 or b > stream_len:
                    raise ValueError(
                        f"`{p.name}`: trial {i} window [{a}:{b}] out of bounds "
                        f"for stream length {stream_len}"
                    )
                X_left[i] = left_stream[:, a:b]
                X_right[i] = right_stream[:, a:b]

            subject_data[subject_id] = SubjectDataDs1(
                subject_id=subject_id,
                sampling_rate=srate,
                X_left_raw=X_left,
                X_right_raw=X_right,
                electrode_locations=electrode_locations,
                preprocess_highpass=preprocess_highpass,
                preprocess_car=preprocess_car,
                preprocess_band=preprocess_band
            )
            
        return subject_data

    def print_info(self):
        total_bytes = sum(s.X_raw().nbytes for s in self.subject_data.values())
        print(f"Total subjects: {len(self.subject_data)}")
        print(f"Epoch window: {self.trial_start_timestamp():.3f}–{self.trial_end_timestamp():.3f} s")
        print(f"Total subject data: {total_bytes / 1024**2:.2f} MB")

    def get_XY(self, subject_id=None):
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


    def _read_mat_file(self, mat_file: Path) -> SubjectData:
        mat = loadmat(mat_file, simplify_cells=True)
        if "eeg" not in mat:
            raise ValueError(f"`{mat_file.name}` missing key 'eeg'")
        eeg = mat["eeg"]
        srate = int(eeg["srate"])
        n_trials = int(eeg["n_imagery_trials"])
        electrode_locations = np.asarray(eeg["psenloc"])
        n_channels = len(electrode_locations)
        onsets = np.where(eeg["imagery_event"] == 1)[0]
        if len(onsets) < n_trials:
            raise ValueError(f"`{p.name}`: found {len(onsets)} imagery onsets, expected at least {n_trials}")

        start = int(round(self.trial_start_timestamp() * srate))
        end = int(round(self.trial_end_timestamp() * srate))
        win = end - start
        if win <= 0:
            raise ValueError("trial_end_timestamp must be > trial_start_timestamp")

        left_stream = np.asarray(eeg["imagery_left"])[: n_channels, :]
        right_stream = np.asarray(eeg["imagery_right"])[: n_channels, :]
        X_left = np.empty(
            (n_trials, n_channels, win),
            dtype=left_stream.dtype
        )
        X_right = np.empty(
            (n_trials, n_channels, win),
            dtype=right_stream.dtype
        )
        stream_len = left_stream.shape[1]
        for i, onset in enumerate(onsets[:n_trials]):
            a = onset + start
            b = onset + end
            if a < 0 or b > stream_len:
                raise ValueError(
                    f"`{p.name}`: trial {i} window [{a}:{b}] out of bounds "
                    f"for stream length {stream_len}"
                )
            X_left[i] = left_stream[:, a:b]
            X_right[i] = right_stream[:, a:b]

        subject_data[subject_id] = SubjectDataDs1(
            subject_id=subject_id,
            sampling_rate=srate,
            X_left_raw=X_left,
            X_right_raw=X_right,
            electrode_locations=electrode_locations,
            preprocess_highpass=preprocess_highpass,
            preprocess_car=preprocess_car,
            preprocess_band=preprocess_band
        )