import re
from pathlib import Path
from typing import Union

import numpy as np
from scipy.io import loadmat
from data import const
from data.subject import SubjectDataDs1, SubjectDataDs4


class Dataset1:
    def __init__(self, dataset_path):
        self._n_channels = None
        self.dataset_path = Path(dataset_path)
        self.subject_data = self._load_subject_data()

    def filename_to_subject_id(self, filename: str):
        return filename.split('s')[1].split('.')[0].strip()

    def trial_start_timestamp(self):
        return 0.5

    def trial_end_timestamp(self):
        return 2.5

    def subject_ids(self):
        return np.array(sorted(self.subject_data.keys()))

    def get_subject(self, sid: Union[int, str]):
        if isinstance(sid, int):
            sid = str(sid)
        if sid in self.subject_data:
            return self.subject_data[sid]
        raise KeyError(f'Subject {sid} not found.')

    def _load_subject_data(self):
        if not self.dataset_path.is_dir():
            raise ValueError(f"Could not find directory `{self.dataset_path}`")

        subject_data = {}
        for p in self.dataset_path.iterdir():
            if not p.is_file() or p.suffix.lower() != ".mat":
                continue
            subject_id = self.filename_to_subject_id(p.name)
            if subject_id in subject_data:
                raise ValueError(f"Duplicate subject id {subject_id} encountered (file {p.name}).")

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
                electrode_locations=electrode_locations
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


class Dataset4(Dataset1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def filename_to_subject_id(self, filename: str):
        return filename.split('Subject')[1].split('6St')[0].strip()

    def trial_start_timestamp(self):
        return 0.0

    def trial_end_timestamp(self):
        return 0.85

    def _load_subject_data(self):
        if not self.dataset_path.is_dir():
            raise ValueError(f"Could not find directory `{self.dataset_path}`")

        subject_data = {}
        for p in self.dataset_path.iterdir():
            if not p.is_file() or p.suffix.lower() != ".mat":
                continue
            subject_id = self.filename_to_subject_id(p.name)
            if subject_id in subject_data:
                raise ValueError(f"Duplicate subject id {subject_id} encountered (file {p.name}).")

            mat = loadmat(p, simplify_cells=True)
            if "o" not in mat:
                raise ValueError(f"`{p.name}` missing key 'o'")
            o = mat["o"]

            srate = int(o["sampFreq"])
            marker = np.asarray(o["marker"]).astype(int).reshape(-1)
            data = np.asarray(o["data"], dtype=float)
            if data.ndim != 2:
                raise ValueError(f"`{p.name}`: expected o['data'] 2D; got {data.shape}")
            if data.shape[0] != marker.shape[0]:
                raise ValueError(f"`{p.name}`: data rows {data.shape[0]} != marker length {marker.shape[0]}")

            data_ch_first = data.T
            if data_ch_first.shape[0] == 22:  # Drop X3 sync channel
                eeg_stream = data_ch_first[:21, :]
                chnames = o.get("chnames", None)
                if chnames is None:
                    electrode_labels = np.array(self.EEG_CHANNEL_NAMES_22[:21], dtype=object)
                else:
                    electrode_labels = np.asarray(chnames).reshape(-1)[:21]
            else:
                eeg_stream = data_ch_first
                chnames = o.get("chnames", None)
                electrode_labels = None if chnames is None else np.asarray(chnames).reshape(-1)

            n_channels, nS = eeg_stream.shape
            start = int(round(self.trial_start_timestamp() * srate))
            end = int(round(self.trial_end_timestamp() * srate))
            win = end - start
            if win <= 0:
                raise ValueError("trial_end_timestamp must be > trial_start_timestamp")
            valid_codes = np.arange(1, 7, dtype=int)
            prev = np.concatenate([[0], marker[:-1]])
            onsets = np.where((prev == 0) & np.isin(marker, valid_codes))[0]

            X_list, y_list = [], []
            for onset in onsets.tolist():
                code = int(marker[onset])
                a = onset + start
                b = a + win
                if a < 0 or b > nS:
                    continue
                X_list.append(eeg_stream[:, a:b])
                y_list.append(code - 1)

            if not X_list:
                continue

            X_raw = np.stack(X_list, axis=0)  # (n_trials, n_channels, win)
            y = np.asarray(y_list, dtype=int)
            electrode_locations = np.asarray([const.EEG_PSENLOC_1020_2D[ch] for ch in electrode_labels])

            subject_data[subject_id] = SubjectDataDs4(
                subject_id=subject_id,
                sampling_rate=srate,
                electrode_locations=electrode_locations,
                electrode_labels=electrode_labels,
                X_raw=X_raw,
                y=y,
            )

        return subject_data
