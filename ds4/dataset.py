from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from data import const
from data.dataset import Dataset
from data.subject import SubjectData
from ds4.subject import SubjectDataDs4


class TrialTimeWindowDs4(Enum):
    DEFAULT = (0.0, 0.85)


class Dataset4(Dataset):

    def __init__(self, *args, trial_time_window: TrialTimeWindowDs4 = TrialTimeWindowDs4.DEFAULT, **kwargs):
        super().__init__(*args, trial_time_window=trial_time_window, **kwargs)

    def filename_to_subject_id(self, filename: str) -> str:
        filename = super().filename_to_subject_id(filename)
        return filename.split('Subject')[1].split('6St')[0].strip()

    def _read_mat_file(self, mat_file: Path, subject_id: str) -> SubjectData:
        mat = loadmat(mat_file, simplify_cells=True)
        if "o" not in mat:
            raise ValueError(f"`{mat_file.name}` missing key 'o'")
        root_content = mat["o"]
        srate = int(root_content["sampFreq"])
        marker = np.asarray(root_content["marker"]).astype(int).reshape(-1)
        data = np.asarray(root_content["data"], dtype=float)
        if data.ndim != 2:
            raise ValueError(f"`{mat_file.name}`: expected root_content['data'] 2D; got {data.shape}")
        if data.shape[0] != marker.shape[0]:
            raise ValueError(f"`{mat_file.name}`: data rows {data.shape[0]} != marker length {marker.shape[0]}")

        data_ch_first = data.T
        if data_ch_first.shape[0] == 22:  # Drop X3 sync channel
            eeg_stream = data_ch_first[:21, :]
            chnames = root_content.get("chnames", None)
            if chnames is None:
                electrode_labels = np.array(self.EEG_CHANNEL_NAMES_22[:21], dtype=object)
            else:
                electrode_labels = np.asarray(chnames).reshape(-1)[:21]
        else:
            eeg_stream = data_ch_first
            chnames = root_content.get("chnames", None)
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
            raise ValueError(f'No trials found for subject {subject_id}, filename {mat_file.name}')

        X_raw = np.stack(X_list, axis=0)  # (n_trials, n_channels, win)
        y = np.asarray(y_list, dtype=int)
        electrode_locations = np.asarray([const.EEG_PSENLOC_1020_2D[ch] for ch in electrode_labels])

        return SubjectDataDs4(
            X_raw=X_raw,
            Y_raw=y,
            subject_id=subject_id,
            sampling_rate=srate,
            electrode_locations=electrode_locations,
            electrode_labels=electrode_labels
        )
