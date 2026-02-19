from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from data.dataset import Dataset
from data.subject import SubjectData
from ds1.subject import SubjectDataDs1


class Dataset1(Dataset):

    def trial_start_timestamp(self):
        return 0.5

    def trial_end_timestamp(self):
        return 2.5

    def filename_to_subject_id(self, filename: str) -> str:
        filename = super().filename_to_subject_id(filename)
        return filename.split('s')[1].strip()

    def _read_mat_file(self, mat_file: Path, subject_id: str) -> SubjectData:
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

        return SubjectDataDs1(
            X_raw=(X_left, X_right),
            subject_id=subject_id,
            sampling_rate=srate,
            electrode_locations=electrode_locations
        )
