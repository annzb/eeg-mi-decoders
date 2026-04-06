from enum import Enum
from pathlib import Path
from typing import Dict, Sequence, Tuple, Callable, Any

import mne
import numpy as np

from data import const
from data.dataset import Dataset
from data.preprocess import PreprocessPipeline, PreprocessResult, validate_preprocess_pipeline
from braindecode.datasets.bbci import BBCIDataset
from ds8.subject import SubjectData, SubjectDataDs8


_LABEL_MAP = {'Right Hand': 0, 'Left Hand': 1, 'Rest': 2, 'Feet': 3}
_DS8_CHANNEL_POS_2D = {
    **const.EEG_PSENLOC_1020_2D,
    **const.EEG_PSENLOC_DS8_MOTOR_2D,
}


class TrialTimeWindowDs8(Enum):
    DEFAULT = (-0.5, 4.0)
    SHORT = (0.5, 4.0)


class Dataset8(Dataset):

    def __init__(self, *args, trial_time_window: TrialTimeWindowDs8 = TrialTimeWindowDs8.DEFAULT, **kwargs):
        # self._train_indices: Dict[str, np.ndarray] = {}
        # self._test_indices: Dict[str, np.ndarray] = {}
        super().__init__(*args, trial_time_window=trial_time_window, **kwargs)

    # def train_indices(self) -> dict:
    #     return self._train_indices

    # def test_indices(self) -> dict:
    #     return self._test_indices

    def filename_to_subject_id(self, filename: str) -> str:
        return super().filename_to_subject_id(filename)

    def _load_subject_data(self, data_path: Path, exclude_subject_ids: Sequence[str] = ()) -> Dict[str, SubjectData]:
        train_dir = data_path / "train"
        test_dir = data_path / "test"
        train_data = super()._load_subject_data(train_dir, exclude_subject_ids)
        print('Loaded train data')
        test_data = super()._load_subject_data(test_dir, exclude_subject_ids)
        print('Loaded test data')
        all_sids = sorted(set(train_data) | set(test_data))
        subject_data = {}

        for sid in all_sids:
            train_subj = train_data.pop(sid, None)
            test_subj = test_data.pop(sid, None)
            if train_subj is not None and test_subj is not None:
                merged = train_subj + test_subj
                trial_subsets = np.concatenate([
                    np.full(train_subj.n_trials(), "train", dtype=object),
                    np.full(test_subj.n_trials(), "test", dtype=object),
                ])
            elif train_subj is not None:
                merged = train_subj
                trial_subsets = np.full(train_subj.n_trials(), "train", dtype=object)
            else:
                merged = test_subj
                trial_subsets = np.full(test_subj.n_trials(), "test", dtype=object)

            merged._trial_subsets = trial_subsets
            subject_data[sid] = merged
            del train_subj, test_subj, merged, trial_subsets

        return subject_data

    def _read_mat_file(self, mat_file: Path, subject_id: str) -> SubjectData:
        bbci = BBCIDataset(filename=str(mat_file), load_sensor_names=None)
        with mne.utils.use_log_level('WARNING'):
            raw = bbci.load()

        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude=[])
        if eeg_picks.size == 0:
            raise ValueError(f"No EEG channels found in {mat_file}")

        sfreq = raw.info['sfreq']
        eeg_data = raw.get_data(picks=eeg_picks)
        channel_names = tuple(raw.ch_names[i] for i in eeg_picks)
        missing = [ch for ch in channel_names if ch not in _DS8_CHANNEL_POS_2D]
        if missing:
            raise ValueError(f"Missing electrode locations for channels: {missing}")

        electrode_locations = np.array(
            [[*_DS8_CHANNEL_POS_2D[ch], 0.0] for ch in channel_names],
            dtype=float,
        )
        win_start_samp = int(round(self.trial_start_timestamp() * sfreq))
        win_end_samp = int(round(self.trial_end_timestamp() * sfreq))

        X_list, y_list = [], []
        for annot in raw.annotations:
            desc = annot['description']
            if desc not in _LABEL_MAP:
                continue
            onset_samp = int(round(annot['onset'] * sfreq))
            extract_start = onset_samp + win_start_samp
            extract_end = onset_samp + win_end_samp
            if extract_start < 0 or extract_end > eeg_data.shape[1]:
                continue
            X_list.append(eeg_data[:, extract_start:extract_end])
            y_list.append(_LABEL_MAP[desc])

        if not X_list:
            raise ValueError(f"No valid trials for subject {subject_id} in {mat_file}")
        X_raw = np.stack(X_list, axis=0)
        Y_raw = np.array(y_list, dtype=np.int64)
        return SubjectDataDs8(
            X_raw=X_raw,
            Y_raw=Y_raw,
            subject_id=subject_id,
            sampling_rate=int(sfreq),
            electrode_locations=electrode_locations,
            channel_names=channel_names,
        )

    def get_XY(self, subject_ids=None, preprocess_pipeline=None):
        if preprocess_pipeline is not None:
            raise ValueError(
                "Dataset8.get_XY(preprocess_pipeline=...) is disabled because DS8 uses "
                "official per-subject train/test indices that must be remapped when trials are removed. "
                "Call ds8.apply_preprocessing(...) first, then call get_XY() without preprocess_pipeline."
            )
        return super().get_XY(subject_ids=subject_ids, preprocess_pipeline=None)
