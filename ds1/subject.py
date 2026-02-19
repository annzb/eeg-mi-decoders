from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data import plots
from data.subject import SubjectData


class SubjectDataDs1(SubjectData):

    def label_names(self) -> tuple:
        return ('handL', 'handR')

    def channel_names(self) -> tuple:
        return (
            # Frontal pole
            "Fp1", 'Fpz', "Fp2",

            # Anterior frontal
            "AF7","AF3","AFz","AF4","AF8",

            # Frontal
            "F7","F5","F3","F1","Fz","F2","F4","F6","F8",

            # Frontocentral
            'FT7', "FC5","FC3","FC1","FCz","FC2","FC4","FC6", 'FT8',

            # Central
            "T7","C5","C3","C1","Cz","C2","C4","C6","T8",

            # Centroparietal
            'TP7', "CP5","CP3","CP1","CPz","CP2","CP4","CP6", 'TP8',

            # Parietal
            'P9', "P7","P5","P3","P1","Pz","P2","P4","P6","P8", 'P10',

            # Parieto-occipital
            "PO7","PO3","POz","PO4","PO8",

            # Occipital
            "O1","Oz","O2",

            # Inter-occipital
            'Iz'
        )

    def _format_X(self, X_raw: Any) -> np.ndarray:
        if not hasattr(X_raw, '__len__') or len(X_raw) != 2:
            raise ValueError(f"X_raw must be a sequence of length 2 (left, right), got {type(X_raw)}")
        X_left_raw, X_right_raw = X_raw
        if not isinstance(X_left_raw, np.ndarray) or not isinstance(X_right_raw, np.ndarray) or X_left_raw.shape != X_right_raw.shape:
            raise ValueError(f"X_left_raw and X_right_raw must be numpy arrays with identical shapes, got {X_left_raw.shape} vs {X_right_raw.shape}")
        X = np.concatenate([X_left_raw, X_right_raw], axis=0)
        return X


    # def _preprocess(self, preprocess_highpass: bool = True, preprocess_car: bool = True, preprocess_band: bool = True):
    #     if preprocess_highpass:
    #         self._X = preprocess.filter_highpass(self._X, sampling_rate=self._sampling_rate, cutoff=0.5)
    #     if preprocess_car:
    #         self._X = preprocess.common_average_reference(self._X)
    #     if preprocess_band:
    #         self._X = preprocess.filter_band(self._X, sampling_rate=self._sampling_rate)
