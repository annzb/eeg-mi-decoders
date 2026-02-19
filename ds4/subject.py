from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data import plots
from data.subject import SubjectData


class SubjectDataDs4(SubjectData):

    def label_names(self) -> tuple:
        return ("handL", "handR", "passive", "legL", "tongue", "legR")

    def channel_names(self) -> tuple:
        return (
            "Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","A1","A2",
            "F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"
        )

    def _post_init(self, X_raw: np.ndarray, y: np.ndarray, *args, **kwargs):
        X_raw = np.asarray(X_raw, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        super()._post_init(X_raw=X_raw, *args, **kwargs)
        if y.shape[0] != self._n_trials:
            raise ValueError(f"y must have length n_trials={self._n_trials}; got {y.shape[0]}")
        self._X = X_raw
        self._Y = y
