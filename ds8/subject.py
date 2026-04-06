from typing import Any

import numpy as np

from data.subject import SubjectData


MOTOR_CH_NAMES = (
    'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5', 'CP1', 'CP2', 'CP6',
    'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
    'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
    'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h',
    'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h', 'CCP2h', 'CPP1h', 'CPP2h',
)


class SubjectDataDs8(SubjectData):

    def label_names(self) -> tuple:
        return ('handR', 'handL', 'rest', 'feet')

    def channel_names(self) -> tuple:
        return self._channel_names

    def _format_X(self, X_raw: Any) -> np.ndarray:
        return np.asarray(X_raw, dtype=float)
