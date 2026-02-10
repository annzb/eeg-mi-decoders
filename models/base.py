from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import copy
import numpy as np


class Model(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def clone(self) -> "Model":
        return copy.deepcopy(self)

    # def score(self, X: np.ndarray, y: np.ndarray) -> float:
    #     yhat = self.predict(X)
    #     return float(np.mean(yhat == y))
