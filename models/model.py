from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from models.feature_extractor import FeatureExtractor
from models.classifier import Classifier


@dataclass
class Model:
    feat: Optional[FeatureExtractor]
    clf: Classifier

    def __post_init__(self):
        if self.feat is not None and not isinstance(self.feat, FeatureExtractor):
            raise ValueError(f"Invalid feature_extractor: {self.feat}")
        if not isinstance(self.clf, Classifier):
            raise ValueError(f"Invalid classifier: {self.clf}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        if self.feat is not None:
            self.feat.fit(X, y)
            F = self.feat.transform(X)
        else:
            F = X
        self.clf.fit(F, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feat is not None:
            F = self.feat.transform(X)
        else:
            F = X
        return self.clf.predict(F)

    def clone(self) -> "Model":
        return Model(self.feat.clone() if self.feat is not None else None, self.clf.clone())
