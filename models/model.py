from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from models.feature_extractor import FeatureExtractor
from models.classifier import Classifier


@dataclass
class Model:
    feat: FeatureExtractor
    clf: Classifier

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        self.feat.fit(X, y)
        F = self.feat.transform(X)
        self.clf.fit(F, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self.feat.transform(X)
        return self.clf.predict(F)

    def clone(self) -> "Model":
        return Model(self.feat.clone(), self.clf.clone())
