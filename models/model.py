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

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.feat is not None:
            self.feat.fit(X, y)
            F = self.feat.transform(X)
        else:
            F = X
        self.clf.fit(F, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feat is not None:
            F = self.feat.transform(X)
        else:
            F = X
        return self.clf.predict(F)

    def clone(self) -> "Model":
        return Model(self.feat.clone() if self.feat is not None else None, self.clf.clone())

    def eval_metric(
        self, split: Split,
        X: np.ndarray, y: np.ndarray,
        metric: Callable[[np.ndarray, np.ndarray], float]
    ) -> Tuple[float, float, float]:
        train_score = val_score = test_score = None
        yhat_train = self.predict(X[split.train_idx])
        train_score = metric(y[split.train_idx], yhat_train)
        if split.val_idx is not None:
            yhat_val = self.predict(X[split.val_idx])
            val_score = metric(y[split.val_idx], yhat_val)
        if split.test_idx is not None:
            yhat_test = self.predict(X[split.test_idx])
            test_score = metric(y[split.test_idx], yhat_test)
        return train_score, val_score, test_score
