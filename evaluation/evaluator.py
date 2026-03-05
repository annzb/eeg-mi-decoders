from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Sequence

import numpy as np

from evaluation import metrics
from evaluation.split import Split
from evaluation.result import DatasetEvalResult, Score, SubjectEvalResult
from models import Model


class Evaluator:
    def eval_subject(
        self, split: Split, model: Model,
        X: np.ndarray, y: np.ndarray,
        alpha: float = 0.05,
        metric: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy,
    ) -> SubjectEvalResult:
        if not isinstance(split, Split):
            raise ValueError(f"split must be a Split instance; got {type(split)}")
        if not isinstance(model, Model):
            raise ValueError(f"model must be a Model instance; got {type(model)}")
        if not callable(metric):
            raise ValueError(f"metric must be a callable; got {type(metric)}")

        guess_accuracy = metrics.calc_guess_accuracy(y)
        ucl_accuracy = metrics.calc_ucl_accuracy(len(y), alpha=alpha, guess_accuracy=guess_accuracy)

        train_scores, val_scores, test_scores = [], [], []
        n_train, n_val, n_test = [], [], []
        for train_idx, val_idx, test_idx in split(y):
            m = model.clone()
            m.fit(X[train_idx], y[train_idx])
            train_scores.append(metric(y[train_idx], m.predict(X[train_idx])))
            n_train.append(int(train_idx.size))
            if val_idx is not None:
                val_scores.append(metric(y[val_idx], m.predict(X[val_idx])))
                n_val.append(int(val_idx.size))
            if test_idx is not None:
                test_scores.append(metric(y[test_idx], m.predict(X[test_idx])))
                n_test.append(int(test_idx.size))
        
        train_score = val_score = test_score = None
        train_score = Score(acc_means=train_scores) if train_scores else None
        val_score = Score(acc_means=val_scores) if val_scores else None
        test_score = Score(acc_means=test_scores) if test_scores else None
        return SubjectEvalResult(train=train_score, val=val_score, test=test_score, guess_accuracy=guess_accuracy, ucl_accuracy=ucl_accuracy)

    def eval_all_subjects(
        self, split: Split, model: Model,
        X: np.ndarray, y: np.ndarray, groups: np.ndarray,
        alpha: float = 0.05,
        metric: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy,
    ) -> DatasetEvalResult:
        if not isinstance(groups, np.ndarray):
            raise ValueError(f"groups must be a numpy array; got {type(groups)}")
        if groups.ndim != 1 or groups.size != X.shape[0]:
            raise ValueError(f"groups must be a 1D numpy array of the same length as X; got {groups.shape}, {groups.dtype}")
        
        per_subject = {}
        means_train, means_val, means_test = [], [], []
        for sid in np.unique(groups):
            mask = (groups == sid)
            scores = self.eval_subject(split=split, model=model, X=X[mask], y=y[mask], alpha=alpha, metric=metric)
            per_subject[str(sid)] = scores
            if scores.train is not None:
                means_train.append(scores.train.acc_mean())
            if scores.val is not None:
                means_val.append(scores.val.acc_mean())
            if scores.test is not None:
                means_test.append(scores.test.acc_mean())
        
        train_score = val_score = test_score = None
        train_score = Score(acc_means=means_train) if means_train else None
        val_score = Score(acc_means=means_val) if means_val else None
        test_score = Score(acc_means=means_test) if means_test else None
        return DatasetEvalResult(per_subject=per_subject, train=train_score, val=val_score, test=test_score)
