from __future__ import annotations
from dataclasses import dataclass
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from typing import Callable, Dict, Iterator, Optional, Sequence

import numpy as np

from data import Dataset, SubjectData, get_dataset
from evaluation import metrics
from evaluation.split import Split, Splitter
from evaluation.result import DatasetEvalResult, Score, SubjectEvalResult, ModelEvalResult
from models import Model


class Evaluator:
    def __init__(self):
        self._results = []

    def reset(self):
        self._results = []

    def get_results(
        self,
        dataset_id: Optional[str] = None,
        subject_id: Optional[str] = None
    ) -> Tuple[ModelEvalResult, ...]:
        if dataset_id is not None and not isinstance(dataset_id, str):
            raise ValueError(f"dataset_id must be a string; got {type(dataset_id)}")
        if subject_id is not None and not isinstance(subject_id, str):
            raise ValueError(f"subject_id must be a string; got {type(subject_id)}")
        return tuple(
            result
            for result in self._results
            if (dataset_id is None or result.dataset_id == dataset_id)
            and (subject_id is None or result.subject_id == subject_id)
        )
    
    def get_scores(self, results: Sequence[ModelEvalResult], mode: str = 'train') -> np.ndarray:
        if not results or not hasattr(results, '__len__'):
            raise ValueError("results must be a non-empty sequence")

        modes = {'train', 'val', 'test'}
        if mode not in modes:
            raise ValueError(f"mode must be one of {modes}; got {mode!r}")
        if mode == "train":
            acc_attr = 'train_acc'
        elif mode == "val":
            acc_attr = 'val_acc'
        elif mode == "test":
            acc_attr = 'test_acc'
        if not hasattr(results[0], acc_attr):
            raise ValueError(f"Model result missing {acc_attr} attribute for mode {mode!r}")

        scores = [getattr(res, acc_attr) for res in results]
        return np.asarray(scores)

    def get_model_scores(self, dataset: Dataset, mode: str = 'train') -> np.ndarray:
        subjects = dataset.subject_ids()
        subject_means, subject_stds, subject_ucls = [], [], []
        for subject in subjects:
            subject_results = self.get_results(dataset_id=dataset.dataset_id, subject_id=subject)
            subject_scores = self.get_scores(subject_results, mode=mode)
            subject_means.append(float(np.mean(subject_scores)))
            subject_stds.append(float(np.std(subject_scores, ddof=0)))
            subject_ucls.append(subject_results[0].ucl_accuracy)
        return np.asarray(subject_means), np.asarray(subject_stds), np.asarray(subject_ucls)
        
    def eval_subject(
        self, splitter: Splitter, model: Model,
        X: np.ndarray, y: np.ndarray,
        alpha: float = 0.05,
        metric: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy,
        dataset_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        reset_results: bool = False
    ) -> Tuple[ModelEvalResult, ...]:
        if reset_results:
            self.reset()

        if not isinstance(splitter, Splitter):
            raise ValueError(f"split must be a Splitter instance; got {type(split)}")
        if not isinstance(model, Model):
            raise ValueError(f"model must be a Model instance; got {type(model)}")
        if not callable(metric):
            raise ValueError(f"metric must be a callable; got {type(metric)}")

        guess_accuracy = metrics.calc_guess_accuracy(y)
        ucl_accuracy = metrics.calc_ucl_accuracy(len(y), alpha=alpha, guess_accuracy=guess_accuracy)

        subject_results = []
        for split in splitter(y, subject_id=subject_id):
            m = model.clone()
            m.fit(X[split.train_idx], y[split.train_idx])
            train_score, val_score, test_score = m.eval_metric(split=split, X=X, y=y, metric=metric)
            subject_results.append(ModelEvalResult(
                model=model,
                split=split,
                guess_accuracy=guess_accuracy,
                ucl_accuracy=ucl_accuracy,
                dataset_id=dataset_id,
                subject_id=subject_id,
                train_acc=train_score,
                val_acc=val_score,
                test_acc=test_score
            ))

        self._results.extend(subject_results)
        return tuple(subject_results)

    def eval_all_subjects(
        self, splitter: Splitter, model: Model,
        X: np.ndarray, y: np.ndarray, groups: np.ndarray,
        alpha: float = 0.05,
        metric: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy,
        dataset_id: Optional[str] = None,
        reset_results: bool = False
    ) -> DatasetEvalResult:
        if reset_results:
            self.reset()
        if self._results:
            raise ValueError("Results are not empty, set reset_results=True")

        if not isinstance(groups, np.ndarray):
            raise ValueError(f"groups must be a numpy array; got {type(groups)}")
        if groups.ndim != 1 or groups.size != X.shape[0]:
            raise ValueError(f"groups must be a 1D numpy array of the same length as X; got {groups.shape}, {groups.dtype}")
        
        results = []
        for sid in np.unique(groups):
            mask = (groups == sid)
            subject_results = self.eval_subject(
                splitter=splitter,
                model=model,
                X=X[mask], y=y[mask],
                alpha=alpha,
                metric=metric,
                dataset_id=dataset_id,
                subject_id=str(sid),
                reset_results=False
            )
            results.extend(subject_results)
        return results

    def rank_subjects(self, dataset: Dataset, k: int = 5, mode: str = 'test') -> None:
        if not isinstance(dataset, Dataset):
            raise ValueError(f"dataset must be a Dataset instance; got {type(dataset)}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive int; got {k!r}")

        subjects = dataset.subject_ids()
        subject_means, _, _ = self.get_model_scores(dataset=dataset, mode=mode)
        if len(subjects) != len(subject_means):
            raise ValueError(f"Mismatch between subjects ({len(subjects)}) and scores ({len(subject_means)})")

        order = np.argsort(subject_means)[::-1]
        top_k = min(k, len(order))

        print(f"Top {top_k} subjects by mean {mode} accuracy:")
        for rank, idx in enumerate(order[:top_k], start=1):
            print(f"{rank}. {subjects[idx]}: {subject_means[idx]:.4f}")
