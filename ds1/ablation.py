from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from data.dataset import Dataset1
from models import classifier as clf, feature_extractor as fe
from models.base import Model
from models.crossval import Crossval
from models.evaluation import SubjectEvalResult



class Ablation:
    def __init__(
        self,
        # Each element is (funcs, kwargs). Default: a single empty pipeline => no preprocessing.
        preprocessing_pipelines: Sequence[PreprocessPipeline] = ((tuple(), tuple()),),

        feature_extractor_values: Sequence[fe.FeatureExtractorType] = (fe.FeatureExtractorType.CSP_LOGVAR,),
        classifier_values: Sequence[clf.ClassifierType] = (clf.ClassifierType.LDA,),

        # --- Feature extractor params
        n_components_values: Sequence[int] = (4,),
        reg_values: Sequence[float] = (1e-10,),
        eps_values: Sequence[float] = (1e-12,),
        normalize_var_values: Sequence[bool] = (True,),
        log_var_values: Sequence[bool] = (True,),

        # --- Classifier params
        shrinkage_values: Sequence[bool] = (False,),
        max_iter_values: Sequence[int] = (2000,),
        threshold_i0_values: Sequence[int] = (0,),
        threshold_i1_values: Sequence[int] = (1,),
    ):
        # Validate non-empty grids (pipelines can be "empty pipeline", but the grid must not be empty)
        for name, param in (
            ("preprocessing_pipelines", preprocessing_pipelines),
            ("feature_extractor_values", feature_extractor_values),
            ("classifier_values", classifier_values),
            ("n_components_values", n_components_values),
            ("reg_values", reg_values),
            ("eps_values", eps_values),
            ("normalize_var_values", normalize_var_values),
            ("log_var_values", log_var_values),
            ("shrinkage_values", shrinkage_values),
            ("max_iter_values", max_iter_values),
            ("threshold_i0_values", threshold_i0_values),
            ("threshold_i1_values", threshold_i1_values),
        ):
            if not isinstance(param, (list, tuple)) or len(param) == 0:
                raise ValueError(f"{name} must be a non-empty list or tuple; got {type(param)} of length {len(param)}")

        for p in preprocessing_pipelines:
            _validate_preprocess_pipeline(p)

        self._results: Dict[AblationSettings, Optional[Dict[str, SubjectEvalResult]]] = {}

        for (pp_funcs, pp_kwargs) in preprocessing_pipelines:
            for feature_extractor in feature_extractor_values:
                for classifier in classifier_values:
                    for csp_n_components in n_components_values:
                        for reg in reg_values:
                            for eps in eps_values:
                                for normalize_var in normalize_var_values:
                                    for log_var in log_var_values:
                                        for lda_shrinkage in shrinkage_values:
                                            for max_iter in max_iter_values:
                                                for threshold_i0 in threshold_i0_values:
                                                    for threshold_i1 in threshold_i1_values:
                                                        settings = AblationSettings(
                                                            preprocessing_funcs=tuple(pp_funcs),
                                                            preprocessing_kwargs=tuple(pp_kwargs),
                                                            feature_extractor=feature_extractor,
                                                            classifier=classifier,
                                                            csp_n_components=csp_n_components,
                                                            reg=reg,
                                                            eps=eps,
                                                            normalize_var=normalize_var,
                                                            log_var=log_var,
                                                            lda_shrinkage=lda_shrinkage,
                                                            max_iter=max_iter,
                                                            threshold_i0=threshold_i0,
                                                            threshold_i1=threshold_i1,
                                                        )
                                                        self._results[settings] = None

    @property
    def results(self) -> Dict[AblationSettings, Optional[Dict[str, SubjectEvalResult]]]:
        return self._results

    def _build_feature_extractor(self, settings: AblationSettings) -> fe.FeatureExtractor:
        if settings.feature_extractor == fe.FeatureExtractorType.CSP_LOGVAR:
            return fe.CSPLogVar(
                csp_n_components=settings.csp_n_components,
                reg=settings.reg,
                eps=settings.eps,
                normalize_var=settings.normalize_var,
                log_var=settings.log_var,
            )
        if settings.feature_extractor == fe.FeatureExtractorType.CHANNEL_LOGVAR:
            return fe.ChannelLogVar(
                eps=settings.eps,
                normalize_var=settings.normalize_var,
                log_var=settings.log_var,
            )
        raise ValueError(f"Unhandled feature_extractor: {settings.feature_extractor}")

    def _build_classifier(self, settings: AblationSettings) -> clf.Classifier:
        if settings.classifier == clf.ClassifierType.LDA:
            return clf.LDAClassifier(lda_shrinkage=settings.lda_shrinkage)
        if settings.classifier == clf.ClassifierType.LOGREG:
            return clf.LogRegClassifier(max_iter=settings.max_iter)
        if settings.classifier == clf.ClassifierType.LINSVM:
            return clf.LinSVMClassifier()
        if settings.classifier == clf.ClassifierType.NEAREST_MEAN:
            return clf.NearestMeanClassifier()
        if settings.classifier == clf.ClassifierType.THRESHOLD:
            return clf.ThresholdClassifier(i0=settings.threshold_i0, i1=settings.threshold_i1)
        raise ValueError(f"Unhandled classifier: {settings.classifier}")

    def run(
        self,
        settings: AblationSettings,
        dataset_path: str,
        crossval: Crossval,
        exclude_subject_ids: Sequence[str] = [],
    ) -> Dict[str, SubjectEvalResult]:
        if not isinstance(crossval, Crossval):
            raise ValueError(f"crossval must be an instance of Crossval; got {type(crossval)}")

        ds = Dataset1(dataset_path=dataset_path, exclude_subject_ids=exclude_subject_ids)
        X, y, groups = ds.get_XY(
            preprocessing_funcs=settings.preprocessing_funcs,
            preprocessing_kwargs=settings.preprocessing_kwargs,
        )

        feat = self._build_feature_extractor(settings)
        c = self._build_classifier(settings)
        model = Model(feat=feat, clf=c)

        return crossval.evaluate_all_subjects(model=model, X=X, y=y, groups=groups)

    def run_all(
        self,
        dataset_path: str,
        crossval: Crossval,
        exclude_subject_ids: Sequence[str] = [],
        skip_completed: bool = True,
    ) -> Dict[AblationSettings, Optional[Dict[str, SubjectEvalResult]]]:
        for settings, cur in list(self._results.items()):
            if skip_completed and cur is not None:
                continue
            self._results[settings] = self.run(
                settings=settings,
                dataset_path=dataset_path,
                crossval=crossval,
                exclude_subject_ids=exclude_subject_ids,
            )
        return self._results
