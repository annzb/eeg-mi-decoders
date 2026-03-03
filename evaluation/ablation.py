from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from data import Dataset, PreprocessPipeline, validate_preprocess_pipeline
from models import ClassifierType, FeatureExtractorType, Model
from evaluation.crossval import Crossval
from evaluation.utils import SubjectEvalResult


@dataclass(frozen=True, slots=True)
class AblationSettings:
    model: Model
    preprocess_pipeline: Optional[PreprocessPipeline] = None
    
    def __post_init__(self):
        if self.preprocess_pipeline is not None:
            validate_preprocess_pipeline(self.preprocess_pipeline)
        if not isinstance(self.model, Model):
            raise ValueError(f"Invalid model: {self.model}")


class Ablation:
    def __init__(
        self,
        # PP
        preprocess_pipeline_values: Sequence[Optional[PreprocessPipeline]] = (None, ),

        # FE
        feature_extractor_values: Sequence[Optional[FeatureExtractorType]] = (None, ),
        fe_reg_values: Sequence[float] = tuple(),
        fe_eps_values: Sequence[float] = tuple(),
        fe_normalize_var_values: Sequence[bool] = tuple(),
        fe_log_var_values: Sequence[bool] = tuple(),
        csp_n_components_values: Sequence[int] = tuple(),

        # CLF
        classifier_values: Sequence[ClassifierType] = (ClassifierType.LDA,),
        lda_shrinkage_values: Sequence[bool] = tuple(),
        logreg_max_iter_values: Sequence[int] = tuple(),
        threshold_i0_values: Sequence[int] = tuple(),
        threshold_i1_values: Sequence[int] = tuple(),
    ):
        for param in (
            preprocess_pipeline_values,
            feature_extractor_values,
            fe_reg_values,
            fe_eps_values,
            fe_normalize_var_values,
            fe_log_var_values,
            csp_n_components_values,
            classifier_values,
            lda_shrinkage_values,
            logreg_max_iter_values,
            threshold_i0_values,
            threshold_i1_values
        ):
            if not hasattr(param, '__len__'):
                raise ValueError(f"All specified ablation settings must be a sequence; got {type(param)}")
        if not preprocess_pipeline_values:
            preprocess_pipeline_values = (None, )
        if not feature_extractor_values:
            feature_extractor_values = (None, )

        self._results: Dict[AblationSettings, Optional[Dict[str, SubjectEvalResult]]] = {}

        extractors = self._init_feature_extractors(
            feature_extractor_values=feature_extractor_values,
            fe_reg_values=fe_reg_values,
            fe_eps_values=fe_eps_values,
            fe_normalize_var_values=fe_normalize_var_values,
            fe_log_var_values=fe_log_var_values,
            csp_n_components_values=csp_n_components_values,
        )
        classifiers = self._init_classifiers(
            classifier_values=classifier_values,
            lda_shrinkage_values=lda_shrinkage_values,
            logreg_max_iter_values=logreg_max_iter_values,
            threshold_i0_values=threshold_i0_values,
            threshold_i1_values=threshold_i1_values,
        )
        for pipeline in preprocess_pipeline_values:
            for feature_extractor in extractors:
                for classifier in classifiers:
                    model = Model(feat=feature_extractor, clf=classifier)
                    settings = AblationSettings(preprocess_pipeline=pipeline, model=model)
                    self._results[settings] = None

    @property
    def results(self) -> Dict[AblationSettings, Optional[Dict[str, SubjectEvalResult]]]:
        return self._results

    def _init_feature_extractors(self, 
        feature_extractor_values: Sequence[FeatureExtractorType] = tuple(),
        fe_reg_values: Sequence[float] = tuple(),
        fe_eps_values: Sequence[float] = tuple(),
        fe_normalize_var_values: Sequence[bool] = tuple(),
        fe_log_var_values: Sequence[bool] = tuple(),
        csp_n_components_values: Sequence[int] = tuple(),
    ) -> Sequence[FeatureExtractor]:
        extractors = []
        for feature_extractor_type in feature_extractor_values:
            if feature_extractor_type is None:
                if None not in extractors:
                    extractors.append(None)
                continue
            elif not isinstance(feature_extractor_type, FeatureExtractorType):
                raise ValueError(f"Invalid feature_extractor_type: {feature_extractor_type}")
            for fe_reg in fe_reg_values:
                for fe_eps in fe_eps_values:
                    for fe_normalize_var in fe_normalize_var_values:
                        for fe_log_var in fe_log_var_values:
                            for csp_n_components in csp_n_components_values:
                                try:
                                    extractor = feature_extractor_type(
                                    reg=fe_reg,
                                    eps=fe_eps,
                                    normalize_var=fe_normalize_var,
                                    log_var=fe_log_var,
                                    csp_n_components=csp_n_components,
                                )
                                except Exception as e:
                                    continue
                                else:
                                    extractors.append(extractor)
        return extractors
    
    def _init_classifiers(
        self,
        classifier_values: Sequence[ClassifierType] = tuple(),
        lda_shrinkage_values: Sequence[bool] = tuple(),
        logreg_max_iter_values: Sequence[int] = tuple(),
        threshold_i0_values: Sequence[int] = tuple(),
        threshold_i1_values: Sequence[int] = tuple(),
    ) -> Sequence[Classifier]:
        classifiers = []
        for classifier_type in classifier_values:
            if not isinstance(classifier_type, ClassifierType):
                raise ValueError(f"Invalid classifier_type: {classifier_type}")
            for lda_shrinkage in lda_shrinkage_values:
                for logreg_max_iter in logreg_max_iter_values:
                    for threshold_i0 in threshold_i0_values:
                        for threshold_i1 in threshold_i1_values:
                            try:
                                classifier = classifier_type(
                                    lda_shrinkage=lda_shrinkage,
                                    logreg_max_iter=logreg_max_iter,
                                    threshold_i0=threshold_i0,
                                    threshold_i1=threshold_i1,
                                )
                            except Exception as e:
                                continue
                            else: 
                                classifiers.append(classifier)
        return classifiers

    def evaluate_settings(
        self,
        settings: AblationSettings,
        dataset: Dataset,
        crossval: Crossval
    ) -> Dict[str, SubjectEvalResult]:
        if not isinstance(settings, AblationSettings):
            raise ValueError(f"settings must be an instance of AblationSettings; got {type(settings)}")
        if not isinstance(dataset, Dataset):
            raise ValueError(f"dataset must be an instance of Dataset; got {type(dataset)}")
        if not isinstance(crossval, Crossval):
            raise ValueError(f"crossval must be an instance of Crossval; got {type(crossval)}")

        X, y, groups = dataset.get_XY(preprocess_pipeline=settings.preprocess_pipeline)
        return crossval.evaluate_all_subjects(model=model, X=X, y=y, groups=groups)

    def evaluate_all_settings(
        self,
        dataset: Dataset,
        crossval: Crossval,
        skip_completed: bool = False,
    ) -> Dict[AblationSettings, Optional[Dict[str, SubjectEvalResult]]]:
        if not isinstance(dataset, Dataset):
            raise ValueError(f"dataset must be an instance of Dataset; got {type(dataset)}")
        for settings, res in list(self._results.items()):
            if skip_completed and isinstance(res, SubjectEvalResult):
                continue
            self._results[settings] = self.evaluate_settings(
                settings=settings,
                dataset=dataset,
                crossval=crossval,
            )
        return self._results
