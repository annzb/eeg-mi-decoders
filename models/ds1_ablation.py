from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from data.dataset import Dataset1
from models import classifier as clf, feature_extractor as fe
from models.base import Model
from models.crossval import Crossval
from models.evaluation import SubjectEvalResult


@dataclass(frozen=True, slots=True)
class Ds1AblationSettings:
    preprocess_highpass: bool = True
    preprocess_car: bool = True
    preprocess_band: bool = True
    feature_extractor: fe.FeatureExtractorType = fe.FeatureExtractorType.CSP_LOGVAR
    classifier: clf.ClassifierType = clf.ClassifierType.LDA
    n_components: int = 4
    reg: float = 1e-10
    normalize_var: bool = True
    log_var: bool = True

    def __post_init__(self):
        if not isinstance(self.preprocess_highpass, bool):
            raise ValueError(f"preprocess_highpass must be a bool, got {self.preprocess_highpass!r}")
        if not isinstance(self.preprocess_car, bool):
            raise ValueError(f"preprocess_car must be a bool, got {self.preprocess_car!r}")
        if not isinstance(self.preprocess_band, bool):
            raise ValueError(f"preprocess_band must be a bool, got {self.preprocess_band!r}")
        if not isinstance(self.feature_extractor, fe.FeatureExtractorType):
            raise ValueError(f"feature_extractor must be a FeatureExtractorType, got {self.feature_extractor!r}. Allowed: {list(fe.FeatureExtractorType)}")
        if not isinstance(self.classifier, clf.ClassifierType):
            raise ValueError(f"classifier must be a ClassifierType, got {self.classifier!r}. Allowed: {list(clf.ClassifierType)}")
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError(f"n_components must be a positive int, got {self.n_components!r}")
        if not isinstance(self.reg, float) or not np.isfinite(self.reg) or self.reg < 0:
            raise ValueError(f"reg must be a finite float >= 0, got {self.reg!r}")
        if not isinstance(self.normalize_var, bool):
            raise ValueError(f"normalize_var must be a bool, got {self.normalize_var!r}")
        if not isinstance(self.log_var, bool):
            raise ValueError(f"log_var must be a bool, got {self.log_var!r}")


class Ablation:
    def __init__(
        self,
        preprocess_highpass_values: Sequence[bool] = (True,),
        preprocess_car_values: Sequence[bool] = (True,),
        preprocess_band_values: Sequence[bool] = (True,),
        feature_extractor_values: Sequence[fe.FeatureExtractorType] = (fe.FeatureExtractorType.CSP_LOGVAR,),
        classifier_values: Sequence[clf.ClassifierType] = (clf.ClassifierType.LDA,),
        n_components_values: Sequence[int] = (4,),
        reg_values: Sequence[float] = (1e-10,),
        normalize_var_values: Sequence[bool] = (True,),
        log_var_values: Sequence[bool] = (True,),
    ):
        for name, param in (
            ("preprocess_highpass_values", preprocess_highpass_values),
            ("preprocess_car_values", preprocess_car_values),
            ("preprocess_band_values", preprocess_band_values),
            ("feature_extractor_values", feature_extractor_values),
            ("classifier_values", classifier_values),
            ("n_components_values", n_components_values),
            ("reg_values", reg_values),
            ("normalize_var_values", normalize_var_values),
            ("log_var_values", log_var_values),
        ):
            if not isinstance(param, (list, tuple)) or len(param) == 0:
                raise ValueError(f"{name} must be a non-empty list or tuple, got {type(param)} of length {len(param)}")

        self._results: Dict[Ds1AblationSettings, Optional[Dict[str, SubjectEvalResult]]] = {}
        for preprocess_highpass in preprocess_highpass_values:
            for preprocess_car in preprocess_car_values:
                for preprocess_band in preprocess_band_values:
                    for feature_extractor in feature_extractor_values:
                        for classifier in classifier_values:
                            for n_components in n_components_values:
                                for reg in reg_values:
                                    for normalize_var in normalize_var_values:
                                        for log_var in log_var_values:
                                            settings = Ds1AblationSettings(
                                                preprocess_highpass=preprocess_highpass, preprocess_car=preprocess_car, preprocess_band=preprocess_band,
                                                feature_extractor=feature_extractor, classifier=classifier, n_components=n_components, reg=reg,
                                                normalize_var=normalize_var, log_var=log_var
                                            )
                                            self._results[settings] = None

    @property
    def results(self) -> Dict[Ds1AblationSettings, Optional[Dict[str, SubjectEvalResult]]]:
        return self._results

    def run(
        self,
        settings: Ds1AblationSettings,
        dataset_path: str,
        crossval: Crossval,
        exclude_subject_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, SubjectEvalResult]:
        if not isinstance(crossval, Crossval):
            raise ValueError(f"crossval must be an instance of Crossval, got {type(crossval)}")

        ds = Dataset1(
            dataset_path=dataset_path,
            exclude_subject_ids=exclude_subject_ids,
            preprocess_highpass=settings.preprocess_highpass,
            preprocess_car=settings.preprocess_car,
            preprocess_band=settings.preprocess_band,
        )
        X, y, groups = ds.get_XY()

        if settings.feature_extractor == fe.FeatureExtractorType.CSP_LOGVAR:
            feat = fe.CSPLogVar(n_components=settings.n_components, reg=settings.reg, normalize_var=settings.normalize_var, log_var=settings.log_var)
        elif settings.feature_extractor == fe.FeatureExtractorType.CHANNEL_LOGVAR:
            feat = fe.ChannelLogVar(normalize_var=settings.normalize_var, log_var=settings.log_var)
        else:
            raise ValueError(f"Unhandled feature_extractor: {settings.feature_extractor}")

        if settings.classifier == clf.ClassifierType.LDA:
            c = clf.LDAClassifier(shrinkage=False)
        elif settings.classifier == clf.ClassifierType.SHRINKAGE_LDA:
            c = clf.ShrinkageLDAClassifier()
        elif settings.classifier == clf.ClassifierType.LOGREG:
            c = clf.LogRegClassifier()
        elif settings.classifier == clf.ClassifierType.LINSVM:
            c = clf.LinSVMClassifier()
        elif settings.classifier == clf.ClassifierType.NEAREST_MEAN:
            c = clf.NearestMeanClassifier()
        elif settings.classifier == clf.ClassifierType.THRESHOLD:
            c = clf.ThresholdClassifier(i0=0, i1=1)
        else:
            raise ValueError(f"Unhandled classifier: {settings.classifier}")

        model = Model(feat=feat, clf=c)
        return crossval.evaluate_all_subjects(model=model, X=X, y=y, groups=groups)

    def run_all(
        self,
        dataset_path: str,
        crossval: Crossval,
        exclude_subject_ids: Optional[Sequence[str]] = None,
        skip_completed: bool = True,
    ) -> Dict[Ds1AblationSettings, Optional[Dict[str, SubjectEvalResult]]]:
        for settings, cur in list(self._results.items()):
            if skip_completed and cur is not None:
                continue
            self._results[settings] = self.run(settings=settings, dataset_path=dataset_path, crossval=crossval, exclude_subject_ids=exclude_subject_ids)
        return self._results
