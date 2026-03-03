from dataclasses import dataclass
from typing import Optional
import numpy as np

from data import PreprocessPipeline, validate_preprocess_pipeline
from models import Classifier, FeatureExtractor


@dataclass(frozen=True, slots=True)
class AblationSettings:
    # preprocessing (PP)
    preprocess_pipeline: PreprocessPipeline = tuple()
    feature_extractor: FeatureExtractor
    classifier: Classifier

    # feature extraction (FE)
    # feature_extractor: Optional[FeatureExtractorType] = None
    # fe_reg: Optional[float] = None
    # fe_eps: Optional[float] = None
    # fe_normalize_var: Optional[bool] = None
    # fe_log_var: Optional[bool] = None
    # csp_n_components: Optional[int] = None

    # classifier (CLF)
    # classifier: ClassifierType = ClassifierType.LDA
    # lda_shrinkage: Optional[bool] = None
    # logreg_max_iter: Optional[int] = None
    # threshold_i0: Optional[int] = None
    # threshold_i1: Optional[int] = None

    def __post_init__(self):
        # PP
        validate_preprocess_pipeline(self.preprocess_pipeline)
        if not isinstance(self.feature_extractor, FeatureExtractor):
            raise ValueError(f"Invalid feature_extractor: {self.feature_extractor}")
        if not isinstance(self.classifier, Classifier):
            raise ValueError(f"Invalid classifier: {self.classifier}")

        ## FE
        # if self.feature_extractor is not None:
        #     if not isinstance(self.feature_extractor, FeatureExtractorType):
        #         raise ValueError(f"Invalid feature_extractor: {self.feature_extractor}")
        #     if not isinstance(self.fe_reg, float) or not np.isfinite(self.fe_reg) or self.fe_reg < 0:
        #         raise ValueError(f"fe_reg must be finite and >= 0; got {self.fe_reg}")
        #     if not isinstance(self.fe_eps, float) or not np.isfinite(self.fe_eps) or self.fe_eps <= 0:
        #         raise ValueError(f"fe_eps must be finite and > 0; got {self.fe_eps}")
        #     if not isinstance(self.fe_normalize_var, bool):
        #         raise ValueError("fe_normalize_var must be bool")
        #     if not isinstance(self.fe_log_var, bool):
        #         raise ValueError("fe_log_var must be bool")
        #             if self.feature_extractor == FeatureExtractorType.CSP_LOGVAR:
        #     if self.csp_n_components is not None and (
        #         not isinstance(self.csp_n_components, int) 
        #         or self.csp_n_components <= 0
        #         or self.csp_n_components % 2 != 0
        #     ):
        #         raise ValueError("csp_n_components must be an even positive int")

        # CLF
        # if not isinstance(self.classifier, ClassifierType):
        #     raise ValueError(f"Invalid classifier: {self.classifier}")
        # if self.lda_shrinkage is not None and not isinstance(self.lda_shrinkage, bool):
        #     raise ValueError("lda_shrinkage must be bool")
        # if self.logreg_max_iter is not None and (
        #     not isinstance(self.logreg_max_iter, int) 
        #     or self.logreg_max_iter <= 0
        # ):
        #     raise ValueError("logreg_max_iter must be a positive int")
        # if self.threshold_i0 is not None and self.threshold_i1 is not None and (
        #     not isinstance(self.threshold_i0, int) 
        #     or not isinstance(self.threshold_i1, int)
        # ):
        #     raise ValueError("threshold_i0 and threshold_i1 must be ints")
        # if self.threshold_i0 == self.threshold_i1:
        #     raise ValueError("threshold_i0 and threshold_i1 must differ")
