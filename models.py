"""
Code providing all the types for this project.
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Union, Dict, Tuple
import numpy as np 
from scipy.stats import multivariate_normal, norm
from sklearn.covariance import EmpiricalCovariance

NaturalParameters = List[Union[np.ndarray, float]] # natural parameters 
ParamDefaults = Dict[str, Union[np.ndarray, float]] # parameter defaults 
Dataset = Tuple[np.ndarray, np.ndarray]
Speeds = List[float]

class Density(BaseModel): 
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # abstract class for a single class-conditional density 
    def fit_density(self, X: np.ndarray) -> None: 
        pass 
    def evaluate_pdf(self, x: np.ndarray) -> float: 
        pass 


class Normal_LDA_Density(Density): 
    mean: np.ndarray
    cov_matrix: np.ndarray # will be shared across all classes.

    def fit_density(self, X: np.ndarray) -> None:
        self.cov_matrix = None # deal with this later!
        self.mean = np.mean(X)

    def evaluate_pdf(self, x: np.ndarray) -> float:
        if len(self.cov_matrix.shape) == 2: 
            return multivariate_normal(mean=self.mean, cov=self.cov_matrix).pdf(x)
        else: 
            return norm(loc=self.mean, scale=np.sqrt(self.cov_matrix)).pdf(x)

class Normal_QDA_Density(Normal_LDA_Density): 
    def fit_density(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X_for_cov = X.reshape(-1, 1)
        else:
            X_for_cov = X

        cov_estimator = EmpiricalCovariance().fit(X_for_cov)
        self.mean = cov_estimator.location_.flatten()
        self.cov_matrix = cov_estimator.covariance_

class ExpoFamilyDensity(Density): 
    natural_params: NaturalParameters

    def set_param_defaults(self, param_defaults: ParamDefaults) -> None: 
        for (param_name, param_value) in param_defaults.items(): 
            setattr(self, param_name, param_value)
        

Class_Conditional_Density = Union[Normal_LDA_Density, Normal_QDA_Density, ExpoFamilyDensity]

class EvaluationObj: 
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auc_roc: float 
    y_pred: np.ndarray
