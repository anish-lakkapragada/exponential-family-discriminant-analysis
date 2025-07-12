"""
Code providing all the types for this project.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Union, Dict, Tuple, Literal
import numpy as np 
from scipy.stats import multivariate_normal, norm
from sklearn.covariance import EmpiricalCovariance

NaturalParameters = List[Union[np.ndarray, float]] # natural parameters 
ParamDefaults = Dict[str, Union[List[float], float]] # parameter defaults 
Dataset = Tuple[np.ndarray, np.ndarray]
Speeds = List[float]

class Density(BaseModel): 
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # abstract class for a single class-conditional density 
    def fit_density(self, X: np.ndarray) -> None: 
        pass 
    def evaluate_pdf(self, x: np.ndarray) -> float: 
        pass 

    def set_param_defaults(self, param_defaults: ParamDefaults) -> None: 
        for (param_name, param_value) in param_defaults.items(): 
            if type(param_value) == list: 
                setattr(self, param_name, np.array(param_value))
            elif type(param_value) == float: 
                setattr(self, param_name, param_value)
            else: 
                raise Exception
        

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
    natural_params: Optional[NaturalParameters] = None

Class_Conditional_Density = Union[Normal_LDA_Density, Normal_QDA_Density, ExpoFamilyDensity]

class EvaluationObj: 
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auc_roc: float 
    y_pred: np.ndarray

class RunData(BaseModel): 
    n_train: int 
    n_test: int 
    class_noise_rate: float = Field(..., ge=0.0, le=1.0)
    data_noise_rate: float = Field(..., ge=0.0, le=1.0)
    class_imbalance_true: List[float]
    distributions: List[Literal["weibull"]]
    true_params: List[ParamDefaults]

class RunConfig(BaseModel): 
    data:           RunData
    methods:        List[Literal["lda", "qda", "efda-weibull"]]
    num_classes:    int
    param_defaults: Optional[List[ParamDefaults]]

class EvaluationConfig(BaseModel): 
    name: str 
    store_dir: Optional[str]
    runs: List[RunConfig]
