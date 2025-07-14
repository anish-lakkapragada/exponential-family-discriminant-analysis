"""
Code providing all the types for this project.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Union, Dict, Tuple, Literal
import numpy as np 
from scipy.stats import multivariate_normal, norm
from sklearn.covariance import EmpiricalCovariance

NaturalParameters = List[Union[np.ndarray, float]] # natural parameters 
ParamDefaults = Dict[str, Union[List[float], List[List[float]], float]] # parameter defaults 
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
    mean: Optional[np.ndarray] = None
    cov_matrix: Optional[np.ndarray] = None # will be shared across all classes.

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

class DataConfig(BaseModel): 
    n_train: int 
    n_test: int 
    num_classes: int
    class_noise_rate: float = Field(..., ge=0.0, le=1.0)
    data_noise_rate: float = Field(..., ge=0.0, le=1.0)
    class_imbalance_true: List[float]
    distributions: List[Literal["weibull", "wishart"]]
    true_params: List[ParamDefaults]

class MethodConfig(BaseModel): 
    densities: List[Literal["lda", "qda", "efda-weibull", "efda-wishart", "log-reg"]]
    param_defaults: Optional[List[ParamDefaults]] = None
    custom: bool = False 

class ExperimentConfig(BaseModel): 
    experiment_name: str
    data: DataConfig
    n_trials: int
    tests: List[MethodConfig]

class EvaluationConfig(BaseModel): 
    name: str 
    store_dir: Optional[str]
    experiments: List[ExperimentConfig]

class EvalResult(BaseModel): 
    model_config = ConfigDict(arbitrary_types_allowed=True)
    auc_roc: float
    acc: float
    # y_pred: np.ndarray

class ExperimentEvaluation: 
    auc_roc: float 
    auc_roc_std: float 
    acc: float
    acc_std: float 
    eval_results : List[EvalResult] = [] 
    
    def add_result(self, eval_result: EvalResult): 
        self.eval_results.append(eval_result)
        aucs, accs = [res.auc_roc for res in self.eval_results], [res.acc for res in self.eval_results]
        self.auc_roc = np.mean(aucs)
        self.auc_roc_std = np.std(aucs)
        self.acc = np.mean(accs)
        self.acc_std = np.std(accs)
    
    def get_auc(self): 
        return f"{round(self.auc_roc, 4)} ± {round(self.auc_roc_std, 4)}"
    
    def get_acc(self): 
        return f"{round(self.acc, 4)} ± {round(self.acc_std, 4)}"
    