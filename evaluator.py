from models import Class_Conditional_Density, Normal_LDA_Density, Speeds, EvaluationObj, EvaluationConfig
from helpers import generate_data, METHOD_TO_DENSITY
from typing import List 
import numpy as np
import time
from rich.panel import Panel
from rich import print 
from pathlib import Path 

class Evaluator: 
    distributions: List[Class_Conditional_Density]
    num_classes: int 
    class_weight_mles: List[float]

    def __init__(self, distributions: List[Class_Conditional_Density], num_classes: int) -> None: 
        self.distributions = distributions 
        self.num_classes = num_classes

    def fit_class_densities(self, X: np.ndarray, y: np.ndarray) -> Speeds:
         
        distribution_types_lda = [type(distribution) == Normal_LDA_Density for distribution in self.distributions]
        
        using_lda = False 
        if not all(distribution_types_lda) and any(distribution_types_lda): 
            return "Cannot do LDA for a single class."
        elif all(distribution_types_lda): 
            using_lda = True 

        assert len(self.distributions) == self.num_classes, "Not required amount of distributions provided."
        
        speeds = [0] * self.num_classes
        self.class_weight_mles = [0] * self.num_classes

        for class_k in range(self.num_classes): 
            idx = y == class_k 
            self.class_weight_mles[class_k] = np.sum(idx) / X.shape[0]

            # fit the actual density. 
            start_time = time.time()
            self.distributions[class_k].fit_density(X[idx])
            end_time = time.time() 
            speeds[class_k] = end_time - start_time 
        
        assert np.abs(np.sum(self.class_weight_mles) - 1.00) < 1e-5, np.abs(np.sum(self.class_weight_mles) - 1.00)
        
        if using_lda: 
            # update all shared covariance matrices. 

            d = X.shape[1]
            total_scatter = np.zeros((d, d))

            for class_k in range(self.num_classes):
                Xk = X[y == class_k]
                mu_k = self.distributions[class_k].mean        
                diff = Xk - mu_k                               
                total_scatter += diff.T.dot(diff)              

            N = X.shape[0]
            pooled_cov = total_scatter / N                  

            for dist in self.distributions:
                dist.cov_matrix = pooled_cov
        
        return speeds 


    def evaluate(self, X: np.ndarray, y: np.ndarray, plot_auc: False) -> EvaluationObj: 
        pass 

if __name__ == "__main__": 
    json_text = Path("config.json").read_text(encoding="utf-8")
    evaluation_cfg = EvaluationConfig.model_validate_json(json_text)
    for run_num, run in enumerate(evaluation_cfg.runs):
        print(Panel(f"[blue] Running Run #{run_num + 1}."))

        X_train, y_train, X_test, y_test = generate_data(run.data)
        
        run_densities = [] 
        for i, method in enumerate(run.methods): 
            density = METHOD_TO_DENSITY[method]()
            if run.param_defaults: 
                param_defaults = run.param_defaults[i]

                density.set_param_defaults(param_defaults)
            run_densities.append(density)

        evaluator = Evaluator(num_classes=run.num_classes, distributions=run_densities)
        speeds = evaluator.fit_class_densities(X_train, y_train)
        
        

        