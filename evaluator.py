from models import Class_Conditional_Density, Normal_LDA_Density, Speeds, EvaluationObj
from typing import List 
import numpy as np
import time 

class Evaluator: 
    distributions: List[Class_Conditional_Density]
    num_classes: int 
    class_weight_mles: List[float]

    def fit_class_densities(self, X: np.ndarray, y: np.ndarray) -> Speeds:
         
        distribution_types_lda = [type(distribution) == Normal_LDA_Density for distribution in self.distributions]
        
        using_lda = False 
        if not all(distribution_types_lda) and any(distribution_types_lda): 
            return "Cannot do LDA for a single class."
        elif all(distribution_types_lda): 
            using_lda = True 

        assert len(self.distributions) == self.num_classes, "Not required amount of distributions provided."
        
        speeds = [0] * self.num_classes
        
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


    def evaluate(self, X: np.ndarray, y: np.ndarray, plot_auc: False) -> EvaluationObj: 
        pass 
