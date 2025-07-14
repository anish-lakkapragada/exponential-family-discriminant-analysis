"""
Code providing all the class-conditional densities & data-generators in this project. 
"""
# %%
from models import ExpoFamilyDensity
import numpy as np
from scipy.stats import weibull_min, wishart    
from typing import Optional

class WeibullDensity(ExpoFamilyDensity): 
    k: Optional[float] = None 

    def fit_density(self, X):
        N_C = X.shape[0]
        eta = - N_C / np.sum(X**self.k)
        self.natural_params = [eta]
    
    def eta_to_lambda(eta, k): 
        return (-1 / eta) ** (1 / k)
    
    def evaluate_pdf(self, x):
        return weibull_min.pdf(x, c=self.k, scale=WeibullDensity.eta_to_lambda(self.natural_params[0], self.k))

class WishartDensity(ExpoFamilyDensity): 
    p: Optional[int] = None
    n: Optional[int]= None # degrees of freedom 
    V: Optional[np.ndarray] = None # scale array 

    def fit_density(self, X):
        assert self.p == X.shape[1], "Dimensions are not matching."
        N_C = X.shape[0]
        Z = np.sum(X, axis=0) / (self.n * N_C)
        self.V = (Z + np.diag(np.diag(Z))) / 2

    def evaluate_pdf(self, x):
        return np.array([
            wishart.pdf(x_i, df=self.n, scale=self.V)
            for x_i in x
        ])


def sample_wishart(n_samples: int, seed: int = 42, data_noise_rate=0.00, **kwargs): 
    return wishart.rvs(size=n_samples, **kwargs) # df and scale 

def sample_weibull(n_samples: int, seed: int = 42, data_noise_rate=0.00, **kwargs): 
    return weibull_min.rvs(size=n_samples, **kwargs)
# %%
