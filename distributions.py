"""
Code providing all the class-conditional densities & data-generators in this project. 
"""
# %%
from models import ExpoFamilyDensity
import numpy as np
from scipy.stats import weibull_min

class WeibullDensity(ExpoFamilyDensity): 
    def fit_density(self, X):
        N_C = X.shape[0]
        eta = - N_C / np.sum(X**self.k)
        self.natural_params = [eta]
    
    def eta_to_lambda(eta, k): 
        return (-1 / eta) ** (1 / k)
    
    def evaluate_pdf(self, x):
        return weibull_min.pdf(x, c=self.k, scale=self.eta_to_lambda(self.natural_params[0], self.k))

def sample_weibull(n_samples: int, seed: int = 42, data_noise_rate=0.00, class_noise_rate=0.00, **kwargs) -> None: 
    return weibull_min.rvs( size=n_samples, random_state=seed, **kwargs)
# %%
