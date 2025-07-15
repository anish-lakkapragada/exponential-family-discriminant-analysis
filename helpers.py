"""
Code for some helper dictionaries.
"""
from distributions import sample_weibull, sample_wishart, sample_laplace, WishartDensity, WeibullDensity, LaplaceDensity
from models import Normal_LDA_Density, Normal_QDA_Density, DataConfig, Dataset, Class_Conditional_Density
from typing import Dict
import numpy as np 
from netcal.metrics import ECE 

METHOD_TO_DENSITY: Dict[str, Class_Conditional_Density]  = {
    "lda": Normal_LDA_Density, 
    "qda": Normal_QDA_Density, 
    "efda-weibull": WeibullDensity, 
    "efda-wishart": WishartDensity,
    "efda-laplace": LaplaceDensity
}

DISTRIBUTION_TO_SAMPLE_GENERATION = {
    "weibull": sample_weibull, 
    "wishart": sample_wishart, 
    "laplace": sample_laplace
}

def generate_data(run_data: DataConfig) -> Dataset: 
    tot = run_data.n_train + run_data.n_test 
    X_tot, y_tot = [], []
    for i, distribution in enumerate(run_data.distributions):
        alpha_i = run_data.class_imbalance_true[i] 
        true_param = run_data.true_params[i]
        n_samples_k = int(tot * alpha_i)

        X_k = DISTRIBUTION_TO_SAMPLE_GENERATION[distribution](n_samples=n_samples_k, 
                                                    data_noise_rate=run_data.data_noise_rate, **true_param)
        y_k = np.zeros((n_samples_k)) + i
        X_tot.append(X_k)
        y_tot.append(y_k)

    X_, y_ = np.concat(X_tot, axis=0), np.concat(y_tot, axis=0)
    perm = np.random.permutation(X_.shape[0])
    X, y = X_[perm], y_[perm]
    X_train, y_train, X_test, y_test = X[:run_data.n_train], y[:run_data.n_train], X[run_data.n_train:], y[run_data.n_train:]
    return X_train, y_train, X_test, y_test  

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins=10): 
    ece = ECE(n_bins) 
    return ece.measure(y_pred, y_true)