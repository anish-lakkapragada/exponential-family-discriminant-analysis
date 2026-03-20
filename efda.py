"""
efda.py — Exponential Family Discriminant Analysis

Core implementation: distribution specifications, EFDA classifier (binary + multiclass),
data generators, and evaluation utilities.
"""

import numpy as np
from scipy import stats, special
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings("ignore")


# ===========================================================================
# Exponential Family Distribution Specifications
# ===========================================================================

class WeibullDist:
    """Weibull(lambda, k) with known shape k.

    Exponential family parametrization (k known):
        eta = -1 / lambda^k,  T(x) = x^k,  A(eta) = log(-1 / (eta * k))
    """
    name = "Weibull"

    def __init__(self, k):
        self.k = k

    def T(self, x):
        return np.asarray(x, float) ** self.k

    def log_h(self, x):
        x = np.asarray(x, float)
        return (self.k - 1) * np.log(np.clip(x, 1e-300, None))

    def A(self, eta):
        return np.log(-1.0 / (eta * self.k))

    def dA(self, eta):
        return -1.0 / eta

    def solve_eta(self, T_mean):
        """MLE: dA(eta) = T_mean  =>  eta = -1/T_mean."""
        return -1.0 / T_mean

    def log_pdf(self, x, eta):
        return self.log_h(x) + eta * self.T(x) - self.A(eta)

    def sample(self, eta, n, rng=None):
        lam = (-1.0 / eta) ** (1.0 / self.k)
        return stats.weibull_min.rvs(c=self.k, scale=lam, size=n, random_state=rng)

    def eta_from_scale(self, lam):
        return -1.0 / (lam ** self.k)


class GammaDist:
    """Gamma(a, theta) with known shape a.

    Exponential family parametrization (a known):
        eta = -1/theta,  T(x) = x,  A(eta) = -a * log(-eta)
    """
    name = "Gamma"

    def __init__(self, a):
        self.a = a

    def T(self, x):
        return np.asarray(x, float)

    def log_h(self, x):
        x = np.asarray(x, float)
        return (self.a - 1) * np.log(np.clip(x, 1e-300, None)) - special.gammaln(self.a)

    def A(self, eta):
        return -self.a * np.log(-eta)

    def dA(self, eta):
        return -self.a / eta

    def solve_eta(self, T_mean):
        """MLE: dA(eta) = T_mean  =>  eta = -a/T_mean."""
        return -self.a / T_mean

    def log_pdf(self, x, eta):
        return self.log_h(x) + eta * self.T(x) - self.A(eta)

    def sample(self, eta, n, rng=None):
        theta = -self.a / eta
        return stats.gamma.rvs(a=self.a, scale=theta, size=n, random_state=rng)

    def eta_from_scale(self, theta):
        return -self.a / theta


class ExponentialDist:
    """Exponential(theta) — special case of Gamma with a=1.

    Exponential family parametrization:
        eta = -1/theta,  T(x) = x,  A(eta) = -log(-eta)
    """
    name = "Exponential"

    def T(self, x):
        return np.asarray(x, float)

    def log_h(self, x):
        return np.zeros(np.asarray(x).shape, dtype=float)

    def A(self, eta):
        return -np.log(-eta)

    def dA(self, eta):
        return -1.0 / eta

    def solve_eta(self, T_mean):
        return -1.0 / T_mean

    def log_pdf(self, x, eta):
        return self.log_h(x) + eta * self.T(x) - self.A(eta)

    def sample(self, eta, n, rng=None):
        theta = -1.0 / eta
        return stats.expon.rvs(scale=theta, size=n, random_state=rng)

    def eta_from_scale(self, theta):
        return -1.0 / theta


class PoissonDist:
    """Poisson(lambda).

    Exponential family parametrization:
        eta = log(lambda),  T(x) = x,  A(eta) = exp(eta)
    """
    name = "Poisson"

    def T(self, x):
        return np.asarray(x, float)

    def log_h(self, x):
        return -special.gammaln(np.asarray(x, float) + 1)

    def A(self, eta):
        return np.exp(eta)

    def dA(self, eta):
        return np.exp(eta)

    def solve_eta(self, T_mean):
        """MLE: exp(eta) = T_mean  =>  eta = log(T_mean)."""
        return np.log(T_mean)

    def log_pdf(self, x, eta):
        return self.log_h(x) + eta * self.T(x) - self.A(eta)

    def sample(self, eta, n, rng=None):
        lam = np.exp(eta)
        return stats.poisson.rvs(mu=lam, size=n, random_state=rng).astype(float)

    def eta_from_rate(self, lam):
        return np.log(lam)


class NegativeBinomialDist:
    """Negative Binomial(r, mu) with known dispersion r.

    Counts the number of failures X before r successes.
    E[X] = r * mu / r = mu  (parameterized by mean mu).

    Exponential family parametrization (r known):
        eta = log(mu / (r + mu))  in (-inf, 0),
        T(x) = x,
        A(eta) = -r * log(1 - exp(eta)),
        h(x)   = C(x + r - 1, x).

    MLE: eta_k = log( x_bar_k / (r + x_bar_k) ),
         where x_bar_k is the within-class sample mean.
    """
    name = "NegativeBinomial"

    def __init__(self, r):
        self.r = float(r)

    def T(self, x):
        return np.asarray(x, float)

    def log_h(self, x):
        x = np.asarray(x, float)
        # log C(x+r-1, x) = log Gamma(x+r) - log Gamma(r) - log Gamma(x+1)
        return (special.gammaln(x + self.r)
                - special.gammaln(self.r)
                - special.gammaln(x + 1))

    def A(self, eta):
        # -r * log(1 - exp(eta));  eta < 0 so exp(eta) in (0,1)
        return -self.r * np.log(1.0 - np.exp(np.clip(eta, -500, -1e-10)))

    def dA(self, eta):
        e = np.exp(np.clip(eta, -500, -1e-10))
        return self.r * e / (1.0 - e)

    def solve_eta(self, T_mean):
        """MLE: dA(eta) = T_mean  =>  eta = log(T_mean / (r + T_mean))."""
        T_mean = max(float(T_mean), 1e-9)
        return np.log(T_mean / (self.r + T_mean))

    def log_pdf(self, x, eta):
        return self.log_h(x) + eta * self.T(x) - self.A(eta)

    def sample(self, eta, n, rng=None):
        # p_success (scipy) = 1 - exp(eta)  (= r/(r+mu))
        p_success = 1.0 - np.exp(np.clip(eta, -500, -1e-10))
        p_success = np.clip(p_success, 1e-9, 1 - 1e-9)
        return stats.nbinom.rvs(n=self.r, p=p_success, size=n, random_state=rng).astype(float)

    def eta_from_mean(self, mu):
        mu = max(float(mu), 1e-9)
        return np.log(mu / (self.r + mu))


# ===========================================================================
# EFDA Classifier
# ===========================================================================

class EFDA(BaseEstimator, ClassifierMixin):
    """Exponential Family Discriminant Analysis.

    Generative classifier for data whose class-conditional distributions
    belong to the same exponential family.  Fits class priors alpha_k and
    natural parameters eta_k by closed-form MLE, then classifies via MAP.

    Parameters
    ----------
    distribution : instance of WeibullDist, GammaDist, etc.
    """

    def __init__(self, distribution):
        self.distribution = distribution

    def fit(self, X, y):
        X = np.asarray(X, float).ravel()
        self.classes_ = np.unique(y)
        n = len(y)
        self.alpha_ = {}
        self.eta_ = {}
        for k in self.classes_:
            mask = y == k
            N_k = mask.sum()
            self.alpha_[k] = N_k / n
            self.eta_[k] = self.distribution.solve_eta(self.distribution.T(X[mask]).mean())
        return self

    def _log_joint(self, X):
        X = np.asarray(X, float).ravel()
        n = len(X)
        log_j = np.zeros((n, len(self.classes_)))
        for i, k in enumerate(self.classes_):
            log_j[:, i] = (
                np.log(self.alpha_[k])
                + self.distribution.log_pdf(X, self.eta_[k])
            )
        return log_j

    def predict(self, X):
        return self.classes_[np.argmax(self._log_joint(X), axis=1)]

    def predict_proba(self, X):
        lj = self._log_joint(X)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)


# ===========================================================================
# Data Generation
# ===========================================================================

def generate_binary_data(dist, eta0, eta1, alpha, n, rng):
    """Sample n observations from a binary EFDA model.

    Y ~ Bernoulli(alpha); X | Y=k ~ dist(eta_k).
    """
    y = (rng.uniform(size=n) < alpha).astype(int)
    X = np.empty(n)
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    if len(idx1):
        X[idx1] = dist.sample(eta1, len(idx1), rng=rng.randint(0, 2**31))
    if len(idx0):
        X[idx0] = dist.sample(eta0, len(idx0), rng=rng.randint(0, 2**31))
    return X, y


def generate_multiclass_data(dist, etas, alphas, n, rng):
    """Sample n observations from a K-class EFDA model."""
    K = len(etas)
    y = rng.choice(K, size=n, p=alphas)
    X = np.empty(n)
    for k in range(K):
        idx = np.where(y == k)[0]
        if len(idx):
            X[idx] = dist.sample(etas[k], len(idx), rng=rng.randint(0, 2**31))
    return X, y


# ===========================================================================
# Evaluation Utilities
# ===========================================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece
