# %% 
"""
Test of Exponential Family Discriminant Analysis (EFDA) for the Weibull distribution. 
"""
import numpy as np 
from scipy.stats import weibull_min
import matplotlib.pyplot as plt 

ALPHA_TRUE = 0.7
K_TRUE = 5 # common scale 

def plot_distributions(class_one_lambda, class_zero_lambda):
    x = np.linspace(0.01, 5, 1000)

    f1 = ALPHA_TRUE * weibull_min.pdf(x, c=K_TRUE, scale=class_one_lambda)  # P[X, Y = 1]
    f0 = (1 - ALPHA_TRUE) * weibull_min.pdf(x, c=K_TRUE, scale=class_zero_lambda)  # P[X, Y = 0]

    denom = f1 + f0
    p_x = f1 / denom
    p_x_complement = f0 / denom

    plt.plot(x,f1, label=r"$\mathbb{P}[Y = 1, X]$")
    plt.plot(x,f0, label=r"$\mathbb{P}[Y = 0, X]$")

    plt.plot(x, p_x, label=r"$\mathbb{P}[Y = 1 \mid X]$")
    plt.plot(x, p_x_complement, label=r"$\mathbb{P}[Y = 0 \mid X]$")
    plt.xlabel("X")
    plt.ylabel("Probability")
    # plt.title(r"Our Data with $\alpha = 0.7$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# %%
CLASS_ONE_TRUE_LAMBDA = 2
CLASS_ZERO_TRUE_LAMBDA = 4
ALPHA_TRUE = 0.7
K_TRUE = 3 # common across both classes

def generate_data(n_samples): 
    X_class_one = weibull_min.rvs(c=K_TRUE, scale=CLASS_ONE_TRUE_LAMBDA, size=n_samples, random_state=42)
    X_class_zero = weibull_min.rvs(c=K_TRUE, scale=CLASS_ZERO_TRUE_LAMBDA, size=n_samples, random_state=42)
    y = (np.random.uniform(0, 1, size=n_samples) <= 0.7).astype(int) 
    X = y * X_class_one + (1 - y) * X_class_zero
    return X, y 

"""
Show the logistic regression vs. LDA Log-Odds Function.
"""
from sklearn.linear_model import LogisticRegression
n = 10000
N_trials = 100

data_values = np.linspace(0.01, 5, 1000)

def compute_logistic_regression_log_odds(log_reg, X): 
    # compute logistic regression log-odds for all values in X
    return (log_reg.coef_ * X + log_reg.intercept_).flatten()

def efda_mles(X, y):
    N_1 = np.sum(y)
    N_0 = y.shape[0] - N_1 
    alpha_hat = np.sum(y) / y.shape[0]
    eta_1_hat = -N_1 / np.sum(y * X**K_TRUE)
    eta_0_hat = -N_0 / np.sum((1 - y) * X**K_TRUE)
    return alpha_hat, eta_1_hat, eta_0_hat

def eta_to_lambda(eta): 
    return (-1 / eta) ** (1 / K_TRUE)

def compute_efda_log_odds(X, X_data, y):
    # compute log-odds for EFDA for all values in X
    alpha_hat, eta_1_hat, eta_0_hat = efda_mles(X_data, y)
    class_one_pdf = weibull_min.pdf(X, c=K_TRUE, scale=eta_to_lambda(eta_1_hat))
    class_zero_pdf =   weibull_min.pdf(X, c=K_TRUE, scale=eta_to_lambda(eta_0_hat))
    return np.log((alpha_hat * class_one_pdf) / ((1 - alpha_hat) * class_zero_pdf)).flatten()


for trial in range(N_trials): 
    X, y = generate_data(n)
    log_reg = LogisticRegression(fit_intercept=True)
    log_reg.fit(X.reshape(-1, 1), y)
    log_reg_log_odds = compute_logistic_regression_log_odds(log_reg, data_values)
    efda_log_odds = compute_efda_log_odds(data_values, X, y)

    label_log_reg = "Logistic Regression" if trial == 0 else None 
    label_efda = "EFDA" if trial == 0 else None
    lw = 1 if trial == 0 else 0.1
    plt.plot(data_values, log_reg_log_odds, color="blue", linewidth=lw, label=label_log_reg)
    plt.plot(data_values, efda_log_odds, color="orange", linewidth=lw, label=label_efda)

true_log_odds = np.log(ALPHA_TRUE / (1 - ALPHA_TRUE)) + (K_TRUE) * np.log(CLASS_ZERO_TRUE_LAMBDA / CLASS_ONE_TRUE_LAMBDA) + np.pow(data_values, K_TRUE) * ((1 / CLASS_ZERO_TRUE_LAMBDA ** K_TRUE) - (1 / CLASS_ONE_TRUE_LAMBDA ** K_TRUE))
plt.plot(data_values, true_log_odds.flatten(), label="True Log Odds Function", color="green")

plt.xlabel("X")
plt.ylabel(r"$\log \frac{p(X)}{1 - p(X)}$")
# plt.title(f"Logistic Regression and EFDA Log-Odds over {N_trials} trials, $n = {n}$")
plt.legend()
plt.show()

# %%
