"""
experiments/efficiency.py — EFDA statistical efficiency experiment.

Measures the variance of the log-odds estimator averaged across a grid of
evaluation points x_0 as a function of training size n.  Compares EFDA against
the Cramér-Rao lower bound, logistic regression, LDA, and QDA to show EFDA's
statistical efficiency advantage.

Usage:
    # Default settings (Weibull k=3, lambda0=4, lambda1=2)
    uv run python experiments/efficiency.py

    # Custom number of x0 evaluation points or number of trials
    uv run python experiments/efficiency.py --n-x0 100 --trials 2000

    # Different Weibull shape and scale parameters
    uv run python experiments/efficiency.py --k 2 --lambda0 5.0 --lambda1 2.0

    # Custom n sweep
    uv run python experiments/efficiency.py --ns 50 100 500 2000 10000

Outputs:
    results/efficiency.json
    Keys per n: EFDA/LR/LDA/QDA.{var,bias_sq,mse}, CR_bound (avg over x0 grid),
                true_lo (list, one per x0 point), x0_points
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efda import EFDA, WeibullDist, generate_binary_data


# ---------------------------------------------------------------------------
# Cramér-Rao utilities (Weibull-specific)
# ---------------------------------------------------------------------------

def _fisher_information_weibull(eta, k):
    """Fisher information I(eta) = A''(eta) = 1/eta^2 for Weibull with known k."""
    return 1.0 / (eta ** 2)


def cramer_rao_variance_log_odds(eta0, eta1, N0, N1, x0, k):
    """Cramér-Rao lower bound for the variance of the log-odds estimate at x0.

    Under EFDA with Weibull(k) class-conditionals the log-odds is
        ell(x0) = C + (eta1 - eta0)*x0^k + (A(eta0) - A(eta1))

    With g1(eta1) = -A(eta1) + eta1*T(x0) and g0(eta0) = A(eta0) - eta0*T(x0):
        g1'(eta1) = T(x0) - A'(eta1)  =  x0^k + 1/eta1
        g0'(eta0) = A'(eta0) - T(x0)  =  -1/eta0 - x0^k  ... squared same
    """
    T0 = x0 ** k
    d_ell_d_eta1 = T0 - (-1.0 / eta1)
    d_ell_d_eta0 = -(T0 - (-1.0 / eta0))
    I1 = _fisher_information_weibull(eta1, k)
    I0 = _fisher_information_weibull(eta0, k)
    return (d_ell_d_eta1 ** 2) / (N1 * I1) + (d_ell_d_eta0 ** 2) / (N0 * I0)

os.makedirs("results", exist_ok=True)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

K_SHAPE  = 3    # Weibull shape parameter (not number of classes)
LAMBDA_0 = 4.0
LAMBDA_1 = 2.0
ALPHA    = 0.7

METHOD_COLORS = {
    "EFDA": "darkorange",
    "LR":   "steelblue",
    "LDA":  "seagreen",
    "QDA":  "tomato",
}
METHOD_LABELS = {
    "EFDA": "EFDA",
    "LR":   "Logistic Regression",
    "LDA":  "LDA",
    "QDA":  "QDA",
}


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _fig_save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def fig_weibull_data():
    from scipy.stats import weibull_min
    x = np.linspace(0.01, 6, 1000)
    f1 = ALPHA       * weibull_min.pdf(x, c=K_SHAPE, scale=LAMBDA_1)
    f0 = (1 - ALPHA) * weibull_min.pdf(x, c=K_SHAPE, scale=LAMBDA_0)
    p_x = f1 / (f1 + f0)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(x, f1,  label=r"$\mathbb{P}[X, Y=1]$", color="steelblue")
    ax.plot(x, f0,  label=r"$\mathbb{P}[X, Y=0]$", color="coral")
    ax.plot(x, p_x, label=r"$\mathbb{P}[Y=1 \mid X]$", color="green", linestyle="--")
    ax.set_xlabel("$X$")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, "weibull-data.png")


def fig_weibull_log_odds(n_trials=100, n=10_000, seed=0):
    from sklearn.linear_model import LogisticRegression as LR
    dist = WeibullDist(k=K_SHAPE)
    eta0 = dist.eta_from_scale(LAMBDA_0)
    eta1 = dist.eta_from_scale(LAMBDA_1)
    xs = np.linspace(0.01, 6, 1000)
    true_lo = (
        np.log(ALPHA / (1 - ALPHA))
        + K_SHAPE * np.log(LAMBDA_0 / LAMBDA_1)
        + xs ** K_SHAPE * (1 / LAMBDA_0 ** K_SHAPE - 1 / LAMBDA_1 ** K_SHAPE)
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    rng = np.random.RandomState(seed)
    for trial in range(n_trials):
        X, y = generate_binary_data(dist, eta0, eta1, ALPHA, n, rng)
        lr   = LR(fit_intercept=True, max_iter=500).fit(X.reshape(-1, 1), y)
        lr_lo = lr.coef_[0, 0] * xs + lr.intercept_[0]
        efda  = EFDA(dist).fit(X, y)
        lj    = efda._log_joint(xs)
        efda_lo = lj[:, 1] - lj[:, 0]
        ax.plot(xs, lr_lo,   color="steelblue",  linewidth=0.3, alpha=0.6,
                label="Logistic Regression" if trial == 0 else None)
        ax.plot(xs, efda_lo, color="darkorange", linewidth=0.3, alpha=0.6,
                label="EFDA" if trial == 0 else None)
    ax.plot(xs, true_lo, color="green", linewidth=2.0, linestyle="--", label="True log-odds")
    ax.set_xlabel("$X$")
    ax.set_ylabel(r"$\log\frac{P(Y=1\mid X)}{P(Y=0\mid X)}$")
    ax.set_ylim(-6, 6)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, "weibull-log-reg-vs-efda.png")


def fig_asymptotic_efficiency(result_file="results/efficiency.json"):
    with open(result_file) as f:
        feff = json.load(f)
    ns = [int(k) for k in feff.keys() if k != "x0_points"]
    methods = ["EFDA", "LR", "LDA", "QDA"]
    vars_by_method = {m: [feff[str(n)][m]["var"] for n in ns] for m in methods}
    mses_by_method = {m: [feff[str(n)][m]["mse"] for n in ns] for m in methods}
    cr_bounds = [feff[str(n)]["CR_bound"] for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    xticks = [n for n in ns if n in {100, 1000, 10000, 100000}]

    # Left: variance (log x, linear y) — shows EFDA converging toward CR bound
    ax = axes[0]
    for m in methods:
        ax.plot(ns, vars_by_method[m], marker="o", markersize=3.5,
                color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.plot(ns, cr_bounds, linestyle="--", color="black", linewidth=1.5,
            label="Cramér–Rao bound")
    ax.set_xscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"$10^{int(np.log10(t))}$" for t in xticks])
    ax.set_xlabel("Training set size $n$")
    ax.set_ylabel(r"Mean $\mathrm{Var}(\hat\ell(x_0))$")
    ax.set_title("Variance")
    ax.legend(fontsize=9)
    ax.grid(True, which="major", alpha=0.3)

    # Right: MSE (log-log) — misspecified methods plateau; EFDA continues to 0
    ax = axes[1]
    for m in methods:
        ax.plot(ns, mses_by_method[m], marker="o", markersize=3.5,
                color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.plot(ns, cr_bounds, linestyle="--", color="black", linewidth=1.5,
            label="Cramér–Rao bound")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"$10^{int(np.log10(t))}$" for t in xticks])
    ax.set_xlabel("Training set size $n$")
    ax.set_ylabel(r"Mean $\mathrm{MSE}(\hat\ell(x_0))$")
    ax.set_title("MSE (log–log)")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _fig_save(fig, "asymptotic-efficiency.png")


# ---------------------------------------------------------------------------
# x0 grid sampling
# ---------------------------------------------------------------------------

def sample_x0_grid(k, lambda0, lambda1, n_per_dist=50, seed=0):
    """Sample x0 evaluation points from both class-conditional distributions.

    Draws n_per_dist samples from Weibull(shape=k, scale=lambda0) and
    n_per_dist from Weibull(shape=k, scale=lambda1), returning the sorted
    union.  Using actual draws from both distributions ensures evaluation
    points are concentrated where the data lives, not at an arbitrary grid.
    """
    from scipy.stats import weibull_min
    rng = np.random.RandomState(seed)
    pts0 = weibull_min.rvs(c=k, scale=lambda0, size=n_per_dist, random_state=rng)
    pts1 = weibull_min.rvs(c=k, scale=lambda1, size=n_per_dist, random_state=rng)
    return np.sort(np.concatenate([pts0, pts1]))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run(args):
    dist = WeibullDist(k=args.k)
    eta0 = dist.eta_from_scale(args.lambda0)
    eta1 = dist.eta_from_scale(args.lambda1)

    # Build x0 grid: sample from both class distributions (reproducible seed)
    x0_pts = sample_x0_grid(args.k, args.lambda0, args.lambda1,
                             n_per_dist=args.n_x0, seed=args.seed)

    # True log-odds at each x0
    true_los = (
        np.log(args.alpha / (1 - args.alpha))
        + args.k * np.log(args.lambda0 / args.lambda1)
        + x0_pts ** args.k * (1 / args.lambda0 ** args.k - 1 / args.lambda1 ** args.k)
    )

    print(f"Weibull(shape k'={args.k})  lambda0={args.lambda0}  lambda1={args.lambda1}")
    print(f"alpha={args.alpha}  x0 grid: {len(x0_pts)} points sampled from both distributions")
    print(f"  x0 range: [{x0_pts.min():.2f}, {x0_pts.max():.2f}]")
    print(f"Running {args.trials} trials per n...")

    results = {"x0_points": list(x0_pts)}
    methods = ["EFDA", "LR", "LDA", "QDA"]

    for n in args.ns:
        # Fix class counts (stratified design): aligns with the CR bound derivation,
        # which conditions on N_0 and N_1 as known constants.
        N1 = int(n * args.alpha)
        N0 = n - N1

        print(f"  n={n} (N0={N0}, N1={N1})...", end=" ", flush=True)
        # lo_samples[method][trial] = array of shape (len(x0_pts),)
        lo_samples = {m: [] for m in methods}
        rng = np.random.RandomState(args.seed)

        for _ in range(args.trials):
            # Stratified sampling: draw exactly N0 class-0 and N1 class-1 samples
            X1 = dist.sample(eta1, N1, rng=rng.randint(0, 2**31))
            X0 = dist.sample(eta0, N0, rng=rng.randint(0, 2**31))
            X_tr = np.concatenate([X0, X1])
            y_tr = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])
            idx = rng.permutation(n)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
            X_tr_2d = X_tr.reshape(-1, 1)

            # EFDA
            try:
                m = EFDA(dist).fit(X_tr, y_tr)
                lj = m._log_joint(x0_pts)
                lo_samples["EFDA"].append(lj[:, 1] - lj[:, 0])
            except Exception:
                lo_samples["EFDA"].append(np.full(len(x0_pts), np.nan))

            # Logistic Regression
            try:
                m = LogisticRegression(max_iter=500, fit_intercept=True).fit(X_tr_2d, y_tr)
                lo_samples["LR"].append(m.coef_[0, 0] * x0_pts + m.intercept_[0])
            except Exception:
                lo_samples["LR"].append(np.full(len(x0_pts), np.nan))

            # LDA
            try:
                m = LinearDiscriminantAnalysis().fit(X_tr_2d, y_tr)
                pp = m.predict_proba(x0_pts.reshape(-1, 1))
                # log-odds = log(P(Y=1|x) / P(Y=0|x))
                lo_samples["LDA"].append(np.log(pp[:, 1] / pp[:, 0]))
            except Exception:
                lo_samples["LDA"].append(np.full(len(x0_pts), np.nan))

            # QDA
            try:
                m = QuadraticDiscriminantAnalysis().fit(X_tr_2d, y_tr)
                pp = m.predict_proba(x0_pts.reshape(-1, 1))
                lo_samples["QDA"].append(np.log(pp[:, 1] / pp[:, 0]))
            except Exception:
                lo_samples["QDA"].append(np.full(len(x0_pts), np.nan))

        # Compute stats: for each method, average variance across x0 grid
        cr_per_x0 = np.array([
            cramer_rao_variance_log_odds(eta0, eta1, N0, N1, x0, args.k)
            for x0 in x0_pts
        ])
        mean_cr = float(np.mean(cr_per_x0))

        def stats_for(method):
            arr = np.array(lo_samples[method])   # (trials, len(x0_pts))
            # variance per x0, then average
            var_per_x0  = np.nanvar(arr, axis=0)
            mean_per_x0 = np.nanmean(arr, axis=0)
            bias_per_x0 = (mean_per_x0 - true_los) ** 2
            mse_per_x0  = np.nanmean((arr - true_los) ** 2, axis=0)
            return dict(
                var=float(np.mean(var_per_x0)),
                bias_sq=float(np.mean(bias_per_x0)),
                mse=float(np.mean(mse_per_x0)),
            )

        results[n] = {m: stats_for(m) for m in methods}
        results[n]["CR_bound"] = mean_cr
        results[n]["true_lo"]  = list(map(float, true_los))

        ratio_lr = results[n]["LR"]["var"] / max(results[n]["EFDA"]["var"], 1e-15)
        ratio_cr = results[n]["EFDA"]["var"] / max(mean_cr, 1e-15)
        print(f"done  (LR/EFDA={ratio_lr:.2f}x, EFDA/CR={ratio_cr:.2f}x)")

    with open("results/efficiency.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  → saved results/efficiency.json")

    if args.generate_figures:
        fig_weibull_data()
        fig_weibull_log_odds()
        fig_asymptotic_efficiency()


def _parse_args():
    p = argparse.ArgumentParser(
        description="EFDA statistical efficiency experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  uv run python experiments/efficiency.py
  uv run python experiments/efficiency.py --n-x0 100 --trials 2000
  uv run python experiments/efficiency.py --k 2 --lambda0 5.0 --lambda1 2.0
  uv run python experiments/efficiency.py --ns 50 100 500 2000 10000 --trials 500
""",
    )
    p.add_argument("--k",        type=float, default=3.0,
                   help="Weibull shape parameter (default: 3.0)")
    p.add_argument("--lambda0",  type=float, default=4.0,
                   help="Class-0 scale parameter lambda_0 (default: 4.0)")
    p.add_argument("--lambda1",  type=float, default=2.0,
                   help="Class-1 scale parameter lambda_1 (default: 2.0)")
    p.add_argument("--alpha",    type=float, default=0.7,
                   help="Class-1 prior P(Y=1) (default: 0.7)")
    p.add_argument("--n-x0", type=int, default=50, dest="n_x0",
                   help="Number of x_0 evaluation points sampled per class distribution "
                        "(default: 50; total grid size = 2 * n_x0)")
    p.add_argument("--ns",       type=int, nargs="+",
                   default=[100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000],
                   help="Training sizes to sweep (default: 100 ... 100000)")
    p.add_argument("--trials",   type=int, default=1000,
                   help="MC trials per n (default: 1000)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--generate-figures", action="store_true", dest="generate_figures",
                   help="Generate figures immediately after the experiment completes")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
