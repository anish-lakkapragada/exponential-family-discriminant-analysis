"""
experiments/calibration.py — EFDA calibration benchmark experiments.

Compares EFDA against LDA, QDA, and Logistic Regression on Expected
Calibration Error (ECE) and accuracy across distributions and conditions.

Usage:
    # Run all experiments (paper results)
    uv run python experiments/calibration.py

    # Quick smoke test
    uv run python experiments/calibration.py --trials 5

    # Only the main benchmark, Weibull and Gamma
    uv run python experiments/calibration.py --only benchmark --dists weibull gamma

    # Benchmark at a specific training size
    uv run python experiments/calibration.py --only benchmark --n-train 500

Outputs:
    results/benchmark.json       — main benchmark (accuracy + ECE)
    results/sample_size.json     — ECE vs n (Weibull)
    results/imbalance.json       — ECE vs class prior alpha (Weibull)
    results/unknown_k.json       — EFDA known-k vs estimated-k (Weibull)
    results/multiclass.json      — multi-class accuracy (K=3, K=5)
    results/negative_binomial.json — NB-EFDA vs Poisson-EFDA
    results/mixed_type.json     — multivariate EFDA (Poisson + Exp)
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efda import (
    EFDA,
    ExponentialDist,
    GammaDist,
    PoissonDist,
    WeibullDist,
    expected_calibration_error,
    generate_binary_data,
    generate_multiclass_data,
)

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

METHOD_LABELS = {"EFDA": "EFDA", "LDA": "LDA", "QDA": "QDA", "LR": "Log. Reg."}
METHOD_COLORS = {"EFDA": "darkorange", "LDA": "steelblue", "QDA": "mediumseagreen", "LR": "tomato"}
METHODS = ["EFDA", "LDA", "QDA", "LR"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _binary_trial(dist, eta0, eta1, alpha, n_train, n_test, rng):
    """Run one MC trial. Returns {method: {acc, auc, ece}}."""
    X_tr, y_tr = generate_binary_data(dist, eta0, eta1, alpha, n_train, rng)
    X_te, y_te = generate_binary_data(dist, eta0, eta1, alpha, n_test, rng)
    X_tr2, X_te2 = X_tr.reshape(-1, 1), X_te.reshape(-1, 1)

    out = {}
    for name, model, Xtr, Xte in [
        ("EFDA", EFDA(dist),                         X_tr,  X_te),
        ("LDA",  LinearDiscriminantAnalysis(),        X_tr2, X_te2),
        ("QDA",  QuadraticDiscriminantAnalysis(),     X_tr2, X_te2),
        ("LR",   LogisticRegression(max_iter=500),   X_tr2, X_te2),
    ]:
        try:
            model.fit(Xtr, y_tr)
            p = model.predict_proba(Xte)[:, 1]
            out[name] = dict(
                acc=float(accuracy_score(y_te, model.predict(Xte))),
                auc=float(roc_auc_score(y_te, p)),
                ece=float(expected_calibration_error(y_te, p)),
            )
        except Exception:
            out[name] = dict(acc=float("nan"), auc=float("nan"), ece=float("nan"))
    return out


def _aggregate(trials, methods=("EFDA", "LDA", "QDA", "LR"), metrics=("acc", "auc", "ece")):
    """Compute mean ± std across trials."""
    return {
        m: {
            k: dict(
                mean=float(np.nanmean([t[m][k] for t in trials])),
                std=float(np.nanstd([t[m][k] for t in trials])),
            )
            for k in metrics
        }
        for m in methods
    }


def _save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → saved {path}")


# Distribution configs: (dist, eta0, eta1, prior, display_name)
_DISTS = {
    "weibull":     ("Weibull",     WeibullDist(k=3),
                    WeibullDist(k=3).eta_from_scale(4.0),
                    WeibullDist(k=3).eta_from_scale(2.0), 0.7),
    "gamma":       ("Gamma",       GammaDist(a=2),
                    GammaDist(a=2).eta_from_scale(4.0),
                    GammaDist(a=2).eta_from_scale(2.0),   0.5),
    "exponential": ("Exponential", ExponentialDist(),
                    ExponentialDist().eta_from_scale(3.0),
                    ExponentialDist().eta_from_scale(1.0), 0.5),
    "poisson":     ("Poisson",     PoissonDist(),
                    PoissonDist().eta_from_rate(5.0),
                    PoissonDist().eta_from_rate(10.0),    0.5),
}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def exp_benchmark(args):
    """Exp 1: Binary benchmark — accuracy and ECE across distributions."""
    print("=== Exp 1: Binary benchmark ===")
    results = {}
    for key in args.dists:
        label, dist, eta0, eta1, alpha = _DISTS[key]
        print(f"  {label}...", end=" ", flush=True)
        rng = np.random.RandomState(args.seed)
        trials = [_binary_trial(dist, eta0, eta1, alpha, args.n_train, args.n_test, rng)
                  for _ in range(args.trials)]
        results[label] = _aggregate(trials)
        print("done")
    _save(results, "results/benchmark.json")
    if args.generate_figures: fig_ece_benchmark()


def exp_sample_size(args):
    """Exp 2: ECE vs training size n (Weibull)."""
    print("=== Exp 2: Sample-size ablation ===")
    _, dist, eta0, eta1, alpha = _DISTS["weibull"]
    sizes = [100, 250, 500, 1000, 2500, 5000, 10000]
    results = {}
    for n in sizes:
        print(f"  n={n}...", end=" ", flush=True)
        rng = np.random.RandomState(args.seed)
        trials = [_binary_trial(dist, eta0, eta1, alpha, n, 5000, rng)
                  for _ in range(args.trials)]
        results[n] = _aggregate(trials)
        print("done")
    _save(results, "results/sample_size.json")
    if args.generate_figures: fig_ece_sample_size()


def exp_imbalance(args):
    """Exp 3: ECE vs class prior alpha (Weibull)."""
    print("=== Exp 3: Class-imbalance ablation ===")
    _, dist, eta0, eta1, _ = _DISTS["weibull"]
    priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = {}
    for a in priors:
        print(f"  alpha={a}...", end=" ", flush=True)
        rng = np.random.RandomState(args.seed)
        trials = [_binary_trial(dist, eta0, eta1, a, args.n_train, args.n_test, rng)
                  for _ in range(args.trials)]
        results[str(a)] = _aggregate(trials)
        print("done")
    _save(results, "results/imbalance.json")
    if args.generate_figures: fig_ece_imbalance()


def exp_unknown_k(args):
    """Exp 4: Weibull with estimated shape k vs known k=3."""
    print("=== Exp 4: Unknown-k ablation ===")
    dist_true = WeibullDist(k=3)
    eta0 = dist_true.eta_from_scale(4.0)
    eta1 = dist_true.eta_from_scale(2.0)
    alpha = 0.7
    sizes = [100, 250, 500, 1000, 2500, 5000]
    results = {}

    for n in sizes:
        print(f"  n={n}...", end=" ", flush=True)
        rows = {m: dict(acc=[], auc=[]) for m in ("EFDA_known_k", "EFDA_est_k", "LR", "LDA")}
        rng = np.random.RandomState(args.seed)

        for _ in range(args.trials):
            X_tr, y_tr = generate_binary_data(dist_true, eta0, eta1, alpha, n, rng)
            X_te, y_te = generate_binary_data(dist_true, eta0, eta1, alpha, 3000, rng)

            # EFDA with known k=3
            m = EFDA(WeibullDist(k=3)).fit(X_tr, y_tr)
            p = m.predict_proba(X_te)[:, 1]
            rows["EFDA_known_k"]["acc"].append(float(accuracy_score(y_te, m.predict(X_te))))
            rows["EFDA_known_k"]["auc"].append(float(roc_auc_score(y_te, p)))

            # EFDA with k estimated from data
            try:
                k_est = float(np.clip(stats.weibull_min.fit(X_tr, floc=0)[0], 0.5, 20.0))
                m = EFDA(WeibullDist(k=k_est)).fit(X_tr, y_tr)
                p = m.predict_proba(X_te)[:, 1]
                rows["EFDA_est_k"]["acc"].append(float(accuracy_score(y_te, m.predict(X_te))))
                rows["EFDA_est_k"]["auc"].append(float(roc_auc_score(y_te, p)))
            except Exception:
                rows["EFDA_est_k"]["acc"].append(float("nan"))
                rows["EFDA_est_k"]["auc"].append(float("nan"))

            # LR and LDA baselines
            for name, model in [("LR", LogisticRegression(max_iter=500)),
                                 ("LDA", LinearDiscriminantAnalysis())]:
                try:
                    Xtr2, Xte2 = X_tr.reshape(-1, 1), X_te.reshape(-1, 1)
                    model.fit(Xtr2, y_tr)
                    p = model.predict_proba(Xte2)[:, 1]
                    rows[name]["acc"].append(float(accuracy_score(y_te, model.predict(Xte2))))
                    rows[name]["auc"].append(float(roc_auc_score(y_te, p)))
                except Exception:
                    rows[name]["acc"].append(float("nan"))
                    rows[name]["auc"].append(float("nan"))

        def s(arr): return dict(mean=float(np.nanmean(arr)), std=float(np.nanstd(arr)))
        results[n] = {m: {k: s(v) for k, v in rows[m].items()} for m in rows}
        print("done")

    _save(results, "results/unknown_k.json")
    if args.generate_figures: fig_unknown_k()


def exp_multiclass(args):
    """Exp 5: Multi-class classification across four distributions (K=3 and K=5)."""
    print("=== Exp 5: Multi-class ===")

    _MC_DISTS = {
        "Weibull":     (WeibullDist(k=3),  "scale",
                        {3: [1.0, 2.0, 3.5],           5: [1.0, 1.8, 2.6, 3.4, 4.5]}),
        "Gamma":       (GammaDist(a=2),    "scale",
                        {3: [1.0, 2.5, 5.0],            5: [1.0, 1.8, 2.6, 3.4, 4.5]}),
        "Exponential": (ExponentialDist(), "scale",
                        {3: [0.5, 1.5, 3.5],            5: [0.5, 1.0, 1.8, 2.8, 4.0]}),
        "Poisson":     (PoissonDist(),     "rate",
                        {3: [3.0, 8.0, 15.0],           5: [2.0, 5.0, 9.0, 14.0, 20.0]}),
    }

    results = {}
    for dist_name, (dist, param_type, params_by_K) in _MC_DISTS.items():
        results[dist_name] = {}
        for K, param_vals in params_by_K.items():
            print(f"  {dist_name} K={K}...", end=" ", flush=True)
            etas = ([dist.eta_from_scale(v) for v in param_vals] if param_type == "scale"
                    else [dist.eta_from_rate(v) for v in param_vals])
            alphas_mc = [1.0 / K] * K
            rng = np.random.RandomState(args.seed)
            rows = {m: dict(acc=[], ece=[]) for m in ("EFDA", "LDA", "QDA", "LR")}

            for _ in range(args.trials):
                X_tr, y_tr = generate_multiclass_data(dist, etas, alphas_mc, 2000, rng)
                X_te, y_te = generate_multiclass_data(dist, etas, alphas_mc, 5000, rng)
                for name, model, Xtr, Xte in [
                    ("EFDA", EFDA(dist),
                     X_tr, X_te),
                    ("LDA",  LinearDiscriminantAnalysis(),
                     X_tr.reshape(-1, 1), X_te.reshape(-1, 1)),
                    ("QDA",  QuadraticDiscriminantAnalysis(),
                     X_tr.reshape(-1, 1), X_te.reshape(-1, 1)),
                    ("LR",   LogisticRegression(multi_class="multinomial", max_iter=500),
                     X_tr.reshape(-1, 1), X_te.reshape(-1, 1)),
                ]:
                    try:
                        model.fit(Xtr, y_tr)
                        preds = model.predict(Xte)
                        conf  = model.predict_proba(Xte).max(axis=1)
                        rows[name]["acc"].append(float(accuracy_score(y_te, preds)))
                        rows[name]["ece"].append(float(
                            expected_calibration_error((preds == y_te).astype(float), conf)))
                    except Exception:
                        rows[name]["acc"].append(float("nan"))
                        rows[name]["ece"].append(float("nan"))

            def s(arr): return dict(mean=float(np.nanmean(arr)), std=float(np.nanstd(arr)))
            results[dist_name][f"K{K}"] = {
                m: {metric: s(rows[m][metric]) for metric in ("acc", "ece")}
                for m in ("EFDA", "LDA", "QDA", "LR")
            }
            print("done")

    _save(results, "results/multiclass.json")



# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _fig_save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def fig_ece_benchmark(result_file="results/benchmark.json"):
    with open(result_file) as f:
        bench = json.load(f)
    dists = ["Weibull", "Gamma", "Exponential", "Poisson"]
    x_pos = np.arange(len(dists))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, m in enumerate(METHODS):
        means = [bench[d][m]["ece"]["mean"] for d in dists]
        stds  = [bench[d][m]["ece"]["std"]  for d in dists]
        ax.bar(x_pos + (i - 1.5) * width, means, width,
               yerr=stds, capsize=3, label=METHOD_LABELS[m],
               color=METHOD_COLORS[m], alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dists)
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, "ece-benchmark.png")


def fig_ece_sample_size(result_file="results/sample_size.json"):
    with open(result_file) as f:
        ss = json.load(f)
    ns = [int(k) for k in ss.keys()]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for m in METHODS:
        means = np.array([ss[str(n)][m]["ece"]["mean"] for n in ns])
        stds  = np.array([ss[str(n)][m]["ece"]["std"]  for n in ns])
        ax.plot(ns, means, marker="o", markersize=4, label=METHOD_LABELS[m], color=METHOD_COLORS[m])
        ax.fill_between(ns, means - stds, means + stds, alpha=0.15, color=METHOD_COLORS[m])
    ax.set_xscale("log")
    ax.set_xlabel("Training set size $n$")
    ax.set_ylabel("ECE")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, "ece-sample-size.png")


def fig_ece_imbalance(result_file="results/imbalance.json"):
    with open(result_file) as f:
        imb = json.load(f)
    alphas = [float(k) for k in imb.keys()]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for m in METHODS:
        means = np.array([imb[str(a)][m]["ece"]["mean"] for a in alphas])
        stds  = np.array([imb[str(a)][m]["ece"]["std"]  for a in alphas])
        ax.plot(alphas, means, marker="o", markersize=4, label=METHOD_LABELS[m], color=METHOD_COLORS[m])
        ax.fill_between(alphas, means - stds, means + stds, alpha=0.15, color=METHOD_COLORS[m])
    ax.set_xlabel(r"Class-prior $\alpha = \mathbb{P}[Y=1]$")
    ax.set_ylabel("ECE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, "ece-imbalance.png")


def fig_unknown_k(result_file="results/unknown_k.json"):
    with open(result_file) as f:
        uk = json.load(f)
    ns = [int(k) for k in uk.keys()]
    uk_methods = ["EFDA_known_k", "EFDA_est_k", "LR", "LDA"]
    uk_labels  = {
        "EFDA_known_k": "EFDA (known $k$)",
        "EFDA_est_k":   "EFDA (estimated $k$)",
        "LR":           "Log. Reg.",
        "LDA":          "LDA",
    }
    uk_colors = {"EFDA_known_k": "darkorange", "EFDA_est_k": "gold",
                 "LR": "tomato", "LDA": "steelblue"}
    uk_ls     = {"EFDA_known_k": "-", "EFDA_est_k": "--", "LR": "-.", "LDA": ":"}
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for m in uk_methods:
        means = np.array([uk[str(n)][m]["acc"]["mean"] for n in ns])
        stds  = np.array([uk[str(n)][m]["acc"]["std"]  for n in ns])
        ax.plot(ns, means, marker="o", markersize=4, label=uk_labels[m],
                color=uk_colors[m], linestyle=uk_ls[m])
        ax.fill_between(ns, means - stds, means + stds, alpha=0.12, color=uk_colors[m])
    ax.set_xscale("log")
    ax.set_xlabel("Training set size $n$")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _fig_save(fig, "unknown-k-ablation.png")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="EFDA calibration benchmark experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  uv run python experiments/calibration.py                            # all experiments
  uv run python experiments/calibration.py --trials 5                 # quick smoke test
  uv run python experiments/calibration.py --only benchmark nb        # selected experiments
  uv run python experiments/calibration.py --dists weibull gamma      # benchmark subset
  uv run python experiments/calibration.py --n-train 500 --n-test 1000
""",
    )
    p.add_argument("--trials",   type=int, default=100,
                   help="MC trials per condition (default: 100)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--n-train",  type=int, default=1000, dest="n_train",
                   help="Training set size for the benchmark (default: 1000)")
    p.add_argument("--n-test",   type=int, default=2000, dest="n_test",
                   help="Test set size for the benchmark (default: 2000)")
    p.add_argument("--dists",    nargs="+",
                   choices=["weibull", "gamma", "exponential", "poisson"],
                   default=["weibull", "gamma", "exponential", "poisson"],
                   help="Distributions for the main benchmark (default: all four)")
    p.add_argument("--only",     nargs="+",
                   choices=["benchmark", "sample-size", "imbalance",
                            "unknown-k", "multiclass"],
                   help="Run only these experiments (default: all)")
    p.add_argument("--generate-figures", action="store_true", dest="generate_figures",
                   help="Generate figures immediately after each experiment completes")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run = set(args.only) if args.only else {
        "benchmark", "sample-size", "imbalance", "unknown-k", "multiclass"
    }

    if "benchmark"   in run: exp_benchmark(args)
    if "sample-size" in run: exp_sample_size(args)
    if "imbalance"   in run: exp_imbalance(args)
    if "unknown-k"   in run: exp_unknown_k(args)
    if "multiclass"  in run: exp_multiclass(args)

    print("\nDone. Results saved to results/")
