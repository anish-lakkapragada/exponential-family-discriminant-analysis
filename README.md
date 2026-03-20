# Code for Exponential Family Discriminant Analysis (EFDA)

This is our code for our paper on exponential family discriminant analysis. It contains all experiment scripts used to get results and generate figures, along with code to run AI-driven Lean 4 Verification of our stated proofs.


## Install

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

Requires Python ≥ 3.9. Dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

## Repository Layout

```
efda.py                      # Core library: EFDA class + distribution classes

experiments/
  calibration.py             # Calibration experiments (ECE benchmark, sample size,
                             #   class imbalance, unknown-k ablation, multi-class)
  efficiency.py              # Statistical efficiency experiment (Var/MSE vs. n,
                             #   Cramér-Rao bound comparison)

results/                     # JSON outputs written by the experiment scripts
lean/                        # Lean 4 formal verification (see lean/README.md)
  EFDAChallenge.lean         # Theorem stubs (challenge file)
  aristotle/                 # Best proofs: all 4 paper theorems, no sorry
  verification/verify.py     # AXLE verification harness
```

## Running Experiments

All scripts run from the repo root.

### Calibration

```bash
uv run python experiments/calibration.py
```

Runs all five calibration experiments and saves results to `results/`.

```bash
# Run a subset of experiments 
uv run python experiments/calibration.py --only benchmark sample-size

# Generate figures into paper/figures/ after running
uv run python experiments/calibration.py --generate-figures
```

Full option list: `uv run python experiments/calibration.py --help`

### Statistical Efficiency

```bash
uv run python experiments/efficiency.py 
```

Similarly runs all efficiency experiments. Saves `results/efficiency.json`. With `--generate-figures`, also writes three figures to
`paper/figures/`: `asymptotic-efficiency.png`, `weibull-data.png`, `weibull-log-reg-vs-efda.png`.

```bash
# Different Weibull parameters or number of trials
uv run python experiments/efficiency.py --k 2 --lambda0 5.0 --lambda1 2.0 --trials 2000
```

Full option list: `uv run python experiments/efficiency.py --help`

## Formal Verification (Lean 4)

The four main propositions of the paper have been formally proved in Lean 4 with Mathlib through provers [Aristotle](http://aristotle.harmonic.fun) and [OpenGauss](https://www.math.inc/opengauss). Both proofs, once written in lean, have been verified through [Axiom Lean Engine (AXLE)](https://axle.axiommath.ai). All four theorems are proved with
no `sorry`. See [`lean/README.md`](lean/README.md) for details and how to run the AXLE verifier.

## Using EFDA

Please see the below example.

```python
from efda import EFDA, WeibullDist, GammaDist, PoissonDist

dist = WeibullDist(k=3)
model = EFDA(dist).fit(X_train, y_train)

model.predict(X_test)         # class labels
model.predict_proba(X_test)   # calibrated posterior probabilities
```

Supported distributions: `WeibullDist`, `GammaDist`, `ExponentialDist`,
`PoissonDist`, `NegativeBinomialDist`.


## Contact

Please reach out to [anish.lakkapragada@yale.edu](mailto:anish.lakkapragada@yale.edu) for any questions or concerns.
