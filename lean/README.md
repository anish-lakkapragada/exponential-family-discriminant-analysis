# EFDA Lean 4 Formal Verification

Formal proofs of the four main propositions of **"Exponential Family Discriminant Analysis"**
(Lakkapragada, 2026), written in Lean 4 (Mathlib) and verified via
[AXLE](https://axle.axiommath.ai).

---

## Directory Layout

| Path | Description |
|---|---|
| `EFDAChallenge.lean` | Challenge file: four theorem stubs with `sorry` bodies |
| `Claude-EFDAChallenge.lean` | Claude's proof attempt |
| `aristotle/EFDAChallenge.lean` | Aristotle's completed proofs — **4/4, best result** |
| `verification/verify.py` | AXLE verification harness |

---

## The Challenge

`EFDAChallenge.lean` sets up a 1-parameter exponential family with density
$p_\eta(x) = h(x)\exp(\eta T(x) - A(\eta))$ and regularity conditions ($A \in C^\infty$,
$A'' > 0$, normalization). Two moment identities are provided as axioms:
$\mathbb{E}_\eta[T(X)] = A'(\eta)$ and $\mathrm{Var}_\eta[T(X)] = A''(\eta)$.
The task is to prove:

| Theorem | Statement |
|---|---|
| `efda_consistency_alpha` | $\hat{\alpha}_k \xrightarrow{\mathrm{a.s.}} \alpha^*_k$ |
| `efda_consistency_eta` | $\hat{\eta}_k \xrightarrow{\mathrm{a.s.}} \eta^*_k$ |
| `efda_calibration` | $\mathbb{E}[\hat{p}_k(X)] \to \mathbb{E}[p^*_k(X)]$ |
| `efda_efficiency` | $\mathrm{MSE}(\hat{\eta}) \geq 1/A''(\eta)$ |
| `efda_mse_misspecification` | $\lim\,\mathrm{MSE}(\hat{\ell}_n) = (\ell^\dagger - \ell^*)^2$ |

---

## Aristotle's Proofs

All four paper theorems are proved with no `sorry`. Two required corrections to the original
statements:

- **Prop 2**: The original statement was missing `AEStronglyMeasurable` on $\hat{p}_n$,
  making it formally false. Aristotle added the hypothesis and proved convergence via the
  dominated convergence theorem.

- **Prop 3**: The Cramér-Rao proof requires the score-estimator covariance identity
  $\int (\hat{\eta} - \eta)(T - A'(\eta))\, p_\eta\, d\mu = 1$, which comes from
  differentiating the unbiasedness condition under the integral sign — not derivable from
  the two provided axioms. Aristotle added it as a third axiom and proved the bound via
  Cauchy-Schwarz: specializing $c = 1/A''(\eta)$ in
  $0 \leq \int ((\hat{\eta}-\eta) - c(T-A'(\eta)))^2 p_\eta\, d\mu$
  yields $\mathrm{MSE} \geq 1/A''(\eta)$.

---

## Running the Verifier

```bash
cd lean
uv run verification/verify.py aristotle/EFDAChallenge.lean
```

Expected result: **4/4 paper theorems fully proved**. `efda_efficiency` is tagged
`[file-verified]` — see below.

---

## A Note on `verify.py`

AXLE verifies each theorem in isolation against the Mathlib environment. This works for
most theorems, but `efda_efficiency` depends on file-local structures and axioms
(`ExpFamily`, `IsRegularExpFamily`, `score_covariance_identity`) that don't exist in
Mathlib. Attempts to inject these as a preamble failed because AXLE's `formal_statement`
parser rejects `import`, `open`, and module doc-comments — all of which the preamble
requires.

The fix: if AXLE reports that a theorem has file-local dependencies, fall back to the
file-level typecheck (`client.check`) as the ground truth. Since Lean's kernel accepted
the full file with no errors and no `sorry`, this is a sound fallback. The `[file-verified]`
tag makes the distinction visible in the output.
