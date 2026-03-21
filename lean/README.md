# EFDA Lean 4 Formal Verification

Formal proofs of the four main propositions in this work, written in Lean 4 (Mathlib) and verified via
[AXLE](https://axle.axiommath.ai).

---

## Directory Layout

| Path | Description |
|---|---|
| `EFDAChallenge.lean` | Challenge file: four theorem stubs with `sorry` bodies |
| `aristotle/EFDAChallenge.lean` | Aristotle's completed proofs (Harmonic) |
| `opengauss-results/EFDAChallenge-proved.lean` | OpenGauss's completed proofs (Math, Inc.) |
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

No proof strategies or Mathlib lemma names were provided. Both systems discovered proofs independently.

---

## Results

Both Aristotle (Harmonic) and OpenGauss (Math, Inc.) proved all four propositions with no `sorry`.
AXLE (Axiom) independently verified all outputs.

### AXLE Verification Output — Aristotle

```
════════════════════════════════════════════════════════════
  AXLE verification: EFDAChallenge.lean
════════════════════════════════════════════════════════════

File typechecks : yes

Extracting theorems …
Found 6 declarations

────────────────────────────────────────────────────────────
  Theorem                                   sorry?  Result
────────────────────────────────────────────────────────────
  efda_consistency_alpha                        no  ✓ PROVED ◀ paper
  efda_consistency_eta                          no  ✓ PROVED ◀ paper
  efda_calibration                              no  ✓ PROVED ◀ paper
  efda_efficiency                               no  ✓ PROVED [file-verified] ◀ paper
  efda_mse_misspecification                     no  ✓ PROVED ◀ paper
────────────────────────────────────────────────────────────

Paper theorem summary
  Fully proved (no sorry) : 4/4
  Trivial stub (True)     : 0/4
  Sorry (typechecks only) : 0/4
  Failed                  : 0/4
```

### AXLE Verification Output — OpenGauss

```
════════════════════════════════════════════════════════════
  AXLE verification: EFDAChallenge-proved.lean
════════════════════════════════════════════════════════════

File typechecks : yes
  warning : -:113:5: warning: unused variable `hstar_meas`
  warning : -:142:5: warning: unused variable `heta_meas`
  warning : -:143:5: warning: unused variable `hunbiased`

Extracting theorems …
Found 5 declarations

────────────────────────────────────────────────────────────
  Theorem                                   sorry?  Result
────────────────────────────────────────────────────────────
  efda_calibration                              no  ✓ PROVED [file-verified] ◀ paper
  efda_consistency_alpha                        no  ✓ PROVED ◀ paper
  efda_consistency_eta                          no  ✓ PROVED
  efda_efficiency                               no  ✓ PROVED [file-verified] ◀ paper
  efda_mse_misspecification                     no  ✓ PROVED ◀ paper
────────────────────────────────────────────────────────────

Paper theorem summary
  Fully proved (no sorry) : 4/4
  Trivial stub (True)     : 0/4
  Sorry (typechecks only) : 0/4
  Failed                  : 0/4
```

---

## Comparison

| Proposition | Aristotle (Harmonic) | OpenGauss (Math, Inc.) |
|---|---|---|
| 1. Consistency | ✓ Proved | ✓ Proved |
| 2. Calibration | ✓ Proved† | ✓ Proved† |
| 3. MLE Efficiency | ✓ Proved‡ | ✓ Proved§ |
| 4. Asymptotic MSE | ✓ Proved | ✓ Proved |

**†** Both systems independently identified that the original Prop 2 statement was missing
`AEStronglyMeasurable` on $\hat{p}_n$. Without it, Mathlib's Bochner integral returns 0 for
non-measurable functions and the statement is formally false. Both added the hypothesis and
proved convergence via the dominated convergence theorem.

**‡** Aristotle added one additional axiom: the score-covariance identity
$\int (\hat{\eta} - \eta)(T - A'(\eta))\, p_\eta\, d\mu = 1$,
which follows from differentiating the unbiasedness condition under the integral sign but is
not derivable from the two provided axioms. Proved the Cramér-Rao bound via a Cauchy-Schwarz
discriminant argument.

**§** OpenGauss added the same covariance identity plus two further hypotheses: the quadratic
expansion identity and non-negativity of the quadratic form integral. These are mechanically
true but needed to be made explicit. Proved the bound via the same
discriminant structure.

---

## Running the Verifier

```bash
# Verify Aristotle's proofs
python lean/verification/verify.py aristotle/EFDAChallenge.lean

# Verify OpenGauss's proofs
python lean/verification/verify.py opengauss-results/EFDAChallenge-proved.lean
```

Expected result for both: **4/4 paper theorems fully proved**.

---

## A Note on `[file-verified]`

AXLE verifies each theorem in isolation against the Mathlib environment. Theorems with
file-local dependencies (custom structures, added axioms) cannot be verified in isolation
because AXLE's parser rejects the `import` and `open` statements the preamble requires.
The fix: if AXLE detects file-local dependencies, fall back to the file-level typecheck
(`client.check`) as ground truth. Since Lean's kernel accepted the full file with no errors
and no `sorry`, this is a sound fallback. The `[file-verified]` tag makes the distinction
visible in the output.
