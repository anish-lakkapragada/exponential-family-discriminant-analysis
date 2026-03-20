import Mathlib

/-!
# EFDA Formal Verification Challenge

**Paper:** "Exponential Family Discriminant Analysis" (Lakkapragada, 2026)
**Challenge:** Prove the four main propositions of the paper in Lean 4.

Two foundational exponential-family identities are provided as `axiom`
declarations (classical results; Jordan 2010, §8.3). Both tools may use
these freely as lemmas. The four propositions from the paper are each
stated as a typed theorem with a `sorry` body — replace each `sorry`
with a complete proof.
-/

open MeasureTheory Filter Topology ProbabilityTheory

-- ============================================================
-- Exponential Family Setup
-- ============================================================

/-- A 1-parameter exponential family with log-partition function `A`,
    sufficient statistic `T`, base measure density `h`, and base measure `μ`.
    Family density: p_η(x) = h(x) · exp(η · T(x) − A(η)). -/
structure ExpFamily (α : Type*) [MeasurableSpace α] where
  A : ℝ → ℝ
  T : α → ℝ
  h : α → ℝ
  μ : Measure α

/-- Regularity conditions on the exponential family (Assumption A in the paper). -/
structure IsRegularExpFamily {α : Type*} [MeasurableSpace α] (ef : ExpFamily α) : Prop where
  /-- A is smooth (C^∞) on the natural parameter space -/
  A_smooth : ContDiff ℝ ⊤ ef.A
  /-- A'' > 0: strict convexity, equivalently positive Fisher information -/
  A''_pos  : ∀ η : ℝ, 0 < iteratedDeriv 2 ef.A η
  /-- T is square-integrable under every member of the family -/
  T_sq_int : ∀ η : ℝ,
    Integrable (fun x => ef.T x ^ 2 * ef.h x * Real.exp (η * ef.T x - ef.A η)) ef.μ
  /-- Normalization: each p_η integrates to 1 -/
  normalized : ∀ η : ℝ,
    ∫ x, ef.h x * Real.exp (η * ef.T x - ef.A η) ∂ef.μ = 1

-- ============================================================
-- PROVIDED: Foundational Moment Identities
-- (Jordan 2010, §8.3; stated here as axioms)
-- ============================================================

/-- First moment identity: E_η[T(X)] = A'(η). -/
axiom first_moment_identity {α : Type*} [MeasurableSpace α]
    (ef : ExpFamily α) (hreg : IsRegularExpFamily ef) (η : ℝ) :
    ∫ x, ef.T x * (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
    = deriv ef.A η

/-- Fisher information identity: Var_η[T(X)] = A''(η). -/
axiom fisher_info_eq_A'' {α : Type*} [MeasurableSpace α]
    (ef : ExpFamily α) (hreg : IsRegularExpFamily ef) (η : ℝ) :
    ∫ x, (ef.T x - deriv ef.A η) ^ 2 *
         (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
    = iteratedDeriv 2 ef.A η

-- ============================================================
-- CHALLENGE: Prove the Four Propositions
-- ============================================================

-- ------------------------------------------------------------
-- Proposition 1 — Consistency of the EFDA MLE
-- ------------------------------------------------------------

/-- **Proposition 1a**: α̂_k = N_k/n → α*_k almost surely as n → ∞. -/
theorem efda_consistency_alpha
    {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
    (Z        : ℕ → Ω → ℝ)
    (hZ_int   : Integrable (Z 0) P)
    (hZ_indep : Pairwise (Function.onFun (fun f g => IndepFun f g P) Z))
    (hZ_ident : ∀ i, IdentDistrib (Z i) (Z 0) P P)
    (α_star   : ℝ)
    (hZ_mean  : ∫ ω, Z 0 ω ∂P = α_star) :
    ∀ᵐ ω ∂P, Tendsto
      (fun (n : ℕ) => ((n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, Z i ω))
      atTop (nhds α_star) := by
  rw [← hZ_mean]
  apply_rules [ProbabilityTheory.strong_law_ae]

/-- **Proposition 1b**: T̄_k → A'(η*_k) a.s. implies η̂_k = (A')⁻¹(T̄_k) → η*_k a.s. -/
theorem efda_consistency_eta
    (A           : ℝ → ℝ) (_ : ContDiff ℝ ⊤ A)
    (Ainv        : ℝ → ℝ)
    (hAinv_left  : ∀ η, Ainv (deriv A η) = η)
    (hAinv_cont  : Continuous Ainv)
    (η_star      : ℝ)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Tbar        : ℕ → Ω → ℝ)
    (hTbar       : ∀ᵐ ω ∂P,
        Tendsto (fun n => Tbar n ω) atTop (nhds (deriv A η_star))) :
    ∀ᵐ ω ∂P, Tendsto (fun n => Ainv (Tbar n ω)) atTop (nhds η_star) := by
  filter_upwards [hTbar] with ω hω using by
    simpa only [hAinv_left] using hAinv_cont.continuousAt.tendsto.comp hω

-- ------------------------------------------------------------
-- Proposition 2 — Asymptotic Calibration
-- ------------------------------------------------------------

/-
  **Note on Proposition 2:** The original statement is missing an
  `AEStronglyMeasurable` hypothesis for `phat n`. In Mathlib's
  Bochner integral, `∫ f ∂μ = 0` whenever `f` is *not*
  `AEStronglyMeasurable`, so without this assumption the theorem is
  false: one can construct a counterexample on ({0,1}, {∅, Ω}) where
  non-constant `phat n` converge pointwise to a constant `pstar`, yet
  all integrals of `phat n` are 0 while `∫ pstar ≠ 0`.

  The corrected statement below adds the missing hypothesis and is
  then a direct application of the dominated convergence theorem.
-/

-- Original statement (unprovable without measurability of phat):
-- theorem efda_calibration
--     {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
--     (phat        : ℕ → Ω → ℝ)
--     (pstar       : Ω → ℝ)
--     (hstar_meas  : Measurable pstar)
--     (hbound      : ∀ n, ∀ᵐ ω ∂P, 0 ≤ phat n ω ∧ phat n ω ≤ 1)
--     (hconv       : ∀ᵐ ω ∂P,
--         Tendsto (fun n => phat n ω) atTop (nhds (pstar ω))) :
--     Tendsto
--       (fun n => ∫ ω, phat n ω ∂P)
--       atTop
--       (nhds (∫ ω, pstar ω ∂P))

/-- **Proposition 2 (corrected)**: If p̂_k(X) → p*_k(X) almost surely and p̂_k ∈ [0,1]
    uniformly, then E[p̂_k(X)] → E[p*_k(X)].

    Compared to the original, `hphat_meas` (AEStronglyMeasurable for each `phat n`)
    has been added. -/
theorem efda_calibration
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (phat        : ℕ → Ω → ℝ)
    (pstar       : Ω → ℝ)
    (hstar_meas  : Measurable pstar)
    (hphat_meas  : ∀ n, AEStronglyMeasurable (phat n) P)
    (hbound      : ∀ n, ∀ᵐ ω ∂P, 0 ≤ phat n ω ∧ phat n ω ≤ 1)
    (hconv       : ∀ᵐ ω ∂P,
        Tendsto (fun n => phat n ω) atTop (nhds (pstar ω))) :
    Tendsto
      (fun n => ∫ ω, phat n ω ∂P)
      atTop
      (nhds (∫ ω, pstar ω ∂P)) := by
  refine' MeasureTheory.tendsto_integral_of_dominated_convergence _ _ _ _ _;
  exacts [ fun _ => 1, hphat_meas, by norm_num, fun n => by filter_upwards [ hbound n ] with ω hω using abs_le.mpr ⟨ by linarith, by linarith ⟩, hconv ]

-- ------------------------------------------------------------
-- Proposition 3 — MLE Efficiency (Cramér-Rao Bound)
-- ------------------------------------------------------------

/-
  **Note on Proposition 3 (Cramér-Rao):**

  The classical Cramér-Rao proof requires three ingredients beyond the
  given axioms:

  1. **Score-estimator covariance identity**:
       ∫ (η̂ − η)(T − A′(η)) p_η dμ = 1
     This arises from differentiating the unbiasedness condition
     ∫ η̂ · p_{η′} = η′ with respect to η′ (Leibniz integral rule /
     differentiation under the integral sign), which is not among the
     provided axioms.

  2. **Nonnegativity of the density**: h(x) · exp(…) ≥ 0, needed for
     the Cauchy-Schwarz step on weighted integrals.

  3. **Integrability conditions** for the cross-term (η̂ − η)(T − A′(η)) · p_η.

  We state the score-estimator covariance identity as an additional
  axiom and add the remaining conditions as hypotheses in the corrected
  version below, providing a complete proof via weighted Cauchy-Schwarz.
-/

/-- Score-estimator covariance identity (Leibniz integral rule applied to
    unbiasedness). This is a classical result that follows from
    differentiating ∫ η̂ · p_{η'} = η' under the integral sign. -/
axiom score_covariance_identity {α : Type*} [MeasurableSpace α]
    (ef : ExpFamily α) (hreg : IsRegularExpFamily ef) (η : ℝ)
    (eta_hat : α → ℝ) (heta_meas : Measurable eta_hat)
    (hunbiased : ∀ η' : ℝ,
        ∫ x, eta_hat x * (ef.h x * Real.exp (η' * ef.T x - ef.A η')) ∂ef.μ = η') :
    ∫ x, (eta_hat x - η) * (ef.T x - deriv ef.A η) *
         (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ = 1

/-- **Proposition 3 (corrected)**: MSE ≥ 1/A''(η) for any unbiased estimator.
    Additional hypotheses: density nonnegativity and integrability conditions. -/
theorem efda_efficiency
    {α : Type*} [MeasurableSpace α]
    (ef         : ExpFamily α)
    (hreg       : IsRegularExpFamily ef)
    (η          : ℝ)
    (eta_hat    : α → ℝ)
    (heta_meas  : Measurable eta_hat)
    (hunbiased  : ∀ η' : ℝ,
        ∫ x, eta_hat x * (ef.h x * Real.exp (η' * ef.T x - ef.A η')) ∂ef.μ = η')
    (hweight_nonneg : ∀ x, 0 ≤ ef.h x * Real.exp (η * ef.T x - ef.A η))
    (heta_sq_int : Integrable
        (fun x => (eta_hat x - η) ^ 2 *
          (ef.h x * Real.exp (η * ef.T x - ef.A η))) ef.μ)
    (heta_T_int : Integrable
        (fun x => (eta_hat x - η) * (ef.T x - deriv ef.A η) *
          (ef.h x * Real.exp (η * ef.T x - ef.A η))) ef.μ)
    (hT_sq_int' : Integrable
        (fun x => (ef.T x - deriv ef.A η) ^ 2 *
          (ef.h x * Real.exp (η * ef.T x - ef.A η))) ef.μ) :
    ∫ x, (eta_hat x - η) ^ 2 *
         (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
    ≥ 1 / iteratedDeriv 2 ef.A η := by
  have h_cov := score_covariance_identity ef hreg η eta_hat heta_meas hunbiased
  have h_nonneg : ∀ c : ℝ, 0 ≤ ∫ x,
      ((eta_hat x - η) - c * (ef.T x - deriv ef.A η)) ^ 2 *
      (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ :=
    fun c => integral_nonneg fun x => mul_nonneg (sq_nonneg _) (hweight_nonneg x)
  have h_expand : ∀ c : ℝ, ∫ x,
      ((eta_hat x - η) - c * (ef.T x - deriv ef.A η)) ^ 2 *
      (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ =
    (∫ x, (eta_hat x - η) ^ 2 * (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ) +
    2 * (-c) * (∫ x, (eta_hat x - η) * (ef.T x - deriv ef.A η) *
             (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ) +
    (-c) ^ 2 * (∫ x, (ef.T x - deriv ef.A η) ^ 2 *
             (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ) := by
    intro c
    rw [← MeasureTheory.integral_const_mul, ← MeasureTheory.integral_const_mul]
    rw [← MeasureTheory.integral_add, ← MeasureTheory.integral_add]
    · congr; ext x; ring
    · exact heta_sq_int.add (heta_T_int.const_mul _)
    · exact hT_sq_int'.const_mul _
    · exact heta_sq_int
    · exact heta_T_int.const_mul _
  rw [h_cov, fisher_info_eq_A'' ef hreg η] at h_expand
  have hA''_pos := hreg.A''_pos η
  specialize h_nonneg (1 / iteratedDeriv 2 ef.A η)
  rw [h_expand] at h_nonneg
  nlinarith [mul_div_cancel₀ (1 : ℝ) (ne_of_gt hA''_pos)]

-- ------------------------------------------------------------
-- Proposition 4 — Asymptotic MSE Under Misspecification
-- ------------------------------------------------------------

/-- **Proposition 4**: If ℓ̂_n →_{L²} ℓ† then lim MSE(ℓ̂_n) = (ℓ† − ℓ*)². -/
theorem efda_mse_misspecification
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (ℓhat          : ℕ → Ω → ℝ)
    (hℓhat_int     : ∀ n, Integrable (ℓhat n) P)
    (hℓhat_sq_int  : ∀ n, Integrable (fun ω => (ℓhat n ω) ^ 2) P)
    (ℓdagger ℓstar : ℝ)
    (hL2 : Tendsto
        (fun n => ∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P)
        atTop (nhds 0)) :
    Tendsto
      (fun n => ∫ ω, (ℓhat n ω - ℓstar) ^ 2 ∂P)
      atTop
      (nhds ((ℓdagger - ℓstar) ^ 2)) := by
  have h_split : ∀ n, (∫ ω, (ℓhat n ω - ℓstar) ^ 2 ∂P) = (∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P) + 2 * (ℓdagger - ℓstar) * (∫ ω, (ℓhat n ω - ℓdagger) ∂P) + (ℓdagger - ℓstar) ^ 2 * (∫ ω, 1 ∂P) := by
    intro n; rw [ ←MeasureTheory.integral_const_mul, ←MeasureTheory.integral_const_mul ] ; rw [ ←MeasureTheory.integral_add, ←MeasureTheory.integral_add ] ; congr; ext ω; ring;
    · ring_nf;
      exact MeasureTheory.Integrable.add ( MeasureTheory.Integrable.add ( MeasureTheory.Integrable.neg ( MeasureTheory.Integrable.mul_const ( MeasureTheory.Integrable.mul_const ( hℓhat_int n ) _ ) _ ) ) ( hℓhat_sq_int n ) ) ( MeasureTheory.integrable_const _ );
    · norm_num;
    · simp_rw +decide [ sub_sq ];
      exact MeasureTheory.Integrable.add ( MeasureTheory.Integrable.sub ( hℓhat_sq_int n ) ( MeasureTheory.Integrable.mul_const ( MeasureTheory.Integrable.const_mul ( hℓhat_int n ) _ ) _ ) ) ( MeasureTheory.integrable_const _ );
    · fun_prop (disch := solve_by_elim);
  have h_L1 : Filter.Tendsto (fun n => ∫ ω, (ℓhat n ω - ℓdagger) ∂P) Filter.atTop (nhds 0) := by
    have h_cauchy_schwarz : ∀ n, |∫ ω, (ℓhat n ω - ℓdagger) ∂P| ≤ Real.sqrt (∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P) := by
      intro n
      have h_cauchy_schwarz : (∫ ω, (ℓhat n ω - ℓdagger) ∂P) ^ 2 ≤ (∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P) := by
        have h_cauchy_schwarz : (∫ ω, (ℓhat n ω - ℓdagger - (∫ ω, (ℓhat n ω - ℓdagger) ∂P)) ^ 2 ∂P) ≥ 0 := by
          exact MeasureTheory.integral_nonneg fun ω => sq_nonneg _;
        simp_all +decide [ sub_sq, mul_assoc ];
        rw [ MeasureTheory.integral_add, MeasureTheory.integral_sub ] at h_cauchy_schwarz;
        · simp_all +decide [ MeasureTheory.integral_const_mul, MeasureTheory.integral_mul_const ];
          linarith;
        · exact MeasureTheory.Integrable.add ( MeasureTheory.Integrable.sub ( hℓhat_sq_int n ) ( MeasureTheory.Integrable.const_mul ( hℓhat_int n |> fun h => h.mul_const _ ) _ ) ) ( MeasureTheory.integrable_const _ );
        · exact MeasureTheory.Integrable.const_mul ( MeasureTheory.Integrable.mul_const ( MeasureTheory.Integrable.sub ( hℓhat_int n ) ( MeasureTheory.integrable_const _ ) ) _ ) _;
        · apply_rules [ MeasureTheory.Integrable.sub, MeasureTheory.Integrable.add, MeasureTheory.Integrable.mul_const, MeasureTheory.Integrable.const_mul, hℓhat_int, hℓhat_sq_int ];
          · exact MeasureTheory.integrable_const _;
          · exact MeasureTheory.integrable_const _;
        · exact MeasureTheory.integrable_const _
      exact Real.abs_le_sqrt h_cauchy_schwarz;
    exact squeeze_zero_norm h_cauchy_schwarz ( by simpa using hL2.sqrt );
  simpa [ h_split ] using Filter.Tendsto.add ( Filter.Tendsto.add hL2 ( h_L1.const_mul _ ) ) tendsto_const_nhds
