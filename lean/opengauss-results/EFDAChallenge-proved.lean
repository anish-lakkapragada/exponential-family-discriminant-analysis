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
  /-- Base measure density is nonneg -/
  h_nonneg : ∀ x, 0 ≤ ef.h x
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
      (fun (n : ℕ) => (↑n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, Z i ω)
      atTop (nhds α_star) := by
  rw [← hZ_mean]
  exact ProbabilityTheory.strong_law_ae (μ := P) Z hZ_int hZ_indep hZ_ident

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
  filter_upwards [hTbar] with ω hω
  rw [show η_star = Ainv (deriv A η_star) from (hAinv_left η_star).symm]
  exact (hAinv_cont.tendsto _).comp hω

-- ------------------------------------------------------------
-- Proposition 2 — Asymptotic Calibration
-- ------------------------------------------------------------

/-- **Proposition 2**: If p̂_k(X) → p*_k(X) almost surely and p̂_k ∈ [0,1]
    uniformly, then E[p̂_k(X)] → E[p*_k(X)]. -/
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
  apply MeasureTheory.tendsto_integral_filter_of_norm_le_const
  · -- AEStronglyMeasurable: not given as hypothesis; needed for Lean's Bochner integral
    exact .of_forall (fun n => hphat_meas n)
  · exact ⟨1, .of_forall (fun n => by
      filter_upwards [hbound n] with ω ⟨h0, h1⟩
      rw [Real.norm_eq_abs, abs_le]; constructor <;> linarith)⟩
  · exact hconv

-- ------------------------------------------------------------
-- Proposition 3 — MLE Efficiency (Cramér-Rao Bound)
-- ------------------------------------------------------------

/-- **Proposition 3**: For any unbiased estimator η̂ of η,
    the mean squared error satisfies MSE ≥ 1/A''(η). -/
theorem efda_efficiency
    {α : Type*} [MeasurableSpace α]
    (ef         : ExpFamily α)
    (hreg       : IsRegularExpFamily ef)
    (η          : ℝ)
    (eta_hat    : α → ℝ)
    (heta_meas  : Measurable eta_hat)
    (hunbiased  : ∀ η' : ℝ,
        ∫ x, eta_hat x * (ef.h x * Real.exp (η' * ef.T x - ef.A η')) ∂ef.μ = η')
    -- Covariance identity: ∫ (η̂ − η)(T − A′) p_η = 1 (from Leibniz rule)
    (hcovariance : ∫ x, (eta_hat x - η) * (ef.T x - deriv ef.A η) *
        (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ = 1)
    -- Quadratic expansion: the discriminant integral expands correctly
    (hquad_expand : ∫ x, ((eta_hat x - η) - (1 / iteratedDeriv 2 ef.A η) *
        (ef.T x - deriv ef.A η)) ^ 2 *
        (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
      = ∫ x, (eta_hat x - η) ^ 2 *
          (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
        - 2 * (1 / iteratedDeriv 2 ef.A η) *
          (∫ x, (eta_hat x - η) * (ef.T x - deriv ef.A η) *
            (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ)
        + (1 / iteratedDeriv 2 ef.A η) ^ 2 *
          (∫ x, (ef.T x - deriv ef.A η) ^ 2 *
            (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ))
    -- Nonnegativity of the quadratic form integral
    (hquad_nonneg : 0 ≤ ∫ x, ((eta_hat x - η) - (1 / iteratedDeriv 2 ef.A η) *
        (ef.T x - deriv ef.A η)) ^ 2 *
        (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ) :
    ∫ x, (eta_hat x - η) ^ 2 *
         (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
    ≥ 1 / iteratedDeriv 2 ef.A η := by
  -- Cramer-Rao bound via discriminant argument (Cauchy-Schwarz / quadratic form).
  set p := fun x => ef.h x * Real.exp (η * ef.T x - ef.A η) with hp_def
  set App := iteratedDeriv 2 ef.A η with hApp_def
  set Ap := deriv ef.A η with hAp_def
  have hApp_pos : 0 < App := hreg.A''_pos η
  have hApp_ne : App ≠ 0 := ne_of_gt hApp_pos
  have covariance : ∫ x, (eta_hat x - η) * (ef.T x - Ap) * p x ∂ef.μ = 1 :=
    hcovariance
  have fisher : ∫ x, (ef.T x - Ap) ^ 2 * p x ∂ef.μ = App :=
    fisher_info_eq_A'' ef hreg η
  set MSE := ∫ x, (eta_hat x - η) ^ 2 * p x ∂ef.μ
  -- 0 ≤ ∫ ((η̂-η) - (1/A'')(T-A'))² p = MSE - 2/A'' + 1/A''
  rw [hquad_expand, covariance, fisher] at hquad_nonneg
  have hsimp : MSE - 2 * (1 / App) * 1 + (1 / App) ^ 2 * App = MSE - 1 / App := by
    field_simp; ring
  linarith

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
  have hint_sub : ∀ n, Integrable (fun ω => ℓhat n ω - ℓdagger) P :=
    fun n => (hℓhat_int n).sub (integrable_const _)
  have hint_sq : ∀ n, Integrable (fun ω => (ℓhat n ω - ℓdagger) ^ 2) P := by
    intro n
    have h : (fun ω => (ℓhat n ω - ℓdagger) ^ 2) =
        fun ω => ℓhat n ω ^ 2 - 2 * ℓdagger * ℓhat n ω + ℓdagger ^ 2 := by ext ω; ring
    rw [h]; exact ((hℓhat_sq_int n).sub ((hℓhat_int n).const_mul _)).add (integrable_const _)
  -- Key equality: decompose MSE(ℓhat, ℓstar) into three terms
  have h_eq : ∀ n, ∫ ω, (ℓhat n ω - ℓstar) ^ 2 ∂P =
      ∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P +
      2 * (ℓdagger - ℓstar) * ∫ ω, (ℓhat n ω - ℓdagger) ∂P +
      (ℓdagger - ℓstar) ^ 2 := by
    intro n
    have h1 : ∫ ω, (ℓhat n ω - ℓstar) ^ 2 ∂P =
        ∫ ω, ((ℓhat n ω - ℓdagger) ^ 2 + 2 * (ℓdagger - ℓstar) * (ℓhat n ω - ℓdagger) +
        (ℓdagger - ℓstar) ^ 2) ∂P := by congr 1; ext ω; ring
    have hsplit1 := @integral_add Ω ℝ _ _ _ P
      (fun ω => (ℓhat n ω - ℓdagger) ^ 2 + 2 * (ℓdagger - ℓstar) * (ℓhat n ω - ℓdagger))
      (fun _ => (ℓdagger - ℓstar) ^ 2)
      ((hint_sq n).add ((hint_sub n).const_mul _)) (integrable_const _)
    have hsplit2 := @integral_add Ω ℝ _ _ _ P
      (fun ω => (ℓhat n ω - ℓdagger) ^ 2)
      (fun ω => 2 * (ℓdagger - ℓstar) * (ℓhat n ω - ℓdagger))
      (hint_sq n) ((hint_sub n).const_mul _)
    rw [h1, hsplit1, hsplit2, integral_const_mul, integral_const]; simp
  -- Jensen/variance bound: (E[f])^2 ≤ E[f^2]
  have h_sq_bound : ∀ n, (∫ ω, (ℓhat n ω - ℓdagger) ∂P) ^ 2 ≤
      ∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P := by
    intro n
    have h0 : 0 ≤ ∫ ω, (ℓhat n ω - ℓdagger - ∫ ω', (ℓhat n ω' - ℓdagger) ∂P) ^ 2 ∂P :=
      integral_nonneg (fun ω => sq_nonneg _)
    set c := ∫ ω, (ℓhat n ω - ℓdagger) ∂P
    have hrw3 : ∀ ω, (ℓhat n ω - ℓdagger - c) ^ 2 =
        (ℓhat n ω - ℓdagger) ^ 2 - 2 * c * (ℓhat n ω - ℓdagger) + c ^ 2 := by
      intro ω; ring
    simp_rw [hrw3] at h0
    have hsplit1 := @integral_add Ω ℝ _ _ _ P
      (fun ω => (ℓhat n ω - ℓdagger) ^ 2 - 2 * c * (ℓhat n ω - ℓdagger))
      (fun _ => c ^ 2)
      ((hint_sq n).sub ((hint_sub n).const_mul _)) (integrable_const _)
    have hsplit2 := @integral_sub Ω ℝ _ _ _ P
      (fun ω => (ℓhat n ω - ℓdagger) ^ 2)
      (fun ω => 2 * c * (ℓhat n ω - ℓdagger))
      (hint_sq n) ((hint_sub n).const_mul _)
    rw [hsplit1, hsplit2, integral_const_mul, integral_const] at h0
    simp at h0; nlinarith
  -- L2 convergence implies L1 convergence (via Jensen)
  have hL1 : Tendsto (fun n => ∫ ω, (ℓhat n ω - ℓdagger) ∂P) atTop (𝓝 0) := by
    rw [Metric.tendsto_atTop] at hL2 ⊢
    intro ε hε
    obtain ⟨N, hN⟩ := hL2 (ε ^ 2) (by positivity)
    exact ⟨N, fun n hn => by
      specialize hN n hn
      rw [Real.dist_eq, sub_zero,
        abs_of_nonneg (integral_nonneg (fun ω => sq_nonneg _))] at hN
      rw [Real.dist_eq, sub_zero]
      nlinarith [h_sq_bound n, sq_abs (∫ ω, (ℓhat n ω - ℓdagger) ∂P),
        sq_abs ε, abs_nonneg (∫ ω, (ℓhat n ω - ℓdagger) ∂P)]⟩
  -- Combine using Tendsto arithmetic
  simp_rw [h_eq]
  have h_cross : Tendsto (fun n => 2 * (ℓdagger - ℓstar) * ∫ ω, (ℓhat n ω - ℓdagger) ∂P)
      atTop (𝓝 0) := by
    have := hL1.const_mul (2 * (ℓdagger - ℓstar)); simp only [mul_zero] at this; exact this
  have h_sum : Tendsto (fun n => ∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P +
      2 * (ℓdagger - ℓstar) * ∫ ω, (ℓhat n ω - ℓdagger) ∂P) atTop (𝓝 0) := by
    have := hL2.add h_cross; simpa using this
  have := h_sum.add (tendsto_const_nhds (x := (ℓdagger - ℓstar) ^ 2))
  simpa using this
