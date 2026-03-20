import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.Probability.StrongLaw
import Mathlib.Probability.IdentDistrib
import Mathlib.Probability.Independence.Basic
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
import Mathlib.Analysis.Calculus.ContDiff.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Convex.Integral
import Mathlib.Analysis.Convex.Mul

set_option linter.style.whitespace false

/-!
# EFDA Formal Verification Challenge
-/

open MeasureTheory Filter Topology ProbabilityTheory

-- ============================================================
-- Exponential Family Setup
-- ============================================================

structure ExpFamily (α : Type*) [MeasurableSpace α] where
A : ℝ → ℝ
T : α → ℝ
h : α → ℝ
μ : Measure α

structure IsRegularExpFamily {α : Type*} [MeasurableSpace α] (ef : ExpFamily α) : Prop where
A_smooth : ContDiff ℝ ⊤ ef.A
A''_pos  : ∀ η : ℝ, 0 < iteratedDeriv 2 ef.A η
T_sq_int : ∀ η : ℝ,
    Integrable (fun x => ef.T x ^ 2 * ef.h x * Real.exp (η * ef.T x - ef.A η)) ef.μ
normalized : ∀ η : ℝ,
    ∫ x, ef.h x * Real.exp (η * ef.T x - ef.A η) ∂ef.μ = 1

-- ============================================================
-- PROVIDED: Foundational Moment Identities (axioms)
-- ============================================================

axiom first_moment_identity {α : Type*} [MeasurableSpace α]
    (ef : ExpFamily α) (hreg : IsRegularExpFamily ef) (η : ℝ) :
    ∫ x, ef.T x * (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
    = deriv ef.A η

axiom fisher_info_eq_A'' {α : Type*} [MeasurableSpace α]
    (ef : ExpFamily α) (hreg : IsRegularExpFamily ef) (η : ℝ) :
    ∫ x, (ef.T x - deriv ef.A η) ^ 2 *
        (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
    = iteratedDeriv 2 ef.A η

-- ============================================================
-- Proposition 1a — SLLN for class proportions
-- ============================================================

theorem efda_consistency_alpha
    {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
    (Z : ℕ → Ω → ℝ)
    (hZ_int : Integrable (Z 0) P)
    (hZ_indep : Pairwise (Function.onFun (fun f g => IndepFun f g P) Z))
    (hZ_ident : ∀ i, IdentDistrib (Z i) (Z 0) P P)
    (α_star : ℝ)
    (hZ_mean : ∫ ω, Z 0 ω ∂P = α_star) :
    ∀ᵐ ω ∂P, Tendsto
    (fun (n : ℕ) => ((n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, Z i ω))
    atTop (nhds α_star) := by
have h := ProbabilityTheory.strong_law_ae Z hZ_int hZ_indep hZ_ident
simp_rw [hZ_mean] at h
exact h

-- ============================================================
-- Proposition 1b — Continuous mapping (inverse A')
-- ============================================================

theorem efda_consistency_eta
    (A : ℝ → ℝ) (_ : ContDiff ℝ ⊤ A)
    (Ainv : ℝ → ℝ)
    (hAinv_left : ∀ η, Ainv (deriv A η) = η)
    (hAinv_cont : Continuous Ainv)
    (η_star : ℝ)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Tbar : ℕ → Ω → ℝ)
    (hTbar : ∀ᵐ ω ∂P,
        Tendsto (fun n => Tbar n ω) atTop (nhds (deriv A η_star))) :
    ∀ᵐ ω ∂P, Tendsto (fun n => Ainv (Tbar n ω)) atTop (nhds η_star) := by
filter_upwards [hTbar] with ω hω
rw [← hAinv_left η_star]
exact hAinv_cont.continuousAt.tendsto.comp hω

-- ============================================================
-- Proposition 2 — Asymptotic Calibration (Bounded DCT)
-- ============================================================

theorem efda_calibration
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (phat : ℕ → Ω → ℝ)
    (pstar : Ω → ℝ)
    (hstar_meas : Measurable pstar)
    (hbound : ∀ n, ∀ᵐ ω ∂P, 0 ≤ phat n ω ∧ phat n ω ≤ 1)
    (hconv : ∀ᵐ ω ∂P,
        Tendsto (fun n => phat n ω) atTop (nhds (pstar ω)))
    (hphat_meas : ∀ n, AEStronglyMeasurable (phat n) P) :
    Tendsto
    (fun n => ∫ ω, phat n ω ∂P)
    atTop
    (nhds (∫ ω, pstar ω ∂P)) := by
apply tendsto_integral_of_dominated_convergence (fun _ => (1 : ℝ))
· exact hphat_meas
· exact integrable_const 1
· intro n
    filter_upwards [hbound n] with ω hω
    rw [Real.norm_eq_abs, abs_le]
    exact ⟨by linarith [hω.1], hω.2⟩
· exact hconv

-- ============================================================
-- Proposition 3 — MLE Efficiency (Cramér–Rao)
-- ============================================================

theorem efda_efficiency
    {α : Type*} [MeasurableSpace α]
    (ef : ExpFamily α)
    (hreg : IsRegularExpFamily ef)
    (η : ℝ)
    (eta_hat : α → ℝ)
    (heta_meas : Measurable eta_hat)
    (hunbiased : ∀ η' : ℝ,
        ∫ x, eta_hat x * (ef.h x * Real.exp (η' * ef.T x - ef.A η')) ∂ef.μ = η')
    (hcov : ∫ x, (eta_hat x - η) * (ef.T x - deriv ef.A η) *
        (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ = 1)
    (hp_nonneg : ∀ x, 0 ≤ ef.h x * Real.exp (η * ef.T x - ef.A η))
    (hmse_int : Integrable
        (fun x => (eta_hat x - η) ^ 2 *
        (ef.h x * Real.exp (η * ef.T x - ef.A η))) ef.μ)
    (hvar_int : Integrable
        (fun x => (ef.T x - deriv ef.A η) ^ 2 *
        (ef.h x * Real.exp (η * ef.T x - ef.A η))) ef.μ) :
    ∫ x, (eta_hat x - η) ^ 2 *
        (ef.h x * Real.exp (η * ef.T x - ef.A η)) ∂ef.μ
    ≥ 1 / iteratedDeriv 2 ef.A η := by
set p := fun x => ef.h x * Real.exp (η * ef.T x - ef.A η) with hp_def
set f := fun x => eta_hat x - η
set g := fun x => ef.T x - deriv ef.A η
set I := iteratedDeriv 2 ef.A η
have hI_pos : (0 : ℝ) < I := hreg.A''_pos η
-- Cauchy-Schwarz via non-negativity of ∫ (f - t·g)² p for t = 1/I
-- Expanding: MSE - 2t·cov + t²·I ≥ 0
-- At t = 1/I: MSE - 2/I + 1/I = MSE - 1/I ≥ 0
suffices h : 0 ≤ ∫ x, (f x - (1 / I) * g x) ^ 2 * p x ∂ef.μ by
    have expand : ∫ x, (f x - (1 / I) * g x) ^ 2 * p x ∂ef.μ =
        ∫ x, f x ^ 2 * p x ∂ef.μ -
        2 * (1 / I) * (∫ x, f x * g x * p x ∂ef.μ) +
        (1 / I) ^ 2 * (∫ x, g x ^ 2 * p x ∂ef.μ) := by
      sorry -- integral linearity expansion
    rw [expand, hcov, fisher_info_eq_A'' ef hreg η] at h
    -- h : 0 ≤ MSE - 2*(1/I)*1 + (1/I)²*I = MSE - 1/I
    have : 2 * (1 / I) * 1 - (1 / I) ^ 2 * I = 1 / I := by
      field_simp; ring
    linarith
-- ∫ (f - tg)² p ≥ 0 since (f - tg)² ≥ 0 and p ≥ 0
apply integral_nonneg
intro x
exact mul_nonneg (sq_nonneg _) (hp_nonneg x)

-- ============================================================
-- Proposition 4 — Asymptotic MSE Under Misspecification
-- ============================================================

theorem efda_mse_misspecification
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (ℓhat : ℕ → Ω → ℝ)
    (hℓhat_int : ∀ n, Integrable (ℓhat n) P)
    (hℓhat_sq_int : ∀ n, Integrable (fun ω => (ℓhat n ω) ^ 2) P)
    (ℓdagger ℓstar : ℝ)
    (hL2 : Tendsto
        (fun n => ∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P)
        atTop (nhds 0)) :
    Tendsto
    (fun n => ∫ ω, (ℓhat n ω - ℓstar) ^ 2 ∂P)
    atTop
    (nhds ((ℓdagger - ℓstar) ^ 2)) := by
-- (ℓ̂ - ℓ*)² = (ℓ̂ - ℓ†)² + 2(ℓ† - ℓ*)(ℓ̂ - ℓ†) + (ℓ† - ℓ*)²
set c := ℓdagger - ℓstar
-- Step 1: rewrite each integral using the algebraic identity
have identity : ∀ n, ∫ ω, (ℓhat n ω - ℓstar) ^ 2 ∂P =
    ∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P +
    2 * c * ∫ ω, (ℓhat n ω - ℓdagger) ∂P + c ^ 2 := by
    intro n
    have h1 : Integrable (fun ω => (ℓhat n ω - ℓdagger) ^ 2) P := by
    have : (fun ω => (ℓhat n ω - ℓdagger) ^ 2) = fun ω => ℓhat n ω ^ 2 - 2 * ℓdagger * ℓhat n ω + ℓdagger ^ 2 := by
        ext ω; ring
    rw [this]
    exact ((hℓhat_sq_int n).sub ((hℓhat_int n).const_mul _)).add (integrable_const _)
    have h2 : Integrable (fun ω => ℓhat n ω - ℓdagger) P :=
    (hℓhat_int n).sub (integrable_const _)
    have h3 : Integrable (fun ω => 2 * c * (ℓhat n ω - ℓdagger)) P :=
    h2.const_mul _
    -- Pointwise identity
    have pw : ∀ ω, (ℓhat n ω - ℓstar) ^ 2 =
        (ℓhat n ω - ℓdagger) ^ 2 + 2 * c * (ℓhat n ω - ℓdagger) + c ^ 2 := by
    intro ω; unfold_let c; ring
    simp_rw [pw]
    rw [integral_add (h1.add h3) (integrable_const _)]
    rw [integral_add h1 h3]
    rw [integral_const, smul_eq_mul, mul_one]
    ring
simp_rw [identity]
-- Step 2: ∫(ℓ̂-ℓ†) → 0 from L² convergence via Jensen: (E[X])² ≤ E[X²]
have hsub_int : ∀ n, Integrable (fun ω => ℓhat n ω - ℓdagger) P :=
    fun n => (hℓhat_int n).sub (integrable_const _)
have hsub_sq_int : ∀ n, Integrable (fun ω => (ℓhat n ω - ℓdagger) ^ 2) P := by
    intro n
    have : (fun ω => (ℓhat n ω - ℓdagger) ^ 2) =
        fun ω => ℓhat n ω ^ 2 - 2 * ℓdagger * ℓhat n ω + ℓdagger ^ 2 := by
    ext ω; ring
    rw [this]
    exact ((hℓhat_sq_int n).sub ((hℓhat_int n).const_mul _)).add (integrable_const _)
-- Jensen: (∫ f)² ≤ ∫ f² on probability space (convexity of x²)
have jensen : ∀ n, (∫ ω, (ℓhat n ω - ℓdagger) ∂P) ^ 2 ≤
    ∫ ω, (ℓhat n ω - ℓdagger) ^ 2 ∂P := by
    intro n
    exact (even_two.convexOn_pow (𝕜 := ℝ)).map_integral_le
    (continuous_pow 2).continuousOn isClosed_univ
    (Eventually.of_forall (fun _ => Set.mem_univ _))
    (hsub_int n) (hsub_sq_int n)
-- From (E[X])² ≤ E[X²] → 0, squeeze (E[X])² → 0, then E[X] → 0
have h_sq_zero : Tendsto (fun n => (∫ ω, (ℓhat n ω - ℓdagger) ∂P) ^ 2) atTop (nhds 0) :=
    squeeze_zero (fun n => sq_nonneg _) jensen hL2
have hL1 : Tendsto (fun n => ∫ ω, (ℓhat n ω - ℓdagger) ∂P) atTop (nhds 0) := by
    rw [tendsto_zero_iff_abs_tendsto_zero]
    have : (fun n => |∫ ω, (ℓhat n ω - ℓdagger) ∂P|) =
        fun n => Real.sqrt ((∫ ω, (ℓhat n ω - ℓdagger) ∂P) ^ 2) := by
    ext n; exact (Real.sqrt_sq_eq_abs _).symm
    rw [this, ← Real.sqrt_zero]
    exact (Real.continuous_sqrt.tendsto 0).comp h_sq_zero
-- Step 3: combine limits
have := hL2.add ((hL1.const_mul (2 * c)).add tendsto_const_nhds)
simp only [zero_add, mul_zero] at this
convert this using 1
ring