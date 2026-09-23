# Phase 7: initial/final snapshot pilot — NOT production PASS

## Scope and preservation

User-approved Phase6 core interval: 0–1652 au (39.959969 fs).
The supplied archive contains initial/final waves for free, resonant F120,
resonant F160, and barrier-frequency F120. All 21 archive hashes matched.
There are no intermediate event-time waves in this archive. No new TDSE was
run and no historical MCEF or Phase1–6 source/result was edited.

Baseline: `results/vsc_polariton/phase7/pilot_v1/baseline/`.
Baseline tests: existing MCEF 181 PASS; then-existing VSC 89 PASS.
After implementation: existing MCEF 181 PASS; VSC 97 PASS, including eight
new pilot tests; historical import/model/coupled-step/reference/TDPES/report
smoke PASS. Existing source hashes were unchanged. Regression scientific
output is under the repository results directory; test fixtures may use tmp.
The older protected manifest differs at the already user-authorized plotting
commit 277c45e only; the baseline explicitly verifies those exact bytes.
This is not a new exception authorizing edits to historical code.

## Actual calculations

`phase7_fock_to_q.py lines 10–37`: Gauss–Hermite transform and derivative
headroom, native `(R,x,n)` to `(R,x,q_c)`. HO functions have physical
quadrature weights. Norm, backprojection and photon moments are checked.

`phase7_native.py lines 9–28`: original Fourier nuclear/electronic kinetic
action and Hermitian Fock light–matter action, atomic units. Reuses packet
operator arrays and independently agrees with Phase6 PFBackend.

`phase7_nested_fields.py lines 17–48`: outer positive-marginal EF, using
`dot(Psi)=-i H Psi` rather than a two-frame time difference. Outer arrays
have shape `(R,)`. Computes two scalar routes and invariant force
`-partial_R epsilon2 + partial_t alpha` (not hydrodynamic acceleration).

`phase7_nested_fields.py lines 51–141`: blocked nested factorization,
conditional electron derivatives, scalar expectation and inversion routes,
middle expectation, `a,b,alpha`, Berry curvature and local BO character.
Fields have shape `(R,q_c)`, BO densities `(R,q_c,j)`. Electron and photon
PNC are checked on occupied support. Units: potentials Ha, nuclear force
Ha/a0; all coordinates/operators use packet atomic-unit conventions.
Formula provenance: PHASE7_NESTED_EF_DERIVATION and Li main Eq.(1), with
full dipole `(R-x)` and full squared DSE retained from Phase6.

`phase7_support.py lines 6–26`: excluded-probability budgets 1e-6/1e-8/1e-10,
connected-component labels and weighted errors, without filling nodes.

`phase7_outer_gauge.py lines 12–66`: component-local outer alpha=0 phase
integration and independent fourth/sixth-order derivative checks. Does not
set both electronic connections to zero or connect separate components.

`run_phase7_pilot.py lines 14–98`: immutable-input verification, two photon
quadratures per wave, support diagnostics and new NPZ/JSON output.
`run_phase7_gauge_pilot.py lines 10–38`: snapshot-only gauge validation.
`phase7_projection_audit.py lines 9–39`: independent analytic cutoff-defect
test on actual final waves. Does not repair/subtract the scalar discrepancy.
`phase7_free_gauge_refinement.py lines 14–46`: free-wave Fourier oversampling
diagnostic, not a new propagation or proof of native nuclear-grid convergence.
`phase7_pilot_plotting.py lines 12–65`: matched-scale provisional contour,
fixed-colour zoom, density/outer potential/force figures; PNG and PDF.

## Results

16 frame/quadrature combinations passed occupied-support factor/transform
checks in `analysis_v2`. Maximum errors:

| Diagnostic | Error |
|---|---:|
| Full reconstruction L2 | 1.057e-16 |
| Fock backprojection L2 | 3.451e-14 |
| Transform orthogonality | 4.313e-13 |
| Electronic PNC, occupied support | 3.553e-15 |
| Photon PNC, occupied support | 1.044e-13 |
| Outer epsilon2 expectation vs inversion, RMS Ha | 1.426e-16 |
| Nested vs outer epsilon2, RMS Ha | 4.126e-16 |

These algebra/representation tests do NOT establish derivative/basis
convergence. `analysis_v1` is preserved: its PNC gate incorrectly included
subnormal empty tails. `analysis_v2` checks the originally declared occupied
support and integrates BO populations from unnormalized wave projections,
avoiding undefined conditional character times zero density. No tolerance
was relaxed. Raw unsupported conditional quantities may remain nonfinite.

### First scalar: finite-Fock projection defect identified

Final-time epsilon1 Route A/B discrepancy, probability budget 1e-8:

| Case | RMS Ha | Remaining after analytic defect accounting, Ha |
|---|---:|---:|
| Resonant F120 | 1.070e-6 | 4.663e-16 |
| Resonant F160 | 1.101e-6 | 3.937e-16 |
| Barrier F120 | 1.839e-6 | 4.225e-16 |

Declared scalar tolerance is **1e-6 Ha**; these coupled final frames FAIL.
The finite-Fock Hamiltonian omits the LM raising action from n=N-1 to n=N:

`D = i dot(Psi) - H_cont Psi = -g_chi sqrt(N) mu_hat Psi_(N-1) phi_N(q)`.

Independently, `epsilon1_A-epsilon1_B = -<Psi|D>_x/rho_qR`.
This explains the discrepancy to roundoff on the actual waves. It is not
an arbitrary gauge offset, and increasing F120 to F160 does not remove it
monotonically. We keep both uncorrected routes and do not call this PASS.
Observable convergence in Phase6 is not automatically conditional-potential
convergence in Phase7. A projected/nonlocal EF formulation would be a
different explicitly documented construction, not a silent replacement.

### Gauge/force: derivative resolution not certified

Initial-frame alpha=0 checks pass. Final-frame native-grid checks fail,
including the free case. Interference makes marginal ratios and integrated
phase much sharper than the original wave. Free-case Fourier oversampling
2/4/8/16 was tested without changing the propagated state. At factor16 the
normalized connection/current gate passes, but independent force residual
is still 0.148 Ha/a0, above the declared 1e-5 tolerance. It is NOT valid to
present the current A=0 force plots as production results.

This does not by itself prove Phase6 dynamics invalid or demand a rerun.
Postprocessing quadrature/phase/derivative convergence must be resolved
first. Algebraically equal scalar routes alone cannot validate force.

## Outputs and next gate

All new output is under `results/vsc_polariton/phase7/pilot_v1/`:
`analysis_v1`, `analysis_v2`, `gauge_v1`, `projection_audit.json`,
`free_gauge_refinement.json`, `figures_v1`, `figures_v2`, `regression`.
Figures are explicitly marked provisional; the colour zoom saturates
outside +/-0.1 Ha and preserves a separate full-range plot.

Phase7 production remains **NOT PASS**. Outstanding work:

1. Resolve phase/force postprocessing resolution with independent checks.
2. Establish epsilon1 continuum/Fock convergence without defect subtraction
   or loosening thresholds; determine required retained-wave refinements.
3. Obtain intermediate saved waves and matching refinement snapshots;
   no movie or all-time claim can come from the present two times.
4. Complete historical potential/vector-field compatibility (current test
   covers historical reconstruction only), time-derivative probes, Phase4
   common-support force comparison, q/Fock/native-R derivative convergence.
5. Only then answer mechanistic questions or run the final global PASS audit.

No new conclusion about barrier lowering, electronic nonadiabaticity, or
full3D vs 2D force agreement is certified by this pilot.
