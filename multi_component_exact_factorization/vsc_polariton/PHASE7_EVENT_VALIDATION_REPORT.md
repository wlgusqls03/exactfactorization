# Phase7 roadmap Steps 1 and 2 — measured local results

Scope: nine imported event waves and the three existing final snapshots.
No TDSE propagation, modified physics, relaxed tolerances, or changes to
historical MCEF / Phase1–6 source or results. Old failed outputs remain.

## Decision

* **Step1 event assessment completed; scalar gate NOT PASS (7/9 pass).**
* **Step2 same-wave gauge/force postprocessing PASS on all 9 events and 3
  endpoints**, after the predeclared resolution extension where necessary.
* **Phase7 overall remains NOT PASS.** This is not propagated-grid, Fock,
  temporal-probe, all-time, or historical-field-compatibility certification.

Authoritative bounded-scope aggregate:
`results/vsc_polariton/phase7/event_validation_v1/step12_review.json`.

## Step1: where scalar agreement fails

Each event was evaluated with two photon quadratures (free32/64;
coupled176/208) and excluded-probability budgets1e-6/1e-8/1e-10.
Native wave derivatives and instantaneous `dotPsi=-i H Psi` were reused.
The scalar Route A/B and imaginary RMS criterion remains1e-6 Ha.

| Case/event | Time fs | epsilon1 Route RMS, budget1e-8 | budget1e-10 |
|---|---:|---:|---:|
| Resonant, return-flux event | 14.7068 | 9.651e-7 Ha | **1.575e-6 Ha** |
| Barrier-frequency, return-flux event | 16.2549 | 6.256e-7 Ha | **1.095e-6 Ha** |

Both fail the unchanged all-budget scalar gate. The other seven imported
events pass the scalar checks; this does not supersede failed final-frame
Fock tests in the original pilot.

At the resonant14.7068fs event, the conditional regional RMS at budget1e-8
is3.645e-6 Ha in |R|<=0.5a0 (about5.53% joint probability), and3.680e-6 Ha
on the product side (about5.73%). These are regionally normalized diagnostics,
not a newly introduced gate. The discrepancy cannot honestly be described
as confined to empty tails. Do not change the mask to manufacture PASS.

Two photon quadratures diagnose quadrature sensitivity, not dynamic Fock
convergence. The nine-event archive contains F120 coupled waves only.
F160 matching event waves are needed for the next Fock check; existing
F160 endpoint data alone do not certify these events. No request for a
new Fock propagation is justified before reusing those existing runs.

## Step2: why the earlier force check failed and what changed

The old native-grid finite differences did not resolve sharp nonlinear
ratios and phase gradients near interference minima. The new method keeps
the SAME finite Fourier wave and SAME instantaneous derivative:

1. Preserve the signed native Nyquist mode when evaluating the Fourier
   wave (do not silently split it as a real-signal interpolation convention).
2. Form density, momentum numerator and their time derivatives at2N before
   taking Fourier coefficients of products, avoiding product aliasing.
3. Evaluate these one-dimensional bilinears on finer R samples; never
   smooth potentials or divide by a density floor.
4. Integrate eta_R=-alpha and eta_t,R=-alpha_t within each connected support.
5. Independently differentiate the transformed scalar and transformed
   conditional wave. The latter uses exact translated-wave overlaps,
   not an algebraic assignment alpha=0.
6. Compare -epsilon_R+alpha_t against the A=0 scalar slope, and compare
   currents and two independent postprocessing resolutions.

The initial2/4/8/16/32/64/128/256 ladder is retained. The two late free
events needed a separately predeclared512/1024/2048 extension. Both highest
resolutions had to pass; no tolerances were relaxed. Endpoint128/256 checks
were sufficient for all three final snapshots.

Representative density-weighted force consistency RMS (Ha/a0), budget1e-8:

| Snapshot | Coarser check | Finest check | Two-resolution difference |
|---|---:|---:|---:|
| Free21.6732fs, factors1024/2048 | 6.415e-7 | 1.127e-8 | 6.312e-7 |
| Free20.8992fs, factors1024/2048 | 2.078e-8 | 2.878e-9 | 2.064e-8 |
| Free39.9600fs, factors128/256 | 4.031e-6 | 7.155e-8 | 3.960e-6 |
| Resonant39.9600fs, factors128/256 | 7.570e-12 | 1.562e-11 | 1.851e-11 |
| Barrier39.9600fs, factors128/256 | 7.571e-12 | 1.456e-11 | 1.704e-11 |

All are below the fixed1e-5 force tolerance at the two accepted resolutions.
The normalized connection/current1e-3 gate also passes across all budgets.
Gauge-phase reconstruction L2 is below7e-17 and density L1 below9e-17
on checked support. Sharp force structures are retained, not smoothed away.
Their physical interpretation still requires propagated-grid convergence:
postprocessing consistency alone cannot certify that such narrow features
are fully converged physical structures.

Negative reconstructed-density roundoff, of order1e-17 integrated mass,
is logged; only support selection clips negative roundoff to zero. Ratios
use the original positive density, with no amplitude floor. Stencil-edge
excluded probability is explicitly reported. No disconnected phases are
joined, and no electronic a=b=0 gauge is assumed.

## Source map, arrays, units, provenance and tests

`phase7_dealiased_gauge.py lines 13–22`: `evaluate`, exact native Fourier
series/derivatives, `(N,channels)` to `(Nfine,channels)`; coordinates in a0.
Verified against analytic modes including the negative Nyquist mode.

`phase7_dealiased_gauge.py lines 25–37`: `bilinears`, sums over physical
dx-weighted electron/Fock channels into `(2N,4)` Fourier coefficients;
native `u,ut` shape `(R,x,n)`. Independent direct-wave tests.

`phase7_dealiased_gauge.py lines 40–47`: `shifted_overlap`,
K_s(R)=sum_n integral dx u*(R)u(R+s); `(2N,)` complex coefficients.
Direct-wave equality test; enables independent transformed-connection test.

`phase7_dealiased_gauge.py lines 50–101`: `probe`, connected gauge, native
force vs fourth/sixth-order transformed scalar slope, current and imaginary
residual. Outer fields `(Rfine,)`; scalars Ha, force Ha/a0, other units au.
Analytic freely interfering-wave test gives zero physical force.

`run_phase7_event_validation.py lines 18–91`: hash-verified archive driver,
two photon grids, regional scalar diagnostics, fixed resolution ladder.
Reuses `outer_fields`, `nested_fields`, `hermite_grid`, `checks_for`,
`budget_support`, `weighted_rms`; does not modify their physical operators.

`phase7_gauge_extension.py lines 15–59`: fixed extension only on failed
event frames; reuses bilinear caches and original waves.

`phase7_endpoint_gauge_check.py lines 12–44`: independent revisit of three
original final snapshots using the same procedure and unchanged packets.

`phase7_event_review.py lines 8–21`: `pair_check`, common-coordinate force,
density and reconstruction comparison. Lines24–53 aggregate gates without
promoting them into full Phase7 PASS.

`phase7_event_validation_plotting.py lines 11–51`: PNG/PDF error-location
and force-resolution figures from stored diagnostics, fixed comparison scales.

`phase7_pilot_baseline.py`: local audit imports moved inside `main` so that
portable constants/helpers do not import missing historical audit modules.
This does not change historical physics or baseline audit functionality.

Equation conventions and physical provenance remain those of
PHASE7_NESTED_EF_DERIVATION and the validated Phase6 full PF packets.

Tests: existing MCEF181 PASS; VSC107 PASS, including three new dealiased
Fourier/gauge tests. Protected historical hash audit reports only the
previously documented user-authorized visualization change277c45e, not a
change introduced here. `git diff --check` must also be clean at handoff.

## Outputs and next action

* `event_validation_v1/`: two-q-grid fields per event, bilinear caches,
  diagnostic JSON, review, tests, figures.
* `event_gauge_extension_v1/`: retained fine-resolution free-case checks.
* `endpoint_gauge_v2/`: three final-frame checks.

Figures: `event_validation_v1/figures/force_resolution.png` and
`scalar_error_locations.png`, with matching PDFs.

No server propagation is needed for the completed Step1/2 postprocessing.
The next unresolved numerical issue is matched-event Fock convergence of
epsilon1. Preserve F160 and any native R-grid refinement waves. Reuse them
before asking for a new propagation. Do not yet claim full-time Phase7 PASS,
barrier lowering, or a converged Phase4/full3D force mechanism.
