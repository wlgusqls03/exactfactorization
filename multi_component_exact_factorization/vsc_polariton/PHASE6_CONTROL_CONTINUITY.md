# Free-control continuity failure: numerical diagnosis and correction

The server `finish_campaign_v2` passed the free-sector stationary check but
stopped in free 2D-A at t=1020 au: |dP/dt-J(0)|=2.0092332766e-6,
above the unchanged 2e-6 limit. Full 3DOF results are not invalidated.

## Independent local reproduction

Exact decoupled photon vacuum (N=1, not a BO-electron approximation to the
full3D run) reproduces the old control: maximum product difference 2.88e-11,
linear-flux residual difference 8.40e-16 over the server's saved interval.
All probes continue to 1652 au, retaining failures rather than promoting them.

| N_R | dR | dt | max residual, linear flux | max residual, spectral point flux |
|---|---|---|---|---|
|352|.025|.125|2.02546186e-6|5.10403635e-7|
|352|.025|.0625|2.02546186e-6|5.10403635e-7|
|440|.020|.125|1.29710734e-6|3.25929119e-7|
|550|.016|.125|8.30483120e-7|2.08297100e-7|
|704|.0125|.125|5.07028559e-7|1.27008970e-7|

Refined-grid PES values use cubic interpolation of the direct-FGH production
array for this numerical sensitivity probe only. They are NOT new independent
electronic-grid validation or production literature settings.

The cell-centred grid straddles R=0 at +/-dR/2. Linear interpolation of sampled
current contributes the dominant O(dR^2) error; the population midpoint
quadrature also has O(dR^2) error. Halving dt does not help. This is not a
propagation instability or a GPU disagreement.

## Correction

`phase6_control_flux.py` evaluates the Fourier-represented wavefunction and
its analytic R derivative at R=0, then computes Im sum(Psi* dPsi)/M with the
same x quadrature. It does not interpolate current linearly. This is the
physical point flux, NOT a redefinition J=dP/dt to force zero residual.

Hamiltonian, propagator, saved cadence, population quadrature and thresholds
are unchanged. Historical `flux_linear` and `continuity_error_linear` are
retained. `flux_box_boundary` separately monitors the other periodic endpoint.
The correction applies to NEW reduced controls only; historical full3D outputs
are never overwritten. Full3D-vs-control flux comparisons therefore retain an
estimator difference: for the free control its maximum is 1.52e-6 au.

For the original N_R=352 grid at 39.96fs, corrected point residual=5.104e-7,
integrated continuity residual=6.269e-5, norm error=4.076e-12,
energy drift=2.492e-12 Ha. All existing absolute checks pass. All tested
grid/time observable comparisons also pass (maximum product grid difference
2.97e-5 < 2e-4). Production coupled controls still require server validation;
these free-control results do not certify all 27 jobs or Phase6.

Raw probe outputs are preserved separately in
`results/vsc_polariton/phase6_control_continuity_probe/` (original) and
`results/vsc_polariton/phase6_control_continuity_spectral_probe/` (corrected).

## Finish without rerunning completed full3D or stationary work

After obtaining the new source via Git, keep all previous directories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
python -u -m multi_component_exact_factorization.vsc_polariton.run_phase6_flux_finish
```

This wraps the existing finish driver, verifies stationary provenance against
the v2 source/input/selection manifest, reuses its passed stationary result,
and sends controls to the spectral-flux adapter. New results go to
`results/vsc_polariton/phase6_gpu/finish_campaign_v3`.
The failed v2 control remains untouched. No Phase7 is started.

```bash
tar -czf phase6_finish_v3_results.tar.gz \
  -C results/vsc_polariton/phase6_gpu finish_campaign_v3
```
