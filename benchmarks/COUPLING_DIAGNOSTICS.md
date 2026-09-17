# Positive-gauge coupling diagnostics

## Six original operator groups (new complete diagnostic)

`coupled_actions.py` reconstructs the PG factors and connections from coherent
Psi, not from previously saved principal-phase bond links. Write
`Dq=-i partial_q-a`, `DR=-i partial_R-b`, `Dp=-i partial_R+b-alpha`.
The six actions are

1. Dq² Phi/(2mp)
2. (a-i partial_q F/F) Dq Phi/mp
3. DR² Phi/(2M)
4. (b-i partial_R F/F) DR Phi/M
5. Dp² Lambda/(2M)
6. (alpha-i partial_R chi/chi) Dp Lambda/M.

Actions 1--4 are summed as complex functions before the electronic norm is
taken. Actions 5--6 are likewise summed before taking a magnitude. Electronic
maps use `sqrt(integral dx |action|²)`. Proton maps use `|action|/Lambda` on
occupied support. With weight rho=chi² Lambda² these yield comparable
wavefunction-weighted RMS energies, not pointwise forces. Signed real and
imaginary expectations are stored separately. Only real means are plotted by
the initial report. Global summaries use the common finite full-resolution
rho>=1e-3 support, normalized to its mass (stored explicitly).

`hBO` excludes the external heavy trap. `Hel=hBO+Ue`;
`Hpr=Tp+epsilon1+Up`. PG epsilon1 is obtained from the independently evaluated
Hel conditional expectation and the supplied native H Psi temporal action,
not defined by a residual of these six action norms. Raw H actions are not the
complete factor time derivatives: the coupled evolution equations also have
scalar/gauge terms. Norm magnitudes are gauge/convention dependent.

q/R operators use central5 continuum diagnostics. x uses the current core
finite electronic operator for BO trajectories and continuum DST-I modes for
spectral trajectories. Finite derivative product rules need not close exactly;
`native_action_difference` explicitly compares the diagnostic full-H action
with the supplied native action. This is an action difference, not a claim
that native discrete/spectral propagation was replaced. No term is adjusted
to force equality. Full-Psi input removes BO analysis truncation, but does not
remove the need for derivative/grid convergence tests.

### Existing BO trajectory, matching original basis cache available

```bash
python -m multi_component_exact_factorization.postprocess_coupled_actions RUN \
  --max-frames 8 --R-block 4 --map-stride 2
python -m multi_component_exact_factorization.coupled_action_report RUN \
  --max-frames 8 --snapshot-count 8 --fps 2
```

The original fingerprinted cache is required; `--basis-dir` can point to a
relocated cache folder containing metadata.json/states.npy. It is never rebuilt
silently with unrelated eigenvector phases. Default preview is eight saved
times; use `--max-frames 0` in the postprocessor for every saved time. Use a
fresh `--outdir` for another analysis; existing diagnostics are not overwritten.
The processor streams coefficient histories and reconstructs coherent x/q/R
blocks. Saved native action coefficients are reused, or the original BO-link
action is evaluated on CPU if absent.

Full-grid archives are rejected by default because their full Psi history is
not retained. `--allow-bo-projection` explicitly produces APPROXIMATE projected
diagnostics and labels their provenance; it is not full-grid recovery.

### Future propagation, full Psi available in memory

Add to the existing TDSE command:

```bash
--save-coupled-actions --coupled-actions-R-block 4 --coupled-actions-map-stride 2
```

This is opt-in because it adds two coherent-Psi passes per saved frame, CPU
derivatives, and disk output. The default propagation remains unchanged. Maps
are decimated only after all derivatives/statistics; default stride 2 keeps
every second q and R point. The sidecar `coupled_action_diagnostics.npz` records
maps, full-resolution summaries, coordinates, times, numerical convention and
source provenance. Separate staging files are retained on failure and removed
only after successful NPZ creation. Updating code cannot enable this observer
inside an already running Python process.

Final visualization automatically includes the new product if this cache is
present. Render only it using `--only six_coupling`. No EF cache is needed for
the six-action report; the full coherent-state observer supplies its own fields.

## Earlier density-only proton--heavy diagnostic

Run the new product alone:

```bash
python -m multi_component_exact_factorization.render_final_visualizations RUN \
  --only coupling --max-frames 415 --fps 12 --snapshot-count 8
```

The default full visualization set also includes `coupling`. No propagation
is performed. Outputs are under `RUN/report/final_visualizations`:
`proton_heavy_coupling_movie.mp4`, individual frames, snapshot summary,
`proton_heavy_coupling_time_summary.png`, and diagnostic JSON.

The implementation follows `core.proton_base_operator` **without tail gates**
as an occupied-support continuum diagnostic, not as a reconstruction of the
native finite-link time propagator. It uses the existing `covariant_square`
and five-point derivatives. Saved forward-bond b and alpha are averaged onto
sites before evaluating D = -i partial_R + b - alpha. In PG:

    U_pn Lambda = D² Lambda/(2M)
                  + alpha D Lambda/M
                  - i (partial_R chi/chi) D Lambda/M.

The map shows b_site-alpha_site and magnitudes of the three action/Lambda
ratios. Ratios are not signed scalar potentials. The time summary shows
rho-weighted RMS and real mean of each ratio, and their **complex sum before
taking magnitudes**. Thus interference/cancellation is retained. Statistics
are normalized to the common finite support rho_qR >= 1e-3; this is not a
full-domain expectation value. JSON records retained support probability and
phase-branch warnings. No phase unwrapping or numerical smoothing is applied.

These amplitudes are reconstructed from the density matching the EF fields.
For full-grid TDSE archives whose EF fields were reconstructed from truncated
BO projections, this is an approximate mixed reconstruction unless projection
loss is negligible. Small global loss alone does not bound derivative errors.

Electronic U_ep Phi and U_en Phi are NOT implemented by this product. They need
coherent Phi and its first/second derivatives, including the BO basis dependence
if reconstructed from Y_j. Density, a/b, scalar geometry and channel populations
alone do not specify those actions. For a BO-propagated run, Y plus its matching
BO basis can support an additional blockwise postprocess. For a full-grid run,
the discarded BO projection cannot be recovered from the saved two coefficients;
exact action diagnostics need full-Psi snapshots or online diagnostics during
propagation. Do not label this product as all electronic coupling actions.
# Coupling report layout

The six-action report uses three synchronized movies and snapshot sets instead
of one nine-panel movie:

- `electronic_coupling_terms`: four electronic action contributions (2 × 2).
- `proton_heavy_coupling_terms`: two proton–heavy contributions and signed
  `b-alpha` (1 × 3).
- `coupling_action_totals`: the two coherent total action magnitudes (1 × 2).

`six_coupled_actions_time_summary.png` separately compares action strengths,
Hamiltonian actions, and signed expectations. Numerical definitions, shared
color limits within each physical group, density masks, and selected times are
unchanged. Existing older output files are not deleted automatically.
