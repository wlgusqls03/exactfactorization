# Expanded positive-gauge proton–heavy analysis

This is a postprocessing extension, not a change to propagation or historical
MCEF equations. Existing `coupling` and `six_coupling` reports are preserved.
Select `--only proton_heavy`, or append it to a full report with
`--with-proton-heavy`. No new TDSE run or VSC data are needed for these plots.

## Implementation map and reuse

`proton_heavy_terms.py` lines 14–26: `TermConfig`, absolute joint-density
display cutoff, heavy-density division floor, fixed q_split, bond/site choice.
Lines 29–31: `conditional_density`, rho_pR/rho_R, undefined where rho_R is
below the floor; no artificial denominator or zero-filling of nodes.
Lines 34–50: `time_rate`, same nonuniform three-frame differentiation as
`tdse_report._add_time_derivative_in_place`, first-order endpoint secants,
fs converted to atomic time. A stride-two probe measures saved-time sensitivity.
Lines 53–117: `frame_terms`, all eight raw complex actions, expanded quad/lin
sums, independent existing-operator action, currents and density sources.
Lines 120–129: `summarize_frame`, joint-density weighted RMS and max-abs on
one common finite occupied mask; support probability is reported too.
Lines 132–147: `peak_integrals`, full q-side integration before display masking,
conditional left/right populations, time derivative, three sources and residual.

`proton_heavy_report.py` lines 48–75: `component`, `category`, `title` select
signed Re/Im data, comparable units/scales, mathematical labels.
Line 77 onward: `render_proton_heavy` orchestrates full-grid diagnostics,
snapshot NPZ, fixed-scale panels/movies, peak sources, and time summaries.
Its nested density/values helpers reconstruct positive factors with a one-frame
cache; build/update reuse the existing figure/movie saving and density contours.

`render_final_visualizations.py` lines 3709–3711, 3742, 3776–3777,
3857–3859: optional product, minimal a/b/alpha loading, rendering hook.
Lines 4259–4290: CLI configuration. Default old products remain unchanged.

Reuse: `core.derivative` (independent periodic central5 D1/D2),
`core.covariant_square` (Hermitian anticommutator), `core.AU_PER_FS`,
`tdse_report.load_observables/calculate_observables/_load_ef_fields`,
existing snapshot/movie frame selection, `_save_figure`, `_save_individual_frames`,
`_save_analysis_movie`, `_absolute_overlay`, report colormaps and MathText ticks.
The old six-action magnitude maps cannot recover T1..T8: taking absolute values
already discarded signs/phases; their decimated grid is also unsuitable for
new spatial derivatives. The original rho and cached connections are reused.

## What is computed

Axes are (q,R), a,b have that shape, alpha/chi have shape (R,).
Alpha is broadcast identically over q at each R. Cached forward-bond a,b,alpha
are averaged with their backward neighbor to sites (the historical diagnostic
convention). All derivatives occur BEFORE display masking/downsampling.

T1..T8 exactly follow the requested expanded formulas, in units Ha*a0^(-1/2).
They are U acting on Lambda, NOT U Lambda/Lambda. With positive real factors,
T1..T4 are real, T5..T8 imaginary (their real parts are zero, not missing data).
Real T terms affect phase directly, not the instantaneous density source
2 Im(Lambda* U Lambda); that distinction matters when judging b-alpha.

Expanded U_quad=T1+T3+T5+T6; U_lin=T2+T4+T7+T8; U_total is their complex sum.
Existing-operator U_quad uses core.covariant_square/(2M), independently.
Existing-operator U_lin=(alpha-i d_Rchi/chi)(-i d_R+delta)Lambda/M.

S_q, S_adv, S_rel have units a0^-1 / atomic_time.
S_U denotes S_adv+S_rel. S_U_operator separately means 2 Im(Lambda* U_operator).
Three distinct non-tautological residuals are retained:

- residual_density = saved-time d_t|Lambda|² - S_q - S_adv - S_rel;
- residual_source_operator = 2 Im(Lambda* U_operator) - S_adv - S_rel;
- residual_product_rule = U_operator - sum(T1..T8).

Also save residual_source_expanded and the algebraic component-closure check.
No residual is added to a physical term to force an identity.

## Discrete consistency / limits

For finite differences D(delta Lambda) != (D delta)Lambda+delta D Lambda.
The exact discrepancy of the expanded and anticommutator quadratic actions is
-i/(2M)[D(delta Lambda)-(D delta)Lambda-delta D Lambda].
The independent D2 stencil is reused, NOT replaced with D1 composed twice.
The temporal rule matches the existing report but is not an instantaneous H Psi
derivative. Endpoint source estimates can be biased, even at an initially real
stationary-density instant; they are flagged in the JSON.

These are the SAME operators as the old coupling diagnostics, NOT the native
spectral or finite-overlap-link TDSE operator. Spectral data plus a bond EF
cache do not provide coherent full-Psi spectral current derivatives. A cache
computed from truncated BO coefficients may also differ from full-grid density.
Do not call these maps exact discrete/spectral source closure. Reliable native
closure would require matched full-Psi currents and instantaneous density rates,
or a separate native-link derivation; a smaller saved time step alone is not a fix.
Phase-branch warnings are counted without silently unwrapping the connections.

## Products and style

Seven figure/movie groups prefixed `proton_heavy_`:
state, density, real_terms, imag_terms, sums, relative, closure.
State includes rho_R and alpha lines. All maps use a common support and fixed
time-global color bounds by physical category; residuals have separate scales.
Positive quantities (including T3) are sequential, signed quantities diverging.
Default fixed bound is the maximum over frames of the 99.5% occupied-site
absolute-value quantile. Colorbar extensions/footer disclose saturation.
Use `--ph-color-quantile 1` for full extrema. No physical energy shift is applied.
Existing absolute density contours are reused (default cutoff 1e-3).

`proton_heavy_expanded_time_summary.png`: source, eight action and closure RMS,
plus b/alpha/delta RMS. Legends are outside the data axes.
`proton_heavy_peak_sources.png/.npz`: P_right(R,t), dP_right/dt, source integrals,
residual, P_left and conditional norm. This is a fixed coordinate partition,
NOT automatic dynamical peak tracking. Lines only show occupied R columns;
NPZ preserves full integrals and undefined columns as NaN.
`proton_heavy_terms_frame_XXXX.npz`: complex fields at selected times, with mask,
q/R and stride metadata. Default map_stride=2 affects saved/display maps ONLY;
all derivatives, integration and RMS statistics use the original full grid.
`proton_heavy_terms_diagnostics.json`: every frame's metrics and conventions.

## Local 415-frame data check (20260916 spectral run)

All eight-term algebraic sums close to max3.47e-18 (action units).
Median/max occupied weighted RMS values:

| Diagnostic | Median | Maximum |
|---|---:|---:|
| density residual | 2.5900e-5 | 1.5466e-3 |
| operator-source residual | 7.3133e-8 | 3.3113e-6 |
| expanded-source residual | 1.0887e-7 | 9.6614e-6 |
| product-rule action residual | 3.1193e-8 | 2.6702e-5 |

Dimensionless closure ratio is RMS(residual_density) divided by the SUM of
RMS(density_dt), RMS(S_q), RMS(S_adv), RMS(S_rel). Interior-frame median is
0.296%; maximum35.23%. The largest absolute density residual is at82.9677fs
(ratio35.12%, 9 phase-branch warning sites). The stride-two temporal probe
there is9.15e-6, much smaller than1.55e-3: temporal sampling alone does not
explain this defect. At t=0 the one-sided derivative produces1.11e-4 despite
zero instantaneous a,b,alpha; do not interpret that endpoint defect physically.

Conclusion: algebraic decomposition works; the full trajectory is NOT certified
as an exact density-source identity. Late branch/cache/discretization problems
must not be interpreted as a new physical source. At50.0709fs, source RMS
Sq=4.904e-3, Sadv=1.292e-3, Srel=4.061e-4; relative transport is visible but
not the only term. Do not infer causality or global importance from this one time.

## Commands

```bash
python -m multi_component_exact_factorization.render_final_visualizations \
  results/20260916/erf_fullgrid_BOcurvature_compact_qR_analysisBO2_100fs \
  --only proton_heavy --format mp4 --max-frames 240 --fps 12 \
  --snapshot-count 6 --dpi 180 --animation-dpi 120 \
  --ph-density-floor 1e-3 --ph-heavy-floor 1e-12 --ph-q-split 0
```

Use --no-animation for snapshots only; --ph-groups density real_terms imag_terms
for fewer movies; --ph-map-stride 1 for full-resolution saved maps.
To append to ALL existing final figures, omit --only and add --with-proton-heavy.
No propagation source, VSC source, or old coupling report is modified.

Validation: 181 repository tests passed (175 existing plus6 new). The new
tests cover component closure, independent operator/product-rule residual,
fourth-order source convergence, nonuniform time units, density-node masking,
peak integrals, all figure families and a short GIF animation. The focused
six tests were rerun after final label changes and passed. Actual415-frame
diagnostics and0/50.0709/99.9997fs snapshots were generated under
`report/final_visualizations/proton_heavy_expanded/` for the local20260916 run.
