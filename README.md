# 1D Shin-Metiu Quantum Dynamics

This repository contains a compact implementation of the 1D Shin-Metiu
nonadiabatic charge-transfer model used in Agostini et al., J. Chem. Phys.
142, 084303 (2015).

The script propagates the full electron-nuclear wavefunction

```text
Psi(r, R, t)
```

with a split-operator FFT method, computes instantaneous Born-Oppenheimer
electronic states, and projects the wavefunction as

```text
Psi(r, R, t) = sum_l F_l(R, t) phi_l(r; R)
P_l(t) = integral dR |F_l(R, t)|^2
```

The default physical parameters match the paper:

```text
L = 19.0 a0
M = 1836
Rf = 5.0 a0
Rl = 3.1 a0
Rr = 4.0 a0
initial Gaussian center R0 = -4.0 a0
initial BO state = state 2
sigma = 1 / sqrt(2.85) a0
```

## Run

Use the Anaconda Python already available on this machine:

```bash
/home/jubjhbjey5/anaconda3/bin/python shin_metiu_1d.py \
  --outdir results/shin_metiu_demo_R8 \
  --nr 192 --nR 192 \
  --t-final-fs 35 \
  --dt-au 0.25 \
  --save-every 50
```

For a closer paper-style time step, use `--dt-au 0.1` and increase the grid,
for example `--nr 384 --nR 384`, if runtime is acceptable.

## Outputs

The script writes:

```text
shin_metiu_1d_results.npz   raw arrays: grids, BO populations, densities
shin_metiu_summary.png      time-R density maps and population curves
shin_metiu_wavepacket.gif   animation of wavepacket splitting and populations
```

In the animation, the left panel shows state-resolved projected nuclear
density overlaid on the first two BO surfaces. The right panel uses time as
the x-axis and tracks the BO populations.

The polished presentation outputs are in `results/shin_metiu_polished/`:

```text
shin_metiu_summary.png              styled time-R maps and diagnostics
shin_metiu_summary.pdf              vector-friendly static summary
shin_metiu_wavepacket.gif           three-panel animation
shin_metiu_wavepacket_final.png     final animation frame
```

To regenerate only the plots and animation from an existing calculation:

```bash
/home/jubjhbjey5/anaconda3/bin/python shin_metiu_1d.py \
  --render-from results/shin_metiu_demo_R8/shin_metiu_1d_results.npz \
  --outdir results/shin_metiu_polished
```

The Korean theory and line-by-line code guide is a 41-page chapter-based
edition with physical intuition, equation explanations, analogies, and
end-of-chapter summaries:

```text
docs/shin_metiu_guide_ko.pdf
```

## Direct exact-factorization prototype

The original full-TDSE/BO-projection workflow above is unchanged. A separate
research implementation under `exact_factorization/` provides:

```text
ef_reference.py   full-TDSE Psi and A=0-gauge EF reference extraction
ef_propagate.py   self-consistent propagation of chi and conditional Phi
ef_compare.py     wavefunction fidelity, density, and population comparison
ef_reference_realspace.py  BO-free Gaussian/full-TDSE reference
ef_propagate_realspace.py  BO-free direct propagation of Phi(r,R) and chi(R)
```

The direct solver contains no surface hopping. See
`exact_factorization/README.md` for matched-grid smoke commands and numerical
limitations.

## Multi-component exact factorization

The separate `multi_component_exact_factorization/` package treats a 1D
electron, quantum proton, and heavy quantum nucleus with the nested product

```text
Psi(x,q,R,t) = Phi_{R,q}(x,t) Lambda_R(q,t) chi(R,t).
```

It initializes the electron in a local Born-Oppenheimer eigenstate, then uses
a pair of nuclear Gaussians whose widths come from independent q/R BO
curvatures, followed by a direct three-factor propagator without surface
hopping. The fixed left site is the electron's
Dirichlet hard-wall boundary. It also contains an independent full-3D-TDSE
reference, explicit two-level gauge functions,
gauge-invariant comparison tools, paper-style plots, wave/density/potential
animations, and local-electronic excited-state population analysis. A
standalone interactive 3D configuration-density HTML provides rotation and a
time slider. See
`multi_component_exact_factorization/README.md` for Korean documentation and
run commands.
