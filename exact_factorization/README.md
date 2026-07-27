# Direct Exact Factorization for the 1D Shin--Metiu model

This directory is independent of the original `shin_metiu_1d.py` workflow.
There is no surface hopping: the nuclear wavefunction and the conditional
electronic state are propagated continuously as coupled quantum fields.

Two direct representations are available:

```text
ef_propagate.py             finite BO-basis representation of Phi
ef_propagate_realspace.py   pure real-space Phi; no BO calculation at all
```

The direct solver uses

```text
Phi_R(r,t) = sum_s C_s(R,t) phi_s^BO(r;R)
```

and propagates `chi(R,t)` and every `C_s(R,t)` self-consistently. The finite
BO basis is a numerical representation of the conditional electronic field,
not a hopping prescription. Increase `--n-states` to test basis convergence.

## Array shape convention

The coordinate/state dimensions are abbreviated as follows:

```text
nr = number of electronic r-grid points
nR = number of nuclear R-grid points
ns = number of retained BO states
nt = number of saved time frames
```

The main saved quantities are:

| NPZ key | Shape | Physical meaning |
|---|---:|---|
| `r` | `(nr,)` | electronic grid |
| `R` | `(nR,)` | nuclear grid |
| `bo_energies` | `(ns,nR)` | BO surfaces `E_s(R)` |
| `bo_states` | `(ns,nR,nr)` | fixed-`R` BO wavefunctions |
| `chi` | `(nt,nR)` | nuclear wavefunction |
| `coefficients` | `(nt,ns,nR)` | conditional-electronic BO coefficients |
| `A` | `(nt,nR)` | EF vector/Berry potential |
| `epsilon` | `(nt,nR)` | exact time-dependent scalar potential |
| `u_coefficients` | `(nt,ns,nR)` | BO projection of `U_en Phi` |
| `phase_S` | `(nt,nR)` | unwrapped phase of `chi` |
| `populations` | `(nt,ns)` | integrated BO populations |
| `norm`, `pnc_error` | `(nt,)` | propagation diagnostics |
| `phi` | `(nt,nr,nR)` | conditional electronic field; `--save-fields` |
| `psi` | `(nt,nr,nR)` | reconstructed molecular field; `--save-fields` |
| `u_phi` | `(nt,nr,nR)` | differential action `U_en Phi`; `--save-fields` |

The reference archive always stores full `psi`. With `--compact`, its
reconstructible `phi` and `u_phi` arrays are omitted to save disk space.

## Pure real-space EF with no BO calculation

In this workflow the only propagated independent fields are

```text
Phi_R(r,t)   (nr,nR)
chi(R,t)     (nR,)
```

`H_BO` is applied directly as `-d_r^2/2 + V(r,R)` using an FFT kinetic step
and pointwise potential multiplication. No eigenvalue problem, BO surface,
BO coefficient, BO projection, or surface hop is used. The initial `Phi` is a
normalized conditional Gaussian.

Run a matched-grid reference and direct calculation with:

```bash
/home/jubjhbjey5/anaconda3/bin/python -m exact_factorization.ef_reference_realspace \
  --nr 64 --nR 96 --dt-au 0.01 --t-final-fs 0.1 --save-every 20 \
  --compact --outdir results/exact_factorization/realspace_reference

/home/jubjhbjey5/anaconda3/bin/python -m exact_factorization.ef_propagate_realspace \
  --nr 64 --nR 96 --dt-au 0.01 --t-final-fs 0.1 --save-every 20 \
  --boundary periodic --outdir results/exact_factorization/realspace_direct

/home/jubjhbjey5/anaconda3/bin/python -m exact_factorization.ef_compare \
  results/exact_factorization/realspace_reference/shin_metiu_ef_reference_realspace.npz \
  results/exact_factorization/realspace_direct/shin_metiu_direct_ef_realspace.npz
```

The direct real-space archive always stores `phi(nt,nr,nR)`, `chi(nt,nR)`,
`A(nt,nR)`, and `epsilon(nt,nR)`. `psi` and `u_phi`, both
`(nt,nr,nR)`, are optional because they can be large; request them with
`--save-psi` and `--save-u-phi`.

For the comparison above, BO population error is intentionally reported as
unavailable. Computing it would require constructing a BO basis and would
violate the purpose of this pure real-space path.

## Short matched-grid smoke calculation

The reference uses a periodic FFT grid, so use the same boundary and grids in
the direct calculation when comparing point-by-point:

```bash
/home/jubjhbjey5/anaconda3/bin/python -m exact_factorization.ef_reference \
  --nr 64 --nR 64 --dt-au 0.05 --t-final-fs 0.05 --save-every 2 \
  --outdir results/exact_factorization/reference_smoke

/home/jubjhbjey5/anaconda3/bin/python -m exact_factorization.ef_propagate \
  --nr 64 --nR 64 --dt-au 0.05 --t-final-fs 0.05 --save-every 2 \
  --boundary periodic --save-fields \
  --outdir results/exact_factorization/direct_smoke

/home/jubjhbjey5/anaconda3/bin/python -m exact_factorization.ef_compare \
  results/exact_factorization/reference_smoke/shin_metiu_ef_reference.npz \
  results/exact_factorization/direct_smoke/shin_metiu_direct_ef.npz
```

## Important numerical controls

- `--r-min`, `--r-max`, `--nr`: electronic grid used to construct the BO basis.
- `--R-min`, `--R-max`, `--nR`: nuclear grid for both `chi` and `C_s`.
- `--dt-au`, `--t-final-fs`: time integration controls.
- `--boundary`: `dirichlet` for isolated direct runs or `periodic` for a
  pointwise comparison with the FFT reference.
- `--derivative-scheme`: defaults to finite differences. This should normally
  remain finite-difference even on a periodic comparison grid because the
  conditional electronic field is not periodic in its parameter `R`.
- `--density-threshold`: smooth regularization of `(d_R chi)/chi` in the
  physically undefined low-density tails.
- `--save-fields`: additionally store reconstructed `Phi`, `Psi`, and
  `U_en Phi`; without it these are reconstructible from the saved BO basis,
  coefficients, and `chi`.

The direct equations are known to have unusual numerical instabilities.
Passing a short smoke test is not evidence of convergence at the avoided
crossing. Increase time, `nR`, the BO basis size, and reduce/scan the time step
in stages while monitoring nuclear norm, partial normalization, fidelity, and
boundary density.
