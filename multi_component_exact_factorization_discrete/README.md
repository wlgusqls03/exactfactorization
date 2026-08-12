# Discrete MCEF reference

This package implements the spatially discrete, time-continuous nested exact
factorization independently of the existing continuum-derived solver.  It
uses the same extended 1D Shin--Metiu Hamiltonian, grids and reusable
Born--Huang cache.

The NumPy code in `core.py` is an algebraic reference and test oracle.  The
production CUDA propagator lives in
`multi_component_exact_factorization_discrete_gpu`.

The defining implementation invariant is

```text
d(C Lambda chi)/dt == -i H_discrete (C Lambda chi)
```

up to the explicitly recorded flat-top mask residual.  No continuum spatial
product rule, logarithmic derivative, vector-potential mask, or product-rule
projection is used.

The production path also verifies the time discretization independently: at
saved steps it compares the actual full-product RK4 increment with the
quadrature of the four recombined factor tangents.  Native discrete geometry
is analyzed as link phase and magnitude; continuum-like `a,b,alpha` are
reconstructed only for output.
