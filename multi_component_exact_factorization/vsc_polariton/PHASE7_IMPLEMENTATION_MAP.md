# Phase7 compatibility audit (read-only historical code)

Inspected at HEAD 4d602b3. No historical implementation is modified.
Production postprocessing is blocked pending validated full wave snapshots.

| Existing location | Meaning | Reuse decision |
|---|---|---|
|core.py 48–86, Model|old x,q,R model and uniform-grid metrics|Do not construct photon physics through this proton model|
|core.py 687–712, derivative|historical central finite differences|Regression comparator only, not native FFT/Fock derivative|
|core.py 723–725, momentum|p=-i derivative|Match sign convention|
|core.py 1106–1138, pnc_project|renormalization of nested factors|Do not silently repair PNC in production postprocessing|
|core.py 1141–1143, reconstruct_psi|Phi*Lambda*chi, arrays (x,q,R),(q,R),(R)|Generic multiplication reusable for compatibility tests|
|core.py 1434–1453, covariant_square|Hermitian anticommutator discretization|Reference sign/ordering; do not replace spectral H by this stencil|
|core.py 1456–1469, geometric_fields|a=<Phi|p_q|Phi>, b=<Phi|p_R|Phi>, alpha=<Lambda|(p_R+b)|Lambda>|Match definitions; use independently converged photon derivatives/weights|
|core.py 1518–1763, instantaneous_functionals|proton-specific operators and parallel-transport scalar convention|Not directly reusable for photon positive-marginal scalar potentials|
|core.py 1766–1781, nested_factorize|positive marginals plus floors and pnc_project|Reference only; probability-budget/node handling must be explicit|
|propagate.py 150–204, coupled_rhs|historical coupled MCEF EOM|Read-only comparison; never propagate photon model with this function|
|propagate.py 227–236, full_step|historical coupled integrator|Existing API smoke only|
|propagate.py 295–555, run|historical research CLI|No changes or reuse as photon driver|
|reference.py 40–197, run|full TDSE and imported core factorization helpers|Read-only smoke, not a photon Hamiltonian|

Phase6 full wave packets are (R,x,n). A future adapter must explicitly
transpose reconstructed coordinate data to (x,q_c,R) if using the generic
historical reconstruction helper. Photon quadrature is not necessarily dq
times an unweighted sum when Gauss–Hermite nodes are used.

Historical snapshot regression requires an actual historical wave plus fields
in the same gauge and derivative convention. Native finite-difference and
spectral scalar fields must not be asserted equal at fixed finite resolution.
No compatibility PASS is claimed yet.
