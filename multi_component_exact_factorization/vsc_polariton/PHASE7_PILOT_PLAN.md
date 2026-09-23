# Phase7 pilot predeclaration

Use only validated transferred initial/final full waves of free, resonant
F120/F160, barrier F120. Full interval/event analysis is NOT certified from
two times. Earlier Phase7 derivation's missing-data note is now superseded by
the verified 21-file pilot transfer; the equations/conventions are unchanged.

No physical parameters are introduced. Atomic units and Hamiltonian arrays
come from each hash-verified Phase6 packet. Work in blocks of R to avoid a
full resident (R,x,q) wave. Native R and x derivatives are Fourier derivatives.

Photon quadrature: Nq=max(32,Nf+16) and max(64,Nf+48), Gauss--Hermite nodes;
ladder derivatives are evaluated with two extra Hermite functions. Preserve
physical quadrature weights when forming conditional densities. Fock cutoff
refinement uses the independently propagated resonant F160 snapshot, not
zero-padding F120 dynamics.
For the resonant F120/F160 pair use the same nodes at Nq=176 and 208 for
both, so a direct common-grid field comparison introduces no interpolation.

Positive-marginal factors and their derivatives are obtained directly from
Psi, its native derivatives and dotPsi=-iH Psi. No floor-filled nodes. Nonzero
tails can be evaluated diagnostically, but displayed/compared potentials use
excluded-probability budgets 1e-6,1e-8,1e-10 and connected-component labels.

Pilot checks fixed BEFORE physical field calculation:

- transform/backprojection/reconstruction/PNC errors: 1e-10;
- normalized density/current and photon moment consistency: 1e-9;
- occupied-density weighted scalar Route A/B RMS: 1e-6 Ha;
- imaginary scalar RMS: 1e-6 Ha;
- scalar q-grid/Fock comparison RMS on common support: 1e-6 Ha;
- gauge/current consistency using numerical support-local phase gradients:
  normalized weighted RMS 1e-3 (not algebraically assigned zero);
- force q/Fock/support comparison RMS: 1e-5 Ha/a0.

Unmet criteria stay FAIL/not-certified; do not loosen them after evaluation.
Algebraic decomposition closure is NOT an independent route test. Compare
the first scalar against marginal Xi inversion and the conditional electronic
EOM action residual; compare the outer scalar against outer inversion and
direct combined electron-photon conditional expectation. Report finite-Fock
projection defects and imaginary residuals separately.

The natural-gauge outer force is -d_R epsilon2 + dot alpha. It is gauge
invariant and is NOT the Bohmian total acceleration. Alpha=0 phase integration
must be validated component by component; never join disconnected support.

Limitations requiring subsequent inputs/work before full Phase7 PASS:
all-time/event frames, independent propagated R-grid refinements, temporal
probe refinement, Phase4 matched force comparison and full historical MCEF
field compatibility. No new TDSE propagation is performed by this pilot.
