# Full photon nested EF: continuum equations and finite-basis audit plan

Atomic units, photon mass 1, nuclear mass M from the validated input. No new
physical parameter. Starting point is Li main Eq.(1) extended with explicit
electron and full operator dipole R-x, as already used in Phase6.

Define p_q=-i partial_q, p_R=-i partial_R and

    h = T_x + V_eN + V_NN + omega_c^2 q_c^2/2
        + sqrt(2 omega_c) g_chi q_c (R-x) + g_chi^2 (R-x)^2/omega_c.
    H = h + p_q^2/2 + p_R^2/(2M).

Here the photon harmonic potential is included in h as an x-independent
scalar. Thus epsilon^(1) includes it; it must NOT be added a second time in
the middle equation. No mu0^2 substitution in the full DSE.

## First factorization, derived by the kinetic product rule

Let Xi=Lambda_R chi and Psi=Phi Xi, with integral dx |Phi|^2=1.
For s=q,R with masses m_q=1,m_R=M, define A_q=a,A_R=b:

    A_s = <Phi|p_s Phi>_x,
    D_s = p_s-A_s,
    U_e,pn Phi = sum_s [D_s^2/(2 m_s)
                      + ((p_s Xi)/Xi + A_s) D_s/m_s] Phi.

This follows by expanding p_s^2(Phi Xi), collecting p_s^2 Xi,
2(p_s Xi)(p_s Phi) and Xi p_s^2 Phi and completing the covariant square.
The derivative in D_s^2 acts on both A_s and Phi.

    (h + U_e,pn - epsilon^(1)) Phi = i partial_t Phi,
    [(p_q+a)^2/2 + (p_R+b)^2/(2M) + epsilon^(1)] Xi
       = i partial_t Xi.

The scalar definition is

    epsilon^(1) = <Phi|h+U_e,pn-i partial_t|Phi>_x.

PNC gives <Phi|D_s Phi>=0 and hence the continuum decomposition

    epsilon^(1) = <h>_x
       + (||partial_q Phi||_x^2-a^2)/2
       + (||partial_R Phi||_x^2-b^2)/(2M)
       + <Phi|-i partial_t Phi>_x.

Separate <T_x+V_eN+V_NN>, harmonic, LM, DSE in saved components. BO projections
of the bare electronic part are diagnostics, not an assumed complete
three-state expansion.

## Second factorization

With integral dq |Lambda_R|^2=1,

    alpha = <Lambda_R|(p_R+b)|Lambda_R>_q,
    D_R = p_R+b-alpha,
    U_p,n Lambda_R = [D_R^2/(2M)
                     + ((p_R chi)/chi+alpha) D_R/M] Lambda_R.

Expanding (p_R+b)^2(Lambda_R chi) yields

    [(p_q+a)^2/2 + epsilon^(1) + U_p,n - epsilon^(2)] Lambda_R
       = i partial_t Lambda_R,
    [(p_R+alpha)^2/(2M) + epsilon^(2)] chi = i partial_t chi,

    epsilon^(2) = <Lambda_R| (p_q+a)^2/2 + epsilon^(1)
                               + U_p,n -i partial_t |Lambda_R>_q.

The derivative b enters the ordered covariant square, not just b^2. The
expectation of the linear D_R term vanishes by the definition of alpha.

## Independent routes and derivatives

Route A uses expectation values above, with time derivatives obtained from
dotPsi=-i H Psi and differentiated marginal densities/factor quotients.
Never derive dotPhi from the very conditional EOM being checked.

Route B checks conditional electronic EOM actions directly, and independently
computes the first scalar from the Xi equation. The outer inversion is

    epsilon^(2)_inv = [i dotchi-(p_R+alpha)^2 chi/(2M)]/chi.

All inversion comparisons retain imaginary residuals. Defining GD as a
residual does not by itself validate an independent scalar identity; directly
evaluate -i<Phi|dotPhi> and -i<Lambda|dotLambda> as well.

Finite FFT/Fock truncation does NOT obey an exact continuum product rule on
arbitrary factor quotients. Thus finite-resolution route differences and
projection/aliasing defects must be saved and converged, never hidden by
defining total as the sum or by taking only real parts.

## Gauges and force

For real theta(q,R,t), eta(R,t),

    Phi' = exp(i theta) Phi,
    Lambda' = exp(-i theta+i eta) Lambda,
    chi' = exp(-i eta) chi.
    a'=a+partial_q theta, b'=b+partial_R theta,
    alpha'=alpha+partial_R eta,
    epsilon1'=epsilon1+partial_t theta,
    epsilon2'=epsilon2+partial_t eta.

These signs agree with the historical code's p=-i partial convention.
Positive chi and Lambda define the initial numerical gauge. On each connected
outer support choose partial_R eta=-alpha for alpha'=0. Never bridge nodes
with an arbitrary phase. Time-dependent component anchors must be tracked.

    F_outer = -partial_R epsilon2 + partial_t alpha

is invariant under this transformation, reducing to -partial_R epsilon2' in
alpha'=0 gauge. It excludes the Madelung quantum-potential acceleration.
The first-level curl partial_q b-partial_R a is gauge invariant; nonzero curl
prevents setting a=b=0 simultaneously.

## Representation and support

Reconstruct Psi(x,q,R)=sum_n Psi_n(x,R) varphi_n(q;omega). Gauss–Hermite
quadrature with enough nodes exactly integrates finite-basis norm and
back-projection in exact arithmetic; derivative and multiplication require
extra headroom (including n=N, N+1 contributions), not a blindly truncated
ladder product. Conditional factors are not finite Hermite polynomials, so
their derivatives need independent q-resolution convergence.

Support budgets 1e-6,1e-8,1e-10 must be tested separately for joint and outer
marginals, with disconnected components explicit. No epsilon-filled nodes,
outside-support smoothing, or extrapolated potential.

Phase4/full3D comparisons use force first; scalar differences require an
explicit common-support alignment. Historical phase4 densities/currents need
not equal full3D densities/currents: that difference is the research question.
Reconstruction consistency is tested against each run's OWN TDSE data.

No production Phase7 fields or PASS are asserted by this derivation. Validated
full-wave snapshots are currently absent locally; failed pilot waves must
not be substituted.
