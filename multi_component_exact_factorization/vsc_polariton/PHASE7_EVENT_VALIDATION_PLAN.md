# Step 1–2 event audit: predeclared postprocessing campaign

No TDSE propagation, physical change, or Phase1–6 edits. Preserve all old
pilot failures. Inputs: nine hash-verified event waves and existing packets.
New outputs: results/vsc_polariton/phase7/event_validation_v1/.

Step 1 reuses native spectral wave derivatives, instantaneous H Psi, nested
expectation/inversion, two photon quadratures, and budgets 1e-6/1e-8/1e-10.
Photon nodes: free 32/64, coupled 176/208 (F120 in both coupled packets).
Report errors on all support, R<0, barrier strip |R|<=0.5 a0, and R>0.
The barrier strip is an analysis window, not a changed physical parameter.
Existing scalar RMS limit 1e-6 Ha; force limit 1e-5 Ha/a0; normalized
gauge/current limit 1e-3. No tolerance changes.

Step 2 isolates nonlinear postprocessing resolution from TDSE resolution.
Predeclared R sampling factors: 2,4,8,16,32,64,128,256. Use the exact same
Fourier interpolant, preserving signed native Nyquist modes. Form density,
momentum numerator and their time derivatives at 2N points before taking
products (dealiasing); then evaluate the resulting finite Fourier series.
No potential interpolation, new propagation, or smoothing of fields.
Cache one-dimensional bilinears in channel blocks to bound memory.

Construct component-local eta_R=-alpha and eta_t,R=-alpha_t; differentiate
the transformed scalar independently with fourth/sixth order formulas.
At the final two resolutions also differentiate actual transformed
conditional wavefunctions using exact Fourier-translated overlap kernels;
do not assign the transformed connection zero by algebra alone.
Compare current invariance, reconstruction, and both force representations.
Disconnected components remain independent; no phase bridge. Support-end
stencil exclusions are reported. Tiny negative Fourier density roundoff may
be clipped ONLY for selecting support, with its negative probability logged;
ratios use original positive density, never an amplitude floor.

Independent tests: known Fourier modes/Nyquist, direct-wave bilinear and
shift-overlap equality, analytic gauge examples, preservation of time/grid
input. Same-frame interpolation is not a propagated-grid convergence test.
Temporal probe tests and Fock/R native refinement remain later gates even
if Step 2 postprocessing passes. Report actual PASS/FAIL per subtest.

## Separate resolution extension (declared before its calculations)

The first fixed ladder reached factor256 and failed at the two late free
events. Preserve that entire run. A separate extension will apply ONLY to
failed Step2 frames: factors512,1024,2048 in this order, all three budgets.
Check independent transformed-wave connections/current at1024 and2048.
Require both highest resolutions to pass the original criteria; stop at2048
and report remaining failures without relaxing tolerances. Reuse bilinear
caches and original waves; this remains postprocessing, not refined TDSE.
Additionally revisit the three previously failing final snapshots at1652au:
128/256 first, and1024/2048 only if that pair fails. This endpoint check is
separate from the nine-event campaign and preserves all earlier failures.
