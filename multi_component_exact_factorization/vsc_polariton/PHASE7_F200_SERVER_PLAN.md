# F200 matched-event refinement — 20 GB disk plan

This is a predeclared larger-cutoff diagnostic, not a promise of Phase7 PASS.
Physical cases: resonant170.6meV and barrier161.76892211069242meV, eta0.094.
Unchanged x/R grids352x160, x halfbox24, R halfbox4.4, dt0.125au,
end1652au=39.9599690752fs, complex128/float64. Only Fock cutoff increases
from160 to200. Initial states are regenerated from the original threshold
preparation and compared with the F160 common coefficients; not snapshot padding.
Original Phase6 source and all historical outputs remain unchanged.

## Gates

Historical CPU action/step/observables adapter must pass on export; Fock
rotation uses the verified evd orthogonal construction. On the server each
packet must pass a fresh128-step CPU/GPU equivalence check before propagation.
All original norm1e-9, energy1e-6Ha, boundary/tail1e-8 and continuity2e-6
limits remain. A failed check or run stops the serial batch. Completed runs
are skipped on restart; interrupted runs resume. Failed runs remain immutable.
Neither completion marker nor CPU/GPU equivalence means scientific Phase7 PASS.

## Storage-only change

Observables/checkpoints remain every4au. Full waves at every64au (~1.548fs),
extra times160/608/672/1408/1440au, initial and final:31 waves per case.
The common event union is retained for both cases. Failed off-cadence or
regular-cadence diagnostic waves are retained. No old scientific file is deleted.
Phase7 derivatives use H Psi, not a coarse saved-frame time difference alone.

Two runs'31 waves plus restarts:10.742GiB. Driver requires13.742GiB total
reserve, including temporary restart, observables/logs and returned archive.
Allow input tar plus extracted packets additionally.20GB decimal is18.63GiB,
so this fits an actually available20GB, subject to other disk users. Disk is
checked before starting. Full original32au cadence would not leave safe margin.
The compact times are adequate for matched-event convergence, not a claim of
uniform sub-fs movie sampling at F200.

Past RTX2080Ti F160 check estimates were1.202/1.208h per full run excluding I/O.
F200 operator cost scales approximately between N and N²; estimated GPU run
time1.5–1.9h per case plus CPU/GPU checks and I/O. Budget4–6h for both serial
runs; new check JSON gives a machine-measured estimate. Not a guaranteed time.

## Transfer / command

Local export: `python -m multi_component_exact_factorization.vsc_polariton.phase7_prepare_f200`
(already executed by assistant; do not rebuild on server).
Transfer results/vsc_polariton/phase6_gpu/phase7_f200_server_bundle_v1.tar.gz.
Extract into a NEW results/vsc_polariton/phase6_gpu/f200_server_transfer_v1
directory. Update Git source separately; result bundles are not in Git.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1:
`python -m multi_component_exact_factorization.vsc_polariton.run_phase7_f200_campaign --device 0`

Default output: results/vsc_polariton/phase6_gpu/phase7_f200_campaign_v1.
On success the driver automatically creates phase7_f200_results.tar.gz there,
containing all observables/logs/gates plus eight selected/final wave snapshots
and SHA256 manifest. Send this archive back. Keep other waves and restart files.

## Optional cleanup

`python -m multi_component_exact_factorization.vsc_polariton.phase7_cleanup_verified_transfers --delete`
removes only archives named phase7_f160_events.tar.gz or phase7_event_waves.tar.gz
whose SHA256 matches the already locally received/audited copy. Default without
--delete is dry-run. Other archives, inputs, wave/restart/status files and all
directories are untouched. Expect about1.7GiB for one copy of each, not tens of
GB. Deletion is not trash/recoverable on server, but a verified local copy exists.
No cleanup was executed locally by the assistant.

## Review requirements after run

Compare160/200 at identical event times, both q quadratures and all budgets.
Require scalar routes/imaginary residual and scalar convergence at existing
1e-6Ha target, force convergence1e-5Ha/a0; retain native R/time refinement
checks independently. Failure is reported, not repaired by altering masks.
No rerun of free control, no physical parameter tuning, no Phase7 PASS shortcut.
