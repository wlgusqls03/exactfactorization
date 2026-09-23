# Reaction movie and polariton-character comparison

These are **Phase6 observable visualizations and Phase2 static eigenstate
analysis**, not a certified Phase7 TDPES movie. Phase7 convergence gates
remain unchanged and NOT PASS.

## Implemented code

`phase7_reaction_media.py lines 31–36`: intersect actual saved atomic-unit
times, no temporal interpolation. Fewer than three common times is an error.

`phase7_reaction_media.py lines 39–48`: identify first signed local flux
extrema above 1% of that sign's global peak, maximum product, endpoints.
This is a declared event-selection threshold, not a physical parameter.

`phase7_reaction_media.py lines 51–75`: load `(time,R)` densities/current and
`(time,)` observables; verify SHA packet association, finite values, monotone
time, norm and product integrals. Reuses Phase7 `bare_action` and packet BO
ground states to draw the static bare PES. No new dynamics or model fitting.

`phase7_reaction_media.py lines 78–101`: cached Phase2 eigenvalues and photon/
vibrational transition doorway strengths. Each channel is normalized within
the original vibrational energy window, not converted into Hopfield weights.
Markers have horizontal offsets +/-0.8 meV solely for readability; physical
pair energies and splitting remain unchanged in the manifest. The two
uncoupled transitions have anharmonic detuning, not a Rabi splitting.

`phase7_reaction_media.py lines 104–185`: fixed-scale 1600x900 movie with
three nuclear densities, product population, signed flux, photon occupation.
Snapshots use the same plots/times. All data are actual saved observables;
no classical point trajectory or invented interpolated TDPES is drawn.

`phase7_wave_inventory.py lines 22–76`: read-only server wave inventory and
optional event-frame archive. No propagation/delete/overwrite. Selects a
saved wave within 0.5 fs of each event; reports actual time deviation.
Initial/final frames already supplied are not duplicated. Files too far
from the event are reported missing, not silently substituted.

Units: R a0, density a0^-1, current atomic inverse time, energy eV in the
explicitly referenced static PES overlay; time au converted by
0.024188843265857 fs/au. Photon occupation/product probability dimensionless.
The gray curve is **bare PES minus its bare well minimum**, NOT a TDPES.
Its scale is separate from the physical density scale; height of density
relative to that curve is not a statement about wavepacket energy.

## Actual output and numerical observations

`results/vsc_polariton/phase7/reaction_media_v1/reaction_comparison.mp4`:
414 genuine matched saved times, 24 fps, 17.25 seconds, 1600x900 H.264.
Physical time is 0–39.959969 fs. All panels use the same clock.
Video is about 4.48 MB; initial complete media folder about 7.9 MiB.
Event snapshots are PNG/PDF; the corrected readable spectral figure is in
`spectral_readability_v2/` (original plot preserved).

| Full3D case | Maximum product | Integrated forward flux | Integrated backward flux |
|---|---:|---:|---:|
| Free | .4684110 | .4701716 | .3466825 |
| 170.6 meV | .0651338 | .0749520 | .0186097 |
| 161.768922 meV | .0660982 | .0758348 | .0148940 |

The movie shows density reaching R>0 and later moving back; these are
population/flux, NOT reaction rates or irreversible yield. Signed net flux
integrals are not individually resolved counts of crossing trajectories.
They also must not be equated to maximum product population.

The static spectrum shows transitions with both photon and vibration
character in the coupled case. This does NOT mean the propagated threshold
wavepacket is one LP/UP eigenstate. Photon occupation is not polariton
population. Both 170.6 and 161.77 meV are near the well frequency; their
comparison is not a far-off-resonant control.

Scientific reference for this separation of static polariton splitting and
dynamical modification: Li–Mandal–Huo, main Figs.1–3 / Eqs.1–2,
https://www.nature.com/articles/s41467-021-21610-9 . Our closed pure-state
data are not the paper's thermal Langevin-rate ensemble.

## What is needed next — do not start another large propagation yet

First inspect/export existing intermediate waves from the server:

```bash
cd /home/hbji/exactfactorization
git pull
OPENBLAS_NUM_THREADS=1 python -m \
  multi_component_exact_factorization.vsc_polariton.phase7_wave_inventory \
  --pack-events
```

Send `results/vsc_polariton/phase7/wave_inventory_v1/inventory.json` and,
if created, `phase7_event_waves.tar.gz` from the same directory. Original
waves stay untouched. The JSON records archive input size and all available
wave times. If output exists, choose a new `--out` name; never erase it just
to rerun. Missing directories are explicit; do not substitute old failed runs.

Event waves allow testing MCEF in the actually occupied crossing region.
The current two endpoint waves alone cannot supply that analysis. Once
gauge/force and Fock-potential convergence are settled, use the full existing
regular wave series for a synchronized TDPES/density/force/flux movie.
Only if snapshots are absent, or a separately identified numerical setting
fails, request targeted server propagation. Do not simply enlarge Fock
cutoff everywhere: F120→F160 did not monotonically remove the current defect.

An 80-meV full3D case would be a meaningful later off-resonance control,
but only after core Phase7 validation; it is not required to render the
already validated density movie and has not been started here.

## Redraw locally from the same data

```bash
OPENBLAS_NUM_THREADS=1 python -m \
  multi_component_exact_factorization.vsc_polariton.phase7_reaction_media \
  --out results/vsc_polariton/phase7/reaction_media_v2
```

Default creates PNG/PDF and MP4. `--no-movie` explicitly disables video.
Local imported data paths are defaults; `--free-dir`, `--resonant-dir`,
`--barrier-dir` override the observable folders. The local pilot input
manifest/packets and Phase2 cached spectrum must also be present.

Validation: 14 Phase7 pilot/media tests PASS, including temporal alignment,
no fabricated recrossing, and archive preservation. Density/product/norm
cross-checks passed for all three 414-frame series. ffprobe independently
confirmed actual video frame count, dimensions, codec and duration.
Existing MCEF and historical Phase1–6 files were not edited.
