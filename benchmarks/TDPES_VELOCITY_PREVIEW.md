# TDPES1 + velocity preview

This is a separate exploratory view, not a replacement for any existing
final-visualization product. It reads existing density, PG a/b, and stored
TDPES1 total arrays. Legacy trap-included scalars are converted with the
existing external-potential helper. No dynamics or EF fields are recomputed.

```bash
python -m multi_component_exact_factorization.preview_tdpes_velocity \
  results/20260909 --start-fs 20 --stop-fs 60 --frames 24 --fps 6
```

Default output: `RUN/report/tdpes_velocity_preview/`, with four PNG snapshots
and a 4-second MP4. Times are selected from existing saved frames, not
interpolated. Only selected frames are retained in memory; compressed NPZ
members still have to be decompressed sequentially up to the last time.

- Background: raw, trap-excluded PG TDPES1, fixed -0.1 to +0.1 Ha scale.
  Values beyond this range saturate in color, not in the stored data.
- Arrows: existing PG velocities (a/proton_mass, b/heavy_mass). They are not
  force vectors or trajectories. Their magnitude is not independently
  normalized at each point or each frame.
- Absolute joint-density support: rho >= 1e-3 a0^-2, same contour helper.
- The panel and colorbar have fixed positions and sizes in every frame.
  Panel width/height is taken from an individual panel in the existing
  six-panel TDPES1 origin layout (not the whole movie canvas).
  Only coordinate limits follow the occupied region. Explicit screen-space
  angles account for unequal coordinate scales, while arrow lengths remain
  proportional to physical hypot(vq,vR), not the stretched vector norm.
- The camera follows each frame's absolute occupied support with padding.
  Sparse fixed sampling sites (38 by 18 before masking) avoid a dense carpet
  of arrows. Green, white-edged arrows replace black arrows.
- Arrow length uses inches, not coordinate units: the reference speed has a
  0.21875-inch arrow even as the camera zooms. Screen length therefore remains
  proportional to speed with the same calibration throughout the movie.
  Shaft width is also fixed in inches (0.020625), independent of panel resizing.
- Sampling changed from 27 by 13 to 38 by 18: interval ratios are
  26/37 = 0.703 (q) and 12/17 = 0.706 (R), before grid rounding.
  Use `--reference-speed` to retain an earlier movie's arrow calibration;
  the latest full movie uses 0.009941261895187755 a0/tau.
- The reference arrow is the 95th percentile of supported sampled speeds in
  the preview interval, fixed for the entire clip. Larger arrows are allowed;
  95% is not a population fraction and not a speed cutoff.

Use `--bound` to choose another fixed symmetric energy scale. The normal
`render_final_visualizations` command is unchanged and does not run this
experimental preview automatically.

Full 0–100 fs movie, matching the 415-frame, 12-FPS report (34.58 seconds):

```bash
python -m multi_component_exact_factorization.preview_tdpes_velocity \
  results/20260909 --start-fs 0 --stop-fs 100 --frames 415 --fps 12 \
  --disk-backed --outdir results/20260909/report/tdpes_velocity_full
```

The disk-backed option keeps the four selected full-resolution arrays in a
temporary directory (about 7 GiB for this run), removed after rendering.
It does not change numeric precision or original archives. Four representative
PNGs and the MP4 are saved separately from the short preview.
