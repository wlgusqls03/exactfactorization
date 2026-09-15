"""Small, separate TDPES1 + PG velocity preview from existing saved arrays."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from . import render_final_visualizations as render
from .external_potential import separate_fields
from .report_plot_style import MASK_COLOR, SIGNED_CMAP
from .tdpes_velocity import Q_POINTS, R_POINTS, REFERENCE_LENGTH, SHAFT_WIDTH
from multi_component_exact_factorization_discrete_gpu.compare_tdse import _stream_arrays


def run(args):
    archive, run_dir = render.find_archive(render.resolve_run_input(args.run))
    cache = run_dir/'tdse_exact_factorization_fields.npz'
    with np.load(archive, allow_pickle=True) as data:
        times = data['times_fs']
        options = data['args'].item()
        obs = dict(q=data['q'], R=data['R'], options=options)
    candidates = np.flatnonzero((times >= args.start_fs) & (times <= args.stop_fs))
    if not len(candidates):
        raise ValueError('No saved frames in requested interval')
    chosen = candidates[np.unique(np.linspace(0, len(candidates)-1,
                                              min(args.frames, len(candidates))).round().astype(int))]
    with np.load(cache) as data:
        if not np.allclose(data['times_fs'], times, atol=1e-10, rtol=0):
            raise ValueError('Source/cache saved times differ')
        scalar = 'tdpes1_total' if 'tdpes1_total' in data.files else 'epsilon_1'
        convention = str(data.get('energy_convention', 'harmonic_included'))
    def read_selected(path, keys):
        with _stream_arrays(path, keys) as readers:
            result = {}
            for key, reader in readers.items():
                print(f'Preview streaming {path.name}: {key}', flush=True)
                first = reader.read(int(chosen[0]))
                shape = (len(chosen),)+first.shape
                if args.scratch_dir:
                    array = np.lib.format.open_memmap(
                        Path(args.scratch_dir)/(key+'.npy'), mode='w+',
                        dtype=first.dtype, shape=shape)
                else:
                    array = np.empty(shape, dtype=first.dtype)
                array[0] = first
                for i, f in enumerate(chosen[1:], 1):
                    array[i] = reader.read(int(f))
                result[key] = array
            return result
    obs.update(read_selected(archive, ['joint_density']))
    obs['times_fs'] = times[chosen]
    ef = read_selected(cache, [scalar, 'a', 'b'])
    ef['energy_convention'] = convention
    separate_fields(ef, obs)
    settings = render.parse_args([str(run_dir)])
    settings.max_frames = len(chosen)
    settings.velocity_q_points, settings.velocity_R_points = Q_POINTS, R_POINTS
    prep = render._joint_velocity_preparation(obs, ef, settings)
    # Keep sparse sampling sites fixed while the camera follows each frame.
    occupied = np.zeros(obs['joint_density'].shape[1:], dtype=bool)
    for rho in obs['joint_density']:
        occupied |= np.isfinite(rho) & (rho >= 1e-3)
    for key, axis, count in (('q', 1, Q_POINTS), ('R', 0, R_POINTS)):
        coordinate = obs[key]
        found = np.flatnonzero(np.any(occupied, axis=axis))
        if found.size:
            pad = max(.3, .1*(coordinate[found[-1]]-coordinate[found[0]]))
            limits = (max(float(coordinate[0]), float(coordinate[found[0]])-pad),
                      min(float(coordinate[-1]), float(coordinate[found[-1]])+pad))
            prep[key+'_limits'] = limits
            prep[key+'_indices'] = render._uniform_sample_indices(coordinate, limits, count)
    prep['q_mesh'], prep['R_mesh'] = np.meshgrid(obs['q'][prep['q_indices']], obs['R'][prep['R_indices']])
    def focus(f):
        occupied = np.isfinite(obs['joint_density'][f]) & (obs['joint_density'][f] >= 1e-3)
        limits = {}
        for key, axis in (('q', 1), ('R', 0)):
            coordinate = obs[key]
            found = np.flatnonzero(np.any(occupied, axis=axis))
            limits[key] = prep[key+'_limits']
            if found.size:
                pad = max(.3, .12*(coordinate[found[-1]]-coordinate[found[0]]))
                limits[key] = (max(float(coordinate[0]), float(coordinate[found[0]])-pad),
                               min(float(coordinate[-1]), float(coordinate[found[-1]])+pad))
        return limits
    def velocity(f):
        peak = max(float(obs['joint_density'][f].max()), 1e-300)
        return render._joint_velocity_frame(obs, ef, prep, f, 1e-3/peak)
    speeds = [np.ma.hypot(*velocity(f)).compressed() for f in range(len(chosen))]
    speed = np.concatenate(speeds)
    reference = max(float(np.percentile(speed, 95)), 1e-14) if speed.size else 1.
    if args.reference_speed is not None:
        reference = args.reference_speed
    # Fixed screen-length calibration prevents zoom from changing arrow size.
    # Explicit screen-space directions below account for unequal axis scales.
    scale = reference/REFERENCE_LENGTH
    output = Path(args.outdir) if args.outdir else run_dir/'report'/'tdpes_velocity_preview'
    output.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(SIGNED_CMAP).copy()
    cmap.set_bad(MASK_COLOR)
    # Derive the physical panel ratio from the actual six-panel TDPES layout,
    # rather than substituting the aspect of its entire movie canvas.
    template = plt.figure(figsize=(16.5, 9.2))
    template_axis = render._tdpes1_origin_axes(template)[0]
    bounds = template_axis.get_position()
    panel_ratio = bounds.width*16.5/(bounds.height*9.2)
    plt.close(template)
    def build():
        fig = plt.figure(figsize=(10, 8))
        panel_width = panel_ratio*.60*8/10
        ax = fig.add_axes([.15, .21, panel_width, .60])
        cax = fig.add_axes([.15+panel_width+.025, .21, .024, .60])
        ax.set_facecolor(MASK_COLOR)
        image = ax.imshow(np.zeros((len(obs['R']), len(obs['q']))), origin='lower',
                          extent=(obs['q'][0], obs['q'][-1], obs['R'][0], obs['R'][-1]),
                          cmap=cmap, vmin=-args.bound, vmax=args.bound, interpolation='nearest')
        u, v = velocity(0)
        arrows = ax.quiver(prep['q_mesh'], prep['R_mesh'], u, v, color='#397d50',
                           edgecolor='white', linewidth=.25, alpha=.85,
                           angles='xy', scale_units='inches', scale=scale,
                           units='inches', width=SHAFT_WIDTH,
                           pivot='mid', minlength=0, zorder=6)
        ax.quiverkey(arrows, .98, 1.10, reference,
                     rf'$v_{{\mathrm{{ref}}}}={render._math_scientific(reference)}\ a_0/t_{{au}}$',
                     labelpos='W', coordinates='axes')
        ax.set(xlim=prep['q_limits'], ylim=prep['R_limits'],
               xlabel=r'proton $q$ ($a_0$)', ylabel=r'heavy $R$ ($a_0$)')
        ax.set_aspect('auto')
        fig.colorbar(image, cax=cax,
                     label='TDPES1 (Ha; trap excluded)', extend='both')
        title = fig.suptitle('', fontsize=15)
        fig.text(.09, .055, r'PG: $\mathbf{v}=(a/m_p,b/M)$; fixed arrow scale; no velocity clipping.'
                 '\n'+r'$\rho_{qR}\geq10^{-3}\,a_0^{-2}$; fixed panel, moving limits. Arrows are flow, not force.', fontsize=10)
        def update(f):
            if f % 50 == 0:
                print(f'Render frame {f+1}/{len(chosen)}', flush=True)
            rho = obs['joint_density'][f]
            image.set_data(np.ma.masked_where((rho < 1e-3) | ~np.isfinite(rho), ef[scalar][f]).T)
            limits = focus(f)
            ax.set_xlim(limits['q'])
            ax.set_ylim(limits['R'])
            # Direction follows the plotted coordinate geometry, while length
            # remains proportional to physical speed, independent of zoom.
            u, v = velocity(f)
            box = ax.get_position()
            sx = box.width*fig.get_figwidth()/np.diff(limits['q'])[0]
            sy = box.height*fig.get_figheight()/np.diff(limits['R'])[0]
            arrows.angles = np.ma.filled(np.degrees(np.ma.arctan2(v*sy, u*sx)), 0).ravel()
            arrows.set_UVC(np.ma.hypot(u, v), np.ma.zeros(u.shape))
            render._absolute_overlay(ax, obs, f)
            title.set_text(r'TDPES1 $\epsilon^{(1)}_{\rm PG}$ + probability-flow velocity'
                           +f" | t={obs['times_fs'][f]:.4f} fs")
            return image, arrows, title
        update(0)
        return fig, update
    for f in np.unique(np.linspace(0, len(chosen)-1, min(4, len(chosen))).round().astype(int)):
        fig, update = build()
        update(int(f))
        fig.savefig(output/f'preview_{obs["times_fs"][f]:08.4f}fs.png', dpi=130)
        plt.close(fig)
    fig, update = build()
    movie = FuncAnimation(fig, update, frames=len(chosen), blit=False)
    movie.save(output/'tdpes1_velocity_preview.mp4', fps=args.fps, dpi=120,
               extra_args=['-crf', '18', '-preset', 'fast'])
    plt.close(fig)
    print(f'Preview: {output}; {len(chosen)} frames; {len(chosen)/args.fps:.2f} s; v95={reference}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--start-fs', type=float, default=20.)
    p.add_argument('--stop-fs', type=float, default=60.)
    p.add_argument('--frames', type=int, default=24)
    p.add_argument('--fps', type=int, default=6)
    p.add_argument('--bound', type=float, default=.1)
    p.add_argument('--outdir')
    p.add_argument('--reference-speed', type=float,
                   help='Fixed arrow calibration, e.g. to match a previous movie')
    p.add_argument('--disk-backed', action='store_true',
                   help='Use temporary disk arrays for long movies; cleaned on completion')
    args = p.parse_args()
    if args.frames < 1 or args.fps < 1 or args.bound <= 0:
        p.error('frames, fps, bound must be positive')
    if args.reference_speed is not None and (not np.isfinite(args.reference_speed) or args.reference_speed <= 0):
        p.error('reference-speed must be finite and positive')
    args.scratch_dir = None
    if args.disk_backed:
        with TemporaryDirectory(prefix='mcef-velocity-') as scratch:
            args.scratch_dir = scratch
            run(args)
    else:
        run(args)
