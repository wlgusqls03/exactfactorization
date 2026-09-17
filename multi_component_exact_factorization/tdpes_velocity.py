"""Shared probability-flow arrows for every two-dimensional TDPES1 view.

No forces are inferred from the surface. Cached, density-supported mechanical
momenta are sampled once; gauge transformations do not change these arrows.
"""
import numpy as np
from matplotlib.quiver import Quiver

DENSITY_FLOOR = 1e-3
Q_POINTS, R_POINTS = 42, 20
# Slightly larger (+10%) and closer-spaced than the original approved preview.
REFERENCE_LENGTH = .240625  # inches at the reference speed
SHAFT_WIDTH = .0226875


class FlowQuiver(Quiver):
    """Recompute direction after layout; keep physical speed as screen length."""

    def set_flow(self, u, v):
        self.flow_u, self.flow_v = u, v
        self.set_UVC(np.ma.hypot(u, v), np.zeros(u.shape))

    def draw(self, renderer):
        # Constrained-layout/colorbars may resize the axes after update().
        sx = self.axes.bbox.width / np.diff(self.axes.get_xlim())[0]
        sy = self.axes.bbox.height / np.diff(self.axes.get_ylim())[0]
        self.angles = np.ma.filled(np.degrees(np.ma.arctan2(
            self.flow_v*sy, self.flow_u*sx)), 0).ravel()
        # Summary panels should not be covered by full-size arrows.
        panel_inches = self.axes.bbox.width/self.figure.dpi
        factor = min(1., panel_inches/6.2546)
        self.scale = self.reference_speed/(REFERENCE_LENGTH*factor)
        self.width = SHAFT_WIDTH*factor
        super().draw(renderer)


def prepare(obs, ef):
    if 'zero' in str(ef.get('gauge', '')) and not all(
            key in ef for key in ('mechanical_q', 'mechanical_R_first')):
        raise ValueError('Zero-gauge arrows require retained mechanical momenta, not zero connections')
    q_momentum = ef.get('mechanical_q', ef.get('a'))
    R_momentum = ef.get('mechanical_R_first', ef.get('b'))
    if q_momentum is None or R_momentum is None:
        raise ValueError('TDPES1 velocity arrows require a/b or mechanical momenta in the EF cache')
    rho = obs['joint_density']
    signature = (id(rho), id(q_momentum), id(R_momentum))
    cached = obs.get('_tdpes_velocity')
    if cached is not None and cached['signature'] == signature:
        return cached
    occupied = np.zeros(rho.shape[1:], bool)
    for density in rho:
        occupied |= np.isfinite(density) & (density >= DENSITY_FLOOR)
    indices = []
    for key, axis, count in (('q', 1, Q_POINTS), ('R', 0, R_POINTS)):
        coordinate = np.asarray(obs[key])
        found = np.flatnonzero(np.any(occupied, axis=axis))
        if found.size:
            pad = max(.3, .1*(coordinate[found[-1]]-coordinate[found[0]]))
            found = np.flatnonzero((coordinate >= coordinate[found[0]]-pad)
                                  & (coordinate <= coordinate[found[-1]]+pad))
        else:
            found = np.arange(len(coordinate))
        indices.append(found[np.unique(np.rint(np.linspace(0, len(found)-1,
                                                          min(count, len(found)))).astype(int))])
    iq, iR = indices
    mesh = np.meshgrid(obs['q'][iq], obs['R'][iR])
    masses = (float(obs['options'].get('proton_mass', 1836.15267343)),
              float(obs['options'].get('heavy_mass', 1836.15267343)))
    velocities, speeds = [], []
    for f, density in enumerate(rho):
        density = density[np.ix_(iq, iR)].T
        u = np.asarray(q_momentum[f])[np.ix_(iq, iR)].T/masses[0]
        v = np.asarray(R_momentum[f])[np.ix_(iq, iR)].T/masses[1]
        invalid = (density < DENSITY_FLOOR) | ~np.isfinite(density) | ~np.isfinite(u) | ~np.isfinite(v)
        u, v = np.ma.array(u, mask=invalid), np.ma.array(v, mask=invalid)
        velocities.append((u, v))
        speeds.append(np.ma.hypot(u, v).compressed())
    speed = np.concatenate(speeds)
    reference = max(float(np.percentile(speed, 95)), 1e-14) if speed.size else 1.
    cached = dict(signature=signature, mesh=mesh, velocities=velocities, reference=reference)
    obs['_tdpes_velocity'] = cached
    return cached


def overlay(axis, obs, ef, frame):
    """Create or update one artist; never accumulate arrows on animation frames."""
    data = prepare(obs, ef)
    state = getattr(axis, '_tdpes_flow', None)
    if state is None or state[0] is not data:
        if state is not None:
            state[1].remove()
            state[2].remove()
        u, v = data['velocities'][frame]
        arrows = FlowQuiver(axis, *data['mesh'], u, v, color='#397d50',
                           edgecolor='white', linewidth=.25, alpha=.85,
                           angles='xy', scale_units='inches', scale=1.,
                           units='inches', width=SHAFT_WIDTH, pivot='mid',
                           minlength=0, zorder=6)
        arrows.reference_speed = data['reference']
        axis.add_collection(arrows, autolim=False)
        exponent = int(np.floor(np.log10(data['reference'])))
        mantissa = data['reference']/10.**exponent
        label = axis.text(.02, .98,
                          rf'$\vec v=(K_q/m_p,K_R/M)$; $v_{{ref}}={mantissa:.2f}\times10^{{{exponent}}}\,a_0/t_{{au}}$',
                          transform=axis.transAxes, va='top', fontsize=6,
                          color='#245632', zorder=8,
                          bbox=dict(facecolor='white', alpha=.7, edgecolor='none', pad=1))
        axis._tdpes_flow = (data, arrows, label)
    else:
        arrows = state[1]
    arrows.set_flow(*data['velocities'][frame])
    return arrows
