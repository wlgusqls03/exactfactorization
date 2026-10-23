"""Read-only molecular parameter search; no artificial double-well term.

Uses the production frozen_potential and electronic DST Hamiltonian. Scan and
validation outputs are separate from every existing TDSE case. A BO double
well is not certification of VSC reaction kinetics or of a production grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh

from .core import configuration, frozen_potential, sine_kinetic
from .workflow import source_record, write_json, write_npz

MEV = 27211.386245988


def stationary_points(R, energy):
    """Interior stationary points of a smooth interpolant; never box ends."""
    spline = CubicSpline(R, energy)
    roots = spline.derivative().roots(extrapolate=False)
    minima, maxima = [], []
    for r in roots:
        if not R[1] < r < R[-2]:
            continue
        item = dict(R=float(r), energy_Ha=float(spline(r)),
                    curvature=float(spline(r, 2)))
        (minima if item['curvature'] > 0 else maxima).append(item)
    barriers = []
    for peak in maxima:
        left = [m for m in minima if m['R'] < peak['R']]
        right = [m for m in minima if m['R'] > peak['R']]
        if left and right:
            l, r = left[-1], right[0]
            barriers.append(dict(left=l, saddle=peak, right=r,
                left_barrier_Ha=peak['energy_Ha']-l['energy_Ha'],
                right_barrier_Ha=peak['energy_Ha']-r['energy_Ha'],
                bias_Ha=r['energy_Ha']-l['energy_Ha']))
    return dict(minima=minima, maxima=maxima, barriers=barriers,
                double_well=bool(barriers))


def bo_curve(config, R, *, nx=359, box=(-22., 30.), states=2, vectors=False):
    """Identical electronic kinetic/potential to production, arbitrary R probe."""
    c = configuration(dict(config, nx=nx, x_left=box[0], x_right=box[1]))
    dx = (box[1]-box[0])/(nx+1)
    x = box[0]+dx*np.arange(1, nx+1)
    tx, _ = sine_kinetic(nx, dx)
    energies, orbitals, mu, mu2 = [], [], [], []
    for r, v in zip(R, frozen_potential(x, R, c)):
        e, u = eigh(tx+np.diag(v), subset_by_index=[0, states-1], check_finite=False)
        energies.append(e)
        if vectors:
            dip = r-x+c['left_fixed']+c['heavy_fixed']
            mu.append((u.T*dip)@u)
            mu2.append((u.T*dip**2)@u)
            orbitals.append(u[:, 0]/np.sqrt(dx))
    result = dict(R=np.asarray(R), energy=np.array(energies), x=x, dx=dx,
                  config=c, topology=stationary_points(R, np.array(energies)[:, 0]))
    if vectors:
        result.update(phi0=np.array(orbitals), mu=np.array(mu), mu2=np.array(mu2))
    return result


def local_levels(R, energy, saddle, mass=1836., spacing=.032, states=6):
    """Left-well Dirichlet preparation with boundary exactly at BO saddle.

    The wall is only a preparation convention. Endpoint leakage and comparisons
    to the global spectrum must be inspected before real-time propagation.
    """
    left = float(R[0])
    n = int(np.ceil((saddle-left)/spacing))-1
    h = (saddle-left)/(n+1)
    grid = left+h*np.arange(1, n+1)
    tx, _ = sine_kinetic(n, h, mass)
    e, u = eigh(tx+np.diag(CubicSpline(R, energy)(grid)),
                subset_by_index=[0, states-1], check_finite=False)
    return dict(grid=grid, energy=e, wave=u/np.sqrt(h), spacing=h,
                left_boundary=left, right_boundary=saddle,
                endpoint_probability_v1=float(u[-1, 1]**2))


def convergence_checks(convergence):
    """Explicit gate for well topology, not for full TDSE or reaction dynamics."""
    ref = convergence['wide']['topology']['barriers']
    unique = all(len(v['topology']['minima']) == 2 and
                 len(v['topology']['barriers']) == 1 for v in convergence.values())
    if not unique:
        return dict(passed=False, reason='Expected exactly two minima and one barrier at every resolution')
    reference = ref[0]
    barrier_change = max(abs(v['topology']['barriers'][0][key]-reference[key])
                         for v in convergence.values()
                         for key in ('left_barrier_Ha','right_barrier_Ha'))
    position_change = max(abs(v['topology']['barriers'][0][key]['R']-reference[key]['R'])
                          for v in convergence.values() for key in ('left','saddle','right'))
    height = min(reference['left_barrier_Ha'], reference['right_barrier_Ha'])
    return dict(passed=bool(height>1e-4 and barrier_change<1e-6 and position_change<.01),
                minimum_barrier_Ha=height, maximum_barrier_change_Ha=barrier_change,
                maximum_stationary_position_change_a0=position_change,
                criteria='two minima + one saddle; both barriers >1e-4 Ha; height change <1e-6 Ha; position change <.01 a0')


def scan(out):
    R = np.linspace(-8.8, 8.8, 145)
    records, curves = [], []
    # First preserve both historical sites while changing only the left range;
    # then sample symmetric fixed-ion electronic interactions more broadly.
    settings = [dict(erf_r_lx=a) for a in (3.1, 3.3, 3.5, 3.7, 3.9, 4.0)]
    settings += [dict(erf_r_lx=a, erf_r_Rx=a, erf_r_qx=b)
                 for a in (2.5, 3.1, 3.5, 4.) for b in (2., 3., 4., 5., 6.)]
    for index, cfg in enumerate(settings):
        p = bo_curve(cfg, R, nx=144)
        records.append(dict(index=index, changes=cfg, topology=p['topology']))
        curves.append(p['energy'][:, 0])
        print('scan', index, cfg, 'double_well=', p['topology']['double_well'], flush=True)
    write_json(out/'scan.json', dict(source=source_record(), records=records,
               scope='Coarse BO screening, not convergence certification'))
    write_npz(out/'scan.npz', R=R, ground=np.array(curves))


def validate(out, moving_range=5.):
    candidate = dict(erf_r_lx=4.0)  # only molecular parameter changed
    if moving_range != 5.:
        candidate['erf_r_qx']=float(moving_range)
    configurations = [('historical', {}), ('symmetric_4', candidate),
                      ('symmetric_3p1', dict(erf_r_Rx=3.1))]
    R = np.linspace(-8.9, 8.9, 401)
    overview = {}
    for name, cfg in configurations:
        print('native electronic grid:', name, flush=True)
        overview[name] = bo_curve(cfg, R, nx=359)
    base = overview['symmetric_4']
    if len(base['topology']['barriers']) != 1:
        raise ValueError('Expected a unique interior double-well barrier')
    barrier = base['topology']['barriers'][0]
    fine_R = np.linspace(-8.9, 8.9, 801)
    print('fine electronic and nuclear sampling', flush=True)
    fine = bo_curve(candidate, fine_R, nx=719)
    print('expanded electronic box, same fine dx', flush=True)
    # 65 / 900 equals 52 / 720: isolate box growth at fixed electronic dx.
    wide = bo_curve(candidate, fine_R, nx=899, box=(-28.5, 36.5))
    points = np.array([barrier[k]['R'] for k in ('left', 'saddle', 'right')])
    convergence = {}
    for name, p in [('native', base), ('fine', fine), ('wide', wide)]:
        top = p['topology']
        convergence[name] = dict(nx=len(p['x']), dx=p['dx'], nR=len(p['R']),
            box=[p['config']['x_left'], p['config']['x_right']], topology=top,
            energy_at_native_stationary_points_Ha=CubicSpline(p['R'], p['energy'][:, 0])(points).tolist())
    finest = wide
    b = finest['topology']['barriers'][0]
    local = local_levels(fine_R, finest['energy'][:, 0], b['saddle']['R'])
    local_fine = local_levels(fine_R, finest['energy'][:, 0], b['saddle']['R'], spacing=.016)
    # Global spectrum identifies tunnelling doublets; it is not confused with
    # a local v=0->1 excitation gap.
    h = fine_R[1]-fine_R[0]
    tx, _ = sine_kinetic(len(fine_R)-2, h, 1836.)
    global_e = eigh(tx+np.diag(finest['energy'][1:-1, 0]),
                    subset_by_index=[0, 9], eigvals_only=True, check_finite=False)
    vib = dict(local_energy_Ha=local['energy'].tolist(),
        local_fine_energy_Ha=local_fine['energy'].tolist(),
        omega_v01_Ha=float(np.diff(local_fine['energy'][:2])[0]),
        barrier_minus_local_v1_Ha=float(b['saddle']['energy_Ha']-local_fine['energy'][1]),
        endpoint_probability_v1=local_fine['endpoint_probability_v1'],
        global_energy_Ha=global_e.tolist(),
        local_wall_R=b['saddle']['R'],
        warning='Local wall is only initial-state preparation. No kick. Not a thermal rate protocol.')
    report = dict(status='BARE_BO_DOUBLE_WELL_VALIDATED_NOT_TDSE_OR_VSC_DYNAMICS',
        source=source_record(), changed_molecular_parameters=candidate,
        preserved_fixed_sites=[-9.5, 9.5], no_added_external_double_well=True,
        overview={k:v['topology'] for k,v in overview.items()},
        convergence=convergence, convergence_checks=convergence_checks(convergence), vibration=vib,
        symmetry_note='Equal electronic fixed-ion smoothing; original bare left ion/proton and softened right ion/proton terms are preserved. Near symmetry holds in the reaction region, not by imposing a globally symmetric Hamiltonian.')
    if not report['convergence_checks']['passed']:
        report['status']='BO_TOPOLOGY_CONVERGENCE_NOT_PASSED'
    write_json(out/'validation.json', report)
    # This is a molecular proposal, deliberately not an approved production preset.
    write_json(out/'candidate_molecular_config.json', dict(candidate,
        local_state_method='local_wall', local_wall_R=float(b['saddle']['R']),
        dividing_R=float(b['saddle']['R']), p0=0.0))
    write_npz(out/'validation_curves.npz', R=R,
        historical=overview['historical']['energy'], native=base['energy'],
        symmetric_3p1=overview['symmetric_3p1']['energy'], fine_R=fine_R,
        fine=fine['energy'], wide=wide['energy'], local_R=local_fine['grid'],
        local_wave=local_fine['wave'], local_energy=local_fine['energy'], global_energy=global_e)
    plot(out, overview, finest, local_fine, report)
    print(json.dumps(report, indent=2), flush=True)


def cavity_check(out, validation_dir):
    """DSE-on static check, NOT a claim of VSC reaction dynamics.

    Local-left vibrational dipoles calibrate the chosen coupling. Fixed-R
    electron+photon ground curves are distinct from vibrational LP/UP levels.
    """
    from .calibration import product_matrix, calibrate_strengths
    validation_dir=Path(validation_dir)
    report=json.loads((validation_dir/'validation.json').read_text())
    if not convergence_checks(report['convergence'])['passed']:
        raise ValueError('Bare double-well validation must pass first')
    with np.load(validation_dir/'validation_curves.npz') as z:
        lr=z['local_R']; wave=z['local_wave']; ev=z['local_energy']
    R=np.linspace(-8.9,8.9,161)
    print('Electronic analysis basis: six BO states, full electronic grid',flush=True)
    parameters=report['changed_molecular_parameters']
    p=bo_curve(parameters,R,nx=359,states=6,vectors=True)
    h=lr[1]-lr[0]
    mu0=CubicSpline(R,p['mu'][:,0,0])(lr)
    mu20=CubicSpline(R,p['mu2'][:,0,0])(lr)
    dip=(wave.T*(mu0*h))@wave;dip2=(wave.T*(mu20*h))@wave
    omega=float(ev[1]-ev[0])
    c=configuration(dict(parameters,n_fock_analysis=16))
    strengths,calibration=calibrate_strengths(ev,dip,dip2,omega,c)
    curves={};topologies={}
    for name in ('zero','weak','strong'):
        g=strengths[name]
        for nb,nf in ((4,32),(6,32),(6,64)):
            label=f'{name}_bo{nb}_fock{nf}'
            print('Fixed-R static ground:',label,flush=True)
            values=[]
            for e,mu,mu2 in zip(p['energy'],p['mu'],p['mu2']):
                mat=product_matrix(e[:nb],mu[:nb,:nb],mu2[:nb,:nb],omega,g,nf,True)
                values.append(eigh(mat,subset_by_index=[0,0],eigvals_only=True,check_finite=False)[0])
            curves[label]=np.array(values)
            topologies[label]=stationary_points(R,curves[label])
    errors={}
    for name in ('zero','weak','strong'):
        reference=curves[f'{name}_bo6_fock64']
        errors[name]=dict(fock32_to64_max_Ha=float(np.max(abs(reference-curves[f'{name}_bo6_fock32']))),
                         bo4_to6_max_Ha=float(np.max(abs(curves[f'{name}_bo6_fock32']-curves[f'{name}_bo4_fock32']))))
    result=dict(scope='DSE-on fixed-R electron+photon ground-curve topology; no TDSE propagation',
                source=source_record(),omega_bare_local_v01_Ha=omega,
                g_cavity=strengths,local_left_BO_coupling_calibration=calibration,
                local_basis_gram_error=float(np.max(abs(wave.T@wave*h-np.eye(wave.shape[1])))),
                topology=topologies,analysis_basis_changes=errors,
                limitations=['Fixed-R ground curves are not vibrational LP/UP curves.',
                             'Calibration uses six local-left BO vibrational states and is not a converged global polariton spectrum.',
                             'Bare BO grid convergence and limited static basis comparisons do not certify full real-time VSC dynamics.'])
    write_json(out/'cavity_static_check.json',result)
    write_npz(out/'cavity_static_curves.npz',R=R,bare=p['energy'][:,0],**curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for name,color in zip(('zero','weak','strong'),('k','#0072B2','#D55E00')):
        energy=curves[f'{name}_bo6_fock64']
        axs[0].plot(R,energy,label=name,color=color)
        axs[1].plot(R,(energy-curves['zero_bo6_fock64'])*MEV,label=name,color=color)
    visible=(R>-4)&(R<4)
    visible_values=np.concatenate([curves[f'{n}_bo6_fock64'][visible] for n in ('zero','weak','strong')])
    lo,hi=visible_values.min(),visible_values.max();margin=.08*(hi-lo)
    axs[0].set(xlim=(-5,5),ylim=(lo-margin,hi+margin),ylabel='Absolute energy (Ha)',title='Fixed-R electron + photon ground reference (DSE on)')
    axs[1].set(xlim=(-5,5),ylabel='Change from uncoupled (meV)',title='Explicit energy differences')
    for ax in axs:
        ax.set_xlabel(r'$R$ ($a_0$)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'cavity_static_check.png',dpi=180);fig.savefig(out/'cavity_static_check.pdf');plt.close(fig)
    print(json.dumps(dict(omega=omega,strengths=strengths,basis_changes=errors),indent=2),flush=True)


def plot(out, overview, finest, local, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    with plt.rc_context({'font.size':13, 'axes.titlesize':16, 'axes.labelsize':14}):
        fig, axs = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
        moving=report['changed_molecular_parameters'].get('erf_r_qx',5.)
        labels=['Original parameters',f'Candidate: fixed ranges 4, moving range {moving:g}','Alternative: fixed ranges 3.1, moving range 5']
        for (name,p),color,label in zip(overview.items(), ['.4','#0072B2','#D55E00'],labels):
            axs[0,0].plot(p['R'],p['energy'][:,0], label=label, color=color)
        axs[0,0].set(xlim=(-7,7), ylim=(-.30,-.08), title='Bare ground BO: molecular parameters only', ylabel='Energy (Ha)')
        axs[0,0].legend(fontsize=10)
        b=finest['topology']['barriers'][0]; zero=b['left']['energy_Ha']
        R=finest['R']; E=finest['energy'][:,0]
        axs[0,1].plot(R,(E-zero)*MEV,color='#0072B2',lw=2.3)
        for key in ('left','saddle','right'):
            p=b[key];axs[0,1].plot(p['R'],(p['energy_Ha']-zero)*MEV,'ko')
        for j in (0,1):
            ev=(local['energy'][j]-zero)*MEV
            axs[0,1].hlines(ev, -4.5, b['saddle']['R'], colors=['#009E73','#D55E00'][j],linestyles='--',label=f'left v={j}')
        height=b['left_barrier_Ha']*MEV
        axs[0,1].set(xlim=(-5,5),ylim=(-.08*height,2.3*height),title='Resolved double well and local vibration',ylabel='Energy above left minimum (meV)')
        axs[0,1].legend()
        axs[1,0].plot(local['grid'],abs(local['wave'][:,1])**2,color='#D55E00',label=r'$|\chi_{L,v=1}|^2$')
        axs[1,0].axvline(b['saddle']['R'],color='k',ls='--',label='preparation boundary at saddle')
        axs[1,0].set(xlim=(-6,2), title='Left-localized v=1, no momentum kick',ylabel=r'Density ($a_0^{-1}$)');axs[1,0].legend()
        z=np.load(out/'validation_curves.npz')
        for key,color in [('fine','#0072B2'),('wide','#D55E00')]:
            delta=(CubicSpline(z['fine_R'],z[key][:,0])(z['R'])-z['native'][:,0])*MEV
            axs[1,1].plot(z['R'],delta,label=key+' − native',color=color)
        z.close();axs[1,1].axhline(0,color='.5',ls='--')
        axs[1,1].set(xlim=(-6,6),title='Electronic grid / box convergence',ylabel='Energy difference (meV)');axs[1,1].legend()
        for ax in axs.flat:
            ax.set_xlabel(r'$R$ ($a_0$)');ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(alpha=.15)
        fig.suptitle('Frozen-heavy Shin–Metiu: double-well candidate (cavity off)',fontsize=20)
        fig.savefig(out/'double_well_summary.png',dpi=180)
        fig.savefig(out/'double_well_summary.pdf')
        plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['scan','validate','cavity']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--validation-dir',type=Path)
    p.add_argument('--moving-range',type=float,default=5.,help='Optional electron/proton smoothing range; default preserves historical 5.0')
    a=p.parse_args(argv)
    a.out.mkdir(parents=True,exist_ok=False)
    if a.stage=='cavity':
        if a.validation_dir is None:p.error('cavity requires --validation-dir')
        cavity_check(a.out,a.validation_dir)
    else:
        if a.stage=='scan':scan(a.out)
        else:validate(a.out,a.moving_range)


if __name__=='__main__':main()
