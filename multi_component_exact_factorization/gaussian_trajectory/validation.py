"""Small reproducible formula/physical-reference audits, never a long pilot.

Run ``python -m ...gaussian_trajectory.validation --out NEW_DIRECTORY``.
Existing output paths are refused. The fixed-R spectral reference below uses
the actual real-q PEN Hamiltonian; it is not a heavy-trajectory simulation.
The RK4 scan is a separately labelled manufactured phase-only eigenmode, not
evidence that explicit RK4 is stable for the full stiff real-q Hamiltonian.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.linalg import eigh

from .gaussian import build_bridge, pack_parameters, split_and_project
from .mls import MLSConfig, reconstruct_R_derivatives
from .models import InternalModel, ModelConfig
from ..pen_perspectives_2026.model import AU_FS, electronic


def fixed_R_hamiltonian(model, R):
    """Explicit small operator from action on identity columns, no projection.

    Constant rectangle dq makes the primitive metric a scalar identity. The
    Hamiltonian entries are unchanged on sqrt(dq)-weighted amplitudes.
    """
    dimension = len(model.q)*model.nk
    identity_columns = np.eye(dimension, dtype=complex).reshape(dimension,len(model.q),model.nk)
    matrix = model.action(np.full(dimension,R),identity_columns).reshape(dimension,dimension).T
    defect = float(np.max(np.abs(matrix-matrix.conj().T)))
    if not np.all(np.isfinite(matrix)) or defect > 1e-11:
        raise ValueError("Fixed-R model action is not finite Hermitian: %g" % defect)
    return matrix, defect


def rabi_reference(coupling=.02, nq=64):
    """Actual PEN-Rabi fixed-R quantum reference, with zero-coupling control.

    A local two-level RWA transfer time only selects the displayed interval.
    Dynamics use full real-q Hamiltonian eigenpairs, including diagonal dipole
    and counterrotating terms. No no-transfer/zero-NAC correction is applied.
    """
    if not np.isfinite(coupling) or coupling <= 0:
        raise ValueError("A positive finite comparison coupling is required")
    config = ModelConfig(name="pen-rabi",nq=nq,qmin=-20.,qmax=20.,g=float(coupling),dse=False)
    coupled = InternalModel(config)
    R = coupled.initial_R_center
    h, dip = electronic(coupled.c,np.array([R]))
    energies, vectors = np.linalg.eigh(h[0])
    transition_dipole = float(abs((vectors.T@dip[0]@vectors)[0,1]))
    resonant_matrix_element = coupling*np.sqrt(coupled.omega/2)*transition_dipole
    if resonant_matrix_element <= 0:
        raise ValueError("No transition dipole for the requested Rabi test")
    transfer_time = np.pi/(2*resonant_matrix_element)
    times = np.linspace(0.,transfer_time,101)
    output = {"label": "FIXED-R real-q PEN-Rabi spectral reference; not trajectory dynamics",
              "config": asdict(config), "R_fixed": R, "dimension": nq*2,
              "bare_gap_Ha": float(energies[1]-energies[0]),
              "omega_Ha": coupled.omega,"transition_dipole_au": transition_dipole,
              "rwa_transfer_time_au": float(transfer_time),
              "time_window_rule": "pi/(2*g*sqrt(omega/2)*abs(mu01)); RWA only selects window",
              "propagator": "exact eigenpair evolution of the finite-grid Hamiltonian",
              "cases": []}
    for g in (0.,coupling):
        model = InternalModel(ModelConfig(**dict(asdict(config),g=g)))
        H, defect = fixed_R_hamiltonian(model,R)
        eigenvalues, U = eigh(H)
        initial = model.initial(np.array([R]))[0]
        initial_weighted = np.sqrt(model.dq)*initial.ravel()
        amplitudes = U.conj().T@initial_weighted
        weighted = np.einsum('ij,tj->ti',U,amplitudes[None,:]*np.exp(-1j*times[:,None]*eigenvalues[None,:]))
        states = weighted.reshape(len(times),len(model.q),2)/np.sqrt(model.dq)
        obs = model.observables(np.full(len(times),R),states)
        bo = obs['bo_population']
        output['cases'].append(dict(g=float(g),times_fs=(times*AU_FS).tolist(),
            excited_population=bo[:,1].tolist(),ground_population=bo[:,0].tolist(),
            photon_number=obs['photon_number'].tolist(),
            max_ground_population=float(np.max(bo[:,0])),
            norm_error=float(np.max(np.abs(obs['norm']-1))),
            energy_drift_Ha=float(np.max(np.abs(obs['internal_energy']-obs['internal_energy'][0]))),
            hermiticity_error=defect,
            max_q_edge_mass=float(np.max(obs['q_edge_probability']))))
    return output


def vacuum_box_audit(nq=64):
    """Actual g=0 model action compared to analytic S1 x photon vacuum energy."""
    rows = []
    for half_width in (8.,12.,16.,20.):
        model = InternalModel(ModelConfig(name="pen-rabi",nq=nq,qmin=-half_width,
                                          qmax=half_width,g=0.,dse=False))
        R = np.array([model.initial_R_center])
        G = model.initial(R)
        eigenvalue = np.linalg.eigvalsh(electronic(model.c,R)[0][0])[1]+.5*model.omega
        defect = model.action(R,G)-eigenvalue*G
        norm_defect = float(np.sqrt(np.einsum('q,iqk->',model.quadrature,np.abs(defect)**2)))
        obs = model.observables(R,G)
        rows.append(dict(half_width=half_width,nq=nq,dq=model.dq,
                         vacuum_action_defect_Ha=norm_defect,
                         photon_number_error=abs(float(obs['photon_number'][0])),
                         q_edge_mass=float(obs['q_edge_probability'][0])))
    return {"label": "Finite periodic q-box action defect; fixed nq, varying box and spacing",
            "rows": rows, "limitation": "Box and spacing change together; not independent grid certification"}


def phase_mode_dt_audit():
    """Literal P3/P8 RK4 vs native complex update in one harmonic eigenmode.

    HG=E*G is supplied analytically for this manufactured invariant sector.
    This removes the high-energy-grid stability issue on purpose and does not
    replace the actual model action in the separate fixed-R physical audit.
    """
    q = np.linspace(-14.,14.,64,endpoint=False)
    w = np.full(len(q),q[1]-q[0])
    omega, energy, end_au = .17, .255, 2.
    initial_z = pack_parameters([1.],[0.],[1/np.sqrt(2*omega)],coordinates="transformed")
    initial_bridge = build_bridge(q,w,initial_z,components=1,coordinates="transformed")
    initial_C = np.zeros((len(q),2),dtype=complex)
    initial_C[:,1] = 1.
    initial_G = initial_bridge.amplitude[:,None]*initial_C
    exact = initial_G*np.exp(-1j*energy*end_au)
    def projected_rate(z,C):
        bridge = build_bridge(q,w,z,components=1,coordinates="transformed")
        rate = split_and_project(-1j*energy*bridge.amplitude[:,None]*C,C,bridge,
                 support_relative=0.,max_excluded_mass=0.,pnc_tolerance=.02)
        return rate.parameter_dot, rate.C_dot
    rows = []
    for dt in (.5,.25,.125,.0625):
        steps = int(round(end_au/dt))
        native = initial_G.copy()
        z, C = initial_z.copy(),initial_C.copy()
        for _ in range(steps):
            k1 = -1j*energy*native
            k2 = -1j*energy*(native+.5*dt*k1)
            k3 = -1j*energy*(native+.5*dt*k2)
            k4 = -1j*energy*(native+dt*k3)
            native += dt*(k1+2*k2+2*k3+k4)/6
            z1,c1 = projected_rate(z,C)
            z2,c2 = projected_rate(z+.5*dt*z1,C+.5*dt*c1)
            z3,c3 = projected_rate(z+.5*dt*z2,C+.5*dt*c2)
            z4,c4 = projected_rate(z+dt*z3,C+dt*c3)
            z += dt*(z1+2*z2+2*z3+z4)/6
            C += dt*(c1+2*c2+2*c3+c4)/6
        compressed = build_bridge(q,w,z,components=1,coordinates="transformed").amplitude[:,None]*C
        error = lambda state: float(np.sqrt(np.einsum('q,qk->',w,np.abs(state-exact)**2)))
        norm_error = lambda state: float(abs(np.einsum('q,qk->',w,np.abs(state)**2)-1))
        rows.append(dict(dt_au=dt,steps=steps,native_l2_error=error(native),
                         K1_l2_error=error(compressed),native_norm_error=norm_error(native),
                         K1_norm_error=norm_error(compressed),
                         parameter_change=float(np.max(np.abs(z-initial_z)))))
    return {"label": "Manufactured phase-only harmonic vacuum sector, not full-grid integrator certification",
            "end_au":end_au,"E_Ha":energy,"integrator":"RK4; no renormalization",
            "rows":rows}


def mls_convergence_audit():
    """Smooth complex phase on current nonuniform positions; known derivatives."""
    rows = []
    for n in (21,41,81):
        u = np.linspace(-1.,1.,n)
        R = u+.08*np.sin(3*u)
        R = R[np.random.default_rng(7).permutation(n)]
        values = np.exp(1j*(.7*R+.2*R**2))[:,None]*np.array([1.,1j])[None,:]/np.sqrt(2.)
        first = 1j*(.7+.4*R)[:,None]*values
        second = (.4j-(.7+.4*R)**2)[:,None]*values
        d,d2,audit = reconstruct_R_derivatives(R,values,config=MLSConfig(degree=4,neighbors=9),
              basis_convention="fixed primitive two-channel basis",gauge_convention="common analytic complex phase")
        rows.append(dict(ntraj=n,first_relative_error=float(np.linalg.norm(d-first)/np.linalg.norm(first)),
            second_relative_error=float(np.linalg.norm(d2-second)/np.linalg.norm(second)),
            maximum_condition=float(np.max(audit['condition']))))
    return {"label":"Manufactured complex-field MLS reconstruction, not a propagating closure",
            "degree":4,"neighbors":9,"rows":rows}


def render_validation(metrics,out):
    """Readable audit curves; comparable errors share log axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    ax = axes[0,0]
    for case in metrics['rabi']['cases']:
        ax.plot(case['times_fs'],case['excited_population'],label=r'$P_{S_1}$, g=%g' % case['g'],lw=2)
        if case['g']:
            ax.plot(case['times_fs'],case['photon_number'],'--',label=r'$\langle n_{ph}\rangle$, g=%g' % case['g'],lw=2)
    ax.set(title='A. Fixed-R real-q PEN reference',xlabel='t (fs)',ylabel='Population / photon number',ylim=(-.03,1.08))
    ax.legend(fontsize=10)
    ax = axes[0,1]
    rows = metrics['phase_mode_dt']['rows']
    ax.loglog([r['dt_au'] for r in rows],[r['native_l2_error'] for r in rows],'o-',label='Native phase-only mode',lw=2)
    ax.loglog([r['dt_au'] for r in rows],[r['K1_l2_error'] for r in rows],'x--',label='K=1 Gaussian projection',lw=2)
    ax.set(title='B. Manufactured RK4 step convergence',xlabel='dt (a.u.)',ylabel='State L2 error')
    ax.legend(fontsize=10)
    ax = axes[1,0]
    rows = metrics['vacuum_box']['rows']
    ax.semilogy([r['half_width'] for r in rows],[r['vacuum_action_defect_Ha'] for r in rows],'o-',lw=2)
    ax.set(title='C. Actual vacuum action / q-box error',xlabel='q half width (a.u.)',ylabel=r'$\|(H-E)G\|$ (Ha)')
    ax = axes[1,1]
    rows = metrics['mls']['rows']
    ax.loglog([r['ntraj'] for r in rows],[r['first_relative_error'] for r in rows],'o-',label=r'$\partial_R G$',lw=2)
    ax.loglog([r['ntraj'] for r in rows],[r['second_relative_error'] for r in rows],'s-',label=r'$\partial_R^2G$',lw=2)
    ax.set(title='D. Current-position MLS convergence',xlabel='Number of R samples',ylabel='Relative error')
    ax.legend(fontsize=10)
    for ax in axes.flat:
        ax.tick_params(labelsize=11)
        ax.grid(alpha=.2)
    fig.suptitle('Small equation / reference audits — not full trajectory validation',fontsize=17)
    fig.savefig(Path(out)/'convergence.png',dpi=180)
    fig.savefig(Path(out)/'convergence.pdf')
    plt.close(fig)


def run_validation(out):
    """Write only to a new output directory, preserving every existing result."""
    root = Path(out)
    if root.exists():
        raise FileExistsError('Validation output already exists; choose a new directory: '+str(root))
    root.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    metrics = {'status':'REFERENCE_AUDIT_NOT_PRODUCTION_CERTIFICATION',
               'python':platform.python_version(), 'numpy':np.__version__,
               'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')},
               'rabi':rabi_reference(),'vacuum_box':vacuum_box_audit(),
               'phase_mode_dt':phase_mode_dt_audit(),'mls':mls_convergence_audit(),
               'limitations':['No moving-heavy trajectory dynamics in this audit.',
                   'Fixed-R spectral evolution is exact only for the chosen finite q grid.',
                   'Phase-only RK4 scan does not certify stability of a stiff full-grid propagator.',
                   'No CTMQC/TSH force or gauge closure is supplied by these tests.']}
    render_validation(metrics,root)
    metrics['wall_seconds'] = time.perf_counter()-started
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('VALIDATION',root/'metrics.json')
    print('FIGURE',root/'convergence.png')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    run_validation(args.out)


if __name__ == '__main__':
    main()
