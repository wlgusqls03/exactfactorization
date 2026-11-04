"""Separate author G-CTMQC execution CLI; never resumes a Python checkpoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from ..pen_perspectives_2026.model import AU_FS
from ..pen_perspectives_2026.resonance import paper_config
from ..pen_ctmqc_2026.core import Engine, Settings, hamiltonian
from .build import digest,HERE
from .data import POL_PERM, analyze_case
from .output_check import prepare_output_dirs, validate_native_outputs


def qml_input(c, view):
    """Explicit parameter read, not QML's compiled defaults.

    QML writes -Mu12 in the off-diagonal dipole. Our Config stores the signed
    dipole itself; hence Mu12_QML=-mu12_Config. The Hamiltonians then match.
    """
    name = 'OneD_Photons' if view == 'el' else 'OneD_Photons2'
    d, k = (2, 2) if view == 'el' else (1, 4)
    return f"""&potential pot_name='{name}', ndim={d}, nsurf={k},
 adiabatic=.true., read_nml=.true., Phase_Following=.true., printlevel=0 /
&{name} MR={c.mass:.17g}, MP=1, k={c.k:.17g}, a={c.a:.17g}, b={c.b:.17g},
 R1={c.r1:.17g}, R2={c.r2:.17g}, R3={c.r3:.17g}, Delta={c.delta:.17g},
 w={c.omega:.17g}, Z={c.z:.17g}, Mu12={-c.mu12:.17g}, lambda={c.g:.17g} /
"""


def input_text(c, s, end_fs, every_fs, m_parameter):
    if c.dse or c.nf != 2:
        raise ValueError('Author OneD_Photons2 baseline supports DSE OFF and nf=2 only')
    d, k = (2, 2) if s.perspective == 'el' else (1, 4)
    steps = int(math.ceil(end_fs/(c.dt*AU_FS)))
    dump = max(1, int(round(every_fs/(c.dt*AU_FS))))
    # Original output uses i4.4 filenames, so fail before their overflow.
    if steps//dump > 9999:
        raise ValueError('Too many native text frames; increase --every-fs')
    typ = 'CTMQC' if s.algorithm == 'ctmqc' else 'EHREN'
    sigma = f'{c.sigma_r:.17g}'+(f', {1/np.sqrt(c.omega):.17g}' if d==2 else '')
    masses = f'{c.mass:.17g}'+(', 1' if d==2 else '')
    text = f"""&system typ_cal='{typ}', model_potential='read_model', n_dof={d}, nstates={k},
 initial_condition_seed={s.seed}, polaritons_surf={'.true.' if d==1 else '.false.'},
 cavity_dof={2 if d==2 else 0}, cavity_w={c.omega:.17g}, M_parameter={m_parameter}, {m_parameter} /
&dynamics dt={c.dt:.17g}, final_time={steps*c.dt:.17g}, nesteps=1,
 ntraj={s.ntraj}, dump={dump}, n_init_BO=1, init_BOstate=2,
 r_init={c.r2:.17g}, 0, k_init=0, 0, sigmaR_init={sigma}, mass_input={masses} /
"""+qml_input(c, s.perspective)+"""&external_files positions_file='initial_positions.dat',
 momenta_file='initial_momenta.dat', output_folder='native' /
"""
    return text, steps, dump


def verify_model(build, out, c, view):
    """Reconstruct V and compare adiabatic gradients using the compiled QML."""
    out = Path(out)
    out.mkdir()
    d, k = (2, 2) if view=='el' else (1, 4)
    points = np.array([[2.,0.], [3.2,-1.2], [3.875,2.1], [4.3,-2.], [6.,.7]])[:, :d]
    np.savetxt(out/'probe_points.dat', points)
    text = f"&adapter_probe model='read_model',n={k},d={d} /\n"+qml_input(c, view)
    (out/'input.in').write_text(text)
    with (out/'input.in').open() as inp, (out/'run.log').open('w') as log:
        subprocess.run([str(build/'model_probe.x')], cwd=out, stdin=inp,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    values = np.loadtxt(out/'probe_values.dat', ndmin=2)
    if len(values) != len(points) or not np.isfinite(values).all():
        raise ValueError('Invalid QML probe output')
    want, grad = hamiltonian(c, view, points)
    errors, gerrors, nerrors = [], [], []
    for i,row in enumerate(values):
        v = row[:k*k].reshape((k,k), order='F')
        g = row[k*k:k*k*(1+d)].reshape((k,k,d), order='F')
        nac = row[k*k*(1+d):k*k*(1+2*d)].reshape((k,k,d), order='F')
        vec = row[k*k*(1+2*d):].reshape((k,k), order='F')
        if view=='pol':
            vec=vec[POL_PERM]
        errors.append(float(np.max(abs(vec@v@vec.T-want[i]))))
        expected_g = np.einsum('ik,dij,jl->kld', vec, grad[i], vec)
        gerrors.append(float(np.max(abs(np.diagonal(g,axis1=0,axis2=1)-np.diagonal(expected_g,axis1=0,axis2=1)))))
        gaps = np.diag(v)[None,:]-np.diag(v)[:,None]
        off = ~np.eye(k,dtype=bool)
        nerrors.append(float(np.max(abs((nac*gaps[:,:,None]-expected_g)[off]))))
    report = dict(hamiltonian_max_abs=max(errors), gradient_max_abs=max(gerrors),
                  nac_gradient_max_abs=max(nerrors), points=points.tolist())
    report['passed'] = max(errors+gerrors+nerrors) < 1e-9
    (out/'model_check.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise ValueError('QML Hamiltonian/gradient/NAC mismatch: '+str(report))
    return report


def run_case(task):
    c, s, out, args = task
    build = args.build.resolve()
    manifest = json.loads((build/'build_manifest.json').read_text())
    if any(digest(HERE/f)!=sha for f,sha in manifest['adapter_sha256'].items()):
        raise ValueError('Stale author build; run build --retry-build or use a fresh build directory')
    for f, key in (('source/src/main.x','binary_sha256'), ('model_probe.x','probe_sha256')):
        if digest(build/f) != manifest[key]:
            raise ValueError('Changed executable: '+f)
    text, steps, dump = input_text(c,s,args.end_fs,args.every_fs,args.m_parameter)
    out = out.resolve()
    if out.exists():
        raise FileExistsError('Refusing overwrite/resume; choose a new --out: '+str(out))
    out.mkdir(parents=True)
    started = time.monotonic()
    # File I/O uses fixed-basis vectors, so relative phases are kept exactly.
    initial = Engine(c,s).initial()
    coeff = initial['coeff'][:,POL_PERM] if s.perspective=='pol' else initial['coeff']
    np.savetxt(out/'initial_positions.dat',initial['X'],fmt='%.17e')
    np.savetxt(out/'initial_momenta.dat',initial['P'],fmt='%.17e')
    np.savetxt(out/'adapter_initial.dat',np.c_[coeff.real,coeff.imag],fmt='%.17e')
    (out/'input.in').write_text(text)
    (out/'config.json').write_text(json.dumps(asdict(c),indent=2)+'\n')
    settings = dict(backend='authors-gctmqc-instrumented',perspective=s.perspective,
        adapter_safety_version=3,boundary_mode='explicitly_open',zero_soc_initialized=True,
        ntraj=s.ntraj,seed=s.seed,initial=s.initial,algorithm=s.algorithm,
        m_parameter=args.m_parameter,force_history_threshold=.005,nesteps=1,
        dump_steps=dump,steps=steps,final_time_fs=steps*c.dt*AU_FS,
        source_manifest=manifest, dt_au=c.dt,initial_sample_sha256=digest(out/'initial_positions.dat'),
        capabilities={'exact_mcef_fields':False,'restart_from_python':False,'native_restart':False})
    settings['adapter_python_sha256']={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}
    (out/'author_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    # Upstream opens these files but never creates their parent directories.
    prepare_output_dirs(out)
    env = dict(os.environ,OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS='1')
    status = dict(status='RUNNING',backend=settings['backend'])
    def save_status():
        status['wall_seconds'] = time.monotonic()-started
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save_status()
    try:
        verify_model(build,out/'model_check',c,s.perspective)
        print('RUN G-CTMQC',out,flush=True)
        propagate_start = time.monotonic()
        with (out/'input.in').open() as inp, (out/'run.log').open('w') as log:
            run = subprocess.run([str(build/'source/src/main.x')],cwd=out,env=env,
                stdin=inp,stdout=log,stderr=subprocess.STDOUT)
        status['propagation_seconds'] = time.monotonic()-propagate_start
        # Fortran STOP may return success on errors. Inspect completion AND data.
        if run.returncode or 'finished successfully' not in (out/'run.log').read_text():
            raise RuntimeError('Author executable failed; see run.log')
        report=analyze_case(out,steps*c.dt*AU_FS)
        if report['max_normalized_norm_error'] > 1e-8 or report['max_surface_error'] > 1e-8 or report['max_mf_force_error'] > 1e-7:
            raise ValueError('Archive validation failed: '+str(report))
        if report['max_open_boundary_k_bound_excess'] > 1e-10:
            raise ValueError('Open-boundary quantum-momentum cutoff contract violated')
        status['binary_validated'] = True
        check_start = time.monotonic()
        validate_native_outputs(out)
        status['native_output_validated'] = True
        status['output_check_seconds'] = time.monotonic()-check_start
        status['numerical_warnings'] = report['warnings']
        status.update(status='COMPLETE_NOT_CERTIFIED',last_saved_fs=report['last_saved_fs'])
        save_status()
        if not args.no_render:
            from .render import render_case
            render_start=time.monotonic()
            try:
                render_case(out,movies=args.movies)
                status['render_seconds']=time.monotonic()-render_start
            except Exception as error:
                # A plotting failure must not relabel valid dynamics as failed.
                status['render_error']=repr(error)
                print('RENDER_FAILED (dynamics saved)',out,repr(error),flush=True)
            save_status()  # Include rendering in total wall time, not only TDSE-free postchecks.
        return str(out),report
    except BaseException as e:
        status.update(status='FAILED',error=repr(e))
        save_status()
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['nonadiabatic','rabi','all'],default='all')
    p.add_argument('--resonance',choices=['on','off','both'],default='both')
    p.add_argument('--perspective',choices=['el','pol','both'],default='both')
    p.add_argument('--case',choices=['zero','weak','strong','all'],default='strong')
    p.add_argument('--ntraj',type=int,default=128)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--end-fs',type=float,default=1.)
    p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--every-fs',type=float,default=.1)
    p.add_argument('--m-parameter',type=float,default=100.)
    p.add_argument('--initial',choices=['paper','bare'],default='paper')
    p.add_argument('--algorithm',choices=['ctmqc','ehrenfest'],default='ctmqc')
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--movies',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if not (args.build/'build_manifest.json').exists():
        raise FileNotFoundError('Build must complete first: '+str(args.build/'build_manifest.json'))
    if not np.isfinite([args.end_fs,args.dt,args.every_fs,args.m_parameter]).all() or min(args.end_fs,args.dt,args.every_fs,args.m_parameter)<=0 or min(args.jobs,args.threads)<1:
        raise ValueError('Need finite positive times, M and job/thread counts')
    choices=lambda value,all_value,options: options if value==all_value else [value]
    tasks=[]
    for model in choices(args.model,'all',['nonadiabatic','rabi']):
        for res in choices(args.resonance,'both',['on','off']):
            for case in choices(args.case,'all',['zero','weak','strong']):
                c=paper_config(model,res,dt=args.dt,g=dict(zero=0.,weak=.001,strong=.01)[case],dse=False,nf=2)
                for view in choices(args.perspective,'both',['el','pol']):
                    s=Settings(view,args.ntraj,args.seed,args.algorithm,args.initial)
                    s.validate()
                    target=args.out/model/res/'dse_off'/case/view
                    if target.exists():
                        raise FileExistsError('Existing case: '+str(target)+'; no overwrite or Python resume')
                    tasks.append((c,s,target,args))
    print(f'{len(tasks)} cases; native Gaussian matrix >= {8*args.ntraj**2/2**20:.1f} MiB / running case. No full wavefunctions.',flush=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_case,t):t[2] for t in tasks}
        for fut in as_completed(futures):
            try:
                out,report=fut.result()
                results.append(dict(path=out,validation=report))
                print('COMPLETE_NOT_CERTIFIED',out,flush=True)
            except Exception as e:
                results.append(dict(path=str(futures[fut]),error=repr(e)))
                print('FAILED',futures[fut],repr(e),flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'campaign.json').write_text(json.dumps(results,indent=2)+'\n')
    if any('error' in r for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
