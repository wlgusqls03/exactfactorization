"""Separate, resumable PEN/VSC workflow. No historical output is overwritten."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import zipfile

import numpy as np

from .backend import Hamiltonian, memory_estimate
from .core import configuration, photon_grid, initial_wave, DEFAULTS


AU_PER_FS = 41.3413745758
ROOT = Path(__file__).resolve().parents[1]
STOP = False


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,default=_json,allow_nan=False)+'\n')
    os.replace(temp,path)


def write_npz(path, **arrays):
    """Lossless fast deflate, atomic replace. Avoid level-9 compression stalls."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for key,value in arrays.items():
            with archive.open(key+'.npy','w',force_zip64=True) as stream:
                np.lib.format.write_array(stream,np.asarray(value),allow_pickle=False)
    os.replace(temp,path)


def save_tree(path, tree):
    arrays={}
    def encode(v):
        if isinstance(v,np.ndarray):
            name='array_'+str(len(arrays)); arrays[name]=v
            return {'__ndarray__':name}
        if isinstance(v,dict): return {k:encode(w) for k,w in v.items()}
        if isinstance(v,(list,tuple)): return [encode(w) for w in v]
        if isinstance(v,np.generic): return v.item()
        return v
    arrays['schema_json']=np.array(json.dumps(encode(tree),allow_nan=False))
    write_npz(path,**arrays)


def load_tree(path):
    with np.load(path,allow_pickle=False) as z:
        def decode(v):
            if isinstance(v,dict) and set(v)=={'__ndarray__'}: return z[v['__ndarray__']].copy()
            if isinstance(v,dict): return {k:decode(w) for k,w in v.items()}
            if isinstance(v,list): return [decode(w) for w in v]
            return v
        return decode(json.loads(str(z['schema_json'])))


def source_record():
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob('*.py'))}
    for rel in ['multi_component_exact_factorization/core.py',
                'multi_component_exact_factorization_discrete_gpu/spectral_tdse.py']:
        sources[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    physics_names={'core.py','backend.py','calibration.py','fields.py','projections.py','spectral_tdse.py'}
    physics={k:v for k,v in sources.items() if Path(k).name in physics_names}
    return dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sources=sources,physics_sources=physics,python=os.sys.version,numpy=np.__version__,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','unset'))


def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,default=_json).encode()).hexdigest()


def cached_prepare(config, cache):
    from .calibration import prepare
    import fcntl
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    model_config={k:v for k,v in config.items() if k in DEFAULTS or k in ('local_wall_R','dividing_R')}
    model_sources={k:v for k,v in source_record()['sources'].items() if Path(k).name in ('core.py','calibration.py')}
    key=fingerprint(dict(config=model_config,sources=model_sources))[:20]
    path=cache/('molecular_'+key+'.npz')
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            print('CACHE REUSE',path,flush=True)
            packet=load_tree(path);packet['config']=dict(config)
            return packet
        print('CALIBRATE bare BO + local vibrational states',path,flush=True)
        packet=prepare(config)
        save_tree(path,packet)
        write_json(cache/('calibration_'+key+'.json'),packet['calibration'])
        return packet


def resources(config, *, gpu=False, jobs=1, frames=201):
    shape=(config['nR'],config['nx'],config['nQ'])
    estimate=memory_estimate(shape,config.get('block',8))
    # Compact maps plus vectors; debug saves complex128 maps and costs more.
    map_bytes=120*16 if config.get('debug_fields',False) else 32*4
    estimate['disk_worst_bytes']=jobs*(frames*config['nR']*config['nQ']*map_bytes+
                                     2*estimate['wave_bytes']+128*2**20)
    raw_every=int(config.get('save_wave_every',0))
    if raw_every>0:
        steps=int(np.ceil(config.get('t_final_fs',.02)*AU_PER_FS/config.get('dt_au',.025)))
        estimate['disk_worst_bytes']+=jobs*(steps//raw_every+1)*estimate['wave_bytes']
    estimate['jobs']=jobs
    if gpu:
        import cupy as cp
        free,total=cp.cuda.runtime.memGetInfo()
        estimate.update(gpu_free_bytes=int(free),gpu_total_bytes=int(total))
        if jobs*estimate['gpu_reservation_bytes']>.85*free:
            raise MemoryError('Requested concurrent cases exceed GPU reservation: '+str(estimate))
    return estimate


def compact_fields(fields, debug=False):
    """Display cache is explicitly float32; validation metrics use float64.

    Full wave checkpoints and observable populations retain complex128/float64.
    Reconstructing new high-precision fields at arbitrary old times requires an
    optional raw wave snapshot or a deterministic resumed/rerun propagation.
    """
    if debug: return fields
    maps={'rho_qR','lambda_density','a','b','epsilon1','epsilon1_GI','epsilon1_GD',
          'S_q','S_adv','S_rel','density_dt','lambda_dt',
          'epsilon1_cond','epsilon1_qgeo','epsilon1_Rgeo',
          'epsilon1_bare','epsilon1_LM','epsilon1_DSE','BO_character',
          'cbo_character','cbo_outside'}
    out={}
    for key,value in fields.items():
        v=np.asarray(value)
        if v.ndim<=1 and v.dtype.kind not in 'OU': out[key]=v
        elif key in maps: out[key]=np.asarray(v.real,dtype=np.float32)
    out['cache_precision']=np.array('float32 maps for display; validation metrics computed before cast')
    return out


def _stop(signum, frame):
    global STOP
    STOP=True


def run_case(config, output, *, campaign='pen', case='zero', gpu=False,
             resume=False, cache=None, movies=True):
    """Run one case and automatically extract/render; coarse runs never certified."""
    from .fields import analyze,metrics
    from .projections import build_analysis,project
    global STOP
    STOP=False
    c=configuration(config); out=Path(output)
    for key,default in (('dt_au',.025),('field_every_fs',.2),('checkpoint_every_fs',5.)):
        if not np.isfinite(c.get(key,default)) or c.get(key,default)<=0:
            raise ValueError(key+' must be positive and finite')
    if not np.isfinite(c.get('t_final_fs',.002)) or c.get('t_final_fs',.002)<0:
        raise ValueError('t_final_fs must be nonnegative and finite')
    identity={k:v for k,v in c.items() if k not in ('t_final_fs','movies')}
    identity.update(campaign=campaign,case=case)
    identity_hash=fingerprint(identity)
    if out.exists() and (out/'config.json').exists():
        if not resume: raise FileExistsError('Existing case preserved; choose new directory or --resume: '+str(out))
        old=json.loads((out/'config.json').read_text())
        if old['identity_hash']!=identity_hash: raise ValueError('Resume configuration mismatch')
    elif out.exists() and any(p.name!='.run.lock' for p in out.iterdir()):
        raise FileExistsError('Nonempty unowned output directory: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    run_lock=(out/'.run.lock').open('a')
    try: fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close()
        raise RuntimeError('This case is already running: '+str(out))
    source=source_record()
    # Validate before replacing ANY provenance/configuration of an old run.
    if resume:
        with np.load(out/'restart.npz',allow_pickle=False) as z:
            if str(z['identity_hash'])!=identity_hash:
                raise ValueError('Checkpoint identity mismatch')
            if str(z['source_hash'])!=fingerprint(source['physics_sources']):
                raise ValueError('Checkpoint source changed: use a new directory; old provenance preserved')
            target_step=int(np.ceil(c.get('t_final_fs',.002)*AU_PER_FS/c.get('dt_au',.025)))
            saved_steps=[int(p.stem.split('_')[-1]) for p in (out/'fields').glob('fields_*.npz')]
            if target_step<max([int(z['step'])]+saved_steps):
                raise ValueError('Resume final time precedes existing history; old files preserved')
    packet=cached_prepare(c, cache or out.parent/'cache')
    packet['dse']=c.get('dse',True)
    if campaign=='vsc' and not packet.get('vsc_available',True):
        raise ValueError('VSC requires an explicitly chosen local-state preparation; see calibration.json')
    omega=float(packet['omega_'+campaign])*(1+float(c.get('detuning',0.)))
    g=float(c.get('g_cavity',packet['g_'+campaign][case]))
    Q=photon_grid(c); dt=float(c.get('dt_au',.025))
    final=float(c.get('t_final_fs',.002))
    nsteps=int(np.ceil(final*AU_PER_FS/dt))
    if final>.1:
        regression=Path(c.get('legacy_regression_file',str(out.parent/'regression'/'regression.json')))
        if not regression.exists() or json.loads(regression.read_text()).get('status')!='PASS':
            raise RuntimeError('Legacy regression must PASS before long propagation: '+str(regression))
        gate=Path(c.get('validation_dir',str(out.parent/'validation')))/f'validation_{campaign}_{case}.json'
        if not gate.exists():
            raise RuntimeError('Long run requires matching short validation first: '+str(gate))
        proof=json.loads(gate.read_text())
        if (proof.get('status')!='SHORT_TIME_CHECK_PASS' or proof.get('config_hash')!=fingerprint(c)
                or proof.get('source_hash')!=fingerprint(source_record()['physics_sources'])):
            raise RuntimeError('Short validation missing, failed, or stale for this configuration/source')
    every=max(1,round(float(c.get('field_every_fs',.2))*AU_PER_FS/dt))
    checkpoint_every=max(every,round(float(c.get('checkpoint_every_fs',5.))*AU_PER_FS/dt))
    estimate=resources(c,gpu=gpu,frames=nsteps//every+2)
    if shutil.disk_usage(out).free<estimate['disk_worst_bytes']+256*2**20:
        raise OSError('Not enough disk for worst-case compact outputs + atomic checkpoint: '+str(estimate))
    write_json(out/'config.json',dict(c,campaign=campaign,case=case,omega=omega,
                                    g_cavity=g,identity_hash=identity_hash))
    write_json(out/'metadata.json',dict(**source,calibration=packet['calibration'],resources=estimate,
                                     status='NOT_CONVERGENCE_CERTIFIED',axis_order=['R','x','Q'],
                                     coordinate='Q=sqrt(omega)*q_c',gauge='positive marginal',
                                     field_precision='float32 display maps; float64 validation',
                                     propagation_basis='full real-space electron + R/Q grids',
                                     dse=c.get('dse',True)))
    analysis_path=out/'analysis_cache.npz'
    if analysis_path.exists() and resume:
        print('CACHE REUSE',analysis_path,flush=True); analysis=load_tree(analysis_path)
    else:
        analysis=build_analysis(packet,omega,g,Q,c)
        save_tree(analysis_path,analysis)
    # Static references are analysis-only; errors never remove a TDSE case.
    try:
        from .projections import references
        if not (out/'references.npz').exists():
            refs=references(packet,Q,omega,g,c,analysis=analysis)
            write_npz(out/'references.npz',**refs)
    except (ImportError,ValueError,np.linalg.LinAlgError) as exc:
        write_json(out/'reference_warning.json',dict(warning=str(exc)))
    h=Hamiltonian(packet,Q,omega,g,gpu=gpu,dse=c.get('dse',True),block=c.get('block',8))
    checkpoint=out/'restart.npz'; rows=[]; start=0; e0=None
    if resume:
        with np.load(checkpoint,allow_pickle=False) as z:
            if str(z['identity_hash'])!=identity_hash: raise ValueError('Checkpoint identity mismatch')
            if str(z['source_hash'])!=fingerprint(source['physics_sources']):
                raise ValueError('Checkpoint source changed: rerun in a new directory')
            u=h.array(z['psi']); start=int(z['step']); e0=float(z['energy0'])
        journal=sorted((out/'observables').glob('obs_*.npz'))
        if journal:
            for path in journal:
                if int(path.stem.split('_')[-1])<=start:
                    with np.load(path,allow_pickle=False) as z:
                        rows.append({k:z[k].copy() for k in z.files})
        elif (out/'observables.npz').exists():
            with np.load(out/'observables.npz',allow_pickle=False) as z:
                for i,s in enumerate(z['step']):
                    if s<=start: rows.append({k:z[k][i] for k in z.files if k not in ('x','R','Q')})
    else: u=h.array(initial_wave(packet,Q,campaign))
    if start>nsteps: raise ValueError('Requested final time precedes restart')
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,_stop)
    wall=time.monotonic(); latest_host=None; latest_step=-1
    def save_checkpoint(step):
        host=latest_host if latest_step==step else h.host(u)
        write_npz(checkpoint,psi=host,step=step,energy0=e0,identity_hash=identity_hash,
                  source_hash=fingerprint(source['physics_sources']))
        if rows:
            data={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
            write_npz(out/'observables.npz',**data,x=packet['x'],R=packet['R'],Q=Q)
    try:
        for step in range(start,nsteps+1):
            sample=step%every==0 or step==nsteps or (step==start and not rows)
            if sample and not (rows and int(rows[-1]['step'])==step):
                if fingerprint(source_record()['physics_sources'])!=fingerprint(source['physics_sources']):
                    raise RuntimeError('Physics source changed during this run; checkpoint preserved, do not git pull during propagation')
                action=h.action(u)
                ob=h.observe(u,action)
                if e0 is None: e0=ob['energy']
                if not np.isfinite(ob['energy']) or abs(ob['norm']-1)>c.get('norm_tolerance',1e-7):
                    raise FloatingPointError('Norm/nonfinite gate failed; last valid checkpoint preserved')
                if abs(ob['energy']-e0)>c.get('energy_tolerance',1e-5):
                    raise FloatingPointError('Energy-drift gate failed; refine dt before production')
                if max(ob['edge_R'],ob['edge_Q'],ob['edge_x'])>c.get('edge_tolerance',1e-5):
                    raise FloatingPointError('Boundary probability gate failed; enlarge/refine box before continuing')
                latest_host=h.host(u); latest_step=step
                projection=project(latest_host,packet,analysis,Q,include_cbo=True)
                cbo_maps={key:projection.pop(key) for key in ('cbo_character','cbo_outside')}
                ob.update(projection)
                fields=analyze(latest_host,packet,Q,omega,g,h.host(action),support=c.get('support',1e-5))
                fields.update(cbo_maps)
                fields.update(t_fs=np.array(step*dt/AU_PER_FS),t_au=np.array(step*dt))
                met=metrics(fields)
                ob.update(step=step,t_fs=step*dt/AU_PER_FS,t_au=step*dt,
                          energy_drift=ob['energy']-e0)
                # Signed-current integrals are backflow diagnostics, not a
                # claim to count individually identifiable quantum crossings.
                if rows:
                    prev=rows[-1]; delta=ob['t_au']-prev['t_au']
                    ob['forward_flux_integral']=prev['forward_flux_integral']+.5*delta*(
                        max(prev['signed_flux'],0)+max(ob['signed_flux'],0))
                    ob['backflow_integral']=prev['backflow_integral']+.5*delta*(
                        max(-prev['signed_flux'],0)+max(-ob['signed_flux'],0))
                else:
                    ob.update(forward_flux_integral=0.,backflow_integral=0.)
                rows.append(ob)
                # Append-only small journal avoids recompressing the complete
                # growing history at every frame (quadratic IO). Consolidate
                # only at checkpoint/final; resume filters journal by step.
                write_npz(out/'observables'/f'obs_{step:07d}.npz',**ob)
                write_npz(out/'fields'/f'fields_{step:07d}.npz',**compact_fields(fields,c.get('debug_fields',False)))
                write_json(out/'metrics'/f'metrics_{step:07d}.json',met)
                if c.get('save_wave_every',0) and step%int(c['save_wave_every'])==0:
                    write_npz(out/'waves'/f'wave_{step:07d}.npz',psi=latest_host,step=step)
                write_json(out/'status.json',dict(status='RUNNING',step=step,total_steps=nsteps,
                     t_fs=ob['t_fs'],norm=ob['norm'],energy_drift=ob['energy_drift'],
                     wall_seconds=time.monotonic()-wall,completed_fields=len(rows)))
                print(campaign,case,step,'/',nsteps,'t_fs=',round(ob['t_fs'],6),
                      'dE=',ob['energy_drift'],flush=True)
                del action,fields
            if step%checkpoint_every==0 or step==nsteps or STOP:
                save_checkpoint(step)
            if STOP: break
            if step<nsteps: u=h.step(u,dt)
        status=dict(status='INTERRUPTED' if STOP else 'COMPLETE_NOT_CERTIFIED',
                    step=step,total_steps=nsteps,t_fs=step*dt/AU_PER_FS,
                    wall_seconds=time.monotonic()-wall)
        write_json(out/'status.json',status)
    except Exception as exc:
        write_json(out/'failure.json',dict(error=repr(exc),step=step,
                                         last_checkpoint_preserved=checkpoint.exists()))
        raise
    if not STOP:
        from .render import render_case
        result=render_case(out,movies=movies,fps=c.get('fps',20),dpi=c.get('dpi',110))
        write_json(out/'render_manifest.json',result)
    run_lock.close()
    return status


def _case_job(job):
    return run_case(**job)


def run_campaign(config,out,*,campaigns=('pen','vsc'),cases=('zero','weak','strong'),
                 jobs=1,gpu=False,resume=False,movies=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=configuration(config)
    resources(c,gpu=gpu,jobs=jobs)
    packet=cached_prepare(c,out/'cache')
    from .schematics import render_schematics
    render_schematics(out/'schematics',packet['calibration'])
    arguments=[dict(config=c,output=out/campaign/case,campaign=campaign,case=case,
                    gpu=gpu,resume=resume,cache=out/'cache',movies=movies)
               for campaign in campaigns for case in cases]
    estimated_frames=int(c.get('t_final_fs',.02)/c.get('field_every_fs',.2))+2
    disk=resources(c,jobs=len(arguments),frames=estimated_frames)['disk_worst_bytes']
    if shutil.disk_usage(out).free<disk+256*2**20:
        raise OSError('Insufficient disk for ALL requested cases; select one case or reduce field sampling: '+str(disk))
    if jobs==1: results=[_case_job(j) for j in arguments]
    else:
        # Spawn avoids inheriting a live CUDA context or BLAS worker pool.
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=jobs,mp_context=mp.get_context('spawn')) as pool:
            results=list(pool.map(_case_job,arguments))
    from .render import render_comparison
    render_comparison(out)
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['init','calibrate','resources','run','campaign','render'])
    p.add_argument('--config',type=Path)
    p.add_argument('--out',type=Path,default=Path('results')/datetime.now().strftime('%Y%m%d')/'cavity_extension_v1')
    p.add_argument('--preset',choices=['native','smoke'],default='native')
    p.add_argument('--campaign',choices=['pen','vsc','both'],default='pen')
    p.add_argument('--case',choices=['zero','weak','strong','all'],default='all')
    p.add_argument('--jobs',type=int,choices=[1,2,3],default=1)
    p.add_argument('--gpu',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--no-movies',action='store_true')
    p.add_argument('--report-out',type=Path,
                   help='render only: separate new report directory; preserve original report')
    p.add_argument('--references',type=Path,
                   help='render only: optional separately refined static reference cache')
    p.add_argument('--set',action='append',default=[],metavar='KEY=JSON')
    a=p.parse_args(argv)
    c=configuration(json.loads(a.config.read_text()) if a.config else {})
    if a.preset=='smoke':
        c.update(nx=96,nR=96,nQ=24,Q_max=7.,n_bo=3,n_vib=4,n_fock_analysis=4,
                 dt_au=.01,t_final_fs=.002,field_every_fs=.0008,checkpoint_every_fs=.001,
                 local_state_method='single_well',validation_scope='coarse numerical smoke only',dpi=80)
    else:
        c.setdefault('dt_au',.025);c.setdefault('t_final_fs',.02)
    for item in a.set:
        key,value=item.split('=',1);c[key]=json.loads(value)
    c=configuration(c)
    if a.command=='init':
        if a.out.exists(): raise FileExistsError(a.out)
        write_json(a.out,c);print(a.out);return
    if a.command=='resources': print(json.dumps(resources(c,gpu=a.gpu,jobs=a.jobs),indent=2));return
    if a.command=='calibrate':
        packet=cached_prepare(c,a.out/'cache');write_json(a.out/'calibration.json',packet['calibration']);return
    if a.command=='render':
        from .render import render_case
        render_case(a.out,movies=not a.no_movies,output_dir=a.report_out,
                    reference_path=a.references);return
    campaigns=['pen','vsc'] if a.campaign=='both' else [a.campaign]
    cases=['zero','weak','strong'] if a.case=='all' else [a.case]
    if a.command=='run':
        if len(campaigns)!=1 or len(cases)!=1: p.error('run requires one --campaign and one --case')
        run_case(c,a.out,campaign=campaigns[0],case=cases[0],gpu=a.gpu,resume=a.resume,movies=not a.no_movies)
    else:
        run_campaign(c,a.out,campaigns=campaigns,cases=cases,jobs=a.jobs,gpu=a.gpu,
                     resume=a.resume,movies=not a.no_movies)


if __name__=='__main__': main()
