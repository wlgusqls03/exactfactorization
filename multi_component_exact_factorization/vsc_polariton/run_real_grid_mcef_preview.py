"""Render actual real-grid TDSE data and diagnostic nested-MCEF fields.

Read-only input; always a NEW output folder under results/vsc_polariton.
CPU postprocessing, no propagation, no 11-GB GPU allocation. Outputs small
fields/JSON/PNG/PDF and optional event slideshow, never full phi/Lambda waves.
This does not certify Phase7 or claim uncoupled/coupled comparison from one case.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
from .real_grid_mcef_fields import analyze, diagnostics
from .phase7_support import budget_support, weighted_rms

ROOT=Path(__file__).resolve().parents[2]/'results/vsc_polariton'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def load_wave(path,packet,input_sha):
    """Check provenance and coordinate normalization before differentiating."""
    with np.load(path,allow_pickle=False) as z:
        if str(z['representation'])!='Psi_Q(R,x,Q)':raise ValueError('Not a native Q-grid wave')
        ident=json.loads(str(z['identity']))
        if ident['input_sha256']!=input_sha:raise ValueError('Wave and packet SHA mismatch')
        for k in ('R','x','mass','omega','g_chi','dx','dR'):
            np.testing.assert_allclose(z[k],packet[k],rtol=1e-12,atol=1e-12)
        Q=z['Q'];w=float(packet['omega'])
        np.testing.assert_allclose(z['q'],Q/np.sqrt(w),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.diff(Q),float(z['dQ']),rtol=1e-12,atol=1e-12)
        u=z['psi']
        if u.shape!=(len(packet['R']),len(packet['x']),len(Q)):raise ValueError('Wrong wave axes')
        if not np.isfinite(u).all():raise ValueError('Nonfinite wave')
        return u,Q,float(z['time_au'])


def compare_outer(base,other):
    """True final-frame resolution comparison; same R and time, no interpolation."""
    np.testing.assert_allclose(base['R'],other['R'],atol=1e-13,rtol=0)
    if abs(float(base['time_au'])-float(other['time_au']))>1e-12:raise ValueError('Time mismatch')
    dr=base['R'][1]-base['R'][0];mass=base['rho_R']*dr
    result={}
    for budget in (1e-6,1e-8,1e-10):
        a,_,_=budget_support(base['rho_R'],dr,budget)
        b,_,_=budget_support(other['rho_R'],dr,budget)
        mask=a&b
        result[str(budget)]={k:weighted_rms(base[k]-other[k],mass,mask)
                            for k in ('force','alpha','alpha_t','epsilon2_A','rho_R')}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path,help='Original Phase6 packet, read only')
    p.add_argument('--waves',required=True,type=Path,help='Campaign root containing full/base_Q384/full')
    p.add_argument('--observables',required=True,type=Path,help='base_Q384/full with observable_*.npz')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--movie',action='store_true')
    p.add_argument('--max-frames',type=int,help='Debug only: process first N saved waves')
    p.add_argument('--block',type=int,default=4)
    p.add_argument('--support-budget',type=float,default=1e-8)
    p.add_argument('--density-floor',type=float,default=1e-5,help='Plotting only, relative to each frame maximum')
    args=p.parse_args()
    if args.block<1 or (args.max_frames is not None and args.max_frames<1):p.error('Invalid count')
    if not 0<args.support_budget<1 or not 0<args.density_floor<1:p.error('Invalid mask')
    out=args.out.resolve();out.relative_to(ROOT.resolve())
    if out.exists():raise FileExistsError('Output is immutable; choose a new --out')
    waves=sorted((args.waves/'full/base_Q384/full').glob('wave_*.npz'))
    obs=sorted(args.observables.glob('observable_*.npz'))
    if not waves or not obs:raise FileNotFoundError('Need actual real-grid waves AND observables')
    sha=digest(args.input)
    with np.load(args.input,allow_pickle=False) as z:
        packet={k:z[k] for k in ('R','x','dR','dx','mass','omega','g_chi','potential','phi')}
    # Metadata validation before creating output. No simulation is launched.
    with np.load(waves[0],allow_pickle=False) as z:
        if json.loads(str(z['identity']))['input_sha256']!=sha:raise ValueError('Wrong packet')
    out.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    from .real_grid_mcef_plotting import style,dynamics,snapshots
    style()
    summary=dict(phase7_pass=False,status='DIAGNOSTIC_PREVIEW',input_sha256=sha,
        packet=str(args.input.resolve()),wave_root=str(args.waves.resolve()),
        scope='Single coupled case; no free comparison; natural gauge, no certified alpha=0 transform',
        missing_validation=['R-derivative resolution','temporal derivative probe','gauge transformation',
                            'uncoupled real-grid comparison','full support/field convergence'],
        source_hashes={name:digest(Path(__file__).with_name(name)) for name in
                      ('run_real_grid_mcef_preview.py','real_grid_mcef_fields.py','real_grid_mcef_plotting.py')},
        frames=[],refinements={})
    label=f"Full 3DOF | cavity {float(packet['omega'])*27211.386245988:.3f} meV"
    rows=[];times=[]
    for path in obs:
        with np.load(path,allow_pickle=False) as z:
            times.append(float(z['time_au']))
            rows.append({k:z[k] for k in ('rho_R','product','flux')})
    summary['dynamics']=dynamics(packet['R'],times,rows,out,label)
    all_frames=[]
    for i,path in enumerate(waves[:args.max_frames]):
        begin=time.monotonic();print('Analyze:',path,flush=True)
        u,Q,t=load_wave(path,packet,sha)
        with np.errstate(divide='ignore',invalid='ignore',over='ignore',under='ignore'):
            f=analyze(u,packet,Q,args.block)
        del u
        f['time_au']=t
        check=diagnostics(f)
        output=out/f'fields_{i:03d}.npz'
        np.savez_compressed(output,**f)
        check.update(wave=str(path),wave_sha256=digest(path),time_au=t,seconds=time.monotonic()-begin)
        (out/f'checks_{i:03d}.json').write_text(json.dumps(check,indent=2,allow_nan=False)+'\n')
        summary['frames'].append(check);all_frames.append(f)
        print(json.dumps(dict(time_au=t,seconds=check['seconds'],errors=check['budgets']['1e-08']['weighted_errors'])),flush=True)
    # Only a time-matched endpoint can test an available refinement.
    final=max(all_frames,key=lambda f:float(f['time_au']))
    for name in ('spacing_Q512','box28_Q448','dt00625_Q384'):
        candidates=sorted((args.waves/'full'/name/'full').glob('wave_*.npz'))
        if not candidates:
            summary['refinements'][name]={'status':'MISSING_WAVE'};continue
        path=candidates[-1]
        try:
            with np.load(path,allow_pickle=False) as z:t=float(z['time_au'])
        except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
            summary['refinements'][name]={'status':'INVALID_INPUT_NOT_CERTIFIED','error':str(exc),'path':str(path)}
            print('Invalid refinement archive; preserved, NOT certified:',path,flush=True)
            continue
        if abs(t-float(final['time_au']))>1e-12:
            summary['refinements'][name]={'status':'NO_COMMON_TIME'};continue
        print('Refinement:',path,flush=True)
        u,Q,t=load_wave(path,packet,sha)
        with np.errstate(divide='ignore',invalid='ignore',over='ignore',under='ignore'):
            f=analyze(u,packet,Q,args.block)
        del u
        f['time_au']=t;np.savez_compressed(out/('refinement_'+name+'.npz'),**f)
        summary['refinements'][name]=dict(status='DIAGNOSTIC_NOT_CERTIFIED',wave_sha256=digest(path),
            outer_weighted_errors=compare_outer(final,f),instantaneous=diagnostics(f))
    summary['plotting']=snapshots(all_frames,out,label,args.support_budget,args.density_floor,args.movie)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Saved:',out,flush=True)
    print('Phase7 NOT certified. Natural-gauge diagnostic figures, not a barrier-lowering claim.',flush=True)


if __name__=='__main__':main()
