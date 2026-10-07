"""Read-only saved-wave audit: quadrature, PG TDPES localization, same-wave sampling.

No propagation, tolerance changes, source-result edits, or certification upgrades.
Atomic units. Wave axes (R,q,electronic label); scalar fields (R,q) or (R,).
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import resample
from .model import Config, Propagator, derivative
from .factorization import analyze, diagnostics
from .run import atomic_json

CASES=('coupled','coupled_grid','coupled_dt','coupled_box','free','free_grid')
TIMES=(1000,1250)
THRESHOLDS=(1e-2,1e-4,1e-6,1e-8)
FIELDS=('epsilon1','cond','geo_q','geo_R','GD','a','b','Q_q','flow_q','Q_R','flow_R')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def integral_fourier(density, grid, lo, hi):
    """Integral of periodic trigonometric interpolant, not a Heaviside sum.

    1D real density [NR], grid [NR] uniform endpoint-excluded; result probability.
    For real even-length data the signed Nyquist convention has the same real integral.
    """
    h=grid[1]-grid[0]
    if not np.allclose(np.diff(grid),h) or not grid[0]<=lo<hi<=grid[0]+len(grid)*h+1e-10:
        raise ValueError('Invalid integration interval/uniform grid')
    c=np.fft.fft(density)/len(grid);k=2*np.pi*np.fft.fftfreq(len(grid),h)
    weights=np.empty(len(grid),complex);weights[0]=hi-lo
    weights[1:]=(np.exp(1j*k[1:]*(hi-grid[0]))-np.exp(1j*k[1:]*(lo-grid[0])))/(1j*k[1:])
    return float(np.sum(c*weights).real)


def populations(u,c):
    """R>=4 historical sum vs endpoint correction vs wave-first spectral integral.

    Last route evaluates the SAME Fourier wave on 2NR points before forming rho.
    It does not constitute independently propagated grid convergence.
    """
    R,q=c.grids();dr=R[1]-R[0];dq=q[1]-q[0]
    rho=np.sum(abs(u)**2,axis=(1,2))*dq;i=int(np.argmin(abs(R-4)))
    if abs(R[i]-4)>1e-9:raise ValueError('Half-weight comparison requires an R=4 grid node')
    historical=float(rho[R>=4].sum()*dr)
    half=historical-.5*dr*rho[i]+.5*dr*rho[0]
    fine=resample(u,2*c.nr,axis=0);Rf=np.linspace(c.rmin,c.rmax,2*c.nr,endpoint=False)
    rhof=np.sum(abs(fine)**2,axis=(1,2))*dq
    return dict(historical=historical,half_endpoint=float(half),
        spectral_native_density=integral_fourier(rho,R,4,c.rmax),
        spectral_wave2R=integral_fourier(rhof,Rf,4,c.rmax),
        rho_R4=float(rho[i]),norm=float(rho.sum()*dr),
        resampled_norm=float(rhof.sum()*dr/2),R_edge_density=float(rho[0]))


def fields(u,c):
    """Reuse native analyze plus four PG QHJ terms, Ha; no masked-field FFT.

    epsilon1 = Fqq/(2F) - a²/2 + FRR/(2MF) - b²/(2M).
    All derivatives act on the original spinor. Node values stay NaN.
    """
    p=Propagator(c);f=analyze(u,p);density=f['rho_qR']
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        for axis,h,key in [(0,p.dr,'R'),(1,p.dq,'q')]:
            du=derivative(u,h,axis);ddu=derivative(u,h,axis,2)
            first=np.sum((u.conj()*du).real,axis=-1)/density
            second=(np.sum((u.conj()*ddu).real,axis=-1)+np.sum(abs(du)**2,axis=-1))/density-first**2
            mass=c.mass if axis==0 else 1.
            f['Q_'+key]=np.where(density>0,second/(2*mass),np.nan)
        f['flow_q']=-f['a']**2/2;f['flow_R']=-f['b']**2/(2*c.mass)
    return f


def stats(delta,weight,mask,R=None,q=None):
    """Weighted error, nonfinite probability, and worst occupied point; no hiding NaN."""
    w=weight[mask];total=float(w.sum());valid=mask&np.isfinite(delta)
    result=dict(selected_mass_fraction=total/float(weight.sum()),
                nonfinite_mass_fraction=float(weight[mask&~np.isfinite(delta)].sum()/weight.sum()))
    if not valid.any():return dict(result,rms=None,max_abs=None)
    values=np.abs(delta[valid]);weights=weight[valid]
    result.update(rms=float(np.sqrt(np.average(values**2,weights=weights))),max_abs=float(values.max()))
    flat=int(np.argmax(np.where(valid,np.abs(delta),-1)))
    loc=np.unravel_index(flat,delta.shape)
    if R is not None:result['worst_R']=float(R[loc[0]])
    if q is not None:result['worst_q']=float(q[loc[1]])
    return result


@np.errstate(invalid='ignore',over='ignore')
def compare_fields(a,b):
    """Native-field linear comparison, thresholds + regions + decomposition residual.

    Same-grid data bypass interpolation. Density is NOT independently normalized.
    Reports correlations of errors, not a mechanistic causal attribution.
    """
    R,q=a['R'],a['q'];same=np.array_equal(R,b['R']) and np.array_equal(q,b['q'])
    rr,qq=np.meshgrid(R,q,indexing='ij');points=np.stack([rr,qq],axis=-1)
    def mapped(k):
        if same:return b[k].real
        return RegularGridInterpolator((b['R'],b['q']),b[k].real,bounds_error=False,fill_value=np.nan)(points)
    other=mapped('rho_qR');rho=a['rho_qR'];dr=R[1]-R[0];dq=q[1]-q[0]
    deltas={k:a[k].real-mapped(k) for k in FIELDS};report={'supports':{},'outer':{}}
    for eta in THRESHOLDS:
        mask=(rho>eta*rho.max())&(other>eta*np.nanmax(other))
        report['supports'][str(eta)]={k:stats(v,rho,mask,R,q) for k,v in deltas.items()}
    common=(rho>1e-6*rho.max())&(other>1e-6*np.nanmax(other))
    report['regions']={name:stats(deltas['epsilon1'],rho,common&mask,R,q) for name,mask in {
        'R_lt3':rr<3,'R_3to4':(rr>=3)&(rr<4),'R_ge4':rr>=4,
        'absq_lt2':abs(qq)<2,'absq_2to6':(abs(qq)>=2)&(abs(qq)<6),'absq_ge6':abs(qq)>=6}.items()}
    for key in ['epsilon2','force','alpha']:
        vals=np.interp(R,b['R'],b[key].real,left=np.nan,right=np.nan)
        rmask=(a['rho_R']>1e-6*a['rho_R'].max())&(np.interp(R,b['R'],b['rho_R'],left=0,right=0)>1e-6*b['rho_R'].max())
        report['outer'][key]=stats(a[key].real-vals,a['rho_R'],rmask,R)
    ok=np.isfinite(other)
    report['rho_L1']=float(np.sum(abs(rho[ok]-other[ok]))*dr*dq)
    report['outside_common_grid_mass']=float(rho[~ok].sum()*dr*dq)
    closure=deltas['epsilon1']-sum(deltas[k] for k in ['Q_q','flow_q','Q_R','flow_R'])
    report['QHJ_error_closure']=stats(closure,rho,common,R,q)
    return report


def same_wave(u,c):
    """2x q/R Fourier evaluation, then derivatives; not new physical propagation."""
    fine=resample(resample(u,2*c.nr,axis=0),2*c.nq,axis=1)
    cf=replace(c,nr=2*c.nr,nq=2*c.nq)
    ff=fields(fine,cf)
    sampled={k:(v[::2,::2] if v.ndim==2 else v[::2]) if isinstance(v,np.ndarray) and v.ndim in (1,2) else v for k,v in ff.items()}
    sampled['fine_sampling_norm']=float(ff['rho_R'].sum()*(c.rmax-c.rmin)/(2*c.nr))
    return sampled


def preflight(root,times):
    """Require ALL requested inputs and original numerical sources before any work."""
    inputs={};configs={};required=[]
    for name in CASES:
        folder=root/name;ident_path=folder/'identity.json';required.append(ident_path)
        ident=json.loads(ident_path.read_text());configs[name]=Config(**ident['config'])
        for source in ['model.py','factorization.py']:
            if sha(Path(__file__).parent/source)!=ident['source'][source]:
                raise ValueError(f'Original numerical source changed: {source}; refusing implicit reinterpretation')
        for t in times:
            path=folder/'waves'/f'wave_{t:04d}.npz'
            if not path.is_file() or path.is_symlink():raise FileNotFoundError(path)
            with np.load(path,allow_pickle=False) as z:
                if float(z['time_au'])!=t or z['psi'].shape!=(configs[name].nr,configs[name].nq,2):
                    raise ValueError(f'Wrong wave time/shape: {path}')
            required.append(path)
        for filename in ['status.json','observables.json','ef_diagnostics.json','backend_check.json']:
            if (folder/filename).exists():required.append(folder/filename)
    for path in required:inputs[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=sha(path))
    return configs,inputs


def audit(root,out,times=TIMES,refine=True,pack=True):
    """Serial CPU event analysis, bounded RAM, outputs only a fresh directory."""
    root=Path(root).resolve();out=Path(out).resolve()
    if out==root or root in out.parents:raise ValueError('Audit output must be outside the original campaign')
    if out.exists():raise FileExistsError('Preserve existing audit; use a NEW --out')
    configs,inputs=preflight(root,times)
    if shutil.disk_usage(out.parent).free<512*1024**2:raise OSError('Need 512 MiB free for the audit package')
    out.mkdir();start=time.perf_counter();summaries={};comparisons=[]
    for t in times:
        for name,c in configs.items():
            print(f'Analyze {name}, t={t} au (no propagation)',flush=True)
            with np.load(root/name/'waves'/f'wave_{t:04d}.npz') as z:u=z['psi']
            f=fields(u,c);summary=dict(case=name,time_au=t,population=populations(u,c),native=diagnostics(f))
            if refine:
                refined=same_wave(u,c)
                summary['same_wave_2x']=compare_fields(f,refined)
                summary['same_wave_2x']['fine_sampling_norm']=refined['fine_sampling_norm']
                del refined
            summaries[f'{name}_{t}']=summary;atomic_json(out/f'{name}_{t}.json',summary)
            del f,u
        for left,right in [('coupled','coupled_grid'),('coupled','coupled_dt'),('coupled','coupled_box'),('free','free_grid')]:
            with np.load(root/left/'waves'/f'wave_{t:04d}.npz') as z:fa=fields(z['psi'],configs[left])
            with np.load(root/right/'waves'/f'wave_{t:04d}.npz') as z:fb=fields(z['psi'],configs[right])
            report=compare_fields(fa,fb)
            pa=summaries[f'{left}_{t}']['population'];pb=summaries[f'{right}_{t}']['population']
            report['population_differences']={k:pa[k]-pb[k] for k in ['historical','half_endpoint','spectral_native_density','spectral_wave2R']}
            report.update(left=left,right=right,time_au=t);comparisons.append(report);del fa,fb
    # Confirm that every input is still byte-identical; never rewrite original gates.
    for name,record in inputs.items():
        if sha(root/name)!=record['sha256']:raise RuntimeError(f'Input changed during audit: {name}')
    summary=dict(status='DIAGNOSTIC_COMPLETE_NOT_CERTIFIED',phase_pass=False,times_au=list(times),
        same_wave_refinement=refine,comparisons=comparisons,inputs=inputs,
        elapsed_seconds=time.perf_counter()-start,
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('factorization.py')]},
        note='No propagation or gate upgrade. Same-wave oversampling cannot prove propagated basis convergence. No scalar offset fitting.')
    atomic_json(out/'audit_summary.json',summary)
    if pack:
        archive=out/'model_a_event_review.tar.gz'
        with tarfile.open(archive,'x:gz') as tar:
            for name in inputs:tar.add(root/name,arcname='campaign/'+name,recursive=False)
            for p in sorted(out.glob('*.json')):tar.add(p,arcname='audit/'+p.name,recursive=False)
        print(f'Done: {archive} ({archive.stat().st_size/1024**2:.1f} MiB)',flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--times',type=int,nargs='+',default=list(TIMES))
    ap.add_argument('--no-same-wave-refinement',action='store_true',help='Diagnostic fallback; marks refinement omitted')
    args=ap.parse_args()
    audit(args.campaign,args.out,tuple(sorted(set(args.times))),not args.no_same_wave_refinement)


if __name__=='__main__':main()
