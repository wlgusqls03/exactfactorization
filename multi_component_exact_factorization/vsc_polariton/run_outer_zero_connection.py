"""Outer-only VSC gauge audit. All existing data read-only; no TDSE replay.

fields: force-work visualization, NOT certified transformed wavefunction.
events: independent saved-wave alpha=0 checks using instantaneous H Psi.
No first-level connection is removed. No spatially varying offset alignment.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .outer_zero_connection import integrate_anchor,fixed_outer_fields,validate_wave
from .qhj_saved_wave import outer_qhj
from .run_real_grid_mcef_preview import load_wave,digest
from .real_grid_mcef_fields import action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('fields','events'),required=True)
    for k in ('input','source','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--anchor-R',type=float,default=-1.7375)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--max-events',type=int,default=3)
    p.add_argument('--steps',type=int,nargs='+',help='Explicit saved event steps, no new propagation')
    a=p.parse_args()
    if a.fps<1 or a.max_events<1:p.error('Counts must be positive')
    root=Path(__file__).resolve().parents[2]/'results/vsc_polariton'
    a.out.resolve().relative_to(root.resolve())
    os.environ.setdefault('MPLCONFIGDIR',str(root/'movie_cache/matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'movie_cache'))
    with np.load(a.input) as z:packet={k:z[k] for k in z.files}
    R=packet['R'];anchor=int(np.argmin(abs(R-a.anchor_R)));report=[];frames=[]
    paths=sorted(a.source.glob('fields_*.npz' if a.mode=='fields' else 'wave_*.npz'))
    if not paths:raise FileNotFoundError('No matching input frames')
    if a.steps:
        paths=[p for p in paths if int(p.stem.split('_')[-1]) in a.steps]
        if len(paths)!=len(set(a.steps)):raise FileNotFoundError('Missing requested steps')
    if a.mode=='events':paths=[paths[i] for i in np.unique(np.linspace(0,len(paths)-1,min(a.max_events,len(paths))).astype(int))]
    a.out.mkdir(parents=True,exist_ok=False)
    for path in paths:
        if a.mode=='fields':
            with np.load(path) as f:
                np.testing.assert_allclose(f['R'],R,atol=1e-12,rtol=0)
                np.testing.assert_allclose(f['Q'],np.sqrt(float(packet['omega']))*f['q'])
                rho=f['rho_R'];force=f['force'];t=float(f['time_au']);ep=f['epsilon2_A'].real
            frame=dict(R=R,rho=rho,time_au=t,PG=ep);rows=[]
            for budget in (1e-6,1e-8,1e-10):
                work,trap,meta,s=integrate_anchor(R,-force,rho,anchor,budget)
                meta.update(budget=budget,quadrature_max=float(np.max(abs(work[s]-trap[s]))) if s.any() else None)
                rows.append(meta)
                if budget==1e-8:frame.update(work=work,support=s)
            frames.append(frame);report.append(dict(time_au=t,support_checks=rows))
            np.savez(a.out/path.name.replace('fields_','work_'),**frame)
        else:
            psi,Q,t=load_wave(path,packet,digest(a.input));omega=float(packet['omega'])
            u=psi*omega**.25;del psi
            q=Q/np.sqrt(omega);volume=float(packet['dx'])*(q[1]-q[0]);M=float(packet['mass'])
            Hu=action(u,packet,q)
            outer=outer_qhj(u,Hu,float(packet['dR']),volume,M);ut=-1j*Hu;del Hu
            rows=[];small=dict(R=R,rho=outer['rho_R'],time_au=t,alpha=outer['alpha'],alpha_t=outer['alpha_t'],force=outer['EF_force'])
            for budget in (1e-6,1e-8,1e-10):
                g,meta=fixed_outer_fields(R,outer['rho_R'],outer['alpha'],outer['alpha_t'],outer['epsilon2'],outer['EF_force'],anchor,budget)
                check=validate_wave(u,ut,R,volume,M,outer,g)
                check.update(budget=budget,support=meta);rows.append(check)
                if budget==1e-8:small.update(g)
            del u,ut
            report.append(dict(time_au=t,checks=rows))
            np.savez(a.out/path.name.replace('wave_','gauge_'),**small)
        print(a.mode,path.name,'t_fs',t*.024188843265857,flush=True)
        (a.out/'validation.json').write_text(json.dumps(dict(mode=a.mode,anchor_R=float(R[anchor]),
            status='FORCE_WORK_ONLY' if a.mode=='fields' else 'INCOMPLETE_AUDIT',phase7_pass=False,
            source=str(a.source.resolve()),packet_sha256=digest(a.input),frames=report),indent=2))
    if a.mode=='events':
        passed=all(c['pass_gate'] for row in report for c in row['checks'])
        summary=json.loads((a.out/'validation.json').read_text())
        summary['status']='SELECTED_SNAPSHOTS_PASS' if passed else 'SELECTED_SNAPSHOTS_FAIL'
        summary['temporal_gauge_certified']=False
        (a.out/'validation.json').write_text(json.dumps(summary,indent=2))
        print(summary['status']);return
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    times=np.array([f['time_au'] for f in frames])*.024188843265857
    if len(times)<2 or np.any(np.diff(times)<=0) or max(np.diff(times))>.2:
        raise ValueError('Dense actual time sequence required for movie; arrays preserved')
    fig,axs=plt.subplots(2,1,figsize=(10,8),layout='constrained')
    writer=FFMpegWriter(fps=a.fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2'])
    peak=max(float(f['rho'].max()) for f in frames)
    with writer.saving(fig,str(a.out/'outer_force_work.mp4'),100):
        for i,f in enumerate(frames):
            for ax in axs:ax.clear()
            m=f['support']&(f['rho']>1e-4*peak)
            value=f['work']*27.211386245988
            axs[0].plot(R,np.where(m,value,np.nan),color='#327ba0')
            cut=m&(abs(value)>2)
            if cut.any():axs[0].text(.02,.95,f'{cut.sum()} occupied grid points outside zoom',transform=axs[0].transAxes,va='top',fontsize=9)
            axs[0].set(ylabel='Force-work potential (eV)',ylim=(-2.1,2.1))
            axs[1].plot(R,f['rho'],color='black');axs[1].set(ylabel=r'$|\chi|^2$',ylim=(0,1.05*peak))
            for ax in axs:ax.axvline(0,color='gray',ls=':');ax.set(xlim=(R[0],R[-1]),xlabel=r'$R$ ($a_0$)')
            fig.suptitle(f'Outer force-work potential | t={times[i]:.3f} fs | {float(packet["omega"])*27211.386245988:.3f} meV')
            fig.supxlabel(f'Fixed anchor R={R[anchor]:.4f}; no bridging across excluded support.\nForce integral only, NOT certified wavefunction gauge. Fixed +/-2 eV zoom; overflow counts annotated.',fontsize=10)
            writer.grab_frame()


if __name__=='__main__':main()
