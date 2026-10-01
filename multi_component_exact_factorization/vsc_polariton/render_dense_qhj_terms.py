"""Stream existing compact MCEF fields to QHJ movies, with no TDSE replay.

Positive marginal gauge only. Total density-curvature terms are inferred from
the QHJ identity, NOT independently differentiated or split into q/R parts.
This is visualization, not new numerical convergence validation or LP/UP analysis.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def terms(f,mass):
    """Return atomic-unit scalar terms: (NR,Nq) epsilon1 and (NR,) epsilon2."""
    a=-f['a']**2/2;b=-f['b']**2/(2*mass);alpha=-f['alpha']**2/(2*mass)
    e1=f['epsilon1_A'].real;e2=f['epsilon2_A'].real
    return (e1-a-b,a,b,e1),(e2-alpha,alpha,e2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('fields','input','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--density-floor',type=float,default=1e-4)
    p.add_argument('--support-budget',type=float,default=1e-8)
    p.add_argument('--vmax-ev',type=float,default=1.)
    args=p.parse_args()
    if args.fps<=0 or args.vmax_ev<=0 or not 0<args.density_floor<1 or not 0<args.support_budget<1:p.error('Invalid setting')
    root=Path(__file__).resolve().parents[2]/'results/vsc_polariton'
    args.out.resolve().relative_to(root.resolve())
    os.environ.setdefault('MPLCONFIGDIR',str(root/'movie_cache/matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'movie_cache'))
    from .vsc_movie_only import read_frame,supports,AU_FS,HA_EV
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    with np.load(args.input) as z:mass=float(z['mass']);omega=float(z['omega'])
    paths=sorted(args.fields.glob('fields_*.npz'))
    times=[];peak=0.;rpeak=0.;reference=None
    for path in paths:
        f=read_frame(path)
        if reference is None:reference=(f['R'],f['Q'])
        np.testing.assert_allclose(f['R'],reference[0]);np.testing.assert_allclose(f['Q'],reference[1])
        np.testing.assert_allclose(f['Q'],np.sqrt(omega)*f['q'],atol=1e-11)
        times.append(float(f['time_au'])*AU_FS)
        peak=max(peak,float(f['rho_qR'].max()));rpeak=max(rpeak,float(f['rho_R'].max()))
    gaps=np.diff(times)
    if len(times)<2 or np.any(gaps<=0) or max(gaps)>.2:raise ValueError('Need dense actual frames, max gap <=0.2 fs')
    args.out.mkdir(parents=True,exist_ok=False)
    saturation=[]
    for family in ('epsilon1','epsilon2'):
        fig=plt.figure(figsize=(12,8),layout='constrained')
        writer=FFMpegWriter(fps=args.fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        with writer.saving(fig,str(args.out/(family+'_qhj_dense.mp4')),100):
            for i,path in enumerate(paths):
                f=read_frame(path);mr,mj=supports(f,args.density_floor,peak,args.support_budget,rpeak)
                if not mr.any() or not mj.any():raise ValueError('Empty occupied support')
                e1,e2=terms(f,mass);fig.clear();axs=fig.subplots(2,2)
                values=e1 if family=='epsilon1' else e2
                titles=(('Total density curvature (QHJ-inferred)',r'$-a^2/2$',r'$-b^2/(2M)$',r'$\epsilon^{(1)}$')
                        if family=='epsilon1' else ('Density curvature (QHJ-inferred)',r'$-\alpha^2/(2M)$',r'$\epsilon^{(2)}$'))
                mask=mj if family=='epsilon1' else mr;rho=f['rho_qR'] if family=='epsilon1' else f['rho_R']
                row=[]
                for ax,v,title in zip(axs.flat,values,titles):
                    v=(v-np.average(v[mask],weights=rho[mask]))*HA_EV
                    row.append(float(rho[mask&(abs(v)>args.vmax_ev)].sum()/rho[mask].sum()))
                    if family=='epsilon1':
                        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e5e5e5')
                        im=ax.pcolormesh(f['R'],f['Q'],np.ma.array(v,mask=~mask).T,shading='auto',cmap=cmap,vmin=-args.vmax_ev,vmax=args.vmax_ev)
                        levels=peak*np.array([1e-4,1e-3,.01,.1]);levels=levels[levels<rho.max()]
                        if len(levels):ax.contour(f['R'],f['Q'],rho.T,levels=levels,colors='gray',linewidths=.5)
                        fig.colorbar(im,ax=ax,extend='both',label='eV, centered')
                        ax.set(ylabel=r'$Q=\sqrt{\omega_c}q_c$',ylim=(-13,11))
                    else:
                        ax.plot(f['R'],np.where(mask,v,np.nan))
                        for sign,mark in ((1,'^'),(-1,'v')):
                            m=mask&(sign*v>args.vmax_ev);ax.scatter(f['R'][m],np.full(m.sum(),sign*args.vmax_ev),marker=mark,s=10)
                        ax.set(ylim=(-1.05*args.vmax_ev,1.05*args.vmax_ev),ylabel='eV, centered')
                    ax.set(title=title,xlabel=r'$R$ ($a_0$)',xlim=(-4.4,4.4))
                if family=='epsilon2':
                    axs.flat[3].plot(f['R'],f['rho_R'],color='black');axs.flat[3].set(title=r'$|\chi|^2$',ylim=(0,rpeak*1.05),xlim=(-4.4,4.4),xlabel=r'$R$ ($a_0$)')
                saturation.append(dict(family=family,time_fs=times[i],clipped_probability_fractions=row))
                fig.suptitle(f'Positive-gauge QHJ | {omega*HA_EV*1000:.3f} meV | t={times[i]:.3f} fs')
                fig.supxlabel('Actual saved frames, no interpolation. Each term has its occupied-density mean removed.\nDensity curvature inferred by identity; no independent derivative validation. Mask '+str(args.density_floor),fontsize=9)
                writer.grab_frame()
                if i%32==0:print(family,i+1,'/',len(paths),flush=True)
        plt.close(fig)
    (args.out/'manifest.json').write_text(json.dumps(dict(frames=len(paths),fps=args.fps,max_gap_fs=float(max(gaps)),duration_seconds=len(paths)/args.fps,
        input_fields=str(args.fields.resolve()),density_curvature='QHJ inferred, not independent',photon_populations=False,phase7_pass=False,saturation=saturation),indent=2))


if __name__=='__main__':main()
