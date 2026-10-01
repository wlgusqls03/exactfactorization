"""Read-only received-field audit and free/coupled nuclear-observable movie.

No propagation, temporal interpolation, inferred free TDPES, or certification.
Array shapes: density (time,R), compact joint fields (R,q). Units: atomic,
except displayed fs and eV. Free data are exact n=0-sector TDSE observables.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from .vsc_movie_only import read_frame, supports, AU_FS, HA_EV


def matched_times(a,b):
    """Reject missing/mismatched actual times rather than interpolate."""
    if len(a)!=len(b) or not np.allclose(a,b,rtol=0,atol=1e-9):
        raise ValueError('Require identical saved physical times; no interpolation')


def load_comparison(fields, free_run, free_packet, mass):
    """Native-grid densities, exact free flux, coupled j=rho*alpha/M."""
    with np.load(free_packet,allow_pickle=False) as p:
        if float(p['g_chi'])!=0:raise ValueError('Control is not uncoupled')
        Rfree=p['R'].copy();omega=float(p['omega'])
    paths=sorted(Path(fields).glob('fields_*.npz'))
    obs=sorted(Path(free_run).glob('observable_*.npz'))
    if not paths or not obs:raise ValueError('Missing fields or free observables')
    free=[];coupled=[];diagnostics=[]
    peak=0.;peakR=0.
    for path in paths:
        f=read_frame(path);R=f['R'];j=f['rho_R']*f['alpha']/mass
        np.testing.assert_allclose(f['Q'],np.sqrt(0.005944898236654274)*f['q'],atol=1e-12,
            err_msg='This comparison labels the received 161.769-meV case only')
        # R grids straddle zero. Interpolation here is spatial only.
        coupled.append((float(f['time_au']),f['rho_R'],float(f['rho_R'][R>0].sum()*(R[1]-R[0])),float(np.interp(0,R,j))))
        peak=max(peak,float(f['rho_qR'].max()));peakR=max(peakR,float(f['rho_R'].max()))
        diagnostics.append(json.loads(path.with_suffix('.json').read_text()))
    for path in obs:
        with np.load(path,allow_pickle=False) as z:
            free.append((float(z['time_au']),z['rho_R'].copy(),float(z['product']),float(z['flux'])))
    matched_times([r[0] for r in free],[r[0] for r in coupled])
    if any(len(r[1])!=len(Rfree) for r in free):raise ValueError('Free packet grid mismatch')
    audit={}
    for floor in (1e-5,1e-4,1e-3):
        rows=[]
        for path in paths:
            f=read_frame(path);mr,mj=supports(f,floor,peak,1e-8,peakR)
            dr=float(f['R'][1]-f['R'][0]);dq=float(f['q'][1]-f['q'][0])
            e=f['epsilon1_A'].real
            offset=float(np.average(e[mj],weights=f['rho_qR'][mj]))
            centered=abs(e-offset)*HA_EV
            flat=int(np.argmax(np.where(mj,centered,-np.inf)));ir,iq=np.unravel_index(flat,e.shape)
            rows.append(dict(time_fs=float(f['time_au'])*AU_FS,
                excluded_joint_probability=float(f['rho_qR'][~mj].sum()*dr*dq),
                epsilon1_max_abs_aligned_eV=float(centered[mj].max()),
                epsilon1_extreme_R=float(f['R'][ir]),epsilon1_extreme_Q=float(f['Q'][iq]),
                extreme_relative_joint_density=float(f['rho_qR'][ir,iq]/peak),
                epsilon1_rms_aligned_eV=float(np.sqrt(np.average(centered[mj]**2,weights=f['rho_qR'][mj]))),
                force_max_abs=float(abs(f['force'][mr]).max()),
                force_rms=float(np.sqrt(np.average(f['force'][mr]**2,weights=f['rho_R'][mr])))))
        audit[str(floor)]=rows
    errors={}
    for budget in ('1e-06','1e-08','1e-10'):
        errors[budget]={k:max(d['budgets'][budget]['weighted_errors'][k] for d in diagnostics)
                        for k in diagnostics[0]['budgets'][budget]['weighted_errors']}
    report=dict(frame_count=len(paths),phase7_pass=False,
        sources=dict(fields=str(fields),free_run=str(free_run),free_packet=str(free_packet)),
        scope='Historical free vs received coupled dynamics; NOT identical grids or a new convergence certificate',
        free_omega_au=omega,free_R_points=len(Rfree),coupled_R_points=len(R),
        coupled_flux_method='linear spatial interpolation of rho_R*alpha/M to R=0',
        max_norm_error=max(abs(d['norm']-1) for d in diagnostics),
        max_reconstruction_L2=max(d['reconstruction_L2'] for d in diagnostics),
        max_energy_drift=max(abs(d['energy_drift']) for d in diagnostics),
        replay_all_pass=all(d['replay_observation']['status']=='PASS' for d in diagnostics),
        weighted_error_maxima=errors,mask_sensitivity=audit)
    for name,rows in [('free',free),('coupled',coupled)]:
        t=np.array([r[0] for r in rows]);pop=np.array([r[2] for r in rows]);flux=np.array([r[3] for r in rows])
        report[name]=dict(max_product=float(pop.max()),time_max_fs=float(t[pop.argmax()]*AU_FS),
            final_product=float(pop[-1]),forward_flux=float(np.trapz(np.maximum(flux,0),t)),
            backward_flux=float(np.trapz(np.maximum(-flux,0),t)),
            population_minus_net_flux=float(pop[-1]-pop[0]-np.trapz(flux,t)))
    return Rfree,R,free,coupled,report


def movie(out,Rf,Rc,free,coupled,fps=24):
    """Three uncluttered panels, native grids, shared linear axes, actual frames."""
    t=np.array([r[0] for r in free])*AU_FS
    if len(t)<2 or np.max(np.diff(t))>.2:raise ValueError('Dense actual frames required')
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    colors=('#546e7a','#c05032');names=('Uncoupled (g=0)','Coupled (eta=0.094)')
    ymax=max(max(float(r[1].max()) for r in rows) for rows in (free,coupled))
    pmax=max(max(r[2] for r in rows) for rows in (free,coupled))
    jmax=max(max(abs(r[3]) for r in rows) for rows in (free,coupled))
    writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-crf','19','-pix_fmt','yuv420p'])
    with writer.saving(fig,str(out/'free_vs_coupled_dynamics.mp4'),100):
        for i in range(len(t)):
            for ax in axs:ax.clear()
            for R,rows,color,name in zip((Rf,Rc),(free,coupled),colors,names):
                axs[0].plot(R,rows[i][1],color=color,label=name,lw=1.8)
                for ax,k in zip(axs[1:],(2,3)):
                    ax.plot(t,[r[k] for r in rows],color=color,alpha=.15)
                    ax.plot(t[:i+1],[r[k] for r in rows[:i+1]],color=color,lw=1.8)
            axs[0].axvline(0,color='black',ls=':',lw=.8)
            axs[0].set(xlim=(min(Rf[0],Rc[0]),max(Rf[-1],Rc[-1])),ylim=(0,ymax*1.05),
                xlabel=r'$R$ ($a_0$)',ylabel=r'$|\chi|^2$ ($a_0^{-1}$)',title='Where is the nucleus?')
            axs[0].legend(fontsize=9)
            axs[1].set(ylim=(0,pmax*1.05),ylabel=r'$P(R>0)$',title='How much has transferred?')
            axs[2].set(ylim=(-1.05*jmax,1.05*jmax),ylabel='Signed flux (a.u.)',title='Forward (+) / backward (-)')
            for ax in axs[1:]:
                ax.set(xlim=(t[0],t[-1]),xlabel='Time (fs)');ax.axvline(t[i],color='0.4',lw=.7)
                ax.axhline(0,color='0.6',lw=.6)
            fig.suptitle(f'Full electron-nuclear dynamics | t = {t[i]:.3f} fs | coupled cavity: 161.769 meV')
            fig.supxlabel('Native grids differ (free 440 / coupled 352 R points). No time interpolation. Not a thermal reaction rate.',fontsize=9)
            writer.grab_frame()
            if i%100==0:print(f'movie {i+1}/{len(t)}',flush=True)
    plt.close(fig)


def momentum_movie(fields,free_run,Rf,out,mass=1836.,floor=1e-4,alpha_limit=30.):
    """Positive-marginal alpha=Mj/rho; free b-alpha=0 by exact separability.

    Conditional RMS is sqrt(integral dq rho(q|R)*(b-alpha)^2), not force.
    No free electronic scalar potential is inferred from these observables.
    """
    paths=sorted(Path(fields).glob('fields_*.npz'));obs=sorted(Path(free_run).glob('observable_*.npz'))
    if not paths or len(paths)!=len(obs):raise ValueError('Missing matched momentum frames')
    rows=[];peak=0.
    for fp,op in zip(paths,obs):
        f=read_frame(fp)
        with np.load(op) as z:
            matched_times([float(f['time_au'])],[float(z['time_au'])])
            rho=z['rho_R'].copy();j=z['current'].copy()
        af=np.divide(mass*j,rho,out=np.full_like(rho,np.nan),where=rho>0)
        delta=f['b']-f['alpha'][:,None];dq=float(f['q'][1]-f['q'][0])
        spread=np.sqrt(np.sum(f['lambda_density']*delta**2,axis=1)*dq)
        rows.append((float(f['time_au'])*AU_FS,rho,f['rho_R'],af,f['alpha'],spread))
        peak=max(peak,float(rho.max()),float(f['rho_R'].max()))
    Rc=f['R'];bound=0.;sbound=0.
    for _,rf,rc,af,ac,s in rows:
        bound=max(bound,float(abs(af[rf>=floor*peak]).max()),float(abs(ac[rc>=floor*peak]).max()))
        sbound=max(sbound,float(s[rc>=floor*peak].max()))
    shown=min(bound,alpha_limit)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    writer=FFMpegWriter(fps=24,codec='libx264',extra_args=['-crf','19','-pix_fmt','yuv420p'])
    with writer.saving(fig,str(out/'free_vs_coupled_mcef_momenta.mp4'),100):
        for i,(t,rf,rc,af,ac,s) in enumerate(rows):
            for ax in axs:ax.clear()
            for R,rho,alpha,color,name in [(Rf,rf,af,'#546e7a','Uncoupled'),(Rc,rc,ac,'#c05032','Coupled')]:
                axs[0].plot(R,np.where(rho>=floor*peak,alpha,np.nan),color=color,label=name)
                for sign,marker in ((1,'^'),(-1,'v')):
                    outside=(rho>=floor*peak)&(sign*alpha>shown)
                    axs[0].scatter(R[outside],np.full(outside.sum(),sign*shown),s=12,marker=marker,color=color)
            axs[0].set(ylim=(-shown*1.05,shown*1.05),ylabel='Momentum (a.u.)',title=r'$\alpha=M j_R/|\chi|^2$')
            axs[0].legend()
            axs[1].plot(Rc,np.where(rc>=floor*peak,s,np.nan),color='#c05032',label='Coupled')
            axs[1].plot(Rf,np.where(rf>=floor*peak,0.,np.nan),color='#546e7a',label='Uncoupled: exactly zero')
            axs[1].set(ylim=(0,1.05*max(sbound,1e-12)),ylabel='Conditional RMS momentum (a.u.)',
                       title=r'$\sqrt{\langle(b-\alpha)^2\rangle_{q|R}}$ (not force)')
            axs[1].legend(fontsize=9)
            for ax in axs:
                ax.set(xlim=(min(Rf[0],Rc[0]),max(Rf[-1],Rc[-1])),xlabel=r'$R$ ($a_0$)')
                ax.axvline(0,color='0.5',ls=':',lw=.8)
                ax.fill_between(Rc,0,.12*rc/peak,transform=ax.get_xaxis_transform(),color='#c05032',alpha=.15)
            fig.suptitle(f'Positive-marginal gauge | t={t:.3f} fs | coupled cavity 161.769 meV')
            fig.supxlabel(f'R support: {floor:g} of shared peak. Alpha display +/-{shown:.1f}; triangles mark out-of-range values (full max {bound:.1f}).\n'
                           'Native grids differ. Conditional RMS is not force.',fontsize=9)
            writer.grab_frame()
            if i%100==0:print(f'momenta {i+1}/{len(rows)}',flush=True)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields',type=Path,required=True);p.add_argument('--free-run',type=Path,required=True)
    p.add_argument('--free-packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit-only',action='store_true');p.add_argument('--mass',type=float,default=1836.)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Choose new output; no overwriting')
    Rf,Rc,free,coupled,report=load_comparison(a.fields,a.free_run,a.free_packet,a.mass)
    a.out.mkdir(parents=True)
    (a.out/'comparison_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(a.out/'comparison_observables.npz',time_au=[r[0] for r in free],
        R_free=Rf,R_coupled=Rc,rho_free=[r[1] for r in free],rho_coupled=[r[1] for r in coupled],
        product_free=[r[2] for r in free],product_coupled=[r[2] for r in coupled],
        flux_free=[r[3] for r in free],flux_coupled=[r[3] for r in coupled])
    if not a.audit_only:
        movie(a.out,Rf,Rc,free,coupled)
        momentum_movie(a.fields,a.free_run,Rf,a.out,a.mass)


if __name__=='__main__':main()
