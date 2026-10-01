"""Saved-wave diagnostics, NOT LP/UP eigenstate populations or Phase7 certification.

Project Psi_Q(R,x,Q) onto displaced oscillator n=0,1; integrate x, leaving
conditional populations (NR,). All other states are retained as a remainder.
QHJ scalar terms use existing wave-jet results, never masked-field derivatives.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from .run_real_grid_mcef_preview import load_wave, digest
from .real_grid_mcef_fields import derivative, divide

EV=27.211386245988
FS=.024188843265857


def populations(psi,Q,shift,dx):
    """Q-normalized wave (NR,Nx,NQ), shift=sqrt(w)*d(R); probabilities unitless.

    Reference modes pi^-1/4 exp(-(Q+shift)^2/2) and sqrt(2)(Q+shift)*mode0.
    These BO-dipole reference modes are not full electron-photon eigenstates.
    """
    y=Q[None,:]+shift[:,None];dq=Q[1]-Q[0]
    h0=np.pi**(-.25)*np.exp(-y*y/2);h1=np.sqrt(2)*y*h0
    rho=np.sum(abs(psi)**2,axis=(1,2))*dx*dq
    weights=np.stack([np.sum(abs(np.einsum('rxq,rq->rx',psi,h)*dq)**2,axis=1)*dx for h in (h0,h1)])
    gram=max(float(np.max(abs(np.sum(h0*h0,axis=1)*dq-1))),
             float(np.max(abs(np.sum(h1*h1,axis=1)*dq-1))),
             float(np.max(abs(np.sum(h0*h1,axis=1)*dq))))
    return rho,weights,gram


def curvature_terms(psi,Q,packet):
    """Separate F_qq/(2F), F_RR/(2MF) from native wave jets; (NR,NQ), Ha.

    Constant Q-to-q wave normalization cancels in logarithmic derivatives.
    Derivatives along Q acquire omega for second physical-q derivatives.
    """
    rho=np.sum(abs(psi)**2,axis=1)
    result=[]
    for axis,spacing,factor in ((2,Q[1]-Q[0],float(packet['omega'])/2),
                                (0,float(packet['dR']),1/(2*float(packet['mass'])))):
        du=derivative(psi,spacing,axis)
        r1=2*np.sum((psi.conj()*du).real,axis=1)
        r2=2*np.sum(abs(du)**2,axis=1);del du
        du=derivative(psi,spacing,axis,2)
        r2+=2*np.sum((psi.conj()*du).real,axis=1);del du
        with np.errstate(invalid='ignore',divide='ignore'):
            result.append(factor*(divide(r2,2*rho)-divide(r1**2,4*rho**2)))
    return result


def render(frames,out,mass,fps=1):
    """Fixed scales across events, actual saved times only; MP4, no smoothing."""
    for family in ('states','epsilon1','epsilon2'):
        fig=plt.figure(figsize=(13,8),layout='constrained')
        writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        with writer.saving(fig,str(out/(family+'_terms_events.mp4')),100):
            for f in frames:
                fig.clear();R=f['R_coupled'];rho=f['coupled_rho_R'];mask=rho>rho.max()*1e-4
                if family=='states':
                    axs=fig.subplots(2,1)
                    for k,label in enumerate(('displaced n=0','displaced n=1','all remaining states')):
                        axs[0].plot(R,np.where(mask,f['pop'][k],np.nan),label=label)
                    axs[0].set(ylim=(-.02,1.02),ylabel='Conditional probability');axs[0].legend()
                    axs[1].plot(R,rho,color='black');axs[1].set(ylabel=r'$|\chi|^2$',ylim=(0,2.5))
                    for ax in axs:ax.set(xlim=(-4.4,4.4),xlabel=r'$R$ ($a_0$)');ax.axvline(0,ls=':',color='gray')
                    fig.suptitle('Displaced-photon reference occupations (NOT LP / UP)\n'+f"t={f['time_au']*FS:.3f} fs | global P0/P1/rest="+str(np.round(f['global_pop'],4)))
                elif family=='epsilon1':
                    axs=fig.subplots(2,3);pref='coupled_joint_';joint=f[pref+'rho_qR'];m=joint>joint.max()*1e-4
                    terms=[f['density_q'],f['density_R'],-f[pref+'a']**2/2,-f[pref+'b']**2/(2*mass),f[pref+'epsilon1_QHJ']]
                    titles=[r'$F_{q_cq_c}/(2F)$',r'$F_{RR}/(2MF)$',r'$-a^2/2$',r'$-b^2/(2M)$',r'Sum: $\epsilon^{(1)}$']
                    for ax,v,title in zip(axs.flat,terms,titles):
                        # Subtract each term's own weighted mean; sums remain additive.
                        v=(v-np.average(v[m],weights=joint[m]))*EV
                        im=ax.pcolormesh(R,f['Q'],np.ma.array(v,mask=~m).T,cmap='RdBu_r',vmin=-1,vmax=1,shading='auto')
                        ax.contour(R,f['Q'],joint.T,levels=joint.max()*np.array([1e-3,.01,.1]),colors='gray',linewidths=.5)
                        ax.set(title=title,xlabel=r'$R$ ($a_0$)',ylabel=r'$Q=\sqrt{\omega_c}q_c$',ylim=(-13,11))
                        fig.colorbar(im,ax=ax,extend='both',label='eV (centered, fixed zoom)')
                    ax=axs.flat[5]
                    im=ax.pcolormesh(R,f['Q'],np.ma.array(joint/joint.max(),mask=~m).T,cmap='viridis',vmin=0,vmax=1,shading='auto')
                    ax.set(title=r'$|\Lambda_R\chi|^2$ (relative)',xlabel=r'$R$ ($a_0$)',ylabel=r'$Q$',ylim=(-13,11))
                    fig.colorbar(im,ax=ax,label='Relative density')
                    fig.suptitle(f"Positive-gauge epsilon1 decomposition | t={f['time_au']*FS:.3f} fs")
                else:
                    axs=fig.subplots(2,2)
                    for col,name in enumerate(('free','coupled')):
                        r=f['R_'+name];w=f[name+'_rho_R'];m=w>w.max()*1e-4
                        den=-f[name+'_Q_nuclear'];mom=-f[name+'_alpha']**2/(2*mass)
                        for v,label in zip((den,mom,den+mom),('Density curvature',r'$-\alpha^2/(2M)$','Sum')):
                            v=(v-np.average(v[m],weights=w[m]))*EV
                            axs[0,col].plot(r,np.where(m,v,np.nan),label=label)
                        axs[0,col].set(title=name,ylim=(-2,2),ylabel='eV (centered, fixed zoom)');axs[0,col].legend()
                        axs[1,col].plot(r,w,color='black');axs[1,col].set(ylabel=r'$|\chi|^2$',ylim=(0,2.5))
                    for ax in axs.flat:ax.set(xlim=(-4.4,4.4),xlabel=r'$R$ ($a_0$)');ax.axvline(0,ls=':',color='gray')
                    fig.suptitle(f"Positive-gauge epsilon2 decomposition | t={f['time_au']*FS:.3f} fs")
                fig.supxlabel('Actual sparse events; no temporal interpolation. Relative density mask 1e-4.\nCoupled cavity 161.769 meV / free 170.6 meV. Diagnostic, NOT convergence certification.',fontsize=10)
                writer.grab_frame()
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('packet','waves','qhj','out'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.packet) as z:packet={k:z[k] for k in z.files}
    mu=packet['R']-np.sum(abs(packet['phi'][:,:,0])**2*packet['x'][None,:],axis=1)*float(packet['dx'])
    shift=np.sqrt(2)*float(packet['g_chi'])*mu/float(packet['omega'])
    frames=[];report=[]
    for path in sorted(args.qhj.glob('qhj_*.npz')):
        with np.load(path) as z:f={k:z[k] for k in z.files}
        wave=args.waves/path.name.replace('qhj_','wave_')
        psi,Q,t=load_wave(wave,packet,digest(args.packet))
        if abs(t-f['time_au'])>1e-10:raise ValueError('Mismatched times')
        rho,weights,gram=populations(psi,Q,shift,float(packet['dx']))
        cq,cr=curvature_terms(psi,Q,packet);del psi
        joint=f['coupled_joint_rho_qR'];support=joint>joint.max()*1e-4
        error=float(np.max(abs((cq+cr+f['coupled_joint_Q_joint'])[support])))
        if error>1e-8:raise ValueError('Density curvature reconstruction failed')
        f.update(density_q=cq,density_R=cr)
        pop=np.divide(weights,rho[None,:],out=np.full_like(weights,np.nan),where=rho[None,:]>0)
        pop=np.vstack((pop,1-pop.sum(axis=0)))
        m=rho>rho.max()*1e-4
        if gram>1e-8 or np.min(pop[:,m]) < -1e-8:raise ValueError('Projection basis invalid')
        np.testing.assert_allclose(rho,f['coupled_rho_R'],atol=1e-11,rtol=1e-10)
        glob=np.array([*np.sum(weights,axis=1)*float(packet['dR']),np.sum(rho)*float(packet['dR'])-np.sum(weights)*float(packet['dR'])])
        f.update(pop=pop,global_pop=glob);frames.append(f)
        report.append(dict(time_fs=t*FS,populations=glob.tolist(),gram_error=gram,curvature_reconstruction_max=error))
        np.savez(args.out/path.name.replace('qhj_','states_'),R=packet['R'],time_au=t,populations=pop,global_populations=glob,density_q=cq,density_R=cr)
        print(t*FS,glob,flush=True)
    (args.out/'validation.json').write_text(json.dumps(dict(status='REFERENCE_BASIS_DIAGNOSTIC',not_LP_UP=True,frames=report),indent=2))
    render(frames,args.out,float(packet['mass']))


if __name__=='__main__':main()
