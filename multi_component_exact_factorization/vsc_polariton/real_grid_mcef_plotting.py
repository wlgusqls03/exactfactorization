"""Readable real-data previews, never an implicit gauge/convergence certificate.

Figures use natural positive-marginal gauge. Force includes alpha_t.
q axis is physical photon quadrature, not position of a photon. Density
contours extend to 1e-5 of each frame's maximum, explicitly labelled.
Energy offsets are for display only and retained in summary metadata.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
from .phase7_support import budget_support

AU_FS=0.024188843265857
HA_EV=27.211386245988


def style():
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
        'figure.dpi':110,'savefig.dpi':180,'axes.spines.top':False,
        'axes.spines.right':False,'font.family':'DejaVu Sans'})


def save(fig, out, name):
    for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)


def dynamics(R, times, rows, out, label):
    """414 stored observables, no fabricated wave interpolation or free control."""
    t=np.array(times)*AU_FS;rho=np.stack([v['rho_R'] for v in rows])
    p=np.array([float(v['product']) for v in rows])
    j=np.array([float(v['flux']) for v in rows])
    fig,axs=plt.subplots(3,1,figsize=(10,9),layout='constrained',sharex=True)
    im=axs[0].pcolormesh(t,R,np.maximum(rho.T,1e-300),shading='auto',rasterized=True,
        norm=LogNorm(vmin=rho.max()*1e-5,vmax=rho.max()),cmap='magma')
    axs[0].axhline(0,color='cyan',ls='--',lw=1)
    axs[0].set(ylabel=r'$R$ ($a_0$)',title='Nuclear probability density | dashed line: dividing surface')
    fig.colorbar(im,ax=axs[0],label=r'$\rho_R$ ($a_0^{-1}$); floor $10^{-5}\rho_{\max}$')
    axs[1].plot(t,p,color='#1864ab',lw=2)
    axs[1].set(ylabel=r'$P_{\rm product}$',title=r'Population on $R>0$ (not a reaction rate)')
    axs[2].plot(t,j,color='#b23a48',lw=1.5)
    axs[2].fill_between(t,0,j,where=j>0,color='#1971c2',alpha=.2,label='Forward')
    axs[2].fill_between(t,0,j,where=j<0,color='#e8590c',alpha=.3,label='Backward')
    axs[2].axhline(0,color='0.4',lw=.8)
    axs[2].set(xlabel='Time (fs)',ylabel=r'$J_{\rm TS}$ (a.u.)',title='Signed flux: positive crossing / negative recrossing')
    axs[2].legend(ncol=2)
    fig.suptitle(label+'\nFull real-grid TDSE | single case, no uncoupled comparison',fontsize=14)
    save(fig,out,'figure1_nuclear_transfer')
    return dict(max_product=float(p.max()),time_max_product_fs=float(t[p.argmax()]),
                final_product=float(p[-1]),forward_integral=float(np.trapz(np.maximum(j,0),times)),
                backward_integral=float(np.trapz(np.maximum(-j,0),times)))


def prepared(f,budget,relative_floor):
    dr=f['R'][1]-f['R'][0];dq=f['q'][1]-f['q'][0]
    mr,_,_=budget_support(f['rho_R'],dr,budget)
    mj,_,_=budget_support(f['rho_qR'],dr*dq,budget)
    mr &= f['rho_R']>=f['rho_R'].max()*relative_floor
    mj &= f['rho_qR']>=f['rho_qR'].max()*relative_floor
    # Natural-gauge scalars: explicitly subtract a single display constant.
    # This is NOT an alpha=0 transformation and does not connect phases at nodes.
    off2=np.average(f['epsilon2_A'].real[mr],weights=f['rho_R'][mr])
    off1=np.average(f['epsilon1_A'].real[mj],weights=f['rho_qR'][mj])
    return dict(mr=mr,mj=mj,e2=np.where(mr,(f['epsilon2_A'].real-off2)*HA_EV,np.nan),
                e1=np.ma.array((f['epsilon1_A'].real-off1)*HA_EV,mask=~mj),
                force=np.where(mr,f['force'],np.nan),offset1_Ha=float(off1),offset2_Ha=float(off2))


def contour(ax,f):
    d=f['rho_qR']/f['rho_qR'].max()
    ax.contour(f['R'],f['q'],d.T,levels=[1e-5,1e-3,.1],colors='white',linewidths=.55,alpha=.65)


def outer_plot(f,p,axs,force_max=None):
    R=f['R'];axs[0].plot(R,p['e2'],color='#1864ab',lw=1.5)
    axs[0].set(ylabel=r'$\epsilon^{(2)}-\langle\epsilon^{(2)}\rangle_\rho$ (eV)',
               title='Positive-marginal gauge scalar (NOT the force potential)')
    density=axs[0].twinx()
    density.fill_between(R,0,f['rho_R'],color='0.4',alpha=.22)
    density.set_ylabel(r'$\rho_R$ ($a_0^{-1}$)',color='0.4')
    axs[0].set_yscale('symlog',linthresh=.1)
    axs[1].plot(R,p['force'],color='#b23a48',lw=1.5,label=r'$-\partial_R\epsilon^{(2)}+\partial_t\alpha$')
    axs[1].plot(R,np.where(p['mr'],-f['epsilon2_R'],np.nan),color='0.55',lw=.8,ls='--',label='Scalar slope only')
    axs[1].set(ylabel='Force (a.u.)',title='Gauge-invariant EF force | symlog, not clipped')
    axs[1].set_yscale('symlog',linthresh=.001)
    if force_max:axs[1].set_ylim(-force_max,force_max)
    axs[1].legend(fontsize=9)
    for ax in axs:
        ax.axvline(0,color='0.3',ls=':',lw=.8);ax.set_xlabel(r'$R$ ($a_0$)')
        ax.grid(alpha=.15)


def snapshots(frames,out,label,budget=1e-8,relative_floor=1e-5,movie=False):
    """Matched limits across real saved event frames. No time interpolation."""
    states=[prepared(f,budget,relative_floor) for f in frames]
    vmax=max(float(np.ma.max(abs(p['e1']))) for p in states)
    vmax=max(vmax,.001)
    fn=max(max(float(np.nanmax(abs(p['force']))),
               float(np.max(abs(f['epsilon2_R'][p['mr']])))) for p,f in zip(states,frames))
    norm=SymLogNorm(linthresh=.1,vmin=-vmax,vmax=vmax)
    # Common occupied-photon viewport; do not rescale each snapshot separately.
    qbound=max(float(np.max(abs(f['q'][p['mj'].any(axis=0)]))) for f,p in zip(frames,states))*1.08
    offsets=[]
    for i,(f,p) in enumerate(zip(frames,states)):
        time=float(f['time_au'])*AU_FS
        fig,axs=plt.subplots(2,1,figsize=(10,7.5),layout='constrained')
        outer_plot(f,p,axs,fn)
        fig.suptitle(f'{label} | t = {time:.3f} fs\nMCEF diagnostic preview; field convergence not certified')
        save(fig,out,f'figure2_outer_{i:03d}')
        fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained',sharex=True,sharey=True)
        im=axs[0].pcolormesh(f['R'],f['q'],p['e1'].T,shading='auto',cmap='RdBu_r',norm=norm,rasterized=True)
        contour(axs[0],f)
        fig.colorbar(im,ax=axs[0],label=r'$\epsilon^{(1)}-\langle\epsilon^{(1)}\rangle$ (eV; symlog)')
        axs[0].set_title('Electronic-level scalar | positive gauge')
        im=axs[1].pcolormesh(f['R'],f['q'],np.ma.array(f['BO_character'][...,0],mask=~p['mj']).T,
                           shading='auto',cmap='viridis',vmin=0,vmax=1,rasterized=True)
        contour(axs[1],f);fig.colorbar(im,ax=axs[1],label=r'$|\langle\phi_0^{BO}|\Phi_{R,q_c}\rangle|^2$')
        axs[1].set_title('Conditional bare-BO ground character')
        rel=f['rho_qR']/f['rho_qR'].max()
        im=axs[2].pcolormesh(f['R'],f['q'],np.ma.array(rel,mask=rel<relative_floor).T,
                           shading='auto',cmap='magma',norm=LogNorm(relative_floor,1),rasterized=True)
        fig.colorbar(im,ax=axs[2],label=r'$\rho_{qR}/\max\rho_{qR}$')
        axs[2].plot(f['R'],np.where(p['mr'],f['conditional_q'],np.nan),color='cyan',lw=1,label=r'$\overline{q}_c(R)$')
        axs[2].legend(fontsize=9);axs[2].set_title('Photon-proton joint density')
        for ax in axs:
            ax.set_xlabel(r'$R$ ($a_0$)');ax.axvline(0,color='0.6',ls=':',lw=.8)
        axs[0].set_ylabel(r'Photon quadrature $q_c$ (a.u.; not photon position)')
        axs[0].set_ylim(-qbound,qbound)
        fig.suptitle(f'{label} | t = {time:.3f} fs\nDiagnostic preview | white contours: relative density 1e-5, 1e-3, 0.1')
        save(fig,out,f'figure3_joint_{i:03d}')
        offsets.append(dict(time_fs=time,epsilon1_offset_Ha=p['offset1_Ha'],epsilon2_offset_Ha=p['offset2_Ha']))
    if movie:
        # This is an explicitly sparse event slideshow, never advertised as
        # smooth real-time propagation. Each frame has its actual physical time.
        extension='mp4' if writers.is_available('ffmpeg') else 'gif'
        writer=FFMpegWriter(fps=1,bitrate=2400) if extension=='mp4' else PillowWriter(fps=1)
        fig=plt.figure(figsize=(12,8),layout='constrained')
        with writer.saving(fig,str(out/('real_grid_mcef_events.'+extension)),dpi=120):
            for f,p in zip(frames,states):
                fig.clear();axs=fig.subplots(2,2)
                outer_plot(f,p,axs[:,0],fn)
                im=axs[0,1].pcolormesh(f['R'],f['q'],p['e1'].T,norm=norm,cmap='RdBu_r',shading='auto',rasterized=True)
                contour(axs[0,1],f);fig.colorbar(im,ax=axs[0,1],label='epsilon1 offset (eV; symlog)')
                rel=f['rho_qR']/f['rho_qR'].max()
                im=axs[1,1].pcolormesh(f['R'],f['q'],np.maximum(rel,1e-300).T,
                    norm=LogNorm(relative_floor,1),cmap='magma',shading='auto',rasterized=True)
                fig.colorbar(im,ax=axs[1,1],label='Relative joint density')
                for ax in axs[:,1]:ax.set(xlabel=r'$R$ ($a_0$)',ylabel=r'$q_c$ (a.u.)',ylim=(-qbound,qbound))
                fig.suptitle(f'{label} | t={float(f["time_au"])*AU_FS:.3f} fs\nSparse saved events (nonuniform physical times); MCEF DIAGNOSTIC, not certified')
                writer.grab_frame()
        plt.close(fig)
    return dict(offsets=offsets,epsilon1_common_abs_limit_eV=vmax,force_common_abs_limit_au=fn,
                common_photon_viewport=[-qbound,qbound],
                support_budget=budget,additional_display_relative_density_floor=relative_floor,
                gauge='Positive chi and Lambda; NOT alpha=0',
                movie='Sparse actual saved frames; no interpolation')
