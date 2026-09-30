"""Small, movie-only real-grid views. No temporal interpolation or PDF output.

Input physical-q fields have (R,q) shape. Densities displayed in Q=sqrt(w)q
include the 1/sqrt(w) Jacobian. Display masks use ONE trajectory maximum;
probability-budget masks separately protect potentials/connections.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.animation import FFMpegWriter, writers
from .phase7_support import budget_support

AU_FS=.024188843265857
HA_EV=27.211386245988
FIELD_KEYS=('R','q','Q','rho_R','rho_qR','lambda_density','a','b','alpha',
            'force','epsilon2_A','epsilon1_A','conditional_q','time_au')


def compact_fields(fields):
    """Lossless plotting subset; complex scalars retain imaginary residuals."""
    return {key:fields[key] for key in FIELD_KEYS}


def read_frame(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in FIELD_KEYS}


def supports(f, floor, peak, budget, peak_R=None):
    """Separate outer/joint probability support; never fill a masked hole."""
    dr=f['R'][1]-f['R'][0];dq=f['q'][1]-f['q'][0]
    mr,_,_=budget_support(f['rho_R'],dr,budget)
    if peak_R is not None:mr &= f['rho_R']>=floor*peak_R
    mj,_,_=budget_support(f['rho_qR'],dr*dq,budget)
    mj &= f['rho_qR']>=floor*peak
    return mr,mj


def inspect(paths,omega,floor,budget,max_gap_fs=.2,allow_sparse=False):
    """Two streaming passes: fix density/field limits without holding all frames."""
    times=[];peak=0.;rpeak=0.;lpeak=0.;products=[];flux=[]
    grid=None
    for path in paths:
        f=read_frame(path)
        if grid is None:grid=(f['R'],f['Q'])
        np.testing.assert_allclose(f['R'],grid[0],atol=1e-12,rtol=0)
        np.testing.assert_allclose(f['Q'],grid[1],atol=1e-12,rtol=0)
        np.testing.assert_allclose(f['Q'],np.sqrt(omega)*f['q'],atol=1e-12,rtol=1e-12)
        times.append(float(f['time_au'])*AU_FS)
        peak=max(peak,float(f['rho_qR'].max()));rpeak=max(rpeak,float(f['rho_R'].max()))
        mr,_=supports(f,floor,float(f['rho_qR'].max()),budget)
        lpeak=max(lpeak,float(f['lambda_density'][mr].max()))
        dr=f['R'][1]-f['R'][0]
        products.append(float(f['rho_R'][f['R']>0].sum()*dr))
    gaps=np.diff(times)
    if len(paths)<2 or np.any(gaps<=0):raise ValueError('Need >=2 strictly time-ordered actual frames')
    sparse=float(gaps.max())>max_gap_fs
    if sparse and not allow_sparse:
        raise ValueError(f'Only {len(paths)} frames, max gap {gaps.max():.3f} fs. '
                         'Replay densely; --allow-sparse-preview is NOT a smooth movie.')
    limits=dict(momentum=1e-12,a=1e-12,force=1e-12,e1=1e-12,e2=1e-12)
    offsets=[];qlo=np.inf;qhi=-np.inf
    for path in paths:
        f=read_frame(path);mr,mj=supports(f,floor,peak,budget,rpeak)
        if not mr.any() or not mj.any():raise ValueError('Empty occupied support')
        delta=f['b']-f['alpha'][:,None]
        vals=[abs(f['b'][mj]).max(),abs(delta[mj]).max(),abs(f['alpha'][mr]).max()]
        limits['momentum']=max(limits['momentum'],*map(float,vals))
        limits['a']=max(limits['a'],float(abs(f['a'][mj]).max()))
        limits['force']=max(limits['force'],float(abs(f['force'][mr]).max()))
        o1=float(np.average(f['epsilon1_A'].real[mj],weights=f['rho_qR'][mj]))
        o2=float(np.average(f['epsilon2_A'].real[mr],weights=f['rho_R'][mr]))
        offsets.append((o1,o2))
        limits['e1']=max(limits['e1'],float(abs(f['epsilon1_A'].real[mj]-o1).max())*HA_EV)
        limits['e2']=max(limits['e2'],float(abs(f['epsilon2_A'].real[mr]-o2).max())*HA_EV)
        ids=np.flatnonzero((f['rho_qR']>=floor*peak).any(axis=0))
        qlo=min(qlo,float(f['Q'][ids[0]]));qhi=max(qhi,float(f['Q'][ids[-1]]))
    pad=.06*(qhi-qlo)
    return dict(times_fs=times,products=products,limits=limits,offsets_Ha=offsets,
                peak_joint_q=peak,peak_conditional_q=lpeak,peak_R=rpeak,
                Q_limits=[qlo-pad,qhi+pad],floor=floor,budget=budget,
                max_gap_fs=float(gaps.max()),sparse_preview=sparse,
                cadence='actual saved frames; no wave/field interpolation')


def contours(ax,f,c):
    """Colored decade contours; black lowest boundary. No dense minor lines."""
    relative=f['rho_qR']/c['peak_joint_q']
    levels=10.**np.arange(int(np.ceil(np.log10(c['floor']))),0)
    levels=levels[(levels>relative.min())&(levels<relative.max())]
    colors=['black' if np.isclose(v,c['floor']) else '#00796b' for v in levels]
    if len(levels):ax.contour(f['R'],f['Q'],relative.T,levels=levels,
                              colors=colors,linewidths=.65,alpha=.85)


def map_panel(fig,ax,f,c,z,title,unit,bound,mask,positive=False):
    norm=LogNorm(max(bound*c['floor'],1e-300),bound) if positive else SymLogNorm(
        max(bound*.01,1e-14),vmin=-bound,vmax=bound)
    cmap=plt.get_cmap('magma' if positive else 'RdBu_r').copy();cmap.set_bad('#f1f1f1')
    artist=ax.pcolormesh(f['R'],f['Q'],np.ma.array(z,mask=~mask).T,
                        shading='auto',norm=norm,cmap=cmap)
    contours(ax,f,c);ax.axvline(0,color='0.4',lw=.7,ls=':')
    ax.set(title=title,xlabel=r'$R$ ($a_0$)',ylabel=r'$Q=\sqrt{\omega_c}q_c$',ylim=c['Q_limits'])
    fig.colorbar(artist,ax=ax,pad=.02,shrink=.8,label=unit+(' (log)' if positive else ' (symlog)'))


def draw(fig,f,c,omega,index,family):
    """At most six panels; force not scalar slope used for motion interpretation."""
    mr,mj=supports(f,c['floor'],c['peak_joint_q'],c['budget'],c['peak_R'])
    R=f['R'];s=np.sqrt(omega);lim=c['limits'];delta=f['b']-f['alpha'][:,None]
    if family=='state':
        axs=fig.subplots(2,3)
        density_mask=f['rho_qR']>=c['floor']*c['peak_joint_q']
        for ax,z,title,bound in ((axs[0,0],f['lambda_density']/s,r'$|\Lambda_R(Q)|^2$',c['peak_conditional_q']/s),
                                (axs[0,1],f['rho_qR']/s,r'$|\Lambda_R(Q)|^2|\chi|^2$',c['peak_joint_q']/s)):
            map_panel(fig,ax,f,c,z,title,'Probability density in Q',bound,density_mask,True)
        axs[0,2].fill_between(R,0,f['rho_R'],color='#426b9a',alpha=.4)
        axs[0,2].set(title=r'$|\chi|^2$',xlabel=r'$R$ ($a_0$)',ylabel=r'$a_0^{-1}$',ylim=(0,1.05*c['peak_R']))
        for ax,z,title in ((axs[1,0],f['b'],r'$b$: conditional nuclear momentum'),
                           (axs[1,2],delta,r'$b-\alpha$')):
            map_panel(fig,ax,f,c,z,title,'Momentum (a.u.)',lim['momentum'],mj)
        axs[1,1].plot(R,np.where(mr,f['alpha'],np.nan),color='#426b9a')
        axs[1,1].set(title=r'$\alpha$: marginal nuclear momentum',xlabel=r'$R$ ($a_0$)',ylabel='Momentum (a.u.)',
                     ylim=(-lim['momentum'],lim['momentum']))
        axs[1,1].set_yscale('symlog',linthresh=max(.01*lim['momentum'],1e-14))
    elif family=='nuclear':
        axs=fig.subplots(2,2)
        axs[0,0].fill_between(R,0,f['rho_R'],alpha=.45,color='#426b9a')
        axs[0,0].axvline(0,color='black',ls=':');axs[0,0].set(title='Where is the nucleus?',
            xlabel=r'$R$ ($a_0$)',ylabel=r'$|\chi|^2$ ($a_0^{-1}$)',ylim=(0,c['peak_R']*1.05))
        axs[0,1].plot(c['times_fs'],c['products'],color='0.75')
        axs[0,1].plot(c['times_fs'][:index+1],c['products'][:index+1],color='#426b9a')
        axs[0,1].axvline(c['times_fs'][index],color='black',lw=.8)
        axs[0,1].set(title='Transferred population (not rate)',xlabel='Time (fs)',ylabel=r'$P(R>0)$',
                     ylim=(0,max(.01,1.05*max(c['products']))))
        force=np.where(mr,f['force'],np.nan)
        axs[1,0].plot(R,force,color='#b23a48');axs[1,0].axhline(0,color='0.5',lw=.6)
        axs[1,0].set(title=r'Force: $-\partial_R\epsilon^{(2)}+\partial_t\alpha$',
            xlabel=r'$R$ ($a_0$)',ylabel='Force (a.u.; symlog)',ylim=(-lim['force'],lim['force']))
        axs[1,0].set_yscale('symlog',linthresh=max(.01*lim['force'],1e-14))
        e=np.where(mr,(f['epsilon2_A'].real-c['offsets_Ha'][index][1])*HA_EV,np.nan)
        axs[1,1].plot(R,e,color='#426b9a');axs[1,1].set(title=r'$\epsilon^{(2)}$: scalar alone is NOT force',
            xlabel=r'$R$ ($a_0$)',ylabel='Mean-aligned energy (eV)',ylim=(-lim['e2'],lim['e2']))
        axs[1,1].set_yscale('symlog',linthresh=max(.01*lim['e2'],1e-12))
        for ax in axs[1,:]:
            ax.fill_between(R,0,.16*f['rho_R']/c['peak_R'],transform=ax.get_xaxis_transform(),
                            color='0.5',alpha=.18)
    else:
        axs=fig.subplots(1,2)
        e=(f['epsilon1_A'].real-c['offsets_Ha'][index][0])*HA_EV
        map_panel(fig,axs[0],f,c,e,r'$\epsilon^{(1)}$: electronic-level scalar','Mean-aligned eV',lim['e1'],mj)
        map_panel(fig,axs[1],f,c,f['a'],r'$a$: photon quadrature momentum','Momentum (a.u.)',lim['a'],mj)
    for ax in np.asarray(axs).flat:
        if not (family=='nuclear' and ax is axs[0,1]):ax.set_xlim(R[0],R[-1])
    fig.suptitle(f'{omega*27211.386245988:.3f} meV | t={c["times_fs"][index]:.3f} fs | positive-marginal gauge',fontsize=14)
    text=(f'Joint-density contours: decades down to {c["floor"]:g} of ONE movie-wide peak; grey = masked. '
          'Diagnostic fields, not Phase7 certification.')
    if c['sparse_preview']:text='SPARSE PREVIEW (not smooth dynamics). '+text
    fig.supxlabel(text,fontsize=9)


def render(paths,out,omega,fps=24,floor=1e-5,budget=1e-8,allow_sparse=False,
           families=('state','nuclear','photon'),dpi=100):
    """MP4 only; one loaded frame at a time. No PNG/PDF/GIF side products."""
    if not writers.is_available('ffmpeg'):raise RuntimeError('ffmpeg required for MP4')
    if fps<=0 or not 0<floor<1 or not 0<budget<1:raise ValueError('Invalid movie settings')
    if set(families)-{'state','nuclear','photon'}:raise ValueError('Unknown movie family')
    out=Path(out)
    if out.exists():raise FileExistsError('Choose new movie output directory')
    c=inspect(paths,omega,floor,budget,allow_sparse=allow_sparse)
    out.mkdir(parents=True)
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False})
    for family in families:
        fig=plt.figure(figsize=(15,8.5) if family=='state' else (12,8) if family=='nuclear' else (12,5.5),layout='constrained')
        writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-crf','19','-pix_fmt','yuv420p',
                             '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        with writer.saving(fig,str(out/f'vsc_{family}_movie.mp4'),dpi=dpi):
            for i,path in enumerate(paths):
                fig.clear();draw(fig,read_frame(path),c,omega,i,family);writer.grab_frame()
                if i%25==0:print(f'{family}: {i+1}/{len(paths)} actual frames',flush=True)
        plt.close(fig)
    c.update(fps=fps,frame_count=len(paths),duration_seconds=len(paths)/fps,
             density_convention='Physical q fields; Q densities divided by sqrt(omega); joint contours fixed to movie-wide peak',
             frame_sources=[str(p) for p in paths],phase7_pass=False)
    (out/'movie_manifest.json').write_text(json.dumps(c,indent=2)+'\n')
    return c
