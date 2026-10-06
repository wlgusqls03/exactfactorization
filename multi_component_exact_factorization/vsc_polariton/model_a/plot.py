"""Paper-layout reference figures and PG nested-MCEF movies, never gauge relabeling.

Fig3-style GI maps are explicitly NOT the total PG scalar; both are exported.
Fig4-style cond/geo cuts follow the paper's caption. Density uses a separate axis.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from .model import Config,potential

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':150,'axes.titlesize':11})


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def fig1(c,out):
    """Paper Fig1: q=0/1.5 R cuts, R=2/4 q cuts, g=0/.01."""
    R=np.linspace(0,8,601);q=np.linspace(-4,4,501)
    fig,ax=plt.subplots(2,2,figsize=(10,7),sharex='col',sharey=True)
    for row,g in enumerate([0.,c.g]):
        for fixed,color in zip([0,1.5],['#0072B2','#AA4499']):
            e=np.linalg.eigvalsh(potential(c,R,[fixed],g))[:,0]
            for j,ls in enumerate(['-','--']):ax[row,0].plot(R,e[:,j],ls,color=color,label=f'q={fixed:g}, S{j}')
        for fixed,color in zip([2,4],['#CC3377','#228833']):
            e=np.linalg.eigvalsh(potential(c,[fixed],q,g))[0]
            for j,ls in enumerate(['-','--']):ax[row,1].plot(q,e[:,j],ls,color=color,label=f'R={fixed:g}, S{j}')
        for a in ax[row]:a.set_ylim(-.03,.35);a.grid(alpha=.15);a.legend(fontsize=8,ncol=2);a.set_title(f'g={g:g}; no DSE')
        ax[row,0].set_ylabel('Cavity-adiabatic energy (Ha)')
    ax[1,0].set_xlabel(r'$R$ ($a_0$)');ax[1,1].set_xlabel(r'$q_c$ (a.u.)')
    fig.suptitle('Model A | Fig.1-style static reference | electronic S0/S1, not LP/UP')
    fig.tight_layout();fig.savefig(out/'fig1_reference.png');plt.close(fig)


def map_panel(ax,f,key,lim,eta=1e-6):
    """R horizontal, q vertical, black relative-density contours; no smoothing."""
    R,q=f['R'],f['q'];rho=f['rho_qR'];ok=rho>eta*rho.max()
    v=np.ma.masked_where(~ok|~np.isfinite(f[key]),f[key])
    im=ax.pcolormesh(R,q,v.T,shading='auto',cmap='RdBu_r',vmin=lim[0],vmax=lim[1])
    ax.contour(R,q,(rho/rho.max()).T,levels=[1e-5,1e-4,1e-3,.01,.03,.1,.2,.4,.6,.8],colors='k',linewidths=.5)
    used=np.where(np.any(rho>1e-5*rho.max(),axis=1))[0]
    if used.size:ax.set_xlim(max(R[0],R[used[0]]-.25),min(R[-1],R[used[-1]]+.25))
    ax.set_ylim(max(-8,q[0]),min(8,q[-1]));ax.set_xlabel(r'$R$ ($a_0$)');ax.set_ylabel(r'$q_c$ (a.u.)')
    ax.set_title(f't={float(f["time_au"]):g} au ({float(f["time_fs"]):.2f} fs)')
    return im


def snapshot_maps(frames,key,limits,out):
    """Six paper times with shared physical color scale and explicit quantity."""
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,f in zip(axes.flat,frames):im=map_panel(ax,f,key,limits)
    label={'GI':'GI = <V_PEN> + geo (NOT total PG TDPES)',
           'epsilon1':'Total positive-gauge epsilon1 = GI + GD'}[key]
    fig.suptitle('Fig.3-style | '+label);fig.colorbar(im,ax=axes,label='Ha',shrink=.8,extend='both')
    fig.savefig(out/f'fig3_{key}.png');plt.close(fig)


def draw_cuts(axes,f,c,total=False):
    """Paper Fig4 q-fixed LEFT, R-fixed RIGHT; colors link curves and density."""
    for col,(fixed_axis,coordinate,targets,colors) in enumerate([
            ('q',f['R_line'],[0,1.5],['#D55E00','#009E73']),
            ('R',f['q_line'],[2,4],['#0072B2','#CC79A7'])]):
        ax=axes[col];density=ax.twinx();density.set_ylim(0,1.0);density.set_ylabel('Joint density (a.u.)',color='#666666')
        for n,(target,color) in enumerate(zip(targets,colors)):
            prefix=f'cut_{fixed_axis}{n}_';at=float(f[prefix+'at'])
            rho=f[prefix+'rho_qR'];ok=rho>1e-5*f['rho_qR'].max()
            cond=f[prefix+'cond'];geo=f[prefix+'geo_q']+f[prefix+'geo_R']
            ax.plot(coordinate,np.where(ok,cond,np.nan),color=color,marker='o',markevery=max(1,len(coordinate)//25),ms=2,label=f'<V> {fixed_axis}={at:g}')
            ax.plot(coordinate,np.where(ok,geo,np.nan),color=color,ls=':',label=f'geo {fixed_axis}={at:g}')
            if total:ax.plot(coordinate,np.where(ok,f[prefix+'epsilon1'],np.nan),color=color,ls='-.',label=f'PG total {fixed_axis}={at:g}')
            refs=np.linalg.eigvalsh(potential(c,coordinate,[at]))[:,0] if fixed_axis=='q' else np.linalg.eigvalsh(potential(c,[at],coordinate))[0]
            for j,ls in enumerate(['-','--']):ax.plot(coordinate,refs[:,j],ls,color=color,alpha=.35,lw=1)
            if rho.max()>0:density.plot(coordinate,rho,color=color,lw=.8,alpha=.55)
        ax.set_xlim((0,8) if col==0 else (-8,8));ax.set_ylim(-.15,.5)
        ax.set_xlabel(r'$R$ ($a_0$)' if col==0 else r'$q_c$ (a.u.)');ax.set_ylabel('Energy (Ha)')
        ax.grid(alpha=.15);ax.legend(fontsize=7,ncol=2,loc='upper right')


def render(run,out,movies=True,fps=24):
    """Stream compact fields. Movies use actual times, never interpolated physics."""
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new plot folder')
    out.mkdir(parents=True,exist_ok=True)
    ident=json.loads((run/'identity.json').read_text());c=Config(**ident['config'])
    paths=sorted((run/'fields').glob('fields_*.npz'))
    if not paths:raise ValueError('No compact fields; rerun without --no-fields')
    times=np.array([float(load(p)['time_au']) for p in paths])
    if movies and len(times)>1 and np.max(np.diff(times))>5.000001:
        raise ValueError('Sparse event waves: use --no-movies; do not present six snapshots as a dense movie')
    selected=[load(paths[int(np.argmin(abs(times-t)))]) for t in [0,250,500,750,1000,1250]]
    fig1(c,out)
    if (run/'observables.json').exists():
        from .photon_number import model_series,draw
        draw([model_series(run)],out/'photon')
    # Time-independent scales predeclared for reading paper-like values, no per-frame normalization.
    snapshot_maps(selected,'GI',(-.05,1.4),out)
    snapshot_maps(selected,'epsilon1',(-.2,.3),out)
    fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for axesrow,t in zip(axes,[250,750,1250]):
        f=load(paths[int(np.argmin(abs(times-t)))]);draw_cuts(axesrow,f,c)
        axesrow[0].set_title(f't={float(f["time_au"]):g} au')
    fig.suptitle('Fig.4-style | <V_PEN> and geo, NOT total scalar | faint solid/dashed: S0/S1')
    fig.savefig(out/'fig4_components.png');plt.close(fig)
    if movies:
        if not matplotlib.animation.writers.is_available('ffmpeg'):raise RuntimeError('ffmpeg required; snapshots already saved')
        for family in ['tdpes1','vectors','outer','cuts']:
            fig=plt.figure(figsize=(12,6))
            writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','20','-threads','2'])
            with writer.saving(fig,str(out/f'{family}_positive.mp4'),110):
                for path in paths:
                    f=load(path);fig.clear()
                    if family=='tdpes1':
                        axes=fig.subplots(1,2)
                        for ax,k,lim,label in zip(axes,['GI','epsilon1'],[(-.05,1.4),(-.2,.3)],['GI (not total)','Total PG epsilon1']):
                            im=map_panel(ax,f,k,lim);ax.set_title(label);fig.colorbar(im,ax=ax,label='Ha',extend='both')
                    elif family=='vectors':
                        axes=fig.subplots(1,3)
                        for ax,k,lim in zip(axes[:2],['a','b'],[(-2,2),(-60,60)]):
                            im=map_panel(ax,f,k,lim);ax.set_title(k+' (momentum, a.u.)');fig.colorbar(im,ax=ax,extend='both')
                        axes[2].plot(f['R_line'],np.where(f['rho_R']>1e-6*f['rho_R'].max(),f['alpha'],np.nan));axes[2].set_ylim(-60,60)
                        axes[2].set(xlabel='R (a0)',ylabel='alpha (momentum, a.u.)')
                    elif family=='outer':
                        axes=fig.subplots(1,2);R=f['R_line'];ok=f['rho_R']>1e-6*f['rho_R'].max()
                        for k,label in [('epsilon2','Total PG'),('epsilon2_cond','Conditional'),('epsilon2_geo','Geometric'),('epsilon2_GD','GD')]:
                            axes[0].plot(R,np.where(ok,f[k],np.nan),label=label)
                        axes[0].set_ylim(-.4,.5);axes[0].set(xlabel='R (a0)',ylabel='epsilon2 (Ha)');axes[0].legend(fontsize=8)
                        density=axes[0].twinx();density.plot(R,f['rho_R'],'k:',lw=1);density.set_ylabel('|chi|^2');density.set_ylim(0,3)
                        axes[1].plot(R,np.where(ok,f['force'],np.nan));axes[1].set_ylim(-.2,.2)
                        axes[1].set(xlabel='R (a0)',ylabel='-dR epsilon2 + dt alpha (Ha/a0)')
                    else:draw_cuts(fig.subplots(1,2),f,c,total=True)
                    fig.suptitle(f'Model A | g={c.g:g}, omega={c.omega:g} Ha | PG | t={float(f["time_au"]):.1f} au / {float(f["time_fs"]):.2f} fs')
                    fig.tight_layout(rect=(0,0,1,.94));writer.grab_frame()
            plt.close(fig)
    clipped={k:[] for k in ['GI','epsilon1','a','b']}
    for path in paths:
        f=load(path);rho=f['rho_qR'];mask=rho>1e-6*rho.max()
        for k,(lo,hi) in zip(clipped,[(-.05,1.4),(-.2,.3),(-2,2),(-60,60)]):
            clipped[k].append(float(rho[mask&((f[k]<lo)|(f[k]>hi))].sum()/rho.sum()))
    (out/'manifest.json').write_text(json.dumps(dict(frames=len(paths),times_au=times.tolist(),fps=fps,
        clipped_joint_probability=clipped,
        interpolation=False,gauge='positive F, chi, Lambda',paper_reproduction_certified=False,
        fig3_GI='Not total PG scalar; exported separately to investigate paper convention',
        display='Maps decimated after native EF; density mask 1e-6, contour down to 1e-5',
        clipping='Fixed limits; extended map colorbars; cut energy range [-.15,.5] Ha'),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--no-movies',action='store_true')
    a=ap.parse_args();render(a.run,a.out,not a.no_movies)
