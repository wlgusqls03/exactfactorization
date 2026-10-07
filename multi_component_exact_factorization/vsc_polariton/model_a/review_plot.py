"""Rerender EXISTING compact fields: shared time/case scales, dense contours.

Display changes only; never change TDSE/EF data or numerical acceptance tests.
Weighted display quantiles are disclosed, and clipped probability is recorded.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from .model import Config,potential
from .plot import load,fig1
from .run import atomic_json

LEVELS=[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,.01,.02,.04,.07,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9]
ETA=1e-6
QUANTILE=.999
MAPS=('GI','epsilon1','a','b')


def quantile(values,weights,p):
    """Display-only weighted quantile, never a scientific support/gate choice."""
    ok=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    if not ok.any():return 0.
    v=values[ok];w=weights[ok];i=np.argsort(v)
    return float(np.interp(p*np.sum(w),np.cumsum(w[i]),v[i]))


def scan(groups):
    """Global limits across BOTH cases and ALL times, not frame autoscaling."""
    extrema={k:[0.,0.] for k in (*MAPS,'alpha','force','epsilon2','cut_energy')}
    density_max=0.;rho_max=0.
    for paths in groups.values():
        for path in paths:
            f=load(path);rho=f['rho_qR'];outer=f['rho_R'];mask=rho>ETA*rho.max();rm=outer>ETA*outer.max()
            for k in MAPS:
                for n,p in enumerate([1-QUANTILE,QUANTILE]):
                    v=quantile(f[k][mask],rho[mask],p)
                    extrema[k][n]=(min if n==0 else max)(extrema[k][n],v)
            for k in ['alpha','force','epsilon2']:
                # Outer lines show the ENTIRE finite occupied range, including alpha>60.
                keys=['epsilon2','epsilon2_cond','epsilon2_geo','epsilon2_GD'] if k=='epsilon2' else [k]
                for key in keys:
                    vals=f[key][rm&np.isfinite(f[key])]
                    if vals.size:
                        extrema[k][0]=min(extrema[k][0],float(vals.min()))
                        extrema[k][1]=max(extrema[k][1],float(vals.max()))
            rho_max=max(rho_max,float(outer.max()))
            for axis in ['q','R']:
                for n in range(2):
                    prefix=f'cut_{axis}{n}_';weight=f[prefix+'rho_qR'];ok=weight>ETA*rho.max()
                    density_max=max(density_max,float(weight.max()))
                    for k in ['cond','GI','GD','epsilon1','geo_q','geo_R']:
                        vals=f[prefix+k]
                        for side,p in enumerate([1-QUANTILE,QUANTILE]):
                            v=quantile(vals[ok],weight[ok],p)
                            extrema['cut_energy'][side]=(min if side==0 else max)(extrema['cut_energy'][side],v)
    for k,(lo,hi) in extrema.items():
        if k in ['a','b','alpha','force']:
            v=max(abs(lo),abs(hi),1e-8)*1.05;extrema[k]=[-v,v]
        else:
            margin=.05*max(hi-lo,1e-3);extrema[k]=[lo-margin,hi+margin]
    return dict(limits=extrema,cut_density_max=density_max*1.05,rho_max=rho_max*1.05,
                density_threshold=ETA,contours=LEVELS,display_quantile=QUANTILE,
                note='Fixed across cases/time; maps/cuts use weighted display quantiles. Outer lines use full occupied extrema.')


def map_axis(ax,f,key,limits):
    rho=f['rho_qR'];mask=rho>ETA*rho.max();v=np.ma.masked_where(~mask|~np.isfinite(f[key]),f[key])
    im=ax.pcolormesh(f['R'],f['q'],v.T,cmap='RdBu_r',vmin=limits[0],vmax=limits[1],shading='auto')
    ax.contour(f['R'],f['q'],(rho/rho.max()).T,levels=LEVELS,colors='k',linewidths=.45,linestyles='dashed',alpha=.65)
    ax.axvline(4,color='#009E73',ls='--',lw=1.4)
    ax.set(xlim=(0,6),ylim=(-8,8),xlabel=r'$R$ ($a_0$)',ylabel=r'$q_c$ (a.u.)')
    lo,hi=limits;clipped=float(rho[mask&((f[key]<lo)|(f[key]>hi))].sum()/rho.sum())
    ax.set_title(f'{key}; scale-clipped mass {100*clipped:.3g}%',fontsize=10)
    return im,clipped


def cuts(axes,f,c,style):
    """Two directions, GI/GD/geo/total and static S0/S1, separate density axes.

    Values outside main energy band are marked at its border, with FULL extrema
    printed; they are not silently discarded or presented as a converged barrier.
    """
    extrema=[];lo,hi=style['limits']['cut_energy']
    for col,(axis,x,colors) in enumerate([('q',f['R_line'],['#D55E00','#0072B2']),('R',f['q_line'],['#D55E00','#0072B2'])]):
        ax=axes[col];density=ax.twinx();density.set_ylim(0,style['cut_density_max'])
        xmin,xmax=(0,6) if col==0 else (-8,8)
        density.set_ylabel(r'$|F|^2$ (joint density)',color='#666666')
        full=[];labels=[]
        for n,color in enumerate(colors):
            prefix=f'cut_{axis}{n}_';at=float(f[prefix+'at']);labels.append(f'{axis}={at:g}')
            rho=f[prefix+'rho_qR'];ok=rho>ETA*f['rho_qR'].max()
            terms={'cond':f[prefix+'cond'],'GI':f[prefix+'GI'],'GD':f[prefix+'GD'],
                   'geo':f[prefix+'geo_q']+f[prefix+'geo_R'],'total':f[prefix+'epsilon1']}
            for key,ls,lw in [('cond','-',.8),('GI','--',1.2),('GD','-.',1.1),('geo',':',1.6),('total','-',2)]:
                v=terms[key];valid=ok&np.isfinite(v)
                ax.plot(x,np.where(valid,v,np.nan),ls,color=color,lw=lw,
                        marker='.' if key=='cond' else None,markevery=max(1,len(x)//30),ms=3)
                for over,y,marker in [(valid&(v>hi),hi,'^'),(valid&(v<lo),lo,'v')]:
                    over=over&(x>=xmin)&(x<=xmax)
                    if over.any():ax.plot(x[over],np.full(over.sum(),y),marker,ms=3,color=color,clip_on=False,ls='none')
                if valid.any():full.extend([float(v[valid].min()),float(v[valid].max())])
            refs=np.linalg.eigvalsh(potential(c,x,[at]))[:,0] if axis=='q' else np.linalg.eigvalsh(potential(c,[at],x))[0]
            for j,ls in enumerate(['-','--']):ax.plot(x,refs[:,j],ls,color=color,alpha=.2,lw=.8)
            density.plot(x,rho,ls=(0,(1,1)),color=color,alpha=.45,lw=1)
        ext=[min(full),max(full)] if full else [None,None];extrema.append(ext)
        ax.set(xlim=(0,6) if col==0 else (-8,8),ylim=(lo,hi),xlabel='R (a0)' if col==0 else 'q_c (a.u.)',ylabel='Energy (Ha)')
        if col==0:ax.axvline(4,color='#009E73',ls='--',lw=1)
        handles=[Line2D([],[],color=color,lw=2,label=label) for color,label in zip(colors,labels)]
        handles += [Line2D([],[],color='k',ls=ls,label=key,marker='.' if key=='cond' else None) for key,ls in [('total','-'),('cond','-'),('GI','--'),('GD','-.'),('geo',':')]]
        handles += [Line2D([],[],color='#666666',ls=(0,(1,1)),label='density (right)')]
        ax.legend(handles=handles,fontsize=7,ncol=3,loc='upper right')
        ax.set_title('No occupied cut' if not full else f'Full occupied component range: [{ext[0]:.3g}, {ext[1]:.3g}] Ha',fontsize=9)
        ax.grid(alpha=.12)
    return extrema


def preflight(root,movies=True):
    """Check dense fields and encoder before spending time on event analysis."""
    root=Path(root)
    groups={name:sorted((root/name/'fields').glob('fields_*.npz')) for name in ['coupled','free']}
    if any(not paths for paths in groups.values()):raise FileNotFoundError('Original coupled/free fields required on server')
    if movies and not matplotlib.animation.writers.is_available('ffmpeg'):raise RuntimeError('ffmpeg is required')
    return groups


def render(root,out,movies=True):
    """New figures/MP4 only; original fields and previous renderings remain intact."""
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    groups=preflight(root,movies)
    out.mkdir();print('Scan fixed common scales across all saved frames',flush=True)
    style=scan(groups);atomic_json(out/'display_scales.json',style);manifests={}
    for name,paths in groups.items():
        folder=out/name;folder.mkdir();c=Config(**json.loads((root/name/'identity.json').read_text())['config'])
        times=[float(load(p)['time_au']) for p in paths]
        if movies and len(times)>1 and np.max(np.diff(times))>5.000001:raise ValueError('Sparse events are not a dense movie')
        fig1(c,folder);clipped=[];cut_ranges=[]
        # Paper Fig3-style six times and Fig4-style three pairs of cuts.
        for key in ['epsilon1','GI']:
            fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
            for axis,t in zip(ax.flat,[0,250,500,750,1000,1250]):
                f=load(paths[int(np.argmin(abs(np.asarray(times)-t)))]);im,_=map_axis(axis,f,key,style['limits'][key]);axis.set_title(f't={float(f["time_au"]):g} au')
            fig.colorbar(im,ax=ax,label='Ha',extend='both');fig.suptitle(f'{name}: {key} | shared scale | diagnostic, not certified')
            fig.savefig(folder/f'fig3_{key}.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(3,2,figsize=(13,11),layout='constrained')
        for row,t in zip(ax,[250,750,1250]):
            f=load(paths[int(np.argmin(abs(np.asarray(times)-t)))]);cuts(row,f,c,style)
            row[0].text(.02,.02,f't={float(f["time_au"]):g} au',transform=row[0].transAxes)
        fig.suptitle('PG decomposition | faint solid/dashed: static S0/S1 | triangles: outside energy band')
        fig.savefig(folder/'fig4_components.png',dpi=150);plt.close(fig)
        if movies:
            for family in ['vectors','tdpes1','outer','cuts']:
                print(f'Render {name}/{family}: {len(paths)} actual frames',flush=True)
                fig=plt.figure(figsize=(13,6));writer=FFMpegWriter(fps=24,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','20','-threads','2'])
                with writer.saving(fig,str(folder/f'{family}_positive.mp4'),110):
                    for path in paths:
                        f=load(path);fig.clear()
                        if family in ['vectors','tdpes1']:
                            axes=fig.subplots(1,3 if family=='vectors' else 2)
                            keys=['a','b'] if family=='vectors' else ['GI','epsilon1'];record=dict(time_au=float(f['time_au']))
                            for ax,key in zip(axes,keys):
                                im,clip=map_axis(ax,f,key,style['limits'][key]);fig.colorbar(im,ax=ax,extend='both',label='Momentum (a.u.)' if family=='vectors' else 'Ha');record[key]=clip
                            clipped.append(record)
                            if family=='vectors':
                                R=f['R_line'];ok=f['rho_R']>ETA*f['rho_R'].max()
                                axes[2].plot(R,np.where(ok,f['alpha'],np.nan));axes[2].set(xlim=(0,6),ylim=style['limits']['alpha'],xlabel='R (a0)',ylabel='alpha (a.u.)')
                                axes[2].axvline(4,color='#009E73',ls='--')
                        elif family=='outer':
                            axes=fig.subplots(1,2);R=f['R_line'];ok=f['rho_R']>ETA*f['rho_R'].max()
                            for key,label in [('epsilon2','Total PG'),('epsilon2_cond','Conditional'),('epsilon2_geo','geo'),('epsilon2_GD','GD')]:axes[0].plot(R,np.where(ok,f[key],np.nan),label=label)
                            axes[0].set(ylim=style['limits']['epsilon2'],xlabel='R (a0)',ylabel='epsilon2 (Ha)');axes[0].legend(fontsize=8)
                            den=axes[0].twinx();den.plot(R,f['rho_R'],'k:',lw=1);den.set(ylim=(0,style['rho_max']),ylabel='|chi|^2')
                            axes[1].plot(R,np.where(ok,f['force'],np.nan));axes[1].set(ylim=style['limits']['force'],xlabel='R (a0)',ylabel='-dR epsilon2 + dt alpha (Ha/a0)')
                            for ax in axes:ax.set_xlim(0,6);ax.axvline(4,color='#009E73',ls='--')
                        else:cut_ranges.append(dict(time_au=float(f['time_au']),ranges=cuts(fig.subplots(1,2),f,c,style)))
                        fig.suptitle(f'Model A | g={c.g:g}, omega={c.omega:g} Ha | PG | t={float(f["time_au"]):g} au ({float(f["time_fs"]):.2f} fs)\nDiagnostic display; TDPES1 convergence NOT certified',fontsize=11)
                        fig.tight_layout(rect=(0,0,1,.92));writer.grab_frame()
                plt.close(fig)
        manifests[name]=dict(frames=len(paths),times_au=times,fps=24,interpolation=False,
            clipped_probability=clipped,full_cut_energy_ranges=cut_ranges,
            axes_R=[0,6],axes_q=[-8,8],note='R=4 is the population dividing surface, not automatically a static barrier.')
    atomic_json(out/'manifest.json',dict(cases=manifests,style=style,certified=False))
    return manifests
