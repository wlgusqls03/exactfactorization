"""Readable component cuts: potential zoom, full geometric range, density apart."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from .review_plot import load,ETA
from .run import atomic_json
from .model import Config,potential


def draw(fig,f,geo_max,density_max,zoom,c=None):
    """Three rows x two directions; no smoothing, no removing off-scale peaks."""
    axes=fig.subplots(3,2,sharex='col');clipped={}
    for col,axis in enumerate(['q','R']):
        x=f['R_line'] if axis=='q' else f['q_line'];ax=axes[0,col]
        for n,color in enumerate(['#D55E00','#0072B2']):
            prefix=f'cut_{axis}{n}_';at=float(f[prefix+'at']);rho=f[prefix+'rho_qR'];ok=rho>ETA*f['rho_qR'].max()
            label=f'{axis}={at:g}'
            if c is not None:
                refs=np.linalg.eigvalsh(potential(c,x,[at]))[:,0] if axis=='q' else np.linalg.eigvalsh(potential(c,[at],x))[0]
                for j,ls in enumerate(['-','--']):ax.plot(x,refs[:,j],ls,color=color,alpha=.25,lw=.8)
            for key,style in [('epsilon1','-'),('cond','--'),('GI',':'),('GD','-.')]:
                y=f[prefix+key];valid=ok&np.isfinite(y)
                ax.plot(x,np.where(valid,y,np.nan),style,color=color,lw=1.3,label=label+' '+('total' if key=='epsilon1' else key))
                clipped[prefix+key]=int(np.sum(valid&((y<zoom[0])|(y>zoom[1]))))
                for m,v in [(valid&(y>zoom[1]),zoom[1]),(valid&(y<zoom[0]),zoom[0])]:
                    if m.any():ax.plot(x[m],np.full(m.sum(),v),'|',color=color,ms=5)
            for key,style in [('geo_q','-'),('geo_R','--')]:
                axes[1,col].plot(x,np.where(ok,f[prefix+key],np.nan),style,color=color,lw=1.2,label=label+' '+key)
            axes[2,col].plot(x,rho,color=color,label=label)
        ax.set(ylim=zoom,ylabel='Energy (Ha)',title='Potential zoom; edge ticks = outside band')
        axes[1,col].set(ylim=(-.01*geo_max,1.05*geo_max),ylabel='Geometric terms (Ha)',title='Full occupied geometric range (fixed in time)')
        axes[2,col].set(ylim=(0,1.05*density_max),ylabel=r'$|F(q_c,R,t)|^2$',xlabel=r'$R$ ($a_0$)' if axis=='q' else r'$q_c$ (a.u.)')
        for a in axes[:,col]:
            a.set_xlim(x[0],x[-1]);a.grid(alpha=.15);a.legend(fontsize=7,ncol=2)
            if axis=='q':a.axvline(4,color='0.4',ls='--',lw=.8)
    fig.suptitle(f'PG components | t={float(f["time_au"]):g} au | diagnostic, not certified\nFaint solid/dashed references: CBO 0/1',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    return clipped


def render(run,out,zoom=(-.5,.6)):
    """Preserve full geometric peaks while resolving ~0.1 Ha potential structure."""
    run=Path(run);out=Path(out)
    if out.exists():raise FileExistsError(out)
    paths=sorted((run/'fields').glob('fields_*.npz'))
    if not paths:raise FileNotFoundError(run/'fields')
    c=Config(**json.loads((run/'identity.json').read_text())['config'])
    gm=1e-6;dm=1e-6
    for path in paths:
        f=load(path)
        for axis in ['q','R']:
            for n in range(2):
                prefix=f'cut_{axis}{n}_';w=f[prefix+'rho_qR'];ok=w>ETA*f['rho_qR'].max();dm=max(dm,float(w.max()))
                for k in ['geo_q','geo_R']:
                    v=f[prefix+k];valid=ok&np.isfinite(v)
                    if valid.any():gm=max(gm,float(abs(v[valid]).max()))
    out.mkdir();records=[];fig=plt.figure(figsize=(13,10))
    writer=FFMpegWriter(fps=24,codec='libx264',extra_args=['-pix_fmt','yuv420p','-threads','2'])
    with writer.saving(fig,str(out/'components_resolved.mp4'),100):
        for path in paths:
            f=load(path);fig.clear();record=draw(fig,f,gm,dm,zoom,c);writer.grab_frame()
            records.append(dict(time_au=float(f['time_au']),off_band_counts=record))
    plt.close(fig)
    atomic_json(out/'manifest.json',dict(zoom_Ha=zoom,geometric_abs_max=gm,density_max=dm,frames=records,
        note='Display zoom only, full geometric scale retained; not a convergence fix.'))
