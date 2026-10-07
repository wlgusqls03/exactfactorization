"""Electronic channel densities on bare BO / cavity-adiabatic references.

Neither electronic pair is the vibrational LP/UP pair. No phase unwrapping,
no two-state truncation of the Model A spinor, no per-frame normalization.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from .model import Config,potential
from .run import atomic_json


class Channels:
    """Reuse a fixed eigenbasis: V_BO(R) and full local V(R,q), atomic units.

    Input psi [NR,Nq,2]; projections <S_j|psi>, not diagonal diabatic labels.
    Sum of the TWO projected densities must reproduce joint density pointwise.
    """
    def __init__(self,c):
        self.c=c;self.R,self.q=c.grids()
        self.Ebo,self.bo=np.linalg.eigh(potential(c,self.R,[0.],0)[:,0])
        self.Ecbo,self.cbo=np.linalg.eigh(potential(c,self.R,self.q))

    def project(self,u):
        c=self.c;dr=self.R[1]-self.R[0];dq=self.q[1]-self.q[0]
        rho=np.sum(abs(u)**2,axis=-1)
        weights={'BO':abs(np.einsum('rjk,rqj->rqk',self.bo.conj(),u))**2,
                 'CBO':abs(np.einsum('rqjk,rqj->rqk',self.cbo.conj(),u))**2}
        out=dict(R=self.R,q=self.q,rho_R=rho.sum(axis=1)*dq)
        for basis,w in weights.items():
            out[basis+'_global']=w.sum(axis=(0,1))*dr*dq
            out[basis+'_rho_R']=w.sum(axis=1)*dq
            out[basis+'_closure']=float(np.max(abs(w.sum(axis=-1)-rho)))
            for axis,targets in [('q',[0.,1.5]),('R',[2.,4.])]:
                grid=self.q if axis=='q' else self.R
                for n,target in enumerate(targets):
                    i=int(np.argmin(abs(grid-target)));prefix=f'{axis}{n}_'
                    out[prefix+'at']=grid[i]
                    out[basis+'_'+prefix+'density']=w[:,i,:] if axis=='q' else w[i,:,:]
                    # BO reference includes common photon harmonic energy only.
                    eb=self.Ebo+.5*c.omega**2*self.q[i]**2 if axis=='q' else self.Ebo[i][None,:]+.5*c.omega**2*self.q[:,None]**2
                    ec=self.Ecbo[:,i,:] if axis=='q' else self.Ecbo[i,:,:]
                    out[basis+'_'+prefix+'energy']=eb if basis=='BO' else ec
        if max(out['BO_closure'],out['CBO_closure'])>1e-10:
            raise ValueError('Electronic channel projection failed')
        return out


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def render(run,out):
    """Two movies: fixed q pair vs R, fixed R pair vs q; also global populations.

    Filled channel density is offset from its energy curve ONLY for display.
    One fixed scale for both channels, both slices, both bases and all times.
    Global BO population vs photon N is plotted separately, not inferred from cuts.
    """
    run=Path(run);out=Path(out)
    if out.exists():raise FileExistsError(out)
    paths=sorted((run/'states').glob('states_*.npz'))
    if not paths:raise FileNotFoundError('No wave-projected state frames; old EF fields alone are insufficient')
    if not matplotlib.animation.writers.is_available('ffmpeg'):raise RuntimeError('ffmpeg required')
    maxima={'q':0.,'R':0.};times=[];pop=[]
    erange={a:[float('inf'),float('-inf')] for a in maxima}
    for p in paths:
        f=load(p);times.append(float(f['time_au']));pop.append(f['CBO_global'])
        for a in maxima:
            for n in range(2):
                for b in ['BO','CBO']:
                    maxima[a]=max(maxima[a],float(f[f'{b}_{a}{n}_density'].max()))
                    e=f[f'{b}_{a}{n}_energy'];erange[a][0]=min(erange[a][0],float(e.min()));erange[a][1]=max(erange[a][1],float(e.max()))
    if len(times)>1 and (np.any(np.diff(times)<=0) or np.max(np.diff(times))>5.000001):
        raise ValueError('State movies require ordered dense frames, not sparse-event interpolation')
    out.mkdir();colors=['#0072B2','#D55E00'];scale={a:.08/max(v,1e-30) for a,v in maxima.items()}
    for fixed,name in [('q','states_R_cuts'),('R','states_q_cuts')]:
        fig=plt.figure(figsize=(12,8));writer=FFMpegWriter(fps=24,codec='libx264',extra_args=['-pix_fmt','yuv420p','-threads','2'])
        with writer.saving(fig,str(out/(name+'.mp4')),100):
            for path in paths:
                f=load(path);fig.clear();axes=fig.subplots(2,2,sharex=True,sharey=True)
                x=f['R'] if fixed=='q' else f['q']
                for row,basis in enumerate(['BO','CBO']):
                    for n in range(2):
                        ax=axes[row,n];key=f'{basis}_{fixed}{n}_'
                        e=f[key+'energy'];w=f[key+'density']
                        for j in range(2):
                            ax.plot(x,e[:,j],color=colors[j],lw=1,label=f'{basis} {j}')
                            ax.fill_between(x,e[:,j],e[:,j]+scale[fixed]*w[:,j],color=colors[j],alpha=.4)
                            ax.plot(x,e[:,j]+scale[fixed]*w[:,j],color=colors[j],lw=1.2)
                        ax.set(title=f'{basis} | fixed {fixed}={float(f[f"{fixed}{n}_at"]):g}',
                               xlabel=r'$R$ ($a_0$)' if fixed=='q' else r'$q_c$ (a.u.)',
                               ylabel='Reference energy + scaled density (Ha)',
                               ylim=(erange[fixed][0]-.02,erange[fixed][1]+.1))
                        if fixed=='q':ax.axvline(4,color='0.4',ls='--',lw=.8)
                        ax.legend(fontsize=8);ax.grid(alpha=.12)
                fig.suptitle(f'Model A electronic channels | t={float(f["time_au"]):g} au ({float(f["time_au"])*.024188843265857:.2f} fs)\n'
                    f'Global bare BO P0/P1={f["BO_global"][0]:.5f}/{f["BO_global"][1]:.5f} | NOT vibrational LP/UP',fontsize=11)
                fig.supxlabel(f'Filled height = joint channel density × {scale[fixed]:.4g}; fixed across time, no slice normalization.\n'
                    'BO row: bare electronic energy + photon harmonic reference. CBO row: eigenvalues of full local V(R,q).',fontsize=9)
                fig.tight_layout(rect=(0,.06,1,.92));writer.grab_frame()
        plt.close(fig)
    rows=json.loads((run/'observables.json').read_text());ts=np.array([r['time_au'] for r in rows])
    fig,ax=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
    for j in range(2):
        ax[0].plot(ts,[r[f'P_S{j}'] for r in rows],color=colors[j],label=rf'Bare BO $P_{{S_{j}}}$')
        ax[0].plot(times,np.asarray(pop)[:,j],'--',color=colors[j],label=f'CBO P{j}')
    ax[0].set(ylabel='Electronic population',ylim=(-.02,1.02));ax[0].legend(ncol=2,fontsize=8)
    ax[1].plot(ts,[r['n_ph']/r['norm'] for r in rows]);ax[1].set_ylabel(r'$\langle N\rangle$')
    ax[2].plot(ts,[r.get('P_right_spectral',r['P_right']) for r in rows]);ax[2].set(ylabel=r'$P(R\geq4)$',xlabel='Time (au)')
    for a in ax:a.axvline(1250,color='0.4',ls='--');a.grid(alpha=.15)
    fig.savefig(out/'populations_and_photons.png',dpi=170);plt.close(fig)
    atomic_json(out/'manifest.json',dict(frames=len(paths),times_au=times,scale=scale,not_LP_UP=True,
        basis='Bare electronic BO and cavity-adiabatic electronic eigenvectors of local V',
        density='Joint state-projected density, not conditional population; sum_j reproduces joint density',
        display='Energy offset plus scaled probability density; height is NOT kinetic/total energy',
        certified=False))
