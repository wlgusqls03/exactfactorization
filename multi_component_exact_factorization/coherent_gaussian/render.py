"""Postprocess small Gaussian checkpoints; plotting q points are NOT propagated.

No exact outer EF scalar/connection is fabricated from a fixed-R calculation
or a classical ensemble. Current/conditional electronic populations are safe
reconstructions of the stored full complex Gaussian state.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.ticker import MaxNLocator
from .core import State, evaluate
from .models import Pen, RealPen, McefLocal
from .experiment import Reference, oscillator_values
from ..pen_perspectives_2026.model import AU_FS


COLORS={'qd':'black','el':'#d95f02','pol':'#1b9e77','gaussian':'#6b46c1'}
NAMES={'qd':'QD (finite Fock)','el':'EL CTMQC','pol':'POL CTMQC',
       'gaussian':'Gaussian-projected POL'}


def style():
    plt.rcParams.update({'font.size':12,'axes.labelsize':14,'axes.titlesize':15,
        'legend.fontsize':10,'figure.titlesize':19,'lines.linewidth':2,
        'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})


def model_from_config(c):
    p=c['physics']
    if c['model']=='pen-real-electron':
        return RealPen(R=c['R'],nx=p['nx'],g=p['g'],dse=p['dse'],omega=p['omega'],
                       representation=c['representation'],nquad=c['nquad'])
    if c['model'].startswith('pen-'):
        return Pen(model=c['model'][4:],g=p['g'],dse=p['dse'],R=c['R'],
            omega=p['omega'],representation=c['representation'],nquad=c['nquad'])
    return McefLocal(R=c['R'],nx=p['nx'],degree=p['degree'],radius=p['radius'],
        representation=c['representation'],nstates=c['ne'],nquad=c['nquad'])


def render_fixed(run, movies=False):
    run=Path(run);style()
    c=json.loads((run/'config.json').read_text())
    rows=json.loads((run/'metrics.json').read_text())
    if not rows:return
    a=np.load(run/'states.npz');t=a['t_fs'];m=model_from_config(c)
    ref=Reference(m,nf=c['nf_reference'])
    out=run/'plots';out.mkdir(exist_ok=True)
    states=[State.unpack(y,c['ne'],c['K']) for y in a['y']]
    # One common range over the COMPLETE saved sequence, integer ticks.
    center=np.concatenate([m.origin+2*m.sigma*s.z.real for s in states])
    q=np.linspace(min(m.origin-5*m.sigma,center.min()-4*m.sigma),
                  max(m.origin+5*m.sigma,center.max()+4*m.sigma),401)
    G=[];density=[];current=[];pop=[];reference=[]
    f,_=oscillator_values(q,m.sigma,m.origin,ref.nf)
    for ti,s in zip(t,states):
        gg,dg,lam,cc=evaluate(s,q,m.sigma,m.origin,m.basis(q))
        G.append(gg);density.append(lam**2)
        aq=np.imag(np.sum(gg.conj()*dg,axis=1))/np.maximum(lam**2,1e-100)
        aq[lam**2<1e-5*max(lam**2)]=np.nan
        current.append(aq)
        u=m.Ubo if isinstance(m,Pen) else np.linalg.eigh(m.potential(q))[1]
        bc=cc@u.conj() if u.ndim==2 else np.einsum('nxk,nx->nk',u.conj(),cc)
        pop.append(abs(bc[:,:min(3,c['ne'])])**2)
        vr=f@ref.state(ti/AU_FS).reshape(m.nprimitive,ref.nf).T
        reference.append(np.sum(abs(vr)**2,axis=1))
    density,reference,current=np.array(density),np.array(reference),np.array(current)
    lim=1.12*max(density.max(),reference.max())
    am=max(.1,np.nanmax(abs(current)))
    fig,ax=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    def draw(i):
        for z in ax.flat:z.clear()
        title='PEN fixed-R benchmark' if isinstance(m,Pen) else 'MCEF LOCAL surrogate, frozen heavy R'
        fig.suptitle(f'{title} | K={c["K"]}, {c["representation"].upper()} | t={t[i]:.2f} fs')
        x=ax[0,0];x.fill_between(q,density[i],color='.75',alpha=.6)
        x.plot(q,reference[i],'k--',label='Uncompressed reference')
        x.plot(q,density[i],color='#6b46c1',label='Gaussian state')
        for Q in m.origin+2*m.sigma*states[i].z.real:x.axvline(Q,color='.6',lw=.7,alpha=.5)
        x.set(title=r'(A) $|\Lambda|^2$: coherent interference retained',ylabel='Density',xlabel='q (a.u.)',ylim=(0,lim),xlim=(q[0],q[-1]))
        x.legend(loc='upper right')
        x=ax[0,1]
        for k in range(pop[i].shape[1]):x.plot(q,np.where(density[i]>1e-5*density[i].max(),pop[i][:,k],np.nan),label=f'BO {k}')
        x.set(title=r'(B) Conditional electronic $|C_k(q)|^2$',ylabel='Population',xlabel='q (a.u.)',ylim=(-.05,1.05),xlim=(q[0],q[-1]));x.legend()
        x=ax[0,2];x.plot(q,current[i]);x.axhline(0,color='.4',ls='--',lw=1)
        x.set(title=r'(C) $a(q)=\mathrm{Im}(G^\dagger\partial_qG)/|\Lambda|^2$',ylabel='Momentum (a.u.)',xlabel='q (a.u.)',ylim=(-am,am),xlim=(q[0],q[-1]))
        x=ax[1,0]
        if isinstance(m,Pen):
            for key,lab,ls,col in [('photon_number',r'$\langle n\rangle$ Gaussian','-','#6b46c1'),('reference_photon_number',r'$\langle n\rangle$ reference','--','black'),('excited_population',r'$P_{S_1}$ Gaussian','-','#d95f02'),('reference_excited_population',r'$P_{S_1}$ reference','--','#777777')]:
                x.plot(t,[r[key] for r in rows],label=lab,ls=ls,color=col)
            x.set(ylabel='Number / probability')
        else:
            x.plot(t,[r['q_mean'] for r in rows],label=r'$\langle q\rangle$')
            x.set(ylabel='Mean proton position (a.u.)')
        x.set(title='(D) Physical observables',xlabel='t (fs)');x.legend()
        x=ax[1,1]
        for key,lab in [('state_l2','Complex-state L2'),('matched_initial_l2','Matched-initial L2'),('density_l1','Density L1')]:x.semilogy(t,np.maximum([r[key] for r in rows],1e-15),label=lab)
        x.set(title='(E) Errors against independent reference',ylabel='Error',xlabel='t (fs)');x.legend()
        x=ax[1,2]
        x.semilogy(t,np.maximum(abs(np.array([r['energy'] for r in rows])-rows[0]['energy']),1e-16),label='Energy drift (Ha)')
        x.semilogy(t,np.maximum(abs(np.array([r['norm'] for r in rows])-1),1e-16),label='Norm error')
        x.semilogy(t,np.maximum([r['schrodinger_residual'] for r in rows],1e-16),label='Schrodinger residual (a.u.)')
        x.set(title='(F) Conservation is not state accuracy',ylabel='Absolute diagnostic',xlabel='t (fs)');x.legend()
        for x in ax[1]:x.axvline(t[i],color='.4',ls=':');x.set_xlim(0,max(t[-1],.01))
        for x in ax[0]:x.xaxis.set_major_locator(MaxNLocator(6,integer=True))
    for i in sorted(set((0,len(t)//2,len(t)-1))):
        draw(i);fig.savefig(out/f'conditional_{i:05d}.png')
    draw(len(t)-1);fig.savefig(out/'conditional_summary.pdf')
    if movies:
        if writers.is_available('ffmpeg'):
            with FFMpegWriter(fps=8,bitrate=2500).saving(fig,str(out/'conditional_dynamics.mp4'),100) as writer:
                for i in range(len(t)):draw(i);writer.grab_frame()
        else:(out/'movie_unavailable.txt').write_text('ffmpeg unavailable; PNG/PDF retained.\n')
    plt.close(fig)


def render_comparison(root):
    root=Path(root);style();fig,ax=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    cfg=json.loads((root/'config.json').read_text());summaries={}
    series={}
    for name in COLORS:
        p=root/name
        if not (p/'metrics.json').exists():continue
        rows=json.loads((p/'metrics.json').read_text())
        if not rows:continue
        series[name]=rows;summaries[name]=json.loads((p/'summary.json').read_text())
        t=np.array([r['t_fs'] for r in rows]);kw=dict(color=COLORS[name],label=NAMES[name],
            linestyle='--' if name=='gaussian' else '-',linewidth=2.5 if name=='pol' else 1.9)
        ax[0,0].plot(t,[r['bo_population'][1] for r in rows],**kw)
        ax[0,1].plot(t,[r['photon_number'] for r in rows],**kw)
        ax[0,2].plot(t,[r['product_R_ge_4'] for r in rows],**kw)
        ax[1,0].plot(t,[r['energy']-rows[0]['energy'] for r in rows],**kw)
        if name!='qd' and 'qd' in series:
            qr=series['qd'];qt=[r['t_fs'] for r in qr]
            common=t<=qt[-1]+1e-9
            err=abs(np.array([r['photon_number'] for r in rows])-np.interp(t,qt,[r['photon_number'] for r in qr]))
            ax[1,1].plot(t[common],err[common],**kw)
    names=list(summaries)
    ax[1,2].bar(range(len(names)),[summaries[n]['wall_seconds'] for n in names],color=[COLORS[n] for n in names])
    ax[1,2].set_xticks(range(len(names)),names)
    ax[0,0].set_ylim(-.02,1.02)
    ax[0,2].set_ylim(-.02,1.02)  # Do not magnify tiny FFT indicator ringing as transfer.
    for a,(title,ylabel) in zip(ax.flat,[('(A) Bare excited population',r'$P_{S_1}$'),('(B) Photon observable',r'$\langle n\rangle$'),('(C) Product side R >= 4','Probability'),('(D) Total-energy drift','Energy (Ha)'),('(E) Photon error vs QD','Absolute error'),('(F) Measured run wall time','Time (s)')]):
        a.set_title(title);a.set_ylabel(ylabel)
        if a is not ax[1,2]:a.set_xlabel('t (fs)');a.axhline(0,color='.5',ls='--',lw=.8)
    handles,labels=ax[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.025),ncol=4)
    suffix=''
    if 'gaussian' in summaries and summaries['gaussian']['status']=='FAILED':
        end=summaries['gaussian']['last_saved_fs']
        suffix=f'; Gaussian stopped at {end:g} fs'
        for a in ax.flat:
            if a is not ax[1,2]:a.axvline(end,color=COLORS['gaussian'],ls=':',alpha=.6)
    fig.suptitle(f"PEN matched-preparation pilot | Ntraj={cfg['ntraj']} | K={cfg['K']} | not convergence certified\nEL: Wigner estimator; Gaussian: projected POL{suffix}",fontsize=16)
    out=root/'plots';out.mkdir(exist_ok=True)
    fig.savefig(out/'method_comparison.png',bbox_inches='tight');fig.savefig(out/'method_comparison.pdf',bbox_inches='tight');plt.close(fig)


def render_convergence(runs,out):
    """Compare K at identical physics AND numerical controls, not cherry-picked minima."""
    style();fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    anchor=None;counts=[];times=[]
    fields=('model','representation','physics','R','sigma','ne','rtol','atol','max_step_au',
            'energy_constraint','shape_regularization','nf_reference','end_fs')
    for run in sorted(map(Path,runs),key=lambda p:json.loads((p/'config.json').read_text())['K']):
        c=json.loads((run/'config.json').read_text());key={k:c[k] for k in fields}
        if anchor is None:anchor=key
        elif key!=anchor:raise ValueError('K comparison requires identical physics and numerical controls: '+str(run))
        rows=json.loads((run/'metrics.json').read_text());s=json.loads((run/'summary.json').read_text())
        if not rows:continue
        t=np.array([r['t_fs'] for r in rows]);label=f'K={c["K"]}'
        if s['status']=='FAILED':label+=' (partial)'
        ax[0,0].semilogy(t,np.maximum([r['state_l2'] for r in rows],1e-12),label=label)
        if 'photon_number' in rows[0]:
            ax[0,1].semilogy(t,np.maximum([abs(r['photon_number']-r['reference_photon_number']) for r in rows],1e-12),label=label)
        else:ax[0,1].semilogy(t,np.maximum([r['density_l1'] for r in rows],1e-12),label=label)
        ax[1,0].semilogy(t,np.maximum([abs(r['energy']-rows[0]['energy']) for r in rows],1e-16),label=label)
        counts.append(c['K']);times.append(s['propagation_seconds'])
    ax[1,1].plot(counts,times,'o-',color='#6b46c1')
    ax[1,1].set(xlabel='Gaussian count K',ylabel='Propagation time (s)',title='(D) Measured CPU-run wall time')
    for a,title,y in [(ax[0,0],'(A) Full complex-state error','L2 error'),(ax[0,1],'(B) Observable error','Absolute error'),(ax[1,0],'(C) Energy conservation','Energy drift (Ha)')]:
        a.set(xlabel='t (fs)',ylabel=y,title=title);a.legend()
    fig.suptitle('Fixed-R Gaussian count study | identical controls\n'+anchor['model']+' | no nuclear trajectory / grid convergence claim',fontsize=18)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix('.png'));fig.savefig(out.with_suffix('.pdf'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path);group.add_argument('--runs',type=Path,nargs='+')
    p.add_argument('--out',type=Path);p.add_argument('--movies',action='store_true')
    a=p.parse_args()
    if a.runs:
        if a.out is None:p.error('--runs requires --out')
        render_convergence(a.runs,a.out);return
    c=json.loads((a.run/'config.json').read_text())
    if 'K' in c and 'representation' in c:render_fixed(a.run,a.movies)
    else:render_comparison(a.run)


if __name__=='__main__':main()
