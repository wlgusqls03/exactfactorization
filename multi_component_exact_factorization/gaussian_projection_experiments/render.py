"""Analysis plots for real saved data, including failed/unfinished experiments."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import MaxNLocator

from ..gaussian_trajectory.run import AU_FS,atomic_json

STYLE={'font.size':12,'axes.titlesize':14,'axes.labelsize':13,'legend.fontsize':10,
       'figure.titlesize':16,'lines.linewidth':1.8,'savefig.dpi':180}


def decorate(axes,legends=True):
    for i,ax in enumerate(np.asarray(axes).ravel()):
        ax.grid(alpha=.18);ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.text(-.075,1.02,chr(65+i),transform=ax.transAxes,va='bottom',fontweight='bold')
        ax.tick_params(labelsize=11)
        if legends and ax.get_legend_handles_labels()[0]:ax.legend(loc='best',framealpha=.85)


def title(config,status):
    return '%s | K=%d, %s, g=%g, %s, dt=%g au\n%s (not convergence-certified)'%(
        config['model']['name'],config['components'],config['mode'],config['model'].get('g',0),
        config['model']['q_operator'],config['dt_au'],status['status'])


def summary_plot(a,config,status,out):
    t=a['t_au']*AU_FS
    fig,axs=plt.subplots(3,2,figsize=(15,12),layout='constrained')
    ax=axs[0,0];drift=a['energy']-a['energy'][0]
    ax.plot(t,drift,label='Energy drift');ax.plot(t,a['energy_integral'],'--',label=r'$\int \dot E_{\mathrm{projection}}dt$')
    if 'reference_energy' in a:ax.plot(t,a['reference_energy']-a['reference_energy'][0],':',label='Matched reference drift')
    ax.set(title='Energy balance',ylabel='Energy (Ha)')
    if np.max(abs(drift))<1e-10:ax.set_ylim(-1e-10,1e-10)
    ax=axs[0,1]
    if 'state_error' in a:
        ax.plot(t,a['state_error'],label=r'$\|G-G_{\mathrm{ref}}\|$')
        ax.plot(t,a['phase_aligned_state_error'],label='Global-phase aligned error')
    ax.plot(t,a['residual_integral'],'--',label='Integrated EOM residual')
    ax.set(title='State error and continuous-time bound',ylabel='State norm')
    ax=axs[1,0]
    if 'amplitude_error' in a:
        ax.plot(t,a['amplitude_error'],label='Propagated amplitude error')
        ax.plot(t,a['density_l1_error'],label='Density L1 error')
        ax.plot(t,a['initial_condition_reference_distance'],':',label='Initial-state mismatch (separate)')
    else:ax.text(.2,.5,'No reference: autonomous propagation',transform=ax.transAxes)
    ax.set(title='Amplitude/density: not offline fitting',ylabel='Error')
    ax=axs[1,1]
    for key,label in [('projection_residual','Full EOM residual'),('support_residual','Occupied-region residual'),('tail_residual','Tail residual')]:ax.plot(t,a[key],label=label)
    ax.set(title='Instantaneous residual',ylabel=r'State norm / $t_{\rm au}$',yscale='symlog');ax.set_yscale('symlog',linthresh=1e-9)
    ax=axs[2,0]
    ax.plot(t,a['high_k_mass'],label='High Fourier modes (>0.6 Nyquist)');ax.plot(t,a['edge_mass'],label='Edge probability')
    ax.set(title='Phase/spatial resolution warnings',ylabel='Probability',ylim=(-.03,1.03))
    ax=axs[2,1];ax.plot(t,a['condition_A'],label='Amplitude tangent condition number')
    ax.set(title='Gaussian parameter conditioning',ylabel='Condition number',yscale='log')
    ax.set_ylim(1,10**max(1,np.ceil(np.log10(max(a['condition_A'])))))
    for ax in axs.flat:ax.set_xlabel('t (fs)')
    decorate(axs);fig.suptitle(title(config,status))
    fig.savefig(out/'diagnostic_summary.png');fig.savefig(out/'diagnostic_summary.pdf');plt.close(fig)


def snapshot_figure(a,config,status,index,limits):
    fig,axs=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    q=a['q'];t=a['t_au'][index]*AU_FS
    ax=axs[0,0];ax.fill_between(q,a['rho'][index],color='.75',alpha=.6)
    ax.plot(q,a['rho'][index],color='#005F73',label=r'Propagated $|\Lambda_K|^2$')
    if 'reference_rho' in a:ax.plot(q,a['reference_rho'][index],'--',color='#BB3E03',label='Matched reference density')
    ax.set(title='Bridge density',xlabel='q (a.u.)',ylabel='Density',ylim=(0,limits['density']))
    ax=axs[0,1]
    ax.plot(q,a['requested_amplitude_rate'][index],label=r'Requested $r=\mathrm{Re}\,C^\dagger(-iHG)$')
    ax.plot(q,a['projected_amplitude_rate'][index],'--',label=r'Actual $\dot\Lambda_K$')
    ax.plot(q,a['amplitude_residual_map'][index],':',label=r'$\dot\Lambda_K-r$')
    ax.axhline(0,color='.35',ls='--',lw=.8)
    ax.set(title='Amplitude projection',xlabel='q (a.u.)',ylabel=r'Amplitude / $t_{\rm au}$',ylim=(-limits['rate'],limits['rate']))
    ax=axs[1,0]
    k=np.fft.fftshift(a['kq']);s=np.fft.fftshift(a['spectrum'][index])
    ax.plot(k,s,label='Propagated Fourier mass')
    if 'reference_G' in a:
        ref=np.sum(abs(np.fft.fft(a['reference_G'][index],axis=0))**2,axis=1);ref/=ref.sum()
        ax.plot(k,np.fft.fftshift(ref),'--',label='Matched reference')
    ax.set(title='Discrete spatial Fourier spectrum',xlabel='k (a.u.)',ylabel='Mode probability',ylim=(0,limits['spectrum']))
    ax=axs[1,1];times=a['t_au']*AU_FS
    ax.plot(times,a['energy']-a['energy'][0],label='Energy drift (full saved interval)')
    ax.plot(times,a['energy_integral'],'--',label='Integrated projection power')
    ax.axvline(t,color='k',ls='--',lw=1.2)
    ax.set(title='Energy balance',xlabel='t (fs)',ylabel='Energy (Ha)')
    if max(abs(a['energy']-a['energy'][0]))<1e-10:ax.set_ylim(-1e-10,1e-10)
    decorate(axs)
    fig.suptitle(title(config,status)+' | t=%.5f fs'%t)
    return fig


def render_case(case,*,movies=False,max_movie_frames=96):
    case=Path(case);out=case/'plots';out.mkdir(exist_ok=True)
    config=json.loads((case/'config.json').read_text());status=json.loads((case/'status.json').read_text())
    with np.load(case/'diagnostics.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with plt.rc_context(STYLE):
        summary_plot(a,config,status,out)
        limits=dict(density=1.1*max(a['rho'].max(),a.get('reference_rho',a['rho']).max()),
            rate=max(1e-12,1.08*max(abs(a[k]).max() for k in ['requested_amplitude_rate','projected_amplitude_rate','amplitude_residual_map'])),
            spectrum=1.05)
        for j in sorted(set([0,len(a['t_au'])//2,len(a['t_au'])-1])):
            fig=snapshot_figure(a,config,status,j,limits);fig.savefig(out/('state_%05d.png'%j));plt.close(fig)
        movie_status='not requested'
        if movies and shutil.which('ffmpeg') and len(a['t_au'])>1:
            writer=FFMpegWriter(fps=10,bitrate=2200)
            indices=np.unique(np.linspace(0,len(a['t_au'])-1,min(max_movie_frames,len(a['t_au']))).astype(int))
            fig=snapshot_figure(a,config,status,int(indices[0]),limits)
            # Fixed figure dimensions and limits for every movie frame.
            with writer.saving(fig,str(out/'state_diagnostics.mp4'),dpi=110):
                for j in indices:
                    if j!=indices[0]:
                        plt.close(fig);fig=snapshot_figure(a,config,status,int(j),limits);writer.fig=fig
                    writer.grab_frame()
            plt.close(fig);movie_status='generated'
        elif movies:movie_status='unavailable: ffmpeg missing or only one saved frame'
        atomic_json(out/'render_status.json',dict(movie=movie_status,scales=limits,data_status=status['status']))
    return out


def render_sweep(root):
    root=Path(root);manifest=json.loads((root/'sweep.json').read_text())
    with plt.rc_context(STYLE):
        for g in sorted(set(c['g'] for c in manifest['cases'])):
            fig,axs=plt.subplots(2,2,figsize=(15,10))
            fig.subplots_adjust(bottom=.14,top=.87,hspace=.35,wspace=.23)
            records=[c for c in manifest['cases'] if c['g']==g];available=[]
            for c in records:
                f=Path(c['path'])/'diagnostics.npz'
                if not f.exists():continue
                with np.load(f) as z:a={k:z[k] for k in z.files}
                t=a['t_au']*AU_FS;color=plt.get_cmap('tab10')((c['K']-1)%10)
                ls='-' if c['mode']=='energy' else '--';label='K%d %s%s'%(c['K'],c['mode'],' [stopped]' if c['status']!='COMPLETE_NOT_CERTIFIED' else '')
                values=[abs(a['energy']-a['energy'][0]),a.get('state_error'),a['projection_residual'],a['condition_A']]
                for ax,y in zip(axs.flat,values):
                    if y is not None:
                        ax.plot(t,y,color=color,ls=ls,label=label)
                        if c['status']!='COMPLETE_NOT_CERTIFIED':ax.plot(t[-1],y[-1],'x',color=color,ms=8)
                available.append(float(t[-1]))
            for ax,ttl,yl in zip(axs.flat,['Energy drift','Matched initial-state error','Actual EOM residual','Tangent conditioning'],
                                 ['|Energy drift| (Ha)','State norm',r'State norm / $t_{\rm au}$','Condition number']):
                ax.set(title=ttl,xlabel='t (fs)',ylabel=yl);ax.set_yscale('symlog',linthresh=1e-10)
                if available:ax.axvline(min(available),color='.5',ls=':',lw=1.)
            decorate(axs,legends=False)
            handles,labels=axs[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,fontsize=11)
            fig.suptitle('Fixed-R K study | g=%g | solid: energy constraint; dashed: original\nCrosses: last accepted state; dotted line: common available interval'%g)
            fig.savefig(root/('K_comparison_g_%s.png'%format(g,'.12g')));fig.savefig(root/('K_comparison_g_%s.pdf'%format(g,'.12g')));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path);p.add_argument('--sweep',type=Path)
    p.add_argument('--movies',action='store_true');p.add_argument('--max-movie-frames',type=int,default=96);a=p.parse_args()
    if a.max_movie_frames<2:p.error('--max-movie-frames must be >=2')
    if a.case:render_case(a.case,movies=a.movies,max_movie_frames=a.max_movie_frames)
    if a.sweep:render_sweep(a.sweep)
    if not a.case and not a.sweep:p.error('Specify --case or --sweep')


if __name__=='__main__':main()
