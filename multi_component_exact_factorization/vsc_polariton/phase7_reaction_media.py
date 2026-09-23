"""Render genuine saved-time reaction observables; never invent TDPES frames.

Read-only Phase2/6 inputs. New figures, movie and provenance under Phase7.
The threshold packet is not an LP/UP eigenstate. Spectral doorway strengths
are separately normalized transition strengths, not Hopfield populations.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
from .phase7_pilot_baseline import INPUT,digest
from .phase7_native import bare_action

AU_FS=0.024188843265857
HA_EV=27.211386245988
COLORS=['#2463A6','#D65B35','#6D529D']
LABELS=['Uncoupled', 'Well resonance · 170.6 meV', 'Barrier frequency · 161.77 meV']
DEFAULTS={
 'free':'results/vsc_polariton/phase6_results/phase6_free_recovery_results/free_recovery_campaign_v1/free_L36_dx0.3/full',
 'resonant':'results/vsc_polariton/phase6_gpu/phase6_orthogonal_results/orthogonal_campaign_v1/F120_dt0125/full',
 'barrier':'results/vsc_polariton/phase6_results/phase6_free_recovery_results/free_recovery_campaign_v1/barrier_F120/full'}


def common_times(series):
    """Exact saved-time intersection, no temporal interpolation."""
    keys=[set(np.round(s['time_au'],9)) for s in series]
    common=np.array(sorted(set.intersection(*keys)))
    if len(common)<3:raise ValueError('At least three common saved times required for a movie')
    return common,[np.array([np.flatnonzero(abs(s['time_au']-t)<1e-8)[0] for t in common]) for s in series]


def events(time,product,flux):
    """First resolved local signed-flux maxima plus global max population."""
    forward=find_peaks(flux)[0];back=find_peaks(-flux)[0]
    forward=forward[flux[forward]>0];back=back[flux[back]<0]
    # Exclude roundoff-only early maxima; relative threshold is documented.
    forward=forward[flux[forward]>.01*np.max(flux)] if len(forward) else forward
    back=back[-flux[back]>.01*np.max(-flux)] if len(back) else back
    return {'initial':0,'first_forward_peak':int(forward[0]) if len(forward) else None,
            'first_backward_peak':int(back[0]) if len(back) else None,
            'max_product':int(np.argmax(product)),'final':len(time)-1}


def load_case(case,folder,manifest):
    """Validate run-to-packet association and saved density/product consistency."""
    record=manifest['cases'][case];packet_path=INPUT/record['packet']
    status_path=folder/'status.json';status=json.loads(status_path.read_text())
    expected=manifest['files'][record['packet']]
    if digest(packet_path)!=expected or status['input_sha256']!=expected:
        raise ValueError('Wrong packet/run association: '+case)
    if status['status']!='COMPLETE_NOT_CERTIFIED' or status.get('failures'):
        raise ValueError('Unexpected Phase6 run status: '+case)
    with np.load(packet_path) as z:p={k:z[k] for k in ('R','dx','dR','phi','potential','tx')}
    paths=sorted(folder.glob('observable_*.npz'));rows=[];hashes={}
    if not paths:raise FileNotFoundError(folder)
    keys=('time_au','rho_R','current','product','flux','nph','P_exc','norm','energy')
    for path in paths:
        with np.load(path) as z:rows.append({k:z[k] for k in keys})
        hashes[str(path)]=digest(path)
    arrays={k:np.array([r[k] for r in rows]) for k in keys}
    if np.any(np.diff(arrays['time_au'])<=0):raise ValueError('Nonmonotone saved times')
    if any(not np.all(np.isfinite(v)) for v in arrays.values()):raise ValueError('Nonfinite observable')
    rho=arrays['rho_R'];R=p['R'];dr=float(p['dR'])
    np.testing.assert_allclose(rho.sum(axis=1)*dr,arrays['norm'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(rho[:,R>0].sum(axis=1)*dr,arrays['product'],atol=1e-10,rtol=0)
    phi=p['phi'][:,:,0,None]
    pes=(np.sum(phi.conj()*bare_action(phi,p),axis=(1,2))*float(p['dx'])).real
    return dict(arrays,R=R,pes=pes,hashes=hashes,status_hash=digest(status_path),packet_hash=expected)


def spectral_figure(out):
    """Cached Phase2 eigenproblem: which transitions have both characters?"""
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True,sharey=True,layout='constrained')
    info={}
    for ax,eta,title in zip(axes,('0','0.094'),('Uncoupled molecule + photon','Coupled: vibrational polaritons')):
        path=Path('results/vsc_polariton/phase2')/f'eigen_C_R400_F50_eta{eta}_w0.0062694344_L2.5.json'
        d=json.loads(path.read_text());gaps=np.array(d['gaps']);weights=np.array(d['doorway_weights'])
        eligible=(gaps>.25*.0062615847)&(gaps<2*.0062615847)
        weights=weights[:,eligible];weights/=weights.sum(axis=1)[:,None]
        energy=gaps[eligible]*HA_EV*1000
        for j,(label,color,offset) in enumerate(zip(('Photon transition','Vibrational transition'),('#2463A6','#D65B35'),(-.8,.8))):
            ax.vlines(energy+offset,0,weights[j],colors=color,lw=3,label=label)
            ax.scatter(energy+offset,weights[j],color=color,s=22)
        ax.set(title=title,xlabel='Excitation energy (meV)',xlim=(120,225),ylim=(0,1.08))
        pair=d['pair_indices'];centres=[gaps[i]*HA_EV*1000 for i in pair]
        if eta!='0':
            for x,label in zip(centres,('LP','UP')):ax.text(x,.94,label,ha='center',fontsize=13)
        info[eta]={'source':str(path),'sha256':digest(path),'pair_meV':centres,
            'splitting_meV':float(d['splitting']*HA_EV*1000),'interpretation':'Transition strengths, NOT populations'}
    axes[0].set_ylabel('Normalized transition strength')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=10)
    fig.suptitle('Static BO-reduced spectrum — photon / vibration character')
    for ext in ('png','pdf'):fig.savefig(out/f'polariton_transition_character.{ext}',dpi=180)
    plt.close(fig);return info


def main():
    parser=argparse.ArgumentParser()
    for case,path in DEFAULTS.items():parser.add_argument('--'+case+'-dir',type=Path,default=Path(path))
    parser.add_argument('--out',type=Path,default=Path('results/vsc_polariton/phase7/reaction_media_v1'))
    parser.add_argument('--fps',type=int,default=24);parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--no-movie',action='store_true');args=parser.parse_args()
    if args.fps<1 or args.stride<1:raise ValueError('fps and stride must be positive')
    if not args.no_movie and not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg required for MP4')
    manifest=json.loads((INPUT/'phase7_pilot_manifest.json').read_text())
    series=[load_case(c,getattr(args,c+'_dir'),manifest) for c in DEFAULTS]
    common,indices=common_times(series)
    if not np.allclose(np.diff(common),np.diff(common)[0],atol=1e-8,rtol=0):
        raise ValueError('Irregular cadence: do not imply constant physical-time playback')
    args.out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12,
                        'axes.spines.top':False,'axes.spines.right':False})
    t=common*AU_FS;maximum=max(s['rho_R'].max() for s in series)*1.1
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    title=fig.suptitle('Full 3DOF quantum transfer',fontsize=19)
    density_lines=[];density_fills=[];texts=[];cursor=[]
    for i,(s,ax,color,label) in enumerate(zip(series,axes[0],COLORS,LABELS)):
        ax.axvspan(0,3.5,color='#E4EFE7',zorder=0)
        ax.axvline(0,color='.35',ls='--',lw=1)
        line,=ax.plot(s['R'],s['rho_R'][0],color=color,lw=2.4);density_lines.append(line)
        density_fills.append(None)
        ax.set(xlim=(-3.5,3.5),ylim=(0,maximum),xlabel=r'$R$ ($a_0$)',ylabel=r'$\rho_R$ ($a_0^{-1}$)',title=label)
        texts.append(ax.text(.03,.94,'',transform=ax.transAxes,va='top',fontsize=12,
                             bbox=dict(facecolor='white',alpha=.85,edgecolor='none')))
        ax.text(.97,.94,'product\nR > 0',transform=ax.transAxes,ha='right',va='top',color='#38744D')
        right=ax.twinx();right.spines['right'].set_visible(True)
        # Explicitly referenced STATIC bare PES, not a purported TDPES movie.
        reference=np.min(s['pes'][s['R']<0]);energy=(s['pes']-reference)*HA_EV
        right.plot(s['R'],energy,color='.6',ls=':',lw=1.2,zorder=0)
        right.set(ylim=(-.05,2.0),ylabel='Bare PES − well minimum (eV)')
        right.tick_params(colors='.5');right.yaxis.label.set_color('.5')
    for col,(key,ylabel) in enumerate((('product',r'$P(R>0)$'),('flux',r'$J(R=0)$ (a.u.)'),('nph',r'$\langle n_{ph}\rangle$'))):
        ax=axes[1,col]
        for s,idx,c,label in zip(series,indices,COLORS,LABELS):ax.plot(t,s[key][idx],color=c,lw=2,label=label)
        ax.set(xlim=(t[0],t[-1]),xlabel='Time (fs)',ylabel=ylabel)
        if key=='flux':ax.axhline(0,color='.5',ls=':',lw=1)
        cursor.append(ax.axvline(0,color='.15',ls='--',lw=1))
    axes[1,0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=10)
    event_records={};snapshots={0,len(t)-1}
    for case,s,idx in zip(DEFAULTS,series,indices):
        ev=events(t,s['product'][idx],s['flux'][idx]);snapshots.update(v for v in ev.values() if v is not None)
        f=s['flux'][idx]
        event_records[case]={'events':{k:None if v is None else float(t[v]) for k,v in ev.items()},
            'max_product':float(s['product'][idx].max()),
            'integrated_forward':float(cumulative_trapezoid(np.maximum(f,0),common,initial=0)[-1]),
            'integrated_backward':float(cumulative_trapezoid(np.maximum(-f,0),common,initial=0)[-1])}
    def update(frame):
        title.set_text(f'Full 3DOF quantum transfer | t = {t[frame]:.2f} fs')
        for j,(s,idx) in enumerate(zip(series,indices)):
            k=idx[frame];rho=s['rho_R'][k];density_lines[j].set_ydata(rho)
            if density_fills[j] is not None:density_fills[j].remove()
            density_fills[j]=axes[0,j].fill_between(s['R'],0,rho,color=COLORS[j],alpha=.18)
            texts[j].set_text(f'P(product) = {s["product"][k]:.3f}\nJ(TS) = {s["flux"][k]:+.2e}')
        for line in cursor:line.set_xdata([t[frame],t[frame]])
    for frame in sorted(snapshots):
        update(frame)
        for ext in ('png','pdf'):fig.savefig(args.out/f'reaction_t{t[frame]:07.3f}fs.{ext}',dpi=140)
    frames=list(range(0,len(t),args.stride))
    if frames[-1]!=len(t)-1:frames.append(len(t)-1)
    if not args.no_movie:
        writer=FFMpegWriter(fps=args.fps,codec='libx264',bitrate=3200,
                           extra_args=['-pix_fmt','yuv420p','-threads','2'])
        with writer.saving(fig,str(args.out/'reaction_comparison.mp4'),dpi=100):
            for n,frame in enumerate(frames):
                update(frame);writer.grab_frame()
                if n%50==0:print(f'Movie {n+1}/{len(frames)}',flush=True)
    plt.close(fig)
    spectrum=spectral_figure(args.out)
    report={'phase7_pass':False,'scope':'Phase6 observables movie plus separate Phase2 static spectrum; NOT TDPES movie',
            'physical_interval_fs':[float(t[0]),float(t[-1])],'frames':len(frames),'fps':args.fps,
            'movie_seconds':None if args.no_movie else len(frames)/args.fps,'events':event_records,
            'spectra':spectrum,'inputs':{c:s['hashes'] for c,s in zip(DEFAULTS,series)},
            'event_rule':'First local extremum exceeding 1% of the same-signed global peak; not a rate',
            'note':'Static PES uses explicitly labelled bare-well reference. Photon number is not polariton population.'}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2));print(event_records,flush=True)


if __name__=='__main__':main()
