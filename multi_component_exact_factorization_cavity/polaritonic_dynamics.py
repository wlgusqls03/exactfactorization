"""Cached state-resolved nuclear motion and bare-product character plots.

These are projections, not classical trajectories on a single PES. Pol k is
the fixed-R energy index, NOT an assumed ground/LP/UP/double-excitation label.
All nuclear probabilities remain absolute (including omitted probability).
Composition weights describe the analysis eigenvectors; they do not determine
their complex phases, and degenerate eigenvectors can rotate within a subspace.
No propagation, eigensolve or changes to the analysis basis occur here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers

from .render import STYLE, REF_COLORS, _nice, _legend, _save, _bounds


PRODUCTS = ((0, 0), (0, 1), (1, 0), (1, 1))
PRODUCT_LABELS = [fr'$|S_{s},{n}\rangle$' for s, n in PRODUCTS] + ['Other products']
PRODUCT_COLORS = ['#4575B4', '#E69F00', '#009E73', '#CC79A7', '#C6C6C6']


def population_panel(ax, data, basis='pol', current=None):
    """One basis per panel; retain probability outside the four plotted states."""
    key=basis+'_population'; title='POL populations' if basis=='pol' else 'BO populations'
    ax.set_title(title)
    if key not in data or 't_fs' not in data:
        ax.text(.5,.5,title+' unavailable',ha='center',transform=ax.transAxes)
        _nice(ax,'t (fs)','Probability',(-.025,1.025))
        return
    t=np.asarray(data['t_fs']); values=np.asarray(data[key])
    if values.ndim!=2 or values.shape[0]!=len(t):
        ax.text(.5,.5,'Population cache shape mismatch',ha='center',transform=ax.transAxes)
        return
    shown=min(4,values.shape[1])
    for k in range(shown):
        label=fr'$|\mathrm{{Pol}}_{k}\rangle$' if basis=='pol' else fr'$S_{k}$'
        ax.plot(t,values[:,k],color=REF_COLORS[k],lw=2.2,label=label)
    # outside is defined relative to ALL stored columns; extra columns belong
    # in the plotted remainder, not silently discarded or renormalized away.
    if basis+'_outside' in data:
        outside=np.asarray(data[basis+'_outside'])+values[:,shown:].sum(axis=1)
    elif 'norm' in data:
        outside=np.asarray(data['norm'])-values[:,:shown].sum(axis=1)
    else:
        outside=None
    if outside is not None:
        ax.plot(t,outside,color='.4',ls='--',lw=1.7,label='Outside shown states')
    if current is not None:
        ax.axvline(current,color='#00A6D6',ls='--',lw=1.5,label='_current time')
    if len(t)>1: ax.set_xlim(float(t[0]),float(t[-1]))
    _nice(ax,'t (fs)','Probability',(-.025,1.025))
    ax.set_yticks(np.arange(0,1.01,.25)); _legend(ax,3)


def product_composition(analysis, nstates=4):
    """Eigenvector rows = s*n_fock+n; columns = energy-ordered Pol states.

    Return all four requested weights and their complement. Missing product
    states in a smaller basis have weight zero, with the basis size recorded.
    Reject nonunit eigenvectors rather than normalizing away a cache mismatch.
    """
    vectors=np.asarray(analysis['pol_vectors'])
    nf=np.asarray(analysis['fock']).shape[1]
    if vectors.ndim!=3 or vectors.shape[1]%nf or vectors.shape[2]!=vectors.shape[1]:
        raise ValueError('Invalid product/eigenvector dimensions')
    nb=vectors.shape[1]//nf; nk=min(nstates,vectors.shape[2])
    coeff=vectors[:,:,:nk]; norm=np.sum(abs(coeff)**2,axis=1)
    if not np.all(np.isfinite(coeff)) or not np.allclose(norm,1.,atol=1e-8,rtol=0):
        raise ValueError('Pol eigenvectors are not normalized in the analysis basis')
    selected=np.zeros((len(vectors),4,nk),dtype=coeff.dtype)
    for i,(s,n) in enumerate(PRODUCTS):
        if s<nb and n<nf: selected[:,i,:]=coeff[:,s*nf+n,:]
    weights=abs(selected)**2
    remaining=1.-weights.sum(axis=1)
    if remaining.min() < -1e-8:
        raise ValueError('Product weights exceed unity')
    weights=np.concatenate((weights,np.maximum(0,remaining)[:,None,:]),axis=1)
    return dict(weights=weights,coefficients=selected,n_bo=nb,n_fock=nf,n_states=nk)


def prepare_wavepacket(data, energies, R):
    """Validate same-grid cached projections and fix scales over ALL times.

    Rectangular quadrature matches projections.project (uniform FFT R grid).
    No interpolation/reprojection of local density onto a different basis.
    """
    R=np.asarray(R); local=np.asarray(data['pol_local_density'])
    rho=np.asarray(data['rho_R']); times=np.asarray(data['t_fs']); e=np.asarray(energies)
    if (R.ndim!=1 or len(R)<2 or local.ndim!=3 or rho.shape!=(len(times),len(R))
            or local.shape[:2]!=rho.shape or e.ndim!=2 or e.shape[0]!=len(R)):
        raise ValueError('State-resolved density/reference shape mismatch')
    if not np.allclose(np.diff(R),R[1]-R[0],rtol=1e-8,atol=1e-12) or R[1]<=R[0]:
        raise ValueError('Expected the native uniform increasing nuclear grid')
    if not all(np.all(np.isfinite(a)) for a in (local,rho,times,e)):
        raise ValueError('Nonfinite density/time/reference cache')
    if np.min(local)<-1e-10 or np.min(rho)<-1e-10 or np.any(np.diff(times)<=0):
        raise ValueError('Invalid density or nonincreasing saved times')
    nk=min(4,local.shape[2],e.shape[1]); local=local[:,:,:nk]; e=e[:,:nk]
    if nk<1: raise ValueError('No polaritonic state available')
    populations=local.sum(axis=1)*(R[1]-R[0])
    if 'pol_population' in data:
        cached=np.asarray(data['pol_population'])[:,:nk]
        if cached.shape!=populations.shape or not np.allclose(populations,cached,atol=1e-8,rtol=1e-6):
            raise ValueError('Integrated local densities disagree with cached POL populations')
    outside=rho-local.sum(axis=2)
    if outside.min() < -1e-8*max(1.,float(rho.max())):
        raise ValueError('Local projected probability exceeds total nuclear density')
    peak=rho.max(axis=1,keepdims=True)
    occupied=np.any(rho>1e-5*peak,axis=0)
    xlim=_bounds(R,occupied,margin=.13); visible=(R>=xlim[0])&(R<=xlim[1])
    lo=float(e[visible].min()); hi=float(e[visible].max()); span=max(hi-lo,1e-5)
    # One gain for all states and times: no per-state normalization hiding
    # a small population. Keep at most 22% of the common energy span as height.
    gain=.22*span/max(float(local.max()),1e-30)
    ylim=(lo-.06*span,hi+.30*span)
    return dict(R=R,t_fs=times,local=local,rho=rho,energy=e,populations=populations,
                outside=outside,n_states=nk,xlim=xlim,ylim=ylim,gain=gain,
                density_ylim=(-.025*max(float(rho.max()),1e-10),1.12*max(float(rho.max()),1e-10)))


def _context(config):
    g=config.get('g_cavity'); omega=config.get('omega')
    return ((' | '+str(config.get('case',''))) if config.get('case') else '')+(
        fr' | $g_{{\mathrm{{cavity}}}}={float(g):.6g}$ a.u.' if g is not None else '')+(
        fr' | $\omega_c={float(omega):.6g}$ Ha' if omega is not None else '')


def wavepacket_figure(prepared, data, index, config=None):
    """Four energy+ribbon panels and two quantitative probability panels."""
    p=prepared; R=p['R']; n=index
    fig,axes=plt.subplots(3,2,figsize=(23,18))
    fig.subplots_adjust(left=.07,right=.97,bottom=.12,top=.88,hspace=.65,wspace=.25)
    for k,ax in enumerate(axes[:2].flat):
        if k<p['n_states']:
            e=p['energy'][:,k]; d=p['local'][n,:,k]
            ax.plot(R,e,color=REF_COLORS[k],lw=2.1)
            ax.fill_between(R,e,e+p['gain']*d,color=REF_COLORS[k],alpha=.42)
            ax.plot(R,e+p['gain']*d,color=REF_COLORS[k],lw=1.2)
            ax.set_title(fr'{chr(65+k)}  $|\mathrm{{Pol}}_{k}\rangle$ | population = {p["populations"][n,k]:.4f}')
        else:
            ax.text(.5,.5,'State unavailable',ha='center',transform=ax.transAxes)
        ax.set_xlim(p['xlim']); _nice(ax,r'$R$ ($a_0$)','Energy (Ha)',p['ylim'])
    ax=axes[2,0]
    # This panel has a true density axis: unlike energy-background silhouettes,
    # its grey fill must use the actual density, not an arbitrary axis fraction.
    ax.fill_between(R,0,p['rho'][n],color='.65',alpha=.35,zorder=0)
    ax.plot(R,p['rho'][n],color='.2',lw=2.1,label=r'$|\chi|^2$ total')
    for k in range(p['n_states']):
        ax.plot(R,p['local'][n,:,k],color=REF_COLORS[k],lw=1.9,label=fr'$|\mathrm{{Pol}}_{k}\rangle$')
    ax.plot(R,p['outside'][n],color='.45',ls='--',lw=1.6,label='Outside shown states')
    ax.set_xlim(p['xlim']);ax.set_title('E  State-resolved nuclear density')
    _nice(ax,r'$R$ ($a_0$)',r'Density ($a_0^{-1}$)',p['density_ylim']);_legend(ax,3)
    panel_data=dict(data,pol_population=p['populations'])
    # The normal cache has four columns. Derive remainder from total density
    # even for smaller/larger input subspaces, with native-grid quadrature.
    panel_data['pol_outside']=p['outside'].sum(axis=1)*(R[1]-R[0])
    population_panel(axes[2,1],panel_data,'pol',p['t_fs'][n])
    axes[2,1].set_title('F  POL populations')
    fig.suptitle('Nuclear wavepacket on polaritonic surfaces'+_context(config or {})+
                 f'\nt = {p["t_fs"][n]:.3f} fs',fontsize=23)
    fig.text(.5,.035,'Colored ribbon height = common fixed gain × projected nuclear density; NOT additional energy or a classical trajectory.',ha='center',fontsize=12)
    fig.text(.5,.012,'Fixed-R electron–photon analysis states; energy indices alone do not identify LP / UP. No probability renormalization.',ha='center',fontsize=12)
    return fig


def composition_figure(composition, R, config=None):
    """Absolute product weights vs R, including the unshown product space."""
    c=composition; fig,axes=plt.subplots(2,2,figsize=(23,13))
    fig.subplots_adjust(left=.07,right=.97,bottom=.17,top=.80,hspace=.36,wspace=.20)
    R0=(config or {}).get('R0')
    for k,ax in enumerate(axes.flat):
        if k<c['n_states']:
            ax.stackplot(R,c['weights'][:,:,k].T,labels=PRODUCT_LABELS,
                         colors=PRODUCT_COLORS,alpha=.85,linewidth=.4,edgecolor='white')
        else: ax.text(.5,.5,'State unavailable',ha='center',transform=ax.transAxes)
        if R0 is not None: ax.axvline(R0,color='.15',ls=':',lw=1.8)
        ax.set_title(fr'{chr(65+k)}  $|\mathrm{{Pol}}_{k}\rangle$ product character')
        ax.set_xlim(R[0],R[-1]);_nice(ax,r'$R$ ($a_0$)','Product weight',(0,1))
        ax.set_yticks(np.arange(0,1.01,.25))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=5,frameon=False,fontsize=14)
    fig.suptitle('Polaritonic-state composition'+_context(config or {})+'\n'+
                 r'$|\mathrm{Pol}_k(R)\rangle=\sum_{j,n}C_{jn}^{(k)}(R)|S_j(R),n\rangle$',fontsize=23)
    fig.text(.5,.055,fr'Weights $|C_{{jn}}^{{(k)}}|^2$; analysis basis: {c["n_bo"]} BO × {c["n_fock"]} Fock. Dotted line: initial $R_0$ (when specified).',ha='center',fontsize=12)
    fig.text(.5,.022,'Other products are retained. Coefficient phases are not shown; degenerate eigenstate compositions depend on the chosen basis.',ha='center',fontsize=12)
    return fig


def render_polaritonic_dynamics(case_dir, output_dir=None, *, data=None, movies=True, fps=20, dpi=110):
    """Standalone or automatic cache-only driver, with explicit missing-data notes.

    Always use native analysis/reference caches, NOT an optional refined static
    reference from render_case: populations belong to the native eigenvectors.
    """
    from .workflow import load_tree
    case=Path(case_dir); out=Path(output_dir) if output_dir is not None else case/'report'
    out.mkdir(parents=True,exist_ok=True); outputs=[]; notes=[]
    config=json.loads((case/'config.json').read_text()) if (case/'config.json').exists() else {}
    def load_npz(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    if data is None:
        data=load_npz(case/'observables.npz') if (case/'observables.npz').exists() else {}
    try:
        native=load_npz(case/'references.npz') if (case/'references.npz').exists() else {}
    except (OSError,ValueError) as exc:
        native={};notes.append('Native references unavailable: '+str(exc))
    R=data.get('R',native.get('R')); analysis=None; composition=None; p=None
    if R is not None and 'R' in native:
        if np.shape(R)!=np.shape(native['R']) or not np.allclose(R,native['R'],atol=1e-12,rtol=0):
            notes.append('Native reference / observable nuclear grids mismatch; projections not relabelled.')
            R=None
    if (case/'analysis_cache.npz').exists():
        try:
            analysis=load_tree(case/'analysis_cache.npz')
            if R is None or analysis['pol_vectors'].shape[0]!=len(R):
                raise ValueError('Nuclear grid missing or eigenvector shape mismatch')
            composition=product_composition(analysis)
            config.setdefault('omega',float(analysis['omega']))
            config.setdefault('g_cavity',float(analysis['g_cavity']))
        except (OSError,ValueError,KeyError,IndexError,TypeError) as exc:
            notes.append('Pol composition unavailable: '+str(exc)); analysis=None;composition=None
    else: notes.append('Pol composition unavailable: analysis_cache.npz missing (no inferred coefficients).')
    energies=analysis.get('pol_energy') if analysis is not None else native.get('pol_energy')
    try:
        if energies is None or R is None: raise ValueError('Native grid/polaritonic energies unavailable')
        p=prepare_wavepacket(data,energies,R)
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        notes.append('State-resolved nuclear movie unavailable: '+str(exc))
    def save(fig,path):
        _save(fig,path,dpi); plt.close(fig)
        outputs.extend([str(path.with_suffix('.png')),str(path.with_suffix('.pdf'))])
    with plt.rc_context(STYLE):
        if composition is not None:
            folder=out/'polaritonic_composition';folder.mkdir(exist_ok=True)
            save(composition_figure(composition,R,config),folder/'product_weights')
            anchor=int(np.argmin(abs(R-config.get('R0',R[len(R)//2]))))
            coefficients=composition['coefficients'][anchor]
            record=dict(R=float(R[anchor]),requested_R=config.get('R0'),
                        n_bo=composition['n_bo'],n_fock=composition['n_fock'],
                        product_order=[list(pair) for pair in PRODUCTS],
                        weights_including_other=composition['weights'][anchor].tolist(),
                        coefficient_real=coefficients.real.tolist(),coefficient_imag=coefficients.imag.tolist(),
                        convention='Rows: four products, then other (weights only). Columns: energy-ordered Pol k. Eigenvector phase is arbitrary; degeneracies allow subspace rotations.')
            target=folder/'initial_composition.json';target.write_text(json.dumps(record,indent=2)+'\n');outputs.append(str(target))
        if p is not None:
            folder=out/'polaritonic_wavepacket';folder.mkdir(exist_ok=True)
            selected=sorted(set([0,len(p['t_fs'])//2,len(p['t_fs'])-1]))
            video=bool(movies and len(p['t_fs'])>1 and writers.is_available('ffmpeg'))
            if movies and not video: notes.append('State-resolved movie skipped: ffmpeg unavailable or fewer than two times.')
            writer=None
            try:
                for n in range(len(p['t_fs'])):
                    if not video and n not in selected:continue
                    fig=wavepacket_figure(p,data,n,config)
                    if n in selected:
                        target=folder/f'snapshot_{n:06d}';_save(fig,target,dpi)
                        outputs.extend([str(target.with_suffix('.png')),str(target.with_suffix('.pdf'))])
                    if video:
                        if writer is None:
                            writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-crf','23','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-threads','1'])
                            writer.setup(fig,str(folder/'polaritonic_wavepacket.mp4'),dpi=dpi)
                        writer.fig=fig;writer.grab_frame(facecolor='white')
                    plt.close(fig)
                if writer:
                    writer.finish();writer=None;outputs.append(str(folder/'polaritonic_wavepacket.mp4'))
            finally:
                if writer:writer.finish()
    manifest=dict(outputs=outputs,notes=notes,cache_policy='Native analysis eigenvectors/energies and absolute cached projections; no propagation or eigensolve.',
                  frame_interpolation=False,physical_frames=0 if p is None else len(p['t_fs']),
                  density_ribbon_gain=None if p is None else p['gain'],
                  fixed_R_limits=None if p is None else list(p['xlim']),
                  fixed_energy_limits=None if p is None else list(p['ylim']))
    target=out/'polaritonic_dynamics_manifest.json';target.write_text(json.dumps(manifest,indent=2)+'\n')
    outputs.append(str(target))
    for note in notes:print('[polaritonic render] '+note)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--no-movies',action='store_true')
    parser.add_argument('--fps',type=int,default=20);parser.add_argument('--dpi',type=int,default=110)
    args=parser.parse_args()
    render_polaritonic_dynamics(args.case,args.out,movies=not args.no_movies,fps=args.fps,dpi=args.dpi)


if __name__=='__main__':main()
