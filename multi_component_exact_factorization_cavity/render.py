"""Streaming, analysis-oriented reports for real-space PEN/VSC campaigns.

Only compact cached fields are read; no TDSE is run by this module.  All
colors/line scales are scanned once and held fixed across frames.  The 2D
camera follows occupied joint density with a small margin, without masking
the potential inside that rectangular window.  References are comparisons,
not claims that the exact time-dependent scalar equals a static surface.
"""
from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.interpolate import RegularGridInterpolator


COLORS = dict(total='#161616', GI='#0072B2', GD='#D55E00', b='#0072B2',
              alpha='#D55E00', delta='#009E73', S_q='#0072B2',
              S_adv='#D55E00', S_rel='#009E73')
REF_COLORS = ['#7B3294', '#E69F00', '#398439', '#CC79A7']
CONTOUR_LEVELS = np.r_[np.geomspace(1e-5,.075,10),np.arange(.10,1.,.05)]
CAMERA_RESET_GAP_FS = 1.0
STYLE = {'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
         'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 11,
         'figure.titlesize': 23, 'axes.spines.top': False,
         'axes.spines.right': False, 'savefig.facecolor': 'white'}


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _real(value):
    return np.asarray(value).real


def _get(f, key):
    if key=='delta_b' and key not in f:
        return _real(f['b'])-_real(f['alpha'])[:,None]
    aliases = {'epsilon1': 'epsilon1_A', 'epsilon2': 'epsilon2_A',
               'S_q': 'Sq', 'S_adv': 'Sadv', 'S_rel': 'Srel',
               'alpha_reconstructed': 'alpha_nested',
               'epsilon2_reconstructed': 'epsilon2_nested'}
    return _real(f[key if key in f else aliases.get(key, key)])


def _time(f):
    return float(f.get('t_fs', f.get('time_fs', 0.)))


def _is_mcef(f):
    return str(np.asarray(f.get('system','cavity')).item()).lower()=='mcef'


def _q_label(f):
    return r'$q$ ($a_0$)' if _is_mcef(f) else r'$q_c$ (a.u.)'


def _q_symbol(f):
    return 'q' if _is_mcef(f) else 'q_c'


def _reference_name(f):
    return 'BO' if _is_mcef(f) else 'CBO'


def _line_xlim(display,coordinate,f):
    return display.get('line_xlim_'+coordinate,(float(f[coordinate][0]),float(f[coordinate][-1])))


def _mcef_outer_bo_cut(f,references,q_index):
    """A fixed-proton BO comparison cut, NOT an outer adiabatic eigenproblem."""
    rr=np.asarray(references.get('R',f['R']));qq=np.asarray(references.get('q',f['q']))
    points=np.column_stack((rr,np.full(len(rr),f['q'][q_index])))
    energy=RegularGridInterpolator((rr,qq),_real(references['cbo'])[:,:,:4],
                                   bounds_error=False,fill_value=np.nan)(points)
    return rr,energy


def _nice(ax, xlabel=None, ylabel=None, limits=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if limits:
        ax.set_ylim(limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3,
                                         integer=bool(xlabel and xlabel.startswith('$'))))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(direction='out', length=5)
    ax.grid(alpha=.12, linewidth=.6)


def _legend(ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc='upper center',
                  bbox_to_anchor=(.5, -.20), ncol=ncol, frameon=False,
                  handlelength=2.8, columnspacing=1.1)


def _silhouette(ax, coordinate, density):
    density = np.maximum(_real(density), 0)
    peak = np.nanmax(density) if len(density) else 0
    if peak > 0:
        # Axis-fraction height: visibly present, never mistaken for energy.
        ax.fill_between(coordinate, 0, .24*density/peak,
                        transform=ax.get_xaxis_transform(), color='0.65',
                        alpha=.32, linewidth=0, zorder=0)


def _bounds(grid, occupied, margin=.10):
    idx = np.flatnonzero(occupied)
    if not len(idx):
        return float(grid[0]), float(grid[-1])
    width = max(grid[idx[-1]]-grid[idx[0]], 4*(grid[1]-grid[0]))
    return max(grid[0], grid[idx[0]]-margin*width), min(grid[-1], grid[idx[-1]]+margin*width)


def _window(f, previous=None):
    density = _real(f['rho_qR'])
    support = density > np.nanmax(density)*1e-5
    target = np.array(_bounds(f['R'], support.any(axis=1))+
                      _bounds(f['q'], support.any(axis=0)))
    if previous is None:
        return target
    # Smooth without ever cropping the current occupied support+margin.
    smoothed = .75*np.asarray(previous)+.25*target
    return np.array([min(target[0], smoothed[0]), max(target[1], smoothed[1]),
                     min(target[2], smoothed[2]), max(target[3], smoothed[3])])


def _map(ax, f, key, limits, window, r_slice, title):
    values = _get(f, key)
    R, q = f['R'], f['q']
    mcef=_is_mcef(f)
    horizontal,vertical=(q,R) if mcef else (R,q)
    shown=values if mcef else values.T
    artist = ax.pcolormesh(horizontal,vertical,shown, shading='auto', cmap='viridis' if key=='lambda_density' else 'RdBu_r',
                           vmin=limits[0], vmax=limits[1], rasterized=True)
    density = _real(f['rho_qR'])
    peak = np.nanmax(density)
    if peak > 0:
        levels = peak*CONTOUR_LEVELS
        if np.nanmin(density) < levels[-1]:
            ax.contour(horizontal,vertical,density if mcef else density.T,
                       levels=levels,colors='black',linewidths=.72,alpha=.88)
    if mcef:
        ax.axhline(r_slice,color='#373737',ls='--',lw=1.6)
        ax.set_xlim(window[2:]);ax.set_ylim(window[:2])
    else:
        ax.axvline(r_slice,color='#373737',ls='--',lw=1.6)
        ax.set_xlim(window[:2]);ax.set_ylim(window[2:])
    ax.set_title(title)
    ax.set_xlabel(_q_label(f) if mcef else r'$R$ ($a_0$)')
    ax.set_ylabel(r'$R$ ($a_0$)' if mcef else _q_label(f))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    bar = ax.figure.colorbar(artist, ax=ax, fraction=.047, pad=.02)
    bar.locator = MaxNLocator(nbins=5); bar.update_ticks()
    bar.ax.tick_params(labelsize=13)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_powerlimits((-3, 3)); bar.formatter = formatter
    bar.update_ticks()
    bar.set_label('Energy (Ha)' if key == 'epsilon1' else
                  'Momentum (a.u.)' if key in ('a', 'b','delta_b') else
                  r'$|\Lambda_R|^2$' if key=='lambda_density' else 'Density / a.u. time', fontsize=13)


def _range(values, symmetric=False):
    parts = [np.asarray(v).ravel() for v in values if np.size(v)]
    v = np.concatenate(parts) if parts else np.array([0.])
    v = v[np.isfinite(v)]
    if not len(v):
        return (-1., 1.)
    lo, hi = np.percentile(v, [.3, 99.7])
    if symmetric:
        hi = max(abs(lo), abs(hi), 1e-12); lo = -hi
    else:
        gap = max(hi-lo, abs(hi)*.04, 1e-9)
        lo -= .06*gap; hi += .06*gap
    ticks = MaxNLocator(nbins=5, symmetric=symmetric).tick_values(lo, hi)
    return float(ticks[0]), float(ticks[-1])


def prepare_display(files, references=None):
    """Pre-scan small fields once; never keep all 3D waves/frames in memory."""
    samples = {k: [] for k in ('epsilon1', 'a', 'b', 'energy1', 'energy2', 'momentum', 'source',
                               'alpha_direct','e2_direct','alpha_residual','e2_residual','bridge_terms',
                               'geometry1','interaction1','lambda_density','force')}
    def collect(key, values):
        flat=np.asarray(values).ravel()
        samples[key].append(flat[::max(1,int(len(flat)/2500))])
    first = _load(files[0]); R, q = first['R'], first['q']
    i = int(np.nanargmax(first['rho_R']))
    j = int(np.nanargmax(np.sum(first['rho_qR'], axis=0)))
    windows = []; previous = None; outer_windows = [];previous_time=None
    integrated = np.zeros_like(first['rho_qR'], dtype=float)
    for path in files:
        f = _load(path); joint = _real(f['rho_qR'])
        support = joint > np.nanmax(joint)*1e-5
        outer = _real(f['rho_R']) > np.nanmax(f['rho_R'])*1e-5
        integrated += joint
        for key in ('epsilon1', 'a', 'b'):
            collect(key,_get(f,key)[support])
        collect('lambda_density',_get(f,'lambda_density')[support])
        for values in (_get(f,'b')[support],(_get(f,'b')-_get(f,'alpha')[:,None])[support],_get(f,'alpha')[outer]):
            collect('momentum',values)
        for key,recon,name in (('alpha','alpha_reconstructed','alpha'),('epsilon2','epsilon2_reconstructed','e2')):
            collect(name+'_direct',_get(f,key)[outer])
            try: collect(name+'_residual',(_get(f,key)-_get(f,recon))[outer])
            except KeyError: pass
        for key in ('mean_e1','Kq','G_R_lambda','Ct'):
            if key in f: collect('bridge_terms',_real(f[key])[outer])
        for key in ('force','alpha_t'):
            if key in f: collect('force',_real(f[key])[outer])
        if 'force' in f and 'alpha_t' in f:
            collect('force',_real(f['force']-f['alpha_t'])[outer])
        for key in ('epsilon1_qgeo','epsilon1_Rgeo'):
            if key in f: collect('geometry1',_real(f[key])[support])
        for key in ('epsilon1_bare','epsilon1_LM','epsilon1_DSE'):
            if key in f: collect('interaction1',_real(f[key])[support])
        for key in ('density_dt','S_q','S_adv','S_rel'):
            if key in f or key in ('S_q','S_adv','S_rel'):
                try:
                    collect('source',_get(f,key)[support])
                except KeyError:
                    pass
        for level in (1,2):
            for suffix in ('','_GI','_GD'):
                key = f'epsilon{level}{suffix}'
                if key in f:
                    collect(f'energy{level}',_real(f[key])[support if level==1 else outer])
        now=_time(f)
        # A sparse 0/40/80 fs snapshot set must not retain the old packet's
        # camera bounds as if those frames were adjacent dense-movie frames.
        if previous_time is not None and now-previous_time>CAMERA_RESET_GAP_FS:
            previous=None
        previous = _window(f, previous); windows.append(previous);previous_time=now
        outer_windows.append(_bounds(f['R'],outer))
    # Draw reference curves on their full x grid, but select energy ranges
    # ONLY from the union of occupied windows + margin. Empty photon-box
    # wings and bare Coulomb poles far from the packet must not flatten all
    # interesting curves. No reference value in this selected region is
    # percentile-trimmed; out-of-window tails can lie outside the y viewport.
    reference_ranges = {}
    if references:
        rr=np.asarray(references.get('R',R))
        if 'cbo' in references:
            qq=np.asarray(references.get('q',q))
            selected=np.zeros((len(rr),len(qq)),dtype=bool)
            for rmin,rmax,qmin,qmax in windows:
                selected |= ((rr[:,None]>=rmin)&(rr[:,None]<=rmax)
                             &(qq[None,:]>=qmin)&(qq[None,:]<=qmax))
            values=_real(references['cbo'])[selected,:4]
            values=values[np.isfinite(values)]
            if len(values): reference_ranges['energy1']=(float(values.min()),float(values.max()))
        if 'pol_energy' in references:
            selected=np.zeros(len(rr),dtype=bool)
            for rmin,rmax in outer_windows:
                selected |= (rr>=rmin)&(rr<=rmax)
            values=_real(references['pol_energy'])[selected,:4]
            values=values[np.isfinite(values)]
            if len(values): reference_ranges['energy2']=(float(values.min()),float(values.max()))
        elif _is_mcef(first) and 'cbo' in references:
            rr,cut=_mcef_outer_bo_cut(first,references,j)
            selected=np.zeros(len(rr),dtype=bool)
            for rmin,rmax in outer_windows:selected|=(rr>=rmin)&(rr<=rmax)
            values=cut[selected];values=values[np.isfinite(values)]
            if len(values):reference_ranges['energy2']=(float(values.min()),float(values.max()))
    scales = {k: _range(v, symmetric=k in ('a','b','source','momentum','alpha_residual','e2_residual','force')) for k,v in samples.items()}
    for key,(reference_lo,reference_hi) in reference_ranges.items():
        lo=min(scales[key][0],reference_lo); hi=max(scales[key][1],reference_hi)
        ticks=MaxNLocator(nbins=5).tick_values(lo,hi)
        scales[key]=(float(ticks[0]),float(ticks[-1]))
    def quantiles(p):
        c = np.cumsum(p); c /= c[-1]
        return [int(np.searchsorted(c, v)) for v in (.25,.75)]
    slices_R = sorted(set([i]+quantiles(integrated.sum(axis=1))))[:2]
    slices_q = sorted(set([j]+quantiles(integrated.sum(axis=0))))[:2]
    line_xlim_R=(float(min(w[0] for w in outer_windows)),float(max(w[1] for w in outer_windows)))
    line_xlim_q=(float(min(w[2] for w in windows)),float(max(w[3] for w in windows)))
    return dict(scales=scales, windows=windows, i=i, j=j,
                slices_R=slices_R, slices_q=slices_q,
                line_xlim_R=line_xlim_R,line_xlim_q=line_xlim_q,
                reference_ranges=reference_ranges,
                reference_range_policy='Full curves drawn; y limits cover all shown reference states inside the union of occupied support+margin windows. Unoccupied reference tails may be outside the y viewport.')


def _references(ax, ref, f, *, level, direction='q', fixed_index=0):
    """Static references are optional; no substitute branch is fabricated."""
    if not ref:
        ax.text(.99,.98,'Static references unavailable', transform=ax.transAxes,
                ha='right',va='top',fontsize=10,color='0.4')
        return
    if level == 2:
        if _is_mcef(f) and 'cbo' in ref:
            rr,e=_mcef_outer_bo_cut(f,ref,fixed_index)
            for k in range(e.shape[1]):
                ax.plot(rr,e[:,k],ls='--',lw=1.5,color=REF_COLORS[k],label=fr'$BO_{k}(q_s,R)$')
            ax.text(.99,.98,fr'BO cut at $q_s={f["q"][fixed_index]:.2f}$ (reference only)',
                    transform=ax.transAxes,ha='right',va='top',fontsize=10,color='.4')
            return
        if 'pol_energy' not in ref:
            ax.text(.99,.98,'Outer references unavailable' if _is_mcef(f) else 'Fixed-R electron–photon references unavailable',
                    transform=ax.transAxes,ha='right',va='top',fontsize=10,color='0.4')
            return
        e = _real(ref['pol_energy']); rr = ref.get('R', f['R'])
        for k in range(min(4,e.shape[-1])):
            label=fr'$E^{{ref}}_{k}$' if _is_mcef(f) else fr'$E^{{x+q_c}}_{k}$'
            ax.plot(rr,e[:,k],ls='--',lw=1.5,color=REF_COLORS[k],label=label)
        return
    if 'cbo' not in ref:
        ax.text(.99,.98,_reference_name(f)+' unavailable',transform=ax.transAxes,ha='right',va='top',fontsize=10,color='0.4')
        return
    rr, qq = ref.get('R', f['R']), ref.get('q',f['q'])
    energy = _real(ref['cbo'])
    coordinate = f['q'] if direction=='q' else f['R']
    points = (np.column_stack((np.full(len(coordinate),f['R'][fixed_index]), coordinate))
              if direction=='q' else
              np.column_stack((coordinate,np.full(len(coordinate),f['q'][fixed_index]))))
    for k in range(min(4,energy.shape[-1])):
        # Interpolation is solely a static-reference display operation.
        values = RegularGridInterpolator((rr,qq),energy[:,:,k],bounds_error=False,fill_value=np.nan)(points)
        ax.plot(coordinate,values,ls='--',lw=1.5,color=REF_COLORS[k],label=f'{_reference_name(f)} {k}')


def _line_support(f,level,selector=None):
    """Display-only masks for dynamic lines, never static references or maps.

    Outer lines use relative marginal density; first-level cuts use their
    own slice peak. A fixed-R cut outside outer support is wholly suppressed
    instead of displaying numerically unstable conditional tail fields.
    """
    density=_real(f['rho_R'] if level==2 else f['rho_qR'])
    if selector is not None: density=density[selector]
    finite=np.isfinite(density)
    peak=float(np.max(density[finite])) if np.any(finite) else 0.
    support=finite&(density>max(0.,peak*1e-5))
    if level==1 and isinstance(selector,(int,np.integer)):
        outer=_real(f['rho_R']);outer_peak=np.nanmax(outer)
        if not np.isfinite(outer[selector]) or outer[selector]<=outer_peak*1e-5:
            support[:]=False
    return support


def _unsupported_note(ax,support):
    if not np.any(support):
        ax.text(.02,.96,'Slice outside occupied support',transform=ax.transAxes,
                ha='left',va='top',fontsize=11,color='.4')


def _scalar_lines(ax, coordinate, f, level, selector=None):
    support=_line_support(f,level,selector)
    for suffix,label,color in (('', 'total', COLORS['total']),('_GI','GI',COLORS['GI']),('_GD','GD',COLORS['GD'])):
        key=f'epsilon{level}{suffix}'
        if key in f:
            values=_real(f[key]); values=values if selector is None else values[selector]
            ax.plot(coordinate,np.where(support,values,np.nan),lw=2.4 if not suffix else 1.8,color=color,label=label)
    _unsupported_note(ax,support)


def _momentum(ax, f, i, limit, xlim=None):
    q=f['q']; b=_get(f,'b')[i]; alpha=float(_get(f,'alpha')[i])
    support=_line_support(f,1,i)
    _silhouette(ax,q,f['lambda_density'][i])
    ax.plot(q,np.where(support,b,np.nan),color=COLORS['b'],lw=2.1,label=r'$b$')
    ax.axhline(alpha,color=COLORS['alpha'],ls='--',lw=2,label=r'$\alpha$')
    ax.plot(q,np.where(support,b-alpha,np.nan),color=COLORS['delta'],lw=2,label=r'$b-\alpha$')
    ax.axhline(0,color='0.45',ls=':',lw=1)
    ax.set_xlim(xlim if xlim is not None else (q[0],q[-1])); _nice(ax,_q_label(f),'Momentum (a.u.)',limit)
    ax.set_title(fr'Momentum-field slice: $R={f["R"][i]:.2f}\,a_0$'); _legend(ax)
    _unsupported_note(ax,support)


def bridge_figure(f, display, index, references=None):
    """A: two EF perspectives, with no bridge reconstruction clutter."""
    fig, axes=plt.subplots(3,2,figsize=(23,18))
    fig.subplots_adjust(left=.065,right=.96,bottom=.075,top=.92,hspace=.70,wspace=.28)
    i=display['i']; scales=display['scales']; R=f['R']; q=f['q']
    effective=_is_mcef(f) and 'effective' in str(f.get('scalar_kind','')).lower()
    scalar_label=r'$\epsilon^{(1)}_{\rm eff}$' if effective else r'$\epsilon^{(1)}$'
    for row,(key,label) in enumerate((('epsilon1',scalar_label),('a',r'$a$'),('b',r'$b$'))):
        _map(axes[row,0],f,key,scales[key],display['windows'][index],R[i],label)
    ax=axes[0,1]; _silhouette(ax,R,f['rho_R']); _scalar_lines(ax,R,f,2)
    _references(ax,references,f,level=2,fixed_index=display['j']); ax.axvline(R[i],ls='--',lw=1,color='0.4')
    ax.set_xlim(_line_xlim(display,'R',f)); _nice(ax,r'$R$ ($a_0$)','Energy (Ha)',scales['energy2'])
    ax.set_title(r'$\epsilon^{(2)}$ analysis'+(' (effective)' if effective else '')); _legend(ax,4)
    ax=axes[1,1]; _silhouette(ax,q,f['lambda_density'][i]); _scalar_lines(ax,q,f,1,i)
    _references(ax,references,f,level=1,fixed_index=i)
    ax.axhline(_get(f,'epsilon2')[i],ls=':',color='#555555',lw=1.7,label=r'$\epsilon^{(2)}(R_s)$')
    ax.set_xlim(_line_xlim(display,'q',f)); _nice(ax,_q_label(f),'Energy (Ha)',scales['energy1'])
    ax.set_title(scalar_label+fr' slice analysis: $R={R[i]:.2f}\,a_0$'); _legend(ax,4)
    _momentum(axes[2,1],f,i,scales['momentum'],_line_xlim(display,'q',f))
    _title(fig,f,'A | Exact-factorization perspectives')
    return fig


def source_figure(f, display, index, references=None):
    """B: conditional-density continuity; RMS belongs in consistency only."""
    fig,axes=plt.subplots(3,2,figsize=(23,18))
    fig.subplots_adjust(left=.065,right=.96,bottom=.075,top=.90,hspace=.68,wspace=.28)
    i=display['i']; R=f['R']; q=f['q']; lim=display['scales']['source']
    sq_title=(r'$S_q=-\partial_q[(a/m_q)|\Lambda_R|^2]$' if _is_mcef(f)
              else r'$S_q=-\partial_{q_c}(a|\Lambda_R|^2)$')
    for ax,key,title in zip(axes[:2].flat,('density_dt','S_q','S_adv','S_rel'),
                            (r'$\partial_t|\Lambda_R|^2$',sq_title,
                             r'$S_{\rm adv}=-(\alpha/M)\partial_R|\Lambda_R|^2$',r'$S_{\rm rel}$: relative nuclear transport')):
        try:
            _map(ax,f,key,lim,display['windows'][index],R[i],title)
        except KeyError:
            ax.text(.5,.5,key+' unavailable',ha='center',transform=ax.transAxes)
    ax=axes[2,0]; _silhouette(ax,q,f['lambda_density'][i]); line_support=_line_support(f,1,i)
    for key,label,color in (('density_dt',r'$\partial_t|\Lambda_R|^2$',COLORS['total']),
                            ('S_q',r'$S_q$',COLORS['S_q']),('S_adv',r'$S_{\rm adv}$',COLORS['S_adv']),
                            ('S_rel',r'$S_{\rm rel}$',COLORS['S_rel'])):
        try:
            ax.plot(q,np.where(line_support,_get(f,key)[i],np.nan),lw=2,color=color,label=label)
        except KeyError:
            pass
    ax.axhline(0,color='0.45',ls='--',lw=1); ax.set_xlim(_line_xlim(display,'q',f))
    _nice(ax,_q_label(f),'Density / a.u. time',lim)
    ax.set_title(fr'Source terms at fixed $R={R[i]:.2f}\,a_0$'); _legend(ax)
    _unsupported_note(ax,line_support)
    _momentum(axes[2,1],f,i,display['scales']['momentum'],_line_xlim(display,'q',f))
    _title(fig,f,r'B | $\partial_t|\Lambda_R|^2=S_q+S_{\rm adv}+S_{\rm rel}$')
    return fig


def consistency_figure(f, display, index, references=None):
    fig,axes=plt.subplots(3,2,figsize=(23,18))
    fig.subplots_adjust(left=.07,right=.97,bottom=.10,top=.91,hspace=.65,wspace=.23)
    R=f['R']; support=f['rho_R']>np.max(f['rho_R'])*1e-5
    for col,key,recon,label in ((0,'alpha','alpha_reconstructed',r'$\alpha$'),
                                 (1,'epsilon2','epsilon2_reconstructed',r'$\epsilon^{(2)}$')):
        ax=axes[0,col]; _silhouette(ax,R,f['rho_R'])
        direct=_get(f,key)
        available=True
        try: rec=_get(f,recon)
        except KeyError: rec=np.full_like(direct,np.nan);available=False
        ax.plot(R,np.where(support,direct,np.nan),lw=2.4,color=COLORS['total'],label='direct')
        if available:
            ax.plot(R,np.where(support,rec,np.nan),lw=1.9,ls='--',color=COLORS['GI'],label='bridge reconstruction')
        else:
            ax.text(.02,.95,'Independent reconstruction unavailable',transform=ax.transAxes,va='top',fontsize=11,color='.4')
        ax.set_title(label+' direct / reconstructed'); _nice(ax,r'$R$ ($a_0$)','Momentum (a.u.)' if col==0 else 'Energy (Ha)',display['scales']['alpha_direct' if col==0 else 'e2_direct'])
        _legend(ax,2)
        if available:
            axes[2,col].plot(R,np.where(support,direct-rec,np.nan),color=COLORS['GD'],lw=1.9,label='direct − reconstructed')
        else:
            axes[2,col].text(.5,.5,'Residual unavailable (not set to zero)',transform=axes[2,col].transAxes,ha='center',fontsize=12,color='.4')
        axes[2,col].axhline(0,color='0.4',ls='--',lw=1)
        axes[2,col].set_title(label+' residual (occupied support)')
        _nice(axes[2,col],r'$R$ ($a_0$)','Momentum (a.u.)' if col==0 else 'Energy (Ha)',display['scales']['alpha_residual' if col==0 else 'e2_residual'])
        _legend(axes[2,col],1)
    ax=axes[1,0]; _silhouette(ax,R,f['rho_R'])
    if not any(key in f for key in ('mean_e1','Kq','G_R_lambda','Ct')):
        ax.text(.5,.5,'Bridge components unavailable',transform=ax.transAxes,ha='center',color='.4')
    for key,label,color in (('mean_e1',r'$\langle\epsilon^{(1)}\rangle_\Lambda$',COLORS['total']),
                            ('Kq',r'$K_q$',COLORS['GI']),('G_R_lambda',r'$G_R^\Lambda$',COLORS['GD']),
                            ('Ct',r'$\langle-i\partial_t\rangle_\Lambda$',COLORS['delta'])):
        if key in f: ax.plot(R,np.where(support,_real(f[key]),np.nan),lw=2,color=color,label=label)
    ax.set_title(r'$\epsilon^{(2)}$ bridge terms'); _nice(ax,r'$R$ ($a_0$)','Energy (Ha)',display['scales']['bridge_terms']); _legend(ax,2)
    ax=axes[1,1]; i=display['i']; q=f['q']; _silhouette(ax,q,f['lambda_density'][i]); line_support=_line_support(f,1,i)
    if 'density_dt' in f:
        ax.plot(q,np.where(line_support,_real(f['density_dt'][i]),np.nan),lw=2.2,color=COLORS['total'],label=r'$\partial_t|\Lambda_R|^2$')
        try:
            total=sum(_get(f,key)[i] for key in ('S_q','S_adv','S_rel'))
            ax.plot(q,np.where(line_support,total,np.nan),lw=1.8,ls='--',color=COLORS['GI'],label='sum of sources')
        except KeyError: pass
    if 'continuity_residual' in f:
        ax.plot(q,np.where(line_support,_real(f['continuity_residual'][i]),np.nan),lw=1.6,color=COLORS['GD'],label='continuity residual')
    ax.set_title(fr'Continuity reconstruction: $R={R[i]:.2f}\,a_0$')
    _nice(ax,_q_label(f),'Density / a.u. time',display['scales']['source']); _legend(ax,2)
    for row in (0,2):
        for column in range(2): axes[row,column].set_xlim(_line_xlim(display,'R',f))
    axes[1,0].set_xlim(_line_xlim(display,'R',f));axes[1,1].set_xlim(_line_xlim(display,'q',f))
    _title(fig,f,'C | Bridge consistency diagnostics')
    return fig


def slice_figure(f, display, index, references=None):
    """Two R cuts and two q cuts; no electronic/vibrational branch conflation."""
    fig,axes=plt.subplots(2,2,figsize=(23,13))
    fig.subplots_adjust(left=.07,right=.97,bottom=.12,top=.89,hspace=.65,wspace=.24)
    for row in range(2):
        i=display['slices_R'][min(row,len(display['slices_R'])-1)]
        j=display['slices_q'][min(row,len(display['slices_q'])-1)]
        ax=axes[row,0]; _silhouette(ax,f['R'],f['rho_qR'][:,j]); _scalar_lines(ax,f['R'],f,1,(slice(None),j))
        _references(ax,references,f,level=1,direction='R',fixed_index=j)
        ax.set_xlim(_line_xlim(display,'R',f)); _nice(ax,r'$R$ ($a_0$)','Energy (Ha)',display['scales']['energy1'])
        ax.set_title(fr'$R$-slice at ${_q_symbol(f)}={f["q"][j]:.2f}$'); _legend(ax,4)
        ax=axes[row,1]; _silhouette(ax,f['q'],f['lambda_density'][i]); _scalar_lines(ax,f['q'],f,1,i)
        _references(ax,references,f,level=1,fixed_index=i)
        ax.set_xlim(_line_xlim(display,'q',f)); _nice(ax,_q_label(f),'Energy (Ha)',display['scales']['energy1'])
        ax.set_title(fr'${_q_symbol(f)}$-slice at $R={f["R"][i]:.2f}\,a_0$'); _legend(ax,4)
    _title(fig,f,r'D | $\epsilon^{(1)}$ slice guidance')
    return fig


def component_figure(f,display):
    """Separate component budget; static snapshots avoid another large movie.

    GI/GD and geometric terms are gauge-labelled PG quantities, not mutually
    independent causal forces. The h expectation includes photon harmonic,
    LM and DSE exactly once; this plot does not add it to epsilon1 again.
    """
    fig,axes=plt.subplots(3,2,figsize=(23,18))
    fig.subplots_adjust(left=.07,right=.97,bottom=.10,top=.91,hspace=.65,wspace=.24)
    i,j=display['i'],display['j']
    groups=((('epsilon1',r'$\epsilon^{(1)}$'),('epsilon1_cond',r'$\langle h\rangle_x$'),
             ('epsilon1_GD','GD')), (('epsilon1_qgeo',fr'$G_{{{_q_symbol(f)}}}$'),('epsilon1_Rgeo',r'$G_R$')),
            (('epsilon1_bare','bare molecular'),('epsilon1_LM','light–matter'),('epsilon1_DSE','DSE')))
    for col in range(2):
        coordinate=f['q'] if col==0 else f['R']
        selector=i if col==0 else (slice(None),j)
        density=f['lambda_density'][i] if col==0 else f['rho_qR'][:,j]
        line_support=_line_support(f,1,selector)
        for row,group in enumerate(groups):
            ax=axes[row,col]; _silhouette(ax,coordinate,density)
            if not any(key in f for key,_ in group):
                ax.text(.5,.7,'Components unavailable in this cache',transform=ax.transAxes,ha='center',color='.4',fontsize=11)
            for k,(key,label) in enumerate(group):
                if key in f:
                    ax.plot(coordinate,np.where(line_support,_real(f[key])[selector],np.nan),lw=2,color=([COLORS['total']]+REF_COLORS)[k],label=label)
            if row==2 and 'omega' in f:
                value=.5*float(f['omega'])**2*np.asarray(f['q'])**2
                ax.plot(coordinate,value if col==0 else np.full(len(coordinate),value[j]),lw=1.6,ls='--',color=REF_COLORS[3],label='photon harmonic')
            ax.axhline(0,ls=':',color='.5',lw=.8)
            ax.set_xlim(_line_xlim(display,'q' if col==0 else 'R',f))
            _nice(ax,_q_label(f) if col==0 else r'$R$ ($a_0$)','Energy (Ha)',
                  display['scales'][('energy1','geometry1','interaction1')[row]])
            title=('Scalar expectation / time term','Geometric contributions','Hamiltonian expectation budget')[row]
            ax.set_title(title+(' | fixed R' if col==0 else ' | fixed q'))
            _legend(ax,3)
            _unsupported_note(ax,line_support)
    _title(fig,f,fr'$\epsilon^{{(1)}}$ component slices | $R_s={f["R"][i]:.2f}$, $q_s={f["q"][j]:.2f}$')
    return fig


def bridge_wavefunction_figure(f,display,index,references=None):
    """Show the intermediate conditional probability and relative R flow."""
    fig,axes=plt.subplots(2,2,figsize=(23,13))
    fig.subplots_adjust(left=.07,right=.97,bottom=.13,top=.89,hspace=.62,wspace=.26)
    i=display['i']; R=f['R']; q=f['q']; scales=display['scales']
    density_limit=(0,scales['lambda_density'][1])
    _map(axes[0,0],f,'lambda_density',density_limit,display['windows'][index],R[i],fr'$|\Lambda_R({_q_symbol(f)})|^2$')
    _map(axes[0,1],f,'delta_b',scales['momentum'],display['windows'][index],R[i],r'$b-\alpha$: relative nuclear momentum')
    ax=axes[1,0]
    for k,ir in enumerate(display['slices_R']):
        ax.plot(q,f['lambda_density'][ir],lw=2.1,color=REF_COLORS[k],label=fr'$R={R[ir]:.2f}\,a_0$')
    ax.set_xlim(_line_xlim(display,'q',f)); ax.set_title('Conditional proton probability' if _is_mcef(f) else 'Conditional photon-coordinate probability')
    _nice(ax,_q_label(f),r'$|\Lambda_R|^2$',density_limit); _legend(ax,2)
    ax=axes[1,1]; dq=q[1]-q[0]; density=f['lambda_density']
    mean=np.sum(density*q[None,:],axis=1)*dq
    variance=np.maximum(0,np.sum(density*(q[None,:]-mean[:,None])**2,axis=1)*dq)
    support=f['rho_R']>np.max(f['rho_R'])*1e-5
    _silhouette(ax,R,f['rho_R'])
    ax.plot(R,np.where(support,mean,np.nan),lw=2.1,color=COLORS['GI'],label=fr'$\langle {_q_symbol(f)}\rangle_{{\Lambda_R}}$')
    ax.fill_between(R,np.where(support,mean-np.sqrt(variance),np.nan),
                    np.where(support,mean+np.sqrt(variance),np.nan),color=COLORS['GI'],alpha=.16,label=r'$\pm$ conditional std')
    ax.axhline(0,ls='--',color='.45',lw=1); ax.set_xlim(_line_xlim(display,'R',f))
    ax.set_title('Conditional proton center / width' if _is_mcef(f) else 'Conditional quadrature center / width')
    _nice(ax,r'$R$ ($a_0$)',_q_label(f),_line_xlim(display,'q',f)); _legend(ax,2)
    _title(fig,f,'Bridge wavefunction and relative flow')
    return fig


def outer_force_figure(f,display):
    """EF electric-like field, not the full Madelung trajectory acceleration."""
    fig,axes=plt.subplots(1,2,figsize=(22,7))
    fig.subplots_adjust(left=.07,right=.97,bottom=.27,top=.80,wspace=.25)
    R=f['R']; support=f['rho_R']>np.max(f['rho_R'])*1e-5
    for ax in axes: _silhouette(ax,R,f['rho_R'])
    axes[0].plot(R,np.where(support,_get(f,'epsilon2'),np.nan),color=COLORS['total'],lw=2.2,label=r'$\epsilon^{(2)}$ (PG)')
    axes[0].set_title('Outer scalar in positive marginal gauge')
    _nice(axes[0],r'$R$ ($a_0$)','Energy (Ha)',display['scales']['e2_direct']); _legend(axes[0],1)
    if 'force' in f:
        axes[1].plot(R,np.where(support,_real(f['force']),np.nan),lw=2.2,color=COLORS['total'],label='total EF field')
        if 'alpha_t' in f:
            axes[1].plot(R,np.where(support,_real(f['alpha_t']),np.nan),lw=1.8,color=COLORS['GD'],label=r'$\partial_t\alpha$')
            # Both fields were obtained from the same instantaneous wave jet.
            axes[1].plot(R,np.where(support,_real(f['force']-f['alpha_t']),np.nan),lw=1.8,color=COLORS['GI'],label=r'$-\partial_R\epsilon^{(2)}$')
    else:
        axes[1].text(.5,.5,'Instantaneous force unavailable',ha='center',transform=axes[1].transAxes)
    axes[1].axhline(0,ls='--',lw=1.1,color='.4')
    axes[1].set_title(r'$F_R=-\partial_R\epsilon^{(2)}+\partial_t\alpha$')
    _nice(axes[1],r'$R$ ($a_0$)','EF field (Ha / $a_0$)',display['scales']['force']); _legend(axes[1],3)
    for ax in axes: ax.set_xlim(_line_xlim(display,'R',f))
    _title(fig,f,'Outer nuclear effective field')
    fig.text(.5,.045,'This field excludes the marginal quantum-potential term; the PG scalar slope alone is not the nuclear EF field.',ha='center',fontsize=12)
    return fig


def _cbo_on_field_grid(f,references):
    R,q=f['R'],f['q']
    if not references or 'cbo' not in references:
        return np.full((len(R),len(q)),np.nan)
    rr=references.get('R',R);qq=references.get('q',q)
    rg,qg=np.meshgrid(R,q,indexing='ij')
    return RegularGridInterpolator((rr,qq),_real(references['cbo'])[:,:,0],
                                    bounds_error=False,fill_value=np.nan)(np.stack((rg,qg),axis=-1))


def cbo_comparison_figure(f,display,index,references=None):
    """Same-geometry CBO reference/character, not a CBO propagation model.

    Static reference interpolation here is display-only. A cached CBO
    character may itself use the documented BO-projected electronic analysis
    basis; it must not be mistaken for a converged complete electronic basis.
    """
    fig,axes=plt.subplots(2,2,figsize=(23,14))
    fig.subplots_adjust(left=.07,right=.95,bottom=.15,top=.88,hspace=.40,wspace=.25)
    R,q=f['R'],f['q']; shape=(len(R),len(q)); window=display['windows'][index]
    has_gi='epsilon1_GI' in f
    gi=_real(f['epsilon1_GI']) if has_gi else np.full(shape,np.nan)
    cbo=_cbo_on_field_grid(f,references); has_cbo=bool(references and 'cbo' in references)
    difference=gi-cbo
    support=f['rho_qR']>np.nanmax(f['rho_qR'])*1e-5
    difference_limits=display['scales'].get('cbo_difference',_range([difference[support]],symmetric=True))
    ref_name=_reference_name(f)
    character=np.full(shape,np.nan); character_title=ref_name+' character unavailable'
    character_note='No electronic character is inferred from energy proximity.'
    if 'cbo_character' in f:
        character=_real(f['cbo_character'])[:,:,0]
        character_title=ref_name+' 0 character (BO-projected analysis)'
        character_note=ref_name+' character: BO-projected electronic analysis basis; omitted population retained in the cache.'
        if 'cbo_outside' in f:
            outside=np.asarray(f['cbo_outside']); weights=np.where(support,f['rho_qR'],0)
            if np.all(np.isfinite(outside[support])) and np.sum(weights)>0:
                omitted=float(np.sum(weights[support]*outside[support])/np.sum(weights))
                character_note+=f' Support-weighted outside fraction: {omitted:.3g}.'
    elif 'BO_character' in f and np.asarray(f['BO_character']).ndim==3:
        character=_real(f['BO_character'])[:,:,0]
        character_title='Bare BO S0 character' if _is_mcef(f) else 'Bare BO S0 character (CBO unavailable)'
        character_note='Bare electronic BO character.' if _is_mcef(f) else 'Fallback is explicitly bare BO character, NOT CBO character.'
    panels=((gi,r'$\epsilon^{(1)}_{\rm GI}$',display['scales']['energy1'],has_gi),
            (cbo,fr'$E^{{\rm {ref_name}}}_0$ at the same $({_q_symbol(f)},R)$',display['scales']['energy1'],has_cbo),
            (difference,fr'$\epsilon^{{(1)}}_{{\rm GI}}-E^{{\rm {ref_name}}}_0$',difference_limits,has_gi and has_cbo),
            (character,character_title,(0,1),np.any(np.isfinite(character))))
    for n,(ax,(values,title,limits,available)) in enumerate(zip(axes.flat,panels)):
        plotted=dict(f);plotted['epsilon1']=values
        _map(ax,plotted,'epsilon1',limits,window,R[display['i']],title)
        if n==3:
            artist=ax.collections[0];artist.set_cmap('viridis')
            artist.colorbar.set_label('Conditional electronic probability',fontsize=13)
        if not available:
            ax.text(.5,.5,'Unavailable in this cache',ha='center',va='center',
                    transform=ax.transAxes,fontsize=15,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    _title(fig,f,'Same-geometry '+ref_name+' / exact-field comparison')
    fig.text(.5,.075,'GI/'+ref_name+' maps share one energy scale; the difference has a separate zero-centered scale. Static reference interpolation is display-only.',ha='center',fontsize=11.5)
    fig.text(.5,.035,character_note,ha='center',fontsize=11)
    return fig


def _focus_profile(f,display,references,level):
    """Absolute GI and two nearby references; selection is NOT state assignment."""
    i=display['i']; grid=f['R'] if level==2 else f['q']
    selector=None if level==2 else i
    key=f'epsilon{level}_GI'
    gi=_real(f[key]) if key in f else np.full((len(f['R']),len(f['q'])) if level==1 else len(grid),np.nan)
    if level==1:gi=gi[i]
    support=_line_support(f,level,selector)
    bounds=_line_xlim(display,'R' if level==2 else 'q',f)
    inside=(grid>=bounds[0])&(grid<=bounds[1])
    curves=None; labels=[]
    if level==2 and references and 'pol_energy' in references:
        rr=references.get('R',f['R']); values=_real(references['pol_energy'])
        curves=np.column_stack([np.interp(grid,rr,values[:,k]) for k in range(min(4,values.shape[1]))])
        labels=[fr'$E^{{ref}}_{k}$' if _is_mcef(f) else fr'$E^{{x+q_c}}_{k}$' for k in range(curves.shape[1])]
    elif level==2 and _is_mcef(f) and references and 'cbo' in references:
        rr,values=_mcef_outer_bo_cut(f,references,display['j'])
        curves=np.column_stack([np.interp(grid,rr,values[:,k]) for k in range(values.shape[1])])
        labels=[fr'$BO_{k}(q_s,R)$' for k in range(curves.shape[1])]
    if level==1 and references and 'cbo' in references:
        points=np.column_stack((np.full(len(grid),f['R'][i]),grid))
        curves=RegularGridInterpolator((references.get('R',f['R']),references.get('q',f['q'])),
                _real(references['cbo'])[:,:,:4],bounds_error=False,fill_value=np.nan)(points)
        labels=[f'{_reference_name(f)} {k}' for k in range(curves.shape[1])]
    anchor=i if level==2 else int(np.nanargmax(f['lambda_density'][i]))
    ids=[]; nearest=None
    if curves is not None and curves.shape[1]:
        target=gi[anchor]
        distance=abs(curves[anchor]-target) if np.isfinite(target) else np.arange(curves.shape[1],dtype=float)
        ordered=np.argsort(np.where(np.isfinite(distance),distance,np.inf))
        nearest=int(ordered[0]);ids=sorted(int(k) for k in ordered[:2])
    offset=np.full(len(grid),np.nan);gap=None
    if nearest is not None:
        offset=(gi-curves[:,nearest])*27211.386245988
    if len(ids)==2:
        gap=(curves[:,ids[1]]-curves[:,ids[0]])*27211.386245988
    return dict(grid=grid,gi=gi,support=support,inside=inside,bounds=bounds,curves=curves,
                labels=labels,anchor=anchor,ids=ids,nearest=nearest,offset=offset,gap=gap)


def _prepare_reference_focus(files,selected,display,references):
    """Keep focus y ranges fixed between the selected, unshifted snapshots."""
    result={}
    for level in (1,2):
        raw=[]; differences=[]
        for n in selected:
            p=_focus_profile(_load(files[n]),display,references,level)
            raw.append(p['gi'][p['support']&p['inside']])
            if p['curves'] is not None:
                raw.append(p['curves'][np.ix_(p['inside'],p['ids'])])
            differences.append(p['offset'][p['support']&p['inside']])
        result[f'energy{level}']=_range(raw)
        result[f'difference{level}']=_range(differences,symmetric=True)
    return result


def reference_focus_figure(f,display,references=None):
    """Separate GI/reference zoom, with explicitly labelled meV differences.

    No frame-dependent energy subtraction or gauge alignment is applied to
    the upper panels. The lower panels plot differences by definition, not
    shifted versions of the original potentials. Fixed-R electron+photon
    references are NOT global vibrational LP/UP levels.
    """
    fig,axes=plt.subplots(2,2,figsize=(23,13))
    fig.subplots_adjust(left=.075,right=.97,bottom=.16,top=.86,hspace=.64,wspace=.24)
    focus=display.get('reference_focus_limits',{})
    for column,level in enumerate((2,1)):
        p=_focus_profile(f,display,references,level);grid=p['grid'];ax=axes[0,column]
        density=f['rho_R'] if level==2 else f['lambda_density'][display['i']]
        _silhouette(ax,grid,density)
        if np.any(np.isfinite(p['gi'])):
            ax.plot(grid,np.where(p['support'],p['gi'],np.nan),lw=2.6,color=COLORS['GI'],label=fr'$\epsilon^{{({level})}}_{{\rm GI}}$')
        else:
            ax.text(.03,.82,'GI unavailable in this cache',transform=ax.transAxes,ha='left',va='top',color='.4')
        for k in p['ids']:
            ax.plot(grid,p['curves'][:,k],ls='--',lw=2.1,color=REF_COLORS[k],label=p['labels'][k])
        if p['curves'] is None:
            ax.text(.03,.92,'Reference not supplied; none inferred',transform=ax.transAxes,ha='left',va='top',color='.4')
        title=(fr'Outer GI | BO cut $q_s={f["q"][display["j"]]:.2f}$ (reference only)' if _is_mcef(f) else 'Outer GI | fixed R: electron + photon') if level==2 else fr'First-level GI | {_reference_name(f)}, $R={f["R"][display["i"]]:.2f}\,a_0$'
        ax.set_title(title);ax.set_xlim(p['bounds'])
        _nice(ax,r'$R$ ($a_0$)' if level==2 else _q_label(f),'Absolute energy (Ha)',focus.get(f'energy{level}'))
        _legend(ax,3)
        lower=axes[1,column];_silhouette(lower,grid,density)
        if p['nearest'] is not None:
            lower.plot(grid,np.where(p['support'],p['offset'],np.nan),lw=2.3,color=COLORS['GI'],
                       label=fr'GI − reference {p["nearest"]}')
        if p['gap'] is not None:
            lower.text(.98,.94,fr'reference {p["ids"][1]} − {p["ids"][0]} gap: {p["gap"][p["anchor"]]:.3g} meV at marker',
                       transform=lower.transAxes,ha='right',va='top',fontsize=12,
                       bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        lower.axhline(0,color='.4',ls=':',lw=1);lower.axvline(grid[p['anchor']],color='.4',ls='--',lw=1)
        lower.set_xlim(p['bounds']);lower.set_title('Explicit energy differences (no gauge re-alignment)')
        _nice(lower,r'$R$ ($a_0$)' if level==2 else _q_label(f),'Energy difference (meV)',focus.get(f'difference{level}'))
        _legend(lower,2)
    _title(fig,f,'GI / reference focus | absolute energies preserved')
    fig.text(.5,.045,'Two references nearest the local GI are shown for visual comparison; this selection does not assign a populated state or a polariton branch.',ha='center',fontsize=11.5)
    if not _is_mcef(f):
        fig.text(.5,.015,'Fixed-R electron + photon levels are not global vibrational LP/UP levels. Original total, GI and GD remain unchanged in A diagnostics.',ha='center',fontsize=11)
    else:
        fig.text(.5,.015,'The outer BO cut fixes the proton coordinate; it is a comparison curve, not an outer heavy-coordinate adiabatic eigenproblem.',ha='center',fontsize=11)
    return fig


def _title(fig,f,title):
    frequency=(fr', $\omega_c={float(f["omega"]):.4g}$ Ha' if 'omega' in f else '')
    coupling='' if _is_mcef(f) else fr', $\lambda={float(f.get("g_cavity",0)):.3g}$'+frequency
    fig.suptitle(title+fr'    $t={_time(f):.3f}$ fs'+coupling,y=.985)
    for i,ax in enumerate([a for a in fig.axes if a.get_title()]):
        ax.text(-.08,1.06,chr(65+i),transform=ax.transAxes,fontweight='bold',fontsize=17)


def _save(fig,path,dpi):
    fig.savefig(path.with_suffix('.png'),dpi=dpi)
    fig.savefig(path.with_suffix('.pdf'))


def _overview_data(case,first):
    if not (case/'observables.npz').exists():
        return None
    data=_load(case/'observables.npz')
    data.setdefault('t_fs',data.get('times_fs',np.arange(len(data.get('norm',[])))))
    for key in ('R','x','Q'):
        data.setdefault(key,first[key])
    return data


def overview_figure(data,current=None):
    fig,axes=plt.subplots(2,3,figsize=(23,12))
    fig.subplots_adjust(left=.065,right=.97,bottom=.15,top=.91,hspace=.62,wspace=.27)
    t=data['t_fs']
    for ax,key,coordinate,title in zip(axes[0],('rho_x','rho_R','rho_Q'),('x','R','Q'),('Electron','Proton','Photon quadrature')):
        if key in data and np.asarray(data[key]).ndim==2:
            density=np.asarray(data[key]); grid=data[coordinate]
            artist=ax.pcolormesh(t,grid,density.T,shading='auto',cmap='magma')
            fig.colorbar(artist,ax=ax,pad=.02,fraction=.045)
            weight=np.sum(density,axis=1)
            mean=np.sum(density*grid[None,:],axis=1)/weight
            std=np.sqrt(np.maximum(0,np.sum(density*(grid[None,:]-mean[:,None])**2,axis=1)/weight))
            ax.plot(t,mean,color='white',lw=1.6)
            ax.plot(t,mean+std,color='white',ls=':',lw=.9)
            ax.plot(t,mean-std,color='white',ls=':',lw=.9)
        ax.set_title(title); _nice(ax,'t (fs)',r'$Q$ (dimensionless)' if coordinate=='Q' else coordinate+' (a.u.)')
    ax=axes[1,0]
    for key,label in (('photon_number',r'$\langle N_{ph}\rangle$'),('product','product side'),('reactant','reactant side')):
        if key in data: ax.plot(t,data[key],lw=2,label=label)
    ax.set_title('Photon / transfer observables'); _nice(ax,'t (fs)','Number / probability'); _legend(ax,2)
    ax=axes[1,1]
    for key,label in (('kinetic_R','proton kinetic'),('photon_energy','photon energy')):
        if key in data: ax.plot(t,data[key],lw=2,label=label)
    if 'energy' in data: ax.plot(t,np.asarray(data['energy'])-data['energy'][0],lw=1.8,ls='--',label='total energy drift')
    ax.set_title('Energy exchange'); _nice(ax,'t (fs)','Energy (Ha)'); _legend(ax,2)
    ax=axes[1,2]
    keys=[k for k in data if ('pop' in k.lower() or k.lower().startswith('p_')) and np.asarray(data[k]).ndim in (1,2)]
    for key in keys[:3]:
        value=np.asarray(data[key])
        if value.ndim==1: ax.plot(t,value,lw=1.6,label=key)
        else:
            for j in range(min(4,value.shape[1])): ax.plot(t,value[:,j],lw=1.6,label=f'{key} {j}')
    if not keys and 'signed_flux' in data:
        ax.plot(t,data['signed_flux'],lw=2,label='signed dividing-surface flux')
    ax.set_title('Populations / flux'); _nice(ax,'t (fs)','Probability / flux (a.u.)'); _legend(ax,3)
    for ax in axes.flat:
        if current is not None: ax.axvline(current,color='#00A6D6',ls='--',lw=1.4)
    fig.suptitle('Real-space dynamics overview'+('' if current is None else f' | t = {current:.3f} fs'))
    return fig


def transfer_photon_figure(data):
    """Transfer across a chosen plane is not automatically a chemical rate."""
    fig,axes=plt.subplots(2,2,figsize=(20,12))
    fig.subplots_adjust(left=.08,right=.97,bottom=.13,top=.90,hspace=.65,wspace=.26)
    t=data['t_fs']
    groups=[(('reactant','left of dividing plane'),('product','right of dividing plane')),
            (('signed_flux','signed flux'),),
            (('forward_flux_integral','integrated forward flux'),('backflow_integral','integrated backflow')),
            (('photon_number','total photon number'),('photon_coherent_number','coherent displacement'),
             ('photon_fluctuation_number','fluctuations'))]
    titles=['Nuclear probability transfer','Signed nuclear current',
            'Forward transport and backflow','Photon number: displacement vs fluctuations']
    for ax,entries,title in zip(axes.flat,groups,titles):
        for key,label in entries:
            if key in data:ax.plot(t,data[key],lw=2,label=label)
        ax.axhline(0,color='0.5',ls=':',lw=1)
        ax.set_title(title);_nice(ax,'t (fs)','Probability / a.u. time' if len(entries)==1 else 'Probability / number')
        _legend(ax,2)
    fig.suptitle('Transfer and photon diagnostics | dividing plane is not assumed to be a barrier')
    return fig


def timeline_figure(files):
    traces={key:[] for key in ('S_q','S_adv','S_rel','delta','continuity')}; times=[]
    for p in files:
        f=_load(p); times.append(_time(f)); w=_real(f['rho_qR'])
        support=w>np.max(w)*1e-5
        w=np.where(support,w,0); w=w/np.sum(w)
        def rms(values):
            values=np.asarray(values)
            if not np.all(np.isfinite(values[support])): return np.nan
            return float(np.sqrt(np.sum(w[support]*abs(values[support])**2)))
        for key in ('S_q','S_adv','S_rel'):
            try: traces[key].append(rms(_get(f,key)))
            except KeyError: traces[key].append(np.nan)
        traces['delta'].append(rms(_get(f,'b')-f['alpha'][:,None]))
        if 'continuity_residual' in f:
            traces['continuity'].append(rms(f['continuity_residual']))
        else: traces['continuity'].append(np.nan)
    fig,axes=plt.subplots(1,3,figsize=(23,6))
    fig.subplots_adjust(bottom=.27,top=.82,wspace=.28)
    for key in ('S_q','S_adv','S_rel'): axes[0].plot(times,traces[key],color=COLORS[key],lw=2,label=key)
    axes[0].set_title('Support-weighted source RMS'); _nice(axes[0],'t (fs)','Density / a.u. time'); _legend(axes[0])
    axes[1].plot(times,traces['delta'],lw=2,color=COLORS['delta']); axes[1].set_title(r'Support-weighted RMS $b-\alpha$'); _nice(axes[1],'t (fs)','Momentum (a.u.)')
    axes[2].plot(times,traces['continuity'],lw=2,color=COLORS['GD']); axes[2].set_title('Continuity identity residual RMS'); _nice(axes[2],'t (fs)','Density / a.u. time')
    selected=sorted(set([0,len(files)//2,len(files)-1]))
    for ax in axes:
        for i in selected: ax.axvline(times[i],ls='--',lw=1,color='0.4')
    fig.suptitle('Event-selection / consistency timeline | markers: saved snapshots')
    return fig


def population_figure(data):
    """Keep electronic, fixed-R pol, and global vib-pol subspaces distinct."""
    fig,axes=plt.subplots(3,2,figsize=(23,18))
    fig.subplots_adjust(left=.07,right=.97,bottom=.11,top=.91,hspace=.66,wspace=.24)
    t=data['t_fs']
    specifications=(('bo_population','BO electronic projection','bo_outside'),
                    ('pol_population','Fixed-R electron–photon projection','pol_basis_outside'),
                    ('vib_population','Bare local vibration projection','vib_outside'),
                    ('vib_pol_population','Global vibration–photon projection','vib_pol_basis_outside'))
    for ax,(key,title,outside) in zip(axes.flat,specifications):
        if key in data:
            values=np.atleast_2d(data[key])
            for j in range(min(4,values.shape[1])):
                ax.plot(t,values[:,j],lw=2,color=REF_COLORS[j],label=f'state {j}')
            if outside in data: ax.plot(t,data[outside],lw=1.8,color='black',ls='--',label='outside basis')
            # Outside PLOTTED states need not equal outside calculation basis.
            omitted=np.asarray(data.get('norm',np.ones(len(t))))-np.sum(values[:,:4],axis=1)
            ax.plot(t,omitted,lw=1.5,color='0.5',ls=':',label='outside plotted states')
        else:
            ax.text(.5,.5,'Projection unavailable',transform=ax.transAxes,ha='center',color='0.4')
        ax.set_title(title); _nice(ax,'t (fs)','Probability',(-.02,1.02)); _legend(ax,3)
    for ax,keys,title in ((axes[2,0],('pol_pair_lower','pol_pair_upper','pol_pair_other'),'Tracked electronic resonant pair'),
                          (axes[2,1],('vib_LP','vib_UP','vib_other'),'Tracked vibrational resonant pair')):
        for key,label,color in zip(keys,('lower member','upper member','outside pair'),('#009E73','#7B3294','0.4')):
            if key in data: ax.plot(t,data[key],lw=2,color=color,label=label)
        if not any(key in data for key in keys):
            ax.text(.5,.5,'Pair projection unavailable',transform=ax.transAxes,ha='center',color='0.4')
        ax.set_title(title); _nice(ax,'t (fs)','Probability',(-.02,1.02)); _legend(ax,3)
    fig.suptitle('Population diagnostics | analysis-basis truncations do not truncate the TDSE')
    return fig


def static_context_figure(references,first):
    if not references or 'bo_energy' not in references:
        return None
    R=references.get('R',first['R']); energies=_real(references['bo_energy'])
    if len(R)!=energies.shape[0]:
        return None
    occupied=np.asarray(first['rho_R'])>np.max(first['rho_R'])*1e-5
    lower,upper=_bounds(first['R'],occupied,margin=0)
    lower-=2.;upper+=2.
    selected=(R>=lower)&(R<=upper)
    # Keep the complete reference curves, but use the initial occupied
    # neighborhood (plus 2 a0) for the vertical scale. A genuinely occupied
    # pole would still be included; this is not an arbitrary outlier filter.
    chosen=energies[selected,:4]
    chosen=chosen[np.isfinite(chosen)]
    energy_limits=_range([chosen]) if len(chosen) else None
    if len(chosen):
        ticks=MaxNLocator(nbins=5).tick_values(float(chosen.min()),float(chosen.max()))
        energy_limits=(float(ticks[0]),float(ticks[-1]))
    gaps=np.diff(energies[:,:4],axis=1)
    selected_gaps=gaps[selected];selected_gaps=selected_gaps[np.isfinite(selected_gaps)]
    gap_limits=None
    if len(selected_gaps):
        ticks=MaxNLocator(nbins=5).tick_values(float(selected_gaps.min()),float(selected_gaps.max()))
        gap_limits=(float(ticks[0]),float(ticks[-1]))
    fig,axes=plt.subplots(1,2,figsize=(20,6))
    fig.subplots_adjust(left=.075,right=.97,bottom=.25,top=.82,wspace=.24)
    _silhouette(axes[0],first['R'],first['rho_R'])
    for k in range(min(4,energies.shape[1])):
        axes[0].plot(R,energies[:,k],lw=1.8,color=REF_COLORS[k],label=f'bare S{k}')
    axes[0].set_title('Bare electronic surfaces + initial nuclear support')
    _nice(axes[0],r'$R$ ($a_0$)','Energy (Ha)',energy_limits); _legend(axes[0],4)
    _silhouette(axes[1],first['R'],first['rho_R'])
    for k in range(min(3,energies.shape[1]-1)):
        axes[1].plot(R,energies[:,k+1]-energies[:,k],lw=1.8,label=f'S{k+1} − S{k}')
    axes[1].set_title('Bare BO gaps (context, not transfer evidence)')
    _nice(axes[1],r'$R$ ($a_0$)','Gap (Ha)',gap_limits); _legend(axes[1],3)
    for ax in axes: ax.set_xlim(R[0],R[-1])
    fig.suptitle('Electronic avoided-crossing context')
    fig.text(.5,.025,'Full reference curves shown; vertical scale uses initial nuclear support + 2 a0. Empty-region poles may exceed the viewport.',ha='center',fontsize=11)
    return fig


def render_case(case_dir,movies=True,fps=20,dpi=110,output_dir=None,reference_path=None):
    """Generate reports automatically after propagation or from cached fields.

    Movies are encoded directly, so no full frame-PNG directory is created.
    A failure of optional reference loading does not abort basic diagnostics;
    movie availability/failures are explicit in render_manifest.json.
    """
    case=Path(case_dir); files=sorted((case/'fields').glob('fields_*.npz'))
    # A failed run may leave fields newer than its durable observable index.
    # Ignore such orphan frames rather than deleting user data or rendering
    # an inconsistent history after rollback/resume.
    if (case/'observables.npz').exists():
        with np.load(case/'observables.npz',allow_pickle=False) as obs:
            if 'step' in obs.files:
                valid_steps=set(int(s) for s in obs['step'])
                files=[p for p in files if int(p.stem.split('_')[-1]) in valid_steps]
    if not files:
        raise ValueError('No compact field frames found in '+str(case/'fields'))
    refs=None; notes=[];refs_path=Path(reference_path) if reference_path is not None else case/'references.npz'
    if refs_path.exists():
        try: refs=_load(refs_path)
        except (OSError,ValueError) as exc: notes.append('Reference cache unavailable: '+str(exc))
    else: notes.append('No references.npz: CBO/polaritonic overlays skipped, no fabricated curves.')
    if reference_path is not None:
        notes.append('External static references: '+str(refs_path.resolve())+'. Cached fields/populations are unchanged; refined overlays do not certify production or population convergence.')
    display=prepare_display(files,refs); first=_load(files[0]); out=Path(output_dir) if output_dir is not None else case/'report'; out.mkdir(parents=True,exist_ok=True)
    selected=sorted(set([0,len(files)//2,len(files)-1])); produced=[]
    has_video=bool(movies and writers.is_available('ffmpeg') and len(files)>1)
    if movies and not has_video: notes.append('Movies skipped: ffmpeg unavailable or fewer than two saved physical times.')
    families={'A_diagnostics_clean':bridge_figure,'B_diagnostics_clean':source_figure,
              'bridge_consistency':consistency_figure,'epsilon1_slice_analysis':slice_figure,
              'bridge_wavefunction':bridge_wavefunction_figure}
    with plt.rc_context(STYLE):
        for family,function in families.items():
            directory=out/family; directory.mkdir(exist_ok=True)
            writer=None; writer_figure=None
            try:
                for n,path in enumerate(files):
                    if not has_video and n not in selected: continue
                    f=_load(path); fig=function(f,display,n,refs)
                    if n in selected:
                        target=directory/f'snapshot_{n:06d}'
                        _save(fig,target,dpi); produced.extend([str(target.with_suffix('.png')),str(target.with_suffix('.pdf'))])
                    if has_video:
                        if writer is None:
                            writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-crf','23','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-threads','1'])
                            writer.setup(fig,str(directory/(family+'.mp4')),dpi=dpi)
                        # The frame dimensions/layout are invariant across figures.
                        writer.fig=fig; writer.grab_frame(facecolor='white')
                    plt.close(fig)
                if writer:
                    writer.finish(); writer=None
                    produced.append(str(directory/(family+'.mp4')))
            finally:
                if writer: writer.finish()
        component_dir=out/'epsilon1_component_slices'; component_dir.mkdir(exist_ok=True)
        for n in selected:
            fig=component_figure(_load(files[n]),display)
            target=component_dir/f'snapshot_{n:06d}'; _save(fig,target,dpi); plt.close(fig)
            produced.extend([str(target.with_suffix('.png')),str(target.with_suffix('.pdf'))])
        force_dir=out/'outer_force'; force_dir.mkdir(exist_ok=True)
        for n in selected:
            fig=outer_force_figure(_load(files[n]),display)
            target=force_dir/f'snapshot_{n:06d}'; _save(fig,target,dpi); plt.close(fig)
            produced.extend([str(target.with_suffix('.png')),str(target.with_suffix('.pdf'))])
        cbo_dir=out/('same_geometry_BO' if _is_mcef(first) else 'same_geometry_CBO');cbo_dir.mkdir(exist_ok=True)
        differences=[]
        for n in selected:
            f=_load(files[n]);support=f['rho_qR']>np.nanmax(f['rho_qR'])*1e-5
            if 'epsilon1_GI' in f:
                differences.append((_real(f['epsilon1_GI'])-_cbo_on_field_grid(f,refs))[support])
        display['scales']['cbo_difference']=_range(differences,symmetric=True)
        for n in selected:
            fig=cbo_comparison_figure(_load(files[n]),display,n,refs)
            target=cbo_dir/f'snapshot_{n:06d}';_save(fig,target,dpi);plt.close(fig)
            produced.extend([str(target.with_suffix('.png')),str(target.with_suffix('.pdf'))])
        focus_dir=out/'reference_focus';focus_dir.mkdir(exist_ok=True)
        display['reference_focus_limits']=_prepare_reference_focus(files,selected,display,refs)
        for n in selected:
            fig=reference_focus_figure(_load(files[n]),display,refs)
            target=focus_dir/f'snapshot_{n:06d}';_save(fig,target,dpi);plt.close(fig)
            produced.extend([str(target.with_suffix('.png')),str(target.with_suffix('.pdf'))])
        timeline=timeline_figure(files); target=out/'event_timeline'; _save(timeline,target,dpi); plt.close(timeline)
        produced.extend([str(target.with_suffix('.png')),str(target.with_suffix('.pdf'))])
        context=static_context_figure(refs,first)
        if context is not None:
            target=out/'bare_BO_context'; _save(context,target,dpi); plt.close(context)
            produced.extend([str(target.with_suffix('.png')),str(target.with_suffix('.pdf'))])
        data=_overview_data(case,first)
        if data is not None:
            fig=overview_figure(data); _save(fig,out/'dynamics_overview',dpi); plt.close(fig)
            produced.extend([str(out/'dynamics_overview.png'),str(out/'dynamics_overview.pdf')])
            fig=population_figure(data); _save(fig,out/'population_diagnostics',dpi); plt.close(fig)
            produced.extend([str(out/'population_diagnostics.png'),str(out/'population_diagnostics.pdf')])
            fig=transfer_photon_figure(data); _save(fig,out/'transfer_photon_diagnostics',dpi);plt.close(fig)
            produced.extend([str(out/'transfer_photon_diagnostics.png'),str(out/'transfer_photon_diagnostics.pdf')])
            if has_video:
                writer=None
                try:
                    for path in files:
                        f=_load(path); fig=overview_figure(data,_time(f))
                        if writer is None:
                            writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-crf','23','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-threads','1'])
                            writer.setup(fig,str(out/'dynamics_overview.mp4'),dpi=dpi)
                        writer.fig=fig; writer.grab_frame(facecolor='white'); plt.close(fig)
                    writer.finish(); writer=None; produced.append(str(out/'dynamics_overview.mp4'))
                finally:
                    if writer: writer.finish()
    manifest=dict(outputs=produced,notes=notes,physical_frames=len(files),fps=fps,
                  frame_interpolation=False,scales=display['scales'],
                  slice_R_values=first['R'][display['slices_R']].tolist(),
                  slice_q_values=first['q'][display['slices_q']].tolist(),
                  slice_selection='initial marginal peak and occupied time-integrated quartiles',
                  density_contour_relative_min=1e-5,rendered_png_frames=selected,
                  density_contour_relative_levels=CONTOUR_LEVELS.tolist(),
                  fixed_1D_viewport=dict(R=display['line_xlim_R'],q=display['line_xlim_q']),
                  camera_policy=f'2D density-following support+10% margin; smoothing resets when adjacent saved times differ by more than {CAMERA_RESET_GAP_FS:g} fs. 1D viewports stay fixed over the occupied union.',
                  dynamic_line_support='Outer rho_R/frame peak >1e-5; first-level joint-density slice/slice peak >1e-5, fixed-R slices also require occupied outer support. Static references remain full-range; 2D maps remain unmasked inside their window.',
                  timeline_RMS_support='rho_qR / frame_peak > 1e-5; renormalized joint-density weights',
                  source_target='partial_t |Lambda_R|^2; NOT i partial_t Lambda_R',
                  scale_policy='fixed density-supported robust .3–99.7 percentile limits; tails may saturate',
                  gauge='positive marginal gauge; scalar slope alone is not the full EF force',
                  reference_policy='static comparison; GI/GD and exact time dependence need not follow a single BO/CBO curve')
    manifest['reference_range_policy']=display['reference_range_policy']
    manifest['reference_energy_ranges']=display['reference_ranges']
    manifest['reference_focus_limits']=display['reference_focus_limits']
    manifest['reference_focus_policy']='Absolute GI and references are unshifted. Two references nearest the local GI are selected descriptively, not as population/branch assignments; differences are explicitly in meV. '+('The outer BO cut fixes q_s and is not an outer adiabatic eigenproblem.' if _is_mcef(first) else 'Fixed-R electron+photon references are not vibrational LP/UP.')
    if refs_path.is_file():
        digest=hashlib.sha256()
        with refs_path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        manifest['reference_cache']=dict(path=str(refs_path.resolve()),sha256=digest.hexdigest(),external=reference_path is not None)
    manifest['output_dir']=str(out.resolve())
    manifest['system']='mcef' if _is_mcef(first) else 'cavity'
    manifest['scalar_kind']=str(first.get('scalar_kind','exact scalar, positive marginal gauge'))
    manifest['energy_convention']=str(first.get('energy_convention','cached absolute energies, unchanged'))
    (out/'render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def render_comparison(root):
    """Coupling comparisons from existing observables; never reruns dynamics."""
    root=Path(root); cases=sorted(root.rglob('observables.npz'))
    if not cases: return {'outputs':[],'notes':['No observables available']}
    outputs=[]
    with plt.rc_context(STYLE):
        for campaign in ('pen','vsc'):
            selected=[p for p in cases if campaign in str(p.relative_to(root)).lower()]
            if not selected: continue
            fig,axes=plt.subplots(2,2,figsize=(20,12)); fig.subplots_adjust(bottom=.14,hspace=.56,wspace=.25)
            for p in selected:
                d=_load(p); t=d.get('t_fs',d.get('times_fs'))
                if t is None: continue
                label=p.parent.name
                for ax,key,y in zip(axes.flat,('product','photon_number','kinetic_R','signed_flux'),('Product-side probability',r'$\langle N_{ph}\rangle$','Proton kinetic (Ha)','Signed flux (a.u.)')):
                    if key in d: ax.plot(t,d[key],lw=2,label=label)
                    _nice(ax,'t (fs)',y); ax.axhline(0,color='0.5',ls=':',lw=.7)
            for ax in axes.flat: _legend(ax)
            fig.suptitle(campaign.upper()+' | cavity coupling comparison')
            path=root/(campaign+'_coupling_comparison'); _save(fig,path,130); plt.close(fig)
            outputs.extend([str(path.with_suffix('.png')),str(path.with_suffix('.pdf'))])
    return {'outputs':outputs}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir'); parser.add_argument('--no-movies',action='store_true')
    parser.add_argument('--fps',type=int,default=20); parser.add_argument('--dpi',type=int,default=110)
    parser.add_argument('--output-dir');parser.add_argument('--references')
    args=parser.parse_args()
    print(json.dumps(render_case(args.case_dir,not args.no_movies,args.fps,args.dpi,
                                 output_dir=args.output_dir,reference_path=args.references),indent=2))
