"""Cached GI budgets and explicit reference-relative magnification.

Definitions come from fields.py: GI1=cond1+qgeo+Rgeo, GI2=cond2+geo2.
No new field extraction, gauge alignment, population assignment or dynamics.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .render import (_load, _real, _line_support, _line_xlim, _nice, _legend,
                     _silhouette, _title, _references, _focus_profile,
                     _q_label, _q_symbol, COLORS, REF_COLORS)

HA_TO_MEV = 27211.386245988


def _limits(parts):
    """Full finite extrema, not percentiles: preserve small real structures."""
    values = [np.asarray(p).ravel() for p in parts if np.size(p)]
    v = np.concatenate(values) if values else np.array([])
    v = v[np.isfinite(v)]
    if not len(v): return (-1., 1.)
    lo, hi = float(v.min()), float(v.max())
    pad = .06*max(hi-lo, abs(hi)*1e-6, 1e-10)
    ticks = MaxNLocator(nbins=5).tick_values(lo-pad, hi+pad)
    return float(ticks[0]), float(ticks[-1])


def _cut(f, display, direction):
    i, j = display['i'], display['j']
    if direction == 'q':
        return f['q'], i, f['lambda_density'][i], _line_support(f, 1, i)
    selector = (slice(None), j)
    return f['R'], selector, f['rho_qR'][:,j], _line_support(f, 1, selector)


def prepare_gi_display(files, display, references):
    """One streaming scan; fixed slices, energy scales and reference baseline.

    Reference-zero subtraction removes common curvature for visibility only.
    It is spatially dependent, so is NOT a gauge transformation or a force PES.
    """
    samples = {k: [] for k in ('energy1','energy2','geo1','geo2','zoom1','zoom2','pair1','pair2')}
    first=_load(files[0])
    pairs={level: _focus_profile(first,display,references,level)['ids'] for level in (1,2)}
    for path in files:
        f = _load(path)
        for direction in ('q','R'):
            x, sel, density, support = _cut(f, display, direction)
            for suffix in ('GI','cond'):
                key = 'epsilon1_'+suffix
                if key in f: samples['energy1'].append(_real(f[key])[sel][support])
            geometry = []
            for suffix in ('qgeo','Rgeo'):
                key = 'epsilon1_'+suffix
                if key in f:
                    v = _real(f[key])[sel][support]
                    samples['geo1'].append(v); geometry.append(v)
            if len(geometry)==2: samples['geo1'].append(geometry[0]+geometry[1])
        support = _line_support(f, 2)
        for suffix in ('GI','cond'):
            if 'epsilon2_'+suffix in f:
                samples['energy2'].append(_real(f['epsilon2_'+suffix])[support])
        if 'epsilon2_geo' in f: samples['geo2'].append(_real(f['epsilon2_geo'])[support])
        for level in (1,2):
            p = _focus_profile(f, display, references, level)
            if p['curves'] is None: continue
            # Same ground-reference baseline for every line and every frame.
            curves = p['curves']; baseline = curves[:,0]
            samples[f'zoom{level}'].append((p['gi']-baseline)[p['support']&p['inside']]*HA_TO_MEV)
            samples[f'zoom{level}'].append((curves-baseline[:,None])[p['inside']]*HA_TO_MEV)
            ids=pairs[level]
            if ids:
                samples[f'pair{level}'].append((curves[:,ids]-curves[:,ids[0],None])[p['inside']]*HA_TO_MEV)
    # Include all four references in the absolute GI viewport, but never GD/total.
    for key, limits in display.get('reference_ranges', {}).items():
        if key in ('energy1','energy2'): samples[key].append(limits)
    result={key: _limits(values) for key, values in samples.items()}
    result['pair_indices']={str(k):v for k,v in pairs.items()}
    return result


def _plot_terms(ax, x, f, selector, support, terms, factor=1.):
    drawn = 0
    for key, label, color in terms:
        if key not in f: continue
        values = _real(f[key])
        if selector is not None: values = values[selector]
        ax.plot(x,np.where(support,values*factor,np.nan),lw=2.3,color=color,label=label)
        drawn += 1
    if drawn < len(terms):
        ax.text(.02,.96,'Some GI components unavailable',transform=ax.transAxes,
                va='top',fontsize=11,color='.4')


def epsilon1_gi_figure(f, display, index=0, references=None):
    fig, axes = plt.subplots(2,2,figsize=(24,14))
    fig.subplots_adjust(left=.07,right=.97,bottom=.13,top=.88,hspace=.64,wspace=.24)
    scales = display['gi_scales']
    for col, direction in enumerate(('q','R')):
        x, sel, density, support = _cut(f,display,direction)
        ax = axes[0,col]; _silhouette(ax,x,density)
        _plot_terms(ax,x,f,sel,support,[('epsilon1_GI','GI total',COLORS['GI']),
                    ('epsilon1_cond',r'$\langle h\rangle_x$','#161616')])
        _references(ax,references,f,level=1,direction=direction,
                    fixed_index=display['i'] if direction=='q' else display['j'])
        fixed = (fr'$R={f["R"][display["i"]]:.2f}\,a_0$' if direction=='q'
                 else fr'${_q_symbol(f)}={f["q"][display["j"]]:.2f}$')
        ax.set_title('GI / static references | '+fixed); _legend(ax,3)
        _nice(ax,_q_label(f) if direction=='q' else r'$R$ ($a_0$)','Energy (Ha)',scales['energy1'])
        ax=axes[1,col];_silhouette(ax,x,density)
        _plot_terms(ax,x,f,sel,support,[('epsilon1_qgeo',fr'$G_{{{_q_symbol(f)}}}^\Phi$','#7B3294'),
                    ('epsilon1_Rgeo',r'$G_R^\Phi$','#009E73')])
        if all(k in f for k in ('epsilon1_qgeo','epsilon1_Rgeo')):
            total = _real(f['epsilon1_qgeo']+f['epsilon1_Rgeo'])[sel]
            ax.plot(x,np.where(support,total,np.nan),color='0.15',ls='--',lw=2,label='Geometric sum')
        ax.axhline(0,color='.5',ls=':',lw=1)
        ax.set_title('Geometric contributions | expanded scale');_legend(ax,3)
        _nice(ax,_q_label(f) if direction=='q' else r'$R$ ($a_0$)','Energy (Ha)',scales['geo1'])
        for ax in axes[:,col]: ax.set_xlim(_line_xlim(display,direction,f))
    _title(fig,f,r'$\epsilon^{(1)}$ GI decomposition')
    fig.text(.5,.025,r'GI = $\langle h\rangle_x+G_{q_c}^{\Phi}+G_R^{\Phi}$ (cavity). Absolute energies unchanged; lower row uses a separate expanded scale.',ha='center',fontsize=12)
    return fig


def outer_gi_figure(f, display, index=0, references=None):
    fig, axes=plt.subplots(1,2,figsize=(24,8))
    fig.subplots_adjust(left=.075,right=.97,bottom=.27,top=.81,wspace=.26)
    R=f['R'];support=_line_support(f,2);scales=display['gi_scales']
    for ax in axes:_silhouette(ax,R,f['rho_R']);ax.set_xlim(_line_xlim(display,'R',f))
    _plot_terms(axes[0],R,f,None,support,[('epsilon2_GI','GI total',COLORS['GI']),
                ('epsilon2_cond',r'$\langle H-T_R\rangle_\Gamma$','#161616')])
    _references(axes[0],references,f,level=2,fixed_index=display['j'])
    axes[0].set_title('Outer GI / static references');_legend(axes[0],3)
    _nice(axes[0],r'$R$ ($a_0$)','Energy (Ha)',scales['energy2'])
    _plot_terms(axes[1],R,f,None,support,[('epsilon2_geo',r'$G_R^\Gamma$','#7B3294')])
    axes[1].axhline(0,color='.5',ls=':',lw=1);axes[1].set_title('Geometric contribution | expanded scale')
    _nice(axes[1],r'$R$ ($a_0$)','Energy (Ha)',scales['geo2']);_legend(axes[1],1)
    _title(fig,f,r'$\epsilon^{(2)}$ GI decomposition')
    fig.text(.5,.035,r'GI = $\langle H-T_R\rangle_\Gamma+G_R^\Gamma$. The geometric contribution alone is not the total nuclear force.',ha='center',fontsize=12)
    return fig


def gi_reference_zoom_figure(f, display, index=0, references=None):
    fig,axes=plt.subplots(2,2,figsize=(24,15))
    fig.subplots_adjust(left=.075,right=.97,bottom=.19,top=.88,wspace=.24,hspace=.68)
    for col,level in enumerate((2,1)):
        ax=axes[0,col];p=_focus_profile(f,display,references,level);x=p['grid']
        _silhouette(ax,x,f['rho_R'] if level==2 else f['lambda_density'][display['i']])
        if p['curves'] is None:
            ax.text(.5,.5,'Reference unavailable: no energy subtraction',ha='center',transform=ax.transAxes)
        else:
            baseline=p['curves'][:,0]
            ax.plot(x,np.where(p['support'],(p['gi']-baseline)*HA_TO_MEV,np.nan),
                    color=COLORS['GI'],lw=2.6,label='GI − reference 0')
            for k in range(p['curves'].shape[1]):
                ax.plot(x,(p['curves'][:,k]-baseline)*HA_TO_MEV,color=REF_COLORS[k],
                        ls='--',lw=2,label=p['labels'][k])
            _legend(ax,3)
        ax.set_xlim(p['bounds']);ax.axhline(0,color='.5',ls=':',lw=.8)
        ax.set_title('Outer GI / Pol gaps' if level==2 else fr'First-level GI / CBO gaps | $R={f["R"][display["i"]]:.2f}\,a_0$')
        _nice(ax,r'$R$ ($a_0$)' if level==2 else _q_label(f),
              r'$E-E_{\mathrm{ref},0}$ (meV)',display['gi_scales'][f'zoom{level}'])
        ax=axes[1,col]
        ids=display['gi_scales']['pair_indices'][str(level)]
        limits=display['gi_scales'][f'pair{level}']
        if p['curves'] is not None and ids:
            baseline=p['curves'][:,ids[0]]
            gi=(p['gi']-baseline)*HA_TO_MEV
            ax.plot(x,np.where(p['support'],gi,np.nan),color=COLORS['GI'],lw=2.6,label=f'GI − reference {ids[0]}')
            for k in ids:
                ax.plot(x,(p['curves'][:,k]-baseline)*HA_TO_MEV,ls='--',lw=2,color=REF_COLORS[k],label=p['labels'][k])
            occupied=p['support']&p['inside']&np.isfinite(gi)
            if np.any(occupied&((gi<limits[0])|(gi>limits[1]))):
                ax.text(.02,.96,'GI extends outside this zoom; see upper panel',transform=ax.transAxes,va='top',fontsize=11,
                        bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            ax.set_title(f'Reference pair {ids[0]}, {ids[-1]} | fixed zoom')
            _nice(ax,r'$R$ ($a_0$)' if level==2 else _q_label(f),
                  fr'$E-E_{{\mathrm{{ref}},{ids[0]}}}$ (meV)',limits)
            _legend(ax,3)
        else:
            ax.text(.5,.5,'Reference pair unavailable',ha='center',transform=ax.transAxes)
        ax.set_xlim(p['bounds']);ax.axhline(0,color='.5',ls=':',lw=.8)
    _title(fig,f,'GI / reference energy differences')
    fig.text(.5,.045,'Each panel subtracts ONE common static reference from every curve. This is NOT a gauge transformation or a force potential.',ha='center',fontsize=12)
    fig.text(.5,.018,'Lower pair: two references nearest initial GI at the density marker, held fixed in time; visual selection, NOT a population / LP–UP assignment.',ha='center',fontsize=11)
    return fig
