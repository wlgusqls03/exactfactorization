"""Expanded PG proton-heavy maps, movies, signed actions and closure audit."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from .proton_heavy_terms import (TermConfig, conditional_density, time_rate,
                                frame_terms, summarize_frame, peak_integrals)
from .report_plot_style import MASK_COLOR, SIGNED_CMAP


LABELS = {
    'lambda_density': r'$|\Lambda_R|^2$', 'joint_density': r'$\rho_{pR}$',
    'heavy_density': r'$\rho_R$', 'a': r'$a$', 'b': r'$b$', 'alpha': r'$\alpha$',
    'delta': r'$b-\alpha$', 'delta_squared': r'$(b-\alpha)^2$',
    'density_dt': r'$\partial_t|\Lambda_R|^2$', 'S_q': r'$S_q$',
    'S_adv': r'$S_{\mathrm{adv}}$', 'S_rel': r'$S_{\mathrm{rel}}$',
    'S_U': r'$S_U=S_{\mathrm{adv}}+S_{\mathrm{rel}}$',
    'residual_density': r'$\partial_t|\Lambda_R|^2-S_q-S_U$',
    'J_rel': r'$J_{\mathrm{rel}}^R$', 'S_U_operator': r'$2\,\mathrm{Im}(\Lambda^* U_{\mathrm{op}}\Lambda)$',
    'residual_source_operator': r'$2\,\mathrm{Im}(\Lambda^*U_{\mathrm{op}}\Lambda)-S_U$',
    'residual_source_expanded': r'$2\,\mathrm{Im}(\Lambda^*U_{\mathrm{exp}}\Lambda)-S_U$',
    'residual_product_rule': r'$U_{\mathrm{op}}\Lambda-\sum_iT_i$',
    'T1': r'$T_1=-\partial_R^2\Lambda/(2M)$',
    'T2': r'$T_2=-(\partial_R\chi/\chi)\partial_R\Lambda/M$',
    'T3': r'$T_3=(b-\alpha)^2\Lambda/(2M)$',
    'T4': r'$T_4=\alpha(b-\alpha)\Lambda/M$',
    'T5': r'$\mathrm{Im}\,T_5=-(b-\alpha)\partial_R\Lambda/M$',
    'T6': r'$\mathrm{Im}\,T_6=-\partial_R(b-\alpha)\Lambda/(2M)$',
    'T7': r'$\mathrm{Im}\,T_7=-\alpha\partial_R\Lambda/M$',
    'T8': r'$\mathrm{Im}\,T_8=-(\partial_R\chi/\chi)(b-\alpha)\Lambda/M$',
}
GROUPS = {
    'state': ('lambda_density','joint_density','heavy_density','a','b','alpha','delta','delta_squared'),
    'density': ('density_dt','S_q','S_adv','S_rel','S_U','residual_density'),
    'real_terms': ('T1','T2','T3','T4'),
    'imag_terms': ('T5','T6','T7','T8'),
    'sums': ('U_quad.real','U_lin.real','U_total.real','U_quad.imag','U_lin.imag','U_total.imag'),
    'relative': ('b','alpha','delta','J_rel','S_rel','S_U'),
    'closure': ('S_U','S_U_operator','residual_source_operator','residual_source_expanded',
                'residual_product_rule.real','residual_product_rule.imag'),
}


def component(fields, key):
    name, _, part = key.partition('.')
    value = fields[name]
    return value.imag if part == 'imag' or name in ('T5','T6','T7','T8') else value.real


def category(key):
    if key in ('lambda_density','joint_density','heavy_density','delta_squared'):
        return key, True
    if key in ('a','b','alpha','delta'):
        return 'momentum', False
    if key == 'J_rel':
        return 'relative_current', False
    if key.startswith('residual_product_rule'):
        return 'action_residual', False
    if key.startswith('residual_'):
        return 'source_residual', False
    if key.startswith('T') or key.startswith('U_') or key.startswith('residual_product_rule'):
        return 'action', key == 'T3'
    return 'density_rate', False


def title(key):
    name, _, part = key.partition('.')
    if name in LABELS:
        return LABELS[name]+(' ('+part+')' if part else '')
    return r'$\mathrm{'+part+r'}\,[U_{\mathrm{'+name[2:]+r'}}\Lambda]$'


def render_proton_heavy(obs, ef, output, args, snapshots):
    from .render_final_visualizations import (_save_figure, _save_analysis_movie,
        _save_individual_frames, _absolute_overlay, _movie_frames)
    if ef.get('gauge') not in ('positive_density','positive_density_marginals'):
        raise ValueError('Expanded coupling analysis requires positive marginals')
    config = TermConfig(getattr(args,'ph_density_floor',1e-3),
                        getattr(args,'ph_heavy_floor',1e-12), getattr(args,'ph_q_split',0.))
    groups = getattr(args,'ph_groups',None) or tuple(GROUPS)
    stride = int(getattr(args,'ph_map_stride',2))
    color_quantile = float(getattr(args,'ph_color_quantile',.995))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    times, q, R = obs['times_fs'], obs['q'], obs['R']
    dq, dR = obs['dq'], obs['dR']
    masses = [float(obs['options'][key]) for key in ('proton_mass','heavy_mass')]
    def density(f):
        return conditional_density(obs['joint_density'][f],obs['heavy_density'][f],config.heavy_floor)
    @lru_cache(maxsize=1)
    def values(f):
        return frame_terms(obs['joint_density'][f],obs['heavy_density'][f],ef['a'][f],
            ef['b'][f],ef['alpha'][f],dq,dR,*masses,time_rate(density,times,f),config)
    rows, peaks, bounds = [], [], {}
    products = []
    for f in range(len(times)):
        result = values(f)
        row = summarize_frame(result,dq,dR)
        row['time_fs'] = float(times[f])
        row['one_sided_time_derivative'] = f in (0,len(times)-1)
        fields, mask = result['fields'], result['valid']
        coarse = time_rate(density,times,f,stride=2)
        diff = coarse-fields['density_dt']
        weight = np.where(mask,fields['joint_density'],0)
        time_mask = mask & np.isfinite(diff)
        tw = np.where(time_mask,weight,0)
        row['time_sampling_rms'] = (float(np.sqrt(np.sum(tw*np.where(time_mask,diff,0)**2)/tw.sum()))
                                    if tw.sum() else None)
        row['phase_branch_sites'] = int(np.count_nonzero((obs['joint_density'][f]>=config.density_floor)
            & ((abs(ef['b'][f]*dR)>.9*np.pi) | (abs(ef['a'][f]*dq)>.9*np.pi))))
        row['component_closure_max'] = (float(np.max(abs((fields['U_total']-
            sum(fields[f'T{i}'] for i in range(1,9)))[mask]))) if mask.any() else None)
        denom = sum(row['rms'][k] or 0 for k in ('density_dt','S_q','S_adv','S_rel'))
        row['density_relative_residual'] = row['rms']['residual_density']/denom if denom else None
        rows.append(row)
        peak = peak_integrals(result,q,dq,config.q_split)
        peak['display_support_R'] = mask.any(axis=0)
        peaks.append(peak)
        for key in set(k for group in groups for k in GROUPS[group]):
            cat, positive = category(key)
            if key == 'heavy_density':
                data = result[key][np.isfinite(result[key])]
            elif key == 'alpha':
                data = result['alpha_line'][mask.any(axis=0)]
            else:
                data = component(fields,key)[mask]
            if data.size:
                # Fixed trajectory-wide envelope of frame quantiles, shared by units.
                # Clipping is explicit (colorbar extensions/footer); raw maxima remain in JSON.
                bounds[cat] = max(bounds.get(cat,0),float(np.quantile(abs(data),color_quantile)))
        if f in snapshots:
            path = output/f'proton_heavy_terms_frame_{f:04d}.npz'
            np.savez_compressed(path, q=q[::stride],R=R[::stride],time_fs=times[f],
                valid=mask[::stride,::stride], map_stride=stride,
                **{k:v[::stride,::stride] for k,v in fields.items()})
            products.append(path)
        if f % 50 == 0:
            print(f'Proton-heavy terms: {f+1}/{len(times)}',flush=True)
    provenance = dict(gauge='positive_density', connections='forward bond cache averaged to sites',
        derivative='core.derivative central5 periodic D1/D2; core.covariant_square anticommutator',
        native_propagator=obs['options'].get('tdse_propagator','unknown'),
        certification='DIAGNOSTIC_ONLY: not a native-link or spectral TDSE identity',
        temporal_derivative='saved-density finite differences; stride1 vs stride2 sensitivity',
        caveat='Cached connections can describe a BO-projected state whereas saved density is full-grid. '
               'Residuals include cache mismatch, bond/site, spatial and temporal discretization; not a missing physical source.',
        density_floor=config.density_floor,heavy_floor=config.heavy_floor,q_split=config.q_split,
        action_units='Ha a0^(-1/2); NOT divided by Lambda',source_units='a0^(-1) atomic_time^(-1)',
        metrics='joint-density weighted RMS on common finite display support',
        map_stride=stride,source_archive=str(obs.get('archive_path','')),ef_cache=str(ef.get('path','')),
        color_quantile=color_quantile, fixed_color_bounds=bounds,
        color_policy='max over frames of per-frame occupied-site absolute-value quantile; residuals separate',
        records=rows)
    path=output/'proton_heavy_terms_diagnostics.json'
    path.write_text(json.dumps(provenance,indent=2,allow_nan=False));products.append(path)
    path=output/'proton_heavy_peak_sources.npz'
    np.savez_compressed(path,times_fs=times,R=R,q_split=config.q_split,
                       **{k:np.array([p[k] for p in peaks]) for k in peaks[0]})
    products.append(path)
    def build(first, group):
        keys = GROUPS[group]; cols = 4 if len(keys)==8 else 2 if len(keys)==4 else 3
        fig,axes=plt.subplots(2,cols,figsize=(6*cols,9),squeeze=False,constrained_layout=True)
        artists=[]
        for ax,key in zip(axes.flat,keys):
            cat,positive=category(key);bound=max(bounds.get(cat,0),1e-15)
            if key in ('heavy_density','alpha'):
                line,=ax.plot(R,np.zeros_like(R),lw=2,color='tab:purple')
                ax.set(ylim=(0,bound*1.05) if positive else (-bound*1.05,bound*1.05),
                       xlabel=r'$R$ ($a_0$)',ylabel='density (a.u.)' if positive else 'momentum (a.u.)')
                ax.axhline(0,color='gray',ls='--',lw=.6);artists.append(line)
            else:
                cmap=plt.get_cmap('magma' if positive else SIGNED_CMAP).copy();cmap.set_bad(MASK_COLOR)
                image=ax.imshow(np.zeros((2,2)),origin='lower',aspect='auto',interpolation='nearest',
                    cmap=cmap,norm=Normalize(0 if positive else -bound,bound))
                fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((-3,3))
                units={'action':r'Ha $a_0^{-1/2}$','density_rate':r'$a_0^{-1}t_{au}^{-1}$',
                       'action_residual':r'Ha $a_0^{-1/2}$','source_residual':r'$a_0^{-1}t_{au}^{-1}$',
                       'momentum':'momentum (a.u.)','joint_density':r'$a_0^{-2}$',
                       'lambda_density':r'$a_0^{-1}$','delta_squared':'momentum squared (a.u.)',
                       'relative_current':'relative current (a.u.)'}
                fig.colorbar(image,ax=ax,pad=.02,shrink=.82,format=fmt,label=units[cat],
                             extend='max' if positive else 'both')
                ax.set(xlabel=r'$q$ ($a_0$)',ylabel=r'$R$ ($a_0$)');artists.append(image)
            ax.set_title(title(key),fontsize=14);ax.tick_params(labelsize=11)
        heading=fig.suptitle('',fontsize=17)
        fig.supxlabel(f'PG | central5 diagnostic, not native TDSE identity | fixed {100*color_quantile:g}% scale; '
                      'colorbar tips indicate saturation; residual scales separate',fontsize=10)
        def update(f):
            result=values(int(f));mask=result['valid']
            found=[np.flatnonzero(mask.any(axis=k)) for k in (1,0)]
            limits=[]
            for coord,ids in zip((q,R),found):
                if not ids.size:ids=np.arange(len(coord))
                lo,hi=max(0,ids[0]-3),min(len(coord)-1,ids[-1]+3)
                limits.append((coord[lo],coord[hi]))
            for ax,key,artist in zip(axes.flat,keys,artists):
                if key in ('heavy_density','alpha'):
                    line=result['heavy_density'] if key=='heavy_density' else result['alpha_line']
                    line=np.where(result['heavy_density']>config.heavy_floor,line,np.nan)
                    artist.set_ydata(line);ax.set_xlim(limits[1])
                else:
                    value=component(result['fields'],key)
                    artist.set_data(np.ma.masked_where(~mask,value)[::stride,::stride].T)
                    artist.set_extent((q[0],q[::stride][-1],R[0],R[::stride][-1]))
                    ax.set(xlim=limits[0],ylim=limits[1]);_absolute_overlay(ax,obs,int(f),compact=True)
            flags=rows[int(f)]['phase_branch_sites']
            warning=f' | connection branch flags: {flags}' if flags else ''
            heading.set_text(f'Proton–heavy: {group.replace("_"," ")} | t={times[int(f)]:.4f} fs'+warning)
            heading.set_color('darkred' if flags else 'black')
            return artists
        update(first);return fig,update
    for group in groups:
        stem='proton_heavy_'+group
        products.extend(_save_individual_frames(lambda f:build(f,group)[0],snapshots,times,
                                               output/(stem+'_frames'),stem,args.dpi))
        if not args.no_animation:
            frames=_movie_frames(obs,args.max_frames)
            fig,update=build(int(frames[0]),group)
            animation=FuncAnimation(fig,lambda i:update(int(frames[i])),frames=len(frames),blit=False)
            products.append(_save_analysis_movie(animation,fig,output,stem+'_movie',args))
    fig,axes=plt.subplots(4,1,figsize=(13,15),constrained_layout=True)
    for ax,keys in zip(axes[:3],(('S_q','S_adv','S_rel','S_U'),tuple(f'T{i}' for i in range(1,9)),
                                ('residual_density','residual_source_operator','residual_source_expanded'))):
        for key in keys:ax.plot(times,[r['rms'][key] for r in rows],label=key)
        ax.legend(ncol=4,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
        ax.set_ylabel('Weighted RMS (a.u.)')
    axes[2].plot(times,[r['time_sampling_rms'] for r in rows],ls='--',label='time sampling')
    axes[2].legend(ncol=2,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
    axes[0].set_ylabel(r'Source RMS ($a_0^{-1}t_{au}^{-1}$)')
    axes[1].set_ylabel(r'Action RMS (Ha $a_0^{-1/2}$)')
    axes[2].set_ylabel(r'Residual RMS ($a_0^{-1}t_{au}^{-1}$)')
    for key in ('delta','b','alpha'):axes[3].plot(times,[r['rms'][key] for r in rows],label=key)
    axes[3].legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
    axes[3].set(xlabel='time (fs)',ylabel='Momentum RMS (a.u.)')
    for ax in axes:
        ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
    path=output/'proton_heavy_expanded_time_summary.png';_save_figure(fig,path,args.dpi);products.append(path)
    fig,axes=plt.subplots(2,3,figsize=(18,9),constrained_layout=True)
    peak_titles = dict(P_right=r'$P_{\mathrm{right}}(R,t)$',
        density_dt=r'$\partial_t P_{\mathrm{right}}$',
        S_q=r'$\int_{q>q_s} S_q\,dq$', S_adv=r'$\int_{q>q_s} S_{\mathrm{adv}}\,dq$',
        S_rel=r'$\int_{q>q_s} S_{\mathrm{rel}}\,dq$',
        residual_density=r'$\partial_t P_{\mathrm{right}}-\int_{q>q_s}(S_q+S_U)\,dq$')
    for f in snapshots:
        p=peaks[f];label=f'{times[f]:.2f} fs'
        for ax,key in zip(axes.flat,('P_right','density_dt','S_q','S_adv','S_rel','residual_density')):
            ax.plot(R,np.where(p['display_support_R'],p[key],np.nan),label=label)
            ax.set(title=peak_titles[key],xlabel=r'$R$ ($a_0$)',
                   ylabel='probability' if key=='P_right' else r'$t_{au}^{-1}$')
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
    axes[0,0].legend(ncol=2,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.01))
    fig.suptitle(f'Right-side conditional population and integrated sources | q > {config.q_split:g}',fontsize=15)
    path=output/'proton_heavy_peak_sources.png';_save_figure(fig,path,args.dpi);products.append(path)
    values.cache_clear()
    return products
