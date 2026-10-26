"""Static PEN zero/weak/strong spectra from existing reference caches only.

No propagation, eigensolve, gauge change or bare-to-polariton relabelling.
Pol k always denotes energy index k. Four bare products are reference guides,
not the full analysis basis or necessarily the four lowest uncoupled levels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .render import STYLE, REF_COLORS, _nice, _save
from .gi_render import _limits, HA_TO_MEV

CASES = ('zero', 'weak', 'strong')
PRODUCTS = ((0,0), (1,0), (0,1), (1,1))
PRODUCT_COLORS = ('#555555', '#0072B2', '#D55E00', '#999999')


def load_cases(root):
    root=Path(root); parent=root/'pen' if (root/'pen').is_dir() else root
    cases={}; notes=[]
    for name in CASES:
        folder=parent/name
        if not (folder/'references.npz').exists() or not (folder/'config.json').exists():
            notes.append(name+': reference/config unavailable');continue
        c=json.loads((folder/'config.json').read_text())
        if c.get('campaign')!='pen':
            raise ValueError('PEN-only surface comparison: '+str(folder))
        with np.load(folder/'references.npz',allow_pickle=False) as z:
            if not all(k in z.files for k in ('R','bo_energy','pol_energy')):
                notes.append(name+': missing energy arrays');continue
            d={k:np.asarray(z[k],dtype=float) for k in ('R','bo_energy','pol_energy')}
            d['pair']=np.array(z['pol_pair_indices'],dtype=int) if 'pol_pair_indices' in z.files else None
        R=d['R']; bo=d['bo_energy']; pol=d['pol_energy']
        if (R.ndim!=1 or len(R)<2 or np.any(np.diff(R)<=0) or bo.ndim!=2 or pol.ndim!=2
                or bo.shape[0]!=len(R) or pol.shape[0]!=len(R) or bo.shape[1]<2 or pol.shape[1]<4
                or not np.all(np.isfinite(R)) or not np.all(np.isfinite(bo)) or not np.all(np.isfinite(pol))):
            raise ValueError('Malformed/nonfinite static reference: '+str(folder))
        d.update(config=c,path=str(folder),omega=float(c['omega']),g=float(c['g_cavity']),R0=float(c['R0']))
        if not np.all(np.isfinite([d['omega'],d['g'],d['R0']])) or d['omega']<=0:
            raise ValueError('Invalid frequency/coupling/R0: '+str(folder))
        if not (R[0]<=d['R0']<=R[-1]):raise ValueError('R0 outside reference grid')
        d['bare']=np.column_stack([bo[:,j]+(n+.5)*d['omega'] for j,n in PRODUCTS])
        d['anchor']=int(np.argmin(abs(R-d['R0'])))
        cases[name]=d
    if not cases:return cases,notes
    first=next(iter(cases.values()))
    for name,d in cases.items():
        # A coupling-only comparison must not silently combine different models,
        # energy zeros, resonance frequencies, or analysis truncations.
        if (d['R'].shape!=first['R'].shape or not np.allclose(d['R'],first['R'],atol=1e-12,rtol=0)
                or d['bo_energy'].shape!=first['bo_energy'].shape
                or not np.allclose(d['bo_energy'],first['bo_energy'],atol=1e-9,rtol=0)
                or not np.isclose(d['omega'],first['omega'],atol=1e-12,rtol=0)
                or not np.isclose(d['R0'],first['R0'],atol=1e-12,rtol=0)):
            raise ValueError('Incompatible grids/bare surfaces/omega/R0: '+name)
        for key in ('n_bo','n_fock_analysis','dse'):
            if d['config'].get(key)!=first['config'].get(key):
                raise ValueError('Incompatible '+key+': '+name)
    return cases,notes


def _header(name,d):
    return (name.capitalize()+fr' | $g_{{\mathrm{{cavity}}}}={d["g"]:.7g}$ a.u.'
            +'\n'+fr'$\omega_c={d["omega"]:.7g}$ Ha; DSE '+('on' if d['config'].get('dse',True) else 'off'))


def surface_figure(cases,zoom_half_width=1.5):
    fig,axes=plt.subplots(2,3,figsize=(26,14))
    fig.subplots_adjust(left=.065,right=.98,bottom=.18,top=.85,hspace=.47,wspace=.23)
    if not cases:raise ValueError('No usable PEN reference caches')
    first=next(iter(cases.values()));R=first['R'];R0=first['R0']
    bounds=(max(float(R[0]),R0-zoom_half_width),min(float(R[-1]),R0+zoom_half_width))
    local=(R>=bounds[0])&(R<=bounds[1])
    if not np.any(local):local[first['anchor']]=True
    limits=[_limits([d[key][:,:4] for d in cases.values() for key in ('pol_energy','bare')]),
            _limits([d[key][local,:4] for d in cases.values() for key in ('pol_energy','bare')])]
    for col,name in enumerate(CASES):
        d=cases.get(name)
        for row in range(2):
            ax=axes[row,col]
            if d:
                for k in range(4):
                    ax.plot(R,d['pol_energy'][:,k],color=REF_COLORS[k],lw=2.5,label=fr'$|\mathrm{{Pol}}_{k}\rangle$')
                for k,(j,n) in enumerate(PRODUCTS):
                    ax.plot(R,d['bare'][:,k],color=PRODUCT_COLORS[k],ls='--',lw=1.3,alpha=.8,
                            label=fr'$|S_{j},{n}\rangle$ bare')
            else:ax.text(.5,.5,name+': cache unavailable',ha='center',transform=ax.transAxes)
            ax.axvline(R0,color='.4',ls=':',lw=1.4)
            ax.set_xlim((R[0],R[-1]) if row==0 else bounds)
            _nice(ax,r'$R$ ($a_0$)','Energy (Ha)',limits[row])
            ax.set_title(_header(name,d) if row==0 and d else ('Initial-region zoom' if row==1 else name))
    handles,labels=axes[0,CASES.index(next(iter(cases)))].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.065),frameon=False,fontsize=13)
    fig.suptitle('PEN | four polaritonic energy surfaces: zero / weak / strong',fontsize=24)
    fig.text(.5,.025,r'Solid: cached fixed-$R$ electron–photon eigenvalues. Dashed: $E_{S_j}^{\rm BO}(R)+(n+1/2)\omega_c$. Dotted: initial $R_0$.',ha='center',fontsize=12)
    fig.text(.5,.005,'Common axes and absolute energies. Four bare guides do not truncate the TDSE or imply a one-to-one state assignment.',ha='center',fontsize=11)
    return fig,dict(full_energy_Ha=limits[0],zoom_energy_Ha=limits[1],zoom_R=bounds)


def pair_figure(cases,zoom_half_width=.6):
    """Fixed energy indices selected by cached product-character tracking at R0.

    A common R-dependent bare pair midpoint is subtracted from all curves in
    every case. This preserves splittings and coupling-induced common shifts.
    Never infer LP/UP from energy order alone.
    """
    first=next(iter(cases.values()));R=first['R'];R0=first['R0']
    bounds=(max(float(R[0]),R0-zoom_half_width),min(float(R[-1]),R0+zoom_half_width))
    mask=(R>=bounds[0])&(R<=bounds[1])
    if not np.any(mask):mask[first['anchor']]=True
    baseline=.5*(first['bare'][:,1]+first['bare'][:,2])
    fig,axes=plt.subplots(1,3,figsize=(26,8))
    fig.subplots_adjust(left=.07,right=.98,bottom=.28,top=.78,wspace=.23)
    selected={};values=[];details={}
    for name,d in cases.items():
        pairs=d['pair']; i=d['anchor']
        ids=sorted(set(int(k) for k in pairs[i])) if pairs is not None and pairs.shape==(len(R),2) else []
        if len(ids)!=2 or min(ids)<0 or max(ids)>=d['pol_energy'].shape[1]:
            details[name]={'unavailable':'Tracked pair outside saved references or missing'};continue
        selected[name]=ids
        values.append((d['pol_energy'][mask][:,ids]-baseline[mask,None])*HA_TO_MEV)
        details[name]=dict(energy_indices=ids,R_sample=float(R[i]),
            splitting_meV=float((d['pol_energy'][i,ids[1]]-d['pol_energy'][i,ids[0]])*HA_TO_MEV))
    values.append((first['bare'][mask,1:3]-baseline[mask,None])*HA_TO_MEV)
    limits=_limits(values)
    for col,name in enumerate(CASES):
        ax=axes[col];d=cases.get(name)
        if d:
            ax.set_title(_header(name,d))
            for k in selected.get(name,[]):
                ax.plot(R,(d['pol_energy'][:,k]-baseline)*HA_TO_MEV,color=REF_COLORS[k%4],lw=2.6,
                        label=fr'$|\mathrm{{Pol}}_{k}\rangle$')
            for k in (1,2):
                j,n=PRODUCTS[k]
                ax.plot(R,(d['bare'][:,k]-baseline)*HA_TO_MEV,color=PRODUCT_COLORS[k],ls='--',lw=1.4,
                        label=fr'$|S_{j},{n}\rangle$ bare')
            info=details[name]
            if name in selected:
                note=fr'$\Delta E={info["splitting_meV"]:.3g}$ meV at $R={info["R_sample"]:.3f}$'
            else:note=info['unavailable']
            ax.text(.03,.96,note,transform=ax.transAxes,va='top',fontsize=12,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False,fontsize=12)
        else:ax.text(.5,.5,name+': cache unavailable',ha='center',transform=ax.transAxes)
        ax.axvline(R0,color='.4',ls=':',lw=1.2);ax.axhline(0,color='.6',ls=':',lw=.8)
        ax.set_xlim(bounds);_nice(ax,r'$R$ ($a_0$)','Relative energy (meV)',limits)
    fig.suptitle('PEN | resonant-product pair: common-scale magnification',fontsize=24)
    fig.text(.5,.04,'Same bare-pair midpoint subtracted in all cases. Not a gauge transformation; splitting is measured at the stated sampled R, not a fitted resonance.',ha='center',fontsize=12)
    fig.text(.5,.01,'Pair indices use cached product-character tracking at the initial region and stay fixed along R; they need not be LP/UP everywhere.',ha='center',fontsize=11)
    return fig,dict(energy_difference_meV=limits,zoom_R=bounds,pairs=details)


def render_surfaces(root,output_dir=None,zoom_half_width=1.5,dpi=150):
    if not np.isfinite(zoom_half_width) or zoom_half_width<=0:raise ValueError('zoom_half_width must be positive')
    root=Path(root);out=Path(output_dir) if output_dir else root/'polaritonic_surfaces'
    cases,notes=load_cases(root)
    if not cases:return dict(outputs=[],notes=notes)
    out.mkdir(parents=True,exist_ok=True);outputs=[]
    with plt.rc_context(STYLE):
        fig,overview=surface_figure(cases,zoom_half_width)
        _save(fig,out/'PEN_polaritonic_surfaces',dpi);plt.close(fig)
        fig,pair=pair_figure(cases,min(.6,zoom_half_width))
        _save(fig,out/'PEN_polaritonic_pair_zoom',dpi);plt.close(fig)
    for name in ('PEN_polaritonic_surfaces','PEN_polaritonic_pair_zoom'):
        outputs.extend(str(out/(name+ext)) for ext in ('.png','.pdf'))
    manifest=dict(outputs=outputs,notes=notes,cases={name:dict(source=d['path'],g_cavity=d['g'],omega_Ha=d['omega'],
        n_BO=d['config'].get('n_bo'),n_Fock=d['config'].get('n_fock_analysis'),dse=d['config'].get('dse',True)) for name,d in cases.items()},
        overview=overview,pair_zoom=pair,
        coupling_definition='Atomic units; H_LM=omega_c*g_cavity*mu*q_c, H_DSE=g_cavity^2*mu^2/2. g_cavity is NOT the Rabi splitting.',
        scope='Static cached projected electron-photon references; no TDSE/eigensolve, no independent convergence certification.')
    (out/'surface_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Campaign root containing pen/zero, pen/weak, pen/strong (or pen directory)')
    p.add_argument('--out',type=Path);p.add_argument('--zoom-half-width',type=float,default=1.5)
    p.add_argument('--dpi',type=int,default=150)
    a=p.parse_args();print(json.dumps(render_surfaces(a.root,a.out,a.zoom_half_width,a.dpi),indent=2))


if __name__=='__main__':main()
