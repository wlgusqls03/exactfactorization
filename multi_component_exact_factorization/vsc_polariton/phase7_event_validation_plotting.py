"""Diagnostic plots for actual Step1/2 failures, not production potentials."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    base=Path('results/vsc_polariton/phase7/event_validation_v1')
    extension=Path('results/vsc_polariton/phase7/event_gauge_extension_v1/validation.json')
    out=base/'figures';out.mkdir(exist_ok=False)
    records=json.loads((base/'validation.json').read_text())['frames']
    extra=json.loads(extension.read_text())['frames'] if extension.exists() else {}
    plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12})
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained',sharex=True,sharey=True)
    for case,ax in zip(('free','resonant','barrier'),axes):
        for name,record in records.items():
            if not name.startswith(case+'_'):continue
            values={int(f):d['1e-08']['force_fd6_error'] for f,d in record['step2'].items()}
            if name in extra:
                values.update({int(f):d['1e-08']['force_fd6_error'] for f,d in extra[name].items() if f.isdigit()})
            x=sorted(values);ax.loglog(x,[values[k] for k in x],'-o',ms=4,label=f'{record["time_au"]*.024188843265857:.2f} fs')
        ax.axhline(1e-5,color='black',ls='--',lw=1)
        ax.set(title=case,xlabel='Postprocessing R sampling factor',xlim=(1.8,2400),ylim=(1e-13,1e3))
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,fontsize=10)
        ax.grid(alpha=.15,which='major')
    axes[0].set_ylabel('Force consistency RMS (Ha / $a_0$)')
    fig.suptitle('Same saved wave — independent gauge-force check | dashed line: fixed tolerance')
    for ext in ('png','pdf'):fig.savefig(out/f'force_resolution.{ext}',dpi=170)
    plt.close(fig)
    picks=[('free_wave_0006912',64),('resonant_wave_0011264',208),('barrier_wave_0011520',208)]
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained',sharex=True,sharey=True)
    for j,(name,nq) in enumerate(picks):
        with np.load(base/f'{name}_Nq{nq}.npz') as z:
            r=z['R'];q=z['q'];rho=z['rho_qR'];mask=z['mask_1e-08']
            err=abs(z['epsilon1_A']-z['epsilon1_B']);tau=float(z['time_au'])*.024188843265857
        for i,(value,norm,cmap) in enumerate(((rho,LogNorm(1e-7,.1),'magma'),(np.maximum(err,1e-16),LogNorm(1e-12,1e-3),'viridis'))):
            image=axes[i,j].pcolormesh(q,r,np.ma.masked_where(~mask,value),norm=norm,cmap=cmap,shading='auto')
            axes[i,j].axhline(0,color='white',ls='--',lw=.8)
            axes[i,j].set(xlim=(-220,220),ylim=(-3.5,3.5),xlabel=r'$q_c$ (a.u.)',ylabel=r'$R$ ($a_0$)')
            if i==0:axes[i,j].set_title(f'{name.split("_")[0]} | {tau:.2f} fs')
            if j==2:fig.colorbar(image,ax=axes[i,:],extend='both',label='Joint density (a.u.)' if i==0 else r'$|\epsilon^{(1)}_A-\epsilon^{(1)}_B|$ (Ha)')
    fig.suptitle('Near maximum product population: density and scalar discrepancy\nDifferent event times; no physical-potential smoothing')
    for ext in ('png','pdf'):fig.savefig(out/f'scalar_error_locations.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
