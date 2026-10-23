"""Code-native explanatory figures, explicitly separated from numerical data.

These reduced diagrams explain resonance and product-state character. They
are NOT four-state approximations used by the real-space TDSE and are NOT
measured energy surfaces. At exact resonance weak coupling already gives
equal two-state mixing; its splitting, not necessarily its mixing, is small.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def _save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)
    return [str(path.with_suffix('.png')),str(path.with_suffix('.pdf'))]


def _calibration_note(calibration):
    if not calibration:
        return 'Frequency and coupling are calibrated by the molecular model; diagram energies are dimensionless.'
    notes=[]
    for campaign in ('pen','vsc'):
        key='omega_'+campaign
        value=calibration.get(key,calibration.get(campaign,{}).get('omega_bare'))
        if value is not None:
            try: notes.append(campaign.upper()+f': bare resonance {float(value):.5g} Ha')
            except (ValueError,TypeError): pass
    return '; '.join(notes) or 'Bare-gap calibration is stored in the campaign metadata; schematic is not production data.'


def resonance_figure(calibration=None):
    fig,axes=plt.subplots(1,2,figsize=(15,6.7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.07,right=.97,bottom=.20,top=.79,wspace=.20)
    coordinate=np.linspace(-2.5,2.5,600)
    excited=.35*(coordinate+.6)**2+1
    ground=.35*(coordinate-.6)**2
    for ax,omega,title in zip(axes,(1.7,1.0),('Off-resonant near preparation','Resonant near preparation')):
        dressed=ground+omega
        coupling=.12
        lower=.5*(excited+dressed)-np.sqrt(.25*(excited-dressed)**2+coupling**2)
        upper=.5*(excited+dressed)+np.sqrt(.25*(excited-dressed)**2+coupling**2)
        ax.plot(coordinate,excited,'--',color='#0072B2',lw=1.7,label=r'$|S_1,0\rangle$ bare')
        ax.plot(coordinate,dressed,'--',color='#D55E00',lw=1.7,label=r'$|S_0,1\rangle$ dressed')
        ax.plot(coordinate,lower,color='#009E73',lw=2.5,label='LP-like eigenvalue')
        ax.plot(coordinate,upper,color='#7B3294',lw=2.5,label='UP-like eigenvalue')
        crossing=coordinate[np.argmin(abs(excited-dressed))]
        ax.axvline(0,color='0.65',ls=':',lw=1.2)
        ax.annotate('avoided crossing',xy=(crossing,np.interp(crossing,coordinate,lower)),
                    xytext=(.55,3.1),arrowprops=dict(arrowstyle='->',color='.3'),fontsize=12)
        ax.set_title(title,fontsize=17,pad=12)
        ax.set_xlabel('Nuclear coordinate relative to preparation (schematic)',fontsize=12)
        ax.set_xlim(-2.5,2.5); ax.set_ylim(.65,4)
        ax.text(.04,.94,fr'$\omega/\Delta_{{el}}={omega:.1f}$',transform=ax.transAxes,fontsize=13,va='top')
        ax.spines[['right','top']].set_visible(False)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel('Relative energy / bare electronic gap',fontsize=14)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.075),frameon=False,fontsize=12)
    fig.suptitle('S1 | Bare-gap frequency tuning and avoided crossing',fontsize=20,y=.96)
    fig.text(.5,.89,'Explanatory diagram — full TDSE retains the real-space electron and all resolved photon amplitudes of one cavity mode.',ha='center',fontsize=10.5)
    fig.text(.5,.025,_calibration_note(calibration),ha='center',fontsize=10)
    return fig


def correlation_figure(campaign='pen',calibration=None):
    fig,axes=plt.subplots(1,3,figsize=(17,7.5),sharey=True)
    fig.subplots_adjust(left=.08,right=.97,bottom=.23,top=.79,wspace=.17)
    electronic=campaign=='pen'
    zero=r'$|S_0,0\rangle$' if electronic else r'$|v_0,0\rangle$'
    matter=r'$|S_1,0\rangle$' if electronic else r'$|v_1,0\rangle$'
    photon=r'$|S_0,1\rangle$' if electronic else r'$|v_0,1\rangle$'
    both=r'$|S_1,1\rangle$' if electronic else r'$|v_1,1\rangle$'
    measurements=(calibration or {}).get(campaign,{}).get('coupling',{}).get('measurements',{})
    splits=[0.]+[float(measurements.get(case,{}).get('measured_splitting_over_omega',fallback))
                 for case,fallback in (('weak',.01),('strong',.1))]
    for ax,split,title in zip(axes,splits,('Cavity uncoupled','Weak splitting','Resolved splitting')):
        ax.set_xlim(0,1); ax.set_ylim(-.18,2.34)
        ax.set_xticks([]); ax.set_yticks([0,1,2])
        ax.spines[['right','top','bottom']].set_visible(False)
        ax.tick_params(labelsize=13)
        ax.set_title(title,fontsize=18,pad=13)
        ax.hlines(0,.15,.85,color='#3F3F3F',lw=3)
        ax.text(.5,.08,zero,ha='center',fontsize=16)
        if split==0:
            ax.hlines(1,.10,.44,color='#0072B2',lw=3)
            ax.hlines(1,.56,.90,color='#D55E00',lw=3)
            ax.text(.26,1.10,matter,ha='center',fontsize=15)
            ax.text(.74,1.10,photon,ha='center',fontsize=15)
            ax.text(.5,.78,'degenerate product states\nno coupling',ha='center',fontsize=12,color='.35')
        else:
            ax.hlines(1-split/2,.14,.86,color='#009E73',lw=1.2 if split<.04 else 3)
            ax.hlines(1+split/2,.14,.86,color='#7B3294',lw=1.2 if split<.04 else 3)
            ax.text(.5,1-split/2-.14,'LP: mixed matter + photon',ha='center',fontsize=12,color='#007C5A')
            ax.text(.5,1+split/2+.10,'UP: mixed matter + photon',ha='center',fontsize=12,color='#7B3294')
            ax.annotate('',xy=(.94,1+split/2),xytext=(.94,1-split/2),arrowprops=dict(arrowstyle='<->',color='.25'))
            prefix='calibrated gap' if measurements else 'target gap'
            ax.text(.5,.45,prefix+fr' $\Omega/\omega={split:.3g}$',ha='center',fontsize=11)
        ax.hlines(2,.15,.85,color='0.4',lw=2,linestyles='--')
        ax.text(.5,2.08,both+' character',ha='center',fontsize=15)
        ax.text(.5,1.79,'higher excitation manifold\n(not generally one pure eigenstate)',ha='center',fontsize=10,color='.35')
    axes[0].set_ylabel('Excitation energy / bare resonant gap',fontsize=14)
    number='S2' if electronic else 'S3'
    subject='electronic–photon' if electronic else 'vibrational–photon'
    fig.suptitle(number+' | '+subject+' state correlation (schematic)',fontsize=22,y=.96)
    fig.text(.5,.88,'Resonance is chosen from the bare transition, before coupling; LP/UP splitting is an output.',ha='center',fontsize=12)
    fig.text(.5,.14,'At exact two-level resonance both nonzero couplings give equal mixing; weak coupling does not imply weak mixing.',ha='center',fontsize=12)
    fig.text(.5,.09,'Energy centers are schematic (DSE/common shifts not shown). These highlighted states do not truncate the real-space TDSE.',ha='center',fontsize=11)
    fig.text(.5,.035,_calibration_note(calibration),ha='center',fontsize=10)
    return fig


def render_schematics(output_dir,calibration=None):
    directory=Path(output_dir); directory.mkdir(parents=True,exist_ok=True)
    outputs=[]
    with plt.rc_context({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'}):
        outputs+=_save(resonance_figure(calibration),directory/'S1_resonance_tuning')
        outputs+=_save(correlation_figure('pen',calibration),directory/'S2_PEN_state_correlation')
        outputs+=_save(correlation_figure('vsc',calibration),directory/'S3_VSC_state_correlation')
    manifest=dict(outputs=outputs,kind='explanatory schematic, not numerical production results',
                  note='Electronic fixed-R polaritonic PES and global vibrational polariton energies are different objects.',
                  warning='Bare |v1,1> or |S1,1> character is not generally an exact fourth eigenstate.')
    (directory/'schematic_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output_dir')
    a=p.parse_args(); print(json.dumps(render_schematics(a.output_dir),indent=2))
