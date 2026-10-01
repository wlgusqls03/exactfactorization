"""Saved free-sector TDSE -> nested EF -> matched-time comparison MP4 only.

No propagation or time interpolation. Scalars are separately density-mean
aligned on displayed support; forces are NOT inferred from scalar slopes.
Original free/coupled grids and photon frequencies are retained and disclosed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FFMpegWriter
from .free_vacuum_fields import analyze_vacuum
from .vsc_movie_only import read_frame,compact_fields,supports,AU_FS,HA_EV


def prepare(packet_path,waves,coupled,out):
    """Validate hashes, native grids and saved marginals before writing fields."""
    sha=hashlib.sha256(packet_path.read_bytes()).hexdigest()
    status=json.loads((waves/'status.json').read_text())
    if status['input_sha256']!=sha:raise ValueError('Free input hash mismatch')
    with np.load(packet_path,allow_pickle=False) as z:p={k:z[k] for k in z.files}
    if float(p['g_chi'])!=0:raise ValueError('Not uncoupled')
    paths=sorted(waves.glob('wave_*.npz'))
    if not paths:raise ValueError('No saved free waves')
    for path in paths:
        if not (coupled/path.name.replace('wave_','fields_')).is_file():raise ValueError('Missing matched coupled frame')
    out.mkdir(parents=True,exist_ok=False);(out/'free_fields').mkdir()
    rows=[];pairs=[]
    for path in paths:
        cp=coupled/path.name.replace('wave_','fields_');c=read_frame(cp)
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['R'],p['R']);np.testing.assert_array_equal(z['x'],p['x'])
            u=z['psi'];t=float(z['time_au'])
            if u.shape!=p['psi'].shape or u.shape[-1]!=1:raise ValueError('Not vacuum-sector wave')
            if abs(t-float(c['time_au']))>1e-10:raise ValueError('Unmatched physical times')
            f=analyze_vacuum(u[:,:,0],p,c['Q']);f['time_au']=t
        with np.load(waves/path.name.replace('wave_','observable_')) as z:
            np.testing.assert_allclose(f['rho_R'],z['rho_R'],atol=2e-12,rtol=2e-12)
            np.testing.assert_allclose(f['current'],z['current'],atol=2e-12,rtol=2e-12)
        # Probability weights, not max tail errors, for independent route checks.
        mask=f['rho_R']>1e-8*f['rho_R'].max();weights=f['rho_R'][mask]
        rms=lambda v:float(np.sqrt(np.average(abs(v[mask])**2,weights=weights)))
        row=dict(time_fs=t*AU_FS,norm=float(f['rho_R'].sum()*float(p['dR'])),
            reconstruction_L2=f['reconstruction_L2'],epsilon2_route_rms=rms(f['epsilon2_A']-f['epsilon2_B']),
            epsilon2_imag_rms=rms(f['epsilon2_A'].imag),PNC_electronic_rms=rms(f['electronic_PNC_error']),
            PNC_photon_max=float(f['photon_PNC_error'].max()))
        if max(row[k] for k in ('reconstruction_L2','epsilon2_route_rms','epsilon2_imag_rms','PNC_electronic_rms','PNC_photon_max'))>1e-8:
            raise ValueError(f'Free postprocessing residual failed: {row}')
        dest=out/'free_fields'/cp.name
        np.savez(dest,**compact_fields(f),alpha_t=f['alpha_t'],epsilon2_R=f['epsilon2_R'])
        rows.append(row);pairs.append((dest,cp))
    return pairs,dict(input_sha256=sha,free_omega_au=float(p['omega']),free_R_points=len(p['R']),
        frames=rows,phase7_pass=False,status='POSTPROCESSING_DIAGNOSTIC',
        note='Algebraic identities pass; not a new spatial/time convergence certificate')


def context(pairs,floor):
    """Common Q-density and R-density peaks; original Q density Jacobians."""
    peak=0.;peakR=0.;qlo=np.inf;qhi=-np.inf
    for pair in pairs:
        for path in pair:
            f=read_frame(path);s=(f['Q'][1]-f['Q'][0])/(f['q'][1]-f['q'][0])
            peak=max(peak,float(f['rho_qR'].max()/s));peakR=max(peakR,float(f['rho_R'].max()))
    for pair in pairs:
        for path in pair:
            f=read_frame(path);s=(f['Q'][1]-f['Q'][0])/(f['q'][1]-f['q'][0])
            ids=np.flatnonzero((f['rho_qR']/s>=floor*peak).any(axis=0))
            qlo=min(qlo,float(f['Q'][ids[0]]));qhi=max(qhi,float(f['Q'][ids[-1]]))
    return dict(peakQ=peak,peakR=peakR,floor=floor,Qlim=(qlo-.5,qhi+.5))


def displayed(f,c):
    s=(f['Q'][1]-f['Q'][0])/(f['q'][1]-f['q'][0])
    mr,mj=supports(f,c['floor'],c['peakQ']*s,1e-8,c['peakR'])
    e1=(f['epsilon1_A'].real-np.average(f['epsilon1_A'].real[mj],weights=f['rho_qR'][mj]))*HA_EV
    e2=(f['epsilon2_A'].real-np.average(f['epsilon2_A'].real[mr],weights=f['rho_R'][mr]))*HA_EV
    return mr,mj,e1,e2,s


def line(ax,R,v,mask,color,limit,label):
    ax.plot(R,np.where(mask,v,np.nan),color=color,label=label,lw=1.6)
    for sign,marker in ((1,'^'),(-1,'v')):
        clip=mask&(sign*v>limit)
        ax.scatter(R[clip],np.full(clip.sum(),sign*limit),s=10,marker=marker,color=color)
    ax.set_ylim(-limit*1.07,limit*1.07)


def heat(fig,ax,f,c,z,mask,limit,title,unit):
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e4e4e4')
    im=ax.pcolormesh(f['R'],f['Q'],np.ma.array(z,mask=~mask).T,shading='auto',
        cmap=cmap,norm=Normalize(-limit,limit))
    s=(f['Q'][1]-f['Q'][0])/(f['q'][1]-f['q'][0]);rel=f['rho_qR']/s/c['peakQ']
    levels=np.array([c['floor'],1e-3,1e-2,1e-1]);levels=np.unique(levels[(levels>rel.min())&(levels<rel.max())])
    if len(levels):ax.contour(f['R'],f['Q'],rel.T,levels=levels,colors='#42645c',linewidths=.65)
    ax.set(title=title,xlabel=r'$R$ ($a_0$)',ylabel=r'$Q=\sqrt{\omega_c}q_c$',xlim=(-4.4,4.4),ylim=c['Qlim'])
    ax.axvline(0,color='0.5',ls=':',lw=.6)
    fig.colorbar(im,ax=ax,shrink=.8,pad=.02,extend='both',label=unit+' (linear zoom)')


def render(pairs,out,report,floor=1e-4,fps=6):
    """Three fixed-scale movie families, actual matched snapshots only."""
    c=context(pairs,floor);records=[]
    for fp,cp in pairs:
        row=[]
        for path in (fp,cp):
            f=read_frame(path);mr,mj,e1,e2,s=displayed(f,c)
            w=f['rho_qR'];r=f['rho_R']
            row.append(dict(epsilon1_offset_Ha=float(np.average(f['epsilon1_A'].real[mj],weights=w[mj])),
                epsilon2_offset_Ha=float(np.average(f['epsilon2_A'].real[mr],weights=r[mr])),
                epsilon1_saturated_displayed_probability=float(w[mj&(abs(e1)>2)].sum()/w[mj].sum()),
                force_saturated_displayed_probability=float(r[mr&(abs(f['force'])>.3)].sum()/r[mr].sum())))
        records.append(row)
    report.update(display=c,display_records_free_then_coupled=records,fps=fps,
        alignment='Separate density-weighted spatial mean on displayed support; not activation barriers',
        original_free_cavity_meV=report['free_omega_au']*27211.386245988,
        coupled_cavity_meV=161.768922,cadence=f'{len(pairs)} actual snapshots; mostly 0.774 fs; no time interpolation',
        limits=dict(epsilon1_eV=2,epsilon2_eV=2,force_au=.3,alpha_au=30,a_au=.5,b_au=30))
    (out/'comparison_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    colors=('#546e7a','#c05032');names=('Uncoupled','Coupled')
    for family in ('nuclear','epsilon1','connections'):
        fig=plt.figure(figsize=(13,8) if family!='epsilon1' else (12,5.5),layout='constrained')
        writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-crf','18','-pix_fmt','yuv420p',
            '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        with writer.saving(fig,str(out/f'mcef_free_coupled_{family}.mp4'),110):
            for i,pair in enumerate(pairs):
                fs=[read_frame(p) for p in pair];ds=[displayed(f,c) for f in fs];fig.clear()
                if family=='nuclear':
                    axs=fig.subplots(2,2)
                    for f,d,color,name in zip(fs,ds,colors,names):
                        mr,mj,e1,e2,s=d;R=f['R']
                        axs[0,0].plot(R,f['rho_R'],color=color,label=name,lw=1.7)
                        line(axs[0,1],R,f['force'],mr,color,.3,name)
                        line(axs[1,0],R,e2,mr,color,2,name)
                        line(axs[1,1],R,f['alpha'],mr,color,30,name)
                        for ax in (axs[0,1],axs[1,0],axs[1,1]):
                            ax.fill_between(R,0,.1*f['rho_R']/c['peakR'],transform=ax.get_xaxis_transform(),color=color,alpha=.12)
                    for ax in axs.flat:
                        ax.set(xlim=(-4.4,4.4),xlabel=r'$R$ ($a_0$)');ax.axvline(0,color='0.5',ls=':',lw=.6)
                    axs[0,0].set(title=r'Nuclear density $|\chi|^2$',ylabel=r'$a_0^{-1}$',ylim=(0,1.05*c['peakR']))
                    axs[0,0].legend()
                    axs[0,1].set(title=r'Total force $-\partial_R\epsilon^{(2)}+\partial_t\alpha$',ylabel='Force (a.u.; zoom)')
                    axs[1,0].set(title=r'$\epsilon^{(2)}$ (mean aligned; NOT force)',ylabel='Energy (eV; zoom)')
                    axs[1,1].set(title=r'$\alpha$: nuclear mechanical momentum',ylabel='Momentum (a.u.; zoom)')
                elif family=='epsilon1':
                    axs=fig.subplots(1,2)
                    for ax,f,d,name in zip(axs,fs,ds,names):heat(fig,ax,f,c,d[2],d[1],2,name+r': $\epsilon^{(1)}$','Mean-aligned eV')
                else:
                    axs=fig.subplots(2,3)
                    for row,f,d,name in zip(axs,fs,ds,names):
                        for ax,z,lim,title in zip(row,(f['a'],f['b'],f['b']-f['alpha'][:,None]),(.5,30,30),('a','b',r'b-\alpha')):
                            heat(fig,ax,f,c,z,d[1],lim,name+': $'+title+'$','Momentum (a.u.)')
                fig.suptitle(f'Positive-marginal MCEF | t={float(fs[0]["time_au"])*AU_FS:.3f} fs\n'
                    'Free cavity 170.6 meV (decoupled) / coupled cavity 161.769 meV; native grids differ',fontsize=12)
                fig.supxlabel('ACTUAL SNAPSHOTS (not dense replay). Shared density mask 1e-4. Triangles / colorbar arrows mark zoom overflow.\n'
                    'Diagnostic fields: algebraic agreement is not spatial convergence. Scalars separately mean aligned.',fontsize=9)
                writer.grab_frame()
                if i%20==0:print(f'{family} {i+1}/{len(pairs)}',flush=True)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','waves','coupled-fields','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('New output required')
    pairs,report=prepare(a.packet,a.waves,a.coupled_fields,a.out)
    render(pairs,a.out,report)


if __name__=='__main__':main()
