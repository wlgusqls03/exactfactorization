"""Sparse saved-wave QHJ force balance, free/coupled, no TDSE replay.

Both marginal and photon-conditioned material momenta are diagnosed.
This is algebraic postprocessing, not Phase7 or derivative-convergence PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from .qhj_saved_wave import outer_qhj,joint_qhj
from .real_grid_mcef_fields import action
from .run_real_grid_mcef_preview import load_wave
from .phase6_gpu_backend import PFBackend
from .vsc_movie_only import AU_FS


def rms(v,w,m):return float(np.sqrt(np.average(abs(v[m])**2,weights=w[m])))


def calculate(args):
    with np.load(args.coupled_packet) as z:pc={k:z[k] for k in z.files}
    with np.load(args.free_packet) as z:pf={k:z[k] for k in z.files}
    sc=hashlib.sha256(args.coupled_packet.read_bytes()).hexdigest()
    sf=hashlib.sha256(args.free_packet.read_bytes()).hexdigest()
    if json.loads((args.free_waves/'status.json').read_text())['input_sha256']!=sf:raise ValueError('Free SHA mismatch')
    if float(pf['g_chi'])!=0:raise ValueError('Free packet is coupled')
    backend=PFBackend(pf,float(pf['dt']))
    paths=sorted(args.coupled_waves.glob('wave_*.npz'))
    if not paths:raise ValueError('No coupled waves')
    args.out.mkdir(parents=True,exist_ok=False);records=[];outputs=[]
    for path in paths:
        uQ,Q,t=load_wave(path,pc,sc);w=float(pc['omega']);q=Q/np.sqrt(w);dq=q[1]-q[0]
        u=uQ*w**.25;del uQ
        H=action(u,pc,q)
        c=outer_qhj(u,H,float(pc['dR']),float(pc['dx'])*dq,float(pc['mass']))
        with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
            cj=joint_qhj(u,H,float(pc['dR']),float(pc['dx']),dq,float(pc['mass']))
        del u,H
        with np.load(args.free_waves/path.name) as z:
            if abs(float(z['time_au'])-t)>1e-10:raise ValueError('Time mismatch')
            np.testing.assert_array_equal(z['R'],pf['R']);np.testing.assert_array_equal(z['x'],pf['x'])
            uf=z['psi'];f=outer_qhj(uf,backend.action(uf),float(pf['dR']),float(pf['dx']),float(pf['mass']))
        del uf
        wf=float(pf['omega']);qf=Q/np.sqrt(wf);vac=np.sqrt(wf/np.pi)*np.exp(-Q**2)
        jointf=f['rho_R'][:,None]*vac
        fj=dict(rho_qR=jointf,photon_scalar_force=np.broadcast_to(-wf**2*qf,jointf.shape),
            photon_quantum_force=np.broadcast_to(wf**2*qf,jointf.shape),
            photon_net_force=np.zeros_like(jointf),photon_material=np.zeros_like(jointf),
            photon_time_force=np.zeros_like(jointf),photon_berry_force=np.zeros_like(jointf))
        with np.load(args.coupled_fields/path.name.replace('wave_','fields_')) as old:
            m=c['rho_R']>1e-4*c['rho_R'].max();mj=cj['rho_qR']>1e-4*cj['rho_qR'].max()
            check=dict(outer_existing_force=rms(c['EF_force']-old['force'],c['rho_R'],m),
                epsilon1_existing=rms(cj['epsilon1_QHJ']-old['epsilon1_A'].real,cj['rho_qR'],mj))
        checks={}
        for name,outer in [('free',f),('coupled',c)]:
            m=outer['rho_R']>1e-4*outer['rho_R'].max()
            checks[name]=dict(outer_balance=rms(outer['balance_residual'],outer['rho_R'],m),
                HJ_identity=rms(outer['QHJ_residual'],outer['rho_R'],m),
                EF_force_rms=rms(outer['EF_force'],outer['rho_R'],m),
                quantum_force_rms=rms(outer['quantum_force'],outer['rho_R'],m),
                net_flow_force_rms=rms(outer['net_flow_force'],outer['rho_R'],m))
        checks['photon_balance']=rms(cj['photon_net_force']-cj['photon_material'],cj['rho_qR'],mj)
        checks['joint_nuclear_balance']=rms(cj['nuclear_net_force']-cj['nuclear_material'],cj['rho_qR'],mj)
        check.update(time_fs=t*AU_FS,checks=checks)
        if max(check['outer_existing_force'],check['epsilon1_existing'],checks['photon_balance'])>1e-8:
            raise ValueError(f'Identity mismatch; diagnostic preserved: {check}')
        dest=args.out/path.name.replace('wave_','qhj_')
        arrays=dict(R_free=pf['R'],R_coupled=pc['R'],Q=Q,time_au=t,
            omega_free=wf,omega_coupled=w)
        for prefix,values in [('free',f),('coupled',c),('free_joint',fj),('coupled_joint',cj)]:
            arrays.update({prefix+'_'+k:v for k,v in values.items()})
        np.savez(dest,**arrays);records.append(check);outputs.append(dest)
        print('QHJ',path.name,'t_fs',t*AU_FS,flush=True)
    report=dict(status='ALGEBRAIC_DIAGNOSTIC',phase7_pass=False,records=records,
        note='Scalar joint force uses wave-jet QHJ inversion, not independent differentiation of expectation scalar. No time interpolation.',
        input_hashes=dict(free=sf,coupled=sc))
    (args.out/'qhj_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return outputs


def render(paths,out):
    frames=[];peak=0.;rpeak=0.
    for path in paths:
        with np.load(path) as z:f={k:z[k] for k in z.files}
        for name in ('free','coupled'):
            peak=max(peak,float((f[name+'_joint_rho_qR']/np.sqrt(float(f['omega_'+name]))).max()))
            rpeak=max(rpeak,float(f[name+'_rho_R'].max()))
        frames.append(f)
    colors=('#b34b40','#327ba0','#222222')
    for family in ('outer','photon'):
        fig=plt.figure(figsize=(13,8),layout='constrained')
        writer=FFMpegWriter(fps=1,codec='libx264',extra_args=['-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        with writer.saving(fig,str(out/f'qhj_{family}_events.mp4'),100):
            for f in frames:
                fig.clear()
                if family=='outer':
                    axs=fig.subplots(2,2)
                    for col,name in enumerate(('free','coupled')):
                        R=f['R_'+name];rho=f[name+'_rho_R'];m=rho>=1e-4*rpeak
                        for key,color,label in zip(('EF_force','quantum_force','net_flow_force'),colors,
                            ('TDPES force','Nuclear quantum force','Sum: M Dv/Dt')):
                            v=f[name+'_'+key];axs[0,col].plot(R,np.where(m,v,np.nan),color=color,label=label,lw=1.5)
                            for sign,marker in ((1,'^'),(-1,'v')):
                                mask=m&(sign*v>.4);axs[0,col].scatter(R[mask],np.full(mask.sum(),sign*.4),s=12,marker=marker,color=color)
                        axs[0,col].set(title=name.capitalize()+': force balance',ylim=(-.43,.43),ylabel='Force (a.u.; fixed zoom)')
                        axs[0,col].legend(fontsize=8)
                        axs[1,col].plot(R,rho,color='#444444');axs[1,col].set(title=r'$|\chi|^2$: where the force is occupied',ylim=(0,rpeak*1.05))
                    for ax in axs.flat:ax.set(xlim=(-4.4,4.4),xlabel=r'$R$ ($a_0$)');ax.axvline(0,color='0.6',ls=':',lw=.7)
                else:
                    axs=fig.subplots(2,3)
                    for row,name in zip(axs,('free','coupled')):
                        R=f['R_'+name];rho=f[name+'_joint_rho_qR']/np.sqrt(float(f['omega_'+name]));m=rho>=1e-4*peak
                        pref=name+'_joint_';ef=f[pref+'photon_scalar_force']+f[pref+'photon_time_force']+f[pref+'photon_berry_force']
                        for ax,v,title in zip(row,(ef,f[pref+'photon_quantum_force'],f[pref+'photon_net_force']),
                            ('Scalar + time + Berry','Photon quantum force','Sum: material derivative of a')):
                            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e4e4e4')
                            im=ax.pcolormesh(R,f['Q'],np.ma.array(v,mask=~m).T,shading='auto',cmap=cmap,norm=Normalize(-.005,.005))
                            levels=np.array([1e-4,1e-3,1e-2,.1])*peak;levels=levels[(levels>rho.min())&(levels<rho.max())]
                            if len(levels):ax.contour(R,f['Q'],rho.T,levels=levels,colors='#45665c',linewidths=.6)
                            ax.set(title=name.capitalize()+': '+title,xlabel=r'$R$ ($a_0$)',ylabel=r'$Q=\sqrt{\omega_c}q_c$',xlim=(-4.4,4.4),ylim=(-13,11))
                            fig.colorbar(im,ax=ax,extend='both',shrink=.8,label='Force (a.u.; zoom)')
                fig.suptitle(f'QHJ force balance | t={float(f["time_au"])*AU_FS:.3f} fs | positive-marginal gauge',fontsize=14)
                fig.supxlabel('SPARSE ACTUAL EVENTS, no interpolation. Shared density cutoff 1e-4. Zoom overflow marked.\n'
                    'Free cavity 170.6 / coupled 161.769 meV; native grids differ. Algebraic check is NOT derivative convergence.',fontsize=9)
                writer.grab_frame()
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('free-packet','coupled-packet','free-waves','coupled-waves','coupled-fields','out'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('New output required')
    paths=calculate(args);render(paths,args.out)


if __name__=='__main__':main()
