"""Positive-gauge MCEF-style panels for native real-grid photon fields.

Arrays are (R,q), not historical proton-heavy (q,R). All fields retain a.u.;
Q=sqrt(omega)*q is a display coordinate only. Photon kinetic mass is 1.
No derivatives, gauge transformations, propagation or interpolation occur here.
Current formulas follow the positive chi/Lambda gauge of real_grid_mcef_fields.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
from .real_grid_mcef_plotting import prepared, style, save, AU_FS, HA_EV


def quantities(f, mass, omega, coordinate='Q'):
    """(R,q) currents, a.u.: j_q=rho*a, j_R=rho*b/M; v_Q=sqrt(w)*a.

    alpha_from_b is an independent quadrature identity, not a gauge change.
    j_R_relative integrates to zero in q on complete normalized support.
    """
    if mass <= 0 or omega <= 0 or coordinate not in ('q','Q'):
        raise ValueError('Invalid mass, frequency or photon coordinate')
    scale=np.sqrt(omega) if coordinate=='Q' else 1.
    rho=f['rho_qR']; delta=f['b']-f['alpha'][:,None]
    dq=f['q'][1]-f['q'][0]
    return dict(y=f['q']*scale,v_y=scale*f['a'],v_R=f['b']/mass,
                j_q=rho*f['a'],j_R=rho*f['b']/mass,
                j_R_relative=rho*delta/mass,delta=delta,delta2=delta**2,
                alpha_from_b=np.sum(f['lambda_density']*f['b'],axis=1)*dq,
                qbar=f['conditional_q']*scale)


def configuration(frames, mass, omega, budget, floor, coordinate):
    """Shared full-range color scales; arrows show direction, not speed magnitude."""
    states=[prepared(f,budget,floor) for f in frames]
    values=[quantities(f,mass,omega,coordinate) for f in frames]
    ranges={}
    for key in ('a','b','alpha','delta','delta2','j_R_relative'):
        samples=[]
        for f,p,v in zip(frames,states,values):
            z=f[key] if key in ('a','b') else (
                np.broadcast_to(f['alpha'][:,None],f['rho_qR'].shape) if key=='alpha' else v[key])
            samples.append(float(np.max(abs(z[p['mj']]))))
        ranges[key]=max(max(samples),1e-12)
    for key in ('epsilon1_cond','epsilon1_geo','epsilon1_GD','epsilon1_total'):
        samples=[]
        for f,p in zip(frames,states):
            z=component(f,key)
            # Display each component relative to its occupied weighted mean.
            z=z-np.average(z[p['mj']],weights=f['rho_qR'][p['mj']])
            samples.append(float(np.max(abs(z[p['mj']])))*HA_EV)
        ranges[key]=max(max(samples),1e-6)
    common=max(ranges[k] for k in ranges if k.startswith('epsilon1'))
    for k in ranges:
        if k.startswith('epsilon1'):ranges[k]=common
    extent=max(float(np.max(abs(v['y'][p['mj'].any(axis=0)]))) for v,p in zip(values,states))*1.06
    return states,values,dict(ranges=ranges,y_extent=extent,
        arrow_length_viewport_fraction=.025,arrow_density_floor=1e-3,
        coordinate=coordinate,probability_budget=budget,display_density_floor=floor)


def component(f,key):
    """Natural-gauge epsilon1 decomposition in Ha, each array (R,q)."""
    if key=='epsilon1_geo':return (f['epsilon1_qgeo']+f['epsilon1_Rgeo']).real
    if key=='epsilon1_total':return f['epsilon1_A'].real
    return f[key].real


def heat(fig,ax,f,p,v,c,z,title,label,limit=None,positive=False):
    """Masked real-grid field, zero-centred diverging or sequential scale."""
    data=np.ma.array(z,mask=~p['mj'])
    if limit is None:limit=max(float(np.ma.max(abs(data))),1e-12)
    norm=SymLogNorm(max(limit*1e-6,1e-12),vmin=0,vmax=limit) if positive else SymLogNorm(max(limit*.01,1e-12),vmin=-limit,vmax=limit)
    im=ax.pcolormesh(f['R'],v['y'],data.T,shading='auto',rasterized=True,
                    cmap='viridis' if positive else 'RdBu_r',norm=norm)
    rel=f['rho_qR']/f['rho_qR'].max()
    ax.contour(f['R'],v['y'],rel.T,levels=[1e-5,1e-3,.1],colors='0.25',linewidths=.5,alpha=.55)
    ax.axvline(0,color='0.4',ls=':',lw=.7)
    ax.set(title=title,xlabel=r'$R$ ($a_0$)',ylabel=(r'$Q=\sqrt{\omega_c}q_c$' if c['coordinate']=='Q' else r'$q_c$ (a.u.)'),
           ylim=(-c['y_extent'],c['y_extent']))
    fig.colorbar(im,ax=ax,label=label,shrink=.85)


def draw(fig,f,p,v,c,family,label):
    """Four-panel families: connections, transport, epsilon1 decomposition."""
    axs=fig.subplots(2,2);lim=c['ranges']
    if family=='connections':
        for ax,key,z,title in zip(axs.flat,('a','b','alpha','delta'),
            (f['a'],f['b'],np.broadcast_to(f['alpha'][:,None],f['b'].shape),v['delta']),
            (r'$a(q_c,R)$: photon connection',r'$b(q_c,R)$: nuclear connection',
             r'$\alpha(R)$: nuclear marginal connection',r'$b-\alpha$: conditional deviation')):
            # b, alpha and delta share one scale; a is a different momentum unit.
            bound=lim['a'] if key=='a' else max(lim[k] for k in ('b','alpha','delta'))
            heat(fig,ax,f,p,v,c,z,title,'Momentum (a.u.; symlog)',bound)
    elif family=='transport':
        ax=axs[0,0];rel=f['rho_qR']/f['rho_qR'].max()
        im=ax.pcolormesh(f['R'],v['y'],np.ma.masked_less(rel,c['display_density_floor']).T,
            norm=LogNorm(c['display_density_floor'],1),cmap='magma',shading='auto',rasterized=True)
        ir=np.arange(0,len(f['R']),max(1,len(f['R'])//22))
        iq=np.arange(0,len(v['y']),max(1,len(v['y'])//26))
        rr,yy=np.meshgrid(f['R'][ir],v['y'][iq],indexing='ij');ix=np.ix_(ir,iq)
        good=p['mj'][ix]&(rel[ix]>=c['arrow_density_floor'])
        vr=v['v_R'][ix];vy=v['v_y'][ix]
        speed=np.hypot(vr/np.ptp(f['R']),vy/(2*c['y_extent']))
        good &= speed>1e-14
        dt=c['arrow_length_viewport_fraction']/speed[good]
        ax.quiver(rr[good],yy[good],dt*vr[good],dt*vy[good],
                  angles='xy',scale_units='xy',scale=1,color='cyan',width=.004)
        ax.set(title='Current direction | arrow length normalized',xlabel=r'$R$ ($a_0$)',
               ylabel=c['coordinate'],ylim=(-c['y_extent'],c['y_extent']))
        ax.axvline(0,color='white',ls=':',lw=.7)
        fig.colorbar(im,ax=ax,label='Relative joint density',shrink=.85)
        heat(fig,axs[0,1],f,p,v,c,v['j_R_relative'],r'$J^R_{\rm rel}=\rho_{qR}(b-\alpha)/M$',
             'Relative nuclear current (a.u.; symlog)',lim['j_R_relative'])
        heat(fig,axs[1,0],f,p,v,c,v['delta2'],r'$(b-\alpha)^2$',
             'Squared momentum (a.u.; sequential symlog)',lim['delta2'],positive=True)
        ax=axs[1,1];mr=p['mr']
        ax.plot(f['R'],np.where(mr,f['alpha'],np.nan),label=r'$\alpha$',lw=2)
        ax.plot(f['R'],np.where(mr,v['alpha_from_b'],np.nan),'--',label=r'$\int dq_c\,|\Lambda|^2 b$',lw=1)
        bound=max(lim[k] for k in ('b','alpha','delta'))
        ax.set(xlabel=r'$R$ ($a_0$)',ylabel='Nuclear momentum (a.u.)',title='Outer momentum: independent contraction',ylim=(-bound,bound))
        ax.set_yscale('symlog',linthresh=max(bound*.01,1e-12));ax.legend(fontsize=10)
        den=ax.twinx();den.fill_between(f['R'],0,f['rho_R'],alpha=.15,color='grey');den.set_ylabel(r'$\rho_R$')
    else:
        for ax,key,title in zip(axs.flat,('epsilon1_cond','epsilon1_geo','epsilon1_GD','epsilon1_total'),
            ('Conditional electronic + cavity energy','Geometric contribution','Time-connection contribution','Total electronic-level TDPES')):
            z=component(f,key);off=np.average(z[p['mj']],weights=f['rho_qR'][p['mj']])
            heat(fig,ax,f,p,v,c,(z-off)*HA_EV,title,'Mean-aligned energy (eV; symlog)',lim[key])
    fig.suptitle(f'{label} | {float(f["time_au"])*AU_FS:.3f} fs\nPositive-marginal gauge | diagnostic, not a mechanism certificate',fontsize=14)


def gallery(frames,out,mass,omega,label,budget=1e-8,floor=1e-5,coordinate='Q',movie=False,fps=1):
    """Save PNG/PDF, three event movies, RMS traces, quadrature consistency JSON data."""
    if not frames or not 0<floor<1 or fps<=0:raise ValueError('Invalid gallery settings')
    style();states,values,c=configuration(frames,mass,omega,budget,floor,coordinate)
    families=('connections','transport','epsilon1_components')
    for family in families:
        for i,(f,p,v) in enumerate(zip(frames,states,values)):
            fig=plt.figure(figsize=(12,9),layout='constrained');draw(fig,f,p,v,c,family,label)
            save(fig,out,f'vsc_{family}_{i:03d}')
        if movie:
            ext='mp4' if writers.is_available('ffmpeg') else 'gif'
            writer=FFMpegWriter(fps=fps,bitrate=2500) if ext=='mp4' else PillowWriter(fps=fps)
            fig=plt.figure(figsize=(12,9),layout='constrained')
            with writer.saving(fig,str(out/f'vsc_{family}_events.{ext}'),dpi=110):
                for f,p,v in zip(frames,states,values):
                    fig.clear();draw(fig,f,p,v,c,family,label);writer.grab_frame()
            plt.close(fig)
    traces={k:[] for k in ('a','b','alpha','delta')};checks=[]
    for f,p,v in zip(frames,states,values):
        dr=f['R'][1]-f['R'][0];dq=f['q'][1]-f['q'][0]
        for key in traces:
            z=f[key] if key in ('a','b') else (np.broadcast_to(f['alpha'][:,None],f['b'].shape) if key=='alpha' else v[key])
            traces[key].append(float(np.sqrt(np.average(z[p['mj']]**2,weights=f['rho_qR'][p['mj']]))))
        mr=p['mr'];checks.append(dict(time_fs=float(f['time_au'])*AU_FS,
            alpha_contraction_max=float(np.max(abs(v['alpha_from_b'][mr]-f['alpha'][mr]))),
            marginal_current_max=float(np.max(abs(np.sum(v['j_R'],axis=1)[mr]*dq-f['current'][mr]))),
            relative_current_integral_max=float(np.max(abs(np.sum(v['j_R_relative'],axis=1)[mr]*dq)))))
    fig,axs=plt.subplots(2,1,figsize=(9,6),layout='constrained',sharex=True)
    t=[float(f['time_au'])*AU_FS for f in frames]
    axs[0].plot(t,traces['a'],'o-',label='a');axs[0].set_ylabel('Photon momentum RMS (a.u.)');axs[0].legend()
    for k in ('b','alpha','delta'):axs[1].plot(t,traces[k],'o-',label='b-alpha' if k=='delta' else k)
    axs[1].set(xlabel='Actual saved time (fs)',ylabel='Nuclear momentum RMS (a.u.)');axs[1].legend()
    fig.suptitle('Occupied-density-weighted RMS | sparse samples, not causal attribution')
    save(fig,out,'vsc_connection_rms')
    return dict(phase7_pass=False,gauge='positive chi and Lambda; alpha is NOT zero',
        configuration=c,checks=checks,traces=traces,
        arrows='Instantaneous probability-current direction, not photon spatial motion or trajectories; equal normalized viewport length, NOT speed magnitude',
        movies='Only actual saved times; nonuniform physical time, no interpolated waves',
        components='Each epsilon1 component has its own weighted mean subtracted; same mask/weights, additive up to roundoff; not a gauge transformation',
        not_included=['No copied historical proton-mass T1..T8 or discrete link-branch flags',
                      'No alpha=0 gauge transformation','No automatic uncoupled comparison'])
