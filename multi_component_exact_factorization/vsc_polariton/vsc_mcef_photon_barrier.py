"""VSC-specific marginal, static-reference and force-work diagnostics.

No claim that a positive-gauge scalar is an activation barrier. Force-work
differences are defined only inside one occupied connected R component.
The bare BO/scalar-DSE references are NOT full electron-photon dressed PESs.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import label
from .real_grid_mcef_plotting import save, AU_FS, HA_EV


def molecular_reference(packet):
    """Bare BO expectations using native FFT T_x, (NR,) Ha and dipole a.u.

    Full projected DSE uses <mu^2>; scalar-DSE uses <mu>^2. Neither includes
    subsequent electron relaxation in a cavity or a nuclear DBOC here.
    """
    phi=packet['phi'][...,0];dx=float(packet['dx'])
    hphi=ifft(fft(phi,axis=1)*packet['tx'][None,:],axis=1)+packet['potential']*phi
    norm=np.sum(abs(phi)**2,axis=1)*dx
    E=np.sum((phi.conj()*hphi).real,axis=1)*dx/norm
    mu=packet['R'][:,None]-packet['x'][None,:]
    mean=np.sum(abs(phi)**2*mu,axis=1)*dx/norm
    second=np.sum(abs(phi)**2*mu**2,axis=1)*dx/norm
    residual=np.sqrt(np.sum(abs(hphi-E[:,None]*phi)**2,axis=1)*dx/norm)
    var=second-mean**2
    g=float(packet['g_chi']);w=float(packet['omega'])
    return dict(E0=E,mu0=mean,variance=var,BO_residual=residual,
                Q_valley=-np.sqrt(2)*g*mean/w,DSE_variance=g*g*var/w)


def force_path(R,force,mask,anchor):
    """W(R)-W(anchor)=-integral F dR in ONE connected support, Ha.

    No extrapolation, gap filling or alpha=0 phase reconstruction. W has the
    force of an alpha=0 scalar where that gauge exists, but is not epsilon2
    in the saved natural gauge. It is not an activation free energy.
    """
    mask=np.asarray(mask,bool)&np.isfinite(force)
    components,_=label(mask);out=np.full(len(R),np.nan)
    if not mask[anchor]:return out
    ids=np.flatnonzero(components==components[anchor])
    y=-cumulative_trapezoid(force[ids],R[ids],initial=0)
    out[ids]=y-y[np.flatnonzero(ids==anchor)[0]]
    return out


def photon_marginal(f):
    """rho_Q=rho_q/sqrt(omega), from saved q/Q Jacobian; both normalized."""
    dr=float(f['R'][1]-f['R'][0]);dq=float(f['q'][1]-f['q'][0]);dQ=float(f['Q'][1]-f['Q'][0])
    rho_q=np.sum(f['rho_qR'],axis=0)*dr
    rho_Q=rho_q*dq/dQ
    n=float(rho_Q.sum()*dQ);mean=float(np.sum(rho_Q*f['Q'])*dQ/n)
    variance=float(np.sum(rho_Q*(f['Q']-mean)**2)*dQ/n)
    return rho_q,rho_Q,dict(norm=n,mean_Q=mean,variance_Q=variance)


def vsc_diagnostics(frames,states,packet,out):
    """PNG/PDF + small NPZ, no full-wave storage; finite-time diagnostics only."""
    ref=molecular_reference(packet);R=frames[0]['R'];E=ref['E0'];dr=R[1]-R[0]
    left=np.flatnonzero(R<0);i0=int(left[np.argmin(E[left])]);its=int(np.argmin(abs(R)))
    barrier=float(E[its]-E[i0]);times=np.array([float(f['time_au'])*AU_FS for f in frames])
    work=[];scalar=[];peaks=[];wp=[];photo=[];means=[];variances=[];lags=[]
    for f,p in zip(frames,states):
        path=force_path(R,f['force'],p['mr'],i0);work.append(path)
        scalar.append(float((f['epsilon2_A'][its]-f['epsilon2_A'][i0]).real) if np.isfinite(path[its]) else np.nan)
        valid=left[p['mr'][left]]
        # A unique occupied maximum; tied maxima are not assigned a reference.
        best=valid[f['rho_R'][valid]==f['rho_R'][valid].max()] if len(valid) else []
        ip=int(best[0]) if len(best)==1 else None
        pp=force_path(R,f['force'],p['mr'],ip) if ip is not None else np.full(len(R),np.nan)
        wp.append(pp[its]);peaks.append(R[ip] if ip is not None else np.nan)
        _,density,stats=photon_marginal(f);photo.append(density);means.append(stats['mean_Q']);variances.append(stats['variance_Q'])
        qbar=f['conditional_q']*np.sqrt(float(packet['omega']))
        lag=qbar-ref['Q_valley'];row=[]
        for region in (R<-.5,abs(R)<=.5,R>.5):
            m=p['mr']&region;weight=f['rho_R'][m]
            row.append(float(np.average(lag[m],weights=weight)) if weight.sum()*dr>1e-8 else np.nan)
        lags.append(row)
    work=np.array(work);lags=np.array(lags);delta=work[:,its]
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axs[0,0].plot(R,(E-E[i0])*HA_EV,label='Bare molecular PES',lw=2)
    axs[0,0].plot(R,(E-E[i0])*HA_EV,'--',label='Relaxed scalar-DSE CBO (identity)',lw=1.4)
    axs[0,0].set(title='Static references: NOT full dressed PES',ylabel='Relative energy (eV)',xlabel=r'$R$ ($a_0$)',xlim=(-3,3),ylim=(-.1,2));axs[0,0].legend(fontsize=8)
    for k,(t,f) in enumerate(zip(times,frames)):
        axs[0,1].plot(R,work[k]*HA_EV,label=f'{t:.2f} fs',lw=1)
    axs[0,1].set(title=r'$W_F(R)-W_F(R_0)=-\int_{R_0}^R F_R\,dR$',ylabel='Force-integrated difference (eV)',xlabel=r'$R$ ($a_0$)')
    axs[0,1].set_yscale('symlog',linthresh=.1);axs[0,1].legend(fontsize=7,ncol=2)
    axs[1,0].plot(times,delta*HA_EV,'o-',label=r'$W_F(R_{TS})-W_F(R_0)$')
    axs[1,0].plot(times,np.array(scalar)*HA_EV,'s--',label='Positive-gauge scalar difference')
    axs[1,0].axhline(barrier*HA_EV,color='black',ls=':',label='Bare static barrier')
    axs[1,0].set(title='Undefined across disconnected support',xlabel='Actual saved time (fs)',ylabel='Difference (eV)');axs[1,0].legend(fontsize=8)
    bare_peak=np.array([E[its]-E[int(np.argmin(abs(R-r)))] if np.isfinite(r) else np.nan for r in peaks])
    axs[1,1].plot(times,np.array(wp)*HA_EV,'o-',label='Force-work: occupied left peak to TS')
    axs[1,1].plot(times,bare_peak*HA_EV,'--',label='Bare PES at SAME moving reference')
    axs[1,1].set(title='Moving reference: not a stationary barrier',xlabel='Actual saved time (fs)',ylabel='Difference (eV)');axs[1,1].legend(fontsize=8)
    fig.suptitle('Barrier diagnostics | force includes alpha_t | NOT activation free energy or Phase7 certification')
    save(fig,out,'vsc_barrier_diagnostics')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for t,f,rho in zip(times,frames,photo):axs[0,0].plot(f['Q'],rho,label=f'{t:.2f} fs',lw=1)
    axs[0,0].set(title='Photon marginal: quadrature probability',xlabel=r'$Q=\sqrt{\omega_c}q_c$',ylabel=r'$\rho_Q$');axs[0,0].legend(fontsize=7,ncol=2)
    axs[0,1].plot(times,means,'o-',label=r'$\langle Q\rangle$')
    axs[0,1].fill_between(times,np.array(means)-np.sqrt(variances),np.array(means)+np.sqrt(variances),alpha=.2,label=r'Mean $\pm$ standard deviation')
    axs[0,1].set(title='Quadrature response (not photon number)',xlabel='Actual saved time (fs)',ylabel='Q');axs[0,1].legend(fontsize=8)
    axs[1,0].plot(R,ref['Q_valley'],'k--',label='Bare BO static valley')
    for t,f,p in zip(times,frames,states):axs[1,0].plot(R,np.where(p['mr'],f['conditional_q']*np.sqrt(float(packet['omega'])),np.nan),lw=1,label=f'{t:.2f} fs')
    axs[1,0].set(title='Conditional mean versus static BO reference',xlabel=r'$R$ ($a_0$)',ylabel=r'$\overline{Q}(R)$');axs[1,0].legend(fontsize=7,ncol=2)
    for k,name in enumerate(('Reactant R < -0.5','Barrier |R| <= 0.5','Product R > 0.5')):
        axs[1,1].plot(times,lags[:,k],'o-',label=name)
    axs[1,1].set(title='Density-weighted deviation from BO valley',xlabel='Actual saved time (fs)',ylabel=r'$\Delta Q$');axs[1,1].legend(fontsize=8)
    fig.suptitle('VSC photon diagnostics | no photon position or phase angle | sparse actual frames')
    save(fig,out,'vsc_photon_marginal_and_lag')
    np.savez_compressed(out/'vsc_photon_barrier_arrays.npz',R=R,times_fs=times,E0=E,
        Q_valley=ref['Q_valley'],DSE_variance=ref['DSE_variance'],work_potential=work,
        fixed_force_work=delta,fixed_scalar=scalar,peak_force_work=wp,peak_R=peaks,
        photon_Q=frames[0]['Q'],photon_marginal_Q=np.array(photo),mean_Q=means,variance_Q=variances,regional_lag_Q=lags)
    return dict(R0_grid=float(R[i0]),RTS_grid=float(R[its]),bare_barrier_eV=barrier*HA_EV,
        BO_eigen_residual_max_Ha=float(ref['BO_residual'].max()),
        defined_fixed_reference_frames=int(np.isfinite(delta).sum()),
        note='All references use nearest saved R grid point. Force-work uses trapezoidal spatial integration and requires separate derivative/quadrature validation; no alpha=0 factor constructed. Full-electron lag is relative to bare BO valley, not exact evolving electronic equilibrium.')
