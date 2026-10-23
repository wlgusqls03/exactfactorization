"""Bare-state preparation and measured resonance/coupling calibration.

Electronic states are diagnostics and initial-condition factors only: the TDSE
still propagates the full electronic grid. The user-selected single-well v=1
uses the interval between fixed sites. Optional local preparation walls are
documented explicitly and never presented as fictitious physical barriers.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

from .core import build_molecular, gaussian, sine_kinetic, HISTORICAL_ARCHIVE


def product_matrix(energy, dipole, dipole2, omega, coupling, n_fock, dse=True):
    """Projected PF Hamiltonian, with P mu^2 P (not (P mu P)^2).

    Product index = matter_index * n_fock + photon_number.  The projection is
    only an analysis basis, not the space used for real-grid time propagation.
    """
    energy=np.asarray(energy); n=len(energy); nf=int(n_fock)
    if energy.ndim not in (1,2) or (energy.ndim==2 and energy.shape!=(n,n)):
        raise ValueError('matter Hamiltonian must be eigenvalues or a square matrix')
    matter=np.diag(energy) if energy.ndim==1 else energy
    ladder=np.diag(np.sqrt(np.arange(1,nf)),1)
    h=np.kron(matter,np.eye(nf))+np.kron(np.eye(n),np.diag(omega*(np.arange(nf)+.5)))
    h=h+coupling*np.sqrt(omega/2)*np.kron(dipole,ladder+ladder.T)
    if dse:
        h=h+.5*coupling**2*np.kron(dipole2,np.eye(nf))
    return h


def pair_character(eigenvectors, n_fock):
    """Select the two states carrying most |1,0> / |0,1> character.

    They need not be the globally second and third eigenstates.  Degenerate
    zero-coupling eigenvectors are never interpreted as physical LP/UP mixing.
    """
    weights=np.abs(eigenvectors[[n_fock,1],:])**2
    pair_weight=weights.sum(axis=0)
    indices=np.sort(np.argsort(pair_weight)[-2:])
    return indices, weights, pair_weight


def tracked_pair(energy, dipole, dipole2, omega, coupling, n_fock, dse=True, steps=16):
    """Continue the |1,0>, |0,1> doublet from zero coupling.

    Ranking bare weights at the target coupling can misidentify a displaced
    cavity ground state as LP. Continuation follows eigenvector overlap as the
    coupling is increased, with a one-to-one assignment at each step. Bare
    weights are retained as character diagnostics, not branch identifiers.
    """
    energy=np.asarray(energy)
    n=len(energy)*n_fock
    if energy.ndim==2:
        matter_energy,matter_vectors=eigh(energy)
        products=np.kron(matter_vectors,np.eye(n_fock))
    else:
        matter_energy=energy
        products=np.eye(n)
    previous=products[:,[n_fock,1]]
    quality=1.
    if coupling==0:
        diagonal=(matter_energy[:,None]+omega*(np.arange(n_fock)+.5)).ravel()
        order=np.argsort(diagonal,kind='stable')
        eigenvalues=diagonal[order];eigenvectors=products[:,order]
        ids=np.sort(np.array([int(np.flatnonzero(order==j)[0]) for j in (n_fock,1)]))
    else:
        for g in np.linspace(coupling/steps,coupling,steps):
            eigenvalues,eigenvectors=eigh(product_matrix(energy,dipole,dipole2,omega,g,n_fock,dse))
            overlap=abs(previous.conj().T@eigenvectors)**2
            rows,ids=linear_sum_assignment(-overlap)
            quality=min(quality,float(np.min(overlap[rows,ids])))
            previous=eigenvectors[:,ids]
        ids=np.sort(ids)
    _,weights,total=pair_character(eigenvectors,n_fock)
    return eigenvalues,eigenvectors,ids,weights,total,quality


def calibrate_strengths(energy, dipole, dipole2, omega, config):
    mu01=float(abs(dipole[0,1]))
    if not np.isfinite(mu01) or mu01<1e-10:
        raise ValueError('Selected transition dipole vanishes; cannot calibrate cavity strength')
    couplings={'zero':0.0}; measurements={}
    for case in ('weak','strong'):
        target=float(config[case+'_splitting_fraction'])
        if not 0<target<1:
            raise ValueError('splitting fractions must be between zero and one')
        coupling=target*np.sqrt(omega/2)/mu01
        couplings[case]=float(coupling)
        nf=config['n_fock_analysis']
        e,v,ids,weights,total,quality=tracked_pair(energy,dipole,dipole2,omega,coupling,
                                                nf,config['dse'])
        ef,vf,jf,_,_,_=tracked_pair(energy,dipole,dipole2,omega,coupling,2*nf,config['dse'])
        measurements[case]={
            'g_cavity':float(coupling),'target_two_level_splitting_over_omega':target,
            'measured_splitting':float(e[ids[1]]-e[ids[0]]),
            'measured_splitting_over_omega':float((e[ids[1]]-e[ids[0]])/omega),
            'estimated_exchange_period_fs':float(2*np.pi/(e[ids[1]]-e[ids[0]])/41.3413745758),
            'branch_indices':ids.tolist(),'pair_retained_weights':total[ids].tolist(),
            'product_weights':weights[:,ids].tolist(),
            'branch_tracking_min_overlap':quality,
            'branch_identification':'adiabatic eigenvector continuation from the resonant product doublet',
            'fock_doubling_gap_difference':float(abs((ef[jf[1]]-ef[jf[0]])-(e[ids[1]]-e[ids[0]]))),
        }
    return couplings,dict(transition_dipole=mu01,measurements=measurements,
        interpretation='Closed-system splitting calibration, not a linewidth criterion; '
                       'weak splitting does not imply weak mixing at exact resonance.')


def topology(R, ground, config):
    """Identify interior wells/saddles without treating fixed-ion poles as barriers."""
    idx=np.flatnonzero((R>config['left_fixed']+.5)&(R<config['heavy_fixed']-.5))
    minima=idx[find_peaks(-ground[idx])[0]]
    maxima=idx[find_peaks(ground[idx])[0]]
    # A reaction saddle must have an interior well on each side.
    saddles=[int(j) for j in maxima if np.any(minima<j) and np.any(minima>j)]
    chosen=None
    if saddles:
        chosen=min(saddles,key=lambda j:abs(R[j]))
    return dict(minima_R=R[minima].tolist(),minima_E=ground[minima].tolist(),
                saddles_R=R[saddles].tolist(),
                natural_barrier_R=None if chosen is None else float(R[chosen]),
                has_double_well_barrier=chosen is not None)


def local_vibrations(packet, wall):
    """Local left-well DST eigenproblem; the wall is removed during full TDSE.

    The preparation domain has Dirichlet ends just beyond selected interior
    grid nodes.  Smoothness at its right endpoint is a convergence diagnostic;
    this is not an eigenstate of the global periodic propagation Hamiltonian.
    """
    c=packet['config']; R=packet['R']; dR=packet['dR']
    idx=np.flatnonzero((R>c['left_fixed'])&(R<float(wall)))
    nv=int(c['n_vib'])
    if len(idx)<max(8,nv+2):
        raise ValueError('Local vibrational interval has too few grid points')
    tx,_=sine_kinetic(len(idx),dR,c['mass'])
    ev,vec=eigh(tx+np.diag(packet['bo_energy'][idx,0]),subset_by_index=[0,nv-1])
    wave=np.zeros((len(R),nv)); wave[idx]=vec/np.sqrt(dR)
    # Deterministic sign, no kick. This leaves the nodal v=1 state genuinely real.
    for j in range(nv):
        wave[:,j]*=1 if wave[np.argmax(abs(wave[:,j])),j]>=0 else -1
    phi=packet['bo_phi'][:,:,0]
    mu0=np.sum(phi**2*packet['mu'],axis=1)*packet['dx']
    mu20=np.sum(phi**2*packet['mu']**2,axis=1)*packet['dx']
    dip=(wave.T*(mu0*dR))@wave
    dip2=(wave.T*(mu20*dR))@wave
    detail=dict(method='left-domain Dirichlet DST-I eigenproblem of T_R+E_S0(R)',
        wall_requested=float(wall),wall_discrete=float(R[idx[-1]]+dR),
        domain_left=float(R[idx[0]]-dR),domain_right=float(R[idx[-1]]+dR),
        preparation_wall_removed_during_TDSE=True,momentum_kick=0.0,
        endpoint_cell_probability_v1=float(wave[idx[-1],1]**2*dR),
        nuclear_norm_v1=float(np.sum(wave[:,1]**2)*dR),
        interpretation='Local preparation, not a global double-well parity eigenstate; '
                       'wall/location and grid convergence remain required.')
    return ev,wave,dip,dip2,detail


def prepare(config=None):
    """Build reusable molecular cache and calibrate PEN and, if defined, VSC.

    The user-confirmed default is the actual single-well v=1 state. An
    artificial local preparation wall requires ``local_state_method=local_wall``
    and ``local_wall_R``. Explicitly clearing the method to None preserves the
    conservative refusal if no natural barrier/wall was specified. No reaction
    barrier is invented: ``barrier_R`` remains a labelled transfer dividing plane.
    """
    p=build_molecular(config); c=p['config']; R=p['R']; dx=p['dx']; nbo=c['n_bo']
    energies=np.empty((len(R),nbo)); phi=np.empty((len(R),len(p['x']),nbo))
    for i,v in enumerate(p['V']):
        e,u=eigh(p['Tx']+np.diag(v),subset_by_index=[0,nbo-1],check_finite=False)
        if i:
            u*=np.where(np.sum(phi[i-1]*u,axis=0)<0,-1.,1.)[None,:]
        energies[i]=e; phi[i]=u/np.sqrt(dx)
    p.update(bo_energy=energies,bo_phi=phi)
    p['chi_pen']=gaussian(R,p['dR'],c['R0'],c['sigma'],c['p0'])
    topo=topology(R,energies[:,0],c)
    # Compute at R0 exactly, not merely nearest coarse nuclear cell.
    from .core import frozen_potential
    vr=frozen_potential(p['x'],np.asarray([c['R0']]),c)[0]
    e0,u0=eigh(p['Tx']+np.diag(vr),subset_by_index=[0,nbo-1])
    mu=c['R0']-p['x']+c['left_fixed']+c['heavy_fixed']
    dip=(u0.T*mu)@u0; dip2=(u0.T*mu**2)@u0
    wp=float(e0[1]-e0[0])
    gp,cp=calibrate_strengths(e0,dip,dip2,wp,c)
    p.update(omega_pen=wp,g_pen=gp)
    method=c.get('local_state_method')
    if method not in (None,'single_well','local_wall'):
        raise ValueError('local_state_method must be single_well or local_wall')
    if method=='single_well':
        if len(topo['minima_R'])!=1:
            raise ValueError('single_well preparation requires exactly one interior BO minimum')
        wall=c['heavy_fixed']
    else:
        wall=c.get('local_wall_R',topo['natural_barrier_R'])
    p['barrier_R']=float(c.get('dividing_R',topo['natural_barrier_R']
                              if topo['natural_barrier_R'] is not None else 0.0))
    meta=dict(topology=topo,pen=dict(omega_bare=wp,resonance_R=float(c['R0']),
        electronic_state=1,nuclear_center=c['R0'],nuclear_density_sigma=c['sigma'],
        nuclear_momentum=c['p0'],coupling=cp),
        dividing_surface_kind='natural BO saddle' if topo['natural_barrier_R'] is not None
                              else 'chosen transfer plane, NOT a BO barrier',
        initial_condition_provenance=HISTORICAL_ARCHIVE,
        paper_scope='2024 JCP 161 084112 motivates S1/vacuum and bare-gap resonance; '
                    'its two-state model, mass, and DSE omission are not reproduced. '
                    'The VSC local-v1 coherent preparation is a selected protocol, '
                    'not the thermal-rate preparation of Li et al. (2021).')
    if wall is None:
        p.update(vsc_available=False,omega_vsc=None,g_vsc={})
        meta['vsc']=dict(available=False,reason='No interior double-well BO barrier found. '
            'Choose local_state_method=single_well or local_wall_R explicitly; R=0 is not a barrier.')
    else:
        ev,wave,dv,dv2,detail=local_vibrations(p,wall)
        if method=='single_well':
            detail.update(method='interior single-well Dirichlet DST-I eigenproblem of T_R+E_S0(R)',
                interpretation='First excited vibration of the actual inherited single well, '
                               'not a double-well left-localized state. Domain endpoint and '
                               'grid convergence remain required.')
        detail['left_of_zero_probability_v1']=float(np.sum(wave[R<0,1]**2)*p['dR'])
        detail['state_method']=method or 'local_wall'
        wv=float(ev[1]-ev[0]); gv,cv=calibrate_strengths(ev,dv,dv2,wv,c)
        p.update(vsc_available=True,vib_energy=ev,vib_phi=wave,
                 vib_dipole=dv,vib_dipole2=dv2,chi_vsc=wave[:,1].astype(complex),
                 omega_vsc=wv,g_vsc=gv)
        meta['vsc']=dict(available=True,omega_bare=wv,preparation=detail,coupling=cv,
            explicit_local_wall='local_wall_R' in c,
            vibrational_analysis='BO-ground projected diagnostic, not full TDSE truncation')
    p['calibration']=meta
    return p
