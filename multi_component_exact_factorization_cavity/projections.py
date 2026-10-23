"""BO, CBO and polaritonic projections, never propagation truncations.

Fixed-R electronic-photon eigenvalues are valid outer-PES references.  A
vibrational polariton includes nuclear kinetic energy; its energies are global
levels, NOT fixed-R electronic-photon PES and not relabelled as such here.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import RegularGridInterpolator

from .calibration import product_matrix, pair_character, tracked_pair
from .core import bare_inverse, erf_inverse


def oscillator_functions(Q, n_fock):
    """Continuous normalized Fock functions evaluated on the actual Q grid."""
    Q=np.asarray(Q); functions=np.empty((len(Q),n_fock))
    functions[:,0]=np.pi**(-.25)*np.exp(-Q**2/2)
    if n_fock>1:
        functions[:,1]=np.sqrt(2)*Q*functions[:,0]
    for n in range(1,n_fock-1):
        functions[:,n+1]=(np.sqrt(2/(n+1))*Q*functions[:,n]
                            -np.sqrt(n/(n+1))*functions[:,n-1])
    return functions


def projected_vibrational_matter(packet):
    """Project the native full molecular H onto phi_0(x;R) chi_v(R).

    The R kinetic operator acts on the entire electron-nuclear product, not
    only chi_v. Thus BO derivative/DBOC and finite-grid contributions are
    included without invoking a continuum product rule. Electronic T_x+V
    acts as the already computed native BO eigenvalue E_0(R). This remains a
    finite S0-local-vibrational *analysis* subspace, not the propagation space.
    """
    p=packet; dR=p['dR']; dx=p['dx']; mass=p['mass']
    basis=p['bo_phi'][:,:,0,None]*p['vib_phi'][:,None,:]
    k=2*np.pi*np.fft.fftfreq(len(p['R']),dR)
    kinetic=np.fft.ifft(np.fft.fft(basis,axis=0)*
                       (k**2/(2*mass))[:,None,None],axis=0)
    action=kinetic+p['bo_energy'][:,0,None,None]*basis
    matrix=np.einsum('Rxv,Rxw->vw',basis.conj(),action,optimize=True)*dR*dx
    overlap=np.einsum('Rxv,Rxw->vw',basis.conj(),basis,optimize=True)*dR*dx
    defect=float(np.max(abs(matrix-matrix.conj().T)))
    matrix=.5*(matrix+matrix.conj().T)
    matrix=np.real_if_close(matrix,tol=1000)
    return matrix,dict(hermiticity_defect=defect,
        basis_gram_error=float(np.max(abs(overlap-np.eye(overlap.shape[0])))),
        bo_reduced_difference_norm=float(np.linalg.norm(matrix-np.diag(p['vib_energy']))))


def build_analysis(packet, omega, g_cavity, Q, config=None):
    """Build small projected electronic/photon and vibrational/photon bases."""
    c=dict(packet['config']); c.update(config or {})
    nf=c['n_fock_analysis']; R=packet['R']; phi=packet['bo_phi']; dx=packet['dx']
    dQ=float(Q[1]-Q[0]); bo=packet['bo_energy']; nb=bo.shape[1]
    fock=oscillator_functions(Q,nf)
    kQ=2*np.pi*np.fft.fftfreq(len(Q),dQ)
    photon_action=.5*omega*(np.fft.ifft(np.fft.fft(fock,axis=0)*kQ[:,None]**2,axis=0)
                              +np.asarray(Q)[:,None]**2*fock)
    photon_projection=fock.T@photon_action*dQ
    photon_error=float(np.max(abs(photon_projection-np.diag(omega*(np.arange(nf)+.5)))))
    eig=np.empty((len(R),nb*nf)); vec=np.empty((len(R),nb*nf,nb*nf))
    cbo_energy=np.empty((len(R),len(Q),nb))
    cbo_vectors=np.empty((len(R),len(Q),nb,nb))
    pairs=np.empty((len(R),2),dtype=np.int64)
    pair_weights=np.empty((len(R),2)); mixing=np.empty((len(R),2))
    for i in range(len(R)):
        dip=(phi[i].T*(packet['mu'][i]*dx))@phi[i]
        dip2=(phi[i].T*(packet['mu'][i]**2*dx))@phi[i]
        # Static fixed-(R,Q) electronic Hamiltonian in the retained BO basis.
        # This is an analysis projection only. Full-x CBO eigenvalues remain
        # the independently computed reference in references().
        cbo_h=np.diag(bo[i])[None,:,:]+np.sqrt(omega)*g_cavity*np.asarray(Q)[:,None,None]*dip[None,:,:]
        if c['dse']:
            cbo_h=cbo_h+.5*g_cavity**2*dip2[None,:,:]
        ce,cv=np.linalg.eigh(cbo_h)
        cbo_energy[i]=ce+.5*omega*np.asarray(Q)[:,None]**2
        cbo_vectors[i]=cv
        eig[i],vec[i],ids,w,total,_=tracked_pair(bo[i],dip,dip2,omega,g_cavity,nf,c['dse'])
        pairs[i]=ids;pair_weights[i]=total[ids]
        mixing[i]=4*w[0,ids]*w[1,ids]/np.maximum(total[ids]**2,1e-300)
    # Retained character must enter location selection: a tiny admixture with
    # equal components is not evidence for an occupied, strongly mixed doublet.
    mixing_score=(mixing*pair_weights).mean(axis=1)
    result=dict(fock=fock,pol_energy=eig,pol_vectors=vec,
        cbo_projected_energy=cbo_energy,cbo_projected_vectors=cbo_vectors,
        cbo_projection_basis=np.asarray('retained bare-BO electronic basis; not full-x CBO eigenvectors'),
        pol_pair_indices=pairs,pol_pair_weight=pair_weights,
        pol_pair_mixing=mixing,omega=np.asarray(omega),g_cavity=np.asarray(g_cavity),
        fock_gram_error=np.asarray(np.max(abs(fock.T@fock*dQ-np.eye(nf)))),
        fock_native_hamiltonian_error=np.asarray(photon_error),
        fock_native_hamiltonian_error_over_omega=np.asarray(photon_error/omega),
        n_pol_plot=np.asarray(min(int(c['n_pol_plot']),nb*nf)),
        lp_up_defined=np.asarray(g_cavity!=0),
        electronic_pair_mixing_score=mixing_score,
        maximum_electronic_pair_mixing_score=np.asarray(mixing_score.max()),
        strongest_electronic_pair_mixing_R=np.asarray(R[np.argmax(mixing_score)]))
    if packet.get('vsc_available',False):
        model=c.get('vsc_analysis_model','projected_full')
        if model=='projected_full':
            matter,checks=projected_vibrational_matter(packet)
        elif model=='bo_reduced':
            matter=np.diag(packet['vib_energy'])
            checks=dict(hermiticity_defect=0.,basis_gram_error=0.,bo_reduced_difference_norm=0.)
        else:
            raise ValueError('vsc_analysis_model must be projected_full or bo_reduced')
        matter_energy=np.linalg.eigvalsh(matter)
        ev,uv,vi,vw,vt,quality=tracked_pair(matter,packet['vib_dipole'],
            packet['vib_dipole2'],omega,g_cavity,nf,c['dse'])
        result.update(vib_pol_energy=ev,vib_pol_vectors=uv,vib_pol_pair_indices=vi,
                      vib_pol_pair_weight=vt[vi],vib_pol_product_weights=vw[:,vi],
                      vib_branch_tracking_min_overlap=np.asarray(quality),
                      vib_matter_matrix=matter,vib_matter_energy=matter_energy,
                      vib_analysis_model=np.asarray(model),
                      vib_analysis_gap=np.asarray(matter_energy[1]-matter_energy[0]),
                      vib_bare_resonance_gap=np.asarray(packet['omega_vsc']),
                      vib_matter_hermiticity_defect=np.asarray(checks['hermiticity_defect']),
                      vib_matter_gram_error=np.asarray(checks['basis_gram_error']),
                      vib_bo_reduced_difference_norm=np.asarray(checks['bo_reduced_difference_norm']))
    return result


def project_cbo(psi_Q, packet, analysis, Q, *, bo_coeff=None):
    """Local CBO character in the retained BO analysis space, plus omissions.

    Character is conditional on (R,Q); global probabilities include joint
    density weights and are not normalized inside the retained subspace. A
    zero-density point stays NaN, with no density floor or invented character.
    ``bo_coeff`` reuses the electronic projection already needed by project().
    """
    p=packet;dx=p['dx'];dR=p['dR'];dQ=float(Q[1]-Q[0])
    if bo_coeff is None:
        bo_coeff=np.einsum('Rxb,RxQ->RbQ',p['bo_phi'].conj(),psi_Q,optimize=True)*dx
    vectors=analysis['cbo_projected_vectors']
    coefficients=np.einsum('RQbk,RbQ->RQk',vectors.conj(),bo_coeff,optimize=True)
    density=abs(coefficients)**2
    joint=np.sum(abs(psi_Q)**2,axis=1)*dx
    char=np.full(density.shape,np.nan,dtype=float)
    np.divide(density,joint[:,:,None],out=char,where=joint[:,:,None]>0)
    population=density.sum(axis=(0,1))*dR*dQ
    norm=float(joint.sum()*dR*dQ)
    return dict(cbo_character=char,cbo_outside=1.-np.sum(char,axis=2),
                cbo_population=population,cbo_omitted=norm-float(population.sum()))


def project(psi_Q, packet, analysis, Q, *, include_cbo=False):
    """Actual projections including populations outside every shown subspace.

    Norms/populations below are absolute probabilities (not renormalized inside
    a retained basis). The VSC basis keeps only S0 and selected local nuclear
    functions; its population deficit is reported explicitly even when the
    projected native full Hamiltonian (default) defines its polaritons.
    """
    p=packet; dQ=float(Q[1]-Q[0]);dR=p['dR'];dx=p['dx']
    norm=float(np.sum(abs(psi_Q)**2)*dx*dR*dQ)
    coeff=np.einsum('Rxb,RxQ->RbQ',p['bo_phi'].conj(),psi_Q,optimize=True)*dx
    bo_pop=np.sum(abs(coeff)**2,axis=(0,2))*dR*dQ
    product=np.einsum('RbQ,Qn->Rbn',coeff,analysis['fock'],optimize=True)*dQ
    flat=product.reshape(len(p['R']),-1)
    projected=np.einsum('Rik,Ri->Rk',analysis['pol_vectors'].conj(),flat,optimize=True)
    pop=np.sum(abs(projected)**2,axis=0)*dR
    nplot=int(analysis['n_pol_plot'])
    pair=np.take_along_axis(projected,analysis['pol_pair_indices'],axis=1)
    pairpop=np.sum(abs(pair)**2,axis=0)*dR
    result=dict(bo_population=bo_pop,bo_outside=norm-float(bo_pop.sum()),
        pol_population=pop[:nplot],pol_outside=norm-float(pop[:nplot].sum()),
        pol_basis_outside=norm-float(pop.sum()),
        pol_pair_lower=float(pairpop[0]),pol_pair_upper=float(pairpop[1]),
        pol_pair_other=norm-float(pairpop.sum()),
        pol_local_density=abs(projected[:,:nplot])**2,
        bo_local_density=np.sum(abs(coeff)**2,axis=2)*dQ,
        product_population=np.sum(abs(product)**2,axis=0)*dR,
        fock_analysis_gram_error=float(analysis['fock_gram_error']),
        fock_native_hamiltonian_error_over_omega=float(analysis['fock_native_hamiltonian_error_over_omega']))
    if 'vib_pol_vectors' in analysis:
        # Projection to S0 electronic factor first, then actual local v states.
        vibcoeff=np.einsum('Rv,RQ->vQ',p['vib_phi'].conj(),coeff[:,0,:],optimize=True)*dR
        vp=vibcoeff@analysis['fock']*dQ
        vc=analysis['vib_pol_vectors'].conj().T@vp.ravel()
        vv=abs(vc)**2;ids=analysis['vib_pol_pair_indices']
        vpop=np.sum(abs(vibcoeff)**2,axis=1)*dQ
        result.update(vib_population=vpop,vib_outside=norm-float(vpop.sum()),
            vib_product_population=abs(vp)**2,vib_pol_population=vv[:nplot],
            vib_pol_outside=norm-float(vv[:nplot].sum()),
            vib_pol_basis_outside=norm-float(vv.sum()),
            vib_LP=float(vv[ids[0]]),vib_UP=float(vv[ids[1]]),
            vib_other=norm-float(vv[ids].sum()))
    if include_cbo:
        result.update(project_cbo(psi_Q,p,analysis,Q,bo_coeff=coeff))
    return result


def references(packet, Q, omega, g_cavity, config=None, analysis=None):
    """CBO map on a documented static sample grid, interpolated for display.

    This avoids nx-sized diagonalization at every movie frame/grid point. The
    un-interpolated sample grid/values are saved too. The exactly known nuclear
    scalar and photon harmonic potential are removed before interpolation and
    restored on the native grid: interpolation must not turn a Coulomb pole or
    a quadratic oscillator into a piecewise-linear reference artifact. Increase
    reference_*_points for the remaining electronic eigenvalue convergence.
    Interpolation is ONLY for static visual reference curves.
    """
    p=packet;c=dict(p['config']);c.update(config or {})
    nr=min(len(p['R']),int(c.get('reference_R_points',32)))
    nq=min(len(Q),int(c.get('reference_Q_points',33)))
    ir=np.unique(np.rint(np.linspace(0,len(p['R'])-1,nr)).astype(int))
    iq=np.unique(np.rint(np.linspace(0,len(Q)-1,nq)).astype(int))
    nb=min(c['n_bo'],int(c.get('n_cbo',4)))
    sampled=np.empty((len(ir),len(iq),nb))
    for ii,i in enumerate(ir):
        for jj,j in enumerate(iq):
            v=p['V'][i]+.5*omega*Q[j]**2+np.sqrt(omega)*g_cavity*Q[j]*p['mu'][i]
            if c['dse']:
                v=v+.5*g_cavity**2*p['mu'][i]**2
            sampled[ii,jj]=eigh(p['Tx']+np.diag(v),subset_by_index=[0,nb-1],
                                eigvals_only=True,check_finite=False)
    rg,qg=np.meshgrid(p['R'],Q,indexing='ij')
    nuclear_scalar=(bare_inverse(p['R']-c['left_fixed'],label='left-fixed/proton')
        +bare_inverse(np.asarray(c['heavy_fixed']-c['left_fixed']),label='left-fixed/heavy')
        +erf_inverse(p['R']-c['heavy_fixed'],c['erf_r_qR'])
        +c['heavy_trap_alpha']*(c['heavy_fixed']-c['heavy_trap_center'])**2)
    photon_harmonic=.5*omega*np.asarray(Q)**2
    smooth_samples=sampled-nuclear_scalar[ir,None,None]-photon_harmonic[None,iq,None]
    interpolator=RegularGridInterpolator((p['R'][ir],Q[iq]),smooth_samples,
                                         bounds_error=False,fill_value=np.nan)
    cbo=(interpolator(np.stack((rg,qg),axis=-1))+nuclear_scalar[:,None,None]
         +photon_harmonic[None,:,None])
    if analysis is None:
        analysis=build_analysis(p,omega,g_cavity,Q,c)
    nplot=int(analysis['n_pol_plot'])
    result=dict(R=p['R'],q=np.asarray(Q)/np.sqrt(omega),Q=Q,
        cbo=cbo,cbo_sample_R=p['R'][ir],cbo_sample_Q=np.asarray(Q)[iq],
        cbo_sample_energy=sampled,cbo_display_interpolated=np.asarray(True),
        cbo_interpolation_method=np.asarray('subtract exact V_NN(R) and omega*Q^2/2; '
            'interpolate electronic remainder; restore exact native-grid scalars'),
        cbo_nuclear_scalar=nuclear_scalar,cbo_photon_harmonic=photon_harmonic,
        pol_energy=analysis['pol_energy'][:,:nplot],bo_energy=p['bo_energy'],
        bare_gap=p['bo_energy'][:,1]-p['bo_energy'][:,0],
        pol_pair_indices=analysis['pol_pair_indices'],
        pol_pair_mixing=analysis['pol_pair_mixing'],
        electronic_pair_mixing_score=analysis['electronic_pair_mixing_score'],
        strongest_electronic_pair_mixing_R=analysis['strongest_electronic_pair_mixing_R'])
    if 'cbo_projected_energy' in analysis:
        projected_samples=analysis['cbo_projected_energy'][ir[:,None],iq[None,:],:nb]
        result['cbo_projected_sample_energy']=projected_samples
        result['cbo_projection_sample_residual']=projected_samples-sampled
        result['cbo_projection_sample_max_error']=np.asarray(np.max(abs(projected_samples-sampled)))
    if 'vib_pol_energy' in analysis:
        result['vib_pol_levels']=analysis['vib_pol_energy'][:nplot]
        result['vib_pol_pair_indices']=analysis['vib_pol_pair_indices']
    return result
