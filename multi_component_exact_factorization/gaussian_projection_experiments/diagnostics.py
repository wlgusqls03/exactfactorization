"""Observations are passive. Reference differences are NEVER propagation rates."""
import numpy as np


def frame(problem,state,detail,t_au,reference=None):
    b,G,HG,p,power,residual=detail;m=problem.model;w=problem.w
    def norm(v):return float(np.sqrt(w@np.sum(abs(v)**2,axis=1)))
    def energy(v):return float(np.real(np.einsum('q,qk,qk->',w,v.conj(),problem.action(v)))/norm(v)**2)
    def kinetic(v):return float(np.real(np.einsum('q,qk,qk->',w,v.conj(),m.kinetic_q_action(v[None])[0]))/norm(v)**2)
    rho=np.sum(abs(G)**2,axis=1);n=norm(G)**2;delta=p.G_dot_actual+1j*HG
    support=rho>1e-6*max(rho)  # diagnostics only, never an evolution mask
    spectrum=np.sum(abs(np.fft.fft(G,axis=0))**2,axis=1)
    spectrum/=spectrum.sum()
    error=p.Lambda_dot-p.amplitude_rate_requested
    result=dict(t_au=float(t_au),norm=n,energy=energy(G),kinetic=kinetic(G),
        pnc_error=float(max(abs(np.sum(abs(state['C'])**2,axis=1)-1))),
        projection_residual=residual,amplitude_residual=float(np.sqrt(w@error**2)),
        support_residual=float(np.sqrt(w[support]@np.sum(abs(delta[support])**2,axis=1))),
        tail_residual=float(np.sqrt(w[~support]@np.sum(abs(delta[~support])**2,axis=1))),
        P15_error=float(np.max(abs(delta-error[:,None]*state['C']))),power=power,
        residual_integral=float(state['residual_integral']),energy_integral=float(state['energy_integral']),
        condition_A=float(p.audit['condition_L']),rank=int(p.audit['rank']),
        parameter_rate_norm=float(np.linalg.norm(p.parameter_dot)),
        energy_correction_rate=float(p.audit.get('correction_amplitude_rate_norm',0.)),
        constraint_resolvability=float(p.audit.get('constraint_resolvability',np.nan)),
        constraint_covector_norm=float(p.audit.get('constraint_covector_norm',np.nan)),
        energy_covector_norm=float(p.audit.get('energy_covector_norm',np.nan)),
        high_k_mass=float(spectrum[abs(m.kq)>.6*max(abs(m.kq))].sum()),
        edge_mass=float(w[[0,-1]]@rho[[0,-1]]),G=G.copy(),rho=rho,
        requested_amplitude_rate=p.amplitude_rate_requested.copy(),projected_amplitude_rate=p.Lambda_dot.copy(),
        amplitude_residual_map=error.copy(),spectrum=spectrum,
        photon_number=float(.5*(m.omega*(w@(m.q**2*rho))/n+2*kinetic(G)/m.omega)-.5) if m.is_photon else np.nan)
    if reference is not None:
        ref,physical_ref=reference.states(t_au);refnorm=norm(ref)**2
        overlap=np.einsum('q,qk,qk->',w,ref.conj(),G)
        ref_rho=np.sum(abs(ref)**2,axis=1)
        result.update(reference_G=ref,reference_rho=ref_rho,reference_energy=energy(ref),
            reference_kinetic=kinetic(ref),state_error=norm(G-ref),
            phase_aligned_state_error=float(np.sqrt(max(0,n+refnorm-2*abs(overlap)))),
            fidelity=float(abs(overlap)**2/(n*refnorm)),
            amplitude_error=float(np.sqrt(w@(np.sqrt(rho)-np.sqrt(ref_rho))**2)),
            density_l1_error=float(w@abs(rho-ref_rho)),physical_target_state_error=norm(G-physical_ref),
            initial_condition_reference_distance=norm(ref-physical_ref))
    return result


def summarize(history,status):
    a={key:np.asarray([f[key] for f in history]) for key in history[0]}
    drift=a['energy']-a['energy'][0]
    summary=dict(status=status,samples=len(history),last_t_au=float(a['t_au'][-1]),
        max_energy_drift_Ha=float(max(abs(drift))),final_energy_drift_Ha=float(drift[-1]),
        max_power_budget_error_Ha=float(max(abs(drift-a['energy_integral']))),
        max_norm_error=float(max(abs(a['norm']-1))),max_pnc_error=float(max(a['pnc_error'])),
        max_projection_residual=float(max(a['projection_residual'])),
        final_integrated_residual=float(a['residual_integral'][-1]),
        max_condition_A=float(max(a['condition_A'])),max_parameter_rate=float(max(a['parameter_rate_norm'])),
        max_high_k_mass=float(max(a['high_k_mass'])),max_edge_mass=float(max(a['edge_mass'])),
        max_energy_constraint_power=float(max(abs(a['power']))),
        numerically_certified=False,energy_conservation_is_not_state_accuracy=True)
    if 'state_error' in a:
        summary.update(max_state_error=float(max(a['state_error'])),final_state_error=float(a['state_error'][-1]),
            max_amplitude_error=float(max(a['amplitude_error'])),final_fidelity=float(a['fidelity'][-1]),
            max_reference_energy_drift=float(max(abs(a['reference_energy']-a['reference_energy'][0]))),
            max_state_error_minus_residual_integral=float(max(a['state_error']-a['state_error'][0]-a['residual_integral'])),
            residual_bound_note='Continuous-time bound; RK quadrature and integration error remain. Not a discrete rigorous certificate.')
    warnings=[]
    if status=='FAILED':warnings.append('Requested interval was not completed; last accepted state only.')
    if summary['max_power_budget_error_Ha']>1e-6:
        warnings.append('Energy change is not resolved by integrated continuous-time power (difference >1e-6 Ha). Refine dt; energy constraint is not a post-step repair.')
    if summary.get('max_state_error_minus_residual_integral',0)>1e-5:
        warnings.append('Discrete state error exceeds integrated continuous-time residual by >1e-5. Time integration/quadrature is unresolved.')
    if summary['max_condition_A']>1e8:
        warnings.append('Gaussian tangent condition number exceeds 1e8; do not infer convergence from K alone.')
    summary['diagnostic_warnings']=warnings
    return a,summary
