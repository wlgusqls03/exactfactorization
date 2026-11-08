"""Cached static Hamiltonian and current-state RHS; no reference feedback.

This experiment deliberately has NO heavy trajectories, forces, MLS, hopping,
or CTMQC quantum momentum. Its energy constraint is not their closure.
"""
import numpy as np
from scipy.linalg import eigh

from ..gaussian_trajectory.gaussian import build_bridge, split_and_project
from ..gaussian_trajectory.models import InternalModel, ModelConfig
from .projection import energy_constrained_rate


class FixedProblem:
    def __init__(self, model_config, R=None):
        self.model=InternalModel(ModelConfig(**model_config))
        m=self.model
        self.R=float(m.initial_R_center if R is None else R)
        if not np.isfinite(self.R):raise ValueError('Finite fixed R required')
        self.q,self.w=m.q,m.quadrature
        self.V=m.potential(np.array([self.R]))[0]
        # Cached once, not re-diagonalized or re-evaluated in every RK stage.
        self.Tq=m.kinetic_q_matrix()
        blocks=[self.Tq,self.V] if m.is_photon else [self.Tq,m.Tx]
        self.hermitian_error=max(float(np.max(abs(a-np.swapaxes(a.conj(),-1,-2)))) for a in blocks)
        if self.hermitian_error>1e-12:raise ValueError('Fixed Hamiltonian is not Hermitian')
        self.C0=m.initial_coefficients(np.array([self.R]))[0]
        self.physical_G0=m.initial(np.array([self.R]))[0]
        check=m.action(np.array([self.R]),self.physical_G0[None])[0]
        self.cached_action_error=float(np.max(abs(self.action(self.physical_G0)-check)))
        if self.cached_action_error>1e-12:raise ValueError('Cached action differs from original model')

    def action(self,G):
        m=self.model
        kinetic=m.kinetic_q_action(G[None])[0]
        if m.is_photon:return kinetic+np.einsum('qkl,ql->qk',self.V,G)
        return kinetic+G@m.Tx.T+self.V*G

    def bridge(self,z,K):
        return build_bridge(self.q,self.w,z,components=K,coordinates='transformed')

    def rate(self,state,K,mode,rank_rtol=1e-12,constraint_rtol=1e-10,power_atol=1e-12):
        b=self.bridge(state['z'],K);C=state['C'];G=b.amplitude[:,None]*C
        HG=self.action(G)
        if mode=='standard':
            p=split_and_project(-1j*HG,C,b,rank_rtol=rank_rtol,rank_policy='raise',
                                pnc_tolerance=1e-10)
        elif mode=='energy':
            p=energy_constrained_rate(HG,C,b,rank_rtol=rank_rtol,
                                     constraint_rtol=constraint_rtol,power_atol=power_atol)
        else:raise ValueError('Select standard or experimental energy projection')
        norm=self.w@np.sum(abs(G)**2,axis=1)
        E=np.real(np.einsum('q,qk,qk->',self.w,G.conj(),HG))/norm
        delta=p.G_dot_actual+1j*HG
        residual=float(np.sqrt(self.w@np.sum(abs(delta)**2,axis=1)))
        power=float(2*np.real(np.einsum('q,qk,qk->',self.w,delta.conj(),HG-E*G))/norm)
        # Passive RK quadratures. These never enter the physical C,z rates.
        derivative=dict(C=p.C_dot,z=p.parameter_dot,
                        residual_integral=np.array(residual),energy_integral=np.array(power))
        return derivative,(b,G,HG,p,power,residual)

    def dense_hamiltonian(self,max_dimension=2048):
        m=self.model;n=len(self.q)*m.nk
        if n>max_dimension:
            raise ValueError('Dense matched reference dimension %d > %d; use --no-reference or explicitly raise --reference-max-dimension'%(n,max_dimension))
        H=np.kron(self.Tq,np.eye(m.nk)).astype(complex)
        if m.is_photon:
            for j,v in enumerate(self.V):H[j*m.nk:(j+1)*m.nk,j*m.nk:(j+1)*m.nk]+=v
        else:H+=np.kron(np.eye(len(self.q)),m.Tx)+np.diag(self.V.ravel())
        rng=np.random.default_rng(801)
        G=rng.normal(size=self.C0.shape)+1j*rng.normal(size=self.C0.shape)
        np.testing.assert_allclose((H@G.ravel()).reshape(G.shape),self.action(G),rtol=1e-12,atol=1e-12)
        return H


class MatchedReference:
    """Same finite Hermitian H, each K's actual initial state. Diagnostic only."""
    def __init__(self,problem,G0,max_dimension=2048):
        self.shape=G0.shape
        self.values,self.vectors=eigh(problem.dense_hamiltonian(max_dimension))
        self.initial=self.vectors.conj().T@G0.ravel()
        self.physical_initial=self.vectors.conj().T@problem.physical_G0.ravel()

    def states(self,t_au):
        phase=np.exp(-1j*self.values*t_au)
        return tuple((self.vectors@(phase*c)).reshape(self.shape)
                     for c in (self.initial,self.physical_initial))
