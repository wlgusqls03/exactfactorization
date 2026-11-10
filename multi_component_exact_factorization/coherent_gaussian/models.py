"""Hamiltonian providers for no-q-grid Gaussian propagation.

PEN reuses the existing two-state 2026 benchmark (not the real-electron PEN
production model). Its BO matrix elements are analytic; CBO uses local
Gaussian-pair Gauss--Hermite quadrature with ALL q derivative terms.
MCEF keeps a real x basis but uses an explicitly named LOCAL POLYNOMIAL
surrogate: the inherited unsmoothed 1/|q+9.5| pole is not integrable against
unrestricted nonzero Gaussians. We do not silently smooth that pole.
"""
from dataclasses import replace, dataclass
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import eigh
from ..pen_perspectives_2026.model import electronic, local_hamiltonian
from ..pen_perspectives_2026.resonance import paper_config
from multi_component_exact_factorization_cavity.core import frozen_potential, sine_kinetic
from .core import overlap, gaussian


class Pen:
    def __init__(self, model='nonadiabatic', resonance='on', g=.01, dse=False,
                 R=None, representation='bo', nquad=32, omega=None):
        self.c = paper_config(model, resonance, g=g, dse=dse)
        if omega is not None:
            self.c = replace(self.c, omega=float(omega))
        self.R = self.c.r2 if R is None else float(R)
        self.h, self.mu = (a[0] for a in electronic(self.c, np.array([self.R])))
        self.Ubo = eigh(self.h)[1]
        self.sigma, self.origin, self.mass = 1/np.sqrt(2*self.c.omega), 0., 1.
        self.ne = self.nprimitive = 2
        self.representation, self.nquad = representation, nquad
        if representation not in ('bo', 'diabatic', 'cbo'):
            raise ValueError('PEN electronic basis must be bo, diabatic, or cbo')
        self.name = 'pen-'+model
        self.energy_shift = float(eigh(self.h)[0][1]+self.c.omega/2)

    def potential(self, q):
        q = np.asarray(q)
        V = (self.h[None]+self.c.omega*self.c.g*q[:, None, None]*self.mu[None]
             +.5*self.c.omega**2*q[:, None, None]**2*np.eye(self.nprimitive))
        if self.c.dse:
            V = V+.5*self.c.g**2*(self.mu@self.mu)
        return V

    def basis(self, q):
        q = np.asarray(q)
        if self.representation != 'cbo':
            U = self.Ubo if self.representation == 'bo' else np.eye(self.ne)
            return np.broadcast_to(U, (len(q), self.ne, self.ne)), np.zeros((len(q), self.ne, self.ne))
        V = self.potential(q)
        dx, dz = V[:, 0, 1], (V[:, 0, 0]-V[:, 1, 1])/2
        theta = np.arctan2(dx, dz)
        # Global real gauge across q, not independent eigensolver signs.
        theta = np.where((dz < 0) & (theta < 0), theta+2*np.pi, theta)
        radius2 = dx*dx+dz*dz
        if np.min(radius2) < 1e-20:
            raise ValueError('CBO degeneracy: derivative basis undefined; use diabatic/BO')
        tq = dz*(self.c.omega*self.c.g*self.mu[0, 1])/radius2
        s, c = np.sin(theta/2), np.cos(theta/2)
        U = np.empty((len(q), 2, 2))
        U[:, 0, 0], U[:, 1, 0] = -s, c
        U[:, 0, 1], U[:, 1, 1] = c, s
        Uq = np.empty_like(U)
        Uq[:, :, 0] = -.5*tq[:, None]*U[:, :, 1]
        Uq[:, :, 1] = .5*tq[:, None]*U[:, :, 0]
        return U, Uq

    def matrices(self, state, *, quadrature=False):
        if self.representation == 'cbo' or quadrature:
            return quadrature_matrices(self, state)
        z = state.z
        S = overlap(z)
        U = self.Ubo if self.representation == 'bo' else np.eye(self.ne)
        h = self.h.copy()
        if self.c.dse:
            h += .5*self.c.g**2*(self.mu@self.mu)
        h = U.T@h@U+np.eye(self.ne)*(self.c.omega/2-self.energy_shift)
        L = self.c.omega*self.c.g*self.sigma*(U.T@self.mu@U)
        left, right = z.conj()[:, None], z[None, :]
        ordered = (h[None, :, None, :]
            +self.c.omega*(left*right)[:, None, :, None]*np.eye(self.ne)[None, :, None, :]
            +(left+right)[:, None, :, None]*L[None, :, None, :])
        H00 = S[:, None, :, None]*ordered
        H10 = S[:, None, :, None]*(right[:, None, :, None]*ordered
            +self.c.omega*right[:, None, :, None]*np.eye(self.ne)[None, :, None, :]
            +L[None, :, None, :])
        return H00, H10

    def initial_physical(self, q):
        return gaussian(q, np.array([0j]), self.sigma)[0]*self.Ubo[:, 1][None]

    def reference(self, nf=32):
        return local_hamiltonian(replace(self.c, nf=nf), np.array([self.R]))[0]


@dataclass
class RealPenParameters:
    nx: int
    omega: float
    g: float
    dse: bool


class RealPen(Pen):
    """Inherited full real-electronic PEN at fixed proton R, no 2-state model.

    All nx electronic channels are retained in both BO and CBO tests. The
    nuclear Coulomb pole is a finite constant at fixed R != -9.5; photon q
    has no such pole, so no molecular surrogate is needed in this provider.
    """
    def __init__(self,R=-4.,nx=17,g=None,dse=True,representation='bo',nquad=32,omega=None):
        if representation not in ('bo','cbo','diabatic'):
            raise ValueError('Real PEN uses bo/cbo/diabatic (real x primitive)')
        self.R,self.ne,self.nprimitive=float(R),nx,nx
        self.dx=52/(nx+1);self.x=-22+self.dx*np.arange(1,nx+1)
        Tx,_=sine_kinetic(nx,self.dx)
        self.h=Tx+np.diag(frozen_potential(self.x,np.array([R]),dict(heavy_fixed=9.5,heavy_trap_alpha=0.))[0])
        E,self.Ubo=eigh(self.h)
        self.mu=np.diag(R-self.x)  # fixed sites -9.5,+9.5 cancel once, not omitted twice
        freq=float(E[1]-E[0]) if omega is None else float(omega)
        dip=abs(self.Ubo[:,0]@self.mu@self.Ubo[:,1])
        coupling=.1*np.sqrt(freq/2)/dip if g is None else float(g)
        self.c=RealPenParameters(nx,freq,coupling,dse)
        self.sigma,self.origin,self.mass=1/np.sqrt(2*freq),0.,1.
        self.representation,self.nquad=representation,nquad
        self.energy_shift=float(E[1]+freq/2)
        self.name='pen-real-electron'
        self._gauge_step=self.sigma/4
        self._anchors={0:eigh(self.potential(np.array([0.]))[0])[1]}

    def _transport(self,qa,Ua,qb,depth=0):
        """Local sign transport, not a propagated wavefunction grid.

        Deterministic electronic gauge anchors are shared by all quadratures.
        Refinement of this chart does not add nuclear/photon state variables.
        """
        U=eigh(self.potential(np.array([qb]))[0])[1]
        ov=np.sum(Ua*U,axis=0)
        if min(abs(ov))<.5:
            if depth>15:raise ValueError('CBO gauge transport unresolved near degeneracy')
            qm=(qa+qb)/2
            return self._transport(qm,self._transport(qa,Ua,qm,depth+1),qb,depth+1)
        return U*np.where(ov<0,-1.,1.)[None]

    def _chart(self,q):
        indices=np.rint(q/self._gauge_step).astype(int)
        for end in (int(min(indices)),int(max(indices))):
            direction=1 if end>=0 else -1
            for j in range(direction,end+direction,direction):
                if j not in self._anchors:
                    self._anchors[j]=self._transport((j-direction)*self._gauge_step,
                        self._anchors[j-direction],j*self._gauge_step)
        return np.stack([self._anchors[j] for j in indices])

    def basis(self,q):
        q=np.asarray(q)
        if self.representation!='cbo':return super().basis(q)
        E,U=np.linalg.eigh(self.potential(q))
        anchor=self._chart(q)
        for k in range(self.ne):
            ov=np.sum(U[:,:,k]*anchor[:,:,k],axis=1)
            if np.min(abs(ov))<.2:raise ValueError('Real-PEN CBO chart too coarse; refine gauge anchors')
            U[:,:,k]*=np.where(ov<0,-1.,1.)[:,None]
        vq=self.c.omega*self.c.g*self.mu
        mat=np.einsum('nxa,xy,nyb->nab',U,vq,U)
        gap=E[:,None,:]-E[:,:,None];np.einsum('nii->ni',gap)[:]=np.inf
        if np.any(abs(gap)<1e-10):raise ValueError('CBO degenerate derivative subspace')
        return U,np.einsum('nxa,nab->nxb',U,mat/gap)

    def reference(self,nf=32):
        a=np.diag(np.sqrt(np.arange(1,nf)),1)
        v=np.kron(self.h,np.eye(nf))+np.kron(np.eye(self.ne),self.c.omega*(a.T@a+.5*np.eye(nf)))
        v+=self.c.omega*self.c.g*self.sigma*np.kron(self.mu,a+a.T)
        if self.c.dse:v+=.5*self.c.g**2*np.kron(self.mu@self.mu,np.eye(nf))
        return v


class McefLocal:
    """Declared local surrogate of the inherited electron--proton--heavy model.

    Full x grid, frozen R. V(x,q,R) is least-squares polynomial-expanded in q
    about q=-4 inside a pole-free interval. It is NOT a replacement for full
    real-space MCEF, and it is NOT a CBO model (there is no cavity).
    """
    def __init__(self, R=9.5, nx=33, degree=6, radius=3., representation='primitive',
                 nstates=None, nquad=32):
        self.R, self.origin, self.sigma, self.mass = R, -4., .3208418789467482, 1836.
        self.nprimitive, self.representation, self.nquad = nx, representation, nquad
        self.degree, self.radius = degree, radius
        if not 0 < radius < 5.5:
            raise ValueError('MCEF local fit must not include the bare q=-9.5 pole')
        self.dx = 52/(nx+1)
        self.x = -22+self.dx*np.arange(1, nx+1)
        self.Tx, _ = sine_kinetic(nx, self.dx)
        self.ne = nx if representation == 'primitive' else (nstates or nx)
        if representation not in ('primitive', 'bo') or not 1 <= self.ne <= nx:
            raise ValueError('MCEF basis must be primitive or bo with 1<=nstates<=nx')
        # Least-squares polynomial, not a claimed exact analytic Taylor series.
        nodes = np.cos(np.pi*(np.arange(max(33, 4*degree+1))+.5)/max(33, 4*degree+1))
        V = self.original_potential(self.origin+radius*nodes)
        self.poly = np.polynomial.polynomial.polyfit(nodes, V, degree)
        self.anchor = eigh(self.Tx+np.diag(self.original_potential(np.array([self.origin]))[0]))[1]
        self.energy_shift = float(eigh(self.potential(np.array([self.origin]))[0])[0][1])
        self.name = 'mcef-local-polynomial'

    def original_potential(self, q):
        return frozen_potential(self.x, q, dict(heavy_fixed=self.R, heavy_trap_alpha=0.))

    def potential(self, q):
        v = np.polynomial.polynomial.polyval((np.asarray(q)-self.origin)/self.radius, self.poly).T
        return self.Tx[None]+v[:, :, None]*np.eye(self.nprimitive)

    def basis(self, q):
        q = np.asarray(q)
        if self.representation == 'primitive':
            return np.broadcast_to(np.eye(self.ne), (len(q), self.ne, self.ne)), np.zeros((len(q), self.ne, self.ne))
        V = self.potential(q)
        e, U = np.linalg.eigh(V)
        # Stable reference gauge in the LOCAL domain; fail at ambiguous overlap.
        for k in range(self.ne):
            ov = U[:, :, k]@self.anchor[:, k]
            if np.min(abs(ov)) < 1e-5:
                raise ValueError('MCEF BO gauge leaves local anchor chart')
            U[:, :, k] *= np.where(ov < 0, -1., 1.)[:, None]
        vp = np.polynomial.polynomial.polyval((q-self.origin)/self.radius,
                    np.polynomial.polynomial.polyder(self.poly)/self.radius).T
        mat = np.einsum('nxa,nx,nxb->nab', U, vp, U)
        gap = e[:, None, :]-e[:, :, None]
        np.einsum('nii->ni', gap)[:] = np.inf
        if np.any(abs(gap) < 1e-10):
            raise ValueError('MCEF BO degenerate derivative subspace; use primitive')
        Up = np.einsum('nxa,nab->nxb', U, mat/gap)
        return U[:, :, :self.ne], Up[:, :, :self.ne]

    def matrices(self, state, **kwargs):
        return quadrature_matrices(self, state)

    def initial_physical(self, q):
        e, U = np.linalg.eigh(self.potential(q))
        u = U[:, :, 1]
        u *= np.where(u@self.anchor[:, 1] < 0, -1., 1.)[:, None]
        return gaussian(q, np.array([0j]), self.sigma, self.origin)[0]*u


def quadrature_matrices(model, state):
    """Local pair quadrature of weak kinetic form, including U_q terms.

    No global q grid and no spatial finite differences: points follow each
    Gaussian pair; only K/ne coefficients and Gaussian parameters propagate.
    Integral <d_q(U g_i)|d_q(U g_j)>/(2m) includes derivative couplings
    and the positive U_q^dagger U_q term even for a truncated BO space.
    """
    z, sigma, origin = state.z, model.sigma, model.origin
    K, ne = len(z), model.ne
    nodes, weights = hermgauss(model.nquad)
    Q, P = 2*sigma*z.real, z.imag/sigma
    mid = (Q[:, None]+Q[None, :])/2
    x = mid[:, :, None]+np.sqrt(2)*sigma*nodes
    q = x+origin
    w = (weights[None, None, :]/np.sqrt(np.pi)*np.exp(
        -(Q[:, None, None]-Q[None, :, None])**2/(8*sigma*sigma)
        +1j*(P[None, :, None]-P[:, None, None])*x
        -.5j*(P[None, :, None]*Q[None, :, None]-P[:, None, None]*Q[:, None, None])))
    U, Uq = model.basis(q.ravel())
    V = model.potential(q.ravel())
    A = np.einsum('nxa,nxb->nab', U.conj(), Uq).reshape(K,K,-1,ne,ne)
    B = np.einsum('nxa,nxb->nab', Uq.conj(), Uq).reshape(K,K,-1,ne,ne)
    Vc = np.einsum('nxa,nxy,nyb->nab', U.conj(), V, U).reshape(K,K,-1,ne,ne)
    Vc -= model.energy_shift*np.eye(ne)
    li = -(x-Q[:,None,None])/(2*sigma*sigma)+1j*P[:,None,None]
    lj = -(x-Q[None,:,None])/(2*sigma*sigma)+1j*P[None,:,None]
    ti = x/sigma-z[:,None,None]
    tpi = 1/sigma+ti*li
    Id = np.eye(ne)
    Ad = np.swapaxes(A.conj(), -1, -2)
    def matrix(f, fp):
        kinetic = (fp.conj()[...,None,None]*lj[...,None,None]*Id
            + fp.conj()[...,None,None]*A
            + f.conj()[...,None,None]*lj[...,None,None]*Ad
            + f.conj()[...,None,None]*B)/(2*model.mass)
        value = kinetic+f.conj()[...,None,None]*Vc
        return np.einsum('ijn,ijnab->iajb', w, value)
    return matrix(np.ones_like(x), li), matrix(ti, tpi)
