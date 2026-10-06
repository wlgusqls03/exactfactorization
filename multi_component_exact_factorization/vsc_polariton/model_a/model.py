"""Gil--Lauvergnat--Agostini, JCP 161, 084112 (2024), Model A.

All units atomic. Psi shape (NR,Nq,2), electronic states are orthonormal
diabatic labels, not an electron grid. No DSE, bath, or fitted parameters.
The q factor missing in printed Eq.28 follows Eq.5 and author QML source.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Config:
    nr: int = 500
    nq: int = 500
    rmin: float = 0.
    rmax: float = 8.
    qmin: float = -10.
    qmax: float = 10.
    dt: float = .05
    mass: float = 20000.
    omega: float = .17
    g: float = .01
    k: float = .02
    a: float = 3.
    b: float = .01
    r1: float = 6.
    r2: float = 2.
    r3: float = 3.875
    delta: float = 0.
    z: float = 1.
    mu12: float = -1.
    sigma_r: float = .223
    sigma_q: float = 2.425

    def grids(self):
        """Periodic endpoint-excluded R,q grids (a0), no absorbing potential."""
        if min(self.nr,self.nq)<8 or self.dt<=0 or self.rmax<=self.rmin or self.qmax<=self.qmin:
            raise ValueError('Invalid numerical grid')
        return (np.linspace(self.rmin,self.rmax,self.nr,endpoint=False),
                np.linspace(self.qmin,self.qmax,self.nq,endpoint=False))


def potential(c, R, q, coupling=None):
    """Eq.27 + photon harmonic + Eq.5 LM: real matrix [NR,Nq,2,2], Ha.

    V12=b exp[-a(R-R3)^2]+omega*g*mu12*q, Vii includes Z*omega*g*q*R.
    Authors' QML uses minus electronic position Mu12=+1: same LM sign.
    """
    g=c.g if coupling is None else coupling
    R=np.asarray(R)[:,None];q=np.asarray(q)[None,:]
    v=np.zeros((R.size,q.size,2,2))
    common=.5*c.omega**2*q*q+c.z*c.omega*g*R*q
    v[...,0,0]=.5*c.k*(R-c.r1)**2+common
    v[...,1,1]=.5*c.k*(R-c.r2)**2+c.delta+common
    v[...,0,1]=v[...,1,0]=c.b*np.exp(-c.a*(R-c.r3)**2)+c.omega*g*c.mu12*q
    return v


def initial(c):
    """Bare adiabatic S1 * real Gaussian, zero momenta, [NR,Nq,2].

    Amplitude exp[-(R-2)^2/(2 sigma_R^2)-q^2/(2 sigma_q^2)]; density
    SD is sigma/sqrt(2). This gives photon vacuum for sigma_q=1/sqrt(w).
    Paper rounded widths retained. Bare-S1 preparation declared explicitly;
    exact author QD input is unavailable, so reproduction is not certified.
    """
    R,q=c.grids();v=potential(c,R,[0.],coupling=0)[:,0]
    _,u=np.linalg.eigh(v);u=u[...,1]
    # Off-diagonal bare b>0, choose positive first component continuously.
    u*=np.where(u[:,0]<0,-1.,1.)[:,None]
    wave=np.exp(-.5*((R[:,None]-2)/c.sigma_r)**2-.5*(q[None,:]/c.sigma_q)**2)
    psi=(wave[...,None]*u[:,None,:]).astype(complex)
    vol=(R[1]-R[0])*(q[1]-q[0]);psi/=np.sqrt(np.sum(abs(psi)**2)*vol)
    return psi


def derivative(u, spacing, axis, order=1, xp=np):
    """Periodic spectral D^order on native grid; used by TDSE and EF."""
    shape=[1]*u.ndim;shape[axis]=u.shape[axis]
    k=2*xp.pi*xp.fft.fftfreq(u.shape[axis],spacing)
    return xp.fft.ifft(xp.fft.fft(u,axis=axis)*(1j*k.reshape(shape))**order,axis=axis)


class Propagator:
    """Unitary second-order V/2--T--V/2 split, complex128 CPU/CuPy."""
    def __init__(self,c,xp=np):
        self.c,self.xp=c,xp;R,q=c.grids()
        self.dr,self.dq=R[1]-R[0],q[1]-q[0]
        self.v=xp.asarray(potential(c,R,q))
        val,vec=np.linalg.eigh(potential(c,R,q))
        self.U=xp.asarray(np.einsum('...ik,...k,...jk->...ij',vec,np.exp(-.5j*c.dt*val),vec))
        kr=2*xp.pi*xp.fft.fftfreq(c.nr,self.dr)
        kq=2*xp.pi*xp.fft.fftfreq(c.nq,self.dq)
        self.kinetic=kr[:,None]**2/(2*c.mass)+kq[None,:]**2/2
        self.phase=xp.exp(-1j*c.dt*self.kinetic)[...,None]

    def action(self,u):
        """H Psi, same FFT kinetic as split, [NR,Nq,2], Ha*wave."""
        xp=self.xp
        return xp.fft.ifftn(xp.fft.fftn(u,axes=(0,1))*self.kinetic[...,None],axes=(0,1))+xp.einsum('...ij,...j->...i',self.v,u)

    def step(self,u):
        xp=self.xp
        u=xp.einsum('...ij,...j->...i',self.U,u)
        u=xp.fft.ifftn(xp.fft.fftn(u,axes=(0,1))*self.phase,axes=(0,1))
        return xp.einsum('...ij,...j->...i',self.U,u)


def observables(u, p):
    """Scalar/marginal diagnostics; states S0/S1 are bare BO, not CBO/LP/UP."""
    xp=p.xp;c=p.c;R,q=c.grids();vol=p.dr*p.dq
    R=xp.asarray(R);q=xp.asarray(q);rho=xp.sum(abs(u)**2,axis=-1)
    ur=derivative(u,p.dr,0,xp=xp);uq=derivative(u,p.dq,1,xp=xp)
    jr=xp.sum((u.conj()*ur).imag,axis=-1)/c.mass
    vals,vec=np.linalg.eigh(potential(c,np.asarray(c.grids()[0]),[0.],0)[:,0])
    bo=xp.einsum('rjk,rqj->rqk',xp.asarray(vec),u)
    pop=xp.sum(abs(bo)**2,axis=(0,1))*vol
    norm=xp.sum(rho)*vol
    q2=xp.sum(q[None,:]**2*rho)*vol
    p2=xp.sum(abs(uq)**2)*vol
    nq=c.omega*q2/2
    npart=p2/(2*c.omega)
    er=(R<c.rmin+.32)|(R>c.rmax-.32)
    eq=(q<c.qmin+.8)|(q>c.qmax-.8)
    scalar=dict(norm=norm,energy=xp.sum(u.conj()*p.action(u)).real*vol,
                P_S0=pop[0],P_S1=pop[1],population_sum=xp.sum(pop),
                mean_R=xp.sum(rho*R[:,None])*vol,
                P_right=xp.sum(rho[R>=4])*vol,
                J_R4=xp.sum(jr[int(xp.argmin(abs(R-4)))])*p.dq,
                q2=q2,p2=p2,n_q=nq,n_p=npart,
                photon_energy=c.omega*(nq+npart),
                n_ph=nq+npart-.5*norm,
                R_edge=xp.sum(rho[er])*vol,q_edge=xp.sum(rho[:,eq])*vol)
    return {k:float(v) for k,v in scalar.items()}
