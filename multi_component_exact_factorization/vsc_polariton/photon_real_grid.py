"""Full PF on periodic (R,x,Q) grids; Q=sqrt(omega)*q, hbar=1.

H = T_R + T_x - omega/2 d_Q^2 + V_mol(x,R)
    + omega/2 [Q + sqrt(2)*g_chi*(R-x)/omega]^2.
The full (R-x)^2 DSE is retained. No BO projection in propagation.
Input: the unchanged, validated Phase6 packet. Output wave shape (NR,Nx,NQ),
normalized with dR*dx*dQ. Physical q-space wave = omega**.25 * wave_Q.

Memory: two full complex potential half-kicks, 1D kinetic phases, no dense
photon rotations, no full kinetic tensor, no per-step host transfer.
The fourth-order V/2-T-V/2 Yoshida split is NOT the old Fock split; timestep
convergence must be re-established. Photon FFT assumes periodic endpoints;
the physical oscillator must be negligible there, tested independently.
"""
import numpy as np


def hermite_basis(Q, nf):
    """Dimensionless normalized Hermite functions, array (nf,NQ)."""
    B=np.empty((nf,len(Q)),float)
    B[0]=np.pi**(-.25)*np.exp(-Q**2/2)
    if nf>1:B[1]=np.sqrt(2)*Q*B[0]
    for n in range(1,nf-1):
        B[n+1]=np.sqrt(2/(n+1))*Q*B[n]-np.sqrt(n/(n+1))*B[n-1]
    return B


def initial_grid(packet, Q, block=4):
    """Lossless-to-tolerance Fock-to-grid conversion, NO renormalization.

    The blockwise back-projection checks actual occupied coefficients rather
    than only the vacuum. No new Gaussian or photon initial state is chosen.
    """
    h=float(Q[1]-Q[0]);c=packet['psi'];B=hermite_basis(Q,c.shape[-1])
    out=np.empty(c.shape[:2]+(len(Q),),complex)
    err2=0.
    for first in range(0,len(c),block):
        sl=slice(first,first+block)
        out[sl]=c[sl]@B
        err2+=float(np.sum(abs(out[sl]@B.T*h-c[sl])**2))
    volume=float(packet['dx'])*float(packet['dR'])
    old=np.sum(abs(c)**2,axis=-1)
    new=np.sum(abs(out)**2,axis=-1)*h
    error=dict(norm_difference=float(abs(np.sum(new-old)*volume)),
        backprojection_L2=float(np.sqrt(err2*volume)),
        joint_Rx_L1=float(np.sum(abs(new-old))*volume))
    if any(not np.isfinite(v) for v in error.values()):raise ValueError('Nonfinite initial transform')
    return out,error


class RealGridPF:
    """CPU/SciPy and GPU/CuPy implementations of the same grid Hamiltonian."""
    def __init__(self,packet,half_Q=24.,nq=384,dt=.125,gpu=False,device=0):
        if half_Q<=0 or nq<8 or dt<=0:raise ValueError('Invalid grid or timestep')
        if gpu:
            import cupy as xp
            xp.cuda.Device(device).use()
            from cupyx.scipy import fft as fftlib
        else:
            xp=np
            from scipy import fft as fftlib
        self.xp,self.fft,self.gpu=xp,fftlib,gpu
        self.R=np.asarray(packet['R']);self.x=np.asarray(packet['x'])
        self.dx=float(packet['dx']);self.dr=float(packet['dR'])
        self.mass=float(packet['mass']);self.omega=float(packet['omega']);self.g=float(packet['g_chi'])
        self.dt=float(dt);self.half_Q=float(half_Q)
        self.Q=np.linspace(-half_Q,half_Q,nq,endpoint=False);self.dQ=2*half_Q/nq
        self.shape=(len(self.R),len(self.x),nq);self.volume=self.dx*self.dr*self.dQ
        if gpu:
            free,total=xp.cuda.runtime.memGetInfo()
            estimate=14*np.prod(self.shape)*16+512*1024**2
            if estimate+1024**3>free:
                raise MemoryError(f'Conservative grid working estimate {estimate/2**30:.2f} GiB + 1 GiB reserve exceeds free VRAM {free/2**30:.2f} GiB')
        self.q=xp.asarray(self.Q)
        self.kr=xp.asarray(2*np.pi*np.fft.fftfreq(len(self.R),self.dr))
        self.kx=xp.asarray(2*np.pi*np.fft.fftfreq(len(self.x),self.dx))
        self.kQ=xp.asarray(2*np.pi*np.fft.fftfreq(nq,self.dQ))
        self.tr=self.kr**2/(2*self.mass);self.tx=self.kx**2/2;self.tQ=self.omega*self.kQ**2/2
        # Enforce unchanged native electron/nuclear kinetic discretization.
        np.testing.assert_allclose(self.host(self.tr),packet['tr'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(self.host(self.tx),packet['tx'],rtol=1e-12,atol=1e-12)
        mu=self.R[:,None]-self.x[None,:]
        np.testing.assert_allclose(mu,packet['mu'],rtol=1e-12,atol=1e-12)
        self.mu=xp.asarray(mu);self.vm=xp.asarray(packet['potential'])
        self.dse=self.g**2*self.mu**2/self.omega
        np.testing.assert_allclose(self.host(self.dse),packet['dse'],rtol=1e-12,atol=1e-12)
        self.phi=xp.asarray(packet['phi'])
        self.V=self.vm[:,:,None]+self.omega/2*(self.q[None,None,:]+np.sqrt(2)*self.g/self.omega*self.mu[:,:,None])**2
        w=1/(2-2**(1/3));self.weights=(w,-2**(1/3)*w,w)
        self.phases={a:(xp.exp(-.5j*dt*a*self.V),
                       tuple(xp.exp(-1j*dt*a*k) for k in (self.tr,self.tx,self.tQ)))
                     for a in set(self.weights)}

    def host(self,a):return self.xp.asnumpy(a) if self.gpu else np.asarray(a)

    def sync(self):
        if self.gpu:self.xp.cuda.get_current_stream().synchronize()

    def step(self,u):
        """In-place half-kicks, three 3D FFT pairs per fourth-order step."""
        for a in self.weights:
            v,(kr,kx,kq)=self.phases[a]
            u*=v
            f=self.fft.fftn(u,axes=(0,1,2),overwrite_x=True)
            f*=kr[:,None,None];f*=kx[None,:,None];f*=kq[None,None,:]
            u=self.fft.ifftn(f,axes=(0,1,2),overwrite_x=True)
            u*=v
        return u

    def action(self,u):
        """Instantaneous H Psi; preserves its input, units Ha*wave_Q."""
        f=self.fft.fftn(u,axes=(0,1,2))
        kinetic=self.tr[:,None,None]+self.tx[None,:,None]+self.tQ[None,None,:]
        f*=kinetic
        del kinetic
        out=self.fft.ifftn(f,axes=(0,1,2),overwrite_x=True)
        out+=self.V*u
        return out

    def observe(self,u):
        """Physical marginals, currents and full energy; transfer only small arrays.

        q,p are physical mass-one photon coordinates. Photon number is bare
        oscillator occupation. BO populations mean bare-BO character, NOT an
        attribution solely to nuclear nonadiabatic transitions.
        """
        xp=self.xp
        rho=abs(u)**2
        rx=xp.sum(rho,axis=2)*self.dQ
        r=xp.sum(rx,axis=1)*self.dx
        x=xp.sum(rx,axis=0)*self.dr
        q=xp.sum(rho,axis=(0,1))*self.dx*self.dr
        qR=xp.sum(rho*self.q[None,None,:],axis=(1,2))*self.dx*self.dQ/np.sqrt(self.omega)
        lm=float(self.host(xp.sum(rho*self.mu[:,:,None]*self.q[None,None,:])*self.volume))*np.sqrt(2)*self.g
        del rho
        f=self.fft.fftn(u,axes=(0,1,2));spectral=abs(f)**2/np.prod(self.shape)
        del f
        spectrumR=xp.sum(spectral,axis=(1,2))*self.volume
        spectrumX=xp.sum(spectral,axis=(0,2))*self.volume
        spectrumQ=xp.sum(spectral,axis=(0,1))*self.volume
        del spectral
        scalar=lambda a:float(self.host(a))
        norm=scalar(xp.sum(r)*self.dr)
        tR=scalar(spectrumR@self.tr);tx=scalar(spectrumX@self.tx)
        tq=scalar(spectrumQ@self.tQ);q2=scalar(xp.sum(q*self.q**2)*self.dQ)
        en=scalar(xp.sum(rx*self.vm)*self.dx*self.dr)
        dse=scalar(xp.sum(rx*self.dse)*self.dx*self.dr)
        hph=tq+self.omega*q2/2
        fr=self.fft.fft(u,axis=0)
        du=self.fft.ifft(fr*(1j*self.kr[:,None,None]),axis=0)
        current=xp.sum((u.conj()*du).imag,axis=(1,2))*self.dx*self.dQ/self.mass
        del du
        tr_u=self.fft.ifft(fr*self.tr[:,None,None],axis=0)
        del fr
        rho_dot=2*xp.sum((u.conj()*tr_u).imag,axis=(1,2))*self.dx*self.dQ
        del tr_u
        coeff=xp.einsum('rxj,rxq->rjq',self.phi.conj(),u)*self.dx
        local=xp.sum(abs(coeff)**2,axis=2)*self.dQ
        del coeff
        pops=xp.sum(local,axis=0)*self.dr
        rhost=self.host(r);jhost=self.host(current)
        flux=float(np.interp(0,self.R,jhost))
        dp=scalar(xp.sum(rho_dot[xp.asarray(self.R>0)])*self.dr)
        pieces=np.array([tx,tR,en,hph,lm,dse])
        conditional=xp.full_like(r,xp.nan)
        occupied=r>0
        conditional[occupied]=qR[occupied]/r[occupied]
        return dict(norm=norm,energy=float(pieces.sum()),energy_parts=pieces,
            rho_R=rhost,rho_x=self.host(x),rho_Q=self.host(q),current=jhost,
            conditional_q=self.host(conditional),
            product=float(np.sum(rhost[self.R>0])*self.dr),flux=flux,
            dproduct_exact=dp,continuity_error=dp-flux,
            mean_R=float(self.R@rhost*self.dr),mean_R2=float(self.R**2@rhost*self.dr),
            nph=hph/self.omega-.5*norm,
            q=scalar(xp.sum(q*self.q)*self.dQ)/np.sqrt(self.omega),
            p=scalar(spectrumQ@self.kQ)*np.sqrt(self.omega),
            BO_populations=self.host(pops),BO_local=self.host(local),
            P_exc=norm-scalar(pops[0]),BO_projection_remainder=norm-scalar(xp.sum(pops)),
            electron_edge=scalar(xp.sum(x[xp.asarray(abs(self.x)>.9*max(abs(self.x)))])*self.dx),
            edge=float(rhost[abs(self.R)>max(abs(self.R))-.3].sum()*self.dr),
            photon_edge=scalar(xp.sum(q[xp.asarray(abs(self.Q)>=self.half_Q-1)])*self.dQ))
