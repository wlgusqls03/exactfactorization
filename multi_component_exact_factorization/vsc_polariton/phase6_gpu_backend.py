"""Portable float64/complex128 PF backend; no imports from historical VSC code.

Input is an audited numerical packet from the existing CPU Hamiltonian.
Psi=(R,x,n), physical quadrature dR*dx, all physical quantities atomic units.
CuPy is imported ONLY when a GPU backend is explicitly requested.
"""
import numpy as np


class PFBackend:
    def __init__(self, packet, dt, gpu=False, device=0):
        if gpu:
            import cupy as xp
            xp.cuda.Device(device).use()
            from cupyx.scipy import fft as fftlib
        else:
            xp=np
            from scipy import fft as fftlib
        self.xp,self.fft,self.gpu=xp,fftlib,gpu
        self.dt=float(dt)
        self.R=np.asarray(packet['R']);self.x=np.asarray(packet['x'])
        self.dx=float(packet['dx']);self.dR=float(packet['dR'])
        self.mass=float(packet['mass']);self.omega=float(packet['omega'])
        self.g=float(packet['g_chi']);self.volume=self.dx*self.dR
        self.shape=packet['psi'].shape;self.nf=self.shape[2]
        for key in ('tx','tr','potential','dse','mu','photon','rotation','displacement','phi'):
            setattr(self,key,xp.asarray(packet[key],dtype=xp.float64))
        self.coupling=self.g*self.mu[:,:,None]*xp.sqrt(xp.arange(1,self.nf))[None,None,:]
        kinetic=self.tr[:,None,None]+self.tx[None,:,None]+self.photon[None,None,:]
        values=(self.potential+self.dse)[:,:,None]+self.g*self.mu[:,:,None]*self.displacement
        w=1/(2-2**(1/3));self.weights=(w,-2**(1/3)*w,w)
        self.phases={v:(xp.exp(-.5j*dt*v*kinetic),xp.exp(-1j*dt*v*values)) for v in set(self.weights)}

    def host(self, value):
        return self.xp.asnumpy(value) if self.gpu else np.asarray(value)

    def sync(self):
        if self.gpu:self.xp.cuda.get_current_stream().synchronize()

    def step(self, u):
        """Same K/2-W-K/2 Yoshida composition as phase6_split.FullSplit, not reduced H."""
        for weight in self.weights:
            k,v=self.phases[weight]
            u=self.fft.ifftn(self.fft.fftn(u,axes=(0,1))*k,axes=(0,1))
            u=(u.reshape(-1,self.nf)@self.rotation).reshape(self.shape)
            u*=v
            u=(u.reshape(-1,self.nf)@self.rotation.T).reshape(self.shape)
            u=self.fft.ifftn(self.fft.fftn(u,axes=(0,1))*k,axes=(0,1))
        return u

    def parts(self,u):
        """Generator avoids holding six full H-part arrays simultaneously."""
        yield self.fft.ifft(self.tx[None,:,None]*self.fft.fft(u,axis=1),axis=1)
        yield self.fft.ifft(self.tr[:,None,None]*self.fft.fft(u,axis=0),axis=0)
        yield self.potential[:,:,None]*u
        yield self.photon[None,None,:]*u
        lm=self.xp.zeros_like(u)
        lm[:,:,:-1]+=self.coupling*u[:,:,1:]
        lm[:,:,1:]+=self.coupling*u[:,:,:-1]
        yield lm
        yield self.dse[:,:,None]*u

    def action(self,u):
        out=self.xp.zeros_like(u)
        for part in self.parts(u):out+=part
        return out

    def observe(self,u):
        """Same observables as historical Phase6, calculated on device; small outputs to CPU."""
        xp=self.xp
        rho=xp.sum(abs(u)**2,axis=(1,2))*self.dx
        rhox=xp.sum(abs(u)**2,axis=(0,2))*self.dR
        k=xp.asarray(2*np.pi*np.fft.fftfreq(len(self.R),self.dR))
        du=self.fft.ifft(1j*k[:,None,None]*self.fft.fft(u,axis=0),axis=0)
        current=xp.sum((u.conj()*du).imag,axis=(1,2))*self.dx/self.mass
        hp=xp.zeros_like(u);energies=[]
        for part in self.parts(u):
            hp+=part;energies.append(complex(self.host(xp.vdot(u,part)*self.volume)))
        rho_dot=2*xp.sum((u.conj()*hp).imag,axis=(1,2))*self.dx
        coefficients=xp.einsum('rxj,rxn->rjn',self.phi,u)*self.dx
        local=xp.sum(abs(coefficients)**2,axis=2)
        pops=xp.sum(local,axis=0)*self.dR
        photon=xp.sum(abs(u)**2,axis=(0,1))*self.volume
        ladder=xp.sum(u[:,:,:-1].conj()*u[:,:,1:]*xp.sqrt(xp.arange(1,self.nf))[None,None,:])*self.volume
        r=self.host(rho);rx=self.host(rhox);j=self.host(current);p=self.host(pops)
        pp=self.host(photon);ladder=complex(self.host(ladder))
        norm=float(r.sum()*self.dR);flux=float(np.interp(0,self.R,j))
        dp=float(self.host(xp.sum(rho_dot[xp.asarray(self.R>0)])*self.dR))
        energy=complex(self.host(xp.vdot(u,hp)*self.volume))
        return dict(rho_R=r,rho_x=rx,current=j,BO_populations=p,BO_local=self.host(local),
            P_exc=float(1-p[0]),BO_projection_remainder=float(norm-p.sum()),norm=norm,
            product=float(r[self.R>0].sum()*self.dR),flux=flux,dproduct_exact=dp,
            continuity_error=dp-flux,mean_R=float(self.R@r*self.dR),
            mean_R2=float((self.R**2)@r*self.dR),nph=float(np.arange(self.nf)@pp),
            q=float(np.sqrt(2/self.omega)*ladder.real),p=float(np.sqrt(2*self.omega)*ladder.imag),
            top=0. if self.nf==1 and self.g==0 else float(pp[-min(5,self.nf):].sum()),
            edge=float(r[abs(self.R)>max(abs(self.R))-.3].sum()*self.dR),
            electron_edge=float(rx[abs(self.x)>.9*max(abs(self.x))].sum()*self.dx),
            energy_parts=np.array(energies).real,imaginary_energy_parts=np.array(energies).imag,
            energy=energy.real,imaginary_energy=energy.imag)
