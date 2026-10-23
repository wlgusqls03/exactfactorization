"""Bounded-memory Strang splitting; historical DST electron + FFT nuclei.

Axis order is (R, x, Q), Q=sqrt(omega)*q_c. No electronic basis truncation
enters propagation. Static BO/product bases are strictly postprocessing.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sf


def memory_estimate(shape, block=8):
    """Conservative per-process device reservation, including FFT workspace.

Not a promise about a vendor FFT plan: allocation errors still stop safely.
CPU field extraction needs more RAM, which is estimated separately.
"""
    wave = int(np.prod(shape))*16
    slab = min(block, shape[0])*shape[1]*shape[2]*16
    return dict(wave_bytes=wave, gpu_reservation_bytes=7*wave+10*slab+256*2**20,
                host_reservation_bytes=18*wave+512*2**20)


class Hamiltonian:
    def __init__(self, packet, Q, omega, g_cavity, *, gpu=False, dse=True,
                 block=8, workers=1):
        self.packet, self.Q = packet, np.asarray(Q)
        self.omega, self.g, self.dse = float(omega), float(g_cavity), bool(dse)
        self.block, self.workers = int(block), int(workers)
        self.gpu = bool(gpu)
        if gpu:
            import cupy as cp
            self.xp = cp
            free, total = cp.cuda.runtime.memGetInfo()
            need = memory_estimate((len(packet['R']), len(packet['x']), len(Q)), block)
            if need['gpu_reservation_bytes'] > .85*free:
                raise MemoryError('Insufficient free GPU memory for conservative reservation: '+str(need))
        else:
            self.xp = np
        xp = self.xp
        self.V = xp.asarray(packet['V'])
        self.mu = xp.asarray(packet['mu'])
        self.qgrid = xp.asarray(Q)[None, None, :]
        self.dQ = float(Q[1]-Q[0])
        nx = len(packet['x'])
        length = (nx+1)*float(packet['dx'])
        self.tx = xp.asarray(.5*(np.pi*np.arange(1,nx+1)/length)**2)
        self.kR = xp.asarray(2*np.pi*np.fft.fftfreq(len(packet['R']), packet['dR']))
        self.kQ = xp.asarray(2*np.pi*np.fft.fftfreq(len(Q), self.dQ))
        self.tr = self.kR**2/(2*packet['mass'])
        self.tq = self.omega*self.kQ**2/2
        self._phases = {}
        self._potential_phase = None
        self._potential_dt = None

    def array(self, u):
        # step mutates its input: owning storage is essential for independent
        # dt/dt2 validation and avoids silently evolving the caller's initial.
        return self.xp.array(u, dtype=self.xp.complex128, copy=True)

    def host(self, u):
        return self.xp.asnumpy(u) if self.gpu else np.asarray(u)

    def _fft(self, u, axis, inverse=False):
        if self.gpu:
            fn = self.xp.fft.ifft if inverse else self.xp.fft.fft
            return fn(u, axis=axis, norm='ortho')
        return (sf.ifft if inverse else sf.fft)(u, axis=axis, norm='ortho', workers=self.workers)

    def _dst(self, u):
        if not self.gpu:
            return sf.dst(u, type=1, axis=1, norm='ortho', workers=self.workers)
        # Reuse the tested historical CUDA odd-extension transform, in slabs.
        from multi_component_exact_factorization_discrete_gpu.spectral_tdse import _dst1_ortho
        return _dst1_ortho(u.swapaxes(0,1)).swapaxes(0,1)

    def potential(self, start, stop):
        mu = self.mu[start:stop,:,None]
        v = self.V[start:stop,:,None] + .5*self.omega*self.qgrid**2
        v = v + np.sqrt(self.omega)*self.g*mu*self.qgrid
        if self.dse:
            v = v + .5*self.g**2*mu**2
        return v

    def step(self, u, dt):
        """Unitary second-order split, cached 1D phases, no full potential cube."""
        xp = self.xp
        if dt not in self._phases:
            self._phases[dt] = tuple(xp.exp(-1j*dt*t) for t in (self.tx,self.tr,self.tq))
        px, pr, pq = self._phases[dt]
        # Static Hamiltonian: do not rebuild the 3D exponentials every step.
        # One complex cube is reserved explicitly in memory_estimate().
        if self._potential_dt != dt:
            if self._potential_phase is None:
                self._potential_phase=xp.empty_like(u)
            for i in range(0,len(u),self.block):
                self._potential_phase[i:i+self.block]=xp.exp(-.5j*dt*self.potential(i,i+self.block))
            self._potential_dt=dt
        u *= self._potential_phase
        for i in range(0,len(u),self.block):
            s=slice(i,i+self.block)
            u[s] = self._dst(self._dst(u[s])*px[None,:,None])
        # R FFT is blocked over x; Q FFT is blocked over R.
        for j in range(0,u.shape[1],self.block):
            s=slice(j,j+self.block)
            u[:,s,:] = self._fft(self._fft(u[:,s,:],0)*pr[:,None,None],0,True)
        for i in range(0,len(u),self.block):
            s=slice(i,i+self.block)
            u[s] = self._fft(self._fft(u[s],2)*pq[None,None,:],2,True)
            u[s] *= self._potential_phase[s]
        return u

    def action(self, u):
        """Same Hamiltonian as propagation, for instantaneous EF derivatives."""
        out = self.xp.empty_like(u)
        for i in range(0,len(u),self.block):
            s=slice(i,i+self.block)
            out[s] = self.potential(i,i+self.block)*u[s]
            out[s] += self._dst(self._dst(u[s])*self.tx[None,:,None])
            out[s] += self._fft(self._fft(u[s],2)*self.tq[None,None,:],2,True)
        for j in range(0,u.shape[1],self.block):
            s=slice(j,j+self.block)
            out[:,s,:] += self._fft(self._fft(u[:,s,:],0)*self.tr[:,None,None],0,True)
        return out

    def observe(self, u, action=None):
        """Bare photon number uses BOTH quadratures, never q density alone."""
        xp=self.xp; p=self.packet
        cell=p['dx']*p['dR']*self.dQ
        rho=xp.abs(u)**2
        norm=float(xp.sum(rho)*cell)
        q2=float(xp.sum(rho*self.qgrid**2)*cell)
        p2=pmean=tr=tx=0.
        for i in range(0,len(u),self.block):
            v=u[i:i+self.block]
            f=self._fft(v,2); w=xp.abs(f)**2
            p2+=float(xp.sum(w*self.kQ[None,None,:]**2)*cell)
            pmean+=float(xp.sum(w*self.kQ[None,None,:])*cell)
            tx+=float(xp.sum(xp.abs(self._dst(v))**2*self.tx[None,:,None])*cell)
        for j in range(0,u.shape[1],self.block):
            f=self._fft(u[:,j:j+self.block,:],0)
            tr+=float(xp.sum(xp.abs(f)**2*self.tr[:,None,None])*cell)
        h=self.action(u) if action is None else action
        energy=float(xp.real(xp.vdot(u,h))*cell)
        rho_R=self.host(xp.sum(rho,axis=(1,2))*p['dx']*self.dQ)
        rho_x=self.host(xp.sum(rho,axis=(0,2))*p['dR']*self.dQ)
        rho_Q=self.host(xp.sum(rho,axis=(0,1))*p['dx']*p['dR'])
        barrier=float(p['barrier_R']); idx=int(np.argmin(abs(p['R']-barrier)))
        ur=self._fft(self._fft(u,0)*(1j*self.kR[:,None,None]),0,True)
        current=self.host(xp.imag(xp.sum(u.conj()*ur,axis=(1,2)))*p['dx']*self.dQ/p['mass'])
        def moments(grid,density,spacing):
            mean=float(np.sum(grid*density)*spacing/norm)
            return mean,float(np.sqrt(np.sum((grid-mean)**2*density)*spacing/norm))
        rm,rs=moments(p['R'],rho_R,p['dR'])
        xm,xs=moments(p['x'],rho_x,p['dx'])
        qm,qs=moments(self.Q/np.sqrt(self.omega),rho_Q,self.dQ)
        number=.5*(q2+p2)/norm-.5
        coherent=.5*(self.omega*qm**2+(pmean/norm)**2)
        return dict(norm=norm,energy=energy,kinetic_R=tr,kinetic_x=tx,
                    photon_number=number,photon_coherent_number=coherent,
                    photon_fluctuation_number=number-coherent,
                    photon_energy=.5*self.omega*(q2+p2),
                    q_mean=float(xp.sum(rho*self.qgrid)*cell)/(norm*np.sqrt(self.omega)),
                    p_mean=np.sqrt(self.omega)*pmean/norm,
                    product=float(np.sum(rho_R[p['R']>=barrier])*p['dR']),
                    reactant=float(np.sum(rho_R[p['R']<barrier])*p['dR']),
                    signed_flux=float(current[idx]),rho_R=rho_R,rho_x=rho_x,rho_Q=rho_Q,
                    mean_R=rm,width_R=rs,mean_x=xm,width_x=xs,width_q=qs,
                    edge_x=float(np.sum(rho_x[np.r_[0:3,len(rho_x)-3:len(rho_x)]])*p['dx']),
                    edge_R=float(np.sum(rho_R[np.r_[0:3,len(rho_R)-3:len(rho_R)]])*p['dR']),
                    edge_Q=float(np.sum(rho_Q[np.r_[0:3,len(rho_Q)-3:len(rho_Q)]])*self.dQ))
