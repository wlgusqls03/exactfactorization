"""Point current from the propagated Fourier wavefunction, atomic units.

No Hamiltonian, propagation, population quadrature or tolerance changes.
The historical linear-current interpolation remains in diagnostic fields.
"""
import sys
from unittest.mock import patch
import numpy as np
from .phase6_gpu_backend import PFBackend


class SpectralFluxBackend(PFBackend):
    def point_current(self,u,position=0.):
        xp=self.xp
        k=xp.asarray(2*np.pi*np.fft.fftfreq(len(self.R),self.dR))
        coefficients=self.fft.fft(u,axis=0)/len(self.R)
        phase=xp.exp(1j*k*(position-self.R[0]))[:,None,None]
        value=xp.sum(coefficients*phase,axis=0)
        derivative=xp.sum(coefficients*phase*(1j*k[:,None,None]),axis=0)
        return float(self.host(xp.sum((value.conj()*derivative).imag)))*self.dx/self.mass

    def observe(self,u):
        row=super().observe(u)
        row['flux_linear']=row['flux']
        row['continuity_error_linear']=row['continuity_error']
        row['flux']=self.point_current(u)
        row['continuity_error']=row['dproduct_exact']-row['flux']
        row['flux_box_boundary']=self.point_current(u,self.R[-1]+self.dR/2)
        return row


def main():
    from . import phase6_completion_controls as controls
    with patch.object(controls,'PFBackend',SpectralFluxBackend):controls.main()


if __name__=='__main__':main()
