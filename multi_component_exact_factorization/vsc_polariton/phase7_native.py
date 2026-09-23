"""Matrix-free native Phase6 spectral Hamiltonian actions, no propagation.

Arrays (R,x,n), au. Independently tested against existing PFBackend.action.
"""
import numpy as np
from scipy.fft import fft,ifft


def R_derivative(u,dR,order=1):
    k=2*np.pi*np.fft.fftfreq(u.shape[0],float(dR))
    return ifft((1j*k[:,None,None])**order*fft(u,axis=0),axis=0)


def bare_action(u,packet):
    return ifft(packet['tx'][None,:,None]*fft(u,axis=1),axis=1)+packet['potential'][:,:,None]*u


def internal_action(u,packet):
    value=bare_action(u,packet)
    value+=(packet['dse'][:,:,None]+packet['photon'][None,None,:])*u
    c=float(packet['g_chi'])*packet['mu'][:,:,None]*np.sqrt(np.arange(1,u.shape[-1]))[None,None,:]
    value[:,:,:-1]+=c*u[:,:,1:]
    value[:,:,1:]+=c*u[:,:,:-1]
    return value


def full_action(u,packet):
    return internal_action(u,packet)-R_derivative(u,packet['dR'],2)/(2*float(packet['mass']))
