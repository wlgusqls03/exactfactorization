"""Read-only-input CPU free-control grid/time diagnosis, not production data.

Exact eta=0 photon vacuum. Cubic PES interpolation is a numerical sensitivity
probe, not a new literature parameter or a certified replacement BO grid.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import resample
from .run_phase6_gpu import load_packet,save_json,save_npz
from .phase6_completion_controls import reduced_packet
from .phase6_control_flux import SpectralFluxBackend as PFBackend


def run():
    folder=Path('results/vsc_polariton/phase6_gpu/completion_transfer/inputs')
    out=Path('results/vsc_polariton/phase6_control_continuity_spectral_probe')
    out.mkdir(exist_ok=True)
    packet,_=load_packet(folder/'free_F120.npz')
    with np.load(folder/'molecular.npz') as z:molecular=dict(z)
    old,_=reduced_packet(packet,molecular,'A')
    report={}
    for nr,dt in [(352,.125),(352,.0625),(440,.125),(550,.125),(704,.125)]:
        name=f'N{nr}_dt{dt}'
        if (out/(name+'.json')).exists():
            report[name]=json.loads((out/(name+'.json')).read_text());continue
        p=dict(old);dr=8.8/nr;R=-4.4+(np.arange(nr)+.5)*dr
        # Fourier resample at correctly shifted cell centres, preserving phase.
        k=2*np.pi*np.fft.fftfreq(len(old['R']),float(old['dR']))
        shifted=np.fft.ifft(np.fft.fft(old['psi'][:,0,0])*np.exp(1j*k*(R[0]-old['R'][0])))
        u=resample(shifted,nr)[:,None,None]
        p.update(R=R,dR=dr,psi=u,phi=np.ones((nr,1,1)),
            tr=(2*np.pi*np.fft.fftfreq(nr,dr))**2/(2*float(p['mass'])),
            potential=CubicSpline(old['R'],old['potential'][:,0])(R)[:,None],
            mu=np.zeros((nr,1)),dse=np.zeros((nr,1)),photon=old['photon'][:1],
            rotation=np.ones((1,1)),displacement=np.zeros(1))
        h=PFBackend(p,dt);rows=[];start=time.monotonic()
        for step in range(round(1652/dt)+1):
            if step%round(4/dt)==0:
                row=h.observe(u)
                row['flux_cubic']=float(CubicSpline(R,row['current'])(0))
                row['time_au']=step*dt;rows.append(row)
            if step<round(1652/dt):u=h.step(u)
        a={k:np.array([r[k] for r in rows]) for k in rows[0]}
        d=dict(nR=nr,dR=dr,dt=dt,wall_seconds=time.monotonic()-start,
            continuity=float(abs(a['continuity_error']).max()),
            continuity_linear=float(abs(a['continuity_error_linear']).max()),
            continuity_cubic_flux=float(abs(a['dproduct_exact']-a['flux_cubic']).max()),
            norm=float(abs(a['norm']-1).max()),energy=float(abs(a['energy']-a['energy'][0]).max()))
        save_npz(out/(name+'.npz'),**a);save_json(out/(name+'.json'),d)
        print(name,d,flush=True);report[name]=d
    save_json(out/'summary.json',report)


if __name__=='__main__':run()
