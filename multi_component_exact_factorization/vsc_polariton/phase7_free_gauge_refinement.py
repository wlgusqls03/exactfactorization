"""Postprocessing resolution probe on unchanged free-case Fourier wave.

No new propagation, potential interpolation, or density-floor repair. 2/4/8/16
oversampling factors are fixed before this probe. Not a native R-grid test.
"""
import json
import numpy as np
from scipy.signal import resample
from .phase7_pilot_baseline import INPUT,OUTPUT
from .phase7_native import full_action,R_derivative
from .phase7_outer_gauge import outer_gauge


def fields_from_wave(u,ut,dx,dr,M):
    ur=R_derivative(u,dr);urr=R_derivative(u,dr,2)
    inner=lambda a:np.sum(u.conj()*a,axis=(1,2))*dx
    rho=inner(u).real;rhot=2*inner(ut).real
    r1=2*inner(ur).real/rho
    r2=2*(inner(urr).real+np.sum(abs(ur)**2,axis=(1,2))*dx)/rho
    alpha=inner(ur).imag/rho;ar=inner(urr).imag/rho-alpha*r1
    at=np.sum((ut.conj()*ur+u.conj()*R_derivative(ut,dr)).imag,axis=(1,2))*dx/rho-alpha*rhot/rho
    r3=(2*inner(R_derivative(u,dr,3)).real+6*np.sum((ur.conj()*urr).real,axis=(1,2))*dx)/rho
    eps=(r2/2-r1*r1/4-alpha*alpha)/(2*M)
    force=-(r3/2-r1*r2+r1**3/2-2*alpha*ar)/(2*M)+at
    return dict(rho_R=rho,chi=np.sqrt(rho),alpha=alpha,alpha_t=at,epsilon2_B=eps,force=force)


def main():
    out=OUTPUT/'free_gauge_refinement.json'
    if out.exists():raise FileExistsError(out)
    record=json.loads((INPUT/'phase7_pilot_manifest.json').read_text())['cases']['free']
    with np.load(INPUT/record['packet']) as z:p={k:z[k] for k in z.files}
    with np.load(INPUT/record['waves'][-1]) as z:u=z['psi']
    ut=-1j*full_action(u,p);reports={}
    for factor in (2,4,8,16):
        n=len(u)*factor;dr=float(p['dR'])/factor
        fine=resample(u,n,axis=0);fine_t=resample(ut,n,axis=0)
        packet=dict(R=p['R'][0]+np.arange(n)*dr,dx=p['dx'],dR=dr,mass=p['mass'])
        outer=fields_from_wave(fine,fine_t,float(p['dx']),dr,float(p['mass']))
        _,report=outer_gauge(fine,packet,outer,1e-8)
        reports[str(factor)]=report;print(factor,report,flush=True)
        del fine,fine_t
    out.write_text(json.dumps(dict(scope='Free final snapshot, Fourier interpolation only',results=reports),indent=2))


if __name__=='__main__':main()
