"""Same-wave spatial refinement, NOT finer TDSE propagation.

Reuse historical read-only Fourier bilinear/overlap machinery for q-grid
waves by supplying volume=dx*dq. Gauge diagnostics use shifted wave overlaps,
not substitution alpha'=0. Snapshot force/gauge only; temporal tracking absent.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .run_real_grid_mcef_preview import load_wave,digest
from .real_grid_mcef_fields import action
from .phase7_dealiased_gauge import bilinears,shifted_overlap,evaluate,probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input','wave','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--factors',nargs='+',type=int,default=[2,4,8])
    a=p.parse_args()
    if any(f<2 for f in a.factors):p.error('factor must be >=2')
    root=Path(__file__).resolve().parents[2]/'results/vsc_polariton'
    a.out.resolve().relative_to(root.resolve());a.out.mkdir(parents=True,exist_ok=False)
    with np.load(a.input) as z:packet={k:z[k] for k in z.files}
    u,Q,t=load_wave(a.wave,packet,digest(a.input));w=float(packet['omega']);u*=w**.25
    q=Q/np.sqrt(w);vol=float(packet['dx'])*(q[1]-q[0]);dr=float(packet['dR']);M=float(packet['mass'])
    ut=-1j*action(u,packet,q)
    cache=bilinears(u,ut,vol,dr);del ut
    np.savez(a.out/'bilinear_cache.npz',cache=cache,R=packet['R'],time_au=t)
    rows=[]
    for factor in a.factors:
        n=len(u)*factor;h=dr/factor;links={}
        for offset in (-3,-2,-1,1,2,3):
            c=shifted_overlap(u,vol,dr,offset*h)
            links[offset]=evaluate(c,n,length=len(u)*dr)
        for budget in (1e-6,1e-8,1e-10):
            fields,report=probe(cache,packet['R'],M,factor,budget,links)
            rows.append(report)
            np.savez(a.out/f'factor{factor}_budget{budget}.npz',**fields)
            print(factor,budget,report['force_fd6_error'],report.get('alpha_fd6_RMS'),flush=True)
        summary=dict(status='SPATIAL_DIAGNOSTIC',time_au=t,packet_sha256=digest(a.input),wave=str(a.wave.resolve()),
            phase7_pass=False,temporal_gauge_certified=False,
            scope='Same Fourier wave oversampling only, not propagated basis convergence. Component-local gauge; no inter-component potential alignment.',records=rows)
        (a.out/'validation.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
