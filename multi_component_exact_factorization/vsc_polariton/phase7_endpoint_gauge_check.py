"""Revisit the original final-frame gauge failures without new propagation."""
import json
from pathlib import Path
import gc
import numpy as np
from .phase7_pilot_baseline import INPUT
from .phase7_transfer_utils import digest
from .phase7_native import full_action
from .phase7_dealiased_gauge import bilinears,shifted_overlap,evaluate,probe


def main():
    out=Path('results/vsc_polariton/phase7/endpoint_gauge_v2');out.mkdir(exist_ok=False)
    manifest=json.loads((INPUT/'phase7_pilot_manifest.json').read_text());results={}
    for case in ('free','resonant','barrier'):
        record=manifest['cases'][case]
        for path in (record['packet'],record['waves'][-1]):
            if digest(INPUT/path)!=manifest['files'][path]:raise ValueError('Input hash mismatch')
        with np.load(INPUT/record['packet']) as z:p={k:z[k] for k in z.files if k!='psi'}
        with np.load(INPUT/record['waves'][-1]) as z:u=z['psi'];tau=float(z['time_au'])
        ut=-1j*full_action(u,p)
        cache=bilinears(u,ut,float(p['dx']),float(p['dR']));del ut
        frame={};passed=False
        for pair in ((128,256),(1024,2048)):
            for factor in pair:
                n=len(u)*factor;h=float(p['dR'])/factor
                links={s:evaluate(shifted_overlap(u,float(p['dx']),float(p['dR']),s*h),n) for s in (-3,-2,-1,1,2,3)}
                checks={}
                for budget in (1e-6,1e-8,1e-10):
                    fields,check=probe(cache,p['R'],float(p['mass']),factor,budget,links);checks[str(budget)]=check
                    if budget==1e-8:np.savez_compressed(out/f'{case}_factor{factor}.npz',**fields)
                    del fields
                frame[str(factor)]=checks
                print('endpoint',case,factor,checks['1e-08'],flush=True)
                del links;gc.collect()
            passed=all(d['force_pass'] and d['gauge_pass'] and d['scalar_imag_RMS']<1e-6
                       for f in pair for d in frame[str(f)].values())
            if passed:break
        results[case]=dict(time_au=tau,postprocessing_pass=passed,resolutions=frame)
        (out/'validation.json').write_text(json.dumps(dict(phase7_pass=False,cases=results),indent=2))
        del u;gc.collect()


if __name__=='__main__':main()
