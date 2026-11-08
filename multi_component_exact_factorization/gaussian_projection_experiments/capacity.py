"""OFFLINE fits to saved matched-reference amplitudes; never used by dynamics."""
import argparse
from pathlib import Path
import numpy as np

from ..gaussian_trajectory.run import AU_FS,atomic_json
from .initialization import fit_distinct


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--K',type=int,nargs='+',default=[1,2,3,4])
    p.add_argument('--times-fs',type=float,nargs='+',default=[0.,.1,.2,.4]);a=p.parse_args()
    with np.load(a.case/'diagnostics.npz') as z:data={k:z[k] for k in z.files}
    if 'reference_rho' not in data:p.error('Offline capacity requires saved reference; dynamics does not')
    records=[]
    for t in a.times_fs:
        if t>data['t_au'][-1]*AU_FS+1e-9:
            records.append(dict(requested_fs=t,status='outside saved interval'));continue
        j=int(np.argmin(abs(data['t_au']*AU_FS-t)))
        for K in a.K:
            item=dict(K=K,requested_fs=t,actual_fs=float(data['t_au'][j]*AU_FS))
            try:
                b,audit=fit_distinct(data['q'],data['w'],data['reference_rho'][j],K,refine_single=True)
                item.update(status='FIT_ONLY_NOT_PROPAGATION',**audit)
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                item.update(status='FIT_FAILED',error=repr(exc),audit=getattr(exc,'audit',{}))
            records.append(item)
    atomic_json(a.case/'offline_capacity.json',dict(reference_feedback=False,
        note='Independent constrained local fits at each saved time, not autonomous propagation or a global-best-fit proof.',fits=records))


if __name__=='__main__':main()
