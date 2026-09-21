"""LOCAL export from existing Phase6 code; server needs only packet + GPU files.

Never overwrite a packet or historical input. Tests the portable CPU map
against the existing FullPF/FullSplit/observe before permitting export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def export(path, case='resonant', setting='base'):
    from .phase6_resume import build,SETTINGS
    from .phase6_split import FullSplit
    from .phase6_observables import observe
    from .phase6_gpu_backend import PFBackend
    path=Path(path)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    config=dict(case=case,setting=SETTINGS[setting])
    h,u,phi,metadata=build(config);dt=config['setting'][-1]
    split=FullSplit(h,dt)
    # Derive in the SAME eigenbasis/order as the old split, without another rotation choice.
    displacement=np.einsum('ni,nm,mi->i',split.rotation,
        np.diag(np.sqrt(np.arange(1,h.nfock)),1)+np.diag(np.sqrt(np.arange(1,h.nfock)),-1),
        split.rotation)
    packet=dict(psi=u,R=h.R,x=h.x,dx=h.xgrid.dx,dR=h.Rgrid.dx,mass=h.cavity.mass,
        omega=h.cavity.omega_c,g_chi=h.cavity.g_chi,tx=h.tx,tr=h.tr,potential=h.potential,
        dse=h.dse,mu=h.mu,photon=h.photon,rotation=split.rotation,displacement=displacement,
        phi=phi,dt=dt)
    backend=PFBackend(packet,dt)
    state_error=float(np.sqrt(np.sum(abs(backend.step(u.copy())-split.step(u.copy()))**2)*h.volume))
    action_error=float(np.sqrt(np.sum(abs(backend.action(u)-h.action(u))**2)*h.volume))
    a,b=backend.observe(u),observe(u,h,phi)
    errors={key:float(np.max(abs(np.asarray(a[key])-np.asarray(b[key])))) for key in b}
    if max(state_error,action_error,max(errors.values()))>1e-10:
        raise RuntimeError(dict(step=state_error,action=action_error,observables=errors))
    metadata.update(config=config,format_version=1,legacy_adapter_validation=dict(
        status='PASS',step_L2=state_error,action_L2=action_error,observable_errors=errors))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez_compressed(stream,**packet,metadata=json.dumps(metadata))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(path=str(path),sha256=digest,bytes=path.stat().st_size,metadata=metadata)
    path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--case',choices=['free','resonant','barrier'],default='resonant')
    p.add_argument('--setting',default='base')
    a=p.parse_args();export(a.out,a.case,a.setting)
