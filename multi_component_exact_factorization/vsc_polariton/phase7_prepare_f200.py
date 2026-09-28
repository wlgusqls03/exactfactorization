"""Local-only export of two orthogonal F200 inputs, preserving old runs."""
import json
import tarfile
from pathlib import Path
import numpy as np
from . import phase6_resume as resume
from .phase6_gpu_export import export
from .phase6_orthogonal_packets import convert
from .run_phase6_gpu import sha, load_packet


def main():
    root=resume.ROOT/'results/vsc_polariton/phase6_gpu/f200_transfer_v1'
    root.mkdir(exist_ok=False)
    original=resume.OUT
    key='phase7_F200_diagnostic'
    assert key not in resume.SETTINGS
    resume.SETTINGS[key]=(24.,160,4.4,352,200,.125)
    manifest=dict(phase7_pass=False,end_au=1652.,setting=list(resume.SETTINGS[key]),cases={})
    try:
        resume.OUT=root/'local_build'
        for case,parent in [('resonant','orthogonal_transfer/inputs/resonant_F160.npz'),('barrier','completion_transfer/inputs/barrier_F160.npz')]:
            raw=root/'local_build'/f'{case}_F200.npz';target=root/'inputs'/raw.name
            export(raw,case,key);convert(raw,target)
            packet,digest=load_packet(target)
            with np.load(root.parent/parent) as z:
                for name in ('R','x','mass','omega','g_chi','tx','tr','potential','dse','mu','phi'):
                    np.testing.assert_allclose(packet[name],z[name],rtol=0,atol=1e-13)
                difference=np.linalg.norm(packet['psi'][:,:,:160]-z['psi'])*np.sqrt(float(packet['dx']*packet['dR']))
            if difference>1e-10:raise RuntimeError('Initial preparation changed')
            manifest['cases'][case]=dict(sha256=digest,parent=parent,initial_common_L2=float(difference))
            del packet
    finally:
        resume.OUT=original;del resume.SETTINGS[key]
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=root.parent/'phase7_f200_server_bundle_v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        tar.add(root/'manifest.json',arcname='manifest.json')
        tar.add(root/'inputs',arcname='inputs')
    print('BUNDLE',archive,'SHA256',sha(archive),'BYTES',archive.stat().st_size,flush=True)


if __name__=='__main__':main()
