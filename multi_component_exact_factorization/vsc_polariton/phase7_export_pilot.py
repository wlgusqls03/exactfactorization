"""Pack validated initial/final waves for local Phase7 development only.

No propagation, no deletion. Keep all original frames for production Phase7.
"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from .run_phase6_gpu import load_packet


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    base=Path('results/vsc_polariton/phase6_gpu')
    gate=base/'finish_campaign_v3/phase6_completion_validation.json'
    if json.loads(gate.read_text())['status']!='COMPUTED_GATES_PASS_REVIEW_PENDING':
        raise RuntimeError('Expected reviewed completed numerical gate')
    cases={
      'free':('free_recovery_campaign_v1/free_L36_dx0.3/full','free_recovery_transfer_v2/inputs/free_L36_dx0.3.npz'),
      'resonant':('orthogonal_campaign_v1/F120_dt0125/full','orthogonal_transfer/inputs/resonant_F120.npz'),
      'resonant_F160':('orthogonal_campaign_v1/F160_dt0125/full','orthogonal_transfer/inputs/resonant_F160.npz'),
      'barrier':('free_recovery_campaign_v1/barrier_F120/full','completion_transfer/inputs/barrier_F120.npz')}
    files={gate};records={}
    for case,(directory,packet_path) in cases.items():
        folder=base/directory;packet_path=base/packet_path
        packet,digest=load_packet(packet_path)
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='COMPLETE_NOT_CERTIFIED' or status['input_sha256']!=digest:
            raise RuntimeError('Unvalidated input/run association: '+case)
        files.update([packet_path,packet_path.with_suffix('.json'),folder/'status.json'])
        selected=[]
        for t in (0.,1652.):
            wave=folder/f"wave_{round(t/status['dt']):07d}.npz"
            if not wave.is_file():raise FileNotFoundError('Keep/copy this successful wave: '+str(wave))
            with np.load(wave) as data:
                if abs(float(data['time_au'])-t)>1e-10:raise RuntimeError('Wave time mismatch')
                np.testing.assert_array_equal(data['R'],packet['R'])
                np.testing.assert_array_equal(data['x'],packet['x'])
            files.add(wave);selected.append(str(wave))
        records[case]=dict(packet=str(packet_path),waves=selected)
    out=Path('results/vsc_polariton/phase7/phase7_pilot_inputs.tar.gz')
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Preserve existing archive: '+str(out))
    size=sum(p.stat().st_size for p in files)
    print(f'Uncompressed input size {size/1024**3:.3f} GiB',flush=True)
    if shutil.disk_usage(out.parent).free<size+512*1024**2:raise RuntimeError('Insufficient temporary archive space')
    manifest=dict(scope='Initial/final pilot only; not all-time Phase7 certification',cases=records,
                  files={str(p):sha(p) for p in sorted(files)})
    data=json.dumps(manifest,indent=2).encode()
    with tarfile.open(out,'x:gz') as tar:
        for p in sorted(files):tar.add(p,arcname=str(p),recursive=False)
        info=tarfile.TarInfo('phase7_pilot_manifest.json');info.size=len(data)
        tar.addfile(info,io.BytesIO(data))
    print('Send',out,'bytes=',out.stat().st_size,flush=True)
    print('Original archives/waves/inputs untouched. Keep all other frames for production.')


if __name__=='__main__':main()
