"""Pack diagnostics and selected F200 waves, never delete original outputs."""
import io
import json
from pathlib import Path
import shutil
import tarfile
from .run_phase6_gpu import sha


def pack(root):
    root=Path(root);target=root/'phase7_f200_results.tar.gz'
    if target.exists():raise FileExistsError('Keep existing archive: '+str(target))
    files=[root/'campaign_identity.json',root/'batch_complete.json']
    for case,steps in [('resonant',(1280,4864,11264,13216)),('barrier',(1280,5376,11520,13216))]:
        folder=root/case
        state=json.loads((folder/'full/status.json').read_text())
        if state['status']!='COMPLETE_NOT_CERTIFIED' or state['failures']:raise ValueError('Run not complete')
        files += [folder/'full/status.json',folder/'check/gpu_validation.json']
        files += sorted(folder.glob('*/run.log'))
        files += sorted((folder/'full').glob('observable_*.npz'))
        files += [folder/'full'/f'wave_{step:07d}.npz' for step in steps]
    for file in files:
        if not file.is_file():raise FileNotFoundError(file)
    if shutil.disk_usage(root).free<sum(f.stat().st_size for f in files)+128*1024**2:
        raise RuntimeError('Insufficient archive space; original results preserved')
    manifest={str(f.relative_to(root)):sha(f) for f in files}
    with tarfile.open(target,'x:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(root)),recursive=False)
        raw=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('SHA256_MANIFEST.json');info.size=len(raw)
        tar.addfile(info,io.BytesIO(raw))
    print('SEND THIS FILE:',target,'SHA256',sha(target),flush=True)


if __name__=='__main__':
    pack(Path(__file__).resolve().parents[2]/'results/vsc_polariton/phase6_gpu/phase7_f200_campaign_v1')
