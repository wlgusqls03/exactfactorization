"""Fixed 512/1024/2048 postprocessing ladder for failed event force checks."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import gc
import numpy as np
from .phase7_pilot_baseline import INPUT
from .phase7_transfer_utils import digest
from .phase7_dealiased_gauge import probe,shifted_overlap,evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=Path('results/vsc_polariton/phase7/event_validation_v1'))
    parser.add_argument('--archive',type=Path,default=Path('results/vsc_polariton/phase7/results/wave_inventory_v1/phase7_event_waves.tar.gz'))
    parser.add_argument('--out',type=Path,default=Path('results/vsc_polariton/phase7/event_gauge_extension_v1'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    reports=json.loads((args.base/'validation.json').read_text())
    if digest(args.archive)!=reports['archive_sha256']:raise ValueError('Archive changed')
    packets=json.loads((INPUT/'phase7_pilot_manifest.json').read_text());results={}
    with tarfile.open(args.archive,'r:gz') as tar:
        manifest=json.load(tar.extractfile('manifest.json'))
        for name,meta in manifest['files'].items():
            case=name.split('/')[0];stem=case+'_'+Path(name).stem
            if reports['frames'][stem]['step2_postprocessing_pass']:continue
            record=packets['cases'][case];packet_path=INPUT/record['packet']
            if digest(packet_path)!=manifest['report'][case]['input_sha256']:raise ValueError('Packet changed')
            with np.load(packet_path) as z:R=z['R'];dr=float(z['dR']);dx=float(z['dx']);M=float(z['mass'])
            raw=tar.extractfile(name).read()
            if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('Wave changed')
            with np.load(io.BytesIO(raw)) as z:u=z['psi']
            del raw
            with np.load(args.base/f'{stem}_bilinears.npz') as z:cache=z['cache']
            frame={}
            for factor in (512,1024,2048):
                links=None
                if factor>=1024:
                    n=len(R)*factor;h=dr/factor
                    links={s:evaluate(shifted_overlap(u,dx,dr,s*h),n) for s in (-3,-2,-1,1,2,3)}
                checks={}
                for budget in (1e-6,1e-8,1e-10):
                    fields,d=probe(cache,R,M,factor,budget,links);checks[str(budget)]=d
                    if factor>=1024 and budget==1e-8:
                        np.savez_compressed(args.out/f'{stem}_gauge{factor}.npz',**fields)
                    del fields
                frame[str(factor)]=checks
                print(stem,factor,checks['1e-08'],flush=True)
                del links;gc.collect()
            frame['postprocessing_pass']=all(d['force_pass'] and d.get('gauge_pass',False) and d['scalar_imag_RMS']<1e-6
                for f in ('1024','2048') for d in frame[f].values())
            results[stem]=frame
            (args.out/'validation.json').write_text(json.dumps(dict(phase7_pass=False,frames=results),indent=2))
            del u;gc.collect()


if __name__=='__main__':main()
