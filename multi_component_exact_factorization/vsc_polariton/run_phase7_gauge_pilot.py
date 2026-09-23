"""Independent snapshot gauge diagnostics; never certifies all-time Phase7."""
import argparse
import json
from pathlib import Path
import numpy as np
from .phase7_pilot_baseline import INPUT, OUTPUT, digest
from .phase7_outer_gauge import outer_gauge


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis',type=Path,default=OUTPUT/'analysis_v2')
    parser.add_argument('--out',type=Path,default=OUTPUT/'gauge_v1')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((INPUT/'phase7_pilot_manifest.json').read_text())
    reports={}
    for case,record in manifest['cases'].items():
        if digest(INPUT/record['packet'])!=manifest['files'][record['packet']]:
            raise RuntimeError('Packet hash changed')
        with np.load(INPUT/record['packet']) as z:
            packet={k:z[k] for k in ('R','dx','dR','mass')}
        for wave in record['waves']:
            if digest(INPUT/wave)!=manifest['files'][wave]:raise RuntimeError('Wave hash changed')
            with np.load(INPUT/wave) as z:u=z['psi'];tau=float(z['time_au'])
            choices=list(args.analysis.glob(f'{case}_t{tau:g}_Nq*.npz'))
            source=max(choices,key=lambda p:int(p.stem.split('Nq')[-1]))
            with np.load(source) as z:outer={k[6:]:z[k] for k in z.files if k.startswith('outer_')}
            for budget in (1e-6,1e-8,1e-10):
                name=f'{case}_t{tau:g}_budget{budget:g}'
                fields,report=outer_gauge(u,packet,outer,budget)
                np.savez_compressed(args.out/(name+'.npz'),R=packet['R'],**fields)
                reports[name]=report
                print(name,report,flush=True)
            del u
    (args.out/'validation.json').write_text(json.dumps(dict(phase7_pass=False,frames=reports),indent=2))


if __name__=='__main__':main()
