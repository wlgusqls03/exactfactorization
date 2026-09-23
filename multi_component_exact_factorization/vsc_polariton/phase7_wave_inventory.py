"""Read-only server inventory; optionally export existing event waves only.

No simulation, deletion, overwrite, or snapshot interpolation. Default root
and outputs are repository results, never /tmp. No required big packet copy.
"""
import argparse
import io
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from .phase7_pilot_baseline import digest
from .phase7_reaction_media import events,AU_FS

RUNS={
 'free':'free_recovery_campaign_v1/free_L36_dx0.3/full',
 'resonant':'orthogonal_campaign_v1/F120_dt0125/full',
 'barrier':'free_recovery_campaign_v1/barrier_F120/full'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/vsc_polariton/phase6_gpu'))
    parser.add_argument('--out',type=Path,default=Path('results/vsc_polariton/phase7/wave_inventory_v1'))
    parser.add_argument('--pack-events',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);report={};selected={}
    for case,relative in RUNS.items():
        folder=args.root/relative
        if not folder.is_dir():
            report[case]={'status':'MISSING_DIRECTORY','path':str(folder)};continue
        status=json.loads((folder/'status.json').read_text())
        if status.get('failures') or status['status']!='COMPLETE_NOT_CERTIFIED':
            raise ValueError('Unexpected run status; refusing export '+case)
        waves=[]
        for path in sorted(folder.glob('wave_*.npz')):
            with np.load(path) as z:time=float(z['time_au'])
            waves.append((time,path))
        rows=[]
        for path in sorted(folder.glob('observable_*.npz')):
            with np.load(path) as z:rows.append([float(z[k]) for k in ('time_au','product','flux')])
        if not rows:
            report[case]={'status':'MISSING_OBSERVABLES','wave_count':len(waves)};continue
        data=np.array(rows);ev=events(data[:,0],data[:,1],data[:,2]);wanted={}
        for name,index in ev.items():
            if name in ('initial','final') or index is None:continue
            target=float(data[index,0])
            if not waves:
                wanted[name]={'status':'MISSING_WAVES','target_fs':target*AU_FS};continue
            time,path=min(waves,key=lambda pair:abs(pair[0]-target))
            error=abs(time-target)*AU_FS
            wanted[name]={'target_fs':target*AU_FS,'saved_fs':time*AU_FS,
                          'time_error_fs':error,'path':str(path),
                          'status':'AVAILABLE' if error<=.5 else 'NO_NEARBY_FRAME'}
            # Never substitute a distant frame silently or duplicate pilot endpoints.
            if error<=.5 and time>0 and time<float(status['time_au']):
                selected[f'{case}/{path.name}']=path
        report[case]={'status':'INVENTORIED','run':str(folder),'input_sha256':status['input_sha256'],
                      'wave_count':len(waves),'events':wanted,
                      'wave_times_fs':[float(t*AU_FS) for t,p in waves]}
    size=sum(p.stat().st_size for p in selected.values())
    report['export']={'files':len(selected),'uncompressed_GiB':size/1024**3,
                      'scope':'Existing event snapshots only; initial/final already supplied; not all-time certification'}
    (args.out/'inventory.json').write_text(json.dumps(report,indent=2))
    if args.pack_events and selected:
        if shutil.disk_usage(args.out).free<size+512*1024**2:raise RuntimeError('Insufficient archive space')
        details=dict(report=report,files={name:{'source':str(path),'sha256':digest(path)} for name,path in selected.items()})
        with tarfile.open(args.out/'phase7_event_waves.tar.gz','x:gz') as tar:
            for name,path in selected.items():tar.add(path,arcname=name,recursive=False)
            raw=json.dumps(details,indent=2).encode();info=tarfile.TarInfo('manifest.json');info.size=len(raw)
            tar.addfile(info,io.BytesIO(raw))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
