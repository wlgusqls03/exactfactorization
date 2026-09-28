"""Steps 1/2: existing event waves only, no dynamics and no tolerance changes."""
import argparse
import gc
import io
import json
from pathlib import Path
import tarfile
import numpy as np
from .phase7_transfer_utils import digest
from .phase7_pilot_baseline import INPUT
from .phase7_nested_fields import outer_fields,nested_fields
from .phase7_fock_to_q import hermite_grid
from .run_phase7_pilot import checks_for
from .phase7_support import weighted_rms
from .phase7_dealiased_gauge import bilinears,shifted_overlap,evaluate,probe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-root',type=Path,default=INPUT)
    parser.add_argument('--archive',type=Path,default=Path('results/vsc_polariton/phase7/results/wave_inventory_v1/phase7_event_waves.tar.gz'))
    parser.add_argument('--out',type=Path,default=Path('results/vsc_polariton/phase7/event_validation_v1'))
    parser.add_argument('--cases',nargs='+',default=['free','resonant','barrier'])
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    packet_manifest=json.loads((args.pilot_root/'phase7_pilot_manifest.json').read_text())
    archive_hash=digest(args.archive)
    (args.out/'source_manifest.json').write_text(json.dumps({str(p):digest(p) for p in Path(__file__).parent.glob('*phase7*.py')},indent=2))
    summary=dict(phase7_pass=False,scope='Step1 event scalar diagnostics and Step2 postprocessing resolution ONLY',archive_sha256=archive_hash,frames={})
    with tarfile.open(args.archive,'r:gz') as tar:
        manifest=json.load(tar.extractfile('manifest.json'))
        for case in args.cases:
            record=packet_manifest['cases'][case];path=args.pilot_root/record['packet']
            if digest(path)!=manifest['report'][case]['input_sha256']:raise ValueError('Input association mismatch')
            with np.load(path) as z:p={k:z[k] for k in z.files if k!='psi'}
            for name,meta in manifest['files'].items():
                if not name.startswith(case+'/'):continue
                raw=tar.extractfile(name).read()
                import hashlib
                if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('Wave hash mismatch')
                with np.load(io.BytesIO(raw)) as z:
                    u=z['psi'];tau=float(z['time_au'])
                    np.testing.assert_array_equal(z['R'],p['R']);np.testing.assert_array_equal(z['x'],p['x'])
                del raw
                stem=case+'_'+Path(name).stem;report=dict(time_au=tau,step1={},step2={})
                print(stem,'native actions',flush=True)
                outer,ur,urr,ut=outer_fields(u,p)
                for nq in ((32,64) if case=='free' else (176,208)):
                    grid=hermite_grid(u.shape[-1],nq,float(p['omega']))
                    fields,checks=nested_fields(u,p,grid,outer,ur,urr,ut)
                    checks=checks_for(fields,outer,grid,p,u,checks)
                    for budget in (1e-6,1e-8,1e-10):
                        mask=fields['mask_'+str(budget)];mass=fields['rho_qR']*grid['weights']*float(p['dR'])
                        regions={}
                        for label,region in [('reactant',p['R']<0),('barrier_strip',abs(p['R'])<=.5),('product',p['R']>0)]:
                            active=mask&region[:,None]
                            regions[label]=dict(probability=float(mass[active].sum()),
                                epsilon1_error=weighted_rms(fields['epsilon1_A']-fields['epsilon1_B'],mass,active),
                                epsilon1_imag=weighted_rms(fields['epsilon1_B'].imag,mass,active))
                        checks['support'][str(budget)]['regions']=regions
                    np.savez_compressed(args.out/f'{stem}_Nq{nq}.npz',R=p['R'],q=grid['q'],time_au=tau,**fields)
                    report['step1'][str(nq)]=checks
                    print(stem,'Nq',nq,'epsilon1',checks['support']['1e-08']['epsilon1_routes_RMS'],flush=True)
                    del fields;gc.collect()
                del ur,urr,outer;gc.collect()
                cache=bilinears(u,ut,float(p['dx']),float(p['dR']))
                del ut;gc.collect()
                np.savez_compressed(args.out/f'{stem}_bilinears.npz',R=p['R'],cache=cache,mass=p['mass'])
                for factor in (2,4,8,16,32,64,128,256):
                    links=None
                    if factor>=128:
                        n=len(u)*factor;h=float(p['dR'])/factor
                        links={s:evaluate(shifted_overlap(u,float(p['dx']),float(p['dR']),s*h),n) for s in (-3,-2,-1,1,2,3)}
                    reports={}
                    for budget in (1e-6,1e-8,1e-10):
                        values,check=probe(cache,p['R'],float(p['mass']),factor,budget,links)
                        reports[str(budget)]=check
                        if factor>=128 and budget==1e-8:
                            np.savez_compressed(args.out/f'{stem}_gauge{factor}.npz',**values)
                    report['step2'][str(factor)]=reports
                    print(stem,'R factor',factor,'force error',reports['1e-08']['force_fd6_error'],flush=True)
                report['step2_postprocessing_pass']=all(d['force_pass'] and d.get('gauge_pass',False) and d['scalar_imag_RMS']<1e-6
                    for f in ('128','256') for d in report['step2'][f].values())
                report['step1_scalar_pass']=all(d['scalar_routes_pass'] for v in report['step1'].values() for d in v['support'].values())
                (args.out/f'{stem}.json').write_text(json.dumps(report,indent=2))
                summary['frames'][stem]=report
                (args.out/'validation.json').write_text(json.dumps(summary,indent=2))
                del u;gc.collect()
    print('Event postprocessing complete; Phase7 overall remains NOT CERTIFIED',flush=True)


if __name__=='__main__':main()
