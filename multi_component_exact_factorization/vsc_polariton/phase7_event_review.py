"""Aggregate actual step1/2 results, retaining failed earlier resolutions."""
import json
from pathlib import Path
import numpy as np
from .phase7_support import weighted_rms


def pair_check(first,second):
    with np.load(first) as z:a={k:z[k] for k in z.files}
    with np.load(second) as z:b={k:z[k] for k in z.files}
    ratio=len(b['R'])//len(a['R'])
    np.testing.assert_allclose(a['R'],b['R'][::ratio],atol=1e-12,rtol=0)
    mask=a['mask']&b['mask'][::ratio]&np.isfinite(a['force_A0_fd6'])&np.isfinite(b['force_A0_fd6'][::ratio])
    h=a['R'][1]-a['R'][0];weight=np.maximum(a['rho'],0)*h
    difference=weighted_rms(a['force_A0_fd6']-b['force_A0_fd6'][::ratio],weight,mask)
    valid=b['mask']&np.isfinite(b['eta']);phase=np.exp(1j*b['eta'][valid])
    mass=b['rho'][valid]*(b['R'][1]-b['R'][0])
    reconstruction=float(np.sqrt(np.sum(mass*abs(phase*phase.conj()-1)**2)))
    density_error=float(np.sum(mass*abs(abs(phase)**2-1)))
    return dict(force_two_resolution_RMS=difference,reconstruction_L2=reconstruction,
                gauge_density_L1=density_error,pass_checks=difference<1e-5 and reconstruction<1e-10 and density_error<1e-9)


def main():
    base=Path('results/vsc_polariton/phase7/event_validation_v1')
    ext=Path('results/vsc_polariton/phase7/event_gauge_extension_v1')
    ends=Path('results/vsc_polariton/phase7/endpoint_gauge_v2')
    out=base/'step12_review.json'
    if out.exists():raise FileExistsError(out)
    records=json.loads((base/'validation.json').read_text())['frames']
    extra=json.loads((ext/'validation.json').read_text())['frames']
    end=json.loads((ends/'validation.json').read_text())['cases']
    if len(records)!=9 or set(end)!= {'free','resonant','barrier'}:
        raise ValueError('Incomplete nine-event / three-endpoint campaign')
    report=dict(phase7_pass=False,scope='Existing-wave event and endpoint postprocessing only',events={},endpoints={})
    for name,record in records.items():
        folder,first,second=(ext,1024,2048) if name in extra else (base,128,256)
        comparison=pair_check(folder/f'{name}_gauge{first}.npz',folder/f'{name}_gauge{second}.npz')
        passed=extra[name]['postprocessing_pass'] if name in extra else record['step2_postprocessing_pass']
        comparison.update(step1_scalar_pass=record['step1_scalar_pass'],step2_postprocessing_pass=bool(passed and comparison['pass_checks']))
        report['events'][name]=comparison
    for name,record in end.items():
        resolutions=sorted(int(k) for k in record['resolutions']);first,second=resolutions[-2:]
        comparison=pair_check(ends/f'{name}_factor{first}.npz',ends/f'{name}_factor{second}.npz')
        comparison['step2_postprocessing_pass']=bool(record['postprocessing_pass'] and comparison['pass_checks'])
        report['endpoints'][name]=comparison
    report['step1_scalar_gate']='PASS' if all(v['step1_scalar_pass'] for v in report['events'].values()) else 'FAIL'
    report['step2_postprocessing_gate']='PASS' if all(v['step2_postprocessing_pass'] for group in ('events','endpoints') for v in report[group].values()) else 'FAIL'
    report['pending']=['Fock convergence including matched refined event waves','propagated R-grid convergence of force','full time series','temporal probe','historical field compatibility','Phase4 force comparison','global audit']
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
