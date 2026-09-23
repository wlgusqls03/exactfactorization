"""Identify finite-Fock projection defect without modifying scalar potentials."""
import json
import numpy as np
from .phase7_pilot_baseline import INPUT,OUTPUT,digest
from .phase7_fock_to_q import hermite_grid,transform
from .phase7_support import weighted_rms


def main():
    root=OUTPUT/'analysis_v2';out=OUTPUT/'projection_audit.json'
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((INPUT/'phase7_pilot_manifest.json').read_text());reports={}
    for case in ('resonant','resonant_F160','barrier'):
        item=manifest['cases'][case]
        with np.load(INPUT/item['packet']) as z:
            dx=float(z['dx']);dr=float(z['dR']);g=float(z['g_chi']);omega=float(z['omega']);mu=z['mu']
        wave=item['waves'][-1]
        if digest(INPUT/wave)!=manifest['files'][wave]:raise RuntimeError('Changed input')
        with np.load(INPUT/wave) as z:u=z['psi'];tau=float(z['time_au'])
        path=max(root.glob(f'{case}_t{tau:g}_Nq*.npz'),key=lambda p:int(p.stem.split('Nq')[-1]))
        with np.load(path) as z:fields={k:z[k] for k in z.files}
        nq=len(fields['q']);nf=u.shape[-1];grid=hermite_grid(nf,nq,omega)
        extra=hermite_grid(nf+1,nq,omega)['basis'][-1]
        expected=np.zeros(fields['rho_qR'].shape,dtype=complex)
        for start in range(0,len(u),8):
            sl=slice(start,start+8);psi=transform(u[sl],grid)
            defect=-g*np.sqrt(nf)*mu[sl,:,None]*u[sl,:,-1,None]*extra
            numerator=np.sum(psi.conj()*defect,axis=1)*dx
            np.divide(numerator,fields['rho_qR'][sl],out=expected[sl],where=fields['rho_qR'][sl]>0)
        mask=fields['mask_1e-08'];mass=fields['rho_qR']*grid['weights']*dr
        difference=fields['epsilon1_A']-fields['epsilon1_B']
        reports[case]={'epsilon1_route_RMS_Ha':weighted_rms(difference,mass,mask),
            'remaining_after_analytic_defect_RMS_Ha':weighted_rms(difference+expected,mass,mask),
            'note':'Diagnostic identity only. No correction applied to production scalar.'}
        del u
    out.write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
