"""Eight validated frames, blocked-R postprocessing; not all-time Phase7 PASS."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from .phase7_pilot_baseline import INPUT,OUTPUT,digest
from .phase7_fock_to_q import hermite_grid
from .phase7_nested_fields import outer_fields,nested_fields
from .phase7_support import budget_support,weighted_rms


def checks_for(fields,outer,grid,p,u,checks):
    dx,dr=map(float,(p['dx'],p['dR']))
    n=np.arange(u.shape[-1]);pop=np.sum(abs(u)**2,axis=(0,1))*dx*dr
    ladder=np.sum(u[:,:,:-1].conj()*u[:,:,1:]*np.sqrt(n[1:]))*dx*dr
    expected=np.array([pop.sum(),np.sqrt(2/float(p['omega']))*ladder.real,
                       np.sqrt(2*float(p['omega']))*ladder.imag,n@pop])
    checks['moment_errors']=(np.array(checks['moments'])-expected).tolist()
    checks['transform_orthogonality']=grid['orthogonality']
    checks['outer_density_relative_L1']=float(np.sum(abs(fields['rho_R_q']-outer['rho_R']))*dr/pop.sum())
    projection=np.einsum('rxj,rxn->rjn',p['phi'].conj(),u)*dx
    bo_expected=np.sum(abs(projection)**2,axis=(0,2))*dr
    bo_got=np.sum(fields['BO_density']*grid['weights'][None,:,None],axis=(0,1))*dr
    checks['BO_population_errors']=(bo_got-bo_expected).tolist()
    checks['support']={}
    for budget in (1e-6,1e-8,1e-10):
        mask,meta,labels=budget_support(fields['rho_qR'],dr*grid['weights'][None,:],budget)
        outer_mask,outer_meta,outer_labels=budget_support(outer['rho_R'],dr,budget)
        fields['mask_'+str(budget)]=mask;fields['components_'+str(budget)]=labels
        outer['mask_'+str(budget)]=outer_mask;outer['components_'+str(budget)]=outer_labels
        mass=fields['rho_qR']*grid['weights'][None,:]*dr;omass=outer['rho_R']*dr
        values=dict(joint=meta,outer=outer_meta,
            electronic_PNC_max=float(np.max(fields['electronic_PNC_error'][mask])),
            photon_PNC_max=float(np.max(fields['photon_PNC_error'][outer_mask])),
            epsilon1_routes_RMS=weighted_rms(fields['epsilon1_A']-fields['epsilon1_B'],mass,mask),
            epsilon1_imag_RMS=weighted_rms(fields['epsilon1_B'].imag,mass,mask),
            electronic_EOM_RMS=weighted_rms(fields['electronic_eom_norm'],mass,mask),
            epsilon2_routes_RMS=weighted_rms(outer['epsilon2_A']-outer['epsilon2_B'],omass,outer_mask),
            epsilon2_nested_RMS=weighted_rms(fields['epsilon2_nested']-outer['epsilon2_B'],omass,outer_mask),
            epsilon2_imag_RMS=weighted_rms(outer['epsilon2_B'].imag,omass,outer_mask),
            alpha_route_RMS=weighted_rms(fields['alpha_nested']-outer['alpha'],omass,outer_mask),
            a_definition_RMS=weighted_rms(fields['connection_a_check'],mass,mask),
            b_definition_RMS=weighted_rms(fields['connection_b_check'],mass,mask),
            berry_RMS=weighted_rms(fields['berry'],mass,mask),
            projection_defect_L2=float(np.sqrt(np.sum(fields['projection_defect_norm'][mask]**2*grid['weights'][None,:].repeat(len(p['R']),axis=0)[mask]*dr))))
        values['scalar_routes_pass']=all(values[k] is not None and values[k]<1e-6 for k in
            ('epsilon1_routes_RMS','epsilon1_imag_RMS','epsilon2_routes_RMS','epsilon2_nested_RMS','epsilon2_imag_RMS'))
        checks['support'][str(budget)]=values
    checks['factor_transform_pass']=all(checks[k]<1e-10 for k in
        ('reconstruction_L2','backprojection_L2','transform_orthogonality'))
    checks['factor_transform_pass'] &= all(v[k]<1e-10 for v in checks['support'].values()
        for k in ('electronic_PNC_max','photon_PNC_max'))
    checks['factor_transform_pass'] &= bool(np.max(abs(bo_got-bo_expected))<1e-9)
    checks['factor_transform_pass'] &= bool(np.max(abs(np.array(checks['moment_errors'])/(1+abs(expected))))<1e-9)
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['free','resonant','resonant_F160','barrier'])
    p.add_argument('--out',type=Path,default=OUTPUT/'analysis_v1')
    p.add_argument('--block',type=int,default=8);args=p.parse_args()
    if json.loads((OUTPUT/'baseline/validation.json').read_text())['status']!='PASS':raise RuntimeError('Baseline not PASS')
    manifest=json.loads((INPUT/'phase7_pilot_manifest.json').read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    # Pin this implementation and imported new Phase7 helpers for reproducibility.
    sources={str(f):digest(f) for f in Path('multi_component_exact_factorization/vsc_polariton').glob('*phase7*.py')}
    (args.out/'sources.json').write_text(json.dumps(sources,indent=2))
    results={}
    for case in args.cases:
        record=manifest['cases'][case];packet_path=INPUT/record['packet']
        if digest(packet_path)!=manifest['files'][record['packet']]:raise RuntimeError('Packet changed')
        with np.load(packet_path) as z:packet={k:z[k] for k in z.files}
        packet.pop('psi') # only archived propagated waves used below
        for wave in record['waves']:
            if digest(INPUT/wave)!=manifest['files'][wave]:raise RuntimeError('Wave changed')
            with np.load(INPUT/wave) as z:u=z['psi'];tau=float(z['time_au'])
            start=time.monotonic();outer,ur,urr,ut=outer_fields(u,packet)
            for nq in ((176,208) if case.startswith('resonant') else (max(32,u.shape[-1]+16),max(64,u.shape[-1]+48))):
                name=f'{case}_t{tau:g}_Nq{nq}'
                grid=hermite_grid(u.shape[-1],nq,float(packet['omega']))
                fields,checks=nested_fields(u,packet,grid,outer,ur,urr,ut,args.block)
                checks=checks_for(fields,outer,grid,packet,u,checks)
                checks.update(case=case,time_au=tau,nq=nq,elapsed=time.monotonic()-start,phase7_pass=False)
                np.savez_compressed(args.out/(name+'.npz'),R=packet['R'],q=grid['q'],weights=grid['weights'],
                    time_au=tau,**{'outer_'+k:v for k,v in outer.items()},**fields)
                (args.out/(name+'.json')).write_text(json.dumps(checks,indent=2))
                results[name]=checks
                print(name,'factor PASS',checks['factor_transform_pass'],'scalar',
                      {b:d['epsilon1_routes_RMS'] for b,d in checks['support'].items()},flush=True)
                del fields;gc.collect()
            del u,ur,urr,ut,outer;gc.collect()
    (args.out/'summary.json').write_text(json.dumps(dict(status='PILOT_ONLY',phase7_pass=False,frames=results,
        pending=['all-time frames','gauge convergence','derivative convergence','Phase4 comparison','historical fields']),indent=2))


if __name__=='__main__':main()
