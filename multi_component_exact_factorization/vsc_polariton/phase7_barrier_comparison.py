"""Read-only, paired saved-wave finite-Fock diagnostics; no propagation."""
import numpy as np
from .phase7_fock_to_q import hermite_grid
from .phase7_support import budget_support,weighted_rms
from .phase7_transfer_utils import digest,AU_FS


def diagnose(packet,wave):
    with np.load(packet,allow_pickle=False) as z:
        dx=float(z['dx']);dr=float(z['dR']);g=float(z['g_chi'])
        omega=float(z['omega']);mu=z['mu']
    with np.load(wave,allow_pickle=False) as z:
        u=z['psi'];time=float(z['time_au'])
    nf=u.shape[-1]
    if mu.shape!=u.shape[:2]:raise ValueError('Packet/wave grid mismatch')
    grid=hermite_grid(nf+1,2*nf+4,omega)
    basis=grid['basis'];weights=grid['weights']
    numerator=0.;denominator=0.;top=0.;total=0.
    # Form the analytic missing raising action, then project onto occupied
    # (R,q) support using the same 1e-8 budget as the pilot scalar audit.
    density=[];overlap=[]
    for start in range(0,len(u),4):
        sl=slice(start,start+4);block=u[sl]
        psi=block@basis[:nf]
        rho=np.sum(abs(psi)**2,axis=1)*dx
        defect=-g*np.sqrt(nf)*mu[sl,:,None]*block[:,:,-1,None]*basis[nf]
        overlap.append(np.sum(psi.conj()*defect,axis=1)*dx)
        density.append(rho)
        top+=float(np.sum(abs(block[:,:,-1])**2)*dx*dr)
        total+=float(np.sum(abs(block)**2)*dx*dr)
    rho=np.concatenate(density);numer=np.concatenate(overlap)
    mass=rho*weights[None,:]*dr
    mask,_,_=budget_support(rho,weights[None,:]*dr,1e-8)
    local=np.zeros_like(numer)
    np.divide(numer,rho,out=local,where=rho>0)
    return {'saved_time_au':time,'saved_time_fs':time*AU_FS,
            'analytic_projection_defect_RMS_Ha':weighted_rms(local,mass,mask),
            'top_fock_population':top/total,'top_fock_amplitude_norm':float(np.sqrt(top/total)),
            'nf':nf,'wave_sha256':digest(wave),
            'note':'Analytic omitted raising action; diagnostic only, no scalar correction.'}


def compare(frames,inventory):
    result={'status':'MISSING','comparisons':[],
            'note':'Only matching saved times and verified packet provenance are compared.'}
    cases=('barrier','barrier_F160')
    if not all(case in frames for case in cases):
        result['cases']={case:inventory.get(case,{}).get('packet_provenance',{}).get('status','MISSING') for case in cases}
        return result
    (p120,w120),(p160,w160)=(frames[case] for case in cases)
    # Restrict to inventory-selected event frames and the saved final frame.
    def candidates(case,waves):
        selected={row['path'] for row in inventory[case]['events'].values() if row.get('status')=='AVAILABLE'}
        if waves:selected.add(str(max(waves,key=lambda pair:pair[0])[1]))
        return [(t,p) for t,p in waves if str(p) in selected]
    a=candidates(cases[0],w120);b=candidates(cases[1],w160)
    for ta,pa in a:
        match=next(((tb,pb) for tb,pb in b if abs(ta-tb)<1e-10),None)
        if match is None:continue
        tb,pb=match
        result['comparisons'].append({'saved_time_au':ta,'F120':diagnose(p120,pa),'F160':diagnose(p160,pb)})
    result['status']='AVAILABLE' if result['comparisons'] else 'UNKNOWN'
    return result
