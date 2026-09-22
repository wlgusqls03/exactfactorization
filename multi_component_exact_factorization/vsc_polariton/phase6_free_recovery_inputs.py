"""Local export: exact free sector with finer nuclear grids, no old writes.

Uses the identical published Hamiltonian and Phase3 threshold preparation.
R grids: [-4.4,4.4), NR440/550. Box scan: 28.8,36,48,72; dx .3/.24.
No interpolation of a fitted PES; BO eigenvalues are independently solved.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .phase6_electronic_basis import aligned_basis
from .fgh_reference import FGHGrid
from .full_pf_3d import FullPF
from .cavity_hamiltonian import Cavity
from .phase6_split import FullSplit
from .phase6_observables import observe
from .phase6_gpu_backend import PFBackend
from .run_phase6_gpu import load_packet
from .model_shin_metiu import HARTREE_EV


def main():
    root=Path('results/vsc_polariton/phase6_gpu/free_recovery_transfer_v2')
    src=Path('results/vsc_polariton/phase6_gpu/completion_transfer/inputs/free_F120.npz')
    original,sha=load_packet(src);prep=json.loads(str(original['metadata']))['preparation']
    mol=json.loads(Path('results/vsc_polariton/phase2/phase2_results.json').read_text())['molecular']
    cavity=Cavity(float(original['omega']),0.,mol['omega_well_meV']/1000/HARTREE_EV,mol['muprime_well'])
    for half in (28.8,36.,48.,72.):
        for dx,nr in ((.3,440),(.24,440),(.3,550)):
            suffix='_R550' if nr==550 else ''
            name=f'free_L{half:g}_dx{dx:g}{suffix}'
            path=root/'inputs'/(name+'.npz')
            if path.exists():raise FileExistsError(path)
            xg,rg=FGHGrid(2*round(half/dx),dx),FGHGrid(nr,8.8/nr)
            energies,phi,_=aligned_basis(xg,rg.points())
            h=FullPF(xg,rg,cavity,1);R=h.R
            chi=np.exp(-float(original['mass'])*prep['omega0']*(R-mol['R_react'])**2/2
                       +1j*prep['momentum']*(R-mol['R_react']))
            chi/=np.sqrt(np.sum(abs(chi)**2)*rg.dx)
            bare_energy=float((np.vdot(chi,np.fft.ifft(h.tr*np.fft.fft(chi))).real
                               +np.sum(abs(chi)**2*energies[:,0]))*rg.dx)
            if abs(bare_energy-prep['target'])>=1e-8:
                raise RuntimeError(('Threshold energy changed',name,bare_energy,prep['target']))
            u=phi[:,:,:1]*chi[:,None,None]
            packet=dict(psi=u,R=R,x=h.x,dx=dx,dR=rg.dx,mass=cavity.mass,
                        omega=cavity.omega_c,g_chi=0.,tx=h.tx,tr=h.tr,potential=h.potential,
                        dse=h.dse,mu=h.mu,photon=h.photon,rotation=np.ones((1,1)),
                        displacement=np.zeros(1),phi=phi,dt=.125)
            backend=PFBackend(packet,.125);legacy=FullSplit(h,.125)
            step=float(np.linalg.norm(backend.step(u)-legacy.step(u))*np.sqrt(h.volume))
            action=float(np.linalg.norm(backend.action(u)-h.action(u))*np.sqrt(h.volume))
            a,b=backend.observe(u),observe(u,h,phi)
            error=max(float(np.max(abs(np.asarray(a[k])-np.asarray(b[k])))) for k in b)
            if max(step,action,error)>=1e-10:raise RuntimeError((step,action,error))
            matching=float(np.max(abs(np.sum(abs(u)**2,axis=(1,2))*dx-abs(chi)**2)))
            if max(matching,abs(a['norm']-1),abs(a['P_exc']))>=1e-10:
                raise RuntimeError('Initial-state matching/normalization failed')
            metadata=dict(format_version=1,config=dict(case='free',setting=[half,xg.n,4.4,nr,1,.125]),
                          preparation=prep,parent_input_sha256=sha,threshold_energy_Ha=bare_energy,
                          nuclear_matching=matching,
                          interpretation='Exact eta=0 photon vacuum sector; explicit electronic propagation',
                          legacy_adapter_validation=dict(status='PASS',step_L2=step,action_L2=action,
                                                         observable_max_error=error))
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:np.savez_compressed(f,**packet,metadata=json.dumps(metadata))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with path.with_suffix('.json').open('x') as f:json.dump(dict(sha256=digest,metadata=metadata),f,indent=2)
            print(name,'initial energy error',bare_energy-prep['target'],'adapter',max(step,action,error),flush=True)


if __name__=='__main__':main()
