import unittest
import tempfile
from pathlib import Path
import numpy as np
from .test_real_grid_mcef_preview import fixture
from ..real_grid_mcef_fields import action
from ..qhj_saved_wave import outer_qhj,joint_qhj


class QHJTests(unittest.TestCase):
    def test_harmonic_packet_balance(self):
        u,p,Q=fixture(boost=.3);q=Q/np.sqrt(p['omega']);u=u*p['omega']**.25
        f=outer_qhj(u,action(u,p,q),p['dR'],p['dx']*(q[1]-q[0]),p['mass'])
        m=f['rho_R']>1e-4*f['rho_R'].max();R=p['R']
        np.testing.assert_allclose(f['EF_force'][m],-R[m],atol=2e-7)
        np.testing.assert_allclose(f['quantum_force'][m],R[m]/p['mass'],atol=2e-7)
        np.testing.assert_allclose(f['net_flow_force'][m],-R[m]+R[m]/p['mass'],atol=2e-7)
        self.assertLess(abs(f['balance_residual'][m]).max(),1e-12)
        self.assertLess(abs(f['QHJ_residual'][m]).max(),1e-12)

    def test_vacuum_photon_force_cancellation(self):
        u,p,Q=fixture();q=Q/np.sqrt(p['omega']);u=u*p['omega']**.25
        f=joint_qhj(u,action(u,p,q),p['dR'],p['dx'],q[1]-q[0],p['mass'])
        mr=abs(p['R'])<2;mq=abs(Q)<2
        np.testing.assert_allclose(f['photon_scalar_force'][mr][:,mq],
            np.broadcast_to(-p['omega']**2*q[mq],(mr.sum(),mq.sum())),atol=1e-7)
        for key in ('a','photon_net_force','photon_material','berry'):
            self.assertLess(abs(f[key][mr][:,mq]).max(),1e-7)
        from ..real_grid_mcef_fields import analyze
        g=analyze(u/p['omega']**.25,p,Q)
        np.testing.assert_allclose(f['epsilon1_QHJ'][mr][:,mq],g['epsilon1_A'].real[mr][:,mq],atol=1e-7)

    def test_event_rendering(self):
        from ..run_qhj_comparison import render
        u,p,Q=fixture();q=Q/np.sqrt(p['omega']);u=u*p['omega']**.25;h=action(u,p,q)
        outer=outer_qhj(u,h,p['dR'],p['dx']*(q[1]-q[0]),p['mass'])
        joint=joint_qhj(u,h,p['dR'],p['dx'],q[1]-q[0],p['mass'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[]
            for i in range(2):
                arrays=dict(R_free=p['R'],R_coupled=p['R'],Q=Q,time_au=4.*i,omega_free=p['omega'],omega_coupled=p['omega'])
                for name in ('free','coupled'):
                    arrays.update({name+'_'+k:v for k,v in outer.items()})
                    arrays.update({name+'_joint_'+k:v for k,v in joint.items()})
                path=root/f'qhj_{i}.npz';np.savez(path,**arrays);paths.append(path)
            render(paths,root)
            for name in ('outer','photon'):self.assertGreater((root/f'qhj_{name}_events.mp4').stat().st_size,0)
