"""Synthetic cache rendering only: these fixtures are not simulated physics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib.pyplot as plt
import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.gaussian import pack_parameters
from multi_component_exact_factorization.gaussian_trajectory.renderer import (
    prepare, plot_snapshot, render,
)


def synthetic_cache(root, *, optional=True):
    root=Path(root)
    (root/"frames").mkdir(parents=True)
    (root/"config.json").write_text(json.dumps({"gaussian_coordinates":"transformed","fixture":"synthetic; not dynamics"}))
    q=np.linspace(-5,5,81); dq=np.full(81,q[1]-q[0]); dq[[0,-1]]*=.5
    R=np.linspace(-1,1,11); eta=np.full(11,1/11)
    for i in range(3):
        t=i*.05; current_R=R+.3*t; center=.25*R+2*t
        rho=np.exp(-.5*((q[None,:]-center[:,None])/.7)**2)
        rho/=np.sum(rho*dq,axis=1,keepdims=True)
        z=np.array([pack_parameters([1.],[c],[.7],coordinates="transformed") for c in center])
        heavy_R=np.linspace(-2,2,128)
        a=np.broadcast_to(.15*np.sin(q),(11,81)).copy()
        b=np.broadcast_to(.3*current_R[:,None]+.1*q,(11,81)).copy()
        alpha=np.einsum("iq,iq,q->i",rho,b,dq)
        data={"t_fs":t,"R":current_R,"eta":eta,"q":q,"quadrature":dq,"rho":rho}
        if optional:
            data.update(P=np.zeros(11),a=a,b=b,delta_b=b-alpha[:,None],alpha=alpha,
                        heavy_eval_R=heavy_R,heavy_density=np.exp(-heavy_R**2)/np.sqrt(np.pi),
                        Q_raw=.5*current_R,z=z,projection_residual=np.full(11,1e-5*(i+1)),
                        pnc_error=1e-12*(i+1),MLS_condition=np.full(11,10.),
                        population=np.tile([.6-.1*i,.4+.1*i],(11,1)),
                        photon_number=np.full(11,.02*i),energy=np.full(11,.4+1e-6*i),
                        kinetic_R=np.full(11,.03))
        np.savez_compressed(root/"frames"/("frame_%06d.npz"%i),**data)


class GaussianRendererTests(unittest.TestCase):
    def test_png_and_missing_encoder_preserve_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            raw={p.name:p.read_bytes() for p in (Path(tmp)/"frames").glob("*.npz")}
            with patch("shutil.which",return_value=None),warnings.catch_warnings(record=True) as seen:
                report=render(tmp,movies=True,dpi=35,snapshot_indices=[1])
            self.assertEqual(report["movie_status"],"ffmpeg_unavailable")
            self.assertTrue(any("ffmpeg unavailable" in str(w.message) for w in seen))
            self.assertGreater((Path(tmp)/"plots"/"summary.png").stat().st_size,1000)
            self.assertGreater((Path(tmp)/"plots"/"snapshot_000001.png").stat().st_size,1000)
            self.assertEqual(raw,{p.name:p.read_bytes() for p in (Path(tmp)/"frames").glob("*.npz")})

    def test_fixed_ranges_and_shared_momentum_color_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            d=prepare(tmp)
            self.assertEqual(d["limits"]["b"],d["limits"]["delta_b"])
            figs=[]
            for i in (0,2):
                from multi_component_exact_factorization.gaussian_trajectory.renderer import _load_frame
                figs.append(plot_snapshot(_load_frame(d["files"][i]),d))
            try:
                for j in range(8):
                    np.testing.assert_allclose(figs[0].axes[j].get_xlim(),figs[1].axes[j].get_xlim())
                    np.testing.assert_allclose(figs[0].axes[j].get_ylim(),figs[1].axes[j].get_ylim())
            finally:
                for f in figs: plt.close(f)

    def test_optional_quantities_skip_without_inventing_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp,optional=False)
            report=render(tmp,dpi=30,snapshot_indices=[0])
            self.assertIn("exact epsilon1/epsilon2",report["unavailable_by_design"])
            self.assertEqual(report["frames"],3)

    def test_native_empty_parameters_and_primitive_grid_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp)
            root=Path(tmp)
            (root/"config.json").write_text(json.dumps({"compression":"native"}))
            (root/"metadata.json").write_text(json.dumps({"model":{"basis":"fixed electronic Dirichlet grid, sqrt(dx)-weighted coefficients"}}))
            for path in (root/"frames").glob("*.npz"):
                with np.load(path) as z: frame={key:z[key] for key in z.files}
                frame["z"]=np.empty((11,0))
                frame["population"]=np.tile(np.linspace(1,8,8)/36,(11,1))
                np.savez_compressed(path,**frame)
            d=prepare(root)
            self.assertEqual(d["run_label"],"Uncompressed MTE reference")
            self.assertEqual(d["traces"]["weights"].ndim,1)
            report=render(root,dpi=30,snapshot_indices=[0])
            self.assertEqual(report["frames"],3)


if __name__=="__main__":
    unittest.main()
