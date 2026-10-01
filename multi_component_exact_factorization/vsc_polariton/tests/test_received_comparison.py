"""Non-mutating comparison safety checks."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from ..compare_received_movies import matched_times,momentum_movie
from .test_real_grid_mcef_preview import fixture
from ..real_grid_mcef_fields import analyze
from ..vsc_movie_only import compact_fields


class ReceivedComparisonTests(unittest.TestCase):
    def test_times_must_match_without_interpolation(self):
        matched_times([0,4,8],[0,4,8])
        with self.assertRaises(ValueError):matched_times([0,4,8],[0,8])
        with self.assertRaises(ValueError):matched_times([0,4,8],[0,4,9])

    def test_momentum_movie_from_current_and_conditional_fields(self):
        u,p,Q=fixture();f=analyze(u,p,Q)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fields=root/'fields';obs=root/'obs';fields.mkdir();obs.mkdir()
            for i in range(2):
                f['time_au']=i*4.
                np.savez(fields/f'fields_{i:07d}.npz',**compact_fields(f))
                np.savez(obs/f'observable_{i:07d}.npz',time_au=i*4.,rho_R=f['rho_R'],current=f['current'])
            momentum_movie(fields,obs,p['R'],root,mass=float(p['mass']))
            self.assertGreater((root/'free_vs_coupled_mcef_momenta.mp4').stat().st_size,0)


if __name__=='__main__':unittest.main()
