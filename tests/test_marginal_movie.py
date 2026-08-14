from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from multi_component_exact_factorization.marginal_movie import (
    make_fixed_scale_marginal_animation,
)
from multi_component_exact_factorization.coordinate_focus_movie import (
    make_coordinate_focus_animation,
)


class MarginalMovieTests(unittest.TestCase):
    def test_fixed_display_movie_does_not_modify_density(self):
        times = np.array([0.0, 1.0])
        x = np.linspace(-22.0, 22.0, 31)
        q = np.linspace(-12.0, 12.0, 21)
        R = np.linspace(-12.0, 12.0, 21)
        electron = np.vstack((np.exp(-x*x), np.exp(-(x-1.0)**2)))
        proton = np.vstack((4.0*np.exp(-q*q), np.exp(-(q+1.0)**2)))
        heavy = np.vstack((3.8*np.exp(-(R-2.0)**2), np.exp(-(R-3.0)**2)))
        originals = tuple(values.copy() for values in (electron, proton, heavy))
        with TemporaryDirectory() as temporary:
            path = make_fixed_scale_marginal_animation(
                times_fs=times,
                particle_series=(
                    ("electron", x, electron),
                    ("proton", q, proton),
                    ("heavy", R, heavy),
                ),
                options={"left_position": -10.0, "right_position": 10.0},
                outdir=temporary, fps=2, max_frames=2, dpi=25, fmt="gif",
                y_max=1.5, x_abs_max=12.0,
            )
            self.assertTrue(Path(path).is_file())
        for values, original in zip((electron, proton, heavy), originals):
            self.assertTrue(np.array_equal(values, original))

    def test_coordinate_movie_changes_only_view_window(self):
        times = np.array([0.0, 1.0])
        q = np.linspace(-12.0, 12.0, 41)
        density = np.vstack((
            np.exp(-((q+2.0)/0.7)**2),
            np.exp(-((q-3.0)/1.5)**2),
        ))
        profiles = (
            (r"$\epsilon^{(1)}$", "energy", q[None, :]+times[:, None],
             "#4c78a8", False),
            (r"$K_q$", "momentum", np.sin(q)[None, :]*(1+times[:, None]),
             "black", True),
            (r"$J_q$", "current", np.cos(q)[None, :]*(1+times[:, None]),
             "#2a9d8f", True),
        )
        original_density = density.copy()
        original_profiles = tuple(item[2].copy() for item in profiles)
        with TemporaryDirectory() as temporary:
            path = make_coordinate_focus_animation(
                times_fs=times, coordinate=q, marginal=density,
                profiles=profiles,
                options={"left_position": -10.0, "right_position": 10.0},
                outdir=temporary, fps=2, max_frames=2, dpi=25, fmt="gif",
                particle_name="Proton", coordinate_symbol="q",
                color="#e76f51", stem="proton_coordinate_dynamics",
                marginal_ymax=1.5, marginal_xmax=12.0,
            )
            self.assertTrue(Path(path).is_file())
        self.assertTrue(np.array_equal(density, original_density))
        for item, original in zip(profiles, original_profiles):
            self.assertTrue(np.array_equal(item[2], original))


if __name__ == "__main__":
    unittest.main()
