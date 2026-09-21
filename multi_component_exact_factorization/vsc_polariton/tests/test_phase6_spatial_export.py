"""Preset isolation and failure safety for the local spatial packet adapter."""
import unittest
from unittest.mock import patch
from multi_component_exact_factorization.vsc_polariton import phase6_spatial_gpu_export as adapter


class SpatialExportTests(unittest.TestCase):
    def test_presets_and_restoration(self):
        original = dict(adapter.resume.SETTINGS)
        old_out = adapter.resume.OUT
        seen = []

        def fake_export(path, case, key):
            setting = adapter.resume.SETTINGS[key]
            self.assertEqual(case, 'resonant')
            self.assertEqual(setting[-2:], (120, .125))
            self.assertIn('spatial_transfer', str(path))
            seen.append(setting)

        with patch.object(adapter, 'export', side_effect=fake_export):
            adapter.main()
        self.assertEqual(seen, list(adapter.CANDIDATES.values()))
        self.assertEqual(adapter.resume.SETTINGS, original)
        self.assertEqual(adapter.resume.OUT, old_out)

    def test_failure_restores_runtime_state(self):
        original = dict(adapter.resume.SETTINGS)
        old_out = adapter.resume.OUT
        with patch.object(adapter, 'export', side_effect=FileExistsError):
            with self.assertRaises(FileExistsError):
                adapter.main()
        self.assertEqual(adapter.resume.SETTINGS, original)
        self.assertEqual(adapter.resume.OUT, old_out)
