import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from multi_component_exact_factorization.audit_tdpes_components import audit


class TDPESAuditTests(unittest.TestCase):
    def test_streaming_detects_a_mismatched_saved_total(self):
        arrays = {}
        for level in (1, 2):
            shape = (3, 4, 2) if level == 1 else (3, 2)
            parts = []
            for i, suffix in enumerate(("wbo_0", "wbo_excited", "gd", "geo_q", "geo_R")):
                part = np.full(shape, float(i))
                arrays[f"tdpes{level}_{suffix}"] = part
                parts.append(part)
            arrays[f"tdpes{level}_total"] = sum(parts)
        arrays["tdpes2_total"][1, 0] += 0.25
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"tdse_exact_factorization_fields.npz"
            np.savez_compressed(path, **arrays)
            result = audit(path)
        np.testing.assert_array_equal(result["tdpes1"], [0, 0, 0])
        np.testing.assert_array_equal(result["tdpes2"], [0, 0.25, 0])
