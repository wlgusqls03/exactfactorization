"""Check stored decomposition closure without extracting large NPZ members."""
import argparse
from pathlib import Path

import numpy as np

from multi_component_exact_factorization_discrete_gpu.compare_tdse import _stream_arrays


def audit(path):
    path = Path(path)
    if path.is_dir():
        path = path / "tdse_exact_factorization_fields.npz"
    result = {}
    for level in (1, 2):
        prefix = f"tdpes{level}_"
        keys = [prefix + suffix for suffix in (
            "total", "wbo_0", "wbo_excited", "gd", "geo_q", "geo_R",
        )]
        errors = []
        with _stream_arrays(path, keys) as readers:
            count = readers[keys[0]].shape[0]
            for frame in range(count):
                total = readers[keys[0]].read(frame)
                parts = [readers[key].read(frame) for key in keys[1:]]
                if not all(np.all(np.isfinite(a)) for a in [total] + parts):
                    raise ValueError(f"nonfinite TDPES{level}, frame {frame}")
                difference = total - sum(parts)
                errors.append(float(np.max(np.abs(difference))))
        result[f"tdpes{level}"] = np.asarray(errors)
        print(f"TDPES{level}: frames={count}, max_abs={max(errors):.16e}, "
              f"worst_frame={int(np.argmax(errors))}", flush=True)
    print("This checks stored component closure, not independent native-Hamiltonian equivalence.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    audit(parser.parse_args().archive)
