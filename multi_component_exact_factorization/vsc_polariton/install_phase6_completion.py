"""Install only NEW completion files from a portable bundle; never overwrite."""
import hashlib
import json
from pathlib import Path


def main():
    bundle = Path(__file__).resolve().parent
    repo = Path.cwd()
    target = repo/'multi_component_exact_factorization/vsc_polariton'
    if not (target/'run_phase6_gpu.py').exists():
        raise RuntimeError('Run installer from existing exactfactorization repository root')
    manifest = json.loads((bundle/'manifest.json').read_text())
    for name, digest in manifest.items():
        src = bundle/name
        src.resolve().relative_to(bundle.resolve())
        if hashlib.sha256(src.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Bundle checksum mismatch: '+name)
    pairs = []
    for src in (bundle/'code').rglob('*.py'):
        dst = target/src.relative_to(bundle/'code')
        dst.resolve().relative_to(target.resolve())
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise RuntimeError('Existing different source preserved: '+str(dst))
        pairs.append((src,dst))
    for src,dst in pairs:
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True)
            with dst.open('xb') as f:
                f.write(src.read_bytes())
    print('Verified inputs; installed new completion files only. Historical files unchanged.')


if __name__=='__main__':main()
