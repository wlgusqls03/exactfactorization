"""Remove ONLY two SHA-verified archives already received and audited locally.

No directory, wave, restart, status, input or diagnostic deletion. Unknown or
changed archives are skipped. Default is dry-run. Symlinks are never removed.
"""
import argparse
from pathlib import Path
from .run_phase6_gpu import sha

RECEIVED={
    'phase7_f160_events.tar.gz':'3c88bd6d33da8ec70baa0631748bffb80940cd57cb46d1deb6bf3c3179cbe8f4',
    'phase7_event_waves.tar.gz':'1b7e1ff8ac98a305f49c2bca1d1f2e46f1b0b9d96a8c485d42622b9debe959d9'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delete',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2];total=0
    for name,digest in RECEIVED.items():
        for path in root.rglob(name):
            if path.is_symlink() or not path.is_file():continue
            if sha(path)!=digest:
                print('SKIP (different hash)',path,flush=True);continue
            size=path.stat().st_size
            print('DELETE' if a.delete else 'WOULD DELETE',path,f'{size/1024**3:.3f} GiB',flush=True)
            if a.delete:path.unlink()
            total+=size
    print(f'Total {total/1024**3:.3f} GiB. All original scientific files preserved.',flush=True)


if __name__=='__main__':main()
