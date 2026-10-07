"""One command: saved-wave audit, display-only rerender, single return archive."""
import argparse
from pathlib import Path
import shutil
import tarfile
from .event_audit import audit,sha
from .review_plot import render,preflight
from .run import atomic_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--no-movies',action='store_true',help='Figures and audit only')
    args=ap.parse_args();root=args.campaign.resolve();out=args.out.resolve()
    if out.exists():raise FileExistsError('Preserve previous review; choose a NEW --out')
    if root==out or root in out.parents:raise ValueError('Output must be outside original campaign')
    preflight(root,movies=not args.no_movies)
    if shutil.disk_usage(out.parent).free<1024**3:raise OSError('Need 1 GiB free for audit, movies and return archive')
    out.mkdir();summary=audit(root,out/'analysis',pack=False)
    render(root,out/'plots',movies=not args.no_movies)
    for name,record in summary['inputs'].items():
        if sha(root/name)!=record['sha256']:raise RuntimeError(f'Original changed: {name}')
    archive=out/'model_a_next_review.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name in summary['inputs']:tar.add(root/name,arcname='campaign/'+name,recursive=False)
        for folder in ['analysis','plots']:
            for p in sorted((out/folder).rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
    atomic_json(out/'transfer.json',dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),
        note='Existing waves reused; no propagation or gate upgrade. Original campaign preserved.'))
    print(f'SEND THIS FILE: {archive}',flush=True)


if __name__=='__main__':main()
