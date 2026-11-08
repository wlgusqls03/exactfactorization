"""Sequential, independently restartable fixed-K/mode/coupling experiments."""
from dataclasses import replace
import json
from pathlib import Path

from ..gaussian_trajectory.run import atomic_json
from .run import parser,config_from_args,run


def main():
    p=parser()
    p.description=__doc__
    p.add_argument('--K',nargs='+',type=int,default=[1,2,3,4])
    p.add_argument('--modes',nargs='+',choices=['standard','energy'],default=['standard','energy'])
    p.add_argument('--couplings',nargs='+',type=float,default=[0.,.01])
    a=p.parse_args();base=config_from_args(a);root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    records=[]
    for g in dict.fromkeys(a.couplings):
        for K in dict.fromkeys(a.K):
            for mode in dict.fromkeys(a.modes):
                out=root/('g_%s'%format(g,'.12g'))/('K%d_%s'%(K,mode))
                config=replace(base,model=dict(base.model,g=g),components=K,mode=mode)
                try:
                    status=run(config,out,resume=a.resume and (out/'checkpoint.npz').exists())
                except (FileExistsError,ValueError) as exc:
                    # Refuse overwriting; do not replace an existing status.
                    status=dict(status='NOT_RUN',error=repr(exc))
                    print('NOT_RUN',out,exc,flush=True)
                records.append(dict(path=str(out.resolve()),g=g,K=K,mode=mode,**status))
                atomic_json(root/'sweep.json',dict(cases=records,scope='Fixed R only; no automatic convergence certification'))
                if (a.render or a.movies) and (out/'diagnostics.npz').exists():
                    from .render import render_case
                    render_case(out,movies=a.movies)
    if a.render or a.movies:
        from .render import render_sweep
        render_sweep(root)
    failed=sum(r['status']!='COMPLETE_NOT_CERTIFIED' for r in records)
    print('%d/%d cases did not reach target. Inspect sweep.json; no silent replacement.'%(failed,len(records)),flush=True)
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
