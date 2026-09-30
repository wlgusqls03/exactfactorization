"""One entry point: existing real-grid waves OR small MCEF fields -> VSC gallery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from .run_real_grid_mcef_preview import ROOT, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--fields',type=Path,help='Existing fields_*.npz directory: fast render only')
    source.add_argument('--waves',type=Path,help='Original real-grid campaign root: compute fields first')
    p.add_argument('--input',type=Path,required=True,help='Original packet with omega and proton mass')
    p.add_argument('--observables',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--movie',action='store_true')
    p.add_argument('--coordinate',choices=('Q','q'),default='Q')
    p.add_argument('--support-budget',type=float,default=1e-8)
    p.add_argument('--density-floor',type=float,default=1e-5)
    p.add_argument('--fps',type=float,default=1)
    p.add_argument('--max-frames',type=int,help='Optional prefix subset; never changes physical times')
    a=p.parse_args()
    if not 0<a.support_budget<1 or not 0<a.density_floor<1 or a.fps<=0:p.error('Invalid plotting parameters')
    if a.max_frames is not None and a.max_frames<1:p.error('--max-frames must be positive')
    if a.waves and not a.observables:p.error('--waves requires --observables')
    out=a.out.resolve();out.relative_to(ROOT.resolve())
    if out.exists():raise FileExistsError('Choose a NEW --out; existing results are immutable')
    with np.load(a.input,allow_pickle=False) as z:
        packet={k:z[k] for k in ('R','x','mass','omega','g_chi','phi','dx','potential','tx')}
        mass=float(packet['mass']);omega=float(packet['omega']);R=packet['R']
    out.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(out/'system_cache'))
    if a.waves:
        a.fields=out/'analysis'
        cmd=[sys.executable,'-m','multi_component_exact_factorization.vsc_polariton.run_real_grid_mcef_preview',
             '--input',str(a.input),'--waves',str(a.waves),'--observables',str(a.observables),
             '--out',str(a.fields),'--support-budget',str(a.support_budget),'--density-floor',str(a.density_floor)]
        if a.max_frames:cmd+=['--max-frames',str(a.max_frames)]
        if a.movie:cmd+=['--movie']
        subprocess.run(cmd,check=True)
    paths=sorted(a.fields.glob('fields_*.npz'))
    if a.max_frames:paths=paths[:a.max_frames]
    if not paths:raise FileNotFoundError('No fields_*.npz found')
    frames=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:f={k:z[k] for k in z.files}
        np.testing.assert_allclose(f['R'],R,atol=1e-12,rtol=0)
        np.testing.assert_allclose(f['Q'],np.sqrt(omega)*f['q'],atol=1e-12,rtol=1e-12)
        frames.append(f)
    if any(float(b['time_au'])<=float(c['time_au']) for c,b in zip(frames,frames[1:])):
        raise ValueError('Frames must have strictly increasing actual times')
    from .vsc_mcef_gallery import gallery
    from .real_grid_mcef_plotting import snapshots,dynamics,prepared
    from .vsc_mcef_photon_barrier import vsc_diagnostics
    label=f'Full real-grid MCEF | cavity {omega*27211.386245988:.3f} meV'
    report=gallery(frames,out,mass,omega,label,a.support_budget,a.density_floor,a.coordinate,a.movie,a.fps)
    states=[prepared(f,a.support_budget,a.density_floor) for f in frames]
    report['photon_barrier']=vsc_diagnostics(frames,states,packet,out)
    # Existing TDPES/force/density figures remain in the SAME gallery family.
    report['tdpes']=snapshots(frames,out,label,a.support_budget,a.density_floor,a.movie)
    if a.observables:
        rows=[];times=[]
        for path in sorted(a.observables.glob('observable_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                times.append(float(z['time_au']));rows.append({k:z[k] for k in ('rho_R','product','flux')})
        if rows:report['dynamics']=dynamics(R,times,rows,out,label)
    report.update(input_sha256=digest(a.input),field_hashes={str(k):digest(k) for k in paths},
                  source_hashes={n:digest(Path(__file__).with_name(n)) for n in ('vsc_mcef_gallery.py','vsc_mcef_photon_barrier.py','run_vsc_mcef_gallery.py')})
    (out/'gallery_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved PNG/PDF and requested movies:',out)
    print('No propagation or model changes. Phase7 is NOT certified by rendering.')


if __name__=='__main__':main()
