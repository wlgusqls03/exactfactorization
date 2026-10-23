"""Refine ONLY static polaritonic references, preserving a completed case.

The TDSE wave, saved fields and time-dependent projected populations are not
changed. A larger Fock basis here does not retroactively improve those old
populations. Original full-x CBO references are copied without alteration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import DEFAULTS
from .projections import build_analysis
from .workflow import load_tree, source_record, write_json, write_npz


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _molecular_cache(case_dir, config):
    """Read an existing matching cache only; do not touch old cache/lock files."""
    names=set(DEFAULTS)|{'local_wall_R','dividing_R'}
    requested={k:v for k,v in config.items() if k in names}
    candidates=[]
    for parent in (case_dir,case_dir.parent,case_dir.parent.parent):
        candidates.extend(sorted((parent/'cache').glob('molecular_*.npz')))
    for path in dict.fromkeys(candidates):
        packet=load_tree(path)
        cached={k:v for k,v in packet['config'].items() if k in names}
        if cached==requested:
            return packet,path
    raise FileNotFoundError('No matching existing molecular cache near case directory; '
                            'restore its cache before static-reference refinement')


def _pair_record(analysis, omega):
    if 'vib_pol_energy' not in analysis:
        return None
    idx=np.asarray(analysis['vib_pol_pair_indices'],dtype=int)
    energies=np.asarray(analysis['vib_pol_energy'])[idx]
    gap=float(energies[1]-energies[0])
    return dict(indices=idx.tolist(),energies=energies.tolist(),gap_Ha=gap,
                gap_over_omega=gap/omega)


def refine(case_dir, out, *, fock=16, compare_fock=8):
    """Calculate every saved R point, and write a new, separately named cache."""
    case_dir=Path(case_dir).resolve();out=Path(out).resolve()
    if out==case_dir or case_dir in out.parents:
        raise ValueError('Refinement output must be outside the original case directory')
    if out.exists():
        raise FileExistsError('Existing refinement output preserved: '+str(out))
    config=json.loads((case_dir/'config.json').read_text())
    metadata=json.loads((case_dir/'metadata.json').read_text())
    current=source_record()
    if metadata.get('physics_sources')!=current['physics_sources']:
        raise ValueError('Original metadata physics source hashes do not match current code')
    original_fock=int(config['n_fock_analysis'])
    fock=int(fock);compare_fock=int(compare_fock)
    if not original_fock<=compare_fock<fock:
        raise ValueError('Require original Fock size <= comparison Fock size < refined Fock size')
    packet,cache_path=_molecular_cache(case_dir,config)
    with np.load(case_dir/'references.npz',allow_pickle=False) as z:
        original={k:z[k].copy() for k in z.files}
    original_analysis=load_tree(case_dir/'analysis_cache.npz')
    inputs=[case_dir/'config.json',case_dir/'metadata.json',case_dir/'references.npz',
            case_dir/'analysis_cache.npz',cache_path]
    hashes={str(path):_digest(path) for path in inputs}
    Q=np.asarray(original['Q']);R=np.asarray(original['R'])
    omega=float(config['omega']);g=float(config['g_cavity'])
    if not np.array_equal(R,packet['R']):
        raise ValueError('Reference and molecular-cache R grids differ')
    compared={}
    for count in (compare_fock,fock):
        print(f'Static reference: nFock={count}, all {len(R)} R points; TDSE/fields unchanged',flush=True)
        settings=dict(config,n_fock_analysis=count)
        compared[count]=build_analysis(packet,omega,g,Q,settings)
    target=compared[fock];nplot=original['pol_energy'].shape[1]
    updated=dict(original)
    updated['pol_energy']=target['pol_energy'][:,:nplot]
    for key in ('pol_pair_indices','pol_pair_mixing','electronic_pair_mixing_score',
                'strongest_electronic_pair_mixing_R'):
        updated[key]=target[key]
    updated['pol_pair_retained_weight']=target['pol_pair_weight']
    if 'vib_pol_energy' in target:
        updated['vib_pol_energy']=target['vib_pol_energy']
        updated['vib_pol_levels']=target['vib_pol_energy'][:nplot]
        updated['vib_pol_pair_indices']=target['vib_pol_pair_indices']
    updated.update(reference_fock_size=np.asarray(fock),
        original_population_fock_size=np.asarray(original_fock),
        reference_source_case=np.asarray(str(case_dir)),
        reference_refinement_scope=np.asarray('static electron-photon and global vibrational-polariton '
            'eigenvalues only; original TDSE, fields, CBO surfaces and projected populations unchanged'))
    rho=None;rho_path=None
    field_files=sorted((case_dir/'fields').glob('fields_*.npz'))
    if field_files:
        rho_path=field_files[0]
        with np.load(rho_path,allow_pickle=False) as z:
            rho=np.asarray(z['rho_R'],dtype=float)
        inputs.append(rho_path);hashes[str(rho_path)]=_digest(rho_path)
    peak=int(np.argmax(rho)) if rho is not None else int(np.argmin(abs(R-config['R0'])))
    arrays={'original':original['pol_energy'],
            'comparison':compared[compare_fock]['pol_energy'][:,:nplot],
            'refined':target['pol_energy'][:,:nplot]}
    differences={}
    for left,right in (('original','comparison'),('comparison','refined'),('original','refined')):
        diff=arrays[right]-arrays[left]
        entry=dict(max_absolute_Ha=float(np.max(abs(diff))),
                   peak_R_difference_Ha=diff[peak].tolist())
        if rho is not None:
            entry['initial_density_weighted_RMS_Ha']=np.sqrt(
                np.sum(rho[:,None]*diff**2,axis=0)/np.sum(rho)).tolist()
        differences[left+'_to_'+right]=entry
    report=dict(scope='Static reference Fock convergence only; no dynamics rerun or population reprojection',
        source_case=str(case_dir),molecular_cache=str(cache_path),
        physics_sources_verified=True,physics_sources=current['physics_sources'],
        refinement_source_sha256=_digest(__file__),git_head=current['git_head'],
        input_hashes=hashes,n_R=len(R),n_BO=int(config['n_bo']),omega=omega,g_cavity=g,
        original_fock=original_fock,compare_fock=compare_fock,refined_fock=fock,
        selected_R=float(R[peak]),selected_R_kind='initial marginal-density peak' if rho is not None else 'nearest R0',
        fixed_R={label:dict(energies_Ha=e[peak].tolist(),gaps_Ha=np.diff(e[peak]).tolist(),
                            gaps_over_omega=(np.diff(e[peak])/omega).tolist()) for label,e in arrays.items()},
        differences=differences,
        global_vibrational_pairs=dict(original=_pair_record(original_analysis,omega),
            comparison=_pair_record(compared[compare_fock],omega),refined=_pair_record(target,omega)),
        limitations=['Electronic BO analysis dimension is unchanged and still requires convergence.',
                     'Original time-dependent populations retain their original Fock basis.',
                     'Old full-x CBO reference arrays are preserved byte-for-value.',
                     'A static Fock convergence check does not validate TDSE spatial/time convergence.'])
    if any(_digest(path)!=hashes[str(path)] for path in inputs):
        raise RuntimeError('An input changed during refinement; do not refine an actively modified case')
    out.mkdir(parents=True,exist_ok=False)
    name=f"refined_refs_{config['campaign']}_{config['case']}.npz"
    write_npz(out/name,**updated)
    report.update(refined_reference=str(out/name),all_input_hashes_unchanged=True)
    write_json(out/'reference_convergence.json',report)
    write_npz(out/'reference_fock_comparison.npz',R=R,
        original_pol_energy=arrays['original'],comparison_pol_energy=arrays['comparison'],
        refined_pol_energy=arrays['refined'],original_fock=original_fock,
        comparison_fock=compare_fock,refined_fock=fock)
    print(out/name,flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fock',type=int,default=16)
    p.add_argument('--compare-fock',type=int,default=8)
    a=p.parse_args(argv)
    refine(a.case_dir,a.out,fock=a.fock,compare_fock=a.compare_fock)


if __name__=='__main__':main()
