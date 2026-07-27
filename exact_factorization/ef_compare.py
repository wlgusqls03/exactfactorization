#!/usr/bin/env python3
"""Compare a direct EF propagation with a full-TDSE EF reference.

The comparison is performed on reconstructed molecular fields
``Psi(r,R,t)=Phi_R(r,t) chi(R,t)``. This avoids comparing gauge-dependent
``Phi`` or ``chi`` from archives that deliberately use different gauges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def reconstruct_direct_phi(data, frame):
    """Return one ``Phi(nr,nR)`` frame, loading or reconstructing it."""
    if "phi" in data:
        return data["phi"][frame]
    if "coefficients" not in data or "bo_states" not in data:
        raise ValueError("Direct archive contains neither real-space phi nor BO reconstruction data")
    return np.einsum(
        "sR,sRr->rR", data["coefficients"][frame], data["bo_states"], optimize=True
    )


def nearest_frame(times, target):
    """Return the scalar index of the saved time closest to ``target``."""
    return int(np.argmin(np.abs(times-target)))


def run(args):
    """Compare matching frames and print one row per saved direct-EF time."""
    with np.load(args.reference,allow_pickle=True) as refz, np.load(args.direct,allow_pickle=True) as dirz:
        ref={k:refz[k] for k in refz.files}
        direct={k:dirz[k] for k in dirz.files}
    if ref["r"].shape != direct["r"].shape or not np.allclose(ref["r"],direct["r"]):
        raise ValueError("Electronic grids differ")
    if ref["R"].shape != direct["R"].shape or not np.allclose(ref["R"],direct["R"]):
        raise ValueError("Nuclear grids differ")
    dr=float(ref["r"][1]-ref["r"][0]); dR=float(ref["R"][1]-ref["R"][0])
    # Each output row is (time, fidelity, nuclear-density L1 error,
    # maximum BO-population error), hence the final table shape is (nmatch,4).
    rows=[]
    for i,t in enumerate(direct["times_fs"]):
        j=nearest_frame(ref["times_fs"],t)
        if abs(float(ref["times_fs"][j]-t)) > args.time_tolerance_fs:
            continue
        phi_d=reconstruct_direct_phi(direct,i)       # (nr,nR)
        psi_d=phi_d*direct["chi"][i][None,:]        # (nr,nR)
        psi_r=ref["psi"][j]                         # (nr,nR)
        # Fidelity is insensitive to a global phase and to the EF gauge.
        overlap=np.sum(psi_r.conj()*psi_d)*dr*dR
        nr=np.sum(np.abs(psi_r)**2)*dr*dR
        nd=np.sum(np.abs(psi_d)**2)*dr*dR
        fidelity=abs(overlap)**2/(nr*nd)
        rho_r=np.sum(np.abs(psi_r)**2,axis=0)*dr     # (nR,)
        rho_d=np.sum(np.abs(psi_d)**2,axis=0)*dr     # (nR,)
        density_l1=np.sum(np.abs(rho_r-rho_d))*dR
        if (
            "populations" in ref and "populations" in direct
            and ref["populations"].shape[1:] == direct["populations"].shape[1:]
            and ref["populations"].shape[1] > 0
        ):
            pop_error=np.max(np.abs(ref["populations"][j]-direct["populations"][i]))
        else:
            # A pure real-space EF calculation deliberately has no BO
            # population because no BO eigenstates were ever constructed.
            pop_error=np.nan
        rows.append((float(t),float(fidelity),float(density_l1),float(pop_error)))
    if not rows:
        raise ValueError("No matching saved times; increase --time-tolerance-fs")
    table=np.asarray(rows)
    print(" time(fs)       fidelity       density_L1     max_pop_error")
    for row in table:
        print(f" {row[0]:8.4f}   {row[1]:.10f}   {row[2]:.3e}      {row[3]:.3e}")
    print(f"Minimum fidelity: {table[:,1].min():.10f}")
    print(f"Maximum density L1 error: {table[:,2].max():.3e}")
    finite_population = np.isfinite(table[:,3])
    if np.any(finite_population):
        print(f"Maximum BO population error: {table[finite_population,3].max():.3e}")
    else:
        print("BO population error: unavailable (pure real-space, no BO basis)")
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        np.savetxt(args.output,table,header="time_fs fidelity density_L1 max_population_error")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path)
    p.add_argument("direct",type=Path)
    p.add_argument("--time-tolerance-fs",type=float,default=1e-8)
    p.add_argument("--output",type=Path)
    return p.parse_args()


if __name__=="__main__":
    run(parse_args())
