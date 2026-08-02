#!/usr/bin/env python3
"""Occupied-support exact potentials, gauge-invariant currents, and BO NACs."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from result_paths import dated_results_dir

from .core import AU_PER_FS, build_model, derivative, local_electronic_basis
from .visualize import archive_arguments, load_archive


def weighted_rms(field, weight):
    """Probability-weighted RMS for arrays whose leading axis is time."""
    axes = tuple(range(1, field.ndim))
    numerator = np.sum(weight*np.abs(field)**2, axis=axes)
    denominator = np.sum(weight, axis=axes)
    return np.sqrt(numerator/np.maximum(denominator, 1.0e-300))


def phase_gradient(wavefunction, spacing, axis, density_floor=1.0e-14):
    """Return ``Im(f* df)/|f|^2`` without explicitly unwrapping its phase.

    For ``f=A exp(iT)`` this is ``dT``.  The ratio is only interpreted after
    applying an occupied-density mask, but the floor keeps node cells finite.
    """
    density = np.abs(wavefunction)**2
    numerator = np.imag(
        np.conj(wavefunction)*derivative(wavefunction, spacing, axis=axis)
    )
    return numerator/np.maximum(density, density_floor)


def gauge_invariant_diagnostics(data):
    """Return covariant currents and electric-field-like exact forces."""
    options = archive_arguments(data)
    q, R = data["q"], data["R"]
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    times_au = np.asarray(data["times_fs"])*AU_PER_FS
    lam = np.asarray(data["lambda_wavefunction"])
    chi = np.asarray(data["chi"])
    a, b = np.asarray(data["a"]), np.asarray(data["b"])
    alpha = np.asarray(data["alpha"])
    eps1, eps2 = np.asarray(data["epsilon_1"]), np.asarray(data["epsilon_2"])
    joint = np.abs(lam)**2*np.abs(chi)[:, None, :]**2
    heavy = np.abs(chi)**2

    phase_q_lam = phase_gradient(lam, dq, axis=1)
    phase_R_lam = phase_gradient(lam, dR, axis=2)
    phase_R_chi = phase_gradient(chi, dR, axis=1)
    momentum_q = phase_q_lam+a
    momentum_R_first = phase_R_lam+phase_R_chi[:, None, :]+b
    momentum_R_outer = phase_R_chi+alpha

    p_q_lam = -1j*derivative(lam, dq, axis=1)
    proton_current = (
        np.abs(chi)[:, None, :]**2
        *np.real(np.conj(lam)*(p_q_lam+a*lam))
        /float(options["proton_mass"])
    )
    p_R_chi = -1j*derivative(chi, dR, axis=1)
    heavy_current = np.real(np.conj(chi)*(p_R_chi+alpha*chi))/float(
        options["heavy_mass"]
    )
    first_heavy_current = joint*momentum_R_first/float(options["heavy_mass"])

    if len(times_au) >= 2:
        edge_order = 2 if len(times_au) >= 3 else 1
        da_dt = np.gradient(a, times_au, axis=0, edge_order=edge_order)
        db_dt = np.gradient(b, times_au, axis=0, edge_order=edge_order)
        dalpha_dt = np.gradient(alpha, times_au, axis=0, edge_order=edge_order)
    else:
        da_dt = np.zeros_like(a)
        db_dt = np.zeros_like(b)
        dalpha_dt = np.zeros_like(alpha)
    force_q = -derivative(eps1, dq, axis=1)+da_dt
    force_R_first = -derivative(eps1, dR, axis=2)+db_dt
    force_R = -derivative(eps2, dR, axis=1)+dalpha_dt
    curvature_qR = derivative(a, dR, axis=2)-derivative(b, dq, axis=1)
    return dict(
        joint_density=joint, heavy_density=heavy,
        phase_gradient_q_lam=phase_q_lam,
        phase_gradient_R_lam=phase_R_lam,
        phase_gradient_R_chi=phase_R_chi,
        momentum_q=momentum_q,
        momentum_R_first=momentum_R_first,
        momentum_R_outer=momentum_R_outer,
        proton_current=proton_current, heavy_current=heavy_current,
        first_heavy_current=first_heavy_current,
        force_q=force_q, force_R_first=force_R_first, force_R=force_R,
        curvature_qR=curvature_qR,
        support_rms_a=weighted_rms(a, joint),
        support_rms_b=weighted_rms(b, joint),
        support_rms_alpha=weighted_rms(alpha, heavy),
        support_rms_momentum_q=weighted_rms(momentum_q, joint),
        support_rms_momentum_R_outer=weighted_rms(momentum_R_outer, heavy),
        support_rms_curvature_qR=weighted_rms(curvature_qR, joint),
        support_rms_force_q=weighted_rms(force_q, joint),
        support_rms_force_R=weighted_rms(force_R, heavy),
    )


def nonadiabatic_couplings(data, n_states=3):
    """Compute |<n|d_coordinate H|m>/(E_m-E_n)| without phase derivatives."""
    if n_states < 3:
        raise ValueError("E2-E1과 d12 분석에는 최소 3개 BO 상태가 필요합니다.")
    options = archive_arguments(data)
    model = build_model(SimpleNamespace(**options))
    energies, states = local_electronic_basis(model, n_states)
    dV_q = derivative(model.potential, model.dq, axis=1)
    dV_R = derivative(model.potential, model.dR, axis=2)
    result = {"energies": energies}
    for lower, upper in ((0, 1), (1, 2)):
        gap = energies[upper]-energies[lower]
        safe_gap = np.where(np.abs(gap) > 1.0e-12, gap, np.inf)
        for label, dV in (("q", dV_q), ("R", dV_R)):
            matrix = np.sum(
                np.conj(states[lower])*dV*states[upper], axis=0
            )*model.dx
            result[f"nac_{lower}{upper}_{label}"] = np.abs(matrix/safe_gap)
    return result


def _masked(values, density, floor):
    cutoff = floor*max(float(np.max(density)), 1.0e-300)
    return np.where(density >= cutoff, values, np.nan)


def plot_nac_maps(data, nac, frame, support_floor, outdir, dpi):
    """Plot two adjacent BO gaps and q/R derivative couplings."""
    q, R = data["q"], data["R"]
    extent = [R[0], R[-1], q[0], q[-1]]
    joint = (
        np.abs(data["lambda_wavefunction"][frame])**2
        *np.abs(data["chi"][frame])[None, :]**2
    )
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13.0), constrained_layout=True)
    rows = (
        (nac["energies"][1]-nac["energies"][0], nac["energies"][2]-nac["energies"][1], "gap"),
        (nac["nac_01_q"], nac["nac_12_q"], r"$|d^q|$"),
        (nac["nac_01_R"], nac["nac_12_R"], r"$|d^R|$"),
    )
    for row, (left, right, label) in enumerate(rows):
        for col, values in enumerate((left, right)):
            shown = _masked(values, joint, support_floor)
            artist = axes[row, col].imshow(
                shown, origin="lower", aspect="auto", extent=extent,
                cmap="viridis",
            )
            axes[row, col].contour(
                R, q, joint, levels=[support_floor*float(np.max(joint))],
                colors="white", linewidths=1.0,
            )
            pair = "01" if col == 0 else "12"
            axes[row, col].set_title(f"{label}, pair {pair}")
            axes[row, col].set_xlabel("heavy R")
            axes[row, col].set_ylabel("proton q")
            fig.colorbar(artist, ax=axes[row, col], pad=0.012)
    fig.suptitle(
        f"BO gaps and derivative couplings | t={data['times_fs'][frame]:.4f} fs\n"
        f"shown where nuclear density >= {support_floor:g} of peak",
        fontsize=13,
    )
    path = outdir/"bo_gap_and_nac_maps.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"BO gap/NAC map 저장: {path}")


def plot_exact_diagnostics(data, diagnostics, frame, support_floor, outdir, dpi):
    """Plot support-masked currents/forces and their time-dependent RMS."""
    q, R, times = data["q"], data["R"], data["times_fs"]
    joint = diagnostics["joint_density"][frame]
    heavy = diagnostics["heavy_density"][frame]
    extent = [R[0], R[-1], q[0], q[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), constrained_layout=True)

    for ax, key, title in (
        (axes[0, 0], "proton_current", r"Gauge-invariant proton current $j_q$"),
        (axes[0, 1], "force_q", r"Exact force $-\partial_q\epsilon_1+\partial_t a$"),
    ):
        shown = _masked(diagnostics[key][frame], joint, support_floor)
        bound = max(float(np.nanmax(np.abs(shown))), 1.0e-14)
        artist = ax.imshow(
            shown, origin="lower", aspect="auto", extent=extent,
            cmap="coolwarm", vmin=-bound, vmax=bound,
        )
        ax.set_title(title)
        ax.set_xlabel("heavy R")
        ax.set_ylabel("proton q")
        fig.colorbar(artist, ax=ax, pad=0.012)

    heavy_current = _masked(diagnostics["heavy_current"][frame], heavy, support_floor)
    force_R = _masked(diagnostics["force_R"][frame], heavy, support_floor)
    axes[1, 0].plot(R, heavy_current, label=r"$j_R$")
    axes[1, 0].plot(R, force_R, label=r"$-\partial_R\epsilon_2+\partial_t\alpha$")
    axes[1, 0].set_title("Outer heavy-coordinate current and force")
    axes[1, 0].set_xlabel("heavy R")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for key, label in (
        ("support_rms_a", "RMS a"), ("support_rms_b", "RMS b"),
        ("support_rms_alpha", "RMS alpha"),
        ("support_rms_force_q", "RMS Fq"),
        ("support_rms_force_R", "RMS FR"),
    ):
        axes[1, 1].plot(times, diagnostics[key], label=label)
    axes[1, 1].set_yscale("symlog", linthresh=1.0e-4)
    axes[1, 1].set_title("Occupied-support field magnitudes")
    axes[1, 1].set_xlabel("time (fs)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    path = outdir/"gauge_invariant_potential_diagnostics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Gauge-invariant potential 진단 저장: {path}")


def run(args, data=None):
    data = data if data is not None else load_archive(
        args.archive, materialize=not getattr(args, "low_memory", False)
    )
    if not 0.0 < args.support_floor < 1.0:
        raise ValueError("--support-floor는 0과 1 사이여야 합니다.")
    frame = args.frame if args.frame >= 0 else len(data["times_fs"])+args.frame
    if not 0 <= frame < len(data["times_fs"]):
        raise IndexError("분석 frame이 저장 시간 범위를 벗어납니다.")
    requested = Path(args.outdir) if args.outdir else Path(args.archive).parent/"potential_analysis"
    outdir = dated_results_dir(requested)
    outdir.mkdir(parents=True, exist_ok=True)
    diagnostics = gauge_invariant_diagnostics(data)
    nac = nonadiabatic_couplings(data, args.nac_states)
    plot_nac_maps(data, nac, frame, args.support_floor, outdir, args.dpi)
    plot_exact_diagnostics(data, diagnostics, frame, args.support_floor, outdir, args.dpi)
    np.savez_compressed(
        outdir/"potential_diagnostics.npz",
        times_fs=data["times_fs"], q=data["q"], R=data["R"],
        **diagnostics, **nac,
    )
    print(f"Potential/NAC 수치자료 저장: {outdir/'potential_diagnostics.npz'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--support-floor", type=float, default=1.0e-3)
    parser.add_argument("--nac-states", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--low-memory", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
