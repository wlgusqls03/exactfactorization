#!/usr/bin/env python3
"""Native geometry and consistency report for discrete-MCEF archives."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from multi_component_exact_factorization.visualize import (
    NUMBER_FORMATTER,
    selected_frames,
)


COLORS = ("#2878B5", "#E07A2D", "#3A9654", "#B05279", "#7B61A8")


def _load(path):
    wanted = {
        "times_fs", "q", "R", "lambda_wavefunction", "chi", "norm",
        "pnc_error", "pnc_projection_correction", "bo_populations",
        "epsilon_1", "epsilon_2", "a", "b", "alpha",
        "sphi_q1_magnitude", "sphi_R1_magnitude", "sgamma_R1_magnitude",
        "relative_unexplained_residual", "recombination_residual_l2",
        "predicted_mask_residual_l2", "unexplained_residual_l2",
        "rk_product_local_defect_l2", "rk_product_local_defect_relative",
        "pnc_product_change_l2", "rk_product_increment_l2",
        "suppressed_probability_phi", "suppressed_probability_lam",
        "mask_transition_fraction_phi", "mask_transition_fraction_lam",
        "max_raw_horizontal_phi", "max_raw_horizontal_lam",
        "max_abs_regularized_F_ratio", "max_abs_regularized_chi_ratio",
        "weighted_link_defect_phi_q", "weighted_link_defect_phi_R",
        "weighted_link_defect_gamma_R", "full_norm_rate",
        "epsilon_1_imaginary_defect", "epsilon_2_imaginary_defect",
    }
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key in wanted}


def _diagnostic(data, name):
    count = len(data["times_fs"])
    value = np.asarray(data.get(name, np.zeros(count)), float)
    return value if value.shape == (count,) else np.zeros(count)


def _positive(values, floor=1.0e-18):
    return np.maximum(np.abs(np.asarray(values, float)), floor)


def _joint(data):
    q, R = np.asarray(data["q"]), np.asarray(data["R"])
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    values = (
        np.abs(np.asarray(data["lambda_wavefunction"]))**2
        *np.abs(np.asarray(data["chi"]))[:, None, :]**2
    )
    norm = np.sum(values, axis=(1, 2))*dq*dR
    return values/np.maximum(norm[:, None, None], 1.0e-300)


def _support(values, density, floor=1.0e-3, shift_peak=False):
    values = np.asarray(values, float).copy()
    if shift_peak:
        values -= values[np.unravel_index(int(np.argmax(density)), density.shape)]
    return np.where(density >= floor*np.max(density), values, np.nan)


def _unwrap_connection(values, spacing, axis):
    """Unwrap a saved principal link phase along its bond coordinate.

    Propagation and archives retain the full complex link and its principal
    ``arg(S)/h`` diagnostic.  Unwrapping is deliberately a plotting-only
    operation, so it cannot alter the discrete dynamics.
    """
    phase = np.asarray(values, float)*float(spacing)
    return np.unwrap(phase, axis=axis)/float(spacing)


def plot_consistency(data, outdir, dpi=180):
    """Plot spatial algebra, temporal integration and constraint diagnostics."""
    t = np.asarray(data["times_fs"], float)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5), constrained_layout=True)
    axes[0, 0].plot(t, np.asarray(data["norm"])-1.0, color="black")
    axes[0, 0].axhline(0.0, color="0.6", lw=0.8)
    axes[0, 0].set_title("Full-product norm", loc="left", fontweight="semibold")
    axes[0, 0].set_ylabel(r"$\|Y\|^2-1$")

    axes[0, 1].semilogy(t, _positive(_diagnostic(data, "pnc_error")), label="saved PNC")
    axes[0, 1].semilogy(
        t, _positive(_diagnostic(data, "pnc_projection_correction")),
        label="PNC retraction load",
    )
    axes[0, 1].semilogy(
        t, _positive(_diagnostic(data, "pnc_product_change_l2")),
        label=r"$\|\Delta Y_{\rm PNC}\|_2$",
    )
    axes[0, 1].set_title("PNC without changing the product", loc="left", fontweight="semibold")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[0, 2].semilogy(
        t, _positive(_diagnostic(data, "relative_unexplained_residual")),
        label="unexplained / $H_hY$",
    )
    axes[0, 2].semilogy(
        t, _positive(_diagnostic(data, "recombination_residual_l2")),
        label="total recombination residual",
    )
    axes[0, 2].semilogy(
        t, _positive(_diagnostic(data, "predicted_mask_residual_l2")),
        ls="--", label="predicted mask residual",
    )
    axes[0, 2].set_title(r"Spatial identity: $i\dot Y-H_hY$", loc="left", fontweight="semibold")
    axes[0, 2].legend(frameon=False, fontsize=8)

    axes[1, 0].semilogy(
        t, _positive(_diagnostic(data, "rk_product_local_defect_relative")),
        label="relative RK product defect",
    )
    axes[1, 0].semilogy(
        t, _positive(_diagnostic(data, "rk_product_local_defect_l2")),
        label="absolute RK product defect",
    )
    axes[1, 0].set_title("Time-continuous ODE / RK4 consistency", loc="left", fontweight="semibold")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].semilogy(
        t, _positive(_diagnostic(data, "suppressed_probability_phi")),
        label=r"$B_F$",
    )
    axes[1, 1].semilogy(
        t, _positive(_diagnostic(data, "suppressed_probability_lam")),
        label=r"$B_\chi$",
    )
    axes[1, 1].semilogy(
        t, _positive(_diagnostic(data, "mask_transition_fraction_phi")),
        ls="--", label="F transition fraction",
    )
    axes[1, 1].set_title("Flat-top generalized inverse", loc="left", fontweight="semibold")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for name, label, color in (
        ("weighted_link_defect_phi_q", r"$\langle1-|S_q^\Phi|\rangle$", COLORS[0]),
        ("weighted_link_defect_phi_R", r"$\langle1-|S_R^\Phi|\rangle$", COLORS[1]),
        ("weighted_link_defect_gamma_R", r"$\langle1-|S_R^\Gamma|\rangle$", COLORS[2]),
    ):
        axes[1, 2].semilogy(t, _positive(_diagnostic(data, name)), label=label, color=color)
    axes[1, 2].set_title("Native discrete geometry", loc="left", fontweight="semibold")
    axes[1, 2].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("time (fs)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "6 | Discrete MCEF structure-preservation diagnostics",
        fontsize=14, fontweight="bold",
    )
    path = Path(outdir)/"06_discrete_mcef_consistency.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Discrete MCEF consistency 저장: {path}")


def _native_frame(data, joint, frame, floor=1.0e-3):
    density = joint[frame]
    heavy = np.sum(density, axis=0)*float(data["q"][1]-data["q"][0])
    dq = float(data["q"][1]-data["q"][0])
    dR = float(data["R"][1]-data["R"][0])
    return {
        "density": density,
        "heavy": heavy,
        "epsilon_1": _support(data["epsilon_1"][frame], density, floor, True),
        "a": _support(
            _unwrap_connection(data["a"][frame], dq, axis=0),
            density, floor,
        ),
        "b": _support(
            _unwrap_connection(data["b"][frame], dR, axis=1),
            density, floor,
        ),
        "link_q": _support(np.maximum(
            0.0, 1.0-data["sphi_q1_magnitude"][frame]
        ), density, floor),
        "link_R": _support(np.maximum(
            0.0, 1.0-data["sphi_R1_magnitude"][frame]
        ), density, floor),
        "epsilon_2": _support(data["epsilon_2"][frame], heavy, floor, True),
        "alpha": _support(
            _unwrap_connection(data["alpha"][frame], dR, axis=0),
            heavy, floor,
        ),
        "gamma_link": _support(np.maximum(
            0.0, 1.0-data["sgamma_R1_magnitude"][frame]
        ), heavy, floor),
    }


def plot_native_geometry(data, outdir, dpi=180, frame=-1):
    q, R = np.asarray(data["q"]), np.asarray(data["R"])
    joint = _joint(data)
    item = _native_frame(data, joint, frame)
    extent = [q[0], q[-1], R[0], R[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.5), constrained_layout=True)
    maps = (
        (item["density"], "Nuclear joint density", "magma", False),
        (item["epsilon_1"], r"shifted $\mathcal{E}^{(1)}$", "coolwarm", True),
        (item["a"], r"unwrapped $\arg S_q^\Phi/\Delta q$", "coolwarm", True),
        (item["b"], r"unwrapped $\arg S_R^\Phi/\Delta R$", "coolwarm", True),
        (item["link_q"], r"$1-|S_q^\Phi|$", "cividis", False),
        (item["link_R"], r"$1-|S_R^\Phi|$", "cividis", False),
    )
    for ax, (values, title, cmap, symmetric) in zip(axes.flat[:6], maps):
        finite = values[np.isfinite(values)]
        kwargs = {}
        if symmetric:
            limit = max(float(np.percentile(np.abs(finite), 99.0)), 1.0e-14)
            kwargs.update(vmin=-limit, vmax=limit)
        image = ax.imshow(values.T, origin="lower", aspect="auto", extent=extent,
                          cmap=cmap, **kwargs)
        ax.set(xlabel="proton q", ylabel="heavy R")
        ax.set_title(title, loc="left", fontweight="semibold")
        fig.colorbar(image, ax=ax, pad=0.01, format=NUMBER_FORMATTER)
    axes[1, 2].plot(R, item["epsilon_2"], color=COLORS[0], lw=2, label=r"$\mathcal{E}^{(2)}$")
    axes[1, 2].plot(R, item["heavy"]/max(np.max(item["heavy"]), 1e-300), color="0.5", label="heavy density (scaled)")
    axes[1, 2].set_title("Outer discrete scalar and support", loc="left", fontweight="semibold")
    axes[1, 2].legend(frameon=False, fontsize=8)
    axes[1, 3].plot(
        R, item["alpha"], label=r"unwrapped $\arg S_R^\Gamma/\Delta R$"
    )
    axes[1, 3].plot(R, item["gamma_link"], label=r"$1-|S_R^\Gamma|$")
    axes[1, 3].set_title("Second-level link geometry", loc="left", fontweight="semibold")
    axes[1, 3].legend(frameon=False, fontsize=8)
    for ax in axes[1, 2:]:
        ax.set_xlabel("heavy R")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"7 | Native discrete fields | t={float(data['times_fs'][frame]):.4f} fs; gray = empty support",
        fontsize=14, fontweight="bold",
    )
    path = Path(outdir)/"07_discrete_link_geometry.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Discrete MCEF link geometry 저장: {path}")


def _save_animation(animation, fig, outdir, fps, dpi, fmt):
    outdir = Path(outdir)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"discrete_mcef_native_geometry.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3200), dpi=dpi)
    else:
        path = outdir/"discrete_mcef_native_geometry.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"Discrete MCEF native dynamics 저장: {path}")


def make_native_animation(data, outdir, fps=12, max_frames=180, dpi=110, fmt="mp4"):
    times, q, R = np.asarray(data["times_fs"]), np.asarray(data["q"]), np.asarray(data["R"])
    joint = _joint(data)
    frames = selected_frames(len(times), min(max_frames, len(times)))
    items = [_native_frame(data, joint, int(frame)) for frame in frames]
    extent = [q[0], q[-1], R[0], R[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.8), constrained_layout=True)
    specifications = (
        ("density", "Nuclear joint density", "magma", False),
        ("epsilon_1", r"shifted $\mathcal{E}^{(1)}$", "coolwarm", True),
        ("link_q", r"$1-|S_q^\Phi|$", "cividis", False),
        ("link_R", r"$1-|S_R^\Phi|$", "cividis", False),
    )
    images = []
    for ax, (key, title, cmap, symmetric) in zip(axes.flat[:4], specifications):
        finite = np.concatenate([np.abs(item[key][np.isfinite(item[key])]) for item in items])
        high = max(float(np.percentile(finite, 99.5)), 1.0e-14)
        low = -high if symmetric else 0.0
        image = ax.imshow(items[0][key].T, origin="lower", aspect="auto", extent=extent,
                          cmap=cmap, vmin=low, vmax=high)
        ax.set(xlabel="proton q", ylabel="heavy R")
        ax.set_title(title, loc="left", fontweight="semibold")
        fig.colorbar(image, ax=ax, pad=0.01, format=NUMBER_FORMATTER)
        images.append((image, key))
    eps_line, = axes[1, 1].plot(R, items[0]["epsilon_2"], color=COLORS[0], lw=2)
    density_line, = axes[1, 1].plot(R, items[0]["heavy"], color="0.5", alpha=0.7)
    axes[1, 1].set_title(r"$\mathcal{E}^{(2)}$ and heavy density", loc="left", fontweight="semibold")
    populations = np.asarray(data["bo_populations"], float)
    populations /= np.maximum(np.sum(populations, axis=1)[:, None], 1.0e-300)
    for state in range(populations.shape[1]):
        axes[1, 2].semilogy(times, np.maximum(populations[:, state], 1e-14),
                            color=COLORS[state % len(COLORS)], lw=1.5, label=rf"$P_{state}$")
    marker = axes[1, 2].axvline(times[frames[0]], color="black", lw=1.1)
    axes[1, 2].set(xlabel="time (fs)", ylabel="BO population", ylim=(1e-12, 1.5))
    axes[1, 2].set_title("Electronic-state transfer", loc="left", fontweight="semibold")
    axes[1, 2].legend(frameon=False, fontsize=7, ncol=2)
    title = fig.suptitle("Native discrete MCEF geometry")

    def update(number):
        frame = int(frames[number])
        item = items[number]
        for image, key in images:
            image.set_data(item[key].T)
        eps_line.set_ydata(item["epsilon_2"])
        density_line.set_ydata(item["heavy"])
        marker.set_xdata([times[frame], times[frame]])
        residual = _diagnostic(data, "relative_unexplained_residual")[frame]
        temporal = _diagnostic(data, "rk_product_local_defect_relative")[frame]
        title.set_text(
            f"Native discrete MCEF | t={times[frame]:.4f} fs | "
            f"spatial={residual:.1e}, temporal={temporal:.1e}"
        )
        return *(item[0] for item in images), eps_line, density_line, marker, title

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    _save_animation(animation, fig, outdir, fps, dpi, fmt)


def run(archive, outdir, *, dpi=180, no_animation=False, fps=12,
        max_frames=180, animation_dpi=110, fmt="mp4"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = _load(archive)
    plot_consistency(data, outdir, dpi)
    plot_native_geometry(data, outdir, dpi)
    if not no_animation:
        make_native_animation(data, outdir, fps, max_frames, animation_dpi, fmt)
    return data
