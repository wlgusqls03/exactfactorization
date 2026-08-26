#!/usr/bin/env python3
"""Native geometry and consistency report for discrete-MCEF archives."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from multi_component_exact_factorization.report_plot_style import (
    COLORS,
    HEAVY_DENSITY_COLOR,
    JOINT_CMAP,
    LINK_CMAP,
    MASK_COLOR,
    SCALAR_CMAP,
    SIGNED_CMAP,
    color_y_axis,
    density_display_alpha,
    density_weighted_shift,
    joint_density_limit,
    masked_cmap,
)
from multi_component_exact_factorization.visualize import (
    NUMBER_FORMATTER,
    selected_frames,
)


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


def _support(values, density, floor=1.0e-4, shift_peak=False):
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


def _scaled_density_line(axis, coordinate, density):
    scaled = np.asarray(density, float)/max(
        float(np.max(density)), 1.0e-300
    )
    line, = axis.plot(
        coordinate, scaled, transform=axis.get_xaxis_transform(),
        color=HEAVY_DENSITY_COLOR, lw=1.1, alpha=0.42,
        label=r"$\rho_R$ (scaled)", zorder=0,
    )
    return line


def _support_tail_lines(
    axis, coordinate, occupied, full, support, *, color, label,
    linewidth=1.8,
):
    tail = np.where(
        ~np.asarray(support, bool) & np.isfinite(full), full, np.nan
    )
    tail_line, = axis.plot(
        coordinate, tail, color=color, lw=0.8, ls=":", alpha=0.55,
    )
    support_line, = axis.plot(
        coordinate, occupied, color=color, lw=linewidth, label=label,
    )
    return support_line, tail_line


def _native_limits(items, joint):
    """Stream trajectory-wide plotting scales without retaining field copies."""
    symmetric = {"connection", "alpha"}
    sources = {
        "epsilon_1": ("epsilon_1",),
        "connection": ("a", "b"),
        "link": ("link_q", "link_R"),
        "epsilon_2": ("epsilon_2",),
        "alpha": ("alpha",),
        "gamma_link": ("gamma_link",),
    }
    bounds = {key: [] for key in sources}
    for item in items:
        for target, keys in sources.items():
            for key in keys:
                finite = np.asarray(item[key])[np.isfinite(item[key])]
                if not finite.size:
                    continue
                if target in symmetric:
                    bounds[target].append(
                        float(np.max(np.abs(finite)))
                    )
                else:
                    bounds[target].append(
                        (float(np.min(finite)), float(np.max(finite)))
                    )

    limits = {"density": (0.0, joint_density_limit(joint))}
    for key, values in bounds.items():
        if not values:
            limits[key] = (-1.0, 1.0)
        elif key in symmetric:
            bound = max(max(values), 1.0e-12)
            limits[key] = (-bound, bound)
        else:
            low = min(value[0] for value in values)
            high = max(value[1] for value in values)
            if high <= low:
                padding = max(abs(low)*1.0e-6, 1.0e-12)
                low, high = low-padding, high+padding
            limits[key] = (low, high)
    limits["a"] = limits["connection"]
    limits["b"] = limits["connection"]
    limits["link_q"] = limits["link"]
    limits["link_R"] = limits["link"]
    del limits["connection"], limits["link"]
    return limits


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


def _native_frame(data, joint, frame, floor=1.0e-4):
    density = joint[frame]
    heavy = np.sum(density, axis=0)*float(data["q"][1]-data["q"][0])
    heavy_support = heavy >= floor*max(float(np.max(heavy)), 1.0e-300)
    dq = float(data["q"][1]-data["q"][0])
    dR = float(data["R"][1]-data["R"][0])
    epsilon_2_full = density_weighted_shift(
        data["epsilon_2"][frame], heavy, floor
    )
    alpha_full = _unwrap_connection(data["alpha"][frame], dR, axis=0)
    gamma_link_full = np.maximum(
        0.0, 1.0-data["sgamma_R1_magnitude"][frame]
    )
    epsilon_1_full = density_weighted_shift(
        data["epsilon_1"][frame], density, floor
    )
    a_full = _unwrap_connection(data["a"][frame], dq, axis=0)
    b_full = _unwrap_connection(data["b"][frame], dR, axis=1)
    link_q_full = np.maximum(
        0.0, 1.0-data["sphi_q1_magnitude"][frame]
    )
    link_R_full = np.maximum(
        0.0, 1.0-data["sphi_R1_magnitude"][frame]
    )
    return {
        "density": density,
        "density_alpha": density_display_alpha(density, floor),
        "heavy": heavy,
        "heavy_support": heavy_support,
        "epsilon_1": _support(epsilon_1_full, density, floor),
        "epsilon_1_full": epsilon_1_full,
        "a": _support(a_full, density, floor),
        "a_full": a_full,
        "b": _support(b_full, density, floor),
        "b_full": b_full,
        "link_q": _support(link_q_full, density, floor),
        "link_q_full": link_q_full,
        "link_R": _support(link_R_full, density, floor),
        "link_R_full": link_R_full,
        "epsilon_2": np.where(heavy_support, epsilon_2_full, np.nan),
        "alpha": np.where(heavy_support, alpha_full, np.nan),
        "gamma_link": np.where(heavy_support, gamma_link_full, np.nan),
        "epsilon_2_full": epsilon_2_full,
        "alpha_full": alpha_full,
        "gamma_link_full": gamma_link_full,
    }


def plot_native_geometry(data, outdir, dpi=180, frame=-1):
    q, R = np.asarray(data["q"]), np.asarray(data["R"])
    joint = _joint(data)
    item = _native_frame(data, joint, frame)
    scale_frames = selected_frames(len(joint), min(180, len(joint)))
    limits = _native_limits(
        (_native_frame(data, joint, int(index)) for index in scale_frames),
        joint,
    )
    extent = [q[0], q[-1], R[0], R[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.5), constrained_layout=True)
    maps = (
        ("density", "Nuclear joint density", JOINT_CMAP),
        ("epsilon_1", r"shifted $\mathcal{E}^{(1)}$", SCALAR_CMAP),
        ("a", r"unwrapped $\arg S_q^\Phi/\Delta q$", SIGNED_CMAP),
        ("b", r"unwrapped $\arg S_R^\Phi/\Delta R$", SIGNED_CMAP),
        ("link_q", r"$1-|S_q^\Phi|$", LINK_CMAP),
        ("link_R", r"$1-|S_R^\Phi|$", LINK_CMAP),
    )
    for ax, (key, title, cmap) in zip(axes.flat[:6], maps):
        values = item[key] if key == "density" else item[f"{key}_full"]
        alpha = None if key == "density" else item["density_alpha"].T
        ax.set_facecolor(MASK_COLOR)
        image = ax.imshow(
            values.T, origin="lower", aspect="auto", extent=extent,
            cmap=masked_cmap(cmap),
            vmin=limits[key][0], vmax=limits[key][1], alpha=alpha,
        )
        ax.set(xlabel="proton q", ylabel="heavy R")
        ax.set_title(title, loc="left", fontweight="semibold")
        fig.colorbar(image, ax=ax, pad=0.01, format=NUMBER_FORMATTER)
    epsilon_line, _epsilon_tail = _support_tail_lines(
        axes[1, 2], R, item["epsilon_2"], item["epsilon_2_full"],
        item["heavy_support"], color=COLORS[0],
        label=r"$\mathcal{E}^{(2)}$", linewidth=2.0,
    )
    _scaled_density_line(axes[1, 2], R, item["heavy"])
    axes[1, 2].set_ylim(limits["epsilon_2"])
    color_y_axis(axes[1, 2], COLORS[0], "shifted energy (Hartree)")
    axes[1, 2].set_title("Outer discrete scalar and support", loc="left", fontweight="semibold")
    axes[1, 2].legend(frameon=False, fontsize=8)
    alpha_line, _alpha_tail = _support_tail_lines(
        axes[1, 3], R, item["alpha"], item["alpha_full"],
        item["heavy_support"], color=COLORS[3],
        label=r"unwrapped $\arg S_R^\Gamma/\Delta R$",
    )
    axes[1, 3].set_ylim(limits["alpha"])
    color_y_axis(axes[1, 3], COLORS[3], r"connection ($a_0^{-1}$)")
    gamma_axis = axes[1, 3].twinx()
    gamma_line, _gamma_tail = _support_tail_lines(
        gamma_axis, R, item["gamma_link"], item["gamma_link_full"],
        item["heavy_support"], color=COLORS[2],
        label=r"$1-|S_R^\Gamma|$",
    )
    gamma_axis.set_ylim(limits["gamma_link"])
    color_y_axis(gamma_axis, COLORS[2], "link magnitude defect")
    axes[1, 3].set_title("Second-level link geometry", loc="left", fontweight="semibold")
    axes[1, 3].legend(
        handles=[alpha_line, gamma_line], frameon=False, fontsize=8,
    )
    for ax in axes[1, 2:]:
        ax.set_xlabel("heavy R")
        ax.set_xlim(float(R[0]), float(R[-1]))
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"7 | Native discrete fields | t={float(data['times_fs'][frame]):.4f} fs\n"
        r"2D gray below $10^{-6}\rho_{\max}$, fully colored above $10^{-4}\rho_{\max}$; "
        "1D solid = occupied, thin dotted = low-density continuation",
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
    limits = _native_limits(
        (_native_frame(data, joint, int(frame)) for frame in frames), joint
    )
    first_frame = int(frames[0])
    first_item = _native_frame(data, joint, first_frame)
    extent = [q[0], q[-1], R[0], R[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.8), constrained_layout=True)
    specifications = (
        ("density", "Nuclear joint density", JOINT_CMAP),
        ("epsilon_1", r"shifted $\mathcal{E}^{(1)}$", SCALAR_CMAP),
        ("link_q", r"$1-|S_q^\Phi|$", LINK_CMAP),
        ("link_R", r"$1-|S_R^\Phi|$", LINK_CMAP),
    )
    images = []
    for ax, (key, title, cmap) in zip(axes.flat[:4], specifications):
        values = (
            first_item[key] if key == "density"
            else first_item[f"{key}_full"]
        )
        alpha = None if key == "density" else first_item["density_alpha"].T
        ax.set_facecolor(MASK_COLOR)
        image = ax.imshow(
            values.T, origin="lower", aspect="auto", extent=extent,
            cmap=masked_cmap(cmap), vmin=limits[key][0], vmax=limits[key][1],
            alpha=alpha,
        )
        ax.set(xlabel="proton q", ylabel="heavy R")
        ax.set_title(title, loc="left", fontweight="semibold")
        fig.colorbar(image, ax=ax, pad=0.01, format=NUMBER_FORMATTER)
        images.append((image, key))
    eps_line, eps_tail = _support_tail_lines(
        axes[1, 1], R, first_item["epsilon_2"],
        first_item["epsilon_2_full"], first_item["heavy_support"],
        color=COLORS[0], label=r"$\mathcal{E}^{(2)}$", linewidth=2.0,
    )
    density_line = _scaled_density_line(axes[1, 1], R, first_item["heavy"])
    axes[1, 1].set_ylim(limits["epsilon_2"])
    axes[1, 1].set_xlim(float(R[0]), float(R[-1]))
    color_y_axis(axes[1, 1], COLORS[0], "shifted energy (Hartree)")
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
    title = fig.suptitle(
        "Native discrete MCEF geometry\n"
        r"2D gray below $10^{-6}\rho_{\max}$; full color above $10^{-4}\rho_{\max}$"
    )

    def update(number):
        frame = int(frames[number])
        item = _native_frame(data, joint, frame)
        for image, key in images:
            values = item[key] if key == "density" else item[f"{key}_full"]
            image.set_data(values.T)
            if key != "density":
                image.set_alpha(item["density_alpha"].T)
        eps_line.set_ydata(item["epsilon_2"])
        eps_tail.set_ydata(np.where(
            ~item["heavy_support"], item["epsilon_2_full"], np.nan
        ))
        density_line.set_ydata(
            item["heavy"]/max(float(np.max(item["heavy"])), 1.0e-300)
        )
        marker.set_xdata([times[frame], times[frame]])
        residual = _diagnostic(data, "relative_unexplained_residual")[frame]
        temporal = _diagnostic(data, "rk_product_local_defect_relative")[frame]
        title.set_text(
            f"Native discrete MCEF | t={times[frame]:.4f} fs | "
            f"spatial={residual:.1e}, temporal={temporal:.1e}\n"
            r"2D gray below $10^{-6}\rho_{\max}$; full color above $10^{-4}\rho_{\max}$"
        )
        return *(entry[0] for entry in images), eps_line, eps_tail, density_line, marker, title

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
