#!/usr/bin/env python3
"""Multi-component EF 결과의 논문형 snapshot과 두 종류의 6분할 동영상.

3차원 복소함수 Phi(x,q,R)와 Psi(x,q,R)는 종이/화면에 그대로 그릴 수
없다. 정적 그림은 논문의 conditional-density 표현을 따른다. 첫 영상은
Phi/Lambda/chi의 Re·Im·density를, 둘째 영상은 세 factor density를 논문식
colormap으로 표시한다. 두 영상 모두 full marginal과 두 TDPES를 포함한다.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter
import numpy as np

from result_paths import dated_results_dir


def readable_number(value, _position=None):
    """1e-3 order 이하는 과학 표기, 나머지는 소수점 둘째 자리까지 표시."""
    if not np.isfinite(value):
        return ""
    # 두 자리 고정소수점에서 0.00으로 뭉개지는 10^-3 order부터 e 표기를 쓴다.
    if value != 0.0 and abs(value) < 1.0e-2:
        return f"{value:.2e}"
    return f"{value:.2f}"


NUMBER_FORMATTER = FuncFormatter(readable_number)


class LoadedArchive(dict):
    """Materialized NPZ data with a small cache for reduced animation frames."""

    def __init__(self, values):
        super().__init__(values)
        self.reduced_frames = {}

    @property
    def files(self):
        return tuple(self.keys())


def load_archive(path, materialize=True):
    """Load an archive once, avoiding repeated decompression of large fields.

    ``numpy.lib.npyio.NpzFile`` reads a compressed member again on every
    ``data[key]`` access. Animation and analysis loops access ``phi`` hundreds
    of times, so the default materialized representation decompresses every
    needed member exactly once. The optional full ``psi`` is skipped because
    all visualizations reconstruct its density from the three factors.
    """
    archive = np.load(path, allow_pickle=True)
    required = {
        "x", "q", "R", "times_fs", "phi", "lambda_wavefunction",
        "chi", "epsilon_2",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        archive.close()
        raise KeyError(f"archive에 필요한 key가 없습니다: {missing}")
    if not materialize:
        return archive
    values = {
        key: archive[key]
        for key in archive.files
        if key != "psi"
    }
    archive.close()
    return LoadedArchive(values)


def archive_arguments(data):
    """계산 NPZ에 함께 저장된 command-line option을 평범한 dict로 읽는다."""
    if "args" not in data.files or data["args"].size != 1:
        return {}
    stored = data["args"].reshape(-1)[0]
    return stored if isinstance(stored, dict) else {}


def common_position_limits(data):
    """세 좌표 profile에 공통으로 쓸 물리적 box 범위."""
    options = archive_arguments(data)
    lower = [float(data[key][0]) for key in ("x", "q", "R")]
    upper = [float(data[key][-1]) for key in ("x", "q", "R")]
    # Hard-wall 전자 grid는 경계점 자체를 저장하지 않으므로 metadata의
    # x_min/x_max를 포함해야 왼쪽 고정점과 오른쪽 벽이 정확히 보인다.
    lower.append(float(options.get("x_min", lower[0])))
    upper.append(float(options.get("x_max", upper[0])))
    return min(lower), max(upper)


def initial_marginals(data):
    """t=0 full wavefunction에서 세 입자의 1D marginal density를 계산한다.

    전자 그림은 특정 ``(q,R)``를 고른 conditional slice가 아니라 나머지 두
    좌표를 모두 적분한 진짜 electron marginal이다. 따라서 초기 입자 위치를
    한눈에 비교하는 용도로 적합하다.
    """
    x, q, R = data["x"], data["q"], data["R"]
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    phi2 = np.abs(data["phi"][0])**2                              # (nx,nq,nR)
    lam2 = np.abs(data["lambda_wavefunction"][0])**2              # (nq,nR)
    chi2 = np.abs(data["chi"][0])**2                              # (nR,)
    electron = np.sum(
        phi2*lam2[None, :, :]*chi2[None, None, :], axis=(1, 2)
    )*dq*dR                                                        # (nx,)
    proton = np.sum(lam2*chi2[None, :], axis=1)*dR                # (nq,)
    return electron, proton, chi2


def initial_summary(data):
    """영상 제목에 반복해서 넣을 간결한 초기 중심·질량 설명."""
    options = archive_arguments(data)
    if not options:
        return "initial parameters unavailable in archive"
    excitation = int(options.get("electron_excitation", 0))
    return (
        f"initial: electron=local H_BO state n={excitation}, "
        f"q0={options['q0']:.2f}, "
        f"R0={options['R0']:.2f} a0  |  masses (me): "
        f"mp={options['proton_mass']:.0f}, MH={options['heavy_mass']:.0f}"
    )


def reduced_frame(data, frame):
    """한 frame의 네 factor/joint density를 계산한다."""
    cache = getattr(data, "reduced_frames", None)
    if cache is not None and frame in cache:
        return cache[frame]
    x, q = data["x"], data["q"]
    dq = float(q[1]-q[0])
    phi = data["phi"][frame]                                      # (nx,nq,nR)
    lam = data["lambda_wavefunction"][frame]                      # (nq,nR)
    chi = data["chi"][frame]                                      # (nR,)
    lam2, chi2 = np.abs(lam)**2, np.abs(chi)**2
    electron_given_R = np.sum(
        np.abs(phi)**2*lam2[None, :, :], axis=1
    )*dq                                                            # (nx,nR)
    full_xR = electron_given_R*chi2[None, :]                        # (nx,nR)
    result = dict(
        phi=phi, lam=lam, chi=chi, heavy=chi2,
        proton=lam2, electron=electron_given_R, full=full_xR,
    )
    if cache is not None:
        cache[frame] = result
    return result


def plot_initial_state(data, outdir, dpi=180):
    """t=0 입자 위치, Gaussian 폭·운동량·질량을 한 장에 요약한다."""
    x, q, R = data["x"], data["q"], data["R"]
    dx, dq, dR = float(x[1]-x[0]), float(q[1]-q[0]), float(R[1]-R[0])
    densities = initial_marginals(data)
    options = archive_arguments(data)
    excitation = int(options.get("electron_excitation", 0))

    grids = (x, q, R)
    names = ("Electron marginal", "Proton marginal", "Heavy-nucleus marginal")
    symbols = ("x", "q", "R")
    colors = ("#2878B5", "#D9534F", "#3A923A")
    spacings = (dx, dq, dR)
    configured_centers = (
        np.nan,
        options.get("q0", np.nan),
        options.get("R0", np.nan),
    )
    sigmas = (
        np.nan,
        options.get("proton_sigma", np.nan),
        options.get("heavy_sigma", np.nan),
    )
    momenta = (
        np.nan,
        options.get("proton_momentum", np.nan),
        options.get("heavy_momentum", np.nan),
    )
    details = (
        f"local H_BO eigenstate n={excitation}",
        f"marginal sigma={sigmas[1]:.3f}, p0={momenta[1]:.2f}",
        f"marginal sigma={sigmas[2]:.3f}, p0={momenta[2]:.2f}",
    )

    common_min, common_max = common_position_limits(data)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=True)
    for ax, grid, rho, name, symbol, color, spacing, center, detail in zip(
        axes, grids, densities, names, symbols, colors, spacings,
        configured_centers, details,
    ):
        mean = float(np.sum(grid*rho)*spacing)
        peak = float(grid[int(np.argmax(rho))])
        ax.fill_between(grid, rho, color=color, alpha=0.25)
        ax.plot(grid, rho, color=color, lw=2.0, label="t=0 density")
        if np.isfinite(center):
            ax.axvline(center, color="black", ls="--", lw=1.1, label="input center")
        # 왼쪽 고정 양전하는 동역학 자유도는 아니지만 초기 공간 배치를 보여준다.
        if "left_position" in options:
            ax.axvline(
                options["left_position"], color="0.45", ls=":", lw=1.1,
                label="fixed site",
            )
        ax.set_xlim(common_min, common_max)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(f"1D position {symbol} (a0)")
        ax.set_ylabel("probability density")
        ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
        ax.set_title(
            f"{name}\nmean={mean:.3f}, grid peak={peak:.3f}\n"
            +detail
        )
        ax.legend(frameon=False, fontsize=8)

    if options:
        mass_line = (
            f"m_p={options['proton_mass']:.0f} m_e,  "
            f"M_H={options['heavy_mass']:.0f} m_e;  "
            f"left wall={options['left_position']:.2f} a0, "
            f"Z_L={options.get('left_charge', np.nan):.2f}, "
            f"Z_R={options.get('right_charge', 0.0):.2f}"
        )
        correlation_line = (
            "independent q/R harmonic Gaussians; "
            f"kq={options.get('initial_proton_force_constant', np.nan):.4f}, "
            f"kR={options.get('initial_heavy_force_constant', np.nan):.4f}"
        )
    else:
        mass_line = "mass and input-center metadata unavailable"
        correlation_line = ""
    fig.suptitle(f"Initial state | BO $n={excitation}$", fontsize=14)
    fig.supxlabel(
        mass_line+"  |  "+correlation_line,
        fontsize=8.5, color="0.35",
    )
    path = outdir/"initial_state_summary.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"초기상태 요약 저장: {path}")


def shifted_tdpes(data, frame, density_floor=1.0e-5):
    """Gauge별 상수 offset을 제거하고 low-density tail을 mask한 TDPES."""
    eps = np.asarray(data["epsilon_2"][frame], float).copy()
    rho = np.abs(data["chi"][frame])**2
    peak = int(np.argmax(rho))
    eps -= eps[peak]
    eps[rho < density_floor*max(float(np.max(rho)), 1.0e-300)] = np.nan
    return eps


def shifted_epsilon_1(data, frame, density_floor=1.0e-5, mask_tails=False):
    """Peak configuration을 0으로 맞춘 epsilon_1 전체 격자.

    기본 그림은 regularized numerical field 전체를 표시한다. 점유 support만
    보고 싶을 때만 ``mask_tails=True``를 사용한다.
    """
    eps = np.asarray(data["epsilon_1"][frame], float).copy()        # (nq,nR)
    lam2 = np.abs(data["lambda_wavefunction"][frame])**2
    chi2 = np.abs(data["chi"][frame])**2
    joint = lam2*chi2[None, :]                                      # (nq,nR)
    peak = np.unravel_index(int(np.argmax(joint)), joint.shape)
    eps -= eps[peak]
    if mask_tails:
        eps[joint < density_floor*max(float(np.max(joint)), 1.0e-300)] = np.nan
    return eps


def supported_potential_fields(data, frame, support_floor=1.0e-3, show_tails=False):
    """Raw potential을 유지하되 probability가 없는 tail은 NaN으로 가린다."""
    if not 0.0 < support_floor < 1.0:
        raise ValueError("potential support floor는 0과 1 사이여야 합니다.")
    support = (
        np.abs(data["lambda_wavefunction"][frame])**2
        *np.abs(data["chi"][frame])[None, :]**2
    )
    heavy = np.abs(data["chi"][frame])**2
    fields = {
        key: np.asarray(data[key][frame], float).copy()
        for key in ("epsilon_1", "a", "b", "epsilon_2", "alpha")
    }
    if not show_tails:
        joint_cutoff = support_floor*max(float(np.max(support)), 1.0e-300)
        heavy_cutoff = support_floor*max(float(np.max(heavy)), 1.0e-300)
        joint_mask = support >= joint_cutoff
        heavy_mask = heavy >= heavy_cutoff
        for key in ("epsilon_1", "a", "b"):
            fields[key][~joint_mask] = np.nan
        for key in ("epsilon_2", "alpha"):
            fields[key][~heavy_mask] = np.nan
    fields["support"] = support
    return fields


def selected_frames(nt, count):
    """처음과 끝을 포함해 균일한 snapshot index를 고른다."""
    count = min(max(1, count), nt)
    return np.unique(np.linspace(0, nt-1, count).round().astype(int))


def robust_limits(arrays, symmetric=False):
    values = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays])
    if values.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(values, [2.0, 98.0])
    if symmetric:
        bound = max(abs(float(lo)), abs(float(hi)), 1.0e-12)
        return -bound, bound
    if hi <= lo:
        hi = lo+1.0
    return float(lo), float(hi)


def plot_snapshots(data, outdir, count=4, dpi=180):
    """논문 Fig. 2–4를 multi-component용 4행 snapshot으로 확장한다."""
    x, q, R, times = data["x"], data["q"], data["R"], data["times_fs"]
    frames = selected_frames(len(times), count)
    reduced = [reduced_frame(data, int(i)) for i in frames]
    epsilons = [shifted_tdpes(data, int(i)) for i in frames]
    eps_lo, eps_hi = robust_limits(epsilons)
    vmax_p = max(float(np.max(item["proton"])) for item in reduced)
    vmax_e = max(float(np.max(item["electron"])) for item in reduced)
    vmax_f = max(float(np.max(item["full"])) for item in reduced)

    fig, axes = plt.subplots(
        4, len(frames), figsize=(3.55*len(frames), 10.5),
        constrained_layout=True, squeeze=False,
    )
    fig.suptitle(
        "Multi-component EF snapshots\n"+initial_summary(data), fontsize=12
    )
    for col, (frame, item, eps) in enumerate(zip(frames, reduced, epsilons)):
        ax = axes[0, col]
        ax.plot(R, eps, color="#275DAD", lw=1.8)
        ax.set_ylim(eps_lo, eps_hi)
        ax.axhline(0.0, color="0.75", lw=0.7)
        density_axis = ax.twinx()
        density_axis.fill_between(R, item["heavy"], color="black", alpha=0.25)
        density_axis.plot(R, item["heavy"], color="black", lw=1.2)
        density_axis.set_ylim(0, 1.12*max(np.max(item["heavy"]), 1.0e-12))
        ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
        density_axis.yaxis.set_major_formatter(NUMBER_FORMATTER)
        if col == len(frames)-1:
            density_axis.set_ylabel(r"$|\chi(R,t)|^2$")
        else:
            density_axis.set_yticklabels([])
        ax.set_title(f"t = {times[frame]:.4f} fs")
        if col == 0:
            ax.set_ylabel(r"shifted $\epsilon^{(2)}$")

        image_p = axes[1, col].imshow(
            item["proton"], origin="lower", aspect="auto", cmap="magma",
            extent=[R[0], R[-1], q[0], q[-1]], vmin=0, vmax=vmax_p,
        )
        image_e = axes[2, col].imshow(
            item["electron"], origin="lower", aspect="auto", cmap="inferno",
            extent=[R[0], R[-1], x[0], x[-1]], vmin=0, vmax=vmax_e,
        )
        image_f = axes[3, col].imshow(
            item["full"], origin="lower", aspect="auto", cmap="viridis",
            extent=[R[0], R[-1], x[0], x[-1]], vmin=0, vmax=vmax_f,
        )
        if col == 0:
            axes[1, col].set_ylabel(r"proton $q$")
            axes[2, col].set_ylabel(r"electron $x$")
            axes[3, col].set_ylabel(r"electron $x$")
        for row in (1, 2, 3):
            axes[row, col].set_xlabel(r"heavy coordinate $R$")

    axes[1, 0].text(
        0.02, 0.96, r"$|\Lambda_R(q,t)|^2$", transform=axes[1, 0].transAxes,
        va="top", color="white", fontsize=10,
    )
    axes[2, 0].text(
        0.02, 0.96, "conditional electron density", transform=axes[2, 0].transAxes,
        va="top", color="white", fontsize=9,
    )
    axes[3, 0].text(
        0.02, 0.96, r"$\int dq\,|\Phi\Lambda\chi|^2$", transform=axes[3, 0].transAxes,
        va="top", color="white", fontsize=10,
    )
    # 같은 행의 모든 snapshot이 동일한 color scale을 쓰므로 행당 하나면 충분하다.
    fig.colorbar(
        image_p, ax=axes[1, :].tolist(), label=r"$|\Lambda_R|^2$",
        pad=0.012, fraction=0.025, format=NUMBER_FORMATTER,
    )
    fig.colorbar(
        image_e, ax=axes[2, :].tolist(), label="conditional electron density",
        pad=0.012, fraction=0.025, format=NUMBER_FORMATTER,
    )
    fig.colorbar(
        image_f, ax=axes[3, :].tolist(), label=r"$\rho_{xR}$",
        pad=0.012, fraction=0.025, format=NUMBER_FORMATTER,
    )
    path = outdir/"multi_component_snapshots.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"정적 snapshot 저장: {path}")


def plot_factor_profiles(data, outdir, frame=-1, dpi=180):
    """peak configuration에서 Phi, Lambda, chi의 복소 profile을 그린다."""
    if frame < 0:
        frame += len(data["times_fs"])
    item = reduced_frame(data, frame)
    x, q, R = data["x"], data["q"], data["R"]
    iR = int(np.argmax(item["heavy"]))
    iq = int(np.argmax(item["proton"][:, iR]))
    phi_line = item["phi"][:, iq, iR]
    lam_line = item["lam"][:, iR]
    chi = item["chi"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
    profiles = [
        (x, phi_line, r"$\Phi_{R,q}(x,t)$", "electron x"),
        (q, lam_line, r"$\Lambda_R(q,t)$", "proton q"),
        (R, chi, r"$\chi(R,t)$", "heavy R"),
    ]
    # 세 입자는 같은 1D 물리 공간에 있으므로 그림의 위치축 범위를 통일한다.
    common_min, common_max = common_position_limits(data)
    for ax, (grid, wave, title, xlabel) in zip(axes, profiles):
        ax.plot(grid, wave.real, label="Real", color="#2369BD")
        ax.plot(grid, wave.imag, label="Imag", color="#D1495B")
        ax.plot(grid, np.abs(wave)**2, label="Density", color="black", lw=1.5)
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xlim(common_min, common_max)
        ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"t={data['times_fs'][frame]:.4f} fs, "
        f"peak configuration q={q[iq]:.3f}, R={R[iR]:.3f}\n"
        +initial_summary(data)
    )
    path = outdir/"factor_wavefunction_profiles.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"factor profile 저장: {path}")


def make_wavefunction_animation(
    data, outdir, fps=12, max_frames=180, dpi=130, fmt="mp4"
):
    """Phi/Lambda/chi/full Psi/epsilon_1/epsilon_2의 6-panel 영상."""
    x, q, R, times = data["x"], data["q"], data["R"], data["times_fs"]
    # 전자, 양성자, heavy 핵을 동일한 물리적 위치 눈금에서 비교한다.
    common_min, common_max = common_position_limits(data)
    frames = selected_frames(len(times), min(max_frames, len(times)))
    # frame마다 y축/색 범위가 흔들리지 않도록 전체 영상에서 limit을 정한다.
    phi_amp = lam_amp = chi_amp = 1.0e-12
    phi_density = lam_density = chi_density = full_max = 1.0e-12
    eps1_arrays, eps2_arrays = [], []
    for frame in frames:
        item = reduced_frame(data, int(frame))
        iR = int(np.argmax(item["heavy"]))
        iq = int(np.argmax(item["proton"][:, iR]))
        phi_line = item["phi"][:, iq, iR]
        lam_line = item["lam"][:, iR]
        phi_amp = max(
            phi_amp, float(np.max(np.abs(phi_line.real))),
            float(np.max(np.abs(phi_line.imag))),
        )
        lam_amp = max(
            lam_amp, float(np.max(np.abs(lam_line.real))),
            float(np.max(np.abs(lam_line.imag))),
        )
        chi_amp = max(
            chi_amp, float(np.max(np.abs(item["chi"].real))),
            float(np.max(np.abs(item["chi"].imag))),
        )
        phi_density = max(phi_density, float(np.max(np.abs(phi_line)**2)))
        lam_density = max(lam_density, float(np.max(np.abs(lam_line)**2)))
        chi_density = max(chi_density, float(np.max(item["heavy"])))
        full_max = max(full_max, float(np.max(item["full"])))
        eps1_arrays.append(shifted_epsilon_1(data, int(frame)))
        eps2_arrays.append(shifted_tdpes(data, int(frame)))
    eps1_lo, eps1_hi = robust_limits(eps1_arrays, symmetric=True)
    eps2_lo, eps2_hi = robust_limits(eps2_arrays)

    first = reduced_frame(data, int(frames[0]))
    iR0 = int(np.argmax(first["heavy"]))
    iq0 = int(np.argmax(first["proton"][:, iR0]))
    first_phi = first["phi"][:, iq0, iR0]
    first_lam = first["lam"][:, iR0]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), constrained_layout=True)

    # 세 factor는 복소함수이므로 Re/Im과 density를 동시에 표시한다.
    def wave_panel(ax, grid, wave, amp_limit, density_limit, title, xlabel):
        real_line, = ax.plot(grid, wave.real, color="#2369BD", label="Real")
        imag_line, = ax.plot(grid, wave.imag, color="#D1495B", label="Imag")
        ax.set_ylim(-1.08*amp_limit, 1.08*amp_limit)
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set_xlabel(xlabel)
        ax.set_xlim(common_min, common_max)
        ax.set_ylabel("amplitude")
        ax.set_title(title)
        density_axis = ax.twinx()
        density_line, = density_axis.plot(
            grid, np.abs(wave)**2, color="black", lw=1.5, label="Density"
        )
        density_axis.set_ylim(0, 1.08*density_limit)
        density_axis.set_ylabel("density")
        ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
        density_axis.yaxis.set_major_formatter(NUMBER_FORMATTER)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        density_axis.legend(loc="upper right", frameon=False, fontsize=8)
        return real_line, imag_line, density_line

    phi_lines = wave_panel(
        axes[0, 0], x, first_phi, phi_amp, phi_density,
        rf"Electron $\Phi$  (q={q[iq0]:.2f}, R={R[iR0]:.2f})", "electron x",
    )
    lam_lines = wave_panel(
        axes[0, 1], q, first_lam, lam_amp, lam_density,
        rf"Proton $\Lambda_R$  (R={R[iR0]:.2f})", "proton q",
    )
    chi_lines = wave_panel(
        axes[0, 2], R, first["chi"], chi_amp, chi_density,
        r"Heavy nucleus $\chi(R,t)$", "heavy R",
    )

    image_f = axes[1, 0].imshow(
        first["full"], origin="lower", aspect="auto", cmap="viridis",
        extent=[R[0], R[-1], x[0], x[-1]], vmin=0, vmax=full_max,
    )
    axes[1, 0].set_title(r"Full $\Psi$: $\int dq|\Phi\Lambda\chi|^2$")
    axes[1, 0].set_xlabel("heavy R")
    axes[1, 0].set_ylabel("electron x")

    eps1_image = axes[1, 1].imshow(
        eps1_arrays[0], origin="lower", aspect="auto", cmap="coolwarm",
        extent=[R[0], R[-1], q[0], q[-1]], vmin=eps1_lo, vmax=eps1_hi,
    )
    axes[1, 1].set_title(
        r"First TDPES $\epsilon^{(1)}(q,R,t)$ (regularized tails)"
    )
    axes[1, 1].set_xlabel("heavy R")
    axes[1, 1].set_ylabel("proton q")
    fig.colorbar(
        image_f, ax=axes[1, 0], label=r"$\rho_{xR}$",
        pad=0.015, fraction=0.048, format=NUMBER_FORMATTER,
    )
    fig.colorbar(
        eps1_image, ax=axes[1, 1], label=r"shifted $\epsilon^{(1)}$",
        pad=0.015, fraction=0.048, format=NUMBER_FORMATTER,
    )

    eps2_line, = axes[1, 2].plot(R, eps2_arrays[0], color="#275DAD", lw=2)
    axes[1, 2].set_ylim(eps2_lo, eps2_hi)
    axes[1, 2].axhline(0, color="0.75", lw=0.7)
    axes[1, 2].set_title(r"Second TDPES $\epsilon^{(2)}(R,t)$")
    axes[1, 2].set_xlabel("heavy R")
    axes[1, 2].set_ylabel("shifted energy")
    axes[1, 2].yaxis.set_major_formatter(NUMBER_FORMATTER)
    title = fig.suptitle(
        f"Multi-component EF   t={times[frames[0]]:.4f} fs\n"
        +initial_summary(data), fontsize=11.5,
    )

    def update(number):
        frame = int(frames[number])
        item = reduced_frame(data, frame)
        iR = int(np.argmax(item["heavy"]))
        iq = int(np.argmax(item["proton"][:, iR]))
        phi_line = item["phi"][:, iq, iR]
        lam_line = item["lam"][:, iR]
        for artists, wave in (
            (phi_lines, phi_line),
            (lam_lines, lam_line),
            (chi_lines, item["chi"]),
        ):
            artists[0].set_ydata(wave.real)
            artists[1].set_ydata(wave.imag)
            artists[2].set_ydata(np.abs(wave)**2)
        axes[0, 0].set_title(
            rf"Electron $\Phi$  (q={q[iq]:.2f}, R={R[iR]:.2f})"
        )
        axes[0, 1].set_title(rf"Proton $\Lambda_R$  (R={R[iR]:.2f})")
        eps1_image.set_data(shifted_epsilon_1(data, frame))
        eps2_line.set_ydata(shifted_tdpes(data, frame))
        image_f.set_data(item["full"])
        title.set_text(
            f"Multi-component EF   t={times[frame]:.4f} fs\n"
            +initial_summary(data)
        )
        return (
            *phi_lines, *lam_lines, *chi_lines,
            image_f, eps1_image, eps2_line, title,
        )

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"multi_component_wavefunction_dynamics.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=2600), dpi=dpi)
    else:
        if fmt == "mp4":
            print("ffmpeg을 찾지 못해 GIF로 대신 저장합니다.")
        path = outdir/"multi_component_wavefunction_dynamics.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"복소 wavefunction 6분할 dynamics 저장: {path}")


def make_density_animation(
    data, outdir, fps=12, max_frames=180, dpi=130, fmt="mp4"
):
    """논문식 colormap으로 세 factor density와 두 TDPES를 보여준다.

    full Psi는 3D configuration-space volume을 직접 투영하지 않고
    ``rho_xR = int dq |Psi(x,q,R)|^2``를 쓴다. 이는 slice가 아니라 q를
    정확히 적분한 marginal이며, 2-component 논문의 total-density 그림을
    multi-component로 확장한 가장 직접적인 표현이다.
    """
    x, q, R, times = data["x"], data["q"], data["R"], data["times_fs"]
    frames = selected_frames(len(times), min(max_frames, len(times)))

    electron_max = proton_max = heavy_max = full_max = 1.0e-12
    eps1_arrays, eps2_arrays = [], []
    for frame in frames:
        item = reduced_frame(data, int(frame))
        electron_max = max(electron_max, float(np.max(item["electron"])))
        proton_max = max(proton_max, float(np.max(item["proton"])))
        heavy_max = max(heavy_max, float(np.max(item["heavy"])))
        full_max = max(full_max, float(np.max(item["full"])))
        eps1_arrays.append(shifted_epsilon_1(data, int(frame)))
        eps2_arrays.append(shifted_tdpes(data, int(frame)))
    eps1_lo, eps1_hi = robust_limits(eps1_arrays, symmetric=True)
    eps2_lo, eps2_hi = robust_limits(eps2_arrays)

    first = reduced_frame(data, int(frames[0]))
    heavy_strip = np.tile(first["heavy"][None, :], (12, 1))
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.0), constrained_layout=True)

    electron_image = axes[0, 0].imshow(
        first["electron"], origin="lower", aspect="auto", cmap="inferno",
        extent=[R[0], R[-1], x[0], x[-1]], vmin=0, vmax=electron_max,
    )
    axes[0, 0].set_title(r"Electron: $\int dq|\Lambda|^2|\Phi|^2$")
    axes[0, 0].set_xlabel("heavy R")
    axes[0, 0].set_ylabel("electron x")

    proton_image = axes[0, 1].imshow(
        first["proton"], origin="lower", aspect="auto", cmap="magma",
        extent=[R[0], R[-1], q[0], q[-1]], vmin=0, vmax=proton_max,
    )
    axes[0, 1].set_title(r"Proton: $|\Lambda_R(q,t)|^2$")
    axes[0, 1].set_xlabel("heavy R")
    axes[0, 1].set_ylabel("proton q")

    heavy_image = axes[0, 2].imshow(
        heavy_strip, origin="lower", aspect="auto", cmap="cividis",
        extent=[R[0], R[-1], -1.0, 1.0], vmin=0, vmax=heavy_max,
    )
    axes[0, 2].set_title(r"Heavy nucleus: $|\chi(R,t)|^2$")
    axes[0, 2].set_xlabel("heavy R")
    axes[0, 2].set_yticks([])

    full_image = axes[1, 0].imshow(
        first["full"], origin="lower", aspect="auto", cmap="viridis",
        extent=[R[0], R[-1], x[0], x[-1]], vmin=0, vmax=full_max,
    )
    axes[1, 0].set_title(r"Full $\Psi$: $\rho_{xR}=\int dq|\Psi|^2$")
    axes[1, 0].set_xlabel("heavy R")
    axes[1, 0].set_ylabel("electron x")

    eps1_image = axes[1, 1].imshow(
        eps1_arrays[0], origin="lower", aspect="auto", cmap="coolwarm",
        extent=[R[0], R[-1], q[0], q[-1]], vmin=eps1_lo, vmax=eps1_hi,
    )
    axes[1, 1].set_title(
        r"First TDPES $\epsilon^{(1)}(q,R,t)$ (regularized tails)"
    )
    axes[1, 1].set_xlabel("heavy R")
    axes[1, 1].set_ylabel("proton q")

    eps2_line, = axes[1, 2].plot(R, eps2_arrays[0], color="#275DAD", lw=2)
    axes[1, 2].set_ylim(eps2_lo, eps2_hi)
    axes[1, 2].axhline(0, color="0.75", lw=0.7)
    axes[1, 2].set_title(r"Second TDPES $\epsilon^{(2)}(R,t)$")
    axes[1, 2].set_xlabel("heavy R")
    axes[1, 2].set_ylabel("shifted energy")
    axes[1, 2].yaxis.set_major_formatter(NUMBER_FORMATTER)

    # 각 colormap의 수치 범위를 해당 panel 바로 옆에 명시한다.
    colorbar_options = dict(
        pad=0.012, fraction=0.046, format=NUMBER_FORMATTER
    )
    fig.colorbar(
        electron_image, ax=axes[0, 0], label="conditional electron density",
        **colorbar_options,
    )
    fig.colorbar(
        proton_image, ax=axes[0, 1], label=r"$|\Lambda_R|^2$",
        **colorbar_options,
    )
    fig.colorbar(
        heavy_image, ax=axes[0, 2], label=r"$|\chi|^2$",
        **colorbar_options,
    )
    fig.colorbar(
        full_image, ax=axes[1, 0], label=r"$\rho_{xR}$",
        **colorbar_options,
    )
    fig.colorbar(
        eps1_image, ax=axes[1, 1], label=r"shifted $\epsilon^{(1)}$",
        **colorbar_options,
    )

    # 2D map은 서로 다른 좌표쌍 (x,R), (q,R)을 나타낸다. 전자 box로
    # 강제 확장하면 실제 R/q 데이터가 가운데 일부만 차지해 '잘린' 것처럼
    # 보이므로 각 map은 계산에 사용한 native grid extent를 그대로 쓴다.
    title = fig.suptitle(
        f"Multi-component EF densities   t={times[frames[0]]:.4f} fs\n"
        +initial_summary(data), fontsize=11.5,
    )

    def update(number):
        frame = int(frames[number])
        item = reduced_frame(data, frame)
        electron_image.set_data(item["electron"])
        proton_image.set_data(item["proton"])
        heavy_image.set_data(np.tile(item["heavy"][None, :], (12, 1)))
        full_image.set_data(item["full"])
        eps1_image.set_data(shifted_epsilon_1(data, frame))
        eps2_line.set_ydata(shifted_tdpes(data, frame))
        title.set_text(
            f"Multi-component EF densities   t={times[frame]:.4f} fs\n"
            +initial_summary(data)
        )
        return (
            electron_image, proton_image, heavy_image, full_image,
            eps1_image, eps2_line, title,
        )

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"multi_component_density_dynamics.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=2800), dpi=dpi)
    else:
        if fmt == "mp4":
            print("ffmpeg을 찾지 못해 GIF로 대신 저장합니다.")
        path = outdir/"multi_component_density_dynamics.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"Density-colormap 6분할 dynamics 저장: {path}")


def make_gauge_potential_animation(
    data, outdir, fps=12, max_frames=180, dpi=130, fmt="mp4",
    support_floor=1.0e-3, show_tails=False,
):
    """두 TDPES, 세 vector potential, 두 gauge function의 시간 변화를 표시.

    기존 wavefunction 영상은 매 frame의 TDPES를 peak에서 0으로 shift하지만,
    이 분석 영상은 선택된 gauge에서 저장된 *raw* scalar potential을 쓴다.
    따라서 ``theta_k``의 시간 미분에 따른 gauge energy offset도 보존된다.

    2x4 panel 배치는 다음과 같다.

        epsilon_1(q,R) | a(q,R)     | b(q,R) | |Lambda chi|^2
        epsilon_2(R)   | alpha(R)   | theta_1(q,R) | theta_2(R)

    마지막 support panel은 potential 값이 물리적으로 신뢰할 만한 점유 영역과
    regularized low-density tail을 구분하기 위해 함께 표시한다.
    """
    q, R, times = data["q"], data["R"], data["times_fs"]
    frames = selected_frames(len(times), min(max_frames, len(times)))

    # ------------------------------------------------------------------
    # 1. 새 archive에는 theta가 저장된다. 이전 archive도 다시 그릴 수 있도록
    #    key가 없으면 그 archive의 representation을 theta=0 기준으로 본다.
    # ------------------------------------------------------------------
    theta1 = (
        data["theta_1"]
        if "theta_1" in data.files else np.zeros_like(data["a"])
    )                                                               # (nt,nq,nR)
    theta2 = (
        data["theta_2"]
        if "theta_2" in data.files else np.zeros_like(data["alpha"])
    )                                                               # (nt,nR)

    # ------------------------------------------------------------------
    # 2. 영상 도중 색 범위가 흔들리지 않도록 모든 표시 frame을 먼저 훑는다.
    # ------------------------------------------------------------------
    # Scalar epsilon_1은 부호 비대칭 범위를, vector/gauge field는 0을 중심으로
    # 한 대칭 범위를 사용한다. 극단적인 tail spike의 지배를 줄이기 위해
    # robust_limits는 전체 유한값의 2--98 percentile을 사용한다.
    displayed = [
        supported_potential_fields(data, int(i), support_floor, show_tails)
        for i in frames
    ]
    eps1_values = [item["epsilon_1"] for item in displayed]
    a_values = [item["a"] for item in displayed]
    b_values = [item["b"] for item in displayed]
    th1_values = [np.asarray(theta1[i], float) for i in frames]
    eps1_lim = robust_limits(eps1_values)
    a_lim = robust_limits(a_values, symmetric=True)
    b_lim = robust_limits(b_values, symmetric=True)
    th1_lim = robust_limits(th1_values, symmetric=True)
    # 점유 support rho_qR=|Lambda_R(q,t)|^2 |chi(R,t)|^2, shape (nq,nR)
    support_max = max(
        float(np.max(
            np.abs(data["lambda_wavefunction"][i])**2
            *np.abs(data["chi"][i])[None, :]**2
        )) for i in frames
    )

    def line_limits(values):
        """1D line의 전 시간 범위에 여백을 더한다.

        theta_2=0처럼 완전히 상수인 field도 y=0 선이 panel 중앙에서 보이도록
        최소 1e-3의 대칭 여백을 만든다.
        """
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return -1.0, 1.0
        low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            pad = max(0.1*abs(low), 1.0e-3)
        else:
            pad = 0.06*(high-low)
        return low-pad, high+pad

    # epsilon_2, alpha, theta_2는 모두 R만의 함수이므로 1D line으로 그린다.
    eps2_lim = line_limits([item["epsilon_2"] for item in displayed])
    alpha_lim = line_limits([item["alpha"] for item in displayed])
    th2_lim = line_limits(theta2[frames])
    first = int(frames[0])
    first_fields = displayed[0]
    support = first_fields["support"]

    # ------------------------------------------------------------------
    # 3. 첫 frame의 artist를 만든다. 이후 update()는 데이터만 교체한다.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.1), constrained_layout=True)
    extent = [R[0], R[-1], q[0], q[-1]]
    eps1_image = axes[0, 0].imshow(
        first_fields["epsilon_1"], origin="lower", aspect="auto",
        extent=extent, cmap="coolwarm", vmin=eps1_lim[0], vmax=eps1_lim[1],
    )
    a_image = axes[0, 1].imshow(
        first_fields["a"], origin="lower", aspect="auto", extent=extent,
        cmap="coolwarm", vmin=a_lim[0], vmax=a_lim[1],
    )
    b_image = axes[0, 2].imshow(
        first_fields["b"], origin="lower", aspect="auto", extent=extent,
        cmap="coolwarm", vmin=b_lim[0], vmax=b_lim[1],
    )
    support_image = axes[0, 3].imshow(
        support, origin="lower", aspect="auto", extent=extent,
        cmap="magma", vmin=0.0, vmax=max(support_max, 1.0e-12),
    )
    for ax in axes[0]:
        ax.set_xlabel("heavy R")
        ax.set_ylabel("proton q")
    axes[0, 0].set_title(r"Raw first TDPES $\epsilon^{(1)}(q,R,t)$")
    axes[0, 1].set_title(r"Vector potential $a(q,R,t)$")
    axes[0, 2].set_title(r"Vector potential $b(q,R,t)$")
    axes[0, 3].set_title(r"Occupied support $|\Lambda_R\chi|^2$")

    eps2_line, = axes[1, 0].plot(R, first_fields["epsilon_2"], lw=2)
    alpha_line, = axes[1, 1].plot(R, first_fields["alpha"], lw=2, color="#8C4F9E")
    theta1_image = axes[1, 2].imshow(
        theta1[first], origin="lower", aspect="auto", extent=extent,
        cmap="twilight", vmin=th1_lim[0], vmax=th1_lim[1],
    )
    theta2_line, = axes[1, 3].plot(R, theta2[first], lw=2, color="#2B8C6B")
    for ax, limits in zip(
        (axes[1, 0], axes[1, 1], axes[1, 3]),
        (eps2_lim, alpha_lim, th2_lim),
    ):
        ax.set_xlim(R[0], R[-1])
        ax.set_ylim(*limits)
        ax.axhline(0.0, color="0.75", lw=0.7)
        ax.set_xlabel("heavy R")
        ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
    axes[1, 0].set_title(r"Raw second TDPES $\epsilon^{(2)}(R,t)$")
    axes[1, 1].set_title(r"Vector potential $\alpha(R,t)$")
    axes[1, 2].set_title(r"Gauge function $\theta_1(q,R,t)$")
    axes[1, 2].set_xlabel("heavy R")
    axes[1, 2].set_ylabel("proton q")
    axes[1, 3].set_title(r"Gauge function $\theta_2(R,t)$")

    # 각 2D field 바로 옆에 고정 colorbar를 둔다. 모든 frame에서 같은 색은
    # 같은 수치를 의미하므로 서로 다른 시간의 값을 눈으로 비교할 수 있다.
    for image_artist, ax, label in (
        (eps1_image, axes[0, 0], r"$\epsilon^{(1)}$"),
        (a_image, axes[0, 1], r"$a$"),
        (b_image, axes[0, 2], r"$b$"),
        (support_image, axes[0, 3], r"$|\Lambda\chi|^2$"),
        (theta1_image, axes[1, 2], r"$\theta_1$"),
    ):
        fig.colorbar(
            image_artist, ax=ax, label=label, pad=0.012, fraction=0.046,
            format=NUMBER_FORMATTER,
        )

    gauge_name = str(data["gauge"].item()) if "gauge" in data.files else "unknown"
    title = fig.suptitle(
        f"Gauge and exact-potential dynamics   t={times[first]:.4f} fs\n"
        f"gauge={gauge_name}; raw values; "
        +("tails shown" if show_tails else f"support >= {support_floor:g} peak"),
        fontsize=12,
    )

    def update(number):
        """선택된 저장 frame의 일곱 field와 support를 한꺼번에 갱신."""
        frame = int(frames[number])
        fields = displayed[number]
        eps1_image.set_data(fields["epsilon_1"])
        a_image.set_data(fields["a"])
        b_image.set_data(fields["b"])
        support_image.set_data(fields["support"])
        eps2_line.set_ydata(fields["epsilon_2"])
        alpha_line.set_ydata(fields["alpha"])
        theta1_image.set_data(theta1[frame])
        theta2_line.set_ydata(theta2[frame])
        title.set_text(
            f"Gauge and exact-potential dynamics   t={times[frame]:.4f} fs\n"
            f"gauge={gauge_name}; raw values; "
            +("tails shown" if show_tails else f"support >= {support_floor:g} peak")
        )
        return (
            eps1_image, a_image, b_image, support_image,
            eps2_line, alpha_line, theta1_image, theta2_line, title,
        )

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"multi_component_gauge_potential_dynamics.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=dpi)
    else:
        if fmt == "mp4":
            print("ffmpeg을 찾지 못해 GIF로 대신 저장합니다.")
        path = outdir/"multi_component_gauge_potential_dynamics.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"Gauge/TDPES/vector-potential dynamics 저장: {path}")


def run(args, data=None):
    data = data if data is not None else load_archive(
        args.archive, materialize=not getattr(args, "low_memory", False)
    )
    requested_outdir = Path(args.outdir) if args.outdir else Path(args.archive).parent/"figures"
    outdir = dated_results_dir(requested_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_initial_state(data, outdir, dpi=args.dpi)
    plot_snapshots(data, outdir, count=args.snapshots, dpi=args.dpi)
    plot_factor_profiles(data, outdir, frame=args.profile_frame, dpi=args.dpi)
    if not args.no_animation:
        if args.animation_style in ("all", "both", "wavefunction"):
            make_wavefunction_animation(
                data, outdir, fps=args.fps, max_frames=args.max_frames,
                dpi=args.animation_dpi, fmt=args.format,
            )
        if args.animation_style in ("all", "both", "density"):
            make_density_animation(
                data, outdir, fps=args.fps, max_frames=args.max_frames,
                dpi=args.animation_dpi, fmt=args.format,
            )
        if args.animation_style in ("all", "potentials"):
            make_gauge_potential_animation(
                data, outdir, fps=args.fps, max_frames=args.max_frames,
                dpi=args.animation_dpi, fmt=args.format,
                support_floor=getattr(args, "potential_support_floor", 1.0e-3),
                show_tails=getattr(args, "show_potential_tails", False),
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="direct 또는 reference NPZ")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--snapshots", type=int, default=4)
    parser.add_argument("--profile-frame", type=int, default=-1)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--animation-dpi", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument(
        "--animation-style",
        choices=("all", "both", "wavefunction", "density", "potentials"),
        default="all",
        help="all은 wavefunction/density/gauge-potential 영상 세 개를 만든다",
    )
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument(
        "--potential-support-floor", type=float, default=1.0e-3,
        help="gauge-potential 영상에서 표시할 frame별 occupied-support 상대 cutoff",
    )
    parser.add_argument(
        "--show-potential-tails", action="store_true",
        help="gauge-potential 영상에서 probability가 없는 tail도 표시",
    )
    parser.add_argument(
        "--low-memory", action="store_true",
        help="큰 배열을 RAM에 유지하지 않지만 compressed NPZ 반복 접근은 느려질 수 있음",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
