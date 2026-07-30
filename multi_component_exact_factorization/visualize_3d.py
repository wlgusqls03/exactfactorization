#!/usr/bin/env python3
"""Full ``|Psi(x,q,R,t)|^2``의 독립적인 interactive 3D HTML 생성.

이 그림의 세 축은 실제 공간의 x/y/z가 아니라 서로 다른 입자의 1D 좌표
``(electron x, proton q, heavy R)``로 이루어진 configuration space이다.
브라우저에서 다음 조작을 할 수 있다.

- 왼쪽 드래그: 회전
- 휠: 확대/축소
- 오른쪽 드래그: 평행 이동
- 아래 slider: 특정 저장 시간 선택
- Play/Pause: 시간 dynamics 재생/정지

HTML 안에 Plotly JavaScript를 포함하므로 생성 후 인터넷 연결이 필요 없다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from result_paths import dated_results_dir


def load_density(data, frame):
    """한 frame의 full molecular probability density ``(nx,nq,nR)``."""
    if "psi" in data:
        return np.abs(data["psi"][frame])**2
    phi2 = np.abs(data["phi"][frame])**2
    lam2 = np.abs(data["lambda_wavefunction"][frame])**2
    chi2 = np.abs(data["chi"][frame])**2
    return phi2*lam2[None, :, :]*chi2[None, None, :]


def choose_indices(size, maximum):
    """양 끝을 포함하면서 한 축을 최대 ``maximum``점으로 downsample."""
    if size <= maximum:
        return np.arange(size)
    return np.unique(np.linspace(0, size-1, maximum).round().astype(int))


def choose_frames(nt, maximum):
    if nt <= maximum:
        return np.arange(nt)
    return np.unique(np.linspace(0, nt-1, maximum).round().astype(int))


def initial_summary(data):
    """3D 제목에 표시할 초기 중심과 질량을 NPZ metadata에서 읽는다."""
    if "args" not in data.files or data["args"].size != 1:
        return "initial parameters unavailable"
    options = data["args"].reshape(-1)[0]
    if not isinstance(options, dict):
        return "initial parameters unavailable"
    excitation = int(options.get("electron_excitation", 0))
    return (
        f"initial: electron=local H_BO state n={excitation}, "
        f"q0={options['q0']:.2f}, "
        f"R0={options['R0']:.2f} a0 &nbsp; | &nbsp; masses (me): "
        f"mp={options['proton_mass']:.0f}, MH={options['heavy_mass']:.0f} "
        f"&nbsp; | &nbsp; xL={options['left_position']:.2f}, "
        f"ZL={options.get('left_charge', np.nan):.2f}"
    )


def run(args):
    data = np.load(args.archive, allow_pickle=True)
    required = {"x", "q", "R", "times_fs", "phi", "lambda_wavefunction", "chi"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"3D 그림에 필요한 archive key가 없습니다: {missing}")

    x, q, R = data["x"], data["q"], data["R"]
    times = data["times_fs"]
    ix = choose_indices(len(x), args.max_axis_points)
    iq = choose_indices(len(q), args.max_axis_points)
    iR = choose_indices(len(R), args.max_axis_points)
    frame_ids = choose_frames(len(times), args.max_frames)

    xs, qs, Rs = x[ix], q[iq], R[iR]
    # indexing='ij'이므로 density(nx,nq,nR)와 좌표 flatten 순서가 정확히 같다.
    X, Q, RR = np.meshgrid(xs, qs, Rs, indexing="ij")
    flat_x, flat_q, flat_R = X.ravel(), Q.ravel(), RR.ravel()

    densities = []
    global_max = 0.0
    for frame in frame_ids:
        rho = load_density(data, int(frame))[np.ix_(ix, iq, iR)]
        densities.append(rho)
        global_max = max(global_max, float(np.max(rho)))
    if global_max <= 0.0:
        raise ValueError("3D로 표시할 양의 molecular density가 없습니다.")
    isomin = args.isomin_fraction*global_max
    number_format = ".2f" if global_max >= 1.0e-2 else ".2e"
    hover_value = "%{value:"+number_format+"}"
    initial_text = initial_summary(data)

    options = data["args"].reshape(-1)[0] if "args" in data.files else {}
    options = options if isinstance(options, dict) else {}
    common_min = min(
        float(x[0]), float(q[0]), float(R[0]),
        float(options.get("x_min", x[0])),
    )
    common_max = max(
        float(x[-1]), float(q[-1]), float(R[-1]),
        float(options.get("x_max", x[-1])),
    )

    def trace_for(rho, show_colorbar):
        """동일한 절대 color scale을 사용하는 한 frame의 isosurface."""
        return go.Isosurface(
            x=flat_x, y=flat_q, z=flat_R, value=rho.ravel(),
            isomin=isomin, isomax=global_max,
            cmin=0.0, cmax=global_max,
            surface_count=args.surface_count,
            opacity=args.opacity,
            colorscale=args.colorscale,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(
                title=dict(text="|Psi|^2"),
                tickformat=number_format, len=0.72,
            ),
            showscale=show_colorbar,
            hovertemplate=(
                "electron x=%{x:.3f}<br>proton q=%{y:.3f}<br>"
                "heavy R=%{z:.3f}<br>|Psi|^2="+hover_value+"<extra></extra>"
            ),
        )

    plot_frames = []
    for number, (frame, rho) in enumerate(zip(frame_ids, densities)):
        plot_frames.append(go.Frame(
            name=str(number),
            data=[trace_for(rho, show_colorbar=True)],
            traces=[0],
            layout=go.Layout(
                title_text=(
                    "Full multi-component density in configuration space"
                    f" &nbsp; t={times[frame]:.5f} fs"
                    f"<br><sup>{initial_text}</sup>"
                )
            ),
        ))

    steps = []
    for number, frame in enumerate(frame_ids):
        steps.append(dict(
            method="animate",
            label=f"{times[frame]:.4f}",
            args=[
                [str(number)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
        ))

    frame_duration = max(1, int(round(1000.0/args.fps)))
    fig = go.Figure(
        data=[trace_for(densities[0], show_colorbar=True)],
        frames=plot_frames,
    )
    fig.update_layout(
        title=(
            "Full multi-component density in configuration space"
            f" &nbsp; t={times[frame_ids[0]]:.5f} fs"
            f"<br><sup>{initial_text}</sup>"
        ),
        width=args.width,
        height=args.height,
        margin=dict(l=10, r=10, t=105, b=85),
        scene=dict(
            xaxis=dict(title="electron x", range=[common_min, common_max]),
            yaxis=dict(title="proton q", range=[common_min, common_max]),
            zaxis=dict(title="heavy R", range=[common_min, common_max]),
            aspectmode="cube",
            uirevision="keep-camera",
            camera=dict(eye=dict(x=1.55, y=1.45, z=1.15)),
        ),
        sliders=[dict(
            active=0, steps=steps,
            currentvalue=dict(prefix="time (fs): "),
            pad=dict(t=40), len=0.78, x=0.12,
        )],
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.01, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.88)", bordercolor="gray",
            buttons=[
                dict(
                    label="Play", method="animate",
                    args=[
                        None,
                        dict(
                            fromcurrent=True, mode="immediate",
                            frame=dict(duration=frame_duration, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="Pause", method="animate",
                    args=[
                        [None],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )],
        annotations=[dict(
            x=0.5, y=-0.11, xref="paper", yref="paper", showarrow=False,
            text=(
                "Drag to rotate · Wheel to zoom · Pause before close inspection · "
                f"isosurface >= {args.isomin_fraction:.1%} of global max"
            ),
        )],
    )

    requested_outdir = Path(args.outdir) if args.outdir else Path(args.archive).parent/"figures"
    outdir = dated_results_dir(requested_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir/"multi_component_full_density_3d.html"
    fig.write_html(
        path, include_plotlyjs=True, full_html=True,
        auto_play=True,
        animation_opts=dict(
            frame=dict(duration=frame_duration, redraw=True),
            transition=dict(duration=0), fromcurrent=True,
        ),
        config={
            "scrollZoom": True, "displaylogo": False,
            "responsive": True,
        },
    )
    print(f"Interactive 3D HTML 저장: {path}")
    print(
        f"표시 grid: {len(xs)} x {len(qs)} x {len(Rs)}, "
        f"frame={len(frame_ids)}, global max density={global_max:.3e}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--max-axis-points", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--surface-count", type=int, default=7)
    parser.add_argument("--isomin-fraction", type=float, default=0.025)
    parser.add_argument("--opacity", type=float, default=0.32)
    parser.add_argument("--colorscale", default="Viridis")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1050)
    parser.add_argument("--height", type=int, default=820)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
