#!/usr/bin/env python3
"""한 NVIDIA GPU에서 multi-component direct EF를 시간 전파한다.

초기 local BO 상태는 CPU에서 한 번 만들고, 실제 time loop의 Phi/Lambda/chi,
미분, coupling, RK4, hard-wall 전자 FFT는 모두 GPU에 유지한다. 저장 frame만
CPU로 복사하므로 기존 visualization/analysis 프로그램이 같은 NPZ를 읽는다.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from result_paths import dated_results_dir

from multi_component_exact_factorization.core import (
    AU_PER_FS,
    add_model_arguments,
    build_model,
    initial_factors,
)
from multi_component_exact_factorization.propagate import output_gauge

from .throttle import gpu_util_percent, throttle_delay

from .gpu_core import (
    DIAGNOSTIC_FIELDS,
    all_finite,
    configure_fused_periodic_derivative,
    cp,
    field_maxima,
    full_step,
    instantaneous_functionals,
    make_gpu_model,
    merge_maxima,
    pnc_error,
    to_gpu_factors,
)


def numpy_dtypes(precision):
    """Archive에 저장할 native propagation dtype."""
    if precision == "double":
        return np.float64, np.complex128
    return np.float32, np.complex64


def device_description(device_id):
    """선택한 CUDA device 이름과 VRAM을 사람이 읽을 문자열로 만든다."""
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    total_gib = properties["totalGlobalMem"]/(1024**3)
    return f"GPU {device_id}: {name}, {total_gib:.2f} GiB"


def run(args):
    """CPU 초기화 -> GPU direct EF 전파 -> CPU-compatible NPZ 저장."""
    if getattr(args, "precision", "double") != "double":
        raise ValueError(
            "production GPU propagation은 검증된 double precision으로 고정됩니다"
        )
    args.precision = "double"
    cp.cuda.Device(args.device).use()
    print(device_description(args.device))
    print(f"계산 정밀도: {args.precision}")
    gpu_util_limit = getattr(args, "gpu_util_limit", 100.0)
    gpu_throttle_every = max(1, getattr(args, "gpu_throttle_every", 20))
    if gpu_util_limit < 100.0:
        print(
            f"GPU 평균 duty-cycle 제한: {gpu_util_limit:g}% "
            f"({gpu_throttle_every} step마다 조절; nvidia-smi 표시는 변동 가능)"
        )

    outdir = dated_results_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Local BO diagonalization은 CPU SciPy가 효율적이고 처음 한 번만 필요하다.
    cpu_model = build_model(args)
    phi_cpu, lam_cpu, chi_cpu = initial_factors(cpu_model, args)
    optimization = getattr(args, "gpu_optimization", "fused")
    configure_fused_periodic_derivative(optimization == "fused")
    gpu_model = make_gpu_model(
        cpu_model, args.precision,
        reuse_stage_derivatives=(optimization != "baseline"),
    )
    phi, lam, chi = to_gpu_factors(phi_cpu, lam_cpu, chi_cpu, gpu_model)
    real_dtype, complex_dtype = numpy_dtypes(args.precision)

    print(f"초기 배열: Phi={phi.shape}, Lambda={lam.shape}, chi={chi.shape}")
    print(
        f"격자: dx={cpu_model.dx:.6f}, dq={cpu_model.dq:.6f}, "
        f"dR={cpu_model.dR:.6f}; hard wall "
        f"{cpu_model.x_left:.3f}..{cpu_model.x_right:.3f}"
    )
    print(
        "초기 Gaussian: "
        f"q0={args.q0:.4f}, sigma_q={args.proton_sigma:.6f}; "
        f"R0={args.R0:.4f}, sigma_R={args.heavy_sigma:.6f}"
    )
    print(
        "수치 scheme: periodic 5-point central D1/D2; "
        "product-preserving gamma transfer + discrete product projection; "
        f"ratio_floor={args.ratio_floor:.1e}, "
        f"mask(Phi,Lambda)=({args.mask_threshold_phi:.1e},"
        f"{args.mask_threshold_lam:.1e})"
    )
    print(
        "GPU 실행 경로: "
        +(
            "stage reuse + one-pass fused periodic stencil (optimized)"
            if optimization == "fused" else
            "stage-local derivative reuse"
            if optimization == "reuse" else
            "baseline repeated derivatives (validation)"
        )
    )

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    # 0은 trajectory 길이에 맞춘 자동 간격이다. 긴 계산에서도 archive와
    # terminal 출력이 무제한 커지지 않도록 각각 약 200 frame, 20 progress,
    # 500 finite check를 목표로 한다.
    args.save_every = (
        args.save_every if args.save_every > 0
        else max(1, int(np.ceil(max(n_steps, 1)/200.0)))
    )
    args.progress_every = (
        args.progress_every if args.progress_every > 0
        else max(1, int(np.ceil(max(n_steps, 1)/20.0)))
    )
    args.check_every = (
        args.check_every if args.check_every > 0
        else max(1, int(np.ceil(max(n_steps, 1)/500.0)))
    )
    print(
        "자동/선택 간격: "
        f"save={args.save_every}, progress={args.progress_every}, "
        f"finite-check={args.check_every} step"
    )
    save_steps = list(range(0, n_steps+1, args.save_every))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    nt = len(save_steps)

    # 저장 배열만 host RAM에 둔다. Production archive는 complex128이다.
    times_fs = np.empty(nt)
    phis = np.empty((nt, args.nx, args.nq, args.nR), dtype=complex_dtype)
    lams = np.empty((nt, args.nq, args.nR), dtype=complex_dtype)
    chis = np.empty((nt, args.nR), dtype=complex_dtype)
    avec = np.empty((nt, args.nq, args.nR), dtype=real_dtype)
    bvec = np.empty_like(avec)
    alpha = np.empty((nt, args.nR), dtype=real_dtype)
    eps1 = np.empty_like(avec)
    eps2 = np.empty((nt, args.nR), dtype=real_dtype)
    theta1 = np.empty_like(avec)
    theta2 = np.empty((nt, args.nR), dtype=real_dtype)
    norm = np.empty(nt)
    pnc = np.empty(nt)
    projection_correction = np.empty(nt)
    diagnostic_history = {
        name: np.empty(nt) for name in DIAGNOSTIC_FIELDS
    }
    psis = []

    def save_frame(frame, step, correction=0.0, interval_diagnostics=None):
        """저장 시점에만 GPU factor/field를 CPU로 내려 동일 NPZ 형식으로 기록."""
        fields_gpu = instantaneous_functionals(
            phi, lam, chi, gpu_model, floor=args.ratio_floor,
            mask_threshold_phi=args.mask_threshold_phi,
            mask_threshold_lam=args.mask_threshold_lam,
        )
        phi_base, lam_base, chi_base = cp.asnumpy(phi), cp.asnumpy(lam), cp.asnumpy(chi)
        fields_base = {
            key: cp.asnumpy(fields_gpu[key])
            for key in ("a", "b", "alpha", "epsilon_1", "epsilon_2")
        }
        time_au = step*args.dt_au
        phi_out, lam_out, chi_out, saved, th1, th2 = output_gauge(
            phi_base, lam_base, chi_base, fields_base,
            time_au, cpu_model, args,
        )
        phi_out = phi_out.astype(complex_dtype, copy=False)
        lam_out = lam_out.astype(complex_dtype, copy=False)
        chi_out = chi_out.astype(complex_dtype, copy=False)
        psi = phi_out*lam_out[None, :, :]*chi_out[None, None, :]

        times_fs[frame] = time_au/AU_PER_FS
        phis[frame], lams[frame], chis[frame] = phi_out, lam_out, chi_out
        avec[frame], bvec[frame] = saved["a"], saved["b"]
        alpha[frame] = saved["alpha"]
        eps1[frame], eps2[frame] = saved["epsilon_1"], saved["epsilon_2"]
        theta1[frame], theta2[frame] = th1, th2
        norm[frame] = np.sum(np.abs(psi)**2, dtype=np.float64)*(
            cpu_model.dx*cpu_model.dq*cpu_model.dR
        )
        pnc[frame] = float(pnc_error(phi, lam, gpu_model).get())
        projection_correction[frame] = float(
            correction.get() if hasattr(correction, "get") else correction
        )
        saved_diagnostics = merge_maxima(
            field_maxima(fields_gpu), interval_diagnostics or {}
        )
        for name, values in diagnostic_history.items():
            values[frame] = float(saved_diagnostics[name].get())
        if args.save_psi:
            psis.append(psi.copy())

    save_frame(0, 0)
    frame = 1
    interval_correction = cp.asarray(0.0)
    interval_diagnostics = merge_maxima()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    wall_start = time.perf_counter()
    throttle_active_seconds = 0.0
    throttle_sleep_seconds = 0.0
    throttle_chunk_start = None
    start.record()

    for step in range(1, n_steps+1):
        if gpu_util_limit < 100.0 and throttle_chunk_start is None:
            throttle_chunk_start = time.perf_counter()
        phi, lam, chi, step_correction, step_diagnostics = full_step(
            phi, lam, chi, args.dt_au, gpu_model, args.ratio_floor,
            args.mask_threshold_phi, args.mask_threshold_lam,
        )
        interval_correction = cp.maximum(
            interval_correction, step_correction
        )
        interval_diagnostics = merge_maxima(
            interval_diagnostics, step_diagnostics
        )
        throttle_now = (
            gpu_util_limit < 100.0
            and (step % gpu_throttle_every == 0 or step == n_steps)
        )
        if throttle_now:
            # CuPy launches asynchronously. Synchronize only at coarse intervals
            # so the measured active time reflects completed GPU work.
            cp.cuda.get_current_stream().synchronize()
            active_seconds = time.perf_counter()-throttle_chunk_start
            delay = throttle_delay(active_seconds, gpu_util_limit)
            throttle_active_seconds += active_seconds
            if delay > 0.0:
                time.sleep(delay)
                throttle_sleep_seconds += delay
            throttle_chunk_start = None
        must_save = frame < nt and step == save_steps[frame]
        must_check = (
            step % max(1, args.check_every) == 0 or must_save or step == n_steps
        )
        if must_check and not all_finite(phi, lam, chi):
            raise FloatingPointError(
                f"step {step}에서 non-finite 값이 발생했습니다. "
                "마지막 정상 frame과 dt/grid/mask 진단을 확인하세요."
            )
        if must_save:
            save_frame(
                frame, step, interval_correction, interval_diagnostics
            )
            frame += 1
            interval_correction = cp.asarray(0.0)
            interval_diagnostics = merge_maxima()
        if step % max(1, args.progress_every) == 0 or step == n_steps:
            print(
                f"step {step:7d}/{n_steps}  "
                f"t={step*args.dt_au/AU_PER_FS:9.4f} fs"
            )

    stop.record()
    stop.synchronize()
    gpu_seconds = cp.cuda.get_elapsed_time(start, stop)/1000.0
    wall_seconds = time.perf_counter()-wall_start

    diagnostic_history["max_abs_support_gamma_phi_dt"] = (
        diagnostic_history["max_abs_support_gamma_phi"]*args.dt_au
    )
    diagnostic_history["max_abs_support_gamma_lam_dt"] = (
        diagnostic_history["max_abs_support_gamma_lam"]*args.dt_au
    )

    # Gauge-dependent time connection은 저장된 작은 시간축을 CPU에서 계산한다.
    if nt >= 2:
        times_au = times_fs*AU_PER_FS
        edge_order = 2 if nt >= 3 else 1
        dphi_dt = np.gradient(phis, times_au, axis=0, edge_order=edge_order)
        dlam_dt = np.gradient(lams, times_au, axis=0, edge_order=edge_order)
        epsilon_gd_1 = (
            np.sum(
                np.conj(phis)*(-1j*dphi_dt), axis=1, dtype=np.complex128
            )*cpu_model.dx
        ).real
        lambda_gd = (
            np.sum(
                np.conj(lams)*(-1j*dlam_dt), axis=1, dtype=np.complex128
            )*cpu_model.dq
        ).real
        epsilon_gd_2 = lambda_gd+np.sum(
            np.abs(lams)**2*epsilon_gd_1, axis=1, dtype=np.float64
        )*cpu_model.dq
    else:
        epsilon_gd_1 = np.zeros_like(eps1)
        epsilon_gd_2 = np.zeros_like(eps2)

    gauge_coefficients = (
        args.theta1_q_gradient, args.theta1_R_gradient, args.theta1_frequency,
        args.theta2_R_gradient, args.theta2_frequency,
    )
    gauge_name = (
        "parallel_transport_two_level"
        if all(value == 0.0 for value in gauge_coefficients)
        else "linear_transform_of_parallel_transport_two_level"
    )
    args.backend = "cupy"
    args.internal_precision = args.precision
    payload = dict(
        kind=np.array("direct_multi_component_exact_factorization"),
        representation=np.array(
            "nested_realspace_independent_harmonic_hardwall_electron"
        ),
        local_norm_correction=np.array(
            "product_preserving_nested_tangent_correction"
        ),
        discrete_product_projection=np.array(
            "nested_tangent_projection_to_periodic_nuclear_D2"
        ),
        spatial_derivative=np.array("periodic_five_point_central_D1_D2"),
        ratio_regularization=np.array(
            "joint_support_mask_on_log_amplitude_gradient_only"
        ),
        ratio_floor=np.array(args.ratio_floor),
        mask_threshold_phi=np.array(args.mask_threshold_phi),
        mask_threshold_lam=np.array(args.mask_threshold_lam),
        backend=np.array("cupy_single_gpu"),
        precision=np.array(args.precision),
        cuda_device=np.array(args.device),
        gauge=np.array(gauge_name),
        base_gauge=np.array("parallel_transport_two_level"),
        x=cpu_model.x, q=cpu_model.q, R=cpu_model.R, times_fs=times_fs,
        phi=phis, lambda_wavefunction=lams, chi=chis,
        a=avec, b=bvec, alpha=alpha,
        epsilon_1=eps1, epsilon_2=eps2,
        theta_1=theta1, theta_2=theta2,
        epsilon_gd_1=epsilon_gd_1, epsilon_gd_2=epsilon_gd_2,
        norm=norm, pnc_error=pnc,
        pnc_projection_correction=projection_correction,
        **diagnostic_history,
        gpu_seconds=np.array(gpu_seconds), wall_seconds=np.array(wall_seconds),
        gpu_util_limit=np.array(gpu_util_limit),
        gpu_throttle_sleep_seconds=np.array(throttle_sleep_seconds),
        gpu_optimization=np.array(optimization),
        args=np.array([vars(args)], dtype=object),
    )
    if args.save_psi:
        payload["psi"] = np.asarray(psis)
    path = outdir/"multi_component_direct_ef_gpu.npz"
    np.savez_compressed(path, **payload)

    used = cp.get_default_memory_pool().used_bytes()/(1024**3)
    total = cp.get_default_memory_pool().total_bytes()/(1024**3)
    print(f"저장 완료: {path}")
    print(f"GPU event 시간: {gpu_seconds:.3f} s; wall 시간: {wall_seconds:.3f} s")
    if n_steps:
        print(f"평균 wall 시간: {wall_seconds/n_steps:.6f} s/step")
    if gpu_util_limit < 100.0:
        controlled_seconds = throttle_active_seconds+throttle_sleep_seconds
        measured_duty = (
            100.0*throttle_active_seconds/controlled_seconds
            if controlled_seconds else 0.0
        )
        print(
            f"GPU throttle 대기: {throttle_sleep_seconds:.3f} s; "
            f"계산/대기 duty cycle: {measured_duty:.1f}%"
        )
    print(f"CuPy memory pool: used={used:.2f} GiB, reserved={total:.2f} GiB")
    print("핵심 수치 진단:")
    print(f"  max |norm-1|:              {np.max(np.abs(norm-1.0)):.3e}")
    print(f"  max saved PNC residual:    {np.max(pnc):.3e}")
    print(f"  max PNC projection load:   {np.max(projection_correction):.3e}")
    for name in (
        "max_abs_support_gamma_phi_dt",
        "max_abs_support_gamma_lam_dt",
        "max_effective_product_residual_l2",
        "max_abs_full_norm_rate_after_product_projection",
    ):
        if name in diagnostic_history:
            print(f"  {name}: {np.max(diagnostic_history[name]):.3e}")
    if getattr(args, "verbose_diagnostics", False):
        print("전체 개발용 진단:")
        for name, values in diagnostic_history.items():
            print(f"  {name}: {np.max(values):.3e}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default="results/multi_component_exact_factorization/gpu"
    )
    parser.add_argument("--device", type=int, default=0, help="사용할 CUDA GPU index")
    # Production propagation은 검증된 complex128/float64로 고정한다. 예전
    # 명령의 ``--precision double``은 계속 받아들이되 help에서는 숨긴다.
    parser.add_argument(
        "--precision", choices=("double",), default="double",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpu-util-limit", type=gpu_util_percent, default=100.0,
        help=(
            "step 사이 대기로 맞출 평균 GPU duty cycle(0 < percent <= 100); "
            "기본 100은 제한 없음"
        ),
    )
    parser.add_argument(
        "--gpu-throttle-every", type=int, default=20,
        help="GPU duty-cycle을 측정하고 대기할 step 간격(기본 20)",
    )
    parser.add_argument(
        "--gpu-optimization", choices=("fused", "reuse", "baseline"),
        default="fused",
        help=(
            "fused는 stage 재사용과 one-pass CUDA 5점 stencil; reuse는 "
            "CuPy roll stencil+stage 재사용; baseline은 기존 반복 계산"
        ),
    )
    parser.add_argument("--dt-au", type=float, default=0.005)
    parser.add_argument("--t-final-fs", type=float, default=0.05)
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="저장 step 간격; 0이면 약 200 frame이 되도록 자동 선택",
    )
    parser.add_argument(
        "--progress-every", type=int, default=0,
        help="진행 출력 간격; 0이면 전체 실행 중 약 20번 출력",
    )
    parser.add_argument(
        "--check-every", type=int, default=0,
        help=(
            "non-finite GPU 검사 간격; 0이면 약 500번 검사하며 저장 frame은 "
            "항상 검사"
        ),
    )
    regularization = parser.add_argument_group("node/tail regularization")
    regularization.add_argument(
        "--ratio-floor", type=float, default=1.0e-14,
        help="logarithmic derivative의 zero division만 막는 numerical floor",
    )
    regularization.add_argument(
        "--mask-threshold-phi", type=float, default=1.0e-10,
        help="joint density 기반 Phi amplitude-gradient mask",
    )
    regularization.add_argument(
        "--mask-threshold-lam", type=float, default=1.0e-10,
        help="heavy density 기반 Lambda amplitude-gradient mask",
    )
    regularization.add_argument(
        "--density-threshold", type=float, default=None,
        help="deprecated: 지정하면 두 mask threshold에 같은 값을 사용",
    )
    parser.add_argument("--save-psi", action="store_true")
    parser.add_argument(
        "--verbose-diagnostics", action="store_true",
        help="종료 시 archive에 저장된 개발용 진단을 모두 terminal에 출력",
    )
    render = parser.add_argument_group("계산 완료 후 자동 렌더링")
    parser.set_defaults(render_after=True, render_fast=True)
    render.add_argument(
        "--no-render-after", dest="render_after", action="store_false",
        help="계산 후 자동 report/동영상 생성을 생략",
    )
    render.add_argument(
        "--render-full", dest="render_fast", action="store_false",
        help="자동 렌더링을 빠른 preview 대신 full 품질로 생성",
    )
    render.add_argument(
        "--render-after", dest="render_after", action="store_true",
        help=argparse.SUPPRESS,
    )
    render.add_argument(
        "--render-fast", dest="render_fast", action="store_true",
        help=argparse.SUPPRESS,
    )

    gauge = parser.add_argument_group("두 단계 gauge")
    gauge.add_argument("--theta1-q-gradient", type=float, default=0.0)
    gauge.add_argument("--theta1-R-gradient", type=float, default=0.0)
    gauge.add_argument("--theta1-frequency", type=float, default=0.0)
    gauge.add_argument("--theta2-R-gradient", type=float, default=0.0)
    gauge.add_argument("--theta2-frequency", type=float, default=0.0)
    add_model_arguments(parser)
    args = parser.parse_args()
    if args.density_threshold is not None:
        args.mask_threshold_phi = args.density_threshold
        args.mask_threshold_lam = args.density_threshold
    return args


def main(args=None):
    """GPU 전파 배열이 해제된 뒤 선택적으로 CPU 렌더링을 실행한다."""
    args = parse_args() if args is None else args
    path = run(args)
    if getattr(args, "render_after", False):
        from multi_component_exact_factorization.render_all import (
            render_completed_run,
        )
        print("계산 완료 후 전체 결과 렌더링을 시작합니다.")
        render_completed_run(path, fast=getattr(args, "render_fast", False))
    return path


if __name__ == "__main__":
    main()
