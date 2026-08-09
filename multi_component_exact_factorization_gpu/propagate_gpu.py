#!/usr/bin/env python3
"""한 NVIDIA GPU에서 multi-component direct EF를 시간 전파한다.

초기 local BO 상태는 CPU에서 한 번 만들고, 실제 time loop의 Phi/Lambda/chi,
미분, coupling, RK4, hard-wall 전자 FFT는 모두 GPU에 유지한다. 저장 frame만
CPU로 복사하므로 기존 visualization/analysis 프로그램이 같은 NPZ를 읽는다.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import shlex
import sys
import time
import traceback

import numpy as np

from result_paths import dated_results_dir

from multi_component_exact_factorization.core import (
    AU_PER_FS,
    add_model_arguments,
    build_model,
    initial_factors,
    mask_threshold_for_probability_budget,
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
    evaluate_mask_residual_diagnostics,
    to_gpu_factors,
)


class _Tee:
    """Line-buffered terminal/file fan-out used by command-line runs."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


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


def probability_budgets(value):
    """Parse a comma-separated list of diagnostic mask mass budgets."""
    try:
        values = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budget은 쉼표로 구분한 실수여야 합니다") from exc
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("각 probability budget은 0과 1 사이여야 합니다")
    return values


def run(args):
    """CPU 초기화 -> GPU direct EF 전파 -> CPU-compatible NPZ 저장."""
    if getattr(args, "precision", "double") != "double":
        raise ValueError(
            "production GPU propagation은 검증된 double precision으로 고정됩니다"
        )
    args.precision = "double"
    args.product_projection_floor_phi = getattr(
        args, "product_projection_floor_phi", args.mask_threshold_phi
    )
    args.product_projection_floor_lam = getattr(
        args, "product_projection_floor_lam", args.mask_threshold_lam
    )
    args.mask_residual_diagnostics = getattr(
        args, "mask_residual_diagnostics", True
    )
    args.mask_probability_budgets = getattr(
        args, "mask_probability_budgets", (1.0e-9, 1.0e-8, 1.0e-7)
    )
    if args.weak_log_delta <= 0.0 or args.weak_log_smoothing < 0.0:
        raise ValueError("weak-log delta는 양수, smoothing은 0 이상이어야 합니다")
    if args.weak_log_tolerance <= 0.0 or args.weak_log_max_iterations < 1:
        raise ValueError("weak-log tolerance/iterations 설정이 잘못되었습니다")
    if min(
        args.projection_tau_phi, args.projection_tau_lam,
        args.projection_tau_chi,
    ) < 0.0:
        raise ValueError("projection tau는 0 이상이어야 합니다")
    if args.projection_support_epsilon <= 0.0:
        raise ValueError("projection support epsilon은 양수여야 합니다")
    if args.deep_tail_zero_threshold < 0.0:
        raise ValueError("--deep-tail-zero-threshold는 0 이상이어야 합니다")
    for name in (
        "mask_threshold_phi", "mask_threshold_lam",
        "product_projection_floor_phi", "product_projection_floor_lam",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')}는 0 이상이어야 합니다")
    if args.ratio_floor <= 0.0:
        raise ValueError("--ratio-floor는 양수여야 합니다")
    if getattr(args, "electronic_representation", "grid") == "born_huang":
        from .propagate_born_huang import run_born_huang
        return run_born_huang(args)
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
        product_projection_floor_phi=args.product_projection_floor_phi,
        product_projection_floor_lam=args.product_projection_floor_lam,
        # 비교 RHS는 check 시점에만 명시적으로 평가한다.
        mask_residual_diagnostics=False,
        factor_stability_diagnostics=args.verbose_diagnostics,
    )
    phi, lam, chi = to_gpu_factors(phi_cpu, lam_cpu, chi_cpu, gpu_model)
    real_dtype, complex_dtype = numpy_dtypes(args.precision)

    print(f"초기 배열: Phi={phi.shape}, Lambda={lam.shape}, chi={chi.shape}")
    print(
        f"격자: dx={cpu_model.dx:.6f}, dq={cpu_model.dq:.6f}, "
        f"dR={cpu_model.dR:.6f}; hard wall "
        f"{cpu_model.x_left:.3f}..{cpu_model.x_right:.3f}; "
        f"Z_L={args.left_charge:.3f}, Z_R={args.right_charge:.3f}"
    )
    if args.symmetric_box_half_width > 0.0:
        print(f"대칭 전자 box preset: [-{args.symmetric_box_half_width:g},+{args.symmetric_box_half_width:g}]")
    if args.full_nuclear_range:
        print(
            "핵 좌표 범위 실험: q와 R 모두 전자 hard-wall 전체 범위 "
            f"[{cpu_model.x_left:.3f},{cpu_model.x_right:.3f}) 사용"
        )
    print(
        "초기 Gaussian: "
        f"q0={args.q0:.4f}, sigma_q={args.proton_sigma:.6f}; "
        f"R0={args.R0:.4f}, sigma_R={args.heavy_sigma:.6f}"
    )
    print(
        "수치 scheme: periodic 5-point central D1/D2; "
        "product-preserving gamma transfer + discrete product projection; "
        f"log_backend={cpu_model.log_derivative_backend}, "
        f"projection_backend={cpu_model.product_projection_backend}; "
        f"ratio_floor={args.ratio_floor:.1e}, "
        f"mask(Phi,Lambda)=({args.mask_threshold_phi:.1e},"
        f"{args.mask_threshold_lam:.1e}), "
        f"deep_tail_zero={args.deep_tail_zero_threshold:.1e}, "
        "product_floor(Phi,Lambda)="
        f"({args.product_projection_floor_phi:.1e},"
        f"{args.product_projection_floor_lam:.1e})"
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
    if args.mask_residual_diagnostics:
        print(
            "Mask 진단: finite-check 시점에서 support mask on/off product "
            "residual을 분해; dynamics에는 비교 RHS를 적용하지 않음"
        )
    print(
        "Probability-budget 진단: "
        +", ".join(f"{value:.0e}" for value in args.mask_probability_budgets)
        +" suppressed mass에 대응하는 eta를 저장 frame에서 역산"
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
    scheduled_nt = len(save_steps)
    # Reserve one extra frame for the latest finite check-point when a later
    # step becomes non-finite between regular save times.
    capacity = scheduled_nt+1

    # 저장 배열만 host RAM에 둔다. Production archive는 complex128이다.
    times_fs = np.empty(capacity)
    saved_step_numbers = np.empty(capacity, dtype=np.int64)
    phis = np.empty(
        (capacity, args.nx, args.nq, args.nR), dtype=complex_dtype
    )
    lams = np.empty((capacity, args.nq, args.nR), dtype=complex_dtype)
    chis = np.empty((capacity, args.nR), dtype=complex_dtype)
    avec = np.empty((capacity, args.nq, args.nR), dtype=real_dtype)
    bvec = np.empty_like(avec)
    alpha = np.empty((capacity, args.nR), dtype=real_dtype)
    eps1 = np.empty_like(avec)
    eps2 = np.empty((capacity, args.nR), dtype=real_dtype)
    theta1 = np.empty_like(avec)
    theta2 = np.empty((capacity, args.nR), dtype=real_dtype)
    norm = np.empty(capacity)
    pnc = np.empty(capacity)
    projection_correction = np.empty(capacity)
    diagnostic_history = {
        name: np.empty(capacity) for name in DIAGNOSTIC_FIELDS
    }
    mask_probability_budgets = np.asarray(
        args.mask_probability_budgets, dtype=float
    )
    mask_budget_eta_phi = np.empty(
        (capacity, len(mask_probability_budgets))
    )
    mask_budget_eta_lam = np.empty_like(mask_budget_eta_phi)
    psis = []

    def save_frame(
        frame, step, correction=0.0, interval_diagnostics=None,
        factor_state=None,
    ):
        """저장 시점에만 GPU factor/field를 CPU로 내려 동일 NPZ 형식으로 기록."""
        state_phi, state_lam, state_chi = (
            (phi, lam, chi) if factor_state is None else factor_state
        )
        fields_gpu = instantaneous_functionals(
            state_phi, state_lam, state_chi, gpu_model,
            floor=args.ratio_floor,
            mask_threshold_phi=args.mask_threshold_phi,
            mask_threshold_lam=args.mask_threshold_lam,
        )
        phi_base = cp.asnumpy(state_phi)
        lam_base = cp.asnumpy(state_lam)
        chi_base = cp.asnumpy(state_chi)
        rho_R_base = np.abs(chi_base)**2
        rho_qR_base = np.abs(lam_base)**2*rho_R_base[None, :]
        for index, budget in enumerate(mask_probability_budgets):
            mask_budget_eta_phi[frame, index] = (
                mask_threshold_for_probability_budget(rho_qR_base, budget)
            )
            mask_budget_eta_lam[frame, index] = (
                mask_threshold_for_probability_budget(rho_R_base, budget)
            )
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
        saved_step_numbers[frame] = step
        phis[frame], lams[frame], chis[frame] = phi_out, lam_out, chi_out
        avec[frame], bvec[frame] = saved["a"], saved["b"]
        alpha[frame] = saved["alpha"]
        eps1[frame], eps2[frame] = saved["epsilon_1"], saved["epsilon_2"]
        theta1[frame], theta2[frame] = th1, th2
        norm[frame] = np.sum(np.abs(psi)**2, dtype=np.float64)*(
            cpu_model.dx*cpu_model.dq*cpu_model.dR
        )
        pnc[frame] = float(
            pnc_error(state_phi, state_lam, state_chi, gpu_model).get()
        )
        projection_correction[frame] = float(
            correction.get() if hasattr(correction, "get") else correction
        )
        if args.mask_residual_diagnostics and interval_diagnostics is None:
            interval_diagnostics = evaluate_mask_residual_diagnostics(
                state_phi, state_lam, state_chi,
                gpu_model, args.ratio_floor,
                args.mask_threshold_phi, args.mask_threshold_lam,
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

    failure_reason = ""
    failure_step = -1
    nonfinite_factors = ()
    failure_nonfinite_counts = np.zeros(3, dtype=np.int64)
    failure_max_finite_abs = np.zeros(3, dtype=float)
    failure_checkpoint_saved = False
    steps_attempted = 0
    last_finite_step = 0
    last_finite_phi = phi.copy()
    last_finite_lam = lam.copy()
    last_finite_chi = chi.copy()
    last_finite_correction = cp.asarray(0.0)
    last_finite_diagnostics = merge_maxima()
    try:
        for step in range(1, n_steps+1):
            steps_attempted = step
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
            must_save = (
                frame < scheduled_nt and step == save_steps[frame]
            )
            must_check = (
                step % max(1, args.check_every) == 0
                or must_save or step == n_steps
            )
            if must_check and not all_finite(phi, lam, chi):
                failure_step = step
                nonfinite_factors = tuple(
                    name for name, factor in (
                        ("Phi", phi), ("Lambda", lam), ("chi", chi)
                    )
                    if not bool(cp.all(cp.isfinite(factor)).get())
                )
                failure_reason = (
                    f"step {step}에서 non-finite 값 검출"
                    +(
                        f" ({', '.join(nonfinite_factors)})"
                        if nonfinite_factors else ""
                    )
                )
                print(f"전파 중단 감지: {failure_reason}")
                for index, factor in enumerate((phi, lam, chi)):
                    finite = cp.isfinite(factor)
                    failure_nonfinite_counts[index] = int(
                        (factor.size-cp.count_nonzero(finite)).get()
                    )
                    failure_max_finite_abs[index] = float(cp.max(
                        cp.where(finite, cp.abs(factor), 0.0)
                    ).get())
                if (
                    last_finite_step > saved_step_numbers[frame-1]
                    and frame < capacity
                ):
                    save_frame(
                        frame, last_finite_step, last_finite_correction,
                        last_finite_diagnostics,
                        factor_state=(
                            last_finite_phi, last_finite_lam,
                            last_finite_chi,
                        ),
                    )
                    frame += 1
                    failure_checkpoint_saved = True
                    print(
                        "마지막 finite check-point 추가 저장: "
                        f"step {last_finite_step} "
                        f"(t={last_finite_step*args.dt_au/AU_PER_FS:.6f} fs)"
                    )
                break
            if must_check and args.mask_residual_diagnostics:
                mask_diagnostics = evaluate_mask_residual_diagnostics(
                    phi, lam, chi, gpu_model, args.ratio_floor,
                    args.mask_threshold_phi, args.mask_threshold_lam,
                )
                interval_diagnostics = merge_maxima(
                    interval_diagnostics, mask_diagnostics
                )
            if must_save:
                save_frame(
                    frame, step, interval_correction, interval_diagnostics
                )
                frame += 1
                interval_correction = cp.asarray(0.0)
                interval_diagnostics = merge_maxima()
            if must_check:
                last_finite_step = step
                last_finite_phi = phi.copy()
                last_finite_lam = lam.copy()
                last_finite_chi = chi.copy()
                last_finite_correction = interval_correction.copy()
                last_finite_diagnostics = {
                    name: value.copy()
                    for name, value in interval_diagnostics.items()
                }
            if step % max(1, args.progress_every) == 0 or step == n_steps:
                print(
                    f"step {step:7d}/{n_steps}  "
                    f"t={step*args.dt_au/AU_PER_FS:9.4f} fs"
                )
    except KeyboardInterrupt:
        failure_step = steps_attempted
        failure_reason = f"사용자가 step {steps_attempted}에서 계산을 중단함"
        print(f"전파 중단 감지: {failure_reason}")

    stop.record()
    stop.synchronize()
    gpu_seconds = cp.cuda.get_elapsed_time(start, stop)/1000.0
    wall_seconds = time.perf_counter()-wall_start

    completed = not failure_reason
    last_saved_step = int(saved_step_numbers[frame-1])
    # 실패한 GPU state는 archive에 섞지 않는다. 마지막으로 finite 판정을 통과해
    # host에 내려온 frame까지만 잘라서 후처리가 일반 완료 archive와 동일하게
    # 동작하도록 한다.
    nt = frame
    times_fs = times_fs[:nt]
    saved_step_numbers = saved_step_numbers[:nt]
    phis, lams, chis = phis[:nt], lams[:nt], chis[:nt]
    avec, bvec, alpha = avec[:nt], bvec[:nt], alpha[:nt]
    eps1, eps2 = eps1[:nt], eps2[:nt]
    theta1, theta2 = theta1[:nt], theta2[:nt]
    norm, pnc = norm[:nt], pnc[:nt]
    projection_correction = projection_correction[:nt]
    diagnostic_history = {
        name: values[:nt] for name, values in diagnostic_history.items()
    }
    mask_budget_eta_phi = mask_budget_eta_phi[:nt]
    mask_budget_eta_lam = mask_budget_eta_lam[:nt]

    diagnostic_history["max_abs_support_gamma_phi_dt"] = (
        diagnostic_history["max_abs_support_gamma_phi"]*args.dt_au
    )
    diagnostic_history["max_abs_support_gamma_lam_dt"] = (
        diagnostic_history["max_abs_support_gamma_lam"]*args.dt_au
    )
    for factor_name in ("phi", "lam", "chi"):
        for statistic in ("support", "weighted_rms"):
            source = (
                f"max_{statistic}_rhs_{factor_name}_after_product_projection"
            )
            diagnostic_history[source+"_dt"] = (
                diagnostic_history[source]*args.dt_au
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
    args.propagation_failed = not completed
    payload = dict(
        kind=np.array("direct_multi_component_exact_factorization"),
        representation=np.array(
            "nested_realspace_independent_harmonic_hardwall_electron"
        ),
        electronic_representation=np.array("grid"),
        local_norm_correction=np.array(
            "product_preserving_nested_tangent_correction"
        ),
        discrete_product_projection=np.array(
            args.product_projection_backend
            +"_tangent_projection_to_periodic_nuclear_D2"
        ),
        spatial_derivative=np.array("periodic_five_point_central_D1_D2"),
        ratio_regularization=np.array(
            args.log_derivative_backend
            +"_amplitude_mask_plus_deep_tail_phase_log_exact_zero"
        ),
        log_derivative_backend=np.array(args.log_derivative_backend),
        weak_log_delta=np.array(args.weak_log_delta),
        weak_log_smoothing=np.array(args.weak_log_smoothing),
        weak_log_tolerance=np.array(args.weak_log_tolerance),
        weak_log_max_iterations=np.array(args.weak_log_max_iterations),
        weak_log_preconditioner=np.array(
            "exact_diagonal" if args.weak_log_smoothing == 0.0
            else "periodic_five_point_fourier_mean_density"
        ),
        product_projection_backend=np.array(args.product_projection_backend),
        projection_tau_phi=np.array(args.projection_tau_phi),
        projection_tau_lam=np.array(args.projection_tau_lam),
        projection_tau_chi=np.array(args.projection_tau_chi),
        projection_support_epsilon=np.array(args.projection_support_epsilon),
        deep_tail_zero_threshold=np.array(args.deep_tail_zero_threshold),
        full_nuclear_range=np.array(args.full_nuclear_range),
        ratio_floor=np.array(args.ratio_floor),
        mask_threshold_phi=np.array(args.mask_threshold_phi),
        mask_threshold_lam=np.array(args.mask_threshold_lam),
        product_projection_floor_phi=np.array(
            args.product_projection_floor_phi
        ),
        product_projection_floor_lam=np.array(
            args.product_projection_floor_lam
        ),
        mask_residual_diagnostics=np.array(args.mask_residual_diagnostics),
        mask_probability_budgets=mask_probability_budgets,
        mask_budget_eta_phi=mask_budget_eta_phi,
        mask_budget_eta_lam=mask_budget_eta_lam,
        backend=np.array("cupy_single_gpu"),
        precision=np.array(args.precision),
        cuda_device=np.array(args.device),
        gauge=np.array(gauge_name),
        base_gauge=np.array("parallel_transport_two_level"),
        x=cpu_model.x, q=cpu_model.q, R=cpu_model.R, times_fs=times_fs,
        saved_steps=saved_step_numbers,
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
        propagation_completed=np.array(completed),
        requested_final_time_fs=np.array(args.t_final_fs),
        requested_steps=np.array(n_steps),
        attempted_steps=np.array(steps_attempted),
        last_saved_step=np.array(last_saved_step),
        failure_detected_step=np.array(failure_step),
        failure_reason=np.array(failure_reason),
        nonfinite_factors=np.asarray(nonfinite_factors, dtype="U16"),
        failure_nonfinite_counts=failure_nonfinite_counts,
        failure_max_finite_abs=failure_max_finite_abs,
        failure_last_finite_check_step=np.array(last_finite_step),
        failure_checkpoint_saved=np.array(failure_checkpoint_saved),
        args=np.array([vars(args)], dtype=object),
    )
    if args.save_psi:
        payload["psi"] = np.asarray(psis)
    path = outdir/"multi_component_direct_ef_gpu.npz"
    np.savez_compressed(path, **payload)

    status_path = outdir/"propagation_status.log"
    status_lines = [
        f"status={'completed' if completed else 'failed'}",
        f"archive={path}",
        f"requested_final_time_fs={args.t_final_fs:.12g}",
        f"requested_steps={n_steps}",
        f"attempted_steps={steps_attempted}",
        f"last_saved_step={last_saved_step}",
        f"last_saved_time_fs={times_fs[-1]:.12g}",
        f"failure_detected_step={failure_step}",
        f"failure_reason={failure_reason or 'none'}",
        f"nonfinite_factors={','.join(nonfinite_factors) or 'none'}",
        "failure_nonfinite_counts="
        +",".join(str(int(value)) for value in failure_nonfinite_counts),
        "failure_max_finite_abs="
        +",".join(f"{value:.12e}" for value in failure_max_finite_abs),
        f"failure_last_finite_check_step={last_finite_step}",
        f"failure_checkpoint_saved={int(failure_checkpoint_saved)}",
        f"mask_threshold_phi={args.mask_threshold_phi:.12e}",
        f"mask_threshold_lam={args.mask_threshold_lam:.12e}",
        f"product_projection_floor_phi={args.product_projection_floor_phi:.12e}",
        f"product_projection_floor_lam={args.product_projection_floor_lam:.12e}",
        f"max_norm_error={np.max(np.abs(norm-1.0)):.12e}",
        f"max_saved_pnc_error={np.max(pnc):.12e}",
        f"max_pnc_projection_correction={np.max(projection_correction):.12e}",
    ]
    for name in (
        "max_abs_support_gamma_phi_dt",
        "max_abs_support_gamma_lam_dt",
        "max_effective_product_residual_l2",
        "max_abs_full_norm_rate_after_product_projection",
        "max_relative_product_projection_l2",
        "max_relative_support_product_projection_l2",
        "deep_tail_suppressed_probability_phi",
        "deep_tail_suppressed_probability_lam",
        "deep_tail_zero_fraction_phi",
        "deep_tail_zero_fraction_lam",
        "max_outer_probability_q",
        "max_outer_probability_R",
        "max_relative_psi_wrap_mismatch_q",
        "max_relative_psi_wrap_mismatch_R",
        "max_product_residual_without_mask_l2",
        "max_product_residual_due_to_mask_l2",
        "max_relative_product_residual_without_mask",
        "max_relative_product_residual_due_to_mask",
        "max_abs_product_mask_nonmask_alignment",
        "max_product_mask_nonmask_alignment_positive",
        "max_product_mask_nonmask_alignment_negative_magnitude",
        "max_support_product_residual_without_mask_l2",
        "max_support_product_residual_due_to_mask_l2",
        "max_relative_support_product_residual_without_mask",
        "max_relative_support_product_residual_due_to_mask",
        "max_support_pnc_phi_projection_load",
        "max_support_pnc_lam_projection_load",
        "max_weighted_rms_pnc_phi_projection_load",
        "max_weighted_rms_pnc_lam_projection_load",
        "max_support_rhs_phi_after_product_projection_dt",
        "max_support_rhs_lam_after_product_projection_dt",
        "max_support_rhs_chi_after_product_projection_dt",
        "max_weighted_rms_rhs_phi_after_product_projection_dt",
        "max_weighted_rms_rhs_lam_after_product_projection_dt",
        "max_weighted_rms_rhs_chi_after_product_projection_dt",
        "max_rk_stage_amplification_phi",
        "max_rk_stage_amplification_lam",
        "max_rk_stage_amplification_chi",
    ):
        if name in diagnostic_history:
            status_lines.append(
                f"{name}={np.max(diagnostic_history[name]):.12e}"
            )
    for index, budget in enumerate(mask_probability_budgets):
        status_lines.extend((
            f"mask_budget_{budget:.0e}_eta_phi_min="
            f"{np.min(mask_budget_eta_phi[:, index]):.12e}",
            f"mask_budget_{budget:.0e}_eta_phi_max="
            f"{np.max(mask_budget_eta_phi[:, index]):.12e}",
            f"mask_budget_{budget:.0e}_eta_lam_min="
            f"{np.min(mask_budget_eta_lam[:, index]):.12e}",
            f"mask_budget_{budget:.0e}_eta_lam_max="
            f"{np.max(mask_budget_eta_lam[:, index]):.12e}",
        ))
    status_path.write_text("\n".join(status_lines)+"\n", encoding="utf-8")

    used = cp.get_default_memory_pool().used_bytes()/(1024**3)
    total = cp.get_default_memory_pool().total_bytes()/(1024**3)
    print(f"{'저장 완료' if completed else '부분 저장 완료'}: {path}")
    print(f"전파 상태 로그: {status_path}")
    print(f"GPU event 시간: {gpu_seconds:.3f} s; wall 시간: {wall_seconds:.3f} s")
    if steps_attempted:
        print(f"평균 wall 시간: {wall_seconds/steps_attempted:.6f} s/step")
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
        "max_relative_product_projection_l2",
        "max_relative_support_product_projection_l2",
        "deep_tail_suppressed_probability_phi",
        "deep_tail_suppressed_probability_lam",
        "deep_tail_zero_fraction_phi",
        "deep_tail_zero_fraction_lam",
        "max_outer_probability_q",
        "max_outer_probability_R",
        "max_relative_psi_wrap_mismatch_q",
        "max_relative_psi_wrap_mismatch_R",
        "max_product_residual_without_mask_l2",
        "max_product_residual_due_to_mask_l2",
        "max_relative_product_residual_without_mask",
        "max_relative_product_residual_due_to_mask",
        "max_abs_product_mask_nonmask_alignment",
        "max_product_mask_nonmask_alignment_positive",
        "max_product_mask_nonmask_alignment_negative_magnitude",
        "max_support_product_residual_without_mask_l2",
        "max_support_product_residual_due_to_mask_l2",
        "max_relative_support_product_residual_without_mask",
        "max_relative_support_product_residual_due_to_mask",
        "max_support_pnc_phi_projection_load",
        "max_support_pnc_lam_projection_load",
        "max_weighted_rms_pnc_phi_projection_load",
        "max_weighted_rms_pnc_lam_projection_load",
        "max_support_rhs_phi_after_product_projection_dt",
        "max_support_rhs_lam_after_product_projection_dt",
        "max_support_rhs_chi_after_product_projection_dt",
        "max_weighted_rms_rhs_phi_after_product_projection_dt",
        "max_weighted_rms_rhs_lam_after_product_projection_dt",
        "max_weighted_rms_rhs_chi_after_product_projection_dt",
        "max_rk_stage_amplification_phi",
        "max_rk_stage_amplification_lam",
        "max_rk_stage_amplification_chi",
    ):
        if name in diagnostic_history:
            print(f"  {name}: {np.max(diagnostic_history[name]):.3e}")
    print("Probability-budget eta 범위(저장 frame):")
    for index, budget in enumerate(mask_probability_budgets):
        print(
            f"  budget={budget:.1e}: "
            f"eta_phi={np.min(mask_budget_eta_phi[:, index]):.3e}.."
            f"{np.max(mask_budget_eta_phi[:, index]):.3e}, "
            f"eta_lam={np.min(mask_budget_eta_lam[:, index]):.3e}.."
            f"{np.max(mask_budget_eta_lam[:, index]):.3e}"
        )
    if getattr(args, "verbose_diagnostics", False):
        print("전체 개발용 진단:")
        for name, values in diagnostic_history.items():
            print(f"  {name}: {np.max(values):.3e}")
    if not completed:
        print(
            f"주의: 요청한 {args.t_final_fs:g} fs 중 마지막 정상 저장 시각 "
            f"{times_fs[-1]:.6f} fs까지만 포함된 partial trajectory입니다."
        )
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
    electronic = parser.add_argument_group("conditional electronic representation")
    electronic.add_argument(
        "--electronic-representation", choices=("grid", "born_huang"),
        default="grid",
        help="Phi를 x-grid 또는 electronic-only Born--Huang coefficients로 전파",
    )
    electronic.add_argument(
        "--bo-states", type=int, default=6,
        help="Born--Huang backend에서 유지할 local BO state 수",
    )
    electronic.add_argument(
        "--bo-save-basis-states", action="store_true",
        help="큰 static BO eigenvector tensor도 archive에 저장",
    )
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
        "--log-derivative-backend", choices=("pointwise", "weak"),
        default="pointwise",
        help="amplitude logarithmic derivative 계산 방식",
    )
    regularization.add_argument(
        "--weak-log-delta", type=float, default=1.0e-10,
        help="weak mass operator의 dimensionless positive floor",
    )
    regularization.add_argument(
        "--weak-log-smoothing", type=float, default=0.04,
        help="weak log-amplitude Tikhonov smoothing length (bohr)",
    )
    regularization.add_argument(
        "--weak-log-tolerance", type=float, default=1.0e-8,
        help="batched weak PCG relative residual tolerance",
    )
    regularization.add_argument(
        "--weak-log-max-iterations", type=int, default=80,
        help="Fourier-preconditioned batched weak PCG 최대 iteration",
    )
    regularization.add_argument(
        "--product-projection-backend",
        choices=("nested_inverse", "weighted_tikhonov"),
        default="nested_inverse",
        help="discrete product residual의 factor tangent 분배 방식",
    )
    regularization.add_argument(
        "--projection-tau-phi", type=float, default=1.0e-10,
        help="weighted projection의 electronic tangent ridge",
    )
    regularization.add_argument(
        "--projection-tau-lam", type=float, default=1.0e-10,
        help="weighted projection의 proton tangent ridge",
    )
    regularization.add_argument(
        "--projection-tau-chi", type=float, default=1.0e-10,
        help="weighted projection의 heavy tangent ridge",
    )
    regularization.add_argument(
        "--projection-support-epsilon", type=float, default=1.0e-12,
        help="inverse-support penalty의 denominator floor",
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
        "--product-projection-floor-phi", type=float, default=1.0e-10,
        help="product projection의 1/(Lambda*chi) numerical support floor",
    )
    regularization.add_argument(
        "--product-projection-floor-lam", type=float, default=1.0e-10,
        help="product projection의 1/chi numerical support floor",
    )
    regularization.add_argument(
        "--no-mask-residual-diagnostics", dest="mask_residual_diagnostics",
        action="store_false",
        help="check 시점의 support-mask on/off residual 분해를 생략",
    )
    regularization.add_argument(
        "--mask-probability-budgets", type=probability_budgets,
        default=(1.0e-9, 1.0e-8, 1.0e-7),
        metavar="B1,B2,...",
        help=(
            "저장 frame에서 동일 suppressed mass를 만드는 eta를 역산할 "
            "diagnostic budget 목록"
        ),
    )
    parser.set_defaults(mask_residual_diagnostics=True)
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


def _execute(args):
    """GPU 전파 배열이 해제된 뒤 선택적으로 CPU 렌더링을 실행한다."""
    path = run(args)
    if getattr(args, "render_after", False):
        from multi_component_exact_factorization.render_all import (
            render_completed_run,
        )
        label = "부분 결과" if getattr(args, "propagation_failed", False) else "전체 결과"
        print(f"계산 종료 후 {label} 렌더링을 시작합니다.")
        try:
            render_completed_run(path, fast=getattr(args, "render_fast", False))
        except Exception as error:
            if not getattr(args, "propagation_failed", False):
                raise
            print(f"부분 결과 렌더링 실패: {type(error).__name__}: {error}")
    if getattr(args, "propagation_failed", False):
        print(
            "전파는 수치 오류로 완료되지 않았습니다. partial archive/report를 "
            "진단한 뒤 dt, grid와 안정성 지표를 확인하세요."
        )
        raise SystemExit(2)
    return path


def main(args=None):
    """Run programmatically, or tee a CLI run into its dated result folder."""
    if args is not None:
        return _execute(args)

    args = parse_args()
    outdir = dated_results_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir/"propagation.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        terminal_out, terminal_err = sys.stdout, sys.stderr
        with redirect_stdout(_Tee(terminal_out, log)), redirect_stderr(
            _Tee(terminal_err, log)
        ):
            print(f"명령: {shlex.join(sys.argv)}")
            print(f"전체 실행 로그: {log_path}")
            try:
                return _execute(args)
            except SystemExit:
                raise
            except BaseException:
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
