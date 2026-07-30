#!/usr/bin/env python3
"""한 NVIDIA GPU에서 multi-component direct EF를 시간 전파한다.

초기 local BO 상태는 CPU에서 한 번 만들고, 실제 time loop의 Phi/Lambda/chi,
미분, coupling, RK4, hard-wall 전자 FFT는 모두 GPU에 유지한다. 저장 frame만
CPU로 복사하므로 기존 visualization/analysis 프로그램이 같은 NPZ를 읽는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from multi_component_exact_factorization.core import (
    AU_PER_FS,
    add_model_arguments,
    build_model,
    initial_factors,
)
from multi_component_exact_factorization.propagate import output_gauge

from .gpu_core import (
    all_finite,
    cp,
    full_step,
    instantaneous_functionals,
    make_gpu_model,
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
    cp.cuda.Device(args.device).use()
    print(device_description(args.device))
    print(f"계산 정밀도: {args.precision}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Local BO diagonalization은 CPU SciPy가 효율적이고 처음 한 번만 필요하다.
    cpu_model = build_model(args)
    phi_cpu, lam_cpu, chi_cpu = initial_factors(cpu_model, args)
    gpu_model = make_gpu_model(cpu_model, args.precision)
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

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    save_steps = list(range(0, n_steps+1, max(1, args.save_every)))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    nt = len(save_steps)

    # 저장 배열만 host RAM에 둔다. Native single/mixed archive는 저장량도 절반이다.
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
    psis = []

    def save_frame(frame, step, correction=0.0):
        """저장 시점에만 GPU factor/field를 CPU로 내려 동일 NPZ 형식으로 기록."""
        fields_gpu = instantaneous_functionals(
            phi, lam, chi, gpu_model, floor=args.density_threshold
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
        if args.save_psi:
            psis.append(psi.copy())

    save_frame(0, 0)
    frame = 1
    last_correction = cp.asarray(0.0)
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    wall_start = time.perf_counter()
    start.record()

    for step in range(1, n_steps+1):
        phi, lam, chi, last_correction = full_step(
            phi, lam, chi, args.dt_au, gpu_model, args.density_threshold
        )
        must_save = frame < nt and step == save_steps[frame]
        must_check = (
            step % max(1, args.check_every) == 0 or must_save or step == n_steps
        )
        if must_check and not all_finite(phi, lam, chi):
            raise FloatingPointError(
                f"step {step}에서 non-finite 값이 발생했습니다. "
                "dt를 줄이거나 density-threshold를 키우세요."
            )
        if must_save:
            save_frame(frame, step, last_correction)
            frame += 1
        if step % max(1, args.progress_every) == 0 or step == n_steps:
            print(
                f"step {step:7d}/{n_steps}  "
                f"t={step*args.dt_au/AU_PER_FS:9.4f} fs"
            )

    stop.record()
    stop.synchronize()
    gpu_seconds = cp.cuda.get_elapsed_time(start, stop)/1000.0
    wall_seconds = time.perf_counter()-wall_start

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
        gpu_seconds=np.array(gpu_seconds), wall_seconds=np.array(wall_seconds),
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
    print(f"CuPy memory pool: used={used:.2f} GiB, reserved={total:.2f} GiB")
    print(f"최대 norm 오차: {np.max(np.abs(norm-1.0)):.3e}")
    print(f"최대 저장 PNC 오차:       {np.max(pnc):.3e}")
    print(f"최대 PNC projection 보정: {np.max(projection_correction):.3e}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default="results/multi_component_exact_factorization/gpu"
    )
    parser.add_argument("--device", type=int, default=0, help="사용할 CUDA GPU index")
    parser.add_argument(
        "--precision", choices=("double", "single", "mixed"), default="mixed",
        help="mixed는 큰 배열 FP32, inner-product/norm reduction FP64",
    )
    parser.add_argument("--dt-au", type=float, default=0.005)
    parser.add_argument("--t-final-fs", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--check-every", type=int, default=100,
        help="non-finite GPU 검사를 수행할 step 간격; 저장 frame에서는 항상 검사",
    )
    parser.add_argument("--density-threshold", type=float, default=1.0e-9)
    parser.add_argument("--save-psi", action="store_true")

    gauge = parser.add_argument_group("두 단계 gauge")
    gauge.add_argument("--theta1-q-gradient", type=float, default=0.0)
    gauge.add_argument("--theta1-R-gradient", type=float, default=0.0)
    gauge.add_argument("--theta1-frequency", type=float, default=0.0)
    gauge.add_argument("--theta2-R-gradient", type=float, default=0.0)
    gauge.add_argument("--theta2-frequency", type=float, default=0.0)
    add_model_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
