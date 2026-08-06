"""Single-GPU driver for the electronic-only Born--Huang MCEF backend."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from result_paths import dated_results_dir

from multi_component_exact_factorization.born_huang import (
    build_born_huang_basis,
    initial_born_huang_factors,
)
from multi_component_exact_factorization.core import AU_PER_FS, build_model
from multi_component_exact_factorization.propagate import output_gauge

from .gpu_born_huang import (
    full_step_bh,
    instantaneous_functionals_bh,
    pnc_project_coefficients,
    to_gpu_basis,
)
from .gpu_core import (
    all_finite,
    configure_fused_periodic_derivative,
    cp,
    make_gpu_model,
)
from .throttle import throttle_delay


def run_born_huang(args):
    """Build the BO/NAC tensors once, then propagate only C/Lambda/chi."""
    outdir = dated_results_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cp.cuda.Device(args.device).use()
    cpu_model = build_model(args)
    n_states = int(args.bo_states)
    if n_states <= int(args.electron_excitation):
        raise ValueError("--bo-states must exceed --electron-excitation")
    print(f"Born--Huang basis 생성: N_BO={n_states}")
    basis_cpu = build_born_huang_basis(cpu_model, n_states)
    coefficients_cpu, lam_cpu, chi_cpu = initial_born_huang_factors(
        cpu_model, args, basis_cpu
    )
    optimization = getattr(args, "gpu_optimization", "fused")
    configure_fused_periodic_derivative(optimization == "fused")
    model = make_gpu_model(
        cpu_model, "double",
        reuse_stage_derivatives=(optimization != "baseline"),
        product_projection_floor_phi=args.product_projection_floor_phi,
        product_projection_floor_lam=args.product_projection_floor_lam,
    )
    basis = to_gpu_basis(basis_cpu, model)
    saved_basis_states = (
        basis_cpu.states if getattr(args, "bo_save_basis_states", False) else None
    )
    if saved_basis_states is None:
        # The x-grid eigenvectors are not used in the time loop.  Releasing
        # them keeps the BO backend's host-memory footprint independent of nt.
        basis_cpu.states = np.empty((0,), dtype=complex)
    coefficients = cp.asarray(coefficients_cpu, dtype=model.complex_dtype)
    lam = cp.asarray(lam_cpu, dtype=model.complex_dtype)
    chi = cp.asarray(chi_cpu, dtype=model.complex_dtype)
    print(
        f"동적 배열: C={coefficients.shape}, Lambda={lam.shape}, chi={chi.shape}; "
        f"direct Phi 대비 원소비={coefficients.size/(args.nx*args.nq*args.nR):.4f}"
    )

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    args.save_every = args.save_every or max(1, int(np.ceil(max(n_steps, 1)/200)))
    args.progress_every = args.progress_every or max(1, int(np.ceil(max(n_steps, 1)/20)))
    args.check_every = args.check_every or max(1, int(np.ceil(max(n_steps, 1)/500)))
    save_steps = list(range(0, n_steps+1, args.save_every))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)

    histories = dict(
        times_fs=[], electronic_coefficients=[], lambda_wavefunction=[], chi=[],
        a=[], b=[], alpha=[], epsilon_1=[], epsilon_2=[], norm=[], pnc_error=[],
        pnc_projection_correction=[], bo_populations=[],
    )
    diagnostic_names = (
        "max_product_residual_l2", "max_effective_product_residual_l2",
        "max_relative_product_projection_l2",
        "max_abs_product_correction_phi", "max_abs_product_correction_lam",
        "max_abs_product_correction_chi",
        "max_inverse_support_product_correction_phi",
        "max_inverse_support_product_correction_lam",
        "max_inverse_support_product_correction_chi",
        "max_abs_full_norm_rate_before_product_projection",
        "max_abs_full_norm_rate_after_product_projection",
        "max_abs_gamma_phi", "max_abs_gamma_lam",
        "max_abs_support_gamma_phi", "max_abs_support_gamma_lam",
        "max_raw_logamp_phi", "max_effective_logamp_phi",
        "max_weak_log_residual_q_xi", "max_weak_log_residual_R_xi",
        "max_weak_log_residual_R_chi", "max_weak_log_iterations",
        "max_weak_log_unconverged_lines",
    )
    diagnostics = {name: [] for name in diagnostic_names}

    def save(step, correction=0.0, interval=None):
        fields = instantaneous_functionals_bh(
            coefficients, lam, chi, model, basis, args.ratio_floor,
            args.mask_threshold_phi, args.mask_threshold_lam,
        )
        c_cpu = cp.asnumpy(coefficients)
        l_cpu = cp.asnumpy(lam)
        h_cpu = cp.asnumpy(chi)
        fields_cpu = {
            key: cp.asnumpy(fields[key])
            for key in ("a", "b", "alpha", "epsilon_1", "epsilon_2")
        }
        c_out, l_out, h_out, transformed, _, _ = output_gauge(
            c_cpu, l_cpu, h_cpu, fields_cpu, step*args.dt_au, cpu_model, args
        )
        joint = np.abs(l_out)**2*np.abs(h_out[None, :])**2
        norm = np.sum(
            np.sum(np.abs(c_out)**2, axis=0)*joint,
            dtype=np.float64,
        )*cpu_model.dq*cpu_model.dR
        histories["times_fs"].append(step*args.dt_au/AU_PER_FS)
        histories["electronic_coefficients"].append(c_out)
        histories["lambda_wavefunction"].append(l_out)
        histories["chi"].append(h_out)
        for key in ("a", "b", "alpha", "epsilon_1", "epsilon_2"):
            histories[key].append(transformed[key])
        histories["norm"].append(norm)
        histories["bo_populations"].append(
            np.sum(
                np.abs(c_out)**2*joint[None, :, :], axis=(1, 2),
                dtype=np.float64,
            )*cpu_model.dq*cpu_model.dR
        )
        _, _, _, error = pnc_project_coefficients(
            coefficients, lam, chi, model
        )
        histories["pnc_error"].append(float(error.get()))
        histories["pnc_projection_correction"].append(float(
            correction.get() if hasattr(correction, "get") else correction
        ))
        interval = interval or {}
        for name in diagnostic_names:
            value = interval.get(name, 0.0)
            diagnostics[name].append(float(
                value.get() if hasattr(value, "get") else value
            ))

    save(0)
    frame = 1
    interval = {}
    interval_correction = cp.asarray(0.0)
    failure = ""
    attempted = 0
    start = time.perf_counter()
    throttle_limit = float(getattr(args, "gpu_util_limit", 100.0))
    throttle_every = max(1, int(getattr(args, "gpu_throttle_every", 20)))
    throttle_chunk_start = None
    throttle_sleep_seconds = 0.0
    for step in range(1, n_steps+1):
        attempted = step
        if throttle_limit < 100.0 and throttle_chunk_start is None:
            throttle_chunk_start = time.perf_counter()
        coefficients, lam, chi, correction, step_diag = full_step_bh(
            coefficients, lam, chi, args.dt_au, model, basis,
            args.ratio_floor, args.mask_threshold_phi, args.mask_threshold_lam,
        )
        interval_correction = cp.maximum(interval_correction, correction)
        for name, value in step_diag.items():
            interval[name] = cp.maximum(interval.get(name, 0.0), value)
        if throttle_limit < 100.0 and (
            step % throttle_every == 0 or step == n_steps
        ):
            cp.cuda.get_current_stream().synchronize()
            active = time.perf_counter()-throttle_chunk_start
            delay = throttle_delay(active, throttle_limit)
            if delay > 0.0:
                time.sleep(delay)
                throttle_sleep_seconds += delay
            throttle_chunk_start = None
        must_save = frame < len(save_steps) and step == save_steps[frame]
        if step % args.check_every == 0 or must_save or step == n_steps:
            if not all_finite(coefficients, lam, chi):
                failure = f"step {step}에서 non-finite C/Lambda/chi 검출"
                print(f"전파 중단 감지: {failure}")
                break
        if must_save:
            save(step, interval_correction, interval)
            frame += 1
            interval, interval_correction = {}, cp.asarray(0.0)
        if step % args.progress_every == 0 or step == n_steps:
            print(f"step {step:7d}/{n_steps}  t={step*args.dt_au/AU_PER_FS:9.4f} fs")

    completed = not failure
    payload = {
        key: np.asarray(value) for key, value in histories.items()
    }
    payload.update({key: np.asarray(value) for key, value in diagnostics.items()})
    payload["max_abs_support_gamma_phi_dt"] = (
        payload["max_abs_support_gamma_phi"]*args.dt_au
    )
    payload["max_abs_support_gamma_lam_dt"] = (
        payload["max_abs_support_gamma_lam"]*args.dt_au
    )
    payload.update(
        kind=np.array("born_huang_multi_component_exact_factorization"),
        representation=np.array("electronic_born_huang_coefficients"),
        electronic_representation=np.array("born_huang"),
        bo_states_count=np.array(n_states),
        bo_energies=basis_cpu.energies,
        bo_d_q=basis_cpu.d_q,
        bo_D_q=basis_cpu.D_q,
        bo_d_R=basis_cpu.d_R,
        bo_D_R=basis_cpu.D_R,
        x=cpu_model.x, q=cpu_model.q, R=cpu_model.R,
        log_derivative_backend=np.array(args.log_derivative_backend),
        product_projection_backend=np.array(args.product_projection_backend),
        weak_log_delta=np.array(args.weak_log_delta),
        weak_log_smoothing=np.array(args.weak_log_smoothing),
        weak_log_preconditioner=np.array(
            "exact_diagonal" if args.weak_log_smoothing == 0.0
            else "periodic_five_point_fourier_mean_density"
        ),
        projection_tau_phi=np.array(args.projection_tau_phi),
        projection_tau_lam=np.array(args.projection_tau_lam),
        projection_tau_chi=np.array(args.projection_tau_chi),
        propagation_completed=np.array(completed),
        requested_final_time_fs=np.array(args.t_final_fs),
        requested_steps=np.array(n_steps), attempted_steps=np.array(attempted),
        failure_reason=np.array(failure),
        wall_seconds=np.array(time.perf_counter()-start),
        gpu_util_limit=np.array(throttle_limit),
        gpu_throttle_sleep_seconds=np.array(throttle_sleep_seconds),
        args=np.array([vars(args)], dtype=object),
    )
    if saved_basis_states is not None:
        payload["bo_basis_states"] = saved_basis_states
    path = outdir/"multi_component_born_huang_ef_gpu.npz"
    np.savez_compressed(path, **payload)
    status = outdir/"propagation_status.log"
    status.write_text(
        f"status={'completed' if completed else 'failed'}\n"
        f"archive={path}\nlast_saved_time_fs={payload['times_fs'][-1]:.12g}\n"
        f"failure_reason={failure or 'none'}\n",
        encoding="utf-8",
    )
    print(f"저장 완료: {path}")
    print(f"BO coefficient 전파 wall 시간: {payload['wall_seconds']:.3f} s")
    print(
        "BO 핵심 진단: "
        f"max|norm-1|={np.max(np.abs(payload['norm']-1.0)):.3e}, "
        "support gamma*dt="
        f"({np.max(payload['max_abs_support_gamma_phi_dt']):.3e}, "
        f"{np.max(payload['max_abs_support_gamma_lam_dt']):.3e}), "
        "effective product residual="
        f"{np.max(payload['max_effective_product_residual_l2']):.3e}, "
        "max highest-state population="
        f"{np.max(payload['bo_populations'][:, -1]):.3e}"
    )
    args.propagation_failed = not completed
    return path
