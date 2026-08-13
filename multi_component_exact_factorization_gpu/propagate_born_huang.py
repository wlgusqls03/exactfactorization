"""Single-GPU driver for the electronic-only Born--Huang MCEF backend."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from result_paths import dated_results_dir

from multi_component_exact_factorization.born_huang import (
    initial_born_huang_factors,
    load_or_build_born_huang_basis,
)
from multi_component_exact_factorization.core import (
    AU_PER_FS, build_model, calibrate_flat_top_args,
    fixed_center_crossing_probabilities,
)
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
    cache_dir = (
        args.bo_basis_cache_dir if getattr(args, "bo_basis_cache", True)
        else None
    )
    basis_cpu, cache_info = load_or_build_born_huang_basis(
        cpu_model, n_states, cache_dir=cache_dir,
        rebuild=getattr(args, "rebuild_bo_basis_cache", False),
    )
    if cache_info["enabled"]:
        state = "HIT: 재사용" if cache_info["hit"] else "MISS: 생성 후 저장"
        print(
            f"BO basis cache {state}; {cache_info['seconds']:.2f} s; "
            f"stored/requested={cache_info['stored_states']}/{n_states}; "
            f"key={cache_info['key'][:16]}; path={cache_info['path']}"
        )
    else:
        print(f"BO basis cache 비활성; 생성 {cache_info['seconds']:.2f} s")
    coefficients_cpu, lam_cpu, chi_cpu = initial_born_huang_factors(
        cpu_model, args, basis_cpu
    )
    coefficient_norm2 = np.sum(np.abs(coefficients_cpu)**2, axis=0)
    rho_qR_initial = coefficient_norm2*np.abs(
        lam_cpu*chi_cpu[None, :]
    )**2
    rho_R_initial = np.sum(rho_qR_initial, axis=0)*cpu_model.dq
    calibrate_flat_top_args(args, rho_qR_initial, rho_R_initial)
    cpu_model.coupling_mask_backend = args.coupling_mask_backend
    cpu_model.flat_top_on_phi = float(args.flat_top_on_phi or 0.0)
    cpu_model.flat_top_on_lam = float(args.flat_top_on_lam or 0.0)
    cpu_model.flat_top_transition_decades = args.flat_top_transition_decades
    optimization = getattr(args, "gpu_optimization", "fused")
    configure_fused_periodic_derivative(optimization == "fused")
    model = make_gpu_model(
        cpu_model, "double",
        reuse_stage_derivatives=(optimization != "baseline"),
        product_projection_floor_phi=args.product_projection_floor_phi,
        product_projection_floor_lam=args.product_projection_floor_lam,
    )
    link_kernel = getattr(args, "bo_link_kernel", "reference")
    basis = to_gpu_basis(basis_cpu, model, link_kernel)
    print(
        "BO 핵 미분: overlap-link periodic 5-point "
        "(D1 anti-Hermitian, D2 Hermitian by construction); "
        f"GPU kernel={link_kernel}"
    )
    saved_basis_states = (
        basis_cpu.states if getattr(args, "bo_save_basis_states", False) else None
    )
    compact_basis_states = None
    if getattr(args, "bo_save_electron_density", True):
        # The clamped electronic Hamiltonian and phase-aligned eigenstates are
        # real.  Keeping only real64 cuts persistent host storage in half
        # relative to the original complex128 basis.  Contraction below is
        # R-blocked, so no full Phi(x,q,R) temporary is formed.
        compact_basis_states = (
            basis_cpu.states.real
            if saved_basis_states is not None
            else np.array(basis_cpu.states.real, dtype=np.float64, copy=True)
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
        bo_state_density_q=[], bo_state_density_R=[],
        electron_density=[],
        outer_probability_q=[], outer_probability_R=[],
        fixed_center_crossing_q_left=[], fixed_center_crossing_q_right=[],
        fixed_center_crossing_q=[], fixed_center_crossing_R_left=[],
        fixed_center_crossing_R_right=[], fixed_center_crossing_R=[],
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
        "suppressed_probability_phi", "suppressed_probability_lam",
        "max_weak_log_residual_q_xi", "max_weak_log_residual_R_xi",
        "max_weak_log_residual_R_chi", "max_weak_log_iterations",
        "max_weak_log_unconverged_lines",
        "deep_tail_suppressed_probability_phi",
        "deep_tail_suppressed_probability_lam",
        "deep_tail_zero_fraction_phi", "deep_tail_zero_fraction_lam",
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
        state_joint = np.abs(c_out)**2*joint[None, :, :]
        q_density = np.sum(state_joint, axis=(0, 2), dtype=np.float64)*cpu_model.dR
        R_density = np.sum(state_joint, axis=(0, 1), dtype=np.float64)*cpu_model.dq
        q_edge = min(5, len(q_density)//2)
        R_edge = min(5, len(R_density)//2)
        histories["outer_probability_q"].append(
            (np.sum(q_density[:q_edge])+np.sum(q_density[-q_edge:]))
            *cpu_model.dq/max(norm, 1.0e-300)
        )
        histories["outer_probability_R"].append(
            (np.sum(R_density[:R_edge])+np.sum(R_density[-R_edge:]))
            *cpu_model.dR/max(norm, 1.0e-300)
        )
        q_cross = fixed_center_crossing_probabilities(
            q_density, cpu_model.q, cpu_model.dq,
            args.left_position, args.right_position, norm,
        )
        R_cross = fixed_center_crossing_probabilities(
            R_density, cpu_model.R, cpu_model.dR,
            args.left_position, args.right_position, norm,
        )
        for suffix, value in zip(("left", "right", ""), q_cross):
            key = "fixed_center_crossing_q" + (f"_{suffix}" if suffix else "")
            histories[key].append(value)
        for suffix, value in zip(("left", "right", ""), R_cross):
            key = "fixed_center_crossing_R" + (f"_{suffix}" if suffix else "")
            histories[key].append(value)
        histories["bo_populations"].append(
            np.sum(state_joint, axis=(1, 2), dtype=np.float64)
            *cpu_model.dq*cpu_model.dR
        )
        # Paper-style BO-surface plots need |F_n|^2 resolved along the two
        # nuclear coordinates.  Saving these reductions costs only
        # O(nt*N_BO*(nq+nR)), unlike saving phi_n(x,q,R).
        histories["bo_state_density_q"].append(
            np.sum(state_joint, axis=2, dtype=np.float64)*cpu_model.dR
        )
        histories["bo_state_density_R"].append(
            np.sum(state_joint, axis=1, dtype=np.float64)*cpu_model.dq
        )
        if compact_basis_states is not None:
            electron_density = np.zeros(len(cpu_model.x), dtype=np.float64)
            block = 32
            for start_R in range(0, len(cpu_model.R), block):
                stop_R = min(start_R+block, len(cpu_model.R))
                phi_block = np.einsum(
                    "nqR,nxqR->xqR",
                    c_out[:, :, start_R:stop_R],
                    compact_basis_states[:, :, :, start_R:stop_R],
                    optimize=True,
                )
                electron_density += np.sum(
                    np.abs(phi_block)**2*joint[None, :, start_R:stop_R],
                    axis=(1, 2), dtype=np.float64,
                )*cpu_model.dq*cpu_model.dR
            histories["electron_density"].append(electron_density)
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
    saved_frame_count = 1
    last_saved_step = 0
    last_completed_step = 0
    interval = {}
    interval_correction = cp.asarray(0.0)
    failure = ""
    attempted = 0
    start = time.perf_counter()
    throttle_limit = float(getattr(args, "gpu_util_limit", 100.0))
    throttle_every = max(1, int(getattr(args, "gpu_throttle_every", 20)))
    throttle_chunk_start = None
    throttle_sleep_seconds = 0.0
    try:
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
            last_completed_step = step
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
                max_norm_drift = float(getattr(args, "max_norm_drift", 1.0e-3))
                if max_norm_drift > 0.0:
                    coefficient_norm2 = cp.sum(
                        cp.real(coefficients*cp.conj(coefficients)), axis=0,
                        dtype=model.reduction_real_dtype,
                    )
                    xi = lam*chi[None, :]
                    current_norm = cp.sum(
                        coefficient_norm2*cp.real(xi*cp.conj(xi)),
                        dtype=model.reduction_real_dtype,
                    )*model.dq*model.dR
                    norm_drift = float(cp.abs(current_norm-1.0).get())
                    if norm_drift > max_norm_drift:
                        failure = (
                            f"step {step}에서 |norm-1|={norm_drift:.3e}가 "
                            f"허용값 {max_norm_drift:.3e} 초과"
                        )
                        print(f"전파 중단 감지: {failure}")
                        save(step, interval_correction, interval)
                        saved_frame_count += 1
                        last_saved_step = step
                        print(
                            "norm-drift finite check-point 추가 저장: "
                            f"step {step} "
                            f"(t={step*args.dt_au/AU_PER_FS:.6f} fs)"
                        )
                        break
            if must_save:
                save(step, interval_correction, interval)
                saved_frame_count += 1
                last_saved_step = step
                frame += 1
                interval, interval_correction = {}, cp.asarray(0.0)
            if step % args.progress_every == 0 or step == n_steps:
                print(f"step {step:7d}/{n_steps}  t={step*args.dt_au/AU_PER_FS:9.4f} fs")
    except KeyboardInterrupt:
        failure = f"사용자가 step {attempted}에서 계산을 중단함"
        print(f"전파 중단 감지: {failure}")
        # A signal during a scheduled save may have appended only part of a
        # frame.  Roll every history back to the last fully completed frame.
        for values in histories.values():
            del values[saved_frame_count:]
        for values in diagnostics.values():
            del values[saved_frame_count:]
        if (
            last_completed_step > last_saved_step
            and all_finite(coefficients, lam, chi)
        ):
            save(last_completed_step, interval_correction, interval)
            saved_frame_count += 1
            last_saved_step = last_completed_step
            print(
                "마지막 완료 step 추가 저장: "
                f"step {last_completed_step} "
                f"(t={last_completed_step*args.dt_au/AU_PER_FS:.6f} fs)"
            )

    completed = not failure
    interrupted = failure.startswith("사용자가 step ")
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
        bo_basis_cache_hit=np.array(cache_info["hit"]),
        bo_basis_cache_key=np.array(cache_info["key"]),
        bo_basis_cache_path=np.array(cache_info["path"]),
        bo_basis_cache_stored_states=np.array(
            cache_info.get("stored_states", n_states)
        ),
        bo_basis_cache_seconds=np.array(cache_info["seconds"]),
        bo_derivative_backend=np.array("overlap_link_five_point"),
        bo_link_kernel=np.array(link_kernel),
        bo_link_kernel_version=np.array(1),
        bo_overlap_links_in_cache=np.array(True),
        bo_energies=basis_cpu.energies,
        x=cpu_model.x, q=cpu_model.q, R=cpu_model.R,
        log_derivative_backend=np.array(args.log_derivative_backend),
        product_projection_backend=np.array(args.product_projection_backend),
        coupling_mask_backend=np.array(args.coupling_mask_backend),
        flat_top_on_phi=np.array(args.flat_top_on_phi or 0.0),
        flat_top_on_lam=np.array(args.flat_top_on_lam or 0.0),
        flat_top_transition_decades=np.array(args.flat_top_transition_decades),
        flat_top_budget_phi=np.array(args.flat_top_budget_phi),
        flat_top_budget_lam=np.array(args.flat_top_budget_lam),
        weak_log_delta=np.array(args.weak_log_delta),
        weak_log_smoothing=np.array(args.weak_log_smoothing),
        weak_log_preconditioner=np.array(
            "exact_diagonal" if args.weak_log_smoothing == 0.0
            else "periodic_five_point_fourier_mean_density"
        ),
        projection_tau_phi=np.array(args.projection_tau_phi),
        projection_tau_lam=np.array(args.projection_tau_lam),
        projection_tau_chi=np.array(args.projection_tau_chi),
        deep_tail_zero_threshold=np.array(args.deep_tail_zero_threshold),
        full_nuclear_range=np.array(args.full_nuclear_range),
        propagation_completed=np.array(completed),
        propagation_interrupted=np.array(interrupted),
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
    status_name = "completed" if completed else ("interrupted" if interrupted else "failed")
    status.write_text(
        f"status={status_name}\n"
        f"archive={path}\nlast_saved_time_fs={payload['times_fs'][-1]:.12g}\n"
        f"failure_reason={failure or 'none'}\n",
        encoding="utf-8",
    )
    print(f"{'저장 완료' if completed else '부분 저장 완료'}: {path}")
    print(f"BO coefficient 전파 wall 시간: {payload['wall_seconds']:.3f} s")
    if attempted:
        print(
            "BO coefficient 평균 wall 시간: "
            f"{float(payload['wall_seconds'])/attempted:.6f} s/step"
        )
    pool = cp.get_default_memory_pool()
    print(
        "CuPy memory pool: "
        f"used={pool.used_bytes()/1024**3:.2f} GiB, "
        f"reserved={pool.total_bytes()/1024**3:.2f} GiB"
    )
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
    print(
        "BO 경계 진단: max outer probability (q,R)="
        f"({np.max(payload['outer_probability_q']):.3e}, "
        f"{np.max(payload['outer_probability_R']):.3e}); "
        "max beyond fixed centers (q,R)="
        f"({np.max(payload['fixed_center_crossing_q']):.3e}, "
        f"{np.max(payload['fixed_center_crossing_R']):.3e})"
    )
    args.propagation_failed = not completed
    return path
