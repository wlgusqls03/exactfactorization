#!/usr/bin/env python3
"""Propagate the discretize-first MCEF equations on one CUDA GPU.

The driver reuses the extended 1D Shin--Metiu model and immutable
Born--Huang cache of the existing solver.  Its evolution equations are a
separate implementation derived from the spatially discrete Hamiltonian.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import shlex
import signal
import sys
import time
import traceback

import numpy as np

from result_paths import dated_results_dir
from multi_component_exact_factorization.born_huang import (
    initial_born_huang_factors,
    load_or_build_born_huang_basis,
)
from multi_component_exact_factorization.core import (
    AU_PER_FS,
    add_model_arguments,
    build_model,
    calibrate_flat_top_args,
    fixed_center_crossing_probabilities,
)
from multi_component_exact_factorization_gpu.gpu_born_huang import (
    PNC_NORM_DIAGNOSTIC_NAMES,
    to_gpu_basis,
)
from multi_component_exact_factorization_gpu.gpu_core import (
    all_finite,
    cp,
)

from .gpu_core import (
    discrete_rhs_gpu,
    full_step_discrete_bh,
    make_discrete_gpu_model,
)
from .checkpoint import (
    load_checkpoint,
    validate_state_shapes,
    write_checkpoint_atomic,
)


DIAGNOSTIC_NAMES = (
    "max_raw_horizontal_phi",
    "max_raw_horizontal_lam",
    "max_raw_pnc_phi_error",
    "max_raw_pnc_lam_error",
    "suppressed_probability_phi",
    "suppressed_probability_lam",
    "recombination_residual_l2",
    "predicted_mask_residual_l2",
    "unexplained_residual_l2",
    "relative_unexplained_residual",
    "direct_action_l2",
    "recombined_rhs_l2",
    "max_abs_regularized_F_ratio",
    "max_abs_regularized_chi_ratio",
    "weighted_link_defect_phi_q",
    "weighted_link_defect_phi_R",
    "weighted_link_defect_gamma_R",
    "epsilon_1_imaginary_defect",
    "epsilon_2_imaginary_defect",
    "full_norm_rate",
    "mask_transition_fraction_phi",
    "mask_transition_fraction_lam",
    "rk_product_local_defect_l2",
    "rk_product_local_defect_relative",
    "pnc_product_change_l2",
    "rk_product_increment_l2",
)+PNC_NORM_DIAGNOSTIC_NAMES


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _as_float(value):
    return float(value.get() if hasattr(value, "get") else value)


def _pnc_errors(coefficients, lam, model):
    c_norm2 = cp.sum(
        cp.real(coefficients*cp.conj(coefficients)), axis=0,
        dtype=model.reduction_real_dtype,
    )
    lam_norm2 = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    return cp.max(cp.abs(c_norm2-1.0)), cp.max(cp.abs(lam_norm2-1.0))


def _checkpoint_metadata(args, cpu_model, n_states, cache_info):
    """Return every invariant needed for a mathematically identical resume."""
    cache_key = str(cache_info.get("key", ""))
    if not cache_key:
        raise ValueError(
            "checkpoint/resume requires the immutable BO basis cache; "
            "remove --no-bo-basis-cache"
        )
    return {
        "formulation": "discretize_first_overlap_link_v1",
        "time_integrator": "classical_rk4_product_preserving_pnc_retraction",
        "dt_au": float(args.dt_au),
        "bo_states": int(n_states),
        "bo_basis_cache_key": cache_key,
        "bo_link_kernel": str(args.bo_link_kernel),
        "nx": int(len(cpu_model.x)),
        "nq": int(len(cpu_model.q)),
        "nR": int(len(cpu_model.R)),
        "dx": float(cpu_model.dx),
        "dq": float(cpu_model.dq),
        "dR": float(cpu_model.dR),
        "x_first": float(cpu_model.x[0]),
        "x_last": float(cpu_model.x[-1]),
        "q_first": float(cpu_model.q[0]),
        "q_last": float(cpu_model.q[-1]),
        "R_first": float(cpu_model.R[0]),
        "R_last": float(cpu_model.R[-1]),
        "proton_mass": float(cpu_model.proton_mass),
        "heavy_mass": float(cpu_model.heavy_mass),
        "flat_top_on_phi": float(cpu_model.flat_top_on_phi),
        "flat_top_on_lam": float(cpu_model.flat_top_on_lam),
        "flat_top_transition_decades": float(
            cpu_model.flat_top_transition_decades
        ),
        "deep_tail_zero_threshold": float(args.deep_tail_zero_threshold),
    }


def _checkpoint_path(args, outdir):
    if args.checkpoint_file:
        return Path(args.checkpoint_file).expanduser().resolve()
    if args.resume_from:
        return Path(args.resume_from).expanduser().resolve()
    return (outdir/"discrete_mcef_checkpoint.npz").resolve()


def _install_termination_handlers(args):
    """Turn HUP/TERM into a safe request handled after the active RK4 step."""
    previous = {}

    def request_termination(signum, _frame):
        # Signal handlers execute between Python bytecodes.  Only set a flag;
        # the propagation loop commits the whole RK4 result before stopping.
        args.termination_signal = int(signum)

    for name in ("SIGHUP", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_termination)
    return previous


def _restore_termination_handlers(previous):
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run(args):
    outdir = dated_results_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cp.cuda.Device(args.device).use()
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    gpu_name = properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")
    print(f"GPU {args.device}: {gpu_name}; precision=complex128/float64")

    cpu_model = build_model(args)
    n_states = int(args.bo_states)
    if n_states <= int(args.electron_excitation):
        raise ValueError("--bo-states must exceed --electron-excitation")
    cache_dir = args.bo_basis_cache_dir if args.bo_basis_cache else None
    print(f"Discrete Born--Huang basis 준비: N_BO={n_states}")
    basis_cpu, cache_info = load_or_build_born_huang_basis(
        cpu_model, n_states, cache_dir=cache_dir,
        rebuild=args.rebuild_bo_basis_cache,
    )
    if cache_info["enabled"]:
        state = "HIT" if cache_info["hit"] else "MISS/build"
        print(
            f"BO cache {state}: {cache_info['seconds']:.2f} s; "
            f"stored/requested={cache_info['stored_states']}/{n_states}; "
            f"path={cache_info['path']}"
        )

    c_cpu, lam_cpu, chi_cpu = initial_born_huang_factors(
        cpu_model, args, basis_cpu
    )
    c_norm2 = np.sum(np.abs(c_cpu)**2, axis=0)
    rho_qR = c_norm2*np.abs(lam_cpu*chi_cpu[None, :])**2
    rho_R = np.sum(rho_qR, axis=0)*cpu_model.dq
    args.coupling_mask_backend = "flat_top"
    calibrate_flat_top_args(args, rho_qR, rho_R)
    cpu_model.coupling_mask_backend = "flat_top"
    cpu_model.flat_top_on_phi = float(args.flat_top_on_phi or 0.0)
    cpu_model.flat_top_on_lam = float(args.flat_top_on_lam or 0.0)
    cpu_model.flat_top_transition_decades = args.flat_top_transition_decades
    pnc_upper = 10.0*args.deep_tail_zero_threshold
    phi_off = cpu_model.flat_top_on_phi*10.0**(-args.flat_top_transition_decades)
    lam_off = cpu_model.flat_top_on_lam*10.0**(-args.flat_top_transition_decades)
    positive_off = [value for value in (phi_off, lam_off) if value > 0.0]
    if positive_off and pnc_upper > min(positive_off):
        print(
            "주의: PNC gate가 ratio-mask active transition과 겹칩니다. "
            "deep-tail threshold를 낮추거나 flat-top transition을 확인하세요: "
            f"PNC upper={pnc_upper:.3e}, min ratio off={min(positive_off):.3e}"
        )
    else:
        print(
            "PNC/mask support ordering 확인: ratio가 active인 곳에서는 "
            "support-aware PNC gate가 1입니다."
        )

    checkpoint_enabled = bool(args.checkpoint_every > 0 or args.resume_from)
    checkpoint_metadata = None
    checkpoint_path = None
    resume_step = 0
    if checkpoint_enabled:
        checkpoint_metadata = _checkpoint_metadata(
            args, cpu_model, n_states, cache_info
        )
        checkpoint_path = _checkpoint_path(args, outdir)
    if args.resume_from:
        resumed = load_checkpoint(
            args.resume_from, expected_metadata=checkpoint_metadata
        )
        validate_state_shapes(
            resumed,
            coefficients_shape=c_cpu.shape,
            lam_shape=lam_cpu.shape,
            chi_shape=chi_cpu.shape,
        )
        c_cpu = resumed["electronic_coefficients"]
        lam_cpu = resumed["lambda_wavefunction"]
        chi_cpu = resumed["chi"]
        resume_step = int(resumed["completed_step"])
        print(
            "checkpoint 재시작: "
            f"{resumed['path']}; completed step={resume_step}, "
            f"t={resume_step*args.dt_au/AU_PER_FS:.9f} fs"
        )

    model = make_discrete_gpu_model(cpu_model)
    basis = to_gpu_basis(basis_cpu, model, args.bo_link_kernel)
    if args.bo_link_kernel != "fused":
        print("주의: reference BO link kernel은 검증용이며 production보다 느립니다.")
    compact_states = None
    if args.bo_save_electron_density:
        # Keep the cache-backed mmap instead of copying the potentially
        # multi-GiB BO tensor into RAM.  Saved-frame reconstruction is
        # R-blocked and the OS page cache provides reuse across frames.
        compact_states = basis_cpu.states
    saved_basis_states = basis_cpu.states if args.bo_save_basis_states else None
    if saved_basis_states is None:
        basis_cpu.states = np.empty((0,), dtype=float)

    coefficients = cp.ascontiguousarray(cp.asarray(c_cpu, dtype=cp.complex128))
    lam = cp.asarray(lam_cpu, dtype=cp.complex128)
    chi = cp.asarray(chi_cpu, dtype=cp.complex128)
    print(
        f"동적 배열: C={coefficients.shape}, Lambda={lam.shape}, chi={chi.shape}; "
        "discrete matrix/link coupling, no spatial product rule"
    )

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    if resume_step > n_steps:
        raise ValueError(
            f"checkpoint step {resume_step} exceeds requested final step {n_steps}"
        )
    args.save_every = args.save_every or max(1, int(np.ceil(max(n_steps, 1)/200)))
    args.progress_every = (
        args.progress_every or max(1, int(np.ceil(max(n_steps, 1)/20)))
    )
    args.check_every = (
        args.check_every or max(1, int(np.ceil(max(n_steps, 1)/500)))
    )
    save_steps = list(range(0, n_steps+1, args.save_every))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    print(
        f"step={n_steps}, dt={args.dt_au:g} au, target={args.t_final_fs:g} fs; "
        f"save/check/progress={args.save_every}/{args.check_every}/"
        f"{args.progress_every}"
    )
    if args.step_sleep_ms > 0.0:
        print(
            "GPU thermal throttle: 각 완료 step 뒤 stream synchronize + "
            f"sleep {args.step_sleep_ms:g} ms"
        )
    if checkpoint_enabled:
        if args.checkpoint_every > 0:
            print(
                "원자적 state checkpoint: "
                f"every {args.checkpoint_every} global steps; {checkpoint_path}"
            )
        else:
            print(
                "주의: checkpoint에서 재시작했지만 새 periodic checkpoint는 "
                "비활성입니다(--checkpoint-every=0)."
            )
        print(
            "checkpoint는 C/Lambda/chi의 완료-step 상태만 저장하며, "
            "재시작 archive는 그 시각부터 시작합니다."
        )

    histories = {
        "times_fs": [], "electronic_coefficients": [],
        "lambda_wavefunction": [], "chi": [],
        "epsilon_1": [], "epsilon_2": [], "a": [], "b": [], "alpha": [],
        "sphi_q1_magnitude": [], "sphi_R1_magnitude": [],
        "sgamma_R1_magnitude": [],
        "norm": [], "pnc_error": [], "pnc_projection_correction": [],
        "outer_probability_q": [], "outer_probability_R": [],
        "fixed_center_crossing_q_left": [],
        "fixed_center_crossing_q_right": [],
        "fixed_center_crossing_q": [],
        "fixed_center_crossing_R_left": [],
        "fixed_center_crossing_R_right": [],
        "fixed_center_crossing_R": [],
        "bo_populations": [], "bo_state_density_q": [],
        "bo_state_density_R": [], "electron_density": [],
    }
    diagnostics = {name: [] for name in DIAGNOSTIC_NAMES}
    checkpoint_writes = 0
    checkpoint_seconds = 0.0

    def checkpoint_state(step):
        nonlocal checkpoint_writes, checkpoint_seconds
        if checkpoint_path is None:
            return
        checkpoint_started = time.perf_counter()
        write_checkpoint_atomic(
            checkpoint_path,
            completed_step=step,
            coefficients=coefficients,
            lam=lam,
            chi=chi,
            metadata=checkpoint_metadata,
        )
        elapsed = time.perf_counter()-checkpoint_started
        checkpoint_writes += 1
        checkpoint_seconds += elapsed
        size_mib = checkpoint_path.stat().st_size/1024**2
        print(
            "checkpoint 저장: "
            f"step {step}, t={step*args.dt_au/AU_PER_FS:.6f} fs; "
            f"{size_mib:.1f} MiB, {elapsed:.3f} s; {checkpoint_path}"
        )

    def save(step, correction=0.0, step_diagnostics=None):
        evaluated = discrete_rhs_gpu(
            coefficients, lam, chi, model, basis,
            collect_diagnostics=True,
        )
        c_out = cp.asnumpy(coefficients)
        l_out = cp.asnumpy(lam)
        h_out = cp.asnumpy(chi)
        epsilon_1 = cp.asnumpy(evaluated.fields["epsilon_1"])
        epsilon_2 = cp.asnumpy(evaluated.fields["epsilon_2"])
        sphi_q1 = cp.asnumpy(evaluated.fields["sphi_q1"])
        sphi_R1 = cp.asnumpy(evaluated.fields["sphi_R1"])
        sgamma_R1 = cp.asnumpy(evaluated.fields["sgamma_R1"])
        joint = np.abs(l_out)**2*np.abs(h_out[None, :])**2
        state_joint = np.abs(c_out)**2*joint[None, :, :]
        total_norm = (
            np.sum(state_joint, dtype=np.float64)*cpu_model.dq*cpu_model.dR
        )
        pnc_c, pnc_l = _pnc_errors(coefficients, lam, model)
        histories["times_fs"].append(step*args.dt_au/AU_PER_FS)
        histories["electronic_coefficients"].append(c_out)
        histories["lambda_wavefunction"].append(l_out)
        histories["chi"].append(h_out)
        histories["epsilon_1"].append(epsilon_1)
        histories["epsilon_2"].append(epsilon_2)
        histories["a"].append(np.angle(sphi_q1)/cpu_model.dq)
        histories["b"].append(np.angle(sphi_R1)/cpu_model.dR)
        histories["alpha"].append(np.angle(sgamma_R1)/cpu_model.dR)
        histories["sphi_q1_magnitude"].append(np.abs(sphi_q1))
        histories["sphi_R1_magnitude"].append(np.abs(sphi_R1))
        histories["sgamma_R1_magnitude"].append(np.abs(sgamma_R1))
        histories["norm"].append(total_norm)
        histories["pnc_error"].append(max(_as_float(pnc_c), _as_float(pnc_l)))
        histories["pnc_projection_correction"].append(_as_float(correction))
        q_density = np.sum(state_joint, axis=(0, 2), dtype=np.float64)*cpu_model.dR
        R_density = np.sum(state_joint, axis=(0, 1), dtype=np.float64)*cpu_model.dq
        q_edge = min(5, len(q_density)//2)
        R_edge = min(5, len(R_density)//2)
        histories["outer_probability_q"].append(
            (np.sum(q_density[:q_edge])+np.sum(q_density[-q_edge:]))
            *cpu_model.dq/max(total_norm, 1.0e-300)
        )
        histories["outer_probability_R"].append(
            (np.sum(R_density[:R_edge])+np.sum(R_density[-R_edge:]))
            *cpu_model.dR/max(total_norm, 1.0e-300)
        )
        q_cross = fixed_center_crossing_probabilities(
            q_density, cpu_model.q, cpu_model.dq,
            args.left_position, args.right_position, total_norm,
        )
        R_cross = fixed_center_crossing_probabilities(
            R_density, cpu_model.R, cpu_model.dR,
            args.left_position, args.right_position, total_norm,
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
        histories["bo_state_density_q"].append(
            np.sum(state_joint, axis=2, dtype=np.float64)*cpu_model.dR
        )
        histories["bo_state_density_R"].append(
            np.sum(state_joint, axis=1, dtype=np.float64)*cpu_model.dq
        )
        if compact_states is not None:
            electron_density = np.zeros(len(cpu_model.x), dtype=np.float64)
            block = 24
            for start in range(0, len(cpu_model.R), block):
                stop = min(start+block, len(cpu_model.R))
                phi_block = np.einsum(
                    "nqR,nxqR->xqR", c_out[:, :, start:stop],
                    compact_states[:, :, :, start:stop], optimize=True,
                )
                electron_density += np.sum(
                    np.abs(phi_block)**2*joint[None, :, start:stop],
                    axis=(1, 2), dtype=np.float64,
                )*cpu_model.dq*cpu_model.dR
            histories["electron_density"].append(electron_density)
        for name in DIAGNOSTIC_NAMES:
            if name in evaluated.diagnostics:
                value = evaluated.diagnostics[name]
            elif step_diagnostics is not None and name in step_diagnostics:
                value = step_diagnostics[name]
            else:
                value = 0.0 if step == resume_step else np.nan
            diagnostics[name].append(_as_float(value))

    save(resume_step)
    future_save_steps = [step for step in save_steps if step > resume_step]
    next_frame = 0
    saved_frame_count = 1
    last_saved_step = resume_step
    last_completed_step = resume_step
    failure = ""
    attempted = resume_step
    correction_peak = cp.asarray(0.0)
    throttle_sleep_seconds = 0.0
    throttled_steps = 0
    started = time.perf_counter()
    last_step_diagnostics = None
    interruption_requested = False
    steps_executed = 0
    committed_state = (
        resume_step, coefficients, lam, chi, correction_peak,
        last_step_diagnostics, steps_executed,
    )
    try:
        for step in range(resume_step+1, n_steps+1):
            attempted = step
            will_save = (
                next_frame < len(future_save_steps)
                and step == future_save_steps[next_frame]
            )
            step_result = full_step_discrete_bh(
                coefficients, lam, chi, args.dt_au, model, basis,
                collect_step_diagnostics=will_save,
            )
            coefficients, lam, chi, correction, step_diagnostics = step_result
            correction_peak = cp.maximum(correction_peak, correction)
            last_step_diagnostics = step_diagnostics
            last_completed_step = step
            steps_executed += 1
            # One reference assignment is the commit point. If Ctrl+C lands
            # during the preceding stores/bookkeeping, the exception path
            # restores the preceding whole state instead of a mixed step.
            committed_state = (
                step, coefficients, lam, chi, correction_peak,
                last_step_diagnostics, steps_executed,
            )
            must_save = will_save
            must_checkpoint = (
                args.checkpoint_every > 0
                and (step % args.checkpoint_every == 0 or step == n_steps)
            )
            must_check = (
                step % args.check_every == 0
                or must_save or must_checkpoint or step == n_steps
            )
            if must_check:
                if not all_finite(coefficients, lam, chi):
                    failure = f"step {step}: non-finite C/Lambda/chi"
                    print(f"전파 중단: {failure}")
                    break
                c_norm2_gpu = cp.sum(
                    cp.real(coefficients*cp.conj(coefficients)), axis=0,
                    dtype=model.reduction_real_dtype,
                )
                F_gpu = lam*chi[None, :]
                current_norm = cp.sum(
                    c_norm2_gpu*cp.real(F_gpu*cp.conj(F_gpu)),
                    dtype=model.reduction_real_dtype,
                )*model.dq*model.dR
                drift = _as_float(cp.abs(current_norm-1.0))
                if args.max_norm_drift > 0.0 and drift > args.max_norm_drift:
                    failure = (
                        f"step {step}: |norm-1|={drift:.3e} exceeds "
                        f"{args.max_norm_drift:.3e}"
                    )
                    print(f"전파 중단: {failure}")
                    save(step, correction_peak, step_diagnostics)
                    saved_frame_count += 1
                    last_saved_step = step
                    break
            if must_save:
                save(step, correction_peak, step_diagnostics)
                saved_frame_count += 1
                last_saved_step = step
                correction_peak = cp.asarray(0.0)
                next_frame += 1
            if must_checkpoint:
                checkpoint_state(step)
            if step % args.progress_every == 0 or step == n_steps:
                print(
                    f"step {step:7d}/{n_steps}  "
                    f"t={step*args.dt_au/AU_PER_FS:9.4f} fs"
                )
            requested_signal = getattr(args, "termination_signal", None)
            if requested_signal is not None:
                signal_name = signal.Signals(requested_signal).name
                failure = (
                    f"{signal_name} 요청을 받아 완료된 step {step}에서 중단함"
                )
                interruption_requested = True
                print(f"전파 중단: {failure}")
                break
            if args.step_sleep_ms > 0.0:
                cp.cuda.get_current_stream().synchronize()
                sleep_started = time.perf_counter()
                time.sleep(args.step_sleep_ms/1000.0)
                throttle_sleep_seconds += time.perf_counter()-sleep_started
                throttled_steps += 1
    except KeyboardInterrupt:
        (
            last_completed_step, coefficients, lam, chi, correction_peak,
            last_step_diagnostics, steps_executed,
        ) = committed_state
        failure = f"사용자가 step {attempted}에서 계산을 중단함"
        interruption_requested = True
        print(f"전파 중단: {failure}")

    if interruption_requested:
        for values in histories.values():
            del values[saved_frame_count:]
        for values in diagnostics.values():
            del values[saved_frame_count:]
        if (
            last_completed_step > last_saved_step
            and all_finite(coefficients, lam, chi)
        ):
            save(last_completed_step, correction_peak, last_step_diagnostics)
            saved_frame_count += 1
            last_saved_step = last_completed_step
            print(
                "마지막 완료 step 추가 저장: "
                f"step {last_completed_step} "
                f"(t={last_completed_step*args.dt_au/AU_PER_FS:.6f} fs)"
            )
        if checkpoint_path is not None and all_finite(coefficients, lam, chi):
            checkpoint_state(last_completed_step)

    cp.cuda.get_current_stream().synchronize()
    wall_seconds = time.perf_counter()-started
    completed = not failure
    interrupted = interruption_requested
    payload = {key: np.asarray(value) for key, value in histories.items()}
    payload.update({key: np.asarray(value) for key, value in diagnostics.items()})
    # Compatibility aliases are diagnostics only; no product projection is
    # performed by this solver.
    payload["max_abs_support_gamma_phi_dt"] = (
        payload["max_raw_horizontal_phi"]*args.dt_au
    )
    payload["max_abs_support_gamma_lam_dt"] = (
        payload["max_raw_horizontal_lam"]*args.dt_au
    )
    payload["max_effective_product_residual_l2"] = payload[
        "unexplained_residual_l2"
    ]
    payload["max_abs_full_norm_rate_after_product_projection"] = payload[
        "full_norm_rate"
    ]
    payload.update(
        kind=np.array("discrete_born_huang_multi_component_exact_factorization"),
        representation=np.array("discrete_electronic_born_huang_coefficients"),
        electronic_representation=np.array("born_huang"),
        discrete_formulation_version=np.array(1),
        spatial_formulation=np.array("discretize_first_overlap_link"),
        time_integrator=np.array("classical_rk4_product_preserving_pnc_retraction"),
        product_projection_backend=np.array("none_by_construction"),
        ratio_regularization=np.array("probability_budget_flat_top_mass_inverse"),
        horizontal_correction=np.array("product_preserving_parallel_transport"),
        pnc_projection_backend=np.array("support_aware_product_preserving"),
        bo_states_count=np.array(n_states),
        bo_energies=np.asarray(basis_cpu.energies),
        bo_basis_cache_hit=np.array(cache_info["hit"]),
        bo_basis_cache_key=np.array(cache_info["key"]),
        bo_basis_cache_path=np.array(cache_info["path"]),
        bo_basis_cache_seconds=np.array(cache_info["seconds"]),
        bo_link_kernel=np.array(args.bo_link_kernel),
        flat_top_on_phi=np.array(cpu_model.flat_top_on_phi),
        flat_top_on_lam=np.array(cpu_model.flat_top_on_lam),
        flat_top_transition_decades=np.array(args.flat_top_transition_decades),
        flat_top_budget_phi=np.array(args.flat_top_budget_phi),
        flat_top_budget_lam=np.array(args.flat_top_budget_lam),
        deep_tail_zero_threshold=np.array(args.deep_tail_zero_threshold),
        x=cpu_model.x, q=cpu_model.q, R=cpu_model.R,
        propagation_completed=np.array(completed),
        propagation_interrupted=np.array(interrupted),
        requested_final_time_fs=np.array(args.t_final_fs),
        requested_steps=np.array(n_steps), attempted_steps=np.array(attempted),
        segment_start_step=np.array(resume_step),
        segment_start_time_fs=np.array(resume_step*args.dt_au/AU_PER_FS),
        resume_from=np.array(str(args.resume_from or "")),
        checkpoint_every=np.array(args.checkpoint_every),
        checkpoint_path=np.array(str(checkpoint_path or "")),
        checkpoint_writes=np.array(checkpoint_writes),
        checkpoint_seconds=np.array(checkpoint_seconds),
        step_sleep_ms=np.array(args.step_sleep_ms),
        throttle_sleep_seconds=np.array(throttle_sleep_seconds),
        throttled_steps=np.array(throttled_steps),
        failure_reason=np.array(failure), wall_seconds=np.array(wall_seconds),
        args=np.array([vars(args)], dtype=object),
    )
    if saved_basis_states is not None:
        payload["bo_basis_states"] = saved_basis_states
    archive = outdir/"multi_component_born_huang_ef_gpu.npz"
    np.savez_compressed(archive, **payload)
    status = outdir/"propagation_status.log"
    status_name = "completed" if completed else ("interrupted" if interrupted else "failed")
    status.write_text(
        f"status={status_name}\n"
        f"archive={archive}\n"
        f"segment_start_step={resume_step}\n"
        f"last_saved_time_fs={payload['times_fs'][-1]:.12g}\n"
        f"checkpoint={checkpoint_path or 'disabled'}\n"
        f"failure_reason={failure or 'none'}\n",
        encoding="utf-8",
    )
    pool = cp.get_default_memory_pool()
    print(f"{'저장 완료' if completed else '부분 저장 완료'}: {archive}")
    print(
        f"wall={wall_seconds:.3f} s; "
        f"{wall_seconds/max(steps_executed, 1):.6f} s/executed-step; "
        f"GPU pool used/reserved={pool.used_bytes()/1024**3:.2f}/"
        f"{pool.total_bytes()/1024**3:.2f} GiB"
    )
    if checkpoint_writes:
        print(
            "checkpoint overhead: "
            f"{checkpoint_seconds:.3f} s/{checkpoint_writes} writes "
            f"({checkpoint_seconds/max(wall_seconds, 1.0e-300):.3%} wall)"
        )
    if throttled_steps:
        active_wall = max(0.0, wall_seconds-throttle_sleep_seconds)
        print(
            "의도적 GPU 휴식: "
            f"{throttle_sleep_seconds:.3f} s/{throttled_steps} steps; "
            f"sleep 제외 wall={active_wall:.3f} s; "
            f"관측 active fraction={active_wall/max(wall_seconds, 1.0e-300):.3f}"
        )
    print("Discrete MCEF 핵심 진단:")
    print(f"  max |norm-1|: {np.max(np.abs(payload['norm']-1.0)):.3e}")
    print(f"  max saved PNC error: {np.max(payload['pnc_error']):.3e}")
    print(
        "  max outer probability (q,R): "
        f"({np.max(payload['outer_probability_q']):.3e}, "
        f"{np.max(payload['outer_probability_R']):.3e})"
    )
    print(
        "  max probability beyond fixed centers (q,R): "
        f"({np.max(payload['fixed_center_crossing_q']):.3e}, "
        f"{np.max(payload['fixed_center_crossing_R']):.3e})"
    )
    print(
        "  max PNC retraction load: "
        f"{np.max(payload['pnc_projection_correction']):.3e}"
    )
    for name in (
        "relative_unexplained_residual", "recombination_residual_l2",
        "predicted_mask_residual_l2", "suppressed_probability_phi",
        "suppressed_probability_lam", "max_raw_horizontal_phi",
        "max_raw_horizontal_lam", "full_norm_rate",
        "rk_product_local_defect_relative", "pnc_product_change_l2",
    ):
        print(f"  max {name}: {np.max(np.abs(payload[name])):.3e}")
    print(
        "  max highest-state population: "
        f"{np.max(payload['bo_populations'][:, -1]):.3e}"
    )
    args.propagation_failed = not completed
    return archive


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default="results/discrete_mcef_gpu",
        help="results/YYYYMMDD 아래의 run folder",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dt-au", type=float, default=0.025)
    parser.add_argument("--t-final-fs", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--check-every", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help=(
            "완료된 global step 기준 atomic state-checkpoint 간격; "
            "0이면 비활성"
        ),
    )
    parser.add_argument(
        "--checkpoint-file", default=None,
        help=(
            "최신 checkpoint를 원자적으로 덮어쓸 경로; 기본값은 "
            "run folder/discrete_mcef_checkpoint.npz"
        ),
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="이 solver가 저장한 state checkpoint에서 global step 재개",
    )
    parser.add_argument(
        "--step-sleep-ms", type=float, default=0.0,
        help=(
            "각 완료 step의 모든 GPU 작업을 동기화한 뒤 쉬는 시간(ms); "
            "0이면 기존 비동기 실행 경로 유지"
        ),
    )
    parser.add_argument("--max-norm-drift", type=float, default=1.0e-4)
    parser.add_argument("--bo-states", type=int, default=10)
    parser.add_argument(
        "--bo-link-kernel", choices=("reference", "fused"), default="fused"
    )
    parser.set_defaults(bo_basis_cache=True)
    parser.add_argument(
        "--bo-basis-cache-dir", default="results/bo_basis_cache"
    )
    parser.add_argument(
        "--no-bo-basis-cache", action="store_false", dest="bo_basis_cache"
    )
    parser.add_argument("--rebuild-bo-basis-cache", action="store_true")
    parser.add_argument("--bo-save-basis-states", action="store_true")
    parser.set_defaults(bo_save_electron_density=True)
    parser.add_argument(
        "--no-bo-save-electron-density", action="store_false",
        dest="bo_save_electron_density",
    )
    mask = parser.add_argument_group("flat-top mass inverse")
    mask.add_argument("--flat-top-budget-phi", type=float, default=1.0e-10)
    mask.add_argument("--flat-top-budget-lam", type=float, default=1.0e-10)
    mask.add_argument("--flat-top-on-phi", type=float, default=None)
    mask.add_argument("--flat-top-on-lam", type=float, default=None)
    mask.add_argument("--flat-top-transition-decades", type=float, default=3.0)
    parser.set_defaults(render_after=False)
    parser.add_argument("--render-after", action="store_true")
    parser.add_argument("--no-render-after", action="store_false", dest="render_after")
    parser.add_argument("--render-fast", action="store_true")
    parser.add_argument("--verbose-diagnostics", action="store_true")
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    if args.dt_au <= 0.0 or args.t_final_fs < 0.0:
        parser.error("dt must be positive and final time nonnegative")
    if not np.isfinite(args.step_sleep_ms) or args.step_sleep_ms < 0.0:
        parser.error("--step-sleep-ms must be a finite nonnegative number")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    if args.checkpoint_file and not (args.checkpoint_every or args.resume_from):
        parser.error(
            "--checkpoint-file requires --checkpoint-every or --resume-from"
        )
    if not 0.0 <= args.flat_top_budget_phi < 1.0:
        parser.error("flat-top phi budget must be in [0,1)")
    if not 0.0 <= args.flat_top_budget_lam < 1.0:
        parser.error("flat-top lambda budget must be in [0,1)")
    return args


def _execute(args):
    args.termination_signal = None
    previous_handlers = _install_termination_handlers(args)
    try:
        archive = run(args)
        if args.render_after:
            from multi_component_exact_factorization.render_all import (
                render_completed_run,
            )
            render_completed_run(archive, fast=args.render_fast)
        if getattr(args, "propagation_failed", False):
            raise SystemExit(2)
        return archive
    finally:
        _restore_termination_handlers(previous_handlers)


def main(args=None):
    if args is not None:
        return _execute(args)
    args = parse_args()
    outdir = dated_results_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir/"propagation.log"
    log_mode = "a" if args.resume_from else "w"
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(
            _Tee(sys.stderr, log)
        ):
            if args.resume_from:
                print("\n===== checkpoint resume invocation =====")
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
