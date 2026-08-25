"""Full-grid Feit--Fleck--Steiger-style spectral TDSE propagation on CUDA."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from multi_component_exact_factorization.born_huang import (
    initial_born_huang_factors,
    load_or_build_born_huang_basis,
)
from multi_component_exact_factorization.core import (
    AU_PER_FS,
    crossing_reference_positions,
    fixed_center_crossing_probabilities,
    print_model_geometry,
)
from multi_component_exact_factorization_gpu.gpu_core import cp

from .spectral_tdse import (
    SpectralTDSEGPU,
    initialize_full_wavefunction_gpu,
    project_full_wavefunction_to_bo,
)


def _scalar(value):
    return float(value.get() if hasattr(value, "get") else value)


def _append_crossing(histories, prefix, density, grid, spacing, references):
    crossing = fixed_center_crossing_probabilities(
        density, grid, spacing, *references,
    )
    for suffix, value in zip(("left", "right", ""), crossing):
        key = prefix+(f"_{suffix}" if suffix else "")
        histories[key].append(value)


def run_split_operator(args, cpu_model, outdir: Path):
    """Run the independent full-TDSE reference and save BO projections."""
    print_model_geometry(cpu_model, args)
    n_states = int(args.bo_states)
    if n_states <= int(args.electron_excitation):
        raise ValueError("--bo-states must exceed --electron-excitation")
    print(
        "TDSE propagation: full Psi(x,q,R), second-order "
        "V/2-T-V/2 spectral split operator"
    )
    print(
        "spatial kinetic: x=Dirichlet DST-I continuum modes; "
        "q,R=periodic FFT continuum modes; no 5-point stencil in TDSE"
    )
    print(
        f"BO states={n_states}: initial state and saved-frame analysis only; "
        "the propagated full Psi is not BO-truncated"
    )
    print(
        "initial/analysis BO basis: shared Dirichlet 5-point basis, so all "
        "three solvers start from the identical Psi(0); only TDSE propagation "
        "uses the independent spectral Hamiltonian"
    )
    cache_dir = args.bo_basis_cache_dir if args.bo_basis_cache else None
    basis, cache_info = load_or_build_born_huang_basis(
        cpu_model, n_states, cache_dir=cache_dir,
        rebuild=args.rebuild_bo_basis_cache,
    )
    if cache_info["enabled"]:
        state = "HIT" if cache_info["hit"] else "MISS/build"
        print(
            f"BO analysis cache {state}: {cache_info['seconds']:.2f} s; "
            f"stored/requested={cache_info['stored_states']}/{n_states}; "
            f"path={cache_info['path']}"
        )

    coefficients, lam, chi = initial_born_huang_factors(
        cpu_model, args, basis
    )
    del coefficients
    marginal = lam*chi[None, :]
    wavefunction = initialize_full_wavefunction_gpu(
        basis, int(args.electron_excitation), marginal,
        block_R=args.tdse_projection_R_block,
    )
    del lam, chi, marginal
    solver = SpectralTDSEGPU(
        cpu_model,
        q_block_R=args.tdse_q_fft_R_block,
        R_block_x=args.tdse_R_fft_x_block,
        x_block_R=args.tdse_x_dst_R_block,
    )
    print(
        f"dynamic full-grid array: Psi={wavefunction.shape}, "
        f"{wavefunction.nbytes/1024**3:.3f} GiB complex128"
    )

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    args.save_every = args.save_every or max(1, int(np.ceil(max(n_steps, 1)/100)))
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
        f"{args.progress_every}; global temporal order=2"
    )
    if abs(args.dt_au-0.1) < 1.0e-14:
        print("time step matches the cited calculation: dt=0.1 au")
    else:
        print(
            "note: the cited calculation used dt=0.1 au; this run uses "
            f"dt={args.dt_au:g} au"
        )

    histories = {
        "times_fs": [], "norm": [], "energy": [],
        "kinetic_energy_x": [], "kinetic_energy_q": [],
        "kinetic_energy_R": [], "potential_energy": [],
        "spectral_energy_decomposition_residual": [],
        "energy_imaginary_defect": [], "norm_rate": [],
        "projected_bo_norm": [], "bo_truncation_loss": [],
        "bo_populations": [], "joint_density": [],
        "proton_density": [], "heavy_density": [],
        "outer_probability_q": [], "outer_probability_R": [],
        "fixed_center_crossing_q_left": [],
        "fixed_center_crossing_q_right": [],
        "fixed_center_crossing_q": [],
        "fixed_center_crossing_R_left": [],
        "fixed_center_crossing_R_right": [],
        "fixed_center_crossing_R": [],
    }
    if args.bo_save_electron_density:
        histories["electron_density"] = []
    stage_path = outdir/".tdse_coefficients.partial.npy"
    action_stage_path = outdir/".tdse_action_coefficients.partial.npy"
    coefficient_stage = np.lib.format.open_memmap(
        stage_path, mode="w+", dtype=np.complex128,
        shape=(len(save_steps), n_states, len(cpu_model.q), len(cpu_model.R)),
    )
    action_stage = np.lib.format.open_memmap(
        action_stage_path, mode="w+", dtype=np.complex128,
        shape=(len(save_steps), n_states, len(cpu_model.q), len(cpu_model.R)),
    )

    def save(step, frame_index):
        energies = solver.energy(wavefunction)
        # Energy uses three temporary transform families.  Release their
        # unused cached blocks before allocating densities and H Psi.
        cp.get_default_memory_pool().free_all_blocks()
        norm = _scalar(energies["norm"])
        # ``conj(Psi)*Psi`` creates two full complex128 temporaries.  A
        # complex absolute-value ufunc followed by an in-place square needs
        # only one full float64 density array (half the bytes of Psi).
        density = cp.abs(wavefunction)
        cp.square(density, out=density)
        joint = cp.sum(density, axis=0, dtype=cp.float64)*cpu_model.dx/norm
        q_density = cp.sum(joint, axis=1, dtype=cp.float64)*cpu_model.dR
        R_density = cp.sum(joint, axis=0, dtype=cp.float64)*cpu_model.dq
        electron_density = None
        if args.bo_save_electron_density:
            electron_density = cp.sum(
                density, axis=(1, 2), dtype=cp.float64
            )*cpu_model.dq*cpu_model.dR/norm
        del density
        y = project_full_wavefunction_to_bo(
            wavefunction, basis.states, cpu_model.dx,
            block_R=args.tdse_projection_R_block,
        )
        coefficient_stage[frame_index] = y
        full_action = solver.action(wavefunction)
        # BLAS dotc performs the conjugation and reduction directly.  The
        # previous elementwise expression allocated conj(Psi) and their
        # product (2*Psi.nbytes), which OOMed an 11-GiB RTX 2080 Ti at the
        # production (300,600,800) grid before the first time step.
        action_expectation = cp.vdot(
            wavefunction, full_action
        )*cpu_model.dx*cpu_model.dq*cpu_model.dR
        action_y = project_full_wavefunction_to_bo(
            full_action, basis.states, cpu_model.dx,
            block_R=args.tdse_projection_R_block,
        )
        action_stage[frame_index] = action_y
        del full_action
        projected_populations = np.sum(
            np.abs(y)**2, axis=(1, 2), dtype=np.float64
        )*cpu_model.dq*cpu_model.dR
        projected_norm = float(np.sum(projected_populations))
        populations = projected_populations/max(norm, np.finfo(float).tiny)
        q_edge = min(5, len(cpu_model.q)//2)
        R_edge = min(5, len(cpu_model.R)//2)
        q_density_cpu = cp.asnumpy(q_density)
        R_density_cpu = cp.asnumpy(R_density)
        histories["times_fs"].append(step*args.dt_au/AU_PER_FS)
        histories["norm"].append(norm)
        histories["energy"].append(_scalar(action_expectation.real)/norm)
        histories["kinetic_energy_x"].append(
            _scalar(energies["kinetic_x"])/norm
        )
        histories["kinetic_energy_q"].append(
            _scalar(energies["kinetic_q"])/norm
        )
        histories["kinetic_energy_R"].append(
            _scalar(energies["kinetic_R"])/norm
        )
        histories["potential_energy"].append(
            _scalar(energies["potential"])/norm
        )
        histories["spectral_energy_decomposition_residual"].append(
            abs(_scalar(action_expectation.real)-_scalar(energies["energy"]))
            /norm
        )
        histories["energy_imaginary_defect"].append(
            abs(_scalar(action_expectation.imag))/norm
        )
        histories["norm_rate"].append(
            abs(2.0*_scalar(action_expectation.imag))
        )
        histories["projected_bo_norm"].append(projected_norm)
        histories["bo_truncation_loss"].append(
            max(0.0, 1.0-projected_norm/max(norm, np.finfo(float).tiny))
        )
        histories["bo_populations"].append(populations)
        histories["joint_density"].append(cp.asnumpy(joint))
        histories["proton_density"].append(q_density_cpu)
        histories["heavy_density"].append(R_density_cpu)
        if electron_density is not None:
            histories["electron_density"].append(cp.asnumpy(electron_density))
        histories["outer_probability_q"].append(float(
            (np.sum(q_density_cpu[:q_edge])+np.sum(q_density_cpu[-q_edge:]))
            *cpu_model.dq
        ))
        histories["outer_probability_R"].append(float(
            (np.sum(R_density_cpu[:R_edge])+np.sum(R_density_cpu[-R_edge:]))
            *cpu_model.dR
        ))
        _append_crossing(
            histories, "fixed_center_crossing_q", q_density_cpu,
            cpu_model.q, cpu_model.dq,
            crossing_reference_positions(cpu_model, args, "q"),
        )
        _append_crossing(
            histories, "fixed_center_crossing_R", R_density_cpu,
            cpu_model.R, cpu_model.dR,
            crossing_reference_positions(cpu_model, args, "R"),
        )
        coefficient_stage.flush()
        action_stage.flush()

    saved_frames = 0
    save(0, saved_frames)
    saved_frames += 1
    next_frame = 1
    last_completed_step = 0
    failure = ""
    attempted = 0
    throttle_sleep_seconds = 0.0
    throttled_steps = 0
    started = time.perf_counter()
    try:
        for step in range(1, n_steps+1):
            attempted = step
            solver.step(wavefunction, args.dt_au)
            last_completed_step = step
            will_save = next_frame < len(save_steps) and step == save_steps[next_frame]
            must_check = step % args.check_every == 0 or will_save or step == n_steps
            if must_check:
                if not bool(cp.all(cp.isfinite(wavefunction)).get()):
                    failure = f"step {step}: non-finite full TDSE wavefunction"
                    print(f"전파 중단: {failure}")
                    break
                norm = cp.real(cp.vdot(
                    wavefunction, wavefunction
                ))*cpu_model.dx*cpu_model.dq*cpu_model.dR
                drift = _scalar(cp.abs(norm-1.0))
                if args.max_norm_drift > 0.0 and drift > args.max_norm_drift:
                    failure = (
                        f"step {step}: |norm-1|={drift:.3e} exceeds "
                        f"{args.max_norm_drift:.3e}"
                    )
                    print(f"전파 중단: {failure}")
                    if not will_save:
                        save(step, saved_frames)
                        saved_frames += 1
                    break
            if will_save:
                save(step, saved_frames)
                saved_frames += 1
                next_frame += 1
            if step % args.progress_every == 0 or step == n_steps:
                print(
                    f"spectral TDSE step {step:7d}/{n_steps}  "
                    f"t={step*args.dt_au/AU_PER_FS:9.4f} fs"
                )
            if args.step_sleep_ms > 0.0:
                cp.cuda.get_current_stream().synchronize()
                sleep_started = time.perf_counter()
                time.sleep(args.step_sleep_ms/1000.0)
                throttle_sleep_seconds += time.perf_counter()-sleep_started
                throttled_steps += 1
    except KeyboardInterrupt:
        failure = f"사용자가 step {attempted}에서 계산을 중단함"
        print(f"전파 중단: {failure}")
        print(
            "in-place split step 중 interrupt 가능성이 있으므로 마지막 "
            "완료 step state를 추가 저장하지 않고 기존 완료 frame만 보존합니다."
        )

    cp.cuda.get_current_stream().synchronize()
    wall_seconds = time.perf_counter()-started
    completed = not failure
    interrupted = failure.startswith("사용자가 step ")
    coefficient_stage.flush()
    action_stage.flush()
    payload = {key: np.asarray(value) for key, value in histories.items()}
    payload["tdse_coefficients"] = coefficient_stage[:saved_frames]
    payload["tdse_action_coefficients"] = action_stage[:saved_frames]
    payload.update(
        kind=np.array("direct_discrete_born_huang_tdse_gpu_spectral_split"),
        representation=np.array("full_grid_wavefunction_with_bo_analysis_projection"),
        spatial_formulation=np.array("full_grid_dst1_fft_continuum_spectral"),
        time_integrator=np.array("second_order_strang_split_operator"),
        precision=np.array("complex128_float64"),
        tdse_wavefunction_bo_truncated=np.array(False),
        initial_analysis_bo_spatial_formulation=np.array(
            "shared_dirichlet_five_point_for_identical_initial_state"
        ),
        tdse_action=np.array("instantaneous_full_spectral_H_psi_projected_to_BO"),
        bo_states_count=np.array(n_states),
        bo_energies=np.asarray(basis.energies),
        bo_basis_cache_hit=np.array(cache_info["hit"]),
        bo_basis_cache_key=np.array(cache_info["key"]),
        bo_basis_cache_path=np.array(cache_info["path"]),
        bo_link_kernel=np.array(args.bo_link_kernel),
        x=cpu_model.x, q=cpu_model.q, R=cpu_model.R,
        propagation_completed=np.array(completed),
        propagation_interrupted=np.array(interrupted),
        requested_final_time_fs=np.array(args.t_final_fs),
        requested_steps=np.array(n_steps), attempted_steps=np.array(attempted),
        completed_steps=np.array(last_completed_step),
        step_sleep_ms=np.array(args.step_sleep_ms),
        throttle_sleep_seconds=np.array(throttle_sleep_seconds),
        throttled_steps=np.array(throttled_steps),
        failure_reason=np.array(failure), wall_seconds=np.array(wall_seconds),
        args=np.array([vars(args)], dtype=object),
    )
    archive = outdir/"multi_component_discrete_tdse_gpu.npz"
    archive_written = False
    try:
        np.savez_compressed(archive, **payload)
        archive_written = True
    finally:
        # Close the mmap before removing only our own staging file.
        del payload["tdse_coefficients"]
        del payload["tdse_action_coefficients"]
        del coefficient_stage, action_stage
        if archive_written and stage_path.exists():
            stage_path.unlink()
        if archive_written and action_stage_path.exists():
            action_stage_path.unlink()
    status_name = "completed" if completed else ("interrupted" if interrupted else "failed")
    (outdir/"propagation_status.log").write_text(
        f"status={status_name}\narchive={archive}\n"
        f"last_saved_time_fs={payload['times_fs'][-1]:.12g}\n"
        f"failure_reason={failure or 'none'}\n",
        encoding="utf-8",
    )
    pool = cp.get_default_memory_pool()
    print(f"{'저장 완료' if completed else '부분 저장 완료'}: {archive}")
    print(
        f"wall={wall_seconds:.3f} s; "
        f"{wall_seconds/max(attempted, 1):.6f} s/step; "
        f"GPU pool used/reserved={pool.used_bytes()/1024**3:.2f}/"
        f"{pool.total_bytes()/1024**3:.2f} GiB"
    )
    if throttled_steps:
        print(
            f"의도적 GPU 휴식={throttle_sleep_seconds:.3f} s/"
            f"{throttled_steps} steps"
        )
    print("Full spectral TDSE 핵심 진단:")
    print(f"  max |norm-1|: {np.max(np.abs(payload['norm']-1.0)):.3e}")
    print(
        "  max |energy-energy(0)|: "
        f"{np.max(np.abs(payload['energy']-payload['energy'][0])):.3e}"
    )
    print(
        "  max BO analysis truncation loss: "
        f"{np.max(payload['bo_truncation_loss']):.3e}"
    )
    print(
        "  max spectral energy decomposition residual: "
        f"{np.max(payload['spectral_energy_decomposition_residual']):.3e}"
    )
    print(
        "  max outer probability (q,R): "
        f"({np.max(payload['outer_probability_q']):.3e}, "
        f"{np.max(payload['outer_probability_R']):.3e})"
    )
    print(
        "  max highest stored BO-state population: "
        f"{np.max(payload['bo_populations'][:, -1]):.3e}"
    )
    args.propagation_failed = not completed
    return archive
