#!/usr/bin/env python3
"""Direct GPU TDSE reference in the same local Born--Huang basis as MCEF."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import shlex
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
)
from multi_component_exact_factorization_gpu.gpu_born_huang import to_gpu_basis
from multi_component_exact_factorization_gpu.gpu_core import cp

from .gpu_core import (
    discrete_tdse_action_gpu,
    full_step_discrete_tdse_gpu,
    make_discrete_gpu_model,
)


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


def _scalar(value):
    return float(value.get() if hasattr(value, "get") else value)


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
    print(f"TDSE Born--Huang basis 준비: N_BO={n_states}")
    cache_dir = args.bo_basis_cache_dir if args.bo_basis_cache else None
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
    y_cpu = c_cpu*(lam_cpu*chi_cpu[None, :])[None, :, :]
    model = make_discrete_gpu_model(cpu_model)
    basis = to_gpu_basis(basis_cpu, model, args.bo_link_kernel)
    y = cp.ascontiguousarray(cp.asarray(y_cpu, dtype=cp.complex128))
    # Eigenstates are not needed after initial projection.  Keep only the
    # immutable energies and links already uploaded by to_gpu_basis().
    basis_cpu.states = np.empty((0,), dtype=float)
    print(
        f"동적 배열: Y={y.shape}; direct i*dY/dt=H_hY; "
        "mask/PNC/retraction 없음"
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
        f"{args.progress_every}; integrator=classical RK4"
    )
    if args.step_sleep_ms > 0.0:
        print(
            "GPU thermal throttle: 각 완료 step 뒤 stream synchronize + "
            f"sleep {args.step_sleep_ms:g} ms"
        )

    histories = {
        "times_fs": [], "tdse_coefficients": [], "norm": [],
        "energy": [], "energy_imaginary_defect": [], "norm_rate": [],
        "bo_populations": [], "joint_density": [],
        "proton_density": [], "heavy_density": [],
        "outer_probability_q": [], "outer_probability_R": [],
    }

    def save(step):
        action = discrete_tdse_action_gpu(y, model, basis)
        density_state = cp.real(y*cp.conj(y))
        joint = cp.sum(
            density_state, axis=0, dtype=model.reduction_real_dtype
        )
        norm = cp.sum(joint, dtype=model.reduction_real_dtype)*model.dq*model.dR
        energy_complex = cp.sum(
            cp.conj(y)*action, dtype=model.reduction_complex_dtype
        )*model.dq*model.dR/norm
        dy = -1j*action
        norm_rate = 2.0*cp.sum(
            cp.conj(y)*dy, dtype=model.reduction_complex_dtype
        ).real*model.dq*model.dR
        populations = cp.sum(
            density_state, axis=(1, 2), dtype=model.reduction_real_dtype
        )*model.dq*model.dR/norm
        q_density = cp.sum(
            joint, axis=1, dtype=model.reduction_real_dtype
        )*model.dR/norm
        R_density = cp.sum(
            joint, axis=0, dtype=model.reduction_real_dtype
        )*model.dq/norm
        q_edge = min(5, len(cpu_model.q)//2)
        R_edge = min(5, len(cpu_model.R)//2)
        histories["times_fs"].append(step*args.dt_au/AU_PER_FS)
        histories["tdse_coefficients"].append(cp.asnumpy(y))
        histories["norm"].append(_scalar(norm))
        histories["energy"].append(_scalar(energy_complex.real))
        histories["energy_imaginary_defect"].append(
            abs(_scalar(energy_complex.imag))
        )
        histories["norm_rate"].append(abs(_scalar(norm_rate)))
        histories["bo_populations"].append(cp.asnumpy(populations))
        histories["joint_density"].append(cp.asnumpy(joint/norm))
        histories["proton_density"].append(cp.asnumpy(q_density))
        histories["heavy_density"].append(cp.asnumpy(R_density))
        histories["outer_probability_q"].append(_scalar(
            (cp.sum(q_density[:q_edge])+cp.sum(q_density[-q_edge:]))*model.dq
        ))
        histories["outer_probability_R"].append(_scalar(
            (cp.sum(R_density[:R_edge])+cp.sum(R_density[-R_edge:]))*model.dR
        ))

    save(0)
    next_frame = 1
    failure = ""
    attempted = 0
    throttle_sleep_seconds = 0.0
    throttled_steps = 0
    started = time.perf_counter()
    for step in range(1, n_steps+1):
        attempted = step
        y = full_step_discrete_tdse_gpu(y, args.dt_au, model, basis)
        will_save = next_frame < len(save_steps) and step == save_steps[next_frame]
        must_check = (
            step % args.check_every == 0 or will_save or step == n_steps
        )
        if must_check:
            if not bool(cp.all(cp.isfinite(y)).get()):
                failure = f"step {step}: non-finite TDSE coefficients"
                print(f"전파 중단: {failure}")
                break
            norm = cp.sum(
                cp.real(y*cp.conj(y)), dtype=model.reduction_real_dtype
            )*model.dq*model.dR
            drift = _scalar(cp.abs(norm-1.0))
            if args.max_norm_drift > 0.0 and drift > args.max_norm_drift:
                failure = (
                    f"step {step}: |norm-1|={drift:.3e} exceeds "
                    f"{args.max_norm_drift:.3e}"
                )
                print(f"전파 중단: {failure}")
                save(step)
                break
        if will_save:
            save(step)
            next_frame += 1
        if step % args.progress_every == 0 or step == n_steps:
            print(
                f"TDSE step {step:7d}/{n_steps}  "
                f"t={step*args.dt_au/AU_PER_FS:9.4f} fs"
            )
        if args.step_sleep_ms > 0.0:
            cp.cuda.get_current_stream().synchronize()
            sleep_started = time.perf_counter()
            time.sleep(args.step_sleep_ms/1000.0)
            throttle_sleep_seconds += time.perf_counter()-sleep_started
            throttled_steps += 1

    cp.cuda.get_current_stream().synchronize()
    wall_seconds = time.perf_counter()-started
    completed = not failure
    payload = {key: np.asarray(value) for key, value in histories.items()}
    payload.update(
        kind=np.array("direct_discrete_born_huang_tdse_gpu"),
        representation=np.array("full_wavefunction_bo_coefficients"),
        spatial_formulation=np.array("discretize_first_overlap_link"),
        time_integrator=np.array("classical_rk4"),
        precision=np.array("complex128_float64"),
        bo_states_count=np.array(n_states),
        bo_energies=np.asarray(basis_cpu.energies),
        bo_basis_cache_hit=np.array(cache_info["hit"]),
        bo_basis_cache_key=np.array(cache_info["key"]),
        bo_basis_cache_path=np.array(cache_info["path"]),
        bo_link_kernel=np.array(args.bo_link_kernel),
        x=cpu_model.x, q=cpu_model.q, R=cpu_model.R,
        propagation_completed=np.array(completed),
        requested_final_time_fs=np.array(args.t_final_fs),
        requested_steps=np.array(n_steps), attempted_steps=np.array(attempted),
        step_sleep_ms=np.array(args.step_sleep_ms),
        throttle_sleep_seconds=np.array(throttle_sleep_seconds),
        throttled_steps=np.array(throttled_steps),
        failure_reason=np.array(failure), wall_seconds=np.array(wall_seconds),
        args=np.array([vars(args)], dtype=object),
    )
    archive = outdir/"multi_component_discrete_tdse_gpu.npz"
    np.savez_compressed(archive, **payload)
    (outdir/"propagation_status.log").write_text(
        f"status={'completed' if completed else 'failed'}\n"
        f"archive={archive}\n"
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
    print("Direct discrete TDSE 핵심 진단:")
    print(f"  max |norm-1|: {np.max(np.abs(payload['norm']-1.0)):.3e}")
    print(
        "  max |energy-energy(0)|: "
        f"{np.max(np.abs(payload['energy']-payload['energy'][0])):.3e}"
    )
    print(
        "  max Im<Y|H|Y>: "
        f"{np.max(payload['energy_imaginary_defect']):.3e}"
    )
    print(f"  max instantaneous norm rate: {np.max(payload['norm_rate']):.3e}")
    print(
        "  max outer probability (q,R): "
        f"({np.max(payload['outer_probability_q']):.3e}, "
        f"{np.max(payload['outer_probability_R']):.3e})"
    )
    print(
        "  max highest-state population: "
        f"{np.max(payload['bo_populations'][:, -1]):.3e}"
    )
    args.propagation_failed = not completed
    return archive


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default="results/discrete_tdse_gpu",
        help="results/YYYYMMDD 아래의 run folder",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dt-au", type=float, default=0.025)
    parser.add_argument("--t-final-fs", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--check-every", type=int, default=0)
    parser.add_argument("--max-norm-drift", type=float, default=1.0e-4)
    parser.add_argument("--step-sleep-ms", type=float, default=0.0)
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
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    if args.dt_au <= 0.0 or args.t_final_fs < 0.0:
        parser.error("dt must be positive and final time nonnegative")
    if not np.isfinite(args.step_sleep_ms) or args.step_sleep_ms < 0.0:
        parser.error("--step-sleep-ms must be a finite nonnegative number")
    return args


def _execute(args):
    archive = run(args)
    if getattr(args, "propagation_failed", False):
        raise SystemExit(2)
    return archive


def main(args=None):
    if args is not None:
        return _execute(args)
    args = parse_args()
    outdir = dated_results_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir/"propagation.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(
            _Tee(sys.stderr, log)
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
