#!/usr/bin/env python3
"""Reconstruct nested exact-factorization fields from a saved TDSE trajectory.

The direct TDSE coefficient wavefunction is factorized frame by frame in the
density gauge (positive real ``F=Lambda*chi`` and ``chi``).  The instantaneous
TDSE action supplies exact frame-time derivatives, so both discrete scalar
potentials include their gauge-dependent temporal terms without differencing
widely separated saved frames.  The large TDSE member is streamed one frame at
a time and never materialized as a complete trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np

from multi_component_exact_factorization.born_huang import (
    load_or_build_born_huang_basis,
)
from multi_component_exact_factorization.core import build_model
from multi_component_exact_factorization.render_all import (
    find_archive,
    resolve_run_input,
)
from multi_component_exact_factorization.tdse_electron import (
    electron_marginal_from_bo,
)
from multi_component_exact_factorization_discrete.core import (
    OFFSETS,
    kinetic_weights,
)
from multi_component_exact_factorization_gpu.gpu_born_huang import (
    neighbor_transports,
    to_gpu_basis,
)
from multi_component_exact_factorization_gpu.gpu_core import cp

from .compare_tdse import _stream_arrays
from .gpu_core import discrete_tdse_action_gpu, make_discrete_gpu_model


OUTPUT_NAME = "tdse_exact_factorization_fields.npz"


def _metadata(archive):
    with np.load(archive, allow_pickle=True) as data:
        if "tdse_coefficients" not in data.files:
            raise ValueError("tdse_coefficients가 없는 archive입니다")
        stored = np.asarray(data["args"], dtype=object).reshape(-1)
        if stored.size != 1 or not isinstance(stored[0], dict):
            raise ValueError("TDSE archive의 args metadata가 없습니다")
        return {
            "args": dict(stored[0]),
            "times_fs": np.asarray(data["times_fs"], float),
            "x": np.asarray(data["x"], float),
            "q": np.asarray(data["q"], float),
            "R": np.asarray(data["R"], float),
            "bo_states": int(np.asarray(data["bo_states_count"]).item()),
            "bo_link_kernel": str(np.asarray(data.get("bo_link_kernel", "fused")).item()),
            "source_kind": str(np.asarray(data.get("kind", "")).item()),
        }


def _safe_density_gauge_factorization(y, action, model):
    """Return C, Lambda, chi and their exact instantaneous density-gauge rates."""
    real_dtype = model.reduction_real_dtype
    rho = cp.sum(cp.real(y*cp.conj(y)), axis=0, dtype=real_dtype)
    rho_R = cp.sum(rho, axis=0, dtype=real_dtype)*model.dq
    F = cp.sqrt(cp.maximum(rho, 0.0))
    chi = cp.sqrt(cp.maximum(rho_R, 0.0))
    active_F = F > 0.0
    active_chi = chi > 0.0
    safe_F = cp.where(active_F, F, 1.0)
    safe_chi = cp.where(active_chi, chi, 1.0)
    c = cp.where(active_F[None, :, :], y/safe_F[None, :, :], 0.0)
    lam = cp.where(active_chi[None, :], F/safe_chi[None, :], 0.0)

    dy = -1j*action
    drho = 2.0*cp.sum(
        cp.real(cp.conj(y)*dy), axis=0, dtype=real_dtype
    )
    drho_R = cp.sum(drho, axis=0, dtype=real_dtype)*model.dq
    dF = cp.where(active_F, drho/(2.0*safe_F), 0.0)
    dchi = cp.where(active_chi, drho_R/(2.0*safe_chi), 0.0)
    dc = cp.where(
        active_F[None, :, :],
        (dy-c*dF[None, :, :])/safe_F[None, :, :],
        0.0,
    )
    dlam = cp.where(
        active_chi[None, :],
        (dF-lam*dchi[None, :])/safe_chi[None, :],
        0.0,
    )
    return c, lam, chi, dc, dlam, dchi


def _frame_fields(y_cpu, model, basis):
    y = cp.ascontiguousarray(cp.asarray(y_cpu, dtype=cp.complex128))
    action = discrete_tdse_action_gpu(y, model, basis)
    state_probability = cp.real(y*cp.conj(y))
    total_probability = cp.maximum(
        cp.sum(state_probability, dtype=model.reduction_real_dtype)
        *model.dq*model.dR,
        cp.asarray(1.0e-300, dtype=model.reduction_real_dtype),
    )
    c, lam, chi, dc, dlam, _dchi = _safe_density_gauge_factorization(
        y, action, model
    )
    tiny = cp.asarray(1.0e-300, dtype=model.reduction_real_dtype)
    c_norm = cp.sum(
        cp.real(c*cp.conj(c)), axis=0, dtype=model.reduction_real_dtype
    )
    c_norm_safe = cp.maximum(c_norm, tiny)
    lam_norm = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    lam_norm_safe = cp.maximum(lam_norm, tiny)

    electronic = cp.sum(
        cp.conj(c)*basis.energies*c, axis=0,
        dtype=model.reduction_complex_dtype,
    )/c_norm_safe
    temporal_1 = -1j*cp.sum(
        cp.conj(c)*dc, axis=0, dtype=model.reduction_complex_dtype,
    )/c_norm_safe
    epsilon_1_complex = electronic+temporal_1

    q_weights = kinetic_weights(model.dq, model.proton_mass)
    q_action_lam = q_weights[0]*lam
    q_transports = neighbor_transports(c, basis, 1)
    sphi_q1 = None
    for index, offset in enumerate(OFFSETS):
        transport = q_transports[index]
        overlap = cp.sum(
            cp.conj(c)*transport, axis=0,
            dtype=model.reduction_complex_dtype,
        )/c_norm_safe
        q_action_lam += (
            q_weights[offset]
            *overlap*cp.roll(lam, -int(offset), axis=0)
        )
        if offset == 1:
            sphi_q1 = overlap

    hpr_local = epsilon_1_complex*lam+q_action_lam
    temporal_2 = -1j*cp.sum(
        cp.conj(lam)*dlam, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dq/lam_norm_safe
    epsilon_2_complex = cp.sum(
        cp.conj(lam)*hpr_local, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dq/lam_norm_safe+temporal_2

    R_transports = neighbor_transports(c, basis, 2)
    sphi_R1 = sgamma_R1 = None
    for index, offset in enumerate(OFFSETS):
        if offset != 1:
            continue
        transport = R_transports[index]
        sphi_R1 = cp.sum(
            cp.conj(c)*transport, axis=0,
            dtype=model.reduction_complex_dtype,
        )/c_norm_safe
        lam_neighbor = cp.roll(lam, -1, axis=1)
        transported_lam = sphi_R1*lam_neighbor
        sgamma_R1 = cp.sum(
            cp.conj(lam)*transported_lam, axis=0,
            dtype=model.reduction_complex_dtype,
        )*model.dq/lam_norm_safe
        break

    reconstructed = c*(lam*chi[None, :])[None, :, :]
    factorization_difference = y-reconstructed
    result = {
        "epsilon_1": epsilon_1_complex.real,
        "epsilon_2": epsilon_2_complex.real,
        "a": cp.angle(sphi_q1)/model.dq,
        "b": cp.angle(sphi_R1)/model.dR,
        "alpha": cp.angle(sgamma_R1)/model.dR,
        "bo_state_density_q": cp.sum(
            state_probability, axis=2, dtype=model.reduction_real_dtype,
        )*model.dR/total_probability,
        "bo_state_density_R": cp.sum(
            state_probability, axis=1, dtype=model.reduction_real_dtype,
        )*model.dq/total_probability,
        "epsilon_1_imaginary_defect": cp.max(cp.abs(epsilon_1_complex.imag)),
        "epsilon_2_imaginary_defect": cp.max(cp.abs(epsilon_2_complex.imag)),
        "factorization_residual": cp.sqrt(cp.sum(
            cp.real(factorization_difference*cp.conj(factorization_difference)),
            dtype=model.reduction_real_dtype,
        )*model.dq*model.dR),
    }
    return {key: cp.asnumpy(value) for key, value in result.items()}


def run(args):
    resolved = resolve_run_input(args.run)
    archive, run_dir = find_archive(resolved)
    if archive.name != "multi_component_discrete_tdse_gpu.npz":
        raise ValueError(f"TDSE archive가 아닙니다: {archive}")
    output = Path(args.output).expanduser().resolve() if args.output else run_dir/OUTPUT_NAME
    if output.exists() and not args.overwrite:
        print(f"TDSE exact-factorization field cache 재사용: {output}")
        return output

    metadata = _metadata(archive)
    options = metadata["args"]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".tdse-ef-write-test-", dir=output.parent
        ):
            pass
    except OSError as error:
        raise PermissionError(f"출력 폴더에 쓸 수 없습니다: {output.parent}") from error
    nt = len(metadata["times_fs"])
    nq, nR = len(metadata["q"]), len(metadata["R"])
    estimated_bytes = nt*(
        3*nq*nR+2*nR+metadata["bo_states"]*(nq+nR)
    )*np.dtype(np.float64).itemsize
    if args.electron_density:
        estimated_bytes += nt*len(metadata["x"])*np.dtype(np.float64).itemsize
    free_bytes = shutil.disk_usage(output.parent).free
    print(
        "EF field cache 예상 raw payload="
        f"{estimated_bytes/1024**3:.2f} GiB; free={free_bytes/1024**3:.2f} GiB"
    )
    if free_bytes < estimated_bytes:
        raise OSError(
            "TDSE EF field cache를 안전하게 저장할 공간이 부족합니다: "
            f"need at least {estimated_bytes/1024**3:.2f} GiB raw"
        )

    cpu_model = build_model(SimpleNamespace(**options))
    cp.cuda.Device(args.device).use()
    cache_dir = options.get("bo_basis_cache_dir", "results/bo_basis_cache")
    basis_cpu, cache_info = load_or_build_born_huang_basis(
        cpu_model, metadata["bo_states"], cache_dir=cache_dir,
    )
    model = make_discrete_gpu_model(cpu_model)
    compact_states = basis_cpu.states if args.electron_density else None
    link_kernel = args.bo_link_kernel or metadata["bo_link_kernel"]
    basis = to_gpu_basis(basis_cpu, model, link_kernel)
    if compact_states is None:
        basis_cpu.states = np.empty((0,), dtype=float)
    print(
        f"TDSE -> exact factorization fields: frames={len(metadata['times_fs'])}, "
        f"N_BO={metadata['bo_states']}, GPU={args.device}, links={link_kernel}"
    )
    print(
        f"BO cache {'HIT' if cache_info['hit'] else 'build'}: "
        f"{cache_info['seconds']:.2f} s; {cache_info['path']}"
    )

    fields = {
        "epsilon_1": np.empty((nt, nq, nR), dtype=np.float64),
        "epsilon_2": np.empty((nt, nR), dtype=np.float64),
        "a": np.empty((nt, nq, nR), dtype=np.float64),
        "b": np.empty((nt, nq, nR), dtype=np.float64),
        "alpha": np.empty((nt, nR), dtype=np.float64),
        "bo_state_density_q": np.empty(
            (nt, metadata["bo_states"], nq), dtype=np.float64
        ),
        "bo_state_density_R": np.empty(
            (nt, metadata["bo_states"], nR), dtype=np.float64
        ),
        "epsilon_1_imaginary_defect": np.empty(nt, dtype=np.float64),
        "epsilon_2_imaginary_defect": np.empty(nt, dtype=np.float64),
        "factorization_residual": np.empty(nt, dtype=np.float64),
    }
    if compact_states is not None:
        fields["electron_density"] = np.empty(
            (nt, len(metadata["x"])), dtype=np.float64
        )
        print(
            "TDSE electron marginal도 정확히 복원합니다. BO states를 "
            f"R-block={args.electron_density_R_block}으로 순차 읽습니다."
        )
    with _stream_arrays(archive, ("tdse_coefficients",)) as readers:
        reader = readers["tdse_coefficients"]
        if reader.shape != (nt, metadata["bo_states"], nq, nR):
            raise ValueError(
                f"tdse_coefficients shape mismatch: {reader.shape}"
            )
        for frame in range(nt):
            y_frame = reader.read(frame)
            current = _frame_fields(y_frame, model, basis)
            if compact_states is not None:
                current["electron_density"] = electron_marginal_from_bo(
                    y_frame, compact_states, model.dq, model.dR,
                    args.electron_density_R_block,
                )
            for key in fields:
                fields[key][frame] = current[key]
            if args.progress_every and (
                (frame+1) % args.progress_every == 0 or frame+1 == nt
            ):
                print(
                    f"TDSE EF fields {frame+1}/{nt}; "
                    f"t={metadata['times_fs'][frame]:.4f} fs; "
                    f"factor residual={fields['factorization_residual'][frame]:.3e}",
                    flush=True,
                )

    np.savez_compressed(
        output,
        **fields,
        kind=np.array("tdse_postprocessed_nested_exact_factorization_fields"),
        source_archive=np.array(str(archive)),
        source_kind=np.array(metadata["source_kind"]),
        gauge=np.array("positive_density_marginals"),
        scalar_time_derivative=np.array("instantaneous_tdse_action"),
        times_fs=metadata["times_fs"], x=metadata["x"],
        q=metadata["q"], R=metadata["R"],
    )
    print(f"TDSE exact-factorization fields 저장: {output}")
    print(
        "  max factorization residual: "
        f"{np.max(fields['factorization_residual']):.3e}"
    )
    print(
        "  max Im scalar defects (E1,E2): "
        f"({np.max(fields['epsilon_1_imaginary_defect']):.3e}, "
        f"{np.max(fields['epsilon_2_imaginary_defect']):.3e})"
    )
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="TDSE run folder or archive")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bo-link-kernel", choices=("reference", "fused"))
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(electron_density=True)
    parser.add_argument(
        "--no-electron-density", action="store_false", dest="electron_density",
        help="전자 marginal 복원을 생략해 field 후처리를 더 빠르게 수행",
    )
    parser.add_argument("--electron-density-R-block", type=int, default=24)
    args = parser.parse_args(argv)
    if args.electron_density_R_block <= 0:
        parser.error("--electron-density-R-block must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
