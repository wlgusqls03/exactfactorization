#!/usr/bin/env python3
"""서버의 CuPy와 custom DST-I/유한차분을 SciPy CPU 결과로 검증한다."""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
from scipy.fft import dst

from multi_component_exact_factorization.core import derivative as cpu_derivative
from .gpu_core import (
    cp,
    derivative,
    dst1_ortho,
    precision_types,
    remove_local_norm_generator,
)


def run(args):
    cp.cuda.Device(args.device).use()
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    print(f"CuPy {cp.__version__}")
    print(f"GPU {args.device}: {name}")
    print(f"CUDA driver/runtime: {cp.cuda.runtime.driverGetVersion()} / {cp.cuda.runtime.runtimeGetVersion()}")

    rng = np.random.default_rng(20260730)
    for np_dtype, tolerance in ((np.complex128, 2.0e-12), (np.complex64, 2.0e-5)):
        values = (
            rng.normal(size=(args.nx, 9, 5))
            +1j*rng.normal(size=(args.nx, 9, 5))
        ).astype(np_dtype)
        gpu_values = cp.asarray(values)
        transformed = cp.asnumpy(dst1_ortho(gpu_values, axis=0))
        reference = dst(values, type=1, axis=0, norm="ortho")
        transform_error = float(np.max(np.abs(transformed-reference)))
        roundtrip = cp.asnumpy(dst1_ortho(dst1_ortho(gpu_values, axis=0), axis=0))
        roundtrip_error = float(np.max(np.abs(roundtrip-values)))
        print(
            f"{np_dtype.__name__}: DST error={transform_error:.3e}, "
            f"roundtrip={roundtrip_error:.3e}"
        )
        if transform_error > tolerance or roundtrip_error > tolerance:
            raise AssertionError(f"{np_dtype.__name__} DST-I 검증 실패")

    values = rng.normal(size=(31, 17, 11))+1j*rng.normal(size=(31, 17, 11))
    for axis in range(3):
        gpu_result = cp.asnumpy(derivative(cp.asarray(values), 0.08, axis))
        error = float(np.max(np.abs(gpu_result-cpu_derivative(values, 0.08, axis))))
        print(f"derivative axis={axis}: error={error:.3e}")
        if error > 1.0e-12:
            raise AssertionError("GPU 유한차분 검증 실패")

    # RK stage처럼 local norm이 1이 아닌 factor에서도 알려진 i*eta*f를
    # 정확히 제거하는지 precision별로 확인한다.
    eta_np = np.array([[0.037, -0.021], [0.012, 0.044], [-0.015, 0.029]])
    weight_np = np.array([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]])
    base = rng.normal(size=(7, 3, 2))+1j*rng.normal(size=(7, 3, 2))
    base *= np.array([0.7, 1.0, 1.3])[None, :, None]
    for precision, tolerance in (
        ("double", 2.0e-12), ("mixed", 2.0e-6), ("single", 2.0e-5),
    ):
        real, complex_, reduction_real, reduction_complex = precision_types(precision)
        precision_model = SimpleNamespace(
            real_dtype=real, complex_dtype=complex_,
            reduction_real_dtype=reduction_real,
            reduction_complex_dtype=reduction_complex,
        )
        factor = cp.asarray(base, dtype=complex_)
        eta = cp.asarray(eta_np, dtype=real)
        weight = cp.asarray(weight_np, dtype=real)
        hermitian_action = weight[None, :, :]*factor
        raw_action = hermitian_action+cp.asarray(1j, dtype=complex_)*eta[None, :, :]*factor
        corrected, gamma, _, corrected_rate = remove_local_norm_generator(
            factor, raw_action, 0.08, axis=0, model=precision_model
        )
        gamma_error = float(cp.max(cp.abs(gamma-eta)).get())
        rate_error = float(cp.max(cp.abs(corrected_rate)).get())
        action_error = float(cp.max(cp.abs(corrected-hermitian_action)).get())
        print(
            f"{precision} norm correction: gamma={gamma_error:.3e}, "
            f"rate={rate_error:.3e}, action={action_error:.3e}"
        )
        if corrected.dtype != complex_:
            raise AssertionError(f"{precision} correction dtype 검증 실패")
        if max(gamma_error, rate_error, action_error) > tolerance:
            raise AssertionError(f"{precision} local-norm correction 검증 실패")

    benchmark = cp.asarray(
        rng.normal(size=(args.nx, args.nq, args.nR))
        +1j*rng.normal(size=(args.nx, args.nq, args.nR)),
        dtype=cp.complex64,
    )
    dst1_ortho(benchmark, axis=0)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(args.repeats):
        benchmark = dst1_ortho(benchmark, axis=0)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter()-start
    print(f"complex64 DST-I: {1e3*elapsed/args.repeats:.3f} ms/call")
    print("GPU 기본 검증: PASS")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nx", type=int, default=174)
    parser.add_argument("--nq", type=int, default=87)
    parser.add_argument("--nR", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
