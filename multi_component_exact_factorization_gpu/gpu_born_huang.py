"""GPU Born--Huang coefficient propagation for electronic-only MCEF expansion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gpu_core import (
    cp,
    covariant_square,
    deep_tail_gate,
    derivative,
    gated_values,
    logarithmic_components,
    flat_top_support_mask,
    occupied_support_mask,
    proton_base_operator,
    remove_local_norm_generator,
    suppressed_probability,
    weak_log_amplitude_gradient,
)


PNC_NORM_DIAGNOSTIC_NAMES = tuple(
    name
    for factor in ("c", "lam")
    for name in (
        f"max_pre_pnc_{factor}_norm",
        f"max_inverse_pre_pnc_{factor}_norm",
        f"max_pre_pnc_tail_{factor}_norm",
        f"max_inverse_pre_pnc_tail_{factor}_norm",
        f"max_pre_pnc_support_{factor}_norm",
        f"max_inverse_pre_pnc_support_{factor}_norm",
        *(
            item
            for threshold in (
                "lt_1e_4", "lt_1e_2", "lt_1e_1",
                "gt_1e1", "gt_1e2", "gt_1e4",
            )
            for item in (
                f"count_pre_pnc_{factor}_norm_{threshold}",
                f"fraction_pre_pnc_{factor}_norm_{threshold}",
            )
        ),
    )
)


@dataclass
class GPUBornHuangBasis:
    energies: cp.ndarray
    link_q1: cp.ndarray
    link_q2: cp.ndarray
    link_R1: cp.ndarray
    link_R2: cp.ndarray
    back_q1: object = None
    back_q2: object = None
    back_R1: object = None
    back_R2: object = None
    link_kernel: str = "reference"
    workspace: object = None


@dataclass
class BHLinkWorkspace:
    """Reusable fused-link temporaries for one coefficient tensor."""

    transports: cp.ndarray
    first: cp.ndarray
    second: cp.ndarray


_BO_LINK_KERNEL_CACHE = {}


def _bo_link_kernels(link_dtype):
    """Compile exact complex128 BO transport/combine kernels lazily."""
    dtype = np.dtype(link_dtype)
    cached = _BO_LINK_KERNEL_CACHE.get(dtype.str)
    if cached is not None:
        return cached
    if dtype == np.dtype(np.float64):
        link_type = "double"
        forward_product = "mul_real(link_value, coefficient)"
        backward_product = "mul_real(link_value, coefficient)"
        forward_load = "links[(left*nstate+right)*ngrid+grid]"
        backward_load = "links[(right*nstate+left)*ngrid+neighbor_grid]"
    elif dtype == np.dtype(np.complex128):
        link_type = "double2"
        forward_product = "mul_complex(link_value, coefficient)"
        backward_product = "mul_complex(mcef_conjugate(link_value), coefficient)"
        forward_load = "links[(left*nstate+right)*ngrid+grid]"
        backward_load = "links[(right*nstate+left)*ngrid+neighbor_grid]"
    else:
        raise TypeError(
            "fused BO link kernel은 float64/complex128 link만 지원합니다."
        )
    suffix = "real" if dtype.kind == "f" else "complex"
    transport_name = f"mcef_bh_transport_c128_{suffix}"
    combine_name = "mcef_bh_covariant_combine_c128"
    code = f'''\
__device__ __forceinline__ double2 add_complex(double2 a, double2 b) {{
    return make_double2(a.x+b.x, a.y+b.y);
}}
__device__ __forceinline__ double2 mul_real(double a, double2 b) {{
    return make_double2(a*b.x, a*b.y);
}}
__device__ __forceinline__ double2 mul_complex(double2 a, double2 b) {{
    return make_double2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}}
__device__ __forceinline__ double2 mcef_conjugate(double2 a) {{
    return make_double2(a.x, -a.y);
}}
__device__ __forceinline__ int wrapped(int value, int length) {{
    value %= length;
    return value < 0 ? value+length : value;
}}
__device__ __forceinline__ long long neighbor_index(
    int iq, int iR, int offset, int axis, int nq, int nR
) {{
    if (axis == 1) iq = wrapped(iq+offset, nq);
    else iR = wrapped(iR+offset, nR);
    return (long long)iq*nR+iR;
}}

extern "C" __global__
void {transport_name}(
    const double2* coefficients,
    const {link_type}* link1,
    const {link_type}* link2,
    double2* transports,
    double2* first,
    double2* second,
    const long long size,
    const int nstate,
    const int nq,
    const int nR,
    const int axis,
    const int write_transports,
    const int write_first,
    const int write_second,
    const double first_scale,
    const double second_scale
) {{
    const long long index = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (index >= size) return;
    const long long ngrid = (long long)nq*nR;
    const int left = (int)(index/ngrid);
    const long long grid = index-(long long)left*ngrid;
    const int iq = (int)(grid/nR);
    const int iR = (int)(grid-(long long)iq*nR);
    const long long gm2 = neighbor_index(iq, iR, -2, axis, nq, nR);
    const long long gm1 = neighbor_index(iq, iR, -1, axis, nq, nR);
    const long long gp1 = neighbor_index(iq, iR, 1, axis, nq, nR);
    const long long gp2 = neighbor_index(iq, iR, 2, axis, nq, nR);
    double2 tm2 = make_double2(0.0, 0.0);
    double2 tm1 = make_double2(0.0, 0.0);
    double2 tp1 = make_double2(0.0, 0.0);
    double2 tp2 = make_double2(0.0, 0.0);
    for (int right = 0; right < nstate; ++right) {{
        double2 coefficient = coefficients[(long long)right*ngrid+gp1];
        {link_type} link_value = {forward_load.replace('links', 'link1')};
        tp1 = add_complex(tp1, {forward_product});
        coefficient = coefficients[(long long)right*ngrid+gp2];
        link_value = {forward_load.replace('links', 'link2')};
        tp2 = add_complex(tp2, {forward_product});
        coefficient = coefficients[(long long)right*ngrid+gm1];
        const long long neighbor_grid = gm1;
        link_value = {backward_load.replace('links', 'link1')};
        tm1 = add_complex(tm1, {backward_product});
        coefficient = coefficients[(long long)right*ngrid+gm2];
        link_value = link2[(right*nstate+left)*ngrid+gm2];
        tm2 = add_complex(tm2, {backward_product});
    }}
    if (write_transports) {{
        transports[index] = tm2;
        transports[size+index] = tm1;
        transports[2*size+index] = tp1;
        transports[3*size+index] = tp2;
    }}
    if (write_first) {{
        first[index] = make_double2(
            (tm2.x-8.0*tm1.x+8.0*tp1.x-tp2.x)*first_scale,
            (tm2.y-8.0*tm1.y+8.0*tp1.y-tp2.y)*first_scale
        );
    }}
    if (write_second) {{
        const double2 center = coefficients[index];
        second[index] = make_double2(
            (-tm2.x+16.0*tm1.x-30.0*center.x+16.0*tp1.x-tp2.x)*second_scale,
            (-tm2.y+16.0*tm1.y-30.0*center.y+16.0*tp1.y-tp2.y)*second_scale
        );
    }}
}}

extern "C" __global__
void {combine_name}(
    const double2* transports,
    const double2* coefficients,
    const double2* phase1,
    const double2* phase2,
    double2* first,
    double2* second,
    const long long size,
    const int nq,
    const int nR,
    const int axis,
    const double first_scale,
    const double second_scale
) {{
    const long long index = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (index >= size) return;
    const long long ngrid = (long long)nq*nR;
    const long long grid = index%ngrid;
    const int iq = (int)(grid/nR);
    const int iR = (int)(grid-(long long)iq*nR);
    const long long gm2 = neighbor_index(iq, iR, -2, axis, nq, nR);
    const long long gm1 = neighbor_index(iq, iR, -1, axis, nq, nR);
    double2 tm2 = mul_complex(mcef_conjugate(phase2[gm2]), transports[index]);
    double2 tm1 = mul_complex(mcef_conjugate(phase1[gm1]), transports[size+index]);
    double2 tp1 = mul_complex(phase1[grid], transports[2*size+index]);
    double2 tp2 = mul_complex(phase2[grid], transports[3*size+index]);
    first[index] = make_double2(
        (tm2.x-8.0*tm1.x+8.0*tp1.x-tp2.x)*first_scale,
        (tm2.y-8.0*tm1.y+8.0*tp1.y-tp2.y)*first_scale
    );
    const double2 center = coefficients[index];
    second[index] = make_double2(
        (-tm2.x+16.0*tm1.x-30.0*center.x+16.0*tp1.x-tp2.x)*second_scale,
        (-tm2.y+16.0*tm1.y-30.0*center.y+16.0*tp1.y-tp2.y)*second_scale
    );
}}
'''
    transport = cp.RawKernel(
        code, transport_name, options=("--std=c++11",)
    )
    combine = cp.RawKernel(
        code, combine_name, options=("--std=c++11",)
    )
    _BO_LINK_KERNEL_CACHE[dtype.str] = (transport, combine)
    return transport, combine


def to_gpu_basis(basis, model, link_kernel="reference"):
    for name in ("link_q1", "link_q2", "link_R1", "link_R2"):
        if getattr(basis, name, None) is None:
            raise ValueError(f"BO overlap link가 없습니다: {name}")

    if link_kernel not in ("reference", "fused"):
        raise ValueError("BO link kernel은 reference 또는 fused여야 합니다.")

    def links(forward, axis, offset):
        # The present one-dimensional electronic Hamiltonian has real BO
        # eigenvectors, but retaining this branch makes the link backend safe
        # for future magnetic/complex bases instead of silently discarding a
        # phase that is required by the adjoint relation.
        dtype = (
            model.complex_dtype if np.iscomplexobj(forward)
            else model.real_dtype
        )
        # A leading BO block sliced from a larger superset cache can retain
        # the parent array's wider strides.  Materialize the upload in the
        # layout used by the explicit CUDA indexing below.  This changes no
        # values and also gives the reference einsum the same dense layout.
        forward_gpu = cp.ascontiguousarray(cp.asarray(forward, dtype=dtype))
        backward_gpu = None
        if link_kernel == "reference":
            backward_gpu = cp.roll(
                cp.swapaxes(cp.conj(forward_gpu), 0, 1),
                offset, axis=axis+1,
            )
        return forward_gpu, backward_gpu

    link_q1, back_q1 = links(basis.link_q1, 1, 1)
    link_q2, back_q2 = links(basis.link_q2, 1, 2)
    link_R1, back_R1 = links(basis.link_R1, 2, 1)
    link_R2, back_R2 = links(basis.link_R2, 2, 2)
    shape = (basis.energies.shape[0],)+basis.energies.shape[1:]
    workspace = None
    if link_kernel == "fused":
        if model.complex_dtype != cp.complex128:
            raise TypeError("fused BO link kernel은 현재 complex128 전용입니다.")
        workspace = BHLinkWorkspace(
            transports=cp.empty((4,)+shape, dtype=model.complex_dtype),
            first=cp.empty(shape, dtype=model.complex_dtype),
            second=cp.empty(shape, dtype=model.complex_dtype),
        )
    return GPUBornHuangBasis(
        energies=cp.asarray(basis.energies, dtype=model.real_dtype),
        link_q1=link_q1, link_q2=link_q2,
        link_R1=link_R1, link_R2=link_R2,
        back_q1=back_q1, back_q2=back_q2,
        back_R1=back_R1, back_R2=back_R2,
        link_kernel=link_kernel, workspace=workspace,
    )


def connection_action(connection, coefficients):
    return cp.einsum("ljqR,jqR->lqR", connection, coefficients)


def _axis_links(basis, axis):
    if axis == 1:
        return basis.link_q1, basis.link_q2, basis.back_q1, basis.back_q2
    if axis == 2:
        return basis.link_R1, basis.link_R2, basis.back_R1, basis.back_R2
    raise ValueError("BO coefficient coordinate axis must be q(1) or R(2)")


def _axis_forward_links(basis, axis):
    if axis == 1:
        return basis.link_q1, basis.link_q2
    if axis == 2:
        return basis.link_R1, basis.link_R2
    raise ValueError("BO coefficient coordinate axis must be q(1) or R(2)")


def _launch_fused_transport(
    coefficients, basis, spacing, axis, *,
    write_transports, write_first, write_second,
):
    """Apply all four BO neighbor contractions in one memory pass."""
    if not coefficients.flags.c_contiguous:
        raise ValueError("fused BO link kernel에는 C-contiguous coefficient가 필요합니다.")
    link1, link2 = _axis_forward_links(basis, axis)
    if not link1.flags.c_contiguous or not link2.flags.c_contiguous:
        raise ValueError("fused BO link kernel에는 C-contiguous link가 필요합니다.")
    workspace = basis.workspace
    transport_kernel, _ = _bo_link_kernels(link1.dtype)
    threads = 128
    blocks = (coefficients.size+threads-1)//threads
    transport_kernel(
        (blocks,), (threads,),
        (
            coefficients, link1, link2,
            workspace.transports, workspace.first, workspace.second,
            np.int64(coefficients.size), np.int32(coefficients.shape[0]),
            np.int32(coefficients.shape[1]), np.int32(coefficients.shape[2]),
            np.int32(axis), np.int32(write_transports),
            np.int32(write_first), np.int32(write_second),
            np.float64(1.0/(12.0*spacing)),
            np.float64(1.0/(12.0*spacing**2)),
        ),
    )
    return workspace.first, workspace.second


def _fused_covariant_from_transports(
    coefficients, basis, vector, spacing, axis,
):
    """Apply Wilson phases to cached transports without another BO contraction."""
    if vector.dtype != cp.float64 or not vector.flags.c_contiguous:
        vector = cp.ascontiguousarray(vector, dtype=cp.float64)
    phase1, phase2 = _forward_gauge_phases(vector, spacing, axis-1)
    first = cp.empty_like(coefficients)
    second = cp.empty_like(coefficients)
    _, combine_kernel = _bo_link_kernels(
        _axis_forward_links(basis, axis)[0].dtype
    )
    threads = 128
    blocks = (coefficients.size+threads-1)//threads
    combine_kernel(
        (blocks,), (threads,),
        (
            basis.workspace.transports, coefficients, phase1, phase2,
            first, second, np.int64(coefficients.size),
            np.int32(coefficients.shape[1]), np.int32(coefficients.shape[2]),
            np.int32(axis),
            np.float64(1.0/(12.0*spacing)),
            np.float64(1.0/(12.0*spacing**2)),
        ),
    )
    return first, second


def _forward_gauge_phases(vector, spacing, vector_axis):
    """Fourth-order symmetric link phases for (d-iA) on a uniform grid."""
    minus1 = cp.roll(vector, 1, axis=vector_axis)
    plus1 = cp.roll(vector, -1, axis=vector_axis)
    plus2 = cp.roll(vector, -2, axis=vector_axis)
    integral1 = spacing*(-minus1+13.0*vector+13.0*plus1-plus2)/24.0
    phase1 = cp.exp(-1j*integral1)
    # The length-two Wilson line is the product of adjacent length-one links.
    phase2 = phase1*cp.roll(phase1, -1, axis=vector_axis)
    return phase1, phase2


def _projected_link_derivatives_reference(
    coefficients, basis, spacing, axis, vector=None,
):
    """Projected D1/D2 with exact discrete adjoint relations.

    Neighbor BO overlaps evaluate Phi(g)^H D[Phi C](g) directly.  Forward
    and backward links are conjugate transposes, hence plain D1 is exactly
    anti-Hermitian and plain D2 exactly Hermitian.  Optional real ``vector``
    inserts mutually conjugate Wilson phases for the covariant derivatives.
    """
    link1, link2, back1, back2 = _axis_links(basis, axis)
    plus1_values = cp.roll(coefficients, -1, axis=axis)
    plus2_values = cp.roll(coefficients, -2, axis=axis)
    minus1_values = cp.roll(coefficients, 1, axis=axis)
    minus2_values = cp.roll(coefficients, 2, axis=axis)
    plus1 = connection_action(link1, plus1_values)
    plus2 = connection_action(link2, plus2_values)
    minus1 = connection_action(back1, minus1_values)
    minus2 = connection_action(back2, minus2_values)
    if vector is not None:
        vector_axis = axis-1
        phase1, phase2 = _forward_gauge_phases(
            vector, spacing, vector_axis
        )
        plus1 = phase1[None, :, :]*plus1
        plus2 = phase2[None, :, :]*plus2
        minus1 = cp.conj(cp.roll(
            phase1, 1, axis=vector_axis
        ))[None, :, :]*minus1
        minus2 = cp.conj(cp.roll(
            phase2, 2, axis=vector_axis
        ))[None, :, :]*minus2
    first = (minus2-8.0*minus1+8.0*plus1-plus2)/(12.0*spacing)
    second = (
        -minus2+16.0*minus1-30.0*coefficients+16.0*plus1-plus2
    )/(12.0*spacing**2)
    return first, second


def projected_link_derivatives(
    coefficients, basis, spacing, axis, vector=None,
):
    """Dispatch to the allocation-heavy reference or fused link stencil."""
    if basis.link_kernel == "reference":
        return _projected_link_derivatives_reference(
            coefficients, basis, spacing, axis, vector=vector
        )
    if vector is None:
        first, second = _launch_fused_transport(
            coefficients, basis, spacing, axis,
            write_transports=False, write_first=True, write_second=True,
        )
        # The public helper owns its outputs.  Production paths below use the
        # workspace views directly and avoid these copies.
        return first.copy(), second.copy()
    _launch_fused_transport(
        coefficients, basis, spacing, axis,
        write_transports=True, write_first=False, write_second=False,
    )
    return _fused_covariant_from_transports(
        coefficients, basis, vector, spacing, axis
    )


def neighbor_transports(coefficients, basis, axis):
    """Return BO-overlap transports at offsets ``(-2,-1,+1,+2)``.

    The returned tensor has shape ``(4,N_BO,nq,nR)`` and evaluates

        S_BO(g,g+s) C(g+s)

    at every current nuclear configuration ``g``.  The fused backend returns
    a reusable workspace view which remains valid only until the next call to
    this function for the same ``basis``.  This low-level primitive is shared
    with the discretize-first MCEF solver; exposing it here avoids duplicating
    the already validated CUDA overlap-link contraction.
    """
    if axis not in (1, 2):
        raise ValueError("BO transport axis must be q(1) or R(2)")
    if basis.link_kernel == "fused":
        _launch_fused_transport(
            coefficients, basis, 1.0, axis,
            write_transports=True, write_first=False, write_second=False,
        )
        return basis.workspace.transports

    link1, link2, back1, back2 = _axis_links(basis, axis)
    return cp.stack((
        connection_action(back2, cp.roll(coefficients, 2, axis=axis)),
        connection_action(back1, cp.roll(coefficients, 1, axis=axis)),
        connection_action(link1, cp.roll(coefficients, -1, axis=axis)),
        connection_action(link2, cp.roll(coefficients, -2, axis=axis)),
    ))


def projected_gradient(coefficients, connection, spacing, axis):
    return (
        derivative(coefficients, spacing, axis=axis)
        +connection_action(connection, coefficients)
    )


def residual_momentum(coefficients, connection, vector, spacing, axis):
    return -1j*projected_gradient(
        coefficients, connection, spacing, axis
    )-vector[None, :, :]*coefficients


def residual_square(
    coefficients, first_connection, second_connection, vector, spacing, axis,
):
    first = derivative(coefficients, spacing, axis=axis)
    second = derivative(coefficients, spacing, axis=axis, order=2)
    vector_derivative = derivative(vector, spacing, axis=axis-1)
    return (
        -second
        -2.0*connection_action(first_connection, first)
        -connection_action(second_connection, coefficients)
        +1j*vector_derivative[None, :, :]*coefficients
        +2j*vector[None, :, :]*(
            first+connection_action(first_connection, coefficients)
        )
        +vector[None, :, :]**2*coefficients
    )


def projected_plain_second(
    coefficients, first_connection, second_connection, spacing, axis,
):
    first = derivative(coefficients, spacing, axis=axis)
    return (
        derivative(coefficients, spacing, axis=axis, order=2)
        +2.0*connection_action(first_connection, first)
        +connection_action(second_connection, coefficients)
    )


def coefficient_vector_potential(coefficients, connection, spacing, axis, model):
    gradient = projected_gradient(coefficients, connection, spacing, axis)
    value = -1j*cp.sum(
        cp.conj(coefficients)*gradient, axis=0,
        dtype=model.reduction_complex_dtype,
    )
    return value.real.astype(model.real_dtype, copy=False)


def _pnc_norm_statistics(norm, gate, prefix, model):
    """Diagnostics immediately before a support-aware PNC rescaling."""
    tiny = cp.asarray(1.0e-300, dtype=model.reduction_real_dtype)
    inverse = 1.0/cp.maximum(norm, tiny)
    tail = gate == 0.0
    support = gate == 1.0
    statistics = {
        f"max_pre_pnc_{prefix}_norm": cp.max(norm),
        f"max_inverse_pre_pnc_{prefix}_norm": cp.max(inverse),
        f"max_pre_pnc_tail_{prefix}_norm": cp.max(cp.where(tail, norm, 0.0)),
        f"max_inverse_pre_pnc_tail_{prefix}_norm": cp.max(
            cp.where(tail, inverse, 0.0)
        ),
        f"max_pre_pnc_support_{prefix}_norm": cp.max(
            cp.where(support, norm, 0.0)
        ),
        f"max_inverse_pre_pnc_support_{prefix}_norm": cp.max(
            cp.where(support, inverse, 0.0)
        ),
    }
    for label, condition in (
        ("lt_1e_4", norm < 1.0e-4),
        ("lt_1e_2", norm < 1.0e-2),
        ("lt_1e_1", norm < 1.0e-1),
        ("gt_1e1", norm > 1.0e1),
        ("gt_1e2", norm > 1.0e2),
        ("gt_1e4", norm > 1.0e4),
    ):
        count = cp.count_nonzero(condition)
        statistics[f"count_pre_pnc_{prefix}_norm_{label}"] = count
        statistics[f"fraction_pre_pnc_{prefix}_norm_{label}"] = count/norm.size
    return statistics


def pnc_project_coefficients(
    coefficients, lam, chi, model, *, return_diagnostics=False,
):
    c_norm2 = cp.sum(
        cp.real(coefficients*cp.conj(coefficients)), axis=0,
        dtype=model.reduction_real_dtype,
    )
    c_error = cp.max(cp.abs(c_norm2-1.0))
    physical_qR = c_norm2*cp.real(
        (lam*chi[None, :])*cp.conj(lam*chi[None, :])
    )
    gate_c = deep_tail_gate(
        physical_qR, model.deep_tail_zero_threshold, model
    )
    c_norm = cp.sqrt(c_norm2).astype(model.real_dtype, copy=False)
    safe_c = cp.where(c_norm > 1.0e-14, c_norm, 1.0)
    scale_c = cp.exp(gated_values(cp.log(safe_c), gate_c))
    c_applied_error = cp.max(cp.abs(scale_c**2-1.0))
    coefficients = coefficients/scale_c[None, :, :]
    lam = lam*scale_c
    lam_norm2 = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    lam_error = cp.max(cp.abs(lam_norm2-1.0))
    physical_R = cp.sum(
        physical_qR, axis=0, dtype=model.reduction_real_dtype
    )*model.dq
    gate_lam = deep_tail_gate(
        physical_R, model.deep_tail_zero_threshold, model
    )
    lam_norm = cp.sqrt(lam_norm2).astype(model.real_dtype, copy=False)
    safe_lam = cp.where(lam_norm > 1.0e-14, lam_norm, 1.0)
    scale_lam = cp.exp(gated_values(cp.log(safe_lam), gate_lam))
    lam_applied_error = cp.max(cp.abs(scale_lam**2-1.0))
    lam = lam/scale_lam[None, :]
    chi = chi*scale_lam
    result = (
        coefficients, lam, chi,
        cp.maximum(c_applied_error, lam_applied_error),
    )
    if not return_diagnostics:
        return result
    diagnostics = {
        "max_raw_pnc_phi_error": c_error,
        "max_raw_pnc_lam_error": lam_error,
    }
    diagnostics.update(_pnc_norm_statistics(c_norm, gate_c, "c", model))
    diagnostics.update(_pnc_norm_statistics(lam_norm, gate_lam, "lam", model))
    return result+ (diagnostics,)


def instantaneous_functionals_bh(
    coefficients, lam, chi, model, basis, ratio_floor,
    mask_threshold_phi, mask_threshold_lam,
):
    # Plain and covariant derivatives use the same four BO transports.  The
    # fused path contracts them once per axis, obtains a/b from plain D1, then
    # applies Wilson phases to the cached transports.  The reference path is
    # intentionally retained for round-off-level validation.
    if basis.link_kernel == "fused":
        gradient_q, _ = _launch_fused_transport(
            coefficients, basis, model.dq, 1,
            write_transports=True, write_first=True, write_second=False,
        )
        a = (-1j*cp.sum(
            cp.conj(coefficients)*gradient_q, axis=0,
            dtype=model.reduction_complex_dtype,
        )).real.astype(model.real_dtype, copy=False)
        cov_gradient_q, cov_second_q = _fused_covariant_from_transports(
            coefficients, basis, a, model.dq, 1
        )
        gradient_R, _ = _launch_fused_transport(
            coefficients, basis, model.dR, 2,
            write_transports=True, write_first=True, write_second=False,
        )
        b = (-1j*cp.sum(
            cp.conj(coefficients)*gradient_R, axis=0,
            dtype=model.reduction_complex_dtype,
        )).real.astype(model.real_dtype, copy=False)
        cov_gradient_R, cov_second_R = _fused_covariant_from_transports(
            coefficients, basis, b, model.dR, 2
        )
    else:
        gradient_q, _ = projected_link_derivatives(
            coefficients, basis, model.dq, axis=1
        )
        gradient_R, _ = projected_link_derivatives(
            coefficients, basis, model.dR, axis=2
        )
        a = (-1j*cp.sum(
            cp.conj(coefficients)*gradient_q, axis=0,
            dtype=model.reduction_complex_dtype,
        )).real.astype(model.real_dtype, copy=False)
        b = (-1j*cp.sum(
            cp.conj(coefficients)*gradient_R, axis=0,
            dtype=model.reduction_complex_dtype,
        )).real.astype(model.real_dtype, copy=False)
        cov_gradient_q, cov_second_q = projected_link_derivatives(
            coefficients, basis, model.dq, axis=1, vector=a
        )
        cov_gradient_R, cov_second_R = projected_link_derivatives(
            coefficients, basis, model.dR, axis=2, vector=b
        )
    p_R_lam = -1j*derivative(lam, model.dR, axis=1)
    alpha = cp.sum(
        cp.conj(lam)*(p_R_lam+b*lam), axis=0,
        dtype=model.reduction_complex_dtype,
    ).real*model.dq
    alpha = alpha.astype(model.real_dtype, copy=False)

    c_norm2 = cp.sum(
        cp.real(coefficients*cp.conj(coefficients)), axis=0,
        dtype=model.reduction_real_dtype,
    )
    rho_qR = c_norm2*cp.real(
        (lam*chi[None, :])*cp.conj(lam*chi[None, :])
    )
    rho_R = cp.sum(
        rho_qR, axis=0, dtype=model.reduction_real_dtype
    )*model.dq
    if model.coupling_mask_backend == "flat_top":
        mask_phi = flat_top_support_mask(
            rho_qR, model.flat_top_on_phi,
            model.flat_top_transition_decades, model,
        )
        mask_lam = flat_top_support_mask(
            rho_R, model.flat_top_on_lam,
            model.flat_top_transition_decades, model,
        )
    else:
        mask_phi = occupied_support_mask(rho_qR, mask_threshold_phi, model)
        mask_lam = occupied_support_mask(rho_R, mask_threshold_lam, model)
    tail_gate_phi = deep_tail_gate(
        rho_qR, model.deep_tail_zero_threshold, model
    )
    tail_gate_lam = deep_tail_gate(
        rho_R, model.deep_tail_zero_threshold, model
    )
    p_q_lam = -1j*derivative(lam, model.dq, axis=0)
    lam_phase_q, lam_logamp_q = logarithmic_components(
        lam, model.dq, axis=0, model=model, numerical_floor=ratio_floor,
        momentum_factor=p_q_lam,
    )
    lam_phase_R, lam_logamp_R = logarithmic_components(
        lam, model.dR, axis=1, model=model, numerical_floor=ratio_floor,
        momentum_factor=p_R_lam,
    )
    p_R_chi = -1j*derivative(chi, model.dR, axis=0)
    chi_phase_R, chi_logamp_R = logarithmic_components(
        chi, model.dR, axis=0, model=model, numerical_floor=ratio_floor,
        momentum_factor=p_R_chi,
    )
    weak_diag = {}
    if model.log_derivative_backend == "weak":
        xi = lam*chi[None, :]
        xi_logamp_q, dq_diag = weak_log_amplitude_gradient(
            xi, model.dq, 0, model
        )
        xi_logamp_R, dR_diag = weak_log_amplitude_gradient(
            xi, model.dR, 1, model
        )
        chi_logamp_used, dc_diag = weak_log_amplitude_gradient(
            chi, model.dR, 0, model
        )
        weak_diag = dict(
            weak_log_residual_q_xi=dq_diag["weak_log_residual"],
            weak_log_residual_R_xi=dR_diag["weak_log_residual"],
            weak_log_residual_R_chi=dc_diag["weak_log_residual"],
            weak_log_iterations=cp.maximum(
                dq_diag["weak_log_iterations"], cp.maximum(
                    dR_diag["weak_log_iterations"],
                    dc_diag["weak_log_iterations"],
                ),
            ),
            weak_log_unconverged_lines=(
                dq_diag["weak_log_unconverged_lines"]
                +dR_diag["weak_log_unconverged_lines"]
                +dc_diag["weak_log_unconverged_lines"]
            ),
        )
    else:
        xi_logamp_q = lam_logamp_q
        xi_logamp_R = lam_logamp_R+chi_logamp_R[None, :]
        chi_logamp_used = chi_logamp_R

    p_q = -1j*cov_gradient_q
    p2_q = -cov_second_q
    p_R = -1j*cov_gradient_R
    p2_R = -cov_second_R
    if model.coupling_mask_backend == "flat_top":
        coefficient_q = gated_values(
            lam_phase_q+a-1j*xi_logamp_q, mask_phi
        )
        coefficient_R = gated_values(
            lam_phase_R+chi_phase_R[None, :]+b-1j*xi_logamp_R,
            mask_phi,
        )
    else:
        coefficient_q = (
            gated_values(lam_phase_q, tail_gate_phi)+a
            -1j*mask_phi*gated_values(xi_logamp_q, tail_gate_phi)
        )
        coefficient_R = (
            gated_values(lam_phase_R+chi_phase_R[None, :], tail_gate_phi)+b
            -1j*mask_phi*gated_values(xi_logamp_R, tail_gate_phi)
        )
    u_c = (
        0.5*p2_q+coefficient_q[None, :, :]*p_q
    )/model.proton_mass+(
        0.5*p2_R+coefficient_R[None, :, :]*p_R
    )/model.heavy_mass
    u_c, gamma_c, raw_rate_c, corrected_rate_c = remove_local_norm_generator(
        coefficients, u_c, 1.0, axis=0, model=model
    )
    hbo_c = basis.energies*coefficients
    epsilon_1 = cp.sum(
        cp.conj(coefficients)*(hbo_c+u_c), axis=0,
        dtype=model.reduction_complex_dtype,
    ).real.astype(model.real_dtype, copy=False)

    base_lam = proton_base_operator(
        lam, a, b, alpha, chi_phase_R, chi_logamp_used, mask_lam,
        tail_gate_lam, model
    )
    hpr_raw = base_lam+epsilon_1*lam+1j*gamma_c*lam
    hpr, gamma_lam, raw_rate_lam, corrected_rate_lam = (
        remove_local_norm_generator(lam, hpr_raw, model.dq, axis=0, model=model)
    )
    epsilon_2 = cp.sum(
        cp.conj(lam)*hpr, axis=0, dtype=model.reduction_complex_dtype,
    ).real*model.dq
    epsilon_2 = epsilon_2.astype(model.real_dtype, copy=False)
    return dict(
        a=a, b=b, alpha=alpha, epsilon_1=epsilon_1, epsilon_2=epsilon_2,
        u_c=u_c, hpr_lam=hpr, gamma_c=gamma_c, gamma_lam=gamma_lam,
        mask_phi=mask_phi, mask_lam=mask_lam,
        tail_gate_phi=tail_gate_phi, tail_gate_lam=tail_gate_lam,
        p_R_chi=p_R_chi,
        raw_rate_phi=raw_rate_c, corrected_rate_phi=corrected_rate_c,
        raw_rate_lam=raw_rate_lam, corrected_rate_lam=corrected_rate_lam,
        raw_logamp_phi=cp.maximum(
            cp.abs(lam_logamp_q),
            cp.abs(lam_logamp_R)+cp.abs(chi_logamp_R)[None, :],
        ),
        effective_logamp_phi=cp.maximum(
            cp.abs(mask_phi*(xi_logamp_q if model.coupling_mask_backend == "flat_top"
                             else gated_values(xi_logamp_q, tail_gate_phi))),
            cp.abs(mask_phi*(xi_logamp_R if model.coupling_mask_backend == "flat_top"
                             else gated_values(xi_logamp_R, tail_gate_phi))),
        ),
        suppressed_probability_phi=suppressed_probability(
            rho_qR, mask_phi, model.dq*model.dR, model
        ),
        suppressed_probability_lam=suppressed_probability(
            rho_R, mask_lam, model.dR, model
        ),
        deep_tail_suppressed_probability_phi=suppressed_probability(
            rho_qR, tail_gate_phi, model.dq*model.dR, model
        ),
        deep_tail_suppressed_probability_lam=suppressed_probability(
            rho_R, tail_gate_lam, model.dR, model
        ),
        deep_tail_zero_fraction_phi=cp.mean(tail_gate_phi == 0.0),
        deep_tail_zero_fraction_lam=cp.mean(tail_gate_lam == 0.0),
        **weak_diag,
    )


def project_product_residual_bh(
    coefficients, lam, chi, dc, dlam, dchi, model, basis,
):
    xi = lam*chi[None, :]
    y = coefficients*xi[None, :, :]
    product_rhs = dc*xi[None, :, :]+coefficients*(
        dlam*chi[None, :]+lam*dchi[None, :]
    )[None, :, :]
    if basis.link_kernel == "fused":
        _, q_second = _launch_fused_transport(
            y, basis, model.dq, 1,
            write_transports=False, write_first=False, write_second=True,
        )
        # Materialize the q contribution before reusing the one D2 workspace
        # for R.  This is algebraically identical to the expression below.
        target = 0.5j*q_second/model.proton_mass
        _, R_second = _launch_fused_transport(
            y, basis, model.dR, 2,
            write_transports=False, write_first=False, write_second=True,
        )
        target = target+0.5j*R_second/model.heavy_mass
    else:
        target = -1j*(
            -0.5*projected_link_derivatives(
                y, basis, model.dq, axis=1
            )[1]/model.proton_mass
            -0.5*projected_link_derivatives(
                y, basis, model.dR, axis=2
            )[1]/model.heavy_mass
        )
    residual = target-product_rhs
    xi_density = cp.real(xi*cp.conj(xi))
    c_norm2 = cp.sum(
        cp.real(coefficients*cp.conj(coefficients)), axis=0,
        dtype=model.reduction_real_dtype,
    )
    physical_qR = c_norm2*xi_density
    tail_gate_phi = deep_tail_gate(
        physical_qR, model.deep_tail_zero_threshold, model
    )
    tiny = cp.asarray(1.0e-30, dtype=xi_density.dtype)
    delta_xi = cp.sum(
        cp.conj(coefficients)*residual, axis=0,
        dtype=model.reduction_complex_dtype,
    )/cp.maximum(c_norm2, tiny)
    delta_xi = gated_values(
        delta_xi.astype(model.complex_dtype, copy=False), tail_gate_phi
    )
    perp_c = residual-coefficients*delta_xi[None, :, :]
    xi_peak = cp.maximum(cp.max(xi_density), tiny)
    support = xi_density/(
        xi_density+model.product_projection_floor_phi*xi_peak+tiny
    )
    if model.product_projection_backend == "weighted_tikhonov":
        ridge = model.projection_tau_phi*xi_peak/(
            support+model.projection_support_epsilon
        )
        inverse_xi = support*cp.conj(xi)/(
            support*xi_density+ridge+tiny
        )
    else:
        inverse_xi = cp.conj(xi)/(
            xi_density+model.product_projection_floor_phi*xi_peak
        )
    inverse_xi = gated_values(inverse_xi, tail_gate_phi)
    delta_c = perp_c*inverse_xi[None, :, :]
    chi_density = cp.real(chi*cp.conj(chi))
    physical_R = cp.sum(
        physical_qR, axis=0, dtype=model.reduction_real_dtype
    )*model.dq
    tail_gate_lam = deep_tail_gate(
        physical_R, model.deep_tail_zero_threshold, model
    )
    lam_norm2 = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    parallel_chi = cp.sum(
        cp.conj(lam)*delta_xi, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dq/cp.maximum(lam_norm2, tiny)
    parallel_chi = gated_values(
        parallel_chi.astype(model.complex_dtype, copy=False), tail_gate_lam
    )
    perp_lam = delta_xi-lam*parallel_chi[None, :]
    chi_peak = cp.maximum(cp.max(chi_density), tiny)
    support_R = chi_density/(
        chi_density+model.product_projection_floor_lam*chi_peak+tiny
    )
    if model.product_projection_backend == "weighted_tikhonov":
        ridge_R = model.projection_tau_lam*chi_peak/(
            support_R+model.projection_support_epsilon
        )
        inverse_chi = support_R*cp.conj(chi)/(
            support_R*chi_density+ridge_R+tiny
        )
        chi_shrink = support_R/(
            support_R
            +model.projection_tau_chi/(
                support_R+model.projection_support_epsilon
            )+tiny
        )
        delta_chi = gated_values(chi_shrink, tail_gate_lam)*parallel_chi
    else:
        inverse_chi = cp.conj(chi)/(
            chi_density+model.product_projection_floor_lam*chi_peak
        )
        delta_chi = gated_values(parallel_chi, tail_gate_lam)
    inverse_chi = gated_values(inverse_chi, tail_gate_lam)
    delta_lam = perp_lam*inverse_chi[None, :]
    dc = dc+delta_c
    dlam = dlam+delta_lam
    dchi = dchi+delta_chi
    corrected = dc*xi[None, :, :]+coefficients*(
        dlam*chi[None, :]+lam*dchi[None, :]
    )[None, :, :]
    volume = model.dq*model.dR
    l2 = lambda value: cp.sqrt(cp.sum(
        cp.real(value*cp.conj(value)), dtype=model.reduction_real_dtype
    )*volume)
    target_l2 = cp.maximum(l2(target), tiny)
    diagnostics = dict(
        max_product_residual_l2=l2(residual),
        max_effective_product_residual_l2=l2(target-corrected),
        max_relative_product_projection_l2=l2(corrected-product_rhs)/target_l2,
        max_abs_product_correction_phi=cp.max(cp.abs(delta_c)),
        max_abs_product_correction_lam=cp.max(cp.abs(delta_lam)),
        max_abs_product_correction_chi=cp.max(cp.abs(delta_chi)),
        max_inverse_support_product_correction_phi=cp.sqrt(cp.sum(
            cp.real(delta_c*cp.conj(delta_c))/(
                support[None, :, :]+model.projection_support_epsilon
            ), dtype=model.reduction_real_dtype,
        )*volume),
        max_inverse_support_product_correction_lam=cp.sqrt(cp.sum(
            cp.real(delta_lam*cp.conj(delta_lam))/(
                support_R[None, :]+model.projection_support_epsilon
            ), dtype=model.reduction_real_dtype,
        )*volume),
        max_inverse_support_product_correction_chi=cp.sqrt(cp.sum(
            cp.real(delta_chi*cp.conj(delta_chi))/(
                support_R+model.projection_support_epsilon
            ), dtype=model.reduction_real_dtype,
        )*model.dR),
        max_abs_full_norm_rate_before_product_projection=cp.abs(
            2.0*cp.sum(
                cp.conj(y)*product_rhs,
                dtype=model.reduction_complex_dtype,
            ).real*volume
        ),
        max_abs_full_norm_rate_after_product_projection=cp.abs(
            2.0*cp.sum(
                cp.conj(y)*corrected,
                dtype=model.reduction_complex_dtype,
            ).real*volume
        ),
    )
    return dc, dlam, dchi, diagnostics


def coupled_rhs_bh(
    coefficients, lam, chi, model, basis, ratio_floor,
    mask_threshold_phi, mask_threshold_lam,
):
    fields = instantaneous_functionals_bh(
        coefficients, lam, chi, model, basis, ratio_floor,
        mask_threshold_phi, mask_threshold_lam,
    )
    dc = -1j*(fields["u_c"]-fields["epsilon_1"][None, :, :]*coefficients)
    dlam = -1j*(fields["hpr_lam"]-fields["epsilon_2"][None, :]*lam)
    p2chi = covariant_square(
        chi, fields["alpha"], model.dR, axis=0, sign=+1,
        momentum_field=fields["p_R_chi"],
    )
    dchi = -1j*(
        0.5*p2chi/model.heavy_mass+fields["epsilon_2"]*chi
    )+fields["gamma_lam"]*chi
    dc, dlam, dchi, diagnostics = project_product_residual_bh(
        coefficients, lam, chi, dc, dlam, dchi, model, basis
    )
    diagnostics.update(
        max_abs_gamma_phi=cp.max(cp.abs(fields["gamma_c"])),
        max_abs_gamma_lam=cp.max(cp.abs(fields["gamma_lam"])),
        max_abs_support_gamma_phi=cp.max(cp.abs(
            fields["tail_gate_phi"]*fields["mask_phi"]*fields["gamma_c"]
        )),
        max_abs_support_gamma_lam=cp.max(cp.abs(
            fields["tail_gate_lam"]*fields["mask_lam"]*fields["gamma_lam"]
        )),
        max_raw_logamp_phi=cp.max(cp.abs(fields["raw_logamp_phi"])),
        max_effective_logamp_phi=cp.max(cp.abs(
            fields["effective_logamp_phi"]
        )),
        suppressed_probability_phi=fields["suppressed_probability_phi"],
        suppressed_probability_lam=fields["suppressed_probability_lam"],
        deep_tail_suppressed_probability_phi=fields[
            "deep_tail_suppressed_probability_phi"
        ],
        deep_tail_suppressed_probability_lam=fields[
            "deep_tail_suppressed_probability_lam"
        ],
        deep_tail_zero_fraction_phi=fields["deep_tail_zero_fraction_phi"],
        deep_tail_zero_fraction_lam=fields["deep_tail_zero_fraction_lam"],
    )
    for key in (
        "weak_log_residual_q_xi", "weak_log_residual_R_xi",
        "weak_log_residual_R_chi", "weak_log_iterations",
        "weak_log_unconverged_lines",
    ):
        if key in fields:
            diagnostics[f"max_{key}"] = cp.max(fields[key])
    return dc, dlam, dchi, fields, diagnostics


def full_step_bh(
    coefficients, lam, chi, dt, model, basis, ratio_floor,
    mask_threshold_phi, mask_threshold_lam, *,
    collect_pnc_norm_diagnostics=False,
):
    phase = cp.exp(-0.5j*dt*basis.energies).astype(
        model.complex_dtype, copy=False
    )
    coefficients = coefficients*phase
    stages = []

    def rhs(c, l, h):
        result = coupled_rhs_bh(
            c, l, h, model, basis, ratio_floor,
            mask_threshold_phi, mask_threshold_lam,
        )
        stages.append(result[4])
        return result[:3]

    k1 = rhs(coefficients, lam, chi)
    k2 = rhs(
        coefficients+0.5*dt*k1[0], lam+0.5*dt*k1[1], chi+0.5*dt*k1[2]
    )
    k3 = rhs(
        coefficients+0.5*dt*k2[0], lam+0.5*dt*k2[1], chi+0.5*dt*k2[2]
    )
    k4 = rhs(coefficients+dt*k3[0], lam+dt*k3[1], chi+dt*k3[2])
    coefficients = coefficients+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0
    lam = lam+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0
    chi = chi+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0
    projected = pnc_project_coefficients(
        coefficients, lam, chi, model,
        return_diagnostics=collect_pnc_norm_diagnostics,
    )
    coefficients, lam, chi, correction = projected[:4]
    pnc_diag1 = projected[4] if collect_pnc_norm_diagnostics else {}
    coefficients = coefficients*phase
    coefficients, lam, chi, correction2 = pnc_project_coefficients(
        coefficients, lam, chi, model,
    )
    merged = {}
    for key in stages[0]:
        value = cp.asarray(0.0, dtype=model.reduction_real_dtype)
        for stage in stages:
            value = cp.maximum(value, stage.get(key, 0.0))
        merged[key] = value
    merged.update(pnc_diag1)
    return coefficients, lam, chi, cp.maximum(correction, correction2), merged
