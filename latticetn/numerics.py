"""Strict numerical validation helpers for scientific tensor contractions.

Roundoff-sized Hermiticity/positivity violations are corrected explicitly.
Non-finite values and violations larger than the dtype-scaled tolerance fail
the run instead of being converted into plausible-looking diagnostics.
"""

from __future__ import annotations

import torch as tc


def _real_dtype(dtype: tc.dtype) -> tc.dtype:
    if dtype in {tc.complex128, tc.float64}:
        return tc.float64
    return tc.float32


def _roundoff_tolerance(value: tc.Tensor, factor: float = 256.0) -> tc.Tensor:
    eps = tc.finfo(_real_dtype(value.dtype)).eps
    return factor * eps * tc.maximum(
        tc.ones((), dtype=value.real.dtype, device=value.device),
        value.real.abs(),
    )


def require_finite(value: tc.Tensor, *, name: str) -> tc.Tensor:
    """Return ``value`` after rejecting any NaN or infinity."""

    if not bool(tc.isfinite(value).all()):
        raise FloatingPointError(f"{name} is non-finite: {value}")
    return value


def real_if_hermitian(
    value: tc.Tensor,
    *,
    name: str,
    tolerance: float | None = None,
) -> tc.Tensor:
    """Return the real view of a Hermitian scalar, rejecting a real violation."""

    require_finite(value, name=name)
    if not tc.is_complex(value):
        return value
    tol = (
        tc.as_tensor(tolerance, dtype=value.real.dtype, device=value.device)
        if tolerance is not None
        else _roundoff_tolerance(value)
    )
    if bool(value.imag.abs() > tol):
        raise FloatingPointError(
            f"{name} has a significant imaginary component: "
            f"real={value.real}, imag={value.imag}, tolerance={tol}"
        )
    return value.real


def nonnegative_if_roundoff(
    value: tc.Tensor,
    *,
    name: str,
    scale: tc.Tensor | float | None = None,
    tolerance: float | None = None,
) -> tc.Tensor:
    """Correct a tiny negative real scalar and reject a significant one."""

    value = real_if_hermitian(value, name=name, tolerance=tolerance)
    require_finite(value, name=name)
    if tolerance is None:
        scale_t = (
            tc.as_tensor(1.0, dtype=value.dtype, device=value.device)
            if scale is None
            else tc.as_tensor(scale, dtype=value.dtype, device=value.device).abs()
        )
        eps = tc.finfo(value.dtype).eps
        tol = 256.0 * eps * tc.maximum(tc.ones_like(scale_t), scale_t)
    else:
        tol = tc.as_tensor(tolerance, dtype=value.dtype, device=value.device)
    if bool(value < -tol):
        raise FloatingPointError(
            f"{name} is significantly negative: {value}, tolerance={tol}"
        )
    return tc.where(value < 0, tc.zeros_like(value), value)


def positive(value: tc.Tensor, *, name: str) -> tc.Tensor:
    """Return a finite positive scalar or fail."""

    value = real_if_hermitian(value, name=name)
    if bool(value <= 0):
        raise FloatingPointError(f"{name} must be positive, got {value}")
    return value


def truncation_error(weights: tc.Tensor, kept: int, *, name: str) -> float:
    """Return discarded relative weight with strict finite/positivity checks."""

    require_finite(weights, name=f"{name} singular-value weights")
    if weights.ndim != 1 or not 1 <= kept <= weights.numel():
        raise ValueError(
            f"{name} kept count must be in [1, {weights.numel()}], got {kept}"
        )
    total_t = weights.sum()
    total = float(positive(total_t, name=f"{name} total spectral weight").cpu())
    kept_weight = float(require_finite(
        weights[:kept].sum(), name=f"{name} retained spectral weight"
    ).cpu())
    error = (total - kept_weight) / total
    tolerance = 256.0 * tc.finfo(weights.dtype).eps
    if not (-tolerance <= error <= 1.0 + tolerance):
        raise FloatingPointError(
            f"{name} truncation error is outside [0, 1]: {error}"
        )
    return min(1.0, max(0.0, error))


__all__ = [
    "require_finite",
    "real_if_hermitian",
    "nonnegative_if_roundoff",
    "positive",
    "truncation_error",
]
