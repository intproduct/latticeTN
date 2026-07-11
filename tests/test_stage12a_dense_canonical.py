import torch as tc

from latticetn.canonical import (
    canonical_residual,
    left_canonicalize,
    mixed_canonicalize,
    normalize_center,
    svd_compress,
)
from latticetn.mps import MPS
from latticetn.operators import tfi_dense
from standalone_stage12a_gauge_retraction_test import run_case


DTYPE = tc.complex128


def _normalized_dense(mps):
    psi = mps.to_dense()
    return psi / psi.norm()


def _rayleigh(psi, h):
    return (tc.vdot(psi, h @ psi) / tc.vdot(psi, psi)).real


def test_exact_qr_and_svd_preserve_dense_state_and_energy():
    tc.manual_seed(1201)
    mps = MPS(5, 2, 4, dtype=DTYPE)
    h = tfi_dense(5, J=0.7, h=1.2, dtype=DTYPE)
    before = _normalized_dense(mps)
    e_before = _rayleigh(before, h)
    for method in ("qr", "svd"):
        out = left_canonicalize(mps, method=method)
        after = _normalized_dense(out)
        assert tc.linalg.vector_norm(after - before) < 1e-12
        assert abs(float((_rayleigh(after, h) - e_before).detach())) < 1e-12
        assert canonical_residual(out, center=out.N - 1) < 1e-12


def test_mixed_center_normalization_produces_unit_physical_norm():
    tc.manual_seed(1202)
    mps = MPS(6, 2, 5, dtype=DTYPE)
    mixed = mixed_canonicalize(mps, center=3)
    assert canonical_residual(mixed, center=3) < 1e-12
    normalized = normalize_center(mixed, center=3)
    assert abs(float(normalized.to_dense().norm()) - 1.0) < 1e-12
    assert canonical_residual(normalized, center=3) < 1e-12


def test_truncated_svd_negative_control_changes_state_and_energy():
    tc.manual_seed(1203)
    mps = MPS(5, 2, 4, dtype=DTYPE)
    h = tfi_dense(5, J=0.9, h=0.4, dtype=DTYPE)
    before = _normalized_dense(mps)
    compressed, _ = svd_compress(mps, chi=1)
    after = _normalized_dense(compressed)
    assert tc.linalg.vector_norm(after - before) > 1e-4
    assert abs(float((_rayleigh(after, h) - _rayleigh(before, h)).detach())) > 1e-6


def test_ad_update_then_exact_canonicalization_preserves_updated_state():
    tc.manual_seed(1204)
    mps = MPS(4, 2, 4, dtype=DTYPE)
    h = tfi_dense(4, J=1.0, h=0.8, dtype=DTYPE)
    opt = tc.optim.SGD(mps.parameters(), lr=1e-3)
    opt.zero_grad()
    psi = mps.to_dense()
    _rayleigh(psi, h).backward()
    opt.step()
    updated = _normalized_dense(mps).detach()
    retracted = left_canonicalize(mps)
    assert tc.linalg.vector_norm(_normalized_dense(retracted) - updated) < 1e-12


def test_periodic_qr_retraction_stabilizes_norm_and_canonical_residual():
    h = tfi_dense(4, J=1.0, h=0.8, dtype=DTYPE)
    _, vectors = tc.linalg.eigh(h)
    kwargs = dict(N=4, chi=4, steps=6, lr=0.01, interval=2, seed=1212)
    pure = run_case("pure", h, vectors[:, 0], **kwargs)
    qr = run_case("qr", h, vectors[:, 0], **kwargs)
    assert abs(qr["physical_norm"] - 1.0) < abs(pure["physical_norm"] - 1.0)
    assert qr["canonical_residual"] < pure["canonical_residual"]
    assert abs(qr["physical_norm"] - 1.0) < 1e-12
    assert qr["canonical_residual"] < 1e-12
