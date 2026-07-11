import torch as tc

from latticetn.charge_sectors import (
    hubbard_hard_sector_product_mps,
    max_forbidden_abs,
    sector_canonical_residual,
    sector_left_canonicalize,
    sector_normalize_center,
    spinless_hard_sector_product_mps,
)
from latticetn.sector_observables import (
    hubbard_sector_leakage_report,
    sector_leakage_report,
)


DTYPE = tc.complex128


def _unit(mps):
    psi = mps.to_dense()
    return psi / psi.norm()


def _random_hermitian(dim, seed):
    tc.manual_seed(seed)
    x = tc.randn(dim, dim, dtype=DTYPE)
    return x + x.conj().t()


def _energy(psi, h):
    return (tc.vdot(psi, h @ psi) / tc.vdot(psi, psi)).real


def test_spinless_block_qr_preserves_state_energy_sector_and_mask():
    camps = spinless_hard_sector_product_mps(6, 3, chi=4, dtype=DTYPE)
    before = _unit(camps.mps)
    h = _random_hermitian(before.numel(), 1210)
    out = sector_left_canonicalize(camps)
    after = _unit(out.mps)
    assert tc.linalg.vector_norm(after - before) < 1e-12
    assert abs(float(_energy(after, h) - _energy(before, h))) < 1e-12
    assert sector_canonical_residual(out, center=5) < 1e-12
    assert max_forbidden_abs(out.mps, out.masks) == 0.0
    report = sector_leakage_report(out.mps, target_n=3)
    assert report["abs_error"] < 1e-12
    assert report["variance"] < 1e-12


def test_hubbard_block_qr_preserves_state_energy_sector_and_mask():
    camps = hubbard_hard_sector_product_mps(4, 2, 2, chi=5, dtype=DTYPE)
    before = _unit(camps.mps)
    h = _random_hermitian(before.numel(), 1211)
    out = sector_left_canonicalize(camps)
    after = _unit(out.mps)
    assert tc.linalg.vector_norm(after - before) < 1e-12
    assert abs(float(_energy(after, h) - _energy(before, h))) < 1e-12
    assert sector_canonical_residual(out, center=3) < 1e-12
    assert max_forbidden_abs(out.mps, out.masks) == 0.0
    report = hubbard_sector_leakage_report(out.mps, target_nup=2, target_ndown=2)
    assert report["n_up_abs_error"] < 1e-12
    assert report["n_down_abs_error"] < 1e-12
    assert report["variance_n_tot"] < 1e-12


def test_sector_center_normalization_has_unit_norm_and_preserves_masks():
    for camps, center in (
        (spinless_hard_sector_product_mps(6, 3, chi=4, dtype=DTYPE), 3),
        (hubbard_hard_sector_product_mps(4, 2, 2, chi=5, dtype=DTYPE), 2),
    ):
        before = _unit(camps.mps)
        out = sector_normalize_center(camps, center=center)
        assert abs(float(out.mps.to_dense().norm()) - 1.0) < 1e-12
        assert tc.linalg.vector_norm(out.mps.to_dense() - before) < 1e-12
        assert sector_canonical_residual(out, center=center) < 1e-12
        assert max_forbidden_abs(out.mps, out.masks) == 0.0
