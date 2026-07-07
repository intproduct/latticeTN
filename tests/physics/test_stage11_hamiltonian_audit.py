from __future__ import annotations

import itertools

import torch as tc

from latticetn.hamiltonian_builder import build_mpo
from latticetn.model_registry import build_model_spec
from latticetn.mpo import MPO


DTYPE = tc.complex128


def _kron_all(mats: list[tc.Tensor]) -> tc.Tensor:
    out = mats[0]
    for mat in mats[1:]:
        out = tc.kron(out, mat)
    return out


def _spin_ops() -> dict[str, tc.Tensor]:
    sx = tc.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    sy = tc.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    sz = tc.tensor([[1, 0], [0, -1]], dtype=DTYPE)
    return {
        "I": tc.eye(2, dtype=DTYPE),
        "Sx": sx / 2,
        "Sy": sy / 2,
        "Sz": sz / 2,
    }


def _fermion_ops() -> dict[str, tc.Tensor]:
    c = tc.tensor([[0, 1], [0, 0]], dtype=DTYPE)
    cdag = tc.tensor([[0, 0], [1, 0]], dtype=DTYPE)
    n = cdag @ c
    return {
        "I": tc.eye(2, dtype=DTYPE),
        "c": c,
        "cdag": cdag,
        "n": n,
        "F": tc.diag(tc.tensor([1, -1], dtype=DTYPE)),
    }


def _dense_heisenberg_independent(N: int, J: float) -> tc.Tensor:
    ops = _spin_ops()
    H = tc.zeros((2**N, 2**N), dtype=DTYPE)
    for i in range(N - 1):
        for name in ["Sx", "Sy", "Sz"]:
            mats = [ops["I"]] * N
            mats[i] = ops[name]
            mats[i + 1] = ops[name]
            H = H + J * _kron_all(mats)
    return H


def _dense_tfi_independent(N: int, J: float, h: float) -> tc.Tensor:
    ops = _spin_ops()
    H = tc.zeros((2**N, 2**N), dtype=DTYPE)
    for i in range(N - 1):
        mats = [ops["I"]] * N
        mats[i] = ops["Sz"]
        mats[i + 1] = ops["Sz"]
        H = H - J * _kron_all(mats)
    for i in range(N):
        mats = [ops["I"]] * N
        mats[i] = ops["Sx"]
        H = H - h * _kron_all(mats)
    return H


def _global_spinless(op: tc.Tensor, site: int, N: int) -> tc.Tensor:
    ops = _fermion_ops()
    mats = []
    for k in range(N):
        if k < site:
            mats.append(ops["F"])
        elif k == site:
            mats.append(op)
        else:
            mats.append(ops["I"])
    return _kron_all(mats)


def _dense_spinless_independent(N: int, t: float, V: float, mu: float) -> tc.Tensor:
    ops = _fermion_ops()
    I = ops["I"]
    nmh = ops["n"] - 0.5 * I
    H = tc.zeros((2**N, 2**N), dtype=DTYPE)
    for i in range(N - 1):
        H = H - t * (
            _global_spinless(ops["cdag"], i, N) @ _global_spinless(ops["c"], i + 1, N)
            + _global_spinless(ops["cdag"], i + 1, N) @ _global_spinless(ops["c"], i, N)
        )
        mats = [I] * N
        mats[i] = nmh
        mats[i + 1] = nmh
        H = H + V * _kron_all(mats)
    for i in range(N):
        mats = [I] * N
        mats[i] = nmh
        H = H - mu * _kron_all(mats)
    return H


def _hubbard_local_ops_independent() -> dict[str, tc.Tensor]:
    I = tc.eye(4, dtype=DTYPE)
    cup = tc.zeros((4, 4), dtype=DTYPE)
    cdagup = tc.zeros((4, 4), dtype=DTYPE)
    cdown = tc.zeros((4, 4), dtype=DTYPE)
    cdagdown = tc.zeros((4, 4), dtype=DTYPE)
    # basis: |0>, |up>, |down>, |up down>
    cup[0, 1] = 1
    cup[2, 3] = 1
    cdagup[1, 0] = 1
    cdagup[3, 2] = 1
    cdown[0, 2] = 1
    cdown[1, 3] = -1
    cdagdown[2, 0] = 1
    cdagdown[3, 1] = -1
    nup = tc.diag(tc.tensor([0, 1, 0, 1], dtype=DTYPE))
    ndown = tc.diag(tc.tensor([0, 0, 1, 1], dtype=DTYPE))
    parity = tc.diag(tc.tensor([1, -1, -1, 1], dtype=DTYPE))
    return {
        "I": I,
        "cup": cup,
        "cdagup": cdagup,
        "cdown": cdown,
        "cdagdown": cdagdown,
        "nup": nup,
        "ndown": ndown,
        "P": parity,
    }


def _global_hubbard(local: tc.Tensor, site: int, N: int) -> tc.Tensor:
    ops = _hubbard_local_ops_independent()
    mats = []
    for k in range(N):
        if k < site:
            mats.append(ops["P"])
        elif k == site:
            mats.append(local)
        else:
            mats.append(ops["I"])
    return _kron_all(mats)


def _dense_hubbard_independent(N: int, t: float, U: float, mu: float, h: float) -> tc.Tensor:
    ops = _hubbard_local_ops_independent()
    I = ops["I"]
    nup = ops["nup"]
    ndown = ops["ndown"]
    H = tc.zeros((4**N, 4**N), dtype=DTYPE)
    for i in range(N - 1):
        for c, cdag in [(ops["cup"], ops["cdagup"]), (ops["cdown"], ops["cdagdown"])]:
            H = H - t * (
                _global_hubbard(cdag, i, N) @ _global_hubbard(c, i + 1, N)
                + _global_hubbard(cdag, i + 1, N) @ _global_hubbard(c, i, N)
            )
    onsite = U * ((nup - 0.5 * I) @ (ndown - 0.5 * I)) - mu * (nup + ndown - I) - h * (nup - ndown)
    for i in range(N):
        mats = [I] * N
        mats[i] = onsite
        H = H + _kron_all(mats)
    return H


def test_spin_operator_conventions_are_spin_not_pauli():
    H = _dense_heisenberg_independent(2, J=1.0)
    eigvals = tc.linalg.eigvalsh(H).real
    expected = tc.tensor([-0.75, 0.25, 0.25, 0.25], dtype=tc.float64)
    assert tc.allclose(eigvals, expected, atol=1e-12)


def test_heisenberg_and_tfi_mpo_match_independent_dense_references():
    for N, J in itertools.product([4, 6], [0.5, 1.0]):
        H_mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_heisenberg(J=J).to_dense()
        H_ref = _dense_heisenberg_independent(N, J=J)
        assert tc.allclose(H_mpo, H_ref, atol=1e-10), float((H_mpo - H_ref).abs().max())
    for J, h in [(1.0, 0.5), (0.7, 1.3)]:
        H_mpo = MPO.from_bonds(4, 2, dtype=DTYPE).generate_tfi(J=J, h=h).to_dense()
        H_ref = _dense_tfi_independent(4, J=J, h=h)
        assert tc.allclose(H_mpo, H_ref, atol=1e-10), float((H_mpo - H_ref).abs().max())


def test_spinless_mpo_matches_independent_jw_dense_and_hopping_signs():
    N = 4
    H_mpo = MPO.from_bonds(N, 2, dtype=DTYPE).generate_spinless_fermion(t=1.0, V=0.4, mu=-0.2).to_dense()
    H_ref = _dense_spinless_independent(N, t=1.0, V=0.4, mu=-0.2)
    assert tc.allclose(H_mpo, H_ref, atol=1e-10)

    hop = _global_spinless(_fermion_ops()["cdag"], 1, N) @ _global_spinless(_fermion_ops()["c"], 2, N)
    assert hop[0b0100, 0b0010] == 1
    assert hop[0b1100, 0b1010] == 1
    assert H_mpo[0b0100, 0b0010] == -1
    assert H_mpo[0b1100, 0b1010] == -1


def test_hubbard_mpo_matches_independent_jw_dense_and_hopping_signs():
    N = 3
    H_mpo = MPO.from_bonds(N, 4, dtype=DTYPE).generate_hubbard(t=1.0, U=2.0, mu=0.3, h=-0.2).to_dense()
    H_ref = _dense_hubbard_independent(N, t=1.0, U=2.0, mu=0.3, h=-0.2)
    assert tc.allclose(H_mpo, H_ref, atol=1e-10), float((H_mpo - H_ref).abs().max())

    ops = _hubbard_local_ops_independent()
    up_hop = _global_hubbard(ops["cdagup"], 1, N) @ _global_hubbard(ops["cup"], 2, N)
    down_hop = _global_hubbard(ops["cdagdown"], 1, N) @ _global_hubbard(ops["cdown"], 2, N)
    # state index is base-4 site-major. These checks isolate the raw
    # cdag_{1,sigma} c_{2,sigma} matrix elements before the Hamiltonian's -t.
    assert up_hop[0 * 16 + 1 * 4 + 0, 0 * 16 + 0 * 4 + 1] == 1
    assert down_hop[0 * 16 + 2 * 4 + 0, 0 * 16 + 0 * 4 + 2] == 1
    assert down_hop[0 * 16 + 2 * 4 + 1, 0 * 16 + 0 * 4 + 3] == -1


def test_modelspec_to_mpo_matches_independent_dense_references():
    cases = [
        (build_model_spec("heisenberg", 4, {"J": 0.8}), _dense_heisenberg_independent(4, 0.8)),
        (build_model_spec("tfi", 4, {"J": 0.9, "h": 1.1}), _dense_tfi_independent(4, 0.9, 1.1)),
        (build_model_spec("spinless_tv", 4, {"t": 0.7, "V": 0.3, "mu": -0.2}), _dense_spinless_independent(4, 0.7, 0.3, -0.2)),
        (build_model_spec("hubbard", 3, {"t": 0.6, "U": 2.0, "mu": 0.1, "h": 0.2}), _dense_hubbard_independent(3, 0.6, 2.0, 0.1, 0.2)),
    ]
    for spec, H_ref in cases:
        H_mpo = build_mpo(spec, dtype=DTYPE, device="cpu").to_dense()
        assert tc.allclose(H_mpo, H_ref, atol=1e-10), spec.name
