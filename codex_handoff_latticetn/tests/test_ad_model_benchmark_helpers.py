"""CPU-only helper tests for the generic AD model benchmark runner.

These tests intentionally stay small. They check runner wiring, product-state
initialization conventions, and the no-ED/no-DMRG result flags. They are not a
large physics benchmark.
"""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ad_model_benchmark.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_ad_model_benchmark", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_product_init_conventions():
    r = load_runner()

    spin_mps = r.make_product_mps("heisenberg", N=4, init="neel",
                                  dtype=tc.complex128, device="cpu")
    assert spin_mps.dim == 2
    assert [int(A.abs().argmax().item()) for A in spin_mps.tensors] == [0, 1, 0, 1]
    assert r.bond_dims(spin_mps) == [1, 1, 1]

    fermion_mps = r.make_product_mps("spinless_tv", N=4, init="cdw",
                                     dtype=tc.complex128, device="cpu")
    assert fermion_mps.dim == 2
    assert [int(A.abs().argmax().item()) for A in fermion_mps.tensors] == [1, 0, 1, 0]

    hubbard_mps = r.make_product_mps("hubbard", N=4, init="hubbard_neel",
                                     dtype=tc.complex128, device="cpu")
    assert hubbard_mps.dim == 4
    assert [int(A.abs().argmax().item()) for A in hubbard_mps.tensors] == [1, 2, 1, 2]
    assert r.bond_dims(hubbard_mps) == [1, 1, 1]


def test_model_builders_do_not_build_dense_hamiltonians():
    r = load_runner()
    dtype = tc.complex128
    for model, dim in [("heisenberg", 2), ("tfi", 2), ("spinless_tv", 2), ("hubbard", 4)]:
        mpo = r.build_mpo(model=model, N=4, dtype=dtype, device="cpu",
                          J=1.0, h=0.7, t=1.0, V=0.5, U=4.0, mu=0.0, field=0.0)
        assert mpo.length == 4
        assert mpo.dim == dim
        assert len(mpo.tensors) == 4


def test_tiny_ad_runner_smoke_no_ed_no_dmrg(tmp_path):
    r = load_runner()
    args = Namespace(
        model="heisenberg",
        N=4,
        chi=2,
        sweeps=1,
        device="cpu",
        dtype="complex128",
        init="neel",
        optimizer="lbfgs",
        local_steps=1,
        lbfgs_iters=2,
        lr=1.0,
        cutoff=None,
        stabilization="tensor_norm",
        grad_clip=None,
        seed=0,
        output=tmp_path / "tiny_ad.json",
        print_bonds=False,
        print_bond_dims=False,
        store_bond_reports=False,
        progress_every=999,
        J=1.0,
        h=1.0,
        t=1.0,
        V=0.0,
        U=4.0,
        mu=0.0,
        field=0.0,
    )
    res = r.run_ad_model_benchmark(args)
    assert res["ad_mainline"] is True
    assert res["ed_skipped"] is True
    assert res["dense_hamiltonian_built"] is False
    assert res["dmrg_lanczos_used"] is False
    assert tc.isfinite(tc.tensor(res["final_energy"])).item()
    assert args.output.exists()
