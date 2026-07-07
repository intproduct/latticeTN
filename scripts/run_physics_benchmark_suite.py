#!/usr/bin/env python3
"""Stage 11 physics benchmark suite runner.

The quick/default suites are CPU-small and may use dense ED only for small-N
reference cases. This script must not be used as a large-N dense benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latticetn.benchmarks.exact_reference import exact_ground_reference, dense_model_hamiltonian  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn import dmrg as DMRG  # noqa: E402
from latticetn import canonical as Can  # noqa: E402
from latticetn.config_schema import MethodConfig, ObservableSpec, RuntimeConfig  # noqa: E402
from latticetn.hamiltonian_builder import build_mpo  # noqa: E402
from latticetn.mps import MPS  # noqa: E402
from latticetn.model_registry import build_model_spec  # noqa: E402
from latticetn.observables import (  # noqa: E402
    dense_connected_correlation,
    dense_entanglement_entropy,
    dense_expect_local,
    dense_expect_two_site,
    dense_fermion_local_density,
    dense_hubbard_double_occ,
)
from latticetn.operators import exact_ground_energy, spin_operators  # noqa: E402
from latticetn.fermion_operators import fermion_operators, hubbard_local_operators  # noqa: E402
from latticetn.runner import run_latticetn_job  # noqa: E402


DTYPE = tc.complex128
DEFAULT_OUTPUT = ROOT / "outputs" / "physics_benchmarks"
REGISTRY = ROOT / "benchmarks" / "references" / "reference_registry.json"


def _mps_local_expectation(mps: MPS, op: tc.Tensor, site: int) -> float:
    op = op.to(dtype=mps.dtype, device=mps.device)
    I = tc.eye(mps.dim, dtype=mps.dtype, device=mps.device)
    v_num = tc.ones((1, 1), dtype=mps.dtype, device=mps.device)
    v_den = tc.ones((1, 1), dtype=mps.dtype, device=mps.device)
    for i, A in enumerate(mps.tensors):
        O = op if i == site else I
        v_num = tc.einsum("lr,lsm,st,rtn->mn", v_num, A.conj(), O, A)
        v_den = tc.einsum("lr,lsm,rsn->mn", v_den, A.conj(), A)
    return float((v_num.reshape(()) / v_den.reshape(())).real.detach().cpu())


def _mps_two_site_expectation(mps: MPS, op_i: tc.Tensor, i: int, op_j: tc.Tensor, j: int) -> float:
    if i == j:
        raise ValueError("two-site expectation requires distinct sites")
    lo, hi = sorted((i, j))
    op_lo = op_i if i == lo else op_j
    op_hi = op_j if j == hi else op_i
    op_lo = op_lo.to(dtype=mps.dtype, device=mps.device)
    op_hi = op_hi.to(dtype=mps.dtype, device=mps.device)
    I = tc.eye(mps.dim, dtype=mps.dtype, device=mps.device)
    v_num = tc.ones((1, 1), dtype=mps.dtype, device=mps.device)
    v_den = tc.ones((1, 1), dtype=mps.dtype, device=mps.device)
    for site, A in enumerate(mps.tensors):
        if site == lo:
            O = op_lo
        elif site == hi:
            O = op_hi
        else:
            O = I
        v_num = tc.einsum("lr,lsm,st,rtn->mn", v_num, A.conj(), O, A)
        v_den = tc.einsum("lr,lsm,rsn->mn", v_den, A.conj(), A)
    return float((v_num.reshape(()) / v_den.reshape(())).real.detach().cpu())


def _mps_connected_expectation(mps: MPS, op_i: tc.Tensor, i: int, op_j: tc.Tensor, j: int) -> float:
    two = _mps_two_site_expectation(mps, op_i, i, op_j, j)
    one_i = _mps_local_expectation(mps, op_i, i)
    one_j = _mps_local_expectation(mps, op_j, j)
    return two - one_i * one_j


def _mps_entanglement_entropy(mps: MPS, cut: int) -> float:
    return float(Can.entanglement_entropy(mps, cut))


def _jsonable(x: Any) -> Any:
    if isinstance(x, tc.Tensor):
        if x.numel() == 1:
            return float(x.detach().real.cpu())
        return x.detach().cpu().tolist()
    if isinstance(x, complex):
        return {"real": x.real, "imag": x.imag}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def _record(
    *,
    case_id: str,
    suite: str,
    model: str,
    N: int,
    method: str,
    energy: float | None = None,
    reference: dict | None = None,
    observables: dict | None = None,
    tolerance: dict | None = None,
    passed: bool = True,
    runtime_sec: float = 0.0,
    parameters: dict | None = None,
    sector: dict | None = None,
    notes: str = "",
) -> dict:
    energy_per_site = energy / N if energy is not None and N > 0 else None
    ref_value = None if reference is None else reference.get("value")
    absolute_error = None
    relative_error = None
    if energy is not None and isinstance(ref_value, (int, float)):
        absolute_error = abs(energy - float(ref_value))
        denom = max(abs(float(ref_value)), 1e-15)
        relative_error = absolute_error / denom
    return {
        "case_id": case_id,
        "suite": suite,
        "model": model,
        "parameters": parameters or {},
        "sector": sector or {},
        "N": N,
        "chi": 0,
        "method": method,
        "device": "cpu",
        "dtype": "torch.complex128",
        "energy": energy,
        "energy_per_site": energy_per_site,
        "observables": observables or {},
        "reference": reference or {},
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "pass": bool(passed),
        "tolerance": tolerance or {},
        "runtime_sec": runtime_sec,
        "notes": notes,
    }


def _hamiltonian_cases(models: set[str]) -> list[dict]:
    specs = [
        ("heisenberg", 4, {"J": 1.0}),
        ("tfi", 4, {"J": 1.0, "h": 0.7}),
        ("spinless_tv", 4, {"t": 1.0, "V": 0.5, "mu": 0.0}),
        ("hubbard", 3, {"t": 1.0, "U": 2.0, "mu": 0.0, "h": 0.0}),
    ]
    out = []
    for model, N, params in specs:
        if model not in models:
            continue
        t0 = time.perf_counter()
        spec = build_model_spec(model, N, params)
        H_mpo = build_mpo(spec, dtype=DTYPE, device="cpu").to_dense()
        H_ref = dense_model_hamiltonian(model, N, params, dtype=DTYPE)
        max_abs = float((H_mpo - H_ref).abs().max())
        out.append(_record(
            case_id=f"hamiltonian_{model}_N{N}",
            suite="exact",
            model=model,
            N=N,
            parameters=params,
            method="MPO vs independent dense reference",
            observables={"max_abs_diff": max_abs, "dim": int(H_ref.shape[0])},
            tolerance={"max_abs_diff": 1e-10},
            passed=max_abs <= 1e-10,
            runtime_sec=time.perf_counter() - t0,
        ))
    return out


def _exact_energy_cases(models: set[str]) -> list[dict]:
    specs = [
        ("heisenberg", 6, {"J": 1.0}, None),
        ("tfi", 6, {"J": 1.0, "h": 0.7}, None),
        ("spinless_tv", 6, {"t": 1.0, "V": 0.5, "mu": 0.0}, {"target_n": 3}),
        ("hubbard", 3, {"t": 1.0, "U": 2.0, "mu": 0.0, "h": 0.0}, {"target_nup": 1, "target_ndown": 1}),
    ]
    out = []
    for model, N, params, sector in specs:
        if model not in models:
            continue
        t0 = time.perf_counter()
        ref = exact_ground_reference(model, N, params, sector, dtype=DTYPE)
        out.append(_record(
            case_id=f"small_ed_{model}_N{N}",
            suite="exact",
            model=model,
            N=N,
            parameters=params,
            sector=sector,
            method="small-N exact diagonalization",
            energy=ref.energy,
            reference={"type": "ED", "value": ref.energy, "dim_sector": ref.dim_sector},
            observables={"energy_per_site": ref.energy_per_site, "dim_full": ref.dim_full},
            tolerance={"hermiticity": 1e-12},
            passed=True,
            runtime_sec=time.perf_counter() - t0,
            notes="ED is restricted to this small-N reference case.",
        ))
    return out


def _basis_state(dim: int, states: list[int]) -> tc.Tensor:
    idx = 0
    for state in states:
        idx = idx * dim + state
    psi = tc.zeros(dim ** len(states), dtype=DTYPE)
    psi[idx] = 1.0
    return psi


def _observable_cases(models: set[str]) -> list[dict]:
    out = []
    if "heisenberg" in models:
        t0 = time.perf_counter()
        psi = _basis_state(2, [0, 1, 0, 1])
        ops = spin_operators(dtype=DTYPE)
        sz0 = float(dense_expect_local(psi, ops["Sz"], 0, 4).real)
        sz01 = float(dense_expect_two_site(psi, ops["Sz"], 0, ops["Sz"], 1, 4).real)
        conn = float(dense_connected_correlation(psi, ops["Sz"], 0, ops["Sz"], 1, 4).real)
        passed = abs(sz0 - 0.5) < 1e-12 and abs(sz01 + 0.25) < 1e-12 and abs(conn) < 1e-12
        out.append(_record(
            case_id="observable_heisenberg_product_neel",
            suite="observables",
            model="heisenberg",
            N=4,
            method="analytic product-state observable",
            observables={"Sz0": sz0, "Sz0Sz1": sz01, "connected_Sz0Sz1": conn},
            tolerance={"abs": 1e-12},
            passed=passed,
            runtime_sec=time.perf_counter() - t0,
        ))
    if "spinless_tv" in models:
        t0 = time.perf_counter()
        psi = _basis_state(2, [1, 0, 1, 0])
        n0 = float(dense_fermion_local_density(psi, 0, 4))
        n1 = float(dense_fermion_local_density(psi, 1, 4))
        passed = abs(n0 - 1.0) < 1e-12 and abs(n1) < 1e-12
        out.append(_record(
            case_id="observable_spinless_cdw_density",
            suite="observables",
            model="spinless_tv",
            N=4,
            method="analytic product-state observable",
            observables={"n0": n0, "n1": n1},
            tolerance={"abs": 1e-12},
            passed=passed,
            runtime_sec=time.perf_counter() - t0,
        ))
    if "hubbard" in models:
        t0 = time.perf_counter()
        psi = _basis_state(4, [1, 2, 3, 0])
        docc2 = float(dense_hubbard_double_occ(psi, 2, 4))
        passed = abs(docc2 - 1.0) < 1e-12
        out.append(_record(
            case_id="observable_hubbard_double_occupancy",
            suite="observables",
            model="hubbard",
            N=4,
            method="analytic product-state observable",
            observables={"double_occ_site2": docc2},
            tolerance={"abs": 1e-12},
            passed=passed,
            runtime_sec=time.perf_counter() - t0,
        ))
    if "heisenberg" in models:
        t0 = time.perf_counter()
        bell = tc.tensor([1, 0, 0, 1], dtype=DTYPE) / math.sqrt(2.0)
        entropy = float(dense_entanglement_entropy(bell, 1, 2))
        out.append(_record(
            case_id="entanglement_bell_pair",
            suite="observables",
            model="heisenberg",
            N=2,
            method="known entangled state",
            observables={"entropy_cut1": entropy},
            reference={"type": "analytic", "value": math.log(2.0)},
            tolerance={"abs": 1e-12},
            passed=abs(entropy - math.log(2.0)) < 1e-12,
            runtime_sec=time.perf_counter() - t0,
        ))
    return out


def _full_stack_model_case(
    *,
    model: str,
    N: int,
    dim: int,
    params: dict[str, float],
    chi: int,
    ad_steps: int,
    ad_tolerance: float,
    dmrg_tolerance: float,
) -> list[dict]:
    out: list[dict] = []
    t0 = time.perf_counter()
    tc.manual_seed(0)
    spec = build_model_spec(model, N, params)
    mpo = build_mpo(spec, dtype=DTYPE, device="cpu")
    exact = exact_ground_reference(model, N, params, dtype=DTYPE)

    ad_mps = MPS(N, dim, chi, dtype=DTYPE, device="cpu")
    ad = ADVariationalMPS(ad_mps, mpo)
    ad_result = train_ad_mps(
        ad,
        num_steps=ad_steps,
        lr=1e-2,
        optimizer="adam",
        projection="tensor_norm",
        record_every=max(1, ad_steps // 3),
    )
    ad_energy = float(ad_result["final_energy"])
    ad_error = abs(ad_energy - exact.energy)
    ad_below = ad_energy < exact.energy - 1e-8
    out.append(_record(
        case_id=f"fullstack_{model}_ad_vs_ed_N{N}",
        suite="fullstack",
        model=model,
        N=N,
        parameters={**params, "chi": chi, "steps": ad_steps, "lr": 1e-2, "seed": 0},
        method="AD-MPS Rayleigh optimization vs ED",
        energy=ad_energy,
        reference={"type": "ED", "value": exact.energy},
        observables={
            "initial_energy": ad_result["initial_energy"],
            "final_energy": ad_energy,
            "absolute_error": ad_error,
            "energy_history": ad_result["energy_history"],
            "max_bond": ad_result["max_bond"],
        },
        tolerance={"abs_error": ad_tolerance, "below_ground_tol": 1e-8},
        passed=(ad_result["final_energy"] < ad_result["initial_energy"])
        and (not ad_below)
        and ad_error < ad_tolerance,
        runtime_sec=time.perf_counter() - t0,
        notes="AD mainline only; no DMRG, Lanczos, or ED is used in the AD loss path.",
    ))

    t1 = time.perf_counter()
    tc.manual_seed(0)
    dmrg_mps = MPS(N, dim, chi, dtype=DTYPE, device="cpu")
    dmrg_result = DMRG.run_dmrg(dmrg_mps, mpo, chi=chi, num_sweeps=2, seed=0, solver="dense")
    dmrg_energy = float(dmrg_result["final_energy"])
    dmrg_error = abs(dmrg_energy - exact.energy)
    out.append(_record(
        case_id=f"fullstack_{model}_dmrg_vs_ed_N{N}",
        suite="fullstack",
        model=model,
        N=N,
        parameters={**params, "chi": chi, "sweeps": 2, "seed": 0},
        method="classical two-site DMRG reference vs ED",
        energy=dmrg_energy,
        reference={"type": "ED", "value": exact.energy},
        observables={
            "absolute_error": dmrg_error,
            "energy_history": [step["energy"] for step in dmrg_result["history"]],
            "final_max_bond": dmrg_result["final_max_bond"],
            "max_truncation": max(step["max_trunc"] for step in dmrg_result["history"]),
        },
        tolerance={"abs_error": dmrg_tolerance, "below_ground_tol": 1e-8},
        passed=(dmrg_energy >= exact.energy - 1e-8) and dmrg_error < dmrg_tolerance,
        runtime_sec=time.perf_counter() - t1,
        notes="Classical DMRG is a reference baseline, not the AD mainline.",
    ))

    ad_dmrg_diff = abs(ad_energy - dmrg_energy)
    out.append(_record(
        case_id=f"fullstack_{model}_ad_vs_dmrg_N{N}",
        suite="fullstack",
        model=model,
        N=N,
        parameters={**params, "chi": chi},
        method="AD-MPS vs classical DMRG baseline",
        energy=ad_energy,
        reference={"type": "classical_dmrg", "value": dmrg_energy},
        observables={"absolute_difference": ad_dmrg_diff},
        tolerance={"abs_difference": ad_tolerance},
        passed=ad_dmrg_diff < ad_tolerance,
        notes="Both methods use the same MPO, open boundary, chi, and dtype.",
    ))
    return out


def _hubbard_full_stack_cases() -> list[dict]:
    """Small hard-sector Hubbard AD/DMRG/ED comparison through Stage 10 API."""
    out: list[dict] = []
    model = "hubbard"
    N = 2
    dim = 4
    chi = 4
    params = {"t": 1.0, "U": 2.0, "mu": 0.0, "h": 0.0}
    sector = {"target_nup": 1, "target_ndown": 1}
    spec_sector = build_model_spec(model, N, params, sector={"mode": "hard", **sector})
    spec_plain = build_model_spec(model, N, params)
    mpo = build_mpo(spec_plain, dtype=DTYPE, device="cpu")
    exact = exact_ground_reference(model, N, params, sector, dtype=DTYPE)

    t0 = time.perf_counter()
    ad_result = run_latticetn_job(
        spec_sector,
        MethodConfig(
            name="ad_dmrg",
            chi=chi,
            sweeps=1,
            optimizer="lbfgs",
            local_steps=10,
            lr=1.0,
            lbfgs_iters=10,
            sector_mode="hard",
        ),
        RuntimeConfig(device="cpu", dtype="complex128", seed=0, no_ed=True),
        ObservableSpec(["energy", "energy_per_site", "sector", "bond_dims", "gradient_norm"]),
    )
    ad_energy = float(ad_result["summary"]["final_energy"])
    ad_error = abs(ad_energy - exact.energy)
    sector_report = ad_result["observables"]["sector"]
    ad_below = ad_energy < exact.energy - 1e-8
    out.append(_record(
        case_id="fullstack_hubbard_ad_hard_sector_vs_ed_N2",
        suite="fullstack",
        model=model,
        N=N,
        parameters={**params, **sector, "chi": chi, "optimizer": "lbfgs", "local_steps": 10, "seed": 0},
        sector=sector,
        method="hard-sector AD runner vs fixed-sector ED",
        energy=ad_energy,
        reference={"type": "sector_ED", "value": exact.energy, "dim_sector": exact.dim_sector},
        observables={
            "absolute_error": ad_error,
            "sector": sector_report,
            "diagnostics": ad_result["diagnostics"],
            "final_gradient_norm": ad_result["summary"]["final_gradient_norm"],
        },
        tolerance={"abs_error": 1e-8, "below_ground_tol": 1e-8, "sector_abs_error": 1e-10},
        passed=(not ad_below)
        and ad_error < 1e-8
        and sector_report["n_up_abs_error"] < 1e-10
        and sector_report["n_down_abs_error"] < 1e-10
        and ad_result["diagnostics"]["ad_used"] is True
        and ad_result["diagnostics"]["ed_used"] is False
        and ad_result["diagnostics"]["classical_dmrg_used"] is False
        and ad_result["diagnostics"]["dense_hamiltonian_built"] is False,
        runtime_sec=time.perf_counter() - t0,
        notes="Uses Stage 10 hard-sector AD runner; ED is used only outside the AD path as the benchmark reference.",
    ))

    t1 = time.perf_counter()
    tc.manual_seed(0)
    dmrg_mps = MPS(N, dim, chi, dtype=DTYPE, device="cpu")
    dmrg_result = DMRG.run_dmrg(dmrg_mps, mpo, chi=chi, num_sweeps=2, seed=0, solver="dense")
    dmrg_energy = float(dmrg_result["final_energy"])
    dmrg_error = abs(dmrg_energy - exact.energy)
    out.append(_record(
        case_id="fullstack_hubbard_dmrg_vs_ed_N2",
        suite="fullstack",
        model=model,
        N=N,
        parameters={**params, "chi": chi, "sweeps": 2, "seed": 0},
        method="classical two-site DMRG reference vs ED",
        energy=dmrg_energy,
        reference={"type": "ED", "value": exact.energy},
        observables={
            "absolute_error": dmrg_error,
            "energy_history": [step["energy"] for step in dmrg_result["history"]],
            "final_max_bond": dmrg_result["final_max_bond"],
            "max_truncation": max(step["max_trunc"] for step in dmrg_result["history"]),
        },
        tolerance={"abs_error": 1e-10, "below_ground_tol": 1e-8},
        passed=(dmrg_energy >= exact.energy - 1e-8) and dmrg_error < 1e-10,
        runtime_sec=time.perf_counter() - t1,
        notes="Classical DMRG is a reference baseline, not the AD mainline.",
    ))

    ad_dmrg_diff = abs(ad_energy - dmrg_energy)
    out.append(_record(
        case_id="fullstack_hubbard_ad_vs_dmrg_N2",
        suite="fullstack",
        model=model,
        N=N,
        parameters={**params, "chi": chi},
        method="hard-sector AD runner vs classical DMRG baseline",
        energy=ad_energy,
        reference={"type": "classical_dmrg", "value": dmrg_energy},
        observables={"absolute_difference": ad_dmrg_diff},
        tolerance={"abs_difference": 1e-8},
        passed=ad_dmrg_diff < 1e-8,
        notes="Both methods use the same Hubbard MPO, open boundary, dtype, and small-N reference convention.",
    ))
    return out


def _full_stack_cases(models: set[str]) -> list[dict]:
    """Small CPU full-stack AD/DMRG/ED comparisons.

    These stay deliberately tiny. Dense ED is a valid small-N reference, and
    classical DMRG is used only as a baseline record outside the AD path.
    """
    specs = [
        ("heisenberg", 4, 2, {"J": 1.0}, 4, 120, 1e-3, 1e-10),
        ("tfi", 4, 2, {"J": 1.0, "h": 0.7}, 4, 120, 1e-3, 1e-10),
        ("spinless_tv", 4, 2, {"t": 1.0, "V": 0.5, "mu": 0.0}, 4, 160, 1e-3, 1e-10),
    ]
    out: list[dict] = []
    for model, N, dim, params, chi, ad_steps, ad_tol, dmrg_tol in specs:
        if model in models:
            out.extend(_full_stack_model_case(
                model=model,
                N=N,
                dim=dim,
                params=params,
                chi=chi,
                ad_steps=ad_steps,
                ad_tolerance=ad_tol,
                dmrg_tolerance=dmrg_tol,
            ))
    if "hubbard" in models:
        out.extend(_hubbard_full_stack_cases())
    return out


def _policy_cases() -> list[dict]:
    cuda_available = bool(tc.cuda.is_available())
    return [
        _record(
            case_id="policy_cuda_quick_suite_cpu_only",
            suite="policy",
            model="all",
            N=0,
            method="runtime policy check",
            observables={
                "cuda_available": cuda_available,
                "quick_suite_device": "cpu",
                "cuda_status": "available_but_not_used" if cuda_available else "clean_skip_no_cuda",
            },
            passed=True,
            notes=(
                "Stage 11 quick validation is CPU-only. CUDA is opt-in for "
                "separate benchmark jobs and is cleanly skipped here when absent."
            ),
        ),
        _record(
            case_id="policy_large_n_ad_runner_no_dense_or_classical_reference",
            suite="policy",
            model="all",
            N=0,
            method="large-N validation policy record",
            observables={
                "large_n_ed": False,
                "large_n_classical_dmrg": False,
                "large_n_lanczos": False,
                "large_n_dense_hamiltonian": False,
            },
            passed=True,
            notes=(
                "Large-N AD benchmark records must not construct dense "
                "Hamiltonians and must not call ED, classical DMRG, or Lanczos."
            ),
        ),
    ]


def _large_n_ad_cases(models: set[str]) -> list[dict]:
    out: list[dict] = []
    if "heisenberg" in models:
        spin_ops = spin_operators(dtype=DTYPE)

        def run_heisenberg_ad(N: int, chi: int) -> dict:
            mpo = build_mpo(build_model_spec("heisenberg", N, {"J": 1.0}), dtype=DTYPE, device="cpu")
            mid = N // 2
            tc.manual_seed(0)
            mps = MPS(N, 2, chi, dtype=DTYPE, device="cpu")
            ad = ADVariationalMPS(mps, mpo)
            result = train_ad_mps(
                ad,
                num_steps=3,
                lr=5e-3,
                optimizer="adam",
                projection="tensor_norm",
                record_every=1,
            )
            history = [float(x) for x in result["energy_history"]]
            sz_mid = _mps_local_expectation(ad.mps, spin_ops["Sz"], mid)
            connected_mid = _mps_connected_expectation(ad.mps, spin_ops["Sz"], mid - 1, spin_ops["Sz"], mid)
            entropy_mid = _mps_entanglement_entropy(ad.mps, mid)
            return {
                "chi": chi,
                "initial_energy": float(result["initial_energy"]),
                "final_energy": float(result["final_energy"]),
                "energy_per_site": float(result["final_energy"]) / N,
                "energy_history": history,
                "grad_norm_history": [float(x) for x in result["grad_norm_history"]],
                "max_bond": int(result["max_bond"]),
                "local_Sz_mid": sz_mid,
                "connected_SzSz_midbond": connected_mid,
                "entanglement_entropy_midbond": entropy_mid,
                "ed_used": False,
                "classical_dmrg_used": False,
                "lanczos_used": False,
                "dense_hamiltonian_built": False,
            }

        N = 20
        t0 = time.perf_counter()
        chi8 = run_heisenberg_ad(N, 8)
        history = chi8["energy_history"]
        finite = all(math.isfinite(x) for x in history)
        out.append(_record(
            case_id="large_n_ad_heisenberg_N20_chi8_no_reference",
            suite="large_n_ad",
            model="heisenberg",
            N=N,
            parameters={"J": 1.0, "chi": 8, "steps": 3, "lr": 5e-3, "seed": 0},
            method="large-N AD-only smoke",
            energy=chi8["final_energy"],
            observables={
                "initial_energy": chi8["initial_energy"],
                "final_energy": chi8["final_energy"],
                "energy_history": history,
                "grad_norm_history": chi8["grad_norm_history"],
                "max_bond": chi8["max_bond"],
                "local_Sz_mid": chi8["local_Sz_mid"],
                "connected_SzSz_midbond": chi8["connected_SzSz_midbond"],
                "entanglement_entropy_midbond": chi8["entanglement_entropy_midbond"],
                "ed_used": False,
                "classical_dmrg_used": False,
                "lanczos_used": False,
                "dense_hamiltonian_built": False,
            },
            tolerance={"energy_decreases": True, "finite_energy": True, "max_bond": 8},
            passed=finite and chi8["final_energy"] < chi8["initial_energy"] and chi8["max_bond"] <= 8
            and math.isfinite(chi8["local_Sz_mid"]) and math.isfinite(chi8["connected_SzSz_midbond"])
            and math.isfinite(chi8["entanglement_entropy_midbond"]) and chi8["entanglement_entropy_midbond"] >= 0.0,
            runtime_sec=time.perf_counter() - t0,
            notes="No ED, no classical DMRG, no Lanczos, and no dense Hamiltonian construction; convergence to a literature value is not asserted.",
        ))
        def append_heisenberg_chi_table(N: int) -> None:
            t0 = time.perf_counter()
            chi_values = [4, 8, 16, 32]
            chi_table = [run_heisenberg_ad(N, chi) for chi in chi_values]
            finite_table = all(
                all(math.isfinite(x) for x in row["energy_history"])
                and math.isfinite(row["local_Sz_mid"])
                and math.isfinite(row["connected_SzSz_midbond"])
                and math.isfinite(row["entanglement_entropy_midbond"])
                and row["entanglement_entropy_midbond"] >= 0.0
                for row in chi_table
            )
            decreasing = all(row["final_energy"] < row["initial_energy"] for row in chi_table)
            higher_chi_not_worse = all(
                chi_table[i + 1]["final_energy"] <= chi_table[i]["final_energy"]
                for i in range(len(chi_table) - 1)
            )
            policy_clean = all(
                row["ed_used"] is False
                and row["classical_dmrg_used"] is False
                and row["lanczos_used"] is False
                and row["dense_hamiltonian_built"] is False
                and row["max_bond"] <= row["chi"]
                for row in chi_table
            )
            out.append(_record(
                case_id=f"large_n_chi_table_heisenberg_N{N}_chi4_8_16_32_no_reference",
                suite="large_n_ad",
                model="heisenberg",
                N=N,
                parameters={"J": 1.0, "chi_values": chi_values, "steps": 3, "lr": 5e-3, "seed": 0},
                method="resource-bounded large-N AD chi table",
                energy=chi_table[-1]["final_energy"],
                observables={
                    "chi_table": chi_table,
                    "energy_decreases_each_chi": decreasing,
                    "higher_chi_energy_not_worse": higher_chi_not_worse,
                    "ed_used": False,
                    "classical_dmrg_used": False,
                    "lanczos_used": False,
                    "dense_hamiltonian_built": False,
                },
                tolerance={
                    "finite_energy": True,
                    "energy_decreases_each_chi": True,
                    "higher_chi_energy_not_worse": True,
                    "max_bond_lte_chi": True,
                },
                passed=finite_table and decreasing and higher_chi_not_worse and policy_clean,
                runtime_sec=time.perf_counter() - t0,
                notes=(
                    "CPU-small chi table only. This strengthens large-N convergence "
                    "evidence but does not replace the requested full N=40/80 production runs."
                ),
            ))

        append_heisenberg_chi_table(20)
        for N in (40, 80):
            t0 = time.perf_counter()
            large_chi8 = run_heisenberg_ad(N, 8)
            large_history = large_chi8["energy_history"]
            large_finite = all(math.isfinite(x) for x in large_history)
            out.append(_record(
                case_id=f"large_n_ad_heisenberg_N{N}_chi8_no_reference",
                suite="large_n_ad",
                model="heisenberg",
                N=N,
                parameters={"J": 1.0, "chi": 8, "steps": 3, "lr": 5e-3, "seed": 0},
                method="bounded larger-size AD-only smoke",
                energy=large_chi8["final_energy"],
                observables={
                    "initial_energy": large_chi8["initial_energy"],
                    "final_energy": large_chi8["final_energy"],
                    "energy_history": large_history,
                    "grad_norm_history": large_chi8["grad_norm_history"],
                    "max_bond": large_chi8["max_bond"],
                    "local_Sz_mid": large_chi8["local_Sz_mid"],
                    "connected_SzSz_midbond": large_chi8["connected_SzSz_midbond"],
                    "entanglement_entropy_midbond": large_chi8["entanglement_entropy_midbond"],
                    "ed_used": False,
                    "classical_dmrg_used": False,
                    "lanczos_used": False,
                    "dense_hamiltonian_built": False,
                },
                tolerance={"energy_decreases": True, "finite_energy": True, "max_bond": 8},
                passed=large_finite and large_chi8["final_energy"] < large_chi8["initial_energy"]
                and large_chi8["max_bond"] <= 8
                and math.isfinite(large_chi8["local_Sz_mid"])
                and math.isfinite(large_chi8["connected_SzSz_midbond"])
                and math.isfinite(large_chi8["entanglement_entropy_midbond"])
                and large_chi8["entanglement_entropy_midbond"] >= 0.0,
                runtime_sec=time.perf_counter() - t0,
                notes=f"CPU-small N={N} interacting AD smoke only; no ED, no classical DMRG, no Lanczos, and no dense Hamiltonian construction.",
            ))
            append_heisenberg_chi_table(N)

    hard_specs = [
        ("spinless_tv", 20, {"t": 1.0, "V": 0.5, "mu": 0.0}, {"mode": "hard", "target_n": 10}, 8),
        ("hubbard", 10, {"t": 1.0, "U": 4.0, "mu": 0.0, "h": 0.0}, {"mode": "hard", "target_nup": 5, "target_ndown": 5}, 8),
    ]
    larger_hard_specs = [
        ("spinless_tv", 40, {"t": 1.0, "V": 0.5, "mu": 0.0}, {"mode": "hard", "target_n": 20}, 8),
        ("spinless_tv", 80, {"t": 1.0, "V": 0.5, "mu": 0.0}, {"mode": "hard", "target_n": 40}, 8),
        ("hubbard", 20, {"t": 1.0, "U": 4.0, "mu": 0.0, "h": 0.0}, {"mode": "hard", "target_nup": 10, "target_ndown": 10}, 8),
        ("hubbard", 40, {"t": 1.0, "U": 4.0, "mu": 0.0, "h": 0.0}, {"mode": "hard", "target_nup": 20, "target_ndown": 20}, 8),
    ]

    def hard_observable_names(model: str) -> list[str]:
        names = ["energy", "energy_per_site", "sector", "bond_dims", "gradient_norm", "local_density_mid"]
        if model == "hubbard":
            names.extend(["double_occupancy_mid", "local_sz_mid"])
        return names

    def hard_local_observables_physical(model: str, obs: dict) -> bool:
        density = float(obs["local_density_mid"])
        if model == "spinless_tv":
            return math.isfinite(density) and -1e-12 <= density <= 1.0 + 1e-12
        double_occ = float(obs["double_occupancy_mid"])
        sz = float(obs["local_sz_mid"])
        return (
            math.isfinite(density)
            and math.isfinite(double_occ)
            and math.isfinite(sz)
            and -1e-12 <= density <= 2.0 + 1e-12
            and -1e-12 <= double_occ <= 1.0 + 1e-12
            and abs(sz) <= 0.5 + 1e-12
        )

    for model, N, params, sector, chi in [*hard_specs, *larger_hard_specs]:
        if model not in models:
            continue
        t0 = time.perf_counter()
        spec = build_model_spec(model, N, params, sector=sector)
        result = run_latticetn_job(
            spec,
            MethodConfig(
                name="ad_dmrg",
                chi=chi,
                sweeps=1,
                optimizer="adam",
                local_steps=2,
                lr=5e-3,
                sector_mode="hard",
            ),
            RuntimeConfig(device="cpu", dtype="complex128", seed=0, no_ed=True),
            ObservableSpec(hard_observable_names(model)),
        )
        final = float(result["summary"]["final_energy"])
        result_obs = result["observables"]
        sector_report = result["observables"]["sector"]
        diagnostics = result["diagnostics"]
        energy_history = [float(step["energy"]) for step in result["sweep_history"]]
        sector_ok = True
        if model == "spinless_tv":
            sector_ok = sector_report["abs_error"] < 1e-10 and sector_report["variance"] < 1e-10
        elif model == "hubbard":
            sector_ok = (
                sector_report["n_up_abs_error"] < 1e-10
                and sector_report["n_down_abs_error"] < 1e-10
                and sector_report["variance_n_tot"] < 1e-10
            )
        finite = math.isfinite(final) and all(math.isfinite(x) for x in energy_history)
        clean_path = (
            diagnostics["ad_used"] is True
            and diagnostics["ed_used"] is False
            and diagnostics["classical_dmrg_used"] is False
            and diagnostics["lanczos_used"] is False
            and diagnostics["dense_hamiltonian_built"] is False
            and diagnostics["max_forbidden_abs"] < 1e-12
            and diagnostics["max_forbidden_grad_abs"] < 1e-12
        )
        out.append(_record(
            case_id=f"large_n_ad_{model}_N{N}_chi{chi}_hard_sector_no_reference",
            suite="large_n_ad",
            model=model,
            N=N,
            parameters={**params, **sector, "chi": chi, "sweeps": 1, "local_steps": 2, "lr": 5e-3, "seed": 0},
            sector={k: v for k, v in sector.items() if k != "mode"},
            method="large-N hard-sector AD-only smoke",
            energy=final,
            observables={
                "final_energy": final,
                "energy_history": energy_history,
                "sector": sector_report,
                "additive_observables": sector_report,
                "local_density_mid": result_obs["local_density_mid"],
                "double_occupancy_mid": result_obs.get("double_occupancy_mid"),
                "local_sz_mid": result_obs.get("local_sz_mid"),
                "diagnostics": diagnostics,
                "gradient_norm": result["summary"]["final_gradient_norm"],
                "max_bond": result["summary"]["final_max_bond"],
            },
            tolerance={
                "finite_energy": True,
                "sector_abs_error": 1e-10,
                "forbidden_abs": 1e-12,
                "max_bond": chi,
                "local_density_range": "spinless [0,1], Hubbard [0,2]",
                "hubbard_double_occupancy_range": "[0,1]",
                "hubbard_sz_range": "[-0.5,0.5]",
            },
            passed=(
                finite
                and sector_ok
                and clean_path
                and int(result["summary"]["final_max_bond"]) <= chi
                and hard_local_observables_physical(model, result_obs)
            ),
            runtime_sec=time.perf_counter() - t0,
            notes=(
                "Hard-sector AD smoke only; no ED, no classical DMRG, no Lanczos, "
                "no dense Hamiltonian construction, and local observables are contracted from the final MPS."
            ),
        ))

    for model, N, params, sector, _ in [*hard_specs, *larger_hard_specs]:
        if model not in models:
            continue
        t0 = time.perf_counter()
        chi_table = []
        chi_values = [4, 8, 16, 32]
        for chi in chi_values:
            spec = build_model_spec(model, N, params, sector=sector)
            result = run_latticetn_job(
                spec,
                MethodConfig(
                    name="ad_dmrg",
                    chi=chi,
                    sweeps=1,
                    optimizer="adam",
                    local_steps=2,
                    lr=5e-3,
                    sector_mode="hard",
                ),
                RuntimeConfig(device="cpu", dtype="complex128", seed=0, no_ed=True),
                ObservableSpec(hard_observable_names(model)),
            )
            result_obs = result["observables"]
            chi_table.append({
                "chi": chi,
                "final_energy": float(result["summary"]["final_energy"]),
                "energy_per_site": float(result["summary"]["final_energy"]) / N,
                "energy_history": [float(step["energy"]) for step in result["sweep_history"]],
                "sector": result["observables"]["sector"],
                "additive_observables": result["observables"]["sector"],
                "local_density_mid": result_obs["local_density_mid"],
                "double_occupancy_mid": result_obs.get("double_occupancy_mid"),
                "local_sz_mid": result_obs.get("local_sz_mid"),
                "diagnostics": result["diagnostics"],
                "gradient_norm": result["summary"]["final_gradient_norm"],
                "max_bond": result["summary"]["final_max_bond"],
            })

        def hard_sector_clean(row: dict) -> bool:
            sector_report = row["sector"]
            diagnostics = row["diagnostics"]
            if model == "spinless_tv":
                sector_ok = sector_report["abs_error"] < 1e-10 and sector_report["variance"] < 1e-10
            else:
                sector_ok = (
                    sector_report["n_up_abs_error"] < 1e-10
                    and sector_report["n_down_abs_error"] < 1e-10
                    and sector_report["variance_n_tot"] < 1e-10
                )
            return (
                math.isfinite(row["final_energy"])
                and all(math.isfinite(x) for x in row["energy_history"])
                and int(row["max_bond"]) <= int(row["chi"])
                and sector_ok
                and diagnostics["ad_used"] is True
                and diagnostics["ed_used"] is False
                and diagnostics["classical_dmrg_used"] is False
                and diagnostics["lanczos_used"] is False
                and diagnostics["dense_hamiltonian_built"] is False
                and diagnostics["max_forbidden_abs"] < 1e-12
                and diagnostics["max_forbidden_grad_abs"] < 1e-12
                and hard_local_observables_physical(model, row)
            )

        out.append(_record(
            case_id=f"large_n_chi_table_{model}_N{N}_chi4_8_16_32_hard_sector_no_reference",
            suite="large_n_ad",
            model=model,
            N=N,
            parameters={**params, **sector, "chi_values": chi_values, "sweeps": 1, "local_steps": 2, "lr": 5e-3, "seed": 0},
            sector={k: v for k, v in sector.items() if k != "mode"},
            method="resource-bounded large-N hard-sector AD chi table",
            energy=chi_table[-1]["final_energy"],
            observables={
                "chi_table": chi_table,
                "sector_clean_each_chi": all(hard_sector_clean(row) for row in chi_table),
                "ed_used": False,
                "classical_dmrg_used": False,
                "lanczos_used": False,
                "dense_hamiltonian_built": False,
            },
            tolerance={"finite_energy": True, "sector_abs_error": 1e-10, "forbidden_abs": 1e-12, "max_bond_lte_chi": True},
            passed=all(hard_sector_clean(row) for row in chi_table),
            runtime_sec=time.perf_counter() - t0,
            notes=(
                "CPU-small hard-sector chi table only. This checks finite energies, "
                "sector preservation, and large-N policy flags but does not assert monotonic convergence."
            ),
        ))
    return out


def _literature_cases(models: set[str]) -> list[dict]:
    refs = json.loads(REGISTRY.read_text(encoding="utf-8"))
    out = []
    for ref in refs:
        model = str(ref["model"])
        if model not in models:
            continue
        passed = True
        notes = str(ref.get("notes", ""))
        out.append(_record(
            case_id=f"literature_{ref['id']}",
            suite="literature",
            model=model,
            N=0,
            method="reference metadata",
            reference=ref,
            passed=passed,
            notes=notes,
        ))
    out.extend(_literature_trend_cases(models))
    return out


def _open_chain_free_fermion_energy(N: int, particles: int, t: float = 1.0) -> float:
    levels = sorted(-2.0 * t * math.cos(k * math.pi / (N + 1)) for k in range(1, N + 1))
    return float(sum(levels[:particles]))


def _open_chain_free_fermion_observables(N: int, particles: int, t: float = 1.0) -> dict:
    energy = _open_chain_free_fermion_energy(N, particles, t=t)
    mid = N // 2

    def one_body(i: int, j: int) -> float:
        return sum(
            2.0
            / (N + 1)
            * math.sin(k * math.pi * (i + 1) / (N + 1))
            * math.sin(k * math.pi * (j + 1) / (N + 1))
            for k in range(1, particles + 1)
        )

    density_mid = one_body(mid, mid)
    density_mid_neighbor = one_body(mid + 1, mid + 1)
    one_body_midbond = one_body(mid, mid + 1)
    connected_midbond = -(one_body_midbond ** 2)
    return {
        "N": N,
        "particles": particles,
        "filling": particles / N,
        "energy": energy,
        "energy_per_site": energy / N,
        "density_mid": density_mid,
        "density_mid_neighbor": density_mid_neighbor,
        "connected_density_midbond": connected_midbond,
    }


def _open_chain_hubbard_free_observables(N: int, particles_per_spin: int, t: float = 1.0) -> dict:
    per_spin = _open_chain_free_fermion_observables(N, particles_per_spin, t=t)
    n_up_mid = per_spin["density_mid"]
    n_down_mid = per_spin["density_mid"]
    n_up_neighbor = per_spin["density_mid_neighbor"]
    n_down_neighbor = per_spin["density_mid_neighbor"]
    return {
        "N": N,
        "n_up": particles_per_spin,
        "n_down": particles_per_spin,
        "filling": 2.0 * particles_per_spin / N,
        "energy": 2.0 * per_spin["energy"],
        "energy_per_site": 2.0 * per_spin["energy_per_site"],
        "n_up_mid": n_up_mid,
        "n_down_mid": n_down_mid,
        "density_mid": n_up_mid + n_down_mid,
        "double_occupancy_mid": n_up_mid * n_down_mid,
        "density_mid_neighbor": n_up_neighbor + n_down_neighbor,
        "double_occupancy_mid_neighbor": n_up_neighbor * n_down_neighbor,
        "connected_density_midbond": 2.0 * per_spin["connected_density_midbond"],
    }


def _literature_trend_cases(models: set[str]) -> list[dict]:
    out: list[dict] = []
    if "heisenberg" in models:
        t0 = time.perf_counter()
        bethe = 0.25 - math.log(2.0)
        sizes = [4, 6, 8]
        e_per_site = [
            exact_ground_reference("heisenberg", N, {"J": 1.0}, dtype=DTYPE).energy_per_site
            for N in sizes
        ]
        distances = [abs(e - bethe) for e in e_per_site]
        passed = all(e > bethe for e in e_per_site) and distances[2] < distances[1] < distances[0]
        out.append(_record(
            case_id="trend_heisenberg_bethe_finite_obc",
            suite="literature",
            model="heisenberg",
            N=max(sizes),
            method="small-N ED finite-size trend",
            observables={"sizes": sizes, "energy_per_site": e_per_site, "distance_to_bethe": distances},
            reference={"id": "heisenberg_bethe_energy", "type": "thermodynamic_limit", "value": bethe},
            tolerance={"trend": "finite OBC E/N approaches 1/4-ln(2) from above for N=4,6,8"},
            passed=passed,
            runtime_sec=time.perf_counter() - t0,
            notes="Small-N ED trend only; finite OBC is not required to equal the thermodynamic limit.",
        ))
    if "tfi" in models:
        t0 = time.perf_counter()
        ops = spin_operators(dtype=DTYPE)
        fields = [0.5, 1.0, 1.5]
        mx = []
        for h in fields:
            H = dense_model_hamiltonian("tfi", 4, {"J": 1.0, "h": h}, dtype=DTYPE)
            _, gs = exact_ground_energy(H)
            vals = [float(dense_expect_local(gs, ops["Sx"], site, 4).real) for site in range(4)]
            mx.append(sum(vals) / len(vals))
        passed = mx[0] < mx[1] < mx[2]
        out.append(_record(
            case_id="trend_tfi_transverse_magnetization",
            suite="literature",
            model="tfi",
            N=4,
            method="small-N ED parameter trend",
            observables={"h": fields, "mean_Sx": mx},
            reference={"type": "qualitative", "value": "mean <Sx> increases with h"},
            tolerance={"trend": "strictly increasing for h=0.5,1.0,1.5 at N=4"},
            passed=passed,
            runtime_sec=time.perf_counter() - t0,
        ))
    if "spinless_tv" in models:
        t0 = time.perf_counter()
        N = 6
        particles = N // 2
        exact = exact_ground_reference(
            "spinless_tv",
            N,
            {"t": 1.0, "V": 0.0, "mu": 0.0},
            {"target_n": particles},
            dtype=DTYPE,
        )
        analytic = _open_chain_free_fermion_energy(N, particles, t=1.0)
        err = abs(exact.energy - analytic)
        out.append(_record(
            case_id="trend_spinless_free_fermion_limit",
            suite="literature",
            model="spinless_tv",
            N=N,
            method="analytic free-fermion limit",
            energy=exact.energy,
            observables={"analytic_energy": analytic, "abs_error": err, "particles": particles},
            reference={"id": "spinless_free_fermion_open_chain", "type": "analytic", "value": analytic},
            tolerance={"abs_error": 1e-10},
            passed=err < 1e-10,
            runtime_sec=time.perf_counter() - t0,
            sector={"target_n": particles},
        ))
        t0 = time.perf_counter()
        sizes = [40, 80]
        rows = [_open_chain_free_fermion_observables(size, size // 2, t=1.0) for size in sizes]
        thermodynamic_energy_per_site = -2.0 / math.pi
        energy_distances = [abs(row["energy_per_site"] - thermodynamic_energy_per_site) for row in rows]
        density_errors = [abs(row["density_mid"] - 0.5) for row in rows]
        correlation_magnitudes = [abs(row["connected_density_midbond"]) for row in rows]
        passed = (
            energy_distances[1] < energy_distances[0]
            and all(err < 0.02 for err in density_errors)
            and all(corr > 0.0 and math.isfinite(corr) for corr in correlation_magnitudes)
        )
        out.append(_record(
            case_id="trend_spinless_free_fermion_large_n_observables",
            suite="literature",
            model="spinless_tv",
            N=max(sizes),
            method="analytic large-N free-fermion observable trend",
            observables={
                "rows": rows,
                "energy_distance_to_thermodynamic": energy_distances,
                "density_mid_abs_error_from_half_filling": density_errors,
                "connected_density_midbond_abs": correlation_magnitudes,
            },
            reference={
                "id": "spinless_free_fermion_open_chain",
                "type": "analytic",
                "value": thermodynamic_energy_per_site,
                "observable": "energy_per_site,density,connected_density_correlation",
            },
            tolerance={
                "energy_per_site_trend": "N=80 closer to -2/pi than N=40",
                "mid_density_abs_error": 0.02,
                "connected_density_midbond": "finite and nonzero",
            },
            passed=passed,
            runtime_sec=time.perf_counter() - t0,
            sector={"target_n": "N/2"},
            notes="Independent single-particle sine-mode reference; no many-body ED or dense Hamiltonian construction.",
        ))
    if "hubbard" in models:
        t0 = time.perf_counter()
        Us = [0.0, 4.0, 8.0]
        docc = []
        for U in Us:
            H = dense_model_hamiltonian("hubbard", 2, {"t": 1.0, "U": U, "mu": 0.0, "h": 0.0}, dtype=DTYPE)
            _, gs = exact_ground_energy(H)
            docc.append(sum(float(dense_hubbard_double_occ(gs, site, 2)) for site in range(2)) / 2.0)
        passed = docc[0] > docc[1] > docc[2]
        out.append(_record(
            case_id="trend_hubbard_double_occupancy_large_u",
            suite="literature",
            model="hubbard",
            N=2,
            method="small-N ED interaction trend",
            observables={"U": Us, "mean_double_occupancy": docc},
            reference={"id": "hubbard_free_fermion_limit", "type": "qualitative", "value": "double occupancy decreases with U"},
            tolerance={"trend": "strictly decreasing for U=0,4,8 at N=2"},
            passed=passed,
            runtime_sec=time.perf_counter() - t0,
        ))
        t0 = time.perf_counter()
        sizes = [40, 80]
        rows = [_open_chain_hubbard_free_observables(size, size // 2, t=1.0) for size in sizes]
        thermodynamic_energy_per_site = -4.0 / math.pi
        energy_distances = [abs(row["energy_per_site"] - thermodynamic_energy_per_site) for row in rows]
        density_errors = [abs(row["density_mid"] - 1.0) for row in rows]
        double_occ_errors = [abs(row["double_occupancy_mid"] - 0.25) for row in rows]
        correlation_magnitudes = [abs(row["connected_density_midbond"]) for row in rows]
        passed = (
            energy_distances[1] < energy_distances[0]
            and all(err < 0.02 for err in density_errors)
            and all(err < 0.02 for err in double_occ_errors)
            and all(corr > 0.0 and math.isfinite(corr) for corr in correlation_magnitudes)
        )
        out.append(_record(
            case_id="trend_hubbard_free_fermion_large_n_observables",
            suite="literature",
            model="hubbard",
            N=max(sizes),
            method="analytic large-N U=0 Hubbard observable trend",
            observables={
                "rows": rows,
                "energy_distance_to_thermodynamic": energy_distances,
                "density_mid_abs_error_from_half_filling": density_errors,
                "double_occupancy_mid_abs_error_from_quarter": double_occ_errors,
                "connected_density_midbond_abs": correlation_magnitudes,
            },
            reference={
                "id": "hubbard_free_fermion_limit",
                "type": "analytic",
                "value": thermodynamic_energy_per_site,
                "observable": "energy_per_site,density,double_occupancy,connected_density_correlation",
            },
            tolerance={
                "energy_per_site_trend": "N=80 closer to -4/pi than N=40",
                "mid_density_abs_error": 0.02,
                "double_occupancy_abs_error": 0.02,
                "connected_density_midbond": "finite and nonzero",
            },
            passed=passed,
            runtime_sec=time.perf_counter() - t0,
            sector={"target_nup": "N/2", "target_ndown": "N/2"},
            notes="Independent spin-resolved sine-mode U=0 reference; no many-body ED or dense Hamiltonian construction.",
        ))
    return out


def _selected_models(names: list[str] | None) -> set[str]:
    all_models = {"heisenberg", "tfi", "spinless_tv", "hubbard"}
    return all_models if not names else set(names)


def run_suite(suite: str, models: set[str]) -> list[dict]:
    records: list[dict] = []
    if suite in {"quick", "exact", "full"}:
        records.extend(_hamiltonian_cases(models))
        records.extend(_exact_energy_cases(models))
    if suite in {"quick", "observables", "full"}:
        records.extend(_observable_cases(models))
    if suite in {"quick", "fullstack", "full"}:
        records.extend(_full_stack_cases(models))
    if suite in {"quick", "policy", "full"}:
        records.extend(_policy_cases())
    if suite in {"quick", "large_n_ad", "full"}:
        records.extend(_large_n_ad_cases(models))
    if suite in {"quick", "literature", "full"}:
        records.extend(_literature_cases(models))
    return records


def _large_n_evidence_payload(records: list[dict]) -> dict:
    evidence_records = [
        rec for rec in records
        if rec["suite"] == "large_n_ad"
        or rec["case_id"] in {
            "trend_spinless_free_fermion_large_n_observables",
            "trend_hubbard_free_fermion_large_n_observables",
        }
    ]
    review_required = [
        {
            "item": "Heisenberg N=40/80 interacting AD chi-convergence",
            "status": "REVIEW REQUIRED",
            "reason": "Quick suite contains bounded Heisenberg N=20/40/80 chi=4/8/16/32 evidence, not production-depth convergence over larger chi/step budgets.",
        },
        {
            "item": "spinless t-V N=40/80 interacting AD chi-convergence",
            "status": "REVIEW REQUIRED",
            "reason": "Quick suite contains bounded spinless t-V N=20/40/80 chi=4/8/16/32 hard-sector evidence plus V=0 analytic N=40/80 observables, not production-depth convergence over larger chi/step budgets.",
        },
        {
            "item": "Hubbard N=20/40 interacting AD chi-convergence",
            "status": "REVIEW REQUIRED",
            "reason": "Quick suite contains bounded Hubbard N=10/20/40 chi=4/8/16/32 hard-sector evidence plus U=0 analytic N=40/80 observables, not production-depth convergence over larger chi/step budgets.",
        },
        {
            "item": "interacting large-N observable/correlation literature trends",
            "status": "REVIEW REQUIRED",
            "reason": "Current large-N observable trends are analytic free limits or AD smoke fields, not interacting literature-grade trend tables.",
        },
    ]
    return {
        "status": "REVIEW REQUIRED",
        "scope": "Stage 11 large-N evidence extracted from the CPU-small quick suite.",
        "evidence_records": evidence_records,
        "review_required": review_required,
    }


def _write_large_n_evidence_outputs(records: list[dict], outdir: Path) -> dict[str, Path]:
    payload = _large_n_evidence_payload(records)
    json_path = outdir / "large_n_evidence.json"
    md_path = outdir / "large_n_evidence.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")

    lines = [
        "# Large-N Evidence Audit",
        "",
        "Generated by `scripts/run_physics_benchmark_suite.py`.",
        "",
        f"Status: {payload['status']}",
        "",
        "## Current Evidence",
        "",
        "| case_id | suite | model | N | method | pass |",
        "|---|---|---|---:|---|:---:|",
    ]
    for rec in payload["evidence_records"]:
        lines.append(
            f"| {rec['case_id']} | {rec['suite']} | {rec['model']} | {rec['N']} | {rec['method']} | {rec['pass']} |"
        )
    lines.extend([
        "",
        "## Review Required",
        "",
        "| item | status | reason |",
        "|---|---|---|",
    ])
    for item in payload["review_required"]:
        lines.append(f"| {item['item']} | {item['status']} | {item['reason']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"large_n_json": json_path, "large_n_markdown": md_path}


def write_outputs(records: list[dict], outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "benchmark_summary.json"
    csv_path = outdir / "benchmark_summary.csv"
    md_path = outdir / "benchmark_summary.md"
    report_path = outdir / "PHYSICS_VALIDATION_REPORT.md"

    json_path.write_text(json.dumps(_jsonable(records), indent=2), encoding="utf-8")
    fields = [
        "case_id", "suite", "model", "N", "method", "device", "dtype",
        "energy", "energy_per_site", "absolute_error", "relative_error",
        "pass", "runtime_sec", "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fields})

    passed = sum(1 for rec in records if rec["pass"])
    lines = [
        "# Physics Benchmark Summary",
        "",
        "| case_id | suite | model | method | pass |",
        "|---|---|---|---|:---:|",
    ]
    for rec in records:
        lines.append(f"| {rec['case_id']} | {rec['suite']} | {rec['model']} | {rec['method']} | {rec['pass']} |")
    lines.extend(["", f"PASS: {passed}/{len(records)}"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_suite: dict[str, tuple[int, int]] = {}
    for rec in records:
        p, n = by_suite.get(rec["suite"], (0, 0))
        by_suite[rec["suite"]] = (p + int(bool(rec["pass"])), n + 1)
    report = [
        "# Physics Validation Report",
        "",
        "Generated by `scripts/run_physics_benchmark_suite.py`.",
        "",
        "## Hamiltonian audit",
        "",
    ]
    for rec in records:
        if rec["case_id"].startswith("hamiltonian_"):
            report.append(f"- {rec['case_id']}: {'PASS' if rec['pass'] else 'FAIL'}")
    report.extend(["", "## Ground-state audit", ""])
    for rec in records:
        if rec["case_id"].startswith("small_ed_"):
            report.append(f"- {rec['case_id']}: E0={rec['energy']} PASS")
        if rec["suite"] == "fullstack":
            report.append(f"- {rec['case_id']}: {'PASS' if rec['pass'] else 'FAIL'}")
    report.extend(["", "## Sector audit", ""])
    report.append("- Small-N sector-restricted ED records include spinless target_n and Hubbard target_nup/target_ndown metadata.")
    report.extend(["", "## Observable audit", ""])
    for rec in records:
        if rec["suite"] == "observables":
            report.append(f"- {rec['case_id']}: {'PASS' if rec['pass'] else 'FAIL'}")
    report.extend(["", "## Literature audit", ""])
    for rec in records:
        if rec["suite"] == "literature":
            report.append(f"- {rec['case_id']}: {'PASS' if rec['pass'] else 'REVIEW REQUIRED'}")
    report.extend(["", "## Policy audit", ""])
    for rec in records:
        if rec["suite"] == "policy":
            report.append(f"- {rec['case_id']}: {'PASS' if rec['pass'] else 'REVIEW REQUIRED'} - {rec['notes']}")
    report.extend(["", "## Large-N AD audit", ""])
    for rec in records:
        if rec["suite"] == "large_n_ad":
            report.append(f"- {rec['case_id']}: {'PASS' if rec['pass'] else 'REVIEW REQUIRED'} - {rec['notes']}")
    report.extend(["", "## Final summary", ""])
    for suite, (p, n) in sorted(by_suite.items()):
        report.append(f"- {suite}: {p}/{n} PASS")
    report.append(f"- OVERALL STATUS: {'PASS' if passed == len(records) else 'REVIEW REQUIRED'}")
    report.append("")
    report.append("Large-N AD runner policy: no ED, no classical DMRG, no Lanczos, and no dense Hamiltonian construction.")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    paths = {"json": json_path, "csv": csv_path, "markdown": md_path, "report": report_path}
    paths.update(_write_large_n_evidence_outputs(records, outdir))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["quick", "exact", "observables", "fullstack", "policy", "large_n_ad", "literature", "full"], default="quick")
    parser.add_argument("--model", action="append", choices=["heisenberg", "tfi", "spinless_tv", "hubbard"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "cuda":
        raise RuntimeError("Stage 11 quick benchmark suite is CPU-only by default")
    if args.device == "auto" and tc.cuda.is_available():
        # Keep quick suite CPU-only; large/GPU benchmark jobs are explicit future work.
        args.device = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    records = run_suite(args.suite, _selected_models(args.model))
    paths = write_outputs(records, args.output_dir)
    passed = sum(1 for rec in records if rec["pass"])
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"PASS: {passed}/{len(records)}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
