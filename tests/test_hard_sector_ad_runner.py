import importlib.util
from pathlib import Path

import pytest
import torch as tc


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ad_model_benchmark.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_ad_model_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_hard(args_list):
    module = _load_runner()
    args = module.build_parser().parse_args(args_list)
    return module.run_ad_model_benchmark(args)


def test_spinless_hard_sector_cpu_runner_smoke(tmp_path):
    data = _run_hard([
        "--model", "spinless_tv",
        "--N", "4",
        "--chi", "4",
        "--sweeps", "1",
        "--device", "cpu",
        "--dtype", "complex128",
        "--init", "spinless_cdw",
        "--optimizer", "adam",
        "--local-steps", "1",
        "--lr", "0.01",
        "--target-n", "2",
        "--sector-mode", "hard",
        "--output", str(tmp_path / "spinless_hard.json"),
        "--no-ed",
    ])
    assert data["sector_mode"] == "hard"
    assert data["ed_status"] == "skipped by design"
    assert data["dmrg_lanczos_used"] is False
    assert data["dense_hamiltonian_built"] is False
    assert data["final_sector_report"]["abs_error"] < 1e-10
    assert data["final_sector_report"]["variance"] < 1e-10
    assert data["final_max_forbidden_abs"] == 0.0


def test_hubbard_hard_sector_cpu_runner_smoke(tmp_path):
    data = _run_hard([
        "--model", "hubbard",
        "--N", "4",
        "--chi", "4",
        "--sweeps", "1",
        "--device", "cpu",
        "--dtype", "complex128",
        "--init", "hubbard_neel",
        "--optimizer", "adam",
        "--local-steps", "1",
        "--lr", "0.01",
        "--target-nup", "2",
        "--target-ndown", "2",
        "--sector-mode", "hard",
        "--output", str(tmp_path / "hubbard_hard.json"),
        "--no-ed",
    ])
    assert data["sector_mode"] == "hard"
    assert data["final_sector_report"]["n_up_abs_error"] < 1e-10
    assert data["final_sector_report"]["n_down_abs_error"] < 1e-10
    assert data["final_sector_report"]["variance_n_tot"] < 1e-10
    assert data["final_max_forbidden_abs"] == 0.0


def test_stage8_soft_sector_runner_still_works(tmp_path):
    data = _run_hard([
        "--model", "spinless_tv",
        "--N", "4",
        "--chi", "2",
        "--sweeps", "1",
        "--device", "cpu",
        "--dtype", "complex128",
        "--init", "spinless_cdw",
        "--optimizer", "adam",
        "--local-steps", "1",
        "--lr", "0.01",
        "--target-n", "2",
        "--lambda-n", "0.1",
        "--sector-mode", "soft",
        "--output", str(tmp_path / "spinless_soft.json"),
        "--no-ed",
    ])
    assert data["sector_mode"] == "soft"
    assert data["optimizer_path"] == "global_ad_with_sector_penalty"
    assert data["initial_sector_report"]["n_target"] == 2


@pytest.mark.skipif(not tc.cuda.is_available(), reason="CUDA not available")
def test_cuda_hard_sector_smoke(tmp_path):
    data = _run_hard([
        "--model", "spinless_tv",
        "--N", "4",
        "--chi", "4",
        "--sweeps", "1",
        "--device", "cuda",
        "--dtype", "complex64",
        "--init", "spinless_cdw",
        "--optimizer", "adam",
        "--local-steps", "1",
        "--lr", "0.01",
        "--target-n", "2",
        "--sector-mode", "hard",
        "--output", str(tmp_path / "spinless_hard_cuda.json"),
        "--no-ed",
    ])
    assert data["device"] == "cuda"
    assert data["final_sector_report"]["abs_error"] < 1e-5
    assert data["final_max_forbidden_abs"] == 0.0
