import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch as tc


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ad_model_benchmark.py"


def _run(tmp_path, model, *extra, device="cpu", dtype="complex128"):
    out = tmp_path / f"{model}.json"
    cmd = [
        sys.executable, str(SCRIPT),
        "--model", model,
        "--N", "4",
        "--chi", "4" if model == "hubbard" else "2",
        "--sweeps", "1",
        "--device", device,
        "--dtype", dtype,
        "--optimizer", "adam",
        "--local-steps", "1",
        "--lr", "0.01",
        "--output", str(out),
        "--no-ed",
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "ED status = skipped by design" in completed.stdout
    assert "classical DMRG/Lanczos = not used" in completed.stdout
    assert data["ed_status"] == "skipped by design"
    assert data["dmrg_lanczos_used"] is False
    assert data["dense_hamiltonian_built"] is False
    return data


def test_cpu_cli_heisenberg_spinless_and_hubbard_smokes(tmp_path):
    heis = _run(tmp_path, "heisenberg", "--init", "neel")
    spinless = _run(tmp_path, "spinless_tv", "--init", "spinless_cdw", "--target-n", "2")
    hubbard = _run(
        tmp_path,
        "hubbard",
        "--init", "hubbard_neel",
        "--target-nup", "2",
        "--target-ndown", "2",
    )
    for data in (heis, spinless, hubbard):
        assert data["device"] == "cpu"
        assert len(data["history"]) == 1
        assert tc.isfinite(tc.tensor(data["final_energy"]))
    assert spinless["initial_sector_report"]["n_target"] == 2
    assert hubbard["initial_sector_report"]["n_up_target"] == 2


def test_runner_source_does_not_import_ed_dmrg_or_lanczos():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden = {"dmrg", "lanczos"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
            assert not (names & forbidden)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[-1] not in forbidden
    text = SCRIPT.read_text(encoding="utf-8")
    assert "exact_ground_energy" not in text
    assert "build_dense" not in text
    assert "to_dense(" not in text


@pytest.mark.skipif(not tc.cuda.is_available(), reason="CUDA not available")
def test_cuda_cli_smoke_when_available(tmp_path):
    spec = importlib.util.spec_from_file_location("run_ad_model_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        [
            "--model", "spinless_tv",
            "--N", "4",
            "--chi", "2",
            "--sweeps", "1",
            "--device", "cuda",
            "--dtype", "complex64",
            "--optimizer", "adam",
            "--local-steps", "1",
            "--lr", "0.01",
            "--init", "spinless_cdw",
            "--target-n", "2",
            "--output", str(tmp_path / "spinless_cuda.json"),
            "--no-ed",
        ]
    )
    data = module.run_ad_model_benchmark(args)
    assert data["device"] == "cuda"
    assert data["ed_status"] == "skipped by design"
    assert data["dmrg_lanczos_used"] is False
