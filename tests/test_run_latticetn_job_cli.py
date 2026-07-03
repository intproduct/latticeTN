import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_latticetn_job.py"
EXAMPLE = ROOT / "examples" / "jobs" / "hubbard_ad_hard_N4.json"


def test_run_latticetn_job_cli_writes_result_json(tmp_path):
    out = tmp_path / "result.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, str(SCRIPT), "--job-json", str(EXAMPLE), "--output", str(out)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model"]["name"] == "hubbard"
    assert data["method"]["name"] == "ad_dmrg"
    assert data["diagnostics"]["ad_used"] is True
    assert data["diagnostics"]["ed_used"] is False
    assert data["observables"]["sector"]["n_up_abs_error"] < 1e-10
