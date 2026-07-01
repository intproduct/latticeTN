"""Stage 4R AD vs classical DMRG reference tests.

DMRG/Lanczos are a classical reference baseline ONLY; the AD solver is the
mainline. These tests confirm the AD final energy is close to the DMRG
reference and that DMRG is NOT used inside the AD optimization path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as tc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latticetn.mps import MPS  # noqa: E402
from latticetn.mpo import MPO  # noqa: E402
from latticetn.ad_variational import ADVariationalMPS, train_ad_mps  # noqa: E402
from latticetn import dmrg as D  # noqa: E402
from latticetn.operators import heisenberg_dense, exact_ground_energy  # noqa: E402
import latticetn.ad_variational as advmod  # noqa: E402

DTYPE = tc.complex128


def test_ad_final_energy_close_to_dmrg_reference():
    N = 5
    E0, _ = exact_ground_energy(heisenberg_dense(N, dtype=DTYPE))

    # DMRG reference (classical, NOT in the AD path)
    tc.manual_seed(0)
    mps_d = MPS(N, 2, 8, dtype=DTYPE)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device="cpu").generate_heisenberg(J=1.0)
    r_dmrg = D.run_dmrg(mps_d, mpo, chi=8, num_sweeps=4, solver="dense")
    e_dmrg = r_dmrg["final_energy"]

    # AD mainline
    tc.manual_seed(0)
    mps_a = MPS(N, 2, 8, dtype=DTYPE)
    ad = ADVariationalMPS(mps_a, mpo)
    r_ad = train_ad_mps(ad, num_steps=300, lr=1e-2, optimizer="adam")
    e_ad = r_ad["final_energy"]

    # both should be at-or-above exact and within 1e-3 of each other
    assert e_ad >= E0 - 1e-6
    assert e_dmrg >= E0 - 1e-6
    assert abs(e_ad - e_dmrg) < 1e-3, (e_ad, e_dmrg)


def test_ad_module_does_not_import_dmrg_into_loss_path():
    # The AD variational module must not call into the DMRG driver / local
    # eigensolver. Structural guard: ad_variational should not reference dmrg
    # or lanczos in its source.
    import inspect
    src = inspect.getsource(advmod)
    assert "from . import dmrg" not in src
    assert "import dmrg" not in src
    assert "from . import lanczos" not in src
    assert "import lanczos" not in src
    assert "run_dmrg" not in src
    assert "two_site_sweep" not in src
    assert "lanczos_lowest_eigenpair" not in src
