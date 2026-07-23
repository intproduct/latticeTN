#!/usr/bin/env python3
"""Small open-boundary Heisenberg variational solve (MPS + MPO + autograd).

Runs gradient descent on the Rayleigh energy E = <psi|H|psi>/<psi|psi> for the
spin-1/2 Heisenberg chain on CPU, and prints/exits a JSON summary:

    {N, chi, steps, lr, seed, exact_E0, initial_E, final_E, abs_err, rel_err,
     pass, below_ground (bool)}

Conventions: H = J * sum_i S.S, S = sigma/2, J=1, open boundary.

Compliance with CLAUDE.md:
- CPU only (CUDA explicitly disabled).
- dtype complex128.
- The differentiable energy path uses NO .detach()/.data/.item()/no_grad.
  Normalization of the MPS after each optimizer step is done under no_grad and
  is OUTSIDE the energy computation, so autograd through the energy is preserved.
- The variational energy must not fall below the exact ground energy beyond a
  tolerance; if it does we report below_ground=true (and still exit non-zero so
  the validation loop surfaces it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch as tc

# allow `import latticetn` and `from tests.reference_models import ...`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from latticetn.mpo import MPO
from latticetn.mps import MPS
from latticetn.operators import heisenberg_dense, exact_ground_energy

DTYPE = tc.complex128


def _full_normalize(mps: MPS):
    """Right-canonical-ish normalization, in place via .data.

    Operates OUTSIDE the autograd energy path (no_grad + .data), so the energy
    computation itself stays fully differentiable. Mutating .data keeps the
    optimizer's reference to each parameter valid.
    """
    with tc.no_grad():
        for i in range(mps.N):
            t = mps.tensors[i]
            # <t|t> over (left_bond, phys) keeping the right bond as structure:
            n = tc.einsum("lsr,msr->lm", t.conj(), t)
            # Frobenius norm of the whole tensor
            total = float(t.norm().item())
            if total > 0:
                mps.tensors[i].data = (t / total).to(t.dtype).data


def solve(N, chi, steps, lr, seed, device="cpu", J=1.0, verbose=False):
    tc.manual_seed(seed)
    mps = MPS(N, 2, chi, dtype=DTYPE, device=device)
    mpo = MPO.from_bonds(N, 2, dtype=DTYPE, device=device).generate_heisenberg(J=J)
    H = heisenberg_dense(N, J=J, device=device)
    E0, _ = exact_ground_energy(H)

    _full_normalize(mps)
    e_init = float(mps.energy_with_MPO(mpo).detach())

    opt = tc.optim.Adam(mps.tensors, lr=lr)
    e_last = e_init
    for step in range(steps):
        e = mps.energy_with_MPO(mpo)
        opt.zero_grad()
        e.backward()
        opt.step()
        _full_normalize(mps)
        e_last = float(e)
        if verbose and (step % max(1, steps // 10) == 0):
            print(f"  step {step}: E={e_last:.8f} (E0={E0:.8f})", file=sys.stderr)

    e_final = float(mps.energy_with_MPO(mpo))
    return {
        "N": N, "chi": chi, "steps": steps, "lr": lr, "seed": seed, "J": J,
        "exact_E0": E0, "initial_E": e_init, "final_E": e_final,
        "abs_err": abs(e_final - E0),
        "rel_err": abs(e_final - E0) / max(1e-12, abs(E0)),
        "below_ground": bool(e_final < E0 - 1e-6),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--chi", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if args.device.startswith("cuda"):
        print("Refusing to use CUDA in validation; forcing cpu.", file=sys.stderr)
        args.device = "cpu"

    res = solve(args.N, args.chi, args.steps, args.lr, args.seed,
                device=args.device, verbose=args.verbose)
    res["pass"] = (not res["below_ground"]) and res["rel_err"] < 0.1
    print(json.dumps(res, indent=2))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
