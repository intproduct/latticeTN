"""Autograd-first variational MPS solver (Stage 4R — the AD mainline).

The PRIMARY optimization path in latticeTN is a differentiable Rayleigh
quotient optimized by PyTorch autograd + a torch optimizer. The classical
Lanczos/DMRG code (Stage 4A/4B) is a reference baseline ONLY and is NOT used
in this path.

This module provides:

- ``ADVariationalMPS`` — an MPS whose tensors are ``torch.nn.Parameter``s
  (trainable), plus helpers to build the Heisenberg MPO and the differentiable
  loss ``E = <psi|H|psi>/<psi|psi>`` via ``contractions.rayleigh_energy_native``.
- ``train_ad_mps`` — a training loop with ``Adam`` or ``LBFGS`` supporting
  energy history, gradient norm, the MPS norm, and max bond dimension.

Autograd rule (CLAUDE.md): the loss path uses NO ``.detach()`` / ``.data`` /
unnecessary ``.item()`` / ``torch.no_grad()``. Per-tensor L2 renormalization
after each optimizer step is performed OUTSIDE the loss graph (under
``no_grad`` mutating ``.data``), exactly as the Stage 1 ``_full_normalize``
routine does — it preserves a valid parameter reference and does NOT touch the
differentiable energy computation.

Conventions unchanged: S = sigma/2, J = 1.0, open boundary, complex128.
"""

from __future__ import annotations

from typing import Literal

import torch as tc

from .mps import MPS
from .mpo import MPO
from . import contractions as K
from . import canonical as Can  # Stage 3A canonicalization (gauge projection)


class ADVariationalMPS:
    """Differentiable variational MPS for an MPO Hamiltonian.

    `mps` is an :class:`MPS` whose tensors are ``nn.Parameter``s with
    ``requires_grad=True`` (the Stage 1 MPS constructor already wraps each site
    tensor in ``nn.Parameter``, so passing an ordinary MPS gives trainable
    parameters). The loss is the Rayleigh quotient
    ``E = <psi|H|psi>/<psi|psi>`` evaluated by the native differentiable
    contraction ``contractions.rayleigh_energy_native``.
    """

    def __init__(self, mps: MPS, mpo: MPO):
        # Ensure tensors are leaf parameters requiring grad.
        params = []
        for t in mps.tensors:
            if not isinstance(t, tc.nn.Parameter):
                t = tc.nn.Parameter(t.detach().clone())
            t.requires_grad_(True)
            params.append(t)
        mps.tensors = tc.nn.ParameterList(params)
        self.mps = mps
        self.mpo = mpo
        self.N = mps.N
        self.dim = mps.dim
        self.dtype = mps.dtype
        self.device = mps.device

    # ---- trainable params ----
    def parameters(self):
        return self.mps.tensors

    # ---- differentiable loss (the AD mainline; autograd-clean) ----
    def energy(self) -> tc.Tensor:
        """Differentiable Rayleigh quotient E = <psi|H|psi>/<psi|psi>.

        Delegates to contractions.rayleigh_energy_native, a fully
        differentiable contraction. Returns a real scalar with requires_grad.
        """
        return K.rayleigh_energy_native(self.mps, self.mpo)

    def loss(self) -> tc.Tensor:
        """Alias of :meth:`energy`; the variational minimization target."""
        return self.energy()

    # ---- diagnostics (report path; these do read scalars, allowed) ----
    def norm(self) -> tc.Tensor:
        """sqrt(<psi|psi>) via the native differentiable contraction."""
        return K.native_norm(self.mps)

    def max_bond_dim(self) -> int:
        return max(int(self.mps.tensors[i].shape[2]) for i in range(self.N - 1)) \
            if self.N > 1 else 1


def _renormalize(mps: MPS) -> None:
    """Per-tensor L2 renormalization, OUTSIDE the loss graph.

    Mirrors Stage 1 ``run_heisenberg_small._full_normalize``: under ``no_grad``
    mutating ``.data``. This keeps parameter magnitudes bounded and the MPS
    norm well-defined; because the Rayleigh quotient is scale-invariant it does
    NOT change the physics. Critically it is NOT part of the differentiable
    energy path.
    """
    with tc.no_grad():
        for i in range(mps.N):
            t = mps.tensors[i]
            total = t.norm()
            if total > 0:
                mps.tensors[i].data = (t / total).to(t.dtype).data


def _grad_norm(params) -> tc.Tensor:
    """Total L2 grad norm over all trainable params (autograd-clean read)."""
    sq = tc.tensor(0.0, dtype=tc.float64)
    seen = False
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = sq + (g.conj() * g).real.sum() if g.is_complex() else sq + (g * g).sum()
        seen = True
    if not seen:
        return tc.tensor(0.0, dtype=tc.float64)
    return sq.sqrt()


def _project(mps: MPS, projection: str) -> None:
    """Post-step gauge projection, OUTSIDE the autograd loss graph.

    projection:
      - 'none'        : no projection.
      - 'tensor_norm' : per-tensor L2 renormalization (Stage 4R behavior).
      - 'canonical'   : left-canonical QR sweep (Stage 3A) written back onto
                        the trainable ``.data`` of each parameter. The dense
                        state is preserved up to a global phase; the gauge is
                        brought toward left-canonical form (sites are made
                        left-orthonormal).

    All variants run under ``torch.no_grad`` and mutate ``.data`` only, so the
    trainable ``nn.Parameter`` leaf identity (and the optimizer's reference) is
    preserved. None of these touch the differentiable Rayleigh-quotient energy.
    """
    if projection == "none":
        return
    with tc.no_grad():
        if projection == "tensor_norm":
            _renormalize(mps)
        elif projection == "canonical":
            # Stage 3A left_canonical returns a new MPS wrapping detached
            # tensors; bring the canonical tensors back onto the live params.
            canon = Can.left_canonical(mps)
            for i in range(mps.N):
                mps.tensors[i].data = canon.tensors[i].detach().to(
                    dtype=mps.tensors[i].dtype, device=mps.tensors[i].device).data
        else:
            raise ValueError(
                f"projection must be 'none'|'tensor_norm'|'canonical', got {projection!r}")


def _canonical_error(mps: MPS) -> float:
    """Max left-orthonormality error over sites [0, N-2] (Stage 3A diagnostic).

    A pure left-canonical MPS has zero error everywhere except the last site.
    This is a REPORT/diagnostic read (uses .detach internally in the Stage 3A
    helper), NOT part of the loss path.
    """
    try:
        return float(Can.left_orthonormal_all(mps))
    except Exception:
        return float("nan")


def _state_norm(mps: MPS) -> float:
    """sqrt(<psi|psi>) via the native differentiable norm, read as a float."""
    return float(K.native_norm(mps).real)


def train_ad_mps(admps: ADVariationalMPS, num_steps: int = 200, lr: float = 1e-2,
                 optimizer: Literal["adam", "lbfgs"] = "adam",
                 lbfgs_iters: int = 20,
                 projection: Literal["none", "tensor_norm", "canonical"] = "tensor_norm",
                 renormalize: bool | None = None,
                 record_every: int = 1, verbose: bool = False
                 ) -> dict:
    """Train an ADVariationalMPS by gradient descent on the Rayleigh quotient.

    projection selects a post-step gauge projection (OUTSIDE the loss graph):
      - 'none'        : no projection
      - 'tensor_norm' : per-tensor L2 renormalization (Stage 4R default)
      - 'canonical'   : left-canonical QR projection (Stage 3A)

    Returns a dict with:
      - ``energy_history``    : list[float] (initial + every record_every steps)
      - ``grad_norm_history`` : list[float]
      - ``norm_history``      : list[float], sqrt(<psi|psi>) ("state norm")
      - ``canonical_error_history`` : list[float], max left-orthonormality error
      - ``initial_energy`` / ``final_energy``
      - ``max_bond``          : int (fixed by init)
      - ``optimizer`` / ``projection`` / ``num_steps`` / ``lr``

    Autograd compliance: the loss = ``admps.energy()`` is the differentiable
    Rayleigh quotient and is the ONLY thing ``backward`` is called on. No
    ``no_grad`` wraps the loss; ``no_grad`` appears only in ``_project`` (a
    post-step projection outside the graph). Projection / canonicalization /
    DMRG / Lanczos are NEVER inside the loss path.
    """
    # Back-compat: `renormalize=True` maps to tensor_norm; False -> none.
    if renormalize is not None:
        projection = "tensor_norm" if renormalize else "none"

    params = list(admps.parameters())
    if optimizer == "adam":
        opt = tc.optim.Adam(params, lr=lr)
    elif optimizer == "lbfgs":
        opt = tc.optim.LBFGS(params, lr=lr, max_iter=lbfgs_iters,
                             line_search_fn="strong_wolfe")
    else:
        raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}")

    energy_history: list[float] = []
    grad_history: list[float] = []
    norm_history: list[float] = []
    canon_history: list[float] = []

    def _record() -> float:
        e = float(admps.energy())          # report-path scalar read
        energy_history.append(e)
        grad_history.append(float(_grad_norm(params)))
        norm_history.append(_state_norm(admps.mps))
        canon_history.append(_canonical_error(admps.mps))
        return e

    e0 = _record()                          # initial energy
    admps.initial_energy = e0

    if optimizer == "adam":
        for step in range(num_steps):
            opt.zero_grad()
            e = admps.energy()
            e.backward()
            opt.step()
            _project(admps.mps, projection)
            if (step + 1) % record_every == 0:
                _record()
            if verbose and (step % max(1, num_steps // 10) == 0):
                print(f"  step {step}: E={float(admps.energy()):.8f}",
                      file=__import__("sys").stderr)
    else:  # lbfgs
        def closure():
            opt.zero_grad()
            e = admps.energy()
            e.backward()
            return e
        for step in range(num_steps):
            opt.step(closure)
            _project(admps.mps, projection)
            if (step + 1) % record_every == 0:
                _record()

    # ensure final state diagnostics reflect the post-projection state
    if (num_steps % record_every) != 0:
        _record()
    final_e = float(admps.energy())
    return {
        "energy_history": energy_history,
        "grad_norm_history": grad_history,
        "state_norm_history": norm_history,
        "norm_history": norm_history,             # back-compat alias (Stage 4R)
        "canonical_error_history": canon_history,
        "initial_energy": e0,
        "final_energy": final_e,
        "max_bond": admps.max_bond_dim(),
        "optimizer": optimizer,
        "projection": projection,
        "num_steps": num_steps,
        "lr": lr,
    }
