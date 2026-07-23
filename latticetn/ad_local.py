"""Autograd local-tensor optimization (Stage 5A — the AD mainline).

This module is the autograd analogue of DMRG's local update, but the local
update is **gradient descent on a differentiable Rayleigh quotient**, not a
local eigensolver.

Main solver path
----------------
Fix every MPS tensor except one *center / local tensor*; make that single tensor
a trainable ``nn.Parameter`` and optimize the differentiable Rayleigh quotient
``E = <psi|H|psi>/<psi|psi>`` with PyTorch autograd + a torch optimizer. Sweep
the active site across the chain (right then left), moving the orthogonality
center by QR between sites, to optimize the whole state.

.. code-block:: text

   mixed-canonical MPS, all tensors frozen except center c
      -> train center tensor c as nn.Parameter
      -> loss = rayleigh_energy_native(mps, mpo)   (differentiable einsum sweep)
      -> loss.backward()   (autograd; only c receives a gradient)
      -> torch optimizer step (Adam / LBFGS) on c
      -> [optional post-step stabilization: none|tensor_norm|qr|canonical]
      -> move orthogonality center to the next site (QR center movement)
      -> repeat; sweep back and forth

Policy (docs/AD_MAINLINE_POLICY.md):
- The loss and the ``backward()`` + optimizer step are the ONLY optimization
  mechanism.
- SVD / QR / canonicalization / compression are permitted ONLY as post-step
  stabilization / projection / center movement, run under ``torch.no_grad()``
  mutating ``.data``, OUTSIDE the loss graph.
- ``dmrg.py`` / ``lanczos.py`` / any classical / dense local eigensolver
  (``eigh``) are reference baselines / oracles ONLY. This module imports NEITHER
  ``dmrg`` NOR ``lanczos``.

Autograd rule (hard): the loss path (``ADLocalOptimizer.energy`` / ``loss`` and
the underlying ``contractions.rayleigh_energy_native``) contains NO
``detach()`` / ``.data`` / ``torch.no_grad()`` / unnecessary ``.item()``, NO
``eigh``/``svd``/``qr``, NO call into ``dmrg``/``lanczos``. Every
``no_grad``/``.data``/``.detach()`` lives in explicitly-marked post-step /
center-movement helpers.

Conventions unchanged: S = sigma/2, J = 1.0, open boundary, complex128, CPU.
"""

from __future__ import annotations

from typing import Literal

import torch as tc

from .mps import MPS
from .mpo import MPO
from . import contractions as K
from . import canonical as Can  # Stage 3A canonicalization (gauge projection)


# ---------------------------------------------------------------------------
# build / freeze center
# ---------------------------------------------------------------------------

def _build_local_mps(mps: MPS, center: int) -> MPS:
    """Mixed-canonical copy of ``mps`` with center ``center``.

    All tensors are detached and frozen (``requires_grad=False``) EXCEPT the
    center tensor, which is promoted to a trainable ``nn.Parameter``. The
    canonicalization here is a preprocessing gauge step (Stage 3A, under
    ``no_grad``) — it is NOT part of the differentiable loss path. The Rayleigh
    quotient is exact for any gauge, so this only improves conditioning.
    """
    mc = Can.mixed_canonical(mps, center)
    new = MPS.from_tensors(mc.tensors, dtype=mc.dtype, device=mc.device,
                           requires_grad=False)
    # promote the center tensor to a trainable leaf parameter
    cdata = new.tensors[center].detach().clone()
    new.tensors[center] = tc.nn.Parameter(cdata)
    new.tensors[center].requires_grad_(True)
    return new


def _set_center(mps: MPS, center: int) -> None:
    """Freeze all tensors except ``center``; make ``center`` trainable.

    Toggles ``requires_grad`` on the leaf Parameters. Called between sites so
    the optimizer only ever trains the current center tensor. The center tensor
    is re-created as a fresh **contiguous** leaf Parameter so its ``.data`` (and
    the gradient autograd populates) are contiguous — LBFGS flattens the grad
    with ``.view(-1)`` and a non-contiguous center (e.g. after a QR/LQ center
    move or a canonical stabilization) would otherwise break it. Done under
    ``no_grad`` (parameter-config mutation, NOT a loss-path operation).
    """
    with tc.no_grad():
        for i, t in enumerate(mps.tensors):
            if i == center:
                # promote to a fresh contiguous trainable leaf
                mps.tensors[i] = tc.nn.Parameter(
                    t.detach().contiguous().clone(), requires_grad=True)
            else:
                t.requires_grad_(False)
                if t.grad is not None:
                    t.grad = None


# ---------------------------------------------------------------------------
# center movement (QR sweep, OUTSIDE the loss graph)
# ---------------------------------------------------------------------------

def _move_center_right(mps: MPS, c: int) -> None:
    """Shift the orthogonality center from site ``c`` to ``c+1`` via QR.

    Site ``c`` becomes left-canonical (orthonormal columns); the R factor is
    absorbed into site ``c+1``, which becomes the new center (carries the norm).
    This is center movement / gauge fixing, NOT the optimizer, and runs under
    ``no_grad`` mutating ``.data`` only.
    """
    with tc.no_grad():
        l, d, r = mps.tensors[c].shape
        Mat = mps.tensors[c].reshape(l * d, r)
        Q, R = tc.linalg.qr(Mat)               # Q:(l*d,k) col-orth, R:(k,r)
        k = Q.shape[1]
        mps.tensors[c].data = Q.reshape(l, d, k).data
        nxt = mps.tensors[c + 1]               # (r, d, r')
        mps.tensors[c + 1].data = tc.einsum("kr,rdc->kdc", R, nxt).data


def _move_center_left(mps: MPS, c: int) -> None:
    """Shift the orthogonality center from site ``c`` to ``c-1`` via LQ (QR).

    Site ``c`` becomes right-canonical (orthonormal rows); the L factor is
    absorbed into site ``c-1``, which becomes the new center. Center movement,
    NOT the optimizer; under ``no_grad`` mutating ``.data``.
    """
    with tc.no_grad():
        l, d, r = mps.tensors[c].shape
        Mat = mps.tensors[c].reshape(l, d * r)        # (l, d*r)
        Qt, Rm = tc.linalg.qr(Mat.t())                # Mat.t():(d*r,l)
        k = Rm.shape[0]
        B = Qt.t().reshape(k, d, r)                   # (k,d,r) row-orth
        mps.tensors[c].data = B.data
        C = Rm.t()                                    # (l, k)
        prev = mps.tensors[c - 1]                     # (l', d, l)
        mps.tensors[c - 1].data = tc.einsum("abc,ce->abe", prev, C).data


# ---------------------------------------------------------------------------
# post-step stabilization (OUTSIDE the loss graph)
# ---------------------------------------------------------------------------

def _stabilize(mps: MPS, center: int,
               stabilization: Literal["none", "tensor_norm", "qr",
                                      "canonical"]) -> None:
    """Post-step gauge/stability projection, OUTSIDE the autograd loss graph.

    - ``none``        : no projection.
    - ``tensor_norm`` : rescale the center tensor to unit Frobenius norm under
                        ``no_grad``. Scale-invariant for the Rayleigh quotient.
    - ``qr``          : mixed-canonical QR projection that keeps ``center`` as
                        the orthogonality center (gauge fixing without moving
                        the center). Written back onto ``.data`` under
                        ``no_grad``.
    - ``canonical``   : full left-canonical Stage-3A QR sweep written back onto
                        ``.data`` under ``no_grad`` (resets the global gauge;
                        the center nominally moves to the last site, but the
                        sweep driver re-promotes the chosen site afterwards).

    None of these touch the differentiable Rayleigh-quotient energy; they never
    appear inside the loss path.
    """
    if stabilization == "none":
        return
    with tc.no_grad():
        if stabilization == "tensor_norm":
            t = mps.tensors[center]
            n = t.norm()
            if n > 0:
                mps.tensors[center].data = (t / n).to(t.dtype).data
        elif stabilization == "qr":
            mc = Can.mixed_canonical(mps, center)
            for i in range(mps.N):
                mps.tensors[i].data = mc.tensors[i].detach().to(
                    dtype=mps.tensors[i].dtype,
                    device=mps.tensors[i].device).data
        elif stabilization == "canonical":
            lc = Can.left_canonical(mps)
            for i in range(mps.N):
                mps.tensors[i].data = lc.tensors[i].detach().to(
                    dtype=mps.tensors[i].dtype,
                    device=mps.tensors[i].device).data
        else:
            raise ValueError(
                "stabilization must be 'none'|'tensor_norm'|'qr'|'canonical', "
                f"got {stabilization!r}")


# ---------------------------------------------------------------------------
# diagnostics (report path; read scalars, allowed)
# ---------------------------------------------------------------------------

def _grad_norm(params) -> tc.Tensor:
    """Total L2 grad norm over trainable params (autograd-clean read)."""
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


def _canonical_error(mps: MPS) -> float:
    """Max left-orthonormality error over sites [0, N-2] (Stage 3A diagnostic)."""
    return float(Can.left_orthonormal_all(mps))


def _state_norm(mps: MPS) -> float:
    return float(K.native_norm(mps).real)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

class ADLocalOptimizer:
    """Differentiable local-tensor optimizer on an MPS / MPO.

    The MPS is held in mixed-canonical form with a chosen orthogonality
    ``center``. Every tensor is frozen except the center tensor, which is the
    only trainable parameter. The loss is the differentiable Rayleigh quotient
    ``E = <psi|H|psi>/<psi|psi>`` via ``contractions.rayleigh_energy_native``;

    ``move_center`` shifts the active site by a QR sweep (center movement, NOT
    the optimizer).
    """

    def __init__(self, mps: MPS, mpo: MPO, center: int = 0):
        if not (0 <= center < mps.N):
            raise ValueError(f"center {center} out of range for N={mps.N}")
        self.mps = _build_local_mps(mps, center)
        self.mpo = mpo
        self.N = mps.N
        self.dim = mps.dim
        self.dtype = mps.dtype
        self.device = mps.device
        self.center = center

    # ---- trainable params ----
    def parameters(self):
        return [self.mps.tensors[self.center]]

    def set_center(self, center: int) -> None:
        """Freeze all tensors except ``center``; record the new active site."""
        if not (0 <= center < self.N):
            raise ValueError(f"center {center} out of range for N={self.N}")
        _set_center(self.mps, center)
        self.center = center

    def move_center(self, new_center: int) -> None:
        """Shift the orthogonality center to ``new_center`` by QR sweeps.

        Steps one site at a time (right via QR, left via LQ) until the center
        reaches ``new_center``. Center movement, NOT the optimizer; runs under
        ``no_grad`` mutating ``.data``. After the move the new center is
        promoted to trainable and the old center is frozen.
        """
        if not (0 <= new_center < self.N):
            raise ValueError(f"new_center {new_center} out of range")
        c = self.center
        while c < new_center:
            _move_center_right(self.mps, c)
            c += 1
        while c > new_center:
            _move_center_left(self.mps, c)
            c -= 1
        self.set_center(new_center)

    # ---- differentiable loss (the AD mainline; autograd-clean) ----
    def energy(self) -> tc.Tensor:
        """Differentiable Rayleigh quotient E = <psi|H|psi>/<psi|psi>.

        Delegates to ``contractions.rayleigh_energy_native`` — a fully
        differentiable einsum sweep. Returns a real scalar with requires_grad.
        Only the center tensor receives a gradient (the others are frozen).
        """
        return K.rayleigh_energy_native(self.mps, self.mpo)

    def loss(self) -> tc.Tensor:
        """Alias of :meth:`energy`; the variational minimization target."""
        return self.energy()

    # ---- diagnostics (report path) ----
    def norm(self) -> tc.Tensor:
        return K.native_norm(self.mps)

    def max_bond_dim(self) -> int:
        if self.N > 1:
            return max(int(self.mps.tensors[i].shape[2]) for i in range(self.N - 1))
        return 1


def train_ad_local(mps: MPS, mpo: MPO, num_sweeps: int = 2,
                   local_steps: int = 50, lr: float = 1e-2,
                   optimizer: Literal["adam", "lbfgs"] = "adam",
                   lbfgs_iters: int = 20,
                   stabilization: Literal["none", "tensor_norm", "qr",
                                          "canonical"] = "none",
                   init_center: int = 0, record_every: int = 1,
                   verbose: bool = False) -> dict:
    """Train an MPS by AD local-tensor optimization (sweep of per-site Adam).

    For each sweep we visit every site (right then left). At each site we freeze
    the rest and run ``local_steps`` of a torch optimizer on the differentiable
    Rayleigh quotient, then optionally apply a post-step ``stabilization``
    projection, then move the orthogonality center to the next site by QR.

    The loss = ``adlo.energy()`` (differentiable Rayleigh quotient) is the ONLY
    thing ``backward()`` is called on. No ``no_grad`` wraps the loss;
    ``no_grad``/``.data`` appear only in ``_set_center`` / ``_move_center_*`` /
    ``_stabilize`` (post-step / center-movement, outside the loss path).
    DMRG/Lanczos/eigh are NEVER used.

    Returns a dict with per-step energy / grad-norm / state-norm /
    canonical-error histories plus per-sweep records, settings, and final
    diagnostics.
    """
    adlo = ADLocalOptimizer(mps, mpo, center=init_center)
    N = adlo.N

    energy_history: list[float] = []
    grad_history: list[float] = []
    norm_history: list[float] = []
    canon_history: list[float] = []
    sweeps: list[dict] = []

    def _record() -> float:
        e = float(adlo.energy().real)        # report-path scalar read
        energy_history.append(e)
        grad_history.append(float(_grad_norm(adlo.parameters())))
        norm_history.append(_state_norm(adlo.mps))
        canon_history.append(_canonical_error(adlo.mps))
        return e

    def _local_optimize(center: int) -> None:
        adlo.set_center(center)
        params = adlo.parameters()
        if optimizer == "adam":
            opt = tc.optim.Adam(params, lr=lr)
            for _ in range(local_steps):
                opt.zero_grad()
                e = adlo.energy()
                e.backward()
                opt.step()
                _stabilize(adlo.mps, center, stabilization)
        elif optimizer == "lbfgs":
            opt = tc.optim.LBFGS(params, lr=lr, max_iter=lbfgs_iters,
                                 line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                e = adlo.energy()
                e.backward()
                return e
            for _ in range(max(1, local_steps // max(1, lbfgs_iters))):
                opt.step(closure)
                _stabilize(adlo.mps, center, stabilization)
        else:
            raise ValueError(
                f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}")

    e0 = _record()                              # initial energy
    sweep_idx = 0
    for s in range(num_sweeps):
        direction = "right" if (s % 2 == 0) else "left"
        order = range(N) if direction == "right" else range(N - 1, -1, -1)
        per_site: list[int] = []
        for c in order:
            _local_optimize(c)
            # move orthogonality center toward the next site in this sweep
            if direction == "right" and c < N - 1:
                adlo.move_center(c + 1)
            elif direction == "left" and c > 0:
                adlo.move_center(c - 1)
            if verbose:
                import sys
                print(f"  sweep {s} site {c}: E={float(adlo.energy()):.8f}",
                      file=sys.stderr)
        e_after = _record()
        sweeps.append({
            "sweep": s, "direction": direction,
            "energy_after": e_after, "center": adlo.center,
        })
        sweep_idx += 1

    if (num_sweeps % record_every) != 0:
        _record()
    final_e = float(adlo.energy().real)
    return {
        "energy_history": energy_history,
        "grad_norm_history": grad_history,
        "state_norm_history": norm_history,
        "norm_history": norm_history,             # back-compat alias
        "canonical_error_history": canon_history,
        "sweeps": sweeps,
        "initial_energy": e0,
        "final_energy": final_e,
        "max_bond": adlo.max_bond_dim(),
        "num_sweeps": num_sweeps,
        "local_steps": local_steps,
        "lr": lr,
        "optimizer": optimizer,
        "stabilization": stabilization,
        "init_center": init_center,
    }
