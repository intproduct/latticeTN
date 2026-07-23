"""Autograd two-site local-tensor optimization (Stage 5B — the AD mainline).

This module is the two-site autograd analogue of DMRG's local update, but the
local update is **gradient descent on a differentiable two-site Rayleigh
quotient**, NOT a local eigensolver.

Main solver path
----------------
Bring the MPS into two-site mixed-canonical form at bond (i, i+1) (sites
[0, i) left-canonical, sites (i+1, N) right-canonical, block (i, i+1) free).
Contract the two adjacent site tensors into a single two-site center tensor
``Θ(l, s_i, s_{i+1}, r)`` and make it the ONLY trainable ``nn.Parameter``.
The loss is the differentiable local Rayleigh quotient

    E(Θ) = <Θ| H_eff |Θ> / <Θ|Θ>

where ``H_eff`` is the frozen two-site effective Hamiltonian built from the
(frozen, detached) left/right MPO environments and the two MPO tensors W_i,
W_{i+1}. Because the rest of the chain is orthonormal, E(Θ) equals the GLOBAL
Rayleigh quotient ``<ψ|H|ψ>/<ψ|ψ>`` — minimizing it lowers the global energy.

.. code-block:: text

   two-site mixed-canonical MPS at bond (i, i+1)
      -> build frozen left/right MPO environments L, R  (constants)
      -> Theta = A_i * A_{i+1}                         (single trainable leaf)
      -> loss = <Theta|H_eff|Theta> / <Theta|Theta>    (differentiable einsum)
      -> loss.backward()                               (autograd; only Theta grads)
      -> torch optimizer step (Adam / LBFGS) on Theta
      -> [post-step split: SVD Theta -> A_i, A_{i+1} with optional chi/cutoff]
      -> move to next bond; sweep left-to-right then right-to-left

Policy (docs/AD_MAINLINE_POLICY.md, docs/AD_TWO_SITE_SPEC.md):
- The loss and the ``backward()`` + optimizer step are the ONLY optimization
  mechanism. Gradient descent on Θ, NOT eigh / Lanczos / classical DMRG.
- SVD / QR / canonicalization are permitted ONLY as the post-step split /
  compression / stabilization and the inter-bond re-canonicalization, run under
  ``torch.no_grad()`` mutating detached data, OUTSIDE the loss graph. They are
  NOT the solver.
- ``dmrg.py`` / ``lanczos.py`` / any classical / dense local eigensolver
  (``eigh``) are reference baselines / oracles ONLY. This module imports NEITHER
  ``dmrg`` NOR ``lanczos``.

Autograd rule (hard): the loss path (``ADTwoSiteOptimizer.energy`` / ``loss``)
contains NO ``detach()`` / ``.data`` / ``torch.no_grad()`` / unnecessary
``.item()``, NO ``eigh``/``svd``/``qr``, NO call into ``dmrg``/``lanczos`` nor
into the split / canonicalization helpers. Every ``no_grad`` / ``.data`` /
``.detach()`` / ``svd`` / ``qr`` lives in explicitly-marked post-step /
preprocessing helpers.

Conventions unchanged: H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open
boundary, complex128, CPU-only.
"""

from __future__ import annotations

import math
from typing import Literal

import torch as tc

from .mps import MPS
from .mpo import MPO
from . import contractions as K
from .numerics import positive, real_if_hermitian, truncation_error


# ---------------------------------------------------------------------------
# preprocessing: two-site mixed canonical form + frozen MPO environments
# (gauge fixing / constant building, OUTSIDE the loss graph)
# ---------------------------------------------------------------------------

def _two_site_mixed_canonical(mps: MPS, i: int) -> None:
    """Bring ``mps`` into two-site mixed-canonical form at bond (i, i+1).

    Sites [0, i) are left-canonical (QR), sites (i+2, N) are right-canonical
    (LQ); the two-site block (i, i+1) is left free. The dense state is
    preserved up to a global phase. In-place gauge fixing under ``no_grad``
    (mutates ``.data``); NOT part of the differentiable loss path.
    """
    N = mps.N
    if not (0 <= i < N - 1):
        raise ValueError(f"bond index i={i} out of range for N={N}")
    t = mps.tensors
    with tc.no_grad():
        # left sweep over [0, i): left-canonicalize site k, push R into k+1
        for k in range(0, i):
            l, d, r = t[k].shape
            Q, R = tc.linalg.qr(t[k].reshape(l * d, r))
            kk = Q.shape[1]
            t[k] = tc.nn.Parameter(Q.reshape(l, d, kk).contiguous(),
                                   requires_grad=False)
            t[k + 1] = tc.nn.Parameter(
                tc.einsum("kr,rdc->kdc", R, t[k + 1]).contiguous(),
                requires_grad=False)
        # right sweep over (i+1, N): right-canonicalize site k, push L into k-1
        for k in range(N - 1, i + 1, -1):
            l, d, r = t[k].shape
            Qt, Rm = tc.linalg.qr(t[k].reshape(l, d * r).t())
            kk = Rm.shape[0]
            B = Qt.t().reshape(kk, d, r).contiguous()
            t[k] = tc.nn.Parameter(B, requires_grad=False)
            C = Rm.t()                                   # (l, k)
            t[k - 1] = tc.nn.Parameter(
                tc.einsum("abc,ce->abe", t[k - 1], C).contiguous(),
                requires_grad=False)


def _left_mpo_env(tensors, Ws, up_to: int) -> tc.Tensor:
    """Left MPO environment at bond ``up_to``: shape (lb_bra, lb_mpo, lb_ket).

    Pure einsum on detached constant tensors (preprocessing, NOT the loss path).
    Verified einsum (mirrors the Stage 4A reference).
    """
    v = tc.ones((1, 1, 1), dtype=tensors[0].dtype, device=tensors[0].device)
    for k in range(up_to):
        v = tc.einsum("lmr,lsb,mtys,ryz->btz", v,
                      tensors[k].conj(), Ws[k], tensors[k])
    return v


def _right_mpo_env(tensors, Ws, from_: int) -> tc.Tensor:
    """Right MPO environment at bond ``from_``: shape (rb_bra, rb_mpo, rb_ket).

    Pure einsum on detached constant tensors (preprocessing, NOT the loss path).
    """
    v = tc.ones((1, 1, 1), dtype=tensors[-1].dtype, device=tensors[-1].device)
    for k in range(len(tensors) - 1, from_ - 1, -1):
        v = tc.einsum("abc,dga,ebfg,hfc->deh", v,
                      tensors[k].conj(), Ws[k], tensors[k])
    return v


# ---------------------------------------------------------------------------
# post-step split: SVD Theta -> two site tensors (compression, NOT the solver)
# ---------------------------------------------------------------------------

def _split_theta(theta: tc.Tensor, max_bond_dim: int | None,
                 cutoff: float | None,
                 direction: Literal["right", "left"]
                 ) -> tuple[list[tc.Tensor], float, int]:
    """SVD-split a two-site block into two site tensors with optional truncation.

    Theta: (l, s_i, s_{i+1}, r).
    - direction='right': A_i = U (left-canonical), A_{i+1} = S Vh (carries norm).
    - direction='left':  A_i = U S (carries norm), A_{i+1} = Vh (right-canonical).

    Truncation: keep the largest ``min(max_bond_dim, k0)`` singular values,
    additionally dropping any singular values with s^2 below ``cutoff``.
    Runs under ``torch.no_grad`` on a detached Theta (post-step compression /
    stabilization, OUTSIDE the loss graph — NEVER the solver).

    Returns ([A_i, A_{i+1}], truncation_error, kept_bond) where
    truncation_error = discarded weight = sum(discarded s^2)/sum(all s^2) in [0,1].
    """
    l, si, si1, r = theta.shape
    M = tc.as_tensor(theta).detach().reshape(l * si, si1 * r)
    U, S, Vh = tc.linalg.svd(M, full_matrices=False)
    s2 = S.real ** 2
    k0 = S.shape[0]

    # cutoff mask (drop tiny singular values)
    keep_mask = tc.ones(k0, dtype=tc.bool)
    if cutoff is not None and cutoff > 0:
        keep_mask = (s2 >= float(cutoff))
    k = int(keep_mask.sum().item())
    if max_bond_dim is not None:
        k = min(k, int(max_bond_dim))
    k = max(1, k)

    trunc = truncation_error(s2, k, name="two-site AD split")

    with tc.no_grad():
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
        if direction == "right":
            A_i = U.reshape(l, si, k)                       # left-canonical
            A_ip1 = (S.reshape(k, 1) * Vh).reshape(k, si1, r)
        elif direction == "left":
            A_i = (U * S.reshape(1, k)).reshape(l, si, k)
            A_ip1 = Vh.reshape(k, si1, r)                   # right-canonical
        else:
            raise ValueError(
                f"direction must be 'right' or 'left', got {direction!r}")
    return [A_i.contiguous(), A_ip1.contiguous()], trunc, k


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


def _bond_dims(mps: MPS) -> list[int]:
    return [int(mps.tensors[i].shape[2]) for i in range(mps.N - 1)]


def _max_bond(mps: MPS) -> int:
    bd = _bond_dims(mps)
    return max(bd) if bd else 1


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

class ADTwoSiteOptimizer:
    """Differentiable two-site local-tensor optimizer on an MPS / MPO.

    The MPS is held in two-site mixed-canonical form at ``bond`` (sites [0, i)
    left-canonical, sites (i+1, N) right-canonical). The two adjacent site
    tensors are contracted into a single trainable two-site center tensor
    ``Θ``; the left/right MPO environments and the two MPO tensors are frozen
    constants. The loss is the differentiable local Rayleigh quotient
    ``E(Θ) = <Θ|H_eff|Θ> / <Θ|Θ>`` (equal to the global Rayleigh quotient while
    the chain is orthonormal around the block).

    ``move_bond`` shifts the active two-site block by re-canonicalizing the chain
    at the new bond (gauge fixing, NOT the optimizer). ``split`` writes the
    optimized Θ back as two site tensors via SVD with optional truncation
    (compression/stabilization, NOT the optimizer).
    """

    def __init__(self, mps: MPS, mpo: MPO, bond: int = 0):
        if not (0 <= bond < mps.N - 1):
            raise ValueError(f"bond {bond} out of range for N={mps.N} "
                             f"(need 0 <= bond < N-1)")
        self.mps = mps
        self.mpo = mpo
        self.N = mps.N
        self.dim = mps.dim
        self.dtype = mps.dtype
        self.device = mps.device
        self.bond = bond
        # frozen constant Theta shape (filled by reset_bond)
        self._L = None
        self._R = None
        self._Wi = None
        self._Wi1 = None
        self.theta = None
        self.reset_bond(bond)

    # ---- preprocessing (gauge fix + frozen environments) ----
    def reset_bond(self, bond: int) -> None:
        """Re-canonicalize at ``bond`` and rebuild frozen environments + Theta.

        Gauge fixing + constant building, OUTSIDE the loss graph. The new Θ is a
        fresh contiguous trainable leaf Parameter.
        """
        if not (0 <= bond < self.N - 1):
            raise ValueError(f"bond {bond} out of range for N={self.N}")
        self.bond = bond
        i = bond
        _two_site_mixed_canonical(self.mps, i)
        # frozen constant environments (detached)
        t = [x.detach() for x in self.mps.tensors]
        Ws = self.mpo.tensors
        with tc.no_grad():
            self._L = _left_mpo_env(t, Ws, up_to=i).detach()
            self._R = _right_mpo_env(t, Ws, from_=i + 2).detach()
            self._Wi = Ws[i].detach()
            self._Wi1 = Ws[i + 1].detach()
            theta = tc.einsum("lsc,cer->lser", t[i], t[i + 1])
        self.theta = tc.nn.Parameter(theta.detach().contiguous().clone(),
                                     requires_grad=True)

    # ---- trainable params ----
    def parameters(self):
        return [self.theta]

    def normalize_theta_(self, eps: float = 1e-300) -> float:
        """Normalize the active two-site center outside the loss graph.

        This is a numerical preconditioner for the scale-invariant Rayleigh
        quotient. It preserves ``theta`` as the same trainable leaf Parameter
        and never enters the differentiable energy graph.
        """
        with tc.no_grad():
            n = self.theta.norm()
            n_float = float(n.detach().real.cpu())
            if (not math.isfinite(n_float)) or n_float <= eps:
                raise FloatingPointError(
                    "cannot normalize two-site theta: Frobenius norm is "
                    f"{n_float!r}; check MPS gauge/initialization"
                )
            self.theta.div_(n.to(dtype=self.theta.dtype, device=self.theta.device))
            return n_float

    # ---- differentiable loss (the AD mainline; autograd-clean) ----
    def energy(self) -> tc.Tensor:
        """Differentiable local Rayleigh quotient E = <Θ|H_eff|Θ>/<Θ|Θ>.

        Pure einsum on the trainable Θ and the frozen constant environments /
        MPO tensors. Returns a real scalar with ``requires_grad``. Only Θ
        receives a gradient. No ``no_grad``/``detach``/``.data``/``.item``, no
        ``eigh``/``svd``/``qr``, no call into any split/canonicalization helper.
        """
        theta = self.theta
        # H_eff |Theta>  ->  same (l, si, si1, r) shape as Theta
        Htheta = tc.einsum("pqr,qsab,stcd,utw,racw->pbdu",
                           self._L, self._Wi, self._Wi1, self._R, theta)
        num = (theta.conj() * Htheta).sum()
        den = positive(
            (theta.conj() * theta).sum(),
            name="two-site AD Rayleigh denominator",
        )
        return real_if_hermitian(
            num / den,
            name="two-site AD Rayleigh energy",
        )

    def loss(self) -> tc.Tensor:
        """Alias of :meth:`energy`; the variational minimization target."""
        return self.energy()

    # ---- post-step split (compression, NOT the solver) ----
    def split(self, max_bond_dim: int | None = None, cutoff: float | None = None,
              direction: Literal["right", "left"] = "right"
              ) -> tuple[float, int]:
        """Write Θ back as two site tensors via SVD with optional truncation.

        Post-step compression / stabilization, OUTSIDE the loss graph. Returns
        (truncation_error, kept_bond). NOT the optimizer.
        """
        new_t, trunc, k = _split_theta(self.theta.detach(), max_bond_dim,
                                       cutoff, direction)
        i = self.bond
        with tc.no_grad():
            self.mps.tensors[i] = tc.nn.Parameter(
                new_t[0].to(dtype=self.mps.tensors[i].dtype,
                             device=self.mps.tensors[i].device).contiguous(),
                requires_grad=False)
            self.mps.tensors[i + 1] = tc.nn.Parameter(
                new_t[1].to(dtype=self.mps.tensors[i + 1].dtype,
                             device=self.mps.tensors[i + 1].device).contiguous(),
                requires_grad=False)
        return trunc, k

    # ---- diagnostics (report path) ----
    def global_energy(self) -> tc.Tensor:
        """Global Rayleigh quotient on the full current MPS (report path)."""
        return K.rayleigh_energy_native(self.mps, self.mpo)

    def norm(self) -> tc.Tensor:
        return K.native_norm(self.mps)

    def max_bond_dim(self) -> int:
        return _max_bond(self.mps)

    def bond_dims(self) -> list[int]:
        return _bond_dims(self.mps)


def train_ad_two_site(mps: MPS, mpo: MPO, num_sweeps: int = 2,
                      local_steps: int = 20, lr: float = 1.0,
                      optimizer: Literal["adam", "lbfgs"] = "lbfgs",
                      lbfgs_iters: int = 20,
                      lbfgs_tolerance_grad: float = 1e-12,
                      lbfgs_tolerance_change: float = 1e-15,
                      max_bond_dim: int | None = None,
                      cutoff: float | None = None,
                      precondition: Literal["theta_norm", "none"] = "theta_norm",
                      stabilization: Literal["none", "tensor_norm"] = "none",
                      init_bond: int = 0,
                      verbose: bool = False) -> dict:
    """Train an MPS by two-site AD local-tensor optimization.

    Alternates right (left-to-right) and left (right-to-left) sweeps. At each
    bond the chain is brought into two-site mixed-canonical form, the two-site
    center tensor Θ is trained as the only ``nn.Parameter`` on the
    differentiable local Rayleigh quotient via ``loss.backward()`` + a torch
    optimizer step, then Θ is split back into two site tensors by SVD with
    optional ``max_bond_dim`` / ``cutoff`` truncation. Re-canonicalization at
    the next bond is gauge fixing; the SVD split is compression/stabilization.
    NEITHER is the solver. No Lanczos / eigh / classical DMRG is ever used.

    Returns a dict with per-sweep energy / grad-norm / bond-dim /
    truncation-error / sweep-direction history plus settings and final
    diagnostics.
    """
    adtso = ADTwoSiteOptimizer(mps, mpo, bond=init_bond)
    N = adtso.N

    energy_history: list[float] = []
    grad_history: list[float] = []
    bond_history: list[int] = []
    trunc_history: list[float] = []
    sweeps: list[dict] = []
    per_bond_trunc: list[list[float]] = []
    precondition_norms: list[float | None] = []
    optimizer_steps = 0
    closure_evals = 0

    trunc_now: list[float] = []

    def _global_e() -> float:
        return float(adtso.global_energy().detach().real.cpu())

    def _local_optimize(bond: int, direction: str) -> tuple[float, int]:
        nonlocal optimizer_steps, closure_evals
        adtso.reset_bond(bond)
        if precondition == "theta_norm":
            precondition_norms.append(adtso.normalize_theta_())
        elif precondition == "none":
            precondition_norms.append(None)
        else:
            raise ValueError(
                "precondition must be 'theta_norm' or 'none', "
                f"got {precondition!r}"
            )
        params = adtso.parameters()
        if optimizer == "adam":
            opt = tc.optim.Adam(params, lr=lr)
            for _ in range(local_steps):
                opt.zero_grad()
                e = adtso.energy()
                closure_evals += 1
                e.backward()
                opt.step()
                optimizer_steps += 1
                if stabilization == "tensor_norm":
                    _stabilize_tensor_norm(adtso)
        elif optimizer == "lbfgs":
            opt = tc.optim.LBFGS(params, lr=lr, max_iter=lbfgs_iters,
                                 tolerance_grad=lbfgs_tolerance_grad,
                                 tolerance_change=lbfgs_tolerance_change,
                                 line_search_fn="strong_wolfe")

            def closure():
                nonlocal closure_evals
                opt.zero_grad()
                e = adtso.energy()
                closure_evals += 1
                e.backward()
                return e
            for _ in range(max(1, local_steps // max(1, lbfgs_iters))):
                opt.step(closure)
                optimizer_steps += 1
                if stabilization == "tensor_norm":
                    _stabilize_tensor_norm(adtso)
        else:
            raise ValueError(
                f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}")
        trunc, kept = adtso.split(max_bond_dim=max_bond_dim, cutoff=cutoff,
                                  direction=direction)
        return trunc, kept

    e0 = _global_e()
    best_energy = e0
    best_step = 0
    best_tensors = [tensor.detach().clone() for tensor in adtso.mps.tensors]
    energy_history.append(e0)
    grad_history.append(0.0)
    bond_history.append(adtso.max_bond_dim())
    trunc_history.append(0.0)

    for s in range(num_sweeps):
        direction = "right" if (s % 2 == 0) else "left"
        bonds = (range(N - 1) if direction == "right"
                 else range(N - 2, -1, -1))
        trunc_now = []
        bond_trunc: list[float] = []
        for b in bonds:
            trunc, kept = _local_optimize(b, direction)
            trunc_now.append(trunc)
            bond_trunc.append(trunc)
            if verbose:
                import sys
                print(f"  sweep {s} bond {b} ({direction}): "
                      f"E={_global_e():.10f} trunc={trunc:.2e} kept={kept}",
                      file=sys.stderr)
        e_after = _global_e()
        if e_after < best_energy:
            best_energy = e_after
            best_step = s + 1
            best_tensors = [
                tensor.detach().clone() for tensor in adtso.mps.tensors
            ]
        energy_history.append(e_after)
        grad_history.append(float(_grad_norm(adtso.parameters())))
        bond_history.append(adtso.max_bond_dim())
        trunc_history.append(max(bond_trunc) if bond_trunc else 0.0)
        per_bond_trunc.append(bond_trunc)
        sweeps.append({
            "sweep": s, "direction": direction,
            "energy_after": e_after, "bond": adtso.bond,
            "max_trunc": max(bond_trunc) if bond_trunc else 0.0,
            "per_bond_trunc": bond_trunc,
            "max_bond": adtso.max_bond_dim(),
        })

    # Restore the state that generated the reported minimum so downstream
    # observables and bond diagnostics refer to that same physical state.
    with tc.no_grad():
        for i, best in enumerate(best_tensors):
            adtso.mps.tensors[i] = tc.nn.Parameter(
                best.to(dtype=adtso.dtype, device=adtso.device).contiguous(),
                requires_grad=False,
            )
    final_e = _global_e()
    tolerance = 1e-10 if adtso.dtype == tc.complex128 else 2e-5
    if abs(final_e - best_energy) > tolerance:
        raise RuntimeError(
            "restored two-site best MPS does not reproduce best energy: "
            f"|{final_e} - {best_energy}| > {tolerance}"
        )
    return {
        "energy_history": energy_history,
        "grad_norm_history": grad_history,
        "bond_dim_history": bond_history,
        "truncation_error_history": trunc_history,
        "sweeps": sweeps,
        "per_bond_truncation": per_bond_trunc,
        "initial_energy": e0,
        "final_energy": float(final_e),
        "best_energy": float(final_e),
        "best_step": best_step,
        "final_state_source": "best_mps",
        "max_bond": adtso.max_bond_dim(),
        "final_bond_dims": adtso.bond_dims(),
        "num_sweeps": num_sweeps,
        "local_steps": local_steps,
        "lr": lr,
        "optimizer": optimizer,
        "optimizer_steps": optimizer_steps,
        "closure_evals": closure_evals,
        "max_bond_dim": max_bond_dim,
        "cutoff": cutoff,
        "precondition": precondition,
        "precondition_norms": precondition_norms,
        "stabilization": stabilization,
        "lbfgs_tolerance_grad": lbfgs_tolerance_grad,
        "lbfgs_tolerance_change": lbfgs_tolerance_change,
        "init_bond": init_bond,
    }


# ---------------------------------------------------------------------------
# post-step tensor-norm stabilization (OUTSIDE the loss graph)
# ---------------------------------------------------------------------------

def _stabilize_tensor_norm(adtso: "ADTwoSiteOptimizer") -> None:
    """Rescale Θ to unit Frobenius norm under no_grad (scale-invariant for the
    Rayleigh quotient). Post-step stability aid, NOT the optimizer."""
    with tc.no_grad():
        t = adtso.theta
        n = t.norm()
        if n > 0:
            adtso.theta.data = (t / n).to(t.dtype).contiguous().data
