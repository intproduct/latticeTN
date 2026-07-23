"""Minimal two-site DMRG for the open-boundary 1D spin-1/2 Heisenberg chain.

Small-system, CORRECTNESS-FOCUSED DMRG. NON-differentiable (classical
optimizer); reuses Stage 3A canonicalization and Stage 3B native contractions.
Not TEBD/TDVP; not a performance benchmark; NOT part of the autograd energy
path (energy_with_MPO / rayleigh_energy_native are unchanged).

Conventions (unchanged): H = J * sum_i S_i.S_{i+1}, S = sigma/2, J = 1.0, open
boundary. MPS tensor A_i: (l, s, r). MPO tensor W_i: (l, r, s_in, s_out).

Two-site DMRG at bond (i, i+1):
- Mixed-canonical form with sites < i left-canonical, sites > i+1
  right-canonical, leaving the two-site block Theta(l, s_i, s_{i+1}, r) free.
- Left/right MPO environments + W_i, W_{i+1} build the Hermitian H_eff acting
  on the flattened Theta.
- Local ground state via torch.linalg.eigh (small dense matrix).
- SVD split of Theta with chi truncation; absorb S according to sweep
  direction so the newly left site is left-canonical (right sweep) or the newly
  right site is right-canonical (left sweep).

All work is under torch.no_grad on detached tensors. Einsum leg mappings are
verified numerically against the dense effective-space ground state in
tests/test_dmrg_effective_hamiltonian.py.
"""

from __future__ import annotations

from typing import Literal

import torch as tc

from .mps import MPS
from .mpo import MPO
from . import contractions as K
from . import lanczos as LZ  # Stage 4B: optional iterative local solver
from .numerics import truncation_error


# ---------------------------------------------------------------------------
# Mixed-canonical two-site form
# ---------------------------------------------------------------------------

def mixed_canonical_two_site(mps: "MPS", i: int) -> list[tc.Tensor]:
    """Return detached tensors in two-site mixed-canonical form at bond (i,i+1).

    Sites [0, i) are left-canonical (QR sweep), sites (i+1, N) are
    right-canonical (LQ sweep); the block (i, i+1) is left free. The dense
    state is preserved up to a global phase.
    """
    t = [x.detach().clone() for x in mps.tensors]
    N = len(t)
    if not (0 <= i < N - 1):
        raise ValueError(f"bond index i={i} out of range for N={N}")
    with tc.no_grad():
        # left sweep over [0, i)
        for k in range(0, i):
            l, d, r = t[k].shape
            Q, R = tc.linalg.qr(t[k].reshape(l * d, r))
            kk = Q.shape[1]
            t[k] = Q.reshape(l, d, kk)
            t[k + 1] = tc.einsum("kr,rdc->kdc", R, t[k + 1])
        # right sweep over (i+1, N)
        for k in range(N - 1, i + 1, -1):
            l, d, r = t[k].shape
            Qt, Rm = tc.linalg.qr(t[k].reshape(l, d * r).t())
            kk = Rm.shape[0]
            B = Qt.t().reshape(kk, d, r)
            t[k] = B
            C = Rm.t()
            t[k - 1] = tc.einsum("abc,ce->abe", t[k - 1], C)
    return t


# ---------------------------------------------------------------------------
# MPO environments
# ---------------------------------------------------------------------------

def _left_mpo_env(t, Ws, up_to: int) -> tc.Tensor:
    """Left MPO environment at bond `up_to`, shape (lb_bra, lb_mpo, lb_ket)."""
    v = tc.ones((1, 1, 1), dtype=t[0].dtype, device=t[0].device)
    for k in range(up_to):
        # v:(lb_bra, lb_mpo, lb_ket); A*:(lb_bra,s_out,rb_bra);
        # W:(lb_mpo, rb_mpo, s_in, s_out); A:(lb_ket, s_in, rb_ket)
        v = tc.einsum("lmr,lsb,mtys,ryz->btz", v, t[k].conj(), Ws[k], t[k])
    return v


def _right_mpo_env(t, Ws, from_: int) -> tc.Tensor:
    """Right MPO environment at bond `from_`, shape (rb_bra, rb_mpo, rb_ket).

    Verified leg mapping: A*.rb -> v.rb_bra; W.rb_mpo -> v.rb_mpo; A.rb -> v.rb_ket;
    output (lb_bra, lb_mpo, lb_ket).
    """
    v = tc.ones((1, 1, 1), dtype=t[-1].dtype, device=t[-1].device)
    for k in range(len(t) - 1, from_ - 1, -1):
        v = tc.einsum("abc,dga,ebfg,hfc->deh", v, t[k].conj(), Ws[k], t[k])
    return v


def mpo_left_env(mps: "MPS", mpo: "MPO", up_to: int) -> tc.Tensor:
    return _left_mpo_env(mps.tensors, mpo.tensors, up_to)


def mpo_right_env(mps: "MPS", mpo: "MPO", from_: int) -> tc.Tensor:
    return _right_mpo_env(mps.tensors, mpo.tensors, from_)


# ---------------------------------------------------------------------------
# Effective Hamiltonian
# ---------------------------------------------------------------------------

def _theta_two_site(t, i: int) -> tc.Tensor:
    """Block Theta(l, s_i, s_{i+1}, r) = t[i] (center) t[i+1]."""
    return tc.einsum("lsc,cer->lser", t[i], t[i + 1])


def apply_heff(L, Wi, Wi1, R, Theta4: tc.Tensor) -> tc.Tensor:
    """H_eff applied to the two-site block Theta4(l, si, si1, r).

    Returns a tensor of the same shape. Verified einsum:
        L:(lA, lM, lK); Wi:(lM, m, si, sout); Wi1:(m, rM, si1, si1out);
        R:(rB, rM, rK); Theta:(lK, si, si1, rK) -> out(lA, sout, si1out, rB)
    """
    return tc.einsum("pqr,qsab,stcd,utw,racw->pbdu", L, Wi, Wi1, R, Theta4)


def effective_hamiltonian(mps: "MPS", mpo: "MPO", i: int) -> tc.Tensor:
    """Dense Hermitian H_eff matrix for bond (i, i+1).

    Built by applying H_eff to each basis vector of the flattened two-site
    block (D = l*d*d*r, small). Requires the MPS to be in two-site
    mixed-canonical form at bond i (use mixed_canonical_two_site). Returns a
    (D, D) matrix whose columns correspond to the ket (lK, si, si1, rK) basis
    and rows to the bra (lA, sout, si1out, rB) basis.

    This DENSE path is the Stage 4A reference; Stage 4B adds a matrix-free
    apply (matrix_free_apply) and a Lanczos solver that avoid materializing it.
    """
    t = [x.detach() for x in mps.tensors]
    Ws = mpo.tensors
    L = _left_mpo_env(t, Ws, up_to=i)
    R = _right_mpo_env(t, Ws, from_=i + 2)
    Wi = Ws[i]
    Wi1 = Ws[i + 1]
    Theta = _theta_two_site(t, i)
    l, si, si1, r = Theta.shape
    D = l * si * si1 * r

    def apply_vec(v: tc.Tensor) -> tc.Tensor:
        T4 = v.reshape(l, si, si1, r)
        o4 = apply_heff(L, Wi, Wi1, R, T4)
        return o4.reshape(-1)

    cols = []
    ident = tc.eye(D, dtype=Theta.dtype, device=Theta.device)
    for k in range(D):
        cols.append(apply_vec(ident[:, k]))
    return tc.stack(cols, dim=1)


def matrix_free_apply(mps: "MPS", mpo: "MPO", i: int):
    """Return a callable ``f(vec) -> vec`` for the matrix-free H_eff apply.

    The returned function acts on a flattened length-D vector (D = l*d*d*r) and
    returns H_eff @ vec as a flattened vector. Verified to match the dense
    ``effective_hamiltonian`` in tests/test_dmrg_matrix_free_heff.py.
    """
    t = [x.detach() for x in mps.tensors]
    Ws = mpo.tensors
    L = _left_mpo_env(t, Ws, up_to=i)
    R = _right_mpo_env(t, Ws, from_=i + 2)
    Wi = Ws[i]
    Wi1 = Ws[i + 1]
    Theta = _theta_two_site(t, i)
    l, si, si1, r = Theta.shape

    def apply_vec(v: tc.Tensor) -> tc.Tensor:
        T4 = v.reshape(l, si, si1, r)
        return apply_heff(L, Wi, Wi1, R, T4).reshape(-1)

    apply_vec.dim = l * si * si1 * r
    apply_vec.dtype = Theta.dtype
    apply_vec.device = Theta.device
    return apply_vec


def local_ground_state(Heff: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
    """Lowest eigenpair of Hermitian Heff. Returns (E0, ground_vector)."""
    E, V = tc.linalg.eigh(Heff)
    return E[0], V[:, 0]


# ---------------------------------------------------------------------------
# Two-site update (SVD split + truncation)
# ---------------------------------------------------------------------------

def two_site_update(Theta4: tc.Tensor, chi: int,
                    direction: Literal["right", "left"]
                    ) -> tuple[list[tc.Tensor], float, int]:
    """SVD-split a two-site block into two site tensors with chi truncation.

    Theta4: (l, s_i, s_{i+1}, r).
    - direction='right': A_i = U (left-canonical), A_{i+1} = S Vh (carries norm).
    - direction='left':  A_i = U S (carries norm), A_{i+1} = Vh (right-canonical).

    Returns ([A_i, A_{i+1}], truncation_error, kept_bond).
    truncation_error = discarded weight = sum(discarded s^2) / sum(all s^2) in [0, 1].
    """
    l, si, si1, r = Theta4.shape
    M = Theta4.reshape(l * si, si1 * r)
    U, S, Vh = tc.linalg.svd(M, full_matrices=False)
    k0 = S.shape[0]
    k = min(chi, k0)
    s2 = S.real ** 2
    trunc = truncation_error(s2, k, name="DMRG two-site split")

    U = U[:, :k]
    S = S[:k]
    Vh = Vh[:k, :]
    with tc.no_grad():
        if direction == "right":
            A_i = U.reshape(l, si, k)                 # left-canonical
            A_ip1 = (S.reshape(k, 1) * Vh).reshape(k, si1, r)
        elif direction == "left":
            A_i = (U * S.reshape(1, k)).reshape(l, si, k)
            A_ip1 = Vh.reshape(k, si1, r)              # right-canonical
        else:
            raise ValueError(f"direction must be 'right' or 'left', got {direction!r}")
    return [A_i, A_ip1], trunc, k


# ---------------------------------------------------------------------------
# Minimal two-site DMRG driver
# ---------------------------------------------------------------------------

def _bond_dims(tensors) -> list[int]:
    return [int(tensors[i].shape[2]) for i in range(len(tensors) - 1)]


def two_site_sweep(tensors: list[tc.Tensor], mpo: "MPO", chi: int,
                   direction: Literal["right", "left"],
                   solver: Literal["dense", "lanczos"] = "dense",
                   lanczos_kwargs: dict | None = None
                   ) -> tuple[list[tc.Tensor], float, list[float]]:
    """One two-site sweep in the given direction.

    Right sweep visits bonds 0,1,...,N-2 (center moves right); left sweep
    visits bonds N-2,...,0. At each bond the MPS is brought into two-site
    mixed-canonical form (cheap re-canonicalization of the current tensors),
    H_eff is built and solved, the block is split with truncation, and the new
    tensors are written back.

    solver:
      - 'dense'    (Stage 4A reference): materialize D x D H_eff and use
                    torch.linalg.eigh.
      - 'lanczos'  (Stage 4B, matrix-free): iterate with lanczos_lowest_eigenpair
                    on the matrix-free apply. Scales better with D.

    Returns (new_tensors, final_local_energy, per_bond_truncation_errors).
    The returned `final_local_energy` is the local H_eff ground energy at the
    last bond (a variational estimate; use rayleigh_energy_native on the
    resulting MPS for the global energy).
    """
    N = len(tensors)
    if N < 2:
        raise ValueError("two-site DMRG requires N >= 2")
    Ws = mpo.tensors
    tensors = [x.detach().clone() for x in tensors]
    bonds = list(range(N - 1)) if direction == "right" else list(range(N - 2, -1, -1))
    trunc_errors: list[float] = []
    last_E: float | None = None
    lanczos_kwargs = dict(lanczos_kwargs or {})
    with tc.no_grad():
        for bi, i in enumerate(bonds):
            # re-canonicalize into two-site form at bond i
            tmp = MPS.from_tensors(tensors, dtype=tensors[0].dtype,
                                   device=tensors[0].device, requires_grad=False)
            t = mixed_canonical_two_site(tmp, i)
            L = _left_mpo_env(t, Ws, up_to=i)
            R = _right_mpo_env(t, Ws, from_=i + 2)
            Wi = Ws[i]
            Wi1 = Ws[i + 1]
            Theta = _theta_two_site(t, i)
            l, si, si1, r = Theta.shape
            D = l * si * si1 * r

            def apply_vec(v, l=l, si=si, si1=si1, r=r, L=L, Wi=Wi, Wi1=Wi1, R=R):
                T4 = v.reshape(l, si, si1, r)
                return apply_heff(L, Wi, Wi1, R, T4).reshape(-1)

            if solver == "dense":
                ident = tc.eye(D, dtype=Theta.dtype, device=Theta.device)
                cols = [apply_vec(ident[:, k]) for k in range(D)]
                Heff = tc.stack(cols, dim=1)
                E, V = tc.linalg.eigh(Heff)
                last_E = float(E[0].real)
                Theta_new = V[:, 0].reshape(l, si, si1, r)
            elif solver == "lanczos":
                E0, V0 = LZ.lanczos_lowest_eigenpair(
                    apply_vec, D, dtype=Theta.dtype, device=Theta.device,
                    **lanczos_kwargs,
                )
                last_E = float(E0.real)
                Theta_new = V0.reshape(l, si, si1, r)
            else:
                raise ValueError(f"solver must be 'dense' or 'lanczos', got {solver!r}")

            new_t, trunc, k = two_site_update(Theta_new, chi, direction)
            trunc_errors.append(trunc)
            # update Theta_new for the write-back above already happened
            t[i] = new_t[0]
            t[i + 1] = new_t[1]
            tensors = t
    if last_E is None:
        raise RuntimeError("two-site sweep completed without visiting a bond")
    return tensors, last_E, trunc_errors


def run_dmrg(mps: "MPS", mpo: "MPO", chi: int, num_sweeps: int = 4,
             seed: int = 0, solver: Literal["dense", "lanczos"] = "dense",
             lanczos_kwargs: dict | None = None) -> dict:
    """Minimal two-site DMRG driver. Returns a result dict.

    Alternates right/left sweeps. Global energy after each sweep is computed
    via the NATIVE MPO Rayleigh quotient (rayleigh_energy_native) on the
    current MPS. Records per-sweep energy history, truncation errors, bond
    dims, and below-ground flag. Non-differentiable.

    solver selects the local two-site eigensolver ('dense' = Stage 4A
    reference; 'lanczos' = Stage 4B matrix-free iterative solver).
    """
    if num_sweeps < 1:
        raise ValueError("num_sweeps must be >= 1")
    N = mps.N
    tensors = [x.detach().clone() for x in mps.tensors]
    history: list[dict] = []
    for s in range(num_sweeps):
        direction = "right" if (s % 2 == 0) else "left"
        tensors, last_local_E, trunc_errs = two_site_sweep(
            tensors, mpo, chi, direction, solver=solver,
            lanczos_kwargs=lanczos_kwargs,
        )
        mps_cur = MPS.from_tensors(tensors, dtype=mps.dtype, device=mps.device,
                                   requires_grad=False)
        E_global = float(K.rayleigh_energy_native(mps_cur, mpo))
        history.append({
            "sweep": s,
            "direction": direction,
            "energy": E_global,
            "local_last_energy": last_local_E,
            "truncation_errors": trunc_errs,
            "max_trunc": max(trunc_errs) if trunc_errs else 0.0,
            "bond_dims": _bond_dims(tensors),
            "max_bond": max(_bond_dims(tensors)) if len(tensors) > 1 else 1,
            "solver": solver,
        })

    # exact ground energy for context
    from .operators import heisenberg_dense, exact_ground_energy
    if N <= 12:
        H = heisenberg_dense(N, dtype=mps.dtype, device=mps.device)
        E_exact, _ = exact_ground_energy(H)
    else:
        E_exact = None

    final_E = history[-1]["energy"]
    below_ground = bool(E_exact is not None and final_E < E_exact - 1e-6)
    return {
        "N": N,
        "chi": chi,
        "num_sweeps": num_sweeps,
        "seed": seed,
        "solver": solver,
        "history": history,
        "final_energy": final_E,
        "energy_per_bond": final_E / max(1, N - 1),
        "exact_energy": E_exact,
        "below_ground": below_ground,
        "final_bond_dims": _bond_dims(tensors),
        "final_max_bond": max(_bond_dims(tensors)) if len(tensors) > 1 else 1,
    }
