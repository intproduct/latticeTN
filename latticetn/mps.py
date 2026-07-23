"""Matrix-product state (MPS).

Index order of each tensor A_i: (left_bond, phys, right_bond).
Open boundary: left bond of site 0 and right bond of site N-1 are size 1.

All energy/norm paths are fully differentiable: no .detach()/.data/.item()/
torch.no_grad() inside the differentiable energy computation.
"""

from __future__ import annotations

import torch as tc
import torch.nn as nn

from .numerics import positive, real_if_hermitian


class MPS:
    """Finite open-boundary MPS (random init, autograd-friendly).

    Tensors are kept in an ``nn.ParameterList`` so an optimizer can hold a
    stable reference to them across steps. The differentiable energy path
    (to_dense / overlap / energy_with_MPO) never uses .detach()/.data/.item()/
    no_grad; normalization that mutates the tensors in place is performed
    OUTSIDE the energy path (see scripts/run_heisenberg_small.py).
    """

    def __init__(self, N, dim, chi, dtype=tc.complex128, device="cpu",
                 boundary="open", rng=None):
        assert boundary == "open", "only open boundary is supported here"
        self.N = N
        self.dim = dim
        self.chi = chi
        self.dtype = dtype
        self.device = device
        self.boundary = boundary
        if N <= 0 or dim <= 0 or chi <= 0:
            raise ValueError(f"N, dim, and chi must be positive, got {N}, {dim}, {chi}")
        if rng is not None and not isinstance(rng, tc.Generator):
            raise TypeError("rng must be a torch.Generator or None")
        # bond_i = left bond of site i (i=0..N), bond_0=bond_N=1.
        # Cap by both chi and the Hilbert-space geometry so bonds match across
        # sites and never exceed what an open chain actually needs.
        bonds = [min(chi, dim ** min(i, N - i)) for i in range(N + 1)]
        tensors = []
        for i in range(N):
            l = bonds[i]
            r = bonds[i + 1]
            # ``generator=None`` deliberately uses PyTorch's global RNG, so
            # manual_seed controls reproducibility while successive MPS
            # constructions do not silently repeat the same tensors.
            t = tc.randn((l, dim, r), dtype=dtype, device=device, generator=rng)
            tensors.append(nn.Parameter(t))
        self.tensors = nn.ParameterList(tensors)

    def parameters(self):
        return self.tensors

    def clone(self):
        new = object.__new__(MPS)
        new.N = self.N
        new.dim = self.dim
        new.chi = self.chi
        new.dtype = self.dtype
        new.device = self.device
        new.boundary = self.boundary
        new.tensors = [t.clone() for t in self.tensors]
        return new

    @classmethod
    def from_tensors(cls, tensors, dtype=None, device=None,
                     requires_grad: bool = False) -> "MPS":
        """Build an MPS from an explicit list of per-site tensors.

        Used by the (non-differentiable) canonicalization/compression path in
        ``latticetn/canonical.py`` to wrap canonical-form tensors back into an
        MPS object so the existing ``to_dense`` / ``overlap`` /
        ``energy_with_MPO`` paths can be reused. Tensors are detached and
        cloned; by default ``requires_grad=False`` because canonicalization is
        a preprocessing/postprocessing step (per Stage 3A constraints),
        NOT part of the differentiable training energy path.
        """
        t0 = tensors[0]
        N = len(tensors)
        dim = int(t0.shape[1])
        if dtype is None:
            dtype = t0.dtype
        if device is None:
            device = t0.device
        obj = object.__new__(cls)
        obj.N = N
        obj.dim = dim
        obj.chi = max(int(t.shape[0]) for t in tensors)
        obj.dtype = dtype
        obj.device = device
        obj.boundary = "open"
        obj.tensors = nn.ParameterList([
            nn.Parameter(t.detach().clone().to(dtype=dtype, device=device),
                         requires_grad=requires_grad)
            for t in tensors
        ])
        return obj

    # ---- dense conversion -------------------------------------------------
    def to_dense(self) -> tc.Tensor:
        """Contract all bonds -> state vector of shape (dim**N,).

        Order: psi[s_0, s_1, ..., s_{N-1}] flattened with site 0 most significant.
        """
        T = self.tensors[0]                       # (1, d, r)
        for A in self.tensors[1:]:
            # T: (..., l_bra) ; A: (l, d, r)
            T = tc.einsum("...a,abc->...bc", T, A)
        # T shape: (1, d, d, ..., d, 1) -> (d, d, ..., d)
        T = T.reshape((self.dim,) * self.N)
        return T.reshape(self.dim ** self.N)

    # ---- norms / overlap --------------------------------------------------
    def overlap(self, other: "MPS") -> tc.Tensor:
        """<self | other> via left-to-right contraction (open boundary).

        Returns the full complex scalar (not rescaled). Small systems only.
        bra = self (conjugated), ket = other.
        """
        v = tc.ones((1, 1), dtype=self.dtype, device=self.device)  # (l_bra, l_ket)
        for A, B in zip(self.tensors, other.tensors):
            # v: (l_bra, l_ket); A*: (l_bra, s, r_bra); B: (l_ket, s, r_ket)
            # NOTE: the left-bond letters must be SHARED between v and A*/B so
            # the bonds actually contract (distinct letters would be summed
            # independently and silently drop the environment).
            v = tc.einsum("lr,lsm,rsn->mn", v, A.conj(), B)
        return v.reshape(())

    def norm_sq(self) -> tc.Tensor:
        """<self|self> as a real-valued scalar."""
        return positive(self.overlap(self), name="MPS norm squared")

    # ---- energy -----------------------------------------------------------
    def energy_with_MPO(self, mpo) -> tc.Tensor:
        """Rayleigh quotient <psi|H|psi> / <psi|psi> via left-to-right contraction.

        Fully differentiable. Returns a real scalar (real part of the ratio).
        """
        assert self.N == mpo.length
        assert self.dim == mpo.dim
        # numerator <psi|H|psi>
        num = self._expect_MPO(mpo)
        den = positive(self.overlap(self), name="MPS Rayleigh denominator")
        e = num / den
        return real_if_hermitian(e, name="MPS Rayleigh energy")

    def _expect_MPO(self, mpo) -> tc.Tensor:
        """<self| H |self> via a (bra_bond, mpo_bond, ket_bond) left env."""
        v = tc.ones((1, 1, 1), dtype=self.dtype, device=self.device)
        for A, W in zip(self.tensors, mpo.tensors):
            # v: (l_bra, l_mpo, l_ket)
            # A*: (l_bra, s_out, r_bra)
            # W : (l_mpo, r_mpo, s_in, s_out)
            # A : (l_ket, s_in, r_ket)
            # Shared letters contract the bonds:
            #   l = l_bra (v & A*), m = l_mpo (v & W), r = l_ket (v & A),
            #   s = s_out (A* & W), y = s_in (W & A).
            v = tc.einsum("lmr,lsb,mtys,ryz->btz", v, A.conj(), W, A)
        return v.reshape(())
