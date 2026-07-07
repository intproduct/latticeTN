"""Matrix-product operator (MPO).

Index order of each tensor W_i: (left_bond, right_bond, phys_in, phys_out).
Open boundary: left bond of site 0 and right bond of site N-1 are size 1.
"""

from __future__ import annotations

import torch as tc

from .operators import spin_operators
from .fermion_operators import fermion_operators, hubbard_local_operators


class MPO:
    """Finite open-boundary MPO with explicit tensors.

    Attributes
    ----------
    tensors : list[tc.Tensor]
        Per-site MPO tensors. For a bulk site: shape (D, D, d, d). For the left
        boundary site: (1, D, d, d). For the right boundary site: (D, 1, d, d).
    """

    def __init__(self, tensors, length=None, dim=None, dtype=tc.complex128,
                 device="cpu"):
        self.tensors = list(tensors)
        self.length = length if length is not None else len(self.tensors)
        self.dim = dim if dim is not None else self.tensors[0].shape[-1]
        self.dtype = dtype
        self.device = device

    @classmethod
    def from_bonds(cls, N, dim, dtype=tc.complex128, device="cpu"):
        """Empty MPO template (tensors filled by a generator)."""
        obj = cls.__new__(cls)
        obj.length = N
        obj.dim = dim
        obj.dtype = dtype
        obj.device = device
        obj.tensors = [None] * N
        return obj

    # ---- generators -------------------------------------------------------
    def generate_heisenberg(self, J: float = 1.0):
        """Heisenberg MPO H = J * sum_i S.S (spin convention, open boundary).

        Uses the standard nearest-neighbor construction with virtual bonds:
            states: 0=idle, 1=carry Sz, 2=carry S-, 3=carry S+, 4=done.
            transitions (no self-loop on carry -> strictly nearest-neighbor):
                0->0 I, 0->k start_op_k (k=1,2,3), k->4 pair_op_k, 4->4 I.
            With start/pair ops:
                Sz:   start Sz  , pair Sz
                Sx/Sy part: start (1/2)S-, pair S+  AND  start (1/2)S+, pair S-.
            J multiplies the start operators (one factor J per bond).
        """
        ops = spin_operators(dtype=self.dtype, device=self.device)
        I = ops["I"]
        Sz = ops["Sz"]
        Sp = ops["S+"]
        Sm = ops["S-"]
        d = self.dim
        D = 5

        def bulk():
            W = tc.zeros((D, D, d, d), dtype=self.dtype, device=self.device)
            # row 0 (idle)
            W[0, 0, :, :] = I
            W[0, 1, :, :] = J * Sz
            W[0, 2, :, :] = J * 0.5 * Sm
            W[0, 3, :, :] = J * 0.5 * Sp
            # carry -> done (next site), pairing operator
            W[1, 4, :, :] = Sz
            W[2, 4, :, :] = Sp
            W[3, 4, :, :] = Sm
            # done -> done
            W[4, 4, :, :] = I
            return W

        for i in range(self.length):
            W = bulk()
            if i == 0:
                W = W[0:1, :, :, :]            # keep only idle row on the left
            if i == self.length - 1:
                W = W[:, 4:5, :, :]            # keep only done col on the right
            self.tensors[i] = W
        return self

    def generate_tfi(self, J: float = 1.0, h: float = 1.0):
        """TFI MPO H = -J Sz Sz - h sum_i Sx (spin convention, open boundary).

        Standard D=3 construction. Each tensor W (left, right, s_in, s_out):
            row 0 (idle):  [ I,   Sz,   -h Sx ]
            row 1 (carry Sz): [ 0,   0,    -J Sz ]
            row 2 (done):  [ 0,   0,     I   ]
        """
        ops = spin_operators(dtype=self.dtype, device=self.device)
        I = ops["I"]
        Sz = ops["Sz"]
        Sx = ops["Sx"]
        d = self.dim
        D = 3

        def bulk():
            W = tc.zeros((D, D, d, d), dtype=self.dtype, device=self.device)
            W[0, 0, :, :] = I
            W[0, 1, :, :] = Sz
            W[0, 2, :, :] = -h * Sx
            W[1, 2, :, :] = -J * Sz
            W[2, 2, :, :] = I
            return W

        for i in range(self.length):
            W = bulk()
            if i == 0:
                W = W[0:1, :, :, :]
            if i == self.length - 1:
                W = W[:, 2:3, :, :]
            self.tensors[i] = W
        return self

    def generate_spinless_fermion(self, t: float = 1.0, V: float = 0.0,
                                  mu: float = 0.0):
        """Spinless-fermion t-V MPO (Jordan-Wigner, open boundary).

            H = -t sum_i (c^d_i c_{i+1} + h.c.)
                + V sum_i (n_i - 1/2)(n_{i+1} - 1/2)
                - mu sum_i (n_i - 1/2)

        Bond dimension D=5. Global single-fermion operators carry
        Jordan-Wigner strings, but for adjacent Hamiltonian products those
        strings cancel on all sites left of the bond.

        Stage 11 correction: the implementation below is the authoritative
        transition table. Any older mention of carrying a left parity string
        for nearest-neighbor hopping is obsolete; such a string would produce
        left-occupation-dependent signs that fail the independent JW audit.

        The Hamiltonian MPO emits the reduced adjacent product directly:
        ``c^d`` (or ``c``) at the left site, then ``c`` (or ``c^d``) at the
        right site. It has no virtual state carrying a left JW parity string.
        Nonlocal single-particle Green functions require explicit JW strings;
        this nearest-neighbor Hamiltonian does not.

        Virtual bond states:

            0: idle
            1: carry c^d
            2: carry c
            3: carry n-1/2
            4: done

        Bulk transitions (row = left virtual, col = right virtual; the entry
        is the local operator acting on the physical index at this site):

            0->0  I                 (idle: identity propagation)
            0->4  -mu*(n-1/2)       (on-site chemical potential, emitted here)
            0->3  (n-1/2)           (start a density-density interaction)
            0->1  -t * c^d          (start c^d_i c_{i+1} here)
            0->2  -t * c            (start c_i c^d_{i+1} here)
            1->4  c                 (complete c^d_i c_{i+1} on the right site)
            2->4  c^d               (complete c_i c^d_{i+1} on the right site)
            3->4  V * (n-1/2)       (complete the interaction on the right site)
            4->4  I                 (done stays done)

        ``to_dense`` of this MPO matches ``operators.spinless_fermion_dense``
        (see tests/test_spinless_fermion_mpo_dense.py).
        """
        ops = fermion_operators(dtype=self.dtype, device=self.device)
        I = ops["I"]
        c = ops["c"]
        cdag = ops["cdag"]
        nmh = ops["n_minus_half"]
        d = self.dim
        D = 5

        def bulk():
            W = tc.zeros((D, D, d, d), dtype=self.dtype, device=self.device)
            # idle -> idle: pure identity (idle propagation; on-site term is
            # emitted via 0->4 below so it is NOT multiplied across sites).
            W[0, 0, :, :] = I
            # idle -> done: on-site chemical potential -mu*(n-1/2) at this site.
            W[0, 4, :, :] = (-mu) * nmh
            # idle -> carry-left-factor: adjacent hopping. The left JW strings
            # cancel for cdag_i c_{i+1} and its h.c., so no parity-carry state
            # is present in the Hamiltonian MPO.
            W[0, 1, :, :] = (-t) * cdag
            W[0, 2, :, :] = (-t) * c
            # idle -> carry-(n-1/2): start a density-density interaction
            W[0, 3, :, :] = nmh
            # carry-c^d -> done: right factor c of c^d_i c_{i+1}
            W[1, 4, :, :] = c
            # carry-c   -> done: right factor c^d of c_i c^d_{i+1}
            W[2, 4, :, :] = cdag
            # carry-(n-1/2) -> done: right factor (n-1/2) of the interaction
            W[3, 4, :, :] = V * nmh
            # done -> done
            W[4, 4, :, :] = I
            return W

        for i in range(self.length):
            W = bulk()
            if i == 0:
                W = W[0:1, :, :, :]            # keep only idle row on the left
            if i == self.length - 1:
                W = W[:, 4:5, :, :]            # keep only done col on the right
            self.tensors[i] = W
        return self

    def generate_hubbard(self, t: float = 1.0, U: float = 4.0,
                         mu: float = 0.0, h: float = 0.0):
        """Spinful-Hubbard MPO (Jordan-Wigner, open boundary), local dim d=4.

            H = -t  sum_{i,sigma} (c^d_{i sigma} c_{i+1,sigma} + h.c.)
                + U  sum_i (n_{i up} - 1/2)(n_{i down} - 1/2)
                - mu sum_i (n_{i up} + n_{i down} - 1)
                - h  sum_i (n_{i up} - n_{i down})

        Local basis ``|0>, |up>, |down>, |up,down>`` (d=4); global mode
        ordering site-major ``(0_up,0_down,1_up,1_down,...)``. This is 1D
        Jordan-Wigner fermions, NOT a spin model and NOT a hard-core-boson
        model: every spin-resolved hop ``c^d_{i sigma} c_{i+1, sigma}`` carries
        the per-site JW parity ``P = F_up x F_down = (-1)^{n_up+n_down}`` on
        the LEFT-FACTOR site ``i`` (the intra-site ``F_up`` for the down mode
        is already inside the local 4x4 ``cdown``/``cdagdown`` operators from
        ``hubbard_local_operators``). ``to_dense`` of this MPO matches
        ``operators.hubbard_dense`` (see tests/test_hubbard_mpo_dense.py).

        Bond dimension D=6 (NO separate parity-carrying virtual state, unlike
        the spinless-fermion MPO). The reason: in the spinful Hubbard chain
        the global operators ``c^d_{i sigma}`` and ``c_{i+1, sigma}`` BOTH
        carry the per-site parity ``P`` on every site left of ``i+1`` (because
        the site-major global mode index of any spin on site ``i+1`` is larger
        than all modes on sites 0..i). In the *product* ``c^d_{i sigma}
        c_{i+1, sigma}`` the two parity strings on sites 0..i-1 therefore
        SQUARE to identity and cancel; no inter-site parity string survives
        in the product. The only surviving parity is the single ``P`` on site
        ``i`` contributed by the RIGHT factor's string (which extends one site
        further right than the left factor's). The MPO captures this by
        emitting the left factor as ``(c^d_sigma @ P)`` / ``(P @ c_sigma)`` at
        site ``i`` (the ``@ P`` / ``P @`` is exactly that surviving
        site-``i`` parity) and the right factor as ``c_sigma`` / ``c^d_sigma``
        at site ``i+1`` (no parity). This is the fermionic, NOT
        hard-core-boson, structure: the ``@ P`` on the left factor and the
        ``F_up`` already inside ``cdown``/``cdagdown`` are what make the
        spin-resolved hops anticommute correctly across the chain; without
        them the MPO would be a spin / hard-core-boson MPO and would NOT match
        ``hubbard_dense`` (verified by tests).

        Virtual bond states (D=6):

            0: idle          - nothing carried yet
            1: carry c^d_up  - left factor of c^d_i c_{i+1} (up) in flight
            2: carry c_up    - left factor of the up h.c. (c_i, for c^d_{i+1} c_i)
            3: carry c^d_down - left factor of c^d_i c_{i+1} (down) in flight
            4: carry c_down  - left factor of the down h.c.
            5: done          - terminal

        Bulk transitions (row = left virtual, col = right virtual; the entry is
        the local 4x4 operator acting on the physical index at this site):

            0->0  I                                          (idle propagation)
            0->5  U*(n_up-1/2)(n_down-1/2) - mu*(n_tot-1)
                  - h*(n_up-n_down)                         (on-site terms)
            0->1  -t * (c^d_up   @ P)    (start up-hop   c^d_i c_{i+1} here)
            0->2  -t * (P   @ c_up)      (start up-hop   h.c.            here)
            0->3  -t * (c^d_down @ P)    (start down-hop c^d_i c_{i+1} here)
            0->4  -t * (P   @ c_down)    (start down-hop h.c.            here)
            1->5  c_up                  (complete up-hop    on the right site)
            2->5  c^d_up                (complete up-hop h.c. on the right site)
            3->5  c_down                (complete down-hop    on the right site)
            4->5  c^d_down              (complete down-hop h.c. on the right site)
            5->5  I                     (done stays done)

        Because no inter-site parity string is needed, the same bulk applies
        at every site (including site 0); the only boundary trimming is the
        usual ``[0:1]`` on the left of site 0 and ``[5:6]`` on the right of
        site ``N-1``.

        The left factors ``c^d @ P`` (for ``c^d_i c_{i+1}``) and ``P @ c`` (for
        the h.c. ``c^d_{i+1} c_i``) DIFFER by a sign (``P`` anticommutes with
        ``c``/``c^d``); this asymmetry is required for Hermiticity and matches
        the dense reference's site-level build exactly.
        """
        ops = hubbard_local_operators(dtype=self.dtype, device=self.device)
        I = ops["I"]
        P = ops["parity"]
        cup = ops["cup"]
        cdagup = ops["cdagup"]
        cdown = ops["cdown"]
        cdagdown = ops["cdagdown"]
        nup = ops["nup"]
        ndown = ops["ndown"]
        nmh_up = nup - 0.5 * I
        nmh_down = ndown - 0.5 * I
        onsite = (U * (nmh_up @ nmh_down)
                  + (-mu) * (nup + ndown - I)
                  + (-h) * (nup - ndown))
        # Left factors at the left-factor site i. The c^d_i c_{i+1} term uses
        # (c^d_sigma @ P); the h.c. c^d_{i+1} c_i term uses (P @ c_sigma).
        # They differ by a sign because P anticommutes with c / c^d. The
        # surviving P at site i comes from the right factor's JW string (it
        # extends one site further right than the left factor's, contributing
        # one P_i that does NOT cancel).
        lf_cdagup = cdagup @ P        # c^d_up   @ P  (for c^d_i c_{i+1}, up)
        lf_cup = P @ cup              # P  @ c_up     (for h.c.,        up)
        lf_cdagdown = cdagdown @ P    # c^d_down @ P  (for c^d_i c_{i+1}, down)
        lf_cdown = P @ cdown          # P  @ c_down   (for h.c.,        down)
        d = self.dim
        D = 6

        def bulk():
            W = tc.zeros((D, D, d, d), dtype=self.dtype, device=self.device)
            # idle -> idle: identity
            W[0, 0, :, :] = I
            # idle -> done: on-site terms (U, mu, h) emitted here
            W[0, 5, :, :] = onsite
            # idle -> carry-left-factor: start a hop left factor at THIS site
            # (with the surviving @P / P@ from the right factor's JW string).
            # No inter-site parity string is needed (it cancels in product).
            W[0, 1, :, :] = (-t) * lf_cdagup
            W[0, 2, :, :] = (-t) * lf_cup
            W[0, 3, :, :] = (-t) * lf_cdagdown
            W[0, 4, :, :] = (-t) * lf_cdown
            # carry-left-factor -> done: complete the hop with the right factor
            W[1, 5, :, :] = cup
            W[2, 5, :, :] = cdagup
            W[3, 5, :, :] = cdown
            W[4, 5, :, :] = cdagdown
            # done -> done
            W[5, 5, :, :] = I
            return W

        for i in range(self.length):
            W = bulk()
            if i == 0:
                W = W[0:1, :, :, :]            # keep only idle row on the left
            if i == self.length - 1:
                W = W[:, 5:6, :, :]            # keep only done col on the right
            self.tensors[i] = W
        return self

    # ---- dense conversion -------------------------------------------------
    def to_dense(self) -> tc.Tensor:
        """Contract all virtual bonds -> dense Hamiltonian (d**N, d**N).

        Result index order: rows = composite phys_in (s_0,...,s_{N-1}),
        cols = composite phys_out (s'_0,...,s'_{N-1}).
        """
        T = self.tensors[0]                        # (l0, r0, s0_in, s0_out)
        n = self.length
        # Track free phys indices explicitly. Label them.
        # We build subscripts dynamically.
        # T currently carries: l0(=1, will vanish as it's size1 but keep), r0,
        # s0_in, s0_out.
        # Use einsum with growing subscript string.
        from string import ascii_lowercase as letters

        L = list(letters)
        # left bond index for site 0 is trivial (size 1); we keep it but it
        # squeezes out. We contract site by site.
        inp_prev = "ab"  # a = right bond (carries), b... we need 4 indices.
        # Simpler: iterative contraction tracking the right-bond idx and phys.
        # Represent running tensor with named axes via einsum letter assignment.
        # axes: [right_bond, s_in_0, s_in_1, ..., s_out_0, s_out_1, ...]
        running = self.tensors[0]                   # (l, r, s0in, s0out)
        # squeeze trivial left bond
        running = running.reshape(running.shape[1], running.shape[2],
                                  running.shape[3])  # (r, s0in, s0out)
        rb = "a"          # right bond letter
        s_in_letters = ["b"]
        s_out_letters = ["c"]
        next_letter_idx = 3
        for i in range(1, n):
            Wi = self.tensors[i]                     # (l, r, s_in, s_out)
            rin = L[next_letter_idx]; next_letter_idx += 1
            sin_i = L[next_letter_idx]; next_letter_idx += 1
            sout_i = L[next_letter_idx]; next_letter_idx += 1
            # running: rb + s_in_letters + s_out_letters
            run_sub = rb + "".join(s_in_letters) + "".join(s_out_letters)
            # Wi: rin(=rb contract), new_rb, sin_i, sout_i
            wi_sub = rb + rin + sin_i + sout_i
            out_sub = rin + "".join(s_in_letters) + sin_i + "".join(s_out_letters) + sout_i
            running = tc.einsum(f"{run_sub},{wi_sub}->{out_sub}", running, Wi)
            rb = rin
            s_in_letters.append(sin_i)
            s_out_letters.append(sout_i)
        # running now has axes: rb(trivial size1) + all s_in + all s_out
        # drop trailing right bond (size 1)
        running = running.reshape(running.shape[1:])
        # current axis order: s_in_0, s_in_1, ..., s_out_0, s_out_1, ...
        n_phys = n
        # move to (s_in_0.., s_out_0..) then reshape to (d**N, d**N)
        # it's already grouped: first n_phys axes are s_in, next n_phys are s_out
        d = self.dim
        shape_in = (d,) * n_phys
        H = running.reshape(d ** n_phys, d ** n_phys)
        return H
