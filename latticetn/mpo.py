"""Matrix-product operator (MPO).

Index order of each tensor W_i: (left_bond, right_bond, phys_in, phys_out).
Open boundary: left bond of site 0 and right bond of site N-1 are size 1.
"""

from __future__ import annotations

import torch as tc

from .operators import spin_operators


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
