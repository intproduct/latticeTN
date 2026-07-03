"""Stage 7C: spinful Hubbard local operator algebra (Jordan-Wigner).

Checks the canonical fermion algebra on the local 4x4 operators returned by
``hubbard_local_operators`` in the standard Hubbard basis
``|0>, |up>, |down>, |up,down>`` (indices 0, 1, 2, 3):

- same-spin CAR:  {c_s, c^d_s} = I,  {c_s, c_s} = 0,  {c^d_s, c^d_s} = 0  (s=up,down)
- cross-spin CAR: {c_up, c_down} = 0, {c_up, c^d_down} = 0, {c^d_up, c_down} = 0
- number ops:     n_up = diag(0,1,0,1),  n_down = diag(0,0,1,1),
                  n_tot = diag(0,1,1,2), sz = diag(0,+1/2,-1/2,0),
                  double_occ = diag(0,0,0,1)
- parity:         P = (-1)^{n_tot} = diag(1,-1,-1,1),  P^2 = I,
                  P c_s = -c_s P,  P c^d_s = -c^d_s P  (same-site JW anticomm.)
- action:         c_up|up>=|0>, c_up|updown>=|down>, c_down|down>=|0>,
                  c_down|updown>=-|up>  (the last minus sign is the JW sign,
                  since the down mode is JW-ordered after the up mode on site).
"""

from __future__ import annotations

import torch as tc

from latticetn.fermion_operators import hubbard_local_operators

DTYPE = tc.complex128


def _ops():
    return hubbard_local_operators(dtype=DTYPE, device="cpu")


def _basis():
    e = {name: tc.zeros(4, dtype=DTYPE) for name in
         ["vac", "up", "down", "ud"]}
    e["vac"][0] = 1.0
    e["up"][1] = 1.0
    e["down"][2] = 1.0
    e["ud"][3] = 1.0
    return e


def test_same_spin_car_is_identity():
    ops = _ops()
    I = ops["I"]
    for s in ("up", "down"):
        c = ops[f"c{s}"]
        cdag = ops[f"cdag{s}"]
        assert tc.allclose(c @ cdag + cdag @ c, I, atol=1e-12), s
        assert tc.allclose(c @ c, tc.zeros_like(c), atol=1e-12), s
        assert tc.allclose(cdag @ cdag, tc.zeros_like(cdag), atol=1e-12), s


def test_cross_spin_anticommutation():
    """{c_up, c_down} = {c_up, c^d_down} = {c^d_up, c_down} = 0."""
    ops = _ops()
    Z = tc.zeros((4, 4), dtype=DTYPE)
    cup, cdagup = ops["cup"], ops["cdagup"]
    cdown, cdagdown = ops["cdown"], ops["cdagdown"]
    assert tc.allclose(cup @ cdown + cdown @ cup, Z, atol=1e-12)
    assert tc.allclose(cup @ cdagdown + cdagdown @ cup, Z, atol=1e-12)
    assert tc.allclose(cdagup @ cdown + cdown @ cdagup, Z, atol=1e-12)
    assert tc.allclose(cdagup @ cdagdown + cdagdown @ cdagup, Z, atol=1e-12)


def test_number_operators_diagonal():
    ops = _ops()
    assert tc.allclose(ops["nup"], tc.diag(tc.tensor([0, 1, 0, 1], dtype=DTYPE)),
                       atol=1e-12)
    assert tc.allclose(ops["ndown"], tc.diag(tc.tensor([0, 0, 1, 1], dtype=DTYPE)),
                       atol=1e-12)
    assert tc.allclose(ops["ntot"], tc.diag(tc.tensor([0, 1, 1, 2], dtype=DTYPE)),
                       atol=1e-12)
    assert tc.allclose(ops["sz"], tc.diag(tc.tensor([0, 0.5, -0.5, 0], dtype=DTYPE)),
                       atol=1e-12)
    assert tc.allclose(ops["double_occ"],
                       tc.diag(tc.tensor([0, 0, 0, 1], dtype=DTYPE)), atol=1e-12)


def test_parity_is_minus_one_to_ntot():
    ops = _ops()
    I = ops["I"]
    P = ops["parity"]
    assert tc.allclose(P, tc.diag(tc.tensor([1, -1, -1, 1], dtype=DTYPE)),
                       atol=1e-12)
    assert tc.allclose(P @ P, I, atol=1e-12)
    # P = (-1)^n_tot:  P and n_tot share eigenvectors; eigenvalue of P is (-1)^n.
    for s in ("cup", "cdown", "cdagup", "cdagdown"):
        c = ops[s]
        assert tc.allclose(P @ c, -(c @ P), atol=1e-12), s


def test_c_and_cdag_actions_on_basis():
    """Verify the standard-basis actions (including the JW sign on down)."""
    ops = _ops()
    e = _basis()
    cup, cdagup = ops["cup"], ops["cdagup"]
    cdown, cdagdown = ops["cdown"], ops["cdagdown"]
    # c_up |up> = |0>,  c_up |updown> = |down>
    assert tc.allclose(cup @ e["up"], e["vac"], atol=1e-12)
    assert tc.allclose(cup @ e["ud"], e["down"], atol=1e-12)
    # c_down |down> = |0>,  c_down |updown> = -|up>  (JW sign)
    assert tc.allclose(cdown @ e["down"], e["vac"], atol=1e-12)
    assert tc.allclose(cdown @ e["ud"], -e["up"], atol=1e-12)
    # c^d_up |0> = |up>,  c^d_down |0> = |down>
    assert tc.allclose(cdagup @ e["vac"], e["up"], atol=1e-12)
    assert tc.allclose(cdagdown @ e["vac"], e["down"], atol=1e-12)
    # c^d_up |down> = +|updown>  (up is JW-first; creating up on top of down
    # carries no extra sign).
    assert tc.allclose(cdagup @ e["down"], e["ud"], atol=1e-12)
    # c^d_down |up> = -|updown>  (down is JW-second; the local c^d_down carries
    # F_up, and F_up|up> = -|up>, so creating down on top of up picks up a
    # minus sign — the on-site JW sign).
    assert tc.allclose(cdagdown @ e["up"], -e["ud"], atol=1e-12)


def test_global_2n_mode_jw_anticommutation():
    """Global c_{i,s} / c^d_{i,s} built with the full 2N-mode JW parity string
    must anticommute on different (site, spin) modes.

    Site-major global ordering (0_up,0_down,1_up,1_down,...). This is the
    defining property that makes the Hubbard Hamiltonian fermionic (not a
    spin / hard-core-boson model)."""
    from latticetn.operators import _jw_global_mode
    from latticetn.fermion_operators import fermion_operators
    fo = fermion_operators(dtype=DTYPE)
    I2, c2, cdag2, F2 = fo["I"], fo["c"], fo["cdag"], fo["F"]

    for N in [2, 3]:
        nm = 2 * N
        # build global c / c^d for every global mode
        cg = [_jw_global_mode(c2, g, nm, DTYPE, "cpu") for g in range(nm)]
        cdagg = [_jw_global_mode(cdag2, g, nm, DTYPE, "cpu") for g in range(nm)]
        Z = tc.zeros((2 ** nm, 2 ** nm), dtype=DTYPE)
        II = tc.eye(2 ** nm, dtype=DTYPE)
        # {c_g, c_h} = 0 and {c_g, c^d_h} = 0 for g != h
        for g in range(nm):
            for h in range(nm):
                if g == h:
                    continue
                assert tc.allclose(cg[g] @ cg[h] + cg[h] @ cg[g], Z, atol=1e-10), (N, g, h)
                assert tc.allclose(cg[g] @ cdagg[h] + cdagg[h] @ cg[g], Z, atol=1e-10), (N, g, h)
        # {c_g, c^d_g} = I (same mode)
        for g in range(nm):
            assert tc.allclose(cg[g] @ cdagg[g] + cdagg[g] @ cg[g], II, atol=1e-10), (N, g)


def test_site_level_standard_basis_factorization_matches_full_jw():
    """The site-level standard-basis build (parity P on left sites, local 4x4
    at the site) must equal the full 2N-mode JW build (permuted to standard
    basis) for every (site, spin). This is the consistency that lets the
    Hubbard MPO (site-level) match the dense reference (full JW)."""
    from latticetn.operators import _jw_global_mode, _kron
    from latticetn.fermion_operators import fermion_operators
    fo = fermion_operators(dtype=DTYPE)
    I2, c2, cdag2, F2 = fo["I"], fo["c"], fo["cdag"], fo["F"]
    hop = _ops()
    I4, P = hop["I"], hop["parity"]
    # per-site basis permutation natural -> standard (swap axes 1<->2)
    Pp = tc.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                   dtype=DTYPE)

    def full_std(local2, g, N):
        nm = 2 * N
        Mnat = _jw_global_mode(local2, g, nm, DTYPE, "cpu")
        Q = None
        for _ in range(N):
            Q = Pp if Q is None else _kron(Q, Pp)
        return Q @ Mnat @ Q

    def site_level(local4, site, N):
        term = None
        for k in range(N):
            op = P if k < site else (local4 if k == site else I4)
            term = op if term is None else _kron(term, op)
        return term

    for N in [2, 3]:
        for site in range(N):
            for spin, (l2, l4) in enumerate(
                    [(c2, hop["cup"]), (cdag2, hop["cdagup"]),
                     (c2, hop["cdown"]), (cdag2, hop["cdagdown"])]):
                # down-mode global index g = 2*site + 1; up-mode g = 2*site
                g = 2 * site + (0 if spin in (0, 1) else 1)
                # spin 0,1 above are (c2,cup) and (cdag2,cdagup) -> up (g=2*site)
                if spin in (2, 3):
                    g = 2 * site + 1
                assert tc.allclose(full_std(l2, g, N), site_level(l4, site, N),
                                   atol=1e-11), (N, site, spin)
