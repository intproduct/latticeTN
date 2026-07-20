"""Matrix-free effective Hamiltonians for one- and two-site TDVP.

MPS tensors use ``(left, physical, right)`` and MPO tensors use
``(left_mpo, right_mpo, physical_in, physical_out)``.  Every action below is
an MPO/MPS environment contraction; no dense many-body Hamiltonian is built.
"""

from __future__ import annotations

import torch as tc


def identity_environment(reference: tc.Tensor) -> tc.Tensor:
    """Open-boundary MPO environment with shape ``(1, 1, 1)``."""
    return tc.ones((1, 1, 1), dtype=reference.dtype, device=reference.device)


def update_left_environment(
    left: tc.Tensor, tensor: tc.Tensor, mpo_tensor: tc.Tensor
) -> tc.Tensor:
    """Contract one MPS/MPO site into a left environment."""
    return tc.einsum(
        "lmr,lsb,mtys,ryz->btz",
        left,
        tensor.conj(),
        mpo_tensor,
        tensor,
    )


def update_right_environment(
    right: tc.Tensor, tensor: tc.Tensor, mpo_tensor: tc.Tensor
) -> tc.Tensor:
    """Contract one MPS/MPO site into a right environment."""
    return tc.einsum(
        "abc,dga,ebfg,hfc->deh",
        right,
        tensor.conj(),
        mpo_tensor,
        tensor,
    )


def build_left_environments(
    tensors: list[tc.Tensor], mpo_tensors: list[tc.Tensor]
) -> list[tc.Tensor]:
    """Return environments at all bonds; entry ``i`` contracts sites ``<i``."""
    envs = [identity_environment(tensors[0])]
    for tensor, mpo_tensor in zip(tensors, mpo_tensors):
        envs.append(update_left_environment(envs[-1], tensor, mpo_tensor))
    return envs


def build_right_environments(
    tensors: list[tc.Tensor], mpo_tensors: list[tc.Tensor]
) -> list[tc.Tensor]:
    """Return environments at all bonds; entry ``i`` contracts sites ``>=i``."""
    n = len(tensors)
    envs: list[tc.Tensor | None] = [None] * (n + 1)
    envs[n] = identity_environment(tensors[-1])
    for i in range(n - 1, -1, -1):
        envs[i] = update_right_environment(envs[i + 1], tensors[i], mpo_tensors[i])
    return envs  # type: ignore[return-value]


def apply_one_site(
    left: tc.Tensor,
    mpo_tensor: tc.Tensor,
    right: tc.Tensor,
    center: tc.Tensor,
) -> tc.Tensor:
    """Apply the one-site effective Hamiltonian to ``center(l,s,r)``."""
    return tc.einsum("pql,qnba,rnk,lbk->par", left, mpo_tensor, right, center)


def apply_zero_site(
    left: tc.Tensor, right: tc.Tensor, bond: tc.Tensor
) -> tc.Tensor:
    """Apply the zero-site/bond effective Hamiltonian to ``bond(l,r)``."""
    return tc.einsum("pql,rqk,lk->pr", left, right, bond)


def apply_two_site(
    left: tc.Tensor,
    mpo_left: tc.Tensor,
    mpo_right: tc.Tensor,
    right: tc.Tensor,
    theta: tc.Tensor,
) -> tc.Tensor:
    """Apply the two-site effective Hamiltonian to ``theta(l,s0,s1,r)``."""
    return tc.einsum(
        "pqr,qsab,stcd,utw,racw->pbdu",
        left,
        mpo_left,
        mpo_right,
        right,
        theta,
    )


def one_site_action(left, mpo_tensor, right, shape):
    """Return a flattened matrix-free one-site Heff callable."""
    def apply(vector: tc.Tensor) -> tc.Tensor:
        return apply_one_site(left, mpo_tensor, right, vector.reshape(shape)).reshape(-1)

    apply.dim = int(shape[0] * shape[1] * shape[2])
    return apply


def zero_site_action(left, right, shape):
    """Return a flattened matrix-free zero-site Heff callable."""
    def apply(vector: tc.Tensor) -> tc.Tensor:
        return apply_zero_site(left, right, vector.reshape(shape)).reshape(-1)

    apply.dim = int(shape[0] * shape[1])
    return apply


def two_site_action(left, mpo_left, mpo_right, right, shape):
    """Return a flattened matrix-free two-site Heff callable."""
    def apply(vector: tc.Tensor) -> tc.Tensor:
        return apply_two_site(
            left, mpo_left, mpo_right, right, vector.reshape(shape)
        ).reshape(-1)

    apply.dim = int(shape[0] * shape[1] * shape[2] * shape[3])
    return apply


__all__ = [
    "identity_environment",
    "update_left_environment",
    "update_right_environment",
    "build_left_environments",
    "build_right_environments",
    "apply_one_site",
    "apply_zero_site",
    "apply_two_site",
    "one_site_action",
    "zero_site_action",
    "two_site_action",
]
