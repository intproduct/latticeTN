"""Projector-splitting TDVP driver for finite open-boundary MPS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping

import torch as tc

from .. import canonical as canonical
from .. import contractions
from ..mps import MPS
from ..mpo import MPO
from . import effective_hamiltonian as effective
from .krylov import lanczos_expm_action


Observable = Callable[[MPS], object]


@dataclass
class TDVPResult:
    """Time series and final state returned by :meth:`TDVP.evolve`."""

    mps: MPS
    times: list[float]
    energy_history: list[float]
    norm_history: list[float]
    observables_history: dict[str, list[object]] = field(default_factory=dict)
    truncation_history: list[dict] = field(default_factory=list)
    method: str = "one_site"

    @property
    def evolved_mps(self) -> MPS:
        return self.mps


def _left_split(center: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
    """``center = left_isometry @ bond`` using a reduced QR."""
    left_dim, physical_dim, right_dim = center.shape
    q, bond = tc.linalg.qr(center.reshape(left_dim * physical_dim, right_dim))
    return q.reshape(left_dim, physical_dim, q.shape[1]), bond


def _right_split(center: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
    """``center = bond @ right_isometry`` using an LQ-via-QR split."""
    left_dim, physical_dim, right_dim = center.shape
    q_transpose, r_transpose = tc.linalg.qr(
        center.reshape(left_dim, physical_dim * right_dim).transpose(0, 1)
    )
    right = q_transpose.transpose(0, 1).reshape(
        q_transpose.shape[1], physical_dim, right_dim
    )
    bond = r_transpose.transpose(0, 1)
    return bond, right


def _absorb_left(bond: tc.Tensor, tensor: tc.Tensor) -> tc.Tensor:
    return tc.einsum("ab,bsr->asr", bond, tensor)


def _absorb_right(tensor: tc.Tensor, bond: tc.Tensor) -> tc.Tensor:
    return tc.einsum("lsa,ab->lsb", tensor, bond)


def _adaptive_two_site_split(
    theta: tc.Tensor,
    *,
    direction: Literal["right", "left"],
    max_bond_dim: int,
    truncation_tol: float,
) -> tuple[tc.Tensor, tc.Tensor, float, int, int]:
    """SVD split with the smallest bond satisfying the discarded-weight tol."""
    left_dim, physical_left, physical_right, right_dim = theta.shape
    matrix = theta.reshape(left_dim * physical_left, physical_right * right_dim)
    u, singular_values, vh = tc.linalg.svd(matrix, full_matrices=False)
    full_rank = int(singular_values.numel())
    allowed = min(int(max_bond_dim), full_rank)
    weights = singular_values.real.square()
    total = float(weights.sum())

    if total == 0.0:
        kept = 1
        truncation = 0.0
    elif truncation_tol == 0.0:
        kept = allowed
        truncation = max(0.0, (total - float(weights[:kept].sum())) / total)
    else:
        cumulative = tc.cumsum(weights, dim=0)
        kept = allowed
        for candidate in range(1, allowed + 1):
            discarded = max(0.0, (total - float(cumulative[candidate - 1])) / total)
            if discarded <= truncation_tol:
                kept = candidate
                break
        truncation = max(0.0, (total - float(cumulative[kept - 1])) / total)

    u = u[:, :kept]
    singular_values = singular_values[:kept]
    vh = vh[:kept, :]
    if direction == "right":
        left_tensor = u.reshape(left_dim, physical_left, kept)
        right_tensor = (
            singular_values[:, None].to(vh.dtype) * vh
        ).reshape(kept, physical_right, right_dim)
    elif direction == "left":
        left_tensor = (
            u * singular_values[None, :].to(u.dtype)
        ).reshape(left_dim, physical_left, kept)
        right_tensor = vh.reshape(kept, physical_right, right_dim)
    else:  # pragma: no cover - guarded by the Literal callers
        raise ValueError("direction must be 'right' or 'left'")
    return left_tensor, right_tensor, truncation, kept, full_rank


def _as_history_value(value: object) -> object:
    if isinstance(value, tc.Tensor) and value.numel() == 1:
        scalar = value.detach().cpu().reshape(()).item()
        if isinstance(scalar, complex) and abs(scalar.imag) < 1e-14:
            return float(scalar.real)
        return scalar
    return value


class TDVP:
    """Traditional one- or two-site TDVP time evolution.

    Parameters
    ----------
    mps, mpo:
        Initial state and time-independent Hamiltonian MPO.
    dt:
        Real time step.  Positive values apply ``exp(-i H dt)``.
    method:
        ``"one_site"`` is the fixed-bond projector-splitting baseline.
        ``"two_site"`` is reserved for Stage 12B-2.
    device:
        Optional target device.  Both MPS and MPO tensors are copied there.
    """

    def __init__(
        self,
        mps: MPS,
        mpo: MPO,
        dt: float = 0.01,
        method: Literal["one_site", "two_site"] = "one_site",
        device: str | tc.device | None = None,
        *,
        krylov_dim: int = 30,
        krylov_tol: float = 1e-13,
        max_bond_dim: int | None = None,
        truncation_tol: float = 1e-12,
    ):
        if mps.N != mpo.length or mps.dim != mpo.dim:
            raise ValueError("MPS and MPO sizes/physical dimensions must match")
        if dt == 0:
            raise ValueError("dt must be nonzero")
        if method not in {"one_site", "two_site"}:
            raise ValueError("method must be 'one_site' or 'two_site'")
        if krylov_dim < 1:
            raise ValueError("krylov_dim must be >= 1")
        if max_bond_dim is not None and max_bond_dim < 1:
            raise ValueError("max_bond_dim must be >= 1")
        if not (0.0 <= truncation_tol < 1.0):
            raise ValueError("truncation_tol must satisfy 0 <= tol < 1")

        target = tc.device(device if device is not None else mps.tensors[0].device)
        if target.type == "cuda" and not tc.cuda.is_available():
            raise RuntimeError("CUDA TDVP requested but torch.cuda.is_available() is false")
        dtype = mps.tensors[0].dtype
        self.mps = MPS.from_tensors(
            [tensor.to(device=target, dtype=dtype) for tensor in mps.tensors],
            dtype=dtype,
            device=target,
            requires_grad=False,
        )
        self.mpo = MPO(
            [tensor.to(device=target, dtype=dtype) for tensor in mpo.tensors],
            length=mpo.length,
            dim=mpo.dim,
            dtype=dtype,
            device=target,
        )
        self.dt = float(dt)
        self.method = method
        self.device = target
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        current_max_bond = max(
            [int(tensor.shape[2]) for tensor in self.mps.tensors[:-1]] or [1]
        )
        self.max_bond_dim = (
            int(max_bond_dim)
            if max_bond_dim is not None
            else (max(2, current_max_bond) if method == "two_site" else current_max_bond)
        )
        self.truncation_tol = float(truncation_tol)

        # Projector splitting starts with the center at site zero and all
        # sites to its right in right-canonical form.  Normalize only once;
        # subsequent norm history measures integrator drift without masking it.
        initial = canonical.mixed_canonical(self.mps, center=0)
        tensors = [tensor.detach().clone() for tensor in initial.tensors]
        norm = tc.linalg.vector_norm(tensors[0])
        if not bool(tc.isfinite(norm)) or float(norm) == 0.0:
            raise ValueError("initial MPS must have finite nonzero norm")
        tensors[0] = tensors[0] / norm
        self.mps = MPS.from_tensors(
            tensors, dtype=dtype, device=target, requires_grad=False
        )

    def _evolve_tensor(self, action, tensor: tc.Tensor, dt: float) -> tc.Tensor:
        return lanczos_expm_action(
            action,
            tensor.reshape(-1),
            dt,
            krylov_dim=self.krylov_dim,
            tol=self.krylov_tol,
        ).reshape(tensor.shape)

    def _one_site_step(self, tensors: list[tc.Tensor]) -> list[tc.Tensor]:
        """Second-order symmetric one-site projector-splitting step."""
        n = len(tensors)
        if n == 1:
            boundary = effective.identity_environment(tensors[0])
            action = effective.one_site_action(
                boundary, self.mpo.tensors[0], boundary, tensors[0].shape
            )
            tensors[0] = self._evolve_tensor(action, tensors[0], self.dt)
            return tensors

        half = 0.5 * self.dt
        right_envs = effective.build_right_environments(tensors, self.mpo.tensors)
        left_envs = [effective.identity_environment(tensors[0])]

        # Left-to-right: forward site evolution, exact QR gauge move, then
        # backward zero-site evolution.  Right environments remain valid
        # because sites to the right have not yet changed.
        for i in range(n - 1):
            site_action = effective.one_site_action(
                left_envs[i], self.mpo.tensors[i], right_envs[i + 1], tensors[i].shape
            )
            tensors[i] = self._evolve_tensor(site_action, tensors[i], half)
            tensors[i], bond = _left_split(tensors[i])
            left_envs.append(
                effective.update_left_environment(
                    left_envs[i], tensors[i], self.mpo.tensors[i]
                )
            )
            bond_action = effective.zero_site_action(
                left_envs[i + 1], right_envs[i + 1], bond.shape
            )
            bond = self._evolve_tensor(bond_action, bond, -half)
            tensors[i + 1] = _absorb_left(bond, tensors[i + 1])

        # The turning-point center receives a full step; all other sites get
        # one half-step in each sweep direction.
        boundary_right = effective.identity_environment(tensors[-1])
        last_action = effective.one_site_action(
            left_envs[n - 1], self.mpo.tensors[-1], boundary_right, tensors[-1].shape
        )
        tensors[-1] = self._evolve_tensor(last_action, tensors[-1], self.dt)

        # Right-to-left: mirror the projector splitting with LQ gauge moves.
        right = boundary_right
        for i in range(n - 1, 0, -1):
            bond, tensors[i] = _right_split(tensors[i])
            right = effective.update_right_environment(
                right, tensors[i], self.mpo.tensors[i]
            )
            bond_action = effective.zero_site_action(left_envs[i], right, bond.shape)
            bond = self._evolve_tensor(bond_action, bond, -half)
            tensors[i - 1] = _absorb_right(tensors[i - 1], bond)
            site_action = effective.one_site_action(
                left_envs[i - 1], self.mpo.tensors[i - 1], right, tensors[i - 1].shape
            )
            tensors[i - 1] = self._evolve_tensor(site_action, tensors[i - 1], half)
        return tensors

    def _two_site_step(
        self, tensors: list[tc.Tensor]
    ) -> tuple[list[tc.Tensor], dict]:
        """Second-order two-site projector splitting with adaptive SVD bonds."""
        n = len(tensors)
        if n < 2:
            return self._one_site_step(tensors), {
                "max_truncation": 0.0,
                "total_truncation": 0.0,
                "bond_dims": [],
                "max_bond": 1,
                "updates": [],
            }

        half = 0.5 * self.dt
        updates: list[dict] = []
        right_envs = effective.build_right_environments(tensors, self.mpo.tensors)
        left_envs = [effective.identity_environment(tensors[0])]

        # Forward two-site projector terms.  Between adjacent bonds the
        # overlapping one-site projector is evolved backward for half a step.
        for i in range(n - 1):
            theta = tc.einsum("lsc,cer->lser", tensors[i], tensors[i + 1])
            action = effective.two_site_action(
                left_envs[i],
                self.mpo.tensors[i],
                self.mpo.tensors[i + 1],
                right_envs[i + 2],
                theta.shape,
            )
            theta = self._evolve_tensor(action, theta, half)
            tensors[i], tensors[i + 1], truncation, kept, full_rank = (
                _adaptive_two_site_split(
                    theta,
                    direction="right",
                    max_bond_dim=self.max_bond_dim,
                    truncation_tol=self.truncation_tol,
                )
            )
            updates.append({
                "direction": "right",
                "bond": i,
                "kept_bond": kept,
                "available_rank": full_rank,
                "truncation_error": truncation,
            })
            left_envs.append(
                effective.update_left_environment(
                    left_envs[i], tensors[i], self.mpo.tensors[i]
                )
            )
            if i < n - 2:
                site_action = effective.one_site_action(
                    left_envs[i + 1],
                    self.mpo.tensors[i + 1],
                    right_envs[i + 2],
                    tensors[i + 1].shape,
                )
                tensors[i + 1] = self._evolve_tensor(
                    site_action, tensors[i + 1], -half
                )

        # Reverse sweep mirrors the forward half and restores center site 0.
        right = effective.identity_environment(tensors[-1])
        for i in range(n - 2, -1, -1):
            theta = tc.einsum("lsc,cer->lser", tensors[i], tensors[i + 1])
            action = effective.two_site_action(
                left_envs[i],
                self.mpo.tensors[i],
                self.mpo.tensors[i + 1],
                right,
                theta.shape,
            )
            theta = self._evolve_tensor(action, theta, half)
            tensors[i], tensors[i + 1], truncation, kept, full_rank = (
                _adaptive_two_site_split(
                    theta,
                    direction="left",
                    max_bond_dim=self.max_bond_dim,
                    truncation_tol=self.truncation_tol,
                )
            )
            updates.append({
                "direction": "left",
                "bond": i,
                "kept_bond": kept,
                "available_rank": full_rank,
                "truncation_error": truncation,
            })
            right = effective.update_right_environment(
                right, tensors[i + 1], self.mpo.tensors[i + 1]
            )
            if i > 0:
                site_action = effective.one_site_action(
                    left_envs[i], self.mpo.tensors[i], right, tensors[i].shape
                )
                tensors[i] = self._evolve_tensor(site_action, tensors[i], -half)

        # Truncated SVD is non-unitary.  Standard two-site TDVP selects the
        # normalized representative after a completed symmetric step; because
        # the center is back at site zero its Frobenius norm is the MPS norm.
        center_norm = tc.linalg.vector_norm(tensors[0])
        if not bool(tc.isfinite(center_norm)) or float(center_norm) == 0.0:
            raise RuntimeError("two-site TDVP produced a non-finite or zero state")
        tensors[0] = tensors[0] / center_norm

        truncations = [float(update["truncation_error"]) for update in updates]
        bond_dims = [int(tensors[i].shape[2]) for i in range(n - 1)]
        diagnostics = {
            "max_truncation": max(truncations, default=0.0),
            "total_truncation": sum(truncations),
            "bond_dims": bond_dims,
            "max_bond": max(bond_dims, default=1),
            "updates": updates,
        }
        return tensors, diagnostics

    def _current_mps(self, tensors: list[tc.Tensor]) -> MPS:
        return MPS.from_tensors(
            tensors,
            dtype=tensors[0].dtype,
            device=tensors[0].device,
            requires_grad=False,
        )

    def evolve(
        self,
        steps: int,
        observables: Mapping[str, Observable] | None = None,
    ) -> TDVPResult:
        """Evolve for ``steps`` and record initial plus per-step diagnostics."""
        if steps < 0:
            raise ValueError("steps must be >= 0")
        observable_fns = dict(observables or {})
        tensors = [tensor.detach().clone() for tensor in self.mps.tensors]
        times: list[float] = []
        energies: list[float] = []
        norms: list[float] = []
        observable_history = {name: [] for name in observable_fns}
        truncation_history: list[dict] = []

        def record(step: int) -> None:
            state = self._current_mps(tensors)
            times.append(step * self.dt)
            energies.append(float(contractions.rayleigh_energy_native(state, self.mpo)))
            norms.append(float(contractions.native_norm(state)))
            for name, observable in observable_fns.items():
                observable_history[name].append(_as_history_value(observable(state)))

        with tc.no_grad():
            record(0)
            for step in range(1, steps + 1):
                if self.method == "one_site":
                    tensors = self._one_site_step(tensors)
                else:
                    tensors, diagnostics = self._two_site_step(tensors)
                    diagnostics = {"step": step, **diagnostics}
                    truncation_history.append(diagnostics)
                record(step)

        self.mps = self._current_mps(tensors)
        return TDVPResult(
            mps=self.mps,
            times=times,
            energy_history=energies,
            norm_history=norms,
            observables_history=observable_history,
            truncation_history=truncation_history,
            method=self.method,
        )


__all__ = ["TDVP", "TDVPResult"]
