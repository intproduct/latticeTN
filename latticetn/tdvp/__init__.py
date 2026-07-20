"""Traditional matrix-product-state TDVP time evolution.

This package implements the non-autograd Stage 12B baseline.  It is kept
separate from the AD optimizers so a future AD-TDVP implementation can reuse
the public evolution/result interfaces without coupling differentiation to
the classical projector-splitting integrator.
"""

from .krylov import lanczos_expm_action
from .tdvp import TDVP, TDVPResult

__all__ = ["TDVP", "TDVPResult", "lanczos_expm_action"]
