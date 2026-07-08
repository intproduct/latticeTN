import torch as tc

from latticetn.ad_two_site import ADTwoSiteOptimizer, train_ad_two_site
from latticetn.mps import MPS
from latticetn.mpo import MPO


def _heisenberg_case(N=4, chi=4):
    tc.manual_seed(0)
    mps = MPS(N, 2, chi, dtype=tc.complex128, device="cpu")
    mpo = MPO.from_bonds(N, 2, dtype=tc.complex128, device="cpu").generate_heisenberg()
    return mps, mpo


def test_theta_normalization_preserves_rayleigh_energy_and_scale():
    mps, mpo = _heisenberg_case()
    opt = ADTwoSiteOptimizer(mps, mpo, bond=1)
    with tc.no_grad():
        opt.theta.mul_(1e12)
    before = opt.energy().detach()
    scale = opt.normalize_theta_()
    after = opt.energy().detach()
    assert scale > 1e10
    assert tc.isfinite(after)
    assert abs(float(before - after)) < 1e-10
    assert abs(float(opt.theta.norm().detach() - 1.0)) < 1e-12
    assert isinstance(opt.theta, tc.nn.Parameter)
    assert opt.theta.is_leaf


def test_train_preconditions_before_first_energy_closure(monkeypatch):
    events = []
    orig_norm = ADTwoSiteOptimizer.normalize_theta_
    orig_energy = ADTwoSiteOptimizer.energy

    def spy_norm(self, *args, **kwargs):
        events.append("normalize")
        return orig_norm(self, *args, **kwargs)

    def spy_energy(self):
        events.append("energy")
        return orig_energy(self)

    monkeypatch.setattr(ADTwoSiteOptimizer, "normalize_theta_", spy_norm)
    monkeypatch.setattr(ADTwoSiteOptimizer, "energy", spy_energy)
    mps, mpo = _heisenberg_case(N=3, chi=2)
    train_ad_two_site(
        mps,
        mpo,
        num_sweeps=1,
        local_steps=1,
        optimizer="adam",
        lr=1e-2,
        max_bond_dim=2,
        precondition="theta_norm",
    )
    assert events[0] == "normalize"
    assert "energy" in events
    assert events.index("normalize") < events.index("energy")


def test_precondition_none_remains_available():
    mps, mpo = _heisenberg_case(N=3, chi=2)
    res = train_ad_two_site(
        mps,
        mpo,
        num_sweeps=1,
        local_steps=1,
        optimizer="adam",
        lr=1e-2,
        max_bond_dim=2,
        precondition="none",
    )
    assert res["precondition"] == "none"
    assert any(x is None for x in res["precondition_norms"])
