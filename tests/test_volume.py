import numpy as np

from robbuffet import L1BallRegion, L2BallRegion, LinfBallRegion, EllipsoidRegion


def test_exact_volumes():
    l2 = L2BallRegion(center=np.zeros(2), radius=1.0)
    l1 = L1BallRegion(center=np.zeros(2), radius=1.0)
    linf = LinfBallRegion(center=np.zeros(2), radius=1.0)
    ell = EllipsoidRegion(center=np.zeros(2), shape_matrix=np.eye(2), radius=1.0)
    assert np.isclose(l2.volume, np.pi, atol=1e-6)
    assert np.isclose(l1.volume, 2.0, atol=1e-6)
    assert np.isclose(linf.volume, 4.0, atol=1e-6)
    assert np.isclose(ell.volume, np.pi, atol=1e-6)


def test_mc_volume():
    l2 = L2BallRegion(center=np.zeros(2), radius=1.0)
    est = l2.volume_mc(bounds=(np.array([-1.5, -1.5]), np.array([1.5, 1.5])), num_samples=5000)
    assert abs(est - np.pi) < 0.5
