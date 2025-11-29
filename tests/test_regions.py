import numpy as np

from avocet.region import L2BallRegion, L1BallRegion, LinfBallRegion, EllipsoidRegion


def test_l2_contains_and_sample():
    center = np.array([0.0, 0.0])
    region = L2BallRegion(center=center, radius=1.0)
    assert region.contains(center)
    assert not region.contains(np.array([2.0, 0.0]))
    samples = region.sample(10)
    assert samples.shape == (10, 2)


def test_l1_and_linf_membership():
    center = np.array([0.0, 0.0])
    l1 = L1BallRegion(center=center, radius=1.0)
    linf = LinfBallRegion(center=center, radius=1.0)
    assert l1.contains(np.array([0.5, 0.5]))  # L1 norm = 1
    assert not l1.contains(np.array([0.8, 0.5]))  # L1 norm > 1
    assert linf.contains(np.array([0.5, -0.5]))
    assert not linf.contains(np.array([1.2, 0.0]))


def test_ellipsoid_membership():
    center = np.array([0.0, 0.0])
    W = np.eye(2)
    region = EllipsoidRegion(center=center, shape_matrix=W, radius=1.0)
    assert region.contains(np.array([0.5, 0.5]))
    assert not region.contains(np.array([2.0, 0.0]))
