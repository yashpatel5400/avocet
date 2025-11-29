import cvxpy as cp
import numpy as np

from avocet import (
    PredictionRegion,
    robustify_affine_leq,
    support_function,
)


def test_support_function_l2():
    region = PredictionRegion.l2_ball(center=np.array([0.0, 0.0]), radius=1.0)
    direction = np.array([1.0, 2.0])
    val = support_function(region, direction)
    # For L2 ball, h(d) = <d, c> + r * ||d||_2 = 0 + 1 * sqrt(5)
    assert np.isclose(val.value, np.sqrt(5))


def test_robust_constraint_linf():
    region = PredictionRegion.linf_ball(center=np.array([0.0, 0.0]), radius=0.5)
    w = cp.Variable(2)
    constr = robustify_affine_leq(theta_direction=w, rhs=1.0, region=region)
    prob = cp.Problem(cp.Minimize(cp.norm(w, 2)), [constr])
    prob.solve(solver="ECOS")
    assert prob.status == cp.OPTIMAL
    assert np.all(np.abs(w.value) <= 1.0)  # should satisfy robust constraint
