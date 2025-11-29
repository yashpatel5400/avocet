"""
Deterministic robust optimization with an L2 prediction region.
Minimize ||w||_2 subject to worst-case linear constraint <w, theta> <= 1
for all theta in an L2 ball around a point prediction.
"""

import cvxpy as cp
import numpy as np

from avocet import PredictionRegion, robustify_affine_leq


def main():
    center = np.array([0.2, -0.1])
    radius = 0.3
    region = PredictionRegion.l2_ball(center=center, radius=radius)

    w = cp.Variable(2)
    constraint = robustify_affine_leq(theta_direction=w, rhs=1.0, region=region)
    problem = cp.Problem(cp.Minimize(cp.norm(w, 2)), [constraint])
    problem.solve(solver="ECOS")
    print("status:", problem.status)
    print("robust w*:", w.value)


if __name__ == "__main__":
    main()
