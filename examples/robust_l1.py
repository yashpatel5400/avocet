"""
Deterministic robust optimization with an L1 prediction region.
Minimize ||w||_1 subject to worst-case linear constraint <w, theta> <= 1
for all theta in an L1 ball around a point prediction.
"""

import cvxpy as cp
import numpy as np

from avocet import PredictionRegion, robustify_affine_leq


def main():
    center = np.array([0.0, 0.0])
    radius = 0.5
    region = PredictionRegion.l1_ball(center=center, radius=radius)

    w = cp.Variable(2)
    constraint = robustify_affine_leq(theta_direction=w, rhs=1.0, region=region)
    problem = cp.Problem(cp.Minimize(cp.norm1(w)), [constraint])
    problem.solve(solver="ECOS")
    print("status:", problem.status)
    print("robust w*:", w.value)


if __name__ == "__main__":
    main()
