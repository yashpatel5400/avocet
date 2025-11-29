"""
Deterministic robust optimization with an ellipsoidal prediction region (Mahalanobis score).
Minimize ||w||_2 subject to worst-case linear constraint <w, theta> <= 1
for all theta in an ellipsoid centered at a point prediction.
"""

import cvxpy as cp
import numpy as np

from avocet import PredictionRegion, robustify_affine_leq


def main():
    center = np.array([0.0, 0.0])
    # Shape matrix W defines (theta-center)^T W (theta-center) <= r^2
    W = np.array([[4.0, 0.0], [0.0, 1.0]])  # tighter along first dimension
    radius = 0.8
    region = PredictionRegion.ellipsoid(center=center, shape_matrix=W, radius=radius)

    w = cp.Variable(2)
    constraint = robustify_affine_leq(theta_direction=w, rhs=1.0, region=region)
    problem = cp.Problem(cp.Minimize(cp.norm(w, 2)), [constraint])
    problem.solve(solver="ECOS")
    print("status:", problem.status)
    print("robust w*:", w.value)


if __name__ == "__main__":
    main()
