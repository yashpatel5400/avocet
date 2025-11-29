from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np

from .region import PredictionRegion


ObjectiveFn = Callable[[cp.Variable, np.ndarray], cp.Expression]
ConstraintFn = Callable[[cp.Variable, np.ndarray], Sequence[cp.Constraint]]


def support_function(region: PredictionRegion, direction) -> cp.Expression:
    """
    Support function h_R(direction) = max_{theta in R} <direction, theta>.
    Assumes direction is a CVXPY expression or NumPy array.
    """
    if region.geometry.name == "union":
        raise ValueError("Support function for union regions should be handled via decomposition.")
    center = region.center
    radius = region.radius
    if center is None or radius is None:
        raise ValueError("Region center and radius required for support computation.")
    # Normalize direction to CVXPY expression
    if not isinstance(direction, cp.Expression):
        direction = cp.Constant(direction)
    base = direction @ center
    if region.geometry.name == "l2_ball":
        return base + radius * cp.norm(direction, 2)
    if region.geometry.name == "l1_ball":
        return base + radius * cp.norm(direction, "inf")
    if region.geometry.name == "linf_ball":
        return base + radius * cp.norm(direction, 1)
    if region.geometry.name == "ellipsoid":
        if region.shape_matrix is None:
            raise ValueError("Ellipsoid requires shape_matrix.")
        w_inv = np.linalg.inv(region.shape_matrix)
        quad = cp.quad_form(direction, w_inv)
        return base + radius * cp.sqrt(quad)
    raise NotImplementedError(f"Support function not implemented for {region.geometry.name}")


def robustify_affine_objective(
    base_obj: cp.Expression, theta_direction: cp.Expression, region: PredictionRegion
) -> cp.Expression:
    """
    Robustify an affine objective term g(w) + <theta_direction(w), theta>.
    Returns g(w) + h_R(theta_direction(w)).
    """
    return base_obj + support_function(region, theta_direction)


def robustify_affine_leq(
    theta_direction: cp.Expression, rhs: cp.Expression, region: PredictionRegion
) -> cp.Constraint:
    """
    Robustify a linear inequality <theta_direction(w), theta> <= rhs for all theta in region.
    Returns constraint: h_R(theta_direction(w)) <= rhs.
    """
    return support_function(region, theta_direction) <= rhs


class ScenarioRobustOptimizer:
    """
    Scenario-based robust optimization over conformal prediction regions.

    The user provides:
    - decision variable shape,
    - objective_fn(decision_var, theta_sample) -> cp.Expression,
    - constraints_fn(decision_var, theta_sample) -> list of cp.Constraint.

    The optimizer samples the region and minimizes the worst-case objective
    over those samples (epigraph formulation). This yields an inner approximation
    of the true robust counterpart; increase num_samples for tighter results.
    """

    def __init__(
        self,
        decision_shape: Tuple[int, ...],
        objective_fn: ObjectiveFn,
        constraints_fn: Optional[ConstraintFn] = None,
        num_samples: int = 128,
        seed: Optional[int] = None,
    ):
        self.decision_shape = decision_shape
        self.objective_fn = objective_fn
        self.constraints_fn = constraints_fn or (lambda w, theta: [])
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def build_problem(
        self, region: PredictionRegion, solver: Optional[str] = None
    ) -> cp.Problem:
        w = cp.Variable(self.decision_shape)
        t = cp.Variable()
        samples = region.sample(self.num_samples, rng=self.rng)
        constraints: List[cp.Constraint] = []
        for theta in samples:
            constraints.append(t >= self.objective_fn(w, theta))
            constraints.extend(self.constraints_fn(w, theta))
        problem = cp.Problem(cp.Minimize(t), constraints)
        if solver is not None:
            problem.solve(solver=solver)
        return problem
