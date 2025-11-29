from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np

from .region import (
    EllipsoidRegion,
    L1BallRegion,
    L2BallRegion,
    LinfBallRegion,
    PredictionRegion,
    UnionRegion,
)


ObjectiveFn = Callable[[cp.Variable, np.ndarray], cp.Expression]
ConstraintFn = Callable[[cp.Variable, np.ndarray], Sequence[cp.Constraint]]


def support_function(region: PredictionRegion, direction) -> cp.Expression:
    """
    Support function h_R(direction) = max_{theta in R} <direction, theta>.
    Assumes direction is a CVXPY expression or NumPy array.
    """
    if isinstance(region, UnionRegion):
        # Exact support of a union is the max of component supports.
        comps = [support_function(r, direction) for r in region.regions]
        return cp.maximum(*comps)
    if not isinstance(direction, cp.Expression):
        direction = cp.Constant(direction)
    if isinstance(region, L2BallRegion):
        base = direction @ region.center
        return base + region.radius * cp.norm(direction, 2)
    if isinstance(region, L1BallRegion):
        base = direction @ region.center
        return base + region.radius * cp.norm(direction, "inf")
    if isinstance(region, LinfBallRegion):
        base = direction @ region.center
        return base + region.radius * cp.norm(direction, 1)
    if isinstance(region, EllipsoidRegion):
        base = direction @ region.center
        w_inv = np.linalg.inv(region.shape_matrix)
        quad = cp.quad_form(direction, w_inv)
        return base + region.radius * cp.sqrt(quad)
    raise NotImplementedError(f"Support function not implemented for {region.name}")


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


class DanskinRobustOptimizer:
    """
    Gradient-based min-max solver using Danskin's Theorem.

    Assumes objective is convex in w and inner maximization over theta can be solved per region.
    The user provides:
      - inner_objective_fn(theta_var, w_np) -> cp.Expression (concave in theta).
      - value_and_grad_fn(w_np, theta_np) -> (value, grad_w_np).
      - project_fn: optional projection on w after each step.
    Supports unions by maximizing over components.
    """

    def __init__(
        self,
        region: PredictionRegion,
        inner_objective_fn: Callable[[cp.Variable, np.ndarray], cp.Expression],
        value_and_grad_fn: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]],
        project_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        solver: str = "ECOS",
    ):
        self.region = region
        self.inner_objective_fn = inner_objective_fn
        self.value_and_grad_fn = value_and_grad_fn
        self.project_fn = project_fn
        self.solver = solver

    def _argmax_theta(self, w: np.ndarray, region: PredictionRegion) -> tuple[np.ndarray, float]:
        if isinstance(region, UnionRegion):
            best_theta = None
            best_val = -np.inf
            for comp in region.regions:
                theta_candidate, val_candidate = self._argmax_theta(w, comp)
                if val_candidate > best_val:
                    best_val = val_candidate
                    best_theta = theta_candidate
            assert best_theta is not None
            return best_theta, best_val
        theta = cp.Variable(region.center.shape)  # type: ignore[attr-defined]
        obj = cp.Maximize(self.inner_objective_fn(theta, w))
        constraints = region.cvxpy_constraints(theta)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=self.solver)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Inner maximization failed with status {prob.status}")
        theta_star = np.array(theta.value).astype(float)
        return theta_star, float(obj.value)

    def solve(
        self,
        w0: np.ndarray,
        step_size: float = 1e-1,
        max_iters: int = 200,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[float]]:
        w = np.array(w0, dtype=float)
        history: list[float] = []
        for it in range(max_iters):
            theta_star, _ = self._argmax_theta(w, self.region)
            val, grad_w = self.value_and_grad_fn(w, theta_star)
            history.append(val)
            w_next = w - step_size * grad_w
            if self.project_fn is not None:
                w_next = self.project_fn(w_next)
            if np.linalg.norm(w_next - w) < tol:
                w = w_next
                break
            w = w_next
            if verbose:
                print(f"iter {it}: val={val:.4f}, grad_norm={np.linalg.norm(grad_w):.4f}")
        return w, history


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
        constraints: List[cp.Constraint] = []
        if isinstance(region, UnionRegion):
            # solve per-component and enforce t >= max objective across components
            for comp in region.regions:
                samples = comp.sample(self.num_samples // len(region.regions) or 1, rng=self.rng)
                for theta in samples:
                    constraints.append(t >= self.objective_fn(w, theta))
                    constraints.extend(self.constraints_fn(w, theta))
        else:
            samples = region.sample(self.num_samples, rng=self.rng)
            for theta in samples:
                constraints.append(t >= self.objective_fn(w, theta))
                constraints.extend(self.constraints_fn(w, theta))
        problem = cp.Problem(cp.Minimize(t), constraints)
        if solver is not None:
            problem.solve(solver=solver)
        return problem
