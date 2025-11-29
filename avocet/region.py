from __future__ import annotations

import dataclasses
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - cvxpy may be optional at import time
    cp = None  # type: ignore


@dataclasses.dataclass
class ScoreGeometry:
    """
    Metadata describing the shape of prediction regions induced by a score function.

    Supported types:
    - name: "l2_ball" (convex), with params {"p": 2}
    - name: "union": union of convex subregions
    - name: "unknown": geometry not available
    """

    name: str
    convex: bool
    union: bool = False
    params: Optional[dict] = None

    def supports_cvxpy(self) -> bool:
        return cp is not None and self.convex


class PredictionRegion:
    """Prediction region abstraction with geometry-aware utilities."""

    def __init__(
        self,
        geometry: ScoreGeometry,
        center: Optional[np.ndarray] = None,
        radius: Optional[float] = None,
        components: Optional[Sequence["PredictionRegion"]] = None,
        shape_matrix: Optional[np.ndarray] = None,
    ):
        self.geometry = geometry
        self.center = center
        self.radius = radius
        self.components = list(components) if components is not None else None
        self.shape_matrix = shape_matrix

    @classmethod
    def l2_ball(cls, center: np.ndarray, radius: float) -> "PredictionRegion":
        geom = ScoreGeometry(name="l2_ball", convex=True, union=False, params={"p": 2})
        return cls(geometry=geom, center=center, radius=float(radius))

    @classmethod
    def l1_ball(cls, center: np.ndarray, radius: float) -> "PredictionRegion":
        geom = ScoreGeometry(name="l1_ball", convex=True, union=False, params={"p": 1})
        return cls(geometry=geom, center=center, radius=float(radius))

    @classmethod
    def linf_ball(cls, center: np.ndarray, radius: float) -> "PredictionRegion":
        geom = ScoreGeometry(name="linf_ball", convex=True, union=False, params={"p": np.inf})
        return cls(geometry=geom, center=center, radius=float(radius))

    @classmethod
    def ellipsoid(cls, center: np.ndarray, shape_matrix: np.ndarray, radius: float = 1.0) -> "PredictionRegion":
        """
        Ellipsoid: {(theta) : (theta - center)^T W (theta - center) <= radius^2}, W PSD.
        """
        geom = ScoreGeometry(name="ellipsoid", convex=True, union=False, params={"shape": "psd"})
        return cls(geometry=geom, center=center, radius=float(radius), shape_matrix=shape_matrix)

    @classmethod
    def union(cls, regions: Sequence["PredictionRegion"]) -> "PredictionRegion":
        geom = ScoreGeometry(name="union", convex=False, union=True)
        return cls(geometry=geom, components=regions)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw samples uniformly from the region (approximate for ball)."""
        if rng is None:
            rng = np.random.default_rng()
        if self.geometry.name == "l2_ball":
            assert self.center is not None and self.radius is not None
            dim = self.center.shape[-1] if self.center.ndim > 0 else 1
            # Sample from isotropic normal and project to the ball radius.
            raw = rng.normal(size=(n, dim))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            directions = raw / norms
            radii = rng.random(size=(n, 1)) ** (1.0 / dim) * self.radius
            return self.center + directions * radii
        if self.geometry.name == "l1_ball":
            assert self.center is not None and self.radius is not None
            dim = self.center.shape[-1] if self.center.ndim > 0 else 1
            # Sample direction from Laplace-like distribution and scale to L1 ball.
            exp_samples = rng.exponential(scale=1.0, size=(n, dim))
            signs = rng.choice([-1.0, 1.0], size=(n, dim))
            directions = signs * exp_samples
            l1_norms = np.sum(np.abs(directions), axis=1, keepdims=True)
            l1_norms[l1_norms == 0] = 1.0
            directions = directions / l1_norms
            radii = rng.random(size=(n, 1)) ** (1.0 / dim) * self.radius
            return self.center + directions * radii
        if self.geometry.name == "linf_ball":
            assert self.center is not None and self.radius is not None
            dim = self.center.shape[-1] if self.center.ndim > 0 else 1
            offsets = rng.uniform(low=-self.radius, high=self.radius, size=(n, dim))
            return self.center + offsets
        if self.geometry.name == "ellipsoid":
            assert self.center is not None and self.radius is not None and self.shape_matrix is not None
            dim = self.center.shape[-1]
            # Sample from unit L2 ball then map through inverse sqrt of W.
            raw = rng.normal(size=(n, dim))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            directions = raw / norms
            radii = rng.random(size=(n, 1)) ** (1.0 / dim) * self.radius
            unit_ball_samples = directions * radii
            # Compute inverse sqrt of shape matrix.
            eigvals, eigvecs = np.linalg.eigh(self.shape_matrix)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-12)) @ eigvecs.T
            transformed = unit_ball_samples @ inv_sqrt.T
            return self.center + transformed
        if self.geometry.name == "union" and self.components:
            counts = rng.multinomial(n, [1 / len(self.components)] * len(self.components))
            samples: List[np.ndarray] = []
            for cnt, region in zip(counts, self.components):
                if cnt > 0:
                    samples.append(region.sample(cnt, rng))
            return np.vstack(samples) if samples else np.empty((0, 0))
        raise NotImplementedError(f"Sampling not implemented for {self.geometry.name}")

    def contains(self, y: np.ndarray) -> bool:
        """Check membership for simple geometries."""
        if self.geometry.name == "l2_ball":
            assert self.center is not None and self.radius is not None
            return float(np.linalg.norm(y - self.center)) <= self.radius + 1e-8
        if self.geometry.name == "l1_ball":
            assert self.center is not None and self.radius is not None
            return float(np.linalg.norm(y - self.center, ord=1)) <= self.radius + 1e-8
        if self.geometry.name == "linf_ball":
            assert self.center is not None and self.radius is not None
            return float(np.linalg.norm(y - self.center, ord=np.inf)) <= self.radius + 1e-8
        if self.geometry.name == "ellipsoid":
            assert self.center is not None and self.radius is not None and self.shape_matrix is not None
            diff = y - self.center
            return float(diff.T @ self.shape_matrix @ diff) <= self.radius**2 + 1e-8
        if self.geometry.name == "union" and self.components:
            return any(region.contains(y) for region in self.components)
        raise NotImplementedError(f"Containment not implemented for {self.geometry.name}")

    def cvxpy_constraints(self, theta_var: "cp.Variable") -> List["cp.Constraint"]:
        """
        Return constraints describing the region for CVXPY-based optimization.

        Only available for convex regions with CVXPY installed.
        """
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        if self.geometry.name == "l2_ball":
            assert self.center is not None and self.radius is not None
            return [cp.norm(theta_var - self.center, 2) <= self.radius]
        if self.geometry.name == "l1_ball":
            assert self.center is not None and self.radius is not None
            return [cp.norm1(theta_var - self.center) <= self.radius]
        if self.geometry.name == "linf_ball":
            assert self.center is not None and self.radius is not None
            return [cp.norm(theta_var - self.center, "inf") <= self.radius]
        if self.geometry.name == "ellipsoid":
            assert self.center is not None and self.radius is not None and self.shape_matrix is not None
            return [cp.quad_form(theta_var - self.center, self.shape_matrix) <= self.radius**2]
        if self.geometry.name == "union":
            raise ValueError("Union regions require handling via decomposition.")
        raise NotImplementedError(f"Constraints not implemented for {self.geometry.name}")

    def is_convex(self) -> bool:
        if self.geometry.name == "union":
            return False
        return self.geometry.convex

    def as_union(self) -> List["PredictionRegion"]:
        if self.geometry.name == "union" and self.components:
            return list(self.components)
        return [self]
