"""
Avocet: Conformal prediction and robust decision-making toolkit.

Exposes:
- ScoreFunction implementations and region geometry helpers.
- SplitConformalCalibrator for calibration and region generation.
- Region visualization utilities.
- Scenario-based robust decision-making helpers.
"""

from .scores import L1Score, L2Score, LinfScore, MahalanobisScore, ScoreFunction
from .calibration import SplitConformalCalibrator
from .region import PredictionRegion, ScoreGeometry
from .decision import (
    ScenarioRobustOptimizer,
    support_function,
    robustify_affine_objective,
    robustify_affine_leq,
)

__all__ = [
    "L1Score",
    "L2Score",
    "LinfScore",
    "MahalanobisScore",
    "ScoreFunction",
    "SplitConformalCalibrator",
    "PredictionRegion",
    "ScoreGeometry",
    "ScenarioRobustOptimizer",
    "support_function",
    "robustify_affine_objective",
    "robustify_affine_leq",
]
