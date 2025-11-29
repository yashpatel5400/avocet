# Example: Predict-Then-Optimize with Conformal Calibration

This directory contains an end-to-end example showing how to conformalize a predictor and use the resulting region in a robust optimization problem.

## `robust_supply_planning.py`
Pipeline:

1. Train a PyTorch predictor $ f(x) $ for a 2D demand vector $ \theta $.
2. Split conformal calibration with L2 score produces an L2-ball region  
   $ \Theta(x) = \{\theta : \|\theta - f(x)\|_2 \le q_\alpha\} $.
3. Robust planning for new $ x_{\text{new}} $:

```math
\Theta = \Theta(x_{\text{new}}), \quad
\min_w \ \lambda \|w\|_2^2 - \min_{\theta \in \Theta} \langle w, \theta \rangle
\quad \text{s.t. } w \ge 0,\ \mathbf{1}^\top w \le 1
```

Robust objective via support:  
$ \min_w \ \lambda \|w\|_2^2 - \langle w, c \rangle + r \|w\|_2 $  
with the same constraints, where $ c $ is the predicted center and $ r $ the conformal radius.

The script also plots empirical calibration curves on held-out test data.

## `robust_shortest_path.py`
Synthetic predict-then-optimize example for robust shortest path:
1. Train a generative predictor (Gaussian) for edge costs given features.
2. Calibrate with a GPCP score using K samples per point; region is a union of L2 balls.
3. Robust shortest path: minimize worst-case cost across sampled centers with an L2 buffer.
4. Compare robust vs nominal cost on a held-out true cost vector.
