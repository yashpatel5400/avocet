"""
Robust capacity planning for a server fleet with conformal uncertainty on arrival rates.

Pipeline:
- Simulate hourly arrival counts with seasonality (time-of-day, weekend) and trend.
- Train a small PyTorch regressor for arrival counts.
- Split conformal calibration with an L2 score to get an interval around the prediction.
- For each test hour, solve a robust capacity problem: choose server capacity to minimize
  capacity cost + worst-case shortage over the conformal interval (via sampled inner maximization).
- Compare robust vs nominal vs oracle decisions on realized arrivals.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp

from robbuffet import L2Score, SplitConformalCalibrator


def simulate_capacity_data(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, 24, size=n)
    weekend = rng.integers(0, 2, size=n)
    trend = np.linspace(0.0, 1.0, n)
    base = 50 + 15 * np.sin(2 * np.pi * hours / 24.0) + 8 * weekend + 10 * trend
    lam_true = np.clip(base + rng.normal(scale=3.0, size=n), 5, None)
    counts = rng.poisson(lam_true)
    X = np.stack([hours / 23.0, weekend, trend], axis=1).astype(np.float32)
    y = counts.astype(np.float32)
    return X, y


class SmallRegressor(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(model: nn.Module, loader: DataLoader, epochs: int = 80, lr: float = 2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()


def solve_deterministic_capacity(
    lam: float, service_rate: float, cap_cost: float, shortage_cost: float, max_cap: float
) -> float:
    w = cp.Variable()
    cost = cap_cost * w + shortage_cost * cp.pos(lam - service_rate * w)
    prob = cp.Problem(cp.Minimize(cost), [w >= 0, w <= max_cap])
    prob.solve(solver=cp.ECOS)
    if w.value is None:
        raise RuntimeError(f"Deterministic capacity solve failed: {prob.status}")
    return float(w.value)


def solve_robust_capacity(
    region,
    service_rate: float,
    cap_cost: float,
    shortage_cost: float,
    max_cap: float,
    num_samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    samples = region.sample(num_samples, rng=rng).reshape(-1)
    w = cp.Variable()
    t = cp.Variable()
    constraints = [w >= 0, w <= max_cap]
    for lam in samples:
        cost = cap_cost * w + shortage_cost * cp.pos(lam - service_rate * w)
        constraints.append(t >= cost)
    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(solver=cp.ECOS)
    if w.value is None:
        raise RuntimeError(f"Robust capacity solve failed: {prob.status}")
    return float(w.value)


def realized_cost(capacity: float, lam: float, service_rate: float, cap_cost: float, shortage_cost: float) -> float:
    shortage = max(lam - service_rate * capacity, 0.0)
    return cap_cost * capacity + shortage_cost * shortage


def run_experiment(
    alpha: float = 0.1,
    service_rate: float = 45.0,
    cap_cost: float = 3.0,
    shortage_cost: float = 10.0,
    max_cap: float = 200.0,
    num_test: int = 200,
    seed: int = 0,
):
    torch.manual_seed(seed)
    X, y = simulate_capacity_data(seed=seed)
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    X = X[idx]
    y = y[idx]

    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train : n_train + n_cal], y[n_train : n_train + n_cal]
    X_test, y_test = X[n_train + n_cal :], y[n_train + n_cal :]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
    cal_loader = DataLoader(TensorDataset(torch.tensor(X_cal), torch.tensor(y_cal)), batch_size=64, shuffle=False)

    model = SmallRegressor(d_in=X.shape[1])
    train_model(model, train_loader)

    score_fn = L2Score()
    calibrator = SplitConformalCalibrator(model, score_fn, cal_loader)
    q = calibrator.calibrate(alpha=alpha)

    selected = min(num_test, len(X_test))
    robust_costs = []
    nominal_costs = []
    oracle_costs = []
    coverages = []
    for i in range(selected):
        x_i = torch.tensor(X_test[i : i + 1])
        lam_true = float(y_test[i])
        region = calibrator.predict_region(x_i)
        lam_pred = float(model(x_i).detach().cpu().numpy().squeeze())
        coverages.append(abs(lam_true - lam_pred) <= q)

        cap_robust = solve_robust_capacity(region, service_rate, cap_cost, shortage_cost, max_cap, num_samples=64, seed=seed + i)
        cap_nom = solve_deterministic_capacity(lam_pred, service_rate, cap_cost, shortage_cost, max_cap)
        cap_oracle = solve_deterministic_capacity(lam_true, service_rate, cap_cost, shortage_cost, max_cap)

        robust_costs.append(realized_cost(cap_robust, lam_true, service_rate, cap_cost, shortage_cost))
        nominal_costs.append(realized_cost(cap_nom, lam_true, service_rate, cap_cost, shortage_cost))
        oracle_costs.append(realized_cost(cap_oracle, lam_true, service_rate, cap_cost, shortage_cost))

    print(f"Calibrated L2 radius q (alpha={alpha}): {q:.2f}")
    print(f"Empirical coverage on held-out set: {np.mean(coverages):.3f}")
    print(f"Avg realized cost (robust):  {np.mean(robust_costs):.2f} ± {np.std(robust_costs):.2f}")
    print(f"Avg realized cost (nominal): {np.mean(nominal_costs):.2f} ± {np.std(nominal_costs):.2f}")
    print(f"Avg realized cost (oracle):  {np.mean(oracle_costs):.2f} ± {np.std(oracle_costs):.2f}")
    print(f"First test point decisions -> robust: {cap_robust:.2f}, nominal: {cap_nom:.2f}, oracle: {cap_oracle:.2f}, true lambda: {lam_true:.2f}")
    return {
        "alpha": alpha,
        "q_calibrated": float(q),
        "avg_cost_robust": float(np.mean(robust_costs)),
        "avg_cost_nominal": float(np.mean(nominal_costs)),
        "avg_cost_oracle": float(np.mean(oracle_costs)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robust capacity planning with conformal intervals.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level.")
    parser.add_argument("--num-test", type=int, default=200, help="Number of test points to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    run_experiment(alpha=args.alpha, num_test=args.num_test, seed=args.seed)
