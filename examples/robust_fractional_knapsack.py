"""
Robust fractional knapsack using SBIBM's two_moons simulator and conformal GPCP regions.

Steps:
- Use sbibm.get_task("two_moons") to simulate (theta, x) pairs.
- Map theta to item values/weights; train an MLP predictor on x -> [values, weights].
- Calibrate with GPCP by sampling noisy forecasts; build a union-of-balls region over values.
- Solve nominal vs robust fractional knapsack and report objectives.
"""

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from robbuffet import SplitConformalCalibrator
from robbuffet.data import SimulationDataset
from robbuffet.scores import GPCPScore, conformal_quantile

try:
    import sbibm
except ImportError as e:  # pragma: no cover - optional dependency
    sbibm = None
    _sbibm_import_error = e
else:
    _sbibm_import_error = None


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, d_out),
        )

    def forward(self, x):
        return self.net(x)


def build_dataset(train_size=800, cal_size=200, test_size=200, n_items=10, capacity=5.0, seed=0):
    if sbibm is None:
        raise ImportError(
            "sbibm is required for this example. Install with `pip install sbibm`."
        ) from _sbibm_import_error

    task = sbibm.get_task("two_moons")
    prior = task.get_prior()
    simulator = task.get_simulator()
    rng = torch.Generator().manual_seed(seed)

    def x_sampler_fn(n: int, _rng: np.random.Generator):
        # Sample theta and simulate x
        theta = prior(num_samples=n)
        x = simulator(theta)
        return x.numpy()

    def y_sampler_fn(X, _rng: np.random.Generator):
        # Map latent theta (not directly available) -> approximate via inverse mapping assumption using x features
        # As a proxy, derive item values from x stats.
        x_t = torch.as_tensor(X, dtype=torch.float32)
        mean_feat = x_t.mean(dim=1, keepdim=True)
        std_feat = x_t.std(dim=1, keepdim=True)
        base_vals = 5 + 2 * mean_feat + 0.5 * torch.randn((x_t.shape[0], n_items))
        base_vals = base_vals + std_feat
        weights = torch.clamp(0.5 + 0.1 * base_vals + 0.05 * torch.randn_like(base_vals), min=0.1)
        return torch.cat([base_vals, weights], dim=1).numpy()

    return SimulationDataset(
        x_sampler=x_sampler_fn,
        y_sampler=y_sampler_fn,
        train_size=train_size,
        cal_size=cal_size,
        test_size=test_size,
        seed=seed,
    )


def solve_nominal(values, weights, capacity):
    # Fractional knapsack nominal: take ratio ordering.
    ratio = values / weights
    order = np.argsort(ratio)[::-1]
    remaining = capacity
    x = np.zeros_like(values)
    for idx in order:
        take = min(1.0, remaining / weights[idx])
        x[idx] = take
        remaining -= take * weights[idx]
        if remaining <= 1e-6:
            break
    return x, values @ x


def solve_robust(values_centers, radius, weights, capacity):
    # Heuristic robust greedy: use worst-case per-item value proxy (min center minus radius)
    worst_vals = values_centers.min(axis=0) - radius
    ratio = worst_vals / weights
    order = np.argsort(ratio)[::-1]
    remaining = capacity
    x = np.zeros_like(weights)
    for idx in order:
        take = min(1.0, remaining / weights[idx])
        x[idx] = take
        remaining -= take * weights[idx]
        if remaining <= 1e-6:
            break
    worst_case_value = values_centers @ x
    worst_case = worst_case_value.min() - radius * np.linalg.norm(x)
    return x, worst_case


def run_experiment(alpha=0.1, K=8, n_items=10, capacity=5.0, seed=0):
    dataset = build_dataset(n_items=n_items, capacity=capacity, seed=seed)

    train_loader = DataLoader(dataset.train, batch_size=64, shuffle=True)
    cal_loader = DataLoader(dataset.calibration, batch_size=64, shuffle=True)
    test_dataset = dataset.test

    model = MLP(d_in=dataset.train.tensors[0].shape[1], d_out=2 * n_items)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(50):
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    # Residual std for sampling
    model.eval()
    with torch.no_grad():
        cal_preds = []
        cal_true = []
        for xb, yb in cal_loader:
            cal_preds.append(model(xb))
            cal_true.append(yb)
    cal_preds = torch.cat(cal_preds)
    cal_true = torch.cat(cal_true)
    resid_std = (cal_true - cal_preds).std(dim=0, keepdim=True) + 1e-3

    def sampler(xb: torch.Tensor) -> torch.Tensor:
        base = model(xb).detach().cpu().numpy()
        noise = np.random.normal(scale=resid_std.cpu().numpy().squeeze(), size=(K, base.shape[0], base.shape[1]))
        samples = base[None, ...] + noise
        return torch.tensor(samples, dtype=torch.float32)

    score_fn = GPCPScore(sampler)
    calibrator = SplitConformalCalibrator(sampler, score_fn, cal_loader)
    q = calibrator.calibrate(alpha=alpha)

    # Calibration curve (optional)
    cal_scores = calibrator.compute_scores(cal_loader).numpy()
    test_scores = calibrator.compute_scores(DataLoader(test_dataset, batch_size=64)).numpy()
    alphas = np.linspace(0.05, 0.5, num=10)
    coverages = [float(np.mean(test_scores <= conformal_quantile(torch.tensor(cal_scores), a))) for a in alphas]

    # Evaluate on first test point
    x_test, y_test = test_dataset.tensors
    x0 = x_test[0:1]
    y0 = y_test[0].numpy()
    values_true = y0[:n_items]
    weights_true = y0[n_items:]

    region = calibrator.predict_region(x0)
    centers = np.stack([r.center for r in region.as_union()])
    radius = q

    mean_pred = centers.mean(axis=0)
    values_pred = mean_pred[:n_items]
    weights_pred = mean_pred[n_items:]

    # Robust and nominal solutions
    x_nom, nominal_obj_pred = solve_nominal(values_pred, weights_pred, capacity)
    x_rob, robust_obj_pred = solve_robust(centers[:, :n_items], radius, weights_pred, capacity)

    true_nominal_obj = float(values_true @ x_nom)
    true_robust_obj = float(values_true @ x_rob)

    print("Fractional knapsack results (single test instance):")
    print(f"  Nominal objective (true values): {true_nominal_obj:.4f}")
    print(f"  Robust objective  (true values): {true_robust_obj:.4f}")
    print(f"  Calibrated radius q: {q:.4f}")

    print("\nTable:")
    print(f"{'method':<12}{'true_obj':>12}")
    print(f"{'nominal':<12}{true_nominal_obj:>12.4f}")
    print(f"{'robust':<12}{true_robust_obj:>12.4f}")

    return {
        "avg_cost_nominal": -true_nominal_obj,  # negative so lower is better for t-test
        "avg_cost_robust": -true_robust_obj,
        "avg_cost_oracle": None,
        "q_calibrated": q,
        "coverage_alphas": alphas,
        "coverage": coverages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n-items", type=int, default=10)
    parser.add_argument("--capacity", type=float, default=5.0)
    args = parser.parse_args()
    run_experiment(alpha=args.alpha, K=args.K, n_items=args.n_items, capacity=args.capacity)
