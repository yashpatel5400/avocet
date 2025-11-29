import torch
from torch.utils.data import DataLoader, TensorDataset

from avocet import L2Score, SplitConformalCalibrator


def test_split_conformal_calibration():
    # Simple predictor: identity mapping
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    predictor = Identity()
    # calibration data: small noise
    x_cal = torch.randn(50, 2)
    y_cal = x_cal + 0.05 * torch.randn_like(x_cal)
    loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=16, shuffle=False)

    cal = SplitConformalCalibrator(predictor, L2Score(), loader)
    alpha = 0.1
    q = cal.calibrate(alpha=alpha)
    assert q > 0.0

    x_new = torch.zeros(1, 2)
    region = cal.predict_region(x_new)
    assert region.radius == q
    assert region.contains(region.center)
