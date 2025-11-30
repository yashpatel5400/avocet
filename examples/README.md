# Example: Predict-Then-Optimize with Conformal Calibration

This directory contains an end-to-end example showing how to conformalize a predictor and use the resulting region in a robust optimization problem.

## `robust_bike_newsvendor.py`
Bike rental demand (UCI Bike Sharing):
1. Train a PyTorch predictor for daily bike demand.
2. Calibrate with split conformal (L2), plot calibration curve.
3. Robust newsvendor decision using scenario sampling over conformal L2-ball regions; compare to nominal decisions on test days.

## `robust_shortest_path_metrla.py`
Robust shortest path on METR-LA with conformalized DCRNN forecasts:
1. Use the `examples/DCRNN_PyTorch` submodule pretrained on METR-LA to forecast edge speeds; derive costs.
2. Calibrate with GPCP over sampled forecasts to get a union-of-balls region.
3. Solve robust vs nominal flows and visualize.
