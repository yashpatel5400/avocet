# Examples

Run any script with `python examples/<script>.py`.

- `robust_shortest_path_metrla.py`: Conformalized DCRNN_PyTorch forecasts + robust shortest path on METR-LA (requires the `examples/DCRNN_PyTorch` submodule and its precomputed predictions NPZ).
- `robust_bike_newsvendor.py`: Conformal calibration on UCI Bike Sharing data + robust newsvendor decisions vs nominal.
- `robust_fractional_knapsack.py`: SBIBM simulator + flow-based posterior samples for robust fractional knapsack.

See `examples/README.md` for mathematical formulations.

## Empirical results (10 trials)

### Newsvendor (Bike Sharing)

| method  | mean objective | std    | paired t-test (robust < nominal) |
|---------|----------------|--------|----------------------------------|
| robust  | 2560.51        | 24.30  | t = -90.94, p = 5.958e-15        |
| nominal | 4370.20        | 83.10  | –                                |

### Shortest path (METR-LA)

| method  | mean objective | std      | paired t-test (robust < nominal) |
|---------|----------------|----------|----------------------------------|
| robust  | 109.58         | 15.56    | t = -9.52, p = 2.682e-06         |
| nominal | 12112.04       | 3780.02  | –                                |
