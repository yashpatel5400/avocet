# Examples

Run any script with `python examples/<script>.py`.

- `robust_shortest_path_metrla.py`: Conformalized DCRNN_PyTorch forecasts + robust shortest path on METR-LA (requires the `examples/DCRNN_PyTorch` submodule and its precomputed predictions NPZ).
- `robust_bike_newsvendor.py`: Conformal calibration on UCI Bike Sharing data + robust newsvendor decisions vs nominal.
- `robust_capacity_planning.py`: Synthetic arrival forecasting + robust server capacity sizing with conformal intervals.
- `robust_fractional_knapsack.py`: SBIBM simulator + flow-based posterior samples for robust fractional knapsack.

### Math formulations

**Bike newsvendor**

- Region: $\mathcal{C}(x) = \{c : \|c - \hat{y}(x)\|_2 \le q\}$
- Objective: $\min_{q \ge 0} \max_{c \in \mathcal{C}(x)} \; c_u (c - q)^+ + c_o (q - c)^+.$

**METR-LA shortest path**

- Region: $\mathcal{C}(x) = \bigcup_{k} \{c : \|c - \hat{c}_k\|_2 \le q\}$
- Objective: $\min_{w} \max_{c \in \mathcal{C}(x)} \langle c, w\rangle \quad \text{s.t. } A w = b,\; 0 \le w \le 1.$

**Fractional knapsack (SBIBM)**

- Region: $\mathcal{C}(x) = \bigcup_k \{v : \|v - \hat{v}_k\|_2 \le q\}$ (weights fixed to nominal proxy).
- Objective: $\max_{x} \; \langle v, x\rangle \quad \text{s.t. } \langle w, x\rangle \le B,\; 0 \le x \le 1$, with robust variant using worst-case $v$.

**Capacity planning (synthetic arrivals)**

- Region: $\mathcal{C}(x) = \{ \lambda : |\lambda - \hat{\lambda}(x)| \le q\}$ (L2 interval around predicted arrival rate).
- Objective: $\min_{0 \le c \le \bar{c}} \; \max_{\lambda \in \mathcal{C}(x)} \; c_{\text{cap}} \, c + c_{\text{short}} \, (\lambda - \mu c)^+$, where $c$ is capacity (servers), $\mu$ is service rate, and $c_{\text{cap}}, c_{\text{short}}$ are cost coefficients. Inner max is approximated by sampling from $\mathcal{C}(x)$.

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
